//! MLX configuration, checkpoint integration, and resident binding for shared Llama.

use eredu_checkpoint::WeightQuantization;
use eredu_runtime::CausalModel;

use std::{collections::HashMap, path::Path};

use safemlx::{
    error::Exception,
    module::Module,
    ops::indexing::TryIndexOp,
    ops::{GgufCheckpoint, GgufMetadataValue},
    Array, Stream,
};
use serde_json::Value;

use eredu_architectures::llama::ModelArgs;

use crate::{
    backend::mlx::error::Error,
    backend::mlx::runtime::cache::{ConcatKeyValueCache, KeyValueCache},
    backend::mlx::runtime::checkpoint::load::{gguf_quantization_configs, GgufTensorNames},
    backend::mlx::runtime::media::input,
};

type SharedTransformerBlock =
    eredu_architectures::llama::TransformerBlock<crate::backend::mlx::nn::shared::MlxBackend>;
/// MLX specialization of the backend-neutral Llama block input.
pub type AttentionInput<'a, C> = eredu_architectures::llama::AttentionInput<'a, Array, C>;

/// MLX specialization of one backend-neutral Llama decoder block.
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    /// Shared architecture implementation specialized to MLX operators.
    pub inner: SharedTransformerBlock,
}

impl TransformerBlock {
    pub(crate) fn new_for_layer(
        args: &ModelArgs,
        layer_index: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let layer = usize::try_from(layer_index)
            .map_err(|_| Exception::custom(format!("invalid Llama layer index {layer_index}")))?;
        eredu_architectures::llama::TransformerBlock::<crate::backend::mlx::nn::shared::MlxBackend>::new(
            args, layer, stream,
        )
        .map(|inner| Self { inner })
        .map_err(|error| Exception::custom(error.to_string()))
    }

    /// Executes one replicated decoder block.
    pub fn forward<C>(
        &mut self,
        input: AttentionInput<'_, C>,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        C: eredu_nn::AttentionCache<Array>,
    {
        self.inner
            .forward(input, stream)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    /// Executes a block with rank-local heads and MLP intermediates.
    pub(crate) fn forward_tensor_parallel<C>(
        &mut self,
        hidden: &Array,
        mask: Option<&Array>,
        cache: Option<&mut C>,
        allow_sliding_prefill: bool,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        C: eredu_nn::AttentionCache<Array>,
    {
        self.inner
            .forward_tensor_parallel(
                AttentionInput {
                    hidden,
                    mask,
                    cache,
                    allow_sliding_prefill,
                },
                group,
                stream,
            )
            .map_err(|error| Exception::custom(error.to_string()))
    }
}

impl safemlx::module::ModuleParameters for TransformerBlock {
    fn num_parameters(&self) -> usize {
        crate::backend::mlx::nn::shared::neutral_parameter_refs(&self.inner, false)
            .entries
            .len()
    }
    fn parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        crate::backend::mlx::nn::shared::neutral_parameter_refs(&self.inner, false)
    }
    fn parameters_mut(&mut self) -> safemlx::module::ModuleParamMut<'_> {
        crate::backend::mlx::nn::shared::neutral_parameter_refs_mut(&mut self.inner)
    }
    fn trainable_parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        crate::backend::mlx::nn::shared::neutral_parameter_refs(&self.inner, true)
    }
    fn freeze_parameters(&mut self, _recursive: bool) {
        eredu_nn::Parameterized::set_trainable(&mut self.inner, false);
    }
    fn unfreeze_parameters(&mut self, _recursive: bool) {
        eredu_nn::Parameterized::set_trainable(&mut self.inner, true);
    }
    fn all_frozen(&self) -> Option<bool> {
        let states = crate::backend::mlx::nn::shared::neutral_parameter_states(&self.inner);
        (!states.is_empty()).then(|| states.iter().all(|trainable| !trainable))
    }
    fn any_frozen(&self) -> Option<bool> {
        let states = crate::backend::mlx::nn::shared::neutral_parameter_states(&self.inner);
        (!states.is_empty()).then(|| states.iter().any(|trainable| !trainable))
    }
}

/// Input for a resident Llama forward pass.
pub struct ModelInput<'a, C> {
    /// Token ids with shape `[batch, sequence]`.
    pub inputs: &'a Array,
    /// Optional caller-provided attention mask.
    pub mask: Option<&'a Array>,
    /// Mutable per-layer key/value cache.
    pub cache: &'a mut Vec<Option<C>>,
}

/// Resident MLX specialization of the shared Llama model.
#[derive(Debug, Clone)]
pub struct ResidentModel {
    /// Normalized model configuration and MLX weight encoding metadata.
    pub args: ModelArgs,
    /// Shared architecture specialized to native MLX modules.
    pub inner: eredu_architectures::llama::Model<crate::backend::mlx::nn::shared::MlxBackend>,
}

impl ResidentModel {
    /// Creates an unloaded resident model.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        let inner =
            eredu_architectures::llama::Model::<crate::backend::mlx::nn::shared::MlxBackend>::new(
                &args, stream,
            )
            .map_err(|error| Exception::custom(error.to_string()))?;
        Ok(Self { args, inner })
    }

    /// Creates one architecture-correct device cache per decoder layer.
    pub fn new_cache(&self) -> Vec<Option<ConcatKeyValueCache>> {
        eredu_architectures::llama::create_caches(&self.args, |_layer, window| match window {
            Some(window) => ConcatKeyValueCache::new_for_sliding_attention(window),
            None => ConcatKeyValueCache::new(),
        })
        .expect("validated Llama configuration has a valid cache schedule")
    }

    fn forward_shared<C>(
        &mut self,
        input: ModelInput<'_, C>,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        C: eredu_nn::AttentionCache<Array> + KeyValueCache,
    {
        self.inner
            .forward(&self.args, input.inputs, input.mask, input.cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))
    }
}

impl<C> Module<ModelInput<'_, C>> for ResidentModel
where
    C: eredu_nn::AttentionCache<Array> + KeyValueCache,
{
    type Output = Array;
    type Error = Exception;
    fn forward(&mut self, input: ModelInput<'_, C>, stream: &Stream) -> Result<Array, Exception> {
        self.forward_shared(input, stream)
    }
    fn training_mode(&mut self, mode: bool) {
        self.inner.decoder.embeddings.training_mode(mode);
        for layer in &mut self.inner.decoder.layers {
            layer.self_attention.query.training_mode(mode);
            layer.self_attention.key.training_mode(mode);
            layer.self_attention.value.training_mode(mode);
            layer.self_attention.output.training_mode(mode);
            layer.mlp.gate.training_mode(mode);
            layer.mlp.up.training_mode(mode);
            layer.mlp.down.training_mode(mode);
            layer.input_norm.training_mode(mode);
            layer.post_attention_norm.training_mode(mode);
        }
        self.inner.decoder.norm.training_mode(mode);
        if let Some(head) = &mut self.inner.lm_head {
            head.training_mode(mode);
        }
    }
}

impl safemlx::module::ModuleParameters for ResidentModel {
    fn num_parameters(&self) -> usize {
        crate::backend::mlx::nn::shared::neutral_parameter_refs(&self.inner, false)
            .entries
            .len()
    }
    fn parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        crate::backend::mlx::nn::shared::neutral_parameter_refs(&self.inner, false)
    }
    fn parameters_mut(&mut self) -> safemlx::module::ModuleParamMut<'_> {
        crate::backend::mlx::nn::shared::neutral_parameter_refs_mut(&mut self.inner)
    }
    fn trainable_parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        crate::backend::mlx::nn::shared::neutral_parameter_refs(&self.inner, true)
    }
    fn freeze_parameters(&mut self, _recursive: bool) {
        eredu_nn::Parameterized::set_trainable(&mut self.inner, false);
    }
    fn unfreeze_parameters(&mut self, _recursive: bool) {
        eredu_nn::Parameterized::set_trainable(&mut self.inner, true);
    }
    fn all_frozen(&self) -> Option<bool> {
        let states = crate::backend::mlx::nn::shared::neutral_parameter_states(&self.inner);
        (!states.is_empty()).then(|| states.iter().all(|trainable| !trainable))
    }
    fn any_frozen(&self) -> Option<bool> {
        let states = crate::backend::mlx::nn::shared::neutral_parameter_states(&self.inner);
        (!states.is_empty()).then(|| states.iter().any(|trainable| !trainable))
    }
}

/// Reads and normalizes Llama model arguments from `config.json`.
pub fn get_llama_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let file = std::fs::File::open(model_dir.as_ref().join("config.json"))?;
    eredu_architectures::llama::model_args_from_config_reader(file)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

pub(crate) fn validate_model_config_value(config: &Value) -> Result<(), Error> {
    eredu_architectures::llama::model_args_from_config_value(config)
        .map(|_| ())
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

pub fn model_args_from_config_value(config: &Value) -> Result<ModelArgs, Error> {
    eredu_architectures::llama::model_args_from_config_value(config)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

pub(crate) struct PreparedLlamaGguf {
    pub(crate) args: ModelArgs,
    pub(crate) eos_token_ids: Vec<u32>,
}

pub(crate) fn prepare_llama_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    quantization: Option<WeightQuantization>,
    _weights_stream: &Stream,
) -> Result<PreparedLlamaGguf, Error> {
    let mut args = model_args_from_gguf_catalog(checkpoint, metadata)?;
    let architecture = args.model_type.clone();
    let gguf_architecture = crate::core::GgufArchitecture::resolve(&architecture)?;
    crate::backend::mlx::structural::validate_gguf(
        gguf_architecture,
        checkpoint,
        metadata,
        crate::backend::mlx::ModelLoadOptions::default(),
    )
    .into_loader_result()?;

    checkpoint
        .catalog()
        .translated_outputs(eredu_architectures::llama::translate_gguf_weight_name)
        .map_err(safemlx::error::IoError::from)?;
    let quantized_weight_configs = gguf_quantization_configs(
        checkpoint,
        eredu_architectures::llama::translate_gguf_weight_name,
    )?;
    if let Some(quantization) = quantization {
        args.quantized_weights = None;
        args.quantization = Some(quantization);
        args.quantized_weight_configs = None;
    } else {
        args.quantized_weights = Some(quantized_weight_configs.keys().cloned().collect());
        args.quantization = None;
        args.quantized_weight_configs = Some(quantized_weight_configs);
    }

    let eos_token_ids = crate::backend::mlx::gguf_eos_token_ids(metadata)?;
    Ok(PreparedLlamaGguf {
        args,
        eos_token_ids,
    })
}

struct NeutralGgufCatalog<'a, T: ?Sized>(&'a T);

impl<T: GgufTensorNames + ?Sized> eredu_architectures::llama::GgufTensorCatalog
    for NeutralGgufCatalog<'_, T>
{
    fn contains(&self, name: &str) -> bool {
        self.0.contains_gguf_tensor(name)
    }

    fn any(&self, predicate: &mut dyn FnMut(&str) -> bool) -> bool {
        self.0.any_gguf_tensor(predicate)
    }
}

/// Parses the GGUF arguments shared by structural preflight and loading.
pub(crate) fn model_args_from_gguf_catalog(
    arrays: &(impl GgufTensorNames + ?Sized),
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<ModelArgs, Error> {
    eredu_architectures::llama::model_args_from_gguf_catalog(&NeutralGgufCatalog(arrays), metadata)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

impl<C> CausalModel<Vec<Option<C>>> for ResidentModel
where
    C: KeyValueCache + eredu_nn::AttentionCache<Array>,
{
    type Tensor = Array;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Vec<Option<C>>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let prompt_tokens = input::text_token_ids(input, stream)?;
        let logits = self.forward(
            ModelInput {
                inputs: &prompt_tokens,
                mask: None,
                cache,
            },
            stream,
        )?;
        logits.try_index_device((.., -1, ..), stream)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Vec<Option<C>>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let logits = self.forward(
            ModelInput {
                inputs: input_tokens,
                mask: None,
                cache,
            },
            stream,
        )?;
        logits.try_index_device((.., -1, ..), stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safemlx::{module::ModuleParameters, Device, DeviceType};

    fn model_args() -> ModelArgs {
        eredu_architectures::llama::model_args_from_config_value(&serde_json::json!({
            "model_type": "llama",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 0.00001,
            "vocab_size": 64,
            "max_position_embeddings": 128,
            "tie_word_embeddings": false
        }))
        .unwrap()
    }

    #[test]
    fn resident_model_exposes_exact_authoritative_parameter_ids() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let mut model = match ResidentModel::new(model_args(), &stream) {
            Ok(model) => model,
            Err(error) if error.to_string().contains("No Metal device available") => return,
            Err(error) => panic!("failed to construct resident Llama: {error}"),
        };
        let mut names = model
            .parameters()
            .flatten()
            .into_keys()
            .map(|name| name.to_string())
            .collect::<Vec<_>>();
        names.sort();
        assert_eq!(
            names,
            [
                "lm_head.weight",
                "model.embed_tokens.weight",
                "model.layers.0.input_layernorm.weight",
                "model.layers.0.mlp.down_proj.weight",
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
                "model.layers.0.post_attention_layernorm.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.o_proj.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
                "model.norm.weight",
            ]
        );
        assert!(names.iter().all(|name| !name.contains(".inner.")));
        assert_eq!(model.num_parameters(), names.len());
        assert_eq!(model.trainable_parameters().flatten().len(), names.len());

        model.freeze_parameters(true);
        assert_eq!(model.trainable_parameters().flatten().len(), 0);
        assert_eq!(model.all_frozen(), Some(true));
        model.unfreeze_parameters(true);
        assert_eq!(model.trainable_parameters().flatten().len(), names.len());
        assert_eq!(model.any_frozen(), Some(false));
    }
}

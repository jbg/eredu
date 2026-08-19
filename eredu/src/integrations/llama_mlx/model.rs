//! MLX configuration, checkpoint integration, and resident binding for shared Llama.

use eredu_checkpoint::WeightQuantization;

use std::{collections::HashMap, path::Path};

use safemlx::{
    error::Exception,
    module::{Module, ModuleParameters},
    ops::indexing::TryIndexOp,
    ops::{GgufCheckpoint, GgufMetadataValue},
    Array, Stream,
};
use serde_json::Value;

use eredu_architectures::llama::ModelArgs;

use crate::{
    backend::mlx::error::Error,
    backend::mlx::nn::generation::CausalLm,
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

fn insert_module_ref<'a>(
    map: &mut safemlx::module::ModuleParamRef<'a>,
    name: &str,
    module: &'a impl safemlx::module::ModuleParameters,
) {
    map.insert(
        name.into(),
        safemlx::nested::NestedValue::Map(module.parameters().entries),
    );
}

fn insert_module_mut<'a>(
    map: &mut safemlx::module::ModuleParamMut<'a>,
    name: &str,
    module: &'a mut impl safemlx::module::ModuleParameters,
) {
    map.insert(
        name.into(),
        safemlx::nested::NestedValue::Map(module.parameters_mut().entries),
    );
}

fn insert_trainable_ref<'a>(
    map: &mut safemlx::module::ModuleParamRef<'a>,
    name: &str,
    module: &'a impl safemlx::module::ModuleParameters,
) {
    map.insert(
        name.into(),
        safemlx::nested::NestedValue::Map(module.trainable_parameters().entries),
    );
}

fn attention_parameters(
    attention: &eredu_architectures::llama::Attention<crate::backend::mlx::nn::shared::MlxBackend>,
) -> safemlx::module::ModuleParamRef<'_> {
    let mut map = safemlx::module::ModuleParamRef::new();
    insert_module_ref(&mut map, "q_proj", &attention.query);
    insert_module_ref(&mut map, "k_proj", &attention.key);
    insert_module_ref(&mut map, "v_proj", &attention.value);
    insert_module_ref(&mut map, "o_proj", &attention.output);
    insert_module_ref(&mut map, "rope", &attention.rotary);
    map
}

fn attention_parameters_mut(
    attention: &mut eredu_architectures::llama::Attention<
        crate::backend::mlx::nn::shared::MlxBackend,
    >,
) -> safemlx::module::ModuleParamMut<'_> {
    let mut map = safemlx::module::ModuleParamMut::new();
    insert_module_mut(&mut map, "q_proj", &mut attention.query);
    insert_module_mut(&mut map, "k_proj", &mut attention.key);
    insert_module_mut(&mut map, "v_proj", &mut attention.value);
    insert_module_mut(&mut map, "o_proj", &mut attention.output);
    insert_module_mut(&mut map, "rope", &mut attention.rotary);
    map
}

fn mlp_parameters(
    mlp: &eredu_architectures::llama::Mlp<crate::backend::mlx::nn::shared::MlxBackend>,
) -> safemlx::module::ModuleParamRef<'_> {
    let mut map = safemlx::module::ModuleParamRef::new();
    insert_module_ref(&mut map, "gate_proj", &mlp.gate);
    insert_module_ref(&mut map, "up_proj", &mlp.up);
    insert_module_ref(&mut map, "down_proj", &mlp.down);
    map
}

fn mlp_parameters_mut(
    mlp: &mut eredu_architectures::llama::Mlp<crate::backend::mlx::nn::shared::MlxBackend>,
) -> safemlx::module::ModuleParamMut<'_> {
    let mut map = safemlx::module::ModuleParamMut::new();
    insert_module_mut(&mut map, "gate_proj", &mut mlp.gate);
    insert_module_mut(&mut map, "up_proj", &mut mlp.up);
    insert_module_mut(&mut map, "down_proj", &mut mlp.down);
    map
}

fn block_parameters(block: &SharedTransformerBlock) -> safemlx::module::ModuleParamRef<'_> {
    let mut map = safemlx::module::ModuleParamRef::new();
    map.insert(
        "self_attn".into(),
        safemlx::nested::NestedValue::Map(attention_parameters(&block.self_attention).entries),
    );
    map.insert(
        "mlp".into(),
        safemlx::nested::NestedValue::Map(mlp_parameters(&block.mlp).entries),
    );
    insert_module_ref(&mut map, "input_layernorm", &block.input_norm);
    insert_module_ref(
        &mut map,
        "post_attention_layernorm",
        &block.post_attention_norm,
    );
    map
}

fn block_parameters_mut(block: &mut SharedTransformerBlock) -> safemlx::module::ModuleParamMut<'_> {
    let mut map = safemlx::module::ModuleParamMut::new();
    map.insert(
        "self_attn".into(),
        safemlx::nested::NestedValue::Map(
            attention_parameters_mut(&mut block.self_attention).entries,
        ),
    );
    map.insert(
        "mlp".into(),
        safemlx::nested::NestedValue::Map(mlp_parameters_mut(&mut block.mlp).entries),
    );
    insert_module_mut(&mut map, "input_layernorm", &mut block.input_norm);
    insert_module_mut(
        &mut map,
        "post_attention_layernorm",
        &mut block.post_attention_norm,
    );
    map
}

fn block_trainable_parameters(
    block: &SharedTransformerBlock,
) -> safemlx::module::ModuleParamRef<'_> {
    let attention = &block.self_attention;
    let mut attention_map = safemlx::module::ModuleParamRef::new();
    insert_trainable_ref(&mut attention_map, "q_proj", &attention.query);
    insert_trainable_ref(&mut attention_map, "k_proj", &attention.key);
    insert_trainable_ref(&mut attention_map, "v_proj", &attention.value);
    insert_trainable_ref(&mut attention_map, "o_proj", &attention.output);
    insert_trainable_ref(&mut attention_map, "rope", &attention.rotary);

    let mut mlp_map = safemlx::module::ModuleParamRef::new();
    insert_trainable_ref(&mut mlp_map, "gate_proj", &block.mlp.gate);
    insert_trainable_ref(&mut mlp_map, "up_proj", &block.mlp.up);
    insert_trainable_ref(&mut mlp_map, "down_proj", &block.mlp.down);

    let mut map = safemlx::module::ModuleParamRef::new();
    map.insert(
        "self_attn".into(),
        safemlx::nested::NestedValue::Map(attention_map.entries),
    );
    map.insert(
        "mlp".into(),
        safemlx::nested::NestedValue::Map(mlp_map.entries),
    );
    insert_trainable_ref(&mut map, "input_layernorm", &block.input_norm);
    insert_trainable_ref(
        &mut map,
        "post_attention_layernorm",
        &block.post_attention_norm,
    );
    map
}

fn block_frozen_states(block: &SharedTransformerBlock) -> impl Iterator<Item = Option<bool>> + '_ {
    [
        block.self_attention.query.all_frozen(),
        block.self_attention.key.all_frozen(),
        block.self_attention.value.all_frozen(),
        block.self_attention.output.all_frozen(),
        block.self_attention.rotary.all_frozen(),
        block.mlp.gate.all_frozen(),
        block.mlp.up.all_frozen(),
        block.mlp.down.all_frozen(),
        block.input_norm.all_frozen(),
        block.post_attention_norm.all_frozen(),
    ]
    .into_iter()
}

fn freeze_block(block: &mut SharedTransformerBlock, mode: bool, recursive: bool) {
    let attention = &mut block.self_attention;
    let mlp = &mut block.mlp;
    if mode {
        attention.query.freeze_parameters(recursive);
        attention.key.freeze_parameters(recursive);
        attention.value.freeze_parameters(recursive);
        attention.output.freeze_parameters(recursive);
        attention.rotary.freeze_parameters(recursive);
        mlp.gate.freeze_parameters(recursive);
        mlp.up.freeze_parameters(recursive);
        mlp.down.freeze_parameters(recursive);
        block.input_norm.freeze_parameters(recursive);
        block.post_attention_norm.freeze_parameters(recursive);
    } else {
        attention.query.unfreeze_parameters(recursive);
        attention.key.unfreeze_parameters(recursive);
        attention.value.unfreeze_parameters(recursive);
        attention.output.unfreeze_parameters(recursive);
        attention.rotary.unfreeze_parameters(recursive);
        mlp.gate.unfreeze_parameters(recursive);
        mlp.up.unfreeze_parameters(recursive);
        mlp.down.unfreeze_parameters(recursive);
        block.input_norm.unfreeze_parameters(recursive);
        block.post_attention_norm.unfreeze_parameters(recursive);
    }
}

impl safemlx::module::ModuleParameters for TransformerBlock {
    fn num_parameters(&self) -> usize {
        self.parameters().flatten().len()
    }
    fn parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        block_parameters(&self.inner)
    }
    fn parameters_mut(&mut self) -> safemlx::module::ModuleParamMut<'_> {
        block_parameters_mut(&mut self.inner)
    }
    fn trainable_parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        block_trainable_parameters(&self.inner)
    }
    fn freeze_parameters(&mut self, recursive: bool) {
        freeze_block(&mut self.inner, true, recursive);
    }
    fn unfreeze_parameters(&mut self, recursive: bool) {
        freeze_block(&mut self.inner, false, recursive);
    }
    fn all_frozen(&self) -> Option<bool> {
        let mut states = block_frozen_states(&self.inner).flatten().peekable();
        states.peek()?;
        Some(states.all(|frozen| frozen))
    }
    fn any_frozen(&self) -> Option<bool> {
        let mut states = block_frozen_states(&self.inner).flatten().peekable();
        states.peek()?;
        Some(states.any(|frozen| frozen))
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
        self.parameters().flatten().len()
    }
    fn parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        let mut model = safemlx::module::ModuleParamRef::new();
        insert_module_ref(&mut model, "embed_tokens", &self.inner.decoder.embeddings);
        let mut layers = safemlx::module::ModuleParamRef::new();
        for (index, layer) in self.inner.decoder.layers.iter().enumerate() {
            layers.insert(
                index.to_string().into(),
                safemlx::nested::NestedValue::Map(block_parameters(layer).entries),
            );
        }
        model.insert(
            "layers".into(),
            safemlx::nested::NestedValue::Map(layers.entries),
        );
        insert_module_ref(&mut model, "norm", &self.inner.decoder.norm);
        let mut root = safemlx::module::ModuleParamRef::new();
        root.insert(
            "model".into(),
            safemlx::nested::NestedValue::Map(model.entries),
        );
        if let Some(head) = &self.inner.lm_head {
            insert_module_ref(&mut root, "lm_head", head);
        }
        root
    }
    fn parameters_mut(&mut self) -> safemlx::module::ModuleParamMut<'_> {
        let mut model = safemlx::module::ModuleParamMut::new();
        insert_module_mut(
            &mut model,
            "embed_tokens",
            &mut self.inner.decoder.embeddings,
        );
        let mut layers = safemlx::module::ModuleParamMut::new();
        for (index, layer) in self.inner.decoder.layers.iter_mut().enumerate() {
            layers.insert(
                index.to_string().into(),
                safemlx::nested::NestedValue::Map(block_parameters_mut(layer).entries),
            );
        }
        model.insert(
            "layers".into(),
            safemlx::nested::NestedValue::Map(layers.entries),
        );
        insert_module_mut(&mut model, "norm", &mut self.inner.decoder.norm);
        let mut root = safemlx::module::ModuleParamMut::new();
        root.insert(
            "model".into(),
            safemlx::nested::NestedValue::Map(model.entries),
        );
        if let Some(head) = &mut self.inner.lm_head {
            insert_module_mut(&mut root, "lm_head", head);
        }
        root
    }
    fn trainable_parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        let mut model = safemlx::module::ModuleParamRef::new();
        insert_trainable_ref(&mut model, "embed_tokens", &self.inner.decoder.embeddings);
        let mut layers = safemlx::module::ModuleParamRef::new();
        for (index, layer) in self.inner.decoder.layers.iter().enumerate() {
            layers.insert(
                index.to_string().into(),
                safemlx::nested::NestedValue::Map(block_trainable_parameters(layer).entries),
            );
        }
        model.insert(
            "layers".into(),
            safemlx::nested::NestedValue::Map(layers.entries),
        );
        insert_trainable_ref(&mut model, "norm", &self.inner.decoder.norm);
        let mut root = safemlx::module::ModuleParamRef::new();
        root.insert(
            "model".into(),
            safemlx::nested::NestedValue::Map(model.entries),
        );
        if let Some(head) = &self.inner.lm_head {
            insert_trainable_ref(&mut root, "lm_head", head);
        }
        root
    }
    fn freeze_parameters(&mut self, recursive: bool) {
        self.inner.decoder.embeddings.freeze_parameters(recursive);
        for layer in &mut self.inner.decoder.layers {
            freeze_block(layer, true, recursive);
        }
        self.inner.decoder.norm.freeze_parameters(recursive);
        if let Some(head) = &mut self.inner.lm_head {
            head.freeze_parameters(recursive);
        }
    }
    fn unfreeze_parameters(&mut self, recursive: bool) {
        self.inner.decoder.embeddings.unfreeze_parameters(recursive);
        for layer in &mut self.inner.decoder.layers {
            freeze_block(layer, false, recursive);
        }
        self.inner.decoder.norm.unfreeze_parameters(recursive);
        if let Some(head) = &mut self.inner.lm_head {
            head.unfreeze_parameters(recursive);
        }
    }
    fn all_frozen(&self) -> Option<bool> {
        let mut states = std::iter::once(self.inner.decoder.embeddings.all_frozen())
            .chain(
                self.inner
                    .decoder
                    .layers
                    .iter()
                    .flat_map(block_frozen_states),
            )
            .chain(std::iter::once(self.inner.decoder.norm.all_frozen()))
            .chain(self.inner.lm_head.iter().map(ModuleParameters::all_frozen))
            .flatten()
            .peekable();
        states.peek()?;
        Some(states.all(|frozen| frozen))
    }
    fn any_frozen(&self) -> Option<bool> {
        let mut states = std::iter::once(self.inner.decoder.embeddings.any_frozen())
            .chain(self.inner.decoder.layers.iter().flat_map(|block| {
                [
                    block.self_attention.query.any_frozen(),
                    block.self_attention.key.any_frozen(),
                    block.self_attention.value.any_frozen(),
                    block.self_attention.output.any_frozen(),
                    block.self_attention.rotary.any_frozen(),
                    block.mlp.gate.any_frozen(),
                    block.mlp.up.any_frozen(),
                    block.mlp.down.any_frozen(),
                    block.input_norm.any_frozen(),
                    block.post_attention_norm.any_frozen(),
                ]
            }))
            .chain(std::iter::once(self.inner.decoder.norm.any_frozen()))
            .chain(self.inner.lm_head.iter().map(ModuleParameters::any_frozen))
            .flatten()
            .peekable();
        states.peek()?;
        Some(states.any(|frozen| frozen))
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

impl<C> CausalLm<Vec<Option<C>>> for ResidentModel
where
    C: KeyValueCache + eredu_nn::AttentionCache<Array>,
{
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

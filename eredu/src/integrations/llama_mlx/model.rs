//! MLX configuration, checkpoint integration, and resident binding for shared Llama.

use eredu_checkpoint::WeightQuantization;

use std::{collections::HashMap, path::Path};

use safemlx::{
    error::Exception,
    ops::{GgufCheckpoint, GgufMetadataValue},
    Array, Stream,
};
use serde_json::Value;

use eredu_architectures::llama::ModelArgs;

use crate::{
    backend::mlx::error::Error,
    backend::mlx::runtime::checkpoint::load::{gguf_quantization_configs, GgufTensorNames},
};

/// MLX specialization of the backend-neutral Llama block input.
pub type AttentionInput<'a, C> = eredu_architectures::llama::AttentionInput<'a, Array, C>;

/// MLX specialization of one backend-neutral Llama decoder block.
pub type TransformerBlock = crate::backend::mlx::nn::shared::MlxModule<
    eredu_architectures::llama::TransformerBlock<crate::backend::mlx::nn::shared::MlxBackend>,
>;

impl
    crate::backend::mlx::nn::shared::MlxModule<
        eredu_architectures::llama::TransformerBlock<crate::backend::mlx::nn::shared::MlxBackend>,
    >
{
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
        .map(Self::new)
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

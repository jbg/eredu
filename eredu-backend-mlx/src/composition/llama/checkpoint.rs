//! MLX checkpoint adaptation for the backend-neutral Llama architecture.
//!
//! The architecture crate owns tensor geometry, canonical plans, and name
//! translation. This module applies those contracts to concrete MLX
//! SafeTensors and GGUF sources during cold-path validation and loading.

use std::collections::HashMap;

use eredu_architectures::llama::ModelArgs;
use eredu_checkpoint::WeightQuantization;
use safemlx::ops::GgufMetadataValue;
use safemlx::Stream;

use crate::backend::error::Error;
use crate::backend::runtime::checkpoint::load::{gguf_quantization_configs, GgufTensorNames};

pub(crate) struct PreparedLlamaGguf {
    pub args: ModelArgs,
}

pub(crate) fn prepare_llama_gguf_checkpoint(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    quantization: Option<WeightQuantization>,
    _weights_stream: &Stream,
) -> Result<PreparedLlamaGguf, Error> {
    if !matches!(
        source.architecture(),
        eredu_architectures::GgufArchitecture::Llama
            | eredu_architectures::GgufArchitecture::Mistral
    ) {
        return Err(Error::ArchitectureModel(format!(
            "Llama GGUF loader received architecture {:?}",
            source.architecture()
        )));
    }
    let checkpoint = source.checkpoint();
    let metadata = source.metadata();
    let mut args = model_args_from_gguf_catalog(checkpoint, metadata)?;
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

    Ok(PreparedLlamaGguf { args })
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
pub fn model_args_from_gguf_catalog(
    arrays: &(impl GgufTensorNames + ?Sized),
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<ModelArgs, Error> {
    eredu_architectures::llama::model_args_from_gguf_catalog(&NeutralGgufCatalog(arrays), metadata)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
}

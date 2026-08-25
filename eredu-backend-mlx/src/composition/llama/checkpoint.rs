//! MLX checkpoint adaptation for the backend-neutral Llama architecture.
//!
//! The architecture crate owns tensor geometry, canonical plans, and name
//! translation. This module applies those contracts to concrete MLX
//! SafeTensors and GGUF sources during cold-path validation and loading.

use eredu_architectures::llama::ModelArgs;
use eredu_checkpoint::WeightQuantization;
use safemlx::Stream;

use crate::backend::error::Error;
use crate::backend::runtime::checkpoint::load::gguf_quantization_configs;

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
    let eredu_architectures::configuration::GgufModelConfig::Llama(args) = source.model() else {
        return Err(Error::ArchitectureModel(
            "Llama GGUF loader received a different prepared model".into(),
        ));
    };
    let mut args = args.clone();
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

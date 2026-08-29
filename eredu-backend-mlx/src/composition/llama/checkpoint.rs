//! MLX checkpoint adaptation for the backend-neutral Llama architecture.
//!
//! The architecture crate owns tensor geometry, canonical plans, and name
//! translation. This module applies those contracts to concrete MLX
//! SafeTensors and GGUF sources during cold-path validation and loading.

use eredu_architectures::llama::ModelArgs;

use crate::backend::error::Error;
use crate::backend::runtime::checkpoint::load::gguf_quantization_configs;

pub(crate) struct PreparedLlamaGguf {
    pub args: ModelArgs,
}

pub(crate) fn prepare_llama_gguf_checkpoint(
    source: &crate::composition::mlx::structural::AdmittedGguf,
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
    let quantized_weight_configs =
        gguf_quantization_configs(checkpoint, source.plan().tensor_mapping())?;
    let args = eredu_architectures::llama::with_checkpoint_formats(args, quantized_weight_configs)
        .map_err(Error::ArchitectureModel)?;

    Ok(PreparedLlamaGguf { args })
}

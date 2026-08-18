//! Model-family detection and model-family-independent MLX load options.

use super::*;
use crate::runtime::execution::layerwise::WeightResidency;
pub use safemlx_lm_core::artifact::ModelKind;

/// Model-family-independent options for loading model weights with MLX.
///
/// When `quantization` is set for a dense checkpoint, eligible parameters are
/// quantized and materialized one tensor at a time. Checkpoints already
/// carrying matching metadata are loaded directly without requantizing.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct ModelLoadOptions {
    /// Optional MLX weight encoding requested during dense checkpoint loading.
    pub quantization: Option<WeightQuantization>,
    /// Optional validated runtime topology and process-local device assignment.
    ///
    /// Singleton topologies preserve replicated model loading. Non-replicated
    /// topologies select a rank-local executable behind [`crate::MlxModel`].
    pub parallel: Option<MlxParallelContext>,
    /// Parameter placement and execution policy for cataloged checkpoint stores.
    pub weight_residency: WeightResidency,
}

impl ModelLoadOptions {
    /// Creates load options that quantize eligible dense weights on load.
    pub fn with_quantization(quantization: impl Into<WeightQuantization>) -> Self {
        Self {
            quantization: Some(quantization.into()),
            parallel: None,
            weight_residency: WeightResidency::fully_resident(),
        }
    }

    /// Adds a validated runtime parallel topology to these options.
    pub fn with_parallel_topology(mut self, topology: MlxParallelContext) -> Self {
        self.parallel = Some(topology);
        self
    }

    /// Creates load options for a validated runtime parallel topology.
    pub fn with_parallel(topology: MlxParallelContext) -> Self {
        Self::default().with_parallel_topology(topology)
    }

    /// Selects fully resident or bounded layer execution for checkpoint weights.
    pub fn with_weight_residency(mut self, residency: WeightResidency) -> Self {
        self.weight_residency = residency;
        self
    }

    pub(crate) fn preparation_policy(self) -> Result<safemlx_lm_core::PreparationPolicy, Error> {
        use crate::runtime::{
            checkpoint::quantization::WeightQuantization,
            execution::layerwise::LayerWeightResidency,
        };
        use safemlx_lm_core::{QuantizationRequest, ResidencyRequest};

        if let Some(quantization) = self.quantization {
            quantization.validate()?;
        }
        let quantization = match self.quantization {
            Some(WeightQuantization::Affine(config)) => Some(QuantizationRequest::Affine {
                group_size: u32::try_from(config.group_size).map_err(|_| {
                    Error::Quantization(format!(
                        "group_size must be non-negative, got {}",
                        config.group_size
                    ))
                })?,
                bits: u8::try_from(config.bits).map_err(|_| {
                    Error::Quantization(format!("bits must fit in u8, got {}", config.bits))
                })?,
            }),
            Some(WeightQuantization::MxFp4) => Some(QuantizationRequest::MxFp4),
            Some(WeightQuantization::GgufIQuant { .. }) | None => None,
        };
        let residency = if self.weight_residency.expert_cache().is_some() {
            ResidencyRequest::ExpertCache
        } else {
            match self.weight_residency.layers() {
                LayerWeightResidency::FullyResident => ResidencyRequest::FullyResident,
                LayerWeightResidency::LayerwiseHost(_) => ResidencyRequest::LayerwiseHost,
                LayerWeightResidency::DenseDiskStream(_) => ResidencyRequest::DenseDiskStream,
            }
        };
        Ok(safemlx_lm_core::PreparationPolicy {
            quantization,
            residency,
            distributed: self
                .parallel
                .is_some_and(|topology| !topology.is_replicated()),
        })
    }

    pub(crate) fn validate_preparation(
        self,
        kind: ModelKind,
        gguf_architecture: Option<safemlx_lm_core::GgufArchitecture>,
        format: safemlx_lm_core::ArtifactFormat,
    ) -> Result<safemlx_lm_core::MaterializationRoute, Error> {
        Ok(safemlx_lm_core::validate_preparation_policy(
            kind,
            gguf_architecture,
            format,
            self.preparation_policy()?,
        )?)
    }
}

pub(crate) fn ensure_replicated_load_options(options: ModelLoadOptions) -> Result<(), Error> {
    if options
        .parallel
        .is_some_and(|topology| !topology.is_replicated())
    {
        return Err(Error::Parallel(
            "this facade owns replicated runtime state; use load_model_with_options and create an MlxModelSession for distributed execution"
                .into(),
        ));
    }
    Ok(())
}

/// Canonical resolution of a validated model config.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct ResolvedModelConfig {
    pub(crate) kind: ModelKind,
    pub(crate) model_type: String,
    pub(crate) effective_model_type: String,
}

#[derive(Debug)]
pub(crate) enum ModelConfigResolutionError {
    InvalidMetadata(serde_json::Error),
    Loader(Error),
}

impl std::fmt::Display for ModelConfigResolutionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMetadata(error) => {
                write!(formatter, "invalid model config metadata: {error}")
            }
            Self::Loader(error) => error.fmt(formatter),
        }
    }
}

pub(crate) fn resolve_model_config(
    config: &Value,
) -> Result<ResolvedModelConfig, ModelConfigResolutionError> {
    let metadata = serde_json::from_value::<ModelMetadata>(config.clone())
        .map_err(ModelConfigResolutionError::InvalidMetadata)?;
    let effective_model_type = effective_model_type(&metadata);
    let kind = ModelKind::from_model_type(&effective_model_type).map_err(|_| {
        ModelConfigResolutionError::Loader(Error::UnsupportedModelType(
            effective_model_type.clone(),
        ))
    })?;
    validate_model_config(kind, config).map_err(ModelConfigResolutionError::Loader)?;
    Ok(ResolvedModelConfig {
        kind,
        model_type: metadata.model_type,
        effective_model_type,
    })
}

fn validate_model_config(kind: ModelKind, config: &Value) -> Result<(), Error> {
    match kind {
        ModelKind::DeepSeekV3 => deepseek_v3::validate_model_config_value(config),
        ModelKind::DeepSeekV4 => deepseek_v4::validate_model_config_value(config),
        ModelKind::Gemma4 => gemma4::validate_model_config_value(config),
        ModelKind::GptOss => gpt_oss::validate_model_config_value(config),
        ModelKind::Inkling => inkling::validate_model_config_value(config),
        ModelKind::KimiLinear => kimi_linear::validate_model_config_value(config),
        ModelKind::Llama => llama::validate_model_config_value(config),
        ModelKind::MuseGlimmer => muse_glimmer::validate_model_config_value(config),
        ModelKind::Lfm2 => lfm2::validate_model_config_value(config),
        ModelKind::NemotronH => nemotron_h::validate_model_config_value(config),
        ModelKind::PersonaPlex => personaplex::validate_model_config_value(config),
        ModelKind::Qwen2 => dense_qwen::config_from_hf_value(config).map(|_| ()),
        ModelKind::Qwen3 => dense_qwen::config_from_hf_value(config).map(|_| ()),
        ModelKind::Qwen3Next => qwen3_next::validate_model_config_value(config),
        ModelKind::Qwen3Vl => qwen3_vl::validate_model_config_value(config),
        ModelKind::Qwen3VlMoe => qwen3_vl_moe::validate_model_config_value(config),
        ModelKind::Qwen35 => qwen3_5::validate_model_config_value(config),
    }
}

#[cfg(test)]
mod tests {
    use super::{ModelKind, ModelLoadOptions};
    use crate::{
        runtime::checkpoint::quantization::WeightQuantization, LayerwiseLoadOptions,
        WeightResidency,
    };

    #[test]
    fn deepseek_v4_quantization_composes_with_nonresident_layers() {
        let options = ModelLoadOptions::with_quantization(WeightQuantization::MxFp4)
            .with_weight_residency(WeightResidency::layerwise_host(
                LayerwiseLoadOptions::default(),
            ));
        options
            .validate_preparation(
                ModelKind::DeepSeekV4,
                None,
                safemlx_lm_core::ArtifactFormat::SafeTensors,
            )
            .unwrap();
    }
}

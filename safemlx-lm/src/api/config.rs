//! Portable model-family detection and configuration validation.

use super::*;
pub use safemlx_lm_core::artifact::ModelKind;

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

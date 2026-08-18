//! MLX architecture validation for portable model configurations.

use serde_json::Value;

use crate::architectures::{
    deepseek_v3::model as deepseek_v3,
    deepseek_v4::model as deepseek_v4,
    gemma4::model as gemma4,
    gpt_oss::model as gpt_oss,
    inkling::model as inkling,
    kimi_linear::model as kimi_linear,
    lfm2::model as lfm2,
    llama::model as llama,
    moshi::personaplex,
    muse_glimmer,
    nemotron_h::model as nemotron_h,
    qwen::{
        dense as dense_qwen,
        hybrid::{qwen3_5, qwen3_next},
        vl::{model as qwen3_vl, moe as qwen3_vl_moe},
    },
};
use crate::backend::mlx::error::Error;

/// Canonical resolution of a model configuration supported by MLX.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct ResolvedModelConfig {
    pub(crate) kind: safemlx_lm_core::ModelKind,
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
    let resolved =
        safemlx_lm_core::resolve_model_configuration(config).map_err(|error| match error {
            safemlx_lm_core::artifact::ArtifactError::Json(error) => {
                ModelConfigResolutionError::InvalidMetadata(error)
            }
            safemlx_lm_core::artifact::ArtifactError::UnsupportedModelType(model_type) => {
                ModelConfigResolutionError::Loader(Error::UnsupportedModelType(model_type))
            }
            error => ModelConfigResolutionError::Loader(Error::Artifact(error)),
        })?;
    validate_model_config(resolved.kind, config).map_err(ModelConfigResolutionError::Loader)?;
    Ok(ResolvedModelConfig {
        kind: resolved.kind,
        model_type: resolved.declared_model_type,
        effective_model_type: resolved.effective_model_type,
    })
}

fn validate_model_config(kind: safemlx_lm_core::ModelKind, config: &Value) -> Result<(), Error> {
    use safemlx_lm_core::ModelKind;

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
        ModelKind::Qwen2 | ModelKind::Qwen3 => dense_qwen::config_from_hf_value(config).map(|_| ()),
        ModelKind::Qwen3Next => qwen3_next::validate_model_config_value(config),
        ModelKind::Qwen3Vl => qwen3_vl::validate_model_config_value(config),
        ModelKind::Qwen3VlMoe => qwen3_vl_moe::validate_model_config_value(config),
        ModelKind::Qwen35 => qwen3_5::validate_model_config_value(config),
    }
}

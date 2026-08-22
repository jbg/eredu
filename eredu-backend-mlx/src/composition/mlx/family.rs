//! MLX architecture validation for portable model configurations.

use serde_json::Value;

use crate::backend::mlx::error::Error;

/// Canonical resolution of a model configuration supported by MLX.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ResolvedModelConfig {
    pub kind: eredu_core::ModelKind,
    pub model_type: String,
    pub effective_model_type: String,
}

#[derive(Debug)]
pub enum ModelConfigResolutionError {
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

pub fn resolve_model_config(
    config: &Value,
) -> Result<ResolvedModelConfig, ModelConfigResolutionError> {
    if let Some(resolved) = resolve_moshi_config(config)? {
        return Ok(resolved);
    }
    let resolved =
        eredu_core::resolve_model_configuration(config).map_err(|error| match error {
            eredu_core::artifact::ArtifactError::Json(error) => {
                ModelConfigResolutionError::InvalidMetadata(error)
            }
            eredu_core::artifact::ArtifactError::UnsupportedModelType(model_type) => {
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

fn resolve_moshi_config(
    config: &Value,
) -> Result<Option<ResolvedModelConfig>, ModelConfigResolutionError> {
    match eredu_architectures::moshi::MoshiConfig::from_config_value(Some(config)) {
        Ok(config) => {
            let effective = config.effective_model_type().as_str().to_string();
            Ok(Some(ResolvedModelConfig {
                kind: eredu_core::ModelKind::Moshi,
                model_type: effective.clone(),
                effective_model_type: effective,
            }))
        }
        Err(error) => {
            let explicitly_moshi = config
                .get("model_type")
                .and_then(Value::as_str)
                .is_some_and(|value| value == "moshi" || value == "personaplex");
            if explicitly_moshi {
                Err(ModelConfigResolutionError::Loader(
                    Error::UnsupportedArchitecture(error.to_string()),
                ))
            } else {
                Ok(None)
            }
        }
    }
}

fn validate_model_config(kind: eredu_core::ModelKind, config: &Value) -> Result<(), Error> {
    use eredu_core::ModelKind;

    match kind {
        ModelKind::DeepSeekV3 => eredu_architectures::deepseek::parse_v3_config(config)
            .map(|_| ())
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string())),
        ModelKind::DeepSeekV4 => eredu_architectures::deepseek::parse_v4_config(config)
            .map(|_| ())
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string())),
        ModelKind::Gemma4 => serde_json::to_vec(config)
            .map_err(Error::from)
            .and_then(|bytes| {
                eredu_architectures::gemma4::FamilyConfig::from_hf_json(&bytes)
                    .map(|_| ())
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
            }),
        ModelKind::GptOss => eredu_architectures::gpt_oss::model_args_from_config_value(config)
            .map(|_| ())
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string())),
        ModelKind::Inkling => serde_json::to_vec(config)
            .map_err(Error::from)
            .and_then(|bytes| {
                eredu_architectures::inkling::ModelArgs::from_hf_json(&bytes)
                    .map(|_| ())
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
            }),
        ModelKind::KimiLinear => {
            eredu_architectures::kimi_linear::model_args_from_config_value(config)
                .map(|_| ())
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        }
        ModelKind::Llama => eredu_architectures::llama::model_args_from_config_value(config)
            .map(|_| ())
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string())),
        ModelKind::MuseGlimmer => {
            serde_json::to_vec(config)
                .map_err(Error::from)
                .and_then(|bytes| {
                    eredu_architectures::muse_glimmer::DecoderConfig::from_hf_json(&bytes)
                        .map(|_| ())
                        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
                })
        }
        ModelKind::Lfm2 => eredu_architectures::lfm2::model_args_from_config_value(config)
            .map(|_| ())
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string())),
        ModelKind::NemotronH => {
            eredu_architectures::nemotron_h::model_args_from_config_value(config)
                .map(|_| ())
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        }
        ModelKind::Moshi => {
            eredu_architectures::moshi::MoshiConfig::from_config_value(Some(config))
                .map(|_| ())
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        }
        ModelKind::Qwen2 | ModelKind::Qwen3 => {
            eredu_architectures::qwen::model_args_from_config_value(config)
                .map(|_| ())
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        }
        ModelKind::Qwen3Next | ModelKind::Qwen35 => {
            eredu_architectures::qwen::hybrid::model_args_from_config_value(config)
                .map(|_| ())
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        }
        ModelKind::Qwen3Vl | ModelKind::Qwen3VlMoe => {
            eredu_architectures::qwen::vl::model_args_from_config_value(config)
                .map(|_| ())
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        }
    }
}

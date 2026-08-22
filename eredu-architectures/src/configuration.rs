//! Authoritative Hugging Face family identity and configuration validation.

use eredu_core::{
    artifact::ArtifactError, ModelConfiguration, ModelConfigurationResolver, ModelKind,
};
use serde::Deserialize;
use serde_json::Value;
use std::path::Path;

/// Stateless registry for every architecture family implemented by this crate.
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelConfigurations;

/// Shared architecture registry used by facade and backend adapters.
pub static MODEL_CONFIGURATIONS: ModelConfigurations = ModelConfigurations;

impl ModelConfigurationResolver for ModelConfigurations {
    fn resolve(&self, json: &Value) -> Result<ModelConfiguration, ArtifactError> {
        resolve_model_configuration(json)
    }
}

#[derive(Deserialize)]
struct ConfigMetadata {
    model_type: String,
    #[serde(default)]
    text_config: Option<TextConfigMetadata>,
}

#[derive(Deserialize)]
struct TextConfigMetadata {
    #[serde(default)]
    model_type: Option<String>,
}

fn model_kind(model_type: &str) -> Result<ModelKind, ArtifactError> {
    match model_type {
        "deepseek_v3" => Ok(ModelKind::DeepSeekV3),
        "deepseek_v4" => Ok(ModelKind::DeepSeekV4),
        "gemma4" | "gemma4_text" | "gemma4_unified" | "gemma4_unified_text" => {
            Ok(ModelKind::Gemma4)
        }
        "gpt_oss" => Ok(ModelKind::GptOss),
        "inkling_mm_model" => Ok(ModelKind::Inkling),
        "kimi_linear" => Ok(ModelKind::KimiLinear),
        "llama" | "mistral" => Ok(ModelKind::Llama),
        "muse_glimmer" | "muse_glimmer_text" => Ok(ModelKind::MuseGlimmer),
        "lfm2" | "lfm2_moe" => Ok(ModelKind::Lfm2),
        "nemotron_h" => Ok(ModelKind::NemotronH),
        "moshi" | "personaplex" => Ok(ModelKind::Moshi),
        "qwen2" => Ok(ModelKind::Qwen2),
        "qwen3" | "qwen3_moe" => Ok(ModelKind::Qwen3),
        "qwen3_next" => Ok(ModelKind::Qwen3Next),
        "qwen3_vl" | "qwen3_vl_text" => Ok(ModelKind::Qwen3Vl),
        "qwen3_vl_moe" | "qwen3_vl_moe_text" => Ok(ModelKind::Qwen3VlMoe),
        "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" => Ok(ModelKind::Qwen35),
        other => Err(ArtifactError::UnsupportedModelType(other.into())),
    }
}

fn effective_model_type(metadata: &ConfigMetadata) -> String {
    if metadata.model_type == "inkling_mm_model" {
        return metadata.model_type.clone();
    }
    if matches!(
        metadata.model_type.as_str(),
        "gemma4" | "gemma4_unified" | "qwen3_vl" | "qwen3_vl_moe" | "qwen3_5" | "qwen3_5_moe"
    ) {
        metadata
            .text_config
            .as_ref()
            .and_then(|text| text.model_type.clone())
            .unwrap_or_else(|| metadata.model_type.clone())
    } else if model_kind(&metadata.model_type).is_ok() {
        metadata.model_type.clone()
    } else {
        metadata
            .text_config
            .as_ref()
            .and_then(|text| text.model_type.clone())
            .unwrap_or_else(|| metadata.model_type.clone())
    }
}

/// Resolves the canonical family and nested text identity of a Hugging Face config.
pub fn resolve_model_configuration(json: &Value) -> Result<ModelConfiguration, ArtifactError> {
    let metadata: ConfigMetadata = serde_json::from_value(json.clone())?;
    let effective_model_type = effective_model_type(&metadata);
    let kind = model_kind(&effective_model_type)?;
    Ok(ModelConfiguration {
        declared_model_type: metadata.model_type,
        effective_model_type,
        kind,
        json: Some(json.clone()),
        gguf_architecture: None,
    })
}

fn invalid_configuration(kind: ModelKind, error: impl std::fmt::Display) -> ArtifactError {
    ArtifactError::InvalidArtifact(format!(
        "invalid {} configuration: {error}",
        kind.model_type_name()
    ))
}

/// Validates a resolved Hugging Face config with its architecture-owned parser.
pub fn validate_model_configuration(
    configuration: &ModelConfiguration,
) -> Result<(), ArtifactError> {
    let json = configuration.json.as_ref().ok_or_else(|| {
        ArtifactError::InvalidArtifact(
            "Hugging Face configuration validation requires raw JSON".into(),
        )
    })?;
    let kind = configuration.kind;
    match kind {
        ModelKind::DeepSeekV3 => crate::deepseek::parse_v3_config(json)
            .map(|_| ())
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::DeepSeekV4 => crate::deepseek::parse_v4_config(json)
            .map(|_| ())
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::Gemma4 => serde_json::to_vec(json)
            .map_err(ArtifactError::from)
            .and_then(|bytes| {
                crate::gemma4::FamilyConfig::from_hf_json(&bytes)
                    .map(|_| ())
                    .map_err(|error| invalid_configuration(kind, error))
            }),
        ModelKind::GptOss => crate::gpt_oss::model_args_from_config_value(json)
            .map(|_| ())
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::Inkling => serde_json::to_vec(json)
            .map_err(ArtifactError::from)
            .and_then(|bytes| {
                crate::inkling::ModelArgs::from_hf_json(&bytes)
                    .map(|_| ())
                    .map_err(|error| invalid_configuration(kind, error))
            }),
        ModelKind::KimiLinear => crate::kimi_linear::model_args_from_config_value(json)
            .map(|_| ())
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::Llama => crate::llama::model_args_from_config_value(json)
            .map(|_| ())
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::MuseGlimmer => serde_json::to_vec(json)
            .map_err(ArtifactError::from)
            .and_then(|bytes| {
                crate::muse_glimmer::DecoderConfig::from_hf_json(&bytes)
                    .map(|_| ())
                    .map_err(|error| invalid_configuration(kind, error))
            }),
        ModelKind::Lfm2 => crate::lfm2::model_args_from_config_value(json)
            .map(|_| ())
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::NemotronH => crate::nemotron_h::model_args_from_config_value(json)
            .map(|_| ())
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::Moshi => crate::moshi::MoshiConfig::from_config_value(Some(json))
            .map(|_| ())
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::Qwen2 | ModelKind::Qwen3 => crate::qwen::model_args_from_config_value(json)
            .map(|_| ())
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::Qwen3Next | ModelKind::Qwen35 => {
            crate::qwen::hybrid::model_args_from_config_value(json)
                .map(|_| ())
                .map_err(|error| invalid_configuration(kind, error))
        }
        ModelKind::Qwen3Vl | ModelKind::Qwen3VlMoe => {
            crate::qwen::vl::model_args_from_config_value(json)
                .map(|_| ())
                .map_err(|error| invalid_configuration(kind, error))
        }
    }
}

/// Resolves and validates one Hugging Face config through the authoritative registry.
pub fn resolve_and_validate_model_configuration(
    json: &Value,
) -> Result<ModelConfiguration, ArtifactError> {
    let configuration = resolve_model_configuration(json)?;
    validate_model_configuration(&configuration)?;
    Ok(configuration)
}

/// Validated family identity consumed by backend composition and inspection.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ResolvedModelConfig {
    /// Canonical architecture family.
    pub kind: ModelKind,
    /// Declared outer model type.
    pub model_type: String,
    /// Effective nested text model type.
    pub effective_model_type: String,
}

/// Resolves and validates one config into its architecture-owned dispatch identity.
pub fn resolve_model_config(json: &Value) -> Result<ResolvedModelConfig, ArtifactError> {
    let configuration = resolve_and_validate_model_configuration(json)?;
    Ok(ResolvedModelConfig {
        kind: configuration.kind,
        model_type: configuration.declared_model_type,
        effective_model_type: configuration.effective_model_type,
    })
}

/// Inspects an artifact using this crate's authoritative family registry.
pub fn inspect_artifact(
    path: impl AsRef<Path>,
) -> Result<eredu_core::ArtifactInspection, ArtifactError> {
    eredu_core::inspect_artifact(path, &MODEL_CONFIGURATIONS)
}

/// Supported architecture-owned external drafter families.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AssistantModelKind {
    /// Gemma 4 draft architecture.
    Gemma4,
    /// Muse-Glimmer DFlash draft architecture.
    MuseGlimmer,
}

/// Resolves the model kind of an external drafter config.
pub fn resolve_assistant_model_kind(json: &Value) -> Result<AssistantModelKind, ArtifactError> {
    match json.get("model_type").and_then(Value::as_str) {
        Some("gemma4_assistant") => Ok(AssistantModelKind::Gemma4),
        Some("muse_glimmer_assistant") => Ok(AssistantModelKind::MuseGlimmer),
        Some(other) => Err(ArtifactError::UnsupportedModelType(other.into())),
        None => Err(ArtifactError::InvalidArtifact(
            "assistant config is missing model_type".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_wrappers_and_aliases_resolve_in_one_registry() {
        let json = serde_json::json!({
            "model_type": "qwen3_5",
            "text_config": { "model_type": "qwen3_5_moe" }
        });
        let resolved = resolve_model_configuration(&json).unwrap();
        assert_eq!(resolved.kind, ModelKind::Qwen35);
        assert_eq!(resolved.declared_model_type, "qwen3_5");
        assert_eq!(resolved.effective_model_type, "qwen3_5_moe");

        for model_type in ["moshi", "personaplex"] {
            let resolved =
                resolve_model_configuration(&serde_json::json!({"model_type": model_type}))
                    .unwrap();
            assert_eq!(resolved.kind, ModelKind::Moshi);
        }
    }

    #[test]
    fn unknown_wrapper_can_delegate_to_a_known_nested_text_family() {
        let resolved = resolve_model_configuration(&serde_json::json!({
            "model_type": "third_party_wrapper",
            "text_config": { "model_type": "qwen3_vl_moe_text" }
        }))
        .unwrap();
        assert_eq!(resolved.kind, ModelKind::Qwen3VlMoe);
        assert_eq!(resolved.effective_model_type, "qwen3_vl_moe_text");
    }
}

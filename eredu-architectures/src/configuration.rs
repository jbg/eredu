//! Authoritative Hugging Face and GGUF family identity and configuration validation.

use eredu_core::{
    artifact::ArtifactError, ModelConfiguration, ModelConfigurationResolver, ModelKind,
};
use eredu_gguf::Checkpoint as GgufCheckpoint;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

/// Stateless registry for every architecture family implemented by this crate.
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelConfigurations;

/// Shared architecture registry used by facade and backend adapters.
pub static MODEL_CONFIGURATIONS: ModelConfigurations = ModelConfigurations;

impl ModelConfigurationResolver for ModelConfigurations {
    fn resolve_safetensors(&self, json: &Value) -> Result<ModelConfiguration, ArtifactError> {
        resolve_model_configuration(json)
    }

    fn resolve_gguf(
        &self,
        architecture: &str,
        checkpoint: &GgufCheckpoint,
    ) -> Result<ModelConfiguration, ArtifactError> {
        resolve_gguf_configuration(architecture, checkpoint)
    }
}

/// Architecture-owned GGUF family identity.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GgufArchitecture {
    /// `kimi-linear`.
    KimiLinear,
    /// `deepseek2`.
    DeepSeek2,
    /// `deepseek4`.
    DeepSeek4,
    /// `gpt-oss`.
    GptOss,
    /// `inkling`.
    Inkling,
    /// `gemma4`.
    Gemma4,
    /// `llama`.
    Llama,
    /// `mistral`.
    Mistral,
    /// `muse-glimmer`.
    MuseGlimmer,
    /// `lfm2`.
    Lfm2,
    /// `lfm2moe`.
    Lfm2Moe,
    /// `nemotron_h`.
    NemotronH,
    /// `nemotron_h_moe`.
    NemotronHMoe,
    /// `qwen2`.
    Qwen2,
    /// `qwen3`.
    Qwen3,
    /// `qwen3moe`.
    Qwen3Moe,
    /// `qwen3vl`.
    Qwen3Vl,
    /// `qwen3vlmoe`.
    Qwen3VlMoe,
    /// `qwen35`.
    Qwen35,
    /// `qwen35moe`.
    Qwen35Moe,
    /// `qwen3next`.
    Qwen3Next,
}

impl GgufArchitecture {
    /// Accepted `general.architecture` values.
    pub const SUPPORTED_NAMES: &'static str = "kimi-linear, deepseek2, deepseek4, gpt-oss, inkling, gemma4, llama, mistral, muse-glimmer, lfm2, lfm2moe, nemotron_h, nemotron_h_moe, qwen2, qwen3, qwen3moe, qwen3vl, qwen3vlmoe, qwen35, qwen35moe, and qwen3next";

    /// Resolves `general.architecture` through the architecture registry.
    pub fn resolve(name: &str) -> Result<Self, ArtifactError> {
        match name {
            "kimi-linear" => Ok(Self::KimiLinear),
            "deepseek2" => Ok(Self::DeepSeek2),
            "deepseek4" => Ok(Self::DeepSeek4),
            "gpt-oss" => Ok(Self::GptOss),
            "inkling" => Ok(Self::Inkling),
            "gemma4" => Ok(Self::Gemma4),
            "llama" => Ok(Self::Llama),
            "mistral" => Ok(Self::Mistral),
            "muse-glimmer" => Ok(Self::MuseGlimmer),
            "lfm2" => Ok(Self::Lfm2),
            "lfm2moe" => Ok(Self::Lfm2Moe),
            "nemotron_h" => Ok(Self::NemotronH),
            "nemotron_h_moe" => Ok(Self::NemotronHMoe),
            "qwen2" => Ok(Self::Qwen2),
            "qwen3" => Ok(Self::Qwen3),
            "qwen3moe" => Ok(Self::Qwen3Moe),
            "qwen3vl" => Ok(Self::Qwen3Vl),
            "qwen3vlmoe" => Ok(Self::Qwen3VlMoe),
            "qwen35" => Ok(Self::Qwen35),
            "qwen35moe" => Ok(Self::Qwen35Moe),
            "qwen3next" => Ok(Self::Qwen3Next),
            other => Err(ArtifactError::UnsupportedGgufArchitecture(other.into())),
        }
    }

    /// General model family implemented by this GGUF architecture.
    pub const fn model_kind(self) -> ModelKind {
        match self {
            Self::KimiLinear => ModelKind::KimiLinear,
            Self::DeepSeek2 => ModelKind::DeepSeekV3,
            Self::DeepSeek4 => ModelKind::DeepSeekV4,
            Self::GptOss => ModelKind::GptOss,
            Self::Inkling => ModelKind::Inkling,
            Self::Gemma4 => ModelKind::Gemma4,
            Self::Llama | Self::Mistral => ModelKind::Llama,
            Self::MuseGlimmer => ModelKind::MuseGlimmer,
            Self::Lfm2 | Self::Lfm2Moe => ModelKind::Lfm2,
            Self::NemotronH | Self::NemotronHMoe => ModelKind::NemotronH,
            Self::Qwen2 => ModelKind::Qwen2,
            Self::Qwen3 | Self::Qwen3Moe => ModelKind::Qwen3,
            Self::Qwen3Vl => ModelKind::Qwen3Vl,
            Self::Qwen3VlMoe => ModelKind::Qwen3VlMoe,
            Self::Qwen35 | Self::Qwen35Moe => ModelKind::Qwen35,
            Self::Qwen3Next => ModelKind::Qwen3Next,
        }
    }

    /// Exact metadata spelling.
    pub const fn metadata_name(self) -> &'static str {
        match self {
            Self::KimiLinear => "kimi-linear",
            Self::DeepSeek2 => "deepseek2",
            Self::DeepSeek4 => "deepseek4",
            Self::GptOss => "gpt-oss",
            Self::Inkling => "inkling",
            Self::Gemma4 => "gemma4",
            Self::Llama => "llama",
            Self::Mistral => "mistral",
            Self::MuseGlimmer => "muse-glimmer",
            Self::Lfm2 => "lfm2",
            Self::Lfm2Moe => "lfm2moe",
            Self::NemotronH => "nemotron_h",
            Self::NemotronHMoe => "nemotron_h_moe",
            Self::Qwen2 => "qwen2",
            Self::Qwen3 => "qwen3",
            Self::Qwen3Moe => "qwen3moe",
            Self::Qwen3Vl => "qwen3vl",
            Self::Qwen3VlMoe => "qwen3vlmoe",
            Self::Qwen35 => "qwen35",
            Self::Qwen35Moe => "qwen35moe",
            Self::Qwen3Next => "qwen3next",
        }
    }
}

fn resolve_gguf_configuration(
    name: &str,
    checkpoint: &GgufCheckpoint,
) -> Result<ModelConfiguration, ArtifactError> {
    let architecture = GgufArchitecture::resolve(name)?;
    validate_gguf_structure(architecture, checkpoint)?;
    Ok(ModelConfiguration {
        declared_model_type: name.into(),
        effective_model_type: name.into(),
        kind: architecture.model_kind(),
        json: None,
    })
}

fn validate_gguf_structure(
    architecture: GgufArchitecture,
    checkpoint: &GgufCheckpoint,
) -> Result<(), ArtifactError> {
    if matches!(
        architecture,
        GgufArchitecture::Qwen35 | GgufArchitecture::Qwen35Moe | GgufArchitecture::Qwen3Next
    ) && checkpoint.tensors().any(|tensor| {
        let name = tensor.descriptor().name.as_str();
        name.starts_with("v.") || name.starts_with("mm.")
    }) {
        return Err(ArtifactError::InvalidArtifact(
            "multimodal Qwen3-Next/Qwen3.5 GGUF checkpoints are not supported".into(),
        ));
    }
    Ok(())
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
    use eredu_gguf::{GgmlType, MetadataValue, TensorInput, Writer};
    use std::{collections::BTreeMap, fs::File};

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

    #[test]
    fn gguf_spellings_resolve_only_through_the_architecture_registry() {
        assert_eq!(
            GgufArchitecture::resolve("qwen2").unwrap(),
            GgufArchitecture::Qwen2
        );
        for nearby in ["qwen", "qwen2moe", "qwen2vl", "qwen2.5"] {
            assert!(matches!(
                GgufArchitecture::resolve(nearby),
                Err(ArtifactError::UnsupportedGgufArchitecture(name)) if name == nearby
            ));
        }
    }

    #[test]
    fn gguf_inspection_applies_architecture_owned_qwen_structure_policy() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("model.gguf");
        let scalar = 1.0_f32.to_le_bytes();
        let metadata = BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("qwen35".into()),
            ),
            ("qwen35.block_count".into(), MetadataValue::Uint32(1)),
            ("qwen35.embedding_length".into(), MetadataValue::Uint32(1)),
        ]);
        Writer::default()
            .write(
                File::create(&path).unwrap(),
                &metadata,
                &[
                    TensorInput {
                        name: "token_embd.weight",
                        dimensions: &[1],
                        ggml_type: GgmlType::F32,
                        data: &scalar,
                    },
                    TensorInput {
                        name: "v.patch_embd.weight",
                        dimensions: &[1],
                        ggml_type: GgmlType::F32,
                        data: &scalar,
                    },
                ],
            )
            .unwrap();

        let error = inspect_artifact(&path).unwrap_err();
        assert!(matches!(
            error,
            ArtifactError::InvalidArtifact(detail)
                if detail.contains("multimodal Qwen3-Next/Qwen3.5")
        ));
    }
}

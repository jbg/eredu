//! Authoritative Hugging Face and GGUF family identity and configuration validation.

use eredu_core::{
    artifact::ArtifactError, LoadingProtocol, ModelConfiguration, ModelConfigurationResolver,
};
use eredu_gguf::{Checkpoint as GgufCheckpoint, MetadataValue};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, path::Path};

/// Stateless registry for every architecture family implemented by this crate.
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelConfigurations;

/// Shared architecture registry used by facade and backend adapters.
pub static MODEL_CONFIGURATIONS: ModelConfigurations = ModelConfigurations;

impl ModelConfigurationResolver for ModelConfigurations {
    fn resolve_safetensors(&self, json: &Value) -> Result<ModelConfiguration, ArtifactError> {
        resolve_portable_model_configuration(json)
    }

    fn resolve_gguf(
        &self,
        architecture: &str,
        checkpoint: &GgufCheckpoint,
    ) -> Result<ModelConfiguration, ArtifactError> {
        resolve_gguf_configuration(architecture, checkpoint)
    }
}

/// Architecture family identity owned by the architecture registry.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    /// DeepSeek-V3/R1 MLA and MoE architecture.
    DeepSeekV3,
    /// DeepSeek-V4 compressed sparse-attention and mHC architecture.
    DeepSeekV4,
    /// Gemma 4 text or unified multimodal architecture.
    Gemma4,
    /// OpenAI GPT-OSS sparse decoder.
    GptOss,
    /// Thinking Machines Lab Inkling multimodal architecture.
    Inkling,
    /// Moonshot Kimi Linear architecture.
    KimiLinear,
    /// Llama-compatible dense decoders, including Mistral.
    Llama,
    /// Meta Muse-Glimmer multimodal decoder.
    MuseGlimmer,
    /// Liquid AI LFM2/LFM2.5 dense or MoE architecture.
    Lfm2,
    /// Nemotron-H hybrid architecture.
    NemotronH,
    /// Moshi-family realtime speech architecture, including PersonaPlex.
    Moshi,
    /// Qwen2/Qwen2.5 dense decoder.
    Qwen2,
    /// Qwen3 dense or MoE decoder.
    Qwen3,
    /// Qwen3-Next hybrid decoder.
    Qwen3Next,
    /// Qwen3-VL multimodal decoder.
    Qwen3Vl,
    /// Qwen3-VL multimodal MoE decoder.
    Qwen3VlMoe,
    /// Qwen3.5 dense or MoE decoder.
    Qwen35,
}

impl ModelKind {
    /// Every architecture family implemented by this crate.
    pub const ALL: [Self; 17] = [
        Self::DeepSeekV3,
        Self::DeepSeekV4,
        Self::Gemma4,
        Self::GptOss,
        Self::Inkling,
        Self::KimiLinear,
        Self::Llama,
        Self::MuseGlimmer,
        Self::Lfm2,
        Self::NemotronH,
        Self::Moshi,
        Self::Qwen2,
        Self::Qwen3,
        Self::Qwen3Next,
        Self::Qwen3Vl,
        Self::Qwen3VlMoe,
        Self::Qwen35,
    ];

    /// Canonical family name published through the neutral artifact protocol.
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::DeepSeekV3 => "deepseek_v3",
            Self::DeepSeekV4 => "deepseek_v4",
            Self::Gemma4 => "gemma4",
            Self::GptOss => "gpt_oss",
            Self::Inkling => "inkling",
            Self::KimiLinear => "kimi_linear",
            Self::Llama => "llama",
            Self::MuseGlimmer => "muse_glimmer",
            Self::Lfm2 => "lfm2",
            Self::NemotronH => "nemotron_h",
            Self::Moshi => "moshi",
            Self::Qwen2 => "qwen2",
            Self::Qwen3 => "qwen3",
            Self::Qwen3Next => "qwen3_next",
            Self::Qwen3Vl => "qwen3_vl",
            Self::Qwen3VlMoe => "qwen3_vl_moe",
            Self::Qwen35 => "qwen3_5",
        }
    }

    /// Neutral loader protocol required by this family.
    pub const fn loading_protocol(self) -> LoadingProtocol {
        match self {
            Self::Moshi => LoadingProtocol::Realtime,
            _ => LoadingProtocol::Model,
        }
    }

    /// Resolves an architecture-owned canonical family name from a neutral artifact.
    pub fn resolve_family(name: &str) -> Result<Self, ArtifactError> {
        Self::ALL
            .into_iter()
            .find(|kind| kind.canonical_name() == name)
            .ok_or_else(|| {
                ArtifactError::InvalidArtifact(format!(
                    "architecture registry returned unknown canonical family {name:?}"
                ))
            })
    }

    /// Resolves a declared or effective model type to its canonical family.
    ///
    /// Wrapper and implementation variants such as `qwen3_moe` and
    /// `qwen3_5_moe_text` intentionally resolve to the same family identity.
    pub fn resolve_model_type(model_type: &str) -> Result<Self, ArtifactError> {
        match model_type {
            "deepseek_v3" => Ok(Self::DeepSeekV3),
            "deepseek_v4" => Ok(Self::DeepSeekV4),
            "gemma4" | "gemma4_text" | "gemma4_unified" | "gemma4_unified_text" => Ok(Self::Gemma4),
            "gpt_oss" => Ok(Self::GptOss),
            "inkling_mm_model" => Ok(Self::Inkling),
            "kimi_linear" => Ok(Self::KimiLinear),
            "llama" | "mistral" => Ok(Self::Llama),
            "muse_glimmer" | "muse_glimmer_text" => Ok(Self::MuseGlimmer),
            "lfm2" | "lfm2_moe" => Ok(Self::Lfm2),
            "nemotron_h" => Ok(Self::NemotronH),
            "moshi" | "personaplex" => Ok(Self::Moshi),
            "qwen2" => Ok(Self::Qwen2),
            "qwen3" | "qwen3_moe" => Ok(Self::Qwen3),
            "qwen3_next" => Ok(Self::Qwen3Next),
            "qwen3_vl" | "qwen3_vl_text" => Ok(Self::Qwen3Vl),
            "qwen3_vl_moe" | "qwen3_vl_moe_text" => Ok(Self::Qwen3VlMoe),
            "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" => Ok(Self::Qwen35),
            other => Err(ArtifactError::UnsupportedModelType(other.into())),
        }
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
        family: architecture.model_kind().canonical_name().into(),
        loading_protocol: architecture.model_kind().loading_protocol(),
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
    } else if ModelKind::resolve_model_type(&metadata.model_type).is_ok() {
        metadata.model_type.clone()
    } else {
        metadata
            .text_config
            .as_ref()
            .and_then(|text| text.model_type.clone())
            .unwrap_or_else(|| metadata.model_type.clone())
    }
}

/// Resolves family aliases and nested wrappers without parsing family geometry.
pub fn resolve_model_identity(json: &Value) -> Result<ResolvedModelConfig, ArtifactError> {
    let metadata: ConfigMetadata = serde_json::from_value(json.clone())?;
    let effective_model_type = effective_model_type(&metadata);
    let kind = ModelKind::resolve_model_type(&effective_model_type)?;
    Ok(ResolvedModelConfig {
        model_type: metadata.model_type,
        effective_model_type,
        kind,
    })
}

fn resolve_portable_model_configuration(json: &Value) -> Result<ModelConfiguration, ArtifactError> {
    let resolved = resolve_model_identity(json)?;
    Ok(ModelConfiguration {
        declared_model_type: resolved.model_type,
        effective_model_type: resolved.effective_model_type,
        family: resolved.kind.canonical_name().into(),
        loading_protocol: resolved.kind.loading_protocol(),
        json: Some(json.clone()),
    })
}

fn invalid_configuration(kind: ModelKind, error: impl std::fmt::Display) -> ArtifactError {
    ArtifactError::InvalidArtifact(format!(
        "invalid {} configuration: {error}",
        kind.canonical_name()
    ))
}

/// Validates a resolved Hugging Face config with its architecture-owned parser.
fn validate_model_configuration(json: &Value, kind: ModelKind) -> Result<(), ArtifactError> {
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

/// Architecture-owned family identity consumed by backend composition and inspection.
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
    let resolved = resolve_model_identity(json)?;
    validate_model_configuration(json, resolved.kind)?;
    Ok(resolved)
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

/// Resolves the model kind of an external SafeTensors drafter config.
pub fn resolve_safetensors_assistant_model_kind(
    json: &Value,
) -> Result<AssistantModelKind, ArtifactError> {
    match json.get("model_type").and_then(Value::as_str) {
        Some("gemma4_assistant") => Ok(AssistantModelKind::Gemma4),
        Some("muse_glimmer_assistant") => Ok(AssistantModelKind::MuseGlimmer),
        Some(other) => Err(ArtifactError::UnsupportedModelType(other.into())),
        None => Err(ArtifactError::InvalidArtifact(
            "assistant config is missing model_type".into(),
        )),
    }
}

/// Resolves the model kind of an external GGUF drafter from neutral metadata.
pub fn resolve_gguf_assistant_model_kind(
    metadata: &HashMap<String, MetadataValue>,
) -> Result<AssistantModelKind, ArtifactError> {
    match metadata
        .get("general.architecture")
        .and_then(MetadataValue::as_str)
    {
        Some("dflash") => Ok(AssistantModelKind::MuseGlimmer),
        Some("gemma4_assistant" | "gemma4-assistant") => Ok(AssistantModelKind::Gemma4),
        Some(other) => Err(ArtifactError::UnsupportedGgufArchitecture(other.into())),
        None => Err(ArtifactError::MissingGgufArchitecture),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_gguf::{GgmlType, TensorInput, Writer};
    use std::{collections::BTreeMap, fs::File};

    #[test]
    fn nested_wrappers_and_aliases_resolve_in_one_registry() {
        let json = serde_json::json!({
            "model_type": "qwen3_5",
            "text_config": { "model_type": "qwen3_5_moe" }
        });
        let resolved = resolve_model_identity(&json).unwrap();
        assert_eq!(resolved.kind, ModelKind::Qwen35);
        assert_eq!(resolved.model_type, "qwen3_5");
        assert_eq!(resolved.effective_model_type, "qwen3_5_moe");

        for model_type in ["moshi", "personaplex"] {
            let json = serde_json::json!({"model_type": model_type});
            let resolved = resolve_model_identity(&json).unwrap();
            assert_eq!(resolved.kind, ModelKind::Moshi);
            let portable = MODEL_CONFIGURATIONS.resolve_safetensors(&json).unwrap();
            assert_eq!(portable.family, "moshi");
            assert_eq!(portable.loading_protocol, LoadingProtocol::Realtime);
        }
    }

    #[test]
    fn effective_model_types_resolve_to_explicit_canonical_families() {
        for (model_type, family) in [
            ("qwen3_moe", ModelKind::Qwen3),
            ("qwen3_vl_moe_text", ModelKind::Qwen3VlMoe),
            ("qwen3_5_text", ModelKind::Qwen35),
            ("qwen3_5_moe_text", ModelKind::Qwen35),
        ] {
            assert_eq!(ModelKind::resolve_model_type(model_type).unwrap(), family);
            assert_eq!(
                ModelKind::resolve_model_type(model_type)
                    .unwrap()
                    .canonical_name(),
                family.canonical_name()
            );
        }
    }

    #[test]
    fn unknown_wrapper_can_delegate_to_a_known_nested_text_family() {
        let resolved = resolve_model_identity(&serde_json::json!({
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
    fn external_assistant_gguf_aliases_resolve_in_the_architecture_registry() {
        for (architecture, kind) in [
            ("dflash", AssistantModelKind::MuseGlimmer),
            ("gemma4_assistant", AssistantModelKind::Gemma4),
            ("gemma4-assistant", AssistantModelKind::Gemma4),
        ] {
            let metadata = HashMap::from([(
                "general.architecture".into(),
                MetadataValue::String(architecture.into()),
            )]);
            assert_eq!(resolve_gguf_assistant_model_kind(&metadata).unwrap(), kind);
        }

        let unsupported = HashMap::from([(
            "general.architecture".into(),
            MetadataValue::String("unknown-assistant".into()),
        )]);
        assert!(matches!(
            resolve_gguf_assistant_model_kind(&unsupported),
            Err(ArtifactError::UnsupportedGgufArchitecture(name))
                if name == "unknown-assistant"
        ));
        assert!(matches!(
            resolve_gguf_assistant_model_kind(&HashMap::new()),
            Err(ArtifactError::MissingGgufArchitecture)
        ));
        let wrong_type = HashMap::from([("general.architecture".into(), MetadataValue::Uint32(1))]);
        assert!(matches!(
            resolve_gguf_assistant_model_kind(&wrong_type),
            Err(ArtifactError::MissingGgufArchitecture)
        ));
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

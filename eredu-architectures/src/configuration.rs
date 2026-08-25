//! Authoritative Hugging Face and GGUF family identity and configuration validation.

use eredu_checkpoint::{
    schema::SafetensorsCheckpointPlan,
    validation::{CheckpointValidation, StrictLoadFailure},
};
use eredu_core::{
    artifact::ArtifactError, ArtifactFormat, GgufCompanionEncoding, GgufCompanionRequirement,
    GgufCompanionRole, LoadingProtocol, ModelConfiguration, ModelConfigurationResolver,
    ResolvedModelConfiguration, ValidatedGguf,
};
use eredu_gguf::Checkpoint as GgufCheckpoint;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{fs, path::Path};

/// Stateless registry for every architecture family implemented by this crate.
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelConfigurations;

/// Shared architecture registry used by facade and backend adapters.
pub static MODEL_CONFIGURATIONS: ModelConfigurations = ModelConfigurations;

impl ModelConfigurationResolver for ModelConfigurations {
    type ArtifactPlan = crate::processor_plan::ArtifactArchitecturePlan;

    fn resolve_safetensors(
        &self,
        json: &Value,
    ) -> Result<ResolvedModelConfiguration<Self::ArtifactPlan>, ArtifactError> {
        let resolved = resolve_model_config(json)?;
        let configuration = portable_model_configuration(json, &resolved);
        Ok(ResolvedModelConfiguration::new(
            configuration,
            crate::processor_plan::ArtifactArchitecturePlan::from_safetensors_architecture(
                resolved.architecture,
            ),
        ))
    }

    fn resolve_gguf(
        &self,
        architecture: &str,
        checkpoint: &GgufCheckpoint,
    ) -> Result<ResolvedModelConfiguration<Self::ArtifactPlan>, ArtifactError> {
        let (configuration, architecture) = resolve_gguf_configuration(architecture, checkpoint)?;
        Ok(ResolvedModelConfiguration::new(
            configuration,
            crate::processor_plan::ArtifactArchitecturePlan::from_gguf_architecture(architecture),
        ))
    }

    fn gguf_companion_requirements(
        &self,
        architecture: &str,
        _checkpoint: &GgufCheckpoint,
    ) -> Result<Vec<GgufCompanionRequirement>, ArtifactError> {
        gguf_companion_requirements(GgufArchitecture::resolve(architecture)?)
    }

    fn artifact_plan(
        &self,
        path: &Path,
        format: ArtifactFormat,
        configuration: &ModelConfiguration,
        validated_gguf: Option<&ValidatedGguf>,
        resolved_plan: Self::ArtifactPlan,
    ) -> Result<Self::ArtifactPlan, ArtifactError> {
        let plan = match format {
            ArtifactFormat::SafeTensors => {
                let kind = resolved_plan.model_kind();
                let model = serde_json::to_vec(configuration.json.as_ref().ok_or_else(|| {
                    ArtifactError::InvalidArtifact(
                        "SafeTensors inspection omitted normalized JSON configuration".into(),
                    )
                })?)?;
                let (image, video) = if matches!(
                    kind,
                    ModelKind::Gemma4
                        | ModelKind::Qwen3Vl
                        | ModelKind::Qwen3VlMoe
                        | ModelKind::Qwen35
                ) {
                    (
                        read_optional_sidecar(
                            path,
                            crate::processor_plan::PROCESSOR_CONFIG_FILENAME,
                        )?,
                        read_optional_sidecar(
                            path,
                            crate::processor_plan::VIDEO_PROCESSOR_CONFIG_FILENAME,
                        )?,
                    )
                } else {
                    (None, None)
                };
                let muse = if kind == ModelKind::MuseGlimmer {
                    read_optional_sidecar(
                        path,
                        crate::processor_plan::MUSE_PROCESSOR_CONFIG_FILENAME,
                    )?
                } else {
                    None
                };
                resolved_plan.with_safetensors_processors(
                    &model,
                    image.as_deref(),
                    video.as_deref(),
                    muse.as_deref(),
                )
            }
            ArtifactFormat::Gguf => {
                let validated = validated_gguf.ok_or_else(|| {
                    ArtifactError::InvalidArtifact(
                        "GGUF inspection omitted its validated checkpoint".into(),
                    )
                })?;
                let projector = validated
                    .companion(&GgufCompanionRole::MediaProjector)
                    .map(|companion| companion.checkpoint().metadata());
                resolved_plan.with_gguf_processors(validated.checkpoint().metadata(), projector)
            }
        };
        plan.map_err(|error| ArtifactError::InvalidArchitecturePlan(error.to_string()))
    }
}

fn read_optional_sidecar(path: &Path, filename: &str) -> Result<Option<Vec<u8>>, ArtifactError> {
    match fs::read(path.join(filename)) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

/// Declares the portable sibling-artifact requirements for one GGUF architecture.
pub fn gguf_companion_requirements(
    architecture: GgufArchitecture,
) -> Result<Vec<GgufCompanionRequirement>, ArtifactError> {
    let policy = match architecture {
        GgufArchitecture::Qwen3Vl | GgufArchitecture::Qwen3VlMoe => {
            Some((true, GgufCompanionEncoding::DensePreferred))
        }
        GgufArchitecture::MuseGlimmer => Some((true, GgufCompanionEncoding::DensePreferred)),
        GgufArchitecture::Gemma4 => Some((false, GgufCompanionEncoding::DenseRequired)),
        GgufArchitecture::Inkling | GgufArchitecture::Qwen35 | GgufArchitecture::Qwen35Moe => {
            Some((false, GgufCompanionEncoding::DensePreferred))
        }
        _ => None,
    };
    policy
        .map(|(required, encoding)| {
            GgufCompanionRequirement::new(
                GgufCompanionRole::MediaProjector,
                required,
                "mmproj",
                1,
                encoding,
            )
            .map(|requirement| vec![requirement])
        })
        .unwrap_or_else(|| Ok(Vec::new()))
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
) -> Result<(ModelConfiguration, GgufArchitecture), ArtifactError> {
    let architecture = GgufArchitecture::resolve(name)?;
    validate_gguf_structure(architecture, checkpoint)?;
    Ok((
        ModelConfiguration {
            declared_model_type: name.into(),
            effective_model_type: name.into(),
            family: architecture.model_kind().canonical_name().into(),
            loading_protocol: architecture.model_kind().loading_protocol(),
            json: None,
        },
        architecture,
    ))
}

fn validate_gguf_structure(
    architecture: GgufArchitecture,
    checkpoint: &GgufCheckpoint,
) -> Result<(), ArtifactError> {
    validate_gguf_checkpoint(architecture, checkpoint)
        .into_loader_result()
        .map_err(gguf_validation_error)
}

/// Parses and validates the complete architecture-owned GGUF checkpoint schema.
///
/// This is the portable admission boundary used by the configuration resolver
/// and by concrete backends that need to validate an already inspected GGUF.
pub fn validate_gguf_checkpoint(
    architecture: GgufArchitecture,
    checkpoint: &GgufCheckpoint,
) -> CheckpointValidation {
    crate::gguf_admission::validate(architecture, checkpoint)
}

fn gguf_validation_error(failure: StrictLoadFailure) -> ArtifactError {
    let mut details = failure
        .missing
        .into_iter()
        .map(|name| format!("missing {name:?}"))
        .collect::<Vec<_>>();
    details.extend(failure.unused);
    ArtifactError::InvalidArtifact(details.join("; "))
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
    if matches!(
        metadata.model_type.as_str(),
        "gemma4" | "gemma4_unified" | "qwen3_vl" | "qwen3_vl_moe" | "qwen3_5" | "qwen3_5_moe"
    ) {
        metadata
            .text_config
            .as_ref()
            .and_then(|text| text.model_type.clone())
            .unwrap_or_else(|| metadata.model_type.clone())
    } else {
        metadata.model_type.clone()
    }
}

/// Family identity resolved before architecture geometry is parsed.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ResolvedModelIdentity {
    /// Canonical architecture family.
    pub kind: ModelKind,
    /// Declared outer model type.
    pub model_type: String,
    /// Effective nested text model type.
    pub effective_model_type: String,
}

/// Resolves family aliases and nested wrappers without parsing family geometry.
pub fn resolve_model_identity(json: &Value) -> Result<ResolvedModelIdentity, ArtifactError> {
    let metadata: ConfigMetadata = serde_json::from_value(json.clone())?;
    let effective_model_type = effective_model_type(&metadata);
    let kind = ModelKind::resolve_model_type(&effective_model_type)?;
    Ok(ResolvedModelIdentity {
        model_type: metadata.model_type,
        effective_model_type,
        kind,
    })
}

fn portable_model_configuration(
    json: &Value,
    resolved: &ResolvedModelConfig,
) -> ModelConfiguration {
    ModelConfiguration {
        declared_model_type: resolved.model_type.clone(),
        effective_model_type: resolved.effective_model_type.clone(),
        family: resolved.kind.canonical_name().into(),
        loading_protocol: resolved.kind.loading_protocol(),
        json: Some(json.clone()),
    }
}

fn invalid_configuration(kind: ModelKind, error: impl std::fmt::Display) -> ArtifactError {
    ArtifactError::InvalidArtifact(format!(
        "invalid {} configuration: {error}",
        kind.canonical_name()
    ))
}

/// Typed, normalized family configuration retained from SafeTensors admission.
#[derive(Debug, Clone)]
pub enum SafetensorsModelConfig {
    /// DeepSeek-V3/R1 family geometry.
    DeepSeekV3(crate::deepseek::V3Args),
    /// DeepSeek-V4 family geometry.
    DeepSeekV4(crate::deepseek::V4Args),
    /// Gemma 4 family geometry.
    Gemma4(crate::gemma4::FamilyConfig),
    /// GPT-OSS family geometry.
    GptOss(crate::gpt_oss::ModelArgs),
    /// Inkling family geometry.
    Inkling(crate::inkling::ModelArgs),
    /// Kimi Linear family geometry.
    KimiLinear(crate::kimi_linear::ModelArgs),
    /// Llama-compatible family geometry.
    Llama(crate::llama::ModelArgs),
    /// Muse-Glimmer family geometry.
    MuseGlimmer(crate::muse_glimmer::DecoderConfig),
    /// LFM2 family geometry.
    Lfm2(crate::lfm2::ModelArgs),
    /// Nemotron-H family geometry.
    NemotronH(crate::nemotron_h::ModelArgs),
    /// Moshi-family realtime geometry.
    Moshi(crate::moshi::MoshiConfig),
    /// Qwen2/Qwen3 family geometry.
    Qwen(crate::qwen::ModelArgs),
    /// Qwen hybrid-family geometry.
    QwenHybrid(crate::qwen::hybrid::ParsedHybridConfig),
    /// Qwen3-VL family geometry.
    QwenVl(crate::qwen::vl::ModelArgs),
}

/// Architecture configuration and checkpoint schema proven valid at admission.
#[derive(Debug, Clone)]
pub struct SafetensorsArchitecturePlan {
    kind: ModelKind,
    model: SafetensorsModelConfig,
    checkpoint: SafetensorsCheckpointPlan,
}

impl SafetensorsArchitecturePlan {
    /// Canonical family selected for this exact configuration.
    pub const fn model_kind(&self) -> ModelKind {
        self.kind
    }

    /// Typed normalized family configuration.
    pub const fn model(&self) -> &SafetensorsModelConfig {
        &self.model
    }

    /// Complete expected SafeTensors catalog derived from the normalized geometry.
    pub const fn checkpoint(&self) -> &SafetensorsCheckpointPlan {
        &self.checkpoint
    }
}

fn resolve_safetensors_architecture(
    json: &Value,
    kind: ModelKind,
) -> Result<SafetensorsArchitecturePlan, ArtifactError> {
    let model = match kind {
        ModelKind::DeepSeekV3 => crate::deepseek::parse_v3_config(json)
            .map(SafetensorsModelConfig::DeepSeekV3)
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::DeepSeekV4 => crate::deepseek::parse_v4_config(json)
            .map(SafetensorsModelConfig::DeepSeekV4)
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::Gemma4 => serde_json::to_vec(json)
            .map_err(ArtifactError::from)
            .and_then(|bytes| {
                crate::gemma4::FamilyConfig::from_hf_json(&bytes)
                    .map(SafetensorsModelConfig::Gemma4)
                    .map_err(|error| invalid_configuration(kind, error))
            }),
        ModelKind::GptOss => crate::gpt_oss::model_args_from_config_value(json)
            .map(SafetensorsModelConfig::GptOss)
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::Inkling => serde_json::to_vec(json)
            .map_err(ArtifactError::from)
            .and_then(|bytes| {
                crate::inkling::ModelArgs::from_hf_json(&bytes)
                    .map(SafetensorsModelConfig::Inkling)
                    .map_err(|error| invalid_configuration(kind, error))
            }),
        ModelKind::KimiLinear => crate::kimi_linear::model_args_from_config_value(json)
            .map(SafetensorsModelConfig::KimiLinear)
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::Llama => crate::llama::model_args_from_config_value(json)
            .map(SafetensorsModelConfig::Llama)
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::MuseGlimmer => serde_json::to_vec(json)
            .map_err(ArtifactError::from)
            .and_then(|bytes| {
                crate::muse_glimmer::DecoderConfig::from_hf_json(&bytes)
                    .map(SafetensorsModelConfig::MuseGlimmer)
                    .map_err(|error| invalid_configuration(kind, error))
            }),
        ModelKind::Lfm2 => crate::lfm2::model_args_from_config_value(json)
            .map(SafetensorsModelConfig::Lfm2)
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::NemotronH => crate::nemotron_h::model_args_from_config_value(json)
            .map(SafetensorsModelConfig::NemotronH)
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::Moshi => crate::moshi::MoshiConfig::from_config_value(Some(json))
            .map(SafetensorsModelConfig::Moshi)
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::Qwen2 | ModelKind::Qwen3 => crate::qwen::model_args_from_config_value(json)
            .map(SafetensorsModelConfig::Qwen)
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::Qwen3Next | ModelKind::Qwen35 => {
            crate::qwen::hybrid::model_args_from_config_value(json)
                .map(SafetensorsModelConfig::QwenHybrid)
                .map_err(|error| invalid_configuration(kind, error))
        }
        ModelKind::Qwen3Vl | ModelKind::Qwen3VlMoe => {
            crate::qwen::vl::model_args_from_config_value(json)
                .map(SafetensorsModelConfig::QwenVl)
                .map_err(|error| invalid_configuration(kind, error))
        }
    }?;
    let checkpoint = match &model {
        SafetensorsModelConfig::DeepSeekV3(args) => {
            crate::deepseek::v3_safetensors_plan(args, true)
                .map_err(|error| invalid_configuration(kind, error))?
        }
        SafetensorsModelConfig::DeepSeekV4(args) => crate::deepseek::v4_safetensors_plan(args)
            .map_err(|error| invalid_configuration(kind, error))?,
        SafetensorsModelConfig::Gemma4(args) => crate::gemma4::safetensors_plan(args)
            .map_err(|error| invalid_configuration(kind, error))?,
        SafetensorsModelConfig::GptOss(args) => crate::gpt_oss::safetensors_plan(args)
            .map_err(|error| invalid_configuration(kind, error))?,
        SafetensorsModelConfig::Inkling(args) => crate::inkling::safetensors_plan(args)
            .map_err(|error| invalid_configuration(kind, error))?,
        SafetensorsModelConfig::KimiLinear(args) => crate::kimi_linear::safetensors_plan(args)
            .map_err(|error| invalid_configuration(kind, error))?,
        SafetensorsModelConfig::Llama(args) => crate::llama::safetensors_plan(args)
            .map_err(|error| invalid_configuration(kind, error))?,
        SafetensorsModelConfig::MuseGlimmer(args) => crate::muse_glimmer::safetensors_plan(args)
            .map_err(|error| invalid_configuration(kind, error))?,
        SafetensorsModelConfig::Lfm2(args) => crate::lfm2::safetensors_plan(args, true)
            .map_err(|error| invalid_configuration(kind, error))?,
        SafetensorsModelConfig::NemotronH(args) => crate::nemotron_h::safetensors_plan(args)
            .map_err(|error| invalid_configuration(kind, error))?,
        SafetensorsModelConfig::Moshi(args) => crate::moshi::safetensors_plan(args)
            .map_err(|error| invalid_configuration(kind, error))?,
        SafetensorsModelConfig::Qwen(args) => crate::qwen::safetensors_plan(args)
            .map_err(|error| invalid_configuration(kind, error))?,
        SafetensorsModelConfig::QwenHybrid(args) => {
            crate::qwen::hybrid::safetensors_plan(&args.text)
                .map_err(|error| invalid_configuration(kind, error))?
        }
        SafetensorsModelConfig::QwenVl(args) => {
            if args.text.is_moe() != (kind == ModelKind::Qwen3VlMoe) {
                return Err(invalid_configuration(
                    kind,
                    "nested text configuration does not match dense/MoE family dispatch",
                ));
            }
            crate::qwen::vl::safetensors_plan(args)
                .map_err(|error| invalid_configuration(kind, error))?
        }
    };
    Ok(SafetensorsArchitecturePlan {
        kind,
        model,
        checkpoint,
    })
}

/// Architecture-owned family identity consumed by backend composition and inspection.
#[derive(Debug, Clone)]
pub struct ResolvedModelConfig {
    /// Canonical architecture family.
    pub kind: ModelKind,
    /// Declared outer model type.
    pub model_type: String,
    /// Effective nested text model type.
    pub effective_model_type: String,
    /// Typed model geometry and checkpoint schema proven valid during resolution.
    pub architecture: SafetensorsArchitecturePlan,
}

/// Resolves and validates one config into its architecture-owned dispatch identity.
pub fn resolve_model_config(json: &Value) -> Result<ResolvedModelConfig, ArtifactError> {
    let resolved = resolve_model_identity(json)?;
    let architecture = resolve_safetensors_architecture(json, resolved.kind)?;
    Ok(ResolvedModelConfig {
        kind: resolved.kind,
        model_type: resolved.model_type,
        effective_model_type: resolved.effective_model_type,
        architecture,
    })
}

/// Inspects an artifact using this crate's authoritative family registry.
pub fn inspect_artifact(
    path: impl AsRef<Path>,
) -> Result<
    eredu_core::ArtifactInspection<crate::processor_plan::ArtifactArchitecturePlan>,
    ArtifactError,
> {
    eredu_core::inspect_artifact(path, &MODEL_CONFIGURATIONS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_gguf::{GgmlType, MetadataValue, TensorInput, Writer};
    use std::{collections::BTreeMap, fs::File};

    fn qwen_vl_config() -> Value {
        serde_json::json!({
            "model_type": "qwen3_vl",
            "image_token_id": 61,
            "video_token_id": 62,
            "vision_start_token_id": 44,
            "vision_end_token_id": 45,
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": "qwen3_vl_text", "hidden_size": 32,
                "num_hidden_layers": 3, "intermediate_size": 64,
                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                "rms_norm_eps": 0.000001, "vocab_size": 64,
                "max_position_embeddings": 128, "rope_theta": 1000000.0,
                "rope_scaling": {"mrope_section": [2, 1, 1], "mrope_interleaved": true}
            },
            "vision_config": {
                "depth": 4, "hidden_size": 16, "intermediate_size": 24,
                "num_heads": 4, "num_position_embeddings": 16, "in_channels": 3,
                "patch_size": 2, "spatial_merge_size": 2, "temporal_patch_size": 2,
                "out_hidden_size": 32, "deepstack_visual_indexes": [1, 3]
            }
        })
    }

    #[test]
    fn safetensors_processor_sidecars_are_snapshotted_during_inspection() {
        let root = tempfile::tempdir().unwrap();
        let processor_path = root
            .path()
            .join(crate::processor_plan::PROCESSOR_CONFIG_FILENAME);
        let visual = |mean: f32| {
            serde_json::to_vec(&serde_json::json!({
                "size": {"shortest_edge": 16, "longest_edge": 64},
                "patch_size": 2,
                "temporal_patch_size": 2,
                "merge_size": 2,
                "image_mean": [mean, mean, mean],
                "image_std": [1.0, 1.0, 1.0]
            }))
            .unwrap()
        };
        std::fs::write(&processor_path, visual(0.25)).unwrap();
        let (configuration, resolved_plan) = MODEL_CONFIGURATIONS
            .resolve_safetensors(&qwen_vl_config())
            .unwrap()
            .into_parts();
        let plan = MODEL_CONFIGURATIONS
            .artifact_plan(
                root.path(),
                ArtifactFormat::SafeTensors,
                &configuration,
                None,
                resolved_plan,
            )
            .unwrap();

        std::fs::write(&processor_path, visual(0.75)).unwrap();

        assert_eq!(plan.model_kind(), ModelKind::Qwen3Vl);
        assert_eq!(
            plan.qwen().unwrap().image(8, 8).unwrap().transform.mean,
            [0.25; 3]
        );
        let (next_configuration, next_resolved_plan) = MODEL_CONFIGURATIONS
            .resolve_safetensors(&qwen_vl_config())
            .unwrap()
            .into_parts();
        let next = MODEL_CONFIGURATIONS
            .artifact_plan(
                root.path(),
                ArtifactFormat::SafeTensors,
                &next_configuration,
                None,
                next_resolved_plan,
            )
            .unwrap();
        assert_eq!(
            next.qwen().unwrap().image(8, 8).unwrap().transform.mean,
            [0.75; 3]
        );
    }

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
            assert_eq!(resolved.kind.loading_protocol(), LoadingProtocol::Realtime);
        }
    }

    #[test]
    fn authoritative_resolver_rejects_malformed_family_geometry() {
        let malformed = serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": 16,
            "num_hidden_layers": 0,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "rms_norm_eps": 0.000001,
            "vocab_size": 64,
            "max_position_embeddings": 128,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": true
        });

        assert_eq!(
            resolve_model_identity(&malformed).unwrap().kind,
            ModelKind::Qwen3
        );
        assert!(matches!(
            MODEL_CONFIGURATIONS.resolve_safetensors(&malformed),
            Err(ArtifactError::InvalidArtifact(_))
        ));
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
    fn unknown_wrapper_cannot_delegate_to_a_known_nested_text_family() {
        let error = resolve_model_identity(&serde_json::json!({
            "model_type": "third_party_wrapper",
            "text_config": { "model_type": "qwen3_vl_moe_text" }
        }))
        .unwrap_err();
        assert!(matches!(
            error,
            ArtifactError::UnsupportedModelType(model_type)
                if model_type == "third_party_wrapper"
        ));
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
    fn gguf_families_declare_companion_presence_and_encoding_policy() {
        let required_quantized = GgufCompanionRequirement::new(
            GgufCompanionRole::MediaProjector,
            true,
            "mmproj",
            1,
            GgufCompanionEncoding::DensePreferred,
        )
        .unwrap();
        assert_eq!(
            gguf_companion_requirements(GgufArchitecture::Qwen3Vl).unwrap(),
            vec![required_quantized]
        );
        let optional_dense = GgufCompanionRequirement::new(
            GgufCompanionRole::MediaProjector,
            false,
            "mmproj",
            1,
            GgufCompanionEncoding::DenseRequired,
        )
        .unwrap();
        assert_eq!(
            gguf_companion_requirements(GgufArchitecture::Gemma4).unwrap(),
            vec![optional_dense]
        );
        assert!(gguf_companion_requirements(GgufArchitecture::Llama)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn gguf_inspection_runs_complete_family_parsing_before_backend_selection() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("model.gguf");
        let scalar = 1.0_f32.to_le_bytes();
        let metadata = BTreeMap::from([(
            "general.architecture".into(),
            MetadataValue::String("llama".into()),
        )]);
        Writer::default()
            .write(
                File::create(&path).unwrap(),
                &metadata,
                &[TensorInput {
                    name: "token_embd.weight",
                    dimensions: &[1],
                    ggml_type: GgmlType::F32,
                    data: &scalar,
                }],
            )
            .unwrap();

        let error = inspect_artifact(&path).unwrap_err();
        assert!(matches!(
            error,
            ArtifactError::InvalidArtifact(detail)
                if detail.contains("llama.embedding_length")
        ));
    }

    #[test]
    fn gguf_inspection_runs_complete_schema_validation_before_backend_selection() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("model.gguf");
        let scalar = 1.0_f32.to_le_bytes();
        let metadata = BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("llama".into()),
            ),
            ("llama.embedding_length".into(), MetadataValue::Uint32(1)),
            (
                "llama.attention.head_count".into(),
                MetadataValue::Uint32(1),
            ),
            ("llama.block_count".into(), MetadataValue::Uint32(1)),
            ("llama.feed_forward_length".into(), MetadataValue::Uint32(1)),
            (
                "llama.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::Float32(1e-5),
            ),
            ("llama.vocab_size".into(), MetadataValue::Uint32(1)),
            ("llama.context_length".into(), MetadataValue::Uint32(1)),
        ]);
        Writer::default()
            .write(
                File::create(&path).unwrap(),
                &metadata,
                &[TensorInput {
                    name: "token_embd.weight",
                    dimensions: &[1, 1],
                    ggml_type: GgmlType::F32,
                    data: &scalar,
                }],
            )
            .unwrap();

        let error = inspect_artifact(&path).unwrap_err();
        assert!(matches!(
            error,
            ArtifactError::InvalidArtifact(detail)
                if detail.contains("output_norm.weight")
        ));
    }
}

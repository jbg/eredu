//! Authoritative Hugging Face and GGUF family identity and configuration validation.

use eredu_checkpoint::{
    recipe::{AtomicRecipeSet, RecipeCatalog},
    schema::{GgufCheckpointPlan, SafetensorsCheckpointPlan},
    store::{StoreError, TensorMetadata},
    validation::{
        resolve_safetensors_plan, CatalogTensorMetadata, CheckpointValidation,
        ResolvedCheckpointPlan, SafetensorsCatalog, StrictLoadFailure,
    },
    StoredDtype,
};
use eredu_core::{
    artifact::ArtifactError,
    checkpoint::{TensorCatalog, TensorDtype},
    ArtifactFormat, GgufCompanionEncoding, GgufCompanionRequirement, GgufCompanionRole,
    LoadingProtocol, ModelConfiguration, ModelConfigurationResolver, ResolvedModelConfiguration,
    ValidatedGguf,
};
use eredu_gguf::Checkpoint as GgufCheckpoint;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::{collections::BTreeSet, fs, path::Path};

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
        let configuration = portable_model_configuration(json, &resolved)?;
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
        tensors: &TensorCatalog,
        validated_gguf: Option<&ValidatedGguf>,
        resolved_plan: Self::ArtifactPlan,
    ) -> Result<Self::ArtifactPlan, ArtifactError> {
        let plan = match format {
            ArtifactFormat::SafeTensors => {
                let resolved_plan = resolved_plan.admit_safetensors_catalog(tensors)?;
                let kind = resolved_plan.model_kind();
                let model = serde_json::to_vec(configuration.json().ok_or_else(|| {
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
                    .map(|companion| companion.checkpoint());
                let media_projector = projector
                    .map(|projector| {
                        crate::gguf_companion::resolve_media_projector(
                            resolved_plan.gguf_plan().ok_or_else(|| {
                                ArtifactError::InvalidArchitecturePlan(
                                    "GGUF companion admission requires a resolved architecture plan"
                                        .into(),
                                )
                            })?,
                            validated.checkpoint(),
                            projector,
                        )
                        .map_err(ArtifactError::InvalidArchitecturePlan)
                    })
                    .transpose()?;
                resolved_plan
                    .with_gguf_media_projector(media_projector)
                    .with_gguf_processors(
                        validated.checkpoint().metadata(),
                        projector.map(GgufCheckpoint::metadata),
                    )
            }
            _ => {
                return Err(ArtifactError::InvalidArtifact(
                    "unsupported artifact format selected during architecture admission".into(),
                ));
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
    use crate::preparation::GgufMediaProjectorRequirement;

    let required = match crate::preparation::gguf_composite_artifact_plan(architecture)
        .media_projector_requirement()
    {
        GgufMediaProjectorRequirement::NotApplicable => return Ok(Vec::new()),
        GgufMediaProjectorRequirement::Optional => false,
        GgufMediaProjectorRequirement::Required => true,
    };
    let encoding = if architecture == GgufArchitecture::Gemma4 {
        GgufCompanionEncoding::DenseRequired
    } else {
        GgufCompanionEncoding::DensePreferred
    };
    GgufCompanionRequirement::new(
        GgufCompanionRole::MediaProjector,
        required,
        "mmproj",
        1,
        encoding,
    )
    .map(|requirement| vec![requirement])
}

/// Architecture family identity owned by the architecture registry.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
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

impl Serialize for ModelKind {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.canonical_name())
    }
}

impl<'de> Deserialize<'de> for ModelKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let name = String::deserialize(deserializer)?;
        Self::resolve_family(&name).map_err(serde::de::Error::custom)
    }
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
) -> Result<(ModelConfiguration, GgufArchitecturePlan), ArtifactError> {
    let architecture = GgufArchitecture::resolve(name)?;
    let plan = resolve_gguf_architecture(architecture, checkpoint)?;
    Ok((
        ModelConfiguration::new(
            name,
            name,
            architecture.model_kind().canonical_name(),
            architecture.model_kind().loading_protocol(),
            None,
        )?,
        plan,
    ))
}

fn resolve_gguf_architecture(
    architecture: GgufArchitecture,
    checkpoint: &GgufCheckpoint,
) -> Result<GgufArchitecturePlan, ArtifactError> {
    let (plan, validation) = crate::gguf_admission::resolve(architecture, checkpoint)
        .map_err(ArtifactError::InvalidArtifact)?;
    validation
        .into_loader_result()
        .map_err(checkpoint_validation_error)?;
    Ok(plan)
}

fn checkpoint_validation_error(failure: StrictLoadFailure) -> ArtifactError {
    let mut details = failure
        .missing
        .into_iter()
        .map(|name| format!("missing {name:?}"))
        .collect::<Vec<_>>();
    details.extend(failure.unused);
    ArtifactError::InvalidArtifact(details.join("; "))
}

fn invalid_checkpoint_validation(validation: CheckpointValidation) -> ArtifactError {
    match validation.into_loader_result() {
        Err(failure) => checkpoint_validation_error(failure),
        Ok(()) => ArtifactError::InvalidArtifact(
            "checkpoint layout resolution failed without validation diagnostics".into(),
        ),
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

fn effective_model_type(
    metadata: &ConfigMetadata,
    declared_kind: ModelKind,
) -> Result<String, ArtifactError> {
    if matches!(
        metadata.model_type.as_str(),
        "gemma4" | "gemma4_unified" | "qwen3_vl" | "qwen3_vl_moe" | "qwen3_5" | "qwen3_5_moe"
    ) {
        let effective_model_type = metadata
            .text_config
            .as_ref()
            .and_then(|text| text.model_type.clone())
            .unwrap_or_else(|| metadata.model_type.clone());
        let effective_kind = ModelKind::resolve_model_type(&effective_model_type)?;
        if effective_kind != declared_kind {
            return Err(ArtifactError::InvalidArtifact(format!(
                "declared model type {:?} resolves to family {:?}, but nested text model type {:?} resolves to family {:?}",
                metadata.model_type,
                declared_kind.canonical_name(),
                effective_model_type,
                effective_kind.canonical_name()
            )));
        }
        Ok(effective_model_type)
    } else {
        Ok(metadata.model_type.clone())
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

/// Resolves compatible family aliases and nested wrappers without parsing
/// family geometry.
pub fn resolve_model_identity(json: &Value) -> Result<ResolvedModelIdentity, ArtifactError> {
    let metadata: ConfigMetadata = serde_json::from_value(json.clone())?;
    let kind = ModelKind::resolve_model_type(&metadata.model_type)?;
    let effective_model_type = effective_model_type(&metadata, kind)?;
    Ok(ResolvedModelIdentity {
        model_type: metadata.model_type,
        effective_model_type,
        kind,
    })
}

fn portable_model_configuration(
    json: &Value,
    resolved: &ResolvedModelConfig,
) -> Result<ModelConfiguration, ArtifactError> {
    ModelConfiguration::new(
        resolved.model_type.clone(),
        resolved.effective_model_type.clone(),
        resolved.kind.canonical_name(),
        resolved.kind.loading_protocol(),
        Some(json.clone()),
    )
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
    checkpoint_resolution: Option<ResolvedCheckpointPlan>,
    moshi_recipes: Option<AtomicRecipeSet>,
}

/// Typed, normalized family configuration retained from GGUF admission.
#[derive(Debug, Clone)]
pub enum GgufModelConfig {
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
    /// LFM2 family geometry.
    Lfm2(crate::lfm2::ModelArgs),
    /// Llama-compatible family geometry.
    Llama(crate::llama::ModelArgs),
    /// Muse-Glimmer family geometry.
    MuseGlimmer(crate::muse_glimmer::DecoderConfig),
    /// Nemotron-H family geometry.
    NemotronH(crate::nemotron_h::ModelArgs),
    /// Qwen2/Qwen3 family geometry, including Qwen3-VL text checkpoints.
    Qwen(crate::qwen::ModelArgs),
    /// Qwen hybrid-family geometry.
    QwenHybrid(crate::qwen::hybrid::ParsedHybridConfig),
}

impl SafetensorsModelConfig {
    /// Whether this exact normalized configuration constructs grouped routed experts.
    pub const fn uses_grouped_routed_experts(&self) -> bool {
        match self {
            Self::DeepSeekV3(args) => args.n_routed_experts > 0,
            Self::DeepSeekV4(args) => args.n_routed_experts > 0,
            Self::Gemma4(args) => matches!(args.text.num_experts, Some(count) if count > 0),
            Self::GptOss(args) => args.num_local_experts > 0,
            Self::Inkling(args) => args.text_config.n_routed_experts > 0,
            Self::KimiLinear(args) => args.num_experts > 0,
            Self::Llama(_) | Self::Moshi(_) => false,
            Self::MuseGlimmer(args) => args.num_experts > 0,
            Self::Lfm2(args) => args.num_experts > 0,
            Self::NemotronH(args) => args.n_routed_experts > 0,
            Self::Qwen(args) => args.num_experts > 0,
            Self::QwenHybrid(args) => args.text.num_experts > 0,
            Self::QwenVl(args) => args.text.num_experts > 0,
        }
    }
}

impl GgufModelConfig {
    /// Whether this exact normalized configuration constructs grouped routed experts.
    pub const fn uses_grouped_routed_experts(&self) -> bool {
        match self {
            Self::DeepSeekV3(args) => args.n_routed_experts > 0,
            Self::DeepSeekV4(args) => args.n_routed_experts > 0,
            Self::Gemma4(args) => matches!(args.text.num_experts, Some(count) if count > 0),
            Self::GptOss(args) => args.num_local_experts > 0,
            Self::Inkling(args) => args.text_config.n_routed_experts > 0,
            Self::KimiLinear(args) => args.num_experts > 0,
            Self::Lfm2(args) => args.num_experts > 0,
            Self::Llama(_) => false,
            Self::MuseGlimmer(args) => args.num_experts > 0,
            Self::NemotronH(args) => args.n_routed_experts > 0,
            Self::Qwen(args) => args.num_experts > 0,
            Self::QwenHybrid(args) => args.text.num_experts > 0,
        }
    }
}

/// GGUF architecture geometry and checkpoint schema proven valid at admission.
#[derive(Debug, Clone)]
pub struct GgufArchitecturePlan {
    architecture: GgufArchitecture,
    model: GgufModelConfig,
    checkpoint: GgufCheckpointPlan,
    tensor_mapping: Vec<eredu_gguf::TranslatedTensorLayout>,
}

impl GgufArchitecturePlan {
    pub(crate) fn new(
        architecture: GgufArchitecture,
        model: GgufModelConfig,
        checkpoint: GgufCheckpointPlan,
        tensor_mapping: Vec<eredu_gguf::TranslatedTensorLayout>,
    ) -> Self {
        Self {
            architecture,
            model,
            checkpoint,
            tensor_mapping,
        }
    }

    /// Exact GGUF architecture selected for this checkpoint.
    pub const fn architecture(&self) -> GgufArchitecture {
        self.architecture
    }

    /// Canonical family selected for this exact checkpoint.
    pub const fn model_kind(&self) -> ModelKind {
        self.architecture.model_kind()
    }

    /// Typed normalized family configuration.
    pub const fn model(&self) -> &GgufModelConfig {
        &self.model
    }

    /// Complete expected GGUF catalog derived from the normalized geometry.
    pub const fn checkpoint(&self) -> &GgufCheckpointPlan {
        &self.checkpoint
    }

    /// Canonical physical-to-logical tensor mapping resolved during admission.
    pub fn tensor_mapping(&self) -> &[eredu_gguf::TranslatedTensorLayout] {
        &self.tensor_mapping
    }
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

    /// Exact physical layout selected and proven during catalog admission.
    ///
    /// This is absent before a configuration-only plan is finalized against an
    /// artifact catalog.
    pub const fn checkpoint_resolution(&self) -> Option<&ResolvedCheckpointPlan> {
        self.checkpoint_resolution.as_ref()
    }

    /// Canonical Moshi binding recipes proven against the admitted catalog.
    pub const fn moshi_recipes(&self) -> Option<&AtomicRecipeSet> {
        self.moshi_recipes.as_ref()
    }

    /// Separates an admitted embedded-prediction artifact into its ordinary
    /// target architecture and an exact prediction-extension contract.
    ///
    /// This projection does not re-admit a different artifact: an exact subset
    /// of the original checkpoint resolution remains the authority for target
    /// source tensors, while the extension retains the omitted admitted
    /// sources under its own typed ownership.
    pub fn prediction_target_projection(
        &self,
    ) -> Result<Option<(Self, PredictionExtensionPlan)>, ArtifactError> {
        let complete_architecture = self.clone();
        let (model, extension) = match &self.model {
            SafetensorsModelConfig::DeepSeekV3(args) if args.num_nextn_predict_layers > 0 => {
                let depth = usize::try_from(args.num_nextn_predict_layers).map_err(|_| {
                    ArtifactError::InvalidArchitecturePlan(
                        "DeepSeek-V3 prediction depth exceeds usize".into(),
                    )
                })?;
                (
                    SafetensorsModelConfig::DeepSeekV3(args.prediction_target().map_err(
                        |error| ArtifactError::InvalidArchitecturePlan(error.to_string()),
                    )?),
                    PredictionExtensionPlan::new(
                        PredictionExtensionKind::DeepSeekV3Mtp,
                        depth,
                        complete_architecture.clone(),
                    )?,
                )
            }
            SafetensorsModelConfig::DeepSeekV4(args) if args.num_nextn_predict_layers > 0 => {
                let depth = usize::try_from(args.num_nextn_predict_layers).map_err(|_| {
                    ArtifactError::InvalidArchitecturePlan(
                        "DeepSeek-V4 prediction depth exceeds usize".into(),
                    )
                })?;
                (
                    SafetensorsModelConfig::DeepSeekV4(args.prediction_target().map_err(
                        |error| ArtifactError::InvalidArchitecturePlan(error.to_string()),
                    )?),
                    PredictionExtensionPlan::new(
                        PredictionExtensionKind::DeepSeekV4Embedded,
                        depth,
                        complete_architecture.clone(),
                    )?,
                )
            }
            SafetensorsModelConfig::Inkling(args)
                if args
                    .mtp_config
                    .as_ref()
                    .is_some_and(|mtp| mtp.num_nextn_predict_layers > 0) =>
            {
                let depth = usize::try_from(
                    args.mtp_config
                        .as_ref()
                        .expect("guarded Inkling prediction configuration")
                        .num_nextn_predict_layers,
                )
                .map_err(|_| {
                    ArtifactError::InvalidArchitecturePlan(
                        "Inkling prediction depth exceeds usize".into(),
                    )
                })?;
                let mut target = args.clone();
                target.mtp_config = None;
                (
                    SafetensorsModelConfig::Inkling(target),
                    PredictionExtensionPlan::new(
                        PredictionExtensionKind::InklingMtp,
                        depth,
                        complete_architecture.clone(),
                    )?,
                )
            }
            SafetensorsModelConfig::QwenHybrid(args) if args.text.mtp_num_hidden_layers > 0 => {
                let depth = usize::try_from(args.text.mtp_num_hidden_layers).map_err(|_| {
                    ArtifactError::InvalidArchitecturePlan(
                        "Qwen hybrid prediction depth exceeds usize".into(),
                    )
                })?;
                let mut target = args.clone();
                target.text.mtp_num_hidden_layers = 0;
                (
                    SafetensorsModelConfig::QwenHybrid(target),
                    PredictionExtensionPlan::new(
                        PredictionExtensionKind::QwenHybridMtp,
                        depth,
                        complete_architecture.clone(),
                    )?,
                )
            }
            SafetensorsModelConfig::NemotronH(args) if args.num_nextn_predict_layers > 0 => {
                let depth = usize::try_from(args.num_nextn_predict_layers).map_err(|_| {
                    ArtifactError::InvalidArchitecturePlan(
                        "Nemotron-H prediction depth exceeds usize".into(),
                    )
                })?;
                (
                    SafetensorsModelConfig::NemotronH(args.prediction_target().map_err(
                        |error| ArtifactError::InvalidArchitecturePlan(error.to_string()),
                    )?),
                    PredictionExtensionPlan::new(
                        PredictionExtensionKind::NemotronHMtp,
                        depth,
                        complete_architecture.clone(),
                    )?,
                )
            }
            _ => return Ok(None),
        };
        let checkpoint = match &model {
            SafetensorsModelConfig::DeepSeekV3(args) => {
                crate::deepseek::v3_safetensors_plan(args, true)
            }
            SafetensorsModelConfig::DeepSeekV4(args) => crate::deepseek::v4_safetensors_plan(args),
            SafetensorsModelConfig::Inkling(args) => crate::inkling::safetensors_plan(args),
            SafetensorsModelConfig::QwenHybrid(args) => {
                crate::qwen::hybrid::composite_safetensors_plan(args)
            }
            SafetensorsModelConfig::NemotronH(args) => crate::nemotron_h::safetensors_plan(args),
            _ => unreachable!("prediction projection admits only typed extension families"),
        }
        .map_err(ArtifactError::InvalidArchitecturePlan)?;
        let checkpoint_resolution = self
            .checkpoint_resolution
            .as_ref()
            .map(|resolution| {
                let target = Self {
                    kind: self.kind,
                    model: model.clone(),
                    checkpoint: checkpoint.clone(),
                    checkpoint_resolution: None,
                    moshi_recipes: None,
                };
                let extension_sources = extension.source_keys(&target)?;
                let target_sources = resolution
                    .source_keys()
                    .difference(&extension_sources)
                    .cloned()
                    .collect();
                resolution
                    .project_claimed_sources(checkpoint.identity.clone(), target_sources)
                    .map_err(ArtifactError::InvalidArchitecturePlan)
            })
            .transpose()?;
        Ok(Some((
            Self {
                kind: self.kind,
                model,
                checkpoint,
                checkpoint_resolution,
                moshi_recipes: None,
            },
            extension,
        )))
    }

    pub(crate) fn admit_catalog(&mut self, tensors: &TensorCatalog) -> Result<(), ArtifactError> {
        let catalog = PortableSafetensorsCatalog(tensors);
        self.checkpoint_resolution = Some(
            resolve_safetensors_plan(&catalog, &self.checkpoint)
                .map_err(invalid_checkpoint_validation)?,
        );
        if let SafetensorsModelConfig::Moshi(config) = &self.model {
            self.moshi_recipes = Some(crate::moshi::canonical_recipes(config, &catalog).map_err(
                |error| {
                    ArtifactError::InvalidArchitecturePlan(format!(
                        "invalid Moshi checkpoint recipes: {error}"
                    ))
                },
            )?);
        }
        Ok(())
    }
}

/// Architecture identity of an adapter-owned prediction extension.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PredictionExtensionKind {
    /// DeepSeek-V3/R1 embedded multi-token prediction.
    DeepSeekV3Mtp,
    /// DeepSeek-V4 embedded multi-token prediction or DSpark proposal extension.
    DeepSeekV4Embedded,
    /// Inkling embedded multi-token prediction.
    InklingMtp,
    /// Qwen3.5/Qwen3-Next hybrid embedded multi-token prediction.
    QwenHybridMtp,
    /// Nemotron-H patterned embedded multi-token prediction.
    NemotronHMtp,
}

/// Exact prediction extension separated from an ordinary target architecture.
#[derive(Debug, Clone)]
pub struct PredictionExtensionPlan {
    kind: PredictionExtensionKind,
    depth: usize,
    complete_architecture: SafetensorsArchitecturePlan,
}

impl PredictionExtensionPlan {
    fn new(
        kind: PredictionExtensionKind,
        depth: usize,
        complete_architecture: SafetensorsArchitecturePlan,
    ) -> Result<Self, ArtifactError> {
        if depth == 0 {
            return Err(ArtifactError::InvalidArchitecturePlan(
                "prediction extension depth must be positive".into(),
            ));
        }
        Ok(Self {
            kind,
            depth,
            complete_architecture,
        })
    }

    /// Exact extension architecture selected from the complete artifact.
    pub const fn kind(&self) -> PredictionExtensionKind {
        self.kind
    }

    /// Ordered embedded prediction depth.
    pub const fn depth(&self) -> usize {
        self.depth
    }

    /// Complete admitted architecture that owns the extension equations and sources.
    ///
    /// The ordinary target projection deliberately removes prediction policy;
    /// this retained value preserves DSpark/MTP mode, prediction-layer
    /// schedules, and every family dimension needed to materialize only the
    /// extension.
    pub const fn complete_architecture(&self) -> &SafetensorsArchitecturePlan {
        &self.complete_architecture
    }

    /// Returns the exact admitted physical sources owned only by this extension.
    ///
    /// The target projection and complete artifact share one already-admitted
    /// source resolution. This subtraction uses architecture-declared target
    /// keys and aliases, so backend composition can reserve extension payloads
    /// without inferring a family prefix or weakening strict target loading.
    pub fn source_keys(
        &self,
        target: &SafetensorsArchitecturePlan,
    ) -> Result<BTreeSet<String>, ArtifactError> {
        let resolution = self
            .complete_architecture
            .checkpoint_resolution()
            .ok_or_else(|| {
                ArtifactError::InvalidArchitecturePlan(
                    "prediction extension has no admitted checkpoint resolution".into(),
                )
            })?;
        let target_names = target
            .checkpoint()
            .common_tensors
            .iter()
            .chain(
                target
                    .checkpoint()
                    .layout_groups
                    .iter()
                    .flat_map(|group| &group.variants)
                    .flat_map(|variant| &variant.tensors),
            )
            .flat_map(|tensor| {
                std::iter::once(tensor.key.as_str())
                    .chain(tensor.aliases.iter().map(String::as_str))
            })
            .collect::<BTreeSet<_>>();
        let extension = resolution
            .source_keys()
            .iter()
            .filter(|key| !target_names.contains(key.as_str()))
            .cloned()
            .collect::<BTreeSet<_>>();
        if extension.is_empty() {
            return Err(ArtifactError::InvalidArchitecturePlan(
                "prediction extension owns no admitted checkpoint sources".into(),
            ));
        }
        Ok(extension)
    }

    /// Maximum logical rank and element count of the target value retained for this extension.
    ///
    /// The ordinary target publication group carries this value in addition to logits.  Keeping
    /// the bound on the architecture-selected extension prevents a backend from inferring hidden
    /// capture geometry from a family name or from a native tensor observed after construction.
    pub fn target_capture_limits(
        &self,
        maximum_batch_size: i32,
        maximum_sequence_length: i32,
    ) -> Result<(usize, usize), ArtifactError> {
        let batch = usize::try_from(maximum_batch_size).map_err(|_| {
            ArtifactError::InvalidArchitecturePlan(
                "prediction target maximum batch size is not positive".into(),
            )
        })?;
        let sequence = usize::try_from(maximum_sequence_length).map_err(|_| {
            ArtifactError::InvalidArchitecturePlan(
                "prediction target maximum sequence length is not positive".into(),
            )
        })?;
        if batch == 0 || sequence == 0 {
            return Err(ArtifactError::InvalidArchitecturePlan(
                "prediction target invocation bounds must be positive".into(),
            ));
        }
        let (rank, dimensions) = match self.complete_architecture.model() {
            SafetensorsModelConfig::DeepSeekV3(args) => (3, vec![args.hidden_size]),
            SafetensorsModelConfig::DeepSeekV4(args) if args.dspark.is_none() => {
                (4, vec![args.hc_mult, args.hidden_size])
            }
            SafetensorsModelConfig::DeepSeekV4(_) => {
                return Err(ArtifactError::InvalidArchitecturePlan(
                    "DSpark target capture requires a dedicated typed target projection".into(),
                ));
            }
            SafetensorsModelConfig::Inkling(args) => (3, vec![args.text_config.hidden_size]),
            SafetensorsModelConfig::QwenHybrid(args) => (3, vec![args.text.hidden_size]),
            SafetensorsModelConfig::NemotronH(args) => (3, vec![args.hidden_size]),
            _ => {
                return Err(ArtifactError::InvalidArchitecturePlan(
                    "prediction extension has no target capture geometry".into(),
                ));
            }
        };
        let initial = batch.checked_mul(sequence).ok_or_else(|| {
            ArtifactError::InvalidArchitecturePlan(
                "prediction target capture geometry overflowed".into(),
            )
        })?;
        let elements = dimensions
            .into_iter()
            .try_fold(initial, |elements, dimension| {
                usize::try_from(dimension)
                    .ok()
                    .and_then(|dimension| elements.checked_mul(dimension))
            })
            .ok_or_else(|| {
                ArtifactError::InvalidArchitecturePlan(
                    "prediction target capture geometry overflowed".into(),
                )
            })?;
        Ok((rank, elements))
    }
}

pub(crate) struct PortableSafetensorsCatalog<'a>(pub(crate) &'a TensorCatalog);

impl SafetensorsCatalog for PortableSafetensorsCatalog<'_> {
    fn keys(&self) -> Vec<String> {
        self.0
            .descriptors()
            .map(|tensor| tensor.name.clone())
            .collect()
    }

    fn metadata(&self, key: &str) -> Result<CatalogTensorMetadata, String> {
        let tensor = self
            .0
            .get(key)
            .ok_or_else(|| format!("unknown checkpoint tensor {key:?}"))?;
        Ok(CatalogTensorMetadata {
            shape: tensor.shape.clone(),
            stored_dtype: portable_stored_dtype(&tensor.dtype),
        })
    }
}

impl RecipeCatalog for PortableSafetensorsCatalog<'_> {
    fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        let tensor = self
            .0
            .get(key)
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })?;
        Ok(TensorMetadata {
            name: tensor.name.clone(),
            logical_shape: tensor.shape.clone(),
            physical_shape: tensor.shape.clone(),
            stored_dtype: portable_stored_dtype(&tensor.dtype),
            encoded_byte_len: tensor.storage.as_ref().map_or(0, |storage| storage.length),
            backing_shard: tensor
                .storage
                .as_ref()
                .map(|storage| storage.member.clone().into()),
        })
    }
}

fn portable_stored_dtype(dtype: &TensorDtype) -> StoredDtype {
    match dtype {
        TensorDtype::Bool => StoredDtype::Bool,
        TensorDtype::U8 => StoredDtype::U8,
        TensorDtype::I8 => StoredDtype::I8,
        TensorDtype::I16 => StoredDtype::I16,
        TensorDtype::U16 => StoredDtype::U16,
        TensorDtype::F16 => StoredDtype::F16,
        TensorDtype::Bf16 => StoredDtype::BF16,
        TensorDtype::I32 => StoredDtype::I32,
        TensorDtype::U32 => StoredDtype::U32,
        TensorDtype::F32 => StoredDtype::F32,
        TensorDtype::F64 => StoredDtype::F64,
        TensorDtype::I64 => StoredDtype::I64,
        TensorDtype::U64 => StoredDtype::U64,
        TensorDtype::Complex64 => StoredDtype::C64,
        TensorDtype::Encoded(name) if name == "F8_E4M3" => StoredDtype::F8E4M3,
        TensorDtype::Encoded(name) if name == "F4" => StoredDtype::F4,
        TensorDtype::Encoded(name) if name == "F8_E8M0" => StoredDtype::F8E8M0,
        TensorDtype::Encoded(name) if name == "F8_E5M2" => StoredDtype::F8E5M2,
        TensorDtype::Encoded(name) => StoredDtype::Other(name.clone()),
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
            crate::qwen::hybrid::composite_safetensors_plan(args)
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
        checkpoint_resolution: None,
        moshi_recipes: None,
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
    use eredu_checkpoint::schema::{SafetensorsTensorConstraint, StoredDtypeConstraint};
    use eredu_gguf::{GgmlType, MetadataValue, TensorInput, Writer};
    use std::{collections::BTreeMap, fs::File};

    fn test_catalog(plan: &SafetensorsCheckpointPlan) -> TensorCatalog {
        let constraints = plan.common_tensors.iter().chain(
            plan.layout_groups
                .iter()
                .filter_map(|group| group.variants.first())
                .flat_map(|variant| variant.tensors.iter()),
        );
        TensorCatalog::new(constraints.map(test_descriptor)).unwrap()
    }

    fn test_descriptor(
        tensor: &SafetensorsTensorConstraint,
    ) -> eredu_core::checkpoint::TensorDescriptor {
        let stored = match &tensor.dtype {
            StoredDtypeConstraint::Exact(dtype) => dtype,
            StoredDtypeConstraint::OneOf(dtypes) => dtypes.first().unwrap(),
            StoredDtypeConstraint::Floating => &StoredDtype::F32,
        };
        eredu_core::checkpoint::TensorDescriptor {
            name: tensor.key.clone(),
            shape: tensor.shape.clone(),
            dtype: match stored {
                StoredDtype::Bool => TensorDtype::Bool,
                StoredDtype::U8 => TensorDtype::U8,
                StoredDtype::I8 => TensorDtype::I8,
                StoredDtype::I16 => TensorDtype::I16,
                StoredDtype::U16 => TensorDtype::U16,
                StoredDtype::F16 => TensorDtype::F16,
                StoredDtype::BF16 => TensorDtype::Bf16,
                StoredDtype::I32 => TensorDtype::I32,
                StoredDtype::U32 => TensorDtype::U32,
                StoredDtype::F32 => TensorDtype::F32,
                StoredDtype::F64 => TensorDtype::F64,
                StoredDtype::I64 => TensorDtype::I64,
                StoredDtype::U64 => TensorDtype::U64,
                StoredDtype::C64 => TensorDtype::Complex64,
                StoredDtype::F8E4M3 => TensorDtype::Encoded("F8_E4M3".into()),
                StoredDtype::F4 => TensorDtype::Encoded("F4".into()),
                StoredDtype::F8E8M0 => TensorDtype::Encoded("F8_E8M0".into()),
                StoredDtype::F8E5M2 => TensorDtype::Encoded("F8_E5M2".into()),
                StoredDtype::Other(name) => TensorDtype::Encoded(name.clone()),
            },
            storage: None,
        }
    }

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

    fn tiny_moshi_config() -> Value {
        serde_json::json!({
            "model_type": "moshi", "dim": 32, "text_card": 101,
            "n_q": 4, "dep_q": 3, "generated_audio_codebooks": 2, "card": 64,
            "num_heads": 4, "num_layers": 2, "dim_feedforward": 48,
            "causal": true, "context": 7, "max_period": 10000.0,
            "positional_embedding": "rope", "depformer_dim": 24,
            "depformer_dim_feedforward": 36, "depformer_num_heads": 4,
            "depformer_num_layers": 2, "depformer_context": 3,
            "depformer_max_period": 10000.0, "depformer_pos_emb": "none",
            "delays": [0, 0, 1, 2, 1]
        })
    }

    fn tiny_deepseek_prediction_config() -> Value {
        serde_json::json!({
            "architectures": ["DeepseekV3ForCausalLM"],
            "model_type": "deepseek_v3", "hidden_size": 16,
            "intermediate_size": 32, "moe_intermediate_size": 8,
            "num_hidden_layers": 4, "num_attention_heads": 2,
            "vocab_size": 128, "max_position_embeddings": 4096,
            "q_lora_rank": 4, "kv_lora_rank": 4,
            "qk_nope_head_dim": 6, "qk_rope_head_dim": 2, "v_head_dim": 8,
            "first_k_dense_replace": 1, "moe_layer_freq": 2,
            "n_routed_experts": 8, "n_shared_experts": 1,
            "num_experts_per_tok": 2, "n_group": 2, "topk_group": 1,
            "topk_method": "noaux_tc", "scoring_func": "sigmoid",
            "norm_topk_prob": true, "routed_scaling_factor": 1.0,
            "tie_word_embeddings": false, "attention_dropout": 0.0,
            "hidden_act": "silu", "num_nextn_predict_layers": 1
        })
    }

    #[test]
    fn prediction_extension_contract_rejects_identity_and_depth_drift() {
        let mut extension = resolve_model_config(&tiny_deepseek_prediction_config())
            .unwrap()
            .architecture
            .prediction_target_projection()
            .unwrap()
            .unwrap()
            .1;
        crate::prediction_extension::validate_extension_contract(&extension).unwrap();

        extension.depth = 2;
        let error = crate::prediction_extension::validate_extension_contract(&extension)
            .unwrap_err()
            .to_string();
        assert!(error.contains("differs from admitted architecture depth"));

        extension.depth = 1;
        extension.kind = PredictionExtensionKind::QwenHybridMtp;
        let error = crate::prediction_extension::validate_extension_contract(&extension)
            .unwrap_err()
            .to_string();
        assert!(error.contains("identity does not match"));
    }

    #[test]
    fn moshi_artifact_plan_retains_catalog_admission_recipes() {
        let (configuration, resolved_plan) = MODEL_CONFIGURATIONS
            .resolve_safetensors(&tiny_moshi_config())
            .unwrap()
            .into_parts();
        let checkpoint = resolved_plan
            .safetensors_architecture()
            .unwrap()
            .checkpoint();
        let tensors = TensorCatalog::new(checkpoint.common_tensors.iter().map(|tensor| {
            eredu_core::checkpoint::TensorDescriptor {
                name: tensor.key.clone(),
                shape: tensor.shape.clone(),
                dtype: TensorDtype::F32,
                storage: None,
            }
        }))
        .unwrap();

        let admitted = MODEL_CONFIGURATIONS
            .artifact_plan(
                Path::new("fixture"),
                ArtifactFormat::SafeTensors,
                &configuration,
                &tensors,
                None,
                resolved_plan,
            )
            .unwrap();
        let plan = admitted.safetensors_architecture().unwrap();

        assert_eq!(plan.model_kind(), ModelKind::Moshi);
        assert!(plan.moshi_recipes().is_some());
        assert!(plan
            .moshi_recipes()
            .unwrap()
            .get("transformer.layers.0.self_attn.in_proj.weight")
            .is_some());
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
        let tensors = test_catalog(
            resolved_plan
                .safetensors_architecture()
                .unwrap()
                .checkpoint(),
        );
        let plan = MODEL_CONFIGURATIONS
            .artifact_plan(
                root.path(),
                ArtifactFormat::SafeTensors,
                &configuration,
                &tensors,
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
        let next_tensors = test_catalog(
            next_resolved_plan
                .safetensors_architecture()
                .unwrap()
                .checkpoint(),
        );
        let next = MODEL_CONFIGURATIONS
            .artifact_plan(
                root.path(),
                ArtifactFormat::SafeTensors,
                &next_configuration,
                &next_tensors,
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
    fn portable_safetensors_admission_validates_non_moshi_checkpoint_schema() {
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "intermediate_size": 8,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "rms_norm_eps": 0.00001,
            "vocab_size": 8,
            "max_position_embeddings": 32,
            "tie_word_embeddings": false,
            "attention_bias": false,
            "mlp_bias": false
        });
        let (configuration, resolved_plan) = MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap()
            .into_parts();
        let catalog = test_catalog(
            resolved_plan
                .safetensors_architecture()
                .unwrap()
                .checkpoint(),
        );
        let admitted = MODEL_CONFIGURATIONS
            .artifact_plan(
                Path::new("fixture"),
                ArtifactFormat::SafeTensors,
                &configuration,
                &catalog,
                None,
                resolved_plan.clone(),
            )
            .unwrap();
        assert!(admitted
            .safetensors_architecture()
            .unwrap()
            .checkpoint_resolution()
            .is_some());

        let error = MODEL_CONFIGURATIONS
            .artifact_plan(
                Path::new("fixture"),
                ArtifactFormat::SafeTensors,
                &configuration,
                &TensorCatalog::new([]).unwrap(),
                None,
                resolved_plan,
            )
            .unwrap_err();

        assert!(matches!(
            error,
            ArtifactError::InvalidArtifact(detail)
                if detail.contains("model.embed_tokens.weight")
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
    fn model_kind_serialization_uses_resolvable_canonical_names() {
        for kind in ModelKind::ALL {
            let serialized = serde_json::to_string(&kind).unwrap();
            assert_eq!(
                serialized,
                serde_json::to_string(kind.canonical_name()).unwrap()
            );

            let family_name: String = serde_json::from_str(&serialized).unwrap();
            assert_eq!(ModelKind::resolve_family(&family_name).unwrap(), kind);
            assert_eq!(
                serde_json::from_str::<ModelKind>(&serialized).unwrap(),
                kind
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
    fn known_wrapper_cannot_delegate_to_a_different_nested_text_family() {
        let error = resolve_model_identity(&serde_json::json!({
            "model_type": "qwen3_vl",
            "text_config": { "model_type": "llama" }
        }))
        .unwrap_err();
        assert!(matches!(
            error,
            ArtifactError::InvalidArtifact(message)
                if message.contains("but nested text model type \"llama\" resolves to family \"llama\"")
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
            vec![required_quantized.clone()]
        );
        assert_eq!(
            gguf_companion_requirements(GgufArchitecture::Qwen3VlMoe).unwrap(),
            vec![required_quantized.clone()]
        );
        let optional_quantized = GgufCompanionRequirement::new(
            GgufCompanionRole::MediaProjector,
            false,
            "mmproj",
            1,
            GgufCompanionEncoding::DensePreferred,
        )
        .unwrap();
        assert_eq!(
            gguf_companion_requirements(GgufArchitecture::MuseGlimmer).unwrap(),
            vec![optional_quantized]
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

    #[test]
    fn gguf_structural_admission_ignores_malformed_tokenizer_policy() {
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
            (
                "tokenizer.ggml.tokens".into(),
                MetadataValue::String("wrong type".into()),
            ),
            (
                "tokenizer.ggml.eos_token_id".into(),
                MetadataValue::String("wrong type".into()),
            ),
            (
                "tokenizer.ggml.padding_token_id".into(),
                MetadataValue::String("wrong type".into()),
            ),
        ]);
        const VECTOR: &[u64] = &[1];
        const MATRIX: &[u64] = &[1, 1];
        let tensors = [
            TensorInput {
                name: "token_embd.weight",
                dimensions: MATRIX,
                ggml_type: GgmlType::F32,
                data: &scalar,
            },
            TensorInput {
                name: "output_norm.weight",
                dimensions: VECTOR,
                ggml_type: GgmlType::F32,
                data: &scalar,
            },
            TensorInput {
                name: "blk.0.attn_norm.weight",
                dimensions: VECTOR,
                ggml_type: GgmlType::F32,
                data: &scalar,
            },
            TensorInput {
                name: "blk.0.ffn_norm.weight",
                dimensions: VECTOR,
                ggml_type: GgmlType::F32,
                data: &scalar,
            },
            TensorInput {
                name: "blk.0.attn_q.weight",
                dimensions: MATRIX,
                ggml_type: GgmlType::F32,
                data: &scalar,
            },
            TensorInput {
                name: "blk.0.attn_k.weight",
                dimensions: MATRIX,
                ggml_type: GgmlType::F32,
                data: &scalar,
            },
            TensorInput {
                name: "blk.0.attn_v.weight",
                dimensions: MATRIX,
                ggml_type: GgmlType::F32,
                data: &scalar,
            },
            TensorInput {
                name: "blk.0.attn_output.weight",
                dimensions: MATRIX,
                ggml_type: GgmlType::F32,
                data: &scalar,
            },
            TensorInput {
                name: "blk.0.ffn_gate.weight",
                dimensions: MATRIX,
                ggml_type: GgmlType::F32,
                data: &scalar,
            },
            TensorInput {
                name: "blk.0.ffn_up.weight",
                dimensions: MATRIX,
                ggml_type: GgmlType::F32,
                data: &scalar,
            },
            TensorInput {
                name: "blk.0.ffn_down.weight",
                dimensions: MATRIX,
                ggml_type: GgmlType::F32,
                data: &scalar,
            },
        ];
        Writer::default()
            .write(File::create(&path).unwrap(), &metadata, &tensors)
            .unwrap();

        let inspection = inspect_artifact(&path).unwrap();
        let plan = inspection.architecture_plan().gguf_plan().unwrap();
        let GgufModelConfig::Llama(args) = plan.model() else {
            panic!("expected Llama GGUF plan");
        };
        assert_eq!(args.vocab_size, 1);
        assert_eq!(plan.tensor_mapping().len(), tensors.len());
        let query = plan
            .tensor_mapping()
            .iter()
            .find(|mapped| mapped.physical_name == "blk.0.attn_q.weight")
            .unwrap();
        assert_eq!(query.original_name, "blk.0.attn_q.weight");
        assert_eq!(query.layout.name, "model.layers.0.self_attn.q_proj.weight");
    }
}

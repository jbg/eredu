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
    ArtifactFormat, GgufCompanionEncoding, GgufCompanionRequirement, ArtifactCompanionRole,
    LoadingProtocol, ModelConfiguration, ModelConfigurationResolver, ResolvedModelConfiguration,
    ValidatedGguf,
};
use eredu_gguf::Checkpoint as GgufCheckpoint;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::{collections::BTreeSet, path::Path};

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
        checkpoint: &GgufCheckpoint,
    ) -> Result<Vec<GgufCompanionRequirement>, ArtifactError> {
        if matches!(architecture, "dream" | "mistral3" | "nemotron_labs_diffusion") {
            let admitted = crate::nemotron_labs_diffusion::gguf::TextGgufPlan::from_checkpoint(checkpoint)
                .map_err(|error| ArtifactError::InvalidArtifact(error.to_string()))?;
            if admitted.config().vision.is_none() { return Ok(Vec::new()); }
            return GgufCompanionRequirement::new(ArtifactCompanionRole::MediaProjector, false,
                "mmproj", 1, GgufCompanionEncoding::DenseRequired).map(|requirement| vec![requirement]);
        }
        gguf_companion_requirements(GgufArchitecture::resolve(architecture)?)
    }

    fn sidecar_names(&self, plan: &Self::ArtifactPlan) -> &[&str] {
        match plan.model_kind() {
            ModelKind::DiffusionGemma
            | ModelKind::Gemma4
            | ModelKind::Qwen3Vl
            | ModelKind::Qwen3VlMoe
            | ModelKind::Qwen35
            | ModelKind::Qwen4Exp => &[
                crate::processor_plan::PROCESSOR_CONFIG_FILENAME,
                crate::processor_plan::VIDEO_PROCESSOR_CONFIG_FILENAME,
            ],
            ModelKind::MuseGlimmer | ModelKind::MuseSpark => &[crate::processor_plan::MUSE_PROCESSOR_CONFIG_FILENAME],
            _ => &[],
        }
    }

    fn artifact_plan(
        &self,
        sidecars: &std::collections::BTreeMap<String, Vec<u8>>,
        format: ArtifactFormat,
        configuration: &ModelConfiguration,
        tensors: &TensorCatalog,
        validated_gguf: Option<&ValidatedGguf>,
        resolved_plan: Self::ArtifactPlan,
    ) -> Result<Self::ArtifactPlan, ArtifactError> {
        let plan = match format {
            ArtifactFormat::SafeTensors => {
                let resolved_plan = resolved_plan.admit_safetensors_catalog(tensors, configuration.json().ok_or_else(|| ArtifactError::InvalidArtifact("SafeTensors inspection omitted its source configuration".into()))?)?;
                let kind = resolved_plan.model_kind();
                let model = serde_json::to_vec(configuration.json().ok_or_else(|| {
                    ArtifactError::InvalidArtifact(
                        "SafeTensors inspection omitted normalized JSON configuration".into(),
                    )
                })?)?;
                let (image, video) = if matches!(
                    kind,
                    ModelKind::DiffusionGemma
                        | ModelKind::Gemma4
                        | ModelKind::Qwen3Vl
                        | ModelKind::Qwen3VlMoe
                        | ModelKind::Qwen35
                        | ModelKind::Qwen4Exp
                ) {
                    (
                        sidecars
                            .get(crate::processor_plan::PROCESSOR_CONFIG_FILENAME)
                            .cloned(),
                        sidecars
                            .get(crate::processor_plan::VIDEO_PROCESSOR_CONFIG_FILENAME)
                            .cloned(),
                    )
                } else {
                    (None, None)
                };
                let muse = if matches!(kind, ModelKind::MuseGlimmer | ModelKind::MuseSpark) {
                    sidecars
                        .get(crate::processor_plan::MUSE_PROCESSOR_CONFIG_FILENAME)
                        .cloned()
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
                    .companion(&ArtifactCompanionRole::MediaProjector)
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
        ArtifactCompanionRole::MediaProjector,
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
    /// DiffusionGemma shared context and iterative block refinement.
    DiffusionGemma,
    /// Gemma 4 text or unified multimodal architecture.
    Gemma4,
    /// OpenAI GPT-OSS sparse decoder.
    GptOss,
    /// Thinking Machines Lab Inkling multimodal architecture.
    Inkling,
    /// IFM K2 Horizon dense, MoE, and routed-value architectures.
    K2Horizon,
    /// Moonshot Kimi Linear architecture.
    KimiLinear,
    /// Llama-compatible dense decoders, including Mistral and Phi-4.
    Llama,
    /// Gemma 2 dense decoder.
    Gemma2,
    /// Nanbeige dense shared-weight repeated decoder.
    Nanbeige,
    /// Meta Muse-Glimmer multimodal decoder.
    MuseGlimmer,
    /// Meta Muse Spark Open multimodal decoder.
    MuseSpark,
    /// Liquid AI LFM2/LFM2.5 dense or MoE architecture.
    Lfm2,
    /// Nemotron-H hybrid architecture.
    NemotronH,
    /// Nemotron-Labs-Diffusion shared text backbone and optional Pixtral context.
    NemotronLabsDiffusion,
    /// Moshi-family realtime speech architecture, including PersonaPlex.
    Moshi,
    /// NemotronLabs VoiceChat coupled realtime speech architecture.
    VoiceChat,
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
    /// Qwen3.8 Flash-Next gated multi-stream and indexed-attention family.
    Qwen4Exp,
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
    pub const ALL: [Self; 25] = [
        Self::DeepSeekV3,
        Self::DeepSeekV4,
        Self::DiffusionGemma,
        Self::Gemma4,
        Self::GptOss,
        Self::Inkling,
        Self::K2Horizon,
        Self::KimiLinear,
        Self::Llama,
        Self::Gemma2,
        Self::Nanbeige,
        Self::MuseGlimmer,
        Self::MuseSpark,
        Self::Lfm2,
        Self::NemotronH,
        Self::NemotronLabsDiffusion,
        Self::Moshi,
        Self::VoiceChat,
        Self::Qwen2,
        Self::Qwen3,
        Self::Qwen3Next,
        Self::Qwen3Vl,
        Self::Qwen3VlMoe,
        Self::Qwen35,
        Self::Qwen4Exp,
    ];

    /// Canonical family name published through the neutral artifact protocol.
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::DeepSeekV3 => "deepseek_v3",
            Self::DeepSeekV4 => "deepseek_v4",
            Self::DiffusionGemma => "diffusion_gemma",
            Self::Gemma4 => "gemma4",
            Self::GptOss => "gpt_oss",
            Self::Inkling => "inkling",
            Self::K2Horizon => "k2_horizon",
            Self::KimiLinear => "kimi_linear",
            Self::Llama => "llama",
            Self::Gemma2 => "gemma2",
            Self::Nanbeige => "nanbeige",
            Self::MuseGlimmer => "muse_glimmer",
            Self::MuseSpark => "muse_spark",
            Self::Lfm2 => "lfm2",
            Self::NemotronH => "nemotron_h",
            Self::NemotronLabsDiffusion => "nemotron_labs_diffusion",
            Self::Moshi => "moshi",
            Self::VoiceChat => "nemotron_voicechat",
            Self::Qwen2 => "qwen2",
            Self::Qwen3 => "qwen3",
            Self::Qwen3Next => "qwen3_next",
            Self::Qwen3Vl => "qwen3_vl",
            Self::Qwen3VlMoe => "qwen3_vl_moe",
            Self::Qwen35 => "qwen3_5",
            Self::Qwen4Exp => "qwen4_exp",
        }
    }

    /// Neutral loader protocol required by this family.
    pub const fn loading_protocol(self) -> LoadingProtocol {
        match self {
            Self::Moshi | Self::VoiceChat => LoadingProtocol::Realtime,
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
            "diffusion_gemma" | "diffusion_gemma_text" => Ok(Self::DiffusionGemma),
            "gemma4" | "gemma4_text" | "gemma4_unified" | "gemma4_unified_text" => Ok(Self::Gemma4),
            "gpt_oss" => Ok(Self::GptOss),
            "inkling_mm_model" => Ok(Self::Inkling),
            "k2_horizon" => Ok(Self::K2Horizon),
            "kimi_linear" => Ok(Self::KimiLinear),
            "llama" | "mistral" | "phi3" => Ok(Self::Llama),
            "nanbeige" => Ok(Self::Nanbeige),
            "gemma2" => Ok(Self::Gemma2),
            "muse_glimmer" | "muse_glimmer_text" => Ok(Self::MuseGlimmer),
            "muse_spark" | "muse_spark_text" => Ok(Self::MuseSpark),
            "lfm2" | "lfm2_moe" => Ok(Self::Lfm2),
            "nemotron_h" => Ok(Self::NemotronH),
            "nemotron_labs_diffusion" | "nemotron_labs_diffusion_vlm" => Ok(Self::NemotronLabsDiffusion),
            "moshi" | "personaplex" => Ok(Self::Moshi),
            "nemotron_voicechat" => Ok(Self::VoiceChat),
            "qwen2" => Ok(Self::Qwen2),
            "qwen3" | "qwen3_moe" => Ok(Self::Qwen3),
            "qwen3_next" => Ok(Self::Qwen3Next),
            "qwen3_vl" | "qwen3_vl_text" => Ok(Self::Qwen3Vl),
            "qwen3_vl_moe" | "qwen3_vl_moe_text" => Ok(Self::Qwen3VlMoe),
            "qwen4_exp" | "qwen4_exp_text" => Ok(Self::Qwen4Exp),
            "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" => Ok(Self::Qwen35),
            other => Err(ArtifactError::UnsupportedModelType(other.into())),
        }
    }
}

/// Architecture-owned GGUF family identity.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GgufArchitecture {
    /// Content-backed Nemotron-Labs-Diffusion conversions.
    NemotronLabsDiffusion,
    /// `k2-horizon`.
    K2Horizon,
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
    /// `diffusion-gemma`.
    DiffusionGemma,
    /// `gemma4`.
    Gemma4,
    /// `llama`.
    Llama,
    /// Gemma 2 dense decoder.
    Gemma2,
    /// `nanbeige`.
    Nanbeige,
    /// `mistral`.
    Mistral,
    /// `phi3`, including the original Phi-4 dense decoder.
    Phi3,
    /// `muse-glimmer`.
    MuseGlimmer,
    /// `muse-spark`.
    MuseSpark,
    /// `lfm2`.
    Lfm2,
    /// `lfm2moe`.
    Lfm2Moe,
    /// `nemotron_h`.
    NemotronH,
    /// `nemotron_h_moe`.
    NemotronHMoe,
    /// Complete `nemotron_voicechat` speech container.
    VoiceChat,
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
    /// Qwen3.8 Flash-Next gated multi-stream and indexed-attention family.
    Qwen4Exp,
    /// `qwen35moe`.
    Qwen35Moe,
    /// `qwen3next`.
    Qwen3Next,
}

impl GgufArchitecture {
    /// Resolves `general.architecture` through the architecture registry.
    pub fn resolve(name: &str) -> Result<Self, ArtifactError> {
        match name {
            "nemotron_labs_diffusion" => Ok(Self::NemotronLabsDiffusion),
            "k2-horizon" => Ok(Self::K2Horizon),
            "kimi-linear" => Ok(Self::KimiLinear),
            "deepseek2" => Ok(Self::DeepSeek2),
            "deepseek4" => Ok(Self::DeepSeek4),
            "gpt-oss" => Ok(Self::GptOss),
            "inkling" => Ok(Self::Inkling),
            "diffusion-gemma" => Ok(Self::DiffusionGemma),
            "gemma4" => Ok(Self::Gemma4),
            "llama" => Ok(Self::Llama),
            "mistral" => Ok(Self::Mistral),
            "phi3" => Ok(Self::Phi3),
            "nanbeige" => Ok(Self::Nanbeige),
            "gemma2" => Ok(Self::Gemma2),
            "muse-glimmer" => Ok(Self::MuseGlimmer),
            "muse-spark" => Ok(Self::MuseSpark),
            "lfm2" => Ok(Self::Lfm2),
            "lfm2moe" => Ok(Self::Lfm2Moe),
            "nemotron_h" => Ok(Self::NemotronH),
            "nemotron_h_moe" => Ok(Self::NemotronHMoe),
            "nemotron_voicechat" => Ok(Self::VoiceChat),
            "qwen2" => Ok(Self::Qwen2),
            "qwen3" => Ok(Self::Qwen3),
            "qwen3moe" => Ok(Self::Qwen3Moe),
            "qwen3vl" => Ok(Self::Qwen3Vl),
            "qwen3vlmoe" => Ok(Self::Qwen3VlMoe),
            "qwen35" => Ok(Self::Qwen35),
            "qwen4exp" => Ok(Self::Qwen4Exp),
            "qwen35moe" => Ok(Self::Qwen35Moe),
            "qwen3next" => Ok(Self::Qwen3Next),
            other => Err(ArtifactError::UnsupportedGgufArchitecture(other.into())),
        }
    }

    /// General model family implemented by this GGUF architecture.
    pub const fn model_kind(self) -> ModelKind {
        match self {
            Self::NemotronLabsDiffusion => ModelKind::NemotronLabsDiffusion,
            Self::K2Horizon => ModelKind::K2Horizon,
            Self::KimiLinear => ModelKind::KimiLinear,
            Self::DeepSeek2 => ModelKind::DeepSeekV3,
            Self::DeepSeek4 => ModelKind::DeepSeekV4,
            Self::GptOss => ModelKind::GptOss,
            Self::Inkling => ModelKind::Inkling,
            Self::DiffusionGemma => ModelKind::DiffusionGemma,
            Self::Gemma4 => ModelKind::Gemma4,
            Self::Llama | Self::Mistral | Self::Phi3 => ModelKind::Llama,
            Self::Nanbeige => ModelKind::Nanbeige,
            Self::Gemma2 => ModelKind::Gemma2,
            Self::MuseGlimmer => ModelKind::MuseGlimmer,
            Self::MuseSpark => ModelKind::MuseSpark,
            Self::Lfm2 | Self::Lfm2Moe => ModelKind::Lfm2,
            Self::NemotronH | Self::NemotronHMoe => ModelKind::NemotronH,
            Self::VoiceChat => ModelKind::VoiceChat,
            Self::Qwen2 => ModelKind::Qwen2,
            Self::Qwen3 | Self::Qwen3Moe => ModelKind::Qwen3,
            Self::Qwen3Vl => ModelKind::Qwen3Vl,
            Self::Qwen3VlMoe => ModelKind::Qwen3VlMoe,
            Self::Qwen35 | Self::Qwen35Moe => ModelKind::Qwen35,
            Self::Qwen4Exp => ModelKind::Qwen4Exp,
            Self::Qwen3Next => ModelKind::Qwen3Next,
        }
    }

    /// Exact metadata spelling.
    pub const fn metadata_name(self) -> &'static str {
        match self {
            Self::NemotronLabsDiffusion => "nemotron_labs_diffusion",
            Self::K2Horizon => "k2-horizon",
            Self::KimiLinear => "kimi-linear",
            Self::DeepSeek2 => "deepseek2",
            Self::DeepSeek4 => "deepseek4",
            Self::GptOss => "gpt-oss",
            Self::Inkling => "inkling",
            Self::DiffusionGemma => "diffusion-gemma",
            Self::Gemma4 => "gemma4",
            Self::Llama => "llama",
            Self::Gemma2 => "gemma2",
            Self::Mistral => "mistral",
            Self::Phi3 => "phi3",
            Self::Nanbeige => "nanbeige",
            Self::MuseGlimmer => "muse-glimmer",
            Self::MuseSpark => "muse-spark",
            Self::Lfm2 => "lfm2",
            Self::Lfm2Moe => "lfm2moe",
            Self::NemotronH => "nemotron_h",
            Self::NemotronHMoe => "nemotron_h_moe",
            Self::VoiceChat => "nemotron_voicechat",
            Self::Qwen2 => "qwen2",
            Self::Qwen3 => "qwen3",
            Self::Qwen3Moe => "qwen3moe",
            Self::Qwen3Vl => "qwen3vl",
            Self::Qwen3VlMoe => "qwen3vlmoe",
            Self::Qwen35 => "qwen35",
            Self::Qwen4Exp => "qwen4exp",
            Self::Qwen35Moe => "qwen35moe",
            Self::Qwen3Next => "qwen3next",
        }
    }
}

fn resolve_gguf_configuration(
    name: &str,
    checkpoint: &GgufCheckpoint,
) -> Result<(ModelConfiguration, GgufArchitecturePlan), ArtifactError> {
    let (architecture, plan) = if matches!(name, "dream" | "mistral3" | "nemotron_labs_diffusion") {
        let admitted = crate::nemotron_labs_diffusion::gguf::TextGgufPlan::from_checkpoint(checkpoint)
            .map_err(|error| ArtifactError::InvalidArtifact(error.to_string()))?;
        let architecture = GgufArchitecture::NemotronLabsDiffusion;
        let plan = GgufArchitecturePlan::new(architecture,
            GgufModelConfig::NemotronLabsDiffusion(admitted.clone()),
            admitted.checkpoint().clone(), admitted.mapping().to_vec());
        (architecture, plan)
    } else {
        let architecture = GgufArchitecture::resolve(name)?;
        (architecture, resolve_gguf_architecture(architecture, checkpoint)?)
    };
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
        "gemma4"
            | "gemma4_unified"
            | "qwen3_vl"
            | "qwen3_vl_moe"
            | "qwen3_5"
            | "qwen3_5_moe"
            | "qwen4_exp"
            | "muse_glimmer"
            | "muse_spark"
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
    // Assistant implementation model_types may overlap target families. The
    // released DFlash2 draft declares qwen3 but requires target captures and
    // draft-only convolution/selector state, so it is not a Qwen target.
    if json
        .get("architectures")
        .and_then(Value::as_array)
        .is_some_and(|architectures| {
            architectures
                .iter()
                .any(|name| name.as_str() == Some("DFlash2DraftModel"))
        })
    {
        return Err(ArtifactError::InvalidArtifact(
            "DFlash2DraftModel is an external draft assistant, not a standalone target".into(),
        ));
    }
    // The released NeMo document has no top-level Hugging Face model_type.
    // Identify its data contract, never its inert Python _target_ strings.
    if json.get("model_type").is_none()
        && json.pointer("/model/stt/model").is_some()
        && json.pointer("/model/speech_generation/model").is_some()
        && json.get("data").is_some()
    {
        let name = ModelKind::VoiceChat.canonical_name().to_owned();
        return Ok(ResolvedModelIdentity {
            kind: ModelKind::VoiceChat,
            model_type: name.clone(),
            effective_model_type: name,
        });
    }
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
    /// Exact shared-backbone and optional Pixtral geometry.
    NemotronLabsDiffusion(crate::nemotron_labs_diffusion::FamilyConfig),
    /// DeepSeek-V3/R1 family geometry.
    DeepSeekV3(crate::deepseek::V3Args),
    /// DeepSeek-V4 family geometry.
    DeepSeekV4(crate::deepseek::V4Args),
    /// Shared context/refinement and optional image geometry.
    DiffusionGemma(crate::diffusion_gemma::FamilyConfig),
    /// Gemma 4 family geometry.
    Gemma4(crate::gemma4::FamilyConfig),
    /// GPT-OSS family geometry.
    GptOss(crate::gpt_oss::ModelArgs),
    /// Inkling family geometry.
    Inkling(crate::inkling::ModelArgs),
    /// K2 Horizon dense, MoE, and routed-value geometry.
    K2Horizon(crate::k2_horizon::ModelArgs),
    /// Kimi Linear family geometry.
    KimiLinear(crate::kimi_linear::ModelArgs),
    /// Llama-compatible family geometry.
    Llama(crate::llama::ModelArgs),
    /// Gemma 2 geometry and equations.
    Gemma2(crate::gemma2::ModelArgs),
    /// Nanbeige physical blocks and logical loop policy.
    Nanbeige(crate::nanbeige::ModelArgs),
    /// Shared Muse geometry with its explicit checkpoint identity.
    Muse(crate::muse::DecoderConfig),
    /// LFM2 family geometry.
    Lfm2(crate::lfm2::ModelArgs),
    /// Nemotron-H family geometry.
    NemotronH(crate::nemotron_h::ModelArgs),
    /// Moshi-family realtime geometry.
    Moshi(crate::moshi::MoshiConfig),
    /// Complete coupled realtime speech geometry.
    VoiceChat(crate::voicechat::config::VoiceChatConfig),
    /// Qwen2/Qwen3 family geometry.
    Qwen(crate::qwen::ModelArgs),
    /// Qwen hybrid-family geometry.
    QwenHybrid(crate::qwen::hybrid::ParsedHybridConfig),
    /// Flash-Next configuration and exact SafeTensors encoding policy.
    Qwen4Exp(
        crate::qwen4_exp::config::Config,
        crate::qwen4_exp::checkpoint::schema::SafetensorsEncoding,
    ),
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
    qwen4_exp_target: Option<std::sync::Arc<crate::qwen4_exp::prepared::SafetensorsTargetPlan>>,
}

/// Typed, normalized family configuration retained from GGUF admission.
#[derive(Debug, Clone)]
pub enum GgufModelConfig {
    /// Pinned normalized text conversion and its already-open source projection.
    NemotronLabsDiffusion(crate::nemotron_labs_diffusion::gguf::TextGgufPlan),
    /// DeepSeek-V3/R1 family geometry.
    DeepSeekV3(crate::deepseek::V3Args),
    /// DeepSeek-V4 family geometry.
    DeepSeekV4(crate::deepseek::V4Args),
    /// Shared context/refinement and optional image geometry.
    DiffusionGemma(crate::diffusion_gemma::FamilyConfig),
    /// Gemma 4 family geometry.
    Gemma4(crate::gemma4::FamilyConfig),
    /// GPT-OSS family geometry.
    GptOss(crate::gpt_oss::ModelArgs),
    /// Inkling family geometry.
    Inkling(crate::inkling::ModelArgs),
    /// K2 Horizon dense, MoE, and routed-value geometry.
    K2Horizon(crate::k2_horizon::ModelArgs),
    /// Kimi Linear family geometry.
    KimiLinear(crate::kimi_linear::ModelArgs),
    /// LFM2 family geometry.
    Lfm2(crate::lfm2::ModelArgs),
    /// Llama-compatible family geometry.
    Llama(crate::llama::ModelArgs),
    /// Gemma 2 geometry and equations.
    Gemma2(crate::gemma2::ModelArgs),
    /// Nanbeige physical blocks and repetition policy.
    Nanbeige(crate::nanbeige::ModelArgs),
    /// Shared Muse geometry with its explicit checkpoint identity.
    Muse(crate::muse::DecoderConfig),
    /// Nemotron-H family geometry.
    NemotronH(crate::nemotron_h::ModelArgs),
    /// Complete coupled realtime speech geometry.
    VoiceChat(crate::voicechat::config::VoiceChatConfig),
    /// Qwen2/Qwen3 family geometry, including Qwen3-VL text checkpoints.
    Qwen(crate::qwen::ModelArgs),
    /// Qwen hybrid-family geometry.
    QwenHybrid(crate::qwen::hybrid::ParsedHybridConfig),
    /// Exact Flash-Next header, representations and target geometry.
    Qwen4Exp(std::sync::Arc<crate::qwen4_exp::prepared::GgufTargetPlan>),
}

impl SafetensorsModelConfig {
    /// Whether this exact normalized configuration constructs grouped routed experts.
    pub fn uses_grouped_routed_experts(&self) -> bool {
        match self {
            Self::DeepSeekV3(args) => args.n_routed_experts > 0,
            Self::DeepSeekV4(args) => args.n_routed_experts > 0,
            Self::DiffusionGemma(args) => matches!(args.text.num_experts, Some(count) if count > 0),
            Self::Gemma4(args) => matches!(args.text.num_experts, Some(count) if count > 0),
            Self::GptOss(args) => args.num_local_experts > 0,
            Self::Inkling(args) => args.text_config.n_routed_experts > 0,
            Self::K2Horizon(args) => args.is_moe(),
            Self::KimiLinear(args) => args.num_experts > 0,
            Self::Gemma2(_)
            | Self::NemotronLabsDiffusion(_)
            | Self::Llama(_)
            | Self::Nanbeige(_)
            | Self::Moshi(_)
            | Self::VoiceChat(_) => false,
            Self::Muse(args) => args.num_experts > 0,
            Self::Lfm2(args) => args.num_experts > 0,
            Self::NemotronH(args) => args.n_routed_experts > 0,
            Self::Qwen(args) => args.num_experts > 0,
            Self::QwenHybrid(args) => args.text.num_experts > 0,
            Self::QwenVl(args) => args.text.num_experts > 0,
            Self::Qwen4Exp(config, _) => config.experts.count > 0,
        }
    }
}

impl GgufModelConfig {
    /// Whether this exact normalized configuration constructs grouped routed experts.
    pub fn uses_grouped_routed_experts(&self) -> bool {
        match self {
            Self::NemotronLabsDiffusion(_) => false,
            Self::DeepSeekV3(args) => args.n_routed_experts > 0,
            Self::DeepSeekV4(args) => args.n_routed_experts > 0,
            Self::DiffusionGemma(args) => matches!(args.text.num_experts, Some(count) if count > 0),
            Self::Gemma4(args) => matches!(args.text.num_experts, Some(count) if count > 0),
            Self::GptOss(args) => args.num_local_experts > 0,
            Self::Inkling(args) => args.text_config.n_routed_experts > 0,
            Self::K2Horizon(args) => args.is_moe(),
            Self::KimiLinear(args) => args.num_experts > 0,
            Self::Lfm2(args) => args.num_experts > 0,
            Self::Gemma2(_) | Self::Llama(_) | Self::Nanbeige(_) | Self::VoiceChat(_) => false,
            Self::Muse(args) => args.num_experts > 0,
            Self::NemotronH(args) => args.n_routed_experts > 0,
            Self::Qwen(args) => args.num_experts > 0,
            Self::QwenHybrid(args) => args.text.num_experts > 0,
            Self::Qwen4Exp(plan) => plan.text_plan().config().experts.count > 0,
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
    pub(crate) fn replace_muse_execution_config(&mut self, config: crate::muse::DecoderConfig) {
        match &mut self.model {
            GgufModelConfig::Muse(current) => *current = config,
            _ => unreachable!("Muse source enrichment retains its admitted family"),
        }
    }

    pub(crate) fn with_draft_adapter(mut self, adapter: crate::nemotron_labs_diffusion::adapter::DraftAdapterConfig) -> Result<Self, String> {
        match &mut self.model {
            GgufModelConfig::NemotronLabsDiffusion(plan) => *plan = plan.clone().with_draft_adapter(adapter),
            _ => return Err("draft-only low-rank parameters require a shared-backbone family".into()),
        }
        Ok(self)
    }

    /// Retained complete Flash-Next header admission.
    pub fn qwen4_exp_target_plan(&self) -> Option<&crate::qwen4_exp::prepared::GgufTargetPlan> {
        match &self.model {
            GgufModelConfig::Qwen4Exp(plan) => Some(plan),
            _ => None,
        }
    }

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

    /// Preserves source representations selected by architecture admission.
    pub(crate) fn prepared_checkpoint<'a>(
        &'a self,
        admitted: &'a eredu_gguf::Checkpoint,
    ) -> &'a eredu_gguf::Checkpoint {
        match &self.model {
            GgufModelConfig::NemotronLabsDiffusion(plan) => plan.prepared_checkpoint(),
            GgufModelConfig::Qwen4Exp(plan) => plan.text_plan().checkpoint(),
            _ => admitted,
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
    pub(crate) fn replace_muse_execution_config(&mut self, config: crate::muse::DecoderConfig) {
        match &mut self.model {
            SafetensorsModelConfig::Muse(current) => *current = config,
            _ => unreachable!("Muse source enrichment retains its admitted family"),
        }
    }

    pub(crate) fn with_draft_adapter(mut self, adapter: crate::nemotron_labs_diffusion::adapter::DraftAdapterConfig) -> Result<Self, String> {
        match &mut self.model {
            SafetensorsModelConfig::NemotronLabsDiffusion(plan) => *plan = plan.clone().with_draft_adapter(adapter),
            _ => return Err("draft-only low-rank parameters require a shared-backbone family".into()),
        }
        Ok(self)
    }

    /// Complete Flash-Next header, available only after catalog admission.
    pub fn qwen4_exp_target_plan(
        &self,
    ) -> Option<&crate::qwen4_exp::prepared::SafetensorsTargetPlan> {
        self.qwen4_exp_target.as_deref()
    }

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
                let mut checkpoint = self.checkpoint.clone();
                checkpoint
                    .common_tensors
                    .retain(|tensor| !tensor.key.starts_with("mtp."));
                checkpoint
                    .layout_groups
                    .retain(|group| !group.id.starts_with("mtp."));
                debug_assert_eq!(args.text.mtp_num_hidden_layers, 0);
                Ok(checkpoint)
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
                    qwen4_exp_target: None,
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
                qwen4_exp_target: None,
            },
            extension,
        )))
    }

    pub(crate) fn resolve_muse_catalog(&mut self, tensors: &TensorCatalog, source: &Value) -> Result<(), ArtifactError> {
        if let SafetensorsModelConfig::Muse(config) = &self.model {
            let config = crate::muse::resolve_safetensors_config(config, source, &PortableSafetensorsCatalog(tensors))
                .map_err(ArtifactError::InvalidArchitecturePlan)?;
            self.checkpoint = crate::muse::safetensors_plan(&config)
                .map_err(ArtifactError::InvalidArchitecturePlan)?;
            self.model = SafetensorsModelConfig::Muse(config);
        }
        Ok(())
    }

    pub(crate) fn admit_catalog(&mut self, tensors: &TensorCatalog) -> Result<(), ArtifactError> {
        let catalog = PortableSafetensorsCatalog(tensors);
        if let SafetensorsModelConfig::Qwen4Exp(config, encoding) = &self.model {
            let header = crate::qwen4_exp::prepared::SafetensorsTargetPlan::prepare(
                &catalog,
                config.clone(),
                encoding.clone(),
            )
            .map_err(|error| ArtifactError::InvalidArchitecturePlan(error.to_string()))?;
            self.checkpoint = header.checkpoint().clone();
            self.checkpoint_resolution = Some(header.resolution().clone());
            self.qwen4_exp_target = Some(std::sync::Arc::new(header));
            return Ok(());
        }
        if let SafetensorsModelConfig::QwenHybrid(config) = &self.model {
            let (config, checkpoint) =
                crate::qwen::hybrid::safetensors_catalog_plan(config, &catalog)
                    .map_err(ArtifactError::InvalidArchitecturePlan)?;
            self.model = SafetensorsModelConfig::QwenHybrid(config);
            self.checkpoint = checkpoint;
        }
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

    /// Returns whether two opaque values came from the same admitted extension contract.
    ///
    /// Normalized kind/depth, architecture family, checkpoint schema, and exact
    /// resolved source set jointly form the cold admission identity. Callers do
    /// not compare or reconstruct family configuration.
    pub fn same_admission(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.depth == other.depth
            && self.complete_architecture.model_kind() == other.complete_architecture.model_kind()
            && self.complete_architecture.checkpoint() == other.complete_architecture.checkpoint()
            && self.complete_architecture.checkpoint_resolution()
                == other.complete_architecture.checkpoint_resolution()
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
            SafetensorsModelConfig::DeepSeekV4(args) => {
                let capture_count = args
                    .target_capture_policy
                    .as_ref()
                    .ok_or_else(|| {
                        ArtifactError::InvalidArchitecturePlan(
                            "DSpark prediction has no target capture policy".into(),
                        )
                    })?
                    .len();
                let capture_count = i32::try_from(capture_count).map_err(|_| {
                    ArtifactError::InvalidArchitecturePlan(
                        "DSpark target capture count exceeds i32".into(),
                    )
                })?;
                (3, vec![capture_count, args.hidden_size])
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
        ModelKind::NemotronLabsDiffusion => crate::nemotron_labs_diffusion::FamilyConfig::from_hf_value(json)
            .map(SafetensorsModelConfig::NemotronLabsDiffusion)
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::DeepSeekV3 => crate::deepseek::parse_v3_config(json)
            .map(SafetensorsModelConfig::DeepSeekV3)
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::DeepSeekV4 => crate::deepseek::parse_v4_config(json)
            .map(SafetensorsModelConfig::DeepSeekV4)
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::DiffusionGemma => serde_json::to_vec(json)
            .map_err(ArtifactError::from)
            .and_then(|bytes| {
                crate::diffusion_gemma::FamilyConfig::from_hf_json(&bytes)
                    .map(SafetensorsModelConfig::DiffusionGemma)
                    .map_err(|error| invalid_configuration(kind, error))
            }),
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
        ModelKind::Gemma2 => crate::gemma2::model_args_from_config_value(json)
            .map(SafetensorsModelConfig::Gemma2)
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::Llama => crate::llama::model_args_from_config_value(json)
            .map(SafetensorsModelConfig::Llama)
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::Nanbeige => crate::nanbeige::model_args_from_config_value(json)
            .map(SafetensorsModelConfig::Nanbeige)
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::MuseGlimmer | ModelKind::MuseSpark => serde_json::to_vec(json)
            .map_err(ArtifactError::from)
            .and_then(|bytes| {
                crate::muse::DecoderConfig::from_hf_json(&bytes)
                    .map(SafetensorsModelConfig::Muse)
                    .map_err(|error| invalid_configuration(kind, error))
            }),
        ModelKind::Lfm2 => crate::lfm2::model_args_from_config_value(json)
            .map(SafetensorsModelConfig::Lfm2)
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::NemotronH => crate::nemotron_h::model_args_from_config_value(json)
            .map(SafetensorsModelConfig::NemotronH)
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::VoiceChat => crate::voicechat::config::VoiceChatConfig::from_value(json)
            .map(SafetensorsModelConfig::VoiceChat)
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::Moshi => crate::moshi::MoshiConfig::from_config_value(Some(json))
            .map(SafetensorsModelConfig::Moshi)
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::K2Horizon => crate::k2_horizon::model_args_from_config_value(json)
            .map(SafetensorsModelConfig::K2Horizon)
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::Qwen2 | ModelKind::Qwen3 => crate::qwen::model_args_from_config_value(json)
            .map(SafetensorsModelConfig::Qwen)
            .map_err(|error| invalid_configuration(kind, error)),
        ModelKind::Qwen4Exp => crate::qwen4_exp::config::Config::from_json(json)
            .map_err(|error| invalid_configuration(kind, error))
            .and_then(|config| {
                crate::qwen4_exp::checkpoint::schema::SafetensorsEncoding::from_json(json)
                    .map(|encoding| SafetensorsModelConfig::Qwen4Exp(config, encoding))
                    .map_err(|error| invalid_configuration(kind, error))
            }),
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
        SafetensorsModelConfig::NemotronLabsDiffusion(args) => crate::nemotron_labs_diffusion::safetensors_plan(args)
            .map_err(|error| invalid_configuration(kind, error))?,
        SafetensorsModelConfig::Qwen4Exp(config, encoding) => {
            crate::qwen4_exp::checkpoint::schema::safetensors_config_plan(config, encoding)
                .map_err(|error| invalid_configuration(kind, error))?
        }
        SafetensorsModelConfig::DeepSeekV3(args) => {
            crate::deepseek::v3_safetensors_plan(args, true)
                .map_err(|error| invalid_configuration(kind, error))?
        }
        SafetensorsModelConfig::DeepSeekV4(args) => crate::deepseek::v4_safetensors_plan(args)
            .map_err(|error| invalid_configuration(kind, error))?,
        SafetensorsModelConfig::DiffusionGemma(args) => {
            crate::diffusion_gemma::safetensors_plan(args)
                .map_err(|error| invalid_configuration(kind, error))?
        }
        SafetensorsModelConfig::Gemma4(args) => crate::gemma4::safetensors_plan(args)
            .map_err(|error| invalid_configuration(kind, error))?,
        SafetensorsModelConfig::GptOss(args) => crate::gpt_oss::safetensors_plan(args)
            .map_err(|error| invalid_configuration(kind, error))?,
        SafetensorsModelConfig::Inkling(args) => crate::inkling::safetensors_plan(args)
            .map_err(|error| invalid_configuration(kind, error))?,
        SafetensorsModelConfig::KimiLinear(args) => crate::kimi_linear::safetensors_plan(args)
            .map_err(|error| invalid_configuration(kind, error))?,
        SafetensorsModelConfig::Gemma2(args) => crate::gemma2::safetensors_plan(args)
            .map_err(|error| invalid_configuration(kind, error))?,
        SafetensorsModelConfig::Llama(args) => crate::llama::safetensors_plan(args)
            .map_err(|error| invalid_configuration(kind, error))?,
        SafetensorsModelConfig::Nanbeige(args) => crate::nanbeige::safetensors_plan(args)
            .map_err(|error| invalid_configuration(kind, error))?,
        SafetensorsModelConfig::Muse(args) => crate::muse::safetensors_plan(args)
            .map_err(|error| invalid_configuration(kind, error))?,
        SafetensorsModelConfig::Lfm2(args) => crate::lfm2::safetensors_plan(args, true)
            .map_err(|error| invalid_configuration(kind, error))?,
        SafetensorsModelConfig::NemotronH(args) => crate::nemotron_h::safetensors_plan(args)
            .map_err(|error| invalid_configuration(kind, error))?,
        SafetensorsModelConfig::VoiceChat(args) => {
            crate::voicechat::checkpoint::safetensors_plan(args, &args.linear_formats)
                .map_err(|error| invalid_configuration(kind, error))?
        }
        SafetensorsModelConfig::Moshi(args) => crate::moshi::safetensors_plan(args)
            .map_err(|error| invalid_configuration(kind, error))?,
        SafetensorsModelConfig::K2Horizon(args) => crate::k2_horizon::safetensors_plan(args)
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
        qwen4_exp_target: None,
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

/// Admits explicitly selected local primary and companion artifacts.
pub fn inspect_artifact_source(
    source: &eredu_core::artifact::ArtifactSource,
) -> Result<
    eredu_core::ArtifactInspection<crate::processor_plan::ArtifactArchitecturePlan>,
    eredu_core::artifact::ArtifactError,
> {
    eredu_core::artifact::inspect_artifact_source(source, &MODEL_CONFIGURATIONS)
}

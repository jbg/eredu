//! Backend-neutral model artifact inspection and preparation planning.
//!
//! Inspection parses configuration and checkpoint headers only. It never
//! materializes tensor payloads or creates a device/runtime object.

use crate::checkpoint::{TensorCatalog, TensorDescriptor, TensorDtype, TensorStorage};
use eredu_gguf::{Checkpoint as GgufCheckpoint, MetadataValue};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::{Component, Path, PathBuf},
};

/// Supported architecture family selected before backend materialization.
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
    /// PersonaPlex realtime speech architecture.
    PersonaPlex,
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
    /// Every architecture family recognized by the general model loader.
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
        Self::PersonaPlex,
        Self::Qwen2,
        Self::Qwen3,
        Self::Qwen3Next,
        Self::Qwen3Vl,
        Self::Qwen3VlMoe,
        Self::Qwen35,
    ];

    /// Resolves a Hugging Face `model_type` without consulting a backend.
    pub fn from_model_type(model_type: &str) -> Result<Self, ArtifactError> {
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
            "personaplex" => Ok(Self::PersonaPlex),
            "qwen2" => Ok(Self::Qwen2),
            "qwen3" | "qwen3_moe" => Ok(Self::Qwen3),
            "qwen3_next" => Ok(Self::Qwen3Next),
            "qwen3_vl" | "qwen3_vl_text" => Ok(Self::Qwen3Vl),
            "qwen3_vl_moe" | "qwen3_vl_moe_text" => Ok(Self::Qwen3VlMoe),
            "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" => Ok(Self::Qwen35),
            other => Err(ArtifactError::UnsupportedModelType(other.into())),
        }
    }

    /// Stable diagnostic name for this family.
    pub const fn model_type_name(self) -> &'static str {
        match self {
            Self::DeepSeekV3 => "deepseek_v3",
            Self::DeepSeekV4 => "deepseek_v4",
            Self::Gemma4 => "gemma4",
            Self::GptOss => "gpt_oss",
            Self::Inkling => "inkling_mm_model",
            Self::KimiLinear => "kimi_linear",
            Self::Llama => "llama/mistral",
            Self::MuseGlimmer => "muse_glimmer",
            Self::Lfm2 => "lfm2/lfm2_moe",
            Self::NemotronH => "nemotron_h",
            Self::PersonaPlex => "personaplex",
            Self::Qwen2 => "qwen2",
            Self::Qwen3 => "qwen3",
            Self::Qwen3Next => "qwen3_next",
            Self::Qwen3Vl => "qwen3_vl",
            Self::Qwen3VlMoe => "qwen3_vl_moe",
            Self::Qwen35 => "qwen3_5",
        }
    }
}

/// GGUF architecture value resolved independently of an execution backend.
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

    /// Resolves `general.architecture`.
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

    /// Whether this architecture has routed experts addressable independently.
    pub const fn supports_expert_cache(self) -> bool {
        matches!(
            self,
            Self::KimiLinear
                | Self::DeepSeek2
                | Self::DeepSeek4
                | Self::GptOss
                | Self::Inkling
                | Self::Gemma4
                | Self::Lfm2Moe
                | Self::NemotronHMoe
                | Self::Qwen3Moe
                | Self::Qwen3VlMoe
                | Self::Qwen35Moe
                | Self::Qwen3Next
        )
    }
}

/// Artifact container selected during inspection.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactFormat {
    /// Hugging Face SafeTensors directory.
    SafeTensors,
    /// Single-file or canonically sharded GGUF checkpoint.
    Gguf,
}

/// Resolved portable model configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfiguration {
    /// Submitted outer `model_type` or GGUF architecture.
    pub declared_model_type: String,
    /// Nested text architecture selected for dispatch where applicable.
    pub effective_model_type: String,
    /// Canonical architecture family.
    pub kind: ModelKind,
    /// Raw JSON configuration for SafeTensors artifacts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json: Option<Value>,
    /// Exact GGUF architecture for GGUF artifacts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gguf_architecture: Option<GgufArchitecture>,
}

/// Resolves portable Hugging Face model identity from a raw `config.json` value.
pub fn resolve_model_configuration(json: &Value) -> Result<ModelConfiguration, ArtifactError> {
    let metadata: ConfigMetadata = serde_json::from_value(json.clone())?;
    let effective_model_type = effective_model_type(&metadata);
    let kind = ModelKind::from_model_type(&effective_model_type)?;
    Ok(ModelConfiguration {
        declared_model_type: metadata.model_type,
        effective_model_type,
        kind,
        json: Some(json.clone()),
        gguf_architecture: None,
    })
}

/// Parses an optional GGUF integer metadata value as lossless `u32` values.
pub fn gguf_u32_metadata_values(
    key: &str,
    value: Option<&MetadataValue>,
) -> Result<Vec<u32>, ArtifactError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    value.to_u32_vec().ok_or_else(|| {
        ArtifactError::InvalidArtifact(format!(
            "GGUF metadata key {key:?} must contain an integer or integer array whose values fit in u32"
        ))
    })
}

/// Header-only artifact inspection result.
#[derive(Debug, Clone)]
pub struct ArtifactInspection {
    path: PathBuf,
    format: ArtifactFormat,
    configuration: ModelConfiguration,
    tensors: TensorCatalog,
    gguf_checkpoint: Option<GgufCheckpoint>,
}

impl ArtifactInspection {
    /// Submitted artifact path.
    pub fn path(&self) -> &Path {
        &self.path
    }
    /// Detected artifact format.
    pub const fn format(&self) -> ArtifactFormat {
        self.format
    }
    /// Resolved model configuration.
    pub fn configuration(&self) -> &ModelConfiguration {
        &self.configuration
    }
    /// Validated portable tensor catalog.
    pub fn tensors(&self) -> &TensorCatalog {
        &self.tensors
    }
    /// Portable GGUF checkpoint handle, when applicable.
    pub fn gguf_checkpoint(&self) -> Option<&GgufCheckpoint> {
        self.gguf_checkpoint.as_ref()
    }
}

/// Requested load-time weight transformation.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum QuantizationRequest {
    /// Per-group affine integer quantization.
    Affine {
        /// Scalars per quantization group.
        group_size: u32,
        /// Packed bits per scalar.
        bits: u8,
    },
    /// Microscaling FP4 with E2M1 values and E8M0 scales.
    MxFp4,
}

/// Coarse backend-neutral weight residency request.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidencyRequest {
    /// Keep all owned weights resident.
    #[default]
    FullyResident,
    /// Keep a bounded layer window resident and stage remaining layers from host.
    LayerwiseHost,
    /// Stream bounded layer units from disk.
    DenseDiskStream,
    /// Manage routed experts independently of non-expert layers.
    ExpertCache,
}

/// Backend-neutral inputs to materialization-route selection.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct PreparationPolicy {
    /// Optional requested load-time transformation.
    pub quantization: Option<QuantizationRequest>,
    /// Requested residency family.
    pub residency: ResidencyRequest,
    /// Whether a non-replicated distributed topology was requested.
    pub distributed: bool,
}

/// Canonical materialization recipe selected by the core planner.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterializationRoute {
    /// Resident materialization, optionally with a load-time transform.
    Resident,
    /// Bounded non-expert layer materialization.
    Layerwise,
    /// Independent routed-expert materialization.
    ExpertCache,
}

/// Fully inspected input supplied to one selected backend for materialization.
#[derive(Debug, Clone)]
pub struct ModelPreparationPlan {
    inspection: ArtifactInspection,
    policy: PreparationPolicy,
    route: MaterializationRoute,
}

impl ModelPreparationPlan {
    /// Header-only inspection owned by the plan.
    pub fn inspection(&self) -> &ArtifactInspection {
        &self.inspection
    }
    /// Validated caller policy.
    pub const fn policy(&self) -> PreparationPolicy {
        self.policy
    }
    /// Canonical materialization route.
    pub const fn route(&self) -> MaterializationRoute {
        self.route
    }
    /// Consume the plan into its portable artifact and policy.
    pub fn into_parts(self) -> (ModelArtifact, PreparationPolicy, MaterializationRoute) {
        let artifact = match self.inspection.gguf_checkpoint {
            Some(checkpoint) => ModelArtifact::Gguf {
                path: self.inspection.path,
                configuration: self.inspection.configuration,
                tensors: self.inspection.tensors,
                checkpoint,
            },
            None => ModelArtifact::SafeTensors {
                path: self.inspection.path,
                configuration: self.inspection.configuration,
                tensors: self.inspection.tensors,
            },
        };
        (artifact, self.policy, self.route)
    }
}

/// Portable artifact payload consumed by a backend materializer.
#[derive(Debug, Clone)]
pub enum ModelArtifact {
    /// SafeTensors directory and validated header catalog.
    SafeTensors {
        /// Model directory.
        path: PathBuf,
        /// Resolved configuration.
        configuration: ModelConfiguration,
        /// Header-only tensor catalog.
        tensors: TensorCatalog,
    },
    /// Validated GGUF checkpoint handle and metadata-derived configuration.
    Gguf {
        /// Submitted first-shard path.
        path: PathBuf,
        /// Resolved configuration.
        configuration: ModelConfiguration,
        /// Header-only tensor catalog.
        tensors: TensorCatalog,
        /// Pure-Rust checkpoint handle used by backend materialization.
        checkpoint: GgufCheckpoint,
    },
}

/// Inspect a local artifact without loading tensor payloads.
pub fn inspect_artifact(path: impl AsRef<Path>) -> Result<ArtifactInspection, ArtifactError> {
    let path = path.as_ref();
    if is_gguf(path) {
        inspect_gguf(path)
    } else if path.is_dir() {
        inspect_safetensors(path)
    } else if !path.exists() {
        Err(ArtifactError::MissingArtifact(path.to_path_buf()))
    } else {
        Err(ArtifactError::UnsupportedContainer(path.to_path_buf()))
    }
}

/// Validate policy and select one backend-independent materialization route.
pub fn plan_model_preparation(
    inspection: ArtifactInspection,
    policy: PreparationPolicy,
) -> Result<ModelPreparationPlan, ArtifactError> {
    let route = validate_preparation_policy(
        inspection.configuration.kind,
        inspection.configuration.gguf_architecture,
        inspection.format,
        policy,
    )?;
    Ok(ModelPreparationPlan {
        inspection,
        policy,
        route,
    })
}

/// Validate a preparation policy against resolved portable artifact facts.
pub fn validate_preparation_policy(
    kind: ModelKind,
    gguf_architecture: Option<GgufArchitecture>,
    format: ArtifactFormat,
    policy: PreparationPolicy,
) -> Result<MaterializationRoute, ArtifactError> {
    if kind == ModelKind::PersonaPlex {
        return Err(ArtifactError::RealtimeModelRequiresRealtimeLoader);
    }
    if policy.quantization.is_some()
        && format == ArtifactFormat::SafeTensors
        && policy.residency != ResidencyRequest::FullyResident
        && policy.residency != ResidencyRequest::ExpertCache
        && !matches!(
            kind,
            ModelKind::DeepSeekV4 | ModelKind::Qwen2 | ModelKind::Qwen3
        )
    {
        return Err(ArtifactError::UnsupportedQuantizationPolicy(format!(
            "load-time quantization is unsupported for {} nonresident loading",
            kind.model_type_name()
        )));
    }
    if policy.residency == ResidencyRequest::ExpertCache {
        let supported = match gguf_architecture {
            Some(architecture) => architecture.supports_expert_cache(),
            None => matches!(
                kind,
                ModelKind::KimiLinear
                    | ModelKind::DeepSeekV3
                    | ModelKind::DeepSeekV4
                    | ModelKind::GptOss
                    | ModelKind::Inkling
                    | ModelKind::Lfm2
                    | ModelKind::NemotronH
                    | ModelKind::Qwen3
                    | ModelKind::Qwen3Next
                    | ModelKind::Qwen3VlMoe
                    | ModelKind::Qwen35
            ),
        };
        if !supported {
            return Err(ArtifactError::UnsupportedResidencyPolicy(format!(
                "independent expert caching is unavailable for {}",
                kind.model_type_name()
            )));
        }
    }
    let route = match policy.residency {
        ResidencyRequest::FullyResident => MaterializationRoute::Resident,
        ResidencyRequest::LayerwiseHost | ResidencyRequest::DenseDiskStream => {
            MaterializationRoute::Layerwise
        }
        ResidencyRequest::ExpertCache => MaterializationRoute::ExpertCache,
    };
    Ok(route)
}

fn inspect_gguf(path: &Path) -> Result<ArtifactInspection, ArtifactError> {
    let checkpoint = GgufCheckpoint::open(path)?;
    let architecture_name = checkpoint
        .metadata()
        .get("general.architecture")
        .and_then(MetadataValue::as_str)
        .ok_or(ArtifactError::MissingGgufArchitecture)?;
    let architecture = GgufArchitecture::resolve(architecture_name)?;
    validate_gguf_floor(architecture, &checkpoint)?;
    let tensors = checkpoint
        .tensors()
        .map(|tensor| {
            let descriptor = tensor.descriptor();
            let shape = descriptor
                .dimensions
                .iter()
                .map(|&dimension| {
                    usize::try_from(dimension).map_err(|_| {
                        ArtifactError::InvalidArtifact(format!(
                            "GGUF tensor {:?} dimension {dimension} exceeds the host address space",
                            descriptor.name
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(TensorDescriptor {
                name: descriptor.name.clone(),
                shape,
                dtype: TensorDtype::Encoded(format!("{:?}", descriptor.ggml_type)),
                storage: None,
            })
        })
        .collect::<Result<Vec<_>, ArtifactError>>()?;
    let tensors = TensorCatalog::new(tensors)?;
    Ok(ArtifactInspection {
        path: path.to_path_buf(),
        format: ArtifactFormat::Gguf,
        configuration: ModelConfiguration {
            declared_model_type: architecture_name.into(),
            effective_model_type: architecture_name.into(),
            kind: architecture.model_kind(),
            json: None,
            gguf_architecture: Some(architecture),
        },
        tensors,
        gguf_checkpoint: Some(checkpoint),
    })
}

fn validate_gguf_floor(
    architecture: GgufArchitecture,
    checkpoint: &GgufCheckpoint,
) -> Result<(), ArtifactError> {
    if checkpoint.physical_tensor_count() == 0 {
        return Err(ArtifactError::InvalidArtifact(
            "GGUF model checkpoint contains no tensors".into(),
        ));
    }
    let prefix = architecture.metadata_name();
    for suffix in ["block_count", "embedding_length"] {
        let key = format!("{prefix}.{suffix}");
        let value = checkpoint
            .metadata()
            .get(&key)
            .and_then(MetadataValue::as_i64)
            .ok_or_else(|| {
                ArtifactError::InvalidArtifact(format!(
                    "GGUF metadata key {key:?} must be a present integer"
                ))
            })?;
        if value <= 0 {
            return Err(ArtifactError::InvalidArtifact(format!(
                "GGUF metadata key {key:?} must be positive, got {value}"
            )));
        }
    }
    if !checkpoint
        .tensors()
        .any(|tensor| tensor.descriptor().name == "token_embd.weight")
    {
        return Err(ArtifactError::InvalidArtifact(
            "GGUF model checkpoint is missing required tensor \"token_embd.weight\"".into(),
        ));
    }
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
    } else if ModelKind::from_model_type(&metadata.model_type).is_ok() {
        metadata.model_type.clone()
    } else {
        metadata
            .text_config
            .as_ref()
            .and_then(|text| text.model_type.clone())
            .unwrap_or_else(|| metadata.model_type.clone())
    }
}

fn inspect_safetensors(path: &Path) -> Result<ArtifactInspection, ArtifactError> {
    let config_path = path.join("config.json");
    let json: Value = serde_json::from_reader(File::open(&config_path)?)?;
    let configuration = resolve_model_configuration(&json)?;
    let shards = safetensors_shards(path)?;
    let mut descriptors = Vec::new();
    let mut names = BTreeSet::new();
    for shard in shards {
        for descriptor in inspect_safetensors_header(&shard)? {
            if !names.insert(descriptor.name.clone()) {
                return Err(ArtifactError::DuplicateTensor(descriptor.name));
            }
            descriptors.push(descriptor);
        }
    }
    let tensors = TensorCatalog::new(descriptors)?;
    if tensors.is_empty() {
        return Err(ArtifactError::InvalidArtifact(
            "SafeTensors checkpoint contains no tensors".into(),
        ));
    }
    Ok(ArtifactInspection {
        path: path.to_path_buf(),
        format: ArtifactFormat::SafeTensors,
        configuration,
        tensors,
        gguf_checkpoint: None,
    })
}

#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: BTreeMap<String, String>,
}

fn safetensors_shards(path: &Path) -> Result<Vec<PathBuf>, ArtifactError> {
    let index_path = path.join("model.safetensors.index.json");
    if index_path.exists() {
        let index: SafetensorsIndex = serde_json::from_reader(File::open(&index_path)?)?;
        if index.weight_map.is_empty() {
            return Err(ArtifactError::InvalidArtifact(
                "SafeTensors index weight_map is empty".into(),
            ));
        }
        let mut shards = BTreeSet::new();
        for relative in index.weight_map.values() {
            let relative = Path::new(relative);
            if relative.is_absolute()
                || relative
                    .components()
                    .any(|component| !matches!(component, Component::Normal(_) | Component::CurDir))
            {
                return Err(ArtifactError::UnsafeShardPath(relative.to_path_buf()));
            }
            shards.insert(path.join(relative));
        }
        return Ok(shards.into_iter().collect());
    }
    Ok(vec![path.join("model.safetensors")])
}

#[derive(Deserialize)]
struct RawSafetensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

fn inspect_safetensors_header(path: &Path) -> Result<Vec<TensorDescriptor>, ArtifactError> {
    const MAX_HEADER_BYTES: u64 = 100_000_000;
    let mut file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut length = [0_u8; 8];
    file.read_exact(&mut length)?;
    let header_len = u64::from_le_bytes(length);
    if header_len > MAX_HEADER_BYTES {
        return Err(ArtifactError::InvalidArtifact(format!(
            "SafeTensors header in {} exceeds {MAX_HEADER_BYTES} bytes",
            path.display()
        )));
    }
    let mut header = vec![
        0_u8;
        usize::try_from(header_len).map_err(|_| {
            ArtifactError::InvalidArtifact("SafeTensors header length overflows usize".into())
        })?
    ];
    file.read_exact(&mut header)?;
    let raw: BTreeMap<String, Value> = serde_json::from_slice(&header)?;
    let payload_start = 8_u64
        .checked_add(header_len)
        .ok_or_else(|| ArtifactError::InvalidArtifact("SafeTensors offset overflow".into()))?;
    let mut entries = raw
        .into_iter()
        .filter(|(name, _)| name != "__metadata__")
        .map(|(name, value)| {
            serde_json::from_value::<RawSafetensorInfo>(value).map(|info| (name, info))
        })
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|(_, info)| info.data_offsets[0]);
    let mut output = Vec::with_capacity(entries.len());
    let mut expected_offset = 0_u64;
    for (name, info) in entries {
        // SafeTensors rank-zero tensors are scalar parameters with one stored
        // element. Gemma media clipping bounds use this representation.
        if info.shape.contains(&0) {
            return Err(ArtifactError::InvalidArtifact(format!(
                "SafeTensors tensor {name:?} has an invalid shape"
            )));
        }
        let [start, end] = info.data_offsets;
        if start != expected_offset || end < start {
            return Err(ArtifactError::InvalidArtifact(format!(
                "SafeTensors tensor {name:?} has non-contiguous data offsets"
            )));
        }
        expected_offset = end;
        let absolute = payload_start
            .checked_add(start)
            .ok_or_else(|| ArtifactError::InvalidArtifact("SafeTensors offset overflow".into()))?;
        output.push(TensorDescriptor {
            name,
            shape: info.shape,
            dtype: safetensors_dtype(&info.dtype),
            storage: Some(TensorStorage {
                member: path.display().to_string(),
                offset: absolute,
                length: end - start,
            }),
        });
    }
    if payload_start
        .checked_add(expected_offset)
        .ok_or_else(|| ArtifactError::InvalidArtifact("SafeTensors length overflow".into()))?
        != file_len
    {
        return Err(ArtifactError::InvalidArtifact(format!(
            "SafeTensors payload length does not match header in {}",
            path.display()
        )));
    }
    Ok(output)
}

fn safetensors_dtype(dtype: &str) -> TensorDtype {
    match dtype {
        "F32" => TensorDtype::F32,
        "F16" => TensorDtype::F16,
        "BF16" => TensorDtype::Bf16,
        "I8" => TensorDtype::I8,
        "U8" => TensorDtype::U8,
        "I32" => TensorDtype::I32,
        other => TensorDtype::Encoded(other.into()),
    }
}

fn is_gguf(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
}

/// Portable artifact inspection/planning failure.
#[derive(Debug, thiserror::Error)]
pub enum ArtifactError {
    /// Artifact path does not exist.
    #[error("model artifact does not exist: {0}")]
    MissingArtifact(PathBuf),
    /// Path is not a supported artifact container.
    #[error("model artifact must be a SafeTensors directory or .gguf file: {0}")]
    UnsupportedContainer(PathBuf),
    /// Model type is not recognized.
    #[error("unsupported model type: {0}")]
    UnsupportedModelType(String),
    /// GGUF architecture is not recognized.
    #[error("unsupported GGUF architecture: {0}")]
    UnsupportedGgufArchitecture(String),
    /// GGUF architecture metadata is absent or has the wrong type.
    #[error("GGUF metadata is missing string key \"general.architecture\"")]
    MissingGgufArchitecture,
    /// Header/catalog content is contradictory.
    #[error("invalid model artifact: {0}")]
    InvalidArtifact(String),
    /// A tensor name occurred more than once.
    #[error("duplicate checkpoint tensor {0:?}")]
    DuplicateTensor(String),
    /// Indexed shard path escapes the artifact root.
    #[error("unsafe SafeTensors shard path {0}")]
    UnsafeShardPath(PathBuf),
    /// Requested quantization transformation is unavailable for the artifact.
    #[error("unsupported model quantization policy: {0}")]
    UnsupportedQuantizationPolicy(String),
    /// Requested residency mode is unavailable for the artifact.
    #[error("unsupported model residency policy: {0}")]
    UnsupportedResidencyPolicy(String),
    /// PersonaPlex uses a distinct realtime model/session contract.
    #[error("PersonaPlex must be prepared through the realtime model loader")]
    RealtimeModelRequiresRealtimeLoader,
    /// Ordinary filesystem error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// JSON configuration/header error.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// GGUF parsing/catalog error.
    #[error(transparent)]
    Gguf(#[from] eredu_gguf::Error),
    /// Neutral tensor catalog error.
    #[error(transparent)]
    Catalog(#[from] crate::checkpoint::CatalogError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_gguf::{GgmlType, MetadataArray, TensorInput, Writer};
    use std::io::Write;

    fn write_safetensors_fixture(root: &Path, model_type: &str) {
        std::fs::write(
            root.join("config.json"),
            format!(r#"{{"model_type":"{model_type}"}}"#),
        )
        .unwrap();
        let header =
            br#"{"token_embd.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let mut file = File::create(root.join("model.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
        file.write_all(&[0_u8; 16]).unwrap();
    }

    #[test]
    fn model_configuration_resolution_is_portable_and_nested() {
        let json = serde_json::json!({
            "model_type": "qwen3_5",
            "text_config": { "model_type": "qwen3_5_moe" }
        });
        let resolved = resolve_model_configuration(&json).unwrap();
        assert_eq!(resolved.kind, ModelKind::Qwen35);
        assert_eq!(resolved.declared_model_type, "qwen3_5");
        assert_eq!(resolved.effective_model_type, "qwen3_5_moe");
        assert_eq!(resolved.json.as_ref(), Some(&json));
    }

    #[test]
    fn gguf_u32_metadata_is_lossless_and_fail_closed() {
        let values = MetadataValue::Array(MetadataArray::Uint64(vec![0, u32::MAX.into()]));
        assert_eq!(
            gguf_u32_metadata_values("tokenizer.ids", Some(&values)).unwrap(),
            vec![0, u32::MAX]
        );
        assert!(gguf_u32_metadata_values(
            "tokenizer.ids",
            Some(&MetadataValue::Uint64(u64::from(u32::MAX) + 1))
        )
        .is_err());
        assert!(
            gguf_u32_metadata_values("tokenizer.ids", Some(&MetadataValue::Int32(-1))).is_err()
        );
        assert!(gguf_u32_metadata_values(
            "tokenizer.ids",
            Some(&MetadataValue::String("1".into()))
        )
        .is_err());
        assert!(gguf_u32_metadata_values("tokenizer.ids", None)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn safetensors_inspection_and_planning_are_backend_neutral() {
        let root = tempfile::tempdir().unwrap();
        write_safetensors_fixture(root.path(), "llama");
        let inspection = inspect_artifact(root.path()).unwrap();
        assert_eq!(inspection.configuration().kind, ModelKind::Llama);
        assert_eq!(inspection.tensors().len(), 1);
        let plan = plan_model_preparation(inspection, PreparationPolicy::default()).unwrap();
        assert_eq!(plan.route(), MaterializationRoute::Resident);
        assert!(matches!(
            plan.into_parts().0,
            ModelArtifact::SafeTensors { .. }
        ));
    }

    #[test]
    fn safetensors_inspection_accepts_rank_zero_scalar_parameters() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("config.json"),
            r#"{"model_type":"gemma4"}"#,
        )
        .unwrap();
        let header = br#"{"clip.output_max":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}"#;
        let mut file = File::create(root.path().join("model.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
        file.write_all(&0.0_f32.to_le_bytes()).unwrap();

        let inspection = inspect_artifact(root.path()).unwrap();
        assert_eq!(
            inspection.tensors().get("clip.output_max").unwrap().shape,
            Vec::<usize>::new()
        );
    }

    #[test]
    fn distributed_policy_uses_the_same_neutral_preparation_plan() {
        let root = tempfile::tempdir().unwrap();
        write_safetensors_fixture(root.path(), "llama");
        let policy = PreparationPolicy {
            distributed: true,
            ..PreparationPolicy::default()
        };

        let plan = plan_model_preparation(inspect_artifact(root.path()).unwrap(), policy).unwrap();

        assert_eq!(plan.policy(), policy);
        assert_eq!(plan.route(), MaterializationRoute::Resident);
    }

    #[test]
    fn policy_fails_closed_for_dense_expert_cache() {
        let root = tempfile::tempdir().unwrap();
        write_safetensors_fixture(root.path(), "llama");
        let error = plan_model_preparation(
            inspect_artifact(root.path()).unwrap(),
            PreparationPolicy {
                residency: ResidencyRequest::ExpertCache,
                ..PreparationPolicy::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ArtifactError::UnsupportedResidencyPolicy(_)
        ));
    }

    #[test]
    fn gguf_plan_owns_the_portable_checkpoint_for_later_materialization() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("model.gguf");
        let data = 1.0_f32.to_le_bytes();
        let metadata = BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("llama".into()),
            ),
            ("llama.block_count".into(), MetadataValue::Uint32(1)),
            ("llama.embedding_length".into(), MetadataValue::Uint32(1)),
        ]);
        Writer::default()
            .write(
                File::create(&path).unwrap(),
                &metadata,
                &[TensorInput {
                    name: "token_embd.weight",
                    dimensions: &[1],
                    ggml_type: GgmlType::F32,
                    data: &data,
                }],
            )
            .unwrap();

        let plan = plan_model_preparation(
            inspect_artifact(&path).unwrap(),
            PreparationPolicy::default(),
        )
        .unwrap();
        let ModelArtifact::Gguf {
            configuration,
            checkpoint,
            ..
        } = plan.into_parts().0
        else {
            panic!("expected GGUF artifact");
        };
        assert_eq!(configuration.kind, ModelKind::Llama);
        assert_eq!(checkpoint.physical_tensor_count(), 1);
    }
}

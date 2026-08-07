//! Model-family detection and architecture-independent load options.

use super::*;

#[derive(Debug, Clone, Copy, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
/// Supported model-family dispatch target.
pub enum ModelKind {
    /// DeepSeek-V3/R1 MLA and MoE architecture.
    DeepSeekV3,
    /// Gemma 4 text architecture.
    Gemma4,
    /// OpenAI GPT-OSS MXFP4 sparse decoder architecture.
    GptOss,
    /// Thinking Machines Lab Inkling multimodal architecture.
    Inkling,
    /// Moonshot Kimi Linear hybrid KDA/MLA sparse decoder architecture.
    KimiLinear,
    /// Llama-compatible dense decoder architecture, including Mistral.
    Llama,
    /// Liquid AI LFM2/LFM2.5 dense or MoE architecture.
    Lfm2,
    /// Nemotron-H hybrid Mamba2/attention/MoE architecture.
    NemotronH,
    /// PersonaPlex realtime speech-to-speech architecture.
    PersonaPlex,
    /// Qwen2 and Qwen2.5 dense text decoder architecture.
    Qwen2,
    /// Qwen3 decoder architecture.
    Qwen3,
    /// Qwen3-Next hybrid attention/MoE architecture.
    Qwen3Next,
    /// Qwen3-VL multimodal architecture.
    Qwen3Vl,
    /// Qwen3-VL multimodal architecture with a sparse MoE text decoder.
    Qwen3VlMoe,
    /// Qwen3.5 dense or mixture-of-experts architecture.
    Qwen35,
}

/// Architecture-independent options for loading model weights.
///
/// When `quantization` is set for a dense checkpoint, eligible parameters are
/// quantized and materialized one tensor at a time. Checkpoints already
/// carrying matching metadata are loaded directly without requantizing.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct ModelLoadOptions {
    /// Optional MLX weight encoding requested during dense checkpoint loading.
    pub quantization: Option<WeightQuantization>,
    /// Optional validated runtime topology and process-local device assignment.
    ///
    /// Singleton topologies preserve normal model loading. Non-replicated
    /// topologies must be loaded through the explicit pipeline/expert APIs or
    /// through the selected architecture's generalized tensor-parallel loader.
    pub parallel: Option<ParallelTopology>,
    /// Parameter placement and execution policy for cataloged checkpoint stores.
    pub weight_residency: WeightResidency,
}

impl ModelLoadOptions {
    /// Creates load options that quantize eligible dense weights on load.
    pub fn with_quantization(quantization: impl Into<WeightQuantization>) -> Self {
        Self {
            quantization: Some(quantization.into()),
            parallel: None,
            weight_residency: WeightResidency::FullyResident,
        }
    }

    /// Adds a validated runtime parallel topology to these options.
    pub fn with_parallel_topology(mut self, topology: ParallelTopology) -> Self {
        self.parallel = Some(topology);
        self
    }

    /// Creates load options for a validated runtime parallel topology.
    pub fn with_parallel(topology: ParallelTopology) -> Self {
        Self::default().with_parallel_topology(topology)
    }

    /// Selects fully resident or bounded layer execution for checkpoint weights.
    pub fn with_weight_residency(mut self, residency: WeightResidency) -> Self {
        self.weight_residency = residency;
        self
    }
}

pub(crate) fn ensure_executable_load_options(options: ModelLoadOptions) -> Result<(), Error> {
    if let Some(topology) = options
        .parallel
        .filter(|topology| !topology.is_replicated())
    {
        Err(Error::Parallel(
            if topology.tensor_parallel_size > 1
                && topology.pipeline_parallel_size == 1
                && topology.expert_parallel_size == 1
            {
                "non-replicated pure tensor-parallel loading requires an architecture adapter; use the selected model family's generalized tensor-parallel loader"
                    .into()
            } else if topology.pipeline_parallel_size > 1
                && topology.tensor_parallel_size == 1
                && topology.expert_parallel_size == 1
            {
                "non-replicated pure pipeline loading cannot return the complete Model type; use architectures::distributed::pipeline::load_pipeline_model_with_options"
                    .into()
            } else if topology.expert_parallel_size > 1
                && topology.tensor_parallel_size == 1
                && topology.pipeline_parallel_size == 1
            {
                "non-replicated pure expert-parallel loading cannot return the complete Model type; use architectures::distributed::expert::load_expert_parallel_model_with_options"
                    .into()
            } else {
                if topology.pipeline_parallel_size > 1 {
                    "Cartesian pipeline topology cannot return the complete Model type; use architectures::distributed::pipeline::load_pipeline_model_with_options and PipelineModel::forward_cartesian"
                        .into()
                } else {
                    "Cartesian TP+EP topology cannot return the complete Model type; use architectures::distributed::expert::load_expert_parallel_model_with_options and ExpertParallelModel::forward_cartesian"
                        .into()
                }
            },
        ))
    } else {
        Ok(())
    }
}

impl ModelKind {
    /// Every model-family dispatch target supported by the high-level loader.
    ///
    /// Inspection parity tests iterate this list. The exhaustive matches in
    /// the loader and preflight planner remain the compile-time backstop when
    /// a new variant is added.
    pub const ALL: [Self; 15] = [
        Self::DeepSeekV3,
        Self::Gemma4,
        Self::GptOss,
        Self::Inkling,
        Self::KimiLinear,
        Self::Llama,
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

    /// Returns a stable model-family name for diagnostics and capability dispatch.
    pub const fn model_type_name(self) -> &'static str {
        match self {
            Self::DeepSeekV3 => "deepseek_v3",
            Self::Gemma4 => "gemma4",
            Self::GptOss => "gpt_oss",
            Self::Inkling => "inkling_mm_model",
            Self::KimiLinear => "kimi_linear",
            Self::Llama => "llama/mistral",
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

    pub(super) fn from_model_type(model_type: &str) -> Result<Self, Error> {
        match model_type {
            "deepseek_v3" => Ok(Self::DeepSeekV3),
            "gemma4" | "gemma4_text" | "gemma4_unified" | "gemma4_unified_text" => Ok(Self::Gemma4),
            "gpt_oss" => Ok(Self::GptOss),
            "inkling_mm_model" => Ok(Self::Inkling),
            "kimi_linear" => Ok(Self::KimiLinear),
            "llama" | "mistral" => Ok(Self::Llama),
            "lfm2" | "lfm2_moe" => Ok(Self::Lfm2),
            "nemotron_h" => Ok(Self::NemotronH),
            "personaplex" => Ok(Self::PersonaPlex),
            "qwen2" => Ok(Self::Qwen2),
            "qwen3" | "qwen3_moe" => Ok(Self::Qwen3),
            "qwen3_next" => Ok(Self::Qwen3Next),
            "qwen3_vl" | "qwen3_vl_text" => Ok(Self::Qwen3Vl),
            "qwen3_vl_moe" | "qwen3_vl_moe_text" => Ok(Self::Qwen3VlMoe),
            "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" => Ok(Self::Qwen35),
            other => Err(Error::UnsupportedModelType(other.to_string())),
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum ArtifactLoadKind {
    Safetensors,
    Gguf,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum GgufArchitecture {
    KimiLinear,
    DeepSeek2,
    GptOss,
    Inkling,
    Gemma4,
    Llama,
    Mistral,
    Lfm2,
    Lfm2Moe,
    NemotronH,
    NemotronHMoe,
    Qwen2,
    Qwen3,
    Qwen3Moe,
    Qwen3Vl,
    Qwen3VlMoe,
    Qwen35,
    Qwen35Moe,
    Qwen3Next,
}

impl GgufArchitecture {
    pub(crate) const SUPPORTED_NAMES: &'static str = "kimi-linear, deepseek2, gpt-oss, inkling, gemma4, llama, mistral, lfm2, lfm2moe, nemotron_h, nemotron_h_moe, qwen2, qwen3, qwen3moe, qwen3vl, qwen3vlmoe, qwen35, qwen35moe, and qwen3next";

    pub(crate) fn resolve(name: &str) -> Result<Self, Error> {
        match name {
            "kimi-linear" => Ok(Self::KimiLinear),
            "deepseek2" => Ok(Self::DeepSeek2),
            "gpt-oss" => Ok(Self::GptOss),
            "inkling" => Ok(Self::Inkling),
            "gemma4" => Ok(Self::Gemma4),
            "llama" => Ok(Self::Llama),
            "mistral" => Ok(Self::Mistral),
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
            other => Err(Error::UnsupportedArchitecture(format!(
                "GGUF architecture {other:?}; supported GGUF architectures are {}",
                Self::SUPPORTED_NAMES
            ))),
        }
    }

    pub(crate) const fn model_kind(self) -> ModelKind {
        match self {
            Self::KimiLinear => ModelKind::KimiLinear,
            Self::DeepSeek2 => ModelKind::DeepSeekV3,
            Self::GptOss => ModelKind::GptOss,
            Self::Inkling => ModelKind::Inkling,
            Self::Gemma4 => ModelKind::Gemma4,
            Self::Llama | Self::Mistral => ModelKind::Llama,
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

    pub(crate) fn validate_load_policy(self, options: ModelLoadOptions) -> Result<(), Error> {
        let kind = self.model_kind();
        validate_load_policy(kind, ArtifactLoadKind::Gguf, options)?;
        let sparse = matches!(
            options.weight_residency,
            WeightResidency::SparseExpertCache(_)
                | WeightResidency::SparseExpertCacheWithDenseLayers(_)
        );
        if sparse
            && !matches!(
                self,
                Self::KimiLinear
                    | Self::DeepSeek2
                    | Self::GptOss
                    | Self::Inkling
                    | Self::Lfm2Moe
                    | Self::NemotronHMoe
                    | Self::Qwen3Moe
                    | Self::Qwen3VlMoe
                    | Self::Qwen35Moe
                    | Self::Qwen3Next
            )
        {
            return Err(Error::UnsupportedArchitecture(format!(
                "sparse expert caching is unavailable for GGUF architecture {:?}",
                self.metadata_name()
            )));
        }
        Ok(())
    }

    pub(crate) const fn metadata_name(self) -> &'static str {
        match self {
            Self::KimiLinear => "kimi-linear",
            Self::DeepSeek2 => "deepseek2",
            Self::GptOss => "gpt-oss",
            Self::Inkling => "inkling",
            Self::Gemma4 => "gemma4",
            Self::Llama => "llama",
            Self::Mistral => "mistral",
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

    /// Validates container facts required by every concrete GGUF loader route.
    /// Architecture modules remain responsible for their additional geometry
    /// checks, but this common floor prevents a structurally valid, metadata-
    /// only GGUF from being admitted as a model checkpoint.
    pub(crate) fn validate_catalog(
        self,
        checkpoint: &GgufCheckpoint,
        metadata: &std::collections::HashMap<String, GgufMetadataValue>,
    ) -> Result<(), Error> {
        if checkpoint.catalog().physical_tensor_count() == 0 {
            return Err(Error::UnsupportedArchitecture(
                "GGUF model checkpoint contains no tensors".into(),
            ));
        }
        let prefix = self.metadata_name();
        for suffix in ["block_count", "embedding_length"] {
            let key = format!("{prefix}.{suffix}");
            let value = metadata.get(&key).ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "GGUF metadata is missing required key {key:?}"
                ))
            })?;
            let value = value.as_i64().ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "GGUF metadata key {key:?} must be an integer"
                ))
            })?;
            if value <= 0 {
                return Err(Error::UnsupportedArchitecture(format!(
                    "GGUF metadata key {key:?} must be positive, got {value}"
                )));
            }
        }
        if !checkpoint
            .catalog()
            .tensors()
            .any(|tensor| tensor.descriptor().name == "token_embd.weight")
        {
            return Err(Error::UnsupportedArchitecture(
                "GGUF model checkpoint is missing required tensor \"token_embd.weight\"".into(),
            ));
        }
        if matches!(self, Self::Qwen35 | Self::Qwen35Moe | Self::Qwen3Next)
            && checkpoint.catalog().tensors().any(|tensor| {
                let name = tensor.descriptor().name.as_str();
                name.starts_with("v.") || name.starts_with("mm.")
            })
        {
            return Err(Error::UnsupportedArchitecture(
                "multimodal Qwen3-Next/Qwen3.5 GGUF checkpoints are not supported".into(),
            ));
        }
        Ok(())
    }
}

/// Validates the architecture-independent part of the exact high-level load
/// route. Both artifact inspection and weight loading call this function.
pub(crate) fn validate_load_policy(
    kind: ModelKind,
    artifact: ArtifactLoadKind,
    options: ModelLoadOptions,
) -> Result<(), Error> {
    ensure_executable_load_options(options)?;
    if kind == ModelKind::PersonaPlex {
        return Err(Error::UnsupportedArchitecture(
            "PersonaPlex must be loaded through the realtime API".into(),
        ));
    }

    if options.quantization.is_some()
        && !matches!(options.weight_residency, WeightResidency::FullyResident)
    {
        return Err(Error::Quantization(match artifact {
            ArtifactLoadKind::Safetensors => format!(
                "load-time quantization is unsupported for {} nonresident loading; use a matching checkpoint-native packed format",
                kind.model_type_name()
            ),
            ArtifactLoadKind::Gguf => "load-time quantization is incompatible with nonresident GGUF policies; use checkpoint-native GGUF quantization".into(),
        }));
    }

    if options.quantization.is_some() && matches!(kind, ModelKind::Inkling | ModelKind::NemotronH) {
        return Err(Error::Quantization(match kind {
            ModelKind::Inkling => "Inkling load-time requantization is unsupported because routed experts use packed rank-3 grouped-matmul weights without a matching quantized grouped-matmul implementation".into(),
            ModelKind::NemotronH => "Nemotron-H load-time quantization is unavailable because routed experts use packed rank-3 grouped-matmul weights without an affine grouped-matmul implementation".into(),
            _ => unreachable!("matched above"),
        }));
    }

    let sparse = matches!(
        options.weight_residency,
        WeightResidency::SparseExpertCache(_)
            | WeightResidency::SparseExpertCacheWithDenseLayers(_)
    );
    if artifact == ArtifactLoadKind::Safetensors
        && sparse
        && !matches!(
            kind,
            ModelKind::KimiLinear
                | ModelKind::DeepSeekV3
                | ModelKind::GptOss
                | ModelKind::Inkling
                | ModelKind::Lfm2
                | ModelKind::NemotronH
                | ModelKind::Qwen3
                | ModelKind::Qwen3Next
                | ModelKind::Qwen3VlMoe
                | ModelKind::Qwen35
        )
    {
        return Err(Error::UnsupportedArchitecture(format!(
            "sparse expert caching requires a supported safetensors MoE architecture, not {}",
            kind.model_type_name()
        )));
    }
    Ok(())
}

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
    let kind = ModelKind::from_model_type(&effective_model_type)
        .map_err(ModelConfigResolutionError::Loader)?;
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
        ModelKind::Gemma4 => gemma4::validate_model_config_value(config),
        ModelKind::GptOss => gpt_oss::validate_model_config_value(config),
        ModelKind::Inkling => inkling::validate_model_config_value(config),
        ModelKind::KimiLinear => kimi_linear::validate_model_config_value(config),
        ModelKind::Llama => llama::validate_model_config_value(config),
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

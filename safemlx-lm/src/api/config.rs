//! Model-family detection and architecture-independent load options.

use super::*;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
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
    /// Llama-compatible dense decoder architecture, including Mistral.
    Llama,
    /// Liquid AI LFM2/LFM2.5 dense or MoE architecture.
    Lfm2,
    /// Nemotron-H hybrid Mamba2/attention/MoE architecture.
    NemotronH,
    /// PersonaPlex realtime speech-to-speech architecture.
    PersonaPlex,
    /// Qwen3 decoder architecture.
    Qwen3,
    /// Qwen3-Next hybrid attention/MoE architecture.
    Qwen3Next,
    /// Qwen3-VL multimodal architecture.
    Qwen3Vl,
    /// Qwen3-VL multimodal architecture with a sparse MoE text decoder.
    Qwen3VlMoe,
    /// Qwen3.5 dense or mixture-of-experts architecture.
    Qwen35Moe,
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
    /// topologies must be loaded through the explicit [`crate::architectures::distributed::pipeline`],
    /// [`crate::architectures::distributed::tensor`], or [`crate::architectures::distributed::expert`] APIs.
    pub parallel: Option<ParallelTopology>,
    /// Parameter placement and execution policy for safetensors checkpoints.
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

    /// Selects fully resident or bounded layer execution for safetensors.
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
                "non-replicated pure tensor-parallel loading cannot return the complete Model type; use architectures::distributed::tensor::load_tensor_parallel_model_with_options"
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
                "hybrid TP+PP, TP+EP, and PP+EP model loading is unsupported; use a pure tensor-, pipeline-, or expert-parallel topology"
                    .into()
            },
        ))
    } else {
        Ok(())
    }
}

impl ModelKind {
    /// Returns a stable model-family name for diagnostics and capability dispatch.
    pub const fn model_type_name(self) -> &'static str {
        match self {
            Self::DeepSeekV3 => "deepseek_v3",
            Self::Gemma4 => "gemma4",
            Self::GptOss => "gpt_oss",
            Self::Inkling => "inkling_mm_model",
            Self::Llama => "llama/mistral",
            Self::Lfm2 => "lfm2/lfm2_moe",
            Self::NemotronH => "nemotron_h",
            Self::PersonaPlex => "personaplex",
            Self::Qwen3 => "qwen3",
            Self::Qwen3Next => "qwen3_next",
            Self::Qwen3Vl => "qwen3_vl",
            Self::Qwen3VlMoe => "qwen3_vl_moe",
            Self::Qwen35Moe => "qwen3_5",
        }
    }

    pub(super) fn from_model_type(model_type: &str) -> Result<Self, Error> {
        match model_type {
            "deepseek_v3" => Ok(Self::DeepSeekV3),
            "gemma4" | "gemma4_text" | "gemma4_unified" | "gemma4_unified_text" => Ok(Self::Gemma4),
            "gpt_oss" => Ok(Self::GptOss),
            "inkling_mm_model" => Ok(Self::Inkling),
            "llama" | "mistral" => Ok(Self::Llama),
            "lfm2" | "lfm2_moe" => Ok(Self::Lfm2),
            "nemotron_h" => Ok(Self::NemotronH),
            "personaplex" => Ok(Self::PersonaPlex),
            "qwen3" => Ok(Self::Qwen3),
            "qwen3_next" => Ok(Self::Qwen3Next),
            "qwen3_vl" | "qwen3_vl_text" => Ok(Self::Qwen3Vl),
            "qwen3_vl_moe" | "qwen3_vl_moe_text" => Ok(Self::Qwen3VlMoe),
            "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" => Ok(Self::Qwen35Moe),
            other => Err(Error::UnsupportedModelType(other.to_string())),
        }
    }
}

/// Details for a model config that this crate can load.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SupportedModelConfig {
    /// The runtime model implementation that will be used.
    pub kind: ModelKind,
    /// The top-level `model_type` from the submitted config.
    pub model_type: String,
    /// The resolved text model type used for dispatch.
    pub effective_model_type: String,
}

/// Result of checking whether a submitted model config is supported.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ModelConfigSupport {
    /// The config is supported by this crate's loader.
    Supported(SupportedModelConfig),
    /// The config is not supported, with a human-readable reason.
    Unsupported {
        /// Human-readable reason the config is unsupported.
        reason: String,
    },
}

impl ModelConfigSupport {
    /// Returns true when this config is supported.
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Supported(_))
    }

    /// Returns the unsupported reason, if this result is unsupported.
    pub fn unsupported_reason(&self) -> Option<&str> {
        match self {
            Self::Supported(_) => None,
            Self::Unsupported { reason } => Some(reason),
        }
    }
}

/// Checks a `config.json` string and reports whether it is supported.
pub fn check_model_config_json(config_json: &str) -> ModelConfigSupport {
    match serde_json::from_str::<Value>(config_json) {
        Ok(config) => check_model_config(&config),
        Err(error) => ModelConfigSupport::Unsupported {
            reason: format!("invalid model config JSON: {error}"),
        },
    }
}

/// Checks a parsed model config value and reports whether it is supported.
pub fn check_model_config(config: &Value) -> ModelConfigSupport {
    let metadata = match serde_json::from_value::<ModelMetadata>(config.clone()) {
        Ok(metadata) => metadata,
        Err(error) => {
            return ModelConfigSupport::Unsupported {
                reason: format!("invalid model config metadata: {error}"),
            };
        }
    };

    let effective_model_type = effective_model_type(&metadata);
    let kind = match ModelKind::from_model_type(&effective_model_type) {
        Ok(kind) => kind,
        Err(error) => {
            return ModelConfigSupport::Unsupported {
                reason: error.to_string(),
            };
        }
    };

    if let Err(error) = validate_model_config(kind, config) {
        return ModelConfigSupport::Unsupported {
            reason: error.to_string(),
        };
    }

    ModelConfigSupport::Supported(SupportedModelConfig {
        kind,
        model_type: metadata.model_type,
        effective_model_type,
    })
}

/// Reads `config.json` from a model directory and reports whether it is supported.
pub fn check_model_dir(model_dir: impl AsRef<Path>) -> ModelConfigSupport {
    let config_path = model_dir.as_ref().join("config.json");
    match std::fs::read_to_string(&config_path) {
        Ok(config_json) => check_model_config_json(&config_json),
        Err(error) => ModelConfigSupport::Unsupported {
            reason: format!("could not read {}: {error}", config_path.display()),
        },
    }
}

fn validate_model_config(kind: ModelKind, config: &Value) -> Result<(), Error> {
    match kind {
        ModelKind::DeepSeekV3 => deepseek_v3::validate_model_config_value(config),
        ModelKind::Gemma4 => gemma4::validate_model_config_value(config),
        ModelKind::GptOss => gpt_oss::validate_model_config_value(config),
        ModelKind::Inkling => inkling::validate_model_config_value(config),
        ModelKind::Llama => llama::validate_model_config_value(config),
        ModelKind::Lfm2 => lfm2::validate_model_config_value(config),
        ModelKind::NemotronH => nemotron_h::validate_model_config_value(config),
        ModelKind::PersonaPlex => personaplex::validate_model_config_value(config),
        ModelKind::Qwen3 => {
            serde_json::from_value::<qwen3::ModelArgs>(config.clone()).map_err(|error| {
                Error::UnsupportedArchitecture(format!("invalid qwen3 config: {error}"))
            })?;
            Ok(())
        }
        ModelKind::Qwen3Next => qwen3_next::validate_model_config_value(config),
        ModelKind::Qwen3Vl => qwen3_vl::validate_model_config_value(config),
        ModelKind::Qwen3VlMoe => qwen3_vl_moe::validate_model_config_value(config),
        ModelKind::Qwen35Moe => qwen3_5_moe::validate_model_config_value(config),
    }
}

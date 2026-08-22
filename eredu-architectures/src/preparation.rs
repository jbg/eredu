//! Normalized architecture capabilities used during materialization planning.

use std::{collections::HashMap, fmt::Display};

use eredu_core::{GgufArchitecture, InputModalities, ModelKind};
use eredu_gguf::{Checkpoint as GgufCheckpoint, MetadataValue};
use serde_json::Value;

/// Preparation-relevant facts derived from one exact normalized architecture.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ArchitectureCapabilities {
    independently_addressable_experts: bool,
    input_modalities: InputModalities,
}

impl ArchitectureCapabilities {
    /// Whether routed expert parameters can be managed independently.
    pub const fn independently_addressable_experts(self) -> bool {
        self.independently_addressable_experts
    }

    /// Input modalities admitted by the exact normalized architecture.
    pub const fn input_modalities(self) -> InputModalities {
        self.input_modalities
    }

    const fn new(
        independently_addressable_experts: bool,
        input_modalities: InputModalities,
    ) -> Self {
        Self {
            independently_addressable_experts,
            input_modalities,
        }
    }
}

impl Default for ArchitectureCapabilities {
    fn default() -> Self {
        Self::new(false, InputModalities::TEXT)
    }
}

/// Failure while normalizing architecture policy for preparation.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("{0}")]
pub struct PreparationCapabilityError(String);

fn invalid(error: impl Display) -> PreparationCapabilityError {
    PreparationCapabilityError(error.to_string())
}

/// Derives preparation capabilities from a normalized SafeTensors family
/// configuration.
pub fn safetensors_capabilities(
    kind: ModelKind,
    config: &Value,
) -> Result<ArchitectureCapabilities, PreparationCapabilityError> {
    let (routed, input_modalities) = match kind {
        ModelKind::DeepSeekV3 => {
            crate::deepseek::parse_v3_config(config).map_err(invalid)?;
            (true, InputModalities::TEXT)
        }
        ModelKind::DeepSeekV4 => {
            crate::deepseek::parse_v4_config(config).map_err(invalid)?;
            (true, InputModalities::TEXT)
        }
        ModelKind::Gemma4 => {
            let bytes = serde_json::to_vec(config).map_err(invalid)?;
            let family = crate::gemma4::FamilyConfig::from_hf_json(&bytes).map_err(invalid)?;
            (family.text.num_experts.is_some(), family.input_modalities())
        }
        ModelKind::GptOss => {
            crate::gpt_oss::model_args_from_config_value(config).map_err(invalid)?;
            (true, InputModalities::TEXT)
        }
        ModelKind::Inkling => {
            let bytes = serde_json::to_vec(config).map_err(invalid)?;
            let args = crate::inkling::ModelArgs::from_hf_json(&bytes).map_err(invalid)?;
            let routed =
                args.text_config.layer_schedule.iter().any(|policy| {
                    policy.feed_forward == crate::inkling::FeedForwardPolicy::SparseMoe
                });
            (routed, args.input_modalities())
        }
        ModelKind::KimiLinear => (
            crate::kimi_linear::model_args_from_config_value(config)
                .map_err(invalid)?
                .has_sparse_moe_layers(),
            InputModalities::TEXT,
        ),
        ModelKind::Lfm2 => (
            crate::lfm2::model_args_from_config_value(config)
                .map_err(invalid)?
                .has_sparse_moe_layers(),
            InputModalities::TEXT,
        ),
        ModelKind::MuseGlimmer => {
            let args =
                crate::muse_glimmer::DecoderConfig::from_hf_value(config).map_err(invalid)?;
            (
                args.is_moe(),
                InputModalities {
                    text: true,
                    image: true,
                    audio: false,
                    video: args.weight_convention
                        == crate::muse_glimmer::WeightConvention::HuggingFace,
                },
            )
        }
        ModelKind::NemotronH => (
            crate::nemotron_h::model_args_from_config_value(config)
                .map_err(invalid)?
                .has_sparse_moe_layers(),
            InputModalities::TEXT,
        ),
        ModelKind::Qwen2 | ModelKind::Qwen3 => (
            crate::qwen::model_args_from_config_value(config)
                .map_err(invalid)?
                .is_moe(),
            InputModalities::TEXT,
        ),
        ModelKind::Qwen3Next | ModelKind::Qwen35 => {
            let args =
                crate::qwen::hybrid::model_args_from_config_value(config).map_err(invalid)?;
            let multimodal = args.vision.is_some();
            (
                args.text.is_moe(),
                InputModalities {
                    text: true,
                    image: multimodal,
                    audio: false,
                    video: multimodal,
                },
            )
        }
        ModelKind::Qwen3Vl | ModelKind::Qwen3VlMoe => {
            let args = crate::qwen::vl::model_args_from_config_value(config).map_err(invalid)?;
            (
                args.text.is_moe(),
                InputModalities {
                    text: true,
                    image: true,
                    audio: false,
                    video: true,
                },
            )
        }
        ModelKind::Moshi => (
            false,
            InputModalities {
                text: true,
                image: false,
                audio: true,
                video: false,
            },
        ),
        ModelKind::Llama => (false, InputModalities::TEXT),
    };
    Ok(ArchitectureCapabilities::new(routed, input_modalities))
}

struct GemmaCatalog<'a>(&'a GgufCheckpoint);

impl crate::gemma4::GgufTensorCatalog for GemmaCatalog<'_> {
    fn contains(&self, name: &str) -> bool {
        self.0
            .tensors()
            .any(|tensor| tensor.descriptor().name == name)
    }
}

/// Derives preparation capabilities from normalized GGUF architecture policy.
pub fn gguf_capabilities(
    architecture: GgufArchitecture,
    checkpoint: &GgufCheckpoint,
) -> Result<ArchitectureCapabilities, PreparationCapabilityError> {
    let metadata: HashMap<String, MetadataValue> = checkpoint
        .metadata()
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let routed = match architecture {
        GgufArchitecture::Gemma4 => {
            crate::gemma4::ModelArgs::from_gguf_metadata(&GemmaCatalog(checkpoint), &metadata)
                .map_err(invalid)?
                .num_experts
                .is_some()
        }
        GgufArchitecture::MuseGlimmer => crate::muse_glimmer::DecoderConfig::from_gguf_metadata(
            &metadata,
            checkpoint
                .tensors()
                .any(|tensor| tensor.descriptor().name == "output.weight"),
        )
        .map_err(invalid)?
        .is_moe(),
        GgufArchitecture::KimiLinear
        | GgufArchitecture::DeepSeek2
        | GgufArchitecture::DeepSeek4
        | GgufArchitecture::GptOss
        | GgufArchitecture::Inkling
        | GgufArchitecture::Lfm2Moe
        | GgufArchitecture::NemotronHMoe
        | GgufArchitecture::Qwen3Moe
        | GgufArchitecture::Qwen3VlMoe
        | GgufArchitecture::Qwen35Moe
        | GgufArchitecture::Qwen3Next => true,
        GgufArchitecture::Llama
        | GgufArchitecture::Mistral
        | GgufArchitecture::Lfm2
        | GgufArchitecture::NemotronH
        | GgufArchitecture::Qwen2
        | GgufArchitecture::Qwen3
        | GgufArchitecture::Qwen3Vl
        | GgufArchitecture::Qwen35 => false,
    };
    let input_modalities = match architecture {
        GgufArchitecture::Inkling => InputModalities {
            text: true,
            image: true,
            audio: true,
            video: false,
        },
        GgufArchitecture::Qwen3Vl | GgufArchitecture::Qwen3VlMoe => InputModalities {
            text: true,
            image: true,
            audio: false,
            video: true,
        },
        _ => InputModalities::TEXT,
    };
    Ok(ArchitectureCapabilities::new(routed, input_modalities))
}

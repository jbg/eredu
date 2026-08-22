//! Normalized architecture capabilities used during materialization planning.

use std::{collections::HashMap, fmt::Display};

use eredu_core::{GgufArchitecture, ModelKind};
use eredu_gguf::{Checkpoint as GgufCheckpoint, MetadataValue};
use serde_json::Value;

/// Preparation-relevant facts derived from one exact normalized architecture.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct ArchitectureCapabilities {
    independently_addressable_experts: bool,
}

impl ArchitectureCapabilities {
    /// Whether routed expert parameters can be managed independently.
    pub const fn independently_addressable_experts(self) -> bool {
        self.independently_addressable_experts
    }

    const fn with_independently_addressable_experts(value: bool) -> Self {
        Self {
            independently_addressable_experts: value,
        }
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
    let routed = match kind {
        ModelKind::DeepSeekV3 => {
            crate::deepseek::parse_v3_config(config).map_err(invalid)?;
            true
        }
        ModelKind::DeepSeekV4 => {
            crate::deepseek::parse_v4_config(config).map_err(invalid)?;
            true
        }
        ModelKind::Gemma4 => {
            let bytes = serde_json::to_vec(config).map_err(invalid)?;
            let family = crate::gemma4::FamilyConfig::from_hf_json(&bytes).map_err(invalid)?;
            family.text.num_experts.is_some()
        }
        ModelKind::GptOss => {
            crate::gpt_oss::model_args_from_config_value(config).map_err(invalid)?;
            true
        }
        ModelKind::Inkling => {
            let bytes = serde_json::to_vec(config).map_err(invalid)?;
            let args = crate::inkling::ModelArgs::from_hf_json(&bytes).map_err(invalid)?;
            let routed =
                args.text_config.layer_schedule.iter().any(|policy| {
                    policy.feed_forward == crate::inkling::FeedForwardPolicy::SparseMoe
                });
            routed
        }
        ModelKind::KimiLinear => crate::kimi_linear::model_args_from_config_value(config)
            .map_err(invalid)?
            .has_sparse_moe_layers(),
        ModelKind::Lfm2 => crate::lfm2::model_args_from_config_value(config)
            .map_err(invalid)?
            .has_sparse_moe_layers(),
        ModelKind::MuseGlimmer => crate::muse_glimmer::DecoderConfig::from_hf_value(config)
            .map_err(invalid)?
            .is_moe(),
        ModelKind::NemotronH => crate::nemotron_h::model_args_from_config_value(config)
            .map_err(invalid)?
            .has_sparse_moe_layers(),
        ModelKind::Qwen2 | ModelKind::Qwen3 => crate::qwen::model_args_from_config_value(config)
            .map_err(invalid)?
            .is_moe(),
        ModelKind::Qwen3Next | ModelKind::Qwen35 => {
            crate::qwen::hybrid::model_args_from_config_value(config)
                .map_err(invalid)?
                .text
                .is_moe()
        }
        ModelKind::Qwen3Vl | ModelKind::Qwen3VlMoe => {
            crate::qwen::vl::model_args_from_config_value(config)
                .map_err(invalid)?
                .text
                .is_moe()
        }
        ModelKind::Llama | ModelKind::Moshi => false,
    };
    Ok(ArchitectureCapabilities::with_independently_addressable_experts(routed))
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
    Ok(ArchitectureCapabilities::with_independently_addressable_experts(routed))
}

//! Normalized architecture capabilities used during materialization planning.

use std::{collections::HashMap, fmt::Display};

use eredu_core::{GgufArchitecture, InputModalities, ModelKind};
use eredu_gguf::{Checkpoint as GgufCheckpoint, MetadataValue};
use serde_json::Value;

/// Preparation-relevant facts derived from one exact normalized architecture.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ArchitectureCapabilities {
    parallel: ParallelCapabilityPlan,
    input_modalities: InputModalities,
}

impl ArchitectureCapabilities {
    /// Whether routed expert parameters can be managed independently.
    pub const fn independently_addressable_experts(self) -> bool {
        self.parallel.independent_expert_residency()
    }

    /// Distributed semantics supported by this exact normalized architecture.
    pub const fn parallel_plan(self) -> ParallelCapabilityPlan {
        self.parallel
    }

    /// Input modalities admitted by the exact normalized architecture.
    pub const fn input_modalities(self) -> InputModalities {
        self.input_modalities
    }

    const fn new(
        kind: ModelKind,
        independently_addressable_experts: bool,
        input_modalities: InputModalities,
    ) -> Self {
        Self {
            parallel: ParallelCapabilityPlan::new(
                true,
                !matches!(kind, ModelKind::Moshi),
                independently_addressable_experts,
                independently_addressable_experts,
            ),
            input_modalities,
        }
    }
}

impl Default for ArchitectureCapabilities {
    fn default() -> Self {
        Self {
            parallel: ParallelCapabilityPlan::default(),
            input_modalities: InputModalities::TEXT,
        }
    }
}

/// Artifact pieces that have passed structural validation for a GGUF model.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum GgufArtifactComposition {
    /// Only the language-model checkpoint has been validated.
    #[default]
    ModelOnly,
    /// The language-model checkpoint and its media projector have been validated.
    ValidatedMediaProjector,
}

/// Architecture-owned modality policy for a possibly composite GGUF artifact.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct GgufCompositeArtifactPlan {
    model_modalities: InputModalities,
    projector_modalities: Option<InputModalities>,
}

impl GgufCompositeArtifactPlan {
    const fn new(
        model_modalities: InputModalities,
        projector_modalities: Option<InputModalities>,
    ) -> Self {
        Self {
            model_modalities,
            projector_modalities,
        }
    }

    /// Input modalities provided by the validated artifact composition.
    pub const fn input_modalities(self, composition: GgufArtifactComposition) -> InputModalities {
        match (composition, self.projector_modalities) {
            (GgufArtifactComposition::ValidatedMediaProjector, Some(modalities)) => modalities,
            _ => self.model_modalities,
        }
    }
}

/// Declares how a validated sibling projector changes one GGUF architecture's
/// accepted input modalities.
pub const fn gguf_composite_artifact_plan(
    architecture: GgufArchitecture,
) -> GgufCompositeArtifactPlan {
    let (model_modalities, projector_modalities) = match architecture {
        GgufArchitecture::Inkling => {
            let modalities = InputModalities {
                text: true,
                image: true,
                audio: true,
                video: false,
            };
            (modalities, Some(modalities))
        }
        GgufArchitecture::Qwen3Vl | GgufArchitecture::Qwen3VlMoe => {
            let modalities = InputModalities {
                text: true,
                image: true,
                audio: false,
                video: true,
            };
            (modalities, Some(modalities))
        }
        GgufArchitecture::Qwen35 | GgufArchitecture::Qwen35Moe => (
            InputModalities::TEXT,
            Some(InputModalities {
                text: true,
                image: true,
                audio: false,
                video: true,
            }),
        ),
        GgufArchitecture::Gemma4 => (
            InputModalities::TEXT,
            Some(InputModalities {
                text: true,
                image: true,
                audio: true,
                video: false,
            }),
        ),
        GgufArchitecture::MuseGlimmer => (
            InputModalities::TEXT,
            Some(InputModalities {
                text: true,
                image: true,
                audio: false,
                video: false,
            }),
        ),
        _ => (InputModalities::TEXT, None),
    };
    GgufCompositeArtifactPlan::new(model_modalities, projector_modalities)
}

/// Architecture-owned distributed capabilities for one normalized model.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct ParallelCapabilityPlan {
    tensor_parallel: bool,
    pipeline_parallel: bool,
    expert_parallel: bool,
    independent_expert_residency: bool,
}

impl ParallelCapabilityPlan {
    const fn new(
        tensor_parallel: bool,
        pipeline_parallel: bool,
        expert_parallel: bool,
        independent_expert_residency: bool,
    ) -> Self {
        Self {
            tensor_parallel,
            pipeline_parallel,
            expert_parallel,
            independent_expert_residency,
        }
    }

    /// Whether the architecture declares tensor-sharded parameter and execution semantics.
    pub const fn tensor_parallel(self) -> bool {
        self.tensor_parallel
    }

    /// Whether the architecture can be partitioned into pipeline stages.
    pub const fn pipeline_parallel(self) -> bool {
        self.pipeline_parallel
    }

    /// Whether routed experts can be partitioned across expert ranks.
    pub const fn expert_parallel(self) -> bool {
        self.expert_parallel
    }

    /// Whether routed expert parameters can be materialized independently.
    pub const fn independent_expert_residency(self) -> bool {
        self.independent_expert_residency
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
    Ok(ArchitectureCapabilities::new(
        kind,
        routed,
        input_modalities,
    ))
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
    let input_modalities = gguf_composite_artifact_plan(architecture)
        .input_modalities(GgufArtifactComposition::ModelOnly);
    Ok(ArchitectureCapabilities::new(
        architecture.model_kind(),
        routed,
        input_modalities,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validated_optional_projectors_expand_gguf_modalities() {
        for (architecture, expected) in [
            (
                GgufArchitecture::Qwen35,
                InputModalities {
                    text: true,
                    image: true,
                    audio: false,
                    video: true,
                },
            ),
            (
                GgufArchitecture::Gemma4,
                InputModalities {
                    text: true,
                    image: true,
                    audio: true,
                    video: false,
                },
            ),
            (
                GgufArchitecture::MuseGlimmer,
                InputModalities {
                    text: true,
                    image: true,
                    audio: false,
                    video: false,
                },
            ),
        ] {
            let plan = gguf_composite_artifact_plan(architecture);
            assert_eq!(
                plan.input_modalities(GgufArtifactComposition::ModelOnly),
                InputModalities::TEXT
            );
            assert_eq!(
                plan.input_modalities(GgufArtifactComposition::ValidatedMediaProjector),
                expected
            );
        }
    }
}

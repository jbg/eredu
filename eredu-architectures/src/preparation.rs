//! Normalized architecture capabilities used during materialization planning.

use std::{collections::HashMap, fmt::Display};

use eredu_core::{GgufArchitecture, InputModalities, ModelKind};
use eredu_gguf::{Checkpoint as GgufCheckpoint, MetadataValue};
use serde_json::Value;

/// Preparation-relevant facts derived from one exact normalized architecture.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ArchitectureCapabilities {
    parallel: ParallelCapabilityPlan,
    independently_addressable_experts: bool,
    input_modalities: InputModalities,
    embedded_draft_layers: Option<usize>,
}

impl ArchitectureCapabilities {
    /// Whether routed expert parameters can be managed independently.
    pub const fn independently_addressable_experts(self) -> bool {
        self.independently_addressable_experts
    }

    /// Distributed semantics supported by this exact normalized architecture.
    pub const fn parallel_plan(self) -> ParallelCapabilityPlan {
        self.parallel
    }

    /// Input modalities admitted by the exact normalized architecture.
    pub const fn input_modalities(self) -> InputModalities {
        self.input_modalities
    }

    /// Exact embedded prediction depth exposed by the normalized architecture.
    ///
    /// `None` means that the artifact convention does not expose enough
    /// architecture-owned information to make an automatic drafting decision.
    pub const fn embedded_draft_layers(self) -> Option<usize> {
        self.embedded_draft_layers
    }

    const fn new(
        parallel: ParallelCapabilityPlan,
        independently_addressable_experts: bool,
        input_modalities: InputModalities,
        embedded_draft_layers: Option<usize>,
    ) -> Self {
        Self {
            parallel,
            independently_addressable_experts,
            input_modalities,
            embedded_draft_layers,
        }
    }
}

impl Default for ArchitectureCapabilities {
    fn default() -> Self {
        Self {
            parallel: ParallelCapabilityPlan::default(),
            independently_addressable_experts: false,
            input_modalities: InputModalities::TEXT,
            embedded_draft_layers: None,
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
}

impl ParallelCapabilityPlan {
    const TENSOR_ONLY: Self = Self::new(true, false, false);
    const TENSOR_PIPELINE: Self = Self::new(true, true, false);
    const TENSOR_PIPELINE_EXPERT: Self = Self::new(true, true, true);

    const fn new(tensor_parallel: bool, pipeline_parallel: bool, expert_parallel: bool) -> Self {
        Self {
            tensor_parallel,
            pipeline_parallel,
            expert_parallel,
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
    let (parallel, independently_addressable_experts, input_modalities, embedded_draft_layers) =
        match kind {
            ModelKind::DeepSeekV3 => {
                let args = crate::deepseek::parse_v3_config(config).map_err(invalid)?;
                (
                    ParallelCapabilityPlan::TENSOR_PIPELINE_EXPERT,
                    true,
                    InputModalities::TEXT,
                    usize::try_from(args.num_nextn_predict_layers).map_err(invalid)?,
                )
            }
            ModelKind::DeepSeekV4 => {
                let args = crate::deepseek::parse_v4_config(config).map_err(invalid)?;
                (
                    ParallelCapabilityPlan::TENSOR_PIPELINE_EXPERT,
                    true,
                    InputModalities::TEXT,
                    usize::try_from(args.num_nextn_predict_layers).map_err(invalid)?,
                )
            }
            ModelKind::Gemma4 => {
                let bytes = serde_json::to_vec(config).map_err(invalid)?;
                let family = crate::gemma4::FamilyConfig::from_hf_json(&bytes).map_err(invalid)?;
                let routed = family.text.num_experts.is_some();
                (
                    if routed {
                        ParallelCapabilityPlan::TENSOR_PIPELINE_EXPERT
                    } else {
                        ParallelCapabilityPlan::TENSOR_PIPELINE
                    },
                    routed,
                    family.input_modalities(),
                    0,
                )
            }
            ModelKind::GptOss => {
                crate::gpt_oss::model_args_from_config_value(config).map_err(invalid)?;
                (
                    ParallelCapabilityPlan::TENSOR_PIPELINE_EXPERT,
                    true,
                    InputModalities::TEXT,
                    0,
                )
            }
            ModelKind::Inkling => {
                let bytes = serde_json::to_vec(config).map_err(invalid)?;
                let args = crate::inkling::ModelArgs::from_hf_json(&bytes).map_err(invalid)?;
                let routed = args.text_config.layer_schedule.iter().any(|policy| {
                    policy.feed_forward == crate::inkling::FeedForwardPolicy::SparseMoe
                });
                let embedded = args
                    .mtp_config
                    .as_ref()
                    .map_or(0, |mtp| mtp.num_nextn_predict_layers);
                (
                    if routed {
                        ParallelCapabilityPlan::TENSOR_PIPELINE_EXPERT
                    } else {
                        ParallelCapabilityPlan::TENSOR_PIPELINE
                    },
                    routed,
                    args.input_modalities(),
                    usize::try_from(embedded).map_err(invalid)?,
                )
            }
            ModelKind::KimiLinear => {
                let args =
                    crate::kimi_linear::model_args_from_config_value(config).map_err(invalid)?;
                let routed = args.has_sparse_moe_layers();
                (
                    if routed {
                        ParallelCapabilityPlan::TENSOR_PIPELINE_EXPERT
                    } else {
                        ParallelCapabilityPlan::TENSOR_PIPELINE
                    },
                    routed,
                    InputModalities::TEXT,
                    0,
                )
            }
            ModelKind::Lfm2 => {
                let args = crate::lfm2::model_args_from_config_value(config).map_err(invalid)?;
                let routed = args.has_sparse_moe_layers();
                (
                    if routed {
                        ParallelCapabilityPlan::TENSOR_PIPELINE_EXPERT
                    } else {
                        ParallelCapabilityPlan::TENSOR_PIPELINE
                    },
                    routed,
                    InputModalities::TEXT,
                    0,
                )
            }
            ModelKind::MuseGlimmer => {
                let args =
                    crate::muse_glimmer::DecoderConfig::from_hf_value(config).map_err(invalid)?;
                let routed = args.is_moe();
                (
                    if routed {
                        ParallelCapabilityPlan::TENSOR_PIPELINE_EXPERT
                    } else {
                        ParallelCapabilityPlan::TENSOR_PIPELINE
                    },
                    routed,
                    InputModalities {
                        text: true,
                        image: true,
                        audio: false,
                        video: args.weight_convention
                            == crate::muse_glimmer::WeightConvention::HuggingFace,
                    },
                    0,
                )
            }
            ModelKind::NemotronH => {
                let args =
                    crate::nemotron_h::model_args_from_config_value(config).map_err(invalid)?;
                let routed = args.has_sparse_moe_layers();
                (
                    if routed {
                        ParallelCapabilityPlan::TENSOR_PIPELINE_EXPERT
                    } else {
                        ParallelCapabilityPlan::TENSOR_PIPELINE
                    },
                    routed,
                    InputModalities::TEXT,
                    usize::try_from(args.num_nextn_predict_layers).map_err(invalid)?,
                )
            }
            ModelKind::Qwen2 | ModelKind::Qwen3 => {
                let args = crate::qwen::model_args_from_config_value(config).map_err(invalid)?;
                let routed = args.is_moe();
                (
                    if routed {
                        ParallelCapabilityPlan::TENSOR_PIPELINE_EXPERT
                    } else {
                        ParallelCapabilityPlan::TENSOR_PIPELINE
                    },
                    routed,
                    InputModalities::TEXT,
                    0,
                )
            }
            ModelKind::Qwen3Next | ModelKind::Qwen35 => {
                let args =
                    crate::qwen::hybrid::model_args_from_config_value(config).map_err(invalid)?;
                let multimodal = args.vision.is_some();
                let routed = args.text.is_moe();
                (
                    if routed {
                        ParallelCapabilityPlan::TENSOR_PIPELINE_EXPERT
                    } else {
                        ParallelCapabilityPlan::TENSOR_PIPELINE
                    },
                    routed,
                    InputModalities {
                        text: true,
                        image: multimodal,
                        audio: false,
                        video: multimodal,
                    },
                    usize::try_from(args.text.mtp_num_hidden_layers).map_err(invalid)?,
                )
            }
            ModelKind::Qwen3Vl | ModelKind::Qwen3VlMoe => {
                let args =
                    crate::qwen::vl::model_args_from_config_value(config).map_err(invalid)?;
                let routed = args.text.is_moe();
                (
                    if routed {
                        ParallelCapabilityPlan::TENSOR_PIPELINE_EXPERT
                    } else {
                        ParallelCapabilityPlan::TENSOR_PIPELINE
                    },
                    routed,
                    InputModalities {
                        text: true,
                        image: true,
                        audio: false,
                        video: true,
                    },
                    0,
                )
            }
            ModelKind::Moshi => (
                ParallelCapabilityPlan::TENSOR_ONLY,
                false,
                InputModalities {
                    text: true,
                    image: false,
                    audio: true,
                    video: false,
                },
                0,
            ),
            ModelKind::Llama => {
                crate::llama::model_args_from_config_value(config).map_err(invalid)?;
                (
                    ParallelCapabilityPlan::TENSOR_PIPELINE,
                    false,
                    InputModalities::TEXT,
                    0,
                )
            }
        };
    Ok(ArchitectureCapabilities::new(
        parallel,
        independently_addressable_experts,
        input_modalities,
        Some(embedded_draft_layers),
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
    let (parallel, independently_addressable_experts) = match architecture {
        GgufArchitecture::Gemma4 => {
            let routed =
                crate::gemma4::ModelArgs::from_gguf_metadata(&GemmaCatalog(checkpoint), &metadata)
                    .map_err(invalid)?
                    .num_experts
                    .is_some();
            (
                if routed {
                    ParallelCapabilityPlan::TENSOR_PIPELINE_EXPERT
                } else {
                    ParallelCapabilityPlan::TENSOR_PIPELINE
                },
                routed,
            )
        }
        GgufArchitecture::MuseGlimmer => {
            let routed = crate::muse_glimmer::DecoderConfig::from_gguf_metadata(
                &metadata,
                checkpoint
                    .tensors()
                    .any(|tensor| tensor.descriptor().name == "output.weight"),
            )
            .map_err(invalid)?
            .is_moe();
            (
                if routed {
                    ParallelCapabilityPlan::TENSOR_PIPELINE_EXPERT
                } else {
                    ParallelCapabilityPlan::TENSOR_PIPELINE
                },
                routed,
            )
        }
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
        | GgufArchitecture::Qwen3Next => (ParallelCapabilityPlan::TENSOR_PIPELINE_EXPERT, true),
        GgufArchitecture::Llama
        | GgufArchitecture::Mistral
        | GgufArchitecture::Lfm2
        | GgufArchitecture::NemotronH
        | GgufArchitecture::Qwen2
        | GgufArchitecture::Qwen3
        | GgufArchitecture::Qwen3Vl
        | GgufArchitecture::Qwen35 => (ParallelCapabilityPlan::TENSOR_PIPELINE, false),
    };
    let input_modalities = gguf_composite_artifact_plan(architecture)
        .input_modalities(GgufArtifactComposition::ModelOnly);
    Ok(ArchitectureCapabilities::new(
        parallel,
        independently_addressable_experts,
        input_modalities,
        None,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen35_text_config(model_type: &str) -> Value {
        serde_json::json!({
            "model_type": model_type,
            "vocab_size": 64,
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "intermediate_size": 48,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 24,
            "num_experts_per_tok": 2,
            "num_experts": 8,
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention"
            ]
        })
    }

    #[test]
    fn parallel_capabilities_follow_the_exact_normalized_variant() {
        let dense =
            safetensors_capabilities(ModelKind::Qwen35, &qwen35_text_config("qwen3_5_text"))
                .unwrap();
        assert!(dense.parallel_plan().tensor_parallel());
        assert!(dense.parallel_plan().pipeline_parallel());
        assert!(!dense.parallel_plan().expert_parallel());
        assert!(!dense.independently_addressable_experts());

        let moe =
            safetensors_capabilities(ModelKind::Qwen35, &qwen35_text_config("qwen3_5_moe_text"))
                .unwrap();
        assert!(moe.parallel_plan().tensor_parallel());
        assert!(moe.parallel_plan().pipeline_parallel());
        assert!(moe.parallel_plan().expert_parallel());
        assert!(moe.independently_addressable_experts());

        let realtime = safetensors_capabilities(ModelKind::Moshi, &Value::Null).unwrap();
        assert!(realtime.parallel_plan().tensor_parallel());
        assert!(!realtime.parallel_plan().pipeline_parallel());
        assert!(!realtime.parallel_plan().expert_parallel());
        assert!(!realtime.independently_addressable_experts());
    }

    #[test]
    fn expert_execution_and_residency_capabilities_are_independent() {
        let capabilities = ArchitectureCapabilities::new(
            ParallelCapabilityPlan::TENSOR_PIPELINE_EXPERT,
            false,
            InputModalities::TEXT,
            None,
        );
        assert!(capabilities.parallel_plan().expert_parallel());
        assert!(!capabilities.independently_addressable_experts());

        let capabilities = ArchitectureCapabilities::new(
            ParallelCapabilityPlan::TENSOR_PIPELINE,
            true,
            InputModalities::TEXT,
            None,
        );
        assert!(!capabilities.parallel_plan().expert_parallel());
        assert!(capabilities.independently_addressable_experts());
    }

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

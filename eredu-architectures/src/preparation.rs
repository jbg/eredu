//! Normalized architecture capabilities used during materialization planning.

use std::fmt::Display;

use eredu_checkpoint::schema::{GgufCheckpointPlan, SafetensorsCheckpointPlan};
use eredu_core::checkpoint::{TensorCatalog, TensorDtype};
use eredu_core::InputModalities;

use crate::GgufArchitecture;

/// Preparation-relevant facts derived from one exact normalized architecture.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ArchitectureCapabilities {
    parallel: ParallelCapabilityPlan,
    independently_addressable_experts: bool,
    nonresident_safetensors_quantization: bool,
    input_modalities: InputModalities,
    embedded_draft_layers: Option<usize>,
}

impl ArchitectureCapabilities {
    /// Whether routed expert parameters can be managed independently.
    pub const fn independently_addressable_experts(self) -> bool {
        self.independently_addressable_experts
    }

    /// Whether the normalized architecture can transform a SafeTensors weight
    /// store before handing it to a nonresident execution policy.
    pub const fn nonresident_safetensors_quantization(self) -> bool {
        self.nonresident_safetensors_quantization
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
        nonresident_safetensors_quantization: bool,
        input_modalities: InputModalities,
        embedded_draft_layers: Option<usize>,
    ) -> Self {
        Self {
            parallel,
            independently_addressable_experts,
            nonresident_safetensors_quantization,
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
            nonresident_safetensors_quantization: false,
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

/// Whether one GGUF architecture admits a separately stored media projector.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum GgufMediaProjectorRequirement {
    /// The architecture has no media-projector composition.
    NotApplicable,
    /// A projector may extend the text-only model with media inputs.
    Optional,
    /// A validated projector is required for the model artifact.
    Required,
}

/// Architecture-owned projector and modality policy for a GGUF artifact.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct GgufCompositeArtifactPlan {
    model_modalities: InputModalities,
    projector_modalities: Option<InputModalities>,
    projector_requirement: GgufMediaProjectorRequirement,
}

impl GgufCompositeArtifactPlan {
    const fn new(
        model_modalities: InputModalities,
        projector_modalities: Option<InputModalities>,
        projector_requirement: GgufMediaProjectorRequirement,
    ) -> Self {
        Self {
            model_modalities,
            projector_modalities,
            projector_requirement,
        }
    }

    /// Architecture-owned presence policy for a sibling media projector.
    pub const fn media_projector_requirement(self) -> GgufMediaProjectorRequirement {
        self.projector_requirement
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
    use GgufMediaProjectorRequirement::{NotApplicable, Optional, Required};

    let (model_modalities, projector_modalities, projector_requirement) = match architecture {
        GgufArchitecture::NemotronLabsDiffusion => (
            InputModalities::TEXT,
            Some(InputModalities {
                text: true,
                image: true,
                audio: false,
                video: false,
            }),
            Optional,
        ),
        GgufArchitecture::VoiceChat => (
            InputModalities {
                text: true,
                image: false,
                audio: true,
                video: false,
            },
            None,
            NotApplicable,
        ),
        GgufArchitecture::Inkling => (
            InputModalities::TEXT,
            Some(InputModalities {
                text: true,
                image: true,
                audio: true,
                video: false,
            }),
            Optional,
        ),
        GgufArchitecture::Qwen3Vl | GgufArchitecture::Qwen3VlMoe => {
            let modalities = InputModalities {
                text: true,
                image: true,
                audio: false,
                video: true,
            };
            (modalities, Some(modalities), Required)
        }
        GgufArchitecture::Qwen35 | GgufArchitecture::Qwen35Moe | GgufArchitecture::Qwen4Exp => (
            InputModalities::TEXT,
            Some(InputModalities {
                text: true,
                image: true,
                audio: false,
                video: true,
            }),
            Optional,
        ),
        GgufArchitecture::Gemma4 => (
            InputModalities::TEXT,
            Some(InputModalities {
                text: true,
                image: true,
                audio: true,
                video: false,
            }),
            Optional,
        ),
        GgufArchitecture::MuseSpark => (
            InputModalities::TEXT,
            Some(InputModalities {
                text: true,
                image: true,
                audio: true,
                video: true,
            }),
            Optional,
        ),
        GgufArchitecture::MuseGlimmer => (
            InputModalities::TEXT,
            Some(InputModalities {
                text: true,
                image: true,
                audio: false,
                video: false,
            }),
            Optional,
        ),
        _ => (InputModalities::TEXT, None, NotApplicable),
    };
    GgufCompositeArtifactPlan::new(
        model_modalities,
        projector_modalities,
        projector_requirement,
    )
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

/// Architecture-resolved source of the dtype used by generic floating runtime
/// state.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FloatingStateDtypeSource {
    parameter: String,
    checkpoint_tensor: String,
    dtype: TensorDtype,
}

impl FloatingStateDtypeSource {
    /// Architecture-declared checkpoint parameter whose loaded values establish
    /// the ordinary decoder activation dtype.
    pub fn parameter(&self) -> &str {
        &self.parameter
    }

    /// Physical tensor name selected from the architecture's admitted aliases.
    pub fn checkpoint_tensor(&self) -> &str {
        &self.checkpoint_tensor
    }

    /// Inspected scalar dtype of the selected physical tensor.
    pub fn dtype(&self) -> &TensorDtype {
        &self.dtype
    }
}

fn resolve_floating_state_dtype_source(
    plan: &SafetensorsCheckpointPlan,
    parameter: &str,
    tensors: &TensorCatalog,
) -> Result<FloatingStateDtypeSource, PreparationCapabilityError> {
    let constraint = plan
        .common_tensors
        .iter()
        .find(|constraint| constraint.key == parameter)
        .ok_or_else(|| {
            invalid(format!(
                "architecture checkpoint plan {:?} does not declare floating-state dtype source {parameter:?} as a common tensor",
                plan.identity
            ))
        })?;
    resolve_declared_floating_state_dtype_source(
        &plan.identity,
        parameter,
        &constraint.aliases,
        tensors,
    )
}

fn resolve_gguf_floating_state_dtype_source(
    plan: &GgufCheckpointPlan,
    parameter: &str,
    tensors: &TensorCatalog,
) -> Result<FloatingStateDtypeSource, PreparationCapabilityError> {
    let constraint = plan
        .common_tensors
        .iter()
        .find(|constraint| constraint.key == parameter)
        .ok_or_else(|| {
            invalid(format!(
                "architecture checkpoint plan {:?} does not declare floating-state dtype source {parameter:?} as a common tensor",
                plan.identity
            ))
        })?;
    resolve_declared_floating_state_dtype_source(
        &plan.identity,
        parameter,
        &constraint.aliases,
        tensors,
    )
}

fn resolve_declared_floating_state_dtype_source(
    plan_identity: &str,
    parameter: &str,
    aliases: &[String],
    tensors: &TensorCatalog,
) -> Result<FloatingStateDtypeSource, PreparationCapabilityError> {
    let present = std::iter::once(parameter)
        .chain(aliases.iter().map(String::as_str))
        .filter_map(|name| tensors.get(name))
        .collect::<Vec<_>>();
    match present.as_slice() {
        [tensor] => Ok(FloatingStateDtypeSource {
            parameter: parameter.into(),
            checkpoint_tensor: tensor.name.clone(),
            dtype: tensor.dtype.clone(),
        }),
        [] => Err(invalid(format!(
            "architecture checkpoint plan {:?} did not find floating-state dtype source {parameter:?} or any of its declared aliases",
            plan_identity
        ))),
        tensors => Err(invalid(format!(
            "architecture checkpoint plan {:?} found multiple physical aliases for floating-state dtype source {parameter:?}: {:?}",
            plan_identity,
            tensors
                .iter()
                .map(|tensor| tensor.name.as_str())
                .collect::<Vec<_>>()
        ))),
    }
}

/// Resolves generic floating-state dtype from the exact SafeTensors plan retained at admission.
pub fn prepared_safetensors_floating_state_dtype_source(
    architecture: &crate::configuration::SafetensorsArchitecturePlan,
    tensors: &TensorCatalog,
) -> Result<FloatingStateDtypeSource, PreparationCapabilityError> {
    use crate::configuration::SafetensorsModelConfig;

    let parameter = match architecture.model() {
        SafetensorsModelConfig::NemotronLabsDiffusion(config) => {
            if config.tensor_fp8() {
                // Scalar-scaled FP8 projection recipes decode to F32. The
                // admitted F32 scale establishes cache arithmetic even when
                // embeddings and normalization gains retain BF16 storage.
                "model.layers.0.self_attn.q_proj.weight_scale".into()
            } else {
                "model.embed_tokens.weight".into()
            }
        }
        SafetensorsModelConfig::DiffusionGemma(_) => "model.decoder.embed_tokens.weight".into(),
        SafetensorsModelConfig::Gemma4(_) | SafetensorsModelConfig::Muse(_) => {
            "model.language_model.embed_tokens.weight".into()
        }
        SafetensorsModelConfig::GptOss(args) => {
            format!("{}.embed_tokens.weight", args.parameter_root)
        }
        SafetensorsModelConfig::Inkling(_) => "model.llm.embed.weight".into(),
        SafetensorsModelConfig::NemotronH(_) => "backbone.embeddings.weight".into(),
        SafetensorsModelConfig::Qwen(args) => {
            format!("{}.embed_tokens.weight", args.parameter_root)
        }
        SafetensorsModelConfig::QwenVl(args) => {
            format!("{}.embed_tokens.weight", args.text.parameter_root)
        }
        SafetensorsModelConfig::Moshi(_) | SafetensorsModelConfig::VoiceChat(_) => {
            return Err(invalid(
                "Realtime floating-state dtype belongs to the realtime loader contract",
            ));
        }
        SafetensorsModelConfig::DeepSeekV4(_) => "embed.weight".into(),
        SafetensorsModelConfig::DeepSeekV3(_)
        | SafetensorsModelConfig::K2Horizon(_)
        | SafetensorsModelConfig::KimiLinear(_)
        | SafetensorsModelConfig::Gemma2(_)
        | SafetensorsModelConfig::Llama(_)
        | SafetensorsModelConfig::Nanbeige(_)
        | SafetensorsModelConfig::Lfm2(_)
        | SafetensorsModelConfig::QwenHybrid(_)
        | SafetensorsModelConfig::Qwen4Exp(_, _) => "model.embed_tokens.weight".into(),
    };
    resolve_floating_state_dtype_source(architecture.checkpoint(), &parameter, tensors)
}

/// Resolves generic floating-state dtype from the exact GGUF plan retained at admission.
pub fn prepared_gguf_floating_state_dtype_source(
    architecture: &crate::configuration::GgufArchitecturePlan,
    tensors: &TensorCatalog,
) -> Result<FloatingStateDtypeSource, PreparationCapabilityError> {
    use crate::configuration::GgufModelConfig;

    let parameter = match architecture.model() {
        GgufModelConfig::NemotronLabsDiffusion(_) => "token_embd.weight",
        GgufModelConfig::VoiceChat(_) => {
            return Err(invalid(
                "Realtime floating-state dtype belongs to the realtime loader contract",
            ));
        }
        GgufModelConfig::DeepSeekV3(_)
        | GgufModelConfig::DeepSeekV4(_)
        | GgufModelConfig::DiffusionGemma(_)
        | GgufModelConfig::Gemma4(_)
        | GgufModelConfig::GptOss(_)
        | GgufModelConfig::Inkling(_)
        | GgufModelConfig::K2Horizon(_)
        | GgufModelConfig::KimiLinear(_)
        | GgufModelConfig::Lfm2(_)
        | GgufModelConfig::Gemma2(_)
        | GgufModelConfig::Llama(_)
        | GgufModelConfig::Nanbeige(_)
        | GgufModelConfig::Muse(_)
        | GgufModelConfig::NemotronH(_)
        | GgufModelConfig::Qwen(_)
        | GgufModelConfig::QwenHybrid(_)
        | GgufModelConfig::Qwen4Exp(_) => "token_embd.weight",
    };
    resolve_gguf_floating_state_dtype_source(architecture.checkpoint(), parameter, tensors)
}

/// Resolves the floating-state dtype source from one completely inspected artifact.
///
/// Container dispatch remains architecture-owned so concrete backends receive only the
/// neutral dtype source selected from the exact admitted plan and tensor catalog.
pub fn prepared_floating_state_dtype_source(
    inspection: &eredu_core::ArtifactInspection<crate::processor_plan::ArtifactArchitecturePlan>,
) -> Result<FloatingStateDtypeSource, PreparationCapabilityError> {
    match inspection.format() {
        eredu_core::ArtifactFormat::SafeTensors => {
            let architecture = inspection
                .architecture_plan()
                .safetensors_architecture()
                .ok_or_else(|| invalid("SafeTensors inspection omitted its architecture plan"))?;
            prepared_safetensors_floating_state_dtype_source(architecture, inspection.tensors())
        }
        eredu_core::ArtifactFormat::Gguf => {
            let architecture = inspection
                .architecture_plan()
                .gguf_plan()
                .ok_or_else(|| invalid("GGUF inspection omitted its architecture plan"))?;
            prepared_gguf_floating_state_dtype_source(architecture, inspection.tensors())
        }
        _ => Err(invalid(
            "unsupported artifact format for floating-state dtype inspection",
        )),
    }
}

/// Derives preparation capabilities from the exact SafeTensors plan retained at admission.
pub fn prepared_safetensors_capabilities(
    architecture: &crate::configuration::SafetensorsArchitecturePlan,
) -> Result<ArchitectureCapabilities, PreparationCapabilityError> {
    use crate::configuration::SafetensorsModelConfig;

    let (parallel, independently_addressable_experts, input_modalities, embedded_draft_layers) =
        match architecture.model() {
            SafetensorsModelConfig::NemotronLabsDiffusion(args) => (
                ParallelCapabilityPlan::TENSOR_PIPELINE,
                false,
                InputModalities {
                    text: true,
                    image: args.vision.is_some(),
                    audio: false,
                    video: false,
                },
                0,
            ),
            SafetensorsModelConfig::DeepSeekV3(args) => {
                let mut capabilities = routed_text(args.has_sparse_moe_layers());
                capabilities.3 = usize::try_from(args.num_nextn_predict_layers).map_err(invalid)?;
                capabilities
            }
            SafetensorsModelConfig::DeepSeekV4(args) => (
                ParallelCapabilityPlan::TENSOR_PIPELINE_EXPERT,
                true,
                InputModalities::TEXT,
                usize::try_from(args.num_nextn_predict_layers).map_err(invalid)?,
            ),
            SafetensorsModelConfig::DiffusionGemma(family) => (
                ParallelCapabilityPlan::TENSOR_PIPELINE_EXPERT,
                true,
                family.input_modalities(),
                0,
            ),
            SafetensorsModelConfig::Gemma4(family) => {
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
            SafetensorsModelConfig::GptOss(_) => (
                ParallelCapabilityPlan::TENSOR_PIPELINE_EXPERT,
                true,
                InputModalities::TEXT,
                0,
            ),
            SafetensorsModelConfig::Inkling(args) => {
                let routed = args.text_config.has_sparse_moe_layers();
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
            SafetensorsModelConfig::KimiLinear(args) => routed_text(args.has_sparse_moe_layers()),
            SafetensorsModelConfig::Lfm2(args) => routed_text(args.has_sparse_moe_layers()),
            SafetensorsModelConfig::Nanbeige(_) => (
                ParallelCapabilityPlan::TENSOR_PIPELINE,
                false,
                InputModalities::TEXT,
                0,
            ),
            SafetensorsModelConfig::Gemma2(_) | SafetensorsModelConfig::Llama(_) => (
                ParallelCapabilityPlan::TENSOR_PIPELINE,
                false,
                InputModalities::TEXT,
                0,
            ),
            SafetensorsModelConfig::Muse(args) => {
                let routed = args.is_moe();
                (
                    routed_parallel(routed),
                    routed,
                    InputModalities {
                        text: true,
                        image: true,
                        audio: false,
                        video: args.weight_convention == crate::muse::WeightConvention::HuggingFace,
                    },
                    0,
                )
            }
            SafetensorsModelConfig::NemotronH(args) => {
                let mut capabilities = routed_text(args.has_sparse_moe_layers());
                capabilities.3 = usize::try_from(args.num_nextn_predict_layers).map_err(invalid)?;
                capabilities
            }
            SafetensorsModelConfig::K2Horizon(args) => routed_text(args.is_moe()),
            SafetensorsModelConfig::Qwen(args) => routed_text(args.is_moe()),
            SafetensorsModelConfig::Qwen4Exp(config, _) => (
                routed_parallel(true),
                true,
                InputModalities {
                    text: true,
                    image: config.vision.is_some(),
                    audio: false,
                    video: config.vision.is_some(),
                },
                config
                    .prediction
                    .as_ref()
                    .map_or(0, |prediction| prediction.layers.len()),
            ),
            SafetensorsModelConfig::QwenHybrid(args) => {
                let routed = args.text.is_moe();
                let multimodal = args.vision.is_some();
                (
                    routed_parallel(routed),
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
            SafetensorsModelConfig::QwenVl(args) => {
                let routed = args.text.is_moe();
                (
                    routed_parallel(routed),
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
            SafetensorsModelConfig::VoiceChat(_) => (
                ParallelCapabilityPlan::default(),
                false,
                InputModalities {
                    text: true,
                    image: false,
                    audio: true,
                    video: false,
                },
                0,
            ),
            SafetensorsModelConfig::Moshi(_) => (
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
        };
    let nonresident_safetensors_quantization = !matches!(
        architecture.model(),
        SafetensorsModelConfig::Moshi(_) | SafetensorsModelConfig::VoiceChat(_)
    );
    Ok(ArchitectureCapabilities::new(
        parallel,
        independently_addressable_experts,
        nonresident_safetensors_quantization,
        input_modalities,
        Some(embedded_draft_layers),
    ))
}

fn routed_parallel(routed: bool) -> ParallelCapabilityPlan {
    if routed {
        ParallelCapabilityPlan::TENSOR_PIPELINE_EXPERT
    } else {
        ParallelCapabilityPlan::TENSOR_PIPELINE
    }
}

fn routed_text(routed: bool) -> (ParallelCapabilityPlan, bool, InputModalities, usize) {
    (routed_parallel(routed), routed, InputModalities::TEXT, 0)
}

/// Derives preparation capabilities from the exact GGUF plan retained at admission.
pub fn prepared_gguf_capabilities(
    plan: &crate::configuration::GgufArchitecturePlan,
) -> ArchitectureCapabilities {
    use crate::configuration::GgufModelConfig;

    let architecture = plan.architecture();
    let routed = match plan.model() {
        GgufModelConfig::NemotronLabsDiffusion(_) => false,
        GgufModelConfig::DiffusionGemma(family) => family.text.num_experts.is_some(),
        GgufModelConfig::Gemma4(family) => family.text.num_experts.is_some(),
        GgufModelConfig::Muse(args) => args.is_moe(),
        GgufModelConfig::DeepSeekV3(args) => args.has_sparse_moe_layers(),
        GgufModelConfig::DeepSeekV4(_) | GgufModelConfig::GptOss(_) => true,
        GgufModelConfig::Inkling(args) => args.text_config.has_sparse_moe_layers(),
        GgufModelConfig::KimiLinear(args) => args.has_sparse_moe_layers(),
        GgufModelConfig::Lfm2(args) => args.has_sparse_moe_layers(),
        GgufModelConfig::NemotronH(args) => args.has_sparse_moe_layers(),
        GgufModelConfig::K2Horizon(args) => args.is_moe(),
        GgufModelConfig::Qwen(args) => args.is_moe(),
        GgufModelConfig::QwenHybrid(args) => args.text.is_moe(),
        GgufModelConfig::Qwen4Exp(_) => true,
        GgufModelConfig::VoiceChat(_) => false,
        GgufModelConfig::Gemma2(_) | GgufModelConfig::Llama(_) | GgufModelConfig::Nanbeige(_) => {
            false
        }
    };
    ArchitectureCapabilities::new(
        if architecture == GgufArchitecture::VoiceChat {
            ParallelCapabilityPlan::default()
        } else {
            routed_parallel(routed)
        },
        routed,
        false,
        gguf_composite_artifact_plan(architecture)
            .input_modalities(GgufArtifactComposition::ModelOnly),
        None,
    )
}

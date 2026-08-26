//! Normalized architecture capabilities used during materialization planning.

use std::fmt::Display;

use eredu_checkpoint::schema::{GgufCheckpointPlan, SafetensorsCheckpointPlan};
use eredu_core::checkpoint::{TensorCatalog, TensorDtype};
use eredu_core::InputModalities;
#[cfg(test)]
use serde_json::Value;

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
        GgufArchitecture::Inkling => (
            InputModalities::TEXT,
            Some(InputModalities {
                text: true,
                image: true,
                audio: true,
                video: false,
            }),
        ),
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

/// Architecture-resolved source of the scalar dtype used by mutable runtime
/// state.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RuntimeStateDtypeSource {
    parameter: String,
    checkpoint_tensor: String,
    dtype: TensorDtype,
}

impl RuntimeStateDtypeSource {
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

fn resolve_runtime_state_dtype_source(
    plan: &SafetensorsCheckpointPlan,
    parameter: &str,
    tensors: &TensorCatalog,
) -> Result<RuntimeStateDtypeSource, PreparationCapabilityError> {
    let constraint = plan
        .common_tensors
        .iter()
        .find(|constraint| constraint.key == parameter)
        .ok_or_else(|| {
            invalid(format!(
                "architecture checkpoint plan {:?} does not declare runtime-state dtype source {parameter:?} as a common tensor",
                plan.identity
            ))
        })?;
    resolve_declared_runtime_state_dtype_source(
        &plan.identity,
        parameter,
        &constraint.aliases,
        tensors,
    )
}

fn resolve_gguf_runtime_state_dtype_source(
    plan: &GgufCheckpointPlan,
    parameter: &str,
    tensors: &TensorCatalog,
) -> Result<RuntimeStateDtypeSource, PreparationCapabilityError> {
    let constraint = plan
        .common_tensors
        .iter()
        .find(|constraint| constraint.key == parameter)
        .ok_or_else(|| {
            invalid(format!(
                "architecture checkpoint plan {:?} does not declare runtime-state dtype source {parameter:?} as a common tensor",
                plan.identity
            ))
        })?;
    resolve_declared_runtime_state_dtype_source(
        &plan.identity,
        parameter,
        &constraint.aliases,
        tensors,
    )
}

fn resolve_declared_runtime_state_dtype_source(
    plan_identity: &str,
    parameter: &str,
    aliases: &[String],
    tensors: &TensorCatalog,
) -> Result<RuntimeStateDtypeSource, PreparationCapabilityError> {
    let present = std::iter::once(parameter)
        .chain(aliases.iter().map(String::as_str))
        .filter_map(|name| tensors.get(name))
        .collect::<Vec<_>>();
    match present.as_slice() {
        [tensor] => Ok(RuntimeStateDtypeSource {
            parameter: parameter.into(),
            checkpoint_tensor: tensor.name.clone(),
            dtype: tensor.dtype.clone(),
        }),
        [] => Err(invalid(format!(
            "architecture checkpoint plan {:?} did not find runtime-state dtype source {parameter:?} or any of its declared aliases",
            plan_identity
        ))),
        tensors => Err(invalid(format!(
            "architecture checkpoint plan {:?} found multiple physical aliases for runtime-state dtype source {parameter:?}: {:?}",
            plan_identity,
            tensors
                .iter()
                .map(|tensor| tensor.name.as_str())
                .collect::<Vec<_>>()
        ))),
    }
}

/// Resolves runtime-state dtype from the exact SafeTensors plan retained at admission.
pub fn prepared_safetensors_runtime_state_dtype_source(
    architecture: &crate::configuration::SafetensorsArchitecturePlan,
    tensors: &TensorCatalog,
) -> Result<RuntimeStateDtypeSource, PreparationCapabilityError> {
    use crate::configuration::SafetensorsModelConfig;

    let parameter = match architecture.model() {
        SafetensorsModelConfig::Gemma4(_) | SafetensorsModelConfig::MuseGlimmer(_) => {
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
        SafetensorsModelConfig::Moshi(_) => {
            return Err(invalid(
                "Moshi runtime-state dtype belongs to the realtime loader contract",
            ));
        }
        SafetensorsModelConfig::DeepSeekV3(_)
        | SafetensorsModelConfig::DeepSeekV4(_)
        | SafetensorsModelConfig::KimiLinear(_)
        | SafetensorsModelConfig::Llama(_)
        | SafetensorsModelConfig::Lfm2(_)
        | SafetensorsModelConfig::QwenHybrid(_) => "model.embed_tokens.weight".into(),
    };
    resolve_runtime_state_dtype_source(architecture.checkpoint(), &parameter, tensors)
}

/// Resolves runtime-state dtype from the exact GGUF plan retained at admission.
pub fn prepared_gguf_runtime_state_dtype_source(
    architecture: &crate::configuration::GgufArchitecturePlan,
    tensors: &TensorCatalog,
) -> Result<RuntimeStateDtypeSource, PreparationCapabilityError> {
    use crate::configuration::GgufModelConfig;

    let parameter = match architecture.model() {
        GgufModelConfig::DeepSeekV3(_)
        | GgufModelConfig::DeepSeekV4(_)
        | GgufModelConfig::Gemma4(_)
        | GgufModelConfig::GptOss(_)
        | GgufModelConfig::Inkling(_)
        | GgufModelConfig::KimiLinear(_)
        | GgufModelConfig::Lfm2(_)
        | GgufModelConfig::Llama(_)
        | GgufModelConfig::MuseGlimmer(_)
        | GgufModelConfig::NemotronH(_)
        | GgufModelConfig::Qwen(_)
        | GgufModelConfig::QwenHybrid(_) => "token_embd.weight",
    };
    resolve_gguf_runtime_state_dtype_source(architecture.checkpoint(), parameter, tensors)
}

/// Derives preparation capabilities from the exact SafeTensors plan retained at admission.
pub fn prepared_safetensors_capabilities(
    architecture: &crate::configuration::SafetensorsArchitecturePlan,
) -> Result<ArchitectureCapabilities, PreparationCapabilityError> {
    use crate::configuration::SafetensorsModelConfig;

    let (parallel, independently_addressable_experts, input_modalities, embedded_draft_layers) =
        match architecture.model() {
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
            SafetensorsModelConfig::Llama(_) => (
                ParallelCapabilityPlan::TENSOR_PIPELINE,
                false,
                InputModalities::TEXT,
                0,
            ),
            SafetensorsModelConfig::MuseGlimmer(args) => {
                let routed = args.is_moe();
                (
                    routed_parallel(routed),
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
            SafetensorsModelConfig::NemotronH(args) => {
                let mut capabilities = routed_text(args.has_sparse_moe_layers());
                capabilities.3 = usize::try_from(args.num_nextn_predict_layers).map_err(invalid)?;
                capabilities
            }
            SafetensorsModelConfig::Qwen(args) => routed_text(args.is_moe()),
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
    let nonresident_safetensors_quantization =
        !matches!(architecture.model(), SafetensorsModelConfig::Moshi(_));
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
        GgufModelConfig::Gemma4(family) => family.text.num_experts.is_some(),
        GgufModelConfig::MuseGlimmer(args) => args.is_moe(),
        GgufModelConfig::DeepSeekV3(args) => args.has_sparse_moe_layers(),
        GgufModelConfig::DeepSeekV4(_) | GgufModelConfig::GptOss(_) => true,
        GgufModelConfig::Inkling(args) => args.text_config.has_sparse_moe_layers(),
        GgufModelConfig::KimiLinear(args) => args.has_sparse_moe_layers(),
        GgufModelConfig::Lfm2(args) => args.has_sparse_moe_layers(),
        GgufModelConfig::NemotronH(args) => args.has_sparse_moe_layers(),
        GgufModelConfig::Qwen(args) => args.is_moe(),
        GgufModelConfig::QwenHybrid(args) => args.text.is_moe(),
        GgufModelConfig::Llama(_) => false,
    };
    ArchitectureCapabilities::new(
        routed_parallel(routed),
        routed,
        false,
        gguf_composite_artifact_plan(architecture)
            .input_modalities(GgufArtifactComposition::ModelOnly),
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dtype_source_plan() -> SafetensorsCheckpointPlan {
        SafetensorsCheckpointPlan::new(
            "test architecture",
            vec![
                eredu_checkpoint::schema::SafetensorsTensorConstraint::required(
                    "released.embedding.weight",
                    vec![32, 16],
                    eredu_checkpoint::schema::StoredDtypeConstraint::Floating,
                )
                .with_aliases(["canonical.embedding.weight"]),
            ],
            Vec::new(),
            eredu_checkpoint::schema::CatalogPolicy::strict(),
        )
        .unwrap()
    }

    fn gguf_dtype_source_plan() -> GgufCheckpointPlan {
        GgufCheckpointPlan::new(
            "test GGUF architecture",
            vec![eredu_checkpoint::schema::GgufTensorConstraint::required(
                "token_embd.weight",
                vec![32, 16],
                eredu_checkpoint::schema::GgufTypeConstraint::OperationClass(
                    eredu_checkpoint::schema::TensorOperation::Matrix,
                ),
            )
            .with_aliases(["released_token_embd.weight"])],
            Vec::new(),
            eredu_checkpoint::schema::CatalogPolicy::strict(),
        )
        .unwrap()
    }

    fn tensor(name: &str, dtype: TensorDtype) -> eredu_core::checkpoint::TensorDescriptor {
        eredu_core::checkpoint::TensorDescriptor {
            name: name.into(),
            shape: vec![32, 16],
            dtype,
            storage: None,
        }
    }

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

    fn deepseek_v3_config(first_k_dense_replace: i32) -> Value {
        serde_json::json!({
            "architectures": ["DeepseekV3ForCausalLM"],
            "model_type": "deepseek_v3",
            "hidden_size": 16,
            "intermediate_size": 32,
            "moe_intermediate_size": 8,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "vocab_size": 128,
            "max_position_embeddings": 4096,
            "q_lora_rank": 4,
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 6,
            "qk_rope_head_dim": 2,
            "v_head_dim": 8,
            "first_k_dense_replace": first_k_dense_replace,
            "moe_layer_freq": 2,
            "n_routed_experts": 8,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "n_group": 2,
            "topk_group": 1,
            "topk_method": "noaux_tc",
            "scoring_func": "sigmoid",
            "norm_topk_prob": true,
            "routed_scaling_factor": 1.0,
            "tie_word_embeddings": false,
            "attention_dropout": 0.0,
            "hidden_act": "silu"
        })
    }

    fn kimi_linear_config(first_k_dense_replace: i32) -> Value {
        serde_json::json!({
            "model_type":"kimi_linear","vocab_size":16,"hidden_size":12,
            "num_hidden_layers":2,"num_attention_heads":3,"num_key_value_heads":3,
            "intermediate_size":17,"head_dim":4,"model_max_length":64,
            "linear_attn_config":{"kda_layers":[1],"full_attn_layers":[2],
                "num_heads":3,"head_dim":4,"short_conv_kernel_size":3},
            "num_experts":2,"moe_intermediate_size":9,"kv_lora_rank":6,
            "qk_nope_head_dim":4,"qk_rope_head_dim":2,"v_head_dim":4,
            "mla_use_nope":true,"num_experts_per_token":1,"num_shared_experts":1,
            "routed_scaling_factor":1.0,"first_k_dense_replace":first_k_dense_replace,
            "num_expert_group":1,"topk_group":1
        })
    }

    fn inkling_config(mlp_layer_types: &[&str]) -> Value {
        serde_json::json!({
            "model_type":"inkling_mm_model","image_token_id":60,"audio_token_id":61,
            "text_config":{
                "hidden_size":16,"num_hidden_layers":3,"vocab_size":64,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,
                "sliding_window_size":8,
                "layer_types":["sliding_attention","full_attention","sliding_attention"],
                "mlp_layer_types":mlp_layer_types,"sconv_kernel_size":4,
                "d_rel":2,"rel_extent":16,"intermediate_size":32,
                "n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1
            }
        })
    }

    fn safetensors_plan(config: &Value) -> crate::configuration::SafetensorsArchitecturePlan {
        crate::configuration::resolve_model_config(config)
            .unwrap()
            .architecture
    }

    #[test]
    fn parallel_capabilities_follow_the_exact_normalized_variant() {
        let dense_plan = safetensors_plan(&qwen35_text_config("qwen3_5_text"));
        let dense = prepared_safetensors_capabilities(&dense_plan).unwrap();
        assert!(dense.parallel_plan().tensor_parallel());
        assert!(dense.parallel_plan().pipeline_parallel());
        assert!(!dense.parallel_plan().expert_parallel());
        assert!(!dense.independently_addressable_experts());

        let moe_plan = safetensors_plan(&qwen35_text_config("qwen3_5_moe_text"));
        let moe = prepared_safetensors_capabilities(&moe_plan).unwrap();
        assert!(moe.parallel_plan().tensor_parallel());
        assert!(moe.parallel_plan().pipeline_parallel());
        assert!(moe.parallel_plan().expert_parallel());
        assert!(moe.independently_addressable_experts());

        let realtime_plan = safetensors_plan(&serde_json::json!({
            "model_type": "moshi", "dim": 32, "text_card": 101,
            "n_q": 4, "dep_q": 3, "generated_audio_codebooks": 2, "card": 64,
            "num_heads": 4, "num_layers": 2, "dim_feedforward": 48,
            "causal": true, "context": 7, "max_period": 10000.0,
            "positional_embedding": "rope", "depformer_dim": 24,
            "depformer_dim_feedforward": 36, "depformer_num_heads": 4,
            "depformer_num_layers": 2, "depformer_context": 3,
            "depformer_max_period": 10000.0, "depformer_pos_emb": "none",
            "delays": [0, 0, 1, 2, 1]
        }));
        let realtime = prepared_safetensors_capabilities(&realtime_plan).unwrap();
        assert!(realtime.parallel_plan().tensor_parallel());
        assert!(!realtime.parallel_plan().pipeline_parallel());
        assert!(!realtime.parallel_plan().expert_parallel());
        assert!(!realtime.independently_addressable_experts());
    }

    fn assert_dense_capabilities(capabilities: ArchitectureCapabilities) {
        assert!(!capabilities.parallel_plan().expert_parallel());
        assert!(!capabilities.independently_addressable_experts());
    }

    #[test]
    fn dense_deepseek_v3_capabilities_follow_the_schedule_in_both_formats() {
        let config = deepseek_v3_config(4);
        let safetensors = safetensors_plan(&config);
        assert_dense_capabilities(prepared_safetensors_capabilities(&safetensors).unwrap());

        let args = crate::deepseek::parse_v3_config(&config).unwrap();
        assert!(!args.has_sparse_moe_layers());
        let checkpoint = crate::deepseek::v3_gguf_plan(&args).unwrap();
        let gguf = crate::configuration::GgufArchitecturePlan::new(
            GgufArchitecture::DeepSeek2,
            crate::configuration::GgufModelConfig::DeepSeekV3(args),
            checkpoint,
        );
        assert_dense_capabilities(prepared_gguf_capabilities(&gguf));
    }

    #[test]
    fn deepseek_v3_prediction_layers_retain_routed_capabilities() {
        let mut config = deepseek_v3_config(4);
        config["num_nextn_predict_layers"] = 1.into();
        let safetensors = safetensors_plan(&config);
        let capabilities = prepared_safetensors_capabilities(&safetensors).unwrap();

        assert!(capabilities.parallel_plan().expert_parallel());
        assert!(capabilities.independently_addressable_experts());
    }

    #[test]
    fn dense_kimi_linear_and_inkling_gguf_capabilities_follow_their_schedules() {
        let kimi =
            crate::kimi_linear::model_args_from_config_value(&kimi_linear_config(2)).unwrap();
        assert!(!kimi.has_sparse_moe_layers());
        let checkpoint = crate::kimi_linear::gguf_plan(&kimi).unwrap();
        let plan = crate::configuration::GgufArchitecturePlan::new(
            GgufArchitecture::KimiLinear,
            crate::configuration::GgufModelConfig::KimiLinear(kimi),
            checkpoint,
        );
        assert_dense_capabilities(prepared_gguf_capabilities(&plan));

        let config = inkling_config(&["dense", "dense", "dense"]);
        let inkling =
            crate::inkling::ModelArgs::from_hf_json(&serde_json::to_vec(&config).unwrap()).unwrap();
        assert!(!inkling.text_config.has_sparse_moe_layers());
        let checkpoint = crate::inkling::gguf_plan(&inkling).unwrap();
        let plan = crate::configuration::GgufArchitecturePlan::new(
            GgufArchitecture::Inkling,
            crate::configuration::GgufModelConfig::Inkling(inkling),
            checkpoint,
        );
        assert_dense_capabilities(prepared_gguf_capabilities(&plan));
    }

    #[test]
    fn runtime_state_dtype_source_uses_architecture_declared_aliases() {
        let catalog =
            TensorCatalog::new([tensor("canonical.embedding.weight", TensorDtype::Bf16)]).unwrap();
        let source = resolve_runtime_state_dtype_source(
            &dtype_source_plan(),
            "released.embedding.weight",
            &catalog,
        )
        .unwrap();

        assert_eq!(source.parameter(), "released.embedding.weight");
        assert_eq!(source.checkpoint_tensor(), "canonical.embedding.weight");
        assert_eq!(source.dtype(), &TensorDtype::Bf16);
    }

    #[test]
    fn gguf_runtime_state_dtype_source_preserves_dense_half_widths() {
        for dtype in [TensorDtype::F16, TensorDtype::Bf16] {
            let catalog =
                TensorCatalog::new([tensor("released_token_embd.weight", dtype.clone())]).unwrap();
            let source = resolve_gguf_runtime_state_dtype_source(
                &gguf_dtype_source_plan(),
                "token_embd.weight",
                &catalog,
            )
            .unwrap();

            assert_eq!(source.parameter(), "token_embd.weight");
            assert_eq!(source.checkpoint_tensor(), "released_token_embd.weight");
            assert_eq!(source.dtype(), &dtype);
        }
    }

    #[test]
    fn normalized_architecture_selects_its_state_dtype_parameter() {
        let config = qwen35_text_config("qwen3_5_text");
        let plan = safetensors_plan(&config);
        let catalog =
            TensorCatalog::new([tensor("model.embed_tokens.weight", TensorDtype::F16)]).unwrap();

        let source = prepared_safetensors_runtime_state_dtype_source(&plan, &catalog).unwrap();
        assert_eq!(source.parameter(), "model.embed_tokens.weight");
        assert_eq!(source.checkpoint_tensor(), "model.embed_tokens.weight");
        assert_eq!(source.dtype(), &TensorDtype::F16);
    }

    #[test]
    fn runtime_state_dtype_source_rejects_missing_or_ambiguous_names() {
        let unknown = TensorCatalog::new([tensor("new.valid.name", TensorDtype::F16)]).unwrap();
        let error = resolve_runtime_state_dtype_source(
            &dtype_source_plan(),
            "released.embedding.weight",
            &unknown,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("did not find runtime-state dtype source"));

        let ambiguous = TensorCatalog::new([
            tensor("released.embedding.weight", TensorDtype::F16),
            tensor("canonical.embedding.weight", TensorDtype::F16),
        ])
        .unwrap();
        let error = resolve_runtime_state_dtype_source(
            &dtype_source_plan(),
            "released.embedding.weight",
            &ambiguous,
        )
        .unwrap_err();
        assert!(error.to_string().contains("multiple physical aliases"));
    }

    #[test]
    fn expert_execution_and_residency_capabilities_are_independent() {
        let capabilities = ArchitectureCapabilities::new(
            ParallelCapabilityPlan::TENSOR_PIPELINE_EXPERT,
            false,
            false,
            InputModalities::TEXT,
            None,
        );
        assert!(capabilities.parallel_plan().expert_parallel());
        assert!(!capabilities.independently_addressable_experts());

        let capabilities = ArchitectureCapabilities::new(
            ParallelCapabilityPlan::TENSOR_PIPELINE,
            true,
            false,
            InputModalities::TEXT,
            None,
        );
        assert!(!capabilities.parallel_plan().expert_parallel());
        assert!(capabilities.independently_addressable_experts());
    }

    #[test]
    fn kimi_linear_declares_nonresident_load_time_quantization() {
        let config = kimi_linear_config(1);
        let plan = safetensors_plan(&config);
        let capabilities = prepared_safetensors_capabilities(&plan).unwrap();
        assert!(capabilities.nonresident_safetensors_quantization());
    }

    #[test]
    fn validated_optional_projectors_expand_gguf_modalities() {
        for (architecture, expected) in [
            (
                GgufArchitecture::Inkling,
                InputModalities {
                    text: true,
                    image: true,
                    audio: true,
                    video: false,
                },
            ),
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

//! MLX architecture binding against portable checkpoint catalogs.

use std::collections::HashMap;

use crate::backend::runtime::checkpoint::gguf::GgufCheckpoint;
use eredu_gguf::MetadataValue as GgufMetadataValue;
#[cfg(test)]
use serde_json::Value;

use eredu_architectures::{GgufArchitecture, ModelKind};

use super::realization::{
    requires_distributed_stage, AddressableParameterBankBinding, FamilyBinding, GgufBinding,
};
#[cfg(test)]
use super::MlxLoadRequest;
use crate::backend::error::Error;

/// Native view of the GGUF source admitted by portable inspection.
///
/// Family composition consumes this type instead of repeating architecture
/// admission or reaching upward from reusable backend runtime code.
pub(crate) struct AdmittedGguf {
    plan: eredu_architectures::configuration::GgufArchitecturePlan,
    checkpoint: GgufCheckpoint,
    metadata: HashMap<String, GgufMetadataValue>,
}

/// Native projector payload paired with its architecture-owned admission proof.
///
/// Family composition may lower this plan into MLX checkpoint sources, but it
/// must not reconstruct family geometry or checkpoint schemas from the payload.
pub(crate) struct AdmittedGgufProjector {
    plan: eredu_architectures::gguf_companion::GgufMediaProjectorPlan,
    checkpoint: GgufCheckpoint,
}

impl AdmittedGgufProjector {
    pub(crate) const fn plan(
        &self,
    ) -> &eredu_architectures::gguf_companion::GgufMediaProjectorPlan {
        &self.plan
    }

    pub(crate) const fn model(
        &self,
    ) -> &eredu_architectures::gguf_companion::GgufMediaProjectorConfig {
        self.plan.model()
    }

    pub(crate) fn checkpoint(&self) -> &GgufCheckpoint {
        &self.checkpoint
    }
}

impl AdmittedGguf {
    pub(crate) fn from_admission(
        plan: eredu_architectures::configuration::GgufArchitecturePlan,
        projector_plan: Option<eredu_architectures::gguf_companion::GgufMediaProjectorPlan>,
        validated: eredu_core::ValidatedGguf,
    ) -> Result<(Self, Option<AdmittedGgufProjector>), Error> {
        let (checkpoint, mut companions) = validated.into_parts();
        let checkpoint = GgufCheckpoint::from_portable(checkpoint);
        let metadata = crate::backend::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        let projector = companions.remove(&eredu_core::GgufCompanionRole::MediaProjector);
        let projector = match (projector_plan, projector) {
            (Some(plan), Some(projector)) => Some(AdmittedGgufProjector {
                plan,
                checkpoint: GgufCheckpoint::from_portable(projector.checkpoint().clone()),
            }),
            (None, None) => None,
            (Some(_), None) => {
                return Err(Error::ArchitectureModel(
                    "GGUF preparation retained a media-projector plan without its admitted checkpoint"
                        .into(),
                ));
            }
            (None, Some(_)) => {
                return Err(Error::ArchitectureModel(
                    "GGUF preparation retained a media-projector checkpoint without its typed architecture plan"
                        .into(),
                ));
            }
        };
        Ok((
            Self {
                plan,
                checkpoint,
                metadata,
            },
            projector,
        ))
    }

    pub(crate) const fn architecture(&self) -> GgufArchitecture {
        self.plan.architecture()
    }

    pub(crate) const fn plan(&self) -> &eredu_architectures::configuration::GgufArchitecturePlan {
        &self.plan
    }

    pub(crate) const fn model(&self) -> &eredu_architectures::configuration::GgufModelConfig {
        self.plan.model()
    }

    pub(crate) fn checkpoint(&self) -> &GgufCheckpoint {
        &self.checkpoint
    }

    pub(crate) fn metadata(&self) -> &HashMap<String, GgufMetadataValue> {
        &self.metadata
    }
}

pub(crate) fn validate_complete_gguf_quantization(
    kind: ModelKind,
    requested: bool,
) -> Result<(), Error> {
    if !requested || GgufBinding::for_kind(kind).is_some_and(GgufBinding::accepts_quantization) {
        return Ok(());
    }
    Err(Error::Artifact(
        eredu_core::artifact::ArtifactError::UnsupportedQuantizationPolicy(format!(
            "load-time quantization is unavailable for complete GGUF {} materialization on MLX",
            kind.canonical_name()
        )),
    ))
}

fn validate_quantization_capability(
    kind: ModelKind,
    format: eredu_core::ArtifactFormat,
    policy: eredu_core::PreparationPolicy,
    capabilities: eredu_architectures::preparation::ArchitectureCapabilities,
) -> Result<(), Error> {
    if policy.quantization().is_none() {
        return Ok(());
    }
    if let Some(topology) = policy
        .topology()
        .filter(|topology| !topology.is_replicated())
    {
        if requires_distributed_stage(kind, topology) {
            return Ok(());
        }
        return Err(Error::Artifact(
            eredu_core::artifact::ArtifactError::UnsupportedQuantizationPolicy(format!(
                "load-time quantization is unavailable for complete tensor-parallel {} materialization on MLX",
                kind.canonical_name()
            )),
        ));
    }
    if format == eredu_core::ArtifactFormat::Gguf {
        return validate_complete_gguf_quantization(kind, true);
    }
    let supported = FamilyBinding::for_kind(kind).is_some()
        && (policy.residency() == eredu_core::ResidencyRequest::FullyResident
            || capabilities.nonresident_safetensors_quantization());
    if supported {
        return Ok(());
    }
    Err(Error::Artifact(
        eredu_core::artifact::ArtifactError::UnsupportedQuantizationPolicy(format!(
            "load-time quantization is unavailable for the normalized {} architecture from SafeTensors with {:?} weights on MLX",
            kind.canonical_name(),
            policy.residency(),
        )),
    ))
}

fn validate_parameter_bank_capability(
    kind: ModelKind,
    capabilities: eredu_architectures::preparation::ArchitectureCapabilities,
) -> Result<(), Error> {
    if capabilities.independently_addressable_experts()
        && AddressableParameterBankBinding::for_kind(kind).is_some()
    {
        return Ok(());
    }
    Err(Error::Artifact(
        eredu_core::artifact::ArtifactError::UnsupportedResidencyPolicy(format!(
            "independent expert caching is unavailable for the normalized {} architecture on MLX",
            kind.canonical_name()
        )),
    ))
}

fn validate_preparation_capability_intersection(
    kind: ModelKind,
    format: eredu_core::ArtifactFormat,
    policy: eredu_core::PreparationPolicy,
    capabilities: eredu_architectures::preparation::ArchitectureCapabilities,
) -> Result<(), Error> {
    if let Some(topology) = policy.topology() {
        let format_name = match format {
            eredu_core::ArtifactFormat::SafeTensors => "SafeTensors",
            eredu_core::ArtifactFormat::Gguf => "GGUF",
            _ => {
                return Err(Error::ArchitectureModel(
                    "unsupported artifact format selected for parallel validation".into(),
                ));
            }
        };
        validate_parallel_capabilities(capabilities, topology, format_name, kind.canonical_name())?;
    }
    validate_quantization_capability(kind, format, policy, capabilities)?;
    if policy.residency() == eredu_core::ResidencyRequest::AddressableParameterBanks {
        validate_parameter_bank_capability(kind, capabilities)?;
    }
    Ok(())
}

fn requires_architecture_capabilities(policy: eredu_core::PreparationPolicy) -> bool {
    policy
        .topology()
        .is_some_and(|topology| !topology.is_replicated())
        || policy.residency() == eredu_core::ResidencyRequest::AddressableParameterBanks
        || policy.quantization().is_some()
}

pub(crate) fn validate_parallel_capabilities(
    capabilities: eredu_architectures::preparation::ArchitectureCapabilities,
    topology: eredu_core::ParallelTopology,
    artifact: &str,
    architecture: &str,
) -> Result<(), Error> {
    let plan = capabilities.parallel_plan();
    let unsupported = |capability: &str| {
        Error::Parallel(format!(
            "{artifact} architecture {architecture:?} has no architecture-owned {capability} plan; no checkpoint payload was materialized"
        ))
    };
    if topology.is_axis_active(eredu_core::ParallelAxis::Pipeline) && !plan.pipeline_parallel() {
        return Err(unsupported("pipeline-parallel"));
    }
    if topology.is_axis_active(eredu_core::ParallelAxis::Tensor) && !plan.tensor_parallel() {
        return Err(unsupported("tensor-parallel"));
    }
    if topology.is_axis_active(eredu_core::ParallelAxis::Expert) && !plan.expert_parallel() {
        return Err(unsupported("expert-parallel"));
    }
    Ok(())
}

#[cfg(test)]
fn validate_safetensors_preparation_for_test(
    kind: ModelKind,
    config: &Value,
    options: MlxLoadRequest,
) -> Result<(), Error> {
    let policy = options.preparation_policy()?;
    eredu_core::validate_preparation_policy(kind.loading_protocol(), policy)?;
    if !requires_architecture_capabilities(policy) {
        return Ok(());
    }
    let resolved = eredu_architectures::configuration::resolve_model_config(config)?;
    if resolved.kind != kind {
        return Err(Error::ArchitectureModel(format!(
            "test requested {kind:?} preparation for a configuration resolved as {:?}",
            resolved.kind
        )));
    }
    let capabilities =
        eredu_architectures::preparation::prepared_safetensors_capabilities(&resolved.architecture)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    validate_preparation_capability_intersection(
        kind,
        eredu_core::ArtifactFormat::SafeTensors,
        policy,
        capabilities,
    )
}

pub(crate) fn validate_inspected_preparation(
    inspection: &eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    policy: eredu_core::PreparationPolicy,
) -> Result<(), Error> {
    eredu_core::validate_preparation_policy(inspection.configuration().loading_protocol(), policy)?;
    if policy
        .topology()
        .is_none_or(|topology| topology.is_replicated())
    {
        let execution =
            eredu_architectures::replicated_text::replicated_text_execution_class(inspection)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        if matches!(
            execution,
            eredu_architectures::replicated_text::ReplicatedTextExecutionClass::Replicated(_)
        ) && policy.residency() == eredu_core::ResidencyRequest::AddressableParameterBanks
        {
            return Err(Error::Artifact(
                eredu_core::artifact::ArtifactError::UnsupportedResidencyPolicy(
                    "independent parameter banks require an architecture-routed text class".into(),
                ),
            ));
        }
        if matches!(
            execution,
            eredu_architectures::replicated_text::ReplicatedTextExecutionClass::Replicated(_)
                | eredu_architectures::replicated_text::ReplicatedTextExecutionClass::Routed(_)
        ) {
            // Generic text classes validate concrete MLX mechanisms through
            // their neutral selection immediately after this protocol check.
            // Excluded-family bindings are only authoritative for `Other`.
            return Ok(());
        }
    }
    if !requires_architecture_capabilities(policy) {
        return Ok(());
    }
    let architecture_plan = inspection.architecture_plan();
    let kind = architecture_plan.model_kind();
    let capabilities = match inspection.format() {
        eredu_core::ArtifactFormat::SafeTensors => {
            eredu_architectures::preparation::prepared_safetensors_capabilities(
                architecture_plan
                    .safetensors_architecture()
                    .ok_or_else(|| {
                        Error::ArchitectureModel(
                            "SafeTensors preparation omitted its validated architecture plan"
                                .into(),
                        )
                    })?,
            )
        }
        eredu_core::ArtifactFormat::Gguf => Ok(
            eredu_architectures::preparation::prepared_gguf_capabilities(
                architecture_plan.gguf_plan().ok_or_else(|| {
                    Error::ArchitectureModel(
                        "GGUF preparation omitted its validated architecture plan".into(),
                    )
                })?,
            ),
        ),
        _ => {
            return Err(Error::ArchitectureModel(
                "unsupported artifact format selected during structural validation".into(),
            ));
        }
    }
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    validate_preparation_capability_intersection(kind, inspection.format(), policy, capabilities)
}

/// Derives exact MLX session capabilities without opening checkpoint payloads.
pub(crate) fn inspected_session_capabilities(
    inspection: &eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    policy: eredu_core::PreparationPolicy,
) -> Result<eredu_core::SessionCapabilities, Error> {
    validate_inspected_preparation(inspection, policy)?;
    Ok(eredu_core::SessionCapabilities::new(true, true, true))
}

#[cfg(test)]
mod admission_policy_tests {
    use super::*;

    fn parameter_bank_options() -> MlxLoadRequest {
        MlxLoadRequest::default().with_weight_residency(
            eredu_runtime::WeightResidency::with_independent_parameter_banks(
                eredu_runtime::OrdinaryWeightResidency::FullyResident,
                eredu_runtime::ParameterBankLoadOptions::default(),
            ),
        )
    }

    fn kimi_linear_config() -> Value {
        serde_json::json!({
            "model_type":"kimi_linear","vocab_size":16,"hidden_size":12,"num_hidden_layers":2,
            "num_attention_heads":3,"num_key_value_heads":3,"intermediate_size":17,"head_dim":4,
            "model_max_length":64,"linear_attn_config":{"kda_layers":[1],"full_attn_layers":[2],"num_heads":3,"head_dim":4,"short_conv_kernel_size":3},
            "num_experts":2,"moe_intermediate_size":9,"kv_lora_rank":6,"qk_nope_head_dim":4,"qk_rope_head_dim":2,"v_head_dim":4,
            "mla_use_nope":true,"num_experts_per_token":1,"num_shared_experts":1,"routed_scaling_factor":1.0,
            "first_k_dense_replace":1,"num_expert_group":1,"topk_group":1
        })
    }

    fn gemma4_config(routed: bool) -> Value {
        let mut config = serde_json::json!({
            "model_type":"gemma4", "tie_word_embeddings":true,
            "text_config":{
                "model_type":"gemma4_text", "hidden_size":32,
                "num_hidden_layers":1, "intermediate_size":64,
                "num_attention_heads":4, "num_key_value_heads":2, "head_dim":8,
                "rms_norm_eps":0.00001, "vocab_size":32,
                "max_position_embeddings":128, "layer_types":["full_attention"],
                "enable_moe_block":false
            }
        });
        if routed {
            config["text_config"]["enable_moe_block"] = true.into();
            config["text_config"]["num_experts"] = 2.into();
            config["text_config"]["top_k_experts"] = 1.into();
            config["text_config"]["moe_intermediate_size"] = 32.into();
        }
        config
    }

    fn muse_glimmer_config(routed: bool) -> Value {
        let mut config = serde_json::json!({
            "architectures":["MuseGlimmerForConditionalGeneration"],
            "model_type":"muse_glimmer", "image_token_id":22, "video_token_id":23,
            "out_hidden_size":32, "projector_hidden_size":16,
            "text_config":{
                "model_type":"muse_glimmer_text", "hidden_size":16,
                "num_hidden_layers":2, "intermediate_size":24,
                "num_attention_heads":4, "num_key_value_heads":2, "head_dim":4,
                "rms_norm_eps":0.00001, "post_norm_eps":0.00001,
                "vocab_size":24, "max_position_embeddings":64,
                "rope_theta":10000.0,
                "layer_types":["sliding_attention","full_attention"],
                "layer_rope_theta":[10000.0,0.0], "sliding_window":8,
                "tie_word_embeddings":false, "hidden_act":"silu",
                "attention_dropout":0.0, "qk_scale_factor":1.0,
                "output_multiplier":1.0, "final_logit_softcapping":30.0
            },
            "vision_config":{
                "model_type":"muse_glimmer_vision", "hidden_size":8,
                "intermediate_size":12, "num_attention_heads":2,
                "num_hidden_layers":1, "patch_size":2, "patch_temporal":1,
                "merge_size":2, "pos_emb_height":2, "pos_emb_width":2,
                "max_position_embeddings":4, "layer_norm_eps":0.00001,
                "hidden_act":"gelu", "layer_types":["full_attention"],
                "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"}
            }
        });
        if routed {
            config["text_config"]["intermediate_size"] = 0.into();
            config["text_config"]["moe_intermediate_size"] = 12.into();
            config["text_config"]["num_experts"] = 8.into();
            config["text_config"]["num_experts_per_tok"] = 2.into();
            config["text_config"]["norm_topk_prob"] = true.into();
        }
        config
    }

    fn dense_qwen3_config() -> Value {
        serde_json::json!({
            "model_type":"qwen3", "hidden_size":16, "num_hidden_layers":3,
            "intermediate_size":32, "num_attention_heads":4,
            "num_key_value_heads":2, "head_dim":4, "rms_norm_eps":0.000001,
            "vocab_size":64, "max_position_embeddings":128,
            "rope_theta":1000000.0, "tie_word_embeddings":true
        })
    }

    fn dense_deepseek_v3_config() -> Value {
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
            "first_k_dense_replace": 4,
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

    #[test]
    fn parameter_bank_admits_normalized_gemma4_and_muse_glimmer_moe() {
        let options = parameter_bank_options();
        validate_safetensors_preparation_for_test(
            ModelKind::Gemma4,
            &gemma4_config(true),
            options.clone(),
        )
        .unwrap();
        validate_safetensors_preparation_for_test(
            ModelKind::MuseGlimmer,
            &muse_glimmer_config(true),
            options,
        )
        .unwrap();
    }

    #[test]
    fn nonresident_quantization_admits_kimi_linear_capability_intersection() {
        let options = MlxLoadRequest::with_quantization(eredu_core::QuantizationRequest::MxFp4)
            .with_weight_residency(eredu_runtime::WeightResidency::layerwise_host(
                eredu_runtime::LayerwiseLoadOptions::default(),
            ));
        validate_safetensors_preparation_for_test(
            ModelKind::KimiLinear,
            &kimi_linear_config(),
            options,
        )
        .unwrap();
    }

    #[test]
    fn complete_tensor_parallel_quantization_is_rejected_during_preflight() {
        let topology = crate::composition::mlx::distributed::topology::MlxParallelPlan::for_rank(
            0,
            2,
            1,
            1,
            crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
        )
        .unwrap();
        let options = MlxLoadRequest::with_quantization(eredu_core::QuantizationRequest::MxFp4)
            .with_parallel_topology(
                topology,
                eredu_runtime::PipelineWireContract::new(
                    eredu_runtime::PipelineActivationDtype::Float32,
                ),
            );

        let error = validate_safetensors_preparation_for_test(
            ModelKind::Qwen3,
            &dense_qwen3_config(),
            options,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            Error::Artifact(
                eredu_core::artifact::ArtifactError::UnsupportedQuantizationPolicy(message)
            ) if message.contains("complete tensor-parallel")
        ));
    }

    #[test]
    fn distributed_stage_quantization_is_admitted_during_preflight() {
        let topology = crate::composition::mlx::distributed::topology::MlxParallelPlan::for_rank(
            0,
            1,
            2,
            1,
            crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
        )
        .unwrap();
        let options = MlxLoadRequest::with_quantization(eredu_core::QuantizationRequest::MxFp4)
            .with_parallel_topology(
                topology,
                eredu_runtime::PipelineWireContract::new(
                    eredu_runtime::PipelineActivationDtype::Float32,
                ),
            );

        validate_safetensors_preparation_for_test(ModelKind::Qwen3, &dense_qwen3_config(), options)
            .unwrap();
    }

    #[test]
    fn replicated_gguf_quantization_requires_a_bound_complete_loader() {
        let policy = eredu_core::PreparationPolicy::new(
            Some(eredu_core::QuantizationRequest::MxFp4),
            eredu_core::ResidencyRequest::FullyResident,
        );
        let capabilities = eredu_architectures::preparation::ArchitectureCapabilities::default();

        validate_quantization_capability(
            ModelKind::Qwen3,
            eredu_core::ArtifactFormat::Gguf,
            policy,
            capabilities,
        )
        .unwrap();
        let error = validate_quantization_capability(
            ModelKind::DeepSeekV3,
            eredu_core::ArtifactFormat::Gguf,
            policy,
            capabilities,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            Error::Artifact(
                eredu_core::artifact::ArtifactError::UnsupportedQuantizationPolicy(message)
            ) if message.contains("deepseek_v3") && message.contains("GGUF")
        ));
    }

    #[test]
    fn parameter_bank_rejects_dense_variants_of_mixed_families() {
        let options = parameter_bank_options();
        for (kind, config) in [
            (ModelKind::DeepSeekV3, dense_deepseek_v3_config()),
            (ModelKind::Gemma4, gemma4_config(false)),
            (ModelKind::MuseGlimmer, muse_glimmer_config(false)),
            (ModelKind::Qwen3, dense_qwen3_config()),
        ] {
            assert!(matches!(
                validate_safetensors_preparation_for_test(kind, &config, options.clone()),
                Err(Error::Artifact(
                    eredu_core::artifact::ArtifactError::UnsupportedResidencyPolicy(_)
                ))
            ));
        }
    }

    #[test]
    fn preparation_rejects_an_unsupported_exact_parallel_axis() {
        let topology = crate::composition::mlx::distributed::topology::MlxParallelPlan::for_rank(
            0,
            1,
            1,
            2,
            crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
        )
        .unwrap();
        let error = validate_safetensors_preparation_for_test(
            ModelKind::Qwen3,
            &dense_qwen3_config(),
            MlxLoadRequest::with_parallel(
                topology,
                eredu_runtime::PipelineWireContract::new(
                    eredu_runtime::PipelineActivationDtype::Float32,
                ),
            ),
        )
        .unwrap_err();

        assert!(matches!(error, Error::Parallel(message) if message.contains("expert-parallel")));
    }

    #[test]
    fn preparation_accepts_supported_exact_parallel_axes() {
        let topology = crate::composition::mlx::distributed::topology::MlxParallelPlan::for_rank(
            0,
            2,
            3,
            1,
            crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
        )
        .unwrap();

        validate_safetensors_preparation_for_test(
            ModelKind::Qwen3,
            &dense_qwen3_config(),
            MlxLoadRequest::with_parallel(
                topology,
                eredu_runtime::PipelineWireContract::new(
                    eredu_runtime::PipelineActivationDtype::Float32,
                ),
            ),
        )
        .unwrap();
    }
}

#[cfg(test)]
mod neutral_qwen_tests {
    use std::collections::BTreeSet;

    fn qwen2_args(tied: bool) -> eredu_architectures::qwen::ModelArgs {
        eredu_architectures::qwen::model_args_from_config_value(&serde_json::json!({
            "model_type": "qwen2", "hidden_size": 8, "num_hidden_layers": 2,
            "intermediate_size": 16, "num_attention_heads": 4,
            "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "vocab_size": 32,
            "max_position_embeddings": 64, "rope_theta": 10000.0,
            "tie_word_embeddings": tied, "use_sliding_window": false
        }))
        .unwrap()
    }

    #[test]
    fn qwen2_plan_is_exactly_biased_and_has_no_qk_norms() {
        let tied = eredu_architectures::qwen::safetensors_plan(&qwen2_args(true)).unwrap();
        let names = tied
            .common_tensors
            .iter()
            .map(|tensor| tensor.key.as_str())
            .collect::<BTreeSet<_>>();
        assert!(names.contains("model.layers.0.self_attn.q_proj.bias"));
        assert!(names.contains("model.layers.0.self_attn.k_proj.bias"));
        assert!(names.contains("model.layers.0.self_attn.v_proj.bias"));
        assert!(!names.contains("model.layers.0.self_attn.q_norm.weight"));
        assert!(!names.contains("model.layers.0.self_attn.k_norm.weight"));
        assert!(!names.contains("lm_head.weight"));
        let redundant_head = tied
            .layout_groups
            .iter()
            .find(|group| group.id == "redundant tied output head")
            .expect("optional redundant tied head layout");
        assert!(!redundant_head.required);
        assert_eq!(redundant_head.variants[0].tensors[0].key, "lm_head.weight");
        let untied = eredu_architectures::qwen::safetensors_plan(&qwen2_args(false)).unwrap();
        assert!(untied
            .common_tensors
            .iter()
            .any(|tensor| tensor.key == "lm_head.weight"));
    }
}

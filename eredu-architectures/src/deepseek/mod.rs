//! Shared backend-neutral DeepSeek architecture policy.

/// Stable segment identity for target decoder state.
pub const TARGET_STATE_SEGMENT: &str = "target";
/// Stable segment identity for checkpoint-embedded prediction state.
pub const PREDICTION_STATE_SEGMENT: &str = "prediction";

/// Family-specific compressed-attention equations.
pub mod attention;
/// Shared normalization, residual, and feed-forward sequencing.
pub mod block;
/// Pure SafeTensors/GGUF schemas and canonical name translation.
pub mod checkpoint;
/// Strict V3/R1 and V4 configuration normalization.
pub mod config;
/// Shared routed-plus-shared expert block.
pub mod moe;
/// Shared embedded-prediction layers and outputs.
pub mod mtp;
/// Semantic tensor/expert/hyper-stream placement plans.
pub mod parallel;
/// Shared normalized low-rank projection assembly.
pub mod projection;
/// Thin V3/R1 architecture policy.
pub mod v3;
/// Thin V4 architecture policy.
pub mod v4;

pub use checkpoint::{
    expert_unit_recipes, normalize_v3_weight_formats, normalize_v4_weight_formats,
    translate_v3_gguf_weight_name, translate_v4_gguf_weight_name, v3_expert_recipes,
    v3_expert_residency_catalog, v3_gguf_kv_b_recipe, v3_gguf_plan, v3_load_time_quantization,
    v3_safetensors_plan, v3_unit_recipes, v3_with_checkpoint_formats, v4_expert_recipes,
    v4_expert_residency_catalog, v4_gguf_plan, v4_load_time_quantization, v4_safetensors_plan,
    v4_with_checkpoint_formats, ExpertUnitRecipes,
};
pub use config::{
    parse_v3_config, parse_v3_gguf, parse_v4_config, parse_v4_gguf, v3_architecture_fingerprint,
    v3_uses_split_kv, v4_architecture_fingerprint, ConfigError, DeepSeekQuantizationConfig,
    DsparkConfig, ExpertFormat, Fp8QuantizationConfig, LayerPolicy, V3Args, V4Args,
    V4AttentionPolicy, YarnConfig,
};

/// Derives complete V3 routed-expert ownership and rank-local bank geometry.
pub fn v3_expert_realization_plan<B>(
    architecture: &v3::Model<B>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<
    Option<crate::ExpertRealizationPlan<eredu_nn::GatedProductExpertBankSpec>>,
    eredu_nn::Error,
>
where
    B: eredu_nn::RoutedNeuralBackend + eredu_nn::BlockwiseAttentionBackend,
{
    let args = architecture.args();
    let global_experts =
        usize::try_from(args.n_routed_experts).map_err(eredu_nn::Error::backend)?;
    let local_experts = i32::try_from(
        eredu_core::balanced_contiguous_range(
            global_experts,
            topology.expert_parallel_size,
            topology.expert_parallel_rank,
            false,
        )
        .map_err(eredu_nn::Error::backend)?
        .len(),
    )
    .map_err(eredu_nn::Error::backend)?;
    let local_args = architecture.shared_parallel_geometry();
    let local_width = local_args
        .as_ref()
        .map_or(args.moe_intermediate_size, |geometry| {
            geometry.args().moe_intermediate_size
        });
    let target = usize::try_from(args.num_hidden_layers).map_err(eredu_nn::Error::backend)?;
    let prediction =
        usize::try_from(args.num_nextn_predict_layers).map_err(eredu_nn::Error::backend)?;
    let target_group =
        eredu_runtime::ExecutionGroupId::new("target").map_err(eredu_nn::Error::backend)?;
    let mut unit_specs = std::collections::BTreeMap::new();
    for layer in 0..target {
        if args.layer_schedule.get(layer) != Some(&LayerPolicy::SparseMoe) {
            continue;
        }
        unit_specs.insert(
            (target_group.clone(), layer),
            v3::localized_expert_bank_spec(args, layer, local_experts, local_width)?,
        );
    }
    for depth in 0..prediction {
        let group = eredu_runtime::ExecutionGroupId::new(format!("mtp.{depth}"))
            .map_err(eredu_nn::Error::backend)?;
        unit_specs.insert(
            (group, 0),
            v3::localized_expert_bank_spec(args, target + depth, local_experts, local_width)?,
        );
    }
    if unit_specs.is_empty() {
        return Ok(None);
    }
    crate::ExpertRealizationPlan::balanced(global_experts, topology, unit_specs)
        .map(Some)
        .map_err(eredu_nn::Error::backend)
}

/// Derives complete V4 routed-expert ownership and rank-local bank geometry.
pub fn v4_expert_realization_plan<B>(
    architecture: &v4::Model<B>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<crate::ExpertRealizationPlan<eredu_nn::GatedProductExpertBankSpec>, eredu_nn::Error>
where
    B: eredu_nn::HyperNeuralBackend + eredu_nn::RoutedNeuralBackend,
{
    let args = architecture.args();
    let global_experts =
        usize::try_from(args.n_routed_experts).map_err(eredu_nn::Error::backend)?;
    let local_experts = i32::try_from(
        eredu_core::balanced_contiguous_range(
            global_experts,
            topology.expert_parallel_size,
            topology.expert_parallel_rank,
            false,
        )
        .map_err(eredu_nn::Error::backend)?
        .len(),
    )
    .map_err(eredu_nn::Error::backend)?;
    let local_args = architecture.shared_parallel_geometry();
    let local_width = local_args
        .as_ref()
        .map_or(args.moe_intermediate_size, |geometry| {
            geometry.args().moe_intermediate_size
        });
    let target = usize::try_from(args.num_hidden_layers).map_err(eredu_nn::Error::backend)?;
    let prediction =
        usize::try_from(args.num_nextn_predict_layers).map_err(eredu_nn::Error::backend)?;
    let target_group =
        eredu_runtime::ExecutionGroupId::new("target").map_err(eredu_nn::Error::backend)?;
    let mut unit_specs = std::collections::BTreeMap::new();
    for layer in 0..target {
        unit_specs.insert(
            (target_group.clone(), layer),
            v4::localized_expert_bank_spec(args, layer, local_experts, local_width)?,
        );
    }
    for depth in 0..prediction {
        let group = eredu_runtime::ExecutionGroupId::new(format!("mtp.{depth}"))
            .map_err(eredu_nn::Error::backend)?;
        unit_specs.insert(
            (group, 0),
            v4::localized_expert_bank_spec(args, target + depth, local_experts, local_width)?,
        );
    }
    crate::ExpertRealizationPlan::balanced(global_experts, topology, unit_specs)
        .map_err(eredu_nn::Error::backend)
}

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
mod component;
/// Strict V3/R1 and V4 configuration normalization.
pub mod config;
mod encoded_expert_residency;
/// Shared routed-plus-shared expert block.
pub mod moe;
/// Shared embedded-prediction layers and outputs.
pub mod mtp;
/// Semantic tensor/expert/hyper-stream placement plans.
pub mod parallel;
/// Thin V3/R1 architecture policy.
pub mod v3;
mod v3_declarations;
pub(crate) mod v3_specs;
/// Thin V4 architecture policy.
pub mod v4;
mod v4_declarations;
pub(crate) mod v4_specs;

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
pub use parallel::{
    v3_partition_local_geometry, v4_partition_local_geometry, PartitionExpertBankOwnership,
    V3PartitionLocalFoundation, V3PartitionLocalGeometry, V4PartitionLocalFoundation,
    V4PartitionLocalGeometry,
};

/// Derives complete V3 expert ownership from declarations and optional admitted local geometry.
pub fn v3_expert_realization_plan(
    args: &V3Args,
    geometry: Option<&parallel::V3LocalGeometry>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<Option<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>, eredu_nn::Error>
{
    args.validate().map_err(eredu_nn::Error::backend_source)?;
    if let Some(geometry) = geometry {
        geometry
            .validate_for(args)
            .map_err(eredu_nn::Error::backend_source)?;
    }
    let banks = v3::expert_bank_specs(args)?;
    if banks.is_empty() {
        return Ok(None);
    }
    let (_, layout) = parallel::execution_layout(
        usize::try_from(args.num_hidden_layers).map_err(eredu_nn::Error::backend)?,
        usize::try_from(args.num_nextn_predict_layers).map_err(eredu_nn::Error::backend)?,
    )
    .map_err(eredu_nn::Error::backend_source)?;
    declared_expert_realization(
        banks,
        &layout,
        geometry.map(|geometry| geometry.args().moe_intermediate_size),
        topology,
    )
    .map(Some)
}

/// Derives complete V4 expert ownership from declarations and optional admitted local geometry.
pub fn v4_expert_realization_plan(
    args: &V4Args,
    geometry: Option<&parallel::V4LocalGeometry>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>, eredu_nn::Error> {
    args.validate().map_err(eredu_nn::Error::backend_source)?;
    if let Some(geometry) = geometry {
        geometry
            .validate_for(args)
            .map_err(eredu_nn::Error::backend_source)?;
    }
    let (_, layout) = parallel::execution_layout(
        usize::try_from(args.num_hidden_layers).map_err(eredu_nn::Error::backend)?,
        usize::try_from(args.num_nextn_predict_layers).map_err(eredu_nn::Error::backend)?,
    )
    .map_err(eredu_nn::Error::backend_source)?;
    declared_expert_realization(
        v4::expert_bank_specs(args)?,
        &layout,
        geometry.map(|geometry| geometry.args().moe_intermediate_size),
        topology,
    )
}

/// Maps a storage-layer identity to its independent execution invocation.
fn expert_unit_owner(
    layout: &eredu_runtime::ExecutionUnitLayout,
    layer: usize,
) -> Result<(eredu_runtime::ExecutionGroupId, usize), eredu_nn::Error> {
    let address = layout
        .address(layer)
        .ok_or_else(|| eredu_nn::Error::backend("unresolved DeepSeek expert invocation"))?;
    let owner = layout
        .group_id(address.group())
        .ok_or_else(|| eredu_nn::Error::backend("unresolved DeepSeek expert owner"))?;
    Ok((owner.clone(), address.index()))
}

/// Localizes construction-owned banks while retaining identities, formats and invocation owners.
fn declared_expert_realization(
    banks: std::collections::BTreeMap<usize, eredu_nn::GroupedGatedProductSpec>,
    layout: &eredu_runtime::ExecutionUnitLayout,
    intermediate: Option<i32>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>, eredu_nn::Error> {
    let global_experts = banks
        .values()
        .next()
        .ok_or_else(|| eredu_nn::Error::backend("no declared routed expert banks"))?
        .group_count();
    if banks
        .values()
        .any(|bank| bank.group_count() != global_experts)
    {
        return Err(eredu_nn::Error::backend(
            "routed expert banks have incompatible group counts",
        ));
    }
    let global_experts = usize::try_from(global_experts).map_err(eredu_nn::Error::backend)?;
    let local_experts = i32::try_from(
        eredu_core::balanced_contiguous_range(
            global_experts,
            topology.expert_parallel_size(),
            topology.expert_parallel_rank(),
            false,
        )
        .map_err(eredu_nn::Error::backend)?
        .len(),
    )
    .map_err(eredu_nn::Error::backend)?;
    let specs = banks
        .into_iter()
        .map(|(layer, bank)| {
            let owner = expert_unit_owner(layout, layer)?;
            let width = intermediate.unwrap_or(bank.intermediate_dimensions());
            Ok((owner, bank.with_group_geometry(local_experts, width)?))
        })
        .collect::<Result<_, eredu_nn::Error>>()?;
    crate::ExpertRealizationPlan::balanced(global_experts, topology, specs)
        .map_err(eredu_nn::Error::backend)
}

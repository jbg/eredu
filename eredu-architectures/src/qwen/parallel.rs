//! Semantic tensor-parallel placement for Qwen decoder blocks.

use eredu_nn::{GroupedNeuralBackend, NeuralBackend};
use eredu_runtime::{
    aligned_partition_units, module_parameter_group, partitioned_module_parameter_group,
    MemberSharding, ParallelPlanError, ParameterGroupSpec, ParameterRole,
};

use crate::decoder::{block_common_parallel_parameter_groups, dense_mlp_parallel_parameter_group};

use super::{FeedForward, ModelArgs, RoutedTransformerBlock, TransformerBlock};

/// Derives rank-local Qwen construction geometry from a semantic placement plan.
pub fn local_block_args(
    args: &ModelArgs,
    layer: usize,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<ModelArgs, ParallelPlanError> {
    let prefix = format!("{}.layers.{layer}", args.parameter_root);
    let tensor = |suffix: &str| {
        layout
            .tensor(&format!("{prefix}.{suffix}.weight"))
            .or_else(|| layout.tensor(&format!("{prefix}.{suffix}")))
            .ok_or_else(|| {
                ParallelPlanError::InvalidTensor(format!(
                    "missing local layout for {prefix}.{suffix}"
                ))
            })
    };
    let query = tensor("self_attn.q_proj")?;
    let key = tensor("self_attn.k_proj")?;
    let query_width = i32::try_from(*query.local_shape().first().ok_or_else(|| {
        ParallelPlanError::InvalidTensor("local Qwen query projection is scalar".into())
    })?)
    .map_err(|_| ParallelPlanError::InvalidTensor("local Qwen query width exceeds i32".into()))?;
    let key_width = i32::try_from(*key.local_shape().first().ok_or_else(|| {
        ParallelPlanError::InvalidTensor("local Qwen key projection is scalar".into())
    })?)
    .map_err(|_| ParallelPlanError::InvalidTensor("local Qwen key width exceeds i32".into()))?;
    if query_width <= 0
        || key_width <= 0
        || query_width % args.head_dim != 0
        || key_width % args.head_dim != 0
    {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "local Qwen attention widths q={query_width}, k={key_width} split head dimension {}",
            args.head_dim
        )));
    }
    let mut local = args.clone();
    local.num_attention_heads = query_width / args.head_dim;
    local.num_key_value_heads = key_width / args.head_dim;
    if args.is_moe() {
        let gate_up = tensor("mlp.experts.gate_up_proj")?;
        let fused_width = *gate_up.local_shape().get(1).ok_or_else(|| {
            ParallelPlanError::InvalidTensor(
                "local Qwen expert gate/up tensor has rank below 2".into(),
            )
        })?;
        if !fused_width.is_multiple_of(2) {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "local Qwen expert gate/up width {fused_width} is not even"
            )));
        }
        local.moe_intermediate_size = i32::try_from(fused_width / 2).map_err(|_| {
            ParallelPlanError::InvalidTensor("local Qwen expert width exceeds i32".into())
        })?;
        if local.moe_intermediate_size <= 0 {
            return Err(ParallelPlanError::InvalidTensor(
                "local Qwen expert width must be positive".into(),
            ));
        }
    } else {
        let gate = tensor("mlp.gate_proj")?;
        local.intermediate_size = i32::try_from(*gate.local_shape().first().ok_or_else(|| {
            ParallelPlanError::InvalidTensor("local Qwen gate projection is scalar".into())
        })?)
        .map_err(|_| ParallelPlanError::InvalidTensor("local Qwen MLP width exceeds i32".into()))?;
        if local.intermediate_size <= 0 {
            return Err(ParallelPlanError::InvalidTensor(
                "local Qwen MLP width must be positive".into(),
            ));
        }
    }
    Ok(local)
}

/// Returns the rank-local key/value head count for every Qwen decoder layer.
pub fn local_key_value_heads(
    args: &ModelArgs,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<Vec<i32>, ParallelPlanError> {
    (0..args.num_hidden_layers as usize)
        .map(|layer| Ok(local_block_args(args, layer, layout)?.num_key_value_heads))
        .collect()
}

/// Complete rank-local geometry for the canonical neutral Qwen text model.
pub type LocalGeometry = crate::decoder::LocalGeometry<ModelArgs>;

/// Derives Qwen unit, vocabulary, and state geometry from one typed plan.
pub fn local_geometry(
    args: &ModelArgs,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<LocalGeometry, ParallelPlanError> {
    crate::decoder::local_geometry(args, layout, local_block_args)
}

/// Derives exact PP×TP×EP-local geometry from the selected routed realization.
pub fn partition_local_routed_geometry(
    args: &ModelArgs,
    layout: &eredu_runtime::LocalModelLayout,
    owned_units: std::ops::Range<usize>,
    topology: eredu_core::ParallelRankTopology,
    realization: &crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
) -> Result<crate::decoder::PartitionLocalGeometry<ModelArgs>, ParallelPlanError> {
    if !args.is_moe() {
        return Err(ParallelPlanError::InvalidGroup(
            "partition-local routed Qwen geometry requires Qwen3-MoE".into(),
        ));
    }
    validate_routed_realization(args, topology, realization)?;
    crate::decoder::partition_local_geometry_with(
        args,
        layout,
        owned_units,
        |args, layer, layout| {
            let mut local = local_block_args(args, layer, layout)
                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
            let spec = realization
                .unit_spec(crate::decoder::TEXT_DECODER_EXECUTION_GROUP, layer)
                .ok_or_else(|| {
                    eredu_nn::Error::backend(format!(
                        "selected routed Qwen realization omits unit {layer}"
                    ))
                })?;
            let local_experts = i32::try_from(realization.local_global_group_indices().len())
                .map_err(eredu_nn::Error::backend)?;
            if spec.group_count() != local_experts
                || spec.intermediate_dimensions() != local.moe_intermediate_size
            {
                return Err(eredu_nn::Error::backend(format!(
                    "selected routed Qwen unit {layer} bank geometry drifted"
                )));
            }
            local.num_experts = local_experts;
            Ok(local)
        },
    )
}

fn validate_routed_realization(
    args: &ModelArgs,
    topology: eredu_core::ParallelRankTopology,
    realization: &crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
) -> Result<(), ParallelPlanError> {
    let global_experts = usize::try_from(args.num_experts)
        .map_err(|_| ParallelPlanError::InvalidGroup("Qwen expert count exceeds usize".into()))?;
    let expected_local = eredu_core::balanced_contiguous_range(
        global_experts,
        topology.expert_parallel_size(),
        topology.expert_parallel_rank(),
        false,
    )
    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?
    .collect::<Vec<_>>();
    if realization.global_expert_count() != global_experts
        || realization.expert_parallel_size() != topology.expert_parallel_size()
        || realization.expert_parallel_rank() != topology.expert_parallel_rank()
        || realization.local_global_group_indices() != expected_local
    {
        return Err(ParallelPlanError::InvalidGroup(
            "selected routed Qwen expert ownership differs from the Cartesian rank".into(),
        ));
    }
    let layers = usize::try_from(args.num_hidden_layers)
        .map_err(|_| ParallelPlanError::InvalidGroup("Qwen layer count exceeds usize".into()))?;
    if realization.unit_specs().len() != layers
        || (0..layers).any(|layer| {
            realization
                .unit_spec(crate::decoder::TEXT_DECODER_EXECUTION_GROUP, layer)
                .is_none()
        })
    {
        return Err(ParallelPlanError::InvalidGroup(
            "selected routed Qwen realization unit schedule drifted".into(),
        ));
    }
    Ok(())
}

/// Declares every rank-local placement group for one dense Qwen block.
pub fn layer_parallel_parameter_groups<B: NeuralBackend>(
    block: &TransformerBlock<B>,
    args: &ModelArgs,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let mut groups = block_common_parallel_parameter_groups(block, args, layer)?;
    groups.push(dense_mlp_parallel_parameter_group(&block.mlp, args, layer)?);
    Ok(groups)
}

/// Declares every rank-local placement group for a dynamically dense-or-routed Qwen block.
pub fn routed_layer_parallel_parameter_groups<
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
>(
    block: &RoutedTransformerBlock<B>,
    args: &ModelArgs,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let mut groups = block_common_parallel_parameter_groups(block, args, layer)?;
    match &block.mlp {
        FeedForward::Dense(mlp) => {
            groups.push(dense_mlp_parallel_parameter_group(mlp, args, layer)?);
        }
        FeedForward::Routed(moe) => {
            let prefix = format!("{}.layers.{layer}.mlp", args.parameter_root);
            groups.push(module_parameter_group::<B::Tensor, _>(
                format!("{prefix}.gate"),
                ParameterRole::Replicated,
                &moe.router,
                |_, _| Ok(MemberSharding::Replicated),
            )?);
            let intermediate = usize::try_from(args.moe_intermediate_size).map_err(|_| {
                ParallelPlanError::InvalidGroup("Qwen expert width exceeds usize".into())
            })?;
            let alignment = args
                .weight_quantization_for(&format!("{prefix}.experts.down_proj"))
                .map_or(Ok(1), |quantization| {
                    usize::try_from(quantization.group_size()).map_err(|_| {
                        ParallelPlanError::InvalidGroup(
                            "Qwen expert quantization group exceeds usize".into(),
                        )
                    })
                })?;
            let units =
                aligned_partition_units(&format!("{prefix}.experts"), intermediate, 1, alignment)?;
            let segments = vec![0..intermediate, intermediate..2 * intermediate];
            groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
                format!("{prefix}.experts.intermediate"),
                ParameterRole::ExpertIntermediate,
                units,
                &moe.experts,
                |metadata, shape| {
                    let name = metadata.id.as_str();
                    if name.contains("gate_up_proj") {
                        if shape.len() < 3 {
                            return Err(ParallelPlanError::InvalidTensor(format!(
                                "packed Qwen gate/up expert parameter {name} has rank {}",
                                shape.len()
                            )));
                        }
                        Ok(MemberSharding::PartitionedSegments {
                            axis: 1,
                            segments: segments.clone(),
                        })
                    } else if name.contains("down_proj") {
                        if shape.len() < 3 {
                            return Err(ParallelPlanError::InvalidTensor(format!(
                                "packed Qwen down expert parameter {name} has rank {}",
                                shape.len()
                            )));
                        }
                        Ok(MemberSharding::Partitioned { axis: 2 })
                    } else {
                        Err(ParallelPlanError::InvalidTensor(format!(
                            "unexpected Qwen expert-bank parameter {name}"
                        )))
                    }
                },
            )?);
        }
    }
    Ok(groups)
}

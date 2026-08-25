//! Semantic tensor-parallel placement for Qwen decoder blocks.

use eredu_nn::{RoutedNeuralBackend, Tensor};
use eredu_runtime::{
    aligned_partition_units, module_parameter_group, partitioned_module_parameter_group,
    ArchitectureParameterDescription, ExecutionGraph, ExecutionUnitLayout, MemberSharding,
    OwnedParameterGroupSpec, ParallelPlanError, ParameterGroupOwner, ParameterGroupSpec,
    ParameterRole,
};

use crate::decoder::{block_common_parallel_parameter_groups, dense_mlp_parallel_parameter_group};

use super::{FeedForward, ModelArgs, TransformerBlock};

/// Describes Qwen pinned parameters and every decoder unit with explicit
/// architecture-global ownership.
pub fn parameter_description<B: RoutedNeuralBackend>(
    model: &super::LayeredModel<B>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    let graph = ExecutionGraph::chain(["text_decoder"])
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let count = usize::try_from(model.args().num_hidden_layers)
        .map_err(|_| ParallelPlanError::InvalidGroup("Qwen layer count exceeds usize".into()))?;
    let layout = ExecutionUnitLayout::new(&graph, [count])
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let static_groups = crate::decoder::static_parallel_parameter_groups::<B>(
        &model.static_modules().embeddings,
        &model.static_modules().norm,
        model.static_modules().lm_head.as_ref(),
        &model.args().parameter_root,
    )?;
    let mut expected = static_groups.clone();
    let mut owned = static_groups
        .into_iter()
        .enumerate()
        .map(|(index, group)| {
            OwnedParameterGroupSpec::new(
                if index == 0 && model.args().tie_word_embeddings {
                    ParameterGroupOwner::static_any_of(["embedding", "output"])
                } else {
                    ParameterGroupOwner::static_role(match index {
                        0 => "embedding",
                        1 => "norm",
                        _ => "output",
                    })
                },
                group,
            )
        })
        .collect::<Vec<_>>();
    let owner_group = layout.group_id(0).expect("Qwen decoder group").clone();
    for index in 0..count {
        let unit = model
            .construct_unit(index, context)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        let groups = layer_parallel_parameter_groups(&unit, model.args(), index)?;
        expected.extend(groups.iter().cloned());
        owned.extend(groups.into_iter().map(|group| {
            OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit(owner_group.clone(), index),
                group,
            )
        }));
    }
    ArchitectureParameterDescription::new(&graph, &layout, expected, owned)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))
}

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

/// Declares every rank-local placement group for one Qwen dense-or-routed block.
pub fn layer_parallel_parameter_groups<B: RoutedNeuralBackend>(
    block: &TransformerBlock<B>,
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

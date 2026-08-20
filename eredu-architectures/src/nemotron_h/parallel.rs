//! Semantic placement and rank-local geometry for Nemotron-H physical units.

use eredu_nn::RoutedNeuralBackend;
use eredu_runtime::{
    aligned_partition_units, module_parameter_group, partitioned_module_parameter_group,
    partitioned_projection_group, MemberSharding, ParallelPlanError, ParameterGroupSpec,
    ParameterRole, ProjectionSharding,
};

use super::{
    Block, DenseMlp, LayerGeometry, LayerPolicy, ModelArgs, Operator, PredictionUnit, Unit,
};
use crate::decoder::StaticModules;

fn local_width(
    layout: &eredu_runtime::LocalModelLayout,
    name: &str,
    axis: usize,
) -> Result<i32, ParallelPlanError> {
    let tensor = layout.tensor(name).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("missing local Nemotron-H layout for {name}"))
    })?;
    i32::try_from(*tensor.local_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("Nemotron-H tensor {name} has no axis {axis}"))
    })?)
    .map_err(|_| {
        ParallelPlanError::InvalidTensor(format!("Nemotron-H width for {name} exceeds i32"))
    })
}

/// Derives one target block's local geometry from the resolved parameter layout.
pub fn local_block_geometry(
    args: &ModelArgs,
    layer: usize,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<LayerGeometry, ParallelPlanError> {
    let root = format!("model.layers.{layer}");
    match args.layer_schedule.get(layer).copied().ok_or_else(|| {
        ParallelPlanError::InvalidGroup(format!("Nemotron-H has no layer {layer}"))
    })? {
        LayerPolicy::Mamba => {
            let heads = local_width(layout, &format!("{root}.mamba.dt_bias"), 0)?;
            let heads_per_group = args.mamba_num_heads / args.n_groups;
            if heads % heads_per_group != 0 {
                return Err(ParallelPlanError::InvalidTensor(
                    "local Mamba heads do not contain complete state groups".into(),
                ));
            }
            Ok(LayerGeometry::Mamba {
                heads,
                groups: heads / heads_per_group,
            })
        }
        LayerPolicy::SelfAttention(_) => {
            let query = local_width(layout, &format!("{root}.attention.q_proj.weight"), 0)?;
            let key = local_width(layout, &format!("{root}.attention.k_proj.weight"), 0)?;
            if query % args.head_dim != 0 || key % args.head_dim != 0 {
                return Err(ParallelPlanError::InvalidTensor(
                    "local attention widths do not contain complete heads".into(),
                ));
            }
            Ok(LayerGeometry::Attention {
                query_heads: query / args.head_dim,
                kv_heads: key / args.head_dim,
            })
        }
        LayerPolicy::DenseMlp => Ok(LayerGeometry::DenseMlp {
            intermediate: local_width(layout, &format!("{root}.mlp.up_proj.weight"), 0)?,
        }),
        LayerPolicy::SparseMoe => Ok(LayerGeometry::SparseMoe {
            routed: local_width(layout, &format!("{root}.moe.experts.up_proj"), 1)?,
            shared: local_width(
                layout,
                &format!("{root}.moe.shared_experts.up_proj.weight"),
                0,
            )?,
        }),
    }
}

/// Returns resolved state geometry for target and appended MTP units.
pub fn local_state_geometry(
    args: &ModelArgs,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<Vec<LayerGeometry>, ParallelPlanError> {
    let mut geometry = (0..args.num_hidden_layers as usize)
        .map(|layer| local_block_geometry(args, layer, layout))
        .collect::<Result<Vec<_>, _>>()?;
    let target = args.num_hidden_layers as usize;
    for (physical, policy) in args
        .mtp_policies()
        .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))?
        .into_iter()
        .enumerate()
    {
        let root = format!("model.mtp.layers.{physical}.mixer");
        geometry.push(match policy {
            LayerPolicy::SelfAttention(_) => LayerGeometry::Attention {
                query_heads: local_width(layout, &format!("{root}.q_proj.weight"), 0)?
                    / args.head_dim,
                kv_heads: local_width(layout, &format!("{root}.k_proj.weight"), 0)? / args.head_dim,
            },
            LayerPolicy::SparseMoe => LayerGeometry::SparseMoe {
                routed: local_width(layout, &format!("{root}.experts.up_proj"), 1)?,
                shared: local_width(layout, &format!("{root}.shared_experts.up_proj.weight"), 0)?,
            },
            _ => {
                return Err(ParallelPlanError::InvalidGroup(format!(
                    "unsupported MTP policy at global state layer {}",
                    target + physical
                )))
            }
        });
    }
    Ok(geometry)
}

/// Declares vocabulary and replicated final-normalization groups.
pub fn static_parallel_parameter_groups<B: RoutedNeuralBackend>(
    modules: &StaticModules<B>,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let mut groups = vec![
        module_parameter_group::<B::Tensor, _>(
            "model.embeddings",
            ParameterRole::Vocabulary,
            &modules.embeddings,
            |_, shape| {
                if shape.is_empty() {
                    Err(ParallelPlanError::InvalidTensor(
                        "Nemotron-H embedding is scalar".into(),
                    ))
                } else {
                    Ok(MemberSharding::Balanced { axis: 0 })
                }
            },
        )?,
        module_parameter_group::<B::Tensor, _>(
            "model.norm_f",
            ParameterRole::Replicated,
            &modules.norm,
            |_, _| Ok(MemberSharding::Replicated),
        )?,
    ];
    if let Some(head) = &modules.lm_head {
        groups.push(module_parameter_group::<B::Tensor, _>(
            "lm_head",
            ParameterRole::Vocabulary,
            head,
            |_, shape| {
                if shape.is_empty() {
                    Err(ParallelPlanError::InvalidTensor(
                        "Nemotron-H output is scalar".into(),
                    ))
                } else {
                    Ok(MemberSharding::Balanced { axis: 0 })
                }
            },
        )?);
    }
    Ok(groups)
}

fn dense_groups<B: RoutedNeuralBackend>(
    root: &str,
    mlp: &DenseMlp<B>,
    width: i32,
    role: ParameterRole,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    Ok(vec![partitioned_projection_group::<B::Tensor, B::Linear>(
        format!("{root}.intermediate"),
        role,
        &[
            (&mlp.up_proj, ProjectionSharding::Column),
            (&mlp.down_proj, ProjectionSharding::Row),
        ],
        aligned_partition_units(
            root,
            usize::try_from(width).map_err(|_| {
                ParallelPlanError::InvalidGroup(
                    "Nemotron-H intermediate width exceeds usize".into(),
                )
            })?,
            1,
            1,
        )?,
    )?])
}

/// Declares semantic groups for one target physical block.
pub fn layer_parallel_parameter_groups<B: RoutedNeuralBackend>(
    block: &Block<B>,
    args: &ModelArgs,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let root = format!("model.layers.{layer}");
    block_parallel_parameter_groups(block, args, &root)
}

fn block_parallel_parameter_groups<B: RoutedNeuralBackend>(
    block: &Block<B>,
    args: &ModelArgs,
    root: &str,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let mut groups = vec![module_parameter_group::<B::Tensor, _>(
        format!("{root}.norm"),
        ParameterRole::Replicated,
        &block.norm,
        |_, _| Ok(MemberSharding::Replicated),
    )?];
    match &block.operator {
        Operator::Mamba(mamba) => {
            let heads = usize::try_from(args.mamba_num_heads)
                .map_err(|_| ParallelPlanError::InvalidGroup("Mamba heads exceed usize".into()))?;
            let intermediate = usize::try_from(args.mamba_num_heads * args.mamba_head_dim)
                .map_err(|_| ParallelPlanError::InvalidGroup("Mamba width exceeds usize".into()))?;
            let grouped = usize::try_from(args.n_groups * args.ssm_state_size).map_err(|_| {
                ParallelPlanError::InvalidGroup("Mamba state width exceeds usize".into())
            })?;
            let segments = vec![
                0..intermediate,
                intermediate..2 * intermediate,
                2 * intermediate..2 * intermediate + grouped,
                2 * intermediate + grouped..2 * intermediate + 2 * grouped,
                2 * intermediate + 2 * grouped..2 * intermediate + 2 * grouped + heads,
            ];
            groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
                format!("{root}.mamba.heads"),
                ParameterRole::Channels,
                usize::try_from(args.n_groups).map_err(|_| {
                    ParallelPlanError::InvalidGroup("Mamba group count exceeds usize".into())
                })?,
                mamba,
                |metadata, shape| {
                    let name = metadata.id.as_str();
                    if name.ends_with("in_proj.weight") || name.ends_with("in_proj.bias") {
                        Ok(MemberSharding::PartitionedSegments {
                            axis: 0,
                            segments: segments.clone(),
                        })
                    } else if name.ends_with("conv1d.weight")
                        || name.ends_with("conv1d.bias")
                        || name.ends_with("dt_bias")
                        || name.ends_with("A_log")
                        || name.ends_with("D")
                        || name.ends_with("norm.weight")
                    {
                        Ok(MemberSharding::Partitioned { axis: 0 })
                    } else if name.ends_with("out_proj.weight") && shape.len() >= 2 {
                        Ok(MemberSharding::Partitioned { axis: 1 })
                    } else {
                        Ok(MemberSharding::Replicated)
                    }
                },
            )?);
        }
        Operator::Attention(attention) => {
            groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
                format!("{root}.attention.heads"),
                ParameterRole::AttentionHeads,
                usize::try_from(args.num_key_value_heads).map_err(|_| {
                    ParallelPlanError::InvalidGroup("attention head count exceeds usize".into())
                })?,
                attention,
                |metadata, shape| {
                    let name = metadata.id.as_str();
                    if name.ends_with("q_proj.weight")
                        || name.ends_with("k_proj.weight")
                        || name.ends_with("v_proj.weight")
                        || name.ends_with("q_proj.bias")
                        || name.ends_with("k_proj.bias")
                        || name.ends_with("v_proj.bias")
                    {
                        Ok(MemberSharding::Partitioned { axis: 0 })
                    } else if name.ends_with("o_proj.weight") && shape.len() >= 2 {
                        Ok(MemberSharding::Partitioned { axis: 1 })
                    } else {
                        Ok(MemberSharding::Replicated)
                    }
                },
            )?)
        }
        Operator::Dense(mlp) => groups.extend(dense_groups(
            &format!("{root}.mlp"),
            mlp,
            args.intermediate_size,
            ParameterRole::FeedForwardIntermediate,
        )?),
        Operator::Sparse(moe) => {
            groups.push(module_parameter_group::<B::Tensor, _>(
                format!("{root}.moe.gate"),
                ParameterRole::Replicated,
                &moe.gate,
                |_, _| Ok(MemberSharding::Replicated),
            )?);
            groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
                format!("{root}.moe.experts.intermediate"),
                ParameterRole::ExpertIntermediate,
                usize::try_from(args.moe_intermediate_size).map_err(|_| {
                    ParallelPlanError::InvalidGroup("expert width exceeds usize".into())
                })?,
                &moe.experts,
                |metadata, _| {
                    if metadata.id.as_str().contains("up_proj") {
                        Ok(MemberSharding::Partitioned { axis: 1 })
                    } else {
                        Ok(MemberSharding::Partitioned { axis: 2 })
                    }
                },
            )?);
            groups.extend(dense_groups(
                &format!("{root}.moe.shared_experts"),
                &moe.shared_experts,
                args.moe_shared_expert_intermediate_size,
                ParameterRole::ExpertIntermediate,
            )?);
        }
    }
    Ok(groups)
}

/// Declares semantic placement for one target or appended prediction unit.
pub fn unit_parallel_parameter_groups<B: RoutedNeuralBackend>(
    unit: &Unit<B>,
    args: &ModelArgs,
    flat: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let target = usize::try_from(args.num_hidden_layers)
        .map_err(|_| ParallelPlanError::InvalidGroup("layer count exceeds usize".into()))?;
    match unit {
        Unit::Target(block) if flat < target => layer_parallel_parameter_groups(block, args, flat),
        Unit::Prediction(prediction) if flat >= target => {
            prediction_parallel_parameter_groups(prediction, args, flat - target)
        }
        _ => Err(ParallelPlanError::InvalidGroup(format!(
            "Nemotron-H unit kind does not match flat position {flat}"
        ))),
    }
}

fn prediction_parallel_parameter_groups<B: RoutedNeuralBackend>(
    unit: &PredictionUnit<B>,
    args: &ModelArgs,
    physical: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let root = format!("model.mtp.layers.{physical}");
    let mut groups = Vec::new();
    if let Some(norm) = &unit.embedding_norm {
        groups.push(module_parameter_group::<B::Tensor, _>(
            format!("{root}.enorm"),
            ParameterRole::Replicated,
            norm,
            |_, _| Ok(MemberSharding::Replicated),
        )?);
    }
    if let Some(norm) = &unit.hidden_norm {
        groups.push(module_parameter_group::<B::Tensor, _>(
            format!("{root}.hnorm"),
            ParameterRole::Replicated,
            norm,
            |_, _| Ok(MemberSharding::Replicated),
        )?);
    }
    if let Some(fusion) = &unit.fusion {
        groups.push(module_parameter_group::<B::Tensor, _>(
            format!("{root}.eh_proj"),
            ParameterRole::Replicated,
            fusion,
            |_, _| Ok(MemberSharding::Replicated),
        )?);
    }
    groups.extend(block_parallel_parameter_groups(
        &unit.block,
        args,
        &format!("{root}.mixer"),
    )?);
    if let Some(norm) = &unit.final_norm {
        groups.push(module_parameter_group::<B::Tensor, _>(
            format!("{root}.final_layernorm"),
            ParameterRole::Replicated,
            norm,
            |_, _| Ok(MemberSharding::Replicated),
        )?);
    }
    Ok(groups)
}

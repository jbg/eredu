//! Semantic tensor-parallel placement for shared Qwen hybrid units.

use eredu_nn::RoutedNeuralBackend;
use eredu_runtime::{
    aligned_partition_units, module_parameter_group, partitioned_module_parameter_group,
    MemberSharding, ParallelPlanError, ParameterGroupSpec, ParameterRole,
};

use super::{
    Block, FeedForward, HybridConfig, HybridLayerPolicy, PredictionUnit, TokenMixer, Unit,
};

fn local_width(
    layout: &eredu_runtime::LocalModelLayout,
    name: &str,
    axis: usize,
) -> Result<i32, ParallelPlanError> {
    let tensor = layout
        .tensor(name)
        .or_else(|| layout.tensor(&format!("{name}.inner.weight")))
        .ok_or_else(|| {
            ParallelPlanError::InvalidTensor(format!("missing local Qwen hybrid layout for {name}"))
        })?;
    i32::try_from(*tensor.local_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("Qwen hybrid tensor {name} has no axis {axis}"))
    })?)
    .map_err(|_| {
        ParallelPlanError::InvalidTensor(format!("Qwen hybrid width for {name} exceeds i32"))
    })
}

/// Derives rank-local recurrent/full-attention and feed-forward widths.
pub fn local_block_config(
    config: &HybridConfig,
    layer: usize,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<HybridConfig, ParallelPlanError> {
    let root = format!("model.layers.{layer}");
    let policy = config.layer_schedule.get(layer).copied().ok_or_else(|| {
        ParallelPlanError::InvalidGroup(format!("Qwen hybrid has no layer {layer}"))
    })?;
    local_config_at(config, &root, policy, layout)
}

/// Derives rank-local construction geometry for one target or prediction unit.
pub fn local_unit_config(
    config: &HybridConfig,
    group: usize,
    index: usize,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<HybridConfig, ParallelPlanError> {
    if group == 0 {
        return local_block_config(config, index, layout);
    }
    if index != 0 || group > config.mtp_num_hidden_layers as usize {
        return Err(ParallelPlanError::InvalidGroup(format!(
            "Qwen hybrid has no unit ({group}, {index})"
        )));
    }
    local_config_at(
        config,
        &format!("mtp.layers.{}", group - 1),
        HybridLayerPolicy::SelfAttention(eredu_core::attention::AttentionPolicy::Full),
        layout,
    )
}

fn local_config_at(
    config: &HybridConfig,
    root: &str,
    policy: HybridLayerPolicy,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<HybridConfig, ParallelPlanError> {
    let mut local = config.clone();
    match policy {
        HybridLayerPolicy::LinearAttention => {
            let key = local_width(layout, &format!("{root}.linear_attn.in_proj_qkv.weight"), 0)?;
            let value = local_width(layout, &format!("{root}.linear_attn.in_proj_z.weight"), 0)?;
            let global_key = config.linear_num_key_heads * config.linear_key_head_dim * 2;
            let global_value = config.linear_num_value_heads * config.linear_value_head_dim;
            let global_total = global_key + global_value;
            if key * global_key % global_total != 0
                || key * global_value % global_total != 0
                || value % config.linear_value_head_dim != 0
            {
                return Err(ParallelPlanError::InvalidTensor(
                    "rank-local recurrent projection splits head geometry".into(),
                ));
            }
            local.linear_num_key_heads =
                (key * global_key / global_total) / (2 * config.linear_key_head_dim);
            local.linear_num_value_heads = value / config.linear_value_head_dim;
        }
        HybridLayerPolicy::SelfAttention(_) => {
            let query = local_width(layout, &format!("{root}.self_attn.q_proj.weight"), 0)?;
            let key = local_width(layout, &format!("{root}.self_attn.k_proj.weight"), 0)?;
            if query % (2 * config.head_dim) != 0 || key % config.head_dim != 0 {
                return Err(ParallelPlanError::InvalidTensor(
                    "rank-local attention projection splits heads".into(),
                ));
            }
            local.num_attention_heads = query / (2 * config.head_dim);
            local.num_key_value_heads = key / config.head_dim;
        }
    }
    if config.is_moe() {
        let fused = local_width(layout, &format!("{root}.mlp.experts.gate_up_proj"), 1)?;
        if fused % 2 != 0 {
            return Err(ParallelPlanError::InvalidTensor(
                "rank-local expert fused width is odd".into(),
            ));
        }
        local.moe_intermediate_size = fused / 2;
        local.shared_expert_intermediate_size = local_width(
            layout,
            &format!("{root}.mlp.shared_expert.gate_proj.weight"),
            0,
        )?;
    } else {
        local.intermediate_size = local_width(layout, &format!("{root}.mlp.gate_proj.weight"), 0)?;
    }
    Ok(local)
}

fn norm_groups<B: RoutedNeuralBackend>(
    block: &Block<B>,
    root: &str,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    Ok(vec![
        module_parameter_group::<B::Tensor, _>(
            format!("{root}.input_norm"),
            ParameterRole::Replicated,
            &block.input_norm,
            |_, _| Ok(MemberSharding::Replicated),
        )?,
        module_parameter_group::<B::Tensor, _>(
            format!("{root}.post_attention_norm"),
            ParameterRole::Replicated,
            &block.post_attention_norm,
            |_, _| Ok(MemberSharding::Replicated),
        )?,
    ])
}

fn block_groups<B: RoutedNeuralBackend>(
    block: &Block<B>,
    config: &HybridConfig,
    root: &str,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let mut groups = norm_groups(block, root)?;
    match &block.mixer {
        TokenMixer::Linear(linear) => {
            let key = usize::try_from(config.linear_num_key_heads).map_err(|_| {
                ParallelPlanError::InvalidGroup("recurrent heads exceed usize".into())
            })?;
            let key_width =
                usize::try_from(config.linear_num_key_heads * config.linear_key_head_dim).map_err(
                    |_| ParallelPlanError::InvalidGroup("recurrent key width exceeds usize".into()),
                )?;
            let value_width =
                usize::try_from(config.linear_num_value_heads * config.linear_value_head_dim)
                    .map_err(|_| {
                        ParallelPlanError::InvalidGroup(
                            "recurrent value width exceeds usize".into(),
                        )
                    })?;
            let segments = vec![
                0..key_width,
                key_width..2 * key_width,
                2 * key_width..2 * key_width + value_width,
            ];
            groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
                format!("{root}.linear_attn.heads"),
                ParameterRole::Channels,
                key,
                linear,
                |metadata, shape| {
                    let name = metadata.id.as_str();
                    if name.contains("in_proj_qkv") {
                        Ok(MemberSharding::PartitionedSegments {
                            axis: 0,
                            segments: segments.clone(),
                        })
                    } else if name.contains("in_proj_z")
                        || name.contains("in_proj_b")
                        || name.contains("in_proj_a")
                        || name.ends_with("conv1d.weight")
                        || name.ends_with("dt_bias")
                        || name.ends_with("A_log")
                    {
                        Ok(MemberSharding::Partitioned { axis: 0 })
                    } else if name.contains("out_proj") && shape.len() >= 2 {
                        Ok(MemberSharding::Partitioned { axis: 1 })
                    } else {
                        Ok(MemberSharding::Replicated)
                    }
                },
            )?);
        }
        TokenMixer::Attention(attention) => {
            groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
                format!("{root}.attention.heads"),
                ParameterRole::AttentionHeads,
                usize::try_from(config.num_key_value_heads).map_err(|_| {
                    ParallelPlanError::InvalidGroup("attention heads exceed usize".into())
                })?,
                attention,
                |metadata, shape| {
                    let name = metadata.id.as_str();
                    if name.contains("q_proj") || name.contains("k_proj") || name.contains("v_proj")
                    {
                        Ok(MemberSharding::Partitioned { axis: 0 })
                    } else if name.contains("o_proj") && shape.len() >= 2 {
                        Ok(MemberSharding::Partitioned { axis: 1 })
                    } else {
                        Ok(MemberSharding::Replicated)
                    }
                },
            )?);
        }
    }
    match &block.feed_forward {
        FeedForward::Dense(mlp) => groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
            format!("{root}.mlp.intermediate"),
            ParameterRole::FeedForwardIntermediate,
            aligned_partition_units(
                root,
                usize::try_from(config.intermediate_size).map_err(|_| {
                    ParallelPlanError::InvalidGroup("dense width exceeds usize".into())
                })?,
                1,
                1,
            )?,
            mlp,
            |metadata, shape| {
                let name = metadata.id.as_str();
                if name.contains("gate_proj") || name.contains("up_proj") {
                    Ok(MemberSharding::Partitioned { axis: 0 })
                } else if name.contains("down_proj") && shape.len() >= 2 {
                    Ok(MemberSharding::Partitioned { axis: 1 })
                } else {
                    Ok(MemberSharding::Replicated)
                }
            },
        )?),
        FeedForward::Routed(moe) => {
            groups.push(module_parameter_group::<B::Tensor, _>(
                format!("{root}.mlp.router"),
                ParameterRole::Replicated,
                &moe.router,
                |_, _| Ok(MemberSharding::Replicated),
            )?);
            let intermediate = usize::try_from(config.moe_intermediate_size).map_err(|_| {
                ParallelPlanError::InvalidGroup("expert width exceeds usize".into())
            })?;
            let segments = vec![0..intermediate, intermediate..2 * intermediate];
            groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
                format!("{root}.mlp.experts.intermediate"),
                ParameterRole::ExpertIntermediate,
                aligned_partition_units(root, intermediate, 1, 1)?,
                &moe.experts,
                |metadata, _| {
                    if metadata.id.as_str().contains("gate_up") {
                        Ok(MemberSharding::PartitionedSegments {
                            axis: 1,
                            segments: segments.clone(),
                        })
                    } else {
                        Ok(MemberSharding::Partitioned { axis: 2 })
                    }
                },
            )?);
            groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
                format!("{root}.mlp.shared.intermediate"),
                ParameterRole::ExpertIntermediate,
                aligned_partition_units(
                    root,
                    usize::try_from(config.shared_expert_intermediate_size).map_err(|_| {
                        ParallelPlanError::InvalidGroup("shared expert width exceeds usize".into())
                    })?,
                    1,
                    1,
                )?,
                &moe.shared_expert,
                |metadata, shape| {
                    let name = metadata.id.as_str();
                    if name.contains("gate_proj") || name.contains("up_proj") {
                        Ok(MemberSharding::Partitioned { axis: 0 })
                    } else if name.contains("down_proj") && shape.len() >= 2 {
                        Ok(MemberSharding::Partitioned { axis: 1 })
                    } else {
                        Ok(MemberSharding::Replicated)
                    }
                },
            )?);
            groups.push(module_parameter_group::<B::Tensor, _>(
                format!("{root}.mlp.shared_gate"),
                ParameterRole::Replicated,
                &moe.shared_expert_gate,
                |_, _| Ok(MemberSharding::Replicated),
            )?);
        }
    }
    Ok(groups)
}

/// Declares semantic placement for a target or configured prediction unit.
pub fn unit_parallel_parameter_groups<B: RoutedNeuralBackend>(
    unit: &Unit<B>,
    config: &HybridConfig,
    group: usize,
    index: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    match unit {
        Unit::Target(block) if group == 0 => {
            block_groups(block, config, &format!("model.layers.{index}"))
        }
        Unit::Prediction(PredictionUnit {
            hidden_norm,
            embedding_norm,
            fusion,
            block,
            final_norm,
        }) if group > 0 && index == 0 => {
            let root = format!("mtp.layers.{}", group - 1);
            let mut groups = vec![
                module_parameter_group::<B::Tensor, _>(
                    format!("{root}.hidden_norm"),
                    ParameterRole::Replicated,
                    hidden_norm,
                    |_, _| Ok(MemberSharding::Replicated),
                )?,
                module_parameter_group::<B::Tensor, _>(
                    format!("{root}.embedding_norm"),
                    ParameterRole::Replicated,
                    embedding_norm,
                    |_, _| Ok(MemberSharding::Replicated),
                )?,
                module_parameter_group::<B::Tensor, _>(
                    format!("{root}.fusion"),
                    ParameterRole::Replicated,
                    fusion,
                    |_, _| Ok(MemberSharding::Replicated),
                )?,
                module_parameter_group::<B::Tensor, _>(
                    format!("{root}.final_norm"),
                    ParameterRole::Replicated,
                    final_norm,
                    |_, _| Ok(MemberSharding::Replicated),
                )?,
            ];
            groups.extend(block_groups(block, config, &root)?);
            Ok(groups)
        }
        _ => Err(ParallelPlanError::InvalidGroup(format!(
            "Qwen hybrid unit kind does not match ({group}, {index})"
        ))),
    }
}

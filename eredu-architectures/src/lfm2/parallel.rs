//! Semantic tensor-parallel placement for LFM2 physical blocks.

use eredu_nn::RoutedNeuralBackend;
use eredu_runtime::{
    aligned_partition_units, module_parameter_group, partitioned_module_parameter_group,
    MemberSharding, ParallelPlanError, ParameterGroupSpec, ParameterRole,
};

use crate::decoder::StaticModules;

use super::{
    Block, BlockGeometry, DenseSwiGlu, FeedForward, LayerCacheGeometry, ModelArgs, OperatorPolicy,
    TokenMixer,
};

fn local_width(
    layout: &eredu_runtime::LocalModelLayout,
    name: &str,
    axis: usize,
) -> Result<i32, ParallelPlanError> {
    let tensor = layout.tensor(name).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("missing local LFM2 layout for {name}"))
    })?;
    i32::try_from(*tensor.local_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("LFM2 tensor {name} has no axis {axis}"))
    })?)
    .map_err(|_| ParallelPlanError::InvalidTensor(format!("LFM2 width for {name} exceeds i32")))
}

/// Derives one block's local widths exclusively from the resolved placement.
pub fn local_block_geometry(
    args: &ModelArgs,
    layer: usize,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<BlockGeometry, ParallelPlanError> {
    let root = format!("model.layers.{layer}");
    let mut geometry = BlockGeometry::replicated(args);
    match args
        .layer_policy(layer)
        .ok_or_else(|| ParallelPlanError::InvalidGroup(format!("LFM2 has no layer {layer}")))?
        .operator
    {
        OperatorPolicy::SelfAttention(_) => {
            let head = args.hidden_size / args.num_attention_heads;
            let query = local_width(layout, &format!("{root}.self_attn.q_proj.weight"), 0)?;
            let key = local_width(layout, &format!("{root}.self_attn.k_proj.weight"), 0)?;
            if query % head != 0 || key % head != 0 {
                return Err(ParallelPlanError::InvalidTensor(format!(
                    "local LFM2 attention widths q={query} k={key} split head dimension {head}"
                )));
            }
            geometry.query_heads = query / head;
            geometry.key_value_heads = key / head;
        }
        OperatorPolicy::CausalConvolution => {
            geometry.convolution_channels =
                local_width(layout, &format!("{root}.conv.conv.weight"), 0)?;
        }
    }
    match args.layer_policy(layer).unwrap().feed_forward {
        super::FeedForwardPolicy::Dense => {
            geometry.dense_intermediate =
                local_width(layout, &format!("{root}.feed_forward.w1.weight"), 0)?;
        }
        super::FeedForwardPolicy::SparseMoe => {
            let fused = local_width(
                layout,
                &format!("{root}.feed_forward.experts.gate_up_proj"),
                1,
            )?;
            if fused % 2 != 0 {
                return Err(ParallelPlanError::InvalidTensor(format!(
                    "local LFM2 packed expert width {fused} is not even"
                )));
            }
            geometry.expert_intermediate = fused / 2;
        }
    }
    Ok(geometry)
}

/// Returns rank-local mutable-state geometry for every scheduled block.
pub fn local_state_geometry(
    args: &ModelArgs,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<Vec<LayerCacheGeometry>, ParallelPlanError> {
    (0..args.num_hidden_layers as usize)
        .map(|layer| {
            let geometry = local_block_geometry(args, layer, layout)?;
            Ok(match args.layer_policy(layer).unwrap().operator {
                OperatorPolicy::SelfAttention(_) => LayerCacheGeometry {
                    kv_heads: Some(geometry.key_value_heads),
                    convolution_channels: None,
                },
                OperatorPolicy::CausalConvolution => LayerCacheGeometry {
                    kv_heads: None,
                    convolution_channels: Some(geometry.convolution_channels),
                },
            })
        })
        .collect()
}

/// Declares vocabulary and replicated final-normalization groups.
pub fn static_parallel_parameter_groups<B: RoutedNeuralBackend>(
    modules: &StaticModules<B>,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let mut groups = vec![
        module_parameter_group::<B::Tensor, _>(
            "model.embed_tokens",
            ParameterRole::Vocabulary,
            &modules.embeddings,
            |_, shape| {
                if shape.is_empty() {
                    Err(ParallelPlanError::InvalidTensor(
                        "LFM2 embedding parameter is scalar".into(),
                    ))
                } else {
                    Ok(MemberSharding::Balanced { axis: 0 })
                }
            },
        )?,
        module_parameter_group::<B::Tensor, _>(
            "model.embedding_norm",
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
                        "LFM2 output parameter is scalar".into(),
                    ))
                } else {
                    Ok(MemberSharding::Balanced { axis: 0 })
                }
            },
        )?);
    }
    Ok(groups)
}

/// Declares semantic groups for one scheduled LFM2 block.
pub fn layer_parallel_parameter_groups<B: RoutedNeuralBackend>(
    block: &Block<B>,
    args: &ModelArgs,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let root = format!("model.layers.{layer}");
    let mut groups = Vec::new();
    match &block.mixer {
        TokenMixer::Attention(attention) => {
            let kv_heads = usize::try_from(args.num_key_value_heads).map_err(|_| {
                ParallelPlanError::InvalidGroup("LFM2 KV head count exceeds usize".into())
            })?;
            groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
                format!("{root}.self_attn.heads"),
                ParameterRole::AttentionHeads,
                kv_heads,
                attention,
                |metadata, shape| {
                    let name = metadata.id.as_str();
                    if name.ends_with("q_proj.weight")
                        || name.ends_with("k_proj.weight")
                        || name.ends_with("v_proj.weight")
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
        TokenMixer::ShortConvolution(convolution) => {
            let channels = usize::try_from(args.hidden_size).map_err(|_| {
                ParallelPlanError::InvalidGroup("LFM2 convolution width exceeds usize".into())
            })?;
            let segments = vec![
                0..channels,
                channels..2 * channels,
                2 * channels..3 * channels,
            ];
            groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
                format!("{root}.conv.channels"),
                ParameterRole::Channels,
                channels,
                convolution,
                |metadata, shape| {
                    let name = metadata.id.as_str();
                    if name.contains("in_proj") {
                        Ok(MemberSharding::PartitionedSegments {
                            axis: 0,
                            segments: segments.clone(),
                        })
                    } else if name.ends_with("conv.weight") || name.ends_with("conv.bias") {
                        Ok(MemberSharding::Partitioned { axis: 0 })
                    } else if name.ends_with("out_proj.weight") && shape.len() >= 2 {
                        Ok(MemberSharding::Partitioned { axis: 1 })
                    } else {
                        Ok(MemberSharding::Replicated)
                    }
                },
            )?);
        }
    }
    for (name, norm) in [
        ("operator_norm", &block.operator_norm),
        ("ffn_norm", &block.feed_forward_norm),
    ] {
        groups.push(module_parameter_group::<B::Tensor, _>(
            format!("{root}.{name}"),
            ParameterRole::Replicated,
            norm,
            |_, _| Ok(MemberSharding::Replicated),
        )?);
    }
    match &block.feed_forward {
        FeedForward::Dense(DenseSwiGlu { gate, down, up }) => {
            let width = usize::try_from(args.dense_intermediate_size).map_err(|_| {
                ParallelPlanError::InvalidGroup("LFM2 dense width exceeds usize".into())
            })?;
            let alignment = args
                .weight_quantization_for(&format!("{root}.feed_forward.w2.weight"))
                .map_or(Ok(1), |quantization| {
                    usize::try_from(quantization.group_size()).map_err(|_| {
                        ParallelPlanError::InvalidGroup(
                            "LFM2 dense quantization group exceeds usize".into(),
                        )
                    })
                })?;
            groups.push(eredu_runtime::partitioned_projection_group::<
                B::Tensor,
                B::Linear,
            >(
                format!("{root}.feed_forward.intermediate"),
                ParameterRole::FeedForwardIntermediate,
                &[
                    (gate, eredu_runtime::ProjectionSharding::Column),
                    (up, eredu_runtime::ProjectionSharding::Column),
                    (down, eredu_runtime::ProjectionSharding::Row),
                ],
                aligned_partition_units(&root, width, 1, alignment)?,
            )?);
        }
        FeedForward::Routed(moe) => {
            groups.push(module_parameter_group::<B::Tensor, _>(
                format!("{root}.feed_forward.gate"),
                ParameterRole::Replicated,
                &moe.router,
                |_, _| Ok(MemberSharding::Replicated),
            )?);
            let width = usize::try_from(args.moe_intermediate_size).map_err(|_| {
                ParallelPlanError::InvalidGroup("LFM2 expert width exceeds usize".into())
            })?;
            let segments = vec![0..width, width..2 * width];
            groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
                format!("{root}.feed_forward.experts.intermediate"),
                ParameterRole::ExpertIntermediate,
                width,
                &moe.experts,
                |metadata, _| {
                    if metadata.id.as_str().contains("gate_up_proj") {
                        Ok(MemberSharding::PartitionedSegments {
                            axis: 1,
                            segments: segments.clone(),
                        })
                    } else {
                        Ok(MemberSharding::Partitioned { axis: 2 })
                    }
                },
            )?);
        }
    }
    Ok(groups)
}

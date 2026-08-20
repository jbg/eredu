//! Semantic tensor-parallel placement for Kimi physical blocks.

use eredu_nn::{BlockwiseAttentionBackend, RoutedNeuralBackend};
use eredu_runtime::{
    aligned_partition_units, module_parameter_group, partitioned_module_parameter_group,
    MemberSharding, ParallelPlanError, ParameterGroupSpec, ParameterRole,
};

use crate::decoder::StaticModules;

use super::{
    AttentionKind, Block, BlockGeometry, DenseSwiGlu, FeedForward, FeedForwardPolicy,
    LayerCacheGeometry, ModelArgs, TokenMixer,
};

fn local_width(
    layout: &eredu_runtime::LocalModelLayout,
    name: &str,
    axis: usize,
) -> Result<i32, ParallelPlanError> {
    let tensor = layout.tensor(name).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("missing local Kimi layout for {name}"))
    })?;
    i32::try_from(*tensor.local_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("Kimi tensor {name} has no axis {axis}"))
    })?)
    .map_err(|_| ParallelPlanError::InvalidTensor(format!("Kimi width for {name} exceeds i32")))
}

/// Derives one block's local widths exclusively from resolved placement.
pub fn local_block_geometry(
    args: &ModelArgs,
    layer: usize,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<BlockGeometry, ParallelPlanError> {
    let root = format!("model.layers.{layer}");
    let policy = args
        .layer_policy(layer)
        .ok_or_else(|| ParallelPlanError::InvalidGroup(format!("Kimi has no layer {layer}")))?;
    let mut geometry = BlockGeometry::replicated(args);
    match policy.attention {
        AttentionKind::Kda => {
            let width = local_width(layout, &format!("{root}.self_attn.q_proj.weight"), 0)?;
            if width % args.kda_config.head_dim != 0 {
                return Err(ParallelPlanError::InvalidTensor(
                    "local KDA width does not contain complete heads".into(),
                ));
            }
            geometry.kda_heads = width / args.kda_config.head_dim;
        }
        AttentionKind::Mla => {
            let query_name = if args.q_lora_rank.is_some() {
                format!("{root}.self_attn.q_b_proj.weight")
            } else {
                format!("{root}.self_attn.q_proj.weight")
            };
            let width = local_width(layout, &query_name, 0)?;
            let head = args.qk_nope_head_dim + args.qk_rope_head_dim;
            if width % head != 0 {
                return Err(ParallelPlanError::InvalidTensor(
                    "local MLA width does not contain complete heads".into(),
                ));
            }
            geometry.mla_heads = width / head;
        }
    }
    match policy.feed_forward {
        FeedForwardPolicy::Dense => {
            geometry.dense_intermediate =
                local_width(layout, &format!("{root}.mlp.gate_proj.weight"), 0)?;
        }
        FeedForwardPolicy::SparseMoe => {
            let fused = local_width(layout, &format!("{root}.mlp.experts.gate_up_proj"), 1)?;
            if fused % 2 != 0 {
                return Err(ParallelPlanError::InvalidTensor(
                    "local Kimi packed expert width is not even".into(),
                ));
            }
            geometry.routed_intermediate = fused / 2;
            geometry.shared_intermediate = local_width(
                layout,
                &format!("{root}.mlp.shared_experts.gate_proj.weight"),
                0,
            )?;
        }
    }
    Ok(geometry)
}

/// Returns rank-local KDA state geometry for every physical layer.
pub fn local_state_geometry(
    args: &ModelArgs,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<Vec<LayerCacheGeometry>, ParallelPlanError> {
    (0..args.num_hidden_layers as usize)
        .map(|layer| {
            let policy = args.layer_policy(layer).expect("validated Kimi schedule");
            Ok(LayerCacheGeometry {
                kda_heads: match policy.attention {
                    AttentionKind::Kda => {
                        Some(local_block_geometry(args, layer, layout)?.kda_heads)
                    }
                    AttentionKind::Mla => None,
                },
            })
        })
        .collect()
}

/// Declares vocabulary and replicated final-normalization groups.
pub fn static_parallel_parameter_groups<B>(
    modules: &StaticModules<B>,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError>
where
    B: RoutedNeuralBackend + BlockwiseAttentionBackend,
{
    let mut groups = vec![
        module_parameter_group::<B::Tensor, _>(
            "model.embed_tokens",
            ParameterRole::Vocabulary,
            &modules.embeddings,
            |_, shape| {
                (!shape.is_empty())
                    .then_some(MemberSharding::Balanced { axis: 0 })
                    .ok_or_else(|| {
                        ParallelPlanError::InvalidTensor("Kimi embedding is scalar".into())
                    })
            },
        )?,
        module_parameter_group::<B::Tensor, _>(
            "model.norm",
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
                (!shape.is_empty())
                    .then_some(MemberSharding::Balanced { axis: 0 })
                    .ok_or_else(|| ParallelPlanError::InvalidTensor("Kimi output is scalar".into()))
            },
        )?);
    }
    Ok(groups)
}

/// Declares semantic groups for one scheduled Kimi block.
pub fn layer_parallel_parameter_groups<B>(
    block: &Block<B>,
    args: &ModelArgs,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError>
where
    B: RoutedNeuralBackend + BlockwiseAttentionBackend,
{
    let root = format!("model.layers.{layer}");
    let mut groups = Vec::new();
    let (units, role) = match &block.mixer {
        TokenMixer::Kda(_) => (args.kda_config.num_heads, ParameterRole::AttentionHeads),
        TokenMixer::Mla(_) => (args.num_attention_heads, ParameterRole::AttentionHeads),
    };
    groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
        format!("{root}.self_attn.heads"),
        role,
        usize::try_from(units)
            .map_err(|_| ParallelPlanError::InvalidGroup("Kimi head count exceeds usize".into()))?,
        &block.mixer,
        |metadata, shape| {
            let name = metadata.id.as_str();
            if name.ends_with("q_proj.weight")
                || name.ends_with("k_proj.weight")
                || name.ends_with("v_proj.weight")
                || name.ends_with("q_b_proj.weight")
                || name.ends_with("kv_b_proj.weight")
                || name.ends_with("k_b_proj.weight")
                || name.ends_with("v_b_proj.weight")
                || name.ends_with("f_b_proj.weight")
                || name.ends_with("b_proj.weight")
                || name.ends_with("g_b_proj.weight")
                || name.ends_with("dt_bias")
                || name.contains("conv1d.weight")
            {
                Ok(MemberSharding::Partitioned { axis: 0 })
            } else if name.ends_with("A_log") && shape.len() >= 3 {
                Ok(MemberSharding::Partitioned { axis: 2 })
            } else if name.ends_with("o_proj.weight") && shape.len() >= 2 {
                Ok(MemberSharding::Partitioned { axis: 1 })
            } else {
                Ok(MemberSharding::Replicated)
            }
        },
    )?);
    for (name, norm) in [
        ("input_layernorm", &block.input_norm),
        ("post_attention_layernorm", &block.post_attention_norm),
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
            let width = usize::try_from(args.intermediate_size).map_err(|_| {
                ParallelPlanError::InvalidGroup("Kimi dense width exceeds usize".into())
            })?;
            groups.push(eredu_runtime::partitioned_projection_group::<
                B::Tensor,
                B::Linear,
            >(
                format!("{root}.mlp.intermediate"),
                ParameterRole::FeedForwardIntermediate,
                &[
                    (gate, eredu_runtime::ProjectionSharding::Column),
                    (up, eredu_runtime::ProjectionSharding::Column),
                    (down, eredu_runtime::ProjectionSharding::Row),
                ],
                aligned_partition_units(&root, width, 1, 1)?,
            )?);
        }
        FeedForward::Sparse(moe) => {
            groups.push(module_parameter_group::<B::Tensor, _>(
                format!("{root}.mlp.gate"),
                ParameterRole::Replicated,
                &moe.router,
                |_, _| Ok(MemberSharding::Replicated),
            )?);
            let width = usize::try_from(args.moe_intermediate_size).map_err(|_| {
                ParallelPlanError::InvalidGroup("Kimi expert width exceeds usize".into())
            })?;
            let segments = vec![0..width, width..2 * width];
            groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
                format!("{root}.mlp.experts.intermediate"),
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
            groups.push(eredu_runtime::partitioned_projection_group::<
                B::Tensor,
                B::Linear,
            >(
                format!("{root}.mlp.shared_experts.intermediate"),
                ParameterRole::ExpertIntermediate,
                &[
                    (&moe.shared.gate, eredu_runtime::ProjectionSharding::Column),
                    (&moe.shared.up, eredu_runtime::ProjectionSharding::Column),
                    (&moe.shared.down, eredu_runtime::ProjectionSharding::Row),
                ],
                aligned_partition_units(&root, width, 1, 1)?,
            )?);
        }
    }
    Ok(groups)
}

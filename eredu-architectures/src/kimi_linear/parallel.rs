//! Semantic tensor-parallel placement for Kimi physical blocks.

use eredu_nn::{BlockwiseAttentionBackend, GroupedNeuralBackend, VocabularyParallelRange};
use eredu_runtime::{
    aligned_partition_units, module_parameter_group, partitioned_module_parameter_group,
    LocalModelLayout, MemberSharding, ParallelPlanError, ParameterGroupSpec, ParameterRole,
    StateLayout, TensorPlacement,
};

use crate::decoder::StaticModules;

use super::{
    prompt_cache_architecture_fingerprint, state_layout_with_geometry, AttentionKind, Block,
    BlockGeometry, DenseSwiGlu, FeedForward, FeedForwardPolicy, LayerCacheGeometry, ModelArgs,
    TokenMixer,
};

/// Complete planner-derived construction and state geometry for one Kimi rank.
///
/// Fields are deliberately private so unit widths, vocabulary ownership, and
/// mutable-state geometry can only be produced together from one typed local
/// layout.
#[derive(Debug, Clone)]
pub struct LocalGeometry {
    blocks: Vec<BlockGeometry>,
    embedding_range: VocabularyParallelRange,
    output_range: Option<VocabularyParallelRange>,
    state_layout: StateLayout,
    architecture_fingerprint: String,
    global_block_geometry: BlockGeometry,
    tied_head: bool,
}

impl LocalGeometry {
    /// Returns local block geometry in physical execution order.
    pub fn blocks(&self) -> &[BlockGeometry] {
        &self.blocks
    }

    /// Returns one validated rank-local block geometry.
    pub fn block(&self, layer: usize) -> Option<&BlockGeometry> {
        self.blocks.get(layer)
    }

    /// Returns input-embedding vocabulary ownership.
    pub const fn embedding_range(&self) -> &VocabularyParallelRange {
        &self.embedding_range
    }

    /// Returns untied output-head vocabulary ownership.
    pub const fn output_range(&self) -> Option<&VocabularyParallelRange> {
        self.output_range.as_ref()
    }

    /// Returns the authoritative state layout derived from these blocks.
    pub const fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    pub(super) fn validate_for(&self, args: &ModelArgs) -> Result<(), ParallelPlanError> {
        let expected_layers = usize::try_from(args.num_hidden_layers).map_err(|_| {
            ParallelPlanError::InvalidGroup("Kimi layer count exceeds usize".into())
        })?;
        if self.blocks.len() != expected_layers
            || self.architecture_fingerprint != prompt_cache_architecture_fingerprint(args)
            || self.global_block_geometry != BlockGeometry::replicated(args)
            || self.tied_head != args.tie_word_embeddings
        {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local Kimi geometry belongs to a different model configuration".into(),
            ));
        }
        self.embedding_range
            .validate_global_rows(args.vocab_size)
            .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
        match (args.tie_word_embeddings, &self.output_range) {
            (true, None) => {}
            (false, Some(range)) => range
                .validate_global_rows(args.vocab_size)
                .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?,
            (true, Some(_)) => {
                return Err(ParallelPlanError::InvalidGroup(
                    "tied Kimi output unexpectedly owns a separate vocabulary range".into(),
                ))
            }
            (false, None) => {
                return Err(ParallelPlanError::InvalidGroup(
                    "untied Kimi output is missing vocabulary ownership".into(),
                ))
            }
        }
        let expected = state_layout_with_geometry(args, &state_geometry(args, &self.blocks))
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        if expected != self.state_layout {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local Kimi state layout drifted from block geometry".into(),
            ));
        }
        Ok(())
    }
}

fn local_width(
    layout: &LocalModelLayout,
    name: &str,
    axis: usize,
) -> Result<i32, ParallelPlanError> {
    let tensor = layout.tensor(name).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("missing local Kimi layout for {name}"))
    })?;
    let local = *tensor.local_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("Kimi tensor {name} has no axis {axis}"))
    })?;
    let global = *tensor.global_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("Kimi tensor {name} has no global axis {axis}"))
    })?;
    if local == 0 || local > global {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "Kimi tensor {name} has invalid local width {local} of global width {global}"
        )));
    }
    i32::try_from(local)
        .map_err(|_| ParallelPlanError::InvalidTensor(format!("Kimi width for {name} exceeds i32")))
}

/// Derives one block's local widths exclusively from resolved placement.
pub fn local_block_geometry(
    args: &ModelArgs,
    layer: usize,
    layout: &LocalModelLayout,
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
    layout: &LocalModelLayout,
) -> Result<Vec<LayerCacheGeometry>, ParallelPlanError> {
    let blocks = (0..args.num_hidden_layers as usize)
        .map(|layer| local_block_geometry(args, layer, layout))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(state_geometry(args, &blocks))
}

fn state_geometry(args: &ModelArgs, blocks: &[BlockGeometry]) -> Vec<LayerCacheGeometry> {
    args.layer_schedule
        .iter()
        .zip(blocks)
        .map(|(policy, block)| LayerCacheGeometry {
            kda_heads: (policy.attention == AttentionKind::Kda).then_some(block.kda_heads),
        })
        .collect()
}

fn vocabulary_range(
    layout: &LocalModelLayout,
    logical_name: &str,
    global_vocabulary: usize,
) -> Result<VocabularyParallelRange, ParallelPlanError> {
    let mut selected = None;
    let mut found = false;
    for (target, tensor) in layout
        .tensors()
        .filter(|(_, tensor)| tensor.logical_name() == logical_name)
    {
        found = true;
        if tensor.global_shape().first().copied() != Some(global_vocabulary) {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "Kimi vocabulary member {target} has global shape {:?}, expected {global_vocabulary} rows",
                tensor.global_shape()
            )));
        }
        let range = match tensor.placement() {
            TensorPlacement::Range {
                axis: 0,
                start,
                end,
            } => *start..*end,
            TensorPlacement::Replicated => 0..global_vocabulary,
            placement => {
                return Err(ParallelPlanError::InvalidTensor(format!(
                    "Kimi vocabulary member {target} has non-row placement {placement:?}"
                )))
            }
        };
        if selected.as_ref().is_some_and(|current| current != &range) {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "Kimi vocabulary group {logical_name} has inconsistent companion selections"
            )));
        }
        selected = Some(range);
    }
    if !found {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "missing local Kimi vocabulary layout for {logical_name}"
        )));
    }
    let range = VocabularyParallelRange {
        global_vocabulary,
        local: selected.expect("a found Kimi vocabulary member supplies a selection"),
    };
    range
        .validate()
        .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
    Ok(range)
}

/// Derives complete rank-local Kimi construction geometry from one typed plan.
pub fn local_geometry(
    args: &ModelArgs,
    layout: &LocalModelLayout,
) -> Result<LocalGeometry, ParallelPlanError> {
    args.validate()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let layers = usize::try_from(args.num_hidden_layers)
        .map_err(|_| ParallelPlanError::InvalidGroup("Kimi layer count exceeds usize".into()))?;
    let blocks = (0..layers)
        .map(|layer| local_block_geometry(args, layer, layout))
        .collect::<Result<Vec<_>, _>>()?;
    let state_layout = state_layout_with_geometry(args, &state_geometry(args, &blocks))
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let global_vocabulary = usize::try_from(args.vocab_size).map_err(|_| {
        ParallelPlanError::InvalidGroup("Kimi vocabulary size exceeds usize".into())
    })?;
    let embedding_range = vocabulary_range(layout, "model.embed_tokens", global_vocabulary)?;
    let output_range = if args.tie_word_embeddings {
        None
    } else {
        Some(vocabulary_range(layout, "lm_head", global_vocabulary)?)
    };
    let geometry = LocalGeometry {
        blocks,
        embedding_range,
        output_range,
        state_layout,
        architecture_fingerprint: prompt_cache_architecture_fingerprint(args),
        global_block_geometry: BlockGeometry::replicated(args),
        tied_head: args.tie_word_embeddings,
    };
    geometry.validate_for(args)?;
    Ok(geometry)
}

/// Declares vocabulary and replicated final-normalization groups.
pub fn static_parallel_parameter_groups<B>(
    modules: &StaticModules<B>,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError>
where
    B: GroupedNeuralBackend + BlockwiseAttentionBackend,
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
    B: GroupedNeuralBackend + BlockwiseAttentionBackend,
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
                ParameterRole::FeedForwardIntermediate,
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

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_runtime::{LocalTensorLayout, ParameterRole};

    fn args() -> ModelArgs {
        crate::kimi_linear::model_args_from_config_value(&serde_json::json!({
            "model_type":"kimi_linear","vocab_size":16,"hidden_size":12,"num_hidden_layers":2,
            "num_attention_heads":3,"num_key_value_heads":3,"intermediate_size":17,"head_dim":4,
            "model_max_length":64,"linear_attn_config":{"kda_layers":[1],"full_attn_layers":[2],"num_heads":3,"head_dim":4,"short_conv_kernel_size":3},
            "num_experts":2,"moe_intermediate_size":9,"kv_lora_rank":6,"qk_nope_head_dim":4,"qk_rope_head_dim":2,"v_head_dim":4,
            "mla_use_nope":true,"num_experts_per_token":1,"num_shared_experts":1,"routed_scaling_factor":1.0,
            "first_k_dense_replace":1,"num_expert_group":1,"topk_group":1
        }))
        .unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn insert(
        layout: &mut LocalModelLayout,
        target: &str,
        logical: &str,
        global: Vec<usize>,
        local: Vec<usize>,
        placement: TensorPlacement,
    ) {
        layout.insert(
            target.into(),
            LocalTensorLayout::new(
                logical,
                ParameterRole::Replicated,
                global,
                local,
                placement,
                None,
                None,
                false,
            ),
        );
    }

    fn valid_layout() -> LocalModelLayout {
        let mut layout = LocalModelLayout::default();
        insert(
            &mut layout,
            "model.embed_tokens.weight",
            "model.embed_tokens",
            vec![16, 12],
            vec![8, 12],
            TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 8,
            },
        );
        insert(
            &mut layout,
            "lm_head.weight",
            "lm_head",
            vec![16, 12],
            vec![8, 12],
            TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 8,
            },
        );
        insert(
            &mut layout,
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.heads",
            vec![12, 12],
            vec![8, 12],
            TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 8,
            },
        );
        insert(
            &mut layout,
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.intermediate",
            vec![17, 12],
            vec![9, 12],
            TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 9,
            },
        );
        insert(
            &mut layout,
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.self_attn.heads",
            vec![18, 12],
            vec![12, 12],
            TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 12,
            },
        );
        insert(
            &mut layout,
            "model.layers.1.mlp.experts.gate_up_proj",
            "model.layers.1.mlp.experts.intermediate",
            vec![2, 18, 12],
            vec![2, 10, 12],
            TensorPlacement::Range {
                axis: 1,
                start: 0,
                end: 10,
            },
        );
        insert(
            &mut layout,
            "model.layers.1.mlp.shared_experts.gate_proj.weight",
            "model.layers.1.mlp.shared_experts.intermediate",
            vec![9, 12],
            vec![5, 12],
            TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 5,
            },
        );
        layout
    }

    #[test]
    fn local_geometry_owns_blocks_vocabulary_and_state_together() {
        let args = args();
        let geometry = local_geometry(&args, &valid_layout()).unwrap();
        assert_eq!(geometry.blocks().len(), 2);
        assert_eq!(geometry.block(0).unwrap().kda_heads, 2);
        assert_eq!(geometry.block(0).unwrap().dense_intermediate, 9);
        assert_eq!(geometry.block(1).unwrap().mla_heads, 2);
        assert_eq!(geometry.block(1).unwrap().routed_intermediate, 5);
        assert_eq!(geometry.block(1).unwrap().shared_intermediate, 5);
        assert_eq!(geometry.embedding_range().local, 0..8);
        assert_eq!(geometry.output_range().unwrap().local, 0..8);
        assert_ne!(
            geometry.state_layout(),
            &crate::kimi_linear::state_layout(&args).unwrap()
        );
        geometry.validate_for(&args).unwrap();
    }

    #[test]
    fn local_geometry_rejects_incomplete_heads_and_vocabulary_drift() {
        let args = args();
        let mut layout = valid_layout();
        insert(
            &mut layout,
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.self_attn.heads",
            vec![18, 12],
            vec![11, 12],
            TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 11,
            },
        );
        assert!(local_geometry(&args, &layout)
            .unwrap_err()
            .to_string()
            .contains("complete heads"));

        let mut layout = valid_layout();
        insert(
            &mut layout,
            "model.embed_tokens.scales",
            "model.embed_tokens",
            vec![16, 1],
            vec![8, 1],
            TensorPlacement::Range {
                axis: 0,
                start: 8,
                end: 16,
            },
        );
        assert!(local_geometry(&args, &layout)
            .unwrap_err()
            .to_string()
            .contains("inconsistent companion selections"));
    }

    #[test]
    fn tied_geometry_uses_embedding_ownership_for_output() {
        let mut args = args();
        args.tie_word_embeddings = true;
        let geometry = local_geometry(&args, &valid_layout()).unwrap();
        assert!(geometry.output_range().is_none());
        geometry.validate_for(&args).unwrap();
    }
}

//! Semantic tensor-parallel placement for LFM2 physical blocks.

use eredu_nn::{GroupedNeuralBackend, VocabularyParallelRange};
use eredu_runtime::{
    aligned_partition_units, module_parameter_group, partitioned_module_parameter_group,
    LocalModelLayout, MemberSharding, ParallelPlanError, ParameterGroupSpec, ParameterRole,
    StateLayout, TensorPlacement,
};

use crate::decoder::StaticModules;

use super::{
    prompt_cache_architecture_fingerprint, state_layout_with_geometry, Block, BlockGeometry,
    DenseSwiGlu, FeedForward, LayerCacheGeometry, ModelArgs, OperatorPolicy, TokenMixer,
};

/// Complete planner-derived construction and state geometry for one LFM2 rank.
///
/// Fields remain private so block widths, vocabulary ownership, and mutable
/// state can only be realized together from the same typed local layout.
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

    /// Returns the authoritative state layout derived from local blocks.
    pub const fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    pub(super) fn validate_for(&self, args: &ModelArgs) -> Result<(), ParallelPlanError> {
        let expected_layers = usize::try_from(args.num_hidden_layers).map_err(|_| {
            ParallelPlanError::InvalidGroup("LFM2 layer count exceeds usize".into())
        })?;
        if self.blocks.len() != expected_layers
            || self.architecture_fingerprint != prompt_cache_architecture_fingerprint(args)
            || self.global_block_geometry != BlockGeometry::replicated(args)
            || self.tied_head != args.tie_word_embeddings
        {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local LFM2 geometry belongs to a different model configuration".into(),
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
                    "tied LFM2 output unexpectedly owns a separate vocabulary range".into(),
                ))
            }
            (false, None) => {
                return Err(ParallelPlanError::InvalidGroup(
                    "untied LFM2 output is missing vocabulary ownership".into(),
                ))
            }
        }
        let expected = state_layout_with_geometry(args, &state_geometry(args, &self.blocks))
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        if expected != self.state_layout {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local LFM2 state layout drifted from block geometry".into(),
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
        ParallelPlanError::InvalidTensor(format!("missing local LFM2 layout for {name}"))
    })?;
    let local = *tensor.local_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("LFM2 tensor {name} has no axis {axis}"))
    })?;
    let global = *tensor.global_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("LFM2 tensor {name} has no global axis {axis}"))
    })?;
    if local == 0 || local > global {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "LFM2 tensor {name} has invalid local width {local} of global width {global}"
        )));
    }
    i32::try_from(local)
        .map_err(|_| ParallelPlanError::InvalidTensor(format!("LFM2 width for {name} exceeds i32")))
}

/// Derives one block's local widths exclusively from the resolved placement.
pub fn local_block_geometry(
    args: &ModelArgs,
    layer: usize,
    layout: &LocalModelLayout,
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
        .map(|(policy, geometry)| match policy.operator {
            OperatorPolicy::SelfAttention(_) => LayerCacheGeometry {
                kv_heads: Some(geometry.key_value_heads),
                convolution_channels: None,
            },
            OperatorPolicy::CausalConvolution => LayerCacheGeometry {
                kv_heads: None,
                convolution_channels: Some(geometry.convolution_channels),
            },
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
                "LFM2 vocabulary member {target} has global shape {:?}, expected {global_vocabulary} rows",
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
                    "LFM2 vocabulary member {target} has non-row placement {placement:?}"
                )))
            }
        };
        if selected.as_ref().is_some_and(|current| current != &range) {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "LFM2 vocabulary group {logical_name} has inconsistent companion selections"
            )));
        }
        selected = Some(range);
    }
    if !found {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "missing local LFM2 vocabulary layout for {logical_name}"
        )));
    }
    let range = VocabularyParallelRange {
        global_vocabulary,
        local: selected.expect("a found LFM2 vocabulary member supplies a selection"),
    };
    range
        .validate()
        .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
    Ok(range)
}

/// Derives complete rank-local LFM2 construction geometry from one typed plan.
pub fn local_geometry(
    args: &ModelArgs,
    layout: &LocalModelLayout,
) -> Result<LocalGeometry, ParallelPlanError> {
    args.validate()
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let layers = usize::try_from(args.num_hidden_layers)
        .map_err(|_| ParallelPlanError::InvalidGroup("LFM2 layer count exceeds usize".into()))?;
    let blocks = (0..layers)
        .map(|layer| local_block_geometry(args, layer, layout))
        .collect::<Result<Vec<_>, _>>()?;
    let state_layout = state_layout_with_geometry(args, &state_geometry(args, &blocks))
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let global_vocabulary = usize::try_from(args.vocab_size).map_err(|_| {
        ParallelPlanError::InvalidGroup("LFM2 vocabulary size exceeds usize".into())
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
pub fn static_parallel_parameter_groups<
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
>(
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
pub fn layer_parallel_parameter_groups<
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
>(
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

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_runtime::{LocalTensorLayout, ParameterRole};

    fn args() -> ModelArgs {
        crate::lfm2::model_args_from_config_value(&serde_json::json!({
            "model_type":"lfm2_moe","vocab_size":16,"hidden_size":12,
            "intermediate_size":18,"num_hidden_layers":3,
            "num_attention_heads":3,"num_key_value_heads":3,
            "max_position_embeddings":64,"layer_types":["conv","full_attention","conv"],
            "conv_L_cache":3,"block_auto_adjust_ff_dim":false,
            "num_dense_layers":1,"moe_intermediate_size":8,
            "num_experts":2,"num_experts_per_tok":1,"tie_word_embeddings":false
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

    fn range(axis: usize, end: usize) -> TensorPlacement {
        TensorPlacement::Range {
            axis,
            start: 0,
            end,
        }
    }

    fn valid_layout() -> LocalModelLayout {
        let mut layout = LocalModelLayout::default();
        insert(
            &mut layout,
            "model.embed_tokens.weight",
            "model.embed_tokens",
            vec![16, 12],
            vec![8, 12],
            range(0, 8),
        );
        insert(
            &mut layout,
            "lm_head.weight",
            "lm_head",
            vec![16, 12],
            vec![8, 12],
            range(0, 8),
        );
        insert(
            &mut layout,
            "model.layers.0.conv.conv.weight",
            "model.layers.0.conv.channels",
            vec![12, 1, 3],
            vec![8, 1, 3],
            range(0, 8),
        );
        insert(
            &mut layout,
            "model.layers.0.feed_forward.w1.weight",
            "model.layers.0.feed_forward.intermediate",
            vec![18, 12],
            vec![9, 12],
            range(0, 9),
        );
        insert(
            &mut layout,
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.self_attn.heads",
            vec![12, 12],
            vec![8, 12],
            range(0, 8),
        );
        insert(
            &mut layout,
            "model.layers.1.self_attn.k_proj.weight",
            "model.layers.1.self_attn.heads",
            vec![12, 12],
            vec![8, 12],
            range(0, 8),
        );
        insert(
            &mut layout,
            "model.layers.1.feed_forward.experts.gate_up_proj",
            "model.layers.1.feed_forward.experts.intermediate",
            vec![2, 16, 12],
            vec![2, 10, 12],
            range(1, 10),
        );
        insert(
            &mut layout,
            "model.layers.2.conv.conv.weight",
            "model.layers.2.conv.channels",
            vec![12, 1, 3],
            vec![8, 1, 3],
            range(0, 8),
        );
        insert(
            &mut layout,
            "model.layers.2.feed_forward.experts.gate_up_proj",
            "model.layers.2.feed_forward.experts.intermediate",
            vec![2, 16, 12],
            vec![2, 10, 12],
            range(1, 10),
        );
        layout
    }

    #[test]
    fn local_geometry_owns_blocks_vocabulary_and_state_together() {
        let args = args();
        let geometry = local_geometry(&args, &valid_layout()).unwrap();
        assert_eq!(geometry.blocks().len(), 3);
        assert_eq!(geometry.block(0).unwrap().convolution_channels, 8);
        assert_eq!(geometry.block(0).unwrap().dense_intermediate, 9);
        assert_eq!(geometry.block(1).unwrap().query_heads, 2);
        assert_eq!(geometry.block(1).unwrap().key_value_heads, 2);
        assert_eq!(geometry.block(1).unwrap().expert_intermediate, 5);
        assert_eq!(geometry.block(2).unwrap().convolution_channels, 8);
        assert_eq!(geometry.embedding_range().local, 0..8);
        assert_eq!(geometry.output_range().unwrap().local, 0..8);
        assert_ne!(
            geometry.state_layout(),
            &crate::lfm2::state_layout(&args).unwrap()
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
            vec![12, 12],
            vec![7, 12],
            range(0, 7),
        );
        assert!(local_geometry(&args, &layout)
            .unwrap_err()
            .to_string()
            .contains("split head dimension"));

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

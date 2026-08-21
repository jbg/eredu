//! Semantic tensor-parallel placement for shared Qwen hybrid units.

use eredu_nn::{RoutedNeuralBackend, VocabularyParallelRange};
use eredu_runtime::{
    aligned_partition_units, module_parameter_group, partitioned_module_parameter_group,
    LocalModelLayout, MemberSharding, ParallelPlanError, ParameterGroupSpec, ParameterRole,
    StateLayout, TensorPlacement,
};

use super::{
    prompt_cache_architecture_fingerprint, state_layout_with_geometry, Block, FeedForward,
    HybridConfig, HybridLayerPolicy, HybridStateGeometry, ParsedHybridConfig, PredictionUnit,
    TokenMixer, Unit,
};

use crate::qwen::vision;

/// Complete planner-derived geometry for target, MTP, vocabulary, and state.
#[derive(Debug, Clone)]
pub struct LocalGeometry {
    target: Vec<HybridConfig>,
    prediction: Vec<HybridConfig>,
    embedding_range: VocabularyParallelRange,
    output_range: Option<VocabularyParallelRange>,
    state_layout: StateLayout,
    architecture_fingerprint: String,
}

impl LocalGeometry {
    /// Returns one target block's rank-local construction policy.
    pub fn target(&self, layer: usize) -> Option<&HybridConfig> {
        self.target.get(layer)
    }

    /// Returns one MTP depth's rank-local construction policy.
    pub fn prediction(&self, depth: usize) -> Option<&HybridConfig> {
        self.prediction.get(depth)
    }

    /// Returns input-vocabulary ownership.
    pub const fn embedding_range(&self) -> &VocabularyParallelRange {
        &self.embedding_range
    }

    /// Returns untied output-vocabulary ownership.
    pub const fn output_range(&self) -> Option<&VocabularyParallelRange> {
        self.output_range.as_ref()
    }

    /// Returns the authoritative heterogeneous rank-local state layout.
    pub const fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    pub(super) fn validate_for(&self, config: &HybridConfig) -> Result<(), ParallelPlanError> {
        let targets = usize::try_from(config.num_hidden_layers)
            .map_err(|_| invalid("Qwen hybrid target layer count is negative"))?;
        let predictions = usize::try_from(config.mtp_num_hidden_layers)
            .map_err(|_| invalid("Qwen hybrid MTP layer count is negative"))?;
        if self.architecture_fingerprint != prompt_cache_architecture_fingerprint(config)
            || self.target.len() != targets
            || self.prediction.len() != predictions
        {
            return Err(invalid(
                "rank-local Qwen hybrid geometry belongs to another configuration",
            ));
        }
        self.embedding_range
            .validate_global_rows(config.vocab_size)
            .map_err(|error| invalid(error.to_string()))?;
        match (config.tie_word_embeddings, self.output_range.as_ref()) {
            (true, None) => {}
            (false, Some(range)) => range
                .validate_global_rows(config.vocab_size)
                .map_err(|error| invalid(error.to_string()))?,
            (true, Some(_)) => return Err(invalid("tied Qwen hybrid output has a separate range")),
            (false, None) => return Err(invalid("untied Qwen hybrid output has no range")),
        }
        let expected = local_state_layout(config, &self.target, &self.prediction)?;
        if expected != self.state_layout {
            return Err(invalid("Qwen hybrid local state geometry drifted"));
        }
        Ok(())
    }
}

/// Derives all target/MTP/vocabulary/state geometry from one typed placement.
pub fn local_geometry(
    config: &HybridConfig,
    layout: &LocalModelLayout,
) -> Result<LocalGeometry, ParallelPlanError> {
    let targets = usize::try_from(config.num_hidden_layers)
        .map_err(|_| invalid("Qwen hybrid target layer count is negative"))?;
    let predictions = usize::try_from(config.mtp_num_hidden_layers)
        .map_err(|_| invalid("Qwen hybrid MTP layer count is negative"))?;
    let target = (0..targets)
        .map(|layer| local_block_config(config, layer, layout))
        .collect::<Result<Vec<_>, _>>()?;
    let prediction = (0..predictions)
        .map(|depth| local_unit_config(config, depth + 1, 0, layout))
        .collect::<Result<Vec<_>, _>>()?;
    let embedding_range = vocabulary_range(layout, "model.embed_tokens", config.vocab_size)?;
    let output_range = if config.tie_word_embeddings {
        None
    } else {
        Some(vocabulary_range(layout, "lm_head", config.vocab_size)?)
    };
    let state_layout = local_state_layout(config, &target, &prediction)?;
    let geometry = LocalGeometry {
        target,
        prediction,
        embedding_range,
        output_range,
        state_layout,
        architecture_fingerprint: prompt_cache_architecture_fingerprint(config),
    };
    geometry.validate_for(config)?;
    Ok(geometry)
}

fn local_state_layout(
    config: &HybridConfig,
    target: &[HybridConfig],
    prediction: &[HybridConfig],
) -> Result<StateLayout, ParallelPlanError> {
    let mut geometry = target
        .iter()
        .enumerate()
        .map(|(layer, local)| match config.layer_schedule.get(layer) {
            Some(HybridLayerPolicy::SelfAttention(_)) => HybridStateGeometry::FullAttention {
                key_value_heads: local.num_key_value_heads,
            },
            Some(HybridLayerPolicy::LinearAttention) => HybridStateGeometry::LinearAttention {
                key_heads: local.linear_num_key_heads,
                value_heads: local.linear_num_value_heads,
            },
            None => HybridStateGeometry::FullAttention { key_value_heads: 0 },
        })
        .collect::<Vec<_>>();
    geometry.extend(
        prediction
            .iter()
            .map(|local| HybridStateGeometry::FullAttention {
                key_value_heads: local.num_key_value_heads,
            }),
    );
    state_layout_with_geometry(config, &geometry).map_err(|error| invalid(error.to_string()))
}

fn vocabulary_range(
    layout: &LocalModelLayout,
    logical_name: &str,
    vocabulary: i32,
) -> Result<VocabularyParallelRange, ParallelPlanError> {
    let vocabulary =
        usize::try_from(vocabulary).map_err(|_| invalid("Qwen hybrid vocabulary is negative"))?;
    let mut selected = None;
    for (target, tensor) in layout
        .tensors()
        .filter(|(_, tensor)| tensor.logical_name() == logical_name)
    {
        if tensor.global_shape().first().copied() != Some(vocabulary) {
            return Err(invalid(format!(
                "Qwen hybrid vocabulary member {target} has global shape {:?}",
                tensor.global_shape()
            )));
        }
        let range = match tensor.placement() {
            TensorPlacement::Range {
                axis: 0,
                start,
                end,
            } => *start..*end,
            TensorPlacement::Replicated | TensorPlacement::Local => 0..vocabulary,
            placement => {
                return Err(invalid(format!(
                    "Qwen hybrid vocabulary member {target} has placement {placement:?}"
                )))
            }
        };
        if selected.as_ref().is_some_and(|current| current != &range) {
            return Err(invalid(format!(
                "Qwen hybrid vocabulary group {logical_name} is inconsistent"
            )));
        }
        selected = Some(range);
    }
    let range = VocabularyParallelRange {
        global_vocabulary: vocabulary,
        local: selected.ok_or_else(|| {
            invalid(format!(
                "missing local Qwen hybrid vocabulary layout for {logical_name}"
            ))
        })?,
    };
    range
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    Ok(range)
}

fn invalid(message: impl Into<String>) -> ParallelPlanError {
    ParallelPlanError::InvalidTensor(message.into())
}

/// Complete local geometry for the conditional vision + target + MTP graph.
#[derive(Debug, Clone)]
pub struct ConditionalLocalGeometry {
    text: LocalGeometry,
    vision_blocks: Vec<(i32, i32)>,
    merger_widths: Vec<i32>,
}

impl ConditionalLocalGeometry {
    /// Returns the local target/MTP/vocabulary/state geometry.
    pub const fn text(&self) -> &LocalGeometry {
        &self.text
    }

    /// Returns one vision unit's local `(heads, intermediate)` geometry.
    pub fn vision_block(&self, layer: usize) -> Option<(i32, i32)> {
        self.vision_blocks.get(layer).copied()
    }

    /// Returns local main/deepstack merger widths.
    pub fn merger_widths(&self) -> &[i32] {
        &self.merger_widths
    }

    /// Returns authoritative heterogeneous target and MTP state.
    pub const fn state_layout(&self) -> &StateLayout {
        self.text.state_layout()
    }

    pub(super) fn validate_for(
        &self,
        parsed: &ParsedHybridConfig,
    ) -> Result<(), ParallelPlanError> {
        self.text.validate_for(&parsed.text)?;
        let vision = parsed
            .vision
            .as_ref()
            .ok_or_else(|| invalid("conditional Qwen geometry has no vision policy"))?;
        if self.vision_blocks.len() != vision.layer_count()
            || self.merger_widths.len() != vision.deepstack_layer_count() + 1
            || self
                .vision_blocks
                .iter()
                .any(|(heads, width)| *heads <= 0 || *width <= 0)
            || self.merger_widths.iter().any(|width| *width <= 0)
        {
            return Err(invalid(
                "conditional Qwen local vision geometry is incomplete",
            ));
        }
        Ok(())
    }
}

/// Derives conditional text, MTP, vocabulary, vision, and state geometry.
pub fn conditional_local_geometry(
    parsed: &ParsedHybridConfig,
    layout: &LocalModelLayout,
) -> Result<ConditionalLocalGeometry, ParallelPlanError> {
    let vision_config = parsed
        .vision
        .as_ref()
        .ok_or_else(|| invalid("conditional Qwen requires vision geometry"))?;
    let text = local_geometry(&parsed.text, layout)?;
    let vision_blocks = (0..vision_config.layer_count())
        .map(|layer| vision::local_block_geometry(vision_config, "model.visual", layer, layout))
        .collect::<Result<Vec<_>, _>>()?;
    let merger_widths = vision::local_merger_widths(vision_config, "model.visual", layout)?;
    let geometry = ConditionalLocalGeometry {
        text,
        vision_blocks,
        merger_widths,
    };
    geometry.validate_for(parsed)?;
    Ok(geometry)
}

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
            if key <= 0
                || value <= 0
                || global_key <= 0
                || global_value <= 0
                || global_total <= 0
                || key * global_key % global_total != 0
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
            if query <= 0
                || key <= 0
                || config.head_dim <= 0
                || query % (2 * config.head_dim) != 0
                || key % config.head_dim != 0
            {
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
        if local.moe_intermediate_size <= 0 {
            return Err(ParallelPlanError::InvalidTensor(
                "rank-local expert width must be positive".into(),
            ));
        }
        local.shared_expert_intermediate_size = local_width(
            layout,
            &format!("{root}.mlp.shared_expert.gate_proj.weight"),
            0,
        )?;
        if local.shared_expert_intermediate_size <= 0 {
            return Err(ParallelPlanError::InvalidTensor(
                "rank-local shared expert width must be positive".into(),
            ));
        }
    } else {
        local.intermediate_size = local_width(layout, &format!("{root}.mlp.gate_proj.weight"), 0)?;
        if local.intermediate_size <= 0 {
            return Err(ParallelPlanError::InvalidTensor(
                "rank-local dense width must be positive".into(),
            ));
        }
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

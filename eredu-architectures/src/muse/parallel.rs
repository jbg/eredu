//! Semantic tensor-parallel placement for Muse-Glimmer text and vision components.

use eredu_core::{cache::LayerCachePolicy, LayerSchedule};
use eredu_nn::VocabularyParallelRange;
use eredu_runtime::{
    expand_linear_format_parameter_groups, ArchitecturePartition, LocalModelLayout, MemberSharding,
    ParallelPlanError, ParameterGroupSpec, ParameterMemberSpec, ParameterRole, PartitionOwnership,
    StateLayout,
};
use std::ops::Range;

use crate::linear_format::standard_parallel_linear_format;

use super::{BlockNormalization, DecoderConfig, RoutedFeedForwardEquation};

/// Exact sums surrounding a routed Muse unit, including inactive PP participants.
pub fn routed_tensor_reductions(
    args: &DecoderConfig,
    routed: bool,
) -> eredu_runtime::RoutedTensorReductions {
    use eredu_runtime::{
        RoutedTensorDimension as Axis, RoutedTensorReduction as Sum, RoutedTensorReductions,
    };
    match (routed, args.routed_feed_forward) {
        (true, RoutedFeedForwardEquation::CompressedSigmoid { expert_width, .. }) => {
            RoutedTensorReductions {
                before: vec![Sum::Hidden],
                after: vec![Sum::Activation {
                    shape: vec![
                        Axis::Tokens(1),
                        Axis::Fixed(args.num_experts_per_tok),
                        Axis::Fixed(expert_width),
                    ],
                }],
            }
        }
        _ => RoutedTensorReductions::hidden(1, 1),
    }
}

/// Complete planner-derived construction and mutable-state geometry for one rank.
#[derive(Debug, Clone)]
pub struct LocalGeometry {
    text_blocks: Vec<DecoderConfig>,
    embedding_range: VocabularyParallelRange,
    output_range: Option<VocabularyParallelRange>,
    state_layout: StateLayout,
    vision_layers: usize,
    architecture_fingerprint: String,
}

/// Exact TP-local and PP-local geometry for one Muse-Glimmer composite partition.
#[derive(Debug, Clone)]
pub struct PartitionLocalGeometry {
    vision_units: Option<Range<usize>>,
    audio_units: Option<Range<usize>>,
    text_units: Range<usize>,
    text_blocks: Vec<DecoderConfig>,
    embedding_range: VocabularyParallelRange,
    output_range: Option<VocabularyParallelRange>,
    complete_state_layout: StateLayout,
    static_roles: Vec<String>,
    architecture_fingerprint: String,
}

/// Validated architecture-owned handoff for one Muse-Glimmer composite partition.
#[derive(Debug, Clone)]
pub struct PartitionLocalFoundation {
    geometry: PartitionLocalGeometry,
    parameter_targets: Vec<String>,
}

impl PartitionLocalFoundation {
    /// Validates group ownership, state offset, boundary schema, and selected tasks together.
    pub fn from_partition(
        args: &DecoderConfig,
        partition: &ArchitecturePartition<
            PartitionLocalGeometry,
            eredu_runtime::NoAuxiliaryBoundarySchema,
        >,
    ) -> Result<Self, ParallelPlanError> {
        let geometry = partition.local_geometry();
        geometry.validate_for(args)?;
        let expected_groups = [
            geometry
                .vision_units()
                .map(|range| (super::VISION_EXECUTION_GROUP, range)),
            Some((super::TEXT_EXECUTION_GROUP, geometry.text_units())),
            geometry
                .audio_units()
                .map(|range| (super::audio::execution::AUDIO_EXECUTION_GROUP, range)),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        let actual_groups = partition
            .groups()
            .iter()
            .map(|group| (group.group().as_str(), group.global_units()))
            .collect::<Vec<_>>();
        if actual_groups != expected_groups {
            return Err(invalid(
                "Muse-Glimmer partition groups differ from family-local geometry",
            ));
        }
        let state = partition
            .state()
            .ok_or_else(|| invalid("Muse-Glimmer text partition has no selected state"))?;
        geometry.validate_state(state)?;
        if partition.boundary_schema()
            != &eredu_runtime::NoAuxiliaryBoundarySchema::new(args.hidden_size)
        {
            return Err(invalid(
                "Muse-Glimmer partition boundary differs from its decoder width",
            ));
        }
        let parameter_targets = partition
            .parameter_bindings()
            .iter()
            .flat_map(|binding| binding.members())
            .map(|member| member.target().to_owned())
            .collect::<Vec<_>>();
        Ok(Self {
            geometry: geometry.clone(),
            parameter_targets,
        })
    }

    /// Exact family-local construction geometry.
    pub const fn geometry(&self) -> &PartitionLocalGeometry {
        &self.geometry
    }

    /// Canonical materialization targets selected for this rank.
    pub fn parameter_targets(&self) -> &[String] {
        &self.parameter_targets
    }
}

impl PartitionLocalGeometry {
    /// Architecture-global units owned from the optional vision root.
    pub fn vision_units(&self) -> Option<Range<usize>> {
        self.vision_units.clone()
    }

    /// Architecture-global text units owned by this pipeline coordinate.
    pub fn text_units(&self) -> Range<usize> {
        self.text_units.clone()
    }

    /// TP-local configuration for one owned architecture-global text unit.
    pub fn text_block(&self, global_unit: usize) -> Option<&DecoderConfig> {
        self.text_units
            .contains(&global_unit)
            .then(|| &self.text_blocks[global_unit - self.text_units.start])
    }

    /// Complete TP-local state geometry before PP slicing.
    pub const fn complete_state_layout(&self) -> &StateLayout {
        &self.complete_state_layout
    }

    /// Architecture-global small waveform units owned by this coordinate.
    pub fn audio_units(&self) -> Option<Range<usize>> {
        self.audio_units.clone()
    }

    pub(super) fn validate_complete_geometry(
        &self,
        args: &DecoderConfig,
        complete: &LocalGeometry,
    ) -> Result<(), ParallelPlanError> {
        self.validate_for(args)?;
        complete.validate_for(args)?;
        if self.complete_state_layout != complete.state_layout
            || self.embedding_range != complete.embedding_range
            || self.output_range != complete.output_range
        {
            return Err(invalid(
                "selected Muse geometry differs from complete placement",
            ));
        }
        for unit in self.text_units.clone() {
            let expected = complete
                .text_block(unit)
                .ok_or_else(|| invalid("missing complete Muse text geometry"))?;
            if self.text_block(unit).is_none_or(|block| {
                block.architecture_fingerprint() != expected.architecture_fingerprint()
            }) {
                return Err(invalid(
                    "selected Muse text block differs from complete placement",
                ));
            }
        }
        Ok(())
    }

    fn validate_state(
        &self,
        state: &eredu_runtime::PartitionState,
    ) -> Result<(), ParallelPlanError> {
        let decoder_layers = self.complete_state_layout.len() - self.audio_state_slots();
        for layer in 0..self.complete_state_layout.len() {
            let expected = self.text_units.contains(&layer)
                || self.audio_units.as_ref().is_some_and(|units| {
                    layer >= decoder_layers && units.contains(&(layer - decoder_layers))
                });
            if state.owns_global_layer(layer) != expected {
                return Err(invalid(
                    "Muse selected state slots differ from decoder and waveform ownership",
                ));
            }
        }
        if state
            .selected_layout(&self.complete_state_layout)
            .map_err(|error| invalid(error.to_string()))?
            != *state.layout()
        {
            return Err(invalid(
                "Muse selected state changes declared component geometry",
            ));
        }
        Ok(())
    }

    fn audio_state_slots(&self) -> usize {
        self.text_blocks
            .first()
            .and_then(|block| block.audio_config.as_ref())
            .map_or(0, |audio| {
                audio
                    .execution_units()
                    .expect("validated audio geometry")
                    .len()
            })
    }

    /// Static roles selected for this pipeline coordinate.
    pub fn static_roles(&self) -> &[String] {
        &self.static_roles
    }

    /// Input-embedding vocabulary ownership for this tensor coordinate.
    pub const fn embedding_range(&self) -> &VocabularyParallelRange {
        &self.embedding_range
    }

    /// Untied output-head vocabulary ownership for this tensor coordinate.
    pub const fn output_range(&self) -> Option<&VocabularyParallelRange> {
        self.output_range.as_ref()
    }

    pub(super) fn validate_for(&self, args: &DecoderConfig) -> Result<(), ParallelPlanError> {
        let text_count = args.num_hidden_layers as usize;
        let vision_count = args
            .vision_config
            .as_ref()
            .map_or(0, |vision| vision.layer_count());
        if self.text_units.is_empty()
            || self.text_units.end > text_count
            || self.text_blocks.len() != self.text_units.len()
            || self.architecture_fingerprint != args.architecture_fingerprint()
        {
            return Err(invalid(
                "partition-local Muse-Glimmer geometry belongs to a different model or text range",
            ));
        }
        if self
            .vision_units
            .as_ref()
            .is_some_and(|range| range.is_empty() || range.end > vision_count)
            || (vision_count == 0 && self.vision_units.is_some())
        {
            return Err(invalid(
                "partition-local Muse-Glimmer vision range is invalid",
            ));
        }
        let mut expected_roles: Vec<String> = Vec::new();
        if self
            .vision_units
            .as_ref()
            .is_some_and(|range| range.start == 0)
        {
            expected_roles.push("vision".into());
        }
        if self.text_units.start == 0 {
            expected_roles.push("embedding".into());
        }
        if self.text_units.end == text_count {
            // The output consumer exists even when its parameter is the
            // shared embedding group retained by either endpoint.
            expected_roles.extend(["norm".into(), "output".into()]);
        }
        if self.static_roles != expected_roles {
            return Err(invalid(format!(
                "partition-local Muse-Glimmer static roles {:?} differ from {:?}",
                self.static_roles, expected_roles
            )));
        }
        let audio_count = args
            .audio_config
            .as_ref()
            .map(|audio| audio.execution_units().map(|units| units.len()))
            .transpose()
            .map_err(|error| invalid(error.to_string()))?
            .unwrap_or(0);
        if self.complete_state_layout.len() != text_count + audio_count {
            return Err(invalid(
                "partition-local Muse complete state length differs from decoder and audio units",
            ));
        }
        if self
            .audio_units
            .as_ref()
            .is_some_and(|units| units.is_empty() || units.end > audio_count)
        {
            return Err(invalid(
                "Muse waveform partition range is outside its declared units",
            ));
        }
        Ok(())
    }
}

impl LocalGeometry {
    /// Returns one rank-local text-block configuration.
    pub fn text_block(&self, layer: usize) -> Option<&DecoderConfig> {
        self.text_blocks.get(layer)
    }

    /// Returns all rank-local text-block configurations in execution order.
    pub fn text_blocks(&self) -> &[DecoderConfig] {
        &self.text_blocks
    }

    /// Returns input-embedding vocabulary ownership.
    pub const fn embedding_range(&self) -> &VocabularyParallelRange {
        &self.embedding_range
    }

    /// Returns untied output-head vocabulary ownership.
    pub const fn output_range(&self) -> Option<&VocabularyParallelRange> {
        self.output_range.as_ref()
    }

    /// Returns the authoritative rank-local text state layout.
    pub const fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    /// Returns the replicated media units owned by this rank.
    pub const fn vision_layers(&self) -> usize {
        self.vision_layers
    }

    pub(super) fn validate_for(&self, args: &DecoderConfig) -> Result<(), ParallelPlanError> {
        if self.architecture_fingerprint != args.architecture_fingerprint()
            || self.text_blocks.len() != args.num_hidden_layers as usize
            || self.vision_layers
                != args
                    .vision_config
                    .as_ref()
                    .map_or(0, |vision| vision.layer_count())
        {
            return Err(invalid(
                "rank-local Muse-Glimmer geometry belongs to a different configuration",
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
                return Err(invalid(
                    "tied Muse-Glimmer output has a separate vocabulary range",
                ))
            }
            (false, None) => {
                return Err(invalid(
                    "untied Muse-Glimmer output has no vocabulary range",
                ))
            }
        }
        let expected = local_state_layout(&self.text_blocks)?;
        if expected != self.state_layout {
            return Err(invalid(
                "rank-local Muse-Glimmer state layout drifted from text geometry",
            ));
        }
        Ok(())
    }
}

/// Derives all rank-local Muse-Glimmer construction geometry from one typed layout.
pub fn local_geometry(
    args: &DecoderConfig,
    layout: &LocalModelLayout,
) -> Result<LocalGeometry, ParallelPlanError> {
    let text_blocks = (0..args.num_hidden_layers as usize)
        .map(|layer| local_decoder_config(args, layer, layout))
        .collect::<Result<Vec<_>, _>>()?;
    let state_layout = local_state_layout(&text_blocks)?;
    let (embedding_range, output_range) = super::text::static_spec(args)
        .vocabulary_ranges(layout, "model.embed_tokens", "model.norm", "lm_head")?;
    let geometry = LocalGeometry {
        text_blocks,
        embedding_range,
        output_range,
        state_layout,
        // Media parameters are deliberately outside the text TP plan and are
        // replicated on every rank. Keeping their unit ownership here makes
        // the canonical multimodal graph authoritative for both lifecycles.
        vision_layers: args
            .vision_config
            .as_ref()
            .map_or(0, |vision| vision.layer_count()),
        architecture_fingerprint: args.architecture_fingerprint(),
    };
    geometry.validate_for(args)?;
    Ok(geometry)
}

/// Derives exact TP-local and PP-local Muse-Glimmer composite geometry.
pub fn partition_local_geometry(
    args: &DecoderConfig,
    layout: &LocalModelLayout,
    group_ranges: impl IntoIterator<Item = (impl AsRef<str>, Range<usize>)>,
    ownership: &PartitionOwnership,
) -> Result<PartitionLocalGeometry, ParallelPlanError> {
    if args.is_moe() {
        return Err(invalid(
            "partition-local Muse-Glimmer foundation does not admit routed text blocks",
        ));
    }
    partition_local_geometry_impl(args, layout, group_ranges, ownership)
}

pub(crate) fn routed_partition_local_geometry(
    args: &DecoderConfig,
    layout: &LocalModelLayout,
    group_ranges: impl IntoIterator<Item = (impl AsRef<str>, Range<usize>)>,
    ownership: &PartitionOwnership,
) -> Result<PartitionLocalGeometry, ParallelPlanError> {
    if !args.is_moe() {
        return Err(invalid(
            "routed Muse-Glimmer partition has no sparse text blocks",
        ));
    }
    partition_local_geometry_impl(args, layout, group_ranges, ownership)
}

fn partition_local_geometry_impl(
    args: &DecoderConfig,
    layout: &LocalModelLayout,
    group_ranges: impl IntoIterator<Item = (impl AsRef<str>, Range<usize>)>,
    ownership: &PartitionOwnership,
) -> Result<PartitionLocalGeometry, ParallelPlanError> {
    let mut vision_units = None;
    let mut audio_units = None;
    let mut text_units = None;
    for (group, range) in group_ranges {
        if range.is_empty() {
            return Err(invalid(
                "Muse-Glimmer partition group range cannot be empty",
            ));
        }
        let slot = match group.as_ref() {
            super::VISION_EXECUTION_GROUP => &mut vision_units,
            super::TEXT_EXECUTION_GROUP => &mut text_units,
            super::audio::execution::AUDIO_EXECUTION_GROUP => &mut audio_units,
            other => {
                return Err(invalid(format!(
                    "unknown Muse-Glimmer partition group {other:?}"
                )))
            }
        };
        if slot.replace(range).is_some() {
            return Err(invalid("duplicate Muse-Glimmer partition group"));
        }
    }
    let text_units = text_units
        .ok_or_else(|| invalid("Muse-Glimmer partition must own a non-empty text decoder range"))?;
    let complete = local_geometry(args, layout)?;
    let text_blocks = text_units
        .clone()
        .map(|global| {
            complete
                .text_block(global)
                .cloned()
                .ok_or_else(|| invalid(format!("Muse-Glimmer has no local text block {global}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let geometry = PartitionLocalGeometry {
        vision_units,
        audio_units,
        text_units,
        text_blocks,
        embedding_range: complete.embedding_range.clone(),
        output_range: complete.output_range.clone(),
        complete_state_layout: complete.state_layout.clone(),
        static_roles: ownership.static_roles().to_vec(),
        architecture_fingerprint: complete.architecture_fingerprint.clone(),
    };
    geometry.validate_for(args)?;
    Ok(geometry)
}

fn local_state_layout(blocks: &[DecoderConfig]) -> Result<StateLayout, ParallelPlanError> {
    let mut layers = blocks
        .iter()
        .enumerate()
        .map(|(layer, block)| {
            let policy = block.attention_schedule.get(layer).ok_or_else(|| {
                invalid(format!(
                    "rank-local Muse-Glimmer block {layer} has no attention policy"
                ))
            })?;
            LayerCachePolicy::key_value(*policy, block.num_key_value_heads, block.head_dim)
                .map_err(|error| invalid(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(audio) = blocks.first().and_then(|block| block.audio_config.as_ref()) {
        let audio = super::audio::execution::state_layout(audio)
            .map_err(|error| invalid(error.to_string()))?;
        layers.extend(audio.layers().iter().cloned());
    }
    StateLayout::new(
        LayerSchedule::new(layers.len(), layers).map_err(|error| invalid(error.to_string()))?,
    )
    .map_err(|error| invalid(error.to_string()))
}

/// Derives rank-local text construction geometry from the semantic placement.
pub fn local_decoder_config(
    args: &DecoderConfig,
    layer: usize,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<DecoderConfig, ParallelPlanError> {
    if layer >= args.num_hidden_layers as usize {
        return Err(invalid(format!(
            "Muse-Glimmer layer {layer} is out of range"
        )));
    }
    let root = format!("model.layers.{layer}");
    let tensor = |suffix: &str| {
        layout
            .tensor(&format!("{root}.{suffix}.weight"))
            .or_else(|| layout.tensor(&format!("{root}.{suffix}")))
            .ok_or_else(|| {
                ParallelPlanError::InvalidTensor(format!(
                    "missing local layout for {root}.{suffix}"
                ))
            })
    };
    let query_width = local_axis(tensor("self_attn.q_proj")?, 0, "query width")?;
    let key_width = local_axis(tensor("self_attn.k_proj")?, 0, "key width")?;
    if query_width % args.head_dim != 0 || key_width % args.head_dim != 0 {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "local Muse-Glimmer attention widths q={query_width}, k={key_width} split head dimension {}",
            args.head_dim
        )));
    }
    let mut local = args.clone();
    local.num_attention_heads = query_width / args.head_dim;
    local.num_key_value_heads = key_width / args.head_dim;
    if args.is_moe() {
        let fused = local_axis(
            tensor("mlp.experts.gate_up_proj")?,
            1,
            "expert gate/up width",
        )?;
        if fused % 2 != 0 {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "local Muse-Glimmer expert gate/up width {fused} is not even"
            )));
        }
        local.moe_intermediate_size = fused / 2;
    } else {
        local.intermediate_size =
            local_axis(tensor("mlp.gate_proj")?, 0, "dense intermediate width")?;
    }
    Ok(local)
}

fn local_axis(
    tensor: &eredu_runtime::LocalTensorLayout,
    axis: usize,
    label: &str,
) -> Result<i32, ParallelPlanError> {
    let value = *tensor.local_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!(
            "local Muse-Glimmer {label} tensor has no axis {axis}"
        ))
    })?;
    i32::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            ParallelPlanError::InvalidTensor(format!(
                "local Muse-Glimmer {label} must be positive and fit i32"
            ))
        })
}

/// Declares pinned text embeddings, final normalization, and vocabulary output.
pub fn static_parameter_groups(
    args: &DecoderConfig,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    Ok(static_resource_parameter_groups(args)?
        .into_iter()
        .map(|(_, group)| group)
        .collect())
}

pub(crate) fn static_resource_parameter_groups(
    args: &DecoderConfig,
) -> Result<Vec<(&'static str, ParameterGroupSpec)>, ParallelPlanError> {
    super::text::static_spec(args).pinned_parameter_groups(
        "model.embed_tokens",
        "model.norm",
        "lm_head",
    )
}

/// Declares one decoder block's head, dense/expert, and replicated groups.
pub fn layer_parameter_groups(
    args: &DecoderConfig,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    if layer >= args.num_hidden_layers as usize {
        return Err(invalid(format!(
            "Muse-Glimmer layer {layer} is out of range"
        )));
    }
    let hidden = dim(args.hidden_size)?;
    let query_heads = dim(args.num_attention_heads)?;
    let key_value_heads = dim(args.num_key_value_heads)?;
    let head = dim(args.head_dim)?;
    let query_width = query_heads
        .checked_mul(head)
        .ok_or_else(|| invalid("Muse-Glimmer query width overflow"))?;
    let key_value_width = key_value_heads
        .checked_mul(head)
        .ok_or_else(|| invalid("Muse-Glimmer key/value width overflow"))?;
    let root = format!("model.layers.{layer}");
    let attention = format!("{root}.self_attn");
    let mut groups = vec![ParameterGroupSpec::partitioned(
        format!("{attention}.query_heads"),
        ParameterRole::AttentionHeads,
        query_heads,
        [
            member(
                format!("{attention}.q_proj.weight"),
                vec![query_width, hidden],
                partitioned(0),
            ),
            member(
                format!("{attention}.gate_proj.weight"),
                vec![query_width, hidden],
                partitioned(0),
            ),
            member(
                format!("{attention}.o_proj.weight"),
                vec![hidden, query_width],
                partitioned(1),
            ),
        ],
    )?
    .with_partition_unit_assignment(eredu_runtime::PartitionUnitAssignment::Even)?];
    groups.push(
        ParameterGroupSpec::partitioned(
            format!("{attention}.key_value_heads"),
            ParameterRole::AttentionHeads,
            key_value_heads,
            [
                member(
                    format!("{attention}.k_proj.weight"),
                    vec![key_value_width, hidden],
                    partitioned(0),
                ),
                member(
                    format!("{attention}.v_proj.weight"),
                    vec![key_value_width, hidden],
                    partitioned(0),
                ),
            ],
        )?
        .with_partition_unit_assignment(
            eredu_runtime::PartitionUnitAssignment::Repeat {
                semantic_units: key_value_heads,
            },
        )?,
    );
    if args.weight_convention == super::WeightConvention::Gguf {
        groups.push(replicated(
            format!("{attention}.qk_norm"),
            [
                (format!("{attention}.q_norm.weight"), vec![head]),
                (format!("{attention}.k_norm.weight"), vec![head]),
            ],
        )?);
    }
    if args.is_moe() {
        let experts = dim(args.num_experts)?;
        let expert_width = match args.routed_feed_forward {
            RoutedFeedForwardEquation::DirectSoftmax => hidden,
            RoutedFeedForwardEquation::CompressedSigmoid { expert_width, .. } => dim(expert_width)?,
        };
        let intermediate = dim(args.moe_intermediate_size)?;
        groups.push(ParameterGroupSpec::partitioned(
            format!("{root}.mlp.experts.intermediate"),
            ParameterRole::ExpertIntermediate,
            intermediate,
            [
                member(
                    format!("{root}.mlp.experts.gate_up_proj"),
                    vec![experts, 2 * intermediate, expert_width],
                    MemberSharding::PartitionedSegments {
                        axis: 1,
                        segments: vec![0..intermediate, intermediate..2 * intermediate],
                    },
                ),
                member(
                    format!("{root}.mlp.experts.down_proj"),
                    vec![experts, expert_width, intermediate],
                    partitioned(2),
                ),
            ],
        )?);
        let mut router = vec![(format!("{root}.mlp.gate.weight"), vec![experts, hidden])];
        if let RoutedFeedForwardEquation::CompressedSigmoid { .. } = args.routed_feed_forward {
            router.push((
                format!("{root}.mlp.gate.e_score_correction_bias"),
                vec![experts],
            ));
            groups.push(replicated(
                format!("{root}.mlp.compression"),
                [
                    (
                        format!("{root}.mlp.pre_expert_proj.weight"),
                        vec![expert_width, hidden],
                    ),
                    (
                        format!("{root}.mlp.pre_expert_norm.weight"),
                        vec![expert_width],
                    ),
                    (
                        format!("{root}.mlp.experts.post_expert_norm.weight"),
                        vec![expert_width],
                    ),
                    (
                        format!("{root}.mlp.post_expert_proj.weight"),
                        vec![hidden, expert_width],
                    ),
                ],
            )?);
        }
        groups.push(replicated(format!("{root}.mlp.router"), router)?);
    } else {
        let intermediate = dim(args.intermediate_size)?;
        groups.push(ParameterGroupSpec::partitioned(
            format!("{root}.mlp.intermediate"),
            ParameterRole::FeedForwardIntermediate,
            intermediate,
            [
                member(
                    format!("{root}.mlp.gate_proj.weight"),
                    vec![intermediate, hidden],
                    partitioned(0),
                ),
                member(
                    format!("{root}.mlp.up_proj.weight"),
                    vec![intermediate, hidden],
                    partitioned(0),
                ),
                member(
                    format!("{root}.mlp.down_proj.weight"),
                    vec![hidden, intermediate],
                    partitioned(1),
                ),
            ],
        )?);
    }
    let mut norms = vec![
        (format!("{root}.input_layernorm.weight"), vec![hidden]),
        (
            format!("{root}.pre_feedforward_layernorm.weight"),
            vec![hidden],
        ),
        (
            format!("{root}.post_feedforward_layernorm.weight"),
            vec![hidden],
        ),
    ];
    if args.block_normalization == BlockNormalization::OutputScaled {
        norms.push((
            format!("{root}.post_attention_layernorm.weight"),
            vec![hidden],
        ));
    }
    groups.push(replicated(format!("{root}.norms"), norms)?);
    let residual_names = super::checkpoint::residual_parameter_names(args);
    if !residual_names.is_empty() {
        groups.push(replicated(
            format!("{root}.residual_gates"),
            residual_names
                .into_iter()
                .map(|name| (format!("{root}.{name}"), vec![hidden])),
        )?);
    }
    expand_linear_format_parameter_groups(groups, |member| {
        standard_parallel_linear_format(member, args.linear_format_for(member.target()))
    })
}

/// Independently materialized pinned vision resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum VisionStaticResource {
    Patch,
    Norms,
    Adapter,
    Projection,
}
impl VisionStaticResource {
    pub(crate) const ALL: [Self; 4] = [Self::Patch, Self::Norms, Self::Adapter, Self::Projection];
    pub(crate) const fn logical_name(self) -> &'static str {
        match self {
            Self::Patch => "model.vision_tower.patch_channels",
            Self::Norms => "model.vision_tower.static_norms",
            Self::Adapter => "model.vision_adapter.intermediate",
            Self::Projection => "model.vision_projection",
        }
    }
    fn owns_target(self, target: &str) -> bool {
        match self {
            Self::Patch => target.starts_with("model.vision_tower.patch_embedder."),
            Self::Norms => {
                target.starts_with("model.vision_tower.ln_pre.")
                    || target.starts_with("model.vision_tower.ln_post.")
            }
            Self::Adapter => target.starts_with("model.vision_adapter."),
            Self::Projection => target.starts_with("model.vision_projection."),
        }
    }
    /// Request preparation uses patch/position and norms on each vision owner;
    /// only the last owner needs the language adapter and projection.
    pub(crate) fn consumers(self, layers: usize) -> std::ops::Range<usize> {
        match self {
            Self::Patch | Self::Norms => 0..layers,
            Self::Adapter | Self::Projection => layers.saturating_sub(1)..layers,
        }
    }
}
#[derive(Clone, Copy)]
enum VisionParameterOwner {
    Static(VisionStaticResource),
    Layer(usize),
}

/// Declares the patch/position roots, vision blocks, merge adapter, and projection.
pub fn vision_parameter_groups(
    args: &DecoderConfig,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    Ok(declared_vision_groups(args)?
        .into_iter()
        .map(|(_, group)| group)
        .collect())
}

fn declared_vision_groups(
    args: &DecoderConfig,
) -> Result<Vec<(VisionParameterOwner, ParameterGroupSpec)>, ParallelPlanError> {
    let Some(config) = &args.vision_config else {
        return Ok(Vec::new());
    };
    let hidden = dim(config.hidden_size)?;
    let _ = dim(config.num_heads)?;
    let patch_input = dim(config.temporal_patch_size * 3 * config.patch_size * config.patch_size)?;
    let positions = dim(config.position_height * config.position_width)?;
    let mut groups = vec![(
        VisionParameterOwner::Static(VisionStaticResource::Patch),
        ParameterGroupSpec::new(
            VisionStaticResource::Patch.logical_name(),
            ParameterRole::Channels,
            [
                member(
                    "model.vision_tower.patch_embedder.patch_embedding.weight",
                    vec![hidden, patch_input],
                    MemberSharding::Replicated,
                ),
                member(
                    "model.vision_tower.patch_embedder.position_embedding_table.weight",
                    vec![positions, hidden],
                    MemberSharding::Replicated,
                ),
            ],
        )?,
    )];
    for layer in 0..config.layer_count() {
        let root = format!("model.vision_tower.layers.{layer}");
        groups.push((
            VisionParameterOwner::Layer(layer),
            ParameterGroupSpec::new(
                format!("{root}.attention_heads"),
                ParameterRole::AttentionHeads,
                [
                    member(
                        format!("{root}.attn.q_proj.weight"),
                        vec![hidden, hidden],
                        MemberSharding::Replicated,
                    ),
                    member(
                        format!("{root}.attn.q_proj.bias"),
                        vec![hidden],
                        MemberSharding::Replicated,
                    ),
                    member(
                        format!("{root}.attn.k_proj.weight"),
                        vec![hidden, hidden],
                        MemberSharding::Replicated,
                    ),
                    member(
                        format!("{root}.attn.k_proj.bias"),
                        vec![hidden],
                        MemberSharding::Replicated,
                    ),
                    member(
                        format!("{root}.attn.v_proj.weight"),
                        vec![hidden, hidden],
                        MemberSharding::Replicated,
                    ),
                    member(
                        format!("{root}.attn.v_proj.bias"),
                        vec![hidden],
                        MemberSharding::Replicated,
                    ),
                    member(
                        format!("{root}.attn.proj.weight"),
                        vec![hidden, hidden],
                        MemberSharding::Replicated,
                    ),
                    member(
                        format!("{root}.attn.proj.bias"),
                        vec![hidden],
                        MemberSharding::Replicated,
                    ),
                ],
            )?,
        ));
        let intermediate = dim(config.intermediate_size)?;
        groups.push((
            VisionParameterOwner::Layer(layer),
            ParameterGroupSpec::new(
                format!("{root}.mlp.intermediate"),
                ParameterRole::FeedForwardIntermediate,
                [
                    member(
                        format!("{root}.mlp.fc1.weight"),
                        vec![intermediate, hidden],
                        MemberSharding::Replicated,
                    ),
                    member(
                        format!("{root}.mlp.fc1.bias"),
                        vec![intermediate],
                        MemberSharding::Replicated,
                    ),
                    member(
                        format!("{root}.mlp.fc2.weight"),
                        vec![hidden, intermediate],
                        MemberSharding::Replicated,
                    ),
                    member(
                        format!("{root}.mlp.fc2.bias"),
                        vec![hidden],
                        MemberSharding::Replicated,
                    ),
                ],
            )?,
        ));
        groups.push((
            VisionParameterOwner::Layer(layer),
            replicated(
                format!("{root}.norms"),
                [
                    (format!("{root}.norm1.weight"), vec![hidden]),
                    (format!("{root}.norm1.bias"), vec![hidden]),
                    (format!("{root}.norm2.weight"), vec![hidden]),
                    (format!("{root}.norm2.bias"), vec![hidden]),
                ],
            )?,
        ));
    }
    let projector = dim(config.projector_hidden_size)?;
    let shuffled = hidden
        .checked_mul(dim(config.merge_size * config.merge_size)?)
        .ok_or_else(|| invalid("Muse-Glimmer shuffled width overflow"))?;
    groups.push((
        VisionParameterOwner::Static(VisionStaticResource::Adapter),
        ParameterGroupSpec::new(
            VisionStaticResource::Adapter.logical_name(),
            ParameterRole::FeedForwardIntermediate,
            [
                member(
                    "model.vision_adapter.fc1.weight",
                    vec![projector, shuffled],
                    MemberSharding::Replicated,
                ),
                member(
                    "model.vision_adapter.fc2.weight",
                    vec![projector, projector],
                    MemberSharding::Replicated,
                ),
            ],
        )?,
    ));
    groups.push((
        VisionParameterOwner::Static(VisionStaticResource::Projection),
        ParameterGroupSpec::new(
            VisionStaticResource::Projection.logical_name(),
            ParameterRole::ColumnProjection,
            [member(
                "model.vision_projection.weight",
                vec![dim(config.language_hidden_size)?, projector],
                MemberSharding::Replicated,
            )],
        )?,
    ));
    groups.push((
        VisionParameterOwner::Static(VisionStaticResource::Norms),
        replicated(
            VisionStaticResource::Norms.logical_name(),
            [
                ("model.vision_tower.ln_pre.weight", vec![hidden]),
                ("model.vision_tower.ln_pre.bias", vec![hidden]),
                ("model.vision_tower.ln_post.weight", vec![hidden]),
                ("model.vision_tower.ln_post.bias", vec![hidden]),
            ],
        )?,
    ));
    eredu_runtime::expand_tagged_linear_format_parameter_groups(groups, |member| {
        standard_parallel_linear_format(member, config.linear_format_for(member.target()))
    })
}

/// Vision units whose owners need each pinned static module. Every vision
/// partition prepares request geometry through the pinned ingress in `begin`;
/// patch/position and static norm groups therefore remain available on those
/// owners. The large merge adapter and projection are needed only at the end.
pub(crate) fn vision_static_consumer_units(
    args: &DecoderConfig,
    target: &str,
) -> Option<std::ops::Range<usize>> {
    let layers = args.vision_config.as_ref()?.layer_count();
    if layers == 0 {
        return None;
    }
    VisionStaticResource::ALL
        .into_iter()
        .find(|resource| resource.owns_target(target))
        .map(|resource| resource.consumers(layers))
}

/// Declares only the pinned patch/position, merge, projection, and norm groups.
pub fn vision_static_parameter_groups(
    args: &DecoderConfig,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    Ok(vision_static_resource_groups(args)?
        .into_iter()
        .map(|(_, group)| group)
        .collect())
}

pub(crate) fn vision_static_resource_groups(
    args: &DecoderConfig,
) -> Result<Vec<(VisionStaticResource, ParameterGroupSpec)>, ParallelPlanError> {
    Ok(declared_vision_groups(args)?
        .into_iter()
        .filter_map(|(owner, group)| match owner {
            VisionParameterOwner::Static(resource) => Some((resource, group)),
            VisionParameterOwner::Layer(_) => None,
        })
        .collect())
}

/// Declares exactly one architecture-global Muse vision execution unit.
pub fn vision_layer_parameter_groups(
    args: &DecoderConfig,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let count = args
        .vision_config
        .as_ref()
        .map_or(0, |vision| vision.layer_count());
    if layer >= count {
        return Err(invalid(format!(
            "Muse-Glimmer vision layer {layer} is outside {count} layers"
        )));
    }
    Ok(declared_vision_groups(args)?
        .into_iter()
        .filter_map(|(owner, group)| match owner {
            VisionParameterOwner::Layer(index) if index == layer => Some(group),
            _ => None,
        })
        .collect())
}

fn group(
    name: impl Into<String>,
    role: ParameterRole,
    members: impl IntoIterator<Item = (impl Into<String>, Vec<usize>, MemberSharding)>,
) -> Result<ParameterGroupSpec, ParallelPlanError> {
    ParameterGroupSpec::new(
        name,
        role,
        members
            .into_iter()
            .map(|(name, shape, sharding)| member(name, shape, sharding)),
    )
}

fn replicated(
    name: impl Into<String>,
    members: impl IntoIterator<Item = (impl Into<String>, Vec<usize>)>,
) -> Result<ParameterGroupSpec, ParallelPlanError> {
    group(
        name,
        ParameterRole::Replicated,
        members
            .into_iter()
            .map(|(name, shape)| (name, shape, MemberSharding::Replicated)),
    )
}

fn member(
    name: impl Into<String>,
    shape: Vec<usize>,
    sharding: MemberSharding,
) -> ParameterMemberSpec {
    ParameterMemberSpec::new(name, shape, sharding)
}

const fn partitioned(axis: usize) -> MemberSharding {
    MemberSharding::Partitioned { axis }
}

fn dim(value: i32) -> Result<usize, ParallelPlanError> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid(format!("invalid Muse-Glimmer dimension {value}")))
}

fn invalid(message: impl Into<String>) -> ParallelPlanError {
    ParallelPlanError::InvalidGroup(message.into())
}

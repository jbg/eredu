//! Semantic parameter placement for Gemma 4 components.

use eredu_nn::{
    AttentionStateSource, AttentionValueSource, NeuralBackend, VocabularyParallelRange,
};
use eredu_runtime::{
    expand_linear_format_parameter_groups, module_parameter_group, ArchitecturePartition,
    LocalModelLayout, MemberSharding, ParallelPlanError, ParameterGroupSpec, ParameterMemberSpec,
    ParameterRole, PartitionOwnership, StateLayout, TensorPlacement,
};
use std::ops::Range;

use crate::linear_format::standard_parallel_linear_format;

use super::{
    state_layout, AudioLayer, AudioStatic, FamilyConfig, FeedForwardPolicy, ModalityProjector,
    ModelArgs, VisionLayer, VisionStatic,
};

/// Declares one replicated Gemma media execution unit.
pub fn media_unit_parameter_groups<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
    logical_name: impl Into<String>,
    unit: &impl eredu_nn::Parameterized<B::Tensor>,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    Ok(vec![module_parameter_group::<B::Tensor, _>(
        logical_name,
        ParameterRole::Replicated,
        unit,
        |_, _| Ok(MemberSharding::Replicated),
    )?])
}

/// Declares pinned Gemma vision modules as one atomic replicated group.
pub fn vision_static_parameter_groups<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
    modules: &VisionStatic<B>,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    media_unit_parameter_groups::<B>("model.vision_tower.static", modules)
}

/// Declares pinned Gemma audio modules as one atomic replicated group.
pub fn audio_static_parameter_groups<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
    modules: &AudioStatic<B>,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    media_unit_parameter_groups::<B>("model.audio_tower.static", modules)
}

/// Declares a Gemma modality projector as one atomic replicated group.
pub fn modality_projection_parameter_groups<
    B: NeuralBackend + eredu_nn::DistributedNeuralBackend,
>(
    logical_name: impl Into<String>,
    projector: &ModalityProjector<B>,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    media_unit_parameter_groups::<B>(logical_name, projector)
}

/// Declares one replicated Gemma vision layer.
pub fn vision_layer_parameter_groups<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
    layer: &VisionLayer<B>,
    index: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    media_unit_parameter_groups::<B>(format!("model.vision_tower.encoder.layers.{index}"), layer)
}

/// Declares one replicated Gemma audio layer.
pub fn audio_layer_parameter_groups<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
    layer: &AudioLayer<B>,
    index: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    media_unit_parameter_groups::<B>(format!("model.audio_tower.layers.{index}"), layer)
}

/// Complete planner-derived construction and state geometry for one Gemma 4 rank.
#[derive(Debug, Clone)]
pub struct LocalGeometry {
    text_blocks: Vec<ModelArgs>,
    embedding_range: VocabularyParallelRange,
    output_range: Option<VocabularyParallelRange>,
    per_layer_range: std::ops::Range<i32>,
    state_layout: StateLayout,
    vision_layers: usize,
    audio_layers: usize,
    architecture_fingerprint: String,
}

/// Exact TP-local and PP-local geometry for one Gemma 4 composite partition.
///
/// Optional media roots retain their own group-local ranges. Decoder state is
/// kept complete until the architecture partition selects the text slice, so
/// its architecture-global offset cannot be confused with a local ordinal.
#[derive(Debug, Clone)]
pub struct PartitionLocalGeometry {
    vision_units: Option<Range<usize>>,
    audio_units: Option<Range<usize>>,
    text_units: Range<usize>,
    text_blocks: Vec<ModelArgs>,
    embedding_range: VocabularyParallelRange,
    output_range: Option<VocabularyParallelRange>,
    per_layer_range: Range<i32>,
    complete_state_layout: StateLayout,
    static_roles: Vec<String>,
    architecture_fingerprint: String,
}

/// Validated architecture-owned handoff for one Gemma 4 composite partition.
#[derive(Debug, Clone)]
pub struct PartitionLocalFoundation {
    geometry: PartitionLocalGeometry,
    parameter_targets: Vec<String>,
}

impl PartitionLocalFoundation {
    /// Validates group ownership, state offset, boundary schema, and selected tasks together.
    pub fn from_partition(
        args: &FamilyConfig,
        partition: &ArchitecturePartition<PartitionLocalGeometry, super::TextBoundarySchema>,
    ) -> Result<Self, ParallelPlanError> {
        let geometry = partition.local_geometry();
        geometry.validate_for(args)?;
        let expected_groups = [
            geometry
                .vision_units()
                .map(|range| (super::VISION_EXECUTION_GROUP, range)),
            geometry
                .audio_units()
                .map(|range| (super::AUDIO_EXECUTION_GROUP, range)),
            Some((super::TEXT_EXECUTION_GROUP, geometry.text_units())),
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
                "Gemma 4 partition groups differ from family-local geometry",
            ));
        }
        let state = partition
            .state()
            .ok_or_else(|| invalid("Gemma 4 text partition has no selected state"))?;
        if state.global_layer_offset() != geometry.text_units.start
            || state.layout() != &geometry.local_state_layout()?
        {
            return Err(invalid(
                "Gemma 4 partition state differs from its text unit range",
            ));
        }
        if partition.boundary_schema()
            != &super::TextBoundarySchema::from_partition_args(&args.text, geometry)
        {
            return Err(invalid(
                "Gemma 4 partition boundary differs from local media-channel geometry",
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

    /// Architecture-global units owned from the optional audio root.
    pub fn audio_units(&self) -> Option<Range<usize>> {
        self.audio_units.clone()
    }

    /// Architecture-global text units owned by this pipeline coordinate.
    pub fn text_units(&self) -> Range<usize> {
        self.text_units.clone()
    }

    /// TP-local configuration for one owned architecture-global text unit.
    pub fn text_block(&self, global_unit: usize) -> Option<&ModelArgs> {
        self.text_units
            .contains(&global_unit)
            .then(|| &self.text_blocks[global_unit - self.text_units.start])
    }

    /// Complete TP-local state geometry before PP slicing.
    pub const fn complete_state_layout(&self) -> &StateLayout {
        &self.complete_state_layout
    }

    /// Exact local state slice and its architecture-global offset.
    pub fn local_state_layout(&self) -> Result<StateLayout, ParallelPlanError> {
        self.complete_state_layout
            .slice(self.text_units.clone())
            .map_err(|error| invalid(error.to_string()))
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

    /// TP-local decoder-wide media-channel ownership.
    pub fn per_layer_range(&self) -> Range<i32> {
        self.per_layer_range.clone()
    }

    pub(super) fn validate_for(&self, args: &FamilyConfig) -> Result<(), ParallelPlanError> {
        let text_count = args.text.num_hidden_layers();
        let vision_count = args
            .vision
            .as_ref()
            .map_or(0, |config| config.num_hidden_layers as usize);
        let audio_count = args
            .audio
            .as_ref()
            .map_or(0, |config| config.num_hidden_layers as usize);
        if self.text_units.is_empty()
            || self.text_units.end > text_count
            || self.text_blocks.len() != self.text_units.len()
            || self.architecture_fingerprint != args.architecture_fingerprint()
        {
            return Err(invalid(
                "partition-local Gemma 4 geometry belongs to a different model or text range",
            ));
        }
        validate_optional_range("vision", &self.vision_units, vision_count)?;
        validate_optional_range("audio", &self.audio_units, audio_count)?;
        for global in self.text_units.clone() {
            if self.text_block(global).is_none() {
                return Err(invalid("missing owned Gemma 4 text block"));
            }
        }
        let expected_roles = expected_partition_static_roles(args, self);
        if self.static_roles != expected_roles {
            return Err(invalid(format!(
                "partition-local Gemma 4 static roles {:?} differ from {:?}",
                self.static_roles, expected_roles
            )));
        }
        if self.complete_state_layout.len() != text_count {
            return Err(invalid(
                "partition-local Gemma 4 complete state length is invalid",
            ));
        }
        self.local_state_layout()?;
        Ok(())
    }
}

fn validate_optional_range(
    name: &str,
    range: &Option<Range<usize>>,
    count: usize,
) -> Result<(), ParallelPlanError> {
    if range
        .as_ref()
        .is_some_and(|range| range.is_empty() || range.end > count)
    {
        return Err(invalid(format!(
            "Gemma 4 {name} partition range {range:?} exceeds {count} units"
        )));
    }
    if count == 0 && range.is_some() {
        return Err(invalid(format!(
            "Gemma 4 partition owns an unconfigured {name} root"
        )));
    }
    Ok(())
}

fn expected_partition_static_roles(
    args: &FamilyConfig,
    geometry: &PartitionLocalGeometry,
) -> Vec<String> {
    let mut roles = Vec::new();
    if geometry
        .vision_units
        .as_ref()
        .is_some_and(|range| range.start == 0)
    {
        roles.extend(["vision".into(), "vision_projection".into()]);
    }
    if geometry
        .audio_units
        .as_ref()
        .is_some_and(|range| range.start == 0)
    {
        roles.extend(["audio".into(), "audio_projection".into()]);
    }
    if geometry.text_units.start == 0 {
        roles.extend([
            "embedding".into(),
            "per_layer_embedding".into(),
            "per_layer_projection".into(),
            "per_layer_norm".into(),
        ]);
    }
    if geometry.text_units.end == args.text.num_hidden_layers() {
        roles.push("norm".into());
        let output = if args.text.tie_word_embeddings {
            "embedding"
        } else {
            "output"
        };
        if !roles.iter().any(|role| role == output) {
            roles.push(output.into());
        }
    }
    roles
}

impl LocalGeometry {
    /// Returns one rank-local decoder configuration.
    pub fn text_block(&self, layer: usize) -> Option<&ModelArgs> {
        self.text_blocks.get(layer)
    }

    /// Returns rank-local decoder configurations in execution order.
    pub fn text_blocks(&self) -> &[ModelArgs] {
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

    /// Returns the local width of each per-layer media input.
    pub const fn per_layer_width(&self) -> i32 {
        self.per_layer_range.end - self.per_layer_range.start
    }

    /// Returns the decoder-media channels owned by every local text block.
    pub fn per_layer_range(&self) -> &std::ops::Range<i32> {
        &self.per_layer_range
    }

    /// Returns the authoritative rank-local decoder state layout.
    pub const fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    /// Returns the replicated vision-unit count owned by this rank.
    pub const fn vision_layers(&self) -> usize {
        self.vision_layers
    }

    /// Returns the replicated audio-unit count owned by this rank.
    pub const fn audio_layers(&self) -> usize {
        self.audio_layers
    }

    pub(super) fn validate_for(&self, args: &FamilyConfig) -> Result<(), ParallelPlanError> {
        let text_layers = args.text.num_hidden_layers();
        let vision_layers = args
            .vision
            .as_ref()
            .map_or(0, |config| config.num_hidden_layers as usize);
        let audio_layers = args
            .audio
            .as_ref()
            .map_or(0, |config| config.num_hidden_layers as usize);
        if self.architecture_fingerprint != args.architecture_fingerprint()
            || self.text_blocks.len() != text_layers
            || self.vision_layers != vision_layers
            || self.audio_layers != audio_layers
        {
            return Err(invalid(
                "rank-local Gemma 4 geometry belongs to a different family configuration",
            ));
        }
        self.embedding_range
            .validate_global_rows(args.text.vocab_size)
            .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
        match (args.text.tie_word_embeddings, &self.output_range) {
            (true, None) => {}
            (false, Some(range)) => range
                .validate_global_rows(args.text.vocab_size)
                .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?,
            (true, Some(_)) => return Err(invalid("tied Gemma 4 output has a separate range")),
            (false, None) => return Err(invalid("untied Gemma 4 output has no local range")),
        }
        let local_text =
            aggregate_local_text(&args.text, &self.text_blocks, self.per_layer_width())?;
        let expected = state_layout(&local_text).map_err(|error| invalid(error.to_string()))?;
        if expected != self.state_layout {
            return Err(invalid(
                "rank-local Gemma 4 state layout drifted from decoder geometry",
            ));
        }
        Ok(())
    }
}

fn aggregate_local_text(
    global: &ModelArgs,
    blocks: &[ModelArgs],
    per_layer_width: i32,
) -> Result<ModelArgs, ParallelPlanError> {
    let mut local = blocks
        .first()
        .cloned()
        .ok_or_else(|| invalid("Gemma 4 local geometry has no text blocks"))?;
    local.layer_schedule = eredu_core::LayerSchedule::new(
        blocks.len(),
        blocks
            .iter()
            .enumerate()
            .map(|(layer, args)| {
                args.layer_policy(layer)
                    .ok_or_else(|| invalid(format!("missing local Gemma 4 layer policy {layer}")))
            })
            .collect::<Result<Vec<_>, _>>()?,
    )
    .map_err(|error| invalid(error.to_string()))?;
    local.hidden_size_per_layer_input = per_layer_width;
    local.vocab_size = global.vocab_size;
    Ok(local)
}

fn vocabulary_range(
    layout: &LocalModelLayout,
    logical_name: &str,
    vocabulary: usize,
) -> Result<VocabularyParallelRange, ParallelPlanError> {
    let mut selected = None;
    for (target, tensor) in layout
        .tensors()
        .filter(|(_, tensor)| tensor.logical_name() == logical_name)
    {
        if tensor.global_shape().first().copied() != Some(vocabulary) {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "Gemma 4 vocabulary member {target} has global shape {:?}, expected {vocabulary} rows",
                tensor.global_shape()
            )));
        }
        let range = match tensor.placement() {
            TensorPlacement::Range {
                axis: 0,
                start,
                end,
            } => *start..*end,
            TensorPlacement::Replicated => 0..vocabulary,
            placement => {
                return Err(ParallelPlanError::InvalidTensor(format!(
                    "Gemma 4 vocabulary member {target} has non-row placement {placement:?}"
                )))
            }
        };
        if selected.as_ref().is_some_and(|current| current != &range) {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "Gemma 4 vocabulary group {logical_name} has inconsistent selections"
            )));
        }
        selected = Some(range);
    }
    let range = VocabularyParallelRange {
        global_vocabulary: vocabulary,
        local: selected.ok_or_else(|| {
            ParallelPlanError::InvalidTensor(format!(
                "missing local Gemma 4 vocabulary layout for {logical_name}"
            ))
        })?,
    };
    range
        .validate()
        .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
    Ok(range)
}

fn local_per_layer_range(
    args: &ModelArgs,
    layout: &LocalModelLayout,
    blocks: &[ModelArgs],
) -> Result<std::ops::Range<i32>, ParallelPlanError> {
    if args.hidden_size_per_layer_input == 0 {
        return Ok(0..0);
    }
    let mut selected = None;
    for layer in 0..args.num_hidden_layers() {
        let target = format!("model.language_model.layers.{layer}.per_layer_input_gate.weight");
        let tensor = layout.tensor(&target).ok_or_else(|| {
            ParallelPlanError::InvalidTensor(format!(
                "missing local Gemma 4 media-channel layout for {target}"
            ))
        })?;
        let range = match tensor.placement() {
            TensorPlacement::Range {
                axis: 0,
                start,
                end,
            } => {
                i32::try_from(*start).map_err(|_| invalid("media range exceeds i32"))?
                    ..i32::try_from(*end).map_err(|_| invalid("media range exceeds i32"))?
            }
            TensorPlacement::Replicated => 0..args.hidden_size_per_layer_input,
            placement => {
                return Err(ParallelPlanError::InvalidTensor(format!(
                    "Gemma 4 media-channel member {target} has invalid placement {placement:?}"
                )))
            }
        };
        if selected.as_ref().is_some_and(|current| current != &range) {
            return Err(ParallelPlanError::InvalidTensor(
                "Gemma 4 text blocks own inconsistent media-channel ranges".into(),
            ));
        }
        selected = Some(range);
    }
    let range = selected.ok_or_else(|| invalid("Gemma 4 has no local media-channel range"))?;
    let width = range.end - range.start;
    if blocks
        .iter()
        .any(|block| block.hidden_size_per_layer_input != width)
    {
        return Err(ParallelPlanError::InvalidTensor(
            "Gemma 4 static and block-local media widths drifted".into(),
        ));
    }
    Ok(range)
}

/// Derives all rank-local Gemma 4 construction geometry from one typed plan.
pub fn local_geometry(
    args: &FamilyConfig,
    layout: &LocalModelLayout,
) -> Result<LocalGeometry, ParallelPlanError> {
    args.validate()
        .map_err(|error| invalid(error.to_string()))?;
    let text_blocks = (0..args.text.num_hidden_layers())
        .map(|layer| local_block_args(&args.text, layer, layout))
        .collect::<Result<Vec<_>, _>>()?;
    let per_layer_range = local_per_layer_range(&args.text, layout, &text_blocks)?;
    let local_text = aggregate_local_text(
        &args.text,
        &text_blocks,
        per_layer_range.end - per_layer_range.start,
    )?;
    let state_layout = state_layout(&local_text).map_err(|error| invalid(error.to_string()))?;
    let vocabulary = dim(args.text.vocab_size)?;
    let embedding_range =
        vocabulary_range(layout, "model.language_model.embed_tokens", vocabulary)?;
    let output_range = if args.text.tie_word_embeddings {
        None
    } else {
        Some(vocabulary_range(layout, "lm_head", vocabulary)?)
    };
    let geometry = LocalGeometry {
        text_blocks,
        embedding_range,
        output_range,
        per_layer_range,
        state_layout,
        vision_layers: args
            .vision
            .as_ref()
            .map_or(0, |config| config.num_hidden_layers as usize),
        audio_layers: args
            .audio
            .as_ref()
            .map_or(0, |config| config.num_hidden_layers as usize),
        architecture_fingerprint: args.architecture_fingerprint(),
    };
    geometry.validate_for(args)?;
    Ok(geometry)
}

/// Derives exact TP-local and PP-local Gemma 4 composite geometry.
///
/// Group ranges use the family execution-group identities. Missing optional
/// roots are represented by omission; the text decoder must be present on
/// every admitted pipeline coordinate.
pub fn partition_local_geometry(
    args: &FamilyConfig,
    layout: &LocalModelLayout,
    group_ranges: impl IntoIterator<Item = (impl AsRef<str>, Range<usize>)>,
    ownership: &PartitionOwnership,
) -> Result<PartitionLocalGeometry, ParallelPlanError> {
    if args
        .text
        .layer_schedule
        .iter()
        .any(|policy| policy.feed_forward == FeedForwardPolicy::DenseWithSparseMoe)
    {
        return Err(invalid(
            "partition-local Gemma 4 foundation does not admit routed text blocks",
        ));
    }
    partition_local_geometry_impl(args, layout, group_ranges, ownership)
}

pub(crate) fn routed_partition_local_geometry(
    args: &FamilyConfig,
    layout: &LocalModelLayout,
    group_ranges: impl IntoIterator<Item = (impl AsRef<str>, Range<usize>)>,
    ownership: &PartitionOwnership,
) -> Result<PartitionLocalGeometry, ParallelPlanError> {
    if !args
        .text
        .layer_schedule
        .iter()
        .any(|policy| policy.feed_forward == FeedForwardPolicy::DenseWithSparseMoe)
    {
        return Err(invalid(
            "routed Gemma 4 partition has no sparse text blocks",
        ));
    }
    partition_local_geometry_impl(args, layout, group_ranges, ownership)
}

fn partition_local_geometry_impl(
    args: &FamilyConfig,
    layout: &LocalModelLayout,
    group_ranges: impl IntoIterator<Item = (impl AsRef<str>, Range<usize>)>,
    ownership: &PartitionOwnership,
) -> Result<PartitionLocalGeometry, ParallelPlanError> {
    let mut vision_units = None;
    let mut audio_units = None;
    let mut text_units = None;
    for (group, range) in group_ranges {
        if range.is_empty() {
            return Err(invalid("Gemma 4 partition group range cannot be empty"));
        }
        let slot = match group.as_ref() {
            super::VISION_EXECUTION_GROUP => &mut vision_units,
            super::AUDIO_EXECUTION_GROUP => &mut audio_units,
            super::TEXT_EXECUTION_GROUP => &mut text_units,
            other => {
                return Err(invalid(format!(
                    "unknown Gemma 4 partition group {other:?}"
                )))
            }
        };
        if slot.replace(range).is_some() {
            return Err(invalid("duplicate Gemma 4 partition group"));
        }
    }
    let text_units = text_units
        .ok_or_else(|| invalid("Gemma 4 partition must own a non-empty text decoder range"))?;
    let complete = local_geometry(args, layout)?;
    let text_blocks = text_units
        .clone()
        .map(|global| {
            complete
                .text_block(global)
                .cloned()
                .ok_or_else(|| invalid(format!("Gemma 4 has no local text block {global}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let geometry = PartitionLocalGeometry {
        vision_units,
        audio_units,
        text_units,
        text_blocks,
        embedding_range: complete.embedding_range.clone(),
        output_range: complete.output_range.clone(),
        per_layer_range: complete.per_layer_range.clone(),
        complete_state_layout: complete.state_layout.clone(),
        static_roles: ownership.static_roles().to_vec(),
        architecture_fingerprint: complete.architecture_fingerprint.clone(),
    };
    geometry.validate_for(args)?;
    Ok(geometry)
}

/// Derives one Gemma 4 decoder block's rank-local construction geometry.
pub fn local_block_args(
    args: &ModelArgs,
    layer: usize,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<ModelArgs, ParallelPlanError> {
    let policy = args
        .layer_policy(layer)
        .ok_or_else(|| invalid(format!("Gemma 4 layer {layer} is out of range")))?;
    let root = format!("model.language_model.layers.{layer}");
    let tensor_at = |layer: usize, suffix: &str| {
        let root = format!("model.language_model.layers.{layer}");
        layout
            .tensor(&format!("{root}.{suffix}.weight"))
            .or_else(|| layout.tensor(&format!("{root}.{suffix}")))
    };
    let tensor = |suffix: &str| {
        tensor_at(layer, suffix).ok_or_else(|| {
            ParallelPlanError::InvalidTensor(format!("missing local layout for {root}.{suffix}"))
        })
    };
    let query_width = local_axis(tensor("self_attn.q_proj")?, 0, "query width")?;
    let head_dim = i32::try_from(policy.head_dim.get()).map_err(|_| {
        ParallelPlanError::InvalidTensor("Gemma 4 head dimension exceeds i32".into())
    })?;
    if query_width % head_dim != 0 {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "local Gemma 4 query width {query_width} splits head dimension {head_dim}"
        )));
    }
    let kv_tensor = if policy.key_value.owns_state() {
        Some(tensor("self_attn.k_proj")?)
    } else {
        (0..layer).rev().find_map(|owner| {
            args.layer_policy(owner)
                .filter(|candidate| {
                    candidate.attention == policy.attention && candidate.key_value.publishes_state()
                })
                .and_then(|_| tensor_at(owner, "self_attn.k_proj"))
        })
    }
    .ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!(
            "Gemma 4 shared-KV layer {layer} has no rank-local publisher"
        ))
    })?;
    let kv_width = local_axis(kv_tensor, 0, "key/value width")?;
    if kv_width % head_dim != 0 {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "local Gemma 4 key/value width {kv_width} splits head dimension {head_dim}"
        )));
    }
    let dense_width = local_axis(tensor("mlp.gate_proj")?, 0, "dense intermediate width")?;
    let mut local = args.clone();
    local.num_attention_heads = query_width / head_dim;
    let mut layers = args.layer_schedule.iter().copied().collect::<Vec<_>>();
    layers[layer].num_key_value_heads =
        std::num::NonZeroU32::new(u32::try_from(kv_width / head_dim).map_err(|_| {
            ParallelPlanError::InvalidTensor("local Gemma 4 KV heads exceed u32".into())
        })?)
        .ok_or_else(|| {
            ParallelPlanError::InvalidTensor("local Gemma 4 KV heads are zero".into())
        })?;
    layers[layer].intermediate_size =
        std::num::NonZeroU32::new(u32::try_from(dense_width).map_err(|_| {
            ParallelPlanError::InvalidTensor("local Gemma 4 dense width exceeds u32".into())
        })?)
        .ok_or_else(|| {
            ParallelPlanError::InvalidTensor("local Gemma 4 dense width is zero".into())
        })?;
    local.layer_schedule = eredu_core::LayerSchedule::new(layers.len(), layers)
        .map_err(|error| invalid(error.to_string()))?;
    if policy.feed_forward == FeedForwardPolicy::DenseWithSparseMoe {
        let fused = local_axis(
            tensor("experts.switch_glu.gate_up_proj")?,
            1,
            "expert gate/up width",
        )?;
        if fused % 2 != 0 {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "local Gemma 4 expert gate/up width {fused} is not even"
            )));
        }
        local.moe_intermediate_size = Some(fused / 2);
    }
    if args.hidden_size_per_layer_input > 0 {
        local.hidden_size_per_layer_input =
            local_axis(tensor("per_layer_input_gate")?, 0, "per-layer media width")?;
    }
    Ok(local)
}

fn local_axis(
    tensor: &eredu_runtime::LocalTensorLayout,
    axis: usize,
    label: &str,
) -> Result<i32, ParallelPlanError> {
    let value = *tensor.local_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("local Gemma 4 {label} tensor has no axis {axis}"))
    })?;
    let global = *tensor.global_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!(
            "global Gemma 4 {label} tensor has no axis {axis}"
        ))
    })?;
    if value == 0 || value > global {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "local Gemma 4 {label} width {value} is invalid for global width {global}"
        )));
    }
    i32::try_from(value)
        .map_err(|_| ParallelPlanError::InvalidTensor(format!("local Gemma 4 {label} exceeds i32")))
}

/// Declares pinned embeddings, final normalization, and vocabulary output.
pub fn static_parameter_groups(
    args: &ModelArgs,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let hidden = dim(args.hidden_size)?;
    let vocab = dim(args.vocab_size)?;
    let mut groups = vec![group(
        "model.language_model.embed_tokens",
        ParameterRole::Vocabulary,
        [(
            "model.language_model.embed_tokens.weight".into(),
            vec![vocab, hidden],
            MemberSharding::Balanced { axis: 0 },
        )],
    )?];
    if args.hidden_size_per_layer_input > 0 {
        let per_layer = dim(args.hidden_size_per_layer_input)?;
        let combined = per_layer
            .checked_mul(args.num_hidden_layers())
            .ok_or_else(|| invalid("Gemma per-layer embedding width overflow"))?;
        groups.push(replicated(
            "model.language_model.per_layer_embedding",
            [(
                "model.language_model.embed_tokens_per_layer.weight".into(),
                vec![
                    dim(args.vocab_size_per_layer_input.unwrap_or(args.vocab_size))?,
                    combined,
                ],
            )],
        )?);
        groups.push(replicated(
            "model.language_model.per_layer_projection",
            [(
                "model.language_model.per_layer_model_projection.weight".into(),
                vec![combined, hidden],
            )],
        )?);
        groups.push(replicated(
            "model.language_model.per_layer_projection_norm",
            [(
                "model.language_model.per_layer_projection_norm.weight".into(),
                vec![per_layer],
            )],
        )?);
    }
    groups.push(replicated(
        "model.language_model.norm",
        [("model.language_model.norm.weight".into(), vec![hidden])],
    )?);
    if !args.tie_word_embeddings {
        groups.push(group(
            "lm_head",
            ParameterRole::Vocabulary,
            [(
                "lm_head.weight".into(),
                vec![vocab, hidden],
                MemberSharding::Balanced { axis: 0 },
            )],
        )?);
    }
    expand_linear_format_parameter_groups(groups, |member| {
        standard_parallel_linear_format(member, args.linear_format_for(member.target()))
    })
}

/// Declares one decoder layer's head, MLP, expert, media, and replicated groups.
pub fn layer_parameter_groups(
    args: &ModelArgs,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let policy = args
        .layer_policy(layer)
        .ok_or_else(|| invalid(format!("Gemma 4 layer {layer} is out of range")))?;
    let root = format!("model.language_model.layers.{layer}");
    let attention = format!("{root}.self_attn");
    let hidden = dim(args.hidden_size)?;
    let head_dim = dim(policy.head_dim.get() as i32)?;
    let query_heads = dim(args.num_attention_heads)?;
    let kv_heads = dim(policy.num_key_value_heads.get() as i32)?;
    let query_width = query_heads
        .checked_mul(head_dim)
        .ok_or_else(|| invalid("Gemma query width overflow"))?;
    let kv_width = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| invalid("Gemma KV width overflow"))?;
    let mut query_members = vec![
        member(
            format!("{attention}.q_proj.weight"),
            vec![query_width, hidden],
            MemberSharding::Partitioned { axis: 0 },
        ),
        member(
            format!("{attention}.o_proj.weight"),
            vec![hidden, query_width],
            MemberSharding::Partitioned { axis: 1 },
        ),
    ];
    if args.attention_bias {
        query_members.extend([
            member(
                format!("{attention}.q_proj.bias"),
                vec![query_width],
                MemberSharding::Partitioned { axis: 0 },
            ),
            member(
                format!("{attention}.o_proj.bias"),
                vec![hidden],
                MemberSharding::Replicated,
            ),
        ]);
    }
    let mut groups = vec![ParameterGroupSpec::partitioned(
        format!("{attention}.query_heads"),
        ParameterRole::AttentionHeads,
        query_heads,
        query_members,
    )?];
    if policy.key_value != AttentionStateSource::Shared {
        let mut kv_members = vec![member(
            format!("{attention}.k_proj.weight"),
            vec![kv_width, hidden],
            MemberSharding::Partitioned { axis: 0 },
        )];
        if policy.key_value.value() == Some(AttentionValueSource::Projected) {
            kv_members.push(member(
                format!("{attention}.v_proj.weight"),
                vec![kv_width, hidden],
                MemberSharding::Partitioned { axis: 0 },
            ));
        }
        if args.attention_bias {
            kv_members.push(member(
                format!("{attention}.k_proj.bias"),
                vec![kv_width],
                MemberSharding::Partitioned { axis: 0 },
            ));
            if policy.key_value.value() == Some(AttentionValueSource::Projected) {
                kv_members.push(member(
                    format!("{attention}.v_proj.bias"),
                    vec![kv_width],
                    MemberSharding::Partitioned { axis: 0 },
                ));
            }
        }
        groups.push(ParameterGroupSpec::partitioned(
            format!("{attention}.key_value_heads"),
            ParameterRole::AttentionHeads,
            kv_heads,
            kv_members,
        )?);
    }
    groups.push(replicated(
        format!("{attention}.head_norms"),
        std::iter::once((format!("{attention}.q_norm.weight"), vec![head_dim])).chain(
            (policy.key_value != AttentionStateSource::Shared)
                .then(|| (format!("{attention}.k_norm.weight"), vec![head_dim])),
        ),
    )?);
    let intermediate = dim(policy.intermediate_size.get() as i32)?;
    groups.push(ParameterGroupSpec::partitioned(
        format!("{root}.mlp.intermediate"),
        ParameterRole::FeedForwardIntermediate,
        intermediate,
        [
            member(
                format!("{root}.mlp.gate_proj.weight"),
                vec![intermediate, hidden],
                MemberSharding::Partitioned { axis: 0 },
            ),
            member(
                format!("{root}.mlp.up_proj.weight"),
                vec![intermediate, hidden],
                MemberSharding::Partitioned { axis: 0 },
            ),
            member(
                format!("{root}.mlp.down_proj.weight"),
                vec![hidden, intermediate],
                MemberSharding::Partitioned { axis: 1 },
            ),
        ],
    )?);
    if policy.feed_forward == FeedForwardPolicy::DenseWithSparseMoe {
        let experts = dim(args
            .num_experts
            .ok_or_else(|| invalid("Gemma sparse layer has no expert count"))?)?;
        let expert_width = dim(args
            .moe_intermediate_size
            .ok_or_else(|| invalid("Gemma sparse layer has no expert width"))?)?;
        let expert_root = format!("{root}.experts.switch_glu");
        groups.push(ParameterGroupSpec::partitioned(
            format!("{expert_root}.intermediate"),
            ParameterRole::ExpertIntermediate,
            expert_width,
            [
                member(
                    format!("{expert_root}.gate_up_proj"),
                    vec![experts, 2 * expert_width, hidden],
                    MemberSharding::PartitionedSegments {
                        axis: 1,
                        segments: vec![0..expert_width, expert_width..2 * expert_width],
                    },
                ),
                member(
                    format!("{expert_root}.down_proj"),
                    vec![experts, hidden, expert_width],
                    MemberSharding::Partitioned { axis: 2 },
                ),
            ],
        )?);
        groups.push(replicated(
            format!("{root}.router"),
            [
                (format!("{root}.router.proj.weight"), vec![experts, hidden]),
                (format!("{root}.router.scale"), vec![hidden]),
                (format!("{root}.router.per_expert_scale"), vec![experts]),
            ],
        )?);
    }
    if args.hidden_size_per_layer_input > 0 {
        let media = dim(args.hidden_size_per_layer_input)?;
        groups.push(ParameterGroupSpec::partitioned(
            format!("{root}.media_channels"),
            ParameterRole::Channels,
            media,
            [
                member(
                    format!("{root}.per_layer_input_gate.weight"),
                    vec![media, hidden],
                    MemberSharding::Partitioned { axis: 0 },
                ),
                member(
                    format!("{root}.per_layer_projection.weight"),
                    vec![hidden, media],
                    MemberSharding::Partitioned { axis: 1 },
                ),
            ],
        )?);
    }
    let mut replicated_members = vec![
        (format!("{root}.input_layernorm.weight"), vec![hidden]),
        (
            format!("{root}.post_attention_layernorm.weight"),
            vec![hidden],
        ),
        (
            format!("{root}.pre_feedforward_layernorm.weight"),
            vec![hidden],
        ),
        (
            format!("{root}.post_feedforward_layernorm.weight"),
            vec![hidden],
        ),
        (format!("{root}.layer_scalar"), vec![1]),
    ];
    if policy.feed_forward == FeedForwardPolicy::DenseWithSparseMoe {
        replicated_members.extend([
            (
                format!("{root}.post_feedforward_layernorm_1.weight"),
                vec![hidden],
            ),
            (
                format!("{root}.pre_feedforward_layernorm_2.weight"),
                vec![hidden],
            ),
            (
                format!("{root}.post_feedforward_layernorm_2.weight"),
                vec![hidden],
            ),
        ]);
    }
    if args.hidden_size_per_layer_input > 0 {
        replicated_members.push((
            format!("{root}.post_per_layer_input_norm.weight"),
            vec![hidden],
        ));
    }
    groups.push(replicated(
        format!("{root}.replicated"),
        replicated_members,
    )?);
    expand_linear_format_parameter_groups(groups, |member| {
        standard_parallel_linear_format(member, args.linear_format_for(member.target()))
    })
}

fn group(
    name: impl Into<String>,
    role: ParameterRole,
    members: impl IntoIterator<Item = (String, Vec<usize>, MemberSharding)>,
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
    members: impl IntoIterator<Item = (String, Vec<usize>)>,
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

fn dim(value: i32) -> Result<usize, ParallelPlanError> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid(format!("invalid Gemma dimension {value}")))
}

fn invalid(message: impl Into<String>) -> ParallelPlanError {
    ParallelPlanError::InvalidGroup(message.into())
}

#[cfg(test)]
mod tests {
    use eredu_checkpoint::AffineQuantization;
    use eredu_runtime::{LocalModelLayout, LocalTensorLayout, TensorPlacement};

    use super::*;

    fn args() -> ModelArgs {
        ModelArgs::from_hf_json(
            br#"{
              "model_type":"gemma4","hidden_size":16,"num_hidden_layers":2,
              "intermediate_size":32,"num_attention_heads":4,"num_key_value_heads":2,
              "head_dim":4,"rms_norm_eps":0.000001,"vocab_size":64,
              "max_position_embeddings":128,"layer_types":["full_attention","full_attention"],
              "num_kv_shared_layers":1,"enable_moe_block":true,"num_experts":4,
              "top_k_experts":2,"moe_intermediate_size":8
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn shared_layer_omits_kv_group_and_sparse_group_is_semantic() {
        let owner = layer_parameter_groups(&args(), 0).unwrap();
        assert!(owner
            .iter()
            .any(|group| group.logical_name().ends_with("key_value_heads")));
        assert!(owner
            .iter()
            .any(|group| group.role() == ParameterRole::ExpertIntermediate));
        let shared = layer_parameter_groups(&args(), 1).unwrap();
        assert!(!shared
            .iter()
            .any(|group| group.logical_name().ends_with("key_value_heads")));
    }

    #[test]
    fn affine_layer_plan_publishes_weight_companions() {
        let mut args = args();
        args.weight_quantization = Some(AffineQuantization::new(16, 4).unwrap().into());
        args.quantized_weights = Some(std::collections::HashSet::from([
            "model.language_model.layers.0.self_attn.q_proj.weight".to_owned(),
            "model.language_model.layers.0.experts.switch_glu.gate_up_proj".to_owned(),
        ]));
        let targets = layer_parameter_groups(&args, 0)
            .unwrap()
            .into_iter()
            .flat_map(|group| group.members().to_vec())
            .map(|member| member.target().to_owned())
            .collect::<Vec<_>>();
        assert!(targets
            .iter()
            .any(|name| name == "model.language_model.layers.0.self_attn.q_proj.scales"));
        assert!(targets.iter().any(|name| {
            name == "model.language_model.layers.0.experts.switch_glu.gate_up_proj_scales"
        }));
    }

    fn local_tensor(shape: Vec<usize>) -> LocalTensorLayout {
        LocalTensorLayout::new(
            "test",
            ParameterRole::AttentionHeads,
            shape.clone(),
            shape,
            TensorPlacement::Local,
            None,
            None,
            false,
        )
    }

    fn family() -> FamilyConfig {
        FamilyConfig::from_hf_json(
            br#"{
              "model_type":"gemma4_unified","tie_word_embeddings":false,
              "image_token_id":60,"audio_token_id":61,
              "text_config":{
                "hidden_size":16,"num_hidden_layers":2,"intermediate_size":32,
                "num_attention_heads":2,"num_key_value_heads":2,"head_dim":8,
                "rms_norm_eps":0.000001,"vocab_size":64,"max_position_embeddings":128,
                "hidden_size_per_layer_input":4,"vocab_size_per_layer_input":64,
                "layer_types":["full_attention","full_attention"]
              },
              "vision_config":{
                "hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,
                "num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,
                "patch_size":4,"pooling_kernel_size":2,"position_embedding_size":16,
                "rms_norm_eps":0.000001
              },
              "audio_config":{
                "hidden_size":16,"num_hidden_layers":1,"num_attention_heads":2,
                "output_proj_dims":8,"conv_kernel_size":3,"attention_chunk_size":4,
                "attention_context_left":5,"attention_context_right":0,
                "attention_invalid_logits_value":-1000000000.0,"attention_logit_cap":50.0,
                "residual_weight":0.5,"rms_norm_eps":0.000001,
                "subsampling_conv_channels":[4,8]
              }
            }"#,
        )
        .unwrap()
    }

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

    fn family_layout() -> LocalModelLayout {
        let mut layout = LocalModelLayout::default();
        for (target, logical) in [
            (
                "model.language_model.embed_tokens.weight",
                "model.language_model.embed_tokens",
            ),
            ("lm_head.weight", "lm_head"),
        ] {
            insert(
                &mut layout,
                target,
                logical,
                vec![64, 16],
                vec![32, 16],
                TensorPlacement::Range {
                    axis: 0,
                    start: 0,
                    end: 32,
                },
            );
        }
        for layer in 0..2 {
            let root = format!("model.language_model.layers.{layer}");
            for suffix in ["self_attn.q_proj.weight", "self_attn.k_proj.weight"] {
                insert(
                    &mut layout,
                    &format!("{root}.{suffix}"),
                    &format!("{root}.heads"),
                    vec![16, 16],
                    vec![8, 16],
                    TensorPlacement::Range {
                        axis: 0,
                        start: 0,
                        end: 8,
                    },
                );
            }
            insert(
                &mut layout,
                &format!("{root}.mlp.gate_proj.weight"),
                &format!("{root}.mlp.intermediate"),
                vec![32, 16],
                vec![16, 16],
                TensorPlacement::Range {
                    axis: 0,
                    start: 0,
                    end: 16,
                },
            );
            insert(
                &mut layout,
                &format!("{root}.per_layer_input_gate.weight"),
                &format!("{root}.media_channels"),
                vec![4, 16],
                vec![2, 16],
                TensorPlacement::Range {
                    axis: 0,
                    start: 0,
                    end: 2,
                },
            );
        }
        layout
    }

    #[test]
    fn local_shared_kv_geometry_is_derived_from_its_publisher() {
        let args = args();
        let mut layout = LocalModelLayout::default();
        layout.insert(
            "model.language_model.layers.0.self_attn.k_proj.weight".into(),
            local_tensor(vec![4, 16]),
        );
        layout.insert(
            "model.language_model.layers.1.self_attn.q_proj.weight".into(),
            local_tensor(vec![8, 16]),
        );
        layout.insert(
            "model.language_model.layers.1.mlp.gate_proj.weight".into(),
            local_tensor(vec![16, 16]),
        );
        layout.insert(
            "model.language_model.layers.1.experts.switch_glu.gate_up_proj".into(),
            local_tensor(vec![4, 8, 16]),
        );
        let local = local_block_args(&args, 1, &layout).unwrap();
        let policy = local.layer_policy(1).unwrap();
        assert_eq!(local.num_attention_heads, 2);
        assert_eq!(policy.num_key_value_heads.get(), 1);
        assert_eq!(policy.intermediate_size.get(), 16);
        assert_eq!(local.moe_intermediate_size, Some(4));
    }

    #[test]
    fn partition_geometry_owns_optional_roots_text_state_and_static_roles_exactly() {
        let family = family();
        let layout = family_layout();
        let first = PartitionOwnership::new(
            true,
            false,
            [
                "vision",
                "vision_projection",
                "audio",
                "audio_projection",
                "embedding",
                "per_layer_embedding",
                "per_layer_projection",
                "per_layer_norm",
            ],
        )
        .unwrap();
        let first = partition_local_geometry(
            &family,
            &layout,
            [
                (super::super::VISION_EXECUTION_GROUP, 0..1),
                (super::super::AUDIO_EXECUTION_GROUP, 0..1),
                (super::super::TEXT_EXECUTION_GROUP, 0..1),
            ],
            &first,
        )
        .unwrap();
        assert_eq!(first.vision_units(), Some(0..1));
        assert_eq!(first.audio_units(), Some(0..1));
        assert_eq!(first.text_units(), 0..1);
        assert_eq!(first.local_state_layout().unwrap().len(), 1);
        assert!(first.text_block(0).is_some());
        assert!(first.text_block(1).is_none());

        let last = PartitionOwnership::new(false, true, ["norm", "output"]).unwrap();
        let last = partition_local_geometry(
            &family,
            &layout,
            [(super::super::TEXT_EXECUTION_GROUP, 1..2)],
            &last,
        )
        .unwrap();
        assert_eq!(last.text_units(), 1..2);
        assert_eq!(last.static_roles(), ["norm", "output"]);
        assert_eq!(last.local_state_layout().unwrap().len(), 1);
        assert_eq!(last.complete_state_layout().len(), 2);
    }

    #[test]
    fn partition_geometry_rejects_bad_ranges_roles_and_task_drift_before_construction() {
        let family = family();
        let layout = family_layout();
        let last = PartitionOwnership::new(false, true, ["norm", "output"]).unwrap();
        assert!(partition_local_geometry(
            &family,
            &layout,
            [(super::super::TEXT_EXECUTION_GROUP, 2..3)],
            &last,
        )
        .is_err());
        assert!(partition_local_geometry(
            &family,
            &layout,
            [(super::super::VISION_EXECUTION_GROUP, 0..1)],
            &last,
        )
        .is_err());
        let wrong = PartitionOwnership::new(false, true, ["embedding"]).unwrap();
        assert!(partition_local_geometry(
            &family,
            &layout,
            [(super::super::TEXT_EXECUTION_GROUP, 1..2)],
            &wrong,
        )
        .is_err());
        let routed_family = FamilyConfig {
            model_type: args().model_type.clone(),
            text: args(),
            vision: None,
            image_token_id: None,
            video_token_id: None,
            audio: None,
            audio_token_id: None,
        };
        let routed_error = partition_local_geometry(
            &routed_family,
            &LocalModelLayout::default(),
            [(super::super::TEXT_EXECUTION_GROUP, 0..1)],
            &PartitionOwnership::new(true, false, ["embedding"]).unwrap(),
        )
        .unwrap_err();
        assert!(routed_error.to_string().contains("does not admit routed"));
    }

    #[test]
    fn family_geometry_owns_text_vocab_state_and_replicated_towers() {
        let family = family();
        let geometry = local_geometry(&family, &family_layout()).unwrap();
        assert_eq!(geometry.text_blocks().len(), 2);
        assert_eq!(geometry.text_block(0).unwrap().num_attention_heads, 1);
        assert_eq!(
            geometry
                .text_block(0)
                .unwrap()
                .layer_policy(0)
                .unwrap()
                .num_key_value_heads
                .get(),
            1
        );
        assert_eq!(geometry.embedding_range().local, 0..32);
        assert_eq!(geometry.output_range().unwrap().local, 0..32);
        assert_eq!(geometry.vision_layers(), 1);
        assert_eq!(geometry.audio_layers(), 1);
        assert_eq!(geometry.per_layer_range(), &(0..2));
        assert_ne!(
            geometry.state_layout(),
            &state_layout(&family.text).unwrap()
        );
        geometry.validate_for(&family).unwrap();
    }

    #[test]
    fn family_geometry_rejects_head_and_vocabulary_selection_drift() {
        let family = family();
        let mut layout = family_layout();
        insert(
            &mut layout,
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.heads",
            vec![16, 16],
            vec![7, 16],
            TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 7,
            },
        );
        assert!(local_geometry(&family, &layout)
            .unwrap_err()
            .to_string()
            .contains("splits head dimension"));

        let mut layout = family_layout();
        insert(
            &mut layout,
            "lm_head.scales",
            "lm_head",
            vec![64, 1],
            vec![32, 1],
            TensorPlacement::Range {
                axis: 0,
                start: 32,
                end: 64,
            },
        );
        assert!(local_geometry(&family, &layout)
            .unwrap_err()
            .to_string()
            .contains("inconsistent selections"));
    }
}

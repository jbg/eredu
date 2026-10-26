//! Semantic parameter placement for Gemma 4 components.

use eredu_nn::{AttentionStateSource, AttentionValueSource, VocabularyParallelRange};
use eredu_runtime::{
    expand_linear_format_parameter_groups, ArchitecturePartition, LocalModelLayout, MemberSharding,
    ParallelPlanError, ParameterGroupSpec, ParameterMemberSpec, ParameterRole, PartitionOwnership,
    StateLayout, TensorPlacement,
};
use std::ops::Range;

use crate::linear_format::standard_parallel_linear_format;

use super::{state_layout, FamilyConfig, FeedForwardPolicy, ModelArgs};

/// Declares pinned Gemma vision modules as one atomic replicated group.
pub fn vision_static_parameter_groups(
    config: &super::VisionConfig,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    super::vision::parameters::static_groups(config, "model.vision_tower")
}

/// Declares one independently leased Gemma vision layer.
pub fn vision_layer_parameter_groups(
    config: &super::VisionConfig,
    index: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    super::vision::parameters::layer_groups(config, "model.vision_tower", index)
}

fn audio_parameter_groups(
    config: &super::AudioConfig,
    text_hidden: i32,
    logical_name: &str,
    layers: Range<usize>,
    include: impl Fn(&str) -> bool,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    config
        .validate()
        .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))?;
    let parameters = super::audio::parameters::declaration(config, text_hidden as usize, layers)
        .map_err(ParallelPlanError::InvalidGroup)?;
    let members: Vec<_> = parameters
        .into_iter()
        .filter(|p| include(&p.name) && (!p.optional || config.output_projection_bias))
        .map(|p| ParameterMemberSpec::new(p.name, p.shape, MemberSharding::Replicated))
        .collect();
    expand_linear_format_parameter_groups(
        vec![ParameterGroupSpec::new(
            logical_name,
            ParameterRole::Replicated,
            members,
        )?],
        |member| {
            let format = if member.global_shape().len() == 2 && member.target().ends_with(".weight")
            {
                config.linear_format_for(member.target(), member.global_shape()[1] as i32)
            } else {
                eredu_checkpoint::LinearFormat::Dense
            };
            standard_parallel_linear_format(member, format)
        },
    )
}

/// Declares pinned audio subsampling and output projection storage.
pub fn audio_static_parameter_groups(
    config: &super::AudioConfig,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    audio_parameter_groups(
        config,
        config.hidden_size,
        "model.audio_tower.static",
        0..0,
        |name| {
            name.starts_with("model.audio_tower.") && !name.starts_with("model.audio_tower.layers.")
        },
    )
}

/// Declares one independently leased Gemma audio layer.
pub fn audio_layer_parameter_groups(
    config: &super::AudioConfig,
    index: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    if index >= config.num_hidden_layers as usize {
        return Err(ParallelPlanError::InvalidGroup(
            "Gemma audio layer outside declared stack".into(),
        ));
    }
    let prefix = format!("model.audio_tower.layers.{index}");
    audio_parameter_groups(
        config,
        config.hidden_size,
        &prefix,
        index..index + 1,
        |name| name.starts_with(&format!("{prefix}.")),
    )
}

/// Declares one media-to-decoder projection with its admitted physical encoding.
pub fn modality_projection_parameter_groups(
    logical_name: impl Into<String>,
    component: &str,
    input: i32,
    output: i32,
    format: eredu_checkpoint::LinearFormat,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    if input <= 0 || output <= 0 {
        return Err(ParallelPlanError::InvalidGroup(
            "Gemma projector dimensions must be positive".into(),
        ));
    }
    expand_linear_format_parameter_groups(
        vec![ParameterGroupSpec::new(
            logical_name,
            ParameterRole::Replicated,
            vec![ParameterMemberSpec::new(
                format!("model.{component}.embedding_projection.weight"),
                vec![output as usize, input as usize],
                MemberSharding::Replicated,
            )],
        )?],
        |member| standard_parallel_linear_format(member, format),
    )
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
    for (group, units, count) in [
        (
            0,
            geometry.vision_units.as_ref(),
            args.vision
                .as_ref()
                .map_or(0, |v| v.num_hidden_layers as usize),
        ),
        (
            1,
            geometry.audio_units.as_ref(),
            args.audio
                .as_ref()
                .map_or(0, |a| a.num_hidden_layers as usize),
        ),
    ] {
        let Some(units) = units else {
            continue;
        };
        let transport = super::pipeline::media_transport(args, group);
        for role in transport
            .first_owner_static_roles
            .into_iter()
            .filter(|_| units.start == 0)
            .chain(
                transport
                    .last_owner_static_roles
                    .into_iter()
                    .filter(|_| units.end == count),
            )
        {
            if !roles.contains(&role) {
                roles.push(role);
            }
        }
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
    let complete_state_layout = super::pipeline::partition_state_layout(
        &args.text,
        &complete.state_layout,
        text_units.clone(),
    )
    .map_err(|error| invalid(error.to_string()))?;
    let geometry = PartitionLocalGeometry {
        vision_units,
        audio_units,
        text_units,
        text_blocks,
        embedding_range: complete.embedding_range.clone(),
        output_range: complete.output_range.clone(),
        per_layer_range: complete.per_layer_range.clone(),
        complete_state_layout,
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
    local_block_args_at(args, layer, layout, "model.language_model.layers")
}

/// Derives shared block geometry under the owning architecture's logical root.
pub(crate) fn local_block_args_at(
    args: &ModelArgs,
    layer: usize,
    layout: &LocalModelLayout,
    layer_root: &str,
) -> Result<ModelArgs, ParallelPlanError> {
    let policy = args
        .layer_policy(layer)
        .ok_or_else(|| invalid(format!("Gemma 4 layer {layer} is out of range")))?;
    let root = format!("{layer_root}.{layer}");
    let tensor_at = |layer: usize, suffix: &str| {
        let root = format!("{layer_root}.{layer}");
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
    static_parameter_groups_at(args, "model.language_model")
}

/// Declares shared vocabulary/readout placement under an architecture-owned root.
pub(crate) fn static_parameter_groups_at(
    args: &ModelArgs,
    root: &str,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    Ok(pinned_parameter_groups_at(args, root)?
        .into_iter()
        .map(|(_, group)| group)
        .collect())
}

/// Declares each pinned resource together with its canonical storage role.
pub(crate) fn pinned_parameter_groups(
    args: &ModelArgs,
) -> Result<Vec<(&'static str, ParameterGroupSpec)>, ParallelPlanError> {
    pinned_parameter_groups_at(args, "model.language_model")
}

fn pinned_parameter_groups_at(
    args: &ModelArgs,
    root: &str,
) -> Result<Vec<(&'static str, ParameterGroupSpec)>, ParallelPlanError> {
    let hidden = dim(args.hidden_size)?;
    let vocab = dim(args.vocab_size)?;
    let mut groups = vec![(
        "embedding",
        group(
            format!("{root}.embed_tokens"),
            ParameterRole::Vocabulary,
            [(
                format!("{root}.embed_tokens.weight"),
                vec![vocab, hidden],
                MemberSharding::Balanced { axis: 0 },
            )],
        )?,
    )];
    if args.hidden_size_per_layer_input > 0 {
        let per_layer = dim(args.hidden_size_per_layer_input)?;
        let combined = per_layer
            .checked_mul(args.num_hidden_layers())
            .ok_or_else(|| invalid("Gemma per-layer embedding width overflow"))?;
        groups.push((
            "per_layer_embedding",
            replicated(
                format!("{root}.per_layer_embedding"),
                [(
                    format!("{root}.embed_tokens_per_layer.weight"),
                    vec![
                        dim(args.vocab_size_per_layer_input.unwrap_or(args.vocab_size))?,
                        combined,
                    ],
                )],
            )?,
        ));
        groups.push((
            "per_layer_projection",
            replicated(
                format!("{root}.per_layer_projection"),
                [(
                    format!("{root}.per_layer_model_projection.weight"),
                    vec![combined, hidden],
                )],
            )?,
        ));
        groups.push((
            "per_layer_norm",
            replicated(
                format!("{root}.per_layer_projection_norm"),
                [(
                    format!("{root}.per_layer_projection_norm.weight"),
                    vec![per_layer],
                )],
            )?,
        ));
    }
    groups.push((
        "norm",
        replicated(
            format!("{root}.norm"),
            [(format!("{root}.norm.weight"), vec![hidden])],
        )?,
    ));
    if !args.tie_word_embeddings {
        groups.push((
            "output",
            group(
                "lm_head",
                ParameterRole::Vocabulary,
                [(
                    "lm_head.weight".into(),
                    vec![vocab, hidden],
                    MemberSharding::Balanced { axis: 0 },
                )],
            )?,
        ));
    }
    eredu_runtime::expand_tagged_linear_format_parameter_groups(groups, |member| {
        standard_parallel_linear_format(member, args.linear_format_for(member.target()))
    })
}

/// Declares one decoder layer's head, MLP, expert, media, and replicated groups.
pub fn layer_parameter_groups(
    args: &ModelArgs,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    layer_parameter_groups_at(args, layer, "model.language_model.layers")
}

/// Declares shared block placement under an architecture-owned logical root.
pub(crate) fn layer_parameter_groups_at(
    args: &ModelArgs,
    layer: usize,
    layer_root: &str,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let policy = args
        .layer_policy(layer)
        .ok_or_else(|| invalid(format!("Gemma 4 layer {layer} is out of range")))?;
    let root = format!("{layer_root}.{layer}");
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

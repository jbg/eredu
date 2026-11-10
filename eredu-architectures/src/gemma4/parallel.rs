//! Semantic parameter placement for Gemma 4 components.

use eredu_nn::VocabularyParallelRange;
use eredu_runtime::{
    ArchitecturePartition, LocalModelLayout, MemberSharding, ParallelPlanError, ParameterGroupSpec,
    ParameterRole, PartitionOwnership, StateLayout, TensorPlacement,
};
use std::ops::Range;

use super::{FamilyConfig, FeedForwardPolicy, ModelArgs};

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
    logical_name: &str,
    spec: &impl eredu_nn::DeclareLogicalParameters,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    Ok(vec![eredu_runtime::logical_parameter_group(
        logical_name,
        ParameterRole::Replicated,
        None,
        spec,
        |_| MemberSharding::Replicated,
    )?])
}

/// Declares pinned audio subsampling and output projection storage.
pub fn audio_static_parameter_groups(
    config: &super::AudioConfig,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let spec = config.pinned();
    let selected = spec.selected();
    audio_parameter_groups(&spec.root, &selected)
}

/// Declares one independently leased Gemma audio layer.
pub fn audio_layer_parameter_groups(
    config: &super::AudioConfig,
    index: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let spec = config
        .layer(index)
        .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))?;
    audio_parameter_groups(&spec.root, spec)
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
        let expected = super::graph::local_state_layout(args, &self.text_blocks)
            .map_err(|error| invalid(error.to_string()))?;
        if expected != self.state_layout {
            return Err(invalid(
                "rank-local Gemma 4 state layout drifted from decoder geometry",
            ));
        }
        Ok(())
    }
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
                )));
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
        .map(|layer| local_block_args(args, layer, layout))
        .collect::<Result<Vec<_>, _>>()?;
    for (layer, local) in text_blocks.iter().enumerate() {
        if let Some(scope) = args.routed(layer).map_err(|e| invalid(e.to_string()))? {
            scope
                .for_invocation(local, layer, None)
                .map_err(|e| invalid(e.to_string()))?;
        }
        let width = local
            .layer_policy(layer)
            .ok_or_else(|| invalid("dense layer outside local schedule"))?
            .intermediate_size
            .get() as i32;
        args.dense(layer)
            .and_then(|spec| spec.clone().with_intermediate(width))
            .map_err(|e| invalid(e.to_string()))?;
        args.block(layer)
            .and_then(|spec| {
                spec.clone()
                    .with_input_width(local.hidden_size_per_layer_input)
            })
            .map_err(|e| invalid(e.to_string()))?;
    }
    let per_layer_range = local_per_layer_range(&args.text, layout, &text_blocks)?;
    let state_layout = super::graph::local_state_layout(args, &text_blocks)
        .map_err(|error| invalid(error.to_string()))?;
    let (embedding_range, output_range) = args.pinned().ordinary.vocabulary_ranges(
        layout,
        "model.language_model.embed_tokens",
        "model.language_model.norm",
        "lm_head",
    )?;
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
                )));
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
    config: &FamilyConfig,
    layer: usize,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<ModelArgs, ParallelPlanError> {
    let owner = config
        .text
        .attention_state_owner(layer)
        .map_err(|e| invalid(e.to_string()))?;
    local_block_args_at(
        &config.text,
        layer,
        layout,
        "model.language_model.layers",
        config
            .attention(layer)
            .map_err(|e| invalid(e.to_string()))?,
        config
            .attention(owner)
            .map_err(|e| invalid(e.to_string()))?,
    )
}

/// Derives shared block geometry under the owning architecture's logical root.
pub(crate) fn local_block_args_at(
    args: &ModelArgs,
    layer: usize,
    layout: &LocalModelLayout,
    layer_root: &str,
    attention: &super::text::specs::AttentionSpec,
    publisher: &super::text::specs::AttentionSpec,
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
    let query = layout
        .tensor(attention.query.weight.id.as_str())
        .ok_or_else(|| {
            invalid(format!(
                "missing selected Gemma query {}",
                attention.query.weight.id.as_str()
            ))
        })?;
    let query_width = local_axis(query, 0, "query width")?;
    let head_dim = attention.geometry.key_dimensions();
    if query_width % head_dim != 0 {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "local Gemma 4 query width {query_width} splits head dimension {head_dim}"
        )));
    }
    let owner = args
        .attention_state_owner(layer)
        .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
    let key = publisher.key.as_ref().ok_or_else(|| {
        invalid(format!(
            "Gemma shared-KV publisher {owner} has no owned key declaration"
        ))
    })?;
    let kv_tensor = layout.tensor(key.weight.id.as_str()).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!(
            "Gemma 4 shared-KV layer {layer} has no rank-local publisher {owner}"
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
    args: &FamilyConfig,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    Ok(pinned_parameter_groups(args)?
        .into_iter()
        .map(|(_, group)| group)
        .collect())
}

/// Declares each pinned resource together with its canonical storage role.
pub(crate) fn pinned_parameter_groups(
    args: &FamilyConfig,
) -> Result<Vec<(&'static str, ParameterGroupSpec)>, ParallelPlanError> {
    let declaration = args.pinned();
    let root = declaration.root();
    let ordinary = declaration.ordinary.pinned_parameter_groups(
        crate::decoder::ParameterEncoding::Stored,
        "model.language_model.embed_tokens",
        "model.language_model.norm",
        "lm_head",
    )?;
    let mut extra = Vec::new();
    if let Some(spec) = &declaration.per_layer {
        for (role, name, slot) in [
            (
                "per_layer_embedding",
                format!("{root}.per_layer_embedding"),
                &spec.embedding as &dyn eredu_nn::DeclareLogicalParameters,
            ),
            (
                "per_layer_projection",
                format!("{root}.per_layer_projection"),
                &spec.projection,
            ),
            (
                "per_layer_norm",
                format!("{root}.per_layer_projection_norm"),
                &spec.norm,
            ),
        ] {
            extra.push((
                role,
                eredu_runtime::logical_parameter_group(
                    name,
                    ParameterRole::Replicated,
                    None,
                    slot,
                    |_| MemberSharding::Replicated,
                )?,
            ));
        }
    }
    let mut extra = extra.into_iter();
    let mut groups = Vec::new();
    for (role, group) in ordinary {
        groups.push((role, group));
        if role == "embedding" {
            groups.extend(extra.by_ref());
        }
    }
    Ok(groups)
}

/// Declares one decoder layer's head, MLP, expert, media, and replicated groups.
pub fn layer_parameter_groups(
    config: &FamilyConfig,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    layer_parameter_groups_at(
        &config.text,
        layer,
        "model.language_model.layers",
        config.block(layer).map_err(|e| invalid(e.to_string()))?,
        config.dense(layer).map_err(|e| invalid(e.to_string()))?,
        config
            .attention(layer)
            .map_err(|e| invalid(e.to_string()))?,
        config.routed(layer).map_err(|e| invalid(e.to_string()))?,
    )
}

/// Declares shared block placement under an architecture-owned logical root.
pub(crate) fn layer_parameter_groups_at(
    args: &ModelArgs,
    layer: usize,
    layer_root: &str,
    block_spec: &super::text::specs::BlockSpec,
    dense_spec: &super::text::specs::DenseSpec,
    attention_spec: &super::text::specs::AttentionSpec,
    routed: Option<&super::text::routed::Declaration>,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    args.layer_policy(layer)
        .ok_or_else(|| invalid(format!("Gemma 4 layer {layer} is out of range")))?;
    let root = format!("{layer_root}.{layer}");
    let attention = format!("{root}.self_attn");
    let mut groups = attention_spec
        .parameter_groups(&attention)
        .map_err(|e| invalid(e.to_string()))?;
    groups.push(
        dense_spec
            .parameter_group(&format!("{root}.mlp"))
            .map_err(|e| invalid(e.to_string()))?,
    );
    if let Some(spec) = &routed {
        groups.extend(
            spec.parameter_groups(&root)
                .map_err(|error| invalid(error.to_string()))?,
        );
    }
    if let Some(spec) = &block_spec.per_layer {
        groups.push(
            spec.parameter_group(&root)
                .map_err(|e| invalid(e.to_string()))?,
        );
    }
    use eredu_nn::DeclareLogicalParameters;
    let mut replicated_members = block_spec
        .replicated_parameters()
        .map_err(|e| invalid(e.to_string()))?;
    if let Some(spec) = &routed {
        use eredu_nn::DeclareLogicalParameters;
        replicated_members.extend(
            spec.norms[..]
                .logical_parameters()
                .map_err(|error| invalid(error.to_string()))?,
        );
    }
    if let Some(spec) = &block_spec.per_layer {
        replicated_members.extend(
            spec.norm
                .logical_parameters()
                .map_err(|e| invalid(e.to_string()))?,
        );
    }
    groups.push(eredu_runtime::logical_parameter_group(
        format!("{root}.replicated"),
        ParameterRole::Replicated,
        None,
        &replicated_members,
        |_| MemberSharding::Replicated,
    )?);
    Ok(groups)
}

fn invalid(message: impl Into<String>) -> ParallelPlanError {
    ParallelPlanError::InvalidGroup(message.into())
}

//! Semantic tensor-parallel placement for Inkling text and media components.

use std::ops::Range;

use eredu_nn::{DeclareLogicalParameters, VocabularyParallelRange};
use eredu_runtime::{
    ArchitecturePartition, LocalModelLayout, MemberSharding, ParallelPlanError, ParameterGroupSpec,
    ParameterRole, PartitionOwnership, PartitionState, StateLayout,
};

use super::{FeedForwardPolicy, ModelArgs};

/// Complete planner-derived construction and state geometry for one Inkling rank.
#[derive(Debug, Clone)]
pub struct LocalGeometry {
    text_layers: Vec<super::TextArgs>,
    embedding_range: VocabularyParallelRange,
    output_range: VocabularyParallelRange,
    state_layout: StateLayout,
    prediction_state: Option<PartitionState>,
    vision_layers: usize,
    architecture_fingerprint: String,
}

/// Exact prediction-free dense Inkling geometry owned by one TP/PP rank.
#[derive(Debug, Clone)]
pub struct PartitionLocalGeometry {
    vision_units: Option<Range<usize>>,
    text_units: Range<usize>,
    text_layers: Vec<super::TextArgs>,
    embedding_range: VocabularyParallelRange,
    output_range: VocabularyParallelRange,
    complete_state_layout: StateLayout,
    static_roles: Vec<String>,
    owns_input: bool,
    owns_output: bool,
    architecture_fingerprint: String,
}

/// Validated architecture-owned handoff for one dense Inkling partition.
#[derive(Debug, Clone)]
pub struct PartitionLocalFoundation {
    geometry: PartitionLocalGeometry,
    parameter_targets: Vec<String>,
}

impl PartitionLocalGeometry {
    /// Architecture-global optional vision units owned here.
    pub fn vision_units(&self) -> Option<Range<usize>> {
        self.vision_units.clone()
    }

    /// Architecture-global decoder units owned here.
    pub fn text_units(&self) -> Range<usize> {
        self.text_units.clone()
    }

    /// Returns one TP-local owned decoder configuration by global unit index.
    pub fn text_layer(&self, global_unit: usize) -> Option<&super::TextArgs> {
        self.text_units
            .contains(&global_unit)
            .then(|| &self.text_layers[global_unit - self.text_units.start])
    }

    /// Input-embedding vocabulary rows owned by this TP rank.
    pub const fn embedding_range(&self) -> &VocabularyParallelRange {
        &self.embedding_range
    }

    /// Output vocabulary rows owned by this TP rank.
    pub const fn output_range(&self) -> &VocabularyParallelRange {
        &self.output_range
    }

    /// Complete TP-local target state before PP slicing.
    pub const fn complete_state_layout(&self) -> &StateLayout {
        &self.complete_state_layout
    }

    /// Exact target-state slice owned by this PP rank.
    pub fn local_state_layout(&self) -> Result<StateLayout, ParallelPlanError> {
        self.complete_state_layout
            .slice(self.text_units.clone())
            .map_err(|error| invalid(error.to_string()))
    }

    /// Selected static roles in canonical vision/audio/text graph order.
    pub fn static_roles(&self) -> &[String] {
        &self.static_roles
    }

    fn validate_for(&self, args: &ModelArgs) -> Result<(), ParallelPlanError> {
        require_prediction_free(args)?;
        let text_count = usize::try_from(args.text_config.num_hidden_layers)
            .map_err(|_| invalid("Inkling text-layer count exceeds usize"))?;
        let vision_count = args
            .vision_config
            .as_ref()
            .map_or(0, |vision| vision.num_hidden_layers as usize);
        if self.architecture_fingerprint != args.architecture_fingerprint()
            || self.text_units.is_empty()
            || self.text_units.end > text_count
            || self.text_layers.len() != self.text_units.len()
            || self.complete_state_layout.len() != text_count
            || self.owns_input != (self.text_units.start == 0)
            || self.owns_output != (self.text_units.end == text_count)
        {
            return Err(invalid(
                "partition-local Inkling geometry belongs to another model or partition",
            ));
        }
        match &self.vision_units {
            Some(range)
                if args.vision_config.is_some()
                    && !range.is_empty()
                    && range.end <= vision_count => {}
            None => {}
            _ => return Err(invalid("partition-local Inkling vision range is invalid")),
        }
        self.embedding_range
            .validate_global_rows(args.text_config.vocab_size)
            .map_err(|error| invalid(error.to_string()))?;
        self.output_range
            .validate_global_rows(args.text_config.vocab_size)
            .map_err(|error| invalid(error.to_string()))?;
        let mut expected_roles = Vec::<String>::new();
        if self.owns_input {
            expected_roles.push("vision".into());
            expected_roles.push("audio".into());
        }
        if self.text_units.start == 0 {
            expected_roles.extend(["embedding".into(), "embedding_norm".into()]);
        }
        if self.text_units.end == text_count {
            expected_roles.extend([
                "norm".into(),
                "output".into(),
                super::MTP_STATIC_ROLE.into(),
            ]);
        }
        if self.static_roles != expected_roles {
            return Err(invalid(format!(
                "partition-local Inkling static roles {:?} differ from {:?}",
                self.static_roles, expected_roles
            )));
        }
        self.local_state_layout()?;
        Ok(())
    }
}

impl PartitionLocalFoundation {
    /// Validates graph ranges, target state, boundary, and selected parameters together.
    pub fn from_partition(
        args: &ModelArgs,
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
                "Inkling partition groups differ from family-local geometry",
            ));
        }
        let state = partition
            .state()
            .ok_or_else(|| invalid("Inkling decoder partition has no target state"))?;
        if state.global_layer_offset() != geometry.text_units.start
            || state.layout() != &geometry.local_state_layout()?
        {
            return Err(invalid(
                "Inkling partition state differs from its decoder range",
            ));
        }
        if partition.boundary_schema()
            != &eredu_runtime::NoAuxiliaryBoundarySchema::new(args.text_config.hidden_size)
        {
            return Err(invalid(
                "Inkling partition boundary differs from decoder hidden width",
            ));
        }
        let parameter_targets = partition
            .parameter_bindings()
            .iter()
            .flat_map(|binding| binding.members())
            .map(|member| member.target().to_owned())
            .collect();
        Ok(Self {
            geometry: geometry.clone(),
            parameter_targets,
        })
    }

    /// Exact family-local construction geometry.
    pub const fn geometry(&self) -> &PartitionLocalGeometry {
        &self.geometry
    }

    /// Canonical selected materialization targets.
    pub fn parameter_targets(&self) -> &[String] {
        &self.parameter_targets
    }
}

impl LocalGeometry {
    /// Returns one rank-local text-layer configuration.
    pub fn text_layer(&self, layer: usize) -> Option<&super::TextArgs> {
        self.text_layers.get(layer)
    }

    /// Returns all rank-local text-layer configurations.
    pub fn text_layers(&self) -> &[super::TextArgs] {
        &self.text_layers
    }

    /// Returns input-embedding vocabulary ownership.
    pub const fn embedding_range(&self) -> &VocabularyParallelRange {
        &self.embedding_range
    }

    /// Returns output-head vocabulary ownership.
    pub const fn output_range(&self) -> &VocabularyParallelRange {
        &self.output_range
    }

    /// Returns the authoritative rank-local decoder state layout.
    pub const fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    /// Returns the architecture-global state placement for embedded prediction.
    pub const fn prediction_state(&self) -> Option<&PartitionState> {
        self.prediction_state.as_ref()
    }

    /// Returns the number of neutral vision units owned by this realization.
    pub const fn vision_layers(&self) -> usize {
        self.vision_layers
    }

    pub(super) fn validate_for(&self, args: &ModelArgs) -> Result<(), ParallelPlanError> {
        let text_layers = usize::try_from(args.text_config.num_hidden_layers)
            .map_err(|_| invalid("Inkling text-layer count exceeds usize"))?;
        let vision_layers = args
            .vision_config
            .as_ref()
            .map_or(0, |vision| vision.num_hidden_layers as usize);
        if self.architecture_fingerprint != args.architecture_fingerprint()
            || self.text_layers.len() != text_layers
            || self.vision_layers != vision_layers
        {
            return Err(invalid(
                "rank-local Inkling geometry belongs to a different model configuration",
            ));
        }
        self.embedding_range
            .validate_global_rows(args.text_config.vocab_size)
            .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
        self.output_range
            .validate_global_rows(args.text_config.vocab_size)
            .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
        let expected = super::parallel_state_layout(args, &self.text_layers)
            .map_err(|error| invalid(error.to_string()))?;
        if expected != self.state_layout {
            return Err(invalid(
                "rank-local Inkling state layout drifted from text geometry",
            ));
        }
        let expected_prediction = prediction_state(args, self.state_layout.len())?;
        if expected_prediction != self.prediction_state {
            return Err(invalid(
                "rank-local Inkling prediction state drifted from composite state geometry",
            ));
        }
        Ok(())
    }
}

/// Derives all rank-local Inkling construction geometry from one typed plan.
pub fn local_geometry(
    args: &ModelArgs,
    layout: &LocalModelLayout,
) -> Result<LocalGeometry, ParallelPlanError> {
    let text_layers = (0..args.text_config.num_hidden_layers as usize)
        .map(|layer| local_text_args(args, layer, layout))
        .collect::<Result<Vec<_>, _>>()?;
    let state_layout = super::parallel_state_layout(args, &text_layers)
        .map_err(|error| invalid(error.to_string()))?;
    let prediction_state = prediction_state(args, state_layout.len())?;
    let (embedding_range, output_range) = args.pinned().ordinary.vocabulary_ranges(
        layout,
        "model.embed_tokens",
        "model.norm",
        "lm_head",
    )?;
    let geometry = LocalGeometry {
        text_layers,
        embedding_range,
        output_range: output_range.ok_or_else(|| invalid("Inkling declares an untied readout"))?,
        state_layout,
        prediction_state,
        vision_layers: args
            .vision_config
            .as_ref()
            .map_or(0, |vision| vision.num_hidden_layers as usize),
        architecture_fingerprint: args.architecture_fingerprint(),
    };
    geometry.validate_for(args)?;
    Ok(geometry)
}

fn require_prediction_free_dense(args: &ModelArgs) -> Result<(), ParallelPlanError> {
    if args.text_config.has_sparse_moe_layers() {
        return Err(invalid(
            "partition-local routed Inkling requires a selected expert realization",
        ));
    }
    require_prediction_free(args)
}

fn require_prediction_free(args: &ModelArgs) -> Result<(), ParallelPlanError> {
    if args
        .mtp_config
        .as_ref()
        .is_some_and(|mtp| mtp.num_nextn_predict_layers > 0)
    {
        return Err(invalid(
            "partition-local prediction Inkling requires an explicit prediction bridge",
        ));
    }
    Ok(())
}

fn partition_ranges(
    group_ranges: impl IntoIterator<Item = (impl AsRef<str>, Range<usize>)>,
) -> Result<(Option<Range<usize>>, Range<usize>), ParallelPlanError> {
    let mut vision = None;
    let mut text = None;
    for (group, range) in group_ranges {
        if range.is_empty() {
            return Err(invalid("Inkling partition group range cannot be empty"));
        }
        let slot = match group.as_ref() {
            super::VISION_EXECUTION_GROUP => &mut vision,
            super::TEXT_EXECUTION_GROUP => &mut text,
            super::AUDIO_EXECUTION_GROUP => {
                return Err(invalid(
                    "Inkling audio is an optional static root and has no execution-unit range",
                ))
            }
            other => {
                return Err(invalid(format!(
                    "unknown Inkling partition group {other:?}"
                )))
            }
        };
        if slot.replace(range).is_some() {
            return Err(invalid("duplicate Inkling partition group"));
        }
    }
    Ok((
        vision,
        text.ok_or_else(|| invalid("Inkling partition has no text decoder range"))?,
    ))
}

/// Derives exact dense prediction-free TP/PP-local Inkling geometry.
pub fn partition_local_geometry(
    args: &ModelArgs,
    layout: &LocalModelLayout,
    group_ranges: impl IntoIterator<Item = (impl AsRef<str>, Range<usize>)>,
    ownership: &PartitionOwnership,
) -> Result<PartitionLocalGeometry, ParallelPlanError> {
    require_prediction_free_dense(args)?;
    partition_local_geometry_impl(args, layout, group_ranges, ownership)
}

pub(crate) fn routed_partition_local_geometry(
    args: &ModelArgs,
    layout: &LocalModelLayout,
    group_ranges: impl IntoIterator<Item = (impl AsRef<str>, Range<usize>)>,
    ownership: &PartitionOwnership,
) -> Result<PartitionLocalGeometry, ParallelPlanError> {
    require_prediction_free(args)?;
    if !args.text_config.has_sparse_moe_layers() {
        return Err(invalid(
            "routed Inkling partition has no sparse text blocks",
        ));
    }
    partition_local_geometry_impl(args, layout, group_ranges, ownership)
}

fn partition_local_geometry_impl(
    args: &ModelArgs,
    layout: &LocalModelLayout,
    group_ranges: impl IntoIterator<Item = (impl AsRef<str>, Range<usize>)>,
    ownership: &PartitionOwnership,
) -> Result<PartitionLocalGeometry, ParallelPlanError> {
    let (vision_units, text_units) = partition_ranges(group_ranges)?;
    let complete = local_geometry(args, layout)?;
    let text_count = complete.text_layers.len();
    if text_units.is_empty() || text_units.end > text_count {
        return Err(invalid(format!(
            "Inkling text range {text_units:?} is outside {text_count} layers"
        )));
    }
    let vision_count = complete.vision_layers;
    if vision_units
        .as_ref()
        .is_some_and(|range| range.is_empty() || range.end > vision_count)
    {
        return Err(invalid(format!(
            "Inkling vision range {vision_units:?} is outside {vision_count} layers"
        )));
    }
    let complete_state_layout = super::composite_state_layout(&complete.state_layout, None)
        .map_err(|error| invalid(error.to_string()))?;
    let geometry = PartitionLocalGeometry {
        vision_units,
        text_layers: complete.text_layers[text_units.clone()].to_vec(),
        text_units,
        embedding_range: complete.embedding_range,
        output_range: complete.output_range,
        complete_state_layout,
        static_roles: ownership.static_roles().to_vec(),
        owns_input: ownership.owns_input(),
        owns_output: ownership.owns_output(),
        architecture_fingerprint: complete.architecture_fingerprint,
    };
    geometry.validate_for(args)?;
    Ok(geometry)
}

fn prediction_state(
    args: &ModelArgs,
    target_layers: usize,
) -> Result<Option<PartitionState>, ParallelPlanError> {
    super::mtp_state_layout(args)
        .map_err(|error| invalid(error.to_string()))?
        .map(|layout| {
            PartitionState::new(layout, target_layers).map_err(|error| invalid(error.to_string()))
        })
        .transpose()
}

/// Derives the construction geometry for one rank-local Inkling decoder layer.
///
/// Hidden-width tensors and replicated state remain global. Head and
/// intermediate dimensions are taken from the semantic placement result so
/// the ordinary layer constructor creates exactly the local projections.
pub fn local_text_args(
    source: &ModelArgs,
    layer: usize,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<super::TextArgs, ParallelPlanError> {
    let args = &source.text_config;
    let policy = args
        .layer_policy(layer)
        .ok_or_else(|| invalid(format!("Inkling layer {layer} is out of range")))?;
    let parameters = &source
        .common(layer)
        .map_err(|error| invalid(error.to_string()))?
        .attention;
    let declared = |weight: &eredu_nn::ParameterSpec| {
        layout.tensor(weight.id.as_str()).ok_or_else(|| {
            invalid(format!(
                "missing selected Inkling parameter {}",
                weight.id.as_str()
            ))
        })
    };
    let query_width = local_axis(
        declared(&parameters.projection.query.weight)?,
        0,
        "query width",
    )?;
    let key_width = local_axis(declared(&parameters.projection.key.weight)?, 0, "key width")?;
    let head_dim = parameters.projection.geometry.key_dimensions();
    if query_width % head_dim != 0 || key_width % head_dim != 0 {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "local Inkling attention widths q={query_width}, k={key_width} split head dimension {head_dim}"
        )));
    }
    let mut local = args.clone();
    if policy.attention.window().is_some() {
        local.swa_num_attention_heads = Some(query_width / head_dim);
        local.swa_num_key_value_heads = Some(key_width / head_dim);
    } else {
        local.num_attention_heads = query_width / head_dim;
        local.num_key_value_heads = key_width / head_dim;
    }
    match policy.feed_forward {
        FeedForwardPolicy::Dense => {
            let parameters = source
                .dense(layer)
                .map_err(|error| invalid(error.to_string()))?;
            let gate = layout
                .tensor(parameters.product.gate.weight.id.as_str())
                .ok_or_else(|| invalid("missing selected Inkling dense gate projection"))?;
            local.dense_intermediate_size = Some(local_axis(gate, 0, "dense intermediate width")?);
        }
        FeedForwardPolicy::SparseMoe => {
            let parameters = source
                .sparse(layer)
                .map_err(|error| invalid(error.to_string()))?;
            let eredu_nn::GatedProductGroupLayout::Packed { gate_up, .. } =
                parameters.routed.layout()
            else {
                return Err(invalid(
                    "Inkling local geometry requires a packed expert bank",
                ));
            };
            let gate_up = layout
                .tensor(gate_up.weight().id.as_str())
                .ok_or_else(|| invalid("missing selected Inkling expert gate/up projection"))?;
            let fused = local_axis(gate_up, 1, "expert gate/up width")?;
            if fused % 2 != 0 {
                return Err(ParallelPlanError::InvalidTensor(format!(
                    "local Inkling expert gate/up width {fused} is not even"
                )));
            }
            local.moe_intermediate_size = Some(fused / 2);
        }
    }
    Ok(local)
}

fn local_axis(
    tensor: &eredu_runtime::LocalTensorLayout,
    axis: usize,
    label: &str,
) -> Result<i32, ParallelPlanError> {
    let value = *tensor.local_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("local Inkling {label} tensor has no axis {axis}"))
    })?;
    let global = *tensor.global_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!(
            "global Inkling {label} tensor has no axis {axis}"
        ))
    })?;
    if value == 0 || value > global {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "local Inkling {label} width {value} is invalid for global width {global}"
        )));
    }
    i32::try_from(value)
        .map_err(|_| ParallelPlanError::InvalidTensor(format!("local Inkling {label} exceeds i32")))
}

/// Declares vocabulary, final normalization, output, and small media roots.
pub fn static_parameter_groups(
    args: &ModelArgs,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    pinned_parameter_groups(args).map(|groups| groups.into_iter().map(|(_, group)| group).collect())
}

pub(super) fn pinned_parameter_groups(
    args: &ModelArgs,
) -> Result<Vec<(&'static str, ParameterGroupSpec)>, ParallelPlanError> {
    let replicated = |name: &str, spec: &dyn eredu_nn::DeclareLogicalParameters| {
        eredu_runtime::logical_parameter_group(name, ParameterRole::Replicated, None, spec, |_| {
            MemberSharding::Replicated
        })
    };
    let embedding_norm = replicated("model.embed_norm", &args.pinned().embedding_norm)?;
    let mut groups = args
        .pinned()
        .ordinary
        .pinned_parameter_groups(
            crate::decoder::ParameterEncoding::Stored,
            "model.embed_tokens",
            "model.norm",
            "lm_head",
        )?
        .into_iter()
        .flat_map(|(role, group)| {
            std::iter::once((role, group))
                .chain((role == "embedding").then(|| ("embedding_norm", embedding_norm.clone())))
        })
        .collect::<Vec<_>>();
    if let Some(config) = &args.audio_config {
        let audio = config.parameters();
        groups.push((
            "audio",
            replicated("audio.codebook_embedding", &audio.embedding)?,
        ));
        groups.push(("audio", replicated("audio.final_norm", &audio.final_norm)?));
    }
    if let Some(config) = &args.vision_config {
        groups.push((
            "vision",
            replicated("visual.final_norm", &config.parameters().final_norm)?,
        ));
    }
    Ok(groups)
}

/// Declares one replicated Inkling vision execution unit.
pub fn vision_layer_parameter_groups(
    config: &super::VisionConfig,
    index: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let spec = config
        .parameters()
        .layer(index)
        .map_err(|error| invalid(error.to_string()))?;
    let root = format!("visual.layers.{index}");
    let projection = eredu_runtime::logical_parameter_group(
        &root,
        ParameterRole::Replicated,
        None,
        &spec.projection,
        |_| MemberSharding::Replicated,
    )?;
    let mut members = projection.members().to_vec();
    members.sort_by(|left, right| left.target().cmp(right.target()));
    if let Some(norm) = &spec.norm {
        members.extend(eredu_runtime::logical_parameter_members(norm, |_| {
            MemberSharding::Replicated
        })?);
    }
    Ok(vec![ParameterGroupSpec::new(
        root,
        ParameterRole::Replicated,
        members,
    )?])
}

/// Declares checkpoint-embedded prediction modules as replicated TP state.
///
/// Predictor depths consume the complete target activation produced after
/// each decoder all-reduce. Keeping their comparatively small blocks
/// replicated preserves the released equations while the shared vocabulary
/// ingress and egress remain sharded by [`static_parameter_groups`].
pub fn mtp_parameter_groups(
    args: &ModelArgs,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let parameters = args.prediction();
    if parameters.depths.is_empty() {
        return Ok(Vec::new());
    }
    let mut declarations = parameters
        .logical_parameters()
        .map_err(|error| invalid(error.to_string()))?;
    declarations.sort_by(|a, b| a.parameter.id.as_str().cmp(b.parameter.id.as_str()));
    Ok(vec![eredu_runtime::logical_parameter_group(
        "model.mtp",
        ParameterRole::Replicated,
        None,
        &declarations,
        |_| MemberSharding::Replicated,
    )?])
}

/// Declares one decoder layer's head, convolution, dense/expert, and replicated
/// parameter groups.
pub fn layer_parameter_groups(
    args: &ModelArgs,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let text = &args.text_config;
    let policy = text
        .layer_policy(layer)
        .ok_or_else(|| invalid(format!("Inkling layer {layer} is out of range")))?;
    let root = format!("model.layers.{layer}");
    let common = args
        .common(layer)
        .map_err(|error| invalid(error.to_string()))?;
    let attention = format!("{root}.self_attn");
    let heads = &common.attention;
    let dense = (policy.feed_forward == FeedForwardPolicy::Dense)
        .then(|| args.dense(layer))
        .transpose()
        .map_err(|error| invalid(error.to_string()))?;
    let sparse = (policy.feed_forward == FeedForwardPolicy::SparseMoe)
        .then(|| args.sparse(layer))
        .transpose()
        .map_err(|error| invalid(error.to_string()))?;
    let mut query = Vec::new();
    for projection in [
        &heads.projection.query,
        &heads.relative,
        &heads.projection.output,
    ] {
        query.extend(
            projection
                .logical_parameters()
                .map_err(|error| invalid(error.to_string()))?,
        );
    }
    let mut groups = vec![eredu_runtime::logical_parameter_group(
        format!("{attention}.query_heads"),
        ParameterRole::AttentionHeads,
        Some(dim(heads.projection.geometry.query_heads())?),
        &query,
        |parameter| {
            if parameter.parameter.id == heads.projection.output.weight.id {
                partitioned(1)
            } else if heads
                .projection
                .output
                .bias
                .as_ref()
                .is_some_and(|bias| bias.id == parameter.parameter.id)
            {
                MemberSharding::Replicated
            } else {
                partitioned(0)
            }
        },
    )?];
    let mut key_value = heads
        .projection
        .key
        .logical_parameters()
        .map_err(|error| invalid(error.to_string()))?;
    key_value.extend(
        heads
            .projection
            .value
            .logical_parameters()
            .map_err(|error| invalid(error.to_string()))?,
    );
    key_value.extend(
        heads
            .key_convolution
            .logical_parameters()
            .map_err(|error| invalid(error.to_string()))?,
    );
    key_value.extend(
        heads
            .value_convolution
            .logical_parameters()
            .map_err(|error| invalid(error.to_string()))?,
    );
    groups.push(eredu_runtime::logical_parameter_group(
        format!("{attention}.key_value_heads"),
        ParameterRole::AttentionHeads,
        Some(dim(heads.projection.geometry.key_value_heads())?),
        &key_value,
        |_| partitioned(0),
    )?);
    match policy.feed_forward {
        FeedForwardPolicy::Dense => {
            let dense = dense.as_ref().expect("declared dense branch");
            let projections = dense
                .product
                .logical_parameters()
                .map_err(|error| invalid(error.to_string()))?;
            groups.push(eredu_runtime::logical_parameter_group(
                format!("{root}.dense.intermediate"),
                ParameterRole::FeedForwardIntermediate,
                Some(dim(dense.product.down.input)?),
                &projections,
                |parameter| {
                    if parameter.parameter.id == dense.product.down.weight.id {
                        partitioned(1)
                    } else if dense
                        .product
                        .down
                        .bias
                        .as_ref()
                        .is_some_and(|bias| bias.id == parameter.parameter.id)
                    {
                        MemberSharding::Replicated
                    } else {
                        partitioned(0)
                    }
                },
            )?);
        }
        FeedForwardPolicy::SparseMoe => {
            let sparse = sparse.as_ref().expect("declared sparse branch");
            for (name, bank, role) in [
                ("experts", &sparse.routed, ParameterRole::ExpertIntermediate),
                (
                    "shared_experts",
                    &sparse.shared,
                    ParameterRole::SharedExpertIntermediate,
                ),
            ] {
                let eredu_nn::GatedProductGroupLayout::Packed { gate_up, .. } = bank.layout()
                else {
                    return Err(invalid(
                        "Inkling placement requires packed gated-product banks",
                    ));
                };
                let intermediate = dim(bank.intermediate_dimensions())?;
                groups.push(eredu_runtime::logical_parameter_group(
                    format!("{root}.moe.{name}.intermediate"),
                    role,
                    Some(intermediate),
                    bank,
                    |parameter| {
                        if parameter.parameter.id == gate_up.weight().id
                            || gate_up
                                .bias()
                                .is_some_and(|bias| bias.id == parameter.parameter.id)
                        {
                            MemberSharding::PartitionedSegments {
                                axis: 1,
                                segments: vec![0..intermediate, intermediate..2 * intermediate],
                            }
                        } else if parameter.role == eredu_nn::LogicalParameterRole::LinearWeight {
                            partitioned(2)
                        } else {
                            MemberSharding::Replicated
                        }
                    },
                )?);
            }
            groups.push(eredu_runtime::logical_parameter_group(
                format!("{root}.moe.router"),
                ParameterRole::Replicated,
                None,
                &sparse.router,
                |_| MemberSharding::Replicated,
            )?);
        }
    }

    let mut replicated = common
        .replicated_parameters()
        .map_err(|error| invalid(error.to_string()))?;
    if let Some(dense) = &dense {
        replicated.push(
            dense
                .scale_declaration()
                .map_err(|error| invalid(error.to_string()))?,
        );
    }
    groups.push(eredu_runtime::logical_parameter_group(
        format!("{root}.replicated"),
        ParameterRole::Replicated,
        None,
        &replicated,
        |_| MemberSharding::Replicated,
    )?);
    Ok(groups)
}

const fn partitioned(axis: usize) -> MemberSharding {
    MemberSharding::Partitioned { axis }
}

fn dim(value: i32) -> Result<usize, ParallelPlanError> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid(format!("invalid Inkling dimension {value}")))
}

fn invalid(message: impl Into<String>) -> ParallelPlanError {
    ParallelPlanError::InvalidGroup(message.into())
}

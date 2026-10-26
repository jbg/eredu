//! Semantic tensor-parallel placement for Inkling text and media components.

use std::{collections::BTreeMap, ops::Range};

use eredu_checkpoint::LinearFormat;
use eredu_nn::VocabularyParallelRange;
use eredu_runtime::{
    expand_linear_format_parameter_groups, ArchitecturePartition, LocalModelLayout, MemberSharding,
    ParallelPlanError, ParameterGroupSpec, ParameterMemberSpec, ParameterRole, PartitionOwnership,
    PartitionState, StateLayout,
};

use crate::linear_format::standard_parallel_linear_format;

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
        .map(|layer| local_text_args(&args.text_config, layer, layout))
        .collect::<Result<Vec<_>, _>>()?;
    let state_layout = super::parallel_state_layout(args, &text_layers)
        .map_err(|error| invalid(error.to_string()))?;
    let prediction_state = prediction_state(args, state_layout.len())?;
    let vocabulary = dim(args.text_config.vocab_size)?;
    let statics = pinned_parameter_groups(args)?;
    let vocabulary_range = |role| -> Result<_, ParallelPlanError> {
        let (_, group) = statics
            .iter()
            .find(|(resource, _)| *resource == role)
            .ok_or_else(|| invalid(format!("Inkling has no declared {role} resource")))?;
        Ok(VocabularyParallelRange {
            global_vocabulary: vocabulary,
            local: eredu_runtime::balanced_parameter_group_range(group, layout, 0, vocabulary)?,
        })
    };
    let geometry = LocalGeometry {
        text_layers,
        embedding_range: vocabulary_range("embedding")?,
        output_range: vocabulary_range("output")?,
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
    args: &super::TextArgs,
    layer: usize,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<super::TextArgs, ParallelPlanError> {
    let policy = args
        .layer_policy(layer)
        .ok_or_else(|| invalid(format!("Inkling layer {layer} is out of range")))?;
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
    let head_dim = args.attention_head_dim(policy.attention.window().is_some());
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
            local.dense_intermediate_size = Some(local_axis(
                tensor("dense.gate_proj")?,
                0,
                "dense intermediate width",
            )?);
        }
        FeedForwardPolicy::SparseMoe => {
            let fused = local_axis(
                tensor("moe.experts.gate_up_proj")?,
                1,
                "expert gate/up width",
            )?;
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
    let hidden = dim(args.text_config.hidden_size)?;
    let embedding_norm = replicated(
        "model.embed_norm",
        [("model.embed_norm.weight", vec![hidden])],
    )?;
    let mut groups = super::static_spec(&args.text_config)
        .logical_pinned_parameter_groups("model.embed_tokens", "model.norm", "lm_head")?
        .into_iter()
        .flat_map(|(role, group)| {
            std::iter::once((role, group))
                .chain((role == "embedding").then(|| ("embedding_norm", embedding_norm.clone())))
        })
        .collect::<Vec<_>>();
    if let Some(audio) = &args.audio_config {
        let audio_vocabulary = dim(audio.num_codebooks)?
            .checked_mul(dim(audio.codebook_size)?)
            .ok_or_else(|| invalid("Inkling audio vocabulary overflow"))?;
        groups.push((
            "audio",
            replicated(
                "audio.codebook_embedding",
                [("audio.encoder.weight", vec![audio_vocabulary, hidden])],
            )?,
        ));
        groups.push((
            "audio",
            replicated(
                "audio.final_norm",
                [("audio.final_norm.weight", vec![hidden])],
            )?,
        ));
    }
    if args.vision_config.is_some() {
        groups.push((
            "vision",
            replicated(
                "visual.final_norm",
                [("visual.final_norm.weight", vec![hidden])],
            )?,
        ));
    }
    eredu_runtime::expand_tagged_linear_format_parameter_groups(groups, |member| {
        let name = member.target();
        let format = if name.starts_with("visual.") {
            args.vision_config
                .as_ref()
                .map_or(LinearFormat::Dense, |config| config.linear_format_for(name))
        } else {
            args.text_config.linear_format_for(name)
        };
        standard_parallel_linear_format(member, format)
    })
}

/// Declares one replicated Inkling vision execution unit.
pub fn vision_layer_parameter_groups(
    config: &super::VisionConfig,
    index: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let (input, output, _, _) = config
        .layer_specs()
        .get(index)
        .copied()
        .filter(|_| index < config.num_hidden_layers as usize)
        .ok_or_else(|| invalid(format!("Inkling vision layer {index} is out of range")))?;
    let root = format!("visual.layers.{index}");
    let projection = expand_linear_format_parameter_groups(
        vec![ParameterGroupSpec::new(
            &root,
            ParameterRole::Replicated,
            [member(
                format!("{root}.projection.weight"),
                vec![dim(output)?, dim(input)?],
                MemberSharding::Replicated,
            )],
        )?],
        |member| standard_parallel_linear_format(member, config.linear_format_for(member.target())),
    )?;
    // Canonical projection traversal orders physical parameters by identity;
    // the following normalization remains a separate module in that traversal.
    let mut members = projection[0].members().to_vec();
    members.sort_by(|left, right| left.target().cmp(right.target()));
    if index + 1 != config.num_hidden_layers as usize {
        members.push(member(
            format!("{root}.layer_norm.weight"),
            vec![dim(output)?],
            MemberSharding::Replicated,
        ));
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
    if args
        .mtp_config
        .as_ref()
        .is_none_or(|mtp| mtp.num_nextn_predict_layers <= 0)
    {
        return Ok(Vec::new());
    }
    let plan = super::safetensors_plan(args).map_err(invalid)?;
    let mut members = BTreeMap::<String, Vec<usize>>::new();
    let mut add = |constraint: &eredu_checkpoint::schema::SafetensorsTensorConstraint| {
        let target = constraint.aliases.last().unwrap_or(&constraint.key);
        if target.starts_with("model.mtp.") {
            members.insert(target.clone(), constraint.shape.clone());
        }
    };
    for constraint in &plan.common_tensors {
        add(constraint);
    }
    for group in &plan.layout_groups {
        if !group.variants.iter().any(|variant| {
            variant.tensors.iter().any(|constraint| {
                constraint.key.starts_with("model.mtp.")
                    || constraint
                        .aliases
                        .iter()
                        .any(|alias| alias.starts_with("model.mtp."))
            })
        }) {
            continue;
        }
        let canonical = group
            .variants
            .iter()
            .find(|variant| variant.id == "canonical split")
            .ok_or_else(|| {
                invalid(format!(
                    "Inkling MTP layout group {:?} has no canonical split",
                    group.id
                ))
            })?;
        for constraint in &canonical.tensors {
            add(constraint);
        }
    }
    if members.is_empty() {
        return Err(invalid(
            "Inkling checkpoint declares MTP depths but no MTP parameters",
        ));
    }
    expand_linear_format_parameter_groups(vec![replicated("model.mtp", members)?], |member| {
        standard_parallel_linear_format(member, args.text_config.linear_format_for(member.target()))
    })
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
    let local = policy.attention.window().is_some();
    let hidden = dim(text.hidden_size)?;
    let query_heads = dim(text.query_heads(local))?;
    let key_value_heads = dim(text.key_value_heads(local))?;
    let head = dim(text.attention_head_dim(local))?;
    let query_width = query_heads
        .checked_mul(head)
        .ok_or_else(|| invalid("Inkling query width overflow"))?;
    let key_value_width = key_value_heads
        .checked_mul(head)
        .ok_or_else(|| invalid("Inkling key/value width overflow"))?;
    let relative_width = query_heads
        .checked_mul(dim(text.d_rel)?)
        .ok_or_else(|| invalid("Inkling relative-query width overflow"))?;
    let root = format!("model.layers.{layer}");
    let attention = format!("{root}.self_attn");
    let mut query_members = vec![
        member(
            format!("{attention}.q_proj.weight"),
            vec![query_width, hidden],
            partitioned(0),
        ),
        member(
            format!("{attention}.r_proj.weight"),
            vec![relative_width, hidden],
            partitioned(0),
        ),
        member(
            format!("{attention}.o_proj.weight"),
            vec![hidden, query_width],
            partitioned(1),
        ),
    ];
    if text.q_bias {
        query_members.push(member(
            format!("{attention}.q_proj.bias"),
            vec![query_width],
            partitioned(0),
        ));
    }
    if text.o_bias {
        query_members.push(member(
            format!("{attention}.o_proj.bias"),
            vec![hidden],
            MemberSharding::Replicated,
        ));
    }
    let mut groups = vec![ParameterGroupSpec::partitioned(
        format!("{attention}.query_heads"),
        ParameterRole::AttentionHeads,
        query_heads,
        query_members,
    )?];
    groups.push(ParameterGroupSpec::partitioned(
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
            member(
                format!("{attention}.k_sconv.weight"),
                vec![key_value_width, 1, dim(text.sconv_kernel_size)?],
                partitioned(0),
            ),
            member(
                format!("{attention}.v_sconv.weight"),
                vec![key_value_width, 1, dim(text.sconv_kernel_size)?],
                partitioned(0),
            ),
        ],
    )?);
    match policy.feed_forward {
        FeedForwardPolicy::Dense => {
            let intermediate = dim(text.dense_intermediate_size())?;
            groups.push(ParameterGroupSpec::partitioned(
                format!("{root}.dense.intermediate"),
                ParameterRole::FeedForwardIntermediate,
                intermediate,
                [
                    member(
                        format!("{root}.dense.gate_proj.weight"),
                        vec![intermediate, hidden],
                        partitioned(0),
                    ),
                    member(
                        format!("{root}.dense.up_proj.weight"),
                        vec![intermediate, hidden],
                        partitioned(0),
                    ),
                    member(
                        format!("{root}.dense.down_proj.weight"),
                        vec![hidden, intermediate],
                        partitioned(1),
                    ),
                ],
            )?);
        }
        FeedForwardPolicy::SparseMoe => {
            let intermediate = dim(text.moe_intermediate_size())?;
            let routed = dim(text.n_routed_experts)?;
            let shared = dim(text.n_shared_experts)?;
            for (name, count, role) in [
                ("experts", routed, ParameterRole::ExpertIntermediate),
                (
                    "shared_experts",
                    shared,
                    ParameterRole::SharedExpertIntermediate,
                ),
            ] {
                groups.push(ParameterGroupSpec::partitioned(
                    format!("{root}.moe.{name}.intermediate"),
                    role,
                    intermediate,
                    [
                        member(
                            format!("{root}.moe.{name}.gate_up_proj"),
                            vec![count, 2 * intermediate, hidden],
                            MemberSharding::PartitionedSegments {
                                axis: 1,
                                segments: vec![0..intermediate, intermediate..2 * intermediate],
                            },
                        ),
                        member(
                            format!("{root}.moe.{name}.down_proj"),
                            vec![count, hidden, intermediate],
                            partitioned(2),
                        ),
                    ],
                )?);
            }
            groups.push(replicated(
                format!("{root}.moe.router"),
                [
                    (
                        format!("{root}.moe.router.weight"),
                        vec![routed + shared, hidden],
                    ),
                    (format!("{root}.moe.router.bias"), vec![routed]),
                    (format!("{root}.moe.router.global_scale"), vec![1]),
                ],
            )?);
        }
    }
    let relative_extent = policy
        .attention
        .window()
        .map(|window| window.get() as usize)
        .unwrap_or(dim(text.rel_extent)?);
    let mut replicated_members = vec![
        (format!("{attention}.q_norm.weight"), vec![head]),
        (format!("{attention}.k_norm.weight"), vec![head]),
        (
            format!("{attention}.rel_proj"),
            vec![dim(text.d_rel)?, relative_extent],
        ),
        (format!("{root}.input_layernorm.weight"), vec![hidden]),
        (
            format!("{root}.post_attention_layernorm.weight"),
            vec![hidden],
        ),
        (
            format!("{root}.attn_sconv.weight"),
            vec![hidden, 1, dim(text.sconv_kernel_size)?],
        ),
        (
            format!("{root}.mlp_sconv.weight"),
            vec![hidden, 1, dim(text.sconv_kernel_size)?],
        ),
    ];
    if policy.feed_forward == FeedForwardPolicy::Dense {
        replicated_members.push((format!("{root}.dense_global_scale"), vec![1]));
    }
    groups.push(replicated(
        format!("{root}.replicated"),
        replicated_members,
    )?);
    expand_linear_format_parameter_groups(groups, |member| {
        standard_parallel_linear_format(member, args.text_config.linear_format_for(member.target()))
    })
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
        .ok_or_else(|| invalid(format!("invalid Inkling dimension {value}")))
}

fn invalid(message: impl Into<String>) -> ParallelPlanError {
    ParallelPlanError::InvalidGroup(message.into())
}

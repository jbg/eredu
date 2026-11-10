//! Semantic tensor-parallel placement for LFM3 physical blocks.

use eredu_nn::{LogicalParameterRole, VocabularyParallelRange};
use eredu_runtime::{
    aligned_partition_units, ArchitectureParameterDescription, ExecutionGraph, ExecutionUnitLayout,
    LocalModelLayout, MemberSharding, OwnedParameterGroupSpec, ParallelPlanError,
    ParameterGroupOwner, ParameterRole, StateLayout,
};
use std::ops::Range;

use super::{prompt_cache_architecture_fingerprint, BlockGeometry, Declaration, ModelArgs};

/// Complete planner-derived construction and state geometry for one LFM3 rank.
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
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    tied_head: bool,
}

/// Backend-neutral geometry retained by one genuinely pipeline-local LFM3 model.
///
/// Block geometry is stored only for the architecture-global units owned by
/// this pipeline partition. The complete state layout remains available so
/// [`eredu_runtime::ArchitecturePartition`] can derive the exact local slice
/// while preserving its architecture-global offset.
#[derive(Debug, Clone)]
pub struct PartitionLocalGeometry {
    owned_units: Range<usize>,
    blocks: Vec<BlockGeometry>,
    embedding_range: VocabularyParallelRange,
    output_range: Option<VocabularyParallelRange>,
    complete_state_layout: StateLayout,
    architecture_fingerprint: String,
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    tied_head: bool,
    static_roles: Vec<String>,
    boundary_schema: eredu_runtime::NoAuxiliaryBoundarySchema,
}

impl PartitionLocalGeometry {
    /// Architecture-global execution-unit range physically owned here.
    pub fn owned_units(&self) -> Range<usize> {
        self.owned_units.clone()
    }

    /// Returns local geometry for one owned architecture-global block.
    pub fn block(&self, global_unit: usize) -> Option<&BlockGeometry> {
        self.owned_units
            .contains(&global_unit)
            .then(|| &self.blocks[global_unit - self.owned_units.start])
    }

    /// Number of block configurations physically retained by this partition.
    pub fn local_unit_count(&self) -> usize {
        self.blocks.len()
    }

    /// Input-embedding vocabulary ownership for this tensor coordinate.
    pub const fn embedding_range(&self) -> &VocabularyParallelRange {
        &self.embedding_range
    }

    /// Untied output-head vocabulary ownership for this tensor coordinate.
    pub const fn output_range(&self) -> Option<&VocabularyParallelRange> {
        self.output_range.as_ref()
    }

    /// Complete TP-local state geometry before pipeline slicing.
    pub const fn complete_state_layout(&self) -> &StateLayout {
        &self.complete_state_layout
    }

    /// Static module roles placed by this pipeline-local range.
    pub fn static_roles(&self) -> &[String] {
        &self.static_roles
    }

    /// Exact no-auxiliary hidden-state boundary for this family.
    pub const fn boundary_schema(&self) -> &eredu_runtime::NoAuxiliaryBoundarySchema {
        &self.boundary_schema
    }

    /// Returns the PP-local state slice in local ordinal order.
    pub fn local_state_layout(&self) -> Result<StateLayout, ParallelPlanError> {
        self.complete_state_layout
            .slice(self.owned_units.clone())
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))
    }

    /// Architecture-global ordinal represented by local state ordinal zero.
    pub const fn state_global_offset(&self) -> usize {
        self.owned_units.start
    }

    /// Validates selected static ownership and role-exact boundary before construction.
    pub fn validate_partition_contract(
        &self,
        ownership: &eredu_runtime::PartitionOwnership,
        boundary: &eredu_runtime::NoAuxiliaryBoundarySchema,
    ) -> Result<(), ParallelPlanError> {
        let owns_input = self.owned_units.start == 0;
        let owns_output = self.static_roles.iter().any(|role| role == "output");
        if ownership.owns_input() != owns_input
            || ownership.owns_output() != owns_output
            || ownership.static_roles() != self.static_roles
            || boundary != &self.boundary_schema
        {
            return Err(ParallelPlanError::InvalidGroup(
                "LFM3 partition ownership or boundary drifted from local geometry".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn validate_for(&self, declaration: &Declaration) -> Result<(), ParallelPlanError> {
        let args = declaration.args();

        let count = declaration.blocks.len();
        if self.owned_units.is_empty()
            || self.owned_units.end > count
            || self.blocks.len() != self.owned_units.len()
            || self.architecture_fingerprint != prompt_cache_architecture_fingerprint(args)
            || self.definition.parameters() != declaration.definition().parameters()
            || self.tied_head != declaration.statics.head_spec().is_none()
            || self.static_roles != expected_static_roles(&self.owned_units, count)
            || self.boundary_schema
                != eredu_runtime::NoAuxiliaryBoundarySchema::new(
                    declaration.statics.embedding_spec().dimensions,
                )
        {
            return Err(ParallelPlanError::InvalidGroup(
                "partition-local LFM3 geometry belongs to a different model or range".into(),
            ));
        }
        self.embedding_range
            .validate_global_rows(declaration.statics.embedding_spec().vocabulary)
            .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
        match (
            declaration.statics.head_spec().is_none(),
            &self.output_range,
        ) {
            (true, None) => {}
            (false, Some(range)) => range
                .validate_global_rows(declaration.statics.embedding_spec().vocabulary)
                .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?,
            _ => {
                return Err(ParallelPlanError::InvalidGroup(
                    "partition-local LFM3 output vocabulary ownership is inconsistent".into(),
                ));
            }
        }
        if self.complete_state_layout.len() != count {
            return Err(ParallelPlanError::InvalidGroup(
                "partition-local LFM3 complete state length is invalid".into(),
            ));
        }
        for global in self.owned_units.clone() {
            let block = self.block(global).expect("validated owned LFM3 block");
            let policy = self.complete_state_layout.layer(global).ok_or_else(|| {
                ParallelPlanError::InvalidGroup("missing LFM3 complete state layer".into())
            })?;
            let expected = declaration.blocks[global]
                .clone()
                .local(*block)
                .and_then(|scope| scope.mixer.state_policy())
                .map_err(invalid)?;
            let matches = *policy == expected;
            if !matches {
                return Err(ParallelPlanError::InvalidGroup(format!(
                    "owned LFM3 block {global} differs from complete TP-local state geometry"
                )));
            }
        }
        Ok(())
    }
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

    pub(super) fn validate_for(&self, declaration: &Declaration) -> Result<(), ParallelPlanError> {
        let args = declaration.args();
        let expected_layers = declaration.blocks.len();
        if self.blocks.len() != expected_layers
            || self.architecture_fingerprint != prompt_cache_architecture_fingerprint(args)
            || self.definition.parameters() != declaration.definition().parameters()
            || self.tied_head != declaration.statics.head_spec().is_none()
        {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local LFM3 geometry belongs to a different model configuration".into(),
            ));
        }
        self.embedding_range
            .validate_global_rows(declaration.statics.embedding_spec().vocabulary)
            .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?;
        match (
            declaration.statics.head_spec().is_none(),
            &self.output_range,
        ) {
            (true, None) => {}
            (false, Some(range)) => range
                .validate_global_rows(declaration.statics.embedding_spec().vocabulary)
                .map_err(|error| ParallelPlanError::InvalidTensor(error.to_string()))?,
            (true, Some(_)) => {
                return Err(ParallelPlanError::InvalidGroup(
                    "tied LFM3 output unexpectedly owns a separate vocabulary range".into(),
                ));
            }
            (false, None) => {
                return Err(ParallelPlanError::InvalidGroup(
                    "untied LFM3 output is missing vocabulary ownership".into(),
                ));
            }
        }
        let expected = declaration
            .state_layout_with_geometry(&self.blocks)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        if expected != self.state_layout {
            return Err(ParallelPlanError::InvalidGroup(
                "rank-local LFM3 state layout drifted from block geometry".into(),
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
        ParallelPlanError::InvalidTensor(format!("missing local LFM3 layout for {name}"))
    })?;
    let local = *tensor.local_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("LFM3 tensor {name} has no axis {axis}"))
    })?;
    let global = *tensor.global_shape().get(axis).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("LFM3 tensor {name} has no global axis {axis}"))
    })?;
    if local == 0 || local > global {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "LFM3 tensor {name} has invalid local width {local} of global width {global}"
        )));
    }
    i32::try_from(local)
        .map_err(|_| ParallelPlanError::InvalidTensor(format!("LFM3 width for {name} exceeds i32")))
}

/// Derives one block's local widths exclusively from the resolved placement.
pub fn local_block_geometry(
    declaration: &Declaration,
    layer: usize,
    layout: &LocalModelLayout,
) -> Result<BlockGeometry, ParallelPlanError> {
    use super::declarations::{FeedForwardParameters as F, MixerParameters as M};
    let scope = declaration
        .blocks
        .get(layer)
        .ok_or_else(|| ParallelPlanError::InvalidGroup(format!("LFM3 has no layer {layer}")))?;
    let mut geometry = scope.geometry(declaration.args());
    match &scope.mixer {
        M::Attention(spec, _) => {
            let head = spec.projection.geometry.key_dimensions();
            let query = local_width(layout, spec.projection.query.weight.id.as_str(), 0)?;
            let key = local_width(layout, spec.projection.key.weight.id.as_str(), 0)?;
            if query % head != 0 || key % head != 0 {
                return Err(ParallelPlanError::InvalidTensor(format!(
                    "local LFM3 attention widths q={query} k={key} split head dimension {head}"
                )));
            }
            geometry.query_heads = query / head;
            geometry.key_value_heads = key / head;
        }
        M::Convolution(spec) => {
            geometry.convolution_channels =
                local_width(layout, spec.convolution.weight.id.as_str(), 0)?;
        }
    }
    match &scope.feed_forward {
        F::Dense(spec) => {
            geometry.dense_intermediate = local_width(layout, spec.gate.weight.id.as_str(), 0)?
        }
    }
    scope.clone().local(geometry).map_err(invalid)?;
    Ok(geometry)
}

/// Derives complete TP-local heterogeneous state before pipeline slicing.
///
/// State follows attention key/value heads or convolution output channels;
/// feed-forward routing does not alter either state-bearing width.
pub fn partitioned_state_layout(
    declaration: &Declaration,
    tensor_rank: usize,
    tensor_size: usize,
) -> Result<StateLayout, ParallelPlanError> {
    let args = declaration.args();
    if tensor_size == 0 || tensor_rank >= tensor_size {
        return Err(ParallelPlanError::InvalidGroup(format!(
            "invalid LFM3 tensor coordinate {tensor_rank}/{tensor_size}"
        )));
    }
    let select = |value: i32, role: &str| {
        let global = usize::try_from(value)
            .map_err(|_| ParallelPlanError::InvalidGroup(format!("LFM3 {role} is not positive")))?;
        let local = eredu_core::balanced_contiguous_range(global, tensor_size, tensor_rank, false)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?
            .len();
        i32::try_from(local)
            .map_err(|_| ParallelPlanError::InvalidGroup(format!("LFM3 local {role} exceeds i32")))
    };
    let geometry = declaration
        .blocks
        .iter()
        .map(|scope| {
            let mut block = scope.geometry(args);
            match &scope.mixer {
                super::declarations::MixerParameters::Attention(spec, _) => {
                    block.query_heads =
                        select(spec.projection.geometry.query_heads(), "query head count")?;
                    block.key_value_heads = select(
                        spec.projection.geometry.key_value_heads(),
                        "key/value head count",
                    )?;
                }
                super::declarations::MixerParameters::Convolution(spec) => {
                    block.convolution_channels =
                        select(spec.channels, "convolution channel count")?;
                }
            }
            Ok(block)
        })
        .collect::<Result<Vec<_>, ParallelPlanError>>()?;
    declaration
        .state_layout_with_geometry(&geometry)
        .map_err(invalid)
}

/// Derives complete rank-local LFM3 construction geometry from one typed plan.
pub fn local_geometry(
    declaration: &Declaration,
    layout: &LocalModelLayout,
) -> Result<LocalGeometry, ParallelPlanError> {
    let args = declaration.args();
    let layers = declaration.blocks.len();
    let blocks = (0..layers)
        .map(|layer| local_block_geometry(declaration, layer, layout))
        .collect::<Result<Vec<_>, _>>()?;
    let state_layout = declaration
        .state_layout_with_geometry(&blocks)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let (embedding_range, output_range) = declaration.statics.vocabulary_ranges(
        layout,
        "model.embed_tokens",
        "model.embedding_norm",
        "lm_head",
    )?;
    let geometry = LocalGeometry {
        blocks,
        embedding_range,
        output_range,
        state_layout,
        architecture_fingerprint: prompt_cache_architecture_fingerprint(args),
        definition: std::sync::Arc::clone(declaration.definition()),
        tied_head: declaration.statics.head_spec().is_none(),
    };
    geometry.validate_for(declaration)?;
    Ok(geometry)
}

/// Derives exact TP-local and PP-local construction geometry for dense LFM3.
pub fn partition_local_geometry(
    declaration: &Declaration,
    layout: &LocalModelLayout,
    owned_units: Range<usize>,
) -> Result<PartitionLocalGeometry, ParallelPlanError> {
    let count = declaration.blocks.len();
    if owned_units.is_empty() || owned_units.end > count {
        return Err(ParallelPlanError::InvalidGroup(format!(
            "LFM3 local unit range {owned_units:?} is outside {count} layers"
        )));
    }
    let complete = local_geometry(declaration, layout)?;
    let blocks = owned_units
        .clone()
        .map(|unit| {
            complete.block(unit).copied().ok_or_else(|| {
                ParallelPlanError::InvalidGroup(format!("LFM3 has no local block {unit}"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let geometry = PartitionLocalGeometry {
        owned_units: owned_units.clone(),
        blocks,
        embedding_range: complete.embedding_range.clone(),
        output_range: complete.output_range.clone(),
        complete_state_layout: complete.state_layout.clone(),
        architecture_fingerprint: complete.architecture_fingerprint.clone(),
        definition: std::sync::Arc::clone(&complete.definition),
        tied_head: complete.tied_head,
        static_roles: expected_static_roles(&owned_units, count),
        boundary_schema: eredu_runtime::NoAuxiliaryBoundarySchema::new(
            declaration.statics.embedding_spec().dimensions,
        ),
    };
    geometry.validate_for(declaration)?;
    Ok(geometry)
}

fn expected_static_roles(owned: &Range<usize>, count: usize) -> Vec<String> {
    let mut roles = Vec::new();
    if owned.start == 0 {
        roles.push("embedding".into());
    }
    if owned.end == count {
        roles.extend(["norm".into(), "output".into()]);
    }
    roles
}

pub(super) fn parameter_description_from_scopes(
    args: &ModelArgs,
    statics: &crate::decoder::StaticModuleConstruction,
    blocks: &[super::declarations::BlockParameters],
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    args.validate().map_err(invalid)?;
    let layers = blocks.len();
    let graph = ExecutionGraph::new(
        vec![eredu_runtime::ExecutionGroupSpec::root(
            crate::decoder::TARGET_EXECUTION_GROUP,
        )],
        crate::decoder::TARGET_EXECUTION_GROUP,
    )
    .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let layout = ExecutionUnitLayout::new(&graph, [layers])
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let mut owned = statics
        .pinned_parameter_groups(
            crate::decoder::ParameterEncoding::Stored,
            "model.embed_tokens",
            "model.embedding_norm",
            "lm_head",
        )?
        .into_iter()
        .map(|(role, group)| {
            let owner = if role == "embedding" && args.tie_word_embeddings {
                ParameterGroupOwner::static_any_of(["embedding", "output"])
            } else {
                ParameterGroupOwner::static_role(role)
            };
            OwnedParameterGroupSpec::new(owner, group)
        })
        .collect::<Vec<_>>();
    for layer in 0..layers {
        let mut groups = Vec::new();
        let root = format!("model.layers.{layer}");
        let scope = &blocks[layer];
        match &scope.mixer {
            super::declarations::MixerParameters::Attention(spec, _) => {
                let output = &spec.projection.output;
                groups.push(eredu_runtime::logical_parameter_group(
                    format!("{root}.self_attn.heads"),
                    ParameterRole::AttentionHeads,
                    Some(spec.projection.geometry.key_value_heads() as usize),
                    spec,
                    |p| {
                        if p.role == LogicalParameterRole::Normalization {
                            MemberSharding::Replicated
                        } else if p.parameter.id == output.weight.id {
                            MemberSharding::Partitioned { axis: 1 }
                        } else {
                            MemberSharding::Partitioned { axis: 0 }
                        }
                    },
                )?);
            }
            super::declarations::MixerParameters::Convolution(spec) => {
                let channels = spec.channels as usize;
                let segments = eredu_nn::GatedShortConvolutionSpec::input_layout(spec.channels)
                    .map_err(invalid)?
                    .ranges()
                    .map(|range| range.start as usize..range.end as usize)
                    .collect::<Vec<_>>();
                groups.push(eredu_runtime::logical_parameter_group(
                    format!("{root}.conv.channels"),
                    ParameterRole::Channels,
                    Some(channels),
                    spec,
                    |declaration| {
                        let id = &declaration.parameter.id;
                        if id == &spec.input_projection.weight.id
                            || spec
                                .input_projection
                                .bias
                                .as_ref()
                                .is_some_and(|p| &p.id == id)
                        {
                            MemberSharding::PartitionedSegments {
                                axis: 0,
                                segments: segments.clone(),
                            }
                        } else if id == &spec.output_projection.weight.id {
                            MemberSharding::Partitioned { axis: 1 }
                        } else if spec
                            .output_projection
                            .bias
                            .as_ref()
                            .is_some_and(|p| &p.id == id)
                        {
                            MemberSharding::Replicated
                        } else {
                            MemberSharding::Partitioned { axis: 0 }
                        }
                    },
                )?);
            }
        }
        for (name, spec) in [
            ("operator_norm", Some(&scope.operator_norm)),
            ("ffn_norm", Some(&scope.feed_forward_norm)),
            ("post_operator_norm", scope.post_operator_norm.as_ref()),
            ("post_ffn_norm", scope.post_ffn_norm.as_ref()),
        ]
        .into_iter()
        .filter_map(|(name, spec)| spec.map(|spec| (name, spec)))
        {
            groups.push(eredu_runtime::logical_parameter_group(
                format!("{root}.{name}"),
                ParameterRole::Replicated,
                None,
                spec,
                |_| MemberSharding::Replicated,
            )?);
        }
        match &scope.feed_forward {
            super::declarations::FeedForwardParameters::Dense(spec) => {
                let down = &spec.down;
                let units = aligned_partition_units(
                    &root,
                    down.input as usize,
                    1,
                    crate::linear_format::input_partition_alignment(down.format.encoding())
                        as usize,
                )?;
                groups.push(eredu_runtime::logical_parameter_group(
                    format!("{root}.feed_forward.intermediate"),
                    ParameterRole::FeedForwardIntermediate,
                    Some(units),
                    spec,
                    |p| {
                        if p.parameter.id == down.weight.id {
                            MemberSharding::Partitioned { axis: 1 }
                        } else {
                            MemberSharding::Partitioned { axis: 0 }
                        }
                    },
                )?);
            }
        }
        let owner = ParameterGroupOwner::execution_unit(
            layout.group_id(0).expect("LFM3 group").clone(),
            layer,
        );
        owned.extend(
            groups
                .into_iter()
                .map(|group| OwnedParameterGroupSpec::new(owner.clone(), group)),
        );
    }
    ArchitectureParameterDescription::from_owned(&graph, &layout, owned)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))
}

fn invalid(error: impl std::fmt::Display) -> ParallelPlanError {
    ParallelPlanError::InvalidGroup(error.to_string())
}

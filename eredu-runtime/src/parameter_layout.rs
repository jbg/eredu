//! Physical parameter-layout lowering over architecture-authored neutral declarations.

use std::{borrow::Cow, ops::Range};

/// Strictly lowers an architecture-owned parameter description to one rank's
/// tensor- and expert-parallel placement.
///
/// This function does not construct modules, select a realization, or project
/// materialization work. It is therefore a layout primitive, not a completed
/// partitioned architecture handoff.
pub fn derive_partitioned_local_layout(
    description: &crate::ArchitectureParameterDescription,
    topology: eredu_core::ParallelRankTopology,
) -> Result<crate::LocalModelLayout, String> {
    partitioned_local_layout(description, topology).map(Cow::into_owned)
}

// Borrow authored placement; allocate only when lowering an unbound declaration.
pub(crate) fn partitioned_local_layout(
    description: &crate::ArchitectureParameterDescription,
    topology: eredu_core::ParallelRankTopology,
) -> Result<Cow<'_, crate::LocalModelLayout>, String> {
    if let Some(layout) = retained_partition_layout(description, topology)? {
        return Ok(Cow::Borrowed(layout));
    }
    local_layout(
        description,
        topology.tensor_parallel_rank(),
        topology.tensor_parallel_size(),
        topology.expert_parallel_rank(),
        topology.expert_parallel_size(),
    )
    .map(Cow::Owned)
}

/// Source-format geometry cannot preserve a selected transform's execution ownership.
#[derive(Debug, thiserror::Error)]
pub enum TransformSourceLayoutError {
    /// Source and target declarations disagree on logical invocation addresses.
    #[error(
        "source and target transform execution-unit addresses differ before local construction"
    )]
    ParameterSpaceMismatch,
    /// The source declaration cannot realize the selected rank.
    #[error("{0}")]
    SourcePlacement(String),
    /// The target declaration cannot realize the selected rank.
    #[error("{0}")]
    TargetPlacement(String),
    /// The target's exact cuts cannot be represented in source coordinates.
    #[error(transparent)]
    Projection(#[from] crate::ParallelPlanError),
}

/// Validates the logical invocation space before constructing conversion resources.
pub fn validate_transform_parameter_space(
    source: &crate::ArchitectureParameterDescription,
    target: &crate::ArchitectureParameterDescription,
) -> Result<(), TransformSourceLayoutError> {
    if source.graph() != target.graph() || source.unit_layout() != target.unit_layout() {
        return Err(TransformSourceLayoutError::ParameterSpaceMismatch);
    }
    Ok(())
}

/// Retains target ownership while constructing source-format coordinates for
/// pointwise conversion of the same canonical parameters. No resources are loaded.
pub fn derive_partitioned_transform_source_layout(
    source: &crate::ArchitectureParameterDescription,
    target: &crate::ArchitectureParameterDescription,
    topology: eredu_core::ParallelRankTopology,
) -> Result<crate::LocalModelLayout, TransformSourceLayoutError> {
    validate_transform_parameter_space(source, target)?;
    let source = partitioned_local_layout(source, topology)
        .map_err(TransformSourceLayoutError::SourcePlacement)?;
    let target = partitioned_local_layout(target, topology)
        .map_err(TransformSourceLayoutError::TargetPlacement)?;
    Ok(crate::derive_transform_source_layout(&source, &target)?)
}

/// Prepares one retained source layout only when admitted lowerings convert payloads.
/// The caller supplies ordinary or exceptional geometry through the same typed
/// builder. Direct loading and recipe-only derivation never invoke it.
pub fn prepare_transform_source_layout<E>(
    lowerings: impl IntoIterator<Item = crate::WeightLoweringKind>,
    build: impl FnOnce() -> Result<crate::LocalModelLayout, E>,
) -> Result<Option<std::sync::Arc<crate::LocalModelLayout>>, E> {
    lowerings
        .into_iter()
        .any(crate::WeightLoweringKind::transforms_payload)
        .then(|| build().map(std::sync::Arc::new))
        .transpose()
}

/// Exact selected parameter placement does not match a tensor communicator.
#[derive(Debug, thiserror::Error)]
pub enum TensorParallelLayoutError {
    /// Communicator size or rank cannot form a tensor-only topology.
    #[error("component {component:?} has invalid tensor communicator: {source}")]
    Topology {
        /// Architecture-owned component identity.
        component: String,
        /// Original neutral topology failure.
        #[source]
        source: eredu_core::TopologyError,
    },
    /// The declaration cannot realize this communicator's placement.
    #[error("component {component:?} cannot derive tensor placement: {reason}")]
    Placement {
        /// Architecture-owned component identity.
        component: String,
        /// Exact layout compiler rejection.
        reason: String,
    },
    /// Selected physical parameters belong to another communicator.
    #[error("component {component:?} parameter layout differs from its tensor communicator")]
    Mismatch {
        /// Architecture-owned component identity.
        component: String,
    },
}

/// Validates ordinary tensor placement before acquisition or collective work.
/// Families retain their additional coupled-dimension and equation constraints.
pub fn validate_tensor_parallel_layout(
    component: &str,
    description: &crate::ArchitectureParameterDescription,
    selected: &crate::LocalModelLayout,
    size: usize,
    rank: usize,
) -> Result<(), TensorParallelLayoutError> {
    let topology = eredu_core::ParallelTopology::new(size, 1, 1, 1)
        .and_then(|topology| eredu_core::ParallelRankTopology::new(topology, rank))
        .map_err(|source| TensorParallelLayoutError::Topology {
            component: component.into(),
            source,
        })?;
    let expected = partitioned_local_layout(description, topology).map_err(|reason| {
        TensorParallelLayoutError::Placement {
            component: component.into(),
            reason,
        }
    })?;
    if selected != expected.as_ref() {
        return Err(TensorParallelLayoutError::Mismatch {
            component: component.into(),
        });
    }
    Ok(())
}

/// Returns an authored layout only for the exact rank whose placement it retains.
/// Descriptions without a retained layout are lowered from their parameter groups.
pub fn retained_partition_layout(
    description: &crate::ArchitectureParameterDescription,
    topology: eredu_core::ParallelRankTopology,
) -> Result<Option<&crate::LocalModelLayout>, String> {
    match description.partition_layout() {
        Some((rank, layout)) if rank == topology => Ok(Some(layout)),
        Some((rank, _)) => Err(format!(
            "parameter placement is retained only for rank {}; projection onto rank {} is not implemented",
            rank.global_rank(),
            topology.global_rank()
        )),
        None => Ok(None),
    }
}

/// Lowers selected parameter groups to one rank's tensor- and expert-parallel placement.
/// This supports bounded inspection of a subset without reconstructing a model description.
pub fn derive_partitioned_group_layout<'a>(
    groups: impl IntoIterator<Item = &'a crate::OwnedParameterGroupSpec>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<crate::LocalModelLayout, String> {
    local_layout_groups(
        groups,
        topology.tensor_parallel_rank(),
        topology.tensor_parallel_size(),
        topology.expert_parallel_rank(),
        topology.expert_parallel_size(),
    )
}

/// Extracts a declared group's exact logical interval from a partitioned member.
///
/// The selected tensor retains the declaration's semantic domain. Physical shape
/// and axis placement never infer that interval, including for chunked members.
/// Use `validate_parameter_layout` to validate the complete physical group.
pub fn logical_parameter_group_range(
    group: &crate::ParameterGroupSpec,
    layout: &crate::LocalModelLayout,
) -> Result<Range<usize>, crate::ParallelPlanError> {
    let units = group.partition_units().ok_or_else(|| {
        crate::ParallelPlanError::InvalidGroup(format!(
            "parameter group {:?} has no logical partition",
            group.logical_name(),
        ))
    })?;
    let member = group
        .members()
        .iter()
        .find(|member| {
            matches!(
                member.sharding(),
                crate::MemberSharding::Partitioned { .. }
                    | crate::MemberSharding::PartitionedSegments { .. }
                    | crate::MemberSharding::PartitionedChunks { .. }
                    | crate::MemberSharding::PartitionedChunkSegments { .. }
            )
        })
        .ok_or_else(|| {
            crate::ParallelPlanError::InvalidGroup(format!(
                "parameter group {:?} has no partitioned member",
                group.logical_name(),
            ))
        })?;
    let invalid = || {
        crate::ParallelPlanError::InvalidTensor(format!(
            "parameter {:?} omits its declared logical partition",
            member.target(),
        ))
    };
    let tensor = layout.tensor(member.target()).ok_or_else(invalid)?;
    if tensor.logical_units() != Some(units) {
        return Err(invalid());
    }
    tensor
        .logical_range()
        .filter(|range| !range.is_empty() && range.end <= units)
        .cloned()
        .ok_or_else(invalid)
}

fn invalid_balanced_axis(
    group: &crate::ParameterGroupSpec,
    axis: usize,
) -> crate::ParallelPlanError {
    crate::ParallelPlanError::InvalidGroup(format!(
        "parameter group {:?} has no independent balanced axis {axis}",
        group.logical_name(),
    ))
}

fn balanced_axis_anchor(
    group: &crate::ParameterGroupSpec,
    axis: usize,
    global_extent: usize,
) -> Result<&crate::ParameterMemberSpec, crate::ParallelPlanError> {
    if group.partition_units().is_some() {
        return Err(invalid_balanced_axis(group, axis));
    }
    group
        .members()
        .iter()
        .find(|member| {
            member.sharding() == &(crate::MemberSharding::Balanced { axis })
                && member.global_shape().get(axis) == Some(&global_extent)
        })
        .ok_or_else(|| invalid_balanced_axis(group, axis))
}

/// Projects an optional physical group. A wholly absent group has no local range;
/// any retained member requires the complete group and its aligned companions.
pub fn balanced_parameter_group_range_if_present(
    group: &crate::ParameterGroupSpec,
    layout: &crate::LocalModelLayout,
    axis: usize,
    global_extent: usize,
) -> Result<Option<Range<usize>>, crate::ParallelPlanError> {
    balanced_axis_anchor(group, axis, global_extent)?;
    if group
        .members()
        .iter()
        .all(|member| layout.tensor(member.target()).is_none())
    {
        return Ok(None);
    }
    balanced_parameter_group_range(group, layout, axis, global_extent).map(Some)
}

/// Extracts and validates an independent balanced axis from its declared group.
/// Complete local storage and replication retain their distinct placement facts.
/// Replicated companions retain their complete shapes, independently of the axis.
/// The layout may contain other groups; only this group's declared targets are read.
pub fn balanced_parameter_group_range(
    group: &crate::ParameterGroupSpec,
    layout: &crate::LocalModelLayout,
    axis: usize,
    global_extent: usize,
) -> Result<Range<usize>, crate::ParallelPlanError> {
    let invalid = || invalid_balanced_axis(group, axis);
    let anchor = balanced_axis_anchor(group, axis, global_extent)?;
    let tensor = layout.tensor(anchor.target()).ok_or_else(invalid)?;
    let range = match tensor.placement() {
        crate::TensorPlacement::Range {
            axis: selected,
            start,
            end,
        } if *selected == axis => *start..*end,
        crate::TensorPlacement::Replicated | crate::TensorPlacement::Local => 0..global_extent,
        _ => return Err(invalid()),
    };
    let complete = range == (0..global_extent);
    for member in group.members() {
        let actual = layout.tensor(member.target()).ok_or_else(invalid)?;
        let (mut placement, shape) = resolve_balanced_member(member, axis, global_extent, &range)
            .map_err(crate::ParallelPlanError::InvalidTensor)?;
        if (complete || member.sharding() == &crate::MemberSharding::Replicated)
            && matches!(
                actual.placement(),
                crate::TensorPlacement::Local | crate::TensorPlacement::Replicated
            )
        {
            placement = actual.placement().clone();
        }
        let fallback = complete
            && placement == crate::TensorPlacement::Replicated
            && actual.fell_back_to_replication();
        let expected = local_tensor(group, member, shape, placement, None, fallback);
        if actual != &expected {
            return Err(crate::ParallelPlanError::InvalidTensor(format!(
                "parameter {:?} differs from declared balanced group {:?}",
                member.target(),
                group.logical_name(),
            )));
        }
    }
    Ok(range)
}

/// Family-selected geometry validated against one declared physical parameter group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParameterGroupSelection {
    /// Semantic group interval, including fused segments and explicit chunks.
    /// `None` selects a group containing only replicated members without a domain.
    Logical(Option<Range<usize>>),
    /// Explicit contiguous interval of an independently balanced tensor axis.
    /// Every sharded member declares the same axis and complete physical extent;
    /// declared replicas retain their full shapes. Rank and communicator admission
    /// remain separate: an uneven interval implies neither.
    BalancedAxis {
        /// Declared tensor axis.
        axis: usize,
        /// Complete physical axis extent shared by all members.
        global_extent: usize,
        /// Exact caller-selected interval on that axis.
        range: Range<usize>,
    },
}

/// Validates exact physical tensors against family-selected group geometry.
///
/// The shared lowering checks logical partitions, explicit balanced-axis ranges,
/// fused segments, chunks, replicas, shapes and companion metadata. The supplied
/// layout contains exactly these groups. No topology is inferred from a tensor's
/// shape or selected interval. Packed expert ownership uses rank-aware lowering.
pub fn validate_parameter_layout<'a, E>(
    groups: impl IntoIterator<Item = &'a crate::ParameterGroupSpec>,
    layout: &crate::LocalModelLayout,
    mut select: impl FnMut(&crate::ParameterGroupSpec) -> Result<ParameterGroupSelection, E>,
) -> Result<(), E>
where
    E: From<crate::ParallelPlanError>,
{
    let mut targets = std::collections::BTreeSet::new();
    for group in groups {
        let selected = select(group)?;
        if let ParameterGroupSelection::BalancedAxis { axis, .. } = &selected {
            if group.partition_units().is_some()
                || !group.members().iter().any(|member| {
                    member.sharding() == &(crate::MemberSharding::Balanced { axis: *axis })
                })
            {
                return Err(crate::ParallelPlanError::InvalidGroup(format!(
                    "parameter group {:?} does not declare independent balanced axis {axis}",
                    group.logical_name(),
                ))
                .into());
            }
        }
        let logical = match &selected {
            ParameterGroupSelection::Logical(range) => range.clone(),
            ParameterGroupSelection::BalancedAxis { .. } => None,
        };
        for member in group.members() {
            let (placement, shape) = match &selected {
                ParameterGroupSelection::Logical(range) => {
                    resolve_logical_member(member, group.partition_units(), range.as_ref())
                }
                ParameterGroupSelection::BalancedAxis {
                    axis,
                    global_extent,
                    range,
                } => resolve_balanced_member(member, *axis, *global_extent, range),
            }
            .map_err(crate::ParallelPlanError::InvalidTensor)?;
            let expected = local_tensor(group, member, shape, placement, logical.clone(), false);
            if !targets.insert(member.target()) {
                return Err(crate::ParallelPlanError::InvalidTensor(format!(
                    "parameter {:?} has duplicate physical ownership",
                    member.target(),
                ))
                .into());
            }
            if layout.tensor(member.target()) != Some(&expected) {
                return Err(crate::ParallelPlanError::InvalidTensor(format!(
                    "parameter {:?} differs from declared group {:?} placement",
                    member.target(),
                    group.logical_name(),
                ))
                .into());
            }
        }
    }
    if let Some((target, _)) = layout.tensors().find(|(name, _)| !targets.contains(name)) {
        return Err(crate::ParallelPlanError::InvalidTensor(format!(
            "parameter {target:?} has no declared physical owner",
        ))
        .into());
    }
    Ok(())
}

fn local_layout(
    description: &crate::ArchitectureParameterDescription,
    tensor_rank: usize,
    tensor_parts: usize,
    expert_rank: usize,
    expert_parts: usize,
) -> Result<crate::LocalModelLayout, String> {
    local_layout_groups(
        description.groups().iter(),
        tensor_rank,
        tensor_parts,
        expert_rank,
        expert_parts,
    )
}

fn local_layout_groups<'a>(
    groups: impl IntoIterator<Item = &'a crate::OwnedParameterGroupSpec>,
    tensor_rank: usize,
    tensor_parts: usize,
    expert_rank: usize,
    expert_parts: usize,
) -> Result<crate::LocalModelLayout, String> {
    if expert_parts == 0 || expert_rank >= expert_parts {
        return Err(format!(
            "invalid expert-parallel coordinate {expert_rank}/{expert_parts}"
        ));
    }
    let mut layout = crate::LocalModelLayout::default();
    for owned in groups {
        let group = owned.group();
        let (member_rank, member_parts) = group
            .partition_coordinates(tensor_rank, tensor_parts)
            .map_err(|error| error.to_string())?;
        // A routed expert bank is packed when at least one member carries the
        // leading expert dimension in addition to its matrix dimensions.  The
        // complete group shares that leading dimension, including rank-two
        // bias companions.  Rank-two shared-expert projections deliberately
        // do not satisfy this test: `ExpertIntermediate` also describes their
        // TP width, but they have no EP-owned packed-expert axis.
        let packed_expert_range = if matches!(
            group.role(),
            crate::ParameterRole::ExpertIntermediate | crate::ParameterRole::ExpertOutput
        ) && expert_parts > 1
            && group
                .members()
                .iter()
                .any(|member| member.global_shape().len() >= 3)
        {
            let global_experts = group
                .members()
                .first()
                .and_then(|member| member.global_shape().first())
                .copied()
                .ok_or_else(|| {
                    format!(
                        "expert parameter group {:?} has no packed expert axis",
                        group.logical_name()
                    )
                })?;
            if group
                .members()
                .iter()
                .any(|member| member.global_shape().first().copied() != Some(global_experts))
            {
                return Err(format!(
                    "expert parameter group {:?} does not share one packed expert axis",
                    group.logical_name()
                ));
            }
            Some(
                eredu_core::balanced_contiguous_range(
                    global_experts,
                    expert_parts,
                    expert_rank,
                    false,
                )
                .map_err(|error| error.to_string())?,
            )
        } else {
            None
        };
        let logical_range = group
            .partition_units()
            .map(|units| {
                eredu_core::balanced_contiguous_range(units, member_parts, member_rank, false)
            })
            .transpose()
            .map_err(|error| error.to_string())?;
        for member in group.members() {
            if layout.contains(member.target()) {
                return Err(format!(
                    "parallel placement target {:?} was registered more than once",
                    member.target()
                ));
            }
            let (placement, local_shape) = if let Some(range) = &packed_expert_range {
                let (tensor_placement, mut shape) =
                    resolve_member(member, group.partition_units(), member_rank, member_parts)?;
                shape[0] = range.len();
                let placement = if tensor_parts == 1 {
                    crate::TensorPlacement::Range {
                        axis: 0,
                        start: range.start,
                        end: range.end,
                    }
                } else {
                    // EP ownership is retained by the exact expert realization;
                    // this physical placement remains the member's TP task.
                    tensor_placement
                };
                (placement, shape)
            } else {
                resolve_member(member, group.partition_units(), member_rank, member_parts)?
            };
            let mut tensor = local_tensor(
                group,
                member,
                local_shape,
                placement,
                logical_range.clone(),
                false,
            );
            if tensor_parts > 1 {
                if let Some(range) = &packed_expert_range {
                    tensor = tensor.with_additional_placement(crate::TensorPlacement::Range {
                        axis: 0,
                        start: range.start,
                        end: range.end,
                    });
                }
            }
            layout.insert(member.target().to_owned(), tensor);
        }
    }
    Ok(layout)
}

fn local_tensor(
    group: &crate::ParameterGroupSpec,
    member: &crate::ParameterMemberSpec,
    shape: Vec<usize>,
    placement: crate::TensorPlacement,
    logical: Option<Range<usize>>,
    fell_back_to_replication: bool,
) -> crate::LocalTensorLayout {
    crate::LocalTensorLayout::new(
        group.logical_name(),
        group.role(),
        member.global_shape().to_vec(),
        shape,
        placement,
        group.partition_units(),
        logical,
        fell_back_to_replication,
    )
    .with_partition_chunk_size(match member.sharding() {
        crate::MemberSharding::PartitionedChunks { chunk_size, .. }
        | crate::MemberSharding::PartitionedChunkSegments { chunk_size, .. } => Some(*chunk_size),
        _ => None,
    })
}

fn resolve_member(
    member: &crate::ParameterMemberSpec,
    partition_units: Option<usize>,
    rank: usize,
    parts: usize,
) -> Result<(crate::TensorPlacement, Vec<usize>), String> {
    use crate::{MemberSharding, TensorPlacement};
    if parts == 0 || rank >= parts {
        return Err(format!("invalid tensor-parallel coordinate {rank}/{parts}"));
    }
    match member.sharding() {
        MemberSharding::Replicated => {
            Ok((TensorPlacement::Replicated, member.global_shape().to_vec()))
        }
        MemberSharding::Equal { axis } => {
            let dimension = member_axis(member, *axis)?;
            if !dimension.is_multiple_of(parts) {
                return Err(format!(
                    "tensor {:?} axis {axis} extent {dimension} is not divisible by {parts}",
                    member.target()
                ));
            }
            let width = dimension / parts;
            let mut shape = member.global_shape().to_vec();
            shape[*axis] = width;
            Ok((
                TensorPlacement::Shard {
                    axis: *axis,
                    index: rank,
                    parts,
                },
                shape,
            ))
        }
        MemberSharding::Balanced { axis } => {
            let dimension = member_axis(member, *axis)?;
            let range = eredu_core::balanced_contiguous_range(dimension, parts, rank, false)
                .map_err(|error| error.to_string())?;
            resolve_balanced_member(member, *axis, dimension, &range)
        }
        MemberSharding::Partitioned { .. }
        | MemberSharding::PartitionedChunks { .. }
        | MemberSharding::PartitionedSegments { .. }
        | MemberSharding::PartitionedChunkSegments { .. } => {
            let units = partition_units
                .ok_or_else(|| format!("tensor {:?} has no logical partition", member.target()))?;
            let logical = eredu_core::balanced_contiguous_range(units, parts, rank, false)
                .map_err(|error| error.to_string())?;
            resolve_logical_member(member, Some(units), Some(&logical))
        }
        MemberSharding::Segmented { axis, segments } => {
            let indices = segmented_indices(member, *axis, segments, |segment| {
                let local =
                    eredu_core::balanced_contiguous_range(segment.len(), parts, rank, false)
                        .map_err(|error| error.to_string())?;
                Ok(segment.start + local.start..segment.start + local.end)
            })?;
            indexed(member, *axis, indices)
        }
    }
}

fn ranged(
    member: &crate::ParameterMemberSpec,
    axis: usize,
    range: Range<usize>,
) -> (crate::TensorPlacement, Vec<usize>) {
    let mut shape = member.global_shape().to_vec();
    shape[axis] = range.len();
    (
        crate::TensorPlacement::Range {
            axis,
            start: range.start,
            end: range.end,
        },
        shape,
    )
}

fn resolve_balanced_member(
    member: &crate::ParameterMemberSpec,
    axis: usize,
    global_extent: usize,
    range: &Range<usize>,
) -> Result<(crate::TensorPlacement, Vec<usize>), String> {
    let replicated = member.sharding() == &crate::MemberSharding::Replicated;
    if (!replicated
        && (member.sharding() != &(crate::MemberSharding::Balanced { axis })
            || member_axis(member, axis)? != global_extent))
        || global_extent == 0
        || range.is_empty()
        || range.end > global_extent
    {
        return Err(format!(
            "tensor {:?} does not preserve axis {axis} extent {global_extent} selection {range:?}",
            member.target(),
        ));
    }
    if replicated {
        Ok((
            crate::TensorPlacement::Replicated,
            member.global_shape().to_vec(),
        ))
    } else {
        Ok(ranged(member, axis, range.clone()))
    }
}

fn resolve_logical_member(
    member: &crate::ParameterMemberSpec,
    units: Option<usize>,
    logical: Option<&Range<usize>>,
) -> Result<(crate::TensorPlacement, Vec<usize>), String> {
    use crate::MemberSharding;
    match (units, logical) {
        (Some(units), Some(range)) if units > 0 && !range.is_empty() && range.end <= units => {}
        (None, None) if matches!(member.sharding(), MemberSharding::Replicated) => {}
        _ => {
            return Err(format!(
                "tensor {:?} has an invalid logical partition range",
                member.target()
            ))
        }
    }
    match member.sharding() {
        MemberSharding::Replicated => Ok((
            crate::TensorPlacement::Replicated,
            member.global_shape().to_vec(),
        )),
        MemberSharding::Partitioned { axis } => {
            let units = units.expect("validated logical partition");
            let logical = logical.expect("validated logical selection");
            let dimension = member_axis(member, *axis)?;
            if !dimension.is_multiple_of(units) {
                return Err(format!(
                    "tensor {:?} axis {axis} does not contain {units} logical units",
                    member.target()
                ));
            }
            let width = dimension / units;
            Ok(ranged(
                member,
                *axis,
                logical.start * width..logical.end * width,
            ))
        }
        MemberSharding::PartitionedChunks { axis, chunk_size } => {
            let units = units.expect("validated logical partition");
            let logical = logical.expect("validated logical selection");
            let dimension = member_axis(member, *axis)?;
            if *chunk_size == 0 || dimension.div_ceil(*chunk_size) != units {
                return Err(format!(
                    "tensor {:?} chunks do not match {units} logical units",
                    member.target()
                ));
            }
            let range = crate::partition_chunk_range(dimension, *chunk_size, logical.clone())
                .map_err(|error| error.to_string())?;
            Ok(ranged(member, *axis, range))
        }
        MemberSharding::PartitionedChunkSegments {
            axis,
            segments,
            chunk_size,
        } => {
            let units = units.expect("validated logical partition");
            let logical = logical.expect("validated logical selection");
            let indices = segmented_indices(member, *axis, segments, |segment| {
                if *chunk_size == 0 || segment.len().div_ceil(*chunk_size) != units {
                    return Err(format!(
                        "segment {segment:?} chunks do not match {units} logical units"
                    ));
                }
                let local =
                    crate::partition_chunk_range(segment.len(), *chunk_size, logical.clone())
                        .map_err(|error| error.to_string())?;
                Ok(segment.start + local.start..segment.start + local.end)
            })?;
            indexed(member, *axis, indices)
        }
        MemberSharding::PartitionedSegments { axis, segments } => {
            let units = units.expect("validated logical partition");
            let logical = logical.expect("validated logical selection");
            let indices = segmented_indices(member, *axis, segments, |segment| {
                if !segment.len().is_multiple_of(units) {
                    return Err(format!(
                        "segment {segment:?} does not contain {units} units"
                    ));
                }
                let width = segment.len() / units;
                Ok(segment.start + logical.start * width..segment.start + logical.end * width)
            })?;
            indexed(member, *axis, indices)
        }
        _ => Err(format!(
            "tensor {:?} does not declare a logical partition",
            member.target()
        )),
    }
}

fn member_axis(member: &crate::ParameterMemberSpec, axis: usize) -> Result<usize, String> {
    member
        .global_shape()
        .get(axis)
        .copied()
        .ok_or_else(|| format!("tensor {:?} has no sharding axis {axis}", member.target()))
}

fn segmented_indices(
    member: &crate::ParameterMemberSpec,
    axis: usize,
    segments: &[Range<usize>],
    mut select: impl FnMut(&Range<usize>) -> Result<Range<usize>, String>,
) -> Result<Vec<usize>, String> {
    let dimension = member_axis(member, axis)?;
    let mut previous_end = 0;
    let mut indices = Vec::new();
    if segments.is_empty() {
        return Err(format!(
            "tensor {:?} has no sharding segments",
            member.target()
        ));
    }
    for segment in segments {
        if segment.start >= segment.end || segment.end > dimension || segment.start < previous_end {
            return Err(format!(
                "tensor {:?} has invalid segment {segment:?}",
                member.target()
            ));
        }
        previous_end = segment.end;
        indices.extend(select(segment)?);
    }
    Ok(indices)
}

fn indexed(
    member: &crate::ParameterMemberSpec,
    axis: usize,
    indices: Vec<usize>,
) -> Result<(crate::TensorPlacement, Vec<usize>), String> {
    if indices.is_empty() {
        return Err(format!("tensor {:?} has no local indices", member.target()));
    }
    let mut shape = member.global_shape().to_vec();
    shape[axis] = indices.len();
    Ok((crate::TensorPlacement::Indices { axis, indices }, shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_logical_units_keep_complete_dense_and_nvfp4_companions() {
        use crate::{
            MemberSharding, OwnedParameterGroupSpec, ParameterGroupOwner, ParameterGroupSpec,
            ParameterMemberSpec, ParameterRole, PartitionUnitAssignment, TensorPlacement,
        };
        let distinct = ParameterGroupSpec::partitioned(
            "kv",
            ParameterRole::AttentionHeads,
            2,
            ["key.weight", "value.weight"].map(|name| {
                ParameterMemberSpec::new(name, [32, 32], MemberSharding::Partitioned { axis: 0 })
            }),
        )
        .unwrap();
        let owned =
            |group| OwnedParameterGroupSpec::new(ParameterGroupOwner::static_role("kv"), group);
        assert!(local_layout_groups([&owned(distinct.clone())], 0, 4, 0, 1).is_err());
        let uneven = ParameterGroupSpec::partitioned(
            "query",
            ParameterRole::AttentionHeads,
            3,
            [ParameterMemberSpec::new(
                "query.weight",
                [48, 32],
                MemberSharding::Partitioned { axis: 0 },
            )],
        )
        .unwrap();
        assert!(local_layout_groups([&owned(uneven.clone())], 0, 2, 0, 1).is_ok());
        let even = uneven
            .with_partition_unit_assignment(PartitionUnitAssignment::Even)
            .unwrap();
        assert!(local_layout_groups([&owned(even)], 0, 2, 0, 1).is_err());
        let repeated = distinct
            .with_partition_unit_assignment(PartitionUnitAssignment::Repeat { semantic_units: 2 })
            .unwrap();
        assert!(local_layout_groups([&owned(repeated.clone())], 0, 3, 0, 1).is_err());
        for packed in [false, true] {
            let groups = if packed {
                crate::expand_linear_format_parameter_groups(vec![repeated.clone()], |member| {
                    Ok(Some(
                        eredu_nn::LinearFormatSpec::nvfp4(
                            eredu_nn::ParameterSpec::trainable(format!(
                                "{}.blocks",
                                member.target()
                            ))
                            .unwrap(),
                            eredu_nn::ParameterSpec::trainable(format!(
                                "{}.global",
                                member.target()
                            ))
                            .unwrap(),
                        )
                        .unwrap(),
                    ))
                })
                .unwrap()
            } else {
                vec![repeated.clone()]
            };
            let group = owned(groups.into_iter().next().unwrap());
            for rank in 0..4 {
                let layout = local_layout_groups([&group], rank, 4, 0, 1).unwrap();
                for name in ["key.weight", "value.weight"] {
                    let tensor = layout.tensor(name).unwrap();
                    let (start, end) = if rank < 2 { (0, 16) } else { (16, 32) };
                    assert_eq!(tensor.local_shape(), [16, if packed { 4 } else { 32 }]);
                    assert_eq!(
                        tensor.placement(),
                        &TensorPlacement::Range {
                            axis: 0,
                            start,
                            end
                        }
                    );
                    assert_eq!(tensor.logical_range(), Some(&(rank / 2..rank / 2 + 1)));
                    if packed {
                        let blocks = layout.tensor(&format!("{name}.blocks")).unwrap();
                        assert_eq!(blocks.local_shape(), [16, 2]);
                        assert_eq!(blocks.placement(), tensor.placement());
                        assert_eq!(
                            layout
                                .tensor(&format!("{name}.global"))
                                .unwrap()
                                .placement(),
                            &TensorPlacement::Replicated
                        );
                    }
                }
            }
        }
        // Coarsening a two-head domain into one physical block must not silently
        // turn head replication into replication of the complete projection.
        let coarsened = crate::expand_linear_format_parameter_groups(vec![repeated], |member| {
            Ok(Some(
                eredu_nn::LinearFormatSpec::scaled(
                    eredu_checkpoint::LinearFormat::E4M3BlockFp8(
                        eredu_checkpoint::BlockFp8Format::new(
                            32,
                            32,
                            eredu_checkpoint::BlockFp8ScaleEncoding::FloatingPoint,
                        )
                        .unwrap(),
                    ),
                    eredu_nn::ParameterSpec::trainable(format!("{}.scale", member.target()))
                        .unwrap(),
                )
                .unwrap(),
            ))
        })
        .unwrap();
        let group = owned(coarsened.into_iter().next().unwrap());
        assert!(local_layout_groups([&group], 0, 1, 0, 1).is_ok());
        assert!(local_layout_groups([&group], 0, 4, 0, 1).is_err());
    }

    #[test]
    fn physical_chunk_placement_covers_short_tail_and_matching_scales() {
        use crate::{MemberSharding, ParameterMemberSpec, TensorPlacement};
        for (shape, axis, chunk, expected) in [
            ([259, 130], 0, 128, [0..256, 256..259]),
            ([130, 259], 1, 128, [0..256, 256..259]),
            ([3, 2], 0, 1, [0..2, 2..3]),
            ([2, 3], 1, 1, [0..2, 2..3]),
        ] {
            let member = ParameterMemberSpec::new(
                "opaque",
                shape,
                MemberSharding::PartitionedChunks {
                    axis,
                    chunk_size: chunk,
                },
            );
            for (rank, range) in expected.into_iter().enumerate() {
                let (placement, local) = resolve_member(&member, Some(3), rank, 2).unwrap();
                assert_eq!(
                    placement,
                    TensorPlacement::Range {
                        axis,
                        start: range.start,
                        end: range.end
                    }
                );
                assert_eq!(local[axis], range.len());
                assert_eq!(local[1 - axis], shape[1 - axis]);
            }
            assert!(resolve_member(&member, Some(2), 0, 2).is_err());
            assert!(resolve_member(&member, Some(3), 0, 4).is_err());
        }
    }

    #[test]
    fn physical_chunk_segments_preserve_each_fused_half_tail() {
        use crate::{MemberSharding, ParameterMemberSpec, TensorPlacement};
        let member = ParameterMemberSpec::new(
            "fused",
            [518, 130],
            MemberSharding::PartitionedChunkSegments {
                axis: 0,
                segments: vec![0..259, 259..518],
                chunk_size: 128,
            },
        );
        for (rank, expected) in [
            (0, (0..256).chain(259..515).collect::<Vec<_>>()),
            (1, vec![256, 257, 258, 515, 516, 517]),
        ] {
            let (placement, shape) = resolve_member(&member, Some(3), rank, 2).unwrap();
            assert_eq!(shape, [expected.len(), 130]);
            assert_eq!(
                placement,
                TensorPlacement::Indices {
                    axis: 0,
                    indices: expected
                }
            );
        }
    }

    #[test]
    fn local_layout_preserves_exact_tp_ranges() {
        use crate::{
            ArchitectureParameterDescription, ExecutionGraph, ExecutionGroupSpec,
            ExecutionUnitLayout, MemberSharding, OwnedParameterGroupSpec, ParameterGroupOwner,
            ParameterGroupSpec, ParameterMemberSpec, ParameterRole, TensorPlacement,
        };

        let graph =
            ExecutionGraph::new(vec![ExecutionGroupSpec::root("decoder")], "decoder").unwrap();
        let units = ExecutionUnitLayout::new(&graph, [1]).unwrap();
        let groups = [
            ParameterGroupSpec::new(
                "query",
                ParameterRole::AttentionHeads,
                [ParameterMemberSpec::new(
                    "q.weight",
                    vec![8, 8],
                    MemberSharding::Equal { axis: 0 },
                )],
            )
            .unwrap(),
            ParameterGroupSpec::new(
                "output",
                ParameterRole::RowProjection,
                [ParameterMemberSpec::new(
                    "o.weight",
                    vec![8, 8],
                    MemberSharding::Equal { axis: 1 },
                )],
            )
            .unwrap(),
            ParameterGroupSpec::new(
                "embedding",
                ParameterRole::Vocabulary,
                [ParameterMemberSpec::new(
                    "embed.weight",
                    vec![15, 8],
                    MemberSharding::Balanced { axis: 0 },
                )],
            )
            .unwrap(),
        ];
        let owned = groups
            .iter()
            .cloned()
            .map(|group| {
                OwnedParameterGroupSpec::new(ParameterGroupOwner::static_role("embedding"), group)
            })
            .collect::<Vec<_>>();
        let description =
            ArchitectureParameterDescription::new(&graph, &units, groups, owned).unwrap();
        let layout = local_layout(&description, 1, 2, 0, 1).unwrap();

        assert_eq!(layout.tensor("q.weight").unwrap().local_shape(), [4, 8]);
        assert_eq!(layout.tensor("o.weight").unwrap().local_shape(), [8, 4]);
        assert_eq!(layout.tensor("embed.weight").unwrap().local_shape(), [7, 8]);
        assert_eq!(
            layout.tensor("embed.weight").unwrap().placement(),
            &TensorPlacement::Range {
                axis: 0,
                start: 8,
                end: 15,
            }
        );
    }

    #[test]
    fn local_layout_projects_packed_experts_to_the_exact_ep_owner_range() {
        use crate::{
            ArchitectureParameterDescription, ExecutionGraph, ExecutionGroupSpec,
            ExecutionUnitLayout, MemberSharding, OwnedParameterGroupSpec, ParameterGroupOwner,
            ParameterGroupSpec, ParameterMemberSpec, ParameterRole, TensorPlacement,
        };

        let graph =
            ExecutionGraph::new(vec![ExecutionGroupSpec::root("decoder")], "decoder").unwrap();
        let units = ExecutionUnitLayout::new(&graph, [1]).unwrap();
        let group = ParameterGroupSpec::new(
            "experts.intermediate",
            ParameterRole::ExpertIntermediate,
            [
                ParameterMemberSpec::new(
                    "experts.gate_up_proj",
                    vec![5, 16, 8],
                    MemberSharding::Segmented {
                        axis: 1,
                        segments: vec![0..8, 8..16],
                    },
                ),
                ParameterMemberSpec::new(
                    "experts.down_proj",
                    vec![5, 8, 8],
                    MemberSharding::Equal { axis: 2 },
                ),
                ParameterMemberSpec::new(
                    "experts.gate_up_proj_bias",
                    vec![5, 16],
                    MemberSharding::Segmented {
                        axis: 1,
                        segments: vec![0..8, 8..16],
                    },
                ),
                ParameterMemberSpec::new(
                    "experts.down_proj_bias",
                    vec![5, 8],
                    MemberSharding::Replicated,
                ),
            ],
        )
        .unwrap();
        let shared = ParameterGroupSpec::new(
            "shared_expert.intermediate",
            ParameterRole::ExpertIntermediate,
            [ParameterMemberSpec::new(
                "shared_expert.gate_proj",
                vec![16, 8],
                MemberSharding::Equal { axis: 0 },
            )],
        )
        .unwrap();
        let description = ArchitectureParameterDescription::new(
            &graph,
            &units,
            [group.clone(), shared.clone()],
            [
                OwnedParameterGroupSpec::new(ParameterGroupOwner::static_role("experts"), group),
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::static_role("shared_expert"),
                    shared,
                ),
            ],
        )
        .unwrap();

        let layout = local_layout(&description, 0, 1, 1, 2).unwrap();
        let expert = layout.tensor("experts.gate_up_proj").unwrap();
        assert_eq!(expert.local_shape(), [2, 16, 8]);
        assert_eq!(
            expert.placement(),
            &TensorPlacement::Range {
                axis: 0,
                start: 3,
                end: 5,
            }
        );

        let layout = local_layout(&description, 1, 2, 0, 2).unwrap();
        let gate_up = layout.tensor("experts.gate_up_proj").unwrap();
        assert_eq!(gate_up.local_shape(), [3, 8, 8]);
        assert_eq!(
            gate_up.placement(),
            &TensorPlacement::Indices {
                axis: 1,
                indices: vec![4, 5, 6, 7, 12, 13, 14, 15],
            }
        );
        assert_eq!(
            gate_up.additional_placements(),
            [TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 3,
            }]
        );
        assert_eq!(
            layout.tensor("experts.down_proj").unwrap().local_shape(),
            [3, 8, 4]
        );
        assert_eq!(
            layout
                .tensor("experts.gate_up_proj_bias")
                .unwrap()
                .local_shape(),
            [3, 8]
        );
        let down_bias = layout.tensor("experts.down_proj_bias").unwrap();
        assert_eq!(down_bias.local_shape(), [3, 8]);
        assert_eq!(down_bias.placement(), &TensorPlacement::Replicated);
        assert_eq!(
            down_bias.additional_placements(),
            [TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 3,
            }]
        );

        let shared = layout.tensor("shared_expert.gate_proj").unwrap();
        assert_eq!(shared.local_shape(), [8, 8]);
        assert_eq!(
            shared.placement(),
            &TensorPlacement::Shard {
                axis: 0,
                index: 1,
                parts: 2,
            }
        );

        let malformed = ParameterGroupSpec::new(
            "malformed.experts",
            ParameterRole::ExpertIntermediate,
            [
                ParameterMemberSpec::new(
                    "malformed.gate_up",
                    vec![5, 16, 8],
                    MemberSharding::Equal { axis: 1 },
                ),
                ParameterMemberSpec::new(
                    "malformed.down_bias",
                    vec![4, 8],
                    MemberSharding::Replicated,
                ),
            ],
        )
        .unwrap();
        let malformed = ArchitectureParameterDescription::new(
            &graph,
            &units,
            [malformed.clone()],
            [OwnedParameterGroupSpec::new(
                ParameterGroupOwner::static_role("malformed"),
                malformed,
            )],
        )
        .unwrap();
        let error = local_layout(&malformed, 0, 2, 0, 2).unwrap_err();
        assert!(error.contains("does not share one packed expert axis"));
    }
}

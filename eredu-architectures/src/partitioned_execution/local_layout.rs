//! Source destinations for the existing parameter-placement worker.
use super::*;
use eredu_core::{HostMetadataFunding, HostMetadataFundingError};

use super::source_allocation::{Allocation, Cause};

/// A genuinely constructed parameter-layout source with its original account.
#[derive(Debug)]
pub struct PreparedLocalModelLayout {
    layout: eredu_runtime::LocalModelLayout,
    _funding: HostMetadataFunding,
}
impl PreparedLocalModelLayout {
    /// Borrow the exact rank-local tensor declarations; this grants no native allocation.
    pub fn layout(&self) -> &eredu_runtime::LocalModelLayout {
        &self.layout
    }
}
/// A failed layout constructor retains its account through all diagnostics.
#[derive(Debug, thiserror::Error)]
#[error("parameter layout source: {cause}")]
pub struct LocalModelLayoutConstructionError {
    #[source]
    cause: Cause,
    _funding: HostMetadataFunding,
}
/// Lower the original parameter source through the same placement worker while
/// funding each control, shape, name, index vector and ordered node before use.
pub fn derive_partitioned_local_layout_with_funding(
    description: &eredu_runtime::ArchitectureParameterDescription,
    topology: eredu_core::ParallelRankTopology,
    funding: HostMetadataFunding,
) -> Result<PreparedLocalModelLayout, LocalModelLayoutConstructionError> {
    let allocation = Allocation(Some(&funding));
    let result = (|| {
        allocation.controls::<(
            PreparedLocalModelLayout,
            LocalModelLayoutConstructionError,
            &eredu_runtime::ArchitectureParameterDescription,
            eredu_core::ParallelRankTopology,
        )>()?;
        local_layout_worker(
            description,
            topology.tensor_parallel_rank(),
            topology.tensor_parallel_size(),
            topology.expert_parallel_rank(),
            topology.expert_parallel_size(),
            allocation,
        )
    })();
    match result {
        Ok(layout) => Ok(PreparedLocalModelLayout {
            layout,
            _funding: funding,
        }),
        Err(cause) => Err(LocalModelLayoutConstructionError {
            cause,
            _funding: funding,
        }),
    }
}

pub(super) fn local_layout(
    description: &eredu_runtime::ArchitectureParameterDescription,
    tensor_rank: usize,
    tensor_parts: usize,
    expert_rank: usize,
    expert_parts: usize,
) -> Result<eredu_runtime::LocalModelLayout, String> {
    local_layout_worker(
        description,
        tensor_rank,
        tensor_parts,
        expert_rank,
        expert_parts,
        Allocation(None),
    )
    .map_err(Cause::ordinary)
}
pub(super) fn local_layout_groups<'a>(
    groups: impl IntoIterator<Item = &'a eredu_runtime::OwnedParameterGroupSpec>,
    tensor_rank: usize,
    tensor_parts: usize,
    expert_rank: usize,
    expert_parts: usize,
) -> Result<eredu_runtime::LocalModelLayout, String> {
    local_layout_groups_worker(
        groups,
        tensor_rank,
        tensor_parts,
        expert_rank,
        expert_parts,
        Allocation(None),
    )
    .map_err(Cause::ordinary)
}
#[cfg(test)]
pub(super) fn resolve_member(
    member: &eredu_runtime::ParameterMemberSpec,
    units: Option<usize>,
    rank: usize,
    parts: usize,
) -> Result<(eredu_runtime::TensorPlacement, Vec<usize>), String> {
    resolve_member_worker(member, units, rank, parts, Allocation(None)).map_err(Cause::ordinary)
}

pub(crate) fn local_layout_worker(
    description: &eredu_runtime::ArchitectureParameterDescription,
    tensor_rank: usize,
    tensor_parts: usize,
    expert_rank: usize,
    expert_parts: usize,
    allocation: Allocation<'_>,
) -> Result<eredu_runtime::LocalModelLayout, Cause> {
    local_layout_groups_worker(
        description.groups().iter(),
        tensor_rank,
        tensor_parts,
        expert_rank,
        expert_parts,
        allocation,
    )
}

fn local_layout_groups_worker<'a>(
    groups: impl IntoIterator<Item = &'a eredu_runtime::OwnedParameterGroupSpec>,
    tensor_rank: usize,
    tensor_parts: usize,
    expert_rank: usize,
    expert_parts: usize,
    allocation: Allocation<'_>,
) -> Result<eredu_runtime::LocalModelLayout, Cause> {
    allocation.controls::<(
        eredu_runtime::LocalModelLayout,
        &eredu_runtime::OwnedParameterGroupSpec,
        &eredu_runtime::ParameterGroupSpec,
        &eredu_runtime::ParameterMemberSpec,
        eredu_runtime::LocalTensorLayout,
        eredu_runtime::TensorPlacement,
        Vec<usize>,
        Option<Range<usize>>,
        std::slice::Iter<'_, eredu_runtime::ParameterMemberSpec>,
        [usize; 4],
    )>()?;
    let groups = groups.into_iter();
    allocation.controls_of(&groups)?;
    if expert_parts == 0 || expert_rank >= expert_parts {
        return Err(allocation.error(format_args!(
            "invalid expert-parallel coordinate {expert_rank}/{expert_parts}"
        )));
    }
    let mut layout = eredu_runtime::LocalModelLayout::default();
    for owned in groups {
        let group = owned.group();
        // A routed expert bank is packed when at least one member carries the
        // leading expert dimension in addition to its matrix dimensions.  The
        // complete group shares that leading dimension, including rank-two
        // bias companions.  Rank-two shared-expert projections deliberately
        // do not satisfy this test: `ExpertIntermediate` also describes their
        // TP width, but they have no EP-owned packed-expert axis.
        let packed_expert_range = if matches!(
            group.role(),
            eredu_runtime::ParameterRole::ExpertIntermediate
                | eredu_runtime::ParameterRole::ExpertOutput
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
                    allocation.error(format_args!(
                        "expert parameter group {:?} has no packed expert axis",
                        group.logical_name()
                    ))
                })?;
            if group
                .members()
                .iter()
                .any(|member| member.global_shape().first().copied() != Some(global_experts))
            {
                return Err(allocation.error(format_args!(
                    "expert parameter group {:?} does not share one packed expert axis",
                    group.logical_name()
                )));
            }
            Some(
                eredu_core::balanced_contiguous_range(
                    global_experts,
                    expert_parts,
                    expert_rank,
                    false,
                )
                .map_err(|error| allocation.error(format_args!("{error}")))?,
            )
        } else {
            None
        };
        let logical_range = group
            .partition_units()
            .map(|units| {
                eredu_core::balanced_contiguous_range(units, tensor_parts, tensor_rank, false)
            })
            .transpose()
            .map_err(|error| allocation.error(format_args!("{error}")))?;
        for member in group.members() {
            if layout.contains(member.target()) {
                return Err(allocation.error(format_args!(
                    "parallel placement target {:?} was registered more than once",
                    member.target()
                )));
            }
            let (placement, local_shape) = if let Some(range) = &packed_expert_range {
                let (tensor_placement, mut shape) = resolve_member_worker(
                    member,
                    group.partition_units(),
                    tensor_rank,
                    tensor_parts,
                    allocation,
                )?;
                shape[0] = range.len();
                let placement = if tensor_parts == 1 {
                    eredu_runtime::TensorPlacement::Range {
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
                resolve_member_worker(
                    member,
                    group.partition_units(),
                    tensor_rank,
                    tensor_parts,
                    allocation,
                )?
            };
            let mut tensor = eredu_runtime::LocalTensorLayout::new(
                allocation.text(group.logical_name())?,
                group.role(),
                allocation.copy(member.global_shape())?,
                local_shape,
                placement,
                group.partition_units(),
                logical_range.clone(),
                false,
            )
            .with_partition_chunk_size(match member.sharding() {
                eredu_runtime::MemberSharding::PartitionedChunks { chunk_size, .. }
                | eredu_runtime::MemberSharding::PartitionedChunkSegments { chunk_size, .. } => {
                    Some(*chunk_size)
                }
                _ => None,
            });
            if tensor_parts > 1 {
                if let Some(range) = &packed_expert_range {
                    tensor = allocation.additional(
                        tensor,
                        eredu_runtime::TensorPlacement::Range {
                            axis: 0,
                            start: range.start,
                            end: range.end,
                        },
                    )?;
                }
            }
            allocation.insert(&mut layout, allocation.text(member.target())?, tensor)?;
        }
    }
    Ok(layout)
}

fn resolve_member_worker(
    member: &eredu_runtime::ParameterMemberSpec,
    partition_units: Option<usize>,
    rank: usize,
    parts: usize,
    allocation: Allocation<'_>,
) -> Result<(eredu_runtime::TensorPlacement, Vec<usize>), Cause> {
    use eredu_runtime::{MemberSharding, TensorPlacement};
    allocation.controls::<(
        &eredu_runtime::ParameterMemberSpec,
        Option<usize>,
        [usize; 3],
        Range<usize>,
        TensorPlacement,
        Vec<usize>,
        Result<(TensorPlacement, Vec<usize>), Cause>,
    )>()?;
    if parts == 0 || rank >= parts {
        return Err(allocation.error(format_args!(
            "invalid tensor-parallel coordinate {rank}/{parts}"
        )));
    }
    let ranged = |axis: usize, range: Range<usize>| {
        let mut shape = allocation.copy(member.global_shape())?;
        shape[axis] = range.len();
        Ok::<_, Cause>((
            TensorPlacement::Range {
                axis,
                start: range.start,
                end: range.end,
            },
            shape,
        ))
    };
    match member.sharding() {
        MemberSharding::Replicated => Ok((
            TensorPlacement::Replicated,
            allocation.copy(member.global_shape())?,
        )),
        MemberSharding::Equal { axis } => {
            let dimension = member_axis(member, *axis, allocation)?;
            if !dimension.is_multiple_of(parts) {
                return Err(allocation.error(format_args!(
                    "tensor {:?} axis {axis} extent {dimension} is not divisible by {parts}",
                    member.target()
                )));
            }
            let width = dimension / parts;
            let mut shape = allocation.copy(member.global_shape())?;
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
            let dimension = member_axis(member, *axis, allocation)?;
            let range = eredu_core::balanced_contiguous_range(dimension, parts, rank, false)
                .map_err(|error| allocation.error(format_args!("{error}")))?;
            ranged(*axis, range)
        }
        MemberSharding::Partitioned { axis } => {
            let units = partition_units.ok_or_else(|| {
                allocation.error(format_args!(
                    "tensor {:?} has no logical partition",
                    member.target()
                ))
            })?;
            let dimension = member_axis(member, *axis, allocation)?;
            if !dimension.is_multiple_of(units) {
                return Err(allocation.error(format_args!(
                    "tensor {:?} axis {axis} does not contain {units} logical units",
                    member.target()
                )));
            }
            let logical = eredu_core::balanced_contiguous_range(units, parts, rank, false)
                .map_err(|error| allocation.error(format_args!("{error}")))?;
            let width = dimension / units;
            ranged(*axis, logical.start * width..logical.end * width)
        }
        MemberSharding::PartitionedChunks { axis, chunk_size } => {
            let units = partition_units.ok_or_else(|| {
                allocation.error(format_args!(
                    "tensor {:?} has no logical partition",
                    member.target()
                ))
            })?;
            let dimension = member_axis(member, *axis, allocation)?;
            if *chunk_size == 0 || dimension.div_ceil(*chunk_size) != units {
                return Err(allocation.error(format_args!(
                    "tensor {:?} chunks do not match {units} logical units",
                    member.target()
                )));
            }
            let logical = eredu_core::balanced_contiguous_range(units, parts, rank, false)
                .map_err(|error| allocation.error(format_args!("{error}")))?;
            let range = eredu_runtime::partition_chunk_range(dimension, *chunk_size, logical)
                .map_err(|error| allocation.error(format_args!("{error}")))?;
            ranged(*axis, range)
        }
        MemberSharding::PartitionedChunkSegments {
            axis,
            segments,
            chunk_size,
        } => {
            let units = partition_units.ok_or_else(|| {
                allocation.error(format_args!(
                    "tensor {:?} has no logical partition",
                    member.target()
                ))
            })?;
            let logical = eredu_core::balanced_contiguous_range(units, parts, rank, false)
                .map_err(|error| allocation.error(format_args!("{error}")))?;
            let indices = segmented_indices(
                member,
                *axis,
                segments,
                |segment| {
                    if *chunk_size == 0 || segment.len().div_ceil(*chunk_size) != units {
                        return Err(allocation.error(format_args!(
                            "segment {segment:?} chunks do not match {units} logical units"
                        )));
                    }
                    let local = eredu_runtime::partition_chunk_range(
                        segment.len(),
                        *chunk_size,
                        logical.clone(),
                    )
                    .map_err(|error| allocation.error(format_args!("{error}")))?;
                    Ok(segment.start + local.start..segment.start + local.end)
                },
                allocation,
            )?;
            indexed(member, *axis, indices, allocation)
        }
        MemberSharding::PartitionedSegments { axis, segments } => {
            let units = partition_units.ok_or_else(|| {
                allocation.error(format_args!(
                    "tensor {:?} has no logical partition",
                    member.target()
                ))
            })?;
            let logical = eredu_core::balanced_contiguous_range(units, parts, rank, false)
                .map_err(|error| allocation.error(format_args!("{error}")))?;
            let indices = segmented_indices(
                member,
                *axis,
                segments,
                |segment| {
                    if !segment.len().is_multiple_of(units) {
                        return Err(allocation.error(format_args!(
                            "segment {segment:?} does not contain {units} units"
                        )));
                    }
                    let width = segment.len() / units;
                    Ok(segment.start + logical.start * width..segment.start + logical.end * width)
                },
                allocation,
            )?;
            indexed(member, *axis, indices, allocation)
        }
        MemberSharding::Segmented { axis, segments } => {
            let indices = segmented_indices(
                member,
                *axis,
                segments,
                |segment| {
                    let local =
                        eredu_core::balanced_contiguous_range(segment.len(), parts, rank, false)
                            .map_err(|error| allocation.error(format_args!("{error}")))?;
                    Ok(segment.start + local.start..segment.start + local.end)
                },
                allocation,
            )?;
            indexed(member, *axis, indices, allocation)
        }
    }
}

fn member_axis(
    member: &eredu_runtime::ParameterMemberSpec,
    axis: usize,
    allocation: Allocation<'_>,
) -> Result<usize, Cause> {
    member.global_shape().get(axis).copied().ok_or_else(|| {
        allocation.error(format_args!(
            "tensor {:?} has no sharding axis {axis}",
            member.target()
        ))
    })
}

fn segmented_indices(
    member: &eredu_runtime::ParameterMemberSpec,
    axis: usize,
    segments: &[Range<usize>],
    mut select: impl FnMut(&Range<usize>) -> Result<Range<usize>, Cause>,
    allocation: Allocation<'_>,
) -> Result<Vec<usize>, Cause> {
    allocation.controls::<(
        &eredu_runtime::ParameterMemberSpec,
        &[Range<usize>],
        Range<usize>,
        std::slice::Iter<'_, Range<usize>>,
        Vec<usize>,
        usize,
    )>()?;
    let dimension = member_axis(member, axis, allocation)?;
    let mut previous_end = 0;
    let mut indices = Vec::new();
    if segments.is_empty() {
        return Err(allocation.error(format_args!(
            "tensor {:?} has no sharding segments",
            member.target()
        )));
    }
    for segment in segments {
        if segment.start >= segment.end || segment.end > dimension || segment.start < previous_end {
            return Err(allocation.error(format_args!(
                "tensor {:?} has invalid segment {segment:?}",
                member.target()
            )));
        }
        previous_end = segment.end;
        let selected = select(segment)?;
        allocation.grow(&mut indices, selected.len())?;
        indices.extend(selected);
    }
    Ok(indices)
}

fn indexed(
    member: &eredu_runtime::ParameterMemberSpec,
    axis: usize,
    indices: Vec<usize>,
    allocation: Allocation<'_>,
) -> Result<(eredu_runtime::TensorPlacement, Vec<usize>), Cause> {
    if indices.is_empty() {
        return Err(allocation.error(format_args!(
            "tensor {:?} has no local indices",
            member.target()
        )));
    }
    let mut shape = allocation.copy(member.global_shape())?;
    shape[axis] = indices.len();
    Ok((
        eredu_runtime::TensorPlacement::Indices { axis, indices },
        shape,
    ))
}

#[cfg(test)]
mod tests;

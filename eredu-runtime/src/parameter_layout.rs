//! Physical parameter-layout lowering over architecture-authored neutral declarations.

use std::ops::Range;

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
    if let Some(layout) = retained_partition_layout(description, topology)? {
        return Ok(layout.clone());
    }
    local_layout(
        description,
        topology.tensor_parallel_rank(),
        topology.tensor_parallel_size(),
        topology.expert_parallel_rank(),
        topology.expert_parallel_size(),
    )
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
                eredu_core::balanced_contiguous_range(units, tensor_parts, tensor_rank, false)
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
                    resolve_member(member, group.partition_units(), tensor_rank, tensor_parts)?;
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
                resolve_member(member, group.partition_units(), tensor_rank, tensor_parts)?
            };
            let mut tensor = crate::LocalTensorLayout::new(
                group.logical_name(),
                group.role(),
                member.global_shape().to_vec(),
                local_shape,
                placement,
                group.partition_units(),
                logical_range.clone(),
                false,
            )
            .with_partition_chunk_size(match member.sharding() {
                crate::MemberSharding::PartitionedChunks { chunk_size, .. }
                | crate::MemberSharding::PartitionedChunkSegments { chunk_size, .. } => {
                    Some(*chunk_size)
                }
                _ => None,
            });
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
    let ranged = |axis: usize, range: Range<usize>| {
        let mut shape = member.global_shape().to_vec();
        shape[axis] = range.len();
        (
            TensorPlacement::Range {
                axis,
                start: range.start,
                end: range.end,
            },
            shape,
        )
    };
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
            Ok(ranged(*axis, range))
        }
        MemberSharding::Partitioned { axis } => {
            let units = partition_units
                .ok_or_else(|| format!("tensor {:?} has no logical partition", member.target()))?;
            let dimension = member_axis(member, *axis)?;
            if !dimension.is_multiple_of(units) {
                return Err(format!(
                    "tensor {:?} axis {axis} does not contain {units} logical units",
                    member.target()
                ));
            }
            let logical = eredu_core::balanced_contiguous_range(units, parts, rank, false)
                .map_err(|error| error.to_string())?;
            let width = dimension / units;
            Ok(ranged(*axis, logical.start * width..logical.end * width))
        }
        MemberSharding::PartitionedChunks { axis, chunk_size } => {
            let units = partition_units
                .ok_or_else(|| format!("tensor {:?} has no logical partition", member.target()))?;
            let dimension = member_axis(member, *axis)?;
            if *chunk_size == 0 || dimension.div_ceil(*chunk_size) != units {
                return Err(format!(
                    "tensor {:?} chunks do not match {units} logical units",
                    member.target()
                ));
            }
            let logical = eredu_core::balanced_contiguous_range(units, parts, rank, false)
                .map_err(|error| error.to_string())?;
            let range = crate::partition_chunk_range(dimension, *chunk_size, logical)
                .map_err(|error| error.to_string())?;
            Ok(ranged(*axis, range))
        }
        MemberSharding::PartitionedChunkSegments {
            axis,
            segments,
            chunk_size,
        } => {
            let units = partition_units
                .ok_or_else(|| format!("tensor {:?} has no logical partition", member.target()))?;
            let logical = eredu_core::balanced_contiguous_range(units, parts, rank, false)
                .map_err(|error| error.to_string())?;
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
            let units = partition_units
                .ok_or_else(|| format!("tensor {:?} has no logical partition", member.target()))?;
            let logical = eredu_core::balanced_contiguous_range(units, parts, rank, false)
                .map_err(|error| error.to_string())?;
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

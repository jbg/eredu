//! Original ownership equations with prospective source destinations.
use super::source_allocation::{Allocation, Cause};
use super::*;

pub(super) struct RankRequirements {
    pub(super) groups: Vec<PartitionedGroupRequirements>,
    pub(super) ownership: PartitionOwnership,
    pub(super) publication_owner: usize,
    pub(super) state: Option<PartitionState>,
    pub(super) parameter_targets: Vec<String>,
}
pub(super) fn rank_requirements(
    execution: &eredu_runtime::ReplicatedTextRequirements,
    state_layout: Option<&eredu_runtime::StateLayout>,
    topology: ParallelRankTopology,
) -> Result<RankRequirements, String> {
    rank_requirements_inner(execution, state_layout, topology, true)
}
pub(super) fn rank_requirements_inner(
    execution: &eredu_runtime::ReplicatedTextRequirements,
    state_layout: Option<&eredu_runtime::StateLayout>,
    topology: ParallelRankTopology,
    include_parameter_targets: bool,
) -> Result<RankRequirements, String> {
    worker(
        execution,
        topology,
        include_parameter_targets,
        Allocation(None),
        |local_state_ranges| {
            let state = if let (Some(state_layout), Some(first)) =
                (state_layout, local_state_ranges.first())
            {
                let start = first.start;
                let end = local_state_ranges
                    .iter()
                    .try_fold(start, |frontier, range| {
                        (range.start == frontier)
                            .then_some(range.end)
                            .ok_or_else(|| {
                                Cause::Semantic(
                                    "rank-local state ranges are not contiguous".to_owned(),
                                )
                            })
                    })?;
                Some(
                    PartitionState::new(
                        state_layout
                            .slice(start..end)
                            .map_err(|error| Cause::Semantic(error.to_string()))?,
                        start,
                    )
                    .map_err(|error| Cause::Semantic(error.to_string()))?,
                )
            } else {
                None
            };

            Ok(state)
        },
    )
    .map(|(rank, _)| rank)
    .map_err(Cause::ordinary)
}
// Only the source projection is qualified here. Native mutable state slicing
// remains the ordinary admission worker above and is never entered by this API.
pub(super) fn source(
    execution: &eredu_runtime::ReplicatedTextRequirements,
    topology: ParallelRankTopology,
    allocation: Allocation<'_>,
) -> Result<RankRequirements, Cause> {
    worker(execution, topology, false, allocation, |_| Ok(None)).map(|(rank, _)| rank)
}
fn copy_roles(
    target: &mut Vec<String>,
    source: &[String],
    allocation: Allocation<'_>,
) -> Result<(), Cause> {
    allocation.controls::<(&mut Vec<String>, std::slice::Iter<'_, String>, String)>()?;
    for role in source {
        let value = allocation.text(role)?;
        allocation.push(target, value)?;
    }
    Ok(())
}
fn worker(
    execution: &eredu_runtime::ReplicatedTextRequirements,
    topology: ParallelRankTopology,
    include_parameter_targets: bool,
    allocation: Allocation<'_>,
    state: impl FnOnce(&[Range<usize>]) -> Result<Option<PartitionState>, Cause>,
) -> Result<(RankRequirements, Vec<Range<usize>>), Cause> {
    allocation.controls::<(
        &eredu_runtime::ReplicatedTextRequirements,
        ParallelRankTopology,
        bool,
        RankRequirements,
        Vec<String>,
        Vec<Range<usize>>,
        Vec<PartitionedGroupRequirements>,
        Vec<(usize, Range<usize>)>,
        Range<usize>,
        Option<usize>,
        [usize; 12],
        [bool; 4],
        eredu_runtime::ArchitecturePartitionError,
    )>()?;
    allocation.controls_of(&state)?;
    let pp_size = topology.pipeline_parallel_size();
    let pp_rank = topology.pipeline_parallel_rank();
    let mut groups = Vec::new();
    let mut static_roles = Vec::new();
    let mut owns_input = false;
    let output_group = execution.execution_graph().output();
    let mut owns_output = false;
    let mut output_pipeline = None;
    let mut decoder_state_offset = 0usize;
    let mut local_state_ranges = Vec::new();

    for (group_index, transport) in execution.group_transports().iter().enumerate() {
        if transport.kind == ArchitectureGroupKind::Prediction {
            return Err(allocation.error(format_args!("embedded prediction is active")));
        }
        let count = execution
            .execution_units()
            .group_range(group_index)
            .ok_or_else(|| {
                allocation.error(format_args!(
                    "execution group {group_index} has no unit geometry"
                ))
            })?
            .len();
        let path = match transport.placement {
            ArchitectureGroupPlacement::Pipeline => 0..pp_size,
            ArchitectureGroupPlacement::OutputOwner => pp_size - 1..pp_size,
        };
        let owners = balanced_ranges_worker(count, path.clone(), allocation)?;
        let first_owner = owners.first().map_or(path.start, |(owner, _)| *owner);
        let last_owner = owners.last().map_or(first_owner, |(owner, _)| *owner);
        let merge_owner = match transport.merge_destination {
            ArchitectureMergeDestination::LastOwner => last_owner,
            ArchitectureMergeDestination::FirstPipelineOwner => 0,
            ArchitectureMergeDestination::OutputOwner => pp_size - 1,
        };
        if execution
            .execution_graph()
            .dependencies(group_index)
            .is_some_and(|dependencies| dependencies.is_empty())
            && first_owner == pp_rank
        {
            owns_input = true;
        }
        if group_index == output_group {
            output_pipeline = Some(merge_owner);
            owns_output = merge_owner == pp_rank;
        }
        let first_role_start = static_roles.len();
        if first_owner == pp_rank {
            copy_roles(
                &mut static_roles,
                &transport.first_owner_static_roles,
                allocation,
            )?;
        }
        let first_role_end = static_roles.len();
        if last_owner == pp_rank {
            for role in &transport.last_owner_static_roles {
                if last_owner != first_owner
                    || !static_roles[first_role_start..first_role_end].contains(role)
                {
                    let value = allocation.text(role)?;
                    allocation.push(&mut static_roles, value)?;
                }
            }
        }
        if let Some((_, units)) = owners.iter().find(|(owner, _)| *owner == pp_rank) {
            let group = PartitionedGroupRequirements {
                group: eredu_runtime::ExecutionGroupId::new(
                    allocation.text(
                        execution
                            .execution_units()
                            .group_id(group_index)
                            .expect("validated execution layout contains every group")
                            .as_str(),
                    )?,
                )
                .expect("existing validated execution group"),
                units: units.clone(),
                pipeline_owner: pp_rank,
            };
            allocation.push(&mut groups, group)?;
            if transport.kind == ArchitectureGroupKind::Decoder {
                allocation.push(
                    &mut local_state_ranges,
                    decoder_state_offset + units.start..decoder_state_offset + units.end,
                )?;
            }
        }
        if transport.kind == ArchitectureGroupKind::Decoder {
            decoder_state_offset = decoder_state_offset.checked_add(count).ok_or_else(|| {
                allocation.error(format_args!("decoder state geometry overflowed"))
            })?;
        }
    }
    // Each nonempty range was appended after advancing the cumulative decoder
    // offset. Their starts are already strictly increasing, so the former sort
    // cannot change order and needs neither scratch nor recursive controls.
    debug_assert!(
        local_state_ranges
            .windows(2)
            .all(|pair| pair[0].start < pair[1].start)
    );
    let state = state(&local_state_ranges)?;
    let mut replicated = Vec::new();
    copy_roles(
        &mut replicated,
        execution.replicated_static_roles(),
        allocation,
    )?;
    let ownership = match allocation.0 {
        Some(funding) => {
            PartitionOwnership::new_with_funding(owns_input, owns_output, static_roles, funding)
                .and_then(|ownership| {
                    ownership.with_replicated_static_roles_funded(replicated, funding)
                })
        }
        None => PartitionOwnership::new(owns_input, owns_output, static_roles)
            .and_then(|ownership| ownership.with_replicated_static_roles(replicated)),
    }
    .map_err(|cause| allocation.partition_error(cause))?;
    let mut parameter_targets = Vec::new();
    let selected_parameters = execution
        .parameters()
        .iter()
        .filter(|_| include_parameter_targets)
        .filter(|parameter| match parameter.owner() {
            eredu_runtime::ReplicatedTextParameterOwner::StaticRole(role) => {
                ownership.stores_static_role(role)
            }
            eredu_runtime::ReplicatedTextParameterOwner::StaticUnitConsumers {
                role,
                consumers,
            } => {
                ownership.replicated_static_roles().contains(role)
                    || consumers.iter().any(|(group, unit)| {
                        groups.iter().any(|owned| {
                            owned.group.as_str() == group && owned.units.contains(unit)
                        })
                    })
            }
            eredu_runtime::ReplicatedTextParameterOwner::ExecutionUnit { group, unit } => groups
                .iter()
                .any(|owned| owned.group.as_str() == group && owned.units.contains(unit)),
            _ => false,
        });
    for parameter in selected_parameters {
        let target = allocation.text(parameter.name())?;
        allocation.push(&mut parameter_targets, target)?;
    }
    let rank = RankRequirements {
        groups,
        ownership,
        publication_owner: topology
            .global_rank_for(eredu_core::ParallelCoordinates::new(
                0,
                output_pipeline.ok_or_else(|| {
                    allocation.error(format_args!("output group has no publication ownership"))
                })?,
                0,
                topology.data_parallel_rank(),
            ))
            .map_err(|error| allocation.error(format_args!("{error}")))?,
        state,
        parameter_targets,
    };
    Ok((rank, local_state_ranges))
}

pub(super) fn balanced_ranges(
    unit_count: usize,
    ranks: Range<usize>,
) -> Vec<(usize, Range<usize>)> {
    balanced_ranges_worker(unit_count, ranks, Allocation(None))
        .unwrap_or_else(|cause| panic!("{}", cause.ordinary()))
}
fn balanced_ranges_worker(
    unit_count: usize,
    ranks: Range<usize>,
    allocation: Allocation<'_>,
) -> Result<Vec<(usize, Range<usize>)>, Cause> {
    allocation.controls::<(usize, Range<usize>, Vec<(usize, Range<usize>)>, [usize; 5])>()?;
    if unit_count == 0 {
        return Ok(Vec::new());
    }
    let active = ranks.len().min(unit_count);
    let base = unit_count / active;
    let remainder = unit_count % active;
    let mut start = 0;
    let mut result = allocation.vector(active)?;
    for (index, rank) in ranks.take(active).enumerate() {
        let end = start + base + usize::from(index < remainder);
        result.push((rank, start..end));
        start = end;
    }
    Ok(result)
}

#[cfg(test)]
mod tests;

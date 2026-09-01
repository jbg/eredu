//! Composition-owned translation from semantic axes to opaque MLX groups.

use eredu_core::{DistributedSession as _, ParallelAxis, ParallelRankTopology};

use crate::backend::{
    distributed::{ADDRESSABLE_GROUP_ID, SHARD_GROUP_ID, STAGE_GROUP_ID},
    error::Error,
    runtime::distributed::topology::{CollectiveGroupRealization, CollectiveRealization},
    MlxParallelContext,
};

pub(crate) fn collective_realization(
    topology: MlxParallelContext,
) -> Result<CollectiveRealization, Error> {
    let mut groups = Vec::new();
    for (id, axis) in [
        (SHARD_GROUP_ID, ParallelAxis::Tensor),
        (STAGE_GROUP_ID, ParallelAxis::Pipeline),
        (ADDRESSABLE_GROUP_ID, ParallelAxis::Expert),
    ] {
        let membership = topology.subgroup(axis)?;
        let routes = matches!(axis, ParallelAxis::Tensor | ParallelAxis::Expert)
            .then(|| logical_stage_group_routes(topology, axis))
            .transpose()?;
        groups.push(CollectiveGroupRealization::new(
            id,
            membership.global_ranks,
            membership.rank,
            membership.color,
            routes,
        )?);
    }
    CollectiveRealization::new(topology.world_size, topology.global_rank, groups)
}

pub(crate) fn validate_session(
    topology: MlxParallelContext,
    session: &crate::backend::MlxDistributedSession<'_>,
) -> Result<(), Error> {
    let expected = collective_realization(topology)?.descriptor();
    let actual = session.descriptor();
    if actual == expected {
        Ok(())
    } else {
        Err(Error::Parallel(format!(
            "distributed session realization {actual:?} does not match selected topology realization {expected:?}"
        )))
    }
}

pub(crate) fn logical_stage_group_routes(
    topology: MlxParallelContext,
    axis: ParallelAxis,
) -> Result<Vec<(usize, Vec<Option<usize>>)>, Error> {
    let axis_size = match axis {
        ParallelAxis::Tensor => topology.tensor_parallel_size,
        ParallelAxis::Expert => topology.expert_parallel_size,
        ParallelAxis::Pipeline | ParallelAxis::Data => {
            return Err(Error::Parallel(
                "logical stage routes require a stage-local collective group".into(),
            ));
        }
    };
    let stage_width = topology
        .tensor_parallel_size
        .checked_mul(topology.expert_parallel_size)
        .ok_or_else(|| Error::Parallel("stage-local route width overflowed usize".into()))?;
    let stage_start = topology
        .pipeline_parallel_rank
        .checked_mul(stage_width)
        .ok_or_else(|| Error::Parallel("stage-local route start overflowed usize".into()))?;
    let cohort = (stage_start..stage_start + stage_width).collect::<Vec<_>>();
    let local_axis_rank = match axis {
        ParallelAxis::Tensor => topology.tensor_parallel_rank,
        ParallelAxis::Expert => topology.expert_parallel_rank,
        ParallelAxis::Pipeline | ParallelAxis::Data => unreachable!(),
    };
    let mut routes = Vec::with_capacity(axis_size);
    for shift in 0..axis_size {
        let mut destinations = cohort
            .iter()
            .map(|&source_rank| -> Result<usize, Error> {
                let source = ParallelRankTopology::new(topology.topology(), source_rank)?;
                let mut coordinates = source.coordinates();
                match axis {
                    ParallelAxis::Tensor => {
                        coordinates.tensor = (coordinates.tensor + shift) % axis_size;
                    }
                    ParallelAxis::Expert => {
                        coordinates.expert = (coordinates.expert + shift) % axis_size;
                    }
                    ParallelAxis::Pipeline | ParallelAxis::Data => unreachable!(),
                }
                topology.global_rank_for(coordinates).map_err(Into::into)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let mut exchanges = Vec::with_capacity(stage_width);
        for round in 0..stage_width {
            let mut local_peer = None;
            for left in (round % 2..stage_width.saturating_sub(1)).step_by(2) {
                let right = left + 1;
                if destinations[left] > destinations[right] {
                    let left_rank = cohort[left];
                    let right_rank = cohort[right];
                    if topology.global_rank == left_rank {
                        local_peer = Some(right_rank);
                    } else if topology.global_rank == right_rank {
                        local_peer = Some(left_rank);
                    }
                    destinations.swap(left, right);
                }
            }
            exchanges.push(local_peer);
        }
        if destinations != cohort {
            return Err(Error::Parallel(format!(
                "failed to construct neighbor route for group shift {shift} within stage cohort {cohort:?}"
            )));
        }
        routes.push(((local_axis_rank + axis_size - shift) % axis_size, exchanges));
    }
    Ok(routes)
}

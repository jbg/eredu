//! Composition-owned translation from semantic axes to opaque MLX groups.

use std::ops::Deref;

use eredu_core::{
    BackendError, DistributedSession as _, ParallelAxis, ParallelRankTopology, ParallelTopology,
};
use safemlx::{distributed::Group, Stream};

use crate::backend::{
    distributed::{ADDRESSABLE_GROUP_ID, SHARD_GROUP_ID, STAGE_GROUP_ID},
    error::Error,
    runtime::distributed::topology::{CollectiveGroupRealization, CollectiveRealization},
    topology::{DeviceAssignment, MlxRankContext},
};

/// Composition-owned semantic topology and sharding fallback supplied to
/// architecture module constructors.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ParallelBuildContext {
    topology: MlxParallelPlan,
    policy: eredu_runtime::ShardingPolicy,
}

impl ParallelBuildContext {
    /// Creates a construction context for a selected semantic topology.
    pub(crate) const fn new(
        topology: MlxParallelPlan,
        policy: eredu_runtime::ShardingPolicy,
    ) -> Self {
        Self { topology, policy }
    }

    /// Returns the composition-owned semantic topology.
    pub(crate) const fn topology(self) -> MlxParallelPlan {
        self.topology
    }

    /// Lowers semantic tensor partitioning into generic rank-local placement.
    pub(crate) fn planner(
        self,
    ) -> crate::backend::runtime::distributed::parallel::ParallelPlanBuilder {
        crate::backend::runtime::distributed::parallel::ParallelPlanBuilder::with_policy(
            self.topology.rank_context(),
            self.topology.tensor_parallel_rank(),
            self.topology.tensor_parallel_size(),
            self.policy,
        )
    }
}

/// Composition-owned semantic parallel plan bound to one MLX rank context.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct MlxParallelPlan {
    rank: ParallelRankTopology,
    context: MlxRankContext,
}

impl MlxParallelPlan {
    /// Binds one semantic topology and rank to a process-local MLX device.
    pub fn new(
        topology: ParallelTopology,
        global_rank: usize,
        device: DeviceAssignment,
    ) -> Result<Self, Error> {
        if topology.data() != 1 {
            return Err(Error::Backend(BackendError::Unsupported {
                backend: "mlx".into(),
                capability: "data-parallel model sessions".into(),
            }));
        }
        Ok(Self {
            rank: ParallelRankTopology::new(topology, global_rank)?,
            context: MlxRankContext::new(topology.world_size(), global_rank, device)?,
        })
    }

    /// Constructs a semantic topology for one explicit world rank.
    pub fn for_rank(
        global_rank: usize,
        tensor_parallel_size: usize,
        pipeline_parallel_size: usize,
        expert_parallel_size: usize,
        device: DeviceAssignment,
    ) -> Result<Self, Error> {
        Self::new(
            ParallelTopology::new(
                tensor_parallel_size,
                pipeline_parallel_size,
                expert_parallel_size,
                1,
            )?,
            global_rank,
            device,
        )
    }

    /// Constructs a semantic topology and snapshots its rank from an MLX world group.
    pub fn for_group(
        group: &Group,
        tensor_parallel_size: usize,
        pipeline_parallel_size: usize,
        expert_parallel_size: usize,
        device: DeviceAssignment,
    ) -> Result<Self, Error> {
        let topology = ParallelTopology::new(
            tensor_parallel_size,
            pipeline_parallel_size,
            expert_parallel_size,
            1,
        )?;
        if group.size() != topology.world_size() {
            return Err(Error::Parallel(format!(
                "parallel topology expects world size {} but received {}",
                topology.world_size(),
                group.size()
            )));
        }
        Self::new(topology, group.rank(), device)
    }

    /// Returns the semantic topology retained by composition.
    pub fn topology(self) -> ParallelTopology {
        self.rank.topology()
    }

    /// Returns the semantic rank snapshot retained by composition.
    pub const fn rank_topology(self) -> ParallelRankTopology {
        self.rank
    }

    pub(crate) const fn rank_context(self) -> MlxRankContext {
        self.context
    }

    /// Resolves the process-local MLX device bound to this composition plan.
    pub fn device(self) -> Result<safemlx::Device, Error> {
        self.context.device()
    }

    pub(crate) fn validate_execution_stream(self, stream: &Stream) -> Result<(), Error> {
        self.context.validate_execution_stream(stream)
    }
}

impl Deref for MlxParallelPlan {
    type Target = ParallelRankTopology;

    fn deref(&self) -> &Self::Target {
        &self.rank
    }
}

/// Translates composition-owned semantic topology into portable cache identity.
pub(crate) fn prompt_cache_topology(
    topology: MlxParallelPlan,
) -> eredu_core::cache::PromptCacheTopology {
    eredu_core::cache::PromptCacheTopology::new(
        (topology.pipeline_parallel_size() > 1).then_some((
            topology.pipeline_parallel_size(),
            topology.pipeline_parallel_rank(),
        )),
        (topology.tensor_parallel_size() > 1).then_some((
            topology.tensor_parallel_size(),
            topology.tensor_parallel_rank(),
        )),
        (topology.expert_parallel_size() > 1).then_some((
            topology.expert_parallel_size(),
            topology.expert_parallel_rank(),
        )),
        true,
    )
    .expect("parallel plan already validated every cache topology rank")
}

pub(crate) fn collective_realization(
    topology: MlxParallelPlan,
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
            membership.global_ranks().to_vec(),
            membership.rank(),
            membership.color(),
            routes,
        )?);
    }
    CollectiveRealization::new(topology.world_size(), topology.global_rank(), groups)
}

pub(crate) fn validate_session(
    topology: MlxParallelPlan,
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
    topology: MlxParallelPlan,
    axis: ParallelAxis,
) -> Result<Vec<(usize, Vec<Option<usize>>)>, Error> {
    let axis_size = match axis {
        ParallelAxis::Tensor => topology.tensor_parallel_size(),
        ParallelAxis::Expert => topology.expert_parallel_size(),
        ParallelAxis::Pipeline | ParallelAxis::Data => {
            return Err(Error::Parallel(
                "logical stage routes require a stage-local collective group".into(),
            ));
        }
        _ => {
            return Err(Error::Parallel(
                "unsupported parallel axis for a stage-local route".into(),
            ));
        }
    };
    let stage_width = topology
        .tensor_parallel_size()
        .checked_mul(topology.expert_parallel_size())
        .ok_or_else(|| Error::Parallel("stage-local route width overflowed usize".into()))?;
    let stage_start = topology
        .pipeline_parallel_rank()
        .checked_mul(stage_width)
        .ok_or_else(|| Error::Parallel("stage-local route start overflowed usize".into()))?;
    let cohort = (stage_start..stage_start + stage_width).collect::<Vec<_>>();
    let local_axis_rank = match axis {
        ParallelAxis::Tensor => topology.tensor_parallel_rank(),
        ParallelAxis::Expert => topology.expert_parallel_rank(),
        ParallelAxis::Pipeline | ParallelAxis::Data => unreachable!(),
        _ => {
            return Err(Error::Parallel(
                "unsupported parallel axis for a stage-local route".into(),
            ));
        }
    };
    let mut routes = Vec::with_capacity(axis_size);
    for shift in 0..axis_size {
        let mut destinations = cohort
            .iter()
            .map(|&source_rank| -> Result<usize, Error> {
                let source = ParallelRankTopology::new(topology.topology(), source_rank)?;
                let coordinates = source.coordinates();
                let coordinates = match axis {
                    ParallelAxis::Tensor => {
                        coordinates.with_tensor((coordinates.tensor() + shift) % axis_size)
                    }
                    ParallelAxis::Expert => {
                        coordinates.with_expert((coordinates.expert() + shift) % axis_size)
                    }
                    ParallelAxis::Pipeline | ParallelAxis::Data => unreachable!(),
                    _ => {
                        return Err(Error::Parallel(
                            "unsupported parallel axis for a stage-local route".into(),
                        ));
                    }
                };
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
                    if topology.global_rank() == left_rank {
                        local_peer = Some(right_rank);
                    } else if topology.global_rank() == right_rank {
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

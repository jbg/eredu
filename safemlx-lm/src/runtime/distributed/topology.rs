//! Rank-aware checkpoint topology, placement planning, and selective loading.
//!
//! Runtime parallelism is deliberately independent of architecture metadata in
//! `config.json`. In particular, a checkpoint's `ep_size` describes how that
//! checkpoint was written, while
//! [`crate::runtime::distributed::topology::ParallelTopology::expert_parallel_size`]
//! describes how the current inference job is arranged.

use std::{
    collections::{BTreeSet, HashMap, HashSet},
    ops::Range,
    path::{Path, PathBuf},
};

use safemlx::{
    distributed::{self, Group},
    Array, Device, DeviceType, Stream,
};

use crate::{
    error::Error,
    runtime::checkpoint::load::StrictLoadConfig,
    runtime::checkpoint::store::{
        SafetensorsWeightStore, TensorSelection, WeightReadPolicy, WeightStore,
    },
};

/// Explicit process-local execution-device assignment.
///
/// This value is never inferred from a global distributed rank. On a
/// one-process-per-visible-GPU launcher it is commonly GPU index zero on every
/// process, even though each process has a different global rank.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct DeviceAssignment {
    /// Device family used by this process.
    pub device_type: DeviceType,
    /// Index within this process's visible devices.
    pub local_index: usize,
}

/// One coordinate in the Cartesian parallel process grid.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct ParallelCoordinates {
    /// Tensor-parallel coordinate.
    pub tensor: usize,
    /// Pipeline-parallel coordinate.
    pub pipeline: usize,
    /// Expert-parallel coordinate.
    pub expert: usize,
}

/// A communication axis in the Cartesian parallel process grid.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum ParallelAxis {
    /// Tensor-parallel collectives at fixed pipeline and expert coordinates.
    Tensor,
    /// Pipeline transport at fixed tensor and expert coordinates.
    Pipeline,
    /// Expert exchange at fixed tensor and pipeline coordinates.
    Expert,
}

/// Topology-derived membership of one rank in an axis subgroup.
///
/// `color` and `key` are the canonical inputs for a backend-native group
/// split. `global_ranks` remains authoritative for backends which implement a
/// logical subgroup without native splitting.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SubgroupMembership {
    /// Axis represented by this subgroup.
    pub axis: ParallelAxis,
    /// Deterministic subgroup color among all groups on this axis.
    pub color: usize,
    /// Rank within the subgroup.
    pub rank: usize,
    /// Number of ranks in the subgroup.
    pub size: usize,
    /// Ordered global ranks, indexed by the changing axis coordinate.
    pub global_ranks: Vec<usize>,
}

/// Weight-independent validation and ownership report for one parallel rank.
///
/// Loaders construct this report after reading lightweight model metadata and
/// before opening checkpoint payload shards. Architecture semantic planners
/// then add their tensor legality and packed-alignment checks before any
/// execution-device materialization.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TopologyPreflightReport {
    /// Complete Cartesian topology.
    pub topology: ParallelTopology,
    /// TP collective membership.
    pub tensor_subgroup: SubgroupMembership,
    /// Pipeline-lane membership.
    pub pipeline_subgroup: SubgroupMembership,
    /// EP exchange membership.
    pub expert_subgroup: SubgroupMembership,
    /// Decoder layers owned by this pipeline coordinate, when supplied.
    pub local_layer_range: Option<Range<usize>>,
    /// Routed experts owned by this expert coordinate, when supplied.
    pub local_expert_range: Option<Range<usize>>,
    /// Whether this rank owns a (possibly TP-sharded) embedding.
    pub owns_embedding: bool,
    /// Whether this rank owns a (possibly TP-sharded) output head.
    pub owns_output_head: bool,
}

impl DeviceAssignment {
    /// Creates an explicit process-local assignment.
    pub const fn new(device_type: DeviceType, local_index: usize) -> Self {
        Self {
            device_type,
            local_index,
        }
    }

    /// Resolves this assignment to an MLX device.
    pub fn device(self) -> Result<Device, Error> {
        Ok(distributed::device_for_local_rank(
            self.device_type,
            self.local_index,
        )?)
    }
}

/// Validated, architecture-independent runtime parallel coordinates.
///
/// Rank ordering is pipeline-major, then tensor, then expert, with expert as
/// the fastest-changing coordinate:
/// `global_rank = ((pipeline_rank * tensor_size) + tensor_rank) * expert_size + expert_rank`.
/// The ordering is stable and should be used by later execution phases.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ParallelTopology {
    /// Number of processes in the communication group.
    pub world_size: usize,
    /// Rank in the communication group.
    pub global_rank: usize,
    /// Tensor-parallel process count.
    pub tensor_parallel_size: usize,
    /// Tensor-parallel coordinate of this process.
    pub tensor_parallel_rank: usize,
    /// Pipeline-parallel process count.
    pub pipeline_parallel_size: usize,
    /// Pipeline-parallel coordinate of this process.
    pub pipeline_parallel_rank: usize,
    /// Expert-parallel process count.
    pub expert_parallel_size: usize,
    /// Expert-parallel coordinate of this process.
    pub expert_parallel_rank: usize,
    /// Explicit process-local device assignment.
    pub device: DeviceAssignment,
}

/// Backend communication contexts materialized from one Cartesian topology.
///
/// Construction is collective when a non-global subgroup must be split. All
/// ranks must call [`Self::new`] in the same order. Singleton axes do not own a
/// communication group, while an axis spanning the complete world borrows the
/// original group without splitting it.
pub(crate) struct ParallelCommunicators<'a> {
    topology: ParallelTopology,
    world: &'a Group,
    tensor: AxisCommunicator,
    pipeline: AxisCommunicator,
    expert: AxisCommunicator,
}

struct AxisCommunicator {
    membership: SubgroupMembership,
    native: Option<Group>,
}

type LogicalRoutePlan = Vec<(usize, Vec<Option<usize>>)>;

impl std::fmt::Debug for ParallelCommunicators<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ParallelCommunicators")
            .field("topology", &self.topology)
            .field("tensor", &self.tensor.membership)
            .field("pipeline", &self.pipeline.membership)
            .field("expert", &self.expert.membership)
            .finish()
    }
}

impl<'a> ParallelCommunicators<'a> {
    /// Validates the world group and materializes every required native subgroup.
    pub(crate) fn new(topology: ParallelTopology, world: &'a Group) -> Result<Self, Error> {
        if world.rank() != topology.global_rank || world.size() != topology.world_size {
            return Err(Error::Parallel(format!(
                "parallel topology expects world rank {}/{} but received {}/{}",
                topology.global_rank,
                topology.world_size,
                world.rank(),
                world.size()
            )));
        }
        let tensor = Self::materialize(topology, world, ParallelAxis::Tensor)?;
        let pipeline = Self::materialize(topology, world, ParallelAxis::Pipeline)?;
        let expert = Self::materialize(topology, world, ParallelAxis::Expert)?;
        Ok(Self {
            topology,
            world,
            tensor,
            pipeline,
            expert,
        })
    }

    fn materialize(
        topology: ParallelTopology,
        world: &Group,
        axis: ParallelAxis,
    ) -> Result<AxisCommunicator, Error> {
        let membership = topology.subgroup(axis)?;
        let native = if membership.size == 1 || membership.size == topology.world_size {
            None
        } else {
            let color = i32::try_from(membership.color)
                .map_err(|_| Error::Parallel(format!("{axis:?} subgroup color exceeds i32")))?;
            let key = i32::try_from(membership.rank)
                .map_err(|_| Error::Parallel(format!("{axis:?} subgroup rank exceeds i32")))?;
            let group = match world.split(color, Some(key)) {
                Ok(group) => group,
                Err(_) if axis != ParallelAxis::Pipeline => world
                    .logical_subgroup_with_routes(
                        &membership.global_ranks,
                        logical_stage_axis_routes(topology, axis)?,
                    )
                    .map_err(|error| {
                        Error::Parallel(format!(
                            "failed to materialize routed logical {axis:?} subgroup color {} with members {:?}: {error}",
                            membership.color, membership.global_ranks
                        ))
                    })?,
                Err(_) => world
                    .logical_subgroup(&membership.global_ranks)
                    .map_err(|error| {
                        Error::Parallel(format!(
                            "failed to materialize native or logical {axis:?} subgroup color {} with members {:?}: {error}",
                            membership.color, membership.global_ranks
                        ))
                    })?,
            };
            if group.rank() != membership.rank || group.size() != membership.size {
                return Err(Error::Parallel(format!(
                    "{axis:?} subgroup expected rank {}/{} but backend produced {}/{}",
                    membership.rank,
                    membership.size,
                    group.rank(),
                    group.size()
                )));
            }
            Some(group)
        };
        Ok(AxisCommunicator { membership, native })
    }

    /// Returns the global communication group.
    pub(crate) const fn world(&self) -> &Group {
        self.world
    }

    /// Returns the native group for a non-singleton axis.
    pub(crate) fn group(&self, axis: ParallelAxis) -> Option<&Group> {
        let communicator = match axis {
            ParallelAxis::Tensor => &self.tensor,
            ParallelAxis::Pipeline => &self.pipeline,
            ParallelAxis::Expert => &self.expert,
        };
        if communicator.membership.size == 1 {
            None
        } else if communicator.membership.size == self.topology.world_size {
            Some(self.world)
        } else {
            communicator.native.as_ref()
        }
    }

    /// Returns the TP collective group, or `None` when TP is inactive.
    pub(crate) fn tensor_group(&self) -> Option<&Group> {
        self.group(ParallelAxis::Tensor)
    }

    /// Returns the pipeline-lane consensus group, or `None` when PP is inactive.
    pub(crate) fn pipeline_group(&self) -> Option<&Group> {
        self.group(ParallelAxis::Pipeline)
    }

    /// Returns the EP exchange group, or `None` when EP is inactive.
    pub(crate) fn expert_group(&self) -> Option<&Group> {
        self.group(ParallelAxis::Expert)
    }
}

fn logical_stage_axis_routes(
    topology: ParallelTopology,
    axis: ParallelAxis,
) -> Result<LogicalRoutePlan, Error> {
    let axis_size = match axis {
        ParallelAxis::Tensor => topology.tensor_parallel_size,
        ParallelAxis::Expert => topology.expert_parallel_size,
        ParallelAxis::Pipeline => {
            return Err(Error::Parallel(
                "pipeline lanes do not use stage-local logical routes".into(),
            ))
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
        ParallelAxis::Pipeline => unreachable!(),
    };
    let mut routes = Vec::with_capacity(axis_size);
    for shift in 0..axis_size {
        let mut destinations = cohort
            .iter()
            .map(|&source_rank| {
                let source = ParallelTopology::from_rank(
                    topology.world_size,
                    source_rank,
                    topology.tensor_parallel_size,
                    topology.pipeline_parallel_size,
                    topology.expert_parallel_size,
                    topology.device,
                )?;
                let mut coordinates = source.coordinates();
                match axis {
                    ParallelAxis::Tensor => {
                        coordinates.tensor = (coordinates.tensor + shift) % axis_size;
                    }
                    ParallelAxis::Expert => {
                        coordinates.expert = (coordinates.expert + shift) % axis_size;
                    }
                    ParallelAxis::Pipeline => unreachable!(),
                }
                topology.global_rank_for(coordinates)
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
                "failed to construct neighbor route for {axis:?} shift {shift} within stage cohort {cohort:?}"
            )));
        }
        let source_rank = (local_axis_rank + axis_size - shift) % axis_size;
        routes.push((source_rank, exchanges));
    }
    Ok(routes)
}

impl ParallelTopology {
    /// Snapshots and validates runtime coordinates from a distributed group.
    pub fn from_group(
        group: &Group,
        tensor_parallel_size: usize,
        pipeline_parallel_size: usize,
        expert_parallel_size: usize,
        device: DeviceAssignment,
    ) -> Result<Self, Error> {
        Self::from_rank(
            group.size(),
            group.rank(),
            tensor_parallel_size,
            pipeline_parallel_size,
            expert_parallel_size,
            device,
        )
    }

    /// Validates a topology snapshot with an explicit world size and rank.
    ///
    /// Most applications should use [`Self::from_group`]. This constructor is
    /// useful to validate launcher metadata before initializing model state.
    pub fn from_rank(
        world_size: usize,
        global_rank: usize,
        tensor_parallel_size: usize,
        pipeline_parallel_size: usize,
        expert_parallel_size: usize,
        device: DeviceAssignment,
    ) -> Result<Self, Error> {
        if world_size == 0
            || tensor_parallel_size == 0
            || pipeline_parallel_size == 0
            || expert_parallel_size == 0
        {
            return Err(Error::Parallel(
                "world, tensor, pipeline, and expert parallel sizes must all be nonzero".into(),
            ));
        }
        if global_rank >= world_size {
            return Err(Error::Parallel(format!(
                "global rank {global_rank} is outside world size {world_size}"
            )));
        }
        if i32::try_from(device.local_index).is_err() {
            return Err(Error::Parallel(format!(
                "local device index {} does not fit in MLX's i32 device index",
                device.local_index
            )));
        }
        let topology_size = pipeline_parallel_size
            .checked_mul(tensor_parallel_size)
            .and_then(|value| value.checked_mul(expert_parallel_size))
            .ok_or_else(|| Error::Parallel("parallel topology size overflowed usize".into()))?;
        if topology_size != world_size {
            return Err(Error::Parallel(format!(
                "TP({tensor_parallel_size}) * PP({pipeline_parallel_size}) * EP({expert_parallel_size}) = {topology_size}, not world size {world_size}"
            )));
        }

        let expert_parallel_rank = global_rank % expert_parallel_size;
        let outer = global_rank / expert_parallel_size;
        let tensor_parallel_rank = outer % tensor_parallel_size;
        let pipeline_parallel_rank = outer / tensor_parallel_size;

        Ok(Self {
            world_size,
            global_rank,
            tensor_parallel_size,
            tensor_parallel_rank,
            pipeline_parallel_size,
            pipeline_parallel_rank,
            expert_parallel_size,
            expert_parallel_rank,
            device,
        })
    }

    /// Returns whether every runtime parallel dimension is a singleton.
    pub const fn is_replicated(self) -> bool {
        self.world_size == 1
            && self.tensor_parallel_size == 1
            && self.pipeline_parallel_size == 1
            && self.expert_parallel_size == 1
    }

    /// Returns whether the selected Cartesian axis has more than one member.
    pub const fn is_axis_active(self, axis: ParallelAxis) -> bool {
        match axis {
            ParallelAxis::Tensor => self.tensor_parallel_size > 1,
            ParallelAxis::Pipeline => self.pipeline_parallel_size > 1,
            ParallelAxis::Expert => self.expert_parallel_size > 1,
        }
    }

    /// Validates combined-axis geometry before checkpoint payloads are opened.
    ///
    /// Every Cartesian combination is accepted. Architecture loaders remain
    /// responsible for declaring whether their semantic layer plans can
    /// compose all active axes.
    pub fn preflight(
        self,
        decoder_layers: Option<usize>,
        routed_experts: Option<usize>,
    ) -> Result<TopologyPreflightReport, Error> {
        let local_layer_range = match decoder_layers {
            Some(layers) => Some(self.layer_range(layers)?),
            None if self.pipeline_parallel_size > 1 => {
                return Err(Error::Parallel(
                    "pipeline topology preflight requires the decoder-layer count".into(),
                ))
            }
            None => None,
        };
        let local_expert_range = match routed_experts {
            Some(experts) => Some(self.expert_range(experts)?),
            None if self.expert_parallel_size > 1 => {
                return Err(Error::Parallel(
                    "expert topology preflight requires the routed-expert count".into(),
                ))
            }
            None => None,
        };
        Ok(TopologyPreflightReport {
            topology: self,
            tensor_subgroup: self.subgroup(ParallelAxis::Tensor)?,
            pipeline_subgroup: self.subgroup(ParallelAxis::Pipeline)?,
            expert_subgroup: self.subgroup(ParallelAxis::Expert)?,
            local_layer_range,
            local_expert_range,
            owns_embedding: self.owns_embedding(),
            owns_output_head: self.owns_output_head(),
        })
    }

    /// Returns this rank's complete Cartesian coordinates.
    pub const fn coordinates(self) -> ParallelCoordinates {
        ParallelCoordinates {
            tensor: self.tensor_parallel_rank,
            pipeline: self.pipeline_parallel_rank,
            expert: self.expert_parallel_rank,
        }
    }

    /// Resolves Cartesian coordinates to the unique global rank.
    pub fn global_rank_for(self, coordinates: ParallelCoordinates) -> Result<usize, Error> {
        if coordinates.tensor >= self.tensor_parallel_size
            || coordinates.pipeline >= self.pipeline_parallel_size
            || coordinates.expert >= self.expert_parallel_size
        {
            return Err(Error::Parallel(format!(
                "parallel coordinates TP={}, PP={}, EP={} are outside topology TP={}, PP={}, EP={}",
                coordinates.tensor,
                coordinates.pipeline,
                coordinates.expert,
                self.tensor_parallel_size,
                self.pipeline_parallel_size,
                self.expert_parallel_size
            )));
        }
        coordinates
            .pipeline
            .checked_mul(self.tensor_parallel_size)
            .and_then(|rank| rank.checked_add(coordinates.tensor))
            .and_then(|rank| rank.checked_mul(self.expert_parallel_size))
            .and_then(|rank| rank.checked_add(coordinates.expert))
            .ok_or_else(|| {
                Error::Parallel("parallel global-rank calculation overflowed usize".into())
            })
    }

    /// Returns topology-derived membership in one communication axis.
    pub fn subgroup(self, axis: ParallelAxis) -> Result<SubgroupMembership, Error> {
        let coordinates = self.coordinates();
        let (color, rank, size, global_ranks) = match axis {
            ParallelAxis::Tensor => {
                let color = coordinates
                    .pipeline
                    .checked_mul(self.expert_parallel_size)
                    .and_then(|value| value.checked_add(coordinates.expert))
                    .ok_or_else(|| Error::Parallel("TP subgroup color overflowed usize".into()))?;
                let ranks = (0..self.tensor_parallel_size)
                    .map(|tensor| {
                        self.global_rank_for(ParallelCoordinates {
                            tensor,
                            ..coordinates
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                (color, coordinates.tensor, self.tensor_parallel_size, ranks)
            }
            ParallelAxis::Pipeline => {
                let color = coordinates
                    .tensor
                    .checked_mul(self.expert_parallel_size)
                    .and_then(|value| value.checked_add(coordinates.expert))
                    .ok_or_else(|| Error::Parallel("PP subgroup color overflowed usize".into()))?;
                let ranks = (0..self.pipeline_parallel_size)
                    .map(|pipeline| {
                        self.global_rank_for(ParallelCoordinates {
                            pipeline,
                            ..coordinates
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                (
                    color,
                    coordinates.pipeline,
                    self.pipeline_parallel_size,
                    ranks,
                )
            }
            ParallelAxis::Expert => {
                let color = coordinates
                    .pipeline
                    .checked_mul(self.tensor_parallel_size)
                    .and_then(|value| value.checked_add(coordinates.tensor))
                    .ok_or_else(|| Error::Parallel("EP subgroup color overflowed usize".into()))?;
                let ranks = (0..self.expert_parallel_size)
                    .map(|expert| {
                        self.global_rank_for(ParallelCoordinates {
                            expert,
                            ..coordinates
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                (color, coordinates.expert, self.expert_parallel_size, ranks)
            }
        };
        if global_ranks.get(rank).copied() != Some(self.global_rank) {
            return Err(Error::Parallel(format!(
                "{axis:?} subgroup geometry does not map local rank {rank} back to global rank {}",
                self.global_rank
            )));
        }
        Ok(SubgroupMembership {
            axis,
            color,
            rank,
            size,
            global_ranks,
        })
    }

    /// Returns the ordered global ranks participating in TP collectives with this rank.
    pub fn tensor_parallel_peers(self) -> Result<Vec<usize>, Error> {
        Ok(self.subgroup(ParallelAxis::Tensor)?.global_ranks)
    }

    /// Returns the ordered global ranks participating in EP exchange with this rank.
    pub fn expert_parallel_peers(self) -> Result<Vec<usize>, Error> {
        Ok(self.subgroup(ParallelAxis::Expert)?.global_ranks)
    }

    /// Returns the preceding pipeline rank with matching TP and EP coordinates.
    pub fn pipeline_predecessor(self) -> Result<Option<usize>, Error> {
        if self.pipeline_parallel_rank == 0 {
            return Ok(None);
        }
        self.global_rank_for(ParallelCoordinates {
            pipeline: self.pipeline_parallel_rank - 1,
            ..self.coordinates()
        })
        .map(Some)
    }

    /// Returns the succeeding pipeline rank with matching TP and EP coordinates.
    pub fn pipeline_successor(self) -> Result<Option<usize>, Error> {
        if self.pipeline_parallel_rank + 1 == self.pipeline_parallel_size {
            return Ok(None);
        }
        self.global_rank_for(ParallelCoordinates {
            pipeline: self.pipeline_parallel_rank + 1,
            ..self.coordinates()
        })
        .map(Some)
    }

    /// Returns whether this rank owns the stage-local token embedding shard.
    pub const fn owns_embedding(self) -> bool {
        self.pipeline_parallel_rank == 0
    }

    /// Returns whether this rank owns the stage-local output normalization and head shard.
    pub const fn owns_output_head(self) -> bool {
        self.pipeline_parallel_rank + 1 == self.pipeline_parallel_size
    }

    /// Returns whether this rank owns a decoder layer under balanced PP placement.
    pub fn owns_layer(self, decoder_layers: usize, layer: usize) -> Result<bool, Error> {
        Ok(self.layer_range(decoder_layers)?.contains(&layer))
    }

    /// Returns whether this rank owns a routed expert under balanced EP placement.
    pub fn owns_expert(self, routed_experts: usize, expert: usize) -> Result<bool, Error> {
        Ok(self.expert_range(routed_experts)?.contains(&expert))
    }

    /// Returns this pipeline stage's balanced contiguous decoder-layer range.
    ///
    /// Empty stages are rejected. Use [`balanced_contiguous_range`] directly
    /// with `allow_empty = true` when an architecture explicitly supports them.
    pub fn layer_range(self, decoder_layers: usize) -> Result<Range<usize>, Error> {
        balanced_contiguous_range(
            decoder_layers,
            self.pipeline_parallel_size,
            self.pipeline_parallel_rank,
            false,
        )
    }

    /// Returns this expert rank's balanced contiguous routed-expert range.
    ///
    /// Empty expert partitions are rejected.
    pub fn expert_range(self, routed_experts: usize) -> Result<Range<usize>, Error> {
        balanced_contiguous_range(
            routed_experts,
            self.expert_parallel_size,
            self.expert_parallel_rank,
            false,
        )
    }

    /// Verifies that an execution stream uses this process's assigned device.
    pub fn validate_execution_stream(self, stream: &Stream) -> Result<(), Error> {
        let actual = stream.get_device()?;
        let actual_type = actual.get_type()?;
        let actual_index = actual.get_index()?;
        let expected_index = i32::try_from(self.device.local_index)
            .expect("topology construction validated the local device index");
        if actual_type == self.device.device_type && actual_index == expected_index {
            Ok(())
        } else {
            Err(Error::Parallel(format!(
                "execution stream uses {actual_type:?} device {actual_index}, but this rank is assigned {:?} device {expected_index}",
                self.device.device_type
            )))
        }
    }
}

/// Computes a deterministic balanced contiguous range.
///
/// The first `total % parts` partitions receive one extra item. Therefore the
/// ranges cover `0..total` without gaps or overlap, including uneven splits.
pub fn balanced_contiguous_range(
    total: usize,
    parts: usize,
    index: usize,
    allow_empty: bool,
) -> Result<Range<usize>, Error> {
    if parts == 0 {
        return Err(Error::Parallel("partition count must be nonzero".into()));
    }
    if index >= parts {
        return Err(Error::Parallel(format!(
            "partition index {index} is outside {parts} parts"
        )));
    }
    if !allow_empty && total < parts {
        return Err(Error::Parallel(format!(
            "cannot divide {total} items among {parts} non-empty partitions"
        )));
    }
    let base = total / parts;
    let extra = total % parts;
    let start = index
        .checked_mul(base)
        .and_then(|value| value.checked_add(index.min(extra)))
        .ok_or_else(|| Error::Parallel("balanced range calculation overflowed usize".into()))?;
    let len = base + usize::from(index < extra);
    let end = start
        .checked_add(len)
        .ok_or_else(|| Error::Parallel("balanced range calculation overflowed usize".into()))?;
    Ok(start..end)
}

/// A validated contiguous slice of a source tensor.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TensorSlice {
    /// Source tensor axis being divided.
    pub axis: usize,
    /// Inclusive element offset on `axis`.
    pub start: usize,
    /// Exclusive element offset on `axis`.
    pub end: usize,
    /// Shard index.
    pub index: usize,
    /// Total number of equal shards.
    pub parts: usize,
}

impl TensorSlice {
    /// Validates and calculates an equal contiguous tensor slice.
    pub fn for_shape(
        shape: &[usize],
        axis: usize,
        index: usize,
        parts: usize,
    ) -> Result<Self, Error> {
        if axis >= shape.len() {
            return Err(Error::Parallel(format!(
                "tensor axis {axis} is outside rank {} shape {shape:?}",
                shape.len()
            )));
        }
        if parts == 0 {
            return Err(Error::Parallel("tensor shard count must be nonzero".into()));
        }
        if index >= parts {
            return Err(Error::Parallel(format!(
                "tensor shard index {index} is outside {parts} parts"
            )));
        }
        let dimension = shape[axis];
        if dimension == 0 || !dimension.is_multiple_of(parts) {
            return Err(Error::Parallel(format!(
                "tensor dimension {dimension} on axis {axis} is not nonzero and divisible by {parts}"
            )));
        }
        let width = dimension / parts;
        let start = index
            .checked_mul(width)
            .ok_or_else(|| Error::Parallel("tensor slice offset overflowed usize".into()))?;
        Ok(Self {
            axis,
            start,
            end: start + width,
            index,
            parts,
        })
    }

    /// Returns the local tensor shape produced by this slice.
    pub fn local_shape(&self, source_shape: &[usize]) -> Vec<usize> {
        let mut shape = source_shape.to_vec();
        shape[self.axis] = self.end - self.start;
        shape
    }
}

/// Typed placement decision for one target tensor.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum TensorPlacement {
    /// Materialize the complete tensor on every rank.
    Replicated,
    /// Materialize the complete tensor on this rank.
    Local,
    /// Intentionally omit this tensor on this rank.
    Omit,
    /// Materialize the complete tensor only on one global rank.
    Rank {
        /// Owning global rank.
        rank: usize,
    },
    /// Materialize the complete tensor only on one pipeline stage.
    PipelineStage {
        /// Owning pipeline-stage coordinate.
        stage: usize,
    },
    /// Materialize an equal contiguous source-tensor slice.
    Shard {
        /// Source tensor axis being sharded.
        axis: usize,
        /// Shard index.
        index: usize,
        /// Total shard count.
        parts: usize,
    },
    /// Materialize an explicit contiguous source-tensor range.
    ///
    /// Unlike [`Self::Shard`], ranges may be uneven. They are intended for
    /// balanced vocabulary partitions and other validated nonuniform layouts.
    Range {
        /// Source tensor axis being sliced.
        axis: usize,
        /// Inclusive element offset on `axis`.
        start: usize,
        /// Exclusive element offset on `axis`.
        end: usize,
    },
    /// Materialize selected source-tensor indices in the supplied order.
    ///
    /// This supports non-contiguous ownership layouts without loading the
    /// enclosing range or retaining unowned rows in the local partition.
    Indices {
        /// Source tensor axis being selected.
        axis: usize,
        /// Distinct source indices, ordered as they should appear locally.
        indices: Vec<usize>,
    },
}

#[derive(Debug, Clone)]
struct TensorPlan {
    placement: TensorPlacement,
    expected_source_shape: Option<Vec<usize>>,
}

/// Inspectable mapping from rewritten target names to typed placement decisions.
#[derive(Debug, Clone)]
pub struct PlacementPlan {
    topology: ParallelTopology,
    tensors: HashMap<String, TensorPlan>,
    default: Option<TensorPlacement>,
}

impl PlacementPlan {
    /// Creates a strict plan in which every checkpoint tensor must be named.
    pub fn new(topology: ParallelTopology) -> Self {
        Self {
            topology,
            tensors: HashMap::new(),
            default: None,
        }
    }

    /// Creates a plan that replicates every checkpoint tensor.
    pub fn replicated(topology: ParallelTopology) -> Self {
        Self::new(topology).with_default(TensorPlacement::Replicated)
    }

    /// Sets the placement used for checkpoint keys without an explicit entry.
    pub fn with_default(mut self, placement: TensorPlacement) -> Self {
        self.default = Some(placement);
        self
    }

    /// Returns the topology captured by this plan.
    pub const fn topology(&self) -> ParallelTopology {
        self.topology
    }

    /// Adds or replaces a target-tensor placement.
    pub fn insert(&mut self, target: impl Into<String>, placement: TensorPlacement) {
        self.tensors.insert(
            target.into(),
            TensorPlan {
                placement,
                expected_source_shape: None,
            },
        );
    }

    /// Adds a placement with a required pre-slice checkpoint shape.
    pub fn insert_expected(
        &mut self,
        target: impl Into<String>,
        expected_source_shape: impl Into<Vec<usize>>,
        placement: TensorPlacement,
    ) -> Result<(), Error> {
        let expected_source_shape = expected_source_shape.into();
        validate_placement(&placement, &expected_source_shape, self.topology)?;
        self.tensors.insert(
            target.into(),
            TensorPlan {
                placement,
                expected_source_shape: Some(expected_source_shape),
            },
        );
        Ok(())
    }

    /// Adds weight, scales, and optional biases using one logical placement.
    ///
    /// Keeping companions in one call prevents a quantized module's metadata
    /// from being accidentally placed differently from its packed weight.
    pub fn insert_quantized_companions(
        &mut self,
        prefix: &str,
        placement: TensorPlacement,
        has_biases: bool,
    ) {
        self.insert(format!("{prefix}.weight"), placement.clone());
        self.insert(format!("{prefix}.scales"), placement.clone());
        if has_biases {
            self.insert(format!("{prefix}.biases"), placement);
        }
    }

    /// Adds a tensor-parallel shard using this rank's TP coordinate.
    pub fn insert_tensor_parallel(&mut self, target: impl Into<String>, axis: usize) {
        self.insert(
            target,
            TensorPlacement::Shard {
                axis,
                index: self.topology.tensor_parallel_rank,
                parts: self.topology.tensor_parallel_size,
            },
        );
    }

    /// Adds this rank's balanced tensor-parallel range on `axis`.
    pub fn insert_balanced_tensor_parallel(
        &mut self,
        target: impl Into<String>,
        axis: usize,
        dimension: usize,
    ) -> Result<Range<usize>, Error> {
        let range = balanced_contiguous_range(
            dimension,
            self.topology.tensor_parallel_size,
            self.topology.tensor_parallel_rank,
            false,
        )?;
        self.insert(
            target,
            TensorPlacement::Range {
                axis,
                start: range.start,
                end: range.end,
            },
        );
        Ok(range)
    }

    /// Returns an explicit tensor placement by rewritten target name.
    pub fn placement(&self, target: &str) -> Option<&TensorPlacement> {
        self.tensors.get(target).map(|plan| &plan.placement)
    }

    /// Validates every placement whose constraints are known before loading.
    ///
    /// Axis bounds and divisibility require `insert_expected`; ownership and
    /// shard-coordinate bounds are validated for all entries.
    pub fn validate(&self) -> Result<(), Error> {
        for (target, tensor) in &self.tensors {
            validate_plan_entry(tensor, self.topology).map_err(|error| {
                Error::Parallel(format!("placement for tensor {target}: {error}"))
            })?;
        }
        if let Some(default) = &self.default {
            validate_plan_entry(
                &TensorPlan {
                    placement: default.clone(),
                    expected_source_shape: None,
                },
                self.topology,
            )?;
        }
        Ok(())
    }

    fn source_plan(&self, source: &str, config: &StrictLoadConfig) -> SourcePlan {
        for candidate in config.candidates(source) {
            if let Some(plan) = self.tensors.get(&candidate) {
                return SourcePlan::Known {
                    target: candidate,
                    tensor: plan.clone(),
                };
            }
        }
        if let Some(placement) = &self.default {
            let target = config
                .candidates(source)
                .into_iter()
                .next()
                .unwrap_or_else(|| source.to_string());
            SourcePlan::Known {
                target,
                tensor: TensorPlan {
                    placement: placement.clone(),
                    expected_source_shape: None,
                },
            }
        } else {
            SourcePlan::Unexpected
        }
    }
}

fn validate_plan_entry(plan: &TensorPlan, topology: ParallelTopology) -> Result<(), Error> {
    match &plan.placement {
        TensorPlacement::Rank { rank } if *rank >= topology.world_size => {
            Err(Error::Parallel(format!(
                "owner rank {rank} is outside world size {}",
                topology.world_size
            )))
        }
        TensorPlacement::PipelineStage { stage } if *stage >= topology.pipeline_parallel_size => {
            Err(Error::Parallel(format!(
                "pipeline owner stage {stage} is outside {} stages",
                topology.pipeline_parallel_size
            )))
        }
        TensorPlacement::Shard { index, parts, .. } if *parts == 0 || *index >= *parts => {
            Err(Error::Parallel(format!(
                "tensor shard index {index} is invalid for {parts} parts"
            )))
        }
        TensorPlacement::Range { start, end, .. } if start >= end => Err(Error::Parallel(format!(
            "tensor range {start}..{end} must be non-empty"
        ))),
        TensorPlacement::Indices { indices, .. } if indices.is_empty() => Err(Error::Parallel(
            "tensor index selection must be non-empty".into(),
        )),
        TensorPlacement::Indices { indices, .. }
            if indices.iter().collect::<HashSet<_>>().len() != indices.len() =>
        {
            Err(Error::Parallel(
                "tensor index selection must not contain duplicates".into(),
            ))
        }
        placement => {
            if let Some(shape) = &plan.expected_source_shape {
                validate_placement(placement, shape, topology)?;
            }
            Ok(())
        }
    }
}

#[derive(Debug, Clone)]
enum SourcePlan {
    Known { target: String, tensor: TensorPlan },
    Unexpected,
}

#[derive(Debug)]
enum ResolvedPlacement {
    Materialize,
    Omit,
    Shard(TensorSlice),
    Indices { axis: usize, indices: Vec<usize> },
}

fn validate_placement(
    placement: &TensorPlacement,
    shape: &[usize],
    topology: ParallelTopology,
) -> Result<(), Error> {
    match placement {
        TensorPlacement::Rank { rank } if *rank >= topology.world_size => {
            Err(Error::Parallel(format!(
                "owner rank {rank} is outside world size {}",
                topology.world_size
            )))
        }
        TensorPlacement::PipelineStage { stage } if *stage >= topology.pipeline_parallel_size => {
            Err(Error::Parallel(format!(
                "pipeline owner stage {stage} is outside {} stages",
                topology.pipeline_parallel_size
            )))
        }
        TensorPlacement::Shard { axis, index, parts } => {
            TensorSlice::for_shape(shape, *axis, *index, *parts).map(|_| ())
        }
        TensorPlacement::Range { axis, start, end } => {
            if *axis >= shape.len() {
                return Err(Error::Parallel(format!(
                    "tensor range axis {axis} is outside rank {} shape {shape:?}",
                    shape.len()
                )));
            }
            if start >= end || *end > shape[*axis] {
                return Err(Error::Parallel(format!(
                    "tensor range {start}..{end} is invalid for dimension {} on axis {axis}",
                    shape[*axis]
                )));
            }
            Ok(())
        }
        TensorPlacement::Indices { axis, indices } => {
            if *axis >= shape.len() {
                return Err(Error::Parallel(format!(
                    "tensor index axis {axis} is outside rank {} shape {shape:?}",
                    shape.len()
                )));
            }
            if indices.is_empty() {
                return Err(Error::Parallel(
                    "tensor index selection must be non-empty".into(),
                ));
            }
            if indices.iter().collect::<HashSet<_>>().len() != indices.len() {
                return Err(Error::Parallel(
                    "tensor index selection must not contain duplicates".into(),
                ));
            }
            if let Some(index) = indices.iter().copied().find(|index| *index >= shape[*axis]) {
                return Err(Error::Parallel(format!(
                    "tensor index {index} is outside dimension {} on axis {axis}",
                    shape[*axis]
                )));
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn resolve_placement(
    plan: &TensorPlan,
    shape: &[usize],
    topology: ParallelTopology,
) -> Result<ResolvedPlacement, Error> {
    if let Some(expected) = &plan.expected_source_shape {
        if expected != shape {
            return Err(Error::Parallel(format!(
                "expected checkpoint shape {expected:?}, got {shape:?}"
            )));
        }
    }
    validate_placement(&plan.placement, shape, topology)?;
    Ok(match &plan.placement {
        TensorPlacement::Replicated | TensorPlacement::Local => ResolvedPlacement::Materialize,
        TensorPlacement::Omit => ResolvedPlacement::Omit,
        TensorPlacement::Rank { rank } => {
            if *rank == topology.global_rank {
                ResolvedPlacement::Materialize
            } else {
                ResolvedPlacement::Omit
            }
        }
        TensorPlacement::PipelineStage { stage } => {
            if *stage == topology.pipeline_parallel_rank {
                ResolvedPlacement::Materialize
            } else {
                ResolvedPlacement::Omit
            }
        }
        TensorPlacement::Shard { axis, index, parts } => {
            ResolvedPlacement::Shard(TensorSlice::for_shape(shape, *axis, *index, *parts)?)
        }
        TensorPlacement::Range { axis, start, end } => ResolvedPlacement::Shard(TensorSlice {
            axis: *axis,
            start: *start,
            end: *end,
            index: 0,
            parts: 1,
        }),
        TensorPlacement::Indices { axis, indices } => ResolvedPlacement::Indices {
            axis: *axis,
            indices: indices.clone(),
        },
    })
}

/// Locally materialized checkpoint partition.
///
/// This is intentionally not an executable model. Later distributed execution
/// phases can consume it together with a communication group without storing a
/// borrowed group inside long-lived model state.
#[derive(Debug)]
pub struct RankPartition {
    topology: ParallelTopology,
    tensors: HashMap<String, Array>,
    opened_shards: Vec<PathBuf>,
}

impl RankPartition {
    /// Returns the validated topology used for this partition.
    pub const fn topology(&self) -> ParallelTopology {
        self.topology
    }

    /// Returns a locally materialized tensor by rewritten target name.
    pub fn get(&self, target: &str) -> Option<&Array> {
        self.tensors.get(target)
    }

    /// Iterates over locally materialized tensors.
    pub fn tensors(&self) -> impl Iterator<Item = (&str, &Array)> {
        self.tensors
            .iter()
            .map(|(key, value)| (key.as_str(), value))
    }

    /// Returns the number of locally materialized tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Returns whether this partition contains no local tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Returns checkpoint payload shards that were actually opened.
    pub fn opened_shards(&self) -> &[PathBuf] {
        &self.opened_shards
    }

    /// Consumes the partition and returns its locally materialized tensors.
    ///
    /// Pipeline stage constructors use this to move arrays directly into
    /// stage-local modules without cloning the partition or its arrays.
    pub fn into_tensors(self) -> HashMap<String, Array> {
        self.tensors
    }
}

#[derive(Default)]
struct PartitionReport {
    loaded: HashSet<String>,
    unexpected: Vec<String>,
}

impl PartitionReport {
    fn finish(self, plan: &PlacementPlan, config: &StrictLoadConfig) -> Result<(), Error> {
        let mut missing = Vec::new();
        for (target, tensor) in &plan.tensors {
            let locally_required = match tensor.placement {
                TensorPlacement::Replicated
                | TensorPlacement::Local
                | TensorPlacement::Shard { .. }
                | TensorPlacement::Range { .. }
                | TensorPlacement::Indices { .. } => true,
                TensorPlacement::Omit => false,
                TensorPlacement::Rank { rank } => rank == plan.topology.global_rank,
                TensorPlacement::PipelineStage { stage } => {
                    stage == plan.topology.pipeline_parallel_rank
                }
            };
            if locally_required && !self.loaded.contains(target) {
                missing.push(target.clone());
            }
        }
        missing.sort();
        let mut unexpected = self
            .unexpected
            .into_iter()
            .filter(|source| !config.is_unused_allowed(source))
            .collect::<Vec<_>>();
        unexpected.sort();
        unexpected.dedup();
        if missing.is_empty() && unexpected.is_empty() {
            Ok(())
        } else {
            Err(Error::StrictLoadValidation {
                missing,
                unused: unexpected,
            })
        }
    }
}

/// Selectively loads a safetensors checkpoint directory according to `plan`.
///
/// For indexed checkpoints, key rewrites and placement are resolved from the
/// index before any payload shard is opened. A shard containing no local
/// tensors is therefore skipped completely. Within an opened shard, omitted
/// tensors never become MLX arrays. Selected source views are sliced before
/// their final stream copy, then explicitly evaluated while the mmap is alive.
/// Peak temporary memory is bounded by the accumulated local partition plus at
/// most the selected source tensor currently being transformed.
pub fn load_safetensors_partition(
    model_dir: impl AsRef<Path>,
    plan: &PlacementPlan,
    stream: &Stream,
    config: &StrictLoadConfig,
) -> Result<RankPartition, Error> {
    load_safetensors_partition_on_streams(model_dir, plan, stream, stream, config)
}

/// Selectively loads on a source/weights stream, then places only local results
/// on `execution_stream`.
///
/// Use a CPU `source_stream` with a GPU `execution_stream` to ensure a full
/// source tensor is never copied to the GPU merely to discard other ranks'
/// slices. The source device holds at most the tensor currently being
/// transformed in addition to the accumulated local partition.
pub fn load_safetensors_partition_on_streams(
    model_dir: impl AsRef<Path>,
    plan: &PlacementPlan,
    source_stream: &Stream,
    execution_stream: &Stream,
    config: &StrictLoadConfig,
) -> Result<RankPartition, Error> {
    let store = SafetensorsWeightStore::open(model_dir)?;
    load_partition_from_store_on_streams(&store, plan, source_stream, execution_stream, config)
}

/// Selectively loads a rank partition from a reusable checkpoint store.
pub fn load_partition_from_store(
    store: &(impl WeightStore + ?Sized),
    plan: &PlacementPlan,
    stream: &Stream,
    config: &StrictLoadConfig,
) -> Result<RankPartition, Error> {
    load_partition_from_store_on_streams(store, plan, stream, stream, config)
}

/// Selectively loads a rank partition from a reusable checkpoint store using
/// explicit source and execution streams.
///
/// Placement is resolved from catalog metadata before a lease materializes an
/// array. Remote-only indexed shards are therefore never acquired or mapped.
pub fn load_partition_from_store_on_streams(
    store: &(impl WeightStore + ?Sized),
    plan: &PlacementPlan,
    source_stream: &Stream,
    execution_stream: &Stream,
    config: &StrictLoadConfig,
) -> Result<RankPartition, Error> {
    plan.validate()?;
    plan.topology.validate_execution_stream(execution_stream)?;
    let mut report = PartitionReport::default();
    let mut tensors = HashMap::new();
    let mut opened_shards = BTreeSet::new();

    for source in store.keys() {
        let SourcePlan::Known { target, tensor } = plan.source_plan(&source, config) else {
            report.unexpected.push(source);
            continue;
        };
        let potentially_local = !matches!(tensor.placement, TensorPlacement::Omit)
            && !matches!(tensor.placement, TensorPlacement::Rank { rank } if rank != plan.topology.global_rank)
            && !matches!(tensor.placement, TensorPlacement::PipelineStage { stage } if stage != plan.topology.pipeline_parallel_rank);
        if !potentially_local {
            continue;
        }

        let metadata = store.metadata(&source)?;
        let resolved =
            resolve_placement(&tensor, &metadata.shape, plan.topology).map_err(|error| {
                Error::Parallel(format!("checkpoint tensor {source} -> {target}: {error}"))
            })?;
        let selection = match resolved {
            ResolvedPlacement::Omit => continue,
            ResolvedPlacement::Materialize => TensorSelection::Full,
            ResolvedPlacement::Shard(slice) => TensorSelection::Range {
                axis: slice.axis,
                start: slice.start,
                end: slice.end,
            },
            ResolvedPlacement::Indices { axis, indices } => {
                TensorSelection::Indices { axis, indices }
            }
        };
        let lease =
            store.acquire_with_policy(&source, selection, WeightReadPolicy::RequireBounded)?;
        let value = lease
            .materialize(source_stream, execution_stream)?
            .synchronize()?;
        opened_shards.insert(lease.backing_shard().to_path_buf());
        report.loaded.insert(target.clone());
        if tensors.insert(target.clone(), value).is_some() {
            return Err(Error::Parallel(format!(
                "multiple checkpoint tensors resolved to local target {target}"
            )));
        }
    }

    report.finish(plan, config)?;
    Ok(RankPartition {
        topology: plan.topology,
        tensors,
        opened_shards: opened_shards.into_iter().collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

    fn stream() -> Stream {
        Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
    }

    fn write_index(dir: &Path, mappings: &[(&str, &str)]) {
        let weight_map = mappings
            .iter()
            .map(|(key, file)| ((*key).to_string(), serde_json::json!(file)))
            .collect::<serde_json::Map<_, _>>();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "metadata": {},
                "weight_map": weight_map,
            }))
            .unwrap(),
        )
        .unwrap();
    }

    fn write_i32_tensor(path: &Path, name: &str, values: &[i32], shape: Vec<usize>) {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let view = TensorView::new(Dtype::I32, shape, &bytes).unwrap();
        serialize_to_file([(name, view)], None, path).unwrap();
    }

    fn topology(world: usize, rank: usize, tp: usize, pp: usize, ep: usize) -> ParallelTopology {
        ParallelTopology::from_rank(
            world,
            rank,
            tp,
            pp,
            ep,
            DeviceAssignment::new(DeviceType::Cpu, 0),
        )
        .unwrap()
    }

    #[test]
    fn validates_topology_and_exhaustive_coordinate_ordering() {
        assert!(ParallelTopology::from_rank(
            8,
            0,
            0,
            2,
            2,
            DeviceAssignment::new(DeviceType::Cpu, 0)
        )
        .is_err());
        assert!(ParallelTopology::from_rank(
            1,
            0,
            1,
            1,
            1,
            DeviceAssignment::new(DeviceType::Cpu, usize::MAX)
        )
        .is_err());
        assert!(ParallelTopology::from_rank(
            7,
            0,
            2,
            2,
            2,
            DeviceAssignment::new(DeviceType::Cpu, 0)
        )
        .is_err());
        assert!(ParallelTopology::from_rank(
            8,
            8,
            2,
            2,
            2,
            DeviceAssignment::new(DeviceType::Cpu, 0)
        )
        .is_err());
        assert!(ParallelTopology::from_rank(
            usize::MAX,
            0,
            usize::MAX,
            2,
            1,
            DeviceAssignment::new(DeviceType::Cpu, 0)
        )
        .is_err());

        for pp_rank in 0..2 {
            for tp_rank in 0..3 {
                for ep_rank in 0..2 {
                    let rank = ((pp_rank * 3) + tp_rank) * 2 + ep_rank;
                    let value = ParallelTopology::from_rank(
                        12,
                        rank,
                        3,
                        2,
                        2,
                        DeviceAssignment::new(DeviceType::Cpu, 99),
                    )
                    .unwrap();
                    assert_eq!(value.pipeline_parallel_rank, pp_rank);
                    assert_eq!(value.tensor_parallel_rank, tp_rank);
                    assert_eq!(value.expert_parallel_rank, ep_rank);
                    assert_eq!(value.device.local_index, 99);
                    assert_ne!(value.device.local_index, value.global_rank);
                }
            }
        }
    }

    #[test]
    fn cartesian_subgroups_and_pipeline_lanes_are_authoritative() {
        let expected_tp = [vec![0, 2, 4], vec![1, 3, 5], vec![6, 8, 10], vec![7, 9, 11]];
        let expected_ep = [
            vec![0, 1],
            vec![2, 3],
            vec![4, 5],
            vec![6, 7],
            vec![8, 9],
            vec![10, 11],
        ];
        for rank in 0..12 {
            let topology = topology(12, rank, 3, 2, 2);
            let coordinates = topology.coordinates();
            assert_eq!(topology.global_rank_for(coordinates).unwrap(), rank);

            let tp = topology.subgroup(ParallelAxis::Tensor).unwrap();
            assert_eq!(tp.rank, coordinates.tensor);
            assert_eq!(tp.size, 3);
            assert_eq!(tp.global_ranks, expected_tp[tp.color]);

            let ep = topology.subgroup(ParallelAxis::Expert).unwrap();
            assert_eq!(ep.rank, coordinates.expert);
            assert_eq!(ep.size, 2);
            assert_eq!(ep.global_ranks, expected_ep[ep.color]);

            let pp = topology.subgroup(ParallelAxis::Pipeline).unwrap();
            assert_eq!(pp.rank, coordinates.pipeline);
            assert_eq!(pp.size, 2);
            assert_eq!(pp.global_ranks[coordinates.pipeline], rank);
            if coordinates.pipeline == 0 {
                assert_eq!(topology.pipeline_predecessor().unwrap(), None);
                assert_eq!(
                    topology.pipeline_successor().unwrap(),
                    Some(pp.global_ranks[1])
                );
                assert!(topology.owns_embedding());
                assert!(!topology.owns_output_head());
            } else {
                assert_eq!(
                    topology.pipeline_predecessor().unwrap(),
                    Some(pp.global_ranks[0])
                );
                assert_eq!(topology.pipeline_successor().unwrap(), None);
                assert!(!topology.owns_embedding());
                assert!(topology.owns_output_head());
            }
        }
    }

    #[test]
    fn pairwise_ownership_combines_pipeline_and_expert_coordinates() {
        for rank in 0..6 {
            let topology = topology(6, rank, 1, 3, 2);
            let layers = topology.layer_range(7).unwrap();
            let experts = topology.expert_range(5).unwrap();
            for layer in 0..7 {
                assert_eq!(
                    topology.owns_layer(7, layer).unwrap(),
                    layers.contains(&layer)
                );
            }
            for expert in 0..5 {
                assert_eq!(
                    topology.owns_expert(5, expert).unwrap(),
                    experts.contains(&expert)
                );
            }
        }
    }

    #[test]
    fn preflight_accepts_pairwise_and_triple_axis_geometry() {
        let tp_pp = topology(4, 3, 2, 2, 1).preflight(Some(5), None).unwrap();
        assert_eq!(tp_pp.local_layer_range, Some(3..5));
        assert_eq!(tp_pp.tensor_subgroup.global_ranks, [2, 3]);
        assert_eq!(tp_pp.pipeline_subgroup.global_ranks, [1, 3]);

        let tp_ep = topology(6, 4, 3, 1, 2).preflight(None, Some(7)).unwrap();
        assert_eq!(tp_ep.local_expert_range, Some(0..4));
        assert_eq!(tp_ep.tensor_subgroup.global_ranks, [0, 2, 4]);
        assert_eq!(tp_ep.expert_subgroup.global_ranks, [4, 5]);

        let pp_ep = topology(6, 5, 1, 3, 2).preflight(Some(7), Some(5)).unwrap();
        assert_eq!(pp_ep.local_layer_range, Some(5..7));
        assert_eq!(pp_ep.local_expert_range, Some(3..5));
        assert_eq!(pp_ep.pipeline_subgroup.global_ranks, [1, 3, 5]);

        let triple = topology(8, 0, 2, 2, 2).preflight(Some(4), Some(4)).unwrap();
        assert_eq!(triple.local_layer_range, Some(0..2));
        assert_eq!(triple.local_expert_range, Some(0..2));
        assert_eq!(triple.tensor_subgroup.global_ranks, [0, 2]);
        assert_eq!(triple.pipeline_subgroup.global_ranks, [0, 4]);
        assert_eq!(triple.expert_subgroup.global_ranks, [0, 1]);
        assert!(topology(4, 0, 1, 4, 1).preflight(None, None).is_err());
        assert!(topology(4, 0, 1, 1, 4).preflight(None, None).is_err());
    }

    #[test]
    fn ring_neighbor_routes_cover_arbitrary_stage_local_axis_degrees() {
        for rank in 0..18 {
            let topology = topology(18, rank, 3, 2, 3);
            for axis in [ParallelAxis::Tensor, ParallelAxis::Expert] {
                let routes = logical_stage_axis_routes(topology, axis).unwrap();
                assert_eq!(routes.len(), 3);
                let mut sources = routes.iter().map(|(source, _)| *source).collect::<Vec<_>>();
                sources.sort_unstable();
                assert_eq!(sources, [0, 1, 2]);
                assert!(routes.iter().flat_map(|(_, rounds)| rounds).all(|peer| {
                    peer.is_none_or(|peer| {
                        (rank + 1) % topology.world_size == peer
                            || (peer + 1) % topology.world_size == rank
                    })
                }));
            }
        }
    }

    #[test]
    fn balanced_ranges_cover_uneven_layers_and_experts() {
        let ranges = (0..3)
            .map(|index| balanced_contiguous_range(8, 3, index, false).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(ranges, [0..3, 3..6, 6..8]);
        assert!(balanced_contiguous_range(2, 3, 0, false).is_err());
        assert_eq!(balanced_contiguous_range(2, 3, 2, true).unwrap(), 2..2);
        assert_eq!(topology(6, 5, 1, 3, 2).layer_range(8).unwrap(), 6..8);
        assert_eq!(topology(6, 5, 1, 3, 2).expert_range(5).unwrap(), 3..5);
    }

    #[test]
    fn validates_tensor_slices() {
        let slice = TensorSlice::for_shape(&[4, 12], 1, 2, 3).unwrap();
        assert_eq!(slice.start, 8);
        assert_eq!(slice.end, 12);
        assert_eq!(slice.local_shape(&[4, 12]), [4, 4]);
        assert!(TensorSlice::for_shape(&[4, 11], 1, 0, 3).is_err());
        assert!(TensorSlice::for_shape(&[4, 12], 2, 0, 3).is_err());
        assert!(TensorSlice::for_shape(&[4, 12], 1, 3, 3).is_err());
    }

    #[test]
    fn validates_explicit_execution_stream_device() {
        let stream = stream();
        topology(1, 0, 1, 1, 1)
            .validate_execution_stream(&stream)
            .unwrap();
        let other_assignment =
            ParallelTopology::from_rank(1, 0, 1, 1, 1, DeviceAssignment::new(DeviceType::Cpu, 1))
                .unwrap();
        assert!(other_assignment.validate_execution_stream(&stream).is_err());
    }

    #[test]
    fn plan_exposes_replicated_omitted_and_quantized_companions() {
        let mut plan = PlacementPlan::new(topology(1, 0, 1, 1, 1));
        plan.insert("replicated", TensorPlacement::Replicated);
        plan.insert("remote", TensorPlacement::Omit);
        plan.insert_quantized_companions("projection", TensorPlacement::Local, true);
        assert_eq!(
            plan.placement("replicated"),
            Some(&TensorPlacement::Replicated)
        );
        assert_eq!(plan.placement("remote"), Some(&TensorPlacement::Omit));
        assert_eq!(
            plan.placement("projection.weight"),
            Some(&TensorPlacement::Local)
        );
        assert_eq!(
            plan.placement("projection.scales"),
            Some(&TensorPlacement::Local)
        );
        assert_eq!(
            plan.placement("projection.biases"),
            Some(&TensorPlacement::Local)
        );

        let mut invalid = PlacementPlan::new(topology(1, 0, 1, 1, 1));
        invalid.insert("bad_owner", TensorPlacement::Rank { rank: 1 });
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn plan_supports_balanced_uneven_ranges() {
        let mut plan = PlacementPlan::new(topology(3, 2, 3, 1, 1));
        let range = plan
            .insert_balanced_tensor_parallel("embedding.weight", 0, 11)
            .unwrap();
        assert_eq!(range, 8..11);
        assert_eq!(
            plan.placement("embedding.weight"),
            Some(&TensorPlacement::Range {
                axis: 0,
                start: 8,
                end: 11,
            })
        );
        plan.insert_expected(
            "head.weight",
            vec![11, 4],
            TensorPlacement::Range {
                axis: 0,
                start: 8,
                end: 11,
            },
        )
        .unwrap();
        plan.validate().unwrap();
    }

    #[test]
    fn model_load_options_preserve_singleton_behavior_and_reject_partial_models() {
        let default = crate::api::ModelLoadOptions::default();
        assert_eq!(default.quantization, None);
        assert_eq!(default.parallel, None);
        crate::api::ensure_executable_load_options(default).unwrap();

        let singleton = crate::api::ModelLoadOptions::with_parallel(topology(1, 0, 1, 1, 1));
        crate::api::ensure_executable_load_options(singleton).unwrap();
        let combined = crate::api::ModelLoadOptions::with_quantization(
            crate::runtime::checkpoint::quantization::WeightQuantization::MxFp4,
        )
        .with_parallel_topology(topology(1, 0, 1, 1, 1));
        assert_eq!(
            combined.quantization,
            Some(crate::runtime::checkpoint::quantization::WeightQuantization::MxFp4)
        );
        assert!(combined.parallel.unwrap().is_replicated());

        let partitioned = crate::api::ModelLoadOptions::with_parallel(topology(2, 0, 2, 1, 1));
        assert!(matches!(
            crate::api::ensure_executable_load_options(partitioned),
            Err(Error::Parallel(_))
        ));
    }

    #[test]
    fn typed_rank_and_pipeline_ownership_resolve_locally() {
        let rank_zero = topology(4, 0, 2, 2, 1);
        let rank_three = topology(4, 3, 2, 2, 1);
        let rank_owned = TensorPlan {
            placement: TensorPlacement::Rank { rank: 3 },
            expected_source_shape: None,
        };
        assert!(matches!(
            resolve_placement(&rank_owned, &[2], rank_zero).unwrap(),
            ResolvedPlacement::Omit
        ));
        assert!(matches!(
            resolve_placement(&rank_owned, &[2], rank_three).unwrap(),
            ResolvedPlacement::Materialize
        ));

        let stage_owned = TensorPlan {
            placement: TensorPlacement::PipelineStage { stage: 1 },
            expected_source_shape: None,
        };
        assert!(matches!(
            resolve_placement(&stage_owned, &[2], rank_zero).unwrap(),
            ResolvedPlacement::Omit
        ));
        assert!(matches!(
            resolve_placement(&stage_owned, &[2], rank_three).unwrap(),
            ResolvedPlacement::Materialize
        ));
    }

    #[test]
    fn selective_loader_skips_remote_shards_and_reconstructs_tp_slices() {
        let dir = tempfile::tempdir().unwrap();
        let stream = stream();
        write_i32_tensor(
            &dir.path().join("local.safetensors"),
            "model.projection.weight",
            &[0, 1, 2, 3, 10, 11, 12, 13],
            vec![2, 4],
        );
        // This is deliberately not a safetensors file. Correct index-level
        // selection must never open it for either rank.
        std::fs::write(dir.path().join("remote.safetensors"), b"must not be opened").unwrap();
        write_index(
            dir.path(),
            &[
                ("model.projection.weight", "local.safetensors"),
                ("model.remote.weight", "remote.safetensors"),
            ],
        );

        let mut reconstructed = Vec::new();
        for rank in 0..2 {
            let topology = topology(2, rank, 2, 1, 1);
            let mut plan = PlacementPlan::new(topology);
            plan.insert_expected(
                "projection.weight",
                vec![2, 4],
                TensorPlacement::Shard {
                    axis: 1,
                    index: rank,
                    parts: 2,
                },
            )
            .unwrap();
            plan.insert("remote.weight", TensorPlacement::Omit);
            let config = StrictLoadConfig::default().strip_prefix("model.");
            let partition =
                load_safetensors_partition(dir.path(), &plan, &stream, &config).unwrap();
            assert_eq!(partition.len(), 1);
            assert_eq!(
                partition.opened_shards(),
                &[dir.path().join("local.safetensors")]
            );
            assert!(partition.get("remote.weight").is_none());
            let local = partition
                .get("projection.weight")
                .unwrap()
                .evaluated()
                .unwrap();
            assert_eq!(local.as_array().shape(), &[2, 2]);
            reconstructed.push(local.as_slice::<i32>().to_vec());
        }
        // Slices are axis-1 contiguous views, so reconstruct each row from
        // the corresponding rows of both rank-local tensors.
        assert_eq!(reconstructed[0], [0, 1, 10, 11]);
        assert_eq!(reconstructed[1], [2, 3, 12, 13]);
        let union = [
            reconstructed[0][0],
            reconstructed[0][1],
            reconstructed[1][0],
            reconstructed[1][1],
            reconstructed[0][2],
            reconstructed[0][3],
            reconstructed[1][2],
            reconstructed[1][3],
        ];
        assert_eq!(union, [0, 1, 2, 3, 10, 11, 12, 13]);
    }

    #[test]
    fn selective_loader_materializes_only_ordered_noncontiguous_indices() {
        let dir = tempfile::tempdir().unwrap();
        let stream = stream();
        write_i32_tensor(
            &dir.path().join("model.safetensors"),
            "experts",
            &[0, 1, 10, 11, 20, 21, 30, 31, 40, 41],
            vec![5, 2],
        );
        let mut plan = PlacementPlan::new(topology(2, 1, 1, 1, 2));
        plan.insert_expected(
            "experts",
            vec![5, 2],
            TensorPlacement::Indices {
                axis: 0,
                indices: vec![3, 1],
            },
        )
        .unwrap();
        let partition =
            load_safetensors_partition(dir.path(), &plan, &stream, &StrictLoadConfig::default())
                .unwrap();
        let local = partition.get("experts").unwrap().evaluated().unwrap();
        assert_eq!(local.as_array().shape(), &[2, 2]);
        assert_eq!(local.as_slice::<i32>(), &[30, 31, 10, 11]);

        for indices in [vec![], vec![1, 1], vec![1, 5]] {
            let mut invalid = PlacementPlan::new(topology(2, 1, 1, 1, 2));
            assert!(invalid
                .insert_expected(
                    "experts",
                    vec![5, 2],
                    TensorPlacement::Indices { axis: 0, indices },
                )
                .is_err());
        }
    }

    #[test]
    fn replicated_default_loads_the_original_full_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let stream = stream();
        write_i32_tensor(
            &dir.path().join("model.safetensors"),
            "weight",
            &[3, 5, 7, 9],
            vec![2, 2],
        );
        let plan = PlacementPlan::replicated(topology(1, 0, 1, 1, 1));
        let partition =
            load_safetensors_partition(dir.path(), &plan, &stream, &StrictLoadConfig::default())
                .unwrap();
        let loaded = partition.get("weight").unwrap().evaluated().unwrap();
        assert_eq!(loaded.as_slice::<i32>(), &[3, 5, 7, 9]);
    }

    #[test]
    fn omitted_unsupported_tensor_is_never_materialized() {
        let dir = tempfile::tempdir().unwrap();
        let bytes = [0u8; 4];
        let unsupported = TensorView::new(Dtype::F8_E5M2, vec![4], &bytes).unwrap();
        serialize_to_file(
            [("remote", unsupported)],
            None,
            &dir.path().join("model.safetensors"),
        )
        .unwrap();
        let mut plan = PlacementPlan::new(topology(1, 0, 1, 1, 1));
        plan.insert("remote", TensorPlacement::Omit);
        let partition =
            load_safetensors_partition(dir.path(), &plan, &stream(), &StrictLoadConfig::default())
                .unwrap();
        assert!(partition.is_empty());
    }

    #[test]
    fn remote_only_index_shard_is_never_opened() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("remote.safetensors"), b"not safetensors").unwrap();
        write_index(dir.path(), &[("remote.weight", "remote.safetensors")]);
        let mut plan = PlacementPlan::new(topology(1, 0, 1, 1, 1));
        plan.insert("remote.weight", TensorPlacement::Omit);
        let partition =
            load_safetensors_partition(dir.path(), &plan, &stream(), &StrictLoadConfig::default())
                .unwrap();
        assert!(partition.is_empty());
        assert!(partition.opened_shards().is_empty());
    }

    #[test]
    fn strict_partition_rejects_missing_malformed_and_unexpected_local_tensors() {
        let dir = tempfile::tempdir().unwrap();
        let stream = stream();
        write_i32_tensor(
            &dir.path().join("model.safetensors"),
            "present",
            &[1, 2, 3, 4],
            vec![2, 2],
        );
        let topology = topology(1, 0, 1, 1, 1);

        let mut malformed = PlacementPlan::new(topology);
        malformed
            .insert_expected("present", vec![4, 2], TensorPlacement::Local)
            .unwrap();
        assert!(matches!(
            load_safetensors_partition(
                dir.path(),
                &malformed,
                &stream,
                &StrictLoadConfig::default()
            ),
            Err(Error::Parallel(_))
        ));

        let mut missing = PlacementPlan::new(topology);
        missing.insert("present", TensorPlacement::Omit);
        missing.insert("required", TensorPlacement::Local);
        let error =
            load_safetensors_partition(dir.path(), &missing, &stream, &StrictLoadConfig::default())
                .unwrap_err();
        match error {
            Error::StrictLoadValidation { missing, unused } => {
                assert_eq!(missing, ["required"]);
                assert!(unused.is_empty());
            }
            other => panic!("unexpected error: {other}"),
        }

        let strict_empty = PlacementPlan::new(topology);
        let error = load_safetensors_partition(
            dir.path(),
            &strict_empty,
            &stream,
            &StrictLoadConfig::default(),
        )
        .unwrap_err();
        match error {
            Error::StrictLoadValidation { missing, unused } => {
                assert!(missing.is_empty());
                assert_eq!(unused, ["present"]);
            }
            other => panic!("unexpected error: {other}"),
        }

        let allowed = load_safetensors_partition(
            dir.path(),
            &strict_empty,
            &stream,
            &StrictLoadConfig::default().allow_unused_prefix("present"),
        )
        .unwrap();
        assert!(allowed.is_empty());
    }
}

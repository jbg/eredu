//! MLX checkpoint placement and selective materialization.

use std::{
    collections::{BTreeSet, HashMap, HashSet},
    ops::Range,
    path::{Path, PathBuf},
};

use eredu_checkpoint::store::{
    CheckpointSource, ReadPolicy, SafetensorsWeightStore, TensorReadRequest, TensorSelection,
};
use eredu_core::{
    balanced_contiguous_range, ParallelAxis, ParallelRankTopology, SubgroupMembership,
};
use eredu_runtime::TensorPlacement;
use safemlx::{distributed::Group, Array, Stream};

use crate::{
    backend::error::Error, backend::runtime::checkpoint::load::StrictLoadConfig,
    backend::runtime::checkpoint::store::MlxParameterMaterializationContext,
};

#[cfg(test)]
use crate::backend::DeviceAssignment;
use crate::backend::MlxParallelContext;
#[cfg(test)]
use safemlx::{Device, DeviceType};

/// Backend communication contexts materialized from one Cartesian topology.
///
/// Construction is collective when a non-global subgroup must be split. All
/// ranks must call [`Self::new`] in the same order. Singleton axes do not own a
/// communication group, while an axis spanning the complete world borrows the
/// original group without splitting it.
pub struct ParallelCommunicators<'a> {
    topology: MlxParallelContext,
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
    pub fn new(topology: MlxParallelContext, world: &'a Group) -> Result<Self, Error> {
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
        topology: MlxParallelContext,
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
    pub const fn world(&self) -> &Group {
        self.world
    }

    /// Returns the native group for a non-singleton axis.
    pub fn group(&self, axis: ParallelAxis) -> Option<&Group> {
        let communicator = match axis {
            ParallelAxis::Tensor => &self.tensor,
            ParallelAxis::Pipeline => &self.pipeline,
            ParallelAxis::Expert => &self.expert,
            ParallelAxis::Data => return None,
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
    pub fn tensor_group(&self) -> Option<&Group> {
        self.group(ParallelAxis::Tensor)
    }

    /// Returns the pipeline-lane consensus group, or `None` when PP is inactive.
    pub fn pipeline_group(&self) -> Option<&Group> {
        self.group(ParallelAxis::Pipeline)
    }

    /// Returns the EP exchange group, or `None` when EP is inactive.
    pub fn expert_group(&self) -> Option<&Group> {
        self.group(ParallelAxis::Expert)
    }
}

fn logical_stage_axis_routes(
    topology: MlxParallelContext,
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
        ParallelAxis::Data => {
            return Err(Error::Parallel(
                "MLX data-parallel logical routes are not implemented".into(),
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
                    ParallelAxis::Pipeline => unreachable!(),
                    ParallelAxis::Data => unreachable!(),
                }
                Ok(topology.global_rank_for(coordinates)?)
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

#[derive(Debug, Clone)]
struct TensorPlan {
    placement: TensorPlacement,
    expected_source_shape: Option<Vec<usize>>,
}

/// Inspectable mapping from rewritten target names to typed placement decisions.
#[derive(Debug, Clone)]
pub struct PlacementPlan {
    topology: MlxParallelContext,
    tensors: HashMap<String, TensorPlan>,
    default: Option<TensorPlacement>,
}

impl PlacementPlan {
    /// Creates a strict plan in which every checkpoint tensor must be named.
    pub fn new(topology: MlxParallelContext) -> Self {
        Self {
            topology,
            tensors: HashMap::new(),
            default: None,
        }
    }

    /// Creates a plan that replicates every checkpoint tensor.
    pub fn replicated(topology: MlxParallelContext) -> Self {
        Self::new(topology).with_default(TensorPlacement::Replicated)
    }

    /// Sets the placement used for checkpoint keys without an explicit entry.
    pub fn with_default(mut self, placement: TensorPlacement) -> Self {
        self.default = Some(placement);
        self
    }

    /// Returns the topology captured by this plan.
    pub const fn topology(&self) -> MlxParallelContext {
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

fn validate_plan_entry(plan: &TensorPlan, topology: MlxParallelContext) -> Result<(), Error> {
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
    topology: MlxParallelContext,
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
    topology: MlxParallelContext,
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
    topology: MlxParallelContext,
    tensors: HashMap<String, Array>,
    opened_shards: Vec<PathBuf>,
}

impl RankPartition {
    /// Returns the validated topology used for this partition.
    pub const fn topology(&self) -> MlxParallelContext {
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
    store: &(impl CheckpointSource + ?Sized),
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
    store: &(impl CheckpointSource + ?Sized),
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
    let context = MlxParameterMaterializationContext::new(source_stream, execution_stream);

    for source in store.source_keys() {
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

        let metadata = store.source_metadata(&source)?;
        let resolved = resolve_placement(&tensor, &metadata.logical_shape, plan.topology).map_err(
            |error| Error::Parallel(format!("checkpoint tensor {source} -> {target}: {error}")),
        )?;
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
        let lease = store.acquire_lease(TensorReadRequest {
            key: source.clone(),
            selection,
            policy: ReadPolicy::RequireBounded,
        })?;
        if let Some(path) = eredu_checkpoint::store::EncodedTensorLease::backing_path(&lease) {
            opened_shards.insert(path.to_path_buf());
        }
        let value = context
            .weight_lease(lease)?
            .materialize(source_stream, execution_stream)?
            .synchronize()?;
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

    fn topology(rank: usize, tp: usize, pp: usize, ep: usize) -> MlxParallelContext {
        MlxParallelContext::for_rank(rank, tp, pp, ep, DeviceAssignment::new(DeviceType::Cpu, 0))
            .unwrap()
    }

    #[test]
    fn ring_neighbor_routes_cover_arbitrary_stage_local_axis_degrees() {
        for rank in 0..18 {
            let topology = topology(rank, 3, 2, 3);
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
        topology(0, 1, 1, 1)
            .validate_execution_stream(&stream)
            .unwrap();
        let other_assignment =
            MlxParallelContext::for_rank(0, 1, 1, 1, DeviceAssignment::new(DeviceType::Cpu, 1))
                .unwrap();
        assert!(other_assignment.validate_execution_stream(&stream).is_err());
    }

    #[test]
    fn plan_exposes_replicated_omitted_and_quantized_companions() {
        let mut plan = PlacementPlan::new(topology(0, 1, 1, 1));
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

        let mut invalid = PlacementPlan::new(topology(0, 1, 1, 1));
        invalid.insert("bad_owner", TensorPlacement::Rank { rank: 1 });
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn plan_supports_balanced_uneven_ranges() {
        let mut plan = PlacementPlan::new(topology(2, 3, 1, 1));
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
    fn replicated_facades_reject_distributed_session_topologies() {
        let default = crate::backend::ModelLoadOptions::default();
        assert_eq!(default.quantization, None);
        assert_eq!(default.parallel, None);
        crate::backend::ensure_replicated_load_options(default).unwrap();

        let singleton = crate::backend::ModelLoadOptions::with_parallel(topology(0, 1, 1, 1));
        crate::backend::ensure_replicated_load_options(singleton).unwrap();
        let combined = crate::backend::ModelLoadOptions::with_quantization(
            eredu_checkpoint::WeightQuantization::MxFp4,
        )
        .with_parallel_topology(topology(0, 1, 1, 1));
        assert_eq!(
            combined.quantization,
            Some(eredu_checkpoint::WeightQuantization::MxFp4)
        );
        assert!(combined.parallel.unwrap().is_replicated());

        let tensor_parallel = crate::backend::ModelLoadOptions::with_parallel(topology(0, 2, 1, 1));
        assert!(crate::backend::ensure_replicated_load_options(tensor_parallel).is_err());

        let pipeline_partitioned =
            crate::backend::ModelLoadOptions::with_parallel(topology(0, 1, 2, 1));
        assert!(crate::backend::ensure_replicated_load_options(pipeline_partitioned).is_err());

        let expert_partitioned =
            crate::backend::ModelLoadOptions::with_parallel(topology(0, 1, 1, 2));
        assert!(crate::backend::ensure_replicated_load_options(expert_partitioned).is_err());
    }

    #[test]
    fn typed_rank_and_pipeline_ownership_resolve_locally() {
        let rank_zero = topology(0, 2, 2, 1);
        let rank_three = topology(3, 2, 2, 1);
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
            let topology = topology(rank, 2, 1, 1);
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
        let mut plan = PlacementPlan::new(topology(1, 1, 1, 2));
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
            let mut invalid = PlacementPlan::new(topology(1, 1, 1, 2));
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
        let plan = PlacementPlan::replicated(topology(0, 1, 1, 1));
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
        let mut plan = PlacementPlan::new(topology(0, 1, 1, 1));
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
        let mut plan = PlacementPlan::new(topology(0, 1, 1, 1));
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
        let topology = topology(0, 1, 1, 1);

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

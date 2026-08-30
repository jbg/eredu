//! Typed, architecture-neutral tensor-parallel planning.
//!
//! Architecture adapters describe logical parameter roles and exact checkpoint
//! members. This module converts those descriptions into rank-local placement
//! and shape information without inspecting checkpoint-name substrings.

use eredu_runtime::{
    LocalModelLayout, LocalTensorLayout, MemberSharding, ParameterGroupSpec, ParameterMemberSpec,
    ParameterRole, Sampler, ShardingPolicy, TensorPlacement,
};

use safemlx::{
    distributed::{self, Group},
    ops::{indexing::TryIndexOp, ones, zeros},
    Array, Dtype, Stream,
};

use crate::{
    backend::error::Error,
    backend::runtime::distributed::{completion::synchronize_outputs, topology::PlacementPlan},
    backend::runtime::generation::MlxSamplingBackend,
    backend::MlxParallelContext,
    MlxTensor,
};
use eredu_core::balanced_contiguous_range;

/// Token selected on one distributed rank together with synchronized stop state.
#[derive(Debug)]
pub struct SynchronizedToken {
    /// Selected token ids with shape `[batch, 1]` on every rank.
    pub token: Array,
    /// Whether every rank should terminate generation.
    pub finished: bool,
}

/// Samples on one designated rank and synchronizes only token ids and stop state.
///
/// `logits` is required on `sampling_rank` and ignored elsewhere. Accepting an
/// optional value lets pipeline stages avoid retaining full logits while TP and
/// EP callers may pass their identical complete logits on every rank.
#[allow(clippy::too_many_arguments)]
pub fn sample_and_synchronize<S: Sampler<MlxSamplingBackend>>(
    logits: Option<&MlxTensor>,
    batch_size: i32,
    sampler: &mut S,
    temperature: f32,
    prng_state: Option<&mut safemlx::random::RandomState>,
    finished: bool,
    sampling_rank: usize,
    group: &Group,
    stream: &Stream,
) -> Result<SynchronizedToken, Error> {
    if sampling_rank >= group.size() {
        return Err(Error::Parallel(format!(
            "sampling rank {sampling_rank} is outside distributed group size {}",
            group.size()
        )));
    }
    if batch_size <= 0 {
        return Err(Error::Parallel(format!(
            "distributed sampling batch size must be positive, got {batch_size}"
        )));
    }
    let local_token = if group.rank() == sampling_rank {
        let logits = logits.ok_or_else(|| {
            Error::Parallel(format!(
                "sampling rank {sampling_rank} requires complete logits"
            ))
        })?;
        if logits.as_array().dim(0) != batch_size {
            return Err(Error::Parallel(format!(
                "sampling logits batch {} does not match declared batch {batch_size}",
                logits.as_array().dim(0)
            )));
        }
        let logits = if logits.as_array().ndim() == 3 {
            MlxTensor::from_array(logits.as_array().try_index_device((.., -1, ..), stream)?)
        } else {
            logits.clone()
        };
        Sampler::<MlxSamplingBackend>::sample(sampler, &logits, temperature, prng_state, stream)?
            .into_array()
            .reshape(&[batch_size, 1], stream)?
            .as_dtype(Dtype::Uint32, stream)?
    } else {
        zeros::<u32>(&[batch_size, 1], stream)?
    };
    // MLX Ring reductions operate on floating payloads. Reducing an integer
    // token array can preserve the integer dtype while interpreting its bytes
    // as floats. Token ids are exactly representable in f32 throughout the
    // supported vocabulary range, so cross the collective boundary as f32 and
    // restore the public u32 contract afterward.
    let token = distributed::all_sum(
        &local_token.as_dtype(Dtype::Float32, stream)?,
        group,
        stream,
    )?
    .as_dtype(Dtype::Uint32, stream)?;
    let local_finished = if group.rank() == sampling_rank && finished {
        ones::<f32>(&[], stream)?
    } else {
        zeros::<f32>(&[], stream)?
    };
    let finished = distributed::all_sum(&local_finished, group, stream)?;
    synchronize_outputs([&token, &finished])?;
    Ok(SynchronizedToken {
        token,
        finished: finished.try_item::<f32>(stream)? != 0.0,
    })
}

/// Builds checkpoint placement and local model geometry from typed roles.
pub struct ParallelPlanBuilder {
    topology: MlxParallelContext,
    policy: ShardingPolicy,
    placement: PlacementPlan,
    local: LocalModelLayout<TensorPlacement>,
}

impl ParallelPlanBuilder {
    /// Creates an empty strict planner for one rank.
    pub fn new(topology: MlxParallelContext) -> Self {
        Self::with_policy(topology, ShardingPolicy::Require)
    }

    /// Creates an empty planner with an explicit fallback policy.
    pub fn with_policy(topology: MlxParallelContext, policy: ShardingPolicy) -> Self {
        Self {
            topology,
            policy,
            placement: PlacementPlan::new(topology),
            local: LocalModelLayout::default(),
        }
    }

    /// Registers one atomic logical parameter group.
    pub fn register(&mut self, group: ParameterGroupSpec) -> Result<(), Error> {
        if let Some(member) = group
            .members()
            .iter()
            .find(|member| self.local.contains(member.target()))
        {
            return Err(Error::Parallel(format!(
                "parallel placement target {:?} was registered more than once",
                member.target()
            )));
        }
        let requested = group
            .members()
            .iter()
            .map(|member| self.resolve_member(member, group.partition_units()))
            .collect::<Result<Vec<_>, _>>();
        let (resolved, fell_back) = match requested {
            Ok(resolved) => (resolved, false),
            Err(_error) if self.policy == ShardingPolicy::ReplicateUnsupported => (
                group
                    .members()
                    .iter()
                    .map(|member| (TensorPlacement::Replicated, member.global_shape().to_vec()))
                    .collect(),
                true,
            ),
            Err(error) => {
                return Err(Error::Parallel(format!(
                    "logical parameter {:?}: {error}",
                    group.logical_name()
                )))
            }
        };
        let logical_range = match (group.partition_units(), fell_back) {
            (Some(units), false) => Some(
                balanced_contiguous_range(
                    units,
                    self.topology.tensor_parallel_size,
                    self.topology.tensor_parallel_rank,
                    false,
                )
                .map_err(|error| Error::Parallel(error.to_string()))?,
            ),
            (Some(units), true) => Some(0..units),
            (None, _) => None,
        };

        for (member, (placement, local_shape)) in group.members().iter().zip(resolved) {
            self.placement.insert_expected(
                member.target().to_owned(),
                member.global_shape().to_vec(),
                placement.clone(),
            )?;
            self.local.insert(
                member.target().to_owned(),
                LocalTensorLayout::new(
                    group.logical_name(),
                    group.role(),
                    member.global_shape().to_vec(),
                    local_shape,
                    placement,
                    group.partition_units(),
                    logical_range.clone(),
                    fell_back,
                ),
            );
        }
        Ok(())
    }

    /// Completes planning and validates every generated placement.
    pub fn finish(self) -> Result<(PlacementPlan, LocalModelLayout<TensorPlacement>), Error> {
        self.placement.validate()?;
        Ok((self.placement, self.local))
    }

    fn resolve_member(
        &self,
        member: &ParameterMemberSpec,
        partition_units: Option<usize>,
    ) -> Result<(TensorPlacement, Vec<usize>), String> {
        let rank = self.topology.tensor_parallel_rank;
        let parts = self.topology.tensor_parallel_size;
        if parts == 0 || rank >= parts {
            return Err(format!("invalid TP coordinate {rank}/{parts}"));
        }
        match member.sharding() {
            MemberSharding::Replicated => {
                Ok((TensorPlacement::Replicated, member.global_shape().to_vec()))
            }
            MemberSharding::Equal { axis } => {
                let dimension = checked_axis(member, *axis)?;
                if dimension % parts != 0 {
                    return Err(format!(
                        "tensor {:?} dimension {dimension} on axis {axis} is not divisible by TP size {parts}",
                        member.target()
                    ));
                }
                let mut local_shape = member.global_shape().to_vec();
                local_shape[*axis] = dimension / parts;
                Ok((
                    TensorPlacement::Shard {
                        axis: *axis,
                        index: rank,
                        parts,
                    },
                    local_shape,
                ))
            }
            MemberSharding::Balanced { axis } => {
                let dimension = checked_axis(member, *axis)?;
                let range = balanced_contiguous_range(dimension, parts, rank, false)
                    .map_err(|error| error.to_string())?;
                let mut local_shape = member.global_shape().to_vec();
                local_shape[*axis] = range.len();
                Ok((
                    TensorPlacement::Range {
                        axis: *axis,
                        start: range.start,
                        end: range.end,
                    },
                    local_shape,
                ))
            }
            MemberSharding::Partitioned { axis } => {
                let units = partition_units.ok_or_else(|| {
                    format!(
                        "tensor {:?} has no group-level logical partition",
                        member.target()
                    )
                })?;
                let dimension = checked_axis(member, *axis)?;
                if dimension % units != 0 {
                    return Err(format!(
                        "tensor {:?} dimension {dimension} on axis {axis} does not contain {units} integral logical units",
                        member.target()
                    ));
                }
                let logical = balanced_contiguous_range(units, parts, rank, false)
                    .map_err(|error| error.to_string())?;
                let elements_per_unit = dimension / units;
                let range = (logical.start * elements_per_unit)..(logical.end * elements_per_unit);
                let mut local_shape = member.global_shape().to_vec();
                local_shape[*axis] = range.len();
                Ok((
                    TensorPlacement::Range {
                        axis: *axis,
                        start: range.start,
                        end: range.end,
                    },
                    local_shape,
                ))
            }
            MemberSharding::PartitionedSegments { axis, segments } => {
                let units = partition_units.ok_or_else(|| {
                    format!(
                        "tensor {:?} has no group-level logical partition",
                        member.target()
                    )
                })?;
                let dimension = checked_axis(member, *axis)?;
                if segments.is_empty() {
                    return Err(format!(
                        "tensor {:?} partitioned placement has no segments",
                        member.target()
                    ));
                }
                let logical = balanced_contiguous_range(units, parts, rank, false)
                    .map_err(|error| error.to_string())?;
                let mut indices = Vec::new();
                let mut previous_end = 0usize;
                for segment in segments {
                    if segment.start >= segment.end || segment.end > dimension {
                        return Err(format!(
                            "tensor {:?} segment {:?} is invalid for axis-{axis} dimension {dimension}",
                            member.target(), segment
                        ));
                    }
                    if segment.start < previous_end {
                        return Err(format!(
                            "tensor {:?} partitioned ranges overlap or are out of order",
                            member.target()
                        ));
                    }
                    previous_end = segment.end;
                    if !segment.len().is_multiple_of(units) {
                        return Err(format!(
                            "tensor {:?} segment {:?} does not contain {units} integral logical units",
                            member.target(), segment
                        ));
                    }
                    let elements_per_unit = segment.len() / units;
                    indices.extend(
                        (segment.start + logical.start * elements_per_unit)
                            ..(segment.start + logical.end * elements_per_unit),
                    );
                }
                let mut local_shape = member.global_shape().to_vec();
                local_shape[*axis] = indices.len();
                Ok((
                    TensorPlacement::Indices {
                        axis: *axis,
                        indices,
                    },
                    local_shape,
                ))
            }
            MemberSharding::Segmented { axis, segments } => {
                let dimension = checked_axis(member, *axis)?;
                if segments.is_empty() {
                    return Err(format!(
                        "tensor {:?} segmented placement has no segments",
                        member.target()
                    ));
                }
                let mut indices = Vec::new();
                let mut previous_end = 0usize;
                for segment in segments {
                    if segment.start >= segment.end || segment.end > dimension {
                        return Err(format!(
                            "tensor {:?} segment {:?} is invalid for axis-{axis} dimension {dimension}",
                            member.target(), segment
                        ));
                    }
                    if segment.start < previous_end {
                        return Err(format!(
                            "tensor {:?} segmented ranges overlap or are out of order",
                            member.target()
                        ));
                    }
                    previous_end = segment.end;
                    let local = balanced_contiguous_range(segment.len(), parts, rank, false)
                        .map_err(|error| error.to_string())?;
                    indices.extend((segment.start + local.start)..(segment.start + local.end));
                }
                if indices.is_empty() {
                    return Err(format!(
                        "tensor {:?} has no local segmented indices on TP rank {rank}",
                        member.target()
                    ));
                }
                let mut local_shape = member.global_shape().to_vec();
                local_shape[*axis] = indices.len();
                Ok((
                    TensorPlacement::Indices {
                        axis: *axis,
                        indices,
                    },
                    local_shape,
                ))
            }
        }
    }
}

/// Returns the planner-owned source-channel range for one routed expert bank.
///
/// Quantized plans may partition in blocks rather than individual channels, so
/// the architecture-local bank width must be mapped through the planner's
/// logical range rather than multiplied by a tensor-parallel rank.
pub(crate) fn routed_expert_intermediate_range(
    layout: &LocalModelLayout<TensorPlacement>,
    global_experts: usize,
    local_width: usize,
) -> Result<std::ops::Range<usize>, Error> {
    let mut selected = None;
    for (target, tensor) in layout.tensors().filter(|(_, tensor)| {
        tensor.role() == ParameterRole::ExpertIntermediate
            && tensor.global_shape().len() == 3
            && tensor.global_shape().first().copied() == Some(global_experts)
    }) {
        let units = tensor.logical_units().ok_or_else(|| {
            Error::Parallel(format!(
                "routed expert tensor {target:?} has no logical partition domain"
            ))
        })?;
        let logical = tensor.logical_range().ok_or_else(|| {
            Error::Parallel(format!(
                "routed expert tensor {target:?} has no local logical range"
            ))
        })?;
        if units == 0 || logical.is_empty() || !local_width.is_multiple_of(logical.len()) {
            return Err(Error::Parallel(format!(
                "routed expert tensor {target:?} maps local width {local_width} onto incompatible logical range {logical:?} of {units} units"
            )));
        }
        let channels_per_unit = local_width / logical.len();
        let range = (logical.start * channels_per_unit)..(logical.end * channels_per_unit);
        if selected.as_ref().is_some_and(|current| current != &range) {
            return Err(Error::Parallel(
                "routed expert tensors have inconsistent intermediate ranges".into(),
            ));
        }
        selected = Some(range);
    }
    selected.ok_or_else(|| {
        Error::Parallel(format!(
            "parallel plan has no routed expert bank with {global_experts} experts"
        ))
    })
}

fn checked_axis(member: &ParameterMemberSpec, axis: usize) -> Result<usize, String> {
    member.global_shape().get(axis).copied().ok_or_else(|| {
        format!(
            "tensor {:?} axis {axis} is outside shape {:?}",
            member.target(),
            member.global_shape()
        )
    })
}

/// Construction-time topology and fallback policy supplied to model builders.
#[derive(Debug, Clone, Copy)]
pub struct ParallelBuildContext {
    topology: MlxParallelContext,
    policy: ShardingPolicy,
}

impl ParallelBuildContext {
    /// Creates a construction context for a validated topology.
    pub const fn new(topology: MlxParallelContext, policy: ShardingPolicy) -> Self {
        Self { topology, policy }
    }

    /// Returns the complete process topology.
    pub const fn topology(self) -> MlxParallelContext {
        self.topology
    }

    /// Returns the configured unsupported-shard behavior.
    pub const fn policy(self) -> ShardingPolicy {
        self.policy
    }

    /// Creates a typed parameter planner for this rank.
    pub fn planner(self) -> ParallelPlanBuilder {
        ParallelPlanBuilder::with_policy(self.topology, self.policy)
    }
}

/// Borrowed execution resources for replicated or tensor-parallel primitives.
///
/// The group is never retained by model state. In hybrid topologies callers
/// supply the TP subgroup, whose rank is the tensor-parallel coordinate.
pub struct ParallelExecutionContext<'a> {
    topology: Option<MlxParallelContext>,
    group: Option<&'a Group>,
    stream: &'a Stream,
}

impl<'a> ParallelExecutionContext<'a> {
    /// Creates a singleton replicated execution context.
    pub const fn replicated(stream: &'a Stream) -> Self {
        Self {
            topology: None,
            group: None,
            stream,
        }
    }

    /// Creates a tensor-parallel context from a topology-derived subgroup.
    pub fn tensor_parallel(
        topology: MlxParallelContext,
        group: &'a Group,
        stream: &'a Stream,
    ) -> Result<Self, Error> {
        if topology.tensor_parallel_size <= 1 {
            return Err(Error::Parallel(
                "tensor-parallel execution context requires TP size greater than one".into(),
            ));
        }
        if group.size() != topology.tensor_parallel_size
            || group.rank() != topology.tensor_parallel_rank
        {
            return Err(Error::Parallel(format!(
                "TP subgroup expects rank {}/{} but received {}/{}",
                topology.tensor_parallel_rank,
                topology.tensor_parallel_size,
                group.rank(),
                group.size()
            )));
        }
        topology.validate_execution_stream(stream)?;
        Ok(Self {
            topology: Some(topology),
            group: Some(group),
            stream,
        })
    }

    /// Returns whether collectives are active.
    pub const fn is_tensor_parallel(&self) -> bool {
        self.group.is_some()
    }

    /// Returns the tensor-parallel rank.
    pub fn rank(&self) -> usize {
        self.topology
            .map_or(0, |topology| topology.tensor_parallel_rank)
    }

    /// Returns the tensor-parallel process count.
    pub fn size(&self) -> usize {
        self.topology
            .map_or(1, |topology| topology.tensor_parallel_size)
    }

    /// Returns the execution stream.
    pub const fn stream(&self) -> &'a Stream {
        self.stream
    }

    /// Returns the TP subgroup when collectives are active.
    pub const fn group(&self) -> Option<&'a Group> {
        self.group
    }

    /// Sums a replicated result across the TP subgroup or returns it unchanged.
    pub fn all_sum(&self, value: &Array) -> Result<Array, Error> {
        match self.group {
            Some(group) => Ok(safemlx::distributed::all_sum(value, group, self.stream)?),
            None => Ok(value.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::DeviceAssignment;
    use safemlx::DeviceType;

    fn topology(rank: usize, parts: usize) -> MlxParallelContext {
        MlxParallelContext::for_rank(rank, parts, 1, 1, DeviceAssignment::new(DeviceType::Cpu, 0))
            .unwrap()
    }

    #[test]
    fn plans_atomic_quantized_row_projection() {
        let mut planner = ParallelPlanBuilder::new(topology(1, 2));
        planner
            .register(
                ParameterGroupSpec::new(
                    "decoder.mlp.down",
                    ParameterRole::RowProjection,
                    [
                        ParameterMemberSpec::new(
                            "down.weight",
                            [8, 16],
                            MemberSharding::Equal { axis: 1 },
                        ),
                        ParameterMemberSpec::new(
                            "down.scales",
                            [8, 4],
                            MemberSharding::Equal { axis: 1 },
                        ),
                        ParameterMemberSpec::new("down.bias", [8], MemberSharding::Replicated),
                    ],
                )
                .unwrap(),
            )
            .unwrap();
        let (placement, local) = planner.finish().unwrap();
        assert_eq!(local.tensor("down.weight").unwrap().local_shape(), &[8, 8]);
        assert_eq!(local.tensor("down.scales").unwrap().local_shape(), &[8, 2]);
        assert_eq!(local.tensor("down.bias").unwrap().local_shape(), &[8]);
        assert!(matches!(
            placement.placement("down.bias"),
            Some(TensorPlacement::Replicated)
        ));
    }

    #[test]
    fn segmented_qkv_preserves_segment_order() {
        let mut planner = ParallelPlanBuilder::new(topology(1, 2));
        planner
            .register(
                ParameterGroupSpec::new(
                    "attention.qkv",
                    ParameterRole::Segmented,
                    [ParameterMemberSpec::new(
                        "qkv.weight",
                        [16, 8],
                        MemberSharding::Segmented {
                            axis: 0,
                            segments: vec![0..8, 8..12, 12..16],
                        },
                    )],
                )
                .unwrap(),
            )
            .unwrap();
        let (_, local) = planner.finish().unwrap();
        let tensor = local.tensor("qkv.weight").unwrap();
        assert_eq!(tensor.local_shape(), &[8, 8]);
        assert_eq!(
            tensor.placement(),
            &TensorPlacement::Indices {
                axis: 0,
                indices: vec![4, 5, 6, 7, 10, 11, 14, 15],
            }
        );
    }

    #[test]
    fn segmented_weight_and_quantization_companions_share_exact_indices() {
        let segments = vec![0..4, 4..8, 8..12];
        let mut planner = ParallelPlanBuilder::new(topology(1, 2));
        planner
            .register(
                ParameterGroupSpec::partitioned(
                    "attention.qkv",
                    ParameterRole::Segmented,
                    4,
                    [
                        ParameterMemberSpec::new(
                            "qkv.weight",
                            [12, 8],
                            MemberSharding::PartitionedSegments {
                                axis: 0,
                                segments: segments.clone(),
                            },
                        ),
                        ParameterMemberSpec::new(
                            "qkv.scales",
                            [12, 2],
                            MemberSharding::PartitionedSegments {
                                axis: 0,
                                segments: segments.clone(),
                            },
                        ),
                        ParameterMemberSpec::new(
                            "qkv.biases",
                            [12, 2],
                            MemberSharding::PartitionedSegments { axis: 0, segments },
                        ),
                    ],
                )
                .unwrap(),
            )
            .unwrap();
        let (_, local) = planner.finish().unwrap();
        let expected = TensorPlacement::Indices {
            axis: 0,
            indices: vec![2, 3, 6, 7, 10, 11],
        };
        for target in ["qkv.weight", "qkv.scales", "qkv.biases"] {
            assert_eq!(local.tensor(target).unwrap().placement(), &expected);
        }
    }

    #[test]
    fn permissive_policy_replicates_the_complete_group() {
        let mut planner =
            ParallelPlanBuilder::with_policy(topology(0, 3), ShardingPolicy::ReplicateUnsupported);
        planner
            .register(
                ParameterGroupSpec::new(
                    "indivisible",
                    ParameterRole::ColumnProjection,
                    [
                        ParameterMemberSpec::new(
                            "weight",
                            [10, 4],
                            MemberSharding::Equal { axis: 0 },
                        ),
                        ParameterMemberSpec::new("bias", [10], MemberSharding::Equal { axis: 0 }),
                    ],
                )
                .unwrap(),
            )
            .unwrap();
        let (_, local) = planner.finish().unwrap();
        for (_, tensor) in local.tensors() {
            assert!(matches!(tensor.placement(), TensorPlacement::Replicated));
            assert!(tensor.fell_back_to_replication());
        }
    }

    #[test]
    fn strict_policy_rejects_indivisible_groups() {
        let mut planner = ParallelPlanBuilder::new(topology(0, 3));
        let error = planner
            .register(
                ParameterGroupSpec::new(
                    "indivisible",
                    ParameterRole::ColumnProjection,
                    [ParameterMemberSpec::new(
                        "weight",
                        [10, 4],
                        MemberSharding::Equal { axis: 0 },
                    )],
                )
                .unwrap(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("not divisible"));
    }

    #[test]
    fn aligned_members_share_an_uneven_logical_range() {
        let expected = [(0..8, 0..4), (8..16, 4..8), (16..20, 8..10)];
        for (rank, (weight_range, scale_range)) in expected.into_iter().enumerate() {
            let mut planner = ParallelPlanBuilder::new(topology(rank, 3));
            planner
                .register(
                    ParameterGroupSpec::partitioned(
                        "aligned.quantized",
                        ParameterRole::RowProjection,
                        5,
                        [
                            ParameterMemberSpec::new(
                                "weight",
                                [8, 20],
                                MemberSharding::Partitioned { axis: 1 },
                            ),
                            ParameterMemberSpec::new(
                                "scales",
                                [8, 10],
                                MemberSharding::Partitioned { axis: 1 },
                            ),
                        ],
                    )
                    .unwrap(),
                )
                .unwrap();
            let (_, local) = planner.finish().unwrap();
            assert_eq!(local.tensor("weight").unwrap().logical_units(), Some(5));
            assert_eq!(
                local.tensor("weight").unwrap().logical_range(),
                Some(&(rank.min(2) * 2..(rank * 2 + 2).min(5)))
            );
            assert_eq!(
                local.tensor("weight").unwrap().placement(),
                &TensorPlacement::Range {
                    axis: 1,
                    start: weight_range.start,
                    end: weight_range.end,
                }
            );
            assert_eq!(
                local.tensor("scales").unwrap().placement(),
                &TensorPlacement::Range {
                    axis: 1,
                    start: scale_range.start,
                    end: scale_range.end,
                }
            );
        }
    }

    #[test]
    fn partitioned_segments_and_packed_axis_share_one_logical_range() {
        let expected = [
            (vec![0, 1, 2, 3, 10, 11, 12, 13], 0..6),
            (vec![4, 5, 6, 7, 14, 15, 16, 17], 6..12),
            (vec![8, 9, 18, 19], 12..15),
        ];
        for (rank, (gate_up_indices, down_range)) in expected.into_iter().enumerate() {
            let mut planner = ParallelPlanBuilder::new(topology(rank, 3));
            planner
                .register(
                    ParameterGroupSpec::partitioned(
                        "experts.intermediate",
                        ParameterRole::ExpertIntermediate,
                        5,
                        [
                            ParameterMemberSpec::new(
                                "gate_up",
                                [4, 20, 8],
                                MemberSharding::PartitionedSegments {
                                    axis: 1,
                                    segments: vec![0..10, 10..20],
                                },
                            ),
                            ParameterMemberSpec::new(
                                "down.packed",
                                [4, 8, 15],
                                MemberSharding::Partitioned { axis: 2 },
                            ),
                        ],
                    )
                    .unwrap(),
                )
                .unwrap();
            let (_, local) = planner.finish().unwrap();
            assert_eq!(
                local.tensor("gate_up").unwrap().placement(),
                &TensorPlacement::Indices {
                    axis: 1,
                    indices: gate_up_indices,
                }
            );
            assert_eq!(
                local.tensor("down.packed").unwrap().placement(),
                &TensorPlacement::Range {
                    axis: 2,
                    start: down_range.start,
                    end: down_range.end,
                }
            );
        }
    }

    #[test]
    fn partitioned_members_require_one_group_level_domain() {
        let error = ParameterGroupSpec::new(
            "missing-domain",
            ParameterRole::AttentionHeads,
            [ParameterMemberSpec::new(
                "query",
                [12, 8],
                MemberSharding::Partitioned { axis: 0 },
            )],
        )
        .unwrap_err();
        assert!(error.to_string().contains("group-level logical partition"));
    }

    #[test]
    fn aligned_groups_reject_more_ranks_than_units() {
        let mut planner = ParallelPlanBuilder::new(topology(0, 3));
        let error = planner
            .register(
                ParameterGroupSpec::partitioned(
                    "too-few-head-groups",
                    ParameterRole::AttentionHeads,
                    2,
                    [ParameterMemberSpec::new(
                        "key",
                        [8, 8],
                        MemberSharding::Partitioned { axis: 0 },
                    )],
                )
                .unwrap(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("non-empty"));
    }

    #[test]
    fn routed_expert_range_preserves_uneven_balanced_slice() {
        let expected = [0..3, 3..5];
        for (rank, expected) in expected.into_iter().enumerate() {
            let mut planner = ParallelPlanBuilder::new(topology(rank, 2));
            planner
                .register(
                    ParameterGroupSpec::partitioned(
                        "experts.intermediate",
                        ParameterRole::ExpertIntermediate,
                        5,
                        [
                            ParameterMemberSpec::new(
                                "experts.gate_up",
                                [4, 10, 8],
                                MemberSharding::PartitionedSegments {
                                    axis: 1,
                                    segments: vec![0..5, 5..10],
                                },
                            ),
                            ParameterMemberSpec::new(
                                "experts.down",
                                [4, 8, 5],
                                MemberSharding::Partitioned { axis: 2 },
                            ),
                        ],
                    )
                    .unwrap(),
                )
                .unwrap();
            let (_, local) = planner.finish().unwrap();

            assert_eq!(
                routed_expert_intermediate_range(&local, 4, expected.len()).unwrap(),
                expected
            );
        }
    }
}

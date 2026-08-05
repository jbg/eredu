//! Typed, architecture-neutral tensor-parallel planning.
//!
//! Architecture adapters describe logical parameter roles and exact checkpoint
//! members. This module converts those descriptions into rank-local placement
//! and shape information without inspecting checkpoint-name substrings.

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
};

use safemlx::{distributed::Group, module::ModuleParameters, Array, Stream};

use crate::{
    error::Error,
    runtime::distributed::topology::{
        balanced_contiguous_range, ParallelTopology, PlacementPlan, TensorPlacement,
    },
};

/// Validates the topology accepted by a pure tensor-parallel execution group.
pub(crate) fn validate_pure_tensor_topology(topology: ParallelTopology) -> Result<(), Error> {
    if topology.tensor_parallel_size <= 1 {
        return Err(Error::Parallel(
            "tensor-parallel loading requires tensor_parallel_size > 1".into(),
        ));
    }
    if topology.pipeline_parallel_size != 1 || topology.expert_parallel_size != 1 {
        return Err(Error::Parallel(format!(
            "pure tensor-parallel execution requires PP=1 and EP=1, got TP={} PP={} EP={}; hybrid TP+PP and TP+EP are unsupported",
            topology.tensor_parallel_size,
            topology.pipeline_parallel_size,
            topology.expert_parallel_size
        )));
    }
    if topology.world_size != topology.tensor_parallel_size {
        return Err(Error::Parallel(
            "pure tensor-parallel world size must equal tensor-parallel size".into(),
        ));
    }
    Ok(())
}

/// Divides a positive model dimension exactly across tensor-parallel ranks.
pub(crate) fn exact_parallel_division(name: &str, value: i32, parts: usize) -> Result<i32, Error> {
    let parts_i32 = i32::try_from(parts)
        .map_err(|_| Error::Parallel("tensor-parallel size does not fit in i32".into()))?;
    if value <= 0 || value % parts_i32 != 0 {
        return Err(Error::Parallel(format!(
            "{name} {value} is not divisible by tensor-parallel size {parts}"
        )));
    }
    Ok(value / parts_i32)
}

/// Validates a local shard dimension against a storage block or group size.
pub(crate) fn require_parallel_alignment(
    tensor: &str,
    dimension: i32,
    alignment: i32,
    topology: ParallelTopology,
) -> Result<(), Error> {
    if alignment <= 0 || dimension % alignment != 0 {
        return Err(Error::Parallel(format!(
            "tensor {tensor} local dimension {dimension} is not aligned to block/group size {alignment} for TP size {}",
            topology.tensor_parallel_size
        )));
    }
    Ok(())
}

/// Returns the balanced vocabulary width assigned to every rank.
pub(crate) fn balanced_parallel_widths(
    vocabulary: usize,
    parts: usize,
) -> Result<Vec<usize>, Error> {
    (0..parts)
        .map(|rank| {
            balanced_contiguous_range(vocabulary, parts, rank, false).map(|range| range.len())
        })
        .collect()
}

/// Semantic role of a logical parameter group.
///
/// Roles are diagnostic and policy inputs. Exact tensor axes remain explicit
/// on each [`ParameterMemberSpec`], so packed weights and their companions do
/// not rely on naming conventions.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ParameterRole {
    /// Small or otherwise non-partitioned state.
    Replicated,
    /// Projection whose output features are rank-local.
    ColumnProjection,
    /// Projection whose input features are rank-local and whose output is reduced.
    RowProjection,
    /// Token embedding or output projection partitioned by vocabulary.
    Vocabulary,
    /// Query, key, or value heads.
    AttentionHeads,
    /// Routed or shared expert intermediate channels.
    ExpertIntermediate,
    /// State-space, convolution, or recurrent channels.
    Channels,
    /// A fused tensor containing independently partitioned segments.
    Segmented,
}

/// Rank-local selection rule for one physical checkpoint tensor.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum MemberSharding {
    /// Materialize the complete member on every tensor-parallel rank.
    Replicated,
    /// Split an axis into equal contiguous shards.
    Equal {
        /// Source tensor axis to partition.
        axis: usize,
    },
    /// Split an axis into balanced, potentially uneven contiguous ranges.
    Balanced {
        /// Source tensor axis to partition.
        axis: usize,
    },
    /// Partition each supplied source range independently and concatenate the
    /// rank-local indices in segment order.
    Segmented {
        /// Source tensor axis containing the fused segments.
        axis: usize,
        /// Ordered, non-overlapping source ranges partitioned independently.
        segments: Vec<Range<usize>>,
    },
}

/// Sharding contract for an architecture-native projection module.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum ProjectionSharding {
    /// Keep every projection parameter complete on every rank.
    Replicated,
    /// Partition projection output features.
    Column,
    /// Partition projection input features and replicate its output bias.
    Row,
}

/// Describes every parameter in a module as one typed logical group.
pub(crate) fn module_parameter_group(
    logical_name: &str,
    role: ParameterRole,
    module: &impl ModuleParameters,
    prefix: &str,
    sharding: impl Fn(&str, &[i32]) -> Result<MemberSharding, Error>,
) -> Result<ParameterGroupSpec, Error> {
    let parameters = module.parameters().flatten();
    ParameterGroupSpec::new(
        logical_name,
        role,
        parameters
            .iter()
            .map(|(name, parameter)| {
                let shape = parameter
                    .shape()
                    .iter()
                    .map(|dimension| {
                        usize::try_from(*dimension).map_err(|_| {
                            Error::Parallel(format!(
                                "parameter {prefix}.{name} has negative dimension {dimension}"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(ParameterMemberSpec::new(
                    format!("{prefix}.{name}"),
                    shape,
                    sharding(name, parameter.shape())?,
                ))
            })
            .collect::<Result<Vec<_>, Error>>()?,
    )
}

/// Registers every parameter in a module as replicated.
pub(crate) fn register_replicated_module(
    planner: &mut ParallelPlanBuilder,
    module: &impl ModuleParameters,
    prefix: &str,
) -> Result<(), Error> {
    if module.parameters().flatten().is_empty() {
        return Ok(());
    }
    planner.register(module_parameter_group(
        prefix,
        ParameterRole::Replicated,
        module,
        prefix,
        |_, _| Ok(MemberSharding::Replicated),
    )?)
}

/// Registers an architecture-native projection and all quantization companions.
pub(crate) fn register_projection_module(
    planner: &mut ParallelPlanBuilder,
    projection: &impl ModuleParameters,
    prefix: &str,
    placement: ProjectionSharding,
) -> Result<(), Error> {
    let role = match placement {
        ProjectionSharding::Replicated => ParameterRole::Replicated,
        ProjectionSharding::Column => ParameterRole::ColumnProjection,
        ProjectionSharding::Row => ParameterRole::RowProjection,
    };
    planner.register(module_parameter_group(
        prefix,
        role,
        projection,
        prefix,
        |name, shape| match placement {
            ProjectionSharding::Replicated => Ok(MemberSharding::Replicated),
            ProjectionSharding::Column => {
                if shape.is_empty() {
                    Err(Error::Parallel(format!(
                        "column projection member {prefix}.{name} has scalar shape"
                    )))
                } else {
                    Ok(MemberSharding::Equal { axis: 0 })
                }
            }
            ProjectionSharding::Row => match name {
                "bias" | "inner.bias" => Ok(MemberSharding::Replicated),
                "weight"
                | "inner.weight"
                | "weight_scale_inv"
                | "inner.weight_scale_inv"
                | "scales"
                | "biases"
                    if shape.len() >= 2 =>
                {
                    Ok(MemberSharding::Equal { axis: 1 })
                }
                _ => Err(Error::Parallel(format!(
                    "unsupported row-projection member {prefix}.{name} with shape {shape:?}"
                ))),
            },
        },
    )?)
}

/// Registers a module partitioned across its leading vocabulary dimension.
pub(crate) fn register_vocabulary_module(
    planner: &mut ParallelPlanBuilder,
    module: &impl ModuleParameters,
    prefix: &str,
) -> Result<(), Error> {
    planner.register(module_parameter_group(
        prefix,
        ParameterRole::Vocabulary,
        module,
        prefix,
        |name, shape| {
            if shape.is_empty() {
                return Err(Error::Parallel(format!(
                    "vocabulary member {prefix}.{name} has scalar shape"
                )));
            }
            Ok(MemberSharding::Balanced { axis: 0 })
        },
    )?)
}

/// Builds one typed member directly from an array parameter.
pub(crate) fn array_parameter_member(
    target: impl Into<String>,
    array: &Array,
    sharding: MemberSharding,
) -> Result<ParameterMemberSpec, Error> {
    let target = target.into();
    let shape = array
        .shape()
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension).map_err(|_| {
                Error::Parallel(format!(
                    "parameter {target} has negative dimension {dimension}"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ParameterMemberSpec::new(target, shape, sharding))
}

/// One physical tensor belonging to a logical parameter group.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ParameterMemberSpec {
    target: String,
    global_shape: Vec<usize>,
    sharding: MemberSharding,
}

impl ParameterMemberSpec {
    /// Creates a member with an exact pre-selection checkpoint shape.
    pub fn new(
        target: impl Into<String>,
        global_shape: impl Into<Vec<usize>>,
        sharding: MemberSharding,
    ) -> Self {
        Self {
            target: target.into(),
            global_shape: global_shape.into(),
            sharding,
        }
    }

    /// Returns the rewritten checkpoint target.
    pub fn target(&self) -> &str {
        &self.target
    }

    /// Returns the complete source shape.
    pub fn global_shape(&self) -> &[usize] {
        &self.global_shape
    }

    /// Returns the requested rank-local selection.
    pub const fn sharding(&self) -> &MemberSharding {
        &self.sharding
    }
}

/// Atomic logical parameter and all of its physical checkpoint companions.
///
/// If permissive planning must fall back, every member in the group is
/// replicated together. This prevents a packed weight, scales, quantization
/// biases, and ordinary post-projection bias from acquiring incompatible
/// layouts.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ParameterGroupSpec {
    logical_name: String,
    role: ParameterRole,
    members: Vec<ParameterMemberSpec>,
}

impl ParameterGroupSpec {
    /// Creates a non-empty logical group.
    pub fn new(
        logical_name: impl Into<String>,
        role: ParameterRole,
        members: impl IntoIterator<Item = ParameterMemberSpec>,
    ) -> Result<Self, Error> {
        let logical_name = logical_name.into();
        if logical_name.trim().is_empty() {
            return Err(Error::Parallel(
                "parallel parameter logical name must not be empty".into(),
            ));
        }
        let members = members.into_iter().collect::<Vec<_>>();
        if members.is_empty() {
            return Err(Error::Parallel(format!(
                "parallel parameter group {logical_name:?} must contain at least one tensor"
            )));
        }
        let mut targets = BTreeSet::new();
        for member in &members {
            if member.target.trim().is_empty() {
                return Err(Error::Parallel(format!(
                    "parallel parameter group {logical_name:?} contains an empty tensor target"
                )));
            }
            if !targets.insert(member.target.clone()) {
                return Err(Error::Parallel(format!(
                    "parallel parameter group {logical_name:?} repeats tensor target {:?}",
                    member.target
                )));
            }
        }
        Ok(Self {
            logical_name,
            role,
            members,
        })
    }

    /// Returns the stable logical name.
    pub fn logical_name(&self) -> &str {
        &self.logical_name
    }

    /// Returns the semantic role.
    pub const fn role(&self) -> ParameterRole {
        self.role
    }

    /// Returns physical checkpoint members.
    pub fn members(&self) -> &[ParameterMemberSpec] {
        &self.members
    }
}

/// Behavior when a requested shard is not legal for the current TP size.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Default)]
pub enum ShardingPolicy {
    /// Reject the complete plan with a precise shape/alignment error.
    #[default]
    Require,
    /// Replicate the complete logical parameter group.
    ReplicateUnsupported,
}

/// Rank-local shape and placement for one planned physical tensor.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct LocalTensorLayout {
    logical_name: String,
    role: ParameterRole,
    global_shape: Vec<usize>,
    local_shape: Vec<usize>,
    placement: TensorPlacement,
    fell_back_to_replication: bool,
}

impl LocalTensorLayout {
    /// Returns the logical parameter group name.
    pub fn logical_name(&self) -> &str {
        &self.logical_name
    }

    /// Returns the semantic parameter role.
    pub const fn role(&self) -> ParameterRole {
        self.role
    }

    /// Returns the checkpoint-global shape.
    pub fn global_shape(&self) -> &[usize] {
        &self.global_shape
    }

    /// Returns the shape materialized on this rank.
    pub fn local_shape(&self) -> &[usize] {
        &self.local_shape
    }

    /// Returns the exact checkpoint placement.
    pub const fn placement(&self) -> &TensorPlacement {
        &self.placement
    }

    /// Returns whether permissive planning replicated an unsupported shard.
    pub const fn fell_back_to_replication(&self) -> bool {
        self.fell_back_to_replication
    }
}

/// Complete rank-local model geometry produced alongside checkpoint placement.
#[derive(Debug, Clone, Default)]
pub struct LocalModelLayout {
    tensors: BTreeMap<String, LocalTensorLayout>,
}

impl LocalModelLayout {
    /// Returns one physical tensor layout by rewritten target name.
    pub fn tensor(&self, target: &str) -> Option<&LocalTensorLayout> {
        self.tensors.get(target)
    }

    /// Iterates physical layouts in deterministic target-name order.
    pub fn tensors(&self) -> impl Iterator<Item = (&str, &LocalTensorLayout)> {
        self.tensors
            .iter()
            .map(|(target, layout)| (target.as_str(), layout))
    }

    /// Returns the number of planned physical tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Returns whether no physical tensors were planned.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Builds checkpoint placement and local model geometry from typed roles.
pub struct ParallelPlanBuilder {
    topology: ParallelTopology,
    policy: ShardingPolicy,
    placement: PlacementPlan,
    local: LocalModelLayout,
}

impl ParallelPlanBuilder {
    /// Creates an empty strict planner for one rank.
    pub fn new(topology: ParallelTopology) -> Self {
        Self::with_policy(topology, ShardingPolicy::Require)
    }

    /// Creates an empty planner with an explicit fallback policy.
    pub fn with_policy(topology: ParallelTopology, policy: ShardingPolicy) -> Self {
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
            .members
            .iter()
            .find(|member| self.local.tensors.contains_key(&member.target))
        {
            return Err(Error::Parallel(format!(
                "parallel placement target {:?} was registered more than once",
                member.target
            )));
        }
        let requested = group
            .members
            .iter()
            .map(|member| self.resolve_member(member))
            .collect::<Result<Vec<_>, _>>();
        let (resolved, fell_back) = match requested {
            Ok(resolved) => (resolved, false),
            Err(_error) if self.policy == ShardingPolicy::ReplicateUnsupported => (
                group
                    .members
                    .iter()
                    .map(|member| (TensorPlacement::Replicated, member.global_shape.clone()))
                    .collect(),
                true,
            ),
            Err(error) => {
                return Err(Error::Parallel(format!(
                    "logical parameter {:?}: {error}",
                    group.logical_name
                )))
            }
        };

        for (member, (placement, local_shape)) in group.members.iter().zip(resolved) {
            self.placement.insert_expected(
                member.target.clone(),
                member.global_shape.clone(),
                placement.clone(),
            )?;
            self.local.tensors.insert(
                member.target.clone(),
                LocalTensorLayout {
                    logical_name: group.logical_name.clone(),
                    role: group.role,
                    global_shape: member.global_shape.clone(),
                    local_shape,
                    placement,
                    fell_back_to_replication: fell_back,
                },
            );
        }
        Ok(())
    }

    /// Completes planning and validates every generated placement.
    pub fn finish(self) -> Result<(PlacementPlan, LocalModelLayout), Error> {
        self.placement.validate()?;
        Ok((self.placement, self.local))
    }

    fn resolve_member(
        &self,
        member: &ParameterMemberSpec,
    ) -> Result<(TensorPlacement, Vec<usize>), String> {
        let rank = self.topology.tensor_parallel_rank;
        let parts = self.topology.tensor_parallel_size;
        if parts == 0 || rank >= parts {
            return Err(format!("invalid TP coordinate {rank}/{parts}"));
        }
        match &member.sharding {
            MemberSharding::Replicated => {
                Ok((TensorPlacement::Replicated, member.global_shape.clone()))
            }
            MemberSharding::Equal { axis } => {
                let dimension = checked_axis(member, *axis)?;
                if dimension % parts != 0 {
                    return Err(format!(
                        "tensor {:?} dimension {dimension} on axis {axis} is not divisible by TP size {parts}",
                        member.target
                    ));
                }
                let mut local_shape = member.global_shape.clone();
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
                let mut local_shape = member.global_shape.clone();
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
            MemberSharding::Segmented { axis, segments } => {
                let dimension = checked_axis(member, *axis)?;
                if segments.is_empty() {
                    return Err(format!(
                        "tensor {:?} segmented placement has no segments",
                        member.target
                    ));
                }
                let mut indices = Vec::new();
                let mut previous_end = 0usize;
                for segment in segments {
                    if segment.start >= segment.end || segment.end > dimension {
                        return Err(format!(
                            "tensor {:?} segment {:?} is invalid for axis-{axis} dimension {dimension}",
                            member.target, segment
                        ));
                    }
                    if segment.start < previous_end {
                        return Err(format!(
                            "tensor {:?} segmented ranges overlap or are out of order",
                            member.target
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
                        member.target
                    ));
                }
                let mut local_shape = member.global_shape.clone();
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

fn checked_axis(member: &ParameterMemberSpec, axis: usize) -> Result<usize, String> {
    member.global_shape.get(axis).copied().ok_or_else(|| {
        format!(
            "tensor {:?} axis {axis} is outside shape {:?}",
            member.target, member.global_shape
        )
    })
}

/// Construction-time topology and fallback policy supplied to model builders.
#[derive(Debug, Clone, Copy)]
pub struct ParallelBuildContext {
    topology: ParallelTopology,
    policy: ShardingPolicy,
}

impl ParallelBuildContext {
    /// Creates a construction context for a validated topology.
    pub const fn new(topology: ParallelTopology, policy: ShardingPolicy) -> Self {
        Self { topology, policy }
    }

    /// Returns the complete process topology.
    pub const fn topology(self) -> ParallelTopology {
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

    /// Returns the local width for an equal logical partition.
    pub fn equal_local_dimension(self, name: &str, global: usize) -> Result<usize, Error> {
        let parts = self.topology.tensor_parallel_size;
        if parts == 0 || global == 0 || !global.is_multiple_of(parts) {
            return Err(Error::Parallel(format!(
                "{name} dimension {global} is not divisible by TP size {parts}"
            )));
        }
        Ok(global / parts)
    }
}

/// Borrowed execution resources for replicated or tensor-parallel primitives.
///
/// The group is never retained by model state. In hybrid topologies callers
/// supply the TP subgroup, whose rank is the tensor-parallel coordinate.
pub struct ParallelExecutionContext<'a> {
    topology: Option<ParallelTopology>,
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
        topology: ParallelTopology,
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
    use crate::runtime::distributed::topology::DeviceAssignment;
    use safemlx::DeviceType;

    fn topology(rank: usize, parts: usize) -> ParallelTopology {
        ParallelTopology::from_rank(
            parts,
            rank,
            parts,
            1,
            1,
            DeviceAssignment::new(DeviceType::Cpu, 0),
        )
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
}

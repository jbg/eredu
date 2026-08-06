//! Typed, architecture-neutral tensor-parallel planning.
//!
//! Architecture adapters describe logical parameter roles and exact checkpoint
//! members. This module converts those descriptions into rank-local placement
//! and shape information without inspecting checkpoint-name substrings.

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
};

use safemlx::{
    distributed::{self, Group},
    module::ModuleParameters,
    ops::{indexing::TryIndexOp, ones, zeros},
    transforms::eval,
    Array, Stream,
};

use crate::{
    error::Error,
    runtime::distributed::topology::{
        balanced_contiguous_range, ParallelTopology, PlacementPlan, TensorPlacement,
    },
    runtime::generation::sampler::Sampler,
};

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
pub fn sample_and_synchronize<S: Sampler>(
    logits: Option<&Array>,
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
        if logits.dim(0) != batch_size {
            return Err(Error::Parallel(format!(
                "sampling logits batch {} does not match declared batch {batch_size}",
                logits.dim(0)
            )));
        }
        let logits = if logits.ndim() == 3 {
            logits.try_index_device((.., -1, ..), stream)?
        } else {
            logits.clone()
        };
        sampler
            .sample(&logits, temperature, prng_state, stream)?
            .reshape(&[batch_size, 1], stream)?
    } else {
        zeros::<u32>(&[batch_size, 1], stream)?
    };
    let token = distributed::all_sum(&local_token, group, stream)?;
    let local_finished = if group.rank() == sampling_rank && finished {
        ones::<i32>(&[], stream)?
    } else {
        zeros::<i32>(&[], stream)?
    };
    let finished = distributed::all_sum(&local_finished, group, stream)?;
    eval([&token, &finished])?;
    stream.synchronize()?;
    Ok(SynchronizedToken {
        token,
        finished: finished.try_item::<i32>(stream)? != 0,
    })
}

/// Returns the finest legal logical-unit count for an aligned row partition.
///
/// `semantic_units` is the number of indivisible semantic groups, while
/// `elements_per_unit` is their width on the row-sharded axis. The returned
/// count combines adjacent semantic groups until every boundary is aligned to
/// `required_alignment` (one for dense tensors, or a quantization block size).
pub(crate) fn aligned_partition_units(
    name: &str,
    semantic_units: usize,
    elements_per_unit: usize,
    required_alignment: usize,
) -> Result<usize, Error> {
    if semantic_units == 0 || elements_per_unit == 0 || required_alignment == 0 {
        return Err(Error::Parallel(format!(
            "{name} aligned partition dimensions must be positive, got units={semantic_units}, width={elements_per_unit}, alignment={required_alignment}"
        )));
    }
    let units_per_partition =
        required_alignment / greatest_common_divisor(elements_per_unit, required_alignment);
    if !semantic_units.is_multiple_of(units_per_partition) {
        return Err(Error::Parallel(format!(
            "{name} has {semantic_units} semantic units of width {elements_per_unit}, which cannot form complete alignment-{required_alignment} partitions"
        )));
    }
    Ok(semantic_units / units_per_partition)
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
    /// Dense feed-forward intermediate channels shared by input and output projections.
    FeedForwardIntermediate,
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
    /// Map the group's logical partition onto one physical tensor axis.
    Partitioned {
        /// Source tensor axis to partition.
        axis: usize,
    },
    /// Map the same group-level logical range into each supplied source
    /// segment and concatenate the selected indices in segment order.
    PartitionedSegments {
        /// Source tensor axis containing the fused segments.
        axis: usize,
        /// Ordered, non-overlapping physical source ranges. Every segment
        /// represents the group's complete logical domain.
        segments: Vec<Range<usize>>,
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

/// Registers projections that must consume the same logical partition.
///
/// `preferred_units` expresses the finest semantically legal partition (KV
/// groups for GQA, intermediate channels for a gated MLP). The planner reduces
/// it to the greatest unit count represented integrally by every physical
/// member, including packed weights and quantization companions. This makes
/// uneven ranges safe without architecture-specific packed-shape arithmetic.
pub(crate) fn register_partitioned_projection_group<M: ModuleParameters>(
    planner: &mut ParallelPlanBuilder,
    logical_name: &str,
    role: ParameterRole,
    projections: &[(&M, &str, ProjectionSharding)],
    preferred_units: usize,
) -> Result<(), Error> {
    let (units, members) = partitioned_projection_members(projections, preferred_units)?;
    planner.register(ParameterGroupSpec::partitioned(
        logical_name,
        role,
        units,
        members,
    )?)
}

/// Builds physical members for projections sharing one logical partition.
/// Architecture-owned compound groups can add non-linear members such as
/// depthwise kernels while retaining the same packed-projection rules.
pub(crate) fn partitioned_projection_members<M: ModuleParameters>(
    projections: &[(&M, &str, ProjectionSharding)],
    preferred_units: usize,
) -> Result<(usize, Vec<ParameterMemberSpec>), Error> {
    if preferred_units == 0 {
        return Err(Error::Parallel(
            "partitioned projection group has zero preferred units".into(),
        ));
    }

    let mut raw_members = Vec::new();
    let mut units = preferred_units;
    for (module, prefix, placement) in projections {
        for (name, parameter) in module.parameters().flatten() {
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
            let axis = projection_partition_axis(prefix, &name, &shape, *placement)?;
            if let Some(axis) = axis {
                units = greatest_common_divisor(units, shape[axis]);
            }
            raw_members.push((format!("{prefix}.{name}"), shape, axis));
        }
    }
    if raw_members.is_empty() {
        return Err(Error::Parallel(
            "partitioned projection group contains no parameters".into(),
        ));
    }

    Ok((
        units,
        raw_members
            .into_iter()
            .map(|(target, shape, axis)| {
                let sharding = axis.map_or(MemberSharding::Replicated, |axis| {
                    MemberSharding::Partitioned { axis }
                });
                ParameterMemberSpec::new(target, shape, sharding)
            })
            .collect(),
    ))
}

fn projection_partition_axis(
    prefix: &str,
    name: &str,
    shape: &[usize],
    placement: ProjectionSharding,
) -> Result<Option<usize>, Error> {
    match placement {
        ProjectionSharding::Replicated => Ok(None),
        ProjectionSharding::Column if shape.is_empty() => Err(Error::Parallel(format!(
            "column projection member {prefix}.{name} has scalar shape"
        ))),
        ProjectionSharding::Column => Ok(Some(0)),
        ProjectionSharding::Row => match name {
            "bias" | "inner.bias" => Ok(None),
            "weight"
            | "inner.weight"
            | "weight_scale_inv"
            | "inner.weight_scale_inv"
            | "scales"
            | "biases"
                if shape.len() >= 2 =>
            {
                Ok(Some(1))
            }
            _ => Err(Error::Parallel(format!(
                "unsupported row-projection member {prefix}.{name} with shape {shape:?}"
            ))),
        },
    }
}

const fn greatest_common_divisor(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
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
    partition_units: Option<usize>,
    members: Vec<ParameterMemberSpec>,
}

impl ParameterGroupSpec {
    /// Creates a non-empty logical group.
    pub fn new(
        logical_name: impl Into<String>,
        role: ParameterRole,
        members: impl IntoIterator<Item = ParameterMemberSpec>,
    ) -> Result<Self, Error> {
        Self::build(logical_name.into(), role, None, members)
    }

    /// Creates a group whose partitioned members share one logical domain.
    pub fn partitioned(
        logical_name: impl Into<String>,
        role: ParameterRole,
        units: usize,
        members: impl IntoIterator<Item = ParameterMemberSpec>,
    ) -> Result<Self, Error> {
        if units == 0 {
            return Err(Error::Parallel(
                "parallel logical partition must contain at least one unit".into(),
            ));
        }
        Self::build(logical_name.into(), role, Some(units), members)
    }

    fn build(
        logical_name: String,
        role: ParameterRole,
        partition_units: Option<usize>,
        members: impl IntoIterator<Item = ParameterMemberSpec>,
    ) -> Result<Self, Error> {
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
        let mut has_partitioned_member = false;
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
            has_partitioned_member |= matches!(
                member.sharding,
                MemberSharding::Partitioned { .. } | MemberSharding::PartitionedSegments { .. }
            );
        }
        if has_partitioned_member != partition_units.is_some() {
            return Err(Error::Parallel(format!(
                "parallel parameter group {logical_name:?} must declare exactly one group-level logical partition for its partitioned members"
            )));
        }
        Ok(Self {
            logical_name,
            role,
            partition_units,
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

    /// Returns the shared logical-unit count, when the group is partitioned.
    pub const fn partition_units(&self) -> Option<usize> {
        self.partition_units
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
    logical_units: Option<usize>,
    logical_range: Option<Range<usize>>,
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

    /// Returns the rank-local range in the parameter group's semantic domain.
    ///
    /// Physical packed weights and their quantization companions can use
    /// different element ranges while still representing the same heads or
    /// channels. This range is therefore the authoritative source for local
    /// execution geometry.
    pub fn logical_range(&self) -> Option<&Range<usize>> {
        self.logical_range.as_ref()
    }

    /// Returns the size of the complete semantic partition domain.
    pub const fn logical_units(&self) -> Option<usize> {
        self.logical_units
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
            .map(|member| self.resolve_member(member, group.partition_units))
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
        let logical_range = match (group.partition_units, fell_back) {
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
                    logical_units: group.partition_units,
                    logical_range: logical_range.clone(),
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
        partition_units: Option<usize>,
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
            MemberSharding::Partitioned { axis } => {
                let units = partition_units.ok_or_else(|| {
                    format!(
                        "tensor {:?} has no group-level logical partition",
                        member.target
                    )
                })?;
                let dimension = checked_axis(member, *axis)?;
                if dimension % units != 0 {
                    return Err(format!(
                        "tensor {:?} dimension {dimension} on axis {axis} does not contain {units} integral logical units",
                        member.target
                    ));
                }
                let logical = balanced_contiguous_range(units, parts, rank, false)
                    .map_err(|error| error.to_string())?;
                let elements_per_unit = dimension / units;
                let range = (logical.start * elements_per_unit)..(logical.end * elements_per_unit);
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
            MemberSharding::PartitionedSegments { axis, segments } => {
                let units = partition_units.ok_or_else(|| {
                    format!(
                        "tensor {:?} has no group-level logical partition",
                        member.target
                    )
                })?;
                let dimension = checked_axis(member, *axis)?;
                if segments.is_empty() {
                    return Err(format!(
                        "tensor {:?} partitioned placement has no segments",
                        member.target
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
                            member.target, segment
                        ));
                    }
                    if segment.start < previous_end {
                        return Err(format!(
                            "tensor {:?} partitioned ranges overlap or are out of order",
                            member.target
                        ));
                    }
                    previous_end = segment.end;
                    if !segment.len().is_multiple_of(units) {
                        return Err(format!(
                            "tensor {:?} segment {:?} does not contain {units} integral logical units",
                            member.target, segment
                        ));
                    }
                    let elements_per_unit = segment.len() / units;
                    indices.extend(
                        (segment.start + logical.start * elements_per_unit)
                            ..(segment.start + logical.end * elements_per_unit),
                    );
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
    fn quantized_alignment_combines_complete_semantic_units() {
        assert_eq!(aligned_partition_units("GQA", 6, 16, 32).unwrap(), 3);
        assert_eq!(aligned_partition_units("SwiGLU", 96, 1, 32).unwrap(), 3);
        let error = aligned_partition_units("GQA", 3, 16, 32).unwrap_err();
        assert!(error.to_string().contains("cannot form complete"));
    }
}

//! Backend-neutral semantic parameter sharding and rank-local layouts.
//!
//! Architectures describe physical checkpoint members in logical groups. An
//! execution backend may then realize the resulting placement without knowing
//! projection names, attention geometry, or other model-family semantics.

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
};

use eredu_nn::{ParameterMetadata, ParameterVisitor, Parameterized, Tensor};

/// Architecture-neutral information for one rank-local parallel model.
#[derive(Debug, Clone)]
pub struct ParallelModelInfo<T> {
    topology: T,
    model_type: String,
    owned_tensors: Vec<String>,
    local_parameter_bytes: u64,
    global_parameter_bytes: u64,
    pinned_device_parameter_bytes: u64,
    maximum_device_parameter_bytes: u64,
}

impl<T> ParallelModelInfo<T> {
    /// Creates a complete rank-local parallel model summary.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        topology: T,
        model_type: impl Into<String>,
        owned_tensors: Vec<String>,
        local_parameter_bytes: u64,
        global_parameter_bytes: u64,
        pinned_device_parameter_bytes: u64,
        maximum_device_parameter_bytes: u64,
    ) -> Self {
        Self {
            topology,
            model_type: model_type.into(),
            owned_tensors,
            local_parameter_bytes,
            global_parameter_bytes,
            pinned_device_parameter_bytes,
            maximum_device_parameter_bytes,
        }
    }

    /// Returns the backend's concrete topology value unchanged.
    pub fn topology(&self) -> T
    where
        T: Clone,
    {
        self.topology.clone()
    }

    /// Returns the architecture's normalized model type.
    pub fn model_type(&self) -> &str {
        &self.model_type
    }

    /// Returns exact checkpoint targets owned or replicated by this rank.
    pub fn owned_tensors(&self) -> &[String] {
        &self.owned_tensors
    }

    /// Returns planned rank-local parameter bytes across static and execution units.
    pub const fn local_parameter_bytes(&self) -> u64 {
        self.local_parameter_bytes
    }

    /// Returns the unsharded model parameter bytes represented by this checkpoint.
    pub const fn global_parameter_bytes(&self) -> u64 {
        self.global_parameter_bytes
    }

    /// Returns rank-local parameter bytes permanently pinned on the execution device.
    pub const fn pinned_device_parameter_bytes(&self) -> u64 {
        self.pinned_device_parameter_bytes
    }

    /// Returns the maximum planned rank-local parameter footprint on device.
    pub const fn maximum_device_parameter_bytes(&self) -> u64 {
        self.maximum_device_parameter_bytes
    }
}

/// Semantic role of a logical parameter group.
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

/// Logical sharding behavior for a parameterized affine projection.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ProjectionSharding {
    /// Keep every projection parameter complete on every rank.
    Replicated,
    /// Partition projection output features.
    Column,
    /// Partition projection input features and replicate output bias.
    Row,
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
    /// Map the same group-level logical range into each supplied source segment.
    PartitionedSegments {
        /// Source tensor axis containing the fused segments.
        axis: usize,
        /// Ordered, non-overlapping physical source ranges.
        segments: Vec<Range<usize>>,
    },
    /// Partition each supplied source range independently.
    Segmented {
        /// Source tensor axis containing the fused segments.
        axis: usize,
        /// Ordered, non-overlapping source ranges.
        segments: Vec<Range<usize>>,
    },
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
    ) -> Result<Self, ParallelPlanError> {
        Self::build(logical_name.into(), role, None, members)
    }

    /// Creates a group whose partitioned members share one logical domain.
    pub fn partitioned(
        logical_name: impl Into<String>,
        role: ParameterRole,
        units: usize,
        members: impl IntoIterator<Item = ParameterMemberSpec>,
    ) -> Result<Self, ParallelPlanError> {
        if units == 0 {
            return Err(ParallelPlanError::InvalidGroup(
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
    ) -> Result<Self, ParallelPlanError> {
        if logical_name.trim().is_empty() {
            return Err(ParallelPlanError::InvalidGroup(
                "parallel parameter logical name must not be empty".into(),
            ));
        }
        let members = members.into_iter().collect::<Vec<_>>();
        if members.is_empty() {
            return Err(ParallelPlanError::InvalidGroup(format!(
                "parallel parameter group {logical_name:?} must contain at least one tensor"
            )));
        }
        let mut targets = BTreeSet::new();
        let mut has_partitioned_member = false;
        for member in &members {
            if member.target.trim().is_empty() {
                return Err(ParallelPlanError::InvalidGroup(format!(
                    "parallel parameter group {logical_name:?} contains an empty tensor target"
                )));
            }
            if !targets.insert(member.target.clone()) {
                return Err(ParallelPlanError::InvalidGroup(format!(
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
            return Err(ParallelPlanError::InvalidGroup(format!(
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

/// Describes every parameter in a neutral module as one logical group.
pub fn module_parameter_group<T, M>(
    logical_name: impl Into<String>,
    role: ParameterRole,
    module: &M,
    mut sharding: impl FnMut(&ParameterMetadata, &[usize]) -> Result<MemberSharding, ParallelPlanError>,
) -> Result<ParameterGroupSpec, ParallelPlanError>
where
    T: Tensor,
    M: Parameterized<T>,
{
    struct Collector<'a, F> {
        members: Vec<ParameterMemberSpec>,
        sharding: &'a mut F,
        error: Option<ParallelPlanError>,
    }

    impl<'a, 'tensor, T, F> ParameterVisitor<'tensor, T> for Collector<'a, F>
    where
        T: Tensor,
        F: FnMut(&ParameterMetadata, &[usize]) -> Result<MemberSharding, ParallelPlanError>,
    {
        fn visit(&mut self, metadata: ParameterMetadata, value: &'tensor T) {
            if self.error.is_some() {
                return;
            }
            let shape = value
                .shape()
                .iter()
                .map(|dimension| {
                    usize::try_from(*dimension).map_err(|_| {
                        ParallelPlanError::InvalidTensor(format!(
                            "parameter {} has negative dimension {dimension}",
                            metadata.id.as_str()
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>();
            let shape = match shape {
                Ok(shape) => shape,
                Err(error) => {
                    self.error = Some(error);
                    return;
                }
            };
            match (self.sharding)(&metadata, &shape) {
                Ok(sharding) => self.members.push(ParameterMemberSpec::new(
                    metadata.id.as_str(),
                    shape,
                    sharding,
                )),
                Err(error) => self.error = Some(error),
            }
        }
    }

    let mut collector = Collector {
        members: Vec::new(),
        sharding: &mut sharding,
        error: None,
    };
    module.visit_parameters(&mut collector);
    if let Some(error) = collector.error {
        return Err(error);
    }
    ParameterGroupSpec::new(logical_name, role, collector.members)
}

/// Describes every parameter in a neutral module as one shared logical partition.
pub fn partitioned_module_parameter_group<T, M>(
    logical_name: impl Into<String>,
    role: ParameterRole,
    preferred_units: usize,
    module: &M,
    mut sharding: impl FnMut(&ParameterMetadata, &[usize]) -> Result<MemberSharding, ParallelPlanError>,
) -> Result<ParameterGroupSpec, ParallelPlanError>
where
    T: Tensor,
    M: Parameterized<T>,
{
    if preferred_units == 0 {
        return Err(ParallelPlanError::InvalidGroup(
            "partitioned module group has zero preferred units".into(),
        ));
    }
    struct Collector<'a, F> {
        members: Vec<ParameterMemberSpec>,
        sharding: &'a mut F,
        error: Option<ParallelPlanError>,
    }
    impl<'a, 'tensor, T, F> ParameterVisitor<'tensor, T> for Collector<'a, F>
    where
        T: Tensor,
        F: FnMut(&ParameterMetadata, &[usize]) -> Result<MemberSharding, ParallelPlanError>,
    {
        fn visit(&mut self, metadata: ParameterMetadata, value: &'tensor T) {
            if self.error.is_some() {
                return;
            }
            let shape = value
                .shape()
                .iter()
                .map(|dimension| {
                    usize::try_from(*dimension).map_err(|_| {
                        ParallelPlanError::InvalidTensor(format!(
                            "parameter {} has negative dimension {dimension}",
                            metadata.id.as_str()
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>();
            match shape.and_then(|shape| {
                (self.sharding)(&metadata, &shape)
                    .map(|sharding| ParameterMemberSpec::new(metadata.id.as_str(), shape, sharding))
            }) {
                Ok(member) => self.members.push(member),
                Err(error) => self.error = Some(error),
            }
        }
    }
    let mut collector = Collector {
        members: Vec::new(),
        sharding: &mut sharding,
        error: None,
    };
    module.visit_parameters(&mut collector);
    if let Some(error) = collector.error {
        return Err(error);
    }
    ParameterGroupSpec::partitioned(logical_name, role, preferred_units, collector.members)
}

/// Describes one affine projection and all encoding companions.
pub fn projection_parameter_group<T, M>(
    logical_name: impl Into<String>,
    role: ParameterRole,
    module: &M,
    placement: ProjectionSharding,
) -> Result<ParameterGroupSpec, ParallelPlanError>
where
    T: Tensor,
    M: Parameterized<T>,
{
    module_parameter_group(
        logical_name,
        role,
        module,
        |metadata, shape| match placement {
            ProjectionSharding::Replicated => Ok(MemberSharding::Replicated),
            ProjectionSharding::Column if shape.is_empty() => {
                Err(ParallelPlanError::InvalidTensor(format!(
                    "column projection parameter {} is scalar",
                    metadata.id.as_str()
                )))
            }
            ProjectionSharding::Column => Ok(MemberSharding::Equal { axis: 0 }),
            ProjectionSharding::Row if shape.len() >= 2 => Ok(MemberSharding::Equal { axis: 1 }),
            ProjectionSharding::Row => Ok(MemberSharding::Replicated),
        },
    )
}

/// Describes projections that consume one shared logical partition.
pub fn partitioned_projection_group<T, M>(
    logical_name: impl Into<String>,
    role: ParameterRole,
    projections: &[(&M, ProjectionSharding)],
    preferred_units: usize,
) -> Result<ParameterGroupSpec, ParallelPlanError>
where
    T: Tensor,
    M: Parameterized<T>,
{
    if preferred_units == 0 {
        return Err(ParallelPlanError::InvalidGroup(
            "partitioned projection group has zero preferred units".into(),
        ));
    }
    let mut units = preferred_units;
    let mut members = Vec::new();
    for (module, placement) in projections {
        let group = projection_parameter_group::<T, M>("projection", role, *module, *placement)?;
        for member in group.members {
            let sharding = match (placement, member.global_shape.len()) {
                (ProjectionSharding::Replicated, _) | (ProjectionSharding::Row, 0 | 1) => {
                    MemberSharding::Replicated
                }
                (ProjectionSharding::Column, 0) => unreachable!("validated above"),
                (ProjectionSharding::Column, _) => {
                    units = greatest_common_divisor(units, member.global_shape[0]);
                    MemberSharding::Partitioned { axis: 0 }
                }
                (ProjectionSharding::Row, _) => {
                    units = greatest_common_divisor(units, member.global_shape[1]);
                    MemberSharding::Partitioned { axis: 1 }
                }
            };
            members.push(ParameterMemberSpec::new(
                member.target,
                member.global_shape,
                sharding,
            ));
        }
    }
    ParameterGroupSpec::partitioned(logical_name, role, units, members)
}

/// Returns the finest legal logical-unit count for an aligned partition.
pub fn aligned_partition_units(
    name: &str,
    semantic_units: usize,
    elements_per_unit: usize,
    required_alignment: usize,
) -> Result<usize, ParallelPlanError> {
    if semantic_units == 0 || elements_per_unit == 0 || required_alignment == 0 {
        return Err(ParallelPlanError::InvalidGroup(format!(
            "{name} aligned partition dimensions must be positive, got units={semantic_units}, width={elements_per_unit}, alignment={required_alignment}"
        )));
    }
    let units_per_partition =
        required_alignment / greatest_common_divisor(elements_per_unit, required_alignment);
    if !semantic_units.is_multiple_of(units_per_partition) {
        return Err(ParallelPlanError::InvalidGroup(format!(
            "{name} has {semantic_units} semantic units of width {elements_per_unit}, which cannot form complete alignment-{required_alignment} partitions"
        )));
    }
    Ok(semantic_units / units_per_partition)
}

const fn greatest_common_divisor(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
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

/// Backend-neutral placement decision for one physical tensor.
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
    Range {
        /// Source tensor axis being sliced.
        axis: usize,
        /// Inclusive element offset on `axis`.
        start: usize,
        /// Exclusive element offset on `axis`.
        end: usize,
    },
    /// Materialize selected source-tensor indices in the supplied order.
    Indices {
        /// Source tensor axis being selected.
        axis: usize,
        /// Distinct source indices in local output order.
        indices: Vec<usize>,
    },
}

/// Rank-local shape and placement for one planned physical tensor.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct LocalTensorLayout<P = TensorPlacement> {
    logical_name: String,
    role: ParameterRole,
    global_shape: Vec<usize>,
    local_shape: Vec<usize>,
    placement: P,
    logical_units: Option<usize>,
    logical_range: Option<Range<usize>>,
    fell_back_to_replication: bool,
}

impl<P> LocalTensorLayout<P> {
    /// Creates one validated-planner output entry.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        logical_name: impl Into<String>,
        role: ParameterRole,
        global_shape: Vec<usize>,
        local_shape: Vec<usize>,
        placement: P,
        logical_units: Option<usize>,
        logical_range: Option<Range<usize>>,
        fell_back_to_replication: bool,
    ) -> Self {
        Self {
            logical_name: logical_name.into(),
            role,
            global_shape,
            local_shape,
            placement,
            logical_units,
            logical_range,
            fell_back_to_replication,
        }
    }

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

    /// Returns the backend-realized placement.
    pub const fn placement(&self) -> &P {
        &self.placement
    }

    /// Returns the rank-local range in the parameter group's semantic domain.
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
#[derive(Debug, Clone)]
pub struct LocalModelLayout<P = TensorPlacement> {
    tensors: BTreeMap<String, LocalTensorLayout<P>>,
}

impl<P> Default for LocalModelLayout<P> {
    fn default() -> Self {
        Self {
            tensors: BTreeMap::new(),
        }
    }
}

impl<P> LocalModelLayout<P> {
    /// Returns whether a physical target has already been planned.
    pub fn contains(&self, target: &str) -> bool {
        self.tensors.contains_key(target)
    }

    /// Inserts one planner-produced physical layout.
    pub fn insert(&mut self, target: String, layout: LocalTensorLayout<P>) {
        self.tensors.insert(target, layout);
    }

    /// Returns one physical tensor layout by rewritten target name.
    pub fn tensor(&self, target: &str) -> Option<&LocalTensorLayout<P>> {
        self.tensors.get(target)
    }

    /// Iterates physical layouts in deterministic target-name order.
    pub fn tensors(&self) -> impl Iterator<Item = (&str, &LocalTensorLayout<P>)> {
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

/// Invalid architecture-declared parallel semantics.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ParallelPlanError {
    /// A logical group is empty, ambiguous, or internally inconsistent.
    #[error("{0}")]
    InvalidGroup(String),
    /// A backend-native parameter exposed invalid logical geometry.
    #[error("{0}")]
    InvalidTensor(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_reject_duplicate_physical_targets() {
        let error = ParameterGroupSpec::new(
            "attention",
            ParameterRole::AttentionHeads,
            [
                ParameterMemberSpec::new("q.weight", [8, 8], MemberSharding::Replicated),
                ParameterMemberSpec::new("q.weight", [8, 8], MemberSharding::Replicated),
            ],
        )
        .unwrap_err();
        assert!(error.to_string().contains("repeats tensor target"));
    }

    #[test]
    fn group_partition_contract_is_explicit() {
        assert!(ParameterGroupSpec::new(
            "query",
            ParameterRole::AttentionHeads,
            [ParameterMemberSpec::new(
                "q.weight",
                [8, 8],
                MemberSharding::Partitioned { axis: 0 },
            )],
        )
        .is_err());
        assert!(ParameterGroupSpec::partitioned(
            "query",
            ParameterRole::AttentionHeads,
            4,
            [ParameterMemberSpec::new(
                "q.weight",
                [8, 8],
                MemberSharding::Partitioned { axis: 0 },
            )],
        )
        .is_ok());
    }

    #[test]
    fn parallel_model_info_preserves_opaque_topology_and_accounting() {
        let info = ParallelModelInfo::new(
            (2usize, 1usize),
            "generic",
            vec!["layer.weight".into()],
            10,
            20,
            4,
            8,
        );
        assert_eq!(info.topology(), (2, 1));
        assert_eq!(info.model_type(), "generic");
        assert_eq!(info.owned_tensors(), ["layer.weight"]);
        assert_eq!(info.local_parameter_bytes(), 10);
        assert_eq!(info.global_parameter_bytes(), 20);
        assert_eq!(info.pinned_device_parameter_bytes(), 4);
        assert_eq!(info.maximum_device_parameter_bytes(), 8);
    }
}

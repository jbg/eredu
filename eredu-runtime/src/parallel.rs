//! Backend-neutral semantic parameter sharding and rank-local layouts.
//!
//! Architectures describe physical checkpoint members in logical groups. An
//! execution backend may then realize the resulting placement without knowing
//! projection names, attention geometry, or other model-family semantics.

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
};

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
}

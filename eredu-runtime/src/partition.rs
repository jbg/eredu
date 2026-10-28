//! Backend-neutral ownership of one rank-local architecture partition.

use std::ops::Deref;
use std::{collections::BTreeMap, collections::BTreeSet, ops::Range};

use crate::{
    ArchitectureStatePartitionError, ArchitectureStatePartitionPlan, ArchitectureStatePlacement,
    ExecutionGraph, ExecutionGroupId, ExecutionUnitLayout, LayeredForwardState,
    LayeredPartitionInput, LayeredPartitionOutput, ParameterGroupSpec,
    PartitionedLayeredArchitecture, RuntimeState, StateLayout,
};

/// Architecture-owned location of one neutral parameter group.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
#[non_exhaustive]
pub enum ParameterGroupOwner {
    /// A pinned module selected by an explicit architecture static role.
    StaticRole(String),
    /// A shared pinned module selected when any declared static consumer is local.
    StaticAnyOf(Vec<String>),
    /// A pinned module used by explicit logical units. Each partition containing
    /// a consumer stores one copy, regardless of how many consumers it owns.
    StaticUnitConsumers {
        /// Canonical static binding role.
        role: String,
        /// Canonical group and unit identities that consume the module.
        consumers: Vec<(ExecutionGroupId, usize)>,
    },
    /// One canonical independently leased module and its logical consumers.
    #[non_exhaustive]
    ExecutionUnit {
        /// Canonical execution-group identity.
        group: ExecutionGroupId,
        /// Canonical physical owner address within its execution group.
        global_unit: usize,
        /// Logical consumers in the same canonical execution group. Parameter
        /// sharing is independent of each consumer's mutable-state ownership.
        consumers: Vec<usize>,
    },
}

impl ParameterGroupOwner {
    /// Creates static-module ownership with a stable, non-empty role.
    pub fn static_role(role: impl Into<String>) -> Self {
        Self::StaticRole(role.into())
    }

    /// Creates shared static-module ownership across explicit consumer roles.
    pub fn static_any_of(roles: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self::StaticAnyOf(roles.into_iter().map(Into::into).collect())
    }

    /// Pins one shared module on each partition executing a declared consumer.
    pub fn static_unit_consumers(
        role: impl Into<String>,
        consumers: impl IntoIterator<Item = (ExecutionGroupId, usize)>,
    ) -> Self {
        Self::StaticUnitConsumers {
            role: role.into(),
            consumers: consumers.into_iter().collect(),
        }
    }

    /// Whether this declaration binds through the given static storage role.
    pub fn accepts_static_role(&self, expected: &str) -> bool {
        match self {
            Self::StaticRole(role) | Self::StaticUnitConsumers { role, .. } => role == expected,
            Self::StaticAnyOf(roles) => roles.iter().any(|role| role == expected),
            Self::ExecutionUnit { .. } => false,
        }
    }

    /// Whether one explicit logical unit consumes this resource.
    /// Static roles alone do not declare unit consumers or mutable-state sharing.
    pub fn has_unit_consumer(&self, group: &ExecutionGroupId, unit: usize) -> bool {
        match self {
            Self::ExecutionUnit {
                group: owner,
                consumers,
                ..
            } => owner == group && consumers.contains(&unit),
            Self::StaticUnitConsumers { consumers, .. } => consumers
                .iter()
                .any(|(owner, index)| owner == group && *index == unit),
            Self::StaticRole(_) | Self::StaticAnyOf(_) => false,
        }
    }

    /// Compares canonical ownership, allowing constructed consumers to refine
    /// a selected binding role or extend its declared invocation consumers.
    pub fn refines_storage_owner(&self, selected: &Self) -> bool {
        self == selected
            || matches!(selected, Self::StaticRole(role) if self.accepts_static_role(role))
            || matches!((self, selected), (
                Self::ExecutionUnit { group, global_unit, consumers },
                Self::ExecutionUnit {
                    group: expected_group,
                    global_unit: expected_unit,
                    consumers: expected_consumers,
                },
            ) if group == expected_group && global_unit == expected_unit
                && expected_consumers.iter().all(|unit| consumers.contains(unit)))
    }

    /// Compares canonical storage without copying consumer declarations or role names.
    pub(crate) fn same_storage_as(&self, other: &Self) -> bool {
        match (self.static_storage_role(), other.static_storage_role()) {
            (Some(left), Some(right)) => left == right,
            (None, None) => matches!((self, other), (
                Self::ExecutionUnit { group: a, global_unit: i, .. },
                Self::ExecutionUnit { group: b, global_unit: j, .. },
            ) if a == b && i == j),
            _ => false,
        }
    }

    /// Creates execution-unit ownership in the architecture-global index space.
    pub fn execution_unit(group: ExecutionGroupId, global_unit: usize) -> Self {
        Self::ExecutionUnit {
            group,
            global_unit,
            consumers: vec![global_unit],
        }
    }

    /// Declares one independently leased owner used by explicit logical invocations.
    pub fn execution_unit_consumers(
        group: ExecutionGroupId,
        global_unit: usize,
        consumers: impl IntoIterator<Item = usize>,
    ) -> Self {
        Self::ExecutionUnit {
            group,
            global_unit,
            consumers: consumers.into_iter().collect(),
        }
    }

    pub(crate) fn validate(
        &self,
        graph: &ExecutionGraph,
        layout: &ExecutionUnitLayout,
    ) -> Result<(), ArchitectureParameterError> {
        match self {
            ParameterGroupOwner::StaticRole(role) => {
                if role.trim().is_empty() {
                    return Err(ArchitectureParameterError::EmptyStaticRole);
                }
            }
            ParameterGroupOwner::StaticAnyOf(roles) => {
                if roles.is_empty() || roles.iter().any(|role| role.trim().is_empty()) {
                    return Err(ArchitectureParameterError::EmptyStaticRole);
                }
                let unique = roles.iter().collect::<BTreeSet<_>>();
                if unique.len() != roles.len() {
                    return Err(ArchitectureParameterError::DuplicateStaticRole);
                }
            }
            ParameterGroupOwner::StaticUnitConsumers { role, consumers } => {
                if role.trim().is_empty() {
                    return Err(ArchitectureParameterError::EmptyStaticRole);
                }
                if consumers.is_empty() {
                    return Err(ArchitectureParameterError::EmptyStaticConsumers);
                }
                if consumers.iter().collect::<BTreeSet<_>>().len() != consumers.len() {
                    return Err(ArchitectureParameterError::DuplicateStaticConsumer);
                }
            }
            ParameterGroupOwner::ExecutionUnit { consumers, .. } => {
                if consumers.is_empty()
                    || consumers.iter().collect::<BTreeSet<_>>().len() != consumers.len()
                {
                    return Err(ArchitectureParameterError::InvalidUnitConsumers);
                }
            }
        }
        let validate_unit = |group: &ExecutionGroupId, global_unit| {
            let Some(group_index) = graph.group_index(group.as_str()) else {
                return Err(ArchitectureParameterError::UnknownExecutionGroup(
                    group.as_str().to_owned(),
                ));
            };
            let available = layout
                .group_range(group_index)
                .expect("validated canonical layout contains every group")
                .len();
            if layout.ordinal(group_index, global_unit).is_none() {
                return Err(ArchitectureParameterError::UnitOutOfRange {
                    group: group.as_str().to_owned(),
                    global_unit,
                    available,
                });
            }
            Ok(())
        };
        match self {
            Self::StaticUnitConsumers { consumers, .. } => consumers
                .iter()
                .try_for_each(|(group, unit)| validate_unit(group, *unit)),
            Self::ExecutionUnit {
                group,
                global_unit,
                consumers,
            } => {
                validate_unit(group, *global_unit)?;
                consumers
                    .iter()
                    .try_for_each(|unit| validate_unit(group, *unit))
            }
            Self::StaticRole(_) | Self::StaticAnyOf(_) => Ok(()),
        }
    }

    fn is_local<G, A>(&self, partition: &ArchitecturePartition<G, A>) -> bool {
        self.is_stored_by(partition.ownership(), |group, unit| {
            partition.owns_unit(group.as_str(), unit)
        })
    }

    /// Tests static consumers and unit ownership without constructing a module.
    pub fn is_owned_by(
        &self,
        ownership: &PartitionOwnership,
        owns_unit: impl Fn(&ExecutionGroupId, usize) -> bool,
    ) -> bool {
        match self {
            Self::StaticRole(role) => ownership.owns_static_role(role),
            Self::StaticAnyOf(roles) => roles.iter().any(|role| ownership.owns_static_role(role)),
            Self::StaticUnitConsumers { consumers, .. } => consumers
                .iter()
                .any(|(group, unit)| owns_unit(group, *unit)),
            Self::ExecutionUnit {
                group, consumers, ..
            } => consumers.iter().any(|unit| owns_unit(group, *unit)),
        }
    }

    /// Tests parameter storage, including replicas for auxiliary invocations.
    /// Use `is_owned_by` for the ordinary model's observation ownership.
    pub fn is_stored_by(
        &self,
        ownership: &PartitionOwnership,
        owns_unit: impl Fn(&ExecutionGroupId, usize) -> bool,
    ) -> bool {
        ownership
            .replicated_static_roles()
            .iter()
            .any(|role| self.accepts_static_role(role))
            || self.is_owned_by(ownership, owns_unit)
    }

    /// Returns the canonical static storage role, independently of its consumers.
    /// Shared static groups use their first declared role as the storage owner.
    /// Execution-unit storage has no static role.
    pub fn static_storage_role(&self) -> Option<&str> {
        match self {
            Self::StaticRole(role) => Some(role),
            Self::StaticAnyOf(roles) => roles.first().map(String::as_str),
            Self::StaticUnitConsumers { role, .. } => Some(role),
            Self::ExecutionUnit { .. } => None,
        }
    }
}

/// Source-format conversion changes canonical ownership of a selected parameter.
#[derive(Debug, thiserror::Error)]
pub enum TransformParameterOwnershipError {
    /// The selected output has no target partition owner.
    #[error("selected transform task {0:?} has no target partition owner")]
    MissingTarget(String),
    /// The selected output has no source partition owner.
    #[error("selected transform task {0:?} has no source partition owner")]
    MissingSource(String),
    /// Source and target assign the same parameter to different storage owners.
    #[error("selected transform task {0:?} changed its source/target owner address")]
    Changed(String),
}

/// One neutral parameter group tagged with its architecture-owned location.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct OwnedParameterGroupSpec {
    owner: ParameterGroupOwner,
    group: ParameterGroupSpec,
}

impl OwnedParameterGroupSpec {
    /// Tags a group with one explicit owner.
    pub fn new(owner: ParameterGroupOwner, group: ParameterGroupSpec) -> Self {
        Self { owner, group }
    }

    /// Returns the architecture-owned location.
    pub const fn owner(&self) -> &ParameterGroupOwner {
        &self.owner
    }

    /// Returns the neutral placement group.
    pub const fn group(&self) -> &ParameterGroupSpec {
        &self.group
    }

    /// Consumes the tag and returns the neutral placement group.
    pub fn into_group(self) -> ParameterGroupSpec {
        self.group
    }
}

impl Deref for OwnedParameterGroupSpec {
    type Target = ParameterGroupSpec;

    fn deref(&self) -> &Self::Target {
        &self.group
    }
}

/// Complete, validated parameter-ownership declaration for an architecture.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ArchitectureParameterDescription {
    graph: ExecutionGraph,
    unit_layout: ExecutionUnitLayout,
    invocations: crate::ExecutionUnitAddressMap,
    groups: Vec<OwnedParameterGroupSpec>,
    partition_layout: Option<(
        eredu_core::ParallelRankTopology,
        std::sync::Arc<crate::LocalModelLayout>,
    )>,
}

impl ArchitectureParameterDescription {
    /// Validates explicit ownership against the canonical graph/layout and an
    /// authoritative set of neutral parameter groups. Every target retains its
    /// member metadata and group semantics, independently of member ordering.
    pub fn new(
        graph: &ExecutionGraph,
        layout: &ExecutionUnitLayout,
        expected: impl IntoIterator<Item = ParameterGroupSpec>,
        groups: impl IntoIterator<Item = OwnedParameterGroupSpec>,
    ) -> Result<Self, ArchitectureParameterError> {
        validate_canonical_layout(graph, layout)
            .map_err(|error| ArchitectureParameterError::InvalidLayout(error.to_string()))?;
        let expected = parameter_declarations(expected)?;
        Self::validate_ownership(graph, layout, Some(expected), groups.into_iter().collect())
    }

    /// Validates a canonical owned declaration without duplicating its parameter catalog.
    /// Use [`Self::new`] when ownership must match independently declared parameters.
    pub fn from_owned(
        graph: &ExecutionGraph,
        layout: &ExecutionUnitLayout,
        groups: impl IntoIterator<Item = OwnedParameterGroupSpec>,
    ) -> Result<Self, ArchitectureParameterError> {
        validate_canonical_layout(graph, layout)
            .map_err(|error| ArchitectureParameterError::InvalidLayout(error.to_string()))?;
        Self::validate_ownership(graph, layout, None, groups.into_iter().collect())
    }

    fn validate_ownership(
        graph: &ExecutionGraph,
        layout: &ExecutionUnitLayout,
        expected: Option<BTreeMap<String, ExpectedParameter>>,
        groups: Vec<OwnedParameterGroupSpec>,
    ) -> Result<Self, ArchitectureParameterError> {
        let mut actual = BTreeMap::new();
        for tagged in &groups {
            tagged.owner().validate(graph, layout)?;
            for member in tagged.group().members() {
                if let Some(previous) = actual.insert(member.target(), tagged.owner()) {
                    return Err(ArchitectureParameterError::DuplicateOwnership {
                        target: member.target().to_owned(),
                        first: previous.clone(),
                        second: tagged.owner().clone(),
                    });
                }
            }
        }
        if let Some(expected) = expected {
            if let Some(target) = expected
                .keys()
                .find(|target| !actual.contains_key(target.as_str()))
            {
                return Err(ArchitectureParameterError::MissingOwnership(target.clone()));
            }
            if let Some(target) = actual.keys().find(|target| !expected.contains_key(**target)) {
                return Err(ArchitectureParameterError::UnexpectedOwnership(
                    target.to_string(),
                ));
            }
            for tagged in &groups {
                for member in tagged.members() {
                    if !expected[member.target()].matches(tagged.group(), member) {
                        return Err(ArchitectureParameterError::InconsistentDeclaration {
                            target: member.target().to_owned(),
                            group: tagged.logical_name().to_owned(),
                        });
                    }
                }
            }
        }
        let mut owners = BTreeMap::new();
        let mut claims = BTreeMap::new();
        for tagged in &groups {
            if let ParameterGroupOwner::ExecutionUnit {
                group,
                global_unit,
                consumers,
            } = tagged.owner()
            {
                let group = graph.group_index(group.as_str()).expect("validated group");
                let owner = layout
                    .address(
                        layout
                            .ordinal(group, *global_unit)
                            .expect("validated owner"),
                    )
                    .expect("declared owner");
                let selected = owners
                    .entry((group, *global_unit))
                    .or_insert_with(|| consumers.iter().copied().collect::<BTreeSet<_>>());
                if selected.len() != consumers.len()
                    || consumers.iter().any(|consumer| !selected.contains(consumer))
                {
                    return Err(ArchitectureParameterError::InconsistentUnitConsumers {
                        group,
                        global_unit: *global_unit,
                    });
                }
                for &consumer in selected.iter() {
                    if let Some(previous) = claims.insert((group, consumer), owner) {
                        if previous != owner {
                            return Err(ArchitectureParameterError::ConflictingUnitOwners {
                                group,
                                global_unit: consumer,
                            });
                        }
                    }
                }
            }
        }
        let references = (0..layout.len())
            .map(|ordinal| {
                let logical = layout.address(ordinal).expect("declared invocation");
                let storage = claims
                    .get(&(logical.group(), logical.index()))
                    .copied()
                    .unwrap_or(logical);
                (logical, storage)
            })
            .collect::<Vec<_>>();
        let used = references
            .iter()
            .map(|(_, storage)| (storage.group(), storage.index()))
            .collect::<BTreeSet<_>>();
        let storage = layout
            .select(
                (0..layout.len())
                    .map(|ordinal| layout.address(ordinal).expect("declared invocation"))
                    .filter(|address| used.contains(&(address.group(), address.index()))),
            )
            .expect("canonical physical subset");
        let invocations = crate::ExecutionUnitAddressMap::new(&storage, references)
            .expect("validated consumer ownership");
        Ok(Self {
            graph: graph.clone(),
            unit_layout: layout.clone(),
            invocations,
            groups,
            partition_layout: None,
        })
    }

    /// Retains architecture-authored placement for this bound rank. Consumers must
    /// not derive other ranks from member rules when this authority is present.
    pub fn with_partition_layout(
        mut self,
        topology: eredu_core::ParallelRankTopology,
        layout: crate::LocalModelLayout,
    ) -> Result<Self, ArchitectureParameterError> {
        let mut count = 0;
        for group in &self.groups {
            for member in group.members() {
                count += 1;
                let tensor = layout.tensor(member.target()).ok_or_else(|| {
                    ArchitectureParameterError::InvalidLayout(format!(
                        "retained partition layout has no parameter {}",
                        member.target()
                    ))
                })?;
                if tensor.global_shape() != member.global_shape()
                    || tensor.logical_name() != group.logical_name()
                    || tensor.role() != group.role()
                {
                    return Err(ArchitectureParameterError::InvalidLayout(format!(
                        "retained partition layout changed parameter {}",
                        member.target()
                    )));
                }
            }
        }
        if count != layout.len() {
            return Err(ArchitectureParameterError::InvalidLayout(
                "retained partition layout contains undeclared parameters".into(),
            ));
        }
        self.partition_layout = Some((topology, std::sync::Arc::new(layout)));
        Ok(self)
    }

    /// Exact placement retained by architecture construction, when member rules
    /// alone cannot reconstruct its rank semantics.
    pub fn partition_layout(
        &self,
    ) -> Option<(eredu_core::ParallelRankTopology, &crate::LocalModelLayout)> {
        self.partition_layout
            .as_ref()
            .map(|(rank, layout)| (*rank, layout.as_ref()))
    }

    /// Returns the canonical execution graph that owns these parameter groups.
    pub const fn graph(&self) -> &ExecutionGraph {
        &self.graph
    }

    /// Returns the canonical architecture-global execution-unit layout that
    /// owns these parameter groups.
    pub const fn unit_layout(&self) -> &ExecutionUnitLayout {
        &self.unit_layout
    }

    /// Canonical physical owners and the logical invocations that consume them.
    pub fn invocations(&self) -> &crate::ExecutionUnitAddressMap {
        &self.invocations
    }

    /// Proves that this description still matches a concrete neutral architecture.
    pub fn validate_architecture<B, S, M>(
        &self,
        architecture: &M,
    ) -> Result<(), ArchitecturePartitionError>
    where
        B: eredu_nn::NeuralBackend,
        S: crate::RuntimeState<B>,
        M: crate::LayeredArchitecture<B, S>,
        M::Error: std::fmt::Display,
    {
        let (graph, unit_layout) = canonical_architecture_layout::<B, S, M>(architecture)?;
        if graph != self.graph {
            return Err(ArchitecturePartitionError::ArchitectureGraphMismatch);
        }
        if unit_layout != self.unit_layout {
            return Err(ArchitecturePartitionError::ArchitectureUnitLayoutMismatch);
        }
        let invocations = architecture
            .execution_unit_storage_map()
            .map_err(|error| ArchitecturePartitionError::ArchitectureTopology(error.to_string()))?;
        if invocations != self.invocations {
            return Err(ArchitecturePartitionError::ArchitectureStorageMapMismatch);
        }
        Ok(())
    }

    /// Returns every explicitly tagged neutral parameter group.
    pub fn groups(&self) -> &[OwnedParameterGroupSpec] {
        &self.groups
    }

    /// Borrows physical member specifications by their authoritative target names.
    ///
    /// Declaration validation guarantees unique ownership. The view includes
    /// encoding companions and retains their exact metadata without copying it;
    /// it grants no source, materialization or parameter-edit authority.
    pub fn members_by_target(&self) -> BTreeMap<&str, &crate::ParameterMemberSpec> {
        self.groups
            .iter()
            .flat_map(|owned| owned.group().members())
            .map(|member| (member.target(), member))
            .collect()
    }

    /// Returns every physical target owned by groups with the supplied semantic role.
    ///
    /// Selection happens after architecture ownership has been assigned, so callers
    /// do not need to rediscover families, aliases, or packed companions from target
    /// name syntax.
    pub fn targets_for_role(&self, role: crate::ParameterRole) -> BTreeSet<String> {
        self.groups
            .iter()
            .filter(|owned| owned.group().role() == role)
            .flat_map(|owned| owned.group().members())
            .map(|member| member.target().to_owned())
            .collect()
    }

    /// Projects named execution groups and pinned roles, preserving canonical owners,
    /// sparse global unit indices and exact retained rank placement. Omitted graph
    /// dependencies become caller-owned ingress; included dependencies stay unchanged.
    pub fn project_groups(
        &self,
        graph: &ExecutionGraph,
        pinned_roles: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, ArchitectureParameterError> {
        let roles = pinned_roles
            .into_iter()
            .map(Into::into)
            .collect::<BTreeSet<_>>();
        for role in &roles {
            if !self
                .groups
                .iter()
                .any(|group| group.owner().accepts_static_role(role))
            {
                return Err(ArchitectureParameterError::UnknownProjectedStaticRole(
                    role.clone(),
                ));
            }
        }
        let layout = self.unit_layout.project_groups(graph)?;
        for group in graph.groups() {
            let parent = self.graph.group_index(group.id()).expect("projected group");
            let dependencies = self.graph.groups()[parent]
                .dependencies()
                .iter()
                .filter(|dependency| graph.group_index(dependency).is_some());
            if !group.dependencies().iter().eq(dependencies) {
                return Err(ArchitectureParameterError::InvalidProjectedDependencies(
                    group.id().to_owned(),
                ));
            }
        }
        let groups = self.groups.iter().filter(|owned| match owned.owner() {
            ParameterGroupOwner::ExecutionUnit { group, .. } => {
                graph.group_index(group.as_str()).is_some()
            }
            ParameterGroupOwner::StaticUnitConsumers { consumers, .. } => {
                roles
                    .iter()
                    .any(|role| owned.owner().accepts_static_role(role))
                    || consumers
                        .iter()
                        .any(|(group, _)| graph.group_index(group.as_str()).is_some())
            }
            ParameterGroupOwner::StaticRole(_) | ParameterGroupOwner::StaticAnyOf(_) => roles
                .iter()
                .any(|role| owned.owner().accepts_static_role(role)),
        });
        let mut selected = Self::from_owned(graph, &layout, groups.cloned())?;
        if let Some((rank, retained)) = self.partition_layout() {
            let mut local = crate::LocalModelLayout::default();
            for group in selected.groups() {
                for member in group.members() {
                    local.insert(
                        member.target().to_owned(),
                        retained
                            .tensor(member.target())
                            .expect("validated retained parameter placement")
                            .clone(),
                    );
                }
            }
            selected = selected.with_partition_layout(rank, local)?;
        }
        Ok(selected)
    }

    /// Selects rank-owned groups without discarding their architecture owner.
    pub fn select_owned<G, A>(
        &self,
        partition: &ArchitecturePartition<G, A>,
    ) -> Vec<OwnedParameterGroupSpec> {
        self.stored_groups(partition.ownership(), |group, unit| {
            partition.owns_unit(group.as_str(), unit)
        })
        .cloned()
        .collect()
    }

    /// Compares retained partition bindings with their exact declared ownership.
    /// The comparison borrows parameter groups instead of cloning a selected catalog.
    pub fn matches_partition_bindings<G, A>(
        &self,
        partition: &ArchitecturePartition<G, A>,
    ) -> bool {
        partition.parameter_bindings().iter().eq(self.stored_groups(
            partition.ownership(),
            |group, unit| partition.owns_unit(group.as_str(), unit),
        ))
    }

    /// Projects every group in a stored canonical resource, including physical replicas.
    /// Logical invocation and observation ownership remain in each group's declaration.
    pub fn stored_groups(
        &self,
        ownership: &PartitionOwnership,
        owns_unit: impl Fn(&ExecutionGroupId, usize) -> bool,
    ) -> impl Iterator<Item = &OwnedParameterGroupSpec> {
        let storage = self
            .groups
            .iter()
            .filter(|tagged| tagged.owner().is_stored_by(ownership, &owns_unit))
            .map(|tagged| crate::ParameterStorageId::from_owner(tagged.owner()))
            .collect::<BTreeSet<_>>();
        self.groups.iter().filter(move |tagged| {
            storage.contains(&crate::ParameterStorageId::from_owner(tagged.owner()))
        })
    }

    /// Resolves one canonical resource without allocating a selected parameter catalog.
    pub fn stores_owner(
        &self,
        owner: &ParameterGroupOwner,
        ownership: &PartitionOwnership,
        owns_unit: impl Fn(&ExecutionGroupId, usize) -> bool,
    ) -> bool {
        self.groups.iter().any(|tagged| {
            tagged.owner().same_storage_as(owner)
                && tagged.owner().is_stored_by(ownership, &owns_unit)
        })
    }

    /// Returns canonical static storage roles selected for one partition.
    ///
    /// Shared owners always return their first declared role, while any later
    /// roles only act as ownership consumers.
    pub fn select_static_roles<'a, G, A>(
        &'a self,
        partition: &ArchitecturePartition<G, A>,
    ) -> Vec<&'a str> {
        self.groups
            .iter()
            .filter(|tagged| tagged.owner().is_local(partition))
            .filter_map(|tagged| tagged.owner().static_storage_role())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}

struct ExpectedParameter {
    group: std::sync::Arc<ParameterGroupSpec>,
    member: usize,
}

impl ExpectedParameter {
    fn matches(&self, group: &ParameterGroupSpec, member: &crate::ParameterMemberSpec) -> bool {
        self.group.logical_name() == group.logical_name()
            && self.group.role() == group.role()
            && self.group.partition_units() == group.partition_units()
            && self.group.partition_unit_assignment() == group.partition_unit_assignment()
            && self.group.members()[self.member] == *member
    }
}

fn parameter_declarations(
    groups: impl IntoIterator<Item = ParameterGroupSpec>,
) -> Result<BTreeMap<String, ExpectedParameter>, ArchitectureParameterError> {
    let mut targets = BTreeMap::new();
    for group in groups {
        let group = std::sync::Arc::new(group);
        for (index, member) in group.members().iter().enumerate() {
            let declaration = ExpectedParameter {
                group: group.clone(),
                member: index,
            };
            if let Some(previous) = targets.insert(member.target().to_owned(), declaration) {
                return Err(ArchitectureParameterError::DuplicateExpectedTarget {
                    target: member.target().to_owned(),
                    first: previous.group.logical_name().to_owned(),
                    second: group.logical_name().to_owned(),
                });
            }
        }
    }
    Ok(targets)
}

/// Invalid architecture-owned parameter ownership declaration.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ArchitectureParameterError {
    /// A projected pinned role is absent from the canonical declaration.
    #[error("unknown projected static parameter role {0:?}")]
    UnknownProjectedStaticRole(String),
    /// A child graph changes dependencies between included execution groups.
    #[error("projected dependencies differ for execution group {0:?}")]
    InvalidProjectedDependencies(String),
    /// Named group selection cannot preserve the canonical invocation layout.
    #[error("invalid projected invocation layout: {0}")]
    ProjectionLayout(#[from] crate::ExecutionUnitLayoutError),
    /// A leased resource has no consumers or repeats one logical consumer.
    #[error("architecture leased parameter owner has empty or repeated consumers")]
    InvalidUnitConsumers,
    /// Parameter groups for one physical owner disagree about its consumers.
    #[error("architecture parameter owner {group}:{global_unit} has inconsistent consumers")]
    InconsistentUnitConsumers {
        /// Canonical group slot.
        group: usize,
        /// Canonical physical owner index.
        global_unit: usize,
    },
    /// One logical invocation claims multiple independently leased owners.
    #[error("architecture invocation {group}:{global_unit} has conflicting leased owners")]
    ConflictingUnitOwners {
        /// Canonical group slot.
        group: usize,
        /// Logical invocation index.
        global_unit: usize,
    },

    /// The supplied graph/layout is not canonical.
    #[error("invalid architecture parameter layout: {0}")]
    InvalidLayout(String),
    /// A pinned parameter group has no semantic role.
    #[error("architecture parameter static role must not be empty")]
    EmptyStaticRole,
    /// A shared pinned parameter repeats one consumer role.
    #[error("architecture shared parameter owner repeats a static role")]
    DuplicateStaticRole,
    /// A unit-consumed static module has no declared invocation.
    #[error("architecture pinned parameter has no unit consumer")]
    EmptyStaticConsumers,
    /// A shared pinned module repeats a logical invocation.
    #[error("architecture pinned parameter repeats a unit consumer")]
    DuplicateStaticConsumer,
    /// A unit owner names no canonical graph group.
    #[error("architecture parameter owner names unknown execution group {0:?}")]
    UnknownExecutionGroup(String),
    /// A unit reference is absent from its declared group.
    #[error("architecture parameter reference {group}:{global_unit} is not among its {available} declared units")]
    UnitOutOfRange {
        /// Canonical execution-group identity.
        group: String,
        /// Invalid group-local global unit.
        global_unit: usize,
        /// Canonical unit count.
        available: usize,
    },
    /// The authoritative neutral group set itself repeats a target.
    #[error("expected parameter target {target:?} appears in both {first:?} and {second:?}")]
    DuplicateExpectedTarget {
        /// Repeated physical target.
        target: String,
        /// First logical group.
        first: String,
        /// Second logical group.
        second: String,
    },
    /// Two explicit owners claim one physical target.
    #[error("parameter target {target:?} is owned by both {first:?} and {second:?}")]
    DuplicateOwnership {
        /// Repeated physical target.
        target: String,
        /// First explicit owner.
        first: ParameterGroupOwner,
        /// Second explicit owner.
        second: ParameterGroupOwner,
    },
    /// An authoritative target was left unowned.
    #[error("parameter target {0:?} has no architecture owner")]
    MissingOwnership(String),
    /// An ownership tag names a target outside the authoritative set.
    #[error("parameter target {0:?} is not present in the authoritative parameter groups")]
    UnexpectedOwnership(String),
    /// An ownership tag changes an authoritative parameter or its group semantics.
    #[error(
        "parameter target {target:?} in group {group:?} differs from its authoritative declaration"
    )]
    InconsistentDeclaration {
        /// Canonical physical parameter target.
        target: String,
        /// Logical group in the ownership declaration.
        group: String,
    },
}

/// Logical scalar kind carried by one architecture-owned boundary tensor.
///
/// `Activation` is resolved by a concrete backend to the execution dtype
/// selected for the surrounding pipeline activation. Integer kinds are exact.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum BoundaryTensorDtype {
    /// Exact FP32 auxiliary values, independent of activation transport precision.
    Float32,
    /// The selected execution activation dtype.
    Activation,
    /// Exact unsigned 32-bit integer values.
    Uint32,
    /// Exact signed 32-bit integer values.
    Int32,
}

/// Portable floating-point dtype carried between pipeline stages.
///
/// This is execution transport policy, not checkpoint storage metadata. A
/// concrete backend must lower the selected dtype to its native tensor dtype
/// and normalize outgoing activations to it before transport.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum PipelineActivationDtype {
    /// IEEE 16-bit floating point.
    Float16,
    /// Brain 16-bit floating point.
    Bfloat16,
    /// IEEE 32-bit floating point.
    Float32,
}

/// Backend-neutral wire contract shared by every stage of one pipeline.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub struct PipelineWireContract {
    activation_dtype: PipelineActivationDtype,
}

impl PipelineWireContract {
    /// Declares the exact dtype used by hidden activations and auxiliary
    /// tensors whose boundary dtype is [`BoundaryTensorDtype::Activation`].
    pub const fn new(activation_dtype: PipelineActivationDtype) -> Self {
        Self { activation_dtype }
    }

    /// Returns the exact floating-point dtype transported between stages.
    pub const fn activation_dtype(self) -> PipelineActivationDtype {
        self.activation_dtype
    }
}

/// One symbolic dimension in an architecture-owned boundary tensor.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum BoundaryTensorDimension {
    /// Invocation batch size.
    Batch,
    /// Invocation sequence length.
    Sequence,
    /// Positive architecture-defined extent.
    Fixed(i32),
}

/// Semantic role, symbolic shape, and logical dtype of one boundary tensor.
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct BoundaryTensorSpec {
    role: String,
    shape: Vec<BoundaryTensorDimension>,
    dtype: BoundaryTensorDtype,
}

impl BoundaryTensorSpec {
    /// Declares one tensor in canonical transport order.
    pub fn new(
        role: impl Into<String>,
        shape: impl IntoIterator<Item = BoundaryTensorDimension>,
        dtype: BoundaryTensorDtype,
    ) -> Self {
        Self {
            role: role.into(),
            shape: shape.into_iter().collect(),
            dtype,
        }
    }

    /// Declares the standard evolving batch/sequence/hidden activation.
    pub fn primary_activation(hidden_size: i32) -> Self {
        Self::new(
            "hidden",
            [
                BoundaryTensorDimension::Batch,
                BoundaryTensorDimension::Sequence,
                BoundaryTensorDimension::Fixed(hidden_size),
            ],
            BoundaryTensorDtype::Activation,
        )
    }

    /// Returns the stable semantic role.
    pub fn role(&self) -> &str {
        &self.role
    }

    /// Returns the symbolic shape.
    pub fn shape(&self) -> &[BoundaryTensorDimension] {
        &self.shape
    }

    /// Returns the logical scalar kind.
    pub const fn dtype(&self) -> BoundaryTensorDtype {
        self.dtype
    }
}

/// One boundary tensor after invocation-dependent dimensions are resolved.
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct ResolvedBoundaryTensorSpec {
    role: String,
    shape: Vec<i32>,
    dtype: BoundaryTensorDtype,
}

impl ResolvedBoundaryTensorSpec {
    /// Returns the stable semantic role.
    pub fn role(&self) -> &str {
        &self.role
    }

    /// Returns the concrete transport shape.
    pub fn shape(&self) -> &[i32] {
        &self.shape
    }

    /// Returns the logical scalar kind.
    pub const fn dtype(&self) -> BoundaryTensorDtype {
        self.dtype
    }
}

/// Complete primary and ordered auxiliary wire schema for one architecture boundary.
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct BoundaryWireSchema {
    identity: &'static str,
    primary: BoundaryTensorSpec,
    auxiliary: Vec<BoundaryTensorSpec>,
}

impl BoundaryWireSchema {
    /// Creates and validates an architecture-owned wire schema.
    pub fn new(
        identity: &'static str,
        primary: BoundaryTensorSpec,
        auxiliary: impl IntoIterator<Item = BoundaryTensorSpec>,
    ) -> Result<Self, ArchitectureBoundaryError> {
        if identity.trim().is_empty() {
            return Err(ArchitectureBoundaryError::EmptyIdentity);
        }
        if primary.dtype != BoundaryTensorDtype::Activation {
            return Err(ArchitectureBoundaryError::InvalidPrimaryDtype { boundary: identity });
        }
        let auxiliary = auxiliary.into_iter().collect::<Vec<_>>();
        let mut roles = BTreeSet::new();
        for tensor in std::iter::once(&primary).chain(&auxiliary) {
            if tensor.role.trim().is_empty() {
                return Err(ArchitectureBoundaryError::EmptyTensorRole { boundary: identity });
            }
            if !roles.insert(tensor.role.as_str()) {
                return Err(ArchitectureBoundaryError::DuplicateTensorRole {
                    boundary: identity,
                    role: tensor.role.clone(),
                });
            }
            if tensor.shape.is_empty() {
                return Err(ArchitectureBoundaryError::EmptyTensorShape {
                    boundary: identity,
                    role: tensor.role.clone(),
                });
            }
            if tensor
                .shape
                .iter()
                .any(|dimension| matches!(dimension, BoundaryTensorDimension::Fixed(value) if *value <= 0))
            {
                return Err(ArchitectureBoundaryError::InvalidTensorDimension {
                    boundary: identity,
                    role: tensor.role.clone(),
                });
            }
        }
        Ok(Self {
            identity,
            primary,
            auxiliary,
        })
    }

    /// Returns the stable schema identity.
    pub const fn identity(&self) -> &'static str {
        self.identity
    }

    /// Returns the primary evolving activation declaration.
    pub const fn primary(&self) -> &BoundaryTensorSpec {
        &self.primary
    }

    /// Returns auxiliary tensor declarations in canonical transport order.
    pub fn auxiliary(&self) -> &[BoundaryTensorSpec] {
        &self.auxiliary
    }

    /// Visits the primary tensor followed by auxiliaries in canonical transport order.
    pub fn tensors(&self) -> impl Iterator<Item = &BoundaryTensorSpec> {
        std::iter::once(&self.primary).chain(&self.auxiliary)
    }

    /// Resolves invocation-dependent dimensions without backend family logic.
    pub fn resolve(
        &self,
        batch_size: i32,
        sequence_length: i32,
    ) -> Result<ResolvedBoundaryWireSchema, ArchitectureBoundaryError> {
        self.resolve_each(
            batch_size,
            std::iter::repeat_n(sequence_length, 1 + self.auxiliary.len()),
        )
    }

    /// Resolves one exact sequence extent per primary/auxiliary tensor.
    ///
    /// This is used by composite boundaries whose evolving internal activation
    /// and learned side outputs have different sequence geometries. The family
    /// supplies values in canonical schema order; the runtime only validates and
    /// substitutes the declared symbolic dimensions.
    pub fn resolve_each(
        &self,
        batch_size: i32,
        sequence_lengths: impl IntoIterator<Item = i32>,
    ) -> Result<ResolvedBoundaryWireSchema, ArchitectureBoundaryError> {
        let sequence_lengths = sequence_lengths.into_iter().collect::<Vec<_>>();
        if sequence_lengths.len() != 1 + self.auxiliary.len() {
            return Err(ArchitectureBoundaryError::TensorCount {
                boundary: self.identity,
                expected: 1 + self.auxiliary.len(),
                actual: sequence_lengths.len(),
            });
        }
        if batch_size <= 0 || sequence_lengths.iter().any(|sequence| *sequence <= 0) {
            return Err(ArchitectureBoundaryError::InvalidInvocationGeometry {
                boundary: self.identity,
                batch_size,
                sequence_length: sequence_lengths
                    .into_iter()
                    .find(|value| *value <= 0)
                    .unwrap_or(0),
            });
        }
        let resolve = |tensor: &BoundaryTensorSpec, sequence_length| ResolvedBoundaryTensorSpec {
            role: tensor.role.clone(),
            shape: tensor
                .shape
                .iter()
                .map(|dimension| match dimension {
                    BoundaryTensorDimension::Batch => batch_size,
                    BoundaryTensorDimension::Sequence => sequence_length,
                    BoundaryTensorDimension::Fixed(value) => *value,
                })
                .collect(),
            dtype: tensor.dtype,
        };
        let mut sequences = sequence_lengths.into_iter();
        Ok(ResolvedBoundaryWireSchema {
            identity: self.identity,
            primary: resolve(
                &self.primary,
                sequences.next().expect("validated primary sequence"),
            ),
            auxiliary: self
                .auxiliary
                .iter()
                .zip(sequences)
                .map(|(tensor, sequence)| resolve(tensor, sequence))
                .collect(),
        })
    }
}

/// One architecture boundary after invocation-dependent dimensions are resolved.
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct ResolvedBoundaryWireSchema {
    identity: &'static str,
    primary: ResolvedBoundaryTensorSpec,
    auxiliary: Vec<ResolvedBoundaryTensorSpec>,
}

impl ResolvedBoundaryWireSchema {
    /// Returns the stable schema identity.
    pub const fn identity(&self) -> &'static str {
        self.identity
    }

    /// Returns the resolved primary evolving activation declaration.
    pub const fn primary(&self) -> &ResolvedBoundaryTensorSpec {
        &self.primary
    }

    /// Returns resolved auxiliary declarations in canonical transport order.
    pub fn auxiliary(&self) -> &[ResolvedBoundaryTensorSpec] {
        &self.auxiliary
    }

    /// Visits the primary tensor followed by auxiliaries in canonical transport order.
    pub fn tensors(&self) -> impl Iterator<Item = &ResolvedBoundaryTensorSpec> {
        std::iter::once(&self.primary).chain(&self.auxiliary)
    }
}

/// Typed architecture-owned tensor state and wire geometry carried across one
/// partition boundary.
///
/// The runtime and a backend transport may resolve and move the encoded tensor
/// vector, but only this family schema assigns semantic roles, shape, dtype,
/// cardinality, or reconstructs the typed value.
pub trait ArchitectureBoundary: Sized {
    /// Typed family value transported by this schema.
    type Boundary<T>;

    /// Stable non-empty semantic identity used in diagnostics and wire schema
    /// validation.
    const IDENTITY: &'static str;

    /// Primary evolving activation declaration.
    fn primary_tensor_spec(&self) -> BoundaryTensorSpec;

    /// Auxiliary tensor declarations in exact encoded order.
    fn auxiliary_tensor_specs(&self) -> Vec<BoundaryTensorSpec>;

    /// Consumes this typed value into exact role-tagged transport tensors.
    ///
    /// Roles are assigned while the architecture-owned typed value is
    /// decomposed; neutral execution must never reconstruct them positionally.
    fn encode<T>(
        &self,
        boundary: Self::Boundary<T>,
    ) -> Result<Vec<ArchitectureBoundaryValue<T>>, ArchitectureBoundaryError>;

    /// Reconstructs the typed value from transport-order tensors.
    fn decode<T>(&self, tensors: Vec<T>) -> Result<Self::Boundary<T>, ArchitectureBoundaryError>;

    /// Returns the validated backend-neutral wire schema.
    fn wire_schema(&self) -> Result<BoundaryWireSchema, ArchitectureBoundaryError> {
        BoundaryWireSchema::new(
            Self::IDENTITY,
            self.primary_tensor_spec(),
            self.auxiliary_tensor_specs(),
        )
    }
}

/// One architecture-tagged auxiliary boundary value.
///
/// The semantic role is assigned while the family-owned typed boundary is
/// decomposed. Keeping it coupled to the tensor prevents a neutral executor
/// from silently reassigning roles by positional zipping.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ArchitectureBoundaryValue<T> {
    role: String,
    tensor: T,
}

impl<T> ArchitectureBoundaryValue<T> {
    /// Couples a non-empty architecture role to its exact tensor value.
    pub fn new(role: impl Into<String>, tensor: T) -> Result<Self, ArchitectureBoundaryError> {
        let role = role.into();
        if role.trim().is_empty() {
            return Err(ArchitectureBoundaryError::EmptyTaggedTensorRole);
        }
        Ok(Self { role, tensor })
    }

    /// Architecture-owned semantic role.
    pub fn role(&self) -> &str {
        &self.role
    }

    /// Borrows the exact tensor assigned to this role.
    pub const fn tensor(&self) -> &T {
        &self.tensor
    }

    /// Decomposes this value without cloning the tensor.
    pub fn into_parts(self) -> (String, T) {
        (self.role, self.tensor)
    }
}

/// Explicit declaration that an architecture partition carries no auxiliary
/// tensors across its boundary.
///
/// This marker is preferable to `()` because it still participates in the
/// typed boundary contract and rejects any unexpected transported tensor.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct NoAuxiliaryBoundary;

/// Schema for an evolving decoder activation with no auxiliary tensors.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct NoAuxiliaryBoundarySchema {
    hidden_size: i32,
}

impl NoAuxiliaryBoundarySchema {
    /// Declares a standard batch/sequence/hidden activation boundary.
    pub const fn new(hidden_size: i32) -> Self {
        Self { hidden_size }
    }
}

impl ArchitectureBoundary for NoAuxiliaryBoundarySchema {
    type Boundary<T> = NoAuxiliaryBoundary;

    const IDENTITY: &'static str = "none";

    fn primary_tensor_spec(&self) -> BoundaryTensorSpec {
        BoundaryTensorSpec::primary_activation(self.hidden_size)
    }

    fn auxiliary_tensor_specs(&self) -> Vec<BoundaryTensorSpec> {
        Vec::new()
    }

    fn encode<T>(
        &self,
        _boundary: Self::Boundary<T>,
    ) -> Result<Vec<ArchitectureBoundaryValue<T>>, ArchitectureBoundaryError> {
        Ok(Vec::new())
    }

    fn decode<T>(&self, tensors: Vec<T>) -> Result<Self::Boundary<T>, ArchitectureBoundaryError> {
        validate_boundary_tensor_count(self, &tensors)?;
        Ok(NoAuxiliaryBoundary)
    }
}

/// Validates the number of tensors before a family boundary decodes any
/// positional value.
pub fn validate_boundary_tensor_count<B, T>(
    boundary: &B,
    tensors: &[T],
) -> Result<(), ArchitectureBoundaryError>
where
    B: ArchitectureBoundary,
{
    let expected = boundary.wire_schema()?.auxiliary().len();
    let actual = tensors.len();
    if actual != expected {
        return Err(ArchitectureBoundaryError::TensorCount {
            boundary: B::IDENTITY,
            expected,
            actual,
        });
    }
    Ok(())
}

/// Invalid architecture-owned partition boundary declaration or payload.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ArchitectureBoundaryError {
    /// A family boundary omitted its stable identity.
    #[error("architecture boundary identity must not be empty")]
    EmptyIdentity,
    /// A role-tagged family value omitted its semantic identity.
    #[error("architecture boundary value contains an empty tensor role")]
    EmptyTaggedTensorRole,
    /// A family boundary assigned a non-activation dtype to its primary tensor.
    #[error("architecture boundary {boundary:?} primary tensor must use activation dtype")]
    InvalidPrimaryDtype {
        /// Stable boundary identity.
        boundary: &'static str,
    },
    /// A family boundary declared an empty tensor role.
    #[error("architecture boundary {boundary:?} contains an empty tensor role")]
    EmptyTensorRole {
        /// Stable boundary identity.
        boundary: &'static str,
    },
    /// A family boundary declared one tensor role more than once.
    #[error("architecture boundary {boundary:?} repeats tensor role {role:?}")]
    DuplicateTensorRole {
        /// Stable boundary identity.
        boundary: &'static str,
        /// Repeated semantic tensor role.
        role: String,
    },
    /// A family boundary declared a rank-zero tensor.
    #[error("architecture boundary {boundary:?} tensor {role:?} has no dimensions")]
    EmptyTensorShape {
        /// Stable boundary identity.
        boundary: &'static str,
        /// Tensor semantic role.
        role: String,
    },
    /// A family boundary declared a non-positive fixed dimension.
    #[error("architecture boundary {boundary:?} tensor {role:?} has a non-positive dimension")]
    InvalidTensorDimension {
        /// Stable boundary identity.
        boundary: &'static str,
        /// Tensor semantic role.
        role: String,
    },
    /// A caller supplied non-positive invocation dimensions.
    #[error(
        "architecture boundary {boundary:?} requires positive invocation geometry, got batch {batch_size} and sequence {sequence_length}"
    )]
    InvalidInvocationGeometry {
        /// Stable boundary identity.
        boundary: &'static str,
        /// Invalid batch size.
        batch_size: i32,
        /// Invalid sequence length.
        sequence_length: i32,
    },
    /// A transported payload has the wrong tensor cardinality.
    #[error(
        "architecture boundary {boundary:?} expected {expected} tensors but received {actual}"
    )]
    TensorCount {
        /// Stable boundary identity.
        boundary: &'static str,
        /// Declared tensor count.
        expected: usize,
        /// Transported tensor count.
        actual: usize,
    },
    /// Family-specific boundary validation failed.
    #[error("architecture boundary {boundary:?} is invalid: {detail}")]
    Invalid {
        /// Stable boundary identity.
        boundary: &'static str,
        /// Family-owned failure detail.
        detail: String,
    },
}

/// Input, output, and pinned static-module ownership for one partition.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PartitionOwnership {
    input: bool,
    output: bool,
    static_roles: Vec<String>,
    replicated_static_roles: Vec<String>,
}

impl PartitionOwnership {
    /// Creates validated boundary and static-module ownership.
    pub fn new(
        input: bool,
        output: bool,
        static_roles: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, ArchitecturePartitionError> {
        let static_roles = static_roles.into_iter().map(Into::into).collect::<Vec<_>>();
        let mut unique = BTreeSet::new();
        for role in &static_roles {
            if role.trim().is_empty() {
                return Err(ArchitecturePartitionError::EmptyStaticRole);
            }
            if !unique.insert(role.clone()) {
                return Err(ArchitecturePartitionError::DuplicateStaticRole(
                    role.clone(),
                ));
            }
        }
        Ok(Self {
            input,
            output,
            static_roles,
            replicated_static_roles: Vec::new(),
        })
    }

    /// Retains static storage needed by auxiliary invocations on every rank.
    /// These replicas do not grant ownership of the ordinary input/output hook.
    pub fn with_replicated_static_roles(
        mut self,
        roles: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, ArchitecturePartitionError> {
        self.replicated_static_roles = Self::new(false, false, roles)?.static_roles;
        Ok(self)
    }

    /// Auxiliary storage dependencies, distinct from ordinary invocation roles.
    pub fn replicated_static_roles(&self) -> &[String] {
        &self.replicated_static_roles
    }

    /// Whether a static parameter is materialized here, including shared replicas.
    pub fn stores_static_role(&self, role: &str) -> bool {
        self.owns_static_role(role)
            || self
                .replicated_static_roles
                .iter()
                .any(|stored| stored == role)
    }

    /// Returns whether this partition owns model input preparation.
    pub const fn owns_input(&self) -> bool {
        self.input
    }

    /// Returns whether this partition owns model output production.
    pub const fn owns_output(&self) -> bool {
        self.output
    }

    /// Returns pinned static roles in architecture declaration order.
    pub fn static_roles(&self) -> &[String] {
        &self.static_roles
    }

    /// Returns whether this partition owns a named pinned static role.
    pub fn owns_static_role(&self, role: &str) -> bool {
        self.static_roles.iter().any(|candidate| candidate == role)
    }
}

/// Rank-local mutable-state geometry and its architecture-global layer range.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PartitionState {
    layout: StateLayout,
    global_layers: Range<usize>,
    selected_layers: Vec<Range<usize>>,
}

impl ArchitectureStatePartitionPlan {
    /// Validates complete state coverage and group geometry without selecting ownership.
    pub fn validate(
        &self,
        complete: &StateLayout,
        units: &ExecutionUnitLayout,
    ) -> Result<(), ArchitectureStatePartitionError> {
        if self.rules().is_empty() {
            if complete
                .layers()
                .iter()
                .all(|policy| matches!(policy, eredu_core::cache::LayerCachePolicy::NoState))
            {
                return Ok(());
            }
            return Err(ArchitectureStatePartitionError::EmptyPlan);
        }

        let mut rules = self.rules().iter().collect::<Vec<_>>();
        rules.sort_by_key(|rule| rule.layers().start);
        let mut frontier = 0usize;
        for rule in &rules {
            let layers = rule.layers();
            if layers.is_empty() {
                return Err(ArchitectureStatePartitionError::EmptyRange {
                    start: layers.start,
                    end: layers.end,
                });
            }
            if layers.end > complete.len() {
                return Err(ArchitectureStatePartitionError::RangeOutOfBounds {
                    start: layers.start,
                    end: layers.end,
                    layers: complete.len(),
                });
            }
            if layers.start < frontier {
                return Err(ArchitectureStatePartitionError::OverlappingRange {
                    start: layers.start,
                    frontier,
                });
            }
            if layers.start > frontier {
                return Err(ArchitectureStatePartitionError::UnassignedLayer { layer: frontier });
            }
            if let ArchitectureStatePlacement::GroupUnits { group } = rule.placement() {
                let units = units
                    .group_range(group)
                    .ok_or(ArchitectureStatePartitionError::UnknownGroup { group })?
                    .len();
                if layers.len() != units {
                    return Err(ArchitectureStatePartitionError::GroupLengthMismatch {
                        group,
                        start: layers.start,
                        end: layers.end,
                        units,
                    });
                }
            }
            if let ArchitectureStatePlacement::Invocation { address } = rule.placement() {
                if units.group_range(address.group()).is_none() {
                    return Err(ArchitectureStatePartitionError::UnknownGroup {
                        group: address.group(),
                    });
                }
                if units.ordinal(address.group(), address.index()).is_none() {
                    return Err(ArchitectureStatePartitionError::UnknownInvocation {
                        group: address.group(),
                        global_unit: address.index(),
                    });
                }
            }
            frontier = layers.end;
        }
        if frontier != complete.len() {
            return Err(ArchitectureStatePartitionError::UnassignedLayer { layer: frontier });
        }

        Ok(())
    }

    fn group_state_ranges(
        &self,
        units: &ExecutionUnitLayout,
        group: usize,
        owned: Range<usize>,
    ) -> Result<Vec<Range<usize>>, ArchitectureStatePartitionError> {
        let complete = units
            .group_range(group)
            .ok_or(ArchitectureStatePartitionError::UnknownGroup { group })?;
        let ordinals = units
            .contiguous_ordinals(group, owned.clone())
            .ok_or_else(|| {
                ArchitectureStatePartitionError::InvalidLayout(format!(
                    "group {group} owned units {owned:?} are not a declared contiguous range"
                ))
            })?;
        let mut selected = Vec::new();
        for rule in self.rules() {
            let layers = rule.layers();
            match rule.placement() {
                ArchitectureStatePlacement::GroupUnits { group: owner } if owner == group => {
                    if layers.len() != complete.len() {
                        return Err(ArchitectureStatePartitionError::GroupLengthMismatch {
                            group,
                            start: layers.start,
                            end: layers.end,
                            units: complete.len(),
                        });
                    }
                    selected.push(
                        layers.start + (ordinals.start - complete.start)
                            ..layers.start + (ordinals.end - complete.start),
                    );
                }
                ArchitectureStatePlacement::Invocation { address } if address.group() == group => {
                    if units.ordinal(group, address.index()).is_none() {
                        return Err(ArchitectureStatePartitionError::UnknownInvocation {
                            group,
                            global_unit: address.index(),
                        });
                    }
                    if owned.contains(&address.index()) {
                        selected.push(layers);
                    }
                }
                _ => {}
            }
        }
        Ok(selected)
    }

    /// Projects one group's selected state span independently of its unit count.
    /// Full plan validation establishes coverage; global-ordinal packing retains
    /// unowned gaps as `NoState` entries in the selected partition layout.
    pub fn group_state_range(
        &self,
        units: &ExecutionUnitLayout,
        group: usize,
        owned: Range<usize>,
    ) -> Result<Option<Range<usize>>, ArchitectureStatePartitionError> {
        Ok(self
            .group_state_ranges(units, group, owned)?
            .into_iter()
            .reduce(|span, range| span.start.min(range.start)..span.end.max(range.end)))
    }

    /// Resolves exact rank-local state from global unit identities. Unit-aligned
    /// state slots follow declared invocation order, independently of global unit IDs.
    /// Global-ordinal packing keeps unowned state gaps stateless.
    pub fn resolve_partition(
        &self,
        complete: &StateLayout,
        units: &ExecutionUnitLayout,
        owned_units: impl Fn(usize) -> Option<Range<usize>>,
        owns_output: bool,
    ) -> Result<Option<PartitionState>, ArchitectureStatePartitionError> {
        self.validate(complete, units)?;
        if self.rules().is_empty() {
            return Ok(None);
        }
        let groups = self
            .rules()
            .iter()
            .filter_map(|rule| match rule.placement() {
                ArchitectureStatePlacement::GroupUnits { group } => Some(group),
                ArchitectureStatePlacement::Invocation { address } => Some(address.group()),
                ArchitectureStatePlacement::OutputOwner => None,
            })
            .collect::<BTreeSet<_>>();
        let mut selected = Vec::new();
        for group in groups {
            if let Some(owned) = owned_units(group) {
                selected.extend(self.group_state_ranges(units, group, owned)?);
            }
        }
        if owns_output {
            selected.extend(
                self.rules()
                    .iter()
                    .filter(|rule| rule.placement() == ArchitectureStatePlacement::OutputOwner)
                    .map(|rule| rule.layers()),
            );
        }
        if selected.is_empty() {
            return Ok(None);
        }
        selected.sort_by_key(|layers| layers.start);
        let start = selected[0].start;
        let mut ranges = vec![selected[0].clone()];
        for layers in selected.into_iter().skip(1) {
            let previous = ranges.last_mut().expect("nonempty state selection");
            if layers.start != previous.end
                && self.packing() == crate::StatePartitionPacking::Contiguous
            {
                return Err(ArchitectureStatePartitionError::DiscontiguousSelection {
                    frontier: previous.end,
                    start: layers.start,
                });
            }
            if layers.start == previous.end {
                previous.end = layers.end;
            } else {
                ranges.push(layers);
            }
        }
        let end = ranges.last().expect("nonempty state selection").end;
        let layout = complete
            .selected_slice(start..end, &ranges)
            .map_err(|error| ArchitectureStatePartitionError::InvalidLayout(error.to_string()))?;
        Ok(Some(PartitionState {
            layout,
            global_layers: start..end,
            selected_layers: ranges,
        }))
    }
}

impl PartitionState {
    /// Attaches a local state layout at one architecture-global layer offset.
    pub fn new(
        layout: StateLayout,
        global_layer_offset: usize,
    ) -> Result<Self, ArchitecturePartitionError> {
        let end = global_layer_offset.checked_add(layout.len()).ok_or(
            ArchitecturePartitionError::StateOffsetOverflow {
                offset: global_layer_offset,
                layers: layout.len(),
            },
        )?;
        Ok(Self {
            layout,
            global_layers: global_layer_offset..end,
            selected_layers: vec![global_layer_offset..end],
        })
    }

    /// Returns the exact rank-local state layout.
    pub const fn layout(&self) -> &StateLayout {
        &self.layout
    }

    /// Returns the first architecture-global layer represented by the layout.
    pub const fn global_layer_offset(&self) -> usize {
        self.global_layers.start
    }

    /// Returns the represented global span, including any stateless unowned gaps.
    pub fn global_layers(&self) -> Range<usize> {
        self.global_layers.clone()
    }

    /// Exact architecture-global slots selected by this coordinate. Adjacent
    /// selections are coalesced; gaps in the represented span remain `NoState`.
    pub fn selected_global_layers(&self) -> &[Range<usize>] {
        &self.selected_layers
    }

    /// Whether this coordinate owns the supplied architecture-global state slot.
    pub fn owns_global_layer(&self, layer: usize) -> bool {
        self.selected_layers
            .iter()
            .any(|range| range.contains(&layer))
    }

    /// Projects the full architecture layout using the retained exact ownership.
    /// Families use this when checking cache identity and local construction.
    pub fn selected_layout(
        &self,
        complete: &StateLayout,
    ) -> Result<StateLayout, crate::StateError> {
        complete.selected_slice(self.global_layers.clone(), &self.selected_layers)
    }

    /// Derives prompt-cache identity from this canonical state partition.
    pub fn prompt_cache_identity<B, M>(
        &self,
        architecture: &M,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_core::cache::PromptCacheModelIdentity, ArchitecturePartitionError>
    where
        B: eredu_nn::NeuralBackend,
        M: crate::ArchitectureParameters<B>,
        M::DefinitionError: std::fmt::Display,
    {
        architecture
            .state_identity(self, topology)
            .map_err(|error| ArchitecturePartitionError::ArchitectureState(error.to_string()))?
            .prompt_cache_identity(self.layout())
            .map_err(|error| ArchitecturePartitionError::PromptCacheIdentity(error.to_string()))
    }
}

/// One validated architecture group and its group-local global unit range.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PartitionGroup {
    group: ExecutionGroupId,
    group_index: usize,
    global_units: Range<usize>,
    ordinals: Range<usize>,
}

impl PartitionGroup {
    /// Returns the canonical execution-group identity.
    pub const fn group(&self) -> &ExecutionGroupId {
        &self.group
    }

    /// Returns the canonical execution-group slot.
    pub const fn group_index(&self) -> usize {
        self.group_index
    }

    /// Returns owned unit indices in the architecture group's global index space.
    pub fn global_units(&self) -> Range<usize> {
        self.global_units.clone()
    }

    /// Returns whether the range contains one group-local global unit index.
    pub fn contains(&self, global_unit: usize) -> bool {
        self.global_units.contains(&global_unit)
    }
}

/// Complete backend-neutral realization of one rank's architecture ownership.
///
/// `G` is family-owned local construction geometry. `A` is the family-owned
/// primary and auxiliary wire schema carried by the partition realization.
#[derive(Debug, Clone)]
pub struct ArchitecturePartition<G, A> {
    graph: ExecutionGraph,
    unit_layout: ExecutionUnitLayout,
    invocations: crate::ExecutionUnitAddressMap,
    groups: Vec<PartitionGroup>,
    ownership: PartitionOwnership,
    state: Option<PartitionState>,
    local_geometry: G,
    boundary_schema: A,
    parameter_bindings: Vec<OwnedParameterGroupSpec>,
}

impl<G, A> ArchitecturePartition<G, A> {
    /// Creates a partition from the topology declared by one concrete neutral
    /// architecture.
    ///
    /// This constructor derives the graph and unit layout from `architecture`,
    /// preventing a backend realization from publishing a parallel topology
    /// that merely resembles, but is not the canonical topology of, the
    /// architecture it will execute. Pre-allocation selection should use
    /// [`Self::from_description`] with the architecture-authored declaration.
    pub fn from_architecture<B, S, M, N>(
        architecture: &M,
        group_ranges: impl IntoIterator<Item = (N, Range<usize>)>,
        ownership: PartitionOwnership,
        local_geometry: G,
        boundary_schema: A,
        parameters: &ArchitectureParameterDescription,
    ) -> Result<Self, ArchitecturePartitionError>
    where
        B: eredu_nn::NeuralBackend,
        S: crate::RuntimeState<B>,
        M: crate::LayeredArchitecture<B, S>,
        M::Error: std::fmt::Display,
        N: Into<String>,
        A: ArchitectureBoundary,
    {
        parameters.validate_architecture::<B, S, M>(architecture)?;
        let complete_state = architecture
            .state_layout()
            .map_err(|error| ArchitecturePartitionError::ArchitectureState(error.to_string()))?;
        let plan = architecture.state_partition_plan(&complete_state);
        Self::from_description(
            parameters,
            group_ranges,
            ownership,
            &complete_state,
            &plan,
            local_geometry,
            boundary_schema,
        )
    }

    /// Creates an authoritative partition from architecture-authored static declarations.
    ///
    /// This path exists for pre-materialization selection: it validates and consumes the
    /// canonical graph/unit declaration, state plan, boundary schema, local geometry, and
    /// parameter topology without constructing a backend module merely to rediscover those
    /// facts. Concrete backends must not synthesize any of these inputs.
    pub fn from_description<N>(
        parameters: &ArchitectureParameterDescription,
        group_ranges: impl IntoIterator<Item = (N, Range<usize>)>,
        ownership: PartitionOwnership,
        complete_state: &StateLayout,
        state_plan: &ArchitectureStatePartitionPlan,
        local_geometry: G,
        boundary_schema: A,
    ) -> Result<Self, ArchitecturePartitionError>
    where
        N: Into<String>,
        A: ArchitectureBoundary,
    {
        boundary_schema.wire_schema()?;
        let mut partition = Self::select_topology(
            parameters,
            group_ranges,
            ownership,
            local_geometry,
            boundary_schema,
        )?;
        partition.state = partition
            .resolve_state_partition(complete_state, state_plan)
            .map_err(|error| ArchitecturePartitionError::ArchitectureState(error.to_string()))?;
        partition.parameter_bindings = parameters.select_owned(&partition);
        Ok(partition)
    }

    /// Selects rank-local topology from one validated parameter declaration.
    fn select_topology<S>(
        parameters: &ArchitectureParameterDescription,
        group_ranges: impl IntoIterator<Item = (S, Range<usize>)>,
        ownership: PartitionOwnership,
        local_geometry: G,
        boundary_schema: A,
    ) -> Result<Self, ArchitecturePartitionError>
    where
        S: Into<String>,
    {
        let graph = parameters.graph();
        let unit_layout = parameters.unit_layout();
        let mut seen_groups = BTreeSet::new();
        let mut groups = Vec::new();
        for (group, global_units) in group_ranges {
            let group = group.into();
            let group_index = graph
                .groups()
                .iter()
                .position(|candidate| candidate.id() == group)
                .ok_or_else(|| ArchitecturePartitionError::UnknownGroup(group.clone()))?;
            if !seen_groups.insert(group.clone()) {
                return Err(ArchitecturePartitionError::DuplicateGroup(group));
            }
            if global_units.is_empty() {
                return Err(ArchitecturePartitionError::EmptyGroupRange { group });
            }
            let available = unit_layout
                .group_range(group_index)
                .expect("canonical layout contains every graph group")
                .len();
            let ordinals = unit_layout
                .contiguous_ordinals(group_index, global_units.clone())
                .ok_or_else(|| ArchitecturePartitionError::GroupRangeOutOfBounds {
                    group: group.clone(),
                    start: global_units.start,
                    end: global_units.end,
                    available,
                })?;
            groups.push(PartitionGroup {
                group: unit_layout
                    .group_id(group_index)
                    .expect("canonical layout contains every graph group identity")
                    .clone(),
                group_index,
                global_units,
                ordinals,
            });
        }
        groups.sort_by_key(PartitionGroup::group_index);

        let invocations = parameters
            .invocations()
            .select_invocations(groups.iter().flat_map(|group| {
                group
                    .ordinals
                    .clone()
                    .map(|ordinal| unit_layout.address(ordinal).expect("owned invocation"))
            }))?;
        Ok(Self {
            graph: graph.clone(),
            unit_layout: unit_layout.clone(),
            invocations,
            groups,
            ownership,
            state: None,
            local_geometry,
            boundary_schema,
            parameter_bindings: Vec::new(),
        })
    }

    /// Returns the canonical architecture execution graph.
    pub const fn graph(&self) -> &ExecutionGraph {
        &self.graph
    }

    /// Returns the canonical complete execution-unit layout.
    pub const fn unit_layout(&self) -> &ExecutionUnitLayout {
        &self.unit_layout
    }

    /// Selected logical invocations and their canonical parameter owners, including replicas.
    pub fn invocations(&self) -> &crate::ExecutionUnitAddressMap {
        &self.invocations
    }

    /// Returns groups and group-local global unit ranges owned by this rank.
    pub fn groups(&self) -> &[PartitionGroup] {
        &self.groups
    }

    /// Traverses rank-owned execution units in canonical architecture order.
    pub fn units(&self) -> impl Iterator<Item = crate::ExecutionUnitAddress> + '_ {
        self.groups.iter().flat_map(move |owned| {
            owned.ordinals.clone().map(move |ordinal| {
                self.unit_layout
                    .address(ordinal)
                    .expect("validated partition invocation")
            })
        })
    }

    /// Returns whether this rank owns one group-local global unit.
    pub fn owns_unit(&self, group: &str, global_unit: usize) -> bool {
        self.groups
            .iter()
            .any(|owned| owned.group.as_str() == group && owned.contains(global_unit))
    }

    /// Returns input, output, and static-module ownership.
    pub const fn ownership(&self) -> &PartitionOwnership {
        &self.ownership
    }

    /// Resolves the unique canonical storage owner of a selected physical target.
    /// Logical consumers and replicated storage retain their declared owner identity.
    pub fn parameter_owner(&self, target: &str) -> Option<&ParameterGroupOwner> {
        self.parameter_bindings.iter().find_map(|binding| {
            binding
                .members()
                .iter()
                .any(|member| member.target() == target)
                .then(|| binding.owner())
        })
    }

    /// Checks that conversion resources keep the selected outputs' canonical owners.
    /// Targets are supplied by admitted materialization tasks; this neither changes
    /// ownership nor combines independent storage resources into one residency unit.
    pub fn validate_transform_parameter_owners<'a, H, W>(
        &self,
        source: &ArchitecturePartition<H, W>,
        targets: impl IntoIterator<Item = &'a str>,
    ) -> Result<(), TransformParameterOwnershipError> {
        for name in targets {
            let target = self
                .parameter_owner(name)
                .ok_or_else(|| TransformParameterOwnershipError::MissingTarget(name.into()))?;
            let source = source
                .parameter_owner(name)
                .ok_or_else(|| TransformParameterOwnershipError::MissingSource(name.into()))?;
            if source != target {
                return Err(TransformParameterOwnershipError::Changed(name.into()));
            }
        }
        Ok(())
    }

    /// Returns rank-local state geometry when this partition owns mutable state.
    pub const fn state(&self) -> Option<&PartitionState> {
        self.state.as_ref()
    }

    /// Derives prompt-cache identity from this partition's canonical state.
    pub fn prompt_cache_identity<B, M>(
        &self,
        architecture: &M,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_core::cache::PromptCacheModelIdentity, ArchitecturePartitionError>
    where
        B: eredu_nn::NeuralBackend,
        M: crate::ArchitectureParameters<B>,
        M::DefinitionError: std::fmt::Display,
    {
        let state = self
            .state()
            .ok_or(ArchitecturePartitionError::MissingArchitectureState)?;
        state.prompt_cache_identity::<B, M>(architecture, topology)
    }

    /// Resolves the architecture-authored state plan for this realized partition.
    ///
    /// Contiguous packing requires adjacent selected ranges. Global-ordinal
    /// preservation instead represents unowned gaps as stateless slots.
    pub fn resolve_state_partition(
        &self,
        complete: &StateLayout,
        plan: &ArchitectureStatePartitionPlan,
    ) -> Result<Option<PartitionState>, ArchitectureStatePartitionError> {
        plan.resolve_partition(
            complete,
            &self.unit_layout,
            |group| {
                self.groups
                    .iter()
                    .find(|owned| owned.group_index() == group)
                    .map(PartitionGroup::global_units)
            },
            self.ownership.owns_output(),
        )
    }

    /// Returns family-owned rank-local construction geometry.
    pub const fn local_geometry(&self) -> &G {
        &self.local_geometry
    }

    /// Returns the family-owned primary and auxiliary boundary schema.
    pub const fn boundary_schema(&self) -> &A {
        &self.boundary_schema
    }

    /// Mutably returns the family-owned primary and auxiliary boundary schema.
    pub fn boundary_schema_mut(&mut self) -> &mut A {
        &mut self.boundary_schema
    }

    /// Returns neutral semantic parameter bindings owned by this rank.
    pub fn parameter_bindings(&self) -> &[OwnedParameterGroupSpec] {
        &self.parameter_bindings
    }

    /// Returns the exact neutral groups assigned to one architecture owner.
    pub fn parameter_bindings_for_owner<'a>(
        &'a self,
        owner: &'a ParameterGroupOwner,
    ) -> impl Iterator<Item = &'a ParameterGroupSpec> + 'a {
        self.parameter_bindings
            .iter()
            .filter(move |binding| binding.owner() == owner)
            .map(OwnedParameterGroupSpec::group)
    }

    /// Proves that this partition still describes the supplied concrete
    /// neutral architecture.
    ///
    /// Loaders may use this when a partition crosses a backend boundary or is
    /// restored from a prepared plan. Both dependency edges and exact unit
    /// counts are compared; matching group names alone are insufficient.
    pub fn validate_architecture<B, S, M>(
        &self,
        architecture: &M,
    ) -> Result<(), ArchitecturePartitionError>
    where
        B: eredu_nn::NeuralBackend,
        S: crate::RuntimeState<B>,
        M: crate::LayeredArchitecture<B, S>,
        M::Error: std::fmt::Display,
    {
        let (graph, unit_layout) = canonical_architecture_layout::<B, S, M>(architecture)?;
        if graph != self.graph {
            return Err(ArchitecturePartitionError::ArchitectureGraphMismatch);
        }
        if unit_layout != self.unit_layout {
            return Err(ArchitecturePartitionError::ArchitectureUnitLayoutMismatch);
        }
        let invocations = architecture
            .execution_unit_storage_map()
            .map_err(|error| ArchitecturePartitionError::ArchitectureTopology(error.to_string()))?;
        if invocations.select_invocations(self.units())? != self.invocations {
            return Err(ArchitecturePartitionError::ArchitectureStorageMapMismatch);
        }
        Ok(())
    }
}

/// Validated execution metadata for one rank-local layered partition.
///
/// This driver is the single owner of partition input/output checks, canonical
/// storage and state ranges, execution-group setup/completion, and final output
/// projection. Concrete backends retain only state storage and unit residency.
#[derive(Debug, Clone)]
pub struct LayeredPartitionDriver {
    group: usize,
    range: Range<usize>,
    state_layout: Option<StateLayout>,
    owns_input: bool,
    owns_output: bool,
}

impl LayeredPartitionDriver {
    /// Validates a canonical partition against its concrete unit storage.
    pub fn new<G, A>(
        partition: &ArchitecturePartition<G, A>,
        group_index: usize,
        storage_range: Range<usize>,
    ) -> Result<Self, LayeredPartitionError> {
        Self::new_with_state_ownership(partition, group_index, storage_range, true)
    }

    /// Validates a partition while explicitly declaring whether this group owns state slots.
    ///
    /// Architecture selection must pass `false` for parameter-only composite roots. A rank can
    /// still own decoder state for another local group without falsely comparing that state range
    /// with this group's unrelated unit indices.
    pub fn new_with_state_ownership<G, A>(
        partition: &ArchitecturePartition<G, A>,
        group_index: usize,
        storage_range: Range<usize>,
        group_owns_state: bool,
    ) -> Result<Self, LayeredPartitionError> {
        let state_range = group_owns_state.then(|| storage_range.clone());
        Self::new_with_state_range(partition, group_index, storage_range, state_range)
    }

    /// Validates architecture-authored state addresses independently of unit indices.
    /// Stateful encoder units can follow the decoder in the global state layout.
    pub fn new_with_state_range<G, A>(
        partition: &ArchitecturePartition<G, A>,
        group_index: usize,
        storage_range: Range<usize>,
        state_range: Option<Range<usize>>,
    ) -> Result<Self, LayeredPartitionError> {
        let group = partition
            .groups()
            .iter()
            .find(|group| group.group_index() == group_index)
            .ok_or(LayeredPartitionError::GroupNotOwned { group: group_index })?;
        let range = group.global_units();
        if storage_range != range {
            return Err(LayeredPartitionError::StorageRange {
                storage: storage_range,
                partition: range,
            });
        }
        if let Some(state_range) = &state_range {
            let state = partition
                .state()
                .ok_or(LayeredPartitionError::MissingState)?;
            if state_range.is_empty()
                || state.global_layers().start > state_range.start
                || state.global_layers().end < state_range.end
                || state_range.clone().any(|layer| {
                    !state.owns_global_layer(layer)
                        && !matches!(
                            state
                                .layout()
                                .layers()
                                .get(layer - state.global_layer_offset()),
                            Some(eredu_core::cache::LayerCachePolicy::NoState)
                        )
                })
            {
                return Err(LayeredPartitionError::StateRange {
                    state: state.global_layers(),
                    partition: state_range.clone(),
                });
            }
        }
        Ok(Self {
            group: group.group_index(),
            range,
            state_layout: state_range
                .is_some()
                .then(|| partition.state().map(|state| state.layout().clone()))
                .flatten(),
            owns_input: partition.ownership().owns_input(),
            owns_output: partition.ownership().owns_output(),
        })
    }

    /// Returns the canonical group-local global unit range.
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }

    /// Returns the canonical architecture execution-group slot.
    pub const fn group_index(&self) -> usize {
        self.group
    }

    /// Returns state geometry for a driver created through the strict stateful constructor.
    pub fn state_layout(&self) -> &StateLayout {
        self.state_layout
            .as_ref()
            .expect("state_layout requires a state-owning layered partition driver")
    }

    /// Returns rank-local state geometry, or `None` for stateless roots and ranks.
    pub const fn optional_state_layout(&self) -> Option<&StateLayout> {
        self.state_layout.as_ref()
    }

    /// Returns whether this partition receives request ingress directly.
    pub const fn owns_input(&self) -> bool {
        self.owns_input
    }

    /// Returns whether this partition owns architecture output projection.
    pub const fn owns_output(&self) -> bool {
        self.owns_output
    }

    /// Validates input form against architecture boundary ownership.
    pub fn input<'a, T, A>(
        &self,
        input: LayeredPartitionInput<'a, T, A>,
    ) -> Result<LayeredPartitionInput<'a, T, A>, LayeredPartitionError> {
        match (&input, self.owns_input) {
            (LayeredPartitionInput::Tokens(_), true)
            | (LayeredPartitionInput::Hidden { .. }, _) => Ok(input),
            (LayeredPartitionInput::Tokens(_), false) => {
                Err(LayeredPartitionError::TokensOnNonInputOwner)
            }
        }
    }

    /// Transports this partition's realized boundary through the selected
    /// opaque collective group.
    ///
    /// Keeping the operation on the validated driver makes boundary movement
    /// part of partition execution rather than an unrelated backend call.
    pub fn exchange_boundary<B>(
        &self,
        value: B::Tensor,
        group: &B::Group,
        executor: &B::Executor,
    ) -> Result<B::Tensor, B::CollectiveError>
    where
        B: crate::CollectiveBackend,
    {
        B::all_to_all(value, group, executor)
    }

    /// Prepares the partition and starts its canonical execution group.
    #[allow(
        clippy::too_many_arguments,
        clippy::type_complexity,
        reason = "the result preserves the concrete architecture error without erased dispatch"
    )]
    pub fn begin<'a, B, S, M>(
        &self,
        architecture: &mut M,
        input: LayeredPartitionInput<
            'a,
            B::Tensor,
            <M::Boundary as ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        mask: Option<&B::Tensor>,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<
        LayeredForwardState<B::Tensor, M::ForwardContext>,
        LayeredPartitionBeginError<M::Error>,
    >
    where
        B: eredu_nn::NeuralBackend,
        S: RuntimeState<B>,
        M: PartitionedLayeredArchitecture<B, S>,
        M::Error: std::fmt::Display,
    {
        self.begin_with_optional_observer(architecture, input, mask, state, parallel, context, None)
    }

    /// Prepares the partition with internal input observations before group entry.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn begin_observed<'a, B, S, M, O>(
        &self,
        architecture: &mut M,
        input: LayeredPartitionInput<
            'a,
            B::Tensor,
            <M::Boundary as ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        mask: Option<&B::Tensor>,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        LayeredForwardState<B::Tensor, M::ForwardContext>,
        LayeredPartitionBeginError<M::Error>,
    >
    where
        B: eredu_nn::NeuralBackend,
        S: RuntimeState<B>,
        M: PartitionedLayeredArchitecture<B, S>,
        M::Error: std::fmt::Display,
        O: crate::ActivationObserver<B::Tensor, M::Error> + ?Sized,
    {
        let mut observer = crate::BorrowedActivationObserver(observer);
        self.begin_with_optional_observer(
            architecture,
            input,
            mask,
            state,
            parallel,
            context,
            Some(&mut observer),
        )
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn begin_with_optional_observer<'a, B, S, M>(
        &self,
        architecture: &mut M,
        input: LayeredPartitionInput<
            'a,
            B::Tensor,
            <M::Boundary as ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        mask: Option<&B::Tensor>,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: Option<&mut dyn crate::ActivationObserver<B::Tensor, M::Error>>,
    ) -> Result<
        LayeredForwardState<B::Tensor, M::ForwardContext>,
        LayeredPartitionBeginError<M::Error>,
    >
    where
        B: eredu_nn::NeuralBackend,
        S: RuntimeState<B>,
        M: PartitionedLayeredArchitecture<B, S>,
        M::Error: std::fmt::Display,
    {
        let expected = self
            .state_layout
            .as_ref()
            .ok_or(LayeredPartitionBeginError::MissingState { group: self.group })?;
        self.begin_with_preparation(
            architecture,
            state,
            parallel,
            context,
            |architecture, state, _| match observer {
                Some(observer) => architecture.begin_partition_observed(
                    input, mask, state, expected, 0, parallel, context, observer,
                ),
                None => match parallel {
                    Some(parallel) => architecture.begin_partition_parallel(
                        input, mask, state, expected, 0, parallel, context,
                    ),
                    None => architecture.begin_partition(input, mask, state, expected, 0, context),
                },
            },
        )
    }

    /// Prepares architecture-owned request ingress and enters the selected group once.
    /// The callback receives the exact local state layout for a state-owning group,
    /// or `None` for a stateless group. Local state ordinals start at zero.
    pub fn begin_with_preparation<B, S, M, F>(
        &self,
        architecture: &mut M,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        prepare: F,
    ) -> Result<
        LayeredForwardState<B::Tensor, M::ForwardContext>,
        LayeredPartitionBeginError<M::Error>,
    >
    where
        B: eredu_nn::NeuralBackend,
        S: RuntimeState<B>,
        M: PartitionedLayeredArchitecture<B, S>,
        M::Error: std::fmt::Display,
        F: FnOnce(
            &mut M,
            &mut S,
            Option<&StateLayout>,
        ) -> Result<LayeredForwardState<B::Tensor, M::ForwardContext>, M::Error>,
    {
        let mut forward = prepare(architecture, state, self.state_layout.as_ref())
            .map_err(LayeredPartitionBeginError::Architecture)?;
        forward.hidden = architecture
            .enter_partition_group(
                self.group,
                &forward.hidden,
                state,
                &mut forward.context,
                parallel,
                context,
            )
            .map_err(LayeredPartitionBeginError::Architecture)?;
        Ok(forward)
    }

    /// Completes the canonical group and applies output projection only on its owner.
    #[allow(
        clippy::too_many_arguments,
        clippy::type_complexity,
        reason = "the signature exposes the backend and architecture boundary types explicitly"
    )]
    pub fn finish<B, S, M>(
        &self,
        architecture: &mut M,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut M::ForwardContext,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<
        LayeredPartitionOutput<
            B::Tensor,
            <M::Boundary as ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        M::Error,
    >
    where
        B: eredu_nn::NeuralBackend,
        S: RuntimeState<B>,
        M: PartitionedLayeredArchitecture<B, S>,
    {
        self.finish_with_optional_observer(
            architecture,
            hidden,
            state,
            forward,
            parallel,
            context,
            None,
        )
    }

    /// Completes the group and observes readout only at the selected output owner.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn finish_observed<B, S, M, O>(
        &self,
        architecture: &mut M,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut M::ForwardContext,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        LayeredPartitionOutput<
            B::Tensor,
            <M::Boundary as ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        M::Error,
    >
    where
        B: eredu_nn::NeuralBackend,
        S: RuntimeState<B>,
        M: PartitionedLayeredArchitecture<B, S>,
        O: crate::ActivationObserver<B::Tensor, M::Error> + ?Sized,
    {
        let mut observer = crate::BorrowedActivationObserver(observer);
        self.finish_with_optional_observer(
            architecture,
            hidden,
            state,
            forward,
            parallel,
            context,
            Some(&mut observer),
        )
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn finish_with_optional_observer<B, S, M>(
        &self,
        architecture: &mut M,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut M::ForwardContext,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: Option<&mut dyn crate::ActivationObserver<B::Tensor, M::Error>>,
    ) -> Result<
        LayeredPartitionOutput<
            B::Tensor,
            <M::Boundary as ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        M::Error,
    >
    where
        B: eredu_nn::NeuralBackend,
        S: RuntimeState<B>,
        M: PartitionedLayeredArchitecture<B, S>,
    {
        let hidden = architecture
            .leave_partition_group(self.group, hidden, state, forward, parallel, context)?;
        match observer {
            Some(observer) => architecture.finish_partition_observed(
                &hidden,
                state,
                forward,
                self.owns_output,
                parallel,
                context,
                observer,
            ),
            None => architecture.finish_partition(
                &hidden,
                state,
                forward,
                self.owns_output,
                parallel,
                context,
            ),
        }
    }
}

/// Failure to enter one concrete rank-local partition group.
#[derive(Debug, thiserror::Error)]
pub enum LayeredPartitionBeginError<E>
where
    E: std::fmt::Display,
{
    /// This group was declared stateless and cannot use the stateful partition entry API.
    #[error("stateless partition group {group} requires an architecture stateless entry strategy")]
    MissingState {
        /// Canonical architecture group slot.
        group: usize,
    },
    /// Architecture-owned partition entry failed.
    #[error("partition architecture entry failed: {0}")]
    Architecture(#[source] E),
}

/// Invalid concrete realization or boundary use of a layered partition.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum LayeredPartitionError {
    /// The selected architecture execution group is not owned by this partition.
    #[error("layered partition does not own execution group {group}")]
    GroupNotOwned {
        /// Canonical architecture group index.
        group: usize,
    },
    /// Concrete unit storage does not match canonical ownership.
    #[error("partition storage range {storage:?} disagrees with canonical range {partition:?}")]
    StorageRange {
        /// Concrete backend storage range.
        storage: Range<usize>,
        /// Canonical partition range.
        partition: Range<usize>,
    },
    /// Partition omitted mutable state geometry.
    #[error("layered partition has no runtime state")]
    MissingState,
    /// Mutable state geometry does not match canonical unit ownership.
    #[error("partition state range {state:?} disagrees with canonical range {partition:?}")]
    StateRange {
        /// Architecture-global state range.
        state: Range<usize>,
        /// Canonical partition range.
        partition: Range<usize>,
    },
    /// Token ids were supplied after the architecture input boundary.
    #[error("non-input partition received token ids")]
    TokensOnNonInputOwner,
}

fn canonical_architecture_layout<B, S, M>(
    architecture: &M,
) -> Result<(ExecutionGraph, ExecutionUnitLayout), ArchitecturePartitionError>
where
    B: eredu_nn::NeuralBackend,
    S: crate::RuntimeState<B>,
    M: crate::LayeredArchitecture<B, S>,
    M::Error: std::fmt::Display,
{
    let graph = architecture
        .execution_graph()
        .map_err(|error| ArchitecturePartitionError::ArchitectureTopology(error.to_string()))?;
    let primary = architecture.primary_execution_group();
    let primary_index = graph.group_index(primary).ok_or_else(|| {
        ArchitecturePartitionError::ArchitectureTopology(format!(
            "primary execution group {primary:?} is not present in the canonical graph"
        ))
    })?;
    let primary_transport = architecture.group_transport(primary_index);
    if !matches!(
        primary_transport.placement,
        crate::ArchitectureGroupPlacement::Pipeline
            | crate::ArchitectureGroupPlacement::PipelineSequence
    ) {
        return Err(ArchitecturePartitionError::ArchitectureTopology(format!(
            "primary execution group {primary:?} must have pipeline placement"
        )));
    }
    let mut declared_groups = BTreeSet::from([primary.to_owned()]);
    for prediction in architecture.prediction_execution_groups() {
        let prediction_index = graph.group_index(&prediction).ok_or_else(|| {
            ArchitecturePartitionError::ArchitectureTopology(format!(
                "prediction execution group {prediction:?} is not present in the canonical graph"
            ))
        })?;
        let prediction_transport = architecture.group_transport(prediction_index);
        if prediction_transport.kind != crate::ArchitectureGroupKind::Prediction
            || prediction_transport.placement != crate::ArchitectureGroupPlacement::OutputOwner
        {
            return Err(ArchitecturePartitionError::ArchitectureTopology(format!(
                "prediction execution group {prediction:?} must be an output-owner prediction"
            )));
        }
        if !declared_groups.insert(prediction.clone()) {
            return Err(ArchitecturePartitionError::ArchitectureTopology(format!(
                "execution group {prediction:?} is declared as a primary or prediction group more than once"
            )));
        }
    }
    let unit_layout = architecture
        .execution_unit_layout()
        .map_err(|error| ArchitecturePartitionError::ArchitectureTopology(error.to_string()))?;
    validate_canonical_layout(&graph, &unit_layout)?;
    let mut paths = BTreeSet::new();
    for group in 0..graph.groups().len() {
        for ordinal in unit_layout
            .group_range(group)
            .expect("canonical execution group")
        {
            let index = unit_layout
                .address(ordinal)
                .expect("canonical invocation")
                .index();
            let path = architecture.unit_path(group, index).map_err(|error| {
                ArchitecturePartitionError::ArchitectureTopology(error.to_string())
            })?;
            if path.trim().is_empty() {
                return Err(ArchitecturePartitionError::EmptyArchitectureUnitPath { group, index });
            }
            if !paths.insert(path.clone()) {
                return Err(ArchitecturePartitionError::DuplicateArchitectureUnitPath(
                    path,
                ));
            }
        }
    }

    Ok((graph, unit_layout))
}

fn validate_canonical_layout(
    graph: &ExecutionGraph,
    layout: &ExecutionUnitLayout,
) -> Result<(), ArchitecturePartitionError> {
    if graph.groups().len() != layout.group_count() {
        return Err(ArchitecturePartitionError::LayoutGroupCountMismatch {
            graph: graph.groups().len(),
            layout: layout.group_count(),
        });
    }
    for (index, group) in graph.groups().iter().enumerate() {
        let layout_group = layout
            .group_id(index)
            .expect("matching group counts provide every layout identity");
        if layout_group.as_str() != group.id() {
            return Err(ArchitecturePartitionError::LayoutGroupMismatch {
                index,
                graph: group.id().to_owned(),
                layout: layout_group.as_str().to_owned(),
            });
        }
    }
    Ok(())
}

/// Invalid backend-neutral architecture partition declaration.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ArchitecturePartitionError {
    /// Logical ownership cannot be projected onto the declared physical owners.
    #[error(transparent)]
    Invocations(#[from] crate::ExecutionUnitAddressMapError),
    /// Canonical physical ownership differs despite matching logical topology.
    #[error("architecture partition storage map differs from the neutral architecture")]
    ArchitectureStorageMapMismatch,

    /// The architecture supplied an invalid partition-boundary wire schema.
    #[error("invalid architecture partition boundary: {0}")]
    InvalidBoundary(#[from] ArchitectureBoundaryError),
    /// The neutral architecture could not declare a canonical graph, unit
    /// count, or unit path.
    #[error("neutral architecture topology is invalid: {0}")]
    ArchitectureTopology(String),
    /// The architecture could not declare or partition its mutable state.
    #[error("neutral architecture state is invalid: {0}")]
    ArchitectureState(String),
    /// The realized partition owns no mutable architecture state.
    #[error("architecture partition owns no mutable state")]
    MissingArchitectureState,
    /// The architecture state could not be converted to prompt-cache identity.
    #[error("architecture prompt-cache identity is invalid: {0}")]
    PromptCacheIdentity(String),
    /// A neutral architecture exposed an empty stable unit path.
    #[error("neutral architecture unit {group}:{index} has an empty path")]
    EmptyArchitectureUnitPath {
        /// Canonical execution-group slot.
        group: usize,
        /// Group-local unit index.
        index: usize,
    },
    /// Two canonical architecture units exposed the same stable path.
    #[error("neutral architecture repeats unit path {0:?}")]
    DuplicateArchitectureUnitPath(String),
    /// The partition dependency graph differs from the concrete architecture.
    #[error("architecture partition dependency graph differs from the neutral architecture")]
    ArchitectureGraphMismatch,
    /// The partition unit counts differ from the concrete architecture.
    #[error("architecture partition unit layout differs from the neutral architecture")]
    ArchitectureUnitLayoutMismatch,
    /// The graph and complete unit layout contain different group counts.
    #[error("execution graph contains {graph} groups but its unit layout contains {layout}")]
    LayoutGroupCountMismatch {
        /// Canonical graph group count.
        graph: usize,
        /// Unit-layout group count.
        layout: usize,
    },
    /// A unit-layout group identity differs from the graph at the same slot.
    #[error("execution group {index} is {graph:?} in the graph but {layout:?} in the unit layout")]
    LayoutGroupMismatch {
        /// Canonical group slot.
        index: usize,
        /// Graph identity.
        graph: String,
        /// Unit-layout identity.
        layout: String,
    },
    /// A rank-local unit range names no canonical architecture group.
    #[error("architecture partition names unknown execution group {0:?}")]
    UnknownGroup(String),
    /// A canonical architecture group was declared more than once.
    #[error("architecture partition repeats execution group {0:?}")]
    DuplicateGroup(String),
    /// A group owns no execution units.
    #[error("architecture partition declares an empty unit range for group {group:?}")]
    EmptyGroupRange {
        /// Canonical group identity.
        group: String,
    },
    /// A global unit range includes identities absent from the canonical layout.
    #[error(
        "architecture partition range {start}..{end} for group {group:?} is not contained in its {available} declared units"
    )]
    GroupRangeOutOfBounds {
        /// Canonical group identity.
        group: String,
        /// Invalid range start.
        start: usize,
        /// Invalid range end.
        end: usize,
        /// Canonical group unit count.
        available: usize,
    },
    /// A static ownership role is blank.
    #[error("architecture partition static role must not be empty")]
    EmptyStaticRole,
    /// A static ownership role was repeated.
    #[error("architecture partition repeats static role {0:?}")]
    DuplicateStaticRole(String),
    /// A local state layout cannot be placed in the global layer index space.
    #[error("state layer offset {offset} plus {layers} local layers overflowed usize")]
    StateOffsetOverflow {
        /// Requested global layer offset.
        offset: usize,
        /// Local state-layer count.
        layers: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MemberSharding, ParameterMemberSpec, ParameterRole};
    use eredu_core::{cache::LayerCachePolicy, LayerSchedule};

    #[derive(Debug, Clone, Eq, PartialEq)]
    struct Geometry(&'static str);

    #[derive(Debug, Clone, Eq, PartialEq)]
    struct Boundary {
        route: usize,
    }

    #[derive(Debug, Clone, Eq, PartialEq)]
    struct PairBoundary<T> {
        tokens: T,
        embedded: T,
    }

    #[derive(Debug, Clone, Copy)]
    struct PairBoundarySchema;

    impl ArchitectureBoundary for PairBoundarySchema {
        type Boundary<T> = PairBoundary<T>;

        const IDENTITY: &'static str = "fixture.target";

        fn primary_tensor_spec(&self) -> BoundaryTensorSpec {
            BoundaryTensorSpec::primary_activation(8)
        }

        fn auxiliary_tensor_specs(&self) -> Vec<BoundaryTensorSpec> {
            vec![
                BoundaryTensorSpec::new(
                    "tokens",
                    [
                        BoundaryTensorDimension::Batch,
                        BoundaryTensorDimension::Sequence,
                    ],
                    BoundaryTensorDtype::Uint32,
                ),
                BoundaryTensorSpec::new(
                    "embedded",
                    [
                        BoundaryTensorDimension::Batch,
                        BoundaryTensorDimension::Sequence,
                        BoundaryTensorDimension::Fixed(16),
                    ],
                    BoundaryTensorDtype::Activation,
                ),
            ]
        }

        fn encode<T>(
            &self,
            boundary: Self::Boundary<T>,
        ) -> Result<Vec<ArchitectureBoundaryValue<T>>, ArchitectureBoundaryError> {
            Ok(vec![
                ArchitectureBoundaryValue::new("tokens", boundary.tokens)?,
                ArchitectureBoundaryValue::new("embedded", boundary.embedded)?,
            ])
        }

        fn decode<T>(
            &self,
            mut tensors: Vec<T>,
        ) -> Result<Self::Boundary<T>, ArchitectureBoundaryError> {
            validate_boundary_tensor_count(self, &tensors)?;
            let embedded = tensors.pop().expect("validated embedded tensor");
            let tokens = tensors.pop().expect("validated token tensor");
            Ok(PairBoundary { tokens, embedded })
        }
    }

    fn graph() -> ExecutionGraph {
        ExecutionGraph::chain(["primary", "prediction"]).unwrap()
    }

    fn layout(graph: &ExecutionGraph) -> ExecutionUnitLayout {
        ExecutionUnitLayout::new(graph, [4, 3]).unwrap()
    }

    fn state_layout(layers: usize) -> StateLayout {
        StateLayout::new(
            LayerSchedule::new(layers, vec![LayerCachePolicy::NoState; layers]).unwrap(),
        )
        .unwrap()
    }

    fn parameter(logical: &str, target: &str) -> ParameterGroupSpec {
        ParameterGroupSpec::new(
            logical,
            ParameterRole::Replicated,
            [ParameterMemberSpec::new(
                target,
                vec![2, 2],
                MemberSharding::Replicated,
            )],
        )
        .unwrap()
    }

    fn identity_description(
        graph: &ExecutionGraph,
        layout: &ExecutionUnitLayout,
    ) -> ArchitectureParameterDescription {
        ArchitectureParameterDescription::from_owned(graph, layout, []).unwrap()
    }

    fn valid_partition() -> ArchitecturePartition<Geometry, Boundary> {
        let graph = graph();
        let layout = layout(&graph);
        ArchitecturePartition::select_topology(
            &identity_description(&graph, &layout),
            [("prediction", 0..2), ("primary", 1..4)],
            PartitionOwnership::new(true, false, ["embedding", "normalization"]).unwrap(),
            Geometry("local"),
            Boundary { route: 3 },
        )
        .map(|mut partition| {
            partition.state = Some(PartitionState::new(state_layout(2), 7).unwrap());
            partition.parameter_bindings = Vec::from([
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::static_role("embedding"),
                    parameter("model.embed_tokens", "model.embed_tokens.weight"),
                ),
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::execution_unit(
                        ExecutionGroupId::new("primary").unwrap(),
                        1,
                    ),
                    parameter("model.layers.1", "model.layers.1.weight"),
                ),
            ]);
            partition
        })
        .unwrap()
    }

    fn state_plan_partition(
        primary: Range<usize>,
        ownership: PartitionOwnership,
    ) -> ArchitecturePartition<(), ()> {
        let graph = graph();
        ArchitecturePartition::select_topology(
            &identity_description(&graph, &layout(&graph)),
            [("primary", primary)],
            ownership,
            (),
            (),
        )
        .unwrap()
    }

    #[test]
    fn empty_state_plan_admits_only_stateless_architectures() {
        let output = state_plan_partition(
            0..4,
            PartitionOwnership::new(true, true, std::iter::empty::<&str>()).unwrap(),
        );
        let plan = ArchitectureStatePartitionPlan::new([]);
        assert!(output
            .resolve_state_partition(&state_layout(1), &plan)
            .unwrap()
            .is_none());
        let stateful = StateLayout::new(
            LayerSchedule::new(
                1,
                vec![LayerCachePolicy::key_value(eredu_core::AttentionPolicy::Full, 1, 8).unwrap()],
            )
            .unwrap(),
        )
        .unwrap();
        assert!(matches!(
            output.resolve_state_partition(&stateful, &plan),
            Err(ArchitectureStatePartitionError::EmptyPlan)
        ));
    }

    #[test]
    fn architecture_state_plan_attaches_declared_tail_to_output_owner() {
        let complete = state_layout(6);
        let plan = ArchitectureStatePartitionPlan::new([
            crate::ArchitectureStatePartitionRule::group_units(0, 0..4),
            crate::ArchitectureStatePartitionRule::output_owner(4..6),
        ]);
        let interior = state_plan_partition(
            1..3,
            PartitionOwnership::new(false, false, std::iter::empty::<&str>()).unwrap(),
        );
        let output = state_plan_partition(
            3..4,
            PartitionOwnership::new(false, true, std::iter::empty::<&str>()).unwrap(),
        );

        assert_eq!(
            interior
                .resolve_state_partition(&complete, &plan)
                .unwrap()
                .unwrap()
                .global_layers(),
            1..3
        );
        assert_eq!(
            output
                .resolve_state_partition(&complete, &plan)
                .unwrap()
                .unwrap()
                .global_layers(),
            3..6
        );
    }

    #[test]
    fn architecture_state_plan_rejects_noncontiguous_local_state() {
        let complete = state_layout(6);
        let plan = ArchitectureStatePartitionPlan::new([
            crate::ArchitectureStatePartitionRule::output_owner(0..2),
            crate::ArchitectureStatePartitionRule::group_units(0, 2..6),
        ]);
        let output = state_plan_partition(
            3..4,
            PartitionOwnership::new(false, true, std::iter::empty::<&str>()).unwrap(),
        );

        assert_eq!(
            output.resolve_state_partition(&complete, &plan),
            Err(ArchitectureStatePartitionError::DiscontiguousSelection {
                frontier: 2,
                start: 5,
            })
        );
    }

    #[test]
    fn architecture_state_plan_rejects_cross_group_cold_ownership() {
        let complete = state_layout(5);
        let partition = state_plan_partition(
            0..2,
            PartitionOwnership::new(true, false, std::iter::empty::<&str>()).unwrap(),
        );
        let plan = ArchitectureStatePartitionPlan::new([
            crate::ArchitectureStatePartitionRule::group_units(0, 0..4),
            crate::ArchitectureStatePartitionRule::output_owner(4..5),
        ]);
        assert!(matches!(
            plan.resolve_partition(&complete, &partition.unit_layout, |_| Some(0..5), false),
            Err(ArchitectureStatePartitionError::InvalidLayout(_))
        ));
    }

    #[test]
    fn architecture_state_plan_preserves_global_keys_without_allocating_unowned_state() {
        use eredu_core::cache::{
            LayerCachePolicy, StateResidencyClass, StateTensorDimension as Dim, StateTensorDtype,
            StateTensorPolicy, StateTensorPresence, StateTensorRole,
        };
        let audio = LayerCachePolicy::FixedState {
            tensors: vec![StateTensorPolicy {
                role: StateTensorRole::Convolution { slot: 0 },
                shape: vec![
                    Dim::Batch,
                    Dim::Fixed(std::num::NonZeroU32::new(2).unwrap()),
                    Dim::Fixed(std::num::NonZeroU32::new(3).unwrap()),
                ],
                dtype: StateTensorDtype::Floating,
                presence: StateTensorPresence::Optional,
                residency: StateResidencyClass::AlwaysDeviceMutable,
            }],
        };
        let kv = LayerCachePolicy::key_value(eredu_core::AttentionPolicy::Full, 2, 4).unwrap();
        let complete = StateLayout::new(
            eredu_core::LayerSchedule::new(
                6,
                vec![audio.clone(), audio, kv.clone(), kv.clone(), kv.clone(), kv],
            )
            .unwrap(),
        )
        .unwrap();
        let plan = ArchitectureStatePartitionPlan::new([
            crate::ArchitectureStatePartitionRule::output_owner(0..2),
            crate::ArchitectureStatePartitionRule::group_units(0, 2..6),
        ])
        .with_packing(crate::StatePartitionPacking::PreserveGlobalOrdinals);
        let mut output = state_plan_partition(
            3..4,
            PartitionOwnership::new(false, true, std::iter::empty::<&str>()).unwrap(),
        );
        let selected = output
            .resolve_state_partition(&complete, &plan)
            .unwrap()
            .unwrap();
        assert_eq!(selected.global_layers(), 0..6);
        assert_eq!(selected.selected_global_layers(), &[0..2, 5..6]);
        assert_eq!(
            selected.selected_layout(&complete).unwrap(),
            *selected.layout()
        );
        for layer in 0..6 {
            let owned = matches!(layer, 0 | 1 | 5);
            assert_eq!(selected.owns_global_layer(layer), owned);
            assert_eq!(
                selected.layout().layer(layer).unwrap(),
                if owned {
                    complete.layer(layer).unwrap()
                } else {
                    &eredu_core::cache::LayerCachePolicy::NoState
                }
            );
            if !owned {
                assert!(selected.layout().components(layer).unwrap().is_empty());
            }
        }
        let non_output = state_plan_partition(
            0..2,
            PartitionOwnership::new(true, false, std::iter::empty::<&str>()).unwrap(),
        );
        let other = non_output
            .resolve_state_partition(&complete, &plan)
            .unwrap()
            .unwrap();
        assert_eq!(other.global_layers(), 2..4);
        assert_eq!(other.selected_global_layers(), &[2..4]);
        // Global layer 5 keeps ordinal 5 in the first coordinate, while global
        // layer 2 is local slot 0 in the other. No unowned KV state is created.
        assert_eq!(selected.global_layer_offset() + 5, 5);
        assert_eq!(other.global_layer_offset(), 2);
        let forged = PartitionState::new(selected.layout().clone(), 0).unwrap();
        assert_ne!(forged.selected_layout(&complete).unwrap(), *forged.layout());
        output.state = Some(selected);
        assert!(
            matches!(
                LayeredPartitionDriver::new(&output, 0, 3..4),
                Err(LayeredPartitionError::StateRange { .. })
            ),
            "a represented gap cannot authorize a stateful unit"
        );
    }

    fn parameter_description(
        expected: Vec<ParameterGroupSpec>,
        groups: Vec<OwnedParameterGroupSpec>,
    ) -> Result<ArchitectureParameterDescription, ArchitectureParameterError> {
        let graph = graph();
        ArchitectureParameterDescription::new(&graph, &layout(&graph), expected, groups)
    }

    #[test]
    fn description_driven_partition_selects_state_and_parameters_before_construction() {
        let embedding = parameter("embedding", "model.embed_tokens.weight");
        let layer = parameter("layer", "model.layers.1.weight");
        let description = parameter_description(
            vec![embedding.clone(), layer.clone()],
            vec![
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::static_role("embedding"),
                    embedding,
                ),
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::execution_unit(
                        ExecutionGroupId::new("primary").unwrap(),
                        1,
                    ),
                    layer,
                ),
            ],
        )
        .unwrap();
        let ownership = PartitionOwnership::new(true, false, ["embedding"]).unwrap();
        let state = state_layout(4);
        let state_plan = ArchitectureStatePartitionPlan::new([
            crate::ArchitectureStatePartitionRule::group_units(0, 0..4),
        ]);

        let partition = ArchitecturePartition::from_description(
            &description,
            [("primary", 1..3)],
            ownership,
            &state,
            &state_plan,
            Geometry("selected-before-allocation"),
            PairBoundarySchema,
        )
        .unwrap();

        assert_eq!(partition.groups()[0].global_units(), 1..3);
        assert_eq!(partition.state().unwrap().global_layers(), 1..3);
        assert_eq!(
            partition.local_geometry(),
            &Geometry("selected-before-allocation")
        );
        assert_eq!(partition.parameter_bindings().len(), 2);
        assert_eq!(
            partition
                .parameter_bindings()
                .iter()
                .flat_map(|group| group.members())
                .map(ParameterMemberSpec::target)
                .collect::<Vec<_>>(),
            ["model.embed_tokens.weight", "model.layers.1.weight"]
        );
    }

    #[test]
    fn parameter_description_selects_static_roles_and_canonical_units() {
        let embedding = parameter("embedding", "model.embed_tokens.weight");
        let layer = parameter("layer", "model.layers.1.weight");
        let description = parameter_description(
            vec![embedding.clone(), layer.clone()],
            vec![
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::static_role("embedding"),
                    embedding,
                ),
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::execution_unit(
                        ExecutionGroupId::new("primary").unwrap(),
                        1,
                    ),
                    layer,
                ),
            ],
        )
        .unwrap();
        let partition = valid_partition();
        assert_eq!(description.graph(), partition.graph());
        assert_eq!(description.unit_layout(), partition.unit_layout());
        let selected = description.select_owned(&partition);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].logical_name(), "embedding");
        assert_eq!(selected[1].logical_name(), "layer");
        assert_eq!(
            selected[0].owner(),
            &ParameterGroupOwner::static_role("embedding")
        );
        assert_eq!(
            selected[1].owner(),
            &ParameterGroupOwner::execution_unit(ExecutionGroupId::new("primary").unwrap(), 1,)
        );
    }

    #[test]
    fn pinned_unit_consumers_select_one_copy_and_canonical_binding_role() {
        let primary = ExecutionGroupId::new("primary").unwrap();
        let shared = parameter("shared", "shared.weight");
        let owner = ParameterGroupOwner::static_unit_consumers(
            "projector",
            [(primary.clone(), 1), (primary.clone(), 2)],
        );
        let description = parameter_description(
            vec![shared.clone()],
            vec![OwnedParameterGroupSpec::new(owner.clone(), shared.clone())],
        )
        .unwrap();
        let partition = valid_partition();
        assert!(!partition.ownership().owns_static_role("projector"));
        assert_eq!(description.select_owned(&partition).len(), 1);
        assert_eq!(description.select_static_roles(&partition), ["projector"]);
        assert!(owner.is_owned_by(partition.ownership(), |group, unit| {
            partition.owns_unit(group.as_str(), unit)
        }));
        assert!(owner.refines_storage_owner(&ParameterGroupOwner::static_role("projector")));
        assert!(!owner.refines_storage_owner(&ParameterGroupOwner::static_role("embedding")));
        let replica = PartitionOwnership::new(false, false, Vec::<String>::new())
            .unwrap()
            .with_replicated_static_roles(["projector"])
            .unwrap();
        assert!(owner.is_stored_by(&replica, |_, _| false));
        assert!(!owner.is_owned_by(&replica, |_, _| false));
        let unowned = parameter_description(
            vec![shared.clone()],
            vec![OwnedParameterGroupSpec::new(
                ParameterGroupOwner::static_unit_consumers("embedding", [(primary, 0)]),
                shared,
            )],
        )
        .unwrap();
        assert!(partition.ownership().owns_static_role("embedding"));
        assert!(unowned.select_owned(&partition).is_empty());
        assert!(unowned.select_static_roles(&partition).is_empty());
    }

    #[test]
    fn pinned_unit_consumers_validate_exact_graph_invocations() {
        let primary = ExecutionGroupId::new("primary").unwrap();
        let make = |consumers| {
            let shared = parameter("shared", "shared.weight");
            parameter_description(
                vec![shared.clone()],
                vec![OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::static_unit_consumers("projector", consumers),
                    shared,
                )],
            )
        };
        assert_eq!(
            make(vec![]).unwrap_err(),
            ArchitectureParameterError::EmptyStaticConsumers
        );
        assert_eq!(
            make(vec![(primary.clone(), 1), (primary.clone(), 1)]).unwrap_err(),
            ArchitectureParameterError::DuplicateStaticConsumer
        );
        assert!(matches!(
            make(vec![(primary, usize::MAX)]),
            Err(ArchitectureParameterError::UnitOutOfRange { .. })
        ));
        assert!(matches!(
            make(vec![(ExecutionGroupId::new("absent").unwrap(), 0)]),
            Err(ArchitectureParameterError::UnknownExecutionGroup(_))
        ));
    }

    #[test]
    fn parameter_description_selects_every_owned_target_for_a_role() {
        let expert = ParameterGroupSpec::new(
            "model.layers.1.expert_intermediate",
            ParameterRole::ExpertIntermediate,
            [
                ParameterMemberSpec::new(
                    "model.layers.1.moe.packed.weight",
                    vec![4, 2],
                    MemberSharding::Replicated,
                ),
                ParameterMemberSpec::new(
                    "model.layers.1.moe.packed.scales",
                    vec![4, 1],
                    MemberSharding::Replicated,
                ),
                ParameterMemberSpec::new(
                    "model.layers.1.moe.alias.biases",
                    vec![4, 1],
                    MemberSharding::Replicated,
                ),
            ],
        )
        .unwrap();
        let replicated = parameter("router", "model.layers.1.moe.router.weight");
        let owner =
            ParameterGroupOwner::execution_unit(ExecutionGroupId::new("primary").unwrap(), 1);
        let description = parameter_description(
            vec![expert.clone(), replicated.clone()],
            vec![
                OwnedParameterGroupSpec::new(owner.clone(), expert),
                OwnedParameterGroupSpec::new(owner, replicated),
            ],
        )
        .unwrap();

        assert_eq!(
            description.targets_for_role(ParameterRole::ExpertIntermediate),
            BTreeSet::from([
                "model.layers.1.moe.alias.biases".to_owned(),
                "model.layers.1.moe.packed.scales".to_owned(),
                "model.layers.1.moe.packed.weight".to_owned(),
            ])
        );
    }

    #[test]
    fn auxiliary_static_replicas_materialize_without_owning_target_hooks() {
        let ownership = PartitionOwnership::new(false, true, ["output"])
            .unwrap()
            .with_replicated_static_roles(["embedding"])
            .unwrap();
        let owner = ParameterGroupOwner::static_role("embedding");
        assert!(!ownership.owns_input());
        assert!(!owner.is_owned_by(&ownership, |_, _| false));
        assert!(owner.is_stored_by(&ownership, |_, _| false));
        assert!(ParameterGroupOwner::static_any_of(["embedding", "unused"])
            .is_stored_by(&ownership, |_, _| false));
        let partition = state_plan_partition(1..4, ownership);
        let embedding = parameter("embedding", "model.embed_tokens.weight");
        let description = parameter_description(
            vec![embedding.clone()],
            vec![OwnedParameterGroupSpec::new(owner, embedding)],
        )
        .unwrap();
        let selected = description.select_owned(&partition);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].logical_name(), "embedding");
        assert!(PartitionOwnership::new(false, false, [] as [&str; 0])
            .unwrap()
            .with_replicated_static_roles(["embedding", "embedding"])
            .is_err());
    }

    #[test]
    fn parameter_description_selects_shared_static_owner_by_any_consumer() {
        let embedding = parameter("embedding", "model.embed_tokens.weight");
        let description = parameter_description(
            vec![embedding.clone()],
            vec![OwnedParameterGroupSpec::new(
                ParameterGroupOwner::static_any_of(["output", "embedding"]),
                embedding,
            )],
        )
        .unwrap();
        assert_eq!(description.select_owned(&valid_partition()).len(), 1);

        let duplicate = parameter("embedding", "model.embed_tokens.weight");
        assert_eq!(
            parameter_description(
                vec![duplicate.clone()],
                vec![OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::static_any_of(["embedding", "embedding"]),
                    duplicate,
                )],
            )
            .unwrap_err(),
            ArchitectureParameterError::DuplicateStaticRole,
        );
    }

    #[test]
    fn parameter_description_rejects_missing_duplicate_and_out_of_range_ownership() {
        let embedding = parameter("embedding", "model.embed_tokens.weight");
        let layer = parameter("layer", "model.layers.1.weight");
        assert_eq!(
            parameter_description(
                vec![embedding.clone(), layer.clone()],
                vec![OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::static_role("embedding"),
                    embedding.clone(),
                )],
            )
            .unwrap_err(),
            ArchitectureParameterError::MissingOwnership("model.layers.1.weight".into())
        );
        assert!(matches!(
            parameter_description(
                vec![embedding.clone()],
                vec![
                    OwnedParameterGroupSpec::new(
                        ParameterGroupOwner::static_role("embedding"),
                        embedding.clone(),
                    ),
                    OwnedParameterGroupSpec::new(
                        ParameterGroupOwner::static_role("output"),
                        embedding.clone(),
                    ),
                ],
            )
            .unwrap_err(),
            ArchitectureParameterError::DuplicateOwnership { .. }
        ));
        assert_eq!(
            parameter_description(
                vec![layer.clone()],
                vec![OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::execution_unit(
                        ExecutionGroupId::new("prediction").unwrap(),
                        3,
                    ),
                    layer,
                )],
            )
            .unwrap_err(),
            ArchitectureParameterError::UnitOutOfRange {
                group: "prediction".into(),
                global_unit: 3,
                available: 3,
            }
        );
    }

    #[test]
    fn retains_canonical_topology_ownership_and_typed_family_values() {
        let mut partition = valid_partition();
        assert_eq!(partition.graph().groups().len(), 2);
        assert_eq!(partition.unit_layout().len(), 7);
        assert_eq!(partition.groups()[0].group().as_str(), "primary");
        assert_eq!(partition.groups()[0].group_index(), 0);
        assert_eq!(partition.groups()[0].global_units(), 1..4);
        assert!(partition.owns_unit("primary", 3));
        assert!(!partition.owns_unit("primary", 0));
        assert!(partition.ownership().owns_input());
        assert!(!partition.ownership().owns_output());
        assert!(partition.ownership().owns_static_role("embedding"));
        assert_eq!(
            partition
                .units()
                .map(|unit| (unit.group(), unit.index()))
                .collect::<Vec<_>>(),
            [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]
        );
        assert_eq!(partition.state().unwrap().global_layers(), 7..9);
        assert_eq!(partition.local_geometry(), &Geometry("local"));
        partition.boundary_schema_mut().route = 5;
        assert_eq!(partition.boundary_schema().route, 5);
        assert_eq!(partition.parameter_bindings().len(), 2);
    }

    #[test]
    fn typed_boundary_owns_roles_order_and_atomic_cardinality_validation() {
        let boundary = PairBoundary {
            tokens: 3,
            embedded: 7,
        };
        let schema = PairBoundarySchema;
        let values = schema.encode(boundary).unwrap();
        assert_eq!(values[0].role(), "tokens");
        assert_eq!(values[1].role(), "embedded");
        let tensors = values
            .into_iter()
            .map(ArchitectureBoundaryValue::into_parts)
            .map(|(_, tensor)| tensor)
            .collect();
        assert_eq!(
            schema.decode(tensors).unwrap(),
            PairBoundary {
                tokens: 3,
                embedded: 7
            }
        );
        let resolved = schema.wire_schema().unwrap().resolve(2, 3).unwrap();
        assert_eq!(resolved.primary().shape(), [2, 3, 8]);
        assert_eq!(resolved.primary().dtype(), BoundaryTensorDtype::Activation);
        assert_eq!(resolved.auxiliary()[0].shape(), [2, 3]);
        assert_eq!(resolved.auxiliary()[0].dtype(), BoundaryTensorDtype::Uint32);
        assert_eq!(resolved.auxiliary()[1].shape(), [2, 3, 16]);
        assert_eq!(
            resolved.auxiliary()[1].dtype(),
            BoundaryTensorDtype::Activation
        );
        assert_eq!(
            schema.decode(vec![3]).unwrap_err(),
            ArchitectureBoundaryError::TensorCount {
                boundary: "fixture.target",
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn boundary_schema_rejects_role_and_geometry_drift_before_transport() {
        let invalid_primary = BoundaryWireSchema::new(
            "fixture.invalid",
            BoundaryTensorSpec::new(
                "hidden",
                [BoundaryTensorDimension::Fixed(8)],
                BoundaryTensorDtype::Uint32,
            ),
            [],
        )
        .unwrap_err();
        assert_eq!(
            invalid_primary,
            ArchitectureBoundaryError::InvalidPrimaryDtype {
                boundary: "fixture.invalid",
            }
        );

        let duplicate = BoundaryWireSchema::new(
            "fixture.invalid",
            BoundaryTensorSpec::primary_activation(8),
            [
                BoundaryTensorSpec::new(
                    "state",
                    [BoundaryTensorDimension::Fixed(1)],
                    BoundaryTensorDtype::Activation,
                ),
                BoundaryTensorSpec::new(
                    "state",
                    [BoundaryTensorDimension::Fixed(2)],
                    BoundaryTensorDtype::Activation,
                ),
            ],
        )
        .unwrap_err();
        assert_eq!(
            duplicate,
            ArchitectureBoundaryError::DuplicateTensorRole {
                boundary: "fixture.invalid",
                role: "state".into(),
            }
        );

        let invalid = BoundaryWireSchema::new(
            "fixture.invalid",
            BoundaryTensorSpec::primary_activation(8),
            [BoundaryTensorSpec::new(
                "state",
                [BoundaryTensorDimension::Fixed(0)],
                BoundaryTensorDtype::Activation,
            )],
        )
        .unwrap_err();
        assert_eq!(
            invalid,
            ArchitectureBoundaryError::InvalidTensorDimension {
                boundary: "fixture.invalid",
                role: "state".into(),
            }
        );
    }

    #[test]
    fn rejects_unknown_and_duplicate_groups() {
        let graph = graph();
        let error = ArchitecturePartition::select_topology(
            &identity_description(&graph, &layout(&graph)),
            [("missing", 0..1)],
            PartitionOwnership::new(false, false, std::iter::empty::<String>()).unwrap(),
            (),
            (),
        )
        .unwrap_err();
        assert_eq!(
            error,
            ArchitecturePartitionError::UnknownGroup("missing".into())
        );

        let error = ArchitecturePartition::select_topology(
            &identity_description(&graph, &layout(&graph)),
            [("primary", 0..1), ("primary", 1..2)],
            PartitionOwnership::new(false, false, std::iter::empty::<String>()).unwrap(),
            (),
            (),
        )
        .unwrap_err();
        assert_eq!(
            error,
            ArchitecturePartitionError::DuplicateGroup("primary".into())
        );
    }

    #[test]
    fn rejects_empty_and_out_of_bounds_group_ranges() {
        let graph = graph();
        let error = ArchitecturePartition::select_topology(
            &identity_description(&graph, &layout(&graph)),
            [("primary", 2..2)],
            PartitionOwnership::new(false, false, std::iter::empty::<String>()).unwrap(),
            (),
            (),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ArchitecturePartitionError::EmptyGroupRange { .. }
        ));

        let error = ArchitecturePartition::select_topology(
            &identity_description(&graph, &layout(&graph)),
            [("prediction", 1..4)],
            PartitionOwnership::new(false, false, std::iter::empty::<String>()).unwrap(),
            (),
            (),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ArchitecturePartitionError::GroupRangeOutOfBounds { .. }
        ));
    }

    #[test]
    fn rejects_state_offset_overflow() {
        assert_eq!(
            PartitionState::new(state_layout(2), usize::MAX).unwrap_err(),
            ArchitecturePartitionError::StateOffsetOverflow {
                offset: usize::MAX,
                layers: 2,
            }
        );
    }

    fn layered_partition(
        storage_state: Range<usize>,
        owns_input: bool,
    ) -> ArchitecturePartition<(), ()> {
        let graph = ExecutionGraph::chain(["decoder"]).unwrap();
        let layout = ExecutionUnitLayout::new(&graph, [4]).unwrap();
        ArchitecturePartition::select_topology(
            &identity_description(&graph, &layout),
            [("decoder", 1..3)],
            PartitionOwnership::new(owns_input, false, std::iter::empty::<String>()).unwrap(),
            (),
            (),
        )
        .map(|mut partition| {
            partition.state = Some(
                PartitionState::new(state_layout(storage_state.len()), storage_state.start)
                    .unwrap(),
            );
            partition
        })
        .unwrap()
    }

    #[test]
    fn layered_driver_rejects_storage_and_state_range_drift() {
        let partition = layered_partition(1..3, true);
        assert!(LayeredPartitionDriver::new(&partition, 0, 1..3).is_ok());
        assert_eq!(
            LayeredPartitionDriver::new(&partition, 0, 0..2).unwrap_err(),
            LayeredPartitionError::StorageRange {
                storage: 0..2,
                partition: 1..3,
            }
        );

        let partition = layered_partition(0..2, true);
        assert_eq!(
            LayeredPartitionDriver::new(&partition, 0, 1..3).unwrap_err(),
            LayeredPartitionError::StateRange {
                state: 0..2,
                partition: 1..3,
            }
        );
    }

    #[test]
    fn layered_driver_represents_stateless_root_without_borrowing_decoder_state() {
        let graph = ExecutionGraph::chain(["vision", "decoder"]).unwrap();
        let layout = ExecutionUnitLayout::new(&graph, [1, 2]).unwrap();
        let partition = ArchitecturePartition::select_topology(
            &identity_description(&graph, &layout),
            [("vision", 0..1), ("decoder", 0..2)],
            PartitionOwnership::new(true, true, std::iter::empty::<String>()).unwrap(),
            (),
            (),
        )
        .map(|mut partition| {
            partition.state = Some(PartitionState::new(state_layout(1), 1).unwrap());
            partition
        })
        .unwrap();

        let vision =
            LayeredPartitionDriver::new_with_state_ownership(&partition, 0, 0..1, false).unwrap();
        assert!(vision.optional_state_layout().is_none());
        assert_eq!(vision.group_index(), 0);

        let without_state = ArchitecturePartition::select_topology(
            &identity_description(
                &ExecutionGraph::chain(["vision"]).unwrap(),
                &ExecutionUnitLayout::new(&ExecutionGraph::chain(["vision"]).unwrap(), [1]).unwrap(),
            ),
            [("vision", 0..1)],
            PartitionOwnership::new(true, false, std::iter::empty::<String>()).unwrap(),
            (),
            (),
        )
        .unwrap();
        assert_eq!(
            LayeredPartitionDriver::new(&without_state, 0, 0..1).unwrap_err(),
            LayeredPartitionError::MissingState
        );
        assert!(
            LayeredPartitionDriver::new_with_state_ownership(&without_state, 0, 0..1, false)
                .unwrap()
                .optional_state_layout()
                .is_none()
        );
    }

    #[test]
    fn layered_driver_restricts_tokens_but_accepts_architecture_prepared_hidden() {
        let input_owner =
            LayeredPartitionDriver::new(&layered_partition(1..3, true), 0, 1..3).unwrap();
        assert!(matches!(
            input_owner.input(LayeredPartitionInput::<i32, NoAuxiliaryBoundary>::Tokens(
                &7
            )),
            Ok(LayeredPartitionInput::Tokens(7))
        ));
        assert!(matches!(
            input_owner.input(LayeredPartitionInput::Hidden {
                hidden: 7,
                auxiliary: NoAuxiliaryBoundary,
            }),
            Ok(LayeredPartitionInput::Hidden {
                hidden: 7,
                auxiliary: NoAuxiliaryBoundary,
            })
        ));

        let hidden_owner =
            LayeredPartitionDriver::new(&layered_partition(1..3, false), 0, 1..3).unwrap();
        assert_eq!(
            hidden_owner
                .input(LayeredPartitionInput::<i32, NoAuxiliaryBoundary>::Tokens(
                    &7
                ))
                .unwrap_err(),
            LayeredPartitionError::TokensOnNonInputOwner
        );
        assert!(matches!(
            hidden_owner.input(LayeredPartitionInput::Hidden {
                hidden: 7,
                auxiliary: NoAuxiliaryBoundary,
            }),
            Ok(LayeredPartitionInput::Hidden {
                hidden: 7,
                auxiliary: NoAuxiliaryBoundary,
            })
        ));
    }
}

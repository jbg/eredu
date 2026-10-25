//! Validated execution-group dependency graphs and ready-set scheduling.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

/// Failure projecting an architecture's canonical execution-unit layout.
#[derive(Debug, thiserror::Error)]
pub enum ArchitectureExecutionLayoutError<E> {
    /// The architecture cannot supply its neutral graph or group counts.
    #[error("architecture execution layout: {0}")]
    Architecture(#[source] E),
    /// Declared counts cannot form a valid canonical layout.
    #[error("execution-unit layout: {0}")]
    Layout(#[from] ExecutionUnitLayoutError),
}

/// Stable non-empty identity for one architecture execution group.
#[derive(Debug, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExecutionGroupId(String);

impl ExecutionGroupId {
    /// Creates a validated execution-group identifier.
    pub fn new(id: impl Into<String>) -> Result<Self, ExecutionGraphError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(ExecutionGraphError::EmptyGroupId);
        }
        Ok(Self(id))
    }

    /// Returns the stable identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ExecutionGroupId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// One named execution group and the groups whose outputs it consumes.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExecutionGroupSpec {
    id: String,
    dependencies: Vec<String>,
}

impl ExecutionGroupSpec {
    /// Declares a root execution group.
    pub fn root(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            dependencies: Vec::new(),
        }
    }

    /// Declares a group with named input dependencies.
    pub fn with_dependencies(
        id: impl Into<String>,
        dependencies: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            id: id.into(),
            dependencies: dependencies.into_iter().map(Into::into).collect(),
        }
    }

    /// Returns the stable group identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns dependency identifiers in declaration order.
    pub fn dependencies(&self) -> &[String] {
        &self.dependencies
    }
}

/// Validated execution-group dependency graph with one authoritative output.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExecutionGraph {
    groups: Vec<ExecutionGroupSpec>,
    dependencies: Vec<Vec<usize>>,
    dependents: Vec<Vec<usize>>,
    execution_order: Vec<usize>,
    output: usize,
}

/// Stable architecture-group and group-local address of one flattened execution unit.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ExecutionUnitAddress {
    group: usize,
    index: usize,
}

impl ExecutionUnitAddress {
    /// Returns the architecture execution-group slot.
    pub const fn group(self) -> usize {
        self.group
    }

    /// Returns the unit's group-local index.
    pub const fn index(self) -> usize {
        self.index
    }

    /// Returns the same execution-group address with a semantic state index.
    pub const fn with_index(self, index: usize) -> Self {
        Self {
            group: self.group,
            index,
        }
    }
}

/// Dense policy-local coordinates of one materialized execution-unit resource.
/// These coordinates do not identify a logical invocation or a mutable-state slot.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ExecutionUnitStorageLocation {
    ordinal: usize,
    address: ExecutionUnitAddress,
}
impl ExecutionUnitStorageLocation {
    /// Dense index in the selected resource pool.
    pub const fn ordinal(self) -> usize {
        self.ordinal
    }
    /// Group and unit coordinates understood by the selected policy.
    pub const fn address(self) -> ExecutionUnitAddress {
        self.address
    }
}

/// Validated invocation and storage coordinates retained with an acquired lease.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ExecutionUnitLeaseAddress {
    ordinal: usize,
    logical: ExecutionUnitAddress,
    storage: ExecutionUnitStorageLocation,
}
impl ExecutionUnitLeaseAddress {
    /// Position in the selected logical invocation sequence.
    pub const fn invocation_ordinal(self) -> usize {
        self.ordinal
    }
    /// Original logical equation address, independent of its physical owner.
    pub const fn invocation_address(self) -> ExecutionUnitAddress {
        self.logical
    }
    /// Resolves the physical pool coordinates without reusing invocation indices.
    pub const fn storage_location(self) -> ExecutionUnitStorageLocation {
        self.storage
    }
    /// Checks that completion returns the lease for its original invocation.
    pub fn validate_return(
        self,
        ordinal: usize,
        logical: ExecutionUnitAddress,
    ) -> Result<(), ExecutionUnitAddressMapError> {
        if ordinal != self.ordinal {
            return Err(ExecutionUnitAddressMapError::ReturnOrdinalMismatch {
                expected: self.ordinal,
                actual: ordinal,
            });
        }
        if logical != self.logical {
            return Err(ExecutionUnitAddressMapError::AddressMismatch {
                ordinal,
                expected: self.logical,
                actual: logical,
            });
        }
        Ok(())
    }
}

/// Validated invocation references to a selected physical resource catalog.
/// Repeated storage references preserve distinct invocation and state identities.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExecutionUnitAddressMap {
    data: std::sync::Arc<ExecutionUnitAddressMapData>,
}
#[derive(Debug, Eq, PartialEq)]
struct ExecutionUnitAddressMapData {
    storage: ExecutionUnitLayout,
    addresses: Vec<(ExecutionUnitAddress, ExecutionUnitStorageLocation)>,
    groups: Vec<Range<usize>>,
}

impl ExecutionUnitAddressMap {
    /// Validates each logical invocation and its explicit physical storage address.
    /// Every selected resource has at least one consumer. Logical groups are
    /// contiguous and retain their physical group; storage may repeat within a group.
    pub fn new(
        local: &ExecutionUnitLayout,
        references: impl IntoIterator<Item = (ExecutionUnitAddress, ExecutionUnitAddress)>,
    ) -> Result<Self, ExecutionUnitAddressMapError> {
        let mut seen = BTreeSet::new();
        let mut used = BTreeSet::new();
        let mut addresses = Vec::new();
        let mut counts = vec![0; local.group_count()];
        let mut previous = 0;
        for (ordinal, (logical, storage)) in references.into_iter().enumerate() {
            let storage_ordinal = local
                .ordinal(storage.group(), storage.index())
                .ok_or(ExecutionUnitAddressMapError::UnknownStorage { address: storage })?;
            if storage.group() != logical.group() {
                return Err(ExecutionUnitAddressMapError::GroupMismatch {
                    ordinal,
                    local: storage.group(),
                    logical: logical.group(),
                });
            }
            if logical.group() < previous {
                return Err(ExecutionUnitAddressMapError::UnorderedGroups { ordinal });
            }
            previous = logical.group();
            if !seen.insert((logical.group(), logical.index())) {
                return Err(ExecutionUnitAddressMapError::DuplicateAddress {
                    group: logical.group(),
                    index: logical.index(),
                });
            }
            used.insert(storage_ordinal);
            counts[logical.group()] += 1;
            addresses.push((
                logical,
                ExecutionUnitStorageLocation {
                    ordinal: storage_ordinal,
                    address: storage,
                },
            ));
        }
        if let Some(ordinal) = (0..local.len()).find(|ordinal| !used.contains(ordinal)) {
            return Err(ExecutionUnitAddressMapError::UnreferencedStorage {
                address: local.address(ordinal).expect("validated storage layout"),
            });
        }
        let mut start = 0;
        let groups = counts
            .into_iter()
            .map(|count| {
                let range = start..start + count;
                start += count;
                range
            })
            .collect();
        Ok(Self {
            data: std::sync::Arc::new(ExecutionUnitAddressMapData {
                storage: local.clone(),
                addresses,
                groups,
            }),
        })
    }

    /// Checks exact logical identities and group schema before resource construction.
    pub fn validate_layout(
        &self,
        layout: &ExecutionUnitLayout,
    ) -> Result<(), ExecutionUnitAddressMapError> {
        if self.data.storage.group_count() != layout.group_count() {
            return Err(ExecutionUnitAddressMapError::LayoutGroupCount {
                expected: layout.group_count(),
                actual: self.data.storage.group_count(),
            });
        }
        for group in 0..layout.group_count() {
            if self.data.storage.group_id(group) != layout.group_id(group) {
                return Err(ExecutionUnitAddressMapError::LayoutGroupIdentity { group });
            }
            let expected = layout.group_range(group).expect("declared group");
            let actual = self.group_range(group).expect("mapped group");
            if expected.len() != actual.len() {
                return Err(ExecutionUnitAddressMapError::InvocationCount {
                    group,
                    expected: expected.len(),
                    actual: actual.len(),
                });
            }
            for ordinal in expected {
                self.resolve(
                    ordinal,
                    layout.address(ordinal).expect("declared invocation"),
                )?;
            }
        }
        Ok(())
    }

    /// Retains logical consumers and their physical owners in the original order.
    /// A resource remains available when its canonical address is not itself a
    /// selected invocation. Selection cannot add or reorder logical authority.
    pub fn select_invocations(
        &self,
        selected: impl IntoIterator<Item = ExecutionUnitAddress>,
    ) -> Result<Self, ExecutionUnitAddressMapError> {
        let lookup = self
            .data
            .addresses
            .iter()
            .enumerate()
            .map(|(ordinal, &(logical, storage))| {
                ((logical.group(), logical.index()), (ordinal, storage))
            })
            .collect::<BTreeMap<_, _>>();
        let mut references = Vec::new();
        let mut resources = BTreeSet::new();
        let mut previous = None;
        for logical in selected {
            let &(ordinal, storage) = lookup
                .get(&(logical.group(), logical.index()))
                .ok_or(ExecutionUnitAddressMapError::UnknownInvocation { address: logical })?;
            if previous.is_some_and(|previous| ordinal <= previous) {
                return Err(ExecutionUnitAddressMapError::NonCanonicalSelection);
            }
            previous = Some(ordinal);
            references.push((logical, storage.address()));
            resources.insert(storage.ordinal());
        }
        let storage = self
            .data
            .storage
            .select(resources.into_iter().map(|ordinal| {
                self.data
                    .storage
                    .address(ordinal)
                    .expect("retained physical resource")
            }))
            .expect("physical owners form a canonical retained subset");
        Self::new(&storage, references)
    }

    /// Exact logical consumers in canonical traversal order, independently of storage.
    pub fn logical_addresses(&self) -> impl ExactSizeIterator<Item = ExecutionUnitAddress> + '_ {
        self.data.addresses.iter().map(|(logical, _)| *logical)
    }

    /// Selected physical resources, independent of their invocation count.
    pub fn storage_layout(&self) -> &ExecutionUnitLayout {
        &self.data.storage
    }

    /// Validated physical factory coordinates at one dense storage ordinal.
    pub fn storage_location(&self, ordinal: usize) -> Option<ExecutionUnitStorageLocation> {
        Some(ExecutionUnitStorageLocation {
            ordinal,
            address: self.data.storage.address(ordinal)?,
        })
    }

    /// Resolves an invocation only when its logical address matches its slot.
    pub fn resolve(
        &self,
        ordinal: usize,
        logical: ExecutionUnitAddress,
    ) -> Result<ExecutionUnitLeaseAddress, ExecutionUnitAddressMapError> {
        let &(expected, storage) = self
            .data
            .addresses
            .get(ordinal)
            .ok_or(ExecutionUnitAddressMapError::UnknownOrdinal { ordinal })?;
        if logical != expected {
            return Err(ExecutionUnitAddressMapError::AddressMismatch {
                ordinal,
                expected,
                actual: logical,
            });
        }
        Ok(ExecutionUnitLeaseAddress {
            ordinal,
            logical,
            storage,
        })
    }

    /// Returns a retained storage slot for completion or abort of an existing lease.
    pub fn local_location(&self, ordinal: usize) -> Option<ExecutionUnitStorageLocation> {
        self.data
            .addresses
            .get(ordinal)
            .map(|&(_, storage)| storage)
    }

    /// Invocations in one execution group, including repeated storage consumers.
    pub fn group_range(&self, group: usize) -> Option<Range<usize>> {
        self.data.groups.get(group).cloned()
    }

    /// Physical storage ordinals in this group's declared invocation order.
    pub fn group_storage_ordinals(&self, group: usize) -> Option<Vec<usize>> {
        Some(
            self.data.addresses[self.group_range(group)?]
                .iter()
                .map(|(_, storage)| storage.ordinal())
                .collect(),
        )
    }

    /// Unique physical resources used by the next bounded group-local invocations.
    pub fn storage_window(
        &self,
        ordinal: usize,
        lookahead: usize,
    ) -> Option<Vec<ExecutionUnitStorageLocation>> {
        let &(logical, _) = self.data.addresses.get(ordinal)?;
        let end = ordinal
            .saturating_add(lookahead)
            .min(self.group_range(logical.group())?.end);
        let mut seen = BTreeSet::new();
        Some(
            self.data.addresses[ordinal..end]
                .iter()
                .map(|&(_, storage)| storage)
                .filter(|storage| seen.insert(storage.ordinal()))
                .collect(),
        )
    }

    /// Whether this invocation closes its execution group.
    pub fn is_group_end(&self, ordinal: usize) -> bool {
        self.data
            .addresses
            .get(ordinal)
            .is_some_and(|(logical, _)| self.data.groups[logical.group()].end - 1 == ordinal)
    }
}

/// An invalid logical-to-local execution address mapping or invocation.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ExecutionUnitAddressMapError {
    /// Physical and logical declarations have different group counts.
    #[error("storage map has {actual} groups, expected {expected}")]
    LayoutGroupCount {
        /// Logical group count.
        expected: usize,
        /// Physical group count.
        actual: usize,
    },
    /// A physical group has another semantic identity.
    #[error("storage map group {group} differs from its logical declaration")]
    LayoutGroupIdentity {
        /// Canonical group slot.
        group: usize,
    },
    /// A group omits or invents logical invocations.
    #[error("storage map group {group} has {actual} invocations, expected {expected}")]
    InvocationCount {
        /// Canonical group slot.
        group: usize,
        /// Declared invocation count.
        expected: usize,
        /// Mapped invocation count.
        actual: usize,
    },

    /// A projection refers to a logical invocation outside its retained authority.
    #[error("unretained logical invocation {address:?}")]
    UnknownInvocation {
        /// Unknown logical address, independently of physical storage availability.
        address: ExecutionUnitAddress,
    },
    /// Logical projection duplicates or reorders retained invocations.
    #[error("invocation selection must retain its original order without duplicates")]
    NonCanonicalSelection,
    /// Completion supplied another invocation's ordinal.
    #[error("execution lease expects ordinal {expected}, received {actual}")]
    ReturnOrdinalMismatch {
        /// Ordinal captured at acquisition.
        expected: usize,
        /// Ordinal supplied at completion.
        actual: usize,
    },
    /// A reference does not name a selected physical resource.
    #[error("unknown physical execution storage {address:?}")]
    UnknownStorage {
        /// Unselected physical resource address.
        address: ExecutionUnitAddress,
    },
    /// A selected physical resource has no invocation consumer.
    #[error("physical execution storage {address:?} has no consumer")]
    UnreferencedStorage {
        /// Unused physical resource address.
        address: ExecutionUnitAddress,
    },
    /// Invocations return to an earlier execution group.
    #[error("execution invocation {ordinal} returns to an earlier group")]
    UnorderedGroups {
        /// Position of the unordered invocation.
        ordinal: usize,
    },
    /// A logical invocation names a different execution group from its local slot.
    #[error("execution slot {ordinal} maps logical group {logical} to local group {local}")]
    GroupMismatch {
        /// Logical invocation position.
        ordinal: usize,
        /// Logical execution group.
        logical: usize,
        /// Local execution group.
        local: usize,
    },
    /// Two entries refer to one logical invocation.
    #[error("logical execution address {group}:{index} occurs more than once")]
    DuplicateAddress {
        /// Logical execution group.
        group: usize,
        /// Global group-local invocation index.
        index: usize,
    },
    /// An invocation names a position outside the retained sequence.
    #[error("execution slot {ordinal} is outside the selected policy")]
    UnknownOrdinal {
        /// Requested logical invocation position.
        ordinal: usize,
    },
    /// The caller's logical address does not match its selected policy slot.
    #[error("execution slot {ordinal} expects {expected:?}, received {actual:?}")]
    AddressMismatch {
        /// Requested logical invocation position.
        ordinal: usize,
        /// Retained logical invocation address.
        expected: ExecutionUnitAddress,
        /// Supplied logical invocation address.
        actual: ExecutionUnitAddress,
    },
}

/// Validated mapping between architecture groups and the flat residency-unit order.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExecutionUnitLayout {
    group_ids: Vec<ExecutionGroupId>,
    group_ranges: Vec<Range<usize>>,
    addresses: Vec<ExecutionUnitAddress>,
}

impl ExecutionUnitLayout {
    /// Builds a stable group-major unit order for one validated execution graph.
    pub fn new(
        graph: &ExecutionGraph,
        group_unit_counts: impl IntoIterator<Item = usize>,
    ) -> Result<Self, ExecutionUnitLayoutError> {
        let counts = group_unit_counts.into_iter().collect::<Vec<_>>();
        if counts.len() != graph.groups().len() {
            return Err(ExecutionUnitLayoutError::GroupCountMismatch {
                graph_groups: graph.groups().len(),
                declared_groups: counts.len(),
            });
        }
        let group_ids = graph
            .groups()
            .iter()
            .map(|group| {
                ExecutionGroupId::new(group.id().to_owned())
                    .expect("validated execution graph has non-empty group identifiers")
            })
            .collect();
        let mut group_ranges = Vec::with_capacity(counts.len());
        let mut addresses = Vec::new();
        for (group, count) in counts.into_iter().enumerate() {
            let start = addresses.len();
            let end = start
                .checked_add(count)
                .ok_or(ExecutionUnitLayoutError::UnitCountOverflow)?;
            addresses.reserve(count);
            addresses.extend((0..count).map(|index| ExecutionUnitAddress { group, index }));
            group_ranges.push(start..end);
        }
        Ok(Self {
            group_ids,
            group_ranges,
            addresses,
        })
    }

    /// Selects canonical unit addresses without rebasing their group or unit indices.
    /// Storage ordinals and group ranges are compacted to the selected units.
    /// Input must follow this layout's order and may not duplicate or add addresses.
    pub fn select(
        &self,
        addresses: impl IntoIterator<Item = ExecutionUnitAddress>,
    ) -> Result<Self, ExecutionUnitLayoutError> {
        let mut selected = Vec::new();
        let mut previous = None;
        for address in addresses {
            let ordinal = self.ordinal(address.group(), address.index()).ok_or(
                ExecutionUnitLayoutError::UnknownSelectedAddress {
                    group: address.group(),
                    index: address.index(),
                },
            )?;
            if previous.is_some_and(|previous| ordinal <= previous) {
                return Err(ExecutionUnitLayoutError::NonCanonicalSelection);
            }
            previous = Some(ordinal);
            selected.push(address);
        }
        let mut start = 0;
        let group_ranges = (0..self.group_count())
            .map(|group| {
                let end = start
                    + selected[start..]
                        .iter()
                        .take_while(|address| address.group() == group)
                        .count();
                let range = start..end;
                start = end;
                range
            })
            .collect();
        Ok(Self {
            group_ids: self.group_ids.clone(),
            group_ranges,
            addresses: selected,
        })
    }

    /// Projects named groups while preserving sparse global invocation indices.
    pub fn project_groups(&self, graph: &ExecutionGraph) -> Result<Self, ExecutionUnitLayoutError> {
        let mut group_ids = Vec::new();
        let mut group_ranges = Vec::new();
        let mut addresses = Vec::new();
        for (group, specification) in graph.groups().iter().enumerate() {
            let parent = self
                .group_ids
                .iter()
                .position(|id| id.as_str() == specification.id())
                .ok_or_else(|| ExecutionUnitLayoutError::UnknownSelectedGroup {
                    group: specification.id().to_owned(),
                })?;
            group_ids.push(self.group_ids[parent].clone());
            let start = addresses.len();
            addresses.extend(
                self.addresses[self.group_ranges[parent].clone()]
                    .iter()
                    .map(|address| ExecutionUnitAddress {
                        group,
                        index: address.index(),
                    }),
            );
            group_ranges.push(start..addresses.len());
        }
        Ok(Self {
            group_ids,
            group_ranges,
            addresses,
        })
    }
    /// Returns the total number of execution units in group-major order.
    pub fn len(&self) -> usize {
        self.addresses.len()
    }

    /// Returns whether the architecture declares no executable units.
    pub fn is_empty(&self) -> bool {
        self.addresses.is_empty()
    }

    /// Returns the number of architecture execution groups.
    pub fn group_count(&self) -> usize {
        self.group_ranges.len()
    }

    /// Returns one architecture execution group's stable identifier.
    pub fn group_id(&self, group: usize) -> Option<&ExecutionGroupId> {
        self.group_ids.get(group)
    }

    /// Returns the group-major flat range for one architecture group.
    pub fn group_range(&self, group: usize) -> Option<Range<usize>> {
        self.group_ranges.get(group).cloned()
    }

    /// Resolves one flat residency-unit slot to its architecture address.
    pub fn address(&self, ordinal: usize) -> Option<ExecutionUnitAddress> {
        self.addresses.get(ordinal).copied()
    }

    /// Projects a nonempty contiguous global unit range into dense flat ordinals.
    /// Unknown identities, gaps and ranges outside the selected layout return `None`.
    pub fn contiguous_ordinals(
        &self,
        group: usize,
        global_units: Range<usize>,
    ) -> Option<Range<usize>> {
        if global_units.is_empty() {
            return None;
        }
        let start = self.ordinal(group, global_units.start)?;
        let end = self.ordinal(group, global_units.end - 1)? + 1;
        (end - start == global_units.len()).then_some(start..end)
    }

    /// Resolves one architecture address to its flat residency-unit slot.
    pub fn ordinal(&self, group: usize, index: usize) -> Option<usize> {
        let range = self.group_ranges.get(group)?;
        // Ordinary complete layouts retain constant-time lookup; selected layouts
        // preserve canonical indices and resolve only within their local group.
        if index < range.len() && self.addresses[range.start + index].index() == index {
            return Some(range.start + index);
        }
        self.addresses[range.clone()]
            .binary_search_by_key(&index, |address| address.index())
            .ok()
            .map(|local| range.start + local)
    }
}

/// Invalid architecture execution-unit grouping.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ExecutionUnitLayoutError {
    /// The architecture did not provide exactly one unit count per graph group.
    #[error(
        "execution graph contains {graph_groups} groups but the architecture declared {declared_groups} group counts"
    )]
    GroupCountMismatch {
        /// Validated graph group count.
        graph_groups: usize,
        /// Architecture-declared group count.
        declared_groups: usize,
    },
    /// The total number of units exceeded the addressable range.
    #[error("execution-unit count overflowed usize")]
    UnitCountOverflow,
    /// A selected address is not present in the source layout.
    #[error("selected execution group {group} unit {index} is absent from the source layout")]
    UnknownSelectedAddress {
        /// Architecture group index.
        group: usize,
        /// Canonical group-local unit index.
        index: usize,
    },
    /// A projected group is not present in the source layout.
    #[error("selected execution group {group:?} is absent from the source layout")]
    UnknownSelectedGroup {
        /// Stable execution-group identity.
        group: String,
    },
    /// Selected addresses repeat or change canonical traversal order.
    #[error("selected execution units must be unique and in canonical order")]
    NonCanonicalSelection,
}

impl ExecutionGraph {
    /// Whether every group lies on one dependency chain ending at the output.
    pub(crate) fn is_serial_chain(&self) -> bool {
        let mut previous = None;
        for &group in &self.execution_order {
            let expected = previous.as_slice();
            if self.dependencies[group] != expected {
                return false;
            }
            previous = Some(group);
        }
        previous == Some(self.output)
    }

    /// Validates names, dependency references, acyclicity, and output reachability.
    pub fn new(
        groups: Vec<ExecutionGroupSpec>,
        output: impl AsRef<str>,
    ) -> Result<Self, ExecutionGraphError> {
        if groups.is_empty() {
            return Err(ExecutionGraphError::EmptyGraph);
        }
        let mut by_id = BTreeMap::new();
        for (index, group) in groups.iter().enumerate() {
            if group.id.trim().is_empty() {
                return Err(ExecutionGraphError::EmptyGroupId);
            }
            if by_id.insert(group.id.clone(), index).is_some() {
                return Err(ExecutionGraphError::DuplicateGroup(group.id.clone()));
            }
        }
        let output_name = output.as_ref();
        let output = by_id
            .get(output_name)
            .copied()
            .ok_or_else(|| ExecutionGraphError::UnknownOutput(output_name.to_owned()))?;
        let mut dependencies = Vec::with_capacity(groups.len());
        let mut dependents = vec![Vec::new(); groups.len()];
        let mut indegree = vec![0usize; groups.len()];
        for (index, group) in groups.iter().enumerate() {
            let mut seen = BTreeSet::new();
            let mut resolved = Vec::with_capacity(group.dependencies.len());
            for dependency in &group.dependencies {
                let dependency_index = by_id.get(dependency).copied().ok_or_else(|| {
                    ExecutionGraphError::UnknownDependency {
                        group: group.id.clone(),
                        dependency: dependency.clone(),
                    }
                })?;
                if dependency_index == index {
                    return Err(ExecutionGraphError::SelfDependency(group.id.clone()));
                }
                if !seen.insert(dependency_index) {
                    return Err(ExecutionGraphError::DuplicateDependency {
                        group: group.id.clone(),
                        dependency: dependency.clone(),
                    });
                }
                resolved.push(dependency_index);
                dependents[dependency_index].push(index);
            }
            indegree[index] = resolved.len();
            dependencies.push(resolved);
        }
        let mut ready = indegree
            .iter()
            .enumerate()
            .filter_map(|(index, &degree)| (degree == 0).then_some(index))
            .collect::<BTreeSet<_>>();
        let mut execution_order = Vec::with_capacity(groups.len());
        while let Some(index) = ready.pop_first() {
            execution_order.push(index);
            for &dependent in &dependents[index] {
                indegree[dependent] -= 1;
                if indegree[dependent] == 0 {
                    ready.insert(dependent);
                }
            }
        }
        if execution_order.len() != groups.len() {
            return Err(ExecutionGraphError::Cycle);
        }
        let mut contributes = BTreeSet::new();
        let mut pending = vec![output];
        while let Some(index) = pending.pop() {
            if contributes.insert(index) {
                pending.extend(dependencies[index].iter().copied());
            }
        }
        if contributes.len() != groups.len() {
            let disconnected = groups
                .iter()
                .enumerate()
                .filter_map(|(index, group)| {
                    (!contributes.contains(&index)).then_some(group.id.clone())
                })
                .collect();
            return Err(ExecutionGraphError::Disconnected { disconnected });
        }
        Ok(Self {
            groups,
            dependencies,
            dependents,
            execution_order,
            output,
        })
    }

    /// Creates a dependency chain whose final group is the output.
    pub fn chain(
        ids: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, ExecutionGraphError> {
        let ids = ids.into_iter().map(Into::into).collect::<Vec<String>>();
        let output = ids.last().cloned().ok_or(ExecutionGraphError::EmptyGraph)?;
        let groups = ids
            .iter()
            .enumerate()
            .map(|(index, id)| match index.checked_sub(1) {
                Some(previous) => Self::group_with_dependency(id.clone(), ids[previous].clone()),
                None => ExecutionGroupSpec::root(id.clone()),
            })
            .collect();
        Self::new(groups, output)
    }

    fn group_with_dependency(id: String, dependency: String) -> ExecutionGroupSpec {
        ExecutionGroupSpec::with_dependencies(id, [dependency])
    }

    /// Returns group specifications in stable architecture slot order.
    pub fn groups(&self) -> &[ExecutionGroupSpec] {
        &self.groups
    }

    /// Resolves a stable execution-group identity to its architecture slot.
    pub fn group_index(&self, id: &str) -> Option<usize> {
        self.groups.iter().position(|group| group.id() == id)
    }

    /// Returns stable topological execution slots.
    pub fn execution_order(&self) -> &[usize] {
        &self.execution_order
    }

    /// Returns dependency slots for an architecture group slot.
    pub fn dependencies(&self, group: usize) -> Option<&[usize]> {
        self.dependencies.get(group).map(Vec::as_slice)
    }

    /// Returns dependent slots in stable declaration order.
    pub fn dependents(&self, group: usize) -> Option<&[usize]> {
        self.dependents.get(group).map(Vec::as_slice)
    }

    /// Returns the authoritative output group slot.
    pub const fn output(&self) -> usize {
        self.output
    }

    /// Returns one consumer count per group slot.
    pub fn consumer_counts(&self) -> Vec<usize> {
        let mut counts = vec![0; self.groups.len()];
        for dependencies in &self.dependencies {
            for &dependency in dependencies {
                counts[dependency] += 1;
            }
        }
        counts
    }
}

/// State of one execution group in a ready-set scheduler.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ReadyGroupState {
    /// Dependencies have not all been ordered yet.
    Pending,
    /// Work was submitted and its consumers may insert completion waits.
    Ordered,
    /// Submission failed.
    Failed,
    /// An upstream failure made this group unreachable.
    Blocked,
}

#[derive(Debug)]
struct ExecutionGroupReadySet<'a> {
    graph: &'a ExecutionGraph,
    remaining_dependencies: Vec<usize>,
    states: Vec<ReadyGroupState>,
    ready: BTreeSet<usize>,
}

/// Backend-neutral execution-group orchestration and dependency-output lifetime tracking.
#[derive(Debug)]
pub struct ExecutionGroupSchedule<'a> {
    graph: &'a ExecutionGraph,
    ready: ExecutionGroupReadySet<'a>,
    started: Vec<bool>,
    remaining_consumers: Vec<usize>,
}

impl<'a> ExecutionGroupSchedule<'a> {
    /// Creates a schedule for one validated execution graph.
    pub fn new(graph: &'a ExecutionGraph) -> Self {
        Self {
            graph,
            ready: ExecutionGroupReadySet::new(graph),
            started: vec![false; graph.groups.len()],
            remaining_consumers: graph.consumer_counts(),
        }
    }

    /// Returns ready groups which have not begun architecture setup.
    pub fn startable_groups(&self) -> impl Iterator<Item = usize> + '_ {
        self.ready
            .ready_groups()
            .filter(|&group| !self.started[group])
    }

    /// Returns dependency slots in architecture declaration order.
    pub fn dependencies(&self, group: usize) -> Result<&[usize], ExecutionScheduleError> {
        self.graph
            .dependencies(group)
            .ok_or(ExecutionScheduleError::UnknownGroup {
                group,
                count: self.started.len(),
            })
    }

    /// Commits successful architecture setup and returns producer outputs whose final
    /// consumer has now captured them.
    pub fn started(&mut self, group: usize) -> Result<Vec<usize>, ExecutionScheduleError> {
        let count = self.started.len();
        let started = self
            .started
            .get_mut(group)
            .ok_or(ExecutionScheduleError::UnknownGroup { group, count })?;
        if *started {
            return Err(ExecutionScheduleError::AlreadyStarted { group });
        }
        if !self.ready.ready.contains(&group) {
            return Err(ExecutionScheduleError::DependenciesPending { group });
        }
        *started = true;
        let mut releasable = Vec::new();
        for &dependency in &self.graph.dependencies[group] {
            self.remaining_consumers[dependency] -= 1;
            if self.remaining_consumers[dependency] == 0 {
                releasable.push(dependency);
            }
        }
        Ok(releasable)
    }

    /// Commits a successfully ordered group and unlocks its dependents.
    pub fn ordered(&mut self, group: usize) -> Result<(), ExecutionScheduleError> {
        match self.started.get(group).copied() {
            None => Err(ExecutionScheduleError::UnknownGroup {
                group,
                count: self.started.len(),
            }),
            Some(false) => Err(ExecutionScheduleError::NotStarted { group }),
            Some(true) if self.ready.state(group) == Some(ReadyGroupState::Pending) => {
                self.ready.ordered(group);
                Ok(())
            }
            Some(true) => Err(ExecutionScheduleError::AlreadyOrdered { group }),
        }
    }

    /// Closes a failed group and its dependent subgraph.
    pub fn fail(&mut self, group: usize) -> Result<(), ExecutionScheduleError> {
        if group >= self.started.len() {
            return Err(ExecutionScheduleError::UnknownGroup {
                group,
                count: self.started.len(),
            });
        }
        self.ready.fail(group);
        Ok(())
    }

    /// Returns one group's ordering state.
    pub fn state(&self, group: usize) -> Option<ReadyGroupState> {
        self.ready.state(group)
    }
}

/// Invalid transition in backend-neutral execution-group orchestration.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ExecutionScheduleError {
    /// The group slot is outside the validated graph.
    #[error("execution group {group} is outside the {count}-group schedule")]
    UnknownGroup {
        /// Requested group slot.
        group: usize,
        /// Number of groups in the schedule.
        count: usize,
    },
    /// Architecture setup was committed more than once.
    #[error("execution group {group} was already started")]
    AlreadyStarted {
        /// Conflicting group slot.
        group: usize,
    },
    /// Architecture setup was attempted before every dependency was ordered.
    #[error("execution group {group} still has unordered dependencies")]
    DependenciesPending {
        /// Premature group slot.
        group: usize,
    },
    /// Ordering was committed without successful architecture setup.
    #[error("execution group {group} was ordered before it started")]
    NotStarted {
        /// Invalid group slot.
        group: usize,
    },
    /// Ordering was committed more than once or after closure.
    #[error("execution group {group} was already ordered or closed")]
    AlreadyOrdered {
        /// Conflicting group slot.
        group: usize,
    },
}

impl<'a> ExecutionGroupReadySet<'a> {
    fn new(graph: &'a ExecutionGraph) -> Self {
        let remaining_dependencies = graph.dependencies.iter().map(Vec::len).collect::<Vec<_>>();
        let ready = remaining_dependencies
            .iter()
            .enumerate()
            .filter_map(|(group, &remaining)| (remaining == 0).then_some(group))
            .collect();
        Self {
            graph,
            remaining_dependencies,
            states: vec![ReadyGroupState::Pending; graph.groups.len()],
            ready,
        }
    }

    fn ready_groups(&self) -> impl Iterator<Item = usize> + '_ {
        self.ready.iter().copied()
    }

    fn ordered(&mut self, group: usize) {
        debug_assert_eq!(self.states[group], ReadyGroupState::Pending);
        self.ready.remove(&group);
        self.states[group] = ReadyGroupState::Ordered;
        for &dependent in &self.graph.dependents[group] {
            if self.states[dependent] != ReadyGroupState::Pending {
                continue;
            }
            self.remaining_dependencies[dependent] -= 1;
            if self.remaining_dependencies[dependent] == 0 {
                self.ready.insert(dependent);
            }
        }
    }

    fn fail(&mut self, group: usize) {
        self.close_subgraph(group, ReadyGroupState::Failed);
    }

    fn close_subgraph(&mut self, group: usize, state: ReadyGroupState) {
        let mut pending = vec![(group, state)];
        while let Some((group, state)) = pending.pop() {
            if self.states[group] != ReadyGroupState::Pending {
                continue;
            }
            self.ready.remove(&group);
            self.states[group] = state;
            pending.extend(
                self.graph.dependents[group]
                    .iter()
                    .copied()
                    .map(|dependent| (dependent, ReadyGroupState::Blocked)),
            );
        }
    }

    fn state(&self, group: usize) -> Option<ReadyGroupState> {
        self.states.get(group).copied()
    }
}

/// Invalid execution graph declaration.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ExecutionGraphError {
    /// No groups were declared.
    #[error("execution-group graph must contain at least one group")]
    EmptyGraph,
    /// A group identity is empty.
    #[error("execution-group identifiers must not be empty")]
    EmptyGroupId,
    /// Two groups share an identity.
    #[error("duplicate execution-group identifier {0:?}")]
    DuplicateGroup(String),
    /// The declared output is unknown.
    #[error("execution-group graph output {0:?} does not exist")]
    UnknownOutput(String),
    /// A dependency is unknown.
    #[error("execution group {group:?} depends on unknown group {dependency:?}")]
    UnknownDependency {
        /// Dependent group.
        group: String,
        /// Missing dependency.
        dependency: String,
    },
    /// A group depends on itself.
    #[error("execution group {0:?} cannot depend on itself")]
    SelfDependency(String),
    /// A dependency is repeated.
    #[error("execution group {group:?} repeats dependency {dependency:?}")]
    DuplicateDependency {
        /// Dependent group.
        group: String,
        /// Repeated dependency.
        dependency: String,
    },
    /// The graph contains a dependency cycle.
    #[error("execution-group graph contains a dependency cycle")]
    Cycle,
    /// Some groups do not contribute to the output.
    #[error("execution groups do not contribute to the graph output: {disconnected:?}")]
    Disconnected {
        /// Disconnected group identities.
        disconnected: Vec<String>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_order_is_stable_and_dependency_driven() {
        let graph = ExecutionGraph::new(
            vec![
                ExecutionGroupSpec::root("image"),
                ExecutionGroupSpec::root("audio"),
                ExecutionGroupSpec::with_dependencies("text", ["image", "audio"]),
            ],
            "text",
        )
        .unwrap();
        assert_eq!(graph.execution_order(), &[0, 1, 2]);
        assert_eq!(graph.dependencies(2), Some([0, 1].as_slice()));

        let mut ready = ExecutionGroupReadySet::new(&graph);
        assert_eq!(ready.ready_groups().collect::<Vec<_>>(), vec![0, 1]);
        ready.ordered(1);
        assert_eq!(ready.ready_groups().collect::<Vec<_>>(), vec![0]);
        ready.ordered(0);
        assert_eq!(ready.ready_groups().collect::<Vec<_>>(), vec![2]);
    }

    #[test]
    fn schedule_releases_dependency_outputs_after_their_final_consumer_starts() {
        let graph = ExecutionGraph::new(
            vec![
                ExecutionGroupSpec::root("root"),
                ExecutionGroupSpec::with_dependencies("left", ["root"]),
                ExecutionGroupSpec::with_dependencies("right", ["root"]),
                ExecutionGroupSpec::with_dependencies("output", ["left", "right"]),
            ],
            "output",
        )
        .unwrap();
        let mut schedule = ExecutionGroupSchedule::new(&graph);
        assert_eq!(schedule.startable_groups().collect::<Vec<_>>(), vec![0]);
        assert!(schedule.started(1).is_err());
        assert!(schedule.started(0).unwrap().is_empty());
        schedule.ordered(0).unwrap();
        assert_eq!(schedule.startable_groups().collect::<Vec<_>>(), vec![1, 2]);
        assert!(schedule.started(1).unwrap().is_empty());
        assert_eq!(schedule.started(2).unwrap(), vec![0]);
        schedule.ordered(1).unwrap();
        schedule.ordered(2).unwrap();
        assert_eq!(schedule.started(3).unwrap(), vec![1, 2]);
        assert!(schedule.ordered(3).is_ok());
        assert_eq!(schedule.state(3), Some(ReadyGroupState::Ordered));
    }

    #[test]
    fn execution_unit_layout_preserves_group_major_residency_order() {
        let graph = ExecutionGraph::new(
            vec![
                ExecutionGroupSpec::root("vision"),
                ExecutionGroupSpec::with_dependencies("text", ["vision"]),
            ],
            "text",
        )
        .unwrap();
        let layout = ExecutionUnitLayout::new(&graph, [2, 3]).unwrap();

        assert_eq!(graph.group_index("vision"), Some(0));
        assert_eq!(graph.group_index("text"), Some(1));
        assert_eq!(graph.group_index("missing"), None);
        assert_eq!(layout.len(), 5);
        assert_eq!(layout.group_count(), 2);
        assert_eq!(layout.group_id(0).unwrap().as_str(), "vision");
        assert_eq!(layout.group_id(1).unwrap().as_str(), "text");
        assert_eq!(layout.group_range(0), Some(0..2));
        assert_eq!(layout.group_range(1), Some(2..5));
        assert_eq!(layout.address(3).unwrap().group(), 1);
        assert_eq!(layout.address(3).unwrap().index(), 1);
        assert_eq!(layout.ordinal(1, 2), Some(4));
        assert_eq!(layout.ordinal(0, 2), None);
    }

    #[test]
    fn selected_unit_layout_preserves_canonical_addresses_and_local_storage() {
        let graph = ExecutionGraph::chain(["input", "blocks", "norm"]).unwrap();
        let full = ExecutionUnitLayout::new(&graph, [1, 4, 1]).unwrap();
        let addresses = [
            full.address(3).unwrap(),
            full.address(4).unwrap(),
            full.address(5).unwrap(),
        ];
        let selected = full.select(addresses).unwrap();
        assert_eq!(selected.group_range(0), Some(0..0));
        assert_eq!(selected.group_range(1), Some(0..2));
        assert_eq!(selected.group_range(2), Some(2..3));
        for (ordinal, address) in addresses.into_iter().enumerate() {
            assert_eq!(selected.address(ordinal), Some(address));
            assert_eq!(
                selected.ordinal(address.group(), address.index()),
                Some(ordinal)
            );
        }
        assert_eq!(selected.ordinal(1, 0), None);
        assert!(selected.select([full.address(1).unwrap()]).is_err());
        assert!(full.select([addresses[1], addresses[0]]).is_err());
        assert!(full.select([addresses[0], addresses[0]]).is_err());
    }

    #[test]
    fn execution_unit_layout_rejects_graph_count_drift() {
        let graph = ExecutionGraph::chain(["vision", "text"]).unwrap();
        assert_eq!(
            ExecutionUnitLayout::new(&graph, [2]).unwrap_err(),
            ExecutionUnitLayoutError::GroupCountMismatch {
                graph_groups: 2,
                declared_groups: 1,
            }
        );
    }

    #[test]
    fn invalid_graphs_fail_closed() {
        let groups = vec![
            ExecutionGroupSpec::with_dependencies("left", ["right"]),
            ExecutionGroupSpec::with_dependencies("right", ["left"]),
        ];
        assert_eq!(
            ExecutionGraph::new(groups, "right"),
            Err(ExecutionGraphError::Cycle)
        );
    }
}

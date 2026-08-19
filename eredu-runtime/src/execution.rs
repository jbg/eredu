//! Validated execution-group dependency graphs and ready-set scheduling.

use std::collections::{BTreeMap, BTreeSet};

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

impl ExecutionGraph {
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
    /// Work was cancelled before ordering.
    Cancelled,
    /// An upstream failure or cancellation made this group unreachable.
    Blocked,
}

/// Deterministic dependency bookkeeping for a concrete backend executor.
#[derive(Debug)]
pub struct ExecutionGroupReadySet<'a> {
    graph: &'a ExecutionGraph,
    remaining_dependencies: Vec<usize>,
    states: Vec<ReadyGroupState>,
    ready: BTreeSet<usize>,
}

impl<'a> ExecutionGroupReadySet<'a> {
    /// Creates a ready set with all root groups ready.
    pub fn new(graph: &'a ExecutionGraph) -> Self {
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

    /// Returns groups which can be ordered now.
    pub fn ready_groups(&self) -> impl Iterator<Item = usize> + '_ {
        self.ready.iter().copied()
    }

    /// Selects a deterministic maximal compatible subset of ready groups.
    pub fn compatible_batch(&self, mut compatible: impl FnMut(usize, usize) -> bool) -> Vec<usize> {
        let mut selected = Vec::new();
        for candidate in self.ready_groups() {
            if selected
                .iter()
                .copied()
                .all(|group| compatible(group, candidate))
            {
                selected.push(candidate);
            }
        }
        selected
    }

    /// Records successful ordering and unlocks newly-ready dependents.
    pub fn ordered(&mut self, group: usize) {
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

    /// Marks a failed group and closes its dependent subgraph.
    pub fn fail(&mut self, group: usize) {
        self.close_subgraph(group, ReadyGroupState::Failed);
    }

    /// Marks a cancelled group and closes its dependent subgraph.
    pub fn cancel(&mut self, group: usize) {
        self.close_subgraph(group, ReadyGroupState::Cancelled);
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

    /// Returns one group's scheduler state.
    pub fn state(&self, group: usize) -> Option<ReadyGroupState> {
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

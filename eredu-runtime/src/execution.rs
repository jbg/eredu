//! Validated execution-group dependency graphs and ready-set scheduling.

use std::ops::Range;

mod construction;

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

    /// Moves an already constructed identifier and dependency row into its
    /// declaration. Validation remains owned by ExecutionGraph::new.
    pub fn from_parts(id: String, dependencies: Vec<String>) -> Self {
        Self { id, dependencies }
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
#[derive(Debug, Eq, PartialEq)]
pub struct ExecutionGraph {
    groups: Vec<ExecutionGroupSpec>,
    dependencies: Vec<Vec<usize>>,
    dependents: Vec<Vec<usize>>,
    execution_order: Vec<usize>,
    output: usize,
}

impl Clone for ExecutionGraph {
    fn clone(&self) -> Self {
        match construction::clone_graph(self, construction::Destination(None)) {
            Ok(graph) => graph,
            Err(_) => unreachable!("ordinary graph cloning has no fallible destination"),
        }
    }
}

/// Actual architecture-owned graph used for allocation-free geometry comparison.
/// A single-group declaration validates its own identifier; the selected graph is
/// only the comparison operand. Owned is the compatibility path for existing producers.
pub struct ArchitectureExecutionGraph<'a>(ArchitectureExecutionGraphKind<'a>);
enum ArchitectureExecutionGraphKind<'a> {
    /// A validated graph retained by the actual architecture.
    Borrowed(&'a ExecutionGraph),
    /// One validated source group with no dependencies.
    Single(&'a str),
    /// The existing owned graph producer.
    Owned(ExecutionGraph),
}
impl<'a> ArchitectureExecutionGraph<'a> {
    /// Borrows a validated graph retained by the actual architecture.
    pub fn borrowed(graph: &'a ExecutionGraph) -> Self {
        Self(ArchitectureExecutionGraphKind::Borrowed(graph))
    }
    /// Retains the compatibility producer's existing owned graph.
    pub fn owned(graph: ExecutionGraph) -> Self {
        Self(ArchitectureExecutionGraphKind::Owned(graph))
    }
    /// Validates the same single root identifier accepted by ExecutionGraph::chain.
    pub fn single(id: &'a str) -> Result<Self, ExecutionGraphError> {
        if id.trim().is_empty() {
            return Err(ExecutionGraphError::EmptyGroupId);
        }
        Ok(Self(ArchitectureExecutionGraphKind::Single(id)))
    }
    /// Number of groups declared by the actual source.
    pub fn group_count(&self) -> usize {
        match &self.0 {
            ArchitectureExecutionGraphKind::Single(_) => 1,
            ArchitectureExecutionGraphKind::Borrowed(graph) => graph.groups.len(),
            ArchitectureExecutionGraphKind::Owned(graph) => graph.groups.len(),
        }
    }
    /// Borrows the retained dependency indices without materializing a graph copy.
    pub fn dependencies(&self, group:usize)->Option<&[usize]> {
        match &self.0 {
            ArchitectureExecutionGraphKind::Borrowed(graph)=>graph.dependencies(group),
            ArchitectureExecutionGraphKind::Owned(graph)=>graph.dependencies(group),
            ArchitectureExecutionGraphKind::Single(_)=>(group==0).then_some(&[]),
        }
    }
    /// Compares the exact source declaration with an independently validated graph.
    pub fn matches(&self, expected: &ExecutionGraph) -> bool {
        match &self.0 {
            ArchitectureExecutionGraphKind::Borrowed(graph) => *graph == expected,
            ArchitectureExecutionGraphKind::Owned(graph) => graph == expected,
            ArchitectureExecutionGraphKind::Single(id) => {
                expected.groups.len() == 1
                    && expected.groups[0].id == *id
                    && expected.groups[0].dependencies.is_empty()
                    && expected.dependencies.len() == 1
                    && expected.dependencies[0].is_empty()
                    && expected.dependents.len() == 1
                    && expected.dependents[0].is_empty()
                    && expected.execution_order.as_slice() == [0]
                    && expected.output == 0
            }
        }
    }
    /// Copies the same validated declaration through the participating metadata
    /// destination. Borrowed declarations never invoke a second graph builder.
    pub fn into_owned_with_metadata(
        self,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<ExecutionGraph, eredu_nn::Error> {
        match self.0 {
            ArchitectureExecutionGraphKind::Borrowed(graph) => graph.clone_with_metadata(context),
            ArchitectureExecutionGraphKind::Owned(graph) => Ok(graph),
            ArchitectureExecutionGraphKind::Single(id) => {
                construction::single(id, construction::Destination(Some(context)))
                    .map_err(construction::Failure::metadata)
            }
        }
    }

    /// Materializes the same declaration for an ordinary owning caller.
    pub fn into_owned(self) -> Result<ExecutionGraph, ExecutionGraphError> {
        match self.0 {
            ArchitectureExecutionGraphKind::Borrowed(graph) => Ok(graph.clone()),
            ArchitectureExecutionGraphKind::Owned(graph) => Ok(graph),
            ArchitectureExecutionGraphKind::Single(id) => {
                construction::single(id, construction::Destination(None))
                    .map_err(construction::Failure::graph)
            }
        }
    }
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

/// Allocation-free storage geometry for the shared layerwise submission driver.
/// Counts cover one complete forward. Retained groups may omit work; these
/// values grant neither request admission nor native submission authority.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct LayerwiseSubmissionGeometry {
    submissions: usize,
    consumer_waits: usize,
    max_consumers: usize,
}

impl LayerwiseSubmissionGeometry {
    const fn single_group() -> Self {
        Self {
            submissions: 0,
            consumer_waits: 0,
            max_consumers: 0,
        }
    }

    fn from_graph(graph: &ExecutionGraph) -> Option<Self> {
        if graph.groups.len() == 1 {
            return Some(Self::single_group());
        }
        let roots = graph
            .dependencies
            .iter()
            .filter(|deps| deps.is_empty())
            .count();
        let mut consumer_waits = roots.checked_add(1)?; // Final readout dependency.
        let mut max_consumers = roots;
        for (group, dependents) in graph.dependents.iter().enumerate() {
            consumer_waits = consumer_waits.checked_add(dependents.len())?;
            let consumers = dependents
                .len()
                .checked_add(usize::from(group == graph.output))?;
            max_consumers = max_consumers.max(consumers);
        }
        Some(Self {
            // Initial hidden state and each group output. Policy finish is separate.
            submissions: graph.groups.len().checked_add(1)?,
            consumer_waits,
            max_consumers,
        })
    }

    /// Initial and group-output submissions in the shared traversal.
    /// A selected policy's final completion adds its own mechanism-specific cost.
    pub const fn group_submissions_per_forward(self) -> usize {
        self.submissions
    }

    /// Root, dependency and final-readout waits in one complete traversal.
    pub const fn consumer_waits_per_forward(self) -> usize {
        self.consumer_waits
    }

    /// Largest consumer population of any single initial/group producer.
    /// A sequential bank may use this bound when omitted groups change which
    /// prepared slot receives a submission. It includes no arbitrary host data.
    pub const fn max_consumers_per_submission(self) -> usize {
        self.max_consumers
    }
}

/// Validated mapping between architecture groups and the flat residency-unit order.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExecutionUnitLayout {
    group_ids: Vec<ExecutionGroupId>,
    group_ranges: Vec<Range<usize>>,
    addresses: Vec<ExecutionUnitAddress>,
    submissions: LayerwiseSubmissionGeometry,
}

impl ExecutionUnitLayout {
    /// Compares the exact single-group declaration without allocating its
    /// group-major rows. This checks the complete owned representation, including
    /// addresses and submission geometry; it grants no execution authority.
    pub fn matches_single_group(&self, id: &str, count: usize) -> bool {
        self.group_ids.len() == 1
            && self.group_ids[0].as_str() == id
            && self.group_ranges.as_slice() == [0..count]
            && self.addresses.len() == count
            && self
                .addresses
                .iter()
                .enumerate()
                .all(|(index, address)| *address == ExecutionUnitAddress { group: 0, index })
            && self.submissions == LayerwiseSubmissionGeometry::single_group()
    }

    /// Requested payload bytes of the vectors and strings made by `Clone`.
    ///
    /// The inline layout value and existing execution graph are separate owners.
    /// This describes the clone producer; it grants no execution authority.
    pub fn cloned_payload_bytes(&self) -> Option<usize> {
        use std::alloc::Layout;
        let bytes = Layout::array::<ExecutionGroupId>(self.group_ids.len())
            .ok()?
            .size()
            .checked_add(
                Layout::array::<Range<usize>>(self.group_ranges.len())
                    .ok()?
                    .size(),
            )?
            .checked_add(
                Layout::array::<ExecutionUnitAddress>(self.addresses.len())
                    .ok()?
                    .size(),
            )?;
        self.group_ids.iter().try_fold(bytes, |bytes, id| {
            bytes.checked_add(Layout::array::<u8>(id.as_str().len()).ok()?.size())
        })
    }

    /// Builds a stable group-major unit order for one validated execution graph.
    pub fn new(
        graph: &ExecutionGraph,
        group_unit_counts: impl IntoIterator<Item = usize>,
    ) -> Result<Self, ExecutionUnitLayoutError> {
        let submissions = LayerwiseSubmissionGeometry::from_graph(graph)
            .ok_or(ExecutionUnitLayoutError::SubmissionCountOverflow)?;
        let counts = group_unit_counts.into_iter().collect::<Vec<_>>();
        construction::layout(graph, submissions, &counts, construction::Destination(None))
            .map_err(construction::Failure::layout)
    }

    /// Constructs the same group-major layout using the caller's counted metadata
    /// destination. Counts are borrowed from the actual architecture declaration.
    pub fn new_with_metadata(
        graph: &ExecutionGraph,
        group_unit_counts: &[usize],
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Self, eredu_nn::Error> {
        let destination = construction::Destination(Some(context));
        destination
            .controls::<(Self, &[usize], LayerwiseSubmissionGeometry)>()
            .map_err(construction::Failure::metadata)?;
        let submissions = LayerwiseSubmissionGeometry::from_graph(graph)
            .ok_or_else(|| {
                destination.layout_error(ExecutionUnitLayoutError::SubmissionCountOverflow)
            })
            .map_err(construction::Failure::metadata)?;
        construction::layout(graph, submissions, group_unit_counts, destination)
            .map_err(construction::Failure::metadata)
    }

    /// Returns the total number of execution units in group-major order.
    pub fn len(&self) -> usize {
        self.addresses.len()
    }

    /// Returns whether the architecture declares no executable units.
    pub fn is_empty(&self) -> bool {
        self.addresses.is_empty()
    }

    /// Retained graph-derived submission storage geometry, without rebuilding
    /// the architecture graph or allocating during request inspection.
    pub const fn submission_geometry(&self) -> LayerwiseSubmissionGeometry {
        self.submissions
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

    /// The forward lookahead window within this unit's architecture group.
    /// A window never crosses a semantic group boundary, including when its
    /// selected depth exceeds the remaining group length.
    pub fn window_range(
        &self,
        ordinal: usize,
        depth: std::num::NonZeroUsize,
    ) -> Option<Range<usize>> {
        let address = self.address(ordinal)?;
        let group = self.group_range(address.group())?;
        Some(ordinal..ordinal.saturating_add(depth.get()).min(group.end))
    }

    /// Resolves one architecture address to its flat residency-unit slot.
    pub fn ordinal(&self, group: usize, index: usize) -> Option<usize> {
        let range = self.group_ranges.get(group)?;
        (index < range.len()).then_some(range.start + index)
    }
}

/// Invalid architecture execution-unit grouping.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ExecutionUnitLayoutError {
    /// Arbitrary architecture mutation invalidated a retained prepared layout.
    #[error("prepared execution geometry was invalidated by architecture mutation")]
    StalePreparedGeometry,
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
    /// Shared producer or dependency-wait population exceeded usize.
    #[error("execution submission count overflowed usize")]
    SubmissionCountOverflow,
}

impl ExecutionGraph {
    /// Copies this already validated graph through the same worker as Clone,
    /// admitting each actual vector and name before construction. No graph
    /// validation, architecture selection, or execution authority is repeated.
    pub fn clone_with_metadata(
        &self,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Self, eredu_nn::Error> {
        construction::clone_graph(self, construction::Destination(Some(context)))
            .map_err(construction::Failure::metadata)
    }

    /// Materializes the exact one-root architecture declaration through counted
    /// row/name destinations. It shares the ordinary Single declaration worker;
    /// arbitrary graph validation and execution scheduling are unchanged.
    pub fn single_with_metadata(
        id: &str,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Self, eredu_nn::Error> {
        construction::single(id, construction::Destination(Some(context)))
            .map_err(construction::Failure::metadata)
    }

    /// Validates names, dependency references, acyclicity, and output reachability.
    pub fn new(
        groups: Vec<ExecutionGroupSpec>,
        output: impl AsRef<str>,
    ) -> Result<Self, ExecutionGraphError> {
        construction::graph::construct(groups, output.as_ref(), construction::Destination(None))
            .map_err(construction::Failure::graph)
    }

    /// Runs the same graph validator with counted index, adjacency and error
    /// destinations. The caller must have paid the supplied owned declarations
    /// and retain its metadata funding with this returned graph or error.
    pub fn new_with_metadata(
        groups: Vec<ExecutionGroupSpec>,
        output: &str,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Self, eredu_nn::Error> {
        construction::graph::construct(groups, output, construction::Destination(Some(context)))
            .map_err(construction::Failure::metadata)
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
        self.fill_consumer_counts(&mut counts);
        counts
    }

    fn fill_consumer_counts(&self, counts: &mut [usize]) {
        for dependencies in &self.dependencies {
            for &dependency in dependencies {
                counts[dependency] += 1;
            }
        }
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
    ready: Vec<bool>,
}

/// Backend-neutral execution-group orchestration and dependency-output lifetime tracking.
#[derive(Debug)]
pub struct ExecutionGroupSchedule<'a> {
    graph: &'a ExecutionGraph,
    ready: ExecutionGroupReadySet<'a>,
    started: Vec<bool>,
    remaining_consumers: Vec<usize>,
    // All schedule slots retire before their paying destination.
    _metadata: Option<eredu_nn::workspace::WorkspaceMetadataFunding>,
}

impl<'a> ExecutionGroupSchedule<'a> {
    /// Creates a schedule for one validated execution graph.
    pub fn new(graph: &'a ExecutionGraph) -> Self {
        Self::new_with_destination(graph, None)
            .expect("ordinary schedule construction is infallible")
    }

    /// Creates the same schedule with actual slot allocations paid before birth.
    pub(crate) fn new_with_metadata(
        graph: &'a ExecutionGraph,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Self, eredu_nn::Error> {
        Self::new_with_destination(graph, Some(context))
    }

    fn new_with_destination(
        graph: &'a ExecutionGraph,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Self, eredu_nn::Error> {
        if let Some(context) = context {
            context.charge_metadata(std::mem::size_of::<(
                Self, Result<Self, eredu_nn::Error>, ExecutionScheduleError,
                Result<(), ExecutionScheduleError>, usize, Option<bool>, ReadyGroupState,
            )>())?;
        }
        let ready = ExecutionGroupReadySet::new_with_destination(graph, context)?;
        let mut started = schedule_vector(context, graph.groups.len())?;
        started.resize(graph.groups.len(), false);
        let mut remaining_consumers = schedule_vector(context, graph.groups.len())?;
        remaining_consumers.resize(graph.groups.len(), 0usize);
        graph.fill_consumer_counts(&mut remaining_consumers);
        Ok(Self {
            graph,
            ready,
            started,
            remaining_consumers,
            _metadata: context.and_then(eredu_nn::workspace::WorkspaceContext::metadata_funding),
        })
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
        let mut releasable = Vec::new();
        self.started_with_release(group, |dependency| releasable.push(dependency))?;
        Ok(releasable)
    }

    /// Applies the same transition and lends released dependency slots in
    /// declaration order without allocating a result container.
    pub fn started_with_release(
        &mut self,
        group: usize,
        mut release: impl FnMut(usize),
    ) -> Result<(), ExecutionScheduleError> {
        let count = self.started.len();
        let started = self
            .started
            .get_mut(group)
            .ok_or(ExecutionScheduleError::UnknownGroup { group, count })?;
        if *started {
            return Err(ExecutionScheduleError::AlreadyStarted { group });
        }
        if !self.ready.ready[group] {
            return Err(ExecutionScheduleError::DependenciesPending { group });
        }
        *started = true;
        for &dependency in &self.graph.dependencies[group] {
            self.remaining_consumers[dependency] -= 1;
            if self.remaining_consumers[dependency] == 0 {
                release(dependency);
            }
        }
        Ok(())
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

fn schedule_vector<T>(
    context: Option<&eredu_nn::workspace::WorkspaceContext>,
    count: usize,
) -> Result<Vec<T>, eredu_nn::Error> {
    match context {
        Some(context) => context.metadata_vec(count),
        None => Ok(Vec::with_capacity(count)),
    }
}

impl<'a> ExecutionGroupReadySet<'a> {
    #[cfg(test)]
    fn new(graph: &'a ExecutionGraph) -> Self {
        Self::new_with_destination(graph, None).expect("ordinary ready set is infallible")
    }

    fn new_with_destination(
        graph: &'a ExecutionGraph,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Self, eredu_nn::Error> {
        if let Some(context) = context {
            context.charge_metadata(std::mem::size_of::<(Self, Result<Self, eredu_nn::Error>)>())?;
        }
        let mut remaining_dependencies = schedule_vector(context, graph.groups.len())?;
        remaining_dependencies.extend(graph.dependencies.iter().map(Vec::len));
        let mut ready = schedule_vector(context, graph.groups.len())?;
        ready.extend(
            remaining_dependencies
                .iter()
                .map(|&remaining| remaining == 0),
        );
        let mut states = schedule_vector(context, graph.groups.len())?;
        states.resize(graph.groups.len(), ReadyGroupState::Pending);
        Ok(Self {
            graph,
            remaining_dependencies,
            states,
            ready,
        })
    }

    fn ready_groups(&self) -> impl Iterator<Item = usize> + '_ {
        self.ready
            .iter()
            .enumerate()
            .filter_map(|(group, ready)| ready.then_some(group))
    }

    fn ordered(&mut self, group: usize) {
        debug_assert_eq!(self.states[group], ReadyGroupState::Pending);
        self.ready[group] = false;
        self.states[group] = ReadyGroupState::Ordered;
        for &dependent in &self.graph.dependents[group] {
            if self.states[dependent] != ReadyGroupState::Pending {
                continue;
            }
            self.remaining_dependencies[dependent] -= 1;
            if self.remaining_dependencies[dependent] == 0 {
                self.ready[dependent] = true;
            }
        }
    }

    fn fail(&mut self, group: usize) {
        self.close_subgraph(group, ReadyGroupState::Failed);
    }

    fn close_subgraph(&mut self, group: usize, state: ReadyGroupState) {
        if self.states[group] != ReadyGroupState::Pending {
            return;
        }
        self.ready[group] = false;
        self.states[group] = state;
        // Every dependency precedes its consumer in this validated order. One
        // pass closes the same reachable pending descendants as the old DFS,
        // without a growing work vector or changing any externally visible order.
        for &dependent in &self.graph.execution_order {
            if self.states[dependent] == ReadyGroupState::Pending
                && self.graph.dependencies[dependent]
                    .iter()
                    .any(|&dependency| {
                        matches!(
                            self.states[dependency],
                            ReadyGroupState::Failed | ReadyGroupState::Blocked
                        )
                    })
            {
                self.ready[dependent] = false;
                self.states[dependent] = ReadyGroupState::Blocked;
            }
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

        // A newly ready earlier declaration precedes an already ready later
        // root. This is the stable order used at composite execution cuts.
        let reordered = ExecutionGraph::new(
            vec![
                ExecutionGroupSpec::with_dependencies("output", ["branch", "other"]),
                ExecutionGroupSpec::with_dependencies("branch", ["root"]),
                ExecutionGroupSpec::root("root"),
                ExecutionGroupSpec::root("other"),
            ],
            "output",
        )
        .unwrap();
        assert_eq!(reordered.execution_order(), &[2, 1, 3, 0]);
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
    fn submission_storage_distinguishes_single_chain_fork_and_multiple_roots() {
        let single = ExecutionGraph::chain(["only"]).unwrap();
        let chain = ExecutionGraph::chain(["first", "middle", "last"]).unwrap();
        let fork = ExecutionGraph::new(
            vec![
                ExecutionGroupSpec::root("input"),
                ExecutionGroupSpec::with_dependencies("left", ["input"]),
                ExecutionGroupSpec::with_dependencies("right", ["input"]),
                ExecutionGroupSpec::with_dependencies("output", ["left", "right"]),
            ],
            "output",
        )
        .unwrap();
        let joined = ExecutionGraph::new(
            vec![
                ExecutionGroupSpec::root("image"),
                ExecutionGroupSpec::root("audio"),
                ExecutionGroupSpec::with_dependencies("text", ["image", "audio"]),
            ],
            "text",
        )
        .unwrap();
        for (graph, expected) in [
            (single, (0, 0, 0)),
            (chain, (4, 4, 1)),
            (fork, (5, 6, 2)),
            (joined, (4, 5, 2)),
        ] {
            // Semantic group dependencies determine submissions; changing the
            // number of resident units, including an empty group, does not.
            for units in [0, 1, 3] {
                let layout = ExecutionUnitLayout::new(
                    &graph,
                    std::iter::repeat_n(units, graph.groups().len()),
                )
                .unwrap();
                let geometry = layout.submission_geometry();
                assert_eq!(
                    (
                        geometry.group_submissions_per_forward(),
                        geometry.consumer_waits_per_forward(),
                        geometry.max_consumers_per_submission()
                    ),
                    expected
                );
            }
        }
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
        // Preserve declaration-order diagnostics even though construction now
        // indexes borrowed names before validating them.
        assert_eq!(
            ExecutionGraph::new(
                vec![
                    ExecutionGroupSpec::root("same"),
                    ExecutionGroupSpec::root("same"),
                    ExecutionGroupSpec::root(" "),
                ],
                "missing",
            ),
            Err(ExecutionGraphError::DuplicateGroup("same".into()))
        );
        assert_eq!(
            ExecutionGraph::new(
                vec![
                    ExecutionGroupSpec::root("root"),
                    ExecutionGroupSpec::with_dependencies("output", ["root", "root", "missing"]),
                ],
                "output",
            ),
            Err(ExecutionGraphError::DuplicateDependency {
                group: "output".into(),
                dependency: "root".into(),
            })
        );
    }
}

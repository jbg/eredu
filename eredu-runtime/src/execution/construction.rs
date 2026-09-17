//! Owning destinations for validated single declarations and group-major layouts.
use super::*;
pub(super) mod graph;
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceMetadataError},
};

#[derive(Clone, Copy)]
pub(super) struct Destination<'a>(pub(super) Option<&'a WorkspaceContext>);
pub(super) enum Failure {
    Graph(ExecutionGraphError),
    Layout(ExecutionUnitLayoutError),
    Metadata(Error),
}
impl Failure {
    pub(super) fn graph(self) -> ExecutionGraphError {
        match self {
            Self::Graph(cause) => cause,
            _ => unreachable!("ordinary graph construction uses only graph failures"),
        }
    }
    pub(super) fn layout(self) -> ExecutionUnitLayoutError {
        match self {
            Self::Layout(cause) => cause,
            _ => unreachable!("ordinary layout uses only layout failures"),
        }
    }
    pub(super) fn metadata(self) -> Error {
        match self {
            Self::Metadata(cause) => cause,
            _ => unreachable!("metadata destination retains every fixed failure"),
        }
    }
}
impl Destination<'_> {
    pub(super) fn controls<T>(self) -> Result<(), Failure> {
        if let Some(context) = self.0 {
            let bytes = [
                size_of::<T>(),
                size_of::<Self>(),
                size_of::<Failure>(),
                size_of::<Result<T, Failure>>(),
            ]
            .into_iter()
            .try_fold(0usize, usize::checked_add)
            .ok_or_else(|| Failure::Metadata(WorkspaceMetadataError::Overflow.into()))?;
            context
                .charge_metadata(bytes)
                .map_err(|cause| Failure::Metadata(cause.into()))?;
        }
        Ok(())
    }
    pub(super) fn graph_error(self, cause: ExecutionGraphError) -> Failure {
        match self.0 {
            Some(context) => Failure::Metadata(context.metadata_source(cause)),
            None => Failure::Graph(cause),
        }
    }
    pub(super) fn layout_error(self, cause: ExecutionUnitLayoutError) -> Failure {
        match self.0 {
            Some(context) => Failure::Metadata(context.metadata_source(cause)),
            None => Failure::Layout(cause),
        }
    }
    pub(super) fn vector<T>(self, count: usize) -> Result<Vec<T>, Failure> {
        match self.0 {
            Some(context) => context.metadata_vec(count).map_err(Failure::Metadata),
            None => Ok(Vec::with_capacity(count)),
        }
    }
    fn reserve<T>(self, values: &mut Vec<T>, additional: usize) -> Result<(), Failure> {
        match self.0 {
            Some(context) => context
                .reserve_metadata_vec(values, additional)
                .map_err(Failure::Metadata),
            None => {
                values.reserve(additional);
                Ok(())
            }
        }
    }
    fn text(self, value: &str) -> Result<String, Failure> {
        match self.0 {
            Some(context) => context
                .metadata_string(format_args!("{value}"))
                .map_err(Failure::Metadata),
            None => Ok(value.to_owned()),
        }
    }
}

pub(super) fn single(id: &str, destination: Destination<'_>) -> Result<ExecutionGraph, Failure> {
    destination.controls::<(
        ExecutionGraph,
        ArchitectureExecutionGraph<'_>,
        ExecutionGroupSpec,
        &str,
    )>()?;
    // The same borrowed declaration establishes all single-root invariants.
    let _source =
        ArchitectureExecutionGraph::single(id).map_err(|cause| destination.graph_error(cause))?;
    let id = destination.text(id)?;
    let mut groups = destination.vector(1)?;
    groups.push(ExecutionGroupSpec::root(id));
    let mut dependencies = destination.vector(1)?;
    dependencies.push(Vec::new());
    let mut dependents = destination.vector(1)?;
    dependents.push(Vec::new());
    let mut execution_order = destination.vector(1)?;
    execution_order.push(0);
    Ok(ExecutionGraph {
        groups,
        dependencies,
        dependents,
        execution_order,
        output: 0,
    })
}

pub(super) fn layout(
    graph: &ExecutionGraph,
    submissions: LayerwiseSubmissionGeometry,
    counts: &[usize],
    destination: Destination<'_>,
) -> Result<ExecutionUnitLayout, Failure> {
    destination.controls::<(
        ExecutionUnitLayout,
        ExecutionGroupId,
        Range<usize>,
        ExecutionUnitAddress,
    )>()?;
    if counts.len() != graph.groups().len() {
        return Err(
            destination.layout_error(ExecutionUnitLayoutError::GroupCountMismatch {
                graph_groups: graph.groups().len(),
                declared_groups: counts.len(),
            }),
        );
    }
    let mut group_ids = destination.vector(graph.groups().len())?;
    for group in graph.groups() {
        group_ids.push(
            ExecutionGroupId::new(destination.text(group.id())?)
                .expect("validated execution graph has non-empty group identifiers"),
        );
    }
    let mut group_ranges = destination.vector(counts.len())?;
    let mut addresses = Vec::new();
    for (group, &count) in counts.iter().enumerate() {
        let start = addresses.len();
        let end = start
            .checked_add(count)
            .ok_or_else(|| destination.layout_error(ExecutionUnitLayoutError::UnitCountOverflow))?;
        destination.reserve(&mut addresses, count)?;
        addresses.extend((0..count).map(|index| ExecutionUnitAddress { group, index }));
        group_ranges.push(start..end);
    }
    Ok(ExecutionUnitLayout {
        group_ids,
        group_ranges,
        addresses,
        submissions,
    })
}

// The ordinary derived-clone layout is now explicit so paid copies use the
// identical field order, names, adjacency rows, execution order and output.
pub(super) fn clone_graph(
    source: &ExecutionGraph,
    destination: Destination<'_>,
) -> Result<ExecutionGraph, Failure> {
    destination.controls::<(
        ExecutionGraph,
        &ExecutionGraph,
        ExecutionGroupSpec,
        Vec<Vec<usize>>,
        Vec<usize>,
        Vec<String>,
        String,
    )>()?;
    let mut groups = destination.vector(source.groups.len())?;
    for group in &source.groups {
        let id = destination.text(&group.id)?;
        let mut dependencies = destination.vector(group.dependencies.len())?;
        for dependency in &group.dependencies {
            dependencies.push(destination.text(dependency)?);
        }
        groups.push(ExecutionGroupSpec { id, dependencies });
    }
    let mut dependencies = destination.vector(source.dependencies.len())?;
    for row in &source.dependencies {
        let mut copied = destination.vector(row.len())?;
        copied.extend_from_slice(row);
        dependencies.push(copied);
    }
    let mut dependents = destination.vector(source.dependents.len())?;
    for row in &source.dependents {
        let mut copied = destination.vector(row.len())?;
        copied.extend_from_slice(row);
        dependents.push(copied);
    }
    let mut execution_order = destination.vector(source.execution_order.len())?;
    execution_order.extend_from_slice(&source.execution_order);
    Ok(ExecutionGraph {
        groups,
        dependencies,
        dependents,
        execution_order,
        output: source.output,
    })
}

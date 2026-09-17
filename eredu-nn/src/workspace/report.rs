//! Fixed supplied destinations for the existing report equations.
use super::*;
use std::{alloc::Layout, collections::TryReserveError};

mod reduce;
mod source;
use reduce::{Frame, Node, Scratch};
use source::{Graph, Ordinary};

/// Fixed rejection; no string, callback or native work is constructed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum WorkspaceReportError {
    /// Count, layout or checked byte addition overflowed.
    #[error("workspace allocation sum overflow")]
    Overflow,
    /// Supplied graph indices or alias ranges do not name their source.
    #[error("workspace report source is malformed")]
    Source,
    /// An exact supplied destination is too short; no growth was attempted.
    #[error("workspace report destination is too short")]
    Capacity,
}

/// Original flat graph node. Indices belong only to the borrowed graph view.
/// This is metadata geometry, never source or execution authority.
#[derive(Clone, Copy, Debug)]
pub struct WorkspaceReportNode {
    /// Complete backing capacity; unknown is preserved.
    pub bytes: Option<u64>,
    /// First alias index in the graph's ordered edge slice.
    pub alias_start: usize,
    /// Number of alias indices, retaining their actual order and duplicates.
    pub alias_count: usize,
}

/// Borrowed flat source for a report; construction validates every range/index.
/// The complete source remains borrowed throughout reduction. No index handle
/// or identity escapes into the resulting scalar report.
///
/// The same source loan survives until its final report use:
/// ```
/// use eredu_nn::workspace::{WorkspaceReportGraph, WorkspaceReportInputs,
///     WorkspaceReportLayout, WorkspaceReportNode, WorkspaceReportWorkspace};
/// let nodes = vec![WorkspaceReportNode { bytes: Some(4), alias_start: 0, alias_count: 0 }];
/// let graph = WorkspaceReportGraph::new(&nodes, &[]).unwrap();
/// let mut workspace = WorkspaceReportWorkspace::new(WorkspaceReportLayout::new(1, 0).unwrap()).unwrap();
/// let report = workspace.report(graph, WorkspaceReportInputs {
///     opening: None, allocations: &[0], closing: &[0], borrowed: None,
///     scratch: 0, host_workspace: Some(0), tensor_complete: true,
/// }).unwrap();
/// drop(nodes);
/// assert_eq!(report.total_bytes, Some(4));
/// ```
/// A source cannot retire while its borrowed graph is still used:
/// ```compile_fail
/// use eredu_nn::workspace::{WorkspaceReportGraph, WorkspaceReportInputs,
///     WorkspaceReportLayout, WorkspaceReportNode, WorkspaceReportWorkspace};
/// let nodes = vec![WorkspaceReportNode { bytes: Some(4), alias_start: 0, alias_count: 0 }];
/// let graph = WorkspaceReportGraph::new(&nodes, &[]).unwrap();
/// drop(nodes);
/// let mut workspace = WorkspaceReportWorkspace::new(WorkspaceReportLayout::new(1, 0).unwrap()).unwrap();
/// workspace.report(graph, WorkspaceReportInputs {
///     opening: None, allocations: &[0], closing: &[0], borrowed: None,
///     scratch: 0, host_workspace: Some(0), tensor_complete: true,
/// }).unwrap();
/// ```
#[derive(Clone, Copy, Debug)]
pub struct WorkspaceReportGraph<'a> {
    nodes: &'a [WorkspaceReportNode],
    edges: &'a [usize],
}
impl<'a> WorkspaceReportGraph<'a> {
    /// Validates the source without allocation or destination mutation.
    pub fn new(
        nodes: &'a [WorkspaceReportNode],
        edges: &'a [usize],
    ) -> Result<Self, WorkspaceReportError> {
        for node in nodes {
            let end = node
                .alias_start
                .checked_add(node.alias_count)
                .ok_or(WorkspaceReportError::Overflow)?;
            if end > edges.len() {
                return Err(WorkspaceReportError::Source);
            }
        }
        if edges.iter().any(|index| *index >= nodes.len()) {
            return Err(WorkspaceReportError::Source);
        }
        Ok(Self { nodes, edges })
    }
}

/// Ordered source-relative roots and scalar operation facts. Root slices may
/// contain more entries than there are distinct graph nodes.
#[derive(Clone, Copy, Debug)]
pub struct WorkspaceReportInputs<'a> {
    /// Absent and present-empty state seeding are deliberately different.
    pub opening: Option<&'a [usize]>,
    /// One ordered entry per actual new allocation, as in the ordinary trace.
    pub allocations: &'a [usize],
    /// Actual completed retained roots; duplicates are allowed.
    pub closing: &'a [usize],
    /// Exact selected existing identities; absence disables residual reporting.
    pub borrowed: Option<&'a [usize]>,
    /// Accumulated tensor scratch.
    pub scratch: u64,
    /// Accumulated disjoint host workspace, when fully described.
    pub host_workspace: Option<u64>,
    /// Whether all executed tensor operations have a bound.
    pub tensor_complete: bool,
}

/// Residual scalar fields, without an owning metadata selection or authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkspaceReportResidual {
    /// Exact unborrowed opening backing population; absent without state seeding.
    pub opening_storage: Option<WorkspaceStoragePopulation>,
    /// Exact unborrowed closing backing population; no scalar source subtraction.
    pub closing_storage: Option<WorkspaceStoragePopulation>,
    /// Complete unborrowed union and scratch/host sum.
    pub total_bytes: Option<u64>,
    /// Unborrowed closing backing.
    pub retained_bytes: Option<u64>,
    /// Unborrowed opening backing absent from closing state.
    pub displaced_bytes: Option<u64>,
    /// Unborrowed union absent from closing state, plus scratch/host.
    pub transient_bytes: Option<u64>,
}

/// Complete possible backing union of an actual closing-root slice. The count
/// includes every distinct nonzero or unknown backing, including possible input
/// aliases; zero-byte view nodes add no allocation. Unknown capacity stays unknown.
/// This describes storage geometry, never native liveness or allocation authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceStoragePopulation {
    /// Full backing bytes, rather than the roots' logical tensor sizes.
    pub bytes: Option<u64>,
    /// Maximum positive backing allocations in the same alias union.
    pub maximum_allocations: usize,
}
impl WorkspaceStoragePopulation {
    /// An observed empty root union.
    pub const EMPTY: Self = Self {
        bytes: Some(0),
        maximum_allocations: 0,
    };
}

/// Same scalar domains as the ordinary report, without copied diagnostics.
#[derive(Clone, Debug)]
pub struct WorkspaceReportScalars {
    /// Complete closing-root backing union, including unchanged input aliases.
    pub closing_storage: WorkspaceStoragePopulation,
    /// Complete seeded opening backing; absent seeding is not an empty source.
    pub opening_storage: Option<WorkspaceStoragePopulation>,
    /// Optional residual reduction.
    pub residual: Option<WorkspaceReportResidual>,
    /// Optional complete state-span accounting.
    pub state: Option<WorkspaceStateSpanReport>,
    /// New tensor plus host workspace.
    pub total_bytes: Option<u64>,
    /// New tensor storage surviving the span.
    pub retained_bytes: Option<u64>,
    /// New tensor/host storage not retained.
    pub transient_bytes: Option<u64>,
    /// Tensor-only domains.
    pub tensor_buffers: WorkspaceTensorBufferReport,
    /// Disjoint host workspace or unknown.
    pub host_workspace_bytes: Option<u64>,
}

/// Exact requested capacities for five report destinations. This does not
/// certify any physical graph, source, completeness or request admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkspaceReportLayout {
    nodes: usize,
    edges: usize,
}
impl WorkspaceReportLayout {
    /// Checks every actual flat destination layout before any reserve.
    pub fn new(nodes: usize, edges: usize) -> Result<Self, WorkspaceReportError> {
        let value = Self { nodes, edges };
        value.bytes_for::<usize>()?;
        Ok(value)
    }
    /// Canonical-node and DFS-frame bound.
    pub fn nodes(self) -> usize {
        self.nodes
    }
    /// Ordered alias-entry bound.
    pub fn edges(self) -> usize {
        self.edges
    }
    /// Five heap requests for the concrete flat representation.
    pub fn heap_bytes(self) -> Result<usize, WorkspaceReportError> {
        self.bytes_for::<usize>()
    }
    /// Concrete flat workspace, result and partial-error control shells. A
    /// runtime must additionally price its source/charge and enclosing wrapper.
    pub fn control_bytes(self) -> Result<usize, WorkspaceReportError> {
        [
            std::mem::size_of::<WorkspaceReportWorkspace>(),
            std::mem::size_of::<WorkspaceReportConstructionError>(),
            std::mem::size_of::<Result<WorkspaceReportWorkspace, WorkspaceReportConstructionError>>(
            ),
            std::mem::size_of::<WorkspaceReportScalars>(),
            std::mem::size_of::<WorkspaceReportGraph<'_>>(),
            std::mem::size_of::<WorkspaceReportInputs<'_>>(),
            std::mem::size_of::<source::Flat<'_>>(),
            std::mem::size_of::<source::Facts>(),
            std::mem::size_of::<Frame>(),
            std::mem::size_of::<Result<WorkspaceReportScalars, WorkspaceReportError>>(),
            std::mem::size_of::<Result<usize, usize>>(),
            std::mem::size_of::<[usize; 4]>(),
            std::mem::size_of::<[&usize; 2]>(),
            // Live merge cursors and the insertion/final-coalescing caller.
            std::mem::size_of::<[usize; 10]>(),
            std::mem::size_of::<std::cmp::Ordering>(),
            std::mem::size_of::<bool>(),
            std::mem::size_of::<WorkspaceStoragePopulation>(),
            std::mem::size_of::<usize>(),
        ]
        .into_iter()
        .try_fold(0usize, |a, b| {
            a.checked_add(b).ok_or(WorkspaceReportError::Overflow)
        })
    }
    fn bytes_for<R>(self) -> Result<usize, WorkspaceReportError> {
        [
            Layout::array::<Node<R>>(self.nodes),
            Layout::array::<usize>(self.edges),
            Layout::array::<Frame>(self.nodes),
            Layout::array::<u8>(self.nodes),
            Layout::array::<usize>(self.nodes),
        ]
        .into_iter()
        .try_fold(0usize, |a, b| {
            a.checked_add(b.map_err(|_| WorkspaceReportError::Overflow)?.size())
                .ok_or(WorkspaceReportError::Overflow)
        })
    }
}

/// Five pre-reserved flat destinations; all source and error custody remains
/// with their enclosing owner. No reduction resizes or allocates a destination.
#[derive(Debug)]
pub struct WorkspaceReportWorkspace {
    scratch: Scratch<usize>,
}
/// Actual successfully allocated prefixes survive a reserve failure. The
/// caller's enclosing original owner must retain its charge until this retires.
#[derive(Debug)]
pub struct WorkspaceReportConstructionError {
    scratch: Scratch<usize>,
    cause: ConstructionCause,
}
#[derive(Debug, thiserror::Error)]
enum ConstructionCause {
    #[error("{0}")]
    Layout(#[source] WorkspaceReportError),
    #[error("workspace report destination {destination} reserve failed")]
    Reserve {
        destination: usize,
        #[source]
        source: TryReserveError,
    },
}
impl fmt::Display for WorkspaceReportConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for WorkspaceReportConstructionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
impl WorkspaceReportConstructionError {
    /// Actual retained vector payload capacities, without source/control shells.
    pub fn retained_heap_bytes(&self) -> usize {
        self.scratch.retained_bytes()
    }
    /// Failed actual reserve, or no reserve for invalid arithmetic/layout.
    pub fn destination(&self) -> Option<usize> {
        match &self.cause {
            ConstructionCause::Reserve { destination, .. } => Some(*destination),
            _ => None,
        }
    }
}
impl WorkspaceReportWorkspace {
    /// Makes each actual destination reserve exactly once.
    pub fn new(layout: WorkspaceReportLayout) -> Result<Self, WorkspaceReportConstructionError> {
        Scratch::new(layout, None)
            .map(|scratch| Self { scratch })
            .map_err(|(scratch, cause)| WorkspaceReportConstructionError { scratch, cause })
    }
    /// Evaluates source-relative metadata into existing buffers, without native
    /// queries, a callback, source construction or another allocation.
    pub fn report(
        &mut self,
        graph: WorkspaceReportGraph<'_>,
        input: WorkspaceReportInputs<'_>,
    ) -> Result<WorkspaceReportScalars, WorkspaceReportError> {
        let source = source::Flat::new(graph, input)?;
        self.scratch.report(&source)
    }
    /// Actual five payload capacities retained by this workspace.
    pub fn retained_heap_bytes(&self) -> usize {
        self.scratch.retained_bytes()
    }
}

#[derive(Debug)]
pub(super) struct Lifecycle {
    started: Cell<bool>,
    counts: Cell<Option<(usize, usize)>>,
}
impl Lifecycle {
    pub(super) fn new() -> Self {
        Self::imported(0)
    }
    fn allocation_bytes() -> Result<usize, WorkspaceReportError> {
        Layout::new::<[Cell<usize>; 2]>()
            .extend(Layout::new::<Self>())
            .map(|(layout, _)| layout.pad_to_align().size())
            .map_err(|_| WorkspaceReportError::Overflow)
    }
    pub(super) fn imported(nodes: usize) -> Self {
        Self {
            started: Cell::new(false),
            counts: Cell::new(Some((nodes, 0))),
        }
    }
    pub(super) fn get(&self) -> bool {
        self.started.get()
    }
    pub(super) fn set(&self, value: bool) {
        self.started.set(value)
    }
    pub(super) fn record(&self, edges: usize) {
        self.counts.set(
            self.counts
                .get()
                .and_then(|(n, e)| Some((n.checked_add(1)?, e.checked_add(edges)?))),
        );
    }
    fn layout(&self) -> Result<WorkspaceReportLayout, WorkspaceReportError> {
        let (nodes, edges) = self.counts.get().ok_or(WorkspaceReportError::Overflow)?;
        let layout = WorkspaceReportLayout { nodes, edges };
        layout.bytes_for::<Rc<Storage>>()?;
        Ok(layout)
    }
}
fn legacy(error: WorkspaceReportError) -> Error {
    match error {
        WorkspaceReportError::Overflow => workspace_overflow("workspace allocation sum overflow"),
        _ => Error::backend_source(error),
    }
}
impl WorkspaceContext {
    pub(super) fn new_storage(
        &self,
        bytes: Option<u64>,
        possible_aliases: Vec<Rc<Storage>>,
    ) -> Rc<Storage> {
        let storage = Rc::new(Storage {
            maximum_allocations: usize::from(bytes != Some(0)),
            bytes,
            possible_aliases,
        });
        self.tracing_started.record(storage.possible_aliases.len());
        storage
    }
    /// Allocation request for the existing shared tracing gate and cumulative
    /// counters, including its concrete Rc header/alignment. This is only the
    /// fifth context owner; the other owners and dynamic payloads remain separate.
    pub fn report_lifecycle_bytes() -> Result<usize, WorkspaceReportError> {
        Lifecycle::allocation_bytes()
    }
    /// Cumulative ordinary metadata capacity, including retired previous spans.
    /// This is diagnostic geometry, not an original source or admission bound.
    pub fn report_workspace_layout(&self) -> Result<WorkspaceReportLayout, WorkspaceReportError> {
        self.tracing_started.layout()
    }
    pub(super) fn fixed_report(
        &self,
        retained: &[WorkspaceTensor],
    ) -> Result<WorkspaceTraceReport, Error> {
        if self.facts.is_some() {
            return Err(WorkspaceMetadataError::ReportClone.into());
        }
        for value in retained {
            value.validate_context(self)?;
        }
        let layout = self.tracing_started.layout().map_err(legacy)?;
        let mut scratch = match Scratch::<Rc<Storage>>::new(layout, None) {
            Ok(s) => s,
            Err((prefix, cause)) => {
                drop(prefix);
                return Err(match cause {
                    ConstructionCause::Layout(e) => legacy(e),
                    ConstructionCause::Reserve { source, .. } => Error::backend_source(source),
                });
            }
        };
        // Destinations outlive both RefCell loans, including every error exit.
        let result = (|| {
            let trace = self.trace.borrow();
            let borrowed = self.borrowed.borrow();
            let source = Ordinary {
                trace: &trace,
                retained,
                borrowed: borrowed.as_ref(),
            };
            let v = scratch.report(&source)?;
            Ok(WorkspaceTraceReport {
                residual: v.residual.map(|r| WorkspaceResidualReport {
                opening_storage: r.opening_storage,
                closing_storage: r.closing_storage,
                    borrowed_storage: borrowed
                        .as_ref()
                        .expect("residual has an installed selection")
                        .clone(),
                    total_bytes: r.total_bytes,
                    retained_bytes: r.retained_bytes,
                    displaced_bytes: r.displaced_bytes,
                    transient_bytes: r.transient_bytes,
                }),
                closing_storage: v.closing_storage,
                opening_storage: v.opening_storage,
                state: v.state,
                total_bytes: v.total_bytes,
                retained_bytes: v.retained_bytes,
                transient_bytes: v.transient_bytes,
                tensor_buffers: v.tensor_buffers,
                host_workspace_bytes: v.host_workspace_bytes,
                tensor_handle_clones: self.identity.tensor_handle_clones.get(),
                fact_construction_bytes: self.identity.fact_bytes.get(),
                metadata_construction: None,
                operations: trace.operations.clone(),
                unpriced_operations: trace.missing.clone(),
                unpriced_host_operations: trace.missing_host.clone(),
                assumptions: trace.assumptions.iter().cloned().collect(),
            })
        })();
        #[cfg(test)]
        tests::probe_retirement(self);
        drop(scratch);
        result.map_err(legacy)
    }
}

#[cfg(test)]
mod tests;

impl WorkspaceContext {
    /// Same five Rc-root report destinations and fixed constructor/error
    /// controls as report_scalars. Counts cover the cumulative context graph,
    /// including roots created in prior spans and now retired.
    pub fn report_metadata_bytes(
        nodes: usize,
        edges: usize,
    ) -> Result<usize, WorkspaceMetadataError> {
        let layout = WorkspaceReportLayout::new(nodes, edges)?;
        let parts = [
            layout
                .bytes_for::<Rc<Storage>>()
                .map_err(WorkspaceMetadataError::from)?,
            layout
                .control_bytes()
                .map_err(WorkspaceMetadataError::from)?,
            std::mem::size_of::<Scratch<Rc<Storage>>>(),
            std::mem::size_of::<
                Result<Scratch<Rc<Storage>>, (Scratch<Rc<Storage>>, ConstructionCause)>,
            >(),
            std::mem::size_of::<Ordinary<'_>>(),
            std::mem::size_of::<WorkspaceTraceReport>(),
            std::mem::size_of::<Result<WorkspaceTraceReport, Error>>(),
            std::mem::size_of::<Option<WorkspaceMetadataEnvelope>>(),
            std::mem::size_of::<Result<WorkspaceReportScalars, Error>>(),
            Error::retained_source_control_bytes::<ConstructionCause>()
                .ok_or(WorkspaceMetadataError::Overflow)?,
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)
    }

    /// Uses the same scalar graph reducer without cloning the trace's owned
    /// operations, shapes or diagnostic text. All five reducer destinations are
    /// checked against the context allowance before the first reserve.
    pub fn report_scalars(
        &self,
        retained: &[WorkspaceTensor],
    ) -> Result<WorkspaceReportScalars, Error> {
        if self.trace.borrow().report_finished {
            return Err(WorkspaceMetadataError::ReportFinished.into());
        }
        for value in retained {
            value.validate_context(self)?;
        }
        let layout = self
            .tracing_started
            .layout()
            .map_err(WorkspaceMetadataError::from)?;
        let bytes = Self::report_metadata_bytes(layout.nodes(), layout.edges())?;
        self.charge_metadata(bytes)?;
        if self.facts.is_some() {
            let reports = self
                .identity
                .metadata_reports
                .get()
                .and_then(|value| value.checked_add(1))
                .ok_or(WorkspaceMetadataError::Overflow)?;
            self.identity.metadata_reports.set(Some(reports));
        }
        let mut scratch =
            Scratch::<Rc<Storage>>::new(layout, None).map_err(|(prefix, cause)| {
                drop(prefix);
                if self.facts.is_some() {
                    Error::backend_retained_source(cause)
                } else {
                    Error::backend_source(cause)
                }
            })?;
        let trace = self.trace.borrow();
        let borrowed = self.borrowed.borrow();
        scratch
            .report(&Ordinary {
                trace: &trace,
                retained,
                borrowed: borrowed.as_ref(),
            })
            .map_err(|cause| {
                if self.facts.is_some() {
                    WorkspaceMetadataError::Report(cause).into()
                } else {
                    legacy(cause)
                }
            })
    }

    /// Finishes a report by moving its already-owned diagnostics. The shared
    /// scalar equations and ordering are identical to `report`; this consumes
    /// diagnostic rows rather than cloning their owned payloads. Begin the next
    /// span before further operations. A legacy context keeps its old behavior.
    pub fn finish_report(
        &self,
        retained: &[WorkspaceTensor],
    ) -> Result<WorkspaceTraceReport, Error> {
        if self.facts.is_none() {
            return self.fixed_report(retained);
        }
        let v = self.report_scalars(retained)?;
        let mut trace = self.trace.borrow_mut();
        let borrowed = self.borrowed.borrow();
        let report = WorkspaceTraceReport {
            residual: v.residual.map(|r| WorkspaceResidualReport {
                opening_storage: r.opening_storage,
                closing_storage: r.closing_storage,
                borrowed_storage: borrowed
                    .as_ref()
                    .expect("residual has installed selection")
                    .clone(),
                total_bytes: r.total_bytes,
                retained_bytes: r.retained_bytes,
                displaced_bytes: r.displaced_bytes,
                transient_bytes: r.transient_bytes,
            }),
            closing_storage: v.closing_storage,
            opening_storage: v.opening_storage,
            state: v.state,
            total_bytes: v.total_bytes,
            retained_bytes: v.retained_bytes,
            transient_bytes: v.transient_bytes,
            tensor_buffers: v.tensor_buffers,
            host_workspace_bytes: v.host_workspace_bytes,
            tensor_handle_clones: self.identity.tensor_handle_clones.get(),
            fact_construction_bytes: self.identity.fact_bytes.get(),
            metadata_construction: self.metadata_census(),
            operations: std::mem::take(&mut trace.operations),
            unpriced_operations: std::mem::take(&mut trace.missing),
            unpriced_host_operations: std::mem::take(&mut trace.missing_host),
            assumptions: std::mem::take(&mut trace.assumptions),
        };
        trace.report_finished = true;
        self.identity.fact_bytes.set(Some(0));
        self.identity.metadata_bytes.set(Some(0));
        self.identity.metadata_reports.set(Some(0));
        Ok(report)
    }
}

//! The same copy program with finite facts and the shared flat report reducer.
use super::*;
use std::{alloc::Layout, collections::TryReserveError, marker::PhantomData, mem::size_of};

/// Preparation failure retaining the selected fact or allocation cause.
#[derive(Debug, thiserror::Error)]
pub enum WorkspaceCopyPreparationError<E> {
    /// The original context refused funded placement/report metadata.
    #[error(transparent)]
    Metadata(#[from] Error),
    /// The actual imported source set does not match its context and selection.
    #[error(transparent)]
    Source(#[from] WorkspaceCopyError),
    /// The selected finite producer rejected this operation.
    #[error("selected workspace facts rejected isolated copy")]
    Mechanism(#[source] E),
    /// A requested finite destination could not be allocated.
    #[error(transparent)]
    Reserve(#[from] TryReserveError),
    /// A population, layout or byte sum overflowed.
    #[error("isolated-copy preparation layout overflow")]
    Overflow,
    /// Counts changed, the accepted layout differs, or emitted ranges are invalid.
    #[error("isolated-copy preparation facts differ from their finite layout")]
    Facts,
    /// The shared tensor/host declaration validator rejected an emitted effect.
    #[error(transparent)]
    Effect(#[from] WorkspaceEffectError),
    /// A shape cannot be represented.
    #[error(transparent)]
    Geometry(#[from] WorkspaceLayoutError),
    /// Emitted assumption text is not UTF-8; its buffer remains in the cause.
    #[error(transparent)]
    Utf8(#[from] std::string::FromUtf8Error),
    /// The shared report rejected its source or finite destinations.
    #[error(transparent)]
    Report(#[from] WorkspaceReportError),
    /// The shared report retains its failed construction prefix.
    #[error(transparent)]
    ReportConstruction(#[from] WorkspaceReportConstructionError),
}

type Result<T, E> = std::result::Result<T, WorkspaceCopyPreparationError<E>>;
fn add<E>(a: usize, b: usize) -> Result<usize, E> {
    a.checked_add(b)
        .ok_or(WorkspaceCopyPreparationError::Overflow)
}
fn mul<E>(a: usize, b: usize) -> Result<usize, E> {
    a.checked_mul(b)
        .ok_or(WorkspaceCopyPreparationError::Overflow)
}
fn array_bytes<T, E>(count: usize) -> Result<usize, E> {
    Layout::array::<T>(count)
        .map(|l| l.size())
        .map_err(|_| WorkspaceCopyPreparationError::Overflow)
}
fn reserve<T, E>(count: usize) -> Result<Vec<T>, E> {
    array_bytes::<T, E>(count)?;
    let mut values = Vec::new();
    values.try_reserve_exact(count)?;
    Ok(values)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Counts {
    sources: usize,
    operations: usize,
    axes: usize,
    aliases: usize,
    alias_scratch: usize,
    assumptions: usize,
    text_bytes: usize,
}

/// Allocation-free accumulator accepting one temporarily borrowed shape at a time.
///
/// A native caller can keep its descriptor guard local to `push_source`. This
/// plans metadata only: source identity, admission and the concrete immutable
/// fact provider remain the caller's responsibility. A rejected push leaves the
/// accumulator unchanged, and no shape or operation is retained here.
#[derive(Debug)]
pub struct WorkspaceCopyPreparationLayoutBuilder<E> {
    counts: Counts,
    error: PhantomData<fn() -> E>,
}
impl<E> Default for WorkspaceCopyPreparationLayoutBuilder<E> {
    fn default() -> Self {
        Self::new()
    }
}
impl<E> WorkspaceCopyPreparationLayoutBuilder<E> {
    /// Starts a fixed-program layout without allocating a context or projection.
    pub fn new() -> Self {
        Self {
            counts: Counts::default(),
            error: PhantomData,
        }
    }

    /// Adds the two actual copy operations using the supplied finite facts.
    pub fn push_source<F: WorkspaceFactMechanisms<Error = E>>(
        &mut self,
        source: WorkspaceLayoutView<'_>,
        facts: &F,
    ) -> Result<(), E> {
        source.bytes()?;
        array_bytes::<i32, E>(source.shape().len())?;
        let counts = RefCell::new(self.counts);
        isolated_copy(Planner {
            counts: &counts,
            source,
            facts,
        })?;
        let mut next = counts.into_inner();
        next.sources = add(next.sources, 1)?;
        next.axes = add(next.axes, source.shape().len())?;
        self.counts = next;
        Ok(())
    }

    /// Checks every actual destination using an upper bound on distinct source
    /// roots. The ordered source count itself is always a valid root bound.
    pub fn finish(self, maximum_source_roots: usize) -> Result<WorkspaceCopyPreparationLayout, E> {
        if maximum_source_roots > self.counts.sources {
            return Err(WorkspaceCopyPreparationError::Facts);
        }
        WorkspaceCopyPreparationLayout::new::<E>(self.counts, maximum_source_roots)
    }
}

struct Planner<'a, F> {
    counts: &'a RefCell<Counts>,
    source: WorkspaceLayoutView<'a>,
    facts: &'a F,
}
impl<F: WorkspaceFactMechanisms> Planner<'_, F> {
    fn operation(&self, kind: WorkspaceOperationKind) -> Result<(), F::Error> {
        let layouts = [self.source];
        let operation = WorkspaceOperationView {
            kind: kind.as_view(),
            inputs: WorkspaceLayoutList::Views(&layouts),
            outputs: WorkspaceLayoutList::Views(&layouts),
        };
        let tensor = self
            .facts
            .operation_facts(operation)
            .map_err(WorkspaceCopyPreparationError::Mechanism)?;
        let host = self
            .facts
            .host_facts(operation)
            .map_err(WorkspaceCopyPreparationError::Mechanism)?;
        let mut count = self.counts.borrow_mut();
        count.operations = add(count.operations, 1)?;
        if let Some(fact) = tensor {
            if fact.layout.outputs != 1 {
                return Err(WorkspaceCopyPreparationError::Facts);
            }
            array_bytes::<usize, F::Error>(fact.layout.aliases)?;
            array_bytes::<u8, F::Error>(fact.layout.assumption_bytes)?;
            count.aliases = add(count.aliases, fact.layout.aliases)?;
            count.alias_scratch = count.alias_scratch.max(fact.layout.aliases);
            count.text_bytes = add(count.text_bytes, fact.layout.assumption_bytes)?;
            count.assumptions = add(count.assumptions, 1)?;
        }
        if let Some(fact) = host {
            array_bytes::<u8, F::Error>(fact.assumption_bytes)?;
            count.text_bytes = add(count.text_bytes, fact.assumption_bytes)?;
            count.assumptions = add(count.assumptions, 1)?;
        }
        Ok(())
    }
}
impl<F: WorkspaceFactMechanisms> IsolatedCopyMechanism for Planner<'_, F> {
    type Value = ();
    type Error = WorkspaceCopyPreparationError<F::Error>;
    fn contiguous(&self) -> Result<(), F::Error> {
        self.operation(WorkspaceOperationKind::Contiguous)
    }
    fn deep_copy(&self, (): ()) -> Result<(), F::Error> {
        self.operation(WorkspaceOperationKind::DeepCopy)
    }
}

/// Checked finite preparation requests, independent of any physical source.
///
/// Heap bytes are the sum of the actual requested Vec/String capacities,
/// including scratch and failed prefixes. They exclude allocator-private
/// overhead and the caller's source projections, retained fact provider and
/// custody. No construction below grows beyond these populations. Controls
/// name this producer's concrete frames and the shared report frames; they do
/// not describe arbitrary caller stack or native execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceCopyPreparationLayout {
    counts: Counts,
    roots: usize,
    report: WorkspaceReportLayout,
    heap_bytes: usize,
    controls: usize,
}
impl WorkspaceCopyPreparationLayout {
    fn new<E>(counts: Counts, roots: usize) -> Result<Self, E> {
        let nodes = add(roots, counts.operations)?;
        let closing = mul(counts.sources, 2)?;
        if counts.operations != closing {
            return Err(WorkspaceCopyPreparationError::Facts);
        }
        let report = WorkspaceReportLayout::new(nodes, counts.aliases)?;
        let mut heap_bytes = report.heap_bytes()?;
        // Final source diagnostics plus two one-input/one-output operations per
        // source own five shape copies in total. Neither the report nor the
        // fixed graph clones those final operation/layout diagnostics.
        for bytes in [
            array_bytes::<WorkspaceReportNode, E>(nodes)?,
            array_bytes::<usize, E>(counts.aliases)?,
            array_bytes::<usize, E>(counts.alias_scratch)?,
            array_bytes::<usize, E>(counts.operations)?,
            array_bytes::<usize, E>(closing)?,
            array_bytes::<usize, E>(roots)?,
            array_bytes::<WorkspaceLayout, E>(counts.sources)?,
            array_bytes::<WorkspaceOperation, E>(counts.operations)?,
            array_bytes::<WorkspaceLayout, E>(mul(counts.operations, 2)?)?,
            array_bytes::<i32, E>(mul(counts.axes, 5)?)?,
            mul(
                mul(counts.sources, 5)?,
                WorkspaceLayout::shape_owner_bytes()
                    .ok_or(WorkspaceCopyPreparationError::Overflow)?,
            )?,
            array_bytes::<usize, E>(counts.operations)?,
            array_bytes::<usize, E>(counts.operations)?,
            array_bytes::<String, E>(counts.assumptions)?,
            counts.text_bytes,
        ] {
            heap_bytes = add(heap_bytes, bytes)?;
        }
        let mut controls = add(
            report.control_bytes()?,
            mul(
                mul(counts.sources, 5)?,
                WorkspaceLayout::shape_control_bytes()
                    .ok_or(WorkspaceCopyPreparationError::Overflow)?,
            )?,
        )?;
        for bytes in [
            size_of::<Self>(),
            size_of::<WorkspaceCopyPreparationLayoutBuilder<E>>(),
            size_of::<RefCell<Counts>>(),
            size_of::<Planner<'_, ()>>(),
            size_of::<WorkspaceIsolatedCopyPreparation<'_, ()>>(),
            size_of::<RefCell<Build<'_, ()>>>(),
            size_of::<Result<Build<'_, ()>, E>>(),
            size_of::<Result<WorkspaceIsolatedCopyPreparation<'_, ()>, E>>(),
            size_of::<FiniteCopy<'_, '_, ()>>(),
            size_of::<WorkspaceIsolatedCopyPlan>(),
            size_of::<WorkspaceTraceReport>(),
            size_of::<WorkspaceCopyPreparationError<E>>(),
            size_of::<Result<WorkspaceIsolatedCopyPlan, E>>(),
            size_of::<Result<WorkspaceCopyPreparationLayout, E>>(),
            size_of::<WorkspaceOperation>(),
            size_of::<WorkspaceLayout>(),
            size_of::<WorkspaceOperationView<'_>>(),
            size_of::<WorkspaceReportNode>(),
            size_of::<WorkspaceOutputEffect>(),
            size_of::<(Option<String>, Option<String>)>(),
            size_of::<Option<WorkspaceOperationFacts>>(),
            size_of::<Option<WorkspaceHostFacts>>(),
            size_of::<Result<Vec<WorkspaceLayout>, E>>(),
            size_of::<Result<WorkspaceLayout, E>>(),
            size_of::<Result<usize, E>>(),
            size_of::<(std::ops::Range<usize>, usize)>(),
            size_of::<[WorkspaceLayoutView<'_>; 1]>(),
            size_of::<[WorkspaceOutputEffect; 1]>(),
            size_of::<WorkspaceEffectDestination<'_>>(),
            size_of::<WorkspaceHostDestination<'_>>(),
            size_of::<Vec<u8>>(),
            size_of::<Result<Vec<u8>, E>>(),
            size_of::<Result<String, E>>(),
            size_of::<std::result::Result<String, std::string::FromUtf8Error>>(),
            size_of::<std::result::Result<Option<WorkspaceOperationFacts>, E>>(),
            size_of::<std::result::Result<Option<WorkspaceHostFacts>, E>>(),
            size_of::<std::slice::Iter<'_, WorkspaceTensor>>(),
            size_of::<std::slice::Iter<'_, WorkspaceExistingStorage>>(),
            size_of::<std::slice::Iter<'_, String>>(),
        ] {
            controls = add(controls, bytes)?;
        }
        add::<E>(heap_bytes, controls)?;
        Ok(Self {
            counts,
            roots,
            report,
            heap_bytes,
            controls,
        })
    }
    /// Ordered sources, including distinct requests sharing one backing.
    pub fn source_count(self) -> usize {
        self.counts.sources
    }
    /// Maximum distinct physical roots admitted to this metadata preparation.
    pub fn maximum_source_roots(self) -> usize {
        self.roots
    }
    /// Sum of the finite requested metadata and scratch heap capacities.
    pub fn requested_heap_bytes(self) -> usize {
        self.heap_bytes
    }
    /// Producer and shared-reducer control frames for the selected error type.
    pub fn control_bytes(self) -> usize {
        self.controls
    }
    /// Checked sum of requested heap storage and named producer controls.
    pub fn total_bytes(self) -> usize {
        self.heap_bytes + self.controls
    }
}

/// Borrowed validated inputs for finite construction after host admission.
/// This contains no private context, tensor clones or ordinary mechanism call.
pub struct WorkspaceIsolatedCopyPreparation<'a, F> {
    context: &'a WorkspaceContext,
    borrowed: &'a WorkspaceBorrowedStorage,
    sources: &'a [WorkspaceTensor],
    facts: &'a F,
    layout: WorkspaceCopyPreparationLayout,
}
impl WorkspaceIsolatedCopyPlan {
    /// Validates sources and queries the finite layout without owning metadata.
    /// Native hooks can instead quote `WorkspaceCopyPreparationLayoutBuilder`
    /// before allocating even their input projections.
    pub fn prepare_finite<'a, F: WorkspaceFactMechanisms>(
        context: &'a WorkspaceContext,
        borrowed: &'a WorkspaceBorrowedStorage,
        sources: &'a [WorkspaceTensor],
        facts: &'a F,
    ) -> Result<WorkspaceIsolatedCopyPreparation<'a, F>, F::Error> {
        validate_sources(context, borrowed, sources)?;
        let mut builder = WorkspaceCopyPreparationLayoutBuilder::new();
        for source in sources {
            builder.push_source(source.layout.as_view(), facts)?;
        }
        let layout = builder.finish(borrowed.roots().len())?;
        Ok(WorkspaceIsolatedCopyPreparation {
            context,
            borrowed,
            sources,
            facts,
            layout,
        })
    }

    /// Uses a layout quoted from borrowed descriptors before input projection.
    /// The exact imported source set is authenticated against `context`; all
    /// actual populations are rechecked within the quoted root bound before
    /// any destination allocation. This does not mint source or funding proof.
    pub fn prepare_finite_with_layout<'a, F: WorkspaceFactMechanisms>(
        context: &'a WorkspaceContext,
        borrowed: &'a WorkspaceBorrowedStorage,
        sources: &'a [WorkspaceTensor],
        facts: &'a F,
        layout: WorkspaceCopyPreparationLayout,
    ) -> Result<WorkspaceIsolatedCopyPreparation<'a, F>, F::Error> {
        let preparation = WorkspaceIsolatedCopyPreparation {
            context,
            borrowed,
            sources,
            facts,
            layout,
        };
        preparation.validate()?;
        Ok(preparation)
    }
}
impl<F: WorkspaceFactMechanisms> WorkspaceIsolatedCopyPreparation<'_, F> {
    /// The prevalidated metadata requests; no native execution is included.
    pub fn layout(&self) -> WorkspaceCopyPreparationLayout {
        self.layout
    }
    fn validate(&self) -> Result<(), F::Error> {
        validate_sources(self.context, self.borrowed, self.sources)?;
        if self.borrowed.roots().len() > self.layout.roots {
            return Err(WorkspaceCopyPreparationError::Facts);
        }
        let mut builder = WorkspaceCopyPreparationLayoutBuilder::new();
        for source in self.sources {
            builder.push_source(source.layout.as_view(), self.facts)?;
        }
        if builder.finish(self.layout.roots)? != self.layout {
            return Err(WorkspaceCopyPreparationError::Facts);
        }
        Ok(())
    }
    /// Constructs the same isolated-copy program in finite supplied graph and
    /// report destinations. Known effects use only `WorkspaceFactMechanisms`;
    /// missing effects remain incomplete, with no ordinary fallback.
    pub fn construct(self) -> Result<WorkspaceIsolatedCopyPlan, F::Error> {
        self.validate()?;
        let mut build = Build::new(self.facts, self.context, self.layout)?;
        for root in self.borrowed.roots() {
            build.borrowed.push(build.nodes.len());
            build.nodes.push(WorkspaceReportNode {
                bytes: root.storage.bytes,
                alias_start: 0,
                alias_count: 0,
            });
            build.placements.push(
                self.context
                    .copy_placement(root.storage.placement.as_ref())?,
            );
            build.host_controls.push(root.storage.host_control_bytes);
        }
        for source in self.sources {
            let root = self
                .borrowed
                .roots()
                .iter()
                .position(|root| Rc::ptr_eq(&root.storage, &source.storage))
                .ok_or(WorkspaceCopyPreparationError::Facts)?;
            build.closing.push(root);
            build.layouts.push(owned_layout(source.layout.as_view())?);
        }
        let build = RefCell::new(build);
        for (index, source) in self.sources.iter().enumerate() {
            let input = build.borrow().closing[index];
            let output = isolated_copy(FiniteCopy {
                build: &build,
                layout: source.layout.as_view(),
                input,
                index,
            })?;
            build.borrow_mut().closing.push(output);
        }
        let mut build = build.into_inner();
        // Fixed scalar cursors keep ordering scratch independent of the
        // selected standard-library sorting implementation. These are only
        // final diagnostic strings, never tensor or execution equations.
        for index in 1..build.assumptions.len() {
            let mut prior = index;
            while prior > 0 && build.assumptions[prior] < build.assumptions[prior - 1] {
                build.assumptions.swap(prior, prior - 1);
                prior -= 1;
            }
        }
        build.assumptions.dedup();
        let graph = WorkspaceReportGraph::new(&build.nodes, &build.edges)?;
        let mut report_workspace = WorkspaceReportWorkspace::new(self.layout.report)?;
        let physical_domains = self
            .facts
            .memory_topology()
            .map(|topology| {
                report_workspace
                    .report_domains_owned(
                        graph,
                        WorkspaceDomainReportInputs {
                            opening: Some(&build.closing[..self.sources.len()]),
                            allocations: &build.allocations,
                            closing: &build.closing,
                            borrowed: Some(&build.borrowed),
                            host_workspace: (!build.host_staging_incomplete).then_some(build.host),
                            tensor_complete: build.missing.is_empty(),
                        },
                        WorkspaceReportPlacements {
                            topology,
                            backings: &build.placements,
                            host_controls: Some(&build.host_controls),
                            scratch: &build.placed_scratch,
                        },
                        self.context,
                    )
                    .and_then(|mut report| {
                        for source in &build.scratch_sources {
                            report.append_scratch_population(source, self.context)?;
                        }
                        Ok(Some(report))
                    })
                    .or_else(|cause| match &cause.storage {
                        crate::ErrorStorage::WorkspaceMetadata(WorkspaceMetadataError::Report(
                            WorkspaceReportError::IncompletePlacement,
                        )) => Ok(None),
                        _ => Err(cause),
                    })
            })
            .transpose()?
            .flatten();
        let input = WorkspaceReportInputs {
            opening: Some(&build.closing[..self.sources.len()]),
            allocations: &build.allocations,
            closing: &build.closing,
            borrowed: Some(&build.borrowed),
            scratch: build.scratch,
            host_workspace: (!build.host_staging_incomplete).then_some(build.host),
            tensor_complete: build.missing.is_empty() && !build.scratch_overflow,
        };
        let extra = build
            .scratch_sources
            .iter()
            .filter(|source| source.domain_population().is_some())
            .count();
        let mut diagnostic_scratch = self.context.metadata_vec(
            build
                .placed_scratch
                .len()
                .checked_add(extra)
                .ok_or(WorkspaceCopyPreparationError::Overflow)?,
        )?;
        for row in &build.placed_scratch {
            diagnostic_scratch.push(WorkspaceScratchAllocation {
                bytes: row.bytes,
                maximum_allocations: row.maximum_allocations,
                host_control_bytes: row.host_control_bytes,
                placement: self.context.copy_placement(row.placement.as_ref())?,
            });
        }
        for source in build
            .scratch_sources
            .iter()
            .filter(|source| source.domain_population().is_some())
        {
            diagnostic_scratch.push(WorkspaceScratchAllocation {
                bytes: 0,
                maximum_allocations: None,
                host_control_bytes: source.host_control_bytes(),
                placement: None,
            });
        }
        let scalar = report_workspace.report_domain_diagnostics(
            graph,
            input,
            &build.host_controls,
            &diagnostic_scratch,
            physical_domains.is_some(),
        )?;
        let residual = scalar.residual.map(|value| WorkspaceResidualReport {
            opening_storage: value.opening_storage,
            closing_storage: value.closing_storage,
            borrowed_storage: self.borrowed.clone(),
            total_bytes: value.total_bytes,
            retained_bytes: value.retained_bytes,
            displaced_bytes: value.displaced_bytes,
            transient_bytes: value.transient_bytes,
        });
        let incremental_bytes = residual.as_ref().and_then(|r| r.total_bytes);
        let report = WorkspaceTraceReport {
            physical_domains,
            closing_storage: scalar.closing_storage,
            opening_storage: scalar.opening_storage,
            residual,
            state: scalar.state,
            total_bytes: scalar.total_bytes,
            retained_bytes: scalar.retained_bytes,
            transient_bytes: scalar.transient_bytes,
            tensor_buffers: scalar.tensor_buffers,
            host_workspace_bytes: scalar.host_workspace_bytes,
            tensor_handle_clones: Some(0),
            fact_construction_bytes: None,
            metadata_construction: None,
            operations: build.operations,
            unpriced_operations: build.missing,
            unpriced_host_operations: build.missing_host,
            assumptions: build.assumptions,
        };
        Ok(WorkspaceIsolatedCopyPlan {
            source_storage: self.borrowed.clone(),
            source_layouts: build.layouts,
            report,
            incremental_bytes,
        })
    }
}

fn owned_layout<E>(layout: WorkspaceLayoutView<'_>) -> Result<WorkspaceLayout, E> {
    let mut shape = reserve(layout.shape().len())?;
    shape.extend_from_slice(layout.shape());
    Ok(WorkspaceLayout::from_owned_shape(shape, layout.dtype()))
}
fn layout_slot<E>(layout: WorkspaceLayoutView<'_>) -> Result<Vec<WorkspaceLayout>, E> {
    let mut slots = reserve(1)?;
    slots.push(owned_layout(layout)?);
    Ok(slots)
}

struct Build<'a, F> {
    facts: &'a F,
    context: &'a WorkspaceContext,
    placements: Vec<Option<eredu_core::MemoryPlacement>>,
    host_controls: Vec<Option<u64>>,
    placed_scratch: Vec<WorkspaceScratchAllocation>,
    scratch_sources: Vec<WorkspaceAllocationPopulation>,
    limit: WorkspaceCopyPreparationLayout,
    nodes: Vec<WorkspaceReportNode>,
    edges: Vec<usize>,
    aliases: Vec<usize>,
    allocations: Vec<usize>,
    closing: Vec<usize>,
    borrowed: Vec<usize>,
    layouts: Vec<WorkspaceLayout>,
    operations: Vec<WorkspaceOperation>,
    missing: Vec<usize>,
    missing_host: Vec<usize>,
    assumptions: Vec<String>,
    scratch: u64,
    scratch_overflow: bool,
    host: u64,
    host_staging_incomplete: bool,
    text_used: usize,
    shapes_used: usize,
}
impl<'a, F: WorkspaceFactMechanisms> Build<'a, F> {
    fn new(
        facts: &'a F,
        context: &'a WorkspaceContext,
        limit: WorkspaceCopyPreparationLayout,
    ) -> Result<Self, F::Error> {
        let c = limit.counts;
        let mut aliases = reserve(c.alias_scratch)?;
        aliases.resize(c.alias_scratch, 0);
        Ok(Self {
            facts,
            context,
            placements: context.metadata_vec(limit.report.nodes())?,
            host_controls: context.metadata_vec(limit.report.nodes())?,
            placed_scratch: context.metadata_vec(c.operations)?,
            scratch_sources: context.metadata_vec(c.operations)?,
            limit,
            nodes: reserve(limit.report.nodes())?,
            edges: reserve(c.aliases)?,
            aliases,
            allocations: reserve(c.operations)?,
            closing: reserve(mul(c.sources, 2)?)?,
            borrowed: reserve(limit.roots)?,
            layouts: reserve(c.sources)?,
            operations: reserve(c.operations)?,
            missing: reserve(c.operations)?,
            missing_host: reserve(c.operations)?,
            assumptions: reserve(c.assumptions)?,
            scratch: 0,
            scratch_overflow: false,
            host: 0,
            host_staging_incomplete: false,
            text_used: 0,
            shapes_used: 0,
        })
    }
    fn text(&mut self, bytes: usize) -> Result<Vec<u8>, F::Error> {
        self.text_used = add(self.text_used, bytes)?;
        if self.text_used > self.limit.counts.text_bytes {
            return Err(WorkspaceCopyPreparationError::Facts);
        }
        let mut text = reserve(bytes)?;
        text.resize(bytes, 0);
        Ok(text)
    }
    fn operation(
        &mut self,
        kind: WorkspaceOperationKind,
        layout: WorkspaceLayoutView<'_>,
        input: usize,
    ) -> Result<usize, F::Error> {
        let ordinal = self.operations.len();
        if ordinal >= self.limit.counts.operations {
            return Err(WorkspaceCopyPreparationError::Facts);
        }
        let layouts = [layout];
        let operation = WorkspaceOperationView {
            kind: kind.as_view(),
            inputs: WorkspaceLayoutList::Views(&layouts),
            outputs: WorkspaceLayoutList::Views(&layouts),
        };
        let tensor = self
            .facts
            .operation_facts(operation)
            .map_err(WorkspaceCopyPreparationError::Mechanism)?;
        let mut outputs = [WorkspaceOutputEffect::Allocate(0)];
        let mut tensor_text = None;
        if let Some(facts) = tensor {
            if facts.layout.outputs != 1 || facts.layout.aliases > self.aliases.len() {
                return Err(WorkspaceCopyPreparationError::Facts);
            }
            let mut text = self.text(facts.layout.assumption_bytes)?;
            let written = self
                .facts
                .write_operation_facts(
                    operation,
                    WorkspaceEffectDestination {
                        outputs: &mut outputs,
                        aliases: &mut self.aliases[..facts.layout.aliases],
                        assumptions: &mut text,
                    },
                )
                .map_err(WorkspaceCopyPreparationError::Mechanism)?;
            if written != tensor {
                return Err(WorkspaceCopyPreparationError::Facts);
            }
            let text = String::from_utf8(text)?;
            validate_workspace_tensor_declaration(1, 1, &text)?;
            let view = outputs[0]
                .as_view(&self.aliases[..facts.layout.aliases])
                .ok_or(WorkspaceCopyPreparationError::Facts)?;
            validate_workspace_output_storage(view, layout, 0, 1)?;
            tensor_text = Some(text);
        }
        let host = self
            .facts
            .host_facts(operation)
            .map_err(WorkspaceCopyPreparationError::Mechanism)?;
        let mut host_text = None;
        if let Some(facts) = host {
            let mut text = self.text(facts.assumption_bytes)?;
            let written = self
                .facts
                .write_host_facts(
                    operation,
                    WorkspaceHostDestination {
                        assumptions: &mut text,
                    },
                )
                .map_err(WorkspaceCopyPreparationError::Mechanism)?;
            if written != host {
                return Err(WorkspaceCopyPreparationError::Facts);
            }
            let text = String::from_utf8(text)?;
            validate_workspace_host_assumptions(&text)?;
            host_text = Some(text);
        }
        let texts = usize::from(tensor_text.is_some()) + usize::from(host_text.is_some());
        if add::<F::Error>(self.assumptions.len(), texts)? > self.limit.counts.assumptions {
            return Err(WorkspaceCopyPreparationError::Facts);
        }
        self.shapes_used = add(self.shapes_used, mul(layout.shape().len(), 2)?)?;
        if self.shapes_used > mul::<F::Error>(self.limit.counts.axes, 4)? {
            return Err(WorkspaceCopyPreparationError::Facts);
        }
        let output = match tensor.map(|_| outputs[0]) {
            Some(WorkspaceOutputEffect::AliasInput(_)) => input,
            Some(WorkspaceOutputEffect::AliasOutput(_)) => {
                return Err(WorkspaceCopyPreparationError::Facts);
            }
            effect => {
                if self.nodes.len() == self.limit.report.nodes() {
                    return Err(WorkspaceCopyPreparationError::Facts);
                }
                let mut node = WorkspaceReportNode {
                    bytes: None,
                    alias_start: self.edges.len(),
                    alias_count: 0,
                };
                match effect {
                    Some(WorkspaceOutputEffect::Allocate(bytes)) => node.bytes = Some(bytes),
                    Some(WorkspaceOutputEffect::AllocateOrAliasInputs {
                        bytes,
                        alias_start,
                        alias_count,
                    }) => {
                        node.bytes = Some(bytes);
                        node.alias_count = alias_count;
                        if add::<F::Error>(self.edges.len(), alias_count)?
                            > self.limit.counts.aliases
                        {
                            return Err(WorkspaceCopyPreparationError::Facts);
                        }
                        let end = add(alias_start, alias_count)?;
                        let aliases = self
                            .aliases
                            .get(alias_start..end)
                            .ok_or(WorkspaceCopyPreparationError::Facts)?;
                        // Shared validation above established every index is the
                        // one actual input. Preserve duplicates and ordering.
                        for _ in aliases {
                            self.edges.push(input);
                        }
                    }
                    _ => {}
                }
                let index = self.nodes.len();
                self.nodes.push(node);
                self.placements.push(
                    self.context
                        .copy_placement(self.facts.output_placement(operation, 0))?,
                );
                self.host_controls
                    .push(self.facts.allocation_host_control_bytes(operation, 0));
                self.allocations.push(index);
                index
            }
        };
        let scratch = tensor.map_or(0, |f| f.scratch_bytes);
        let mut missing_scratch_controls = false;
        if let Some(rows) = self
            .facts
            .scratch_allocations(operation)
            .map_err(WorkspaceCopyPreparationError::Mechanism)?
        {
            self.scratch_sources.push(rows.clone());
            missing_scratch_controls |= rows.host_control_bytes().is_none();
            let rows = self.context.copy_scratch_population(rows, scratch)?;
            self.context
                .reserve_metadata_vec(&mut self.placed_scratch, rows.len())?;
            for row in rows {
                missing_scratch_controls |= row.host_control_bytes.is_none();
                self.placed_scratch.push(row);
            }
        } else if scratch != 0 {
            let host_control_bytes = self
                .facts
                .scratch_host_control_bytes(operation)
                .map_err(WorkspaceCopyPreparationError::Mechanism)?;
            missing_scratch_controls = host_control_bytes.is_none();
            self.placed_scratch.push(WorkspaceScratchAllocation {
                bytes: scratch,
                maximum_allocations: None,
                host_control_bytes,
                placement: self
                    .context
                    .copy_placement(self.facts.scratch_placement(operation))?,
            });
        }
        self.scratch = match self.scratch.checked_add(scratch) {
            Some(total) => total,
            None if self.facts.memory_topology().is_some() => {
                self.scratch_overflow = true;
                self.scratch
            }
            None => return Err(WorkspaceCopyPreparationError::Overflow),
        };
        self.host = self
            .host
            .checked_add(host.map_or(0, |f| f.bytes))
            .ok_or(WorkspaceCopyPreparationError::Overflow)?;
        let missing_host = host_text.is_none();
        self.host_staging_incomplete |= missing_host;
        if let Some(text) = host_text {
            self.assumptions.push(text);
        }
        if missing_host || missing_scratch_controls {
            self.missing_host.push(ordinal);
        }
        if let Some(text) = tensor_text {
            self.assumptions.push(text);
        } else {
            self.missing.push(ordinal);
        }
        self.operations.push(WorkspaceOperation {
            kind,
            inputs: layout_slot(layout)?,
            outputs: layout_slot(layout)?,
        });
        Ok(output)
    }
}

struct FiniteCopy<'a, 'b, F> {
    build: &'a RefCell<Build<'b, F>>,
    layout: WorkspaceLayoutView<'a>,
    input: usize,
    index: usize,
}
impl<F: WorkspaceFactMechanisms> IsolatedCopyMechanism for FiniteCopy<'_, '_, F> {
    type Value = usize;
    type Error = WorkspaceCopyPreparationError<F::Error>;
    fn contiguous(&self) -> Result<usize, F::Error> {
        self.build.borrow_mut().operation(
            WorkspaceOperationKind::Contiguous,
            self.layout,
            self.input,
        )
    }
    fn deep_copy(&self, contiguous: usize) -> Result<usize, F::Error> {
        let mut build = self.build.borrow_mut();
        let output = build.operation(WorkspaceOperationKind::DeepCopy, self.layout, contiguous)?;
        let node = &build.nodes[output];
        if node.bytes.is_some()
            && (output == contiguous || build.closing.contains(&output) || node.alias_count != 0)
        {
            return Err(WorkspaceCopyError::NonIndependentDestination { index: self.index }.into());
        }
        Ok(output)
    }
}

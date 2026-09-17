//! Allocation-free source-specific census of concrete Context producers.
use super::super::facts::owned::{FactPreparationError, WorkspaceFactPreparation};
use super::*;
use std::marker::PhantomData;

/// Fixed cold refusal. The selected fact cause remains typed and is not erased,
/// formatted, or allocated before the enclosing host admission.
#[derive(Debug, thiserror::Error)]
pub enum WorkspaceContextMetadataError<E> {
    /// The selected borrowed fact query failed.
    #[error("workspace source metadata facts were refused")]
    Mechanism(#[source] E),
    /// A concrete requested capacity cannot be represented.
    #[error(transparent)]
    Metadata(#[from] WorkspaceMetadataError),
    /// Source/output geometry is invalid.
    #[error(transparent)]
    Geometry(#[from] WorkspaceLayoutError),
    /// A finite fact header does not describe its declared output.
    #[error(transparent)]
    Effect(#[from] WorkspaceEffectError),
    /// This query does not yet describe an operation's owned options/temporaries.
    #[error("source metadata operation has no complete constructor query")]
    Unqualified,
}

type QueryResult<T, E> = Result<T, WorkspaceContextMetadataError<E>>;
fn overflow<E>() -> WorkspaceContextMetadataError<E> {
    WorkspaceMetadataError::Overflow.into()
}
fn add<E>(a: usize, b: usize) -> QueryResult<usize, E> {
    a.checked_add(b).ok_or_else(overflow)
}
fn known<E>(value: Option<usize>) -> QueryResult<usize, E> {
    value.ok_or_else(overflow)
}

#[derive(Clone, Copy, Debug, Default)]
struct Span {
    operations: usize,
    allocations: usize,
    missing: usize,
    missing_host: usize,
    assumptions: usize,
}

/// Pure cumulative constructor plan for source imports, simple copied values,
/// and their actual report spans. It borrows every operation only while pushing;
/// it owns no context, native descriptor, shape, source, or execution authority.
///
/// The query covers Context::execute plus the shared one-output tensor worker.
/// Only copy, reshape and dtype-cast primitives are qualified here. General
/// generation equations use their retained semantic producer census instead.
#[derive(Debug)]
pub struct WorkspaceContextMetadataBuilder<E> {
    envelope: WorkspaceMetadataEnvelope,
    nodes: usize,
    edges: usize,
    span: Span,
    error: PhantomData<fn() -> E>,
}
impl<E> Default for WorkspaceContextMetadataBuilder<E> {
    fn default() -> Self {
        Self::new()
    }
}
impl<E> WorkspaceContextMetadataBuilder<E> {
    /// Starts a metadata-only census with no allocations or grants.
    pub fn new() -> Self {
        Self {
            envelope: WorkspaceMetadataEnvelope::default(),
            nodes: 0,
            edges: 0,
            span: Span::default(),
            error: PhantomData,
        }
    }

    /// Adds a separately described concrete constructor in the same Context
    /// domain. The caller must query its actual worker, not an execution budget.
    pub fn add_context_bytes(&mut self, bytes: usize) -> QueryResult<(), E> {
        self.envelope.context = add(self.envelope.context, bytes)?;
        Ok(())
    }

    /// One imported shape and one possible distinct existing backing root.
    /// Repeated physical aliases may use fewer roots; no source identity is
    /// deduplicated or manufactured by this count-only plan.
    pub fn push_import(&mut self, layout: WorkspaceLayoutView<'_>) -> QueryResult<(), E> {
        layout.bytes()?;
        let bytes = add(
            known(WorkspaceLayout::construction_bytes(layout.shape().len()))?,
            known(WorkspaceExistingStorage::construction_bytes())?,
        )?;
        let nodes = add(self.nodes, 1)?;
        self.add_context_bytes(bytes)?;
        self.nodes = nodes;
        Ok(())
    }

    fn flush_span(&mut self) -> QueryResult<(), E> {
        let span = self.span;
        let amounts = [
            WorkspaceContext::metadata_vec_growth_bytes::<WorkspaceOperation>(0, span.operations),
            WorkspaceContext::metadata_vec_growth_bytes::<Rc<Storage>>(0, span.allocations),
            WorkspaceContext::metadata_vec_growth_bytes::<usize>(0, span.missing),
            WorkspaceContext::metadata_vec_growth_bytes::<usize>(0, span.missing_host),
            WorkspaceContext::metadata_vec_growth_bytes::<String>(0, span.assumptions),
        ];
        let mut bytes = 0;
        for amount in amounts {
            bytes = add(bytes, known(amount)?)?;
        }
        self.add_context_bytes(bytes)?;
        self.span = Span::default();
        Ok(())
    }

    /// Mirrors begin_state_span: previous trace payloads retire without refund,
    /// while all created graph nodes remain in the report capacity census.
    pub fn begin_state_span(&mut self, opening_roots: usize) -> QueryResult<(), E> {
        let bytes = known(WorkspaceContext::metadata_vec_growth_bytes::<Rc<Storage>>(
            0,
            opening_roots,
        ))?;
        self.flush_span()?;
        self.add_context_bytes(bytes)
    }

    /// Adds the same four report destinations and moves diagnostics, preserving
    /// cumulative node/edge bounds across spans. No report clone is assumed.
    pub fn finish_report(&mut self) -> QueryResult<(), E> {
        let bytes = WorkspaceContext::report_metadata_bytes(self.nodes, self.edges)?;
        self.flush_span()?;
        self.add_context_bytes(bytes)?;
        self.envelope.reports = add(self.envelope.reports, 1)?;
        Ok(())
    }

    /// The source-created graph stays in the same context's cumulative report
    /// layout during future equations. Add its four scratch-array increments for
    /// each actual recorded future report; fixed report controls are already in
    /// that program's separate envelope and are not added again.
    pub fn add_later_report_graph(&mut self, reports: usize) -> QueryResult<(), E> {
        let whole = WorkspaceContext::report_metadata_bytes(self.nodes, self.edges)?;
        let fixed = WorkspaceContext::report_metadata_bytes(0, 0)?;
        let graph = whole.checked_sub(fixed).ok_or_else(overflow)?;
        let bytes = graph.checked_mul(reports).ok_or_else(overflow)?;
        self.add_context_bytes(bytes)
    }

    /// Returns the represented cumulative allowance, including an unfinished
    /// trace's vector requests. This never certifies a whole request or source.
    pub fn finish(mut self) -> QueryResult<WorkspaceMetadataEnvelope, E> {
        self.flush_span()?;
        Ok(self.envelope)
    }
}

impl<E: std::error::Error + Send + Sync + 'static> WorkspaceContextMetadataBuilder<E> {
    /// Runs the existing isolated-copy sequence through a borrowed visitor.
    pub fn push_isolated_copy<M: WorkspaceFactMechanisms<Error = E> + ?Sized>(
        &mut self,
        source: WorkspaceLayoutView<'_>,
        facts: &M,
    ) -> QueryResult<(), E> {
        visit_isolated_copy_operations(source, &mut |operation| {
            self.push_operation(operation, facts)
        })
    }

    /// Describes the same concrete tensor primitive without emitting facts or
    /// constructing layouts. Known alias ranges can overlap, so each output's
    /// candidate-owner Vec is bounded by the complete emitted alias extent.
    pub fn push_operation<M: WorkspaceFactMechanisms<Error = E> + ?Sized>(
        &mut self,
        operation: WorkspaceOperationView<'_>,
        facts: &M,
    ) -> QueryResult<(), E> {
        if operation.inputs.len() != 1 || operation.outputs.len() != 1 {
            return Err(WorkspaceContextMetadataError::Unqualified);
        }
        let output = operation
            .outputs
            .get(0)
            .ok_or(WorkspaceContextMetadataError::Unqualified)?;
        output.bytes()?;
        let temporary = match operation.kind {
            WorkspaceOperationKindView::Contiguous | WorkspaceOperationKindView::DeepCopy => 0,
            WorkspaceOperationKindView::View("reshape") => known(
                WorkspaceContext::metadata_vec_bytes::<i32>(output.shape().len()),
            )?,
            WorkspaceOperationKindView::Elementwise("cast_u32") => 0,
            _ => return Err(WorkspaceContextMetadataError::Unqualified),
        };
        let prepared =
            WorkspaceFactPreparation::inspect(operation, facts).map_err(|cause| match cause {
                FactPreparationError::Mechanism(cause) => {
                    WorkspaceContextMetadataError::Mechanism(cause)
                }
                FactPreparationError::Overflow => overflow(),
                FactPreparationError::Effect(cause) => WorkspaceContextMetadataError::Effect(cause),
                // Inspection never reserves, emits, or converts owned UTF-8.
                FactPreparationError::Changed
                | FactPreparationError::Reserve(_)
                | FactPreparationError::Utf8(_) => WorkspaceContextMetadataError::Unqualified,
            })?;
        let (tensor, host) = prepared.headers();
        let aliases = tensor.map_or(0, |value| value.layout.aliases);
        let parts = [
            known(super::super::representation::operation_control_bytes())?,
            temporary,
            known(WorkspaceContext::metadata_vec_bytes::<WorkspaceLayout>(1))?,
            known(WorkspaceLayout::construction_bytes(output.shape().len()))?,
            known(WorkspaceContext::metadata_vec_bytes::<WorkspaceLayout>(
                operation.inputs.len(),
            ))?,
            known(WorkspaceContext::metadata_vec_bytes::<WorkspaceTensor>(1))?,
            known(WorkspaceExistingStorage::construction_bytes())?,
            known(WorkspaceContext::metadata_vec_bytes::<Rc<Storage>>(aliases))?,
        ];
        let bytes = parts.into_iter().try_fold(0, add)?;
        let next = Span {
            operations: add(self.span.operations, 1)?,
            allocations: add(self.span.allocations, 1)?,
            missing: add(self.span.missing, usize::from(tensor.is_none()))?,
            missing_host: add(self.span.missing_host, usize::from(host.is_none()))?,
            assumptions: add(
                self.span.assumptions,
                usize::from(tensor.is_some()) + usize::from(host.is_some()),
            )?,
        };
        let nodes = add(self.nodes, 1)?;
        let edges = add(self.edges, aliases)?;
        let facts = add(self.envelope.facts, prepared.charged_bytes())?;
        self.add_context_bytes(bytes)?;
        self.envelope.facts = facts;
        self.nodes = nodes;
        self.edges = edges;
        self.span = next;
        Ok(())
    }
}

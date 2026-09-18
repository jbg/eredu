//! Complete-request composition of cold, completed equation spans.

use super::{InferenceExecutionIdentity, InferenceRequest};
use crate::prefill::{PrefillChunk, PrefillDriver, PrefillError, PrefillExecutor, PrefillOutcome};
use eredu_core::{
    CapabilityError, Completion, ExecutionWorkspaceEstimate, GenerationCancellationToken,
    InferenceGeometry, OutputDemand, RuntimeStateEstimate, Submission, WorkspaceBound,
};
use eredu_nn::workspace::{WorkspaceBorrowedStorage, WorkspaceTraceReport};
use std::convert::Infallible;

mod metadata;
mod spans;
use metadata::Metadata;
pub use spans::{InferenceSpanWorkspacePlan, InferenceSpanWorkspaceRecord, SamplingWorkspacePlanCollector};

/// One equation invocation in the admitted prompt and output allowance.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum InferenceWorkspaceSpan {
    /// Actual sampling-only preparation or prediction in an admitted extension.
    /// This describes no model invocation, cache update, or prefill work.
    Sampling(super::SamplingWorkspacePhase),
    /// The same semantic span scheduled by ordinary/controlled prefill.
    Prefill(PrefillChunk),
    /// One cached decode invocation, including every possible cache-growth edge.
    Decode {
        /// Zero-based invocation within the admitted output allowance.
        index: u64,
        /// Absolute decoder position before this invocation.
        position: u64,
        /// A cached generation invocation projects its one new position.
        output: OutputDemand,
    },
}

/// Cold observer program used by ordinary equation workspace quotation.
///
/// Implementations borrow the actual admitted plan and retain every generated or
/// copied output until its original delivery/copy lifetime ends. These methods
/// describe metadata equations, not source registration, host capacity, native
/// work permission, or a complete capture quote. All tensors belong to the same
/// WorkspaceContext supplied by the enclosing ordinary traversal.
///
/// Runtime graph/observer bookkeeping and any non-tensor sidecar construction
/// remain separate preparation/host mechanism obligations.
pub trait InferenceWorkspaceObserver:
    crate::ActivationObserver<eredu_nn::workspace::WorkspaceTensor, eredu_nn::Error>
{
    /// Exact retained semantic companion actually used by this observer's p0
    /// program. Default None preserves whole-prompt attribution. A companion
    /// grants no source registration, native work, capacity or completion; the
    /// equation adapter revalidates its candidate and physical path owner.
    fn prefill_selection(&self) -> Option<crate::layered::BoundCaptureSelection<'_>> {
        None
    }

    /// Selects one eligible ordinary prediction before its equations. Prediction
    /// zero is prefill; decode index k maps to k+1. Phantom final decode spans and
    /// zero-output requests never invoke this callback. It may inspect schedules
    /// and shape metadata but must not trace operations: the original span opens
    /// afterward. Active partial prefill requires the exact retained companion
    /// above; default observers retain the typed whole-prompt attribution gate.
    fn begin_span(
        &mut self,
        geometry: InferenceGeometry,
        span: &InferenceWorkspaceSpan,
        prediction: u64,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<bool, eredu_nn::Error>;

    /// Records the selected publication owner's output on a receiving rank.
    /// The architecture invokes this only after its ordinary publication. A
    /// receiver may validate geometry and advance host receipt progression, but
    /// must not trace the owner's native capture transform. Default observers
    /// cannot certify that distinct ownership contract.
    fn observe_remote_output(
        &mut self,
        _value: &eredu_nn::workspace::WorkspaceTensor,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<(), eredu_nn::Error> {
        Err(context.metadata_source(InferenceObservationError::RemoteOutput))
    }

    /// Enumerates every still-live observer output, including earlier spans.
    /// The traversal borrows these actual metadata values into the next span's
    /// opening roots. Missing native source ownership or allocation facts cannot
    /// be supplied by this visitor; the enclosing closed provider owns that proof.
    fn visit_retained(&self, visitor: &mut dyn FnMut(&eredu_nn::workspace::WorkspaceTensor));

    /// Models the selected mechanism's completed span after its report is fixed.
    /// It may retire temporary native metadata roots; later visit_retained calls
    /// must include only resources whose real mechanism keeps them across spans.
    /// Scheduled host receipts retain their separate cumulative host custody.
    /// This is not native completion evidence and must not trace new operations.
    fn end_span(
        &mut self,
        _span: &InferenceWorkspaceSpan,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<(), eredu_nn::Error> {
        Ok(())
    }
}

/// Concrete observation/readout geometry rejected before affected equations.
#[derive(Debug, thiserror::Error)]
pub enum InferenceObservationError {
    /// The original request did not include required complete-sequence readout.
    #[error("workspace observer requires sequence readout in original request geometry")]
    SequenceReadoutRequired,
    /// Whole-prompt ordinary capture cannot be assigned to one partial chunk.
    #[error("ordinary prefill observation requires one complete prompt span")]
    PartialPrefill,
    /// A selected score-producing pass did not return logits.
    #[error("observed score-producing equation returned no logits")]
    MissingLogits,
    /// A text output cannot use the existing batch/sequence/vocabulary index.
    #[error("text logits must have positive batch, sequence and vocabulary axes")]
    InvalidLogitsGeometry,
    /// The selected observer has no source-derived remote receipt program.
    #[error("workspace observer has no selected remote output receipt producer")]
    RemoteOutput,
    /// Checked host composition or local prediction arithmetic overflowed.
    #[error("observed workspace arithmetic overflow")]
    Overflow,
}

/// Failure before a complete request workspace quote exists.
#[derive(Debug, thiserror::Error)]
pub enum InferenceWorkspaceError<E> {
    /// Invalid request geometry or overflowing accounting.
    #[error(transparent)]
    Geometry(#[from] CapabilityError),
    /// Fixed geometry validation used before checked metadata construction.
    #[error(transparent)]
    FixedGeometry(#[from] eredu_core::AdmissionPolicyError),
    /// A participating metadata constructor refused or failed. This exact
    /// source is already paid or inline, so callers must not format it again.
    #[error(transparent)]
    Metadata(#[from] eredu_nn::Error),
    /// The selected equation could not be inspected at this exact span.
    #[error("workspace inspection failed at {span:?}: {source}")]
    Quote {
        /// Invocation whose cold inspection failed.
        span: InferenceWorkspaceSpan,
        /// Original architecture or mechanism failure.
        source: E,
    },
}

/// A complete traversal of the requested geometry, possibly with explicit gaps.
/// Construction runs every prefill chunk and every allowed cached decode; callers
/// cannot accidentally compose only the first chunk or the final cache size.
#[derive(Debug, Clone)]
pub struct InferenceWorkspaceReport {
    geometry: InferenceGeometry,
    spans: InferenceSpanWorkspacePlan,
    completed_spans: u64,
    state_spans_complete: bool,
    transient: WorkspaceBound,
    retained_peak: Option<u64>,
    tensor_transient_peak: Option<u64>,
    host_peak: Option<u64>,
    peak_span: Option<InferenceWorkspaceSpan>,
    first_gap: Option<InferenceWorkspaceSpan>,
    residual: Option<InferenceResidualWorkspace>,
}

/// Incremental equation demand after excluding one exact borrowed-root set.
/// This is numerical metadata; registered ownership must independently prove
/// that the excluded allocations remain charged in the admission domain.
#[derive(Debug, Clone)]
pub struct InferenceResidualWorkspace {
    borrowed: WorkspaceBorrowedStorage,
    peak_bytes: Option<u64>,
    association_complete: bool,
}

impl InferenceResidualWorkspace {
    pub(super) fn has_complete_association(&self) -> bool {
        self.association_complete
    }
    /// Original immutable borrowed roots, shared by every inspected span.
    pub fn borrowed_storage(&self) -> &WorkspaceBorrowedStorage {
        &self.borrowed
    }

    /// Complete peak of unborrowed state, tensor and disjoint host storage.
    pub fn peak_bytes(&self) -> Option<u64> {
        self.peak_bytes
    }
}

impl InferenceWorkspaceReport {
    pub(super) fn has_complete_state_spans(&self) -> bool {
        self.state_spans_complete && (self.completed_spans != 0 || self.has_empty_schedule())
    }
    fn has_empty_schedule(&self) -> bool {
        self.geometry.input_positions == 0 && self.geometry.max_output_tokens == 0
    }
    /// Exact geometry inspected by this report.
    pub const fn geometry(&self) -> InferenceGeometry {
        self.geometry
    }
    /// Actual immutable span allocation diagnostics. Retention past admission
    /// requires the separate opt-in quote contribution; this alone grants nothing.
    pub fn span_workspace_plan(&self) -> &InferenceSpanWorkspacePlan {
        &self.spans
    }
    /// Number of safely separated equation spans inspected.
    pub const fn completed_spans(&self) -> u64 {
        self.completed_spans
    }
    /// Simultaneous tensor-plus-host transient peak, or an explicit missing bound.
    pub const fn transient(&self) -> &WorkspaceBound {
        &self.transient
    }
    /// Largest complete closing state backing, including older allocations.
    /// This must fit the enclosing persistent-state capacity estimate.
    pub const fn retained_peak_bytes(&self) -> Option<u64> {
        self.retained_peak
    }
    /// Independent tensor-only diagnostic; never substitutes for a missing host bound.
    pub const fn tensor_transient_peak_bytes(&self) -> Option<u64> {
        self.tensor_transient_peak
    }
    /// Largest disjoint host payload. Its peak need not coincide with the tensor peak.
    pub const fn host_peak_bytes(&self) -> Option<u64> {
        self.host_peak
    }
    /// Span with the largest known complete transient bound.
    pub const fn peak_span(&self) -> Option<&InferenceWorkspaceSpan> {
        self.peak_span.as_ref()
    }
    /// First invocation lacking either required workspace domain.
    pub const fn first_gap(&self) -> Option<&InferenceWorkspaceSpan> {
        self.first_gap.as_ref()
    }

    /// Separate identity-aware equation bound. Full diagnostics remain intact.
    pub fn residual_workspace(&self) -> Option<&InferenceResidualWorkspace> {
        self.residual.as_ref()
    }

    /// Adds the completed-span peak once to the existing admission estimate.
    /// Preparation, materialization, sampling and separately retained resources
    /// still require explicit enclosing bounds; missing components stay unknown.
    pub fn compose(
        &self,
        state: RuntimeStateEstimate,
        outside: ExecutionWorkspaceEstimate,
    ) -> Result<RuntimeStateEstimate, CapabilityError> {
        self.compose_metadata(state, outside, super::WorkspaceReportMetadata::ordinary())
            .map_err(super::WorkspaceReportError::into_capability)
    }

    /// The same complete composition using supplied owning report destinations.
    pub fn compose_metadata(
        &self,
        state: RuntimeStateEstimate,
        outside: ExecutionWorkspaceEstimate,
        metadata: super::WorkspaceReportMetadata<'_>,
    ) -> Result<RuntimeStateEstimate, super::WorkspaceReportError> {
        if outside.geometry != self.geometry {
            return Err(eredu_core::AdmissionPolicyError::InvalidConfiguration {
                field: "inference_workspace",
                detail: "outside workspace does not match inspected request geometry",
            }
            .into());
        }
        super::trace::with_transient_workspace_metadata(
            state,
            outside,
            metadata.clone_bound(&self.transient)?,
            self.retained_peak,
            metadata,
        )
    }

    /// Refines logical state with complete selected backing, preserving its
    /// logical fixed/context fields and requiring the complete projection.
    pub fn refine_state_backing(
        &self,
        state: RuntimeStateEstimate,
    ) -> Result<RuntimeStateEstimate, CapabilityError> {
        self.refine_state_backing_metadata(state, super::WorkspaceReportMetadata::ordinary())
            .map_err(super::WorkspaceReportError::into_capability)
    }

    /// The same backing refinement with a counted final diagnostic destination.
    pub fn refine_state_backing_metadata(
        &self,
        state: RuntimeStateEstimate,
        metadata: super::WorkspaceReportMetadata<'_>,
    ) -> Result<RuntimeStateEstimate, super::WorkspaceReportError> {
        // A terminal placement inspects no equation and therefore says nothing
        // new about the existing state's backing. Preserve its actual bound,
        // including unknown coverage, instead of replacing it with zero.
        if self.has_empty_schedule() {
            return Ok(state);
        }
        let backing = match self.retained_peak {
            Some(bytes) => metadata.bounded(bytes, format_args!(
                "largest complete retained decoder-state backing over all selected prefill and decode spans, including capacity padding and distinct retained views"))?,
            None => metadata.unknown(format_args!(
                "selected equation spans have no complete decoder-state backing bound"))?,
        };
        metadata.admit::<RuntimeStateEstimate>()?;
        Ok(state.with_selected_state_backing_fixed(self.geometry, backing)?)
    }
}

/// Inspects the entire admitted prompt and decode allowance without native work.
///
/// `quote` must advance a fresh metadata execution of the retained selection,
/// initialized with the exact cached-state projection. It must record ordinary
/// architecture equations, begin a new trace at each completed span, and identify
/// retained state roots. It must not allocate native tensors or prepare media.
/// A new metadata execution is required for each candidate chunk geometry.
///
/// Prefill uses the ordinary [`PrefillDriver`], including its readout policy.
/// Decodes are all enumerated because native workspace and cache growth need not
/// be monotone. This conservatively includes one invocation per reserved output
/// position, even when generation samples the first token from prefill. Decode
/// requires its one score row even after state-only prompt processing.
/// Safely separated spans contribute their maximum simultaneous
/// tensor-plus-host transient bound, not a sum of their graphs or independent
/// maxima of overlapping domains. One unknown span makes the request unknown.
pub fn quote_inference_workspace<E>(
    geometry: InferenceGeometry,
    quote: impl FnMut(&InferenceWorkspaceSpan) -> Result<WorkspaceTraceReport, E>,
) -> Result<InferenceWorkspaceReport, InferenceWorkspaceError<E>> {
    quote_inference_workspace_inner(geometry, quote, Metadata::ordinary(), None)
}

/// The same complete schedule with every owned quote/report destination charged
/// through this exact Context. Ordinary contexts retain their ordinary route.
/// This does not create a native request grant or replace unknown equation facts.
pub fn quote_inference_workspace_with_context<E>(
    geometry: InferenceGeometry,
    context: &eredu_nn::workspace::WorkspaceContext,
    quote: impl FnMut(&InferenceWorkspaceSpan) -> Result<WorkspaceTraceReport, E>,
) -> Result<InferenceWorkspaceReport, InferenceWorkspaceError<E>> {
    quote_inference_workspace_inner(
        geometry,
        quote,
        Metadata::from_context(context),
        context.borrowed_storage_selection(),
    )
}

/// Reduces the same exact completed reports while their caller retains an
/// independent owner. This avoids copying report graphs for interval diagnostics;
/// the owner remains live through reduction and grants no native authority.
pub fn quote_inference_workspace_with_report_owner<E, R>(
    geometry: InferenceGeometry,
    context: &eredu_nn::workspace::WorkspaceContext,
    quote: impl FnMut(&InferenceWorkspaceSpan) -> Result<R, E>,
) -> Result<InferenceWorkspaceReport, InferenceWorkspaceError<E>>
where
    R: std::borrow::Borrow<WorkspaceTraceReport>,
{
    quote_inference_workspace_inner(
        geometry,
        quote,
        Metadata::from_context(context),
        context.borrowed_storage_selection(),
    )
}

fn quote_inference_workspace_inner<F, E, R>(
    geometry: InferenceGeometry,
    quote: F,
    metadata: Metadata<'_>,
    borrowed: Option<WorkspaceBorrowedStorage>,
) -> Result<InferenceWorkspaceReport, InferenceWorkspaceError<E>>
where
    F: FnMut(&InferenceWorkspaceSpan) -> Result<R, E>,
    R: std::borrow::Borrow<WorkspaceTraceReport>,
{
    metadata.validate_geometry(geometry)?;
    metadata.admit_schedule::<F, E, R>()?;
    let execution = InferenceExecutionIdentity::default();
    // This authority drives metadata scheduling only and never escapes as native
    // submission authority. Successful native admission must reserve the result.
    let request = InferenceRequest::without_memory_budget(&execution, geometry)?;
    let mut driver = PrefillDriver::<(), ColdCompletion>::new(
        &execution,
        request,
        geometry,
        GenerationCancellationToken::new(),
    )
    .map_err(|error| metadata.prefill_error(error))?;
    let mut inspection = Inspection {
        quote,
        metadata,
        spans: Vec::new(),
        report: InferenceWorkspaceReport {
            geometry,
            spans: InferenceSpanWorkspacePlan::new_metadata(
                geometry,
                Vec::new(),
                metadata.context(),
            )?,
            completed_spans: 0,
            state_spans_complete: true,
            transient: WorkspaceBound::bounded(
                0,
                metadata.text(format_args!("no equation spans inspected"))?,
            ),
            retained_peak: Some(0),
            tensor_transient_peak: Some(0),
            host_peak: Some(0),
            peak_span: None,
            first_gap: None,
            residual: None,
        },
        known_peak: 0,
    };
    match driver.run(&mut inspection, |_, _| {}) {
        Ok(PrefillOutcome::Complete) => {}
        Ok(PrefillOutcome::Cancelled) => {
            return Err(metadata.invalid("cold inspection unexpectedly cancelled"));
        }
        Err(PrefillError::Submission(error)) => return Err(error),
        Err(PrefillError::Completion(never)) => match never {},
        Err(_) => {
            return Err(metadata.invalid("cold inspection violated the prefill output contract"));
        }
    }
    let decode_start = geometry.cached_positions + geometry.input_positions;
    for index in 0..geometry.max_output_tokens {
        inspection.record(InferenceWorkspaceSpan::Decode {
            index,
            position: decode_start + index,
            output: OutputDemand::LastPosition,
        })?;
    }
    inspection.report.spans =
        InferenceSpanWorkspacePlan::new_metadata(geometry, inspection.spans, metadata.context())?;
    if inspection.report.has_empty_schedule() {
        // No equation can create new storage. Residual composition still needs
        // this exact context selection and its separately registered owner;
        // absence of spans must not invent an empty source association.
        inspection.report.residual = borrowed.map(|borrowed| InferenceResidualWorkspace {
            borrowed,
            peak_bytes: Some(0),
            association_complete: true,
        });
    }
    Ok(inspection.report)
}

struct Inspection<'a, F> {
    quote: F,
    metadata: Metadata<'a>,
    spans: Vec<InferenceSpanWorkspaceRecord>,
    report: InferenceWorkspaceReport,
    known_peak: u64,
}
impl<F, E, R> Inspection<'_, F>
where
    F: FnMut(&InferenceWorkspaceSpan) -> Result<R, E>,
    R: std::borrow::Borrow<WorkspaceTraceReport>,
{
    fn record(&mut self, span: InferenceWorkspaceSpan) -> Result<(), InferenceWorkspaceError<E>> {
        let owner = (self.quote)(&span).map_err(|source| InferenceWorkspaceError::Quote {
            span: span.clone(),
            source,
        })?;
        let trace = std::borrow::Borrow::<WorkspaceTraceReport>::borrow(&owner);
        self.metadata.reserve(&mut self.spans, 1)?;
        self.spans
            .push(InferenceSpanWorkspaceRecord::new(span.clone(), &trace));
        self.report.state_spans_complete &= trace.state.is_some();
        // A partial family trace cannot become complete merely because its
        // borrowed-root union is known. Preserve every full-coverage gap.
        let residual_bytes = trace
            .inference_transient_bytes()
            .and_then(|_| trace.residual.as_ref()?.total_bytes);
        if self.report.completed_spans == 0 {
            self.report.residual =
                trace
                    .residual
                    .as_ref()
                    .map(|residual| InferenceResidualWorkspace {
                        borrowed: residual.borrowed_storage.clone(),
                        peak_bytes: residual_bytes,
                        association_complete: true,
                    });
        } else if let Some(existing) = self.report.residual.as_mut() {
            existing.association_complete &= trace.residual.is_some();
            if let Some(residual) = &trace.residual {
                if !existing.borrowed.same_identity(&residual.borrowed_storage) {
                    return Err(self
                        .metadata
                        .invalid("borrowed storage changed between equation spans"));
                }
            }
            existing.peak_bytes = existing
                .peak_bytes
                .zip(residual_bytes)
                .map(|(a, b)| a.max(b));
        }
        self.report.completed_spans = self
            .report
            .completed_spans
            .checked_add(1)
            .ok_or_else(|| self.metadata.invalid("completed span count overflow"))?;
        let maximum = |old: Option<u64>, next: Option<u64>| old.zip(next).map(|(a, b)| a.max(b));
        let state = trace.state.as_ref();
        self.report.retained_peak = maximum(
            self.report.retained_peak,
            state.and_then(|state| state.retained_bytes),
        );
        self.report.tensor_transient_peak = maximum(
            self.report.tensor_transient_peak,
            trace
                .tensor_buffers
                .transient_bytes
                .zip(state.and_then(|state| state.displaced_bytes))
                .map(|(new, displaced)| {
                    new.checked_add(displaced).ok_or_else(|| {
                        self.metadata
                            .invalid("tensor and displaced state workspace overflow")
                    })
                })
                .transpose()?,
        );
        self.report.host_peak = maximum(self.report.host_peak, trace.host_workspace_bytes);
        match trace.inference_transient_bytes() {
            Some(bytes) => {
                if self.report.peak_span.is_none() || bytes > self.known_peak {
                    self.known_peak = bytes;
                    self.report.peak_span = Some(span);
                    if self.report.first_gap.is_none() {
                        self.report.transient = WorkspaceBound::bounded(
                            bytes,
                            self.metadata.text(format_args!(
                                "maximum simultaneous tensor, displaced opening state and disjoint host transient storage across all completed prefill/decode spans; closing state roots excluded once; {}",
                                metadata::Assumptions(&trace.assumptions)
                            ))?,
                        );
                    }
                }
            }
            None if self.report.first_gap.is_none() => {
                self.report.transient = WorkspaceBound::Unknown {
                    reason: self.metadata.text(format_args!(
                        "inference equation span {span:?} has missing state accounting {:?}, unpriced tensor operations {:?} or managed-host operations {:?}; {}",
                        trace.state,
                        trace.unpriced_operations,
                        trace.unpriced_host_operations,
                        metadata::Assumptions(&trace.assumptions)
                    ))?,
                };
                self.report.first_gap = Some(span);
            }
            None => {}
        }
        Ok(())
    }
}

struct ColdCompletion;
impl Completion for ColdCompletion {
    type Error = Infallible;
    fn is_complete(&self) -> Result<bool, Infallible> {
        Ok(true)
    }
    fn wait(&self) -> Result<(), Infallible> {
        Ok(())
    }
}
impl<F, E, R> PrefillExecutor for Inspection<'_, F>
where
    F: FnMut(&InferenceWorkspaceSpan) -> Result<R, E>,
    R: std::borrow::Borrow<WorkspaceTraceReport>,
{
    type Output = ();
    type Completion = ColdCompletion;
    type Error = InferenceWorkspaceError<E>;
    fn submit_chunk(
        &mut self,
        chunk: &PrefillChunk,
        _: InferenceRequest,
    ) -> Result<Submission<Option<()>, ColdCompletion>, Self::Error> {
        self.record(InferenceWorkspaceSpan::Prefill(chunk.clone()))?;
        Ok(Submission {
            output: (chunk.output != OutputDemand::StateOnly).then_some(()),
            completion: ColdCompletion,
        })
    }
}

fn invalid(detail: &str) -> CapabilityError {
    CapabilityError::InvalidConfiguration {
        field: "inference_workspace",
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests;

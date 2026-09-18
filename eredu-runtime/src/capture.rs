//! Budget admission at the observation boundary, before native handles are retained.

mod preflight_budget;
pub(crate) use preflight_budget::PreflightBudget;

use eredu_core::capture::*;

mod checkpoint;
pub(crate) mod funded;
mod ordinary_delivery;
mod ordinary_error;
mod ordinary_prefill;
mod policy;
mod prefill_transaction;
pub use funded::{
    FundedCaptureCheckpoint, FundedCaptureCheckpointError, FundedCaptureDrainError,
    FundedCaptureError, FundedCaptureSession, FundedEmbeddedCaptureInvocation,
    FundedSpeculativeCaptureInvocation, PreparedFundedCaptureCheckpoint, ScheduledCaptureBackend,
};
pub use ordinary_prefill::OrdinaryPrefillCapture;
pub use policy::prefill::{
    CapturePrefillHookDecision, CapturePrefillObservationPolicy, CapturePrefillObservationRow,
    CapturePrefillProgressError, CapturePrefillRowProgress,
};
pub use policy::{CaptureObservationStep, CaptureProtocolError, CaptureRecordStatus};
mod encoded;
pub(crate) use encoded::{
    RECORD_ENCODING_CONTROL_BYTES, intervention_fits_encoding, record_fits_encoding,
};
mod host_preparation;
use host_preparation::CaptureHostOwner;
mod empty;
mod generated;
mod invocation;
pub(crate) use invocation::InvocationPreflightBudget;
mod prefix;
pub(crate) use prefix::{PrefixDestination,PrefixError};
pub(crate) mod reduction;
pub use invocation::CaptureInvocationSelection;
pub use reduction::{
    CaptureHistogramError, CaptureSummaryError, histogram_prefill_usage, summary_prefill_usage,
};
mod speculative;
pub use speculative::{
    CaptureBackendProvider, OriginalSpeculativeCapture, OriginalSpeculativeCaptureError,
    OriginalSpeculativeCaptureInvocation, OriginalSpeculativeCapturePrefix,
    PartitionCaptureBackendProvider, PreparedSpeculativeActivationRestore,
    SpeculativeActivationCheckpoint, SpeculativeCaptureErrorTransport, SpeculativeCaptureObserver,
    SpeculativeCaptureScope,
};
mod routed;
use empty::empty_payload;
pub use generated::generated_capture_source;
pub mod partition;
#[cfg(test)]
mod tests;
#[cfg(test)]
#[path = "capture/tests/text_origin.rs"]
mod text_origin_tests;
pub use checkpoint::{
    CaptureCheckpoint, CaptureForkRequest, InterventionForkRequest, PreparedCaptureRestore,
};

/// A run owns one ledger and at most one step of host records. Consumers must drain
/// each step before another is started; there is no producer queue.
pub struct CaptureSession {
    pub(crate) checkpoint_ready: bool,
    has_step: bool,
    ordinary_prefill: Option<OrdinaryPrefillCapture>,
    // Only checkpoint::fork sets this payload-free identity of the exact saved
    // frontier/usage instance. No predecessor session, ledger or account is held.
    ordinary_prefill_parent: Option<std::sync::Arc<()>>,
    ordinary_progress: Option<ordinary_prefill::Progress>,
    pub(crate) invocation: Option<CaptureInvocationShape>,
    pub(crate) invocation_window: Option<CaptureInvocationWindow>,
    // Only the original speculative collector can enable host reduction fragments.
    window_reductions: bool,
    transaction: Option<(eredu_core::DistributedCommitEpoch, CaptureTransactionStatus)>,
    last_transaction_epoch: Option<eredu_core::DistributedCommitEpoch>,
    partition: Option<partition::PartitionCaptureRun>,
    pub(crate) plan: SharedCapturePlan,
    pub(crate) ledger: CaptureLedger,
    pub(crate) records: Option<Vec<CaptureRecord>>,
    pub(crate) prediction: u64,
    pub(crate) phase: CapturePhase,
    pub(crate) capture_seconds: f64,
    pub(crate) interventions: Option<crate::intervention::InterventionRun>,
    // Records/partial state retire before the preallocated final frame custody.
    ordinary_frame: Option<PreparedCapturedStep>,
    // Identity is shared by checkpoints/partition handles, never child runs.
    // All payloads retire before this owner and final immutable error custody.
    owner: std::sync::Arc<CaptureHostOwner>,
    // Independent of the mutable host mutex: poison diagnostics also retain custody.
    ordinary_error_custody: Option<eredu_core::HostPreparationAuthority>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CaptureTransactionStatus {
    Pending,
    Committed,
    Aborted,
}

impl CaptureSession {
    /// Creates an unstarted capture run retaining its exact immutable source.
    /// Existing source attachments remain on that owner; session storage,
    /// logical capture quotas and native admission remain separate obligations.
    pub fn new(plan: SharedCapturePlan) -> Self {
        Self {
            owner: std::sync::Arc::new(CaptureHostOwner::default()),
            ordinary_error_custody: None,
            checkpoint_ready: true,
            has_step: false,
            ordinary_prefill: None,
            ordinary_prefill_parent: None,
            ordinary_progress: None,
            ordinary_frame: None,
            invocation: None,
            invocation_window: None,
            window_reductions: false,
            transaction: None,
            last_transaction_epoch: None,
            partition: None,
            ledger: CaptureLedger::new(&plan),
            plan,
            records: None,
            prediction: 0,
            phase: CapturePhase::Prefill,
            capture_seconds: 0.0,
            interventions: None,
        }
    }

    /// Retains already-acquired host preparation custody for this run and all
    /// existing checkpoint/partition aliases. This grants no allocation authority
    /// or byte bound; callers acquire before constructing or copying payloads.
    /// Borrowed admission DTO clones remain the caller's ownership contract.
    pub fn retain_host_preparation(
        &self,
        authority: &eredu_core::HostPreparationAuthority,
    ) -> Result<(), CaptureError> {
        let host = self.ordinary_error_custody.clone();
        let result = self.retain_host_preparation_unretained(authority);
        ordinary_error::retain_result(host, result)
    }

    fn retain_host_preparation_unretained(
        &self,
        authority: &eredu_core::HostPreparationAuthority,
    ) -> Result<(), CaptureError> {
        self.owner.retain(authority)
    }

    /// Borrows this run's immutable admission.
    pub fn plan(&self) -> &AdmittedCapturePlan {
        &self.plan
    }

    /// Borrow this run's exact retained source without creating a registration.
    pub fn shared_plan_source(&self) -> &SharedCapturePlan {
        &self.plan
    }

    pub(crate) fn prepare_transaction(
        &mut self,
        epoch: eredu_core::DistributedCommitEpoch,
        pass: crate::ExpertPass,
    ) -> Result<(), CaptureError> {
        let host = self.ordinary_error_custody.clone();
        let result = self.prepare_transaction_unretained(epoch, pass);
        ordinary_error::retain_result(host, result)
    }

    fn prepare_transaction_unretained(
        &mut self,
        epoch: eredu_core::DistributedCommitEpoch,
        pass: crate::ExpertPass,
    ) -> Result<(), CaptureError> {
        if self.records.is_none()
            || self.transaction.is_some()
            || self
                .last_transaction_epoch
                .is_some_and(|previous| previous >= epoch)
        {
            return Err(CaptureError::Invalid(
                "capture transaction requires a fresh epoch and an undelivered step".into(),
            ));
        }
        self.last_transaction_epoch = Some(epoch);
        self.bind_partition_run_epoch(epoch);
        self.transaction = Some((epoch, CaptureTransactionStatus::Pending));
        let phase = match pass {
            crate::ExpertPass::Prefill => CapturePhase::Prefill,
            crate::ExpertPass::Decode => CapturePhase::Decode,
        };
        if self.phase != phase {
            return Err(CaptureError::Invalid(
                "capture phase differs from the actual forward".into(),
            ));
        }
        Ok(())
    }

    /// Starts this forward's step inside shared observer preparation. Claim the
    /// epoch before fallible reservation so even rejected attempts cannot reuse it.
    pub(crate) fn prepare_step_transaction(
        &mut self,
        epoch: eredu_core::DistributedCommitEpoch,
        pass: crate::ExpertPass,
        prediction: u64,
    ) -> Result<(), CaptureError> {
        let host = self.ordinary_error_custody.clone();
        let result = self.prepare_step_transaction_unretained(epoch, pass, prediction);
        ordinary_error::retain_result(host, result)
    }

    fn prepare_step_transaction_unretained(
        &mut self,
        epoch: eredu_core::DistributedCommitEpoch,
        pass: crate::ExpertPass,
        prediction: u64,
    ) -> Result<(), CaptureError> {
        self.claim_step_epoch(epoch, false)
            .map_err(policy::public_error)?;
        let phase = match pass {
            crate::ExpertPass::Prefill => CapturePhase::Prefill,
            crate::ExpertPass::Decode => CapturePhase::Decode,
        };
        self.begin_step_inner(phase, prediction)
    }

    pub(crate) fn complete_transaction(
        &mut self,
        epoch: eredu_core::DistributedCommitEpoch,
    ) -> Result<(), CaptureError> {
        let host = self.ordinary_error_custody.clone();
        let result = self.complete_transaction_unretained(epoch);
        ordinary_error::retain_result(host, result)
    }

    fn complete_transaction_unretained(
        &mut self,
        epoch: eredu_core::DistributedCommitEpoch,
    ) -> Result<(), CaptureError> {
        if self.records.is_none()
            || self.transaction != Some((epoch, CaptureTransactionStatus::Pending))
        {
            return Err(CaptureError::Invalid(
                "capture completion has no matching pending transaction".into(),
            ));
        }
        if self
            .partition
            .as_ref()
            .is_some_and(|run| !run.delivery_complete(epoch))
        {
            return Err(CaptureError::Invalid(
                "partition capture delivery is incomplete".into(),
            ));
        }
        self.finish_routed_captures()?;
        self.finish_interventions()
    }

    pub(crate) fn finish_transaction(
        &mut self,
        epoch: eredu_core::DistributedCommitEpoch,
        committed: bool,
    ) {
        if let Some(partition) = &mut self.partition {
            partition.finish(
                epoch,
                committed,
                self.records.as_mut(),
                self.interventions
                    .as_mut()
                    .and_then(|run| run.records.as_mut()),
            );
        }
        if self.transaction.is_some_and(|(active, _)| active == epoch) {
            self.transaction = Some((
                epoch,
                if committed {
                    CaptureTransactionStatus::Committed
                } else {
                    CaptureTransactionStatus::Aborted
                },
            ));
        }
        if !committed {
            self.checkpoint_ready = false;
            if self.transaction.is_none() && self.records.is_some() {
                self.transaction = Some((epoch, CaptureTransactionStatus::Aborted));
            }
        }
    }

    /// Immutable intervention admission currently paired with this shared owner.
    pub fn intervention_plan(&self) -> Option<&eredu_core::intervention::AdmittedInterventionPlan> {
        self.interventions.as_ref().map(|run| &run.plan)
    }

    /// Reserves diagnostic envelopes before execution, including scheduled skips and
    /// missing values. Exhaustion here fails the step: emitting an unaccounted skip
    /// record would itself violate the export limit.
    pub fn begin_step(&mut self, phase: CapturePhase, prediction: u64) -> Result<(), CaptureError> {
        let host = self.ordinary_error_custody.clone();
        let result = self.begin_step_unretained(phase, prediction);
        ordinary_error::retain_result(host, result)
    }

    fn begin_step_unretained(
        &mut self,
        phase: CapturePhase,
        prediction: u64,
    ) -> Result<(), CaptureError> {
        if self.plan.invocation_bounds().is_some() {
            return Err(CaptureError::Invalid(
                "independent capture requires explicit invocation geometry".into(),
            ));
        }
        if self.transaction.is_some() {
            return Err(CaptureError::Invalid(
                "capture step cannot replace an undrained transaction".into(),
            ));
        }
        self.begin_step_inner(phase, prediction)
    }

    fn begin_step_inner(
        &mut self,
        phase: CapturePhase,
        prediction: u64,
    ) -> Result<(), CaptureError> {
        self.validate_step_start(prediction)
            .map_err(policy::public_error)?;
        self.reset_step_ledger()?;
        // Required control is charged and preallocated before any record field.
        // Keep it local until records are installed; partial construction errors
        // drop the records before this actual host authority.
        let ordinary_frame = self.prepare_frame()?;
        let mut records = Vec::new();
        for (selection, point) in self.plan.plan().selections.iter().zip(self.plan.points()) {
            let charged = policy::reserve_metadata(
                &mut self.ledger,
                selection,
                point,
                self.partition
                    .as_ref()
                    .map_or(1, |run| run.world_size() as u64),
            )?;
            records.push(CaptureRecord {
                schema_version: CAPTURE_SCHEMA_VERSION,
                selection_id: selection.id.clone(),
                path: selection.path.clone(),
                node_id: point.node_id.clone(),
                position: point.position,
                source_shape: None,
                source_dtype: None,
                selected_shape: None,
                outcome: if selection.schedule.includes(phase, prediction) {
                    CaptureOutcome::Missing
                } else {
                    CaptureOutcome::Skipped {
                        reason: CaptureSkipReason::Schedule,
                    }
                },
                payload: None,
                charged,
            });
        }
        self.records = Some(records);
        self.ordinary_frame = Some(ordinary_frame);
        self.phase = phase;
        self.prediction = prediction;
        self.capture_seconds = 0.0;
        if let Some(interventions) = &mut self.interventions {
            interventions.begin_step(&mut self.ledger, phase, prediction)?;
        }
        Ok(())
    }

    /// Invoked while the architecture borrows a tensor. A skipped point never calls
    /// the native transform and never clones a native tensor handle.
    pub fn observe<B: CaptureBackend>(
        &mut self,
        backend: &mut B,
        path: &str,
        tensor: &B::Tensor,
    ) -> Result<(), CaptureExecutionError<B::Error>> {
        let host = self.ordinary_error_custody.clone();
        let result = self.observe_unretained(backend, path, tensor);
        ordinary_error::retain_admission(host, result)
    }

    fn observe_unretained<B: CaptureBackend>(
        &mut self,
        backend: &mut B,
        path: &str,
        tensor: &B::Tensor,
    ) -> Result<(), CaptureExecutionError<B::Error>> {
        if self.ordinary_progress.is_some() {
            return self.observe_ordinary_prefill(backend, path, tensor);
        }
        self.observe_classified(backend, path, tensor, |error| error)
    }

    // Generated sources can fail portable contract checks after reservation.
    // Classify those before finalizing the record, using the same forward driver.
    fn observe_classified<B: CaptureBackend>(
        &mut self,
        backend: &mut B,
        path: &str,
        tensor: &B::Tensor,
        classify: impl Fn(CaptureExecutionError<B::Error>) -> CaptureExecutionError<B::Error>,
    ) -> Result<(), CaptureExecutionError<B::Error>> {
        if self.partition.is_some() {
            return Err(CaptureError::Invalid(
                "partition-bound capture requires its live producer authority".into(),
            )
            .into());
        }
        let geometry = self.tensor_geometry()?;
        let Some(records) = self.records.as_mut() else {
            return Err(CaptureError::Invalid("capture step not started".into()).into());
        };
        for ((selection, point), record) in self
            .plan
            .plan()
            .selections
            .iter()
            .zip(self.plan.points())
            .zip(records)
        {
            match policy::selected_record(selection, record, path) {
                Ok(false) => continue,
                Ok(true) => {}
                Err(_) => {
                    return Err(CaptureError::Invalid(format!(
                        "observation emitted twice: {path}"
                    ))
                    .into());
                }
            }
            let started = std::time::Instant::now();
            let result = capture_window_value(
                self.invocation_window,
                backend,
                tensor,
                selection,
                point,
                record,
                geometry,
                &mut self.ledger,
            )
            .map_err(&classify);
            self.capture_seconds += started.elapsed().as_secs_f64();
            if let Err(error) = result {
                let reason = match &error {
                    CaptureExecutionError::Admission(error) => policy::failure_reason(error),
                    CaptureExecutionError::Backend(_) => CaptureFailureReason::Native,
                };
                record.payload = None;
                record.outcome = CaptureOutcome::Failed {
                    reason,
                    message: bounded_diagnostic(&error),
                };
                return Err(error);
            }
        }
        Ok(())
    }

    /// Test diagnostics are explicit caller-owned copies of the delivered frame.
    #[cfg(test)]
    pub(crate) fn take_step(&mut self) -> Option<CapturedStep> {
        self.take_shared_step().map(|step| step.as_step().clone())
    }

    fn take_step_inner(&mut self) -> Option<CapturedStep> {
        if self
            .transaction
            .is_some_and(|(_, status)| status == CaptureTransactionStatus::Pending)
        {
            return None;
        }
        // The low-level, nontransactional owner must also finalize sparse
        // receipts before handing out records. Failed/incomplete payloads become
        // explicit failed outcomes; no tentative sparse rows escape as Missing.
        let _ = self.finish_routed_captures();
        let outcome = match self.transaction.take() {
            None => CaptureStepOutcome::Untracked,
            Some((_, CaptureTransactionStatus::Committed)) => CaptureStepOutcome::Committed,
            Some((_, CaptureTransactionStatus::Aborted)) => CaptureStepOutcome::Aborted,
            Some((_, CaptureTransactionStatus::Pending)) => {
                unreachable!("pending step cannot drain")
            }
        };
        let committed = outcome != CaptureStepOutcome::Aborted;
        if let Some(records) = &self.records {
            self.checkpoint_ready = committed
                && self.finish_interventions().is_ok()
                && !records
                    .iter()
                    .any(|record| matches!(record.outcome, CaptureOutcome::Failed { .. }));
        }
        self.invocation_window = None;
        self.records.take().map(|records| CapturedStep {
            outcome,
            phase: self.phase,
            invocation: self.invocation.take(),
            prediction_index: self.prediction,
            records,
            partitions: self
                .partition
                .as_mut()
                .map_or_else(Vec::new, |run| run.take_evidence()),
            interventions: self
                .interventions
                .as_mut()
                .map_or_else(Vec::new, |run| run.take_records()),
            step_usage: self.ledger.step(),
            cumulative_usage: self.ledger.total(),
            capture_seconds: self.capture_seconds,
        })
    }
}

pub(crate) fn capture_window_value<B: CaptureBackend>(
    window: Option<CaptureInvocationWindow>,
    backend: &mut B,
    tensor: &B::Tensor,
    selection: &CaptureSelection,
    point: &eredu_core::ObservationPoint,
    record: &mut CaptureRecord,
    physical: CaptureInvocationShape,
    ledger: &mut CaptureLedger,
) -> Result<(), CaptureExecutionError<B::Error>> {
    let Some(window) = window else {
        return capture_value(backend, tensor, selection, point, record, physical, ledger);
    };
    // The same bounded metadata envelope used by the existing native fragment
    // path is reserved before constructing its vector/control projection.
    let usage = partition::fragment_metadata_usage(
        selection,
        point,
        point.axes.as_ref().map_or(32, Vec::len),
    )?;
    if let Some(reason) = ledger.reserve(usage)? {
        record.outcome = CaptureOutcome::Skipped { reason };
        return Ok(());
    }
    record.charged = record.charged.checked_add(usage)?;
    let shape = backend
        .shape(tensor)
        .map_err(CaptureExecutionError::Backend)?;
    record.source_dtype = backend.source_dtype(tensor);
    let projection = window.project(physical, point, selection, &shape)?;
    let Some(fragment) = projection.fragments().first() else {
        record.source_shape = Some(shape);
        record.outcome = CaptureOutcome::Skipped {
            reason: CaptureSkipReason::NotInvoked,
        };
        return Ok(());
    };
    capture_resolved_value(
        backend,
        tensor,
        selection,
        record,
        shape,
        fragment.local(),
        ledger,
    )
}

/// Shared transformation path for ordinary captures and intervention evidence.
#[allow(clippy::too_many_arguments)]
pub(crate) fn capture_value<B: CaptureBackend>(
    backend: &mut B,
    tensor: &B::Tensor,
    selection: &CaptureSelection,
    point: &eredu_core::ObservationPoint,
    record: &mut CaptureRecord,
    geometry: CaptureInvocationShape,
    ledger: &mut CaptureLedger,
) -> Result<(), CaptureExecutionError<B::Error>> {
    let path = &selection.path;
    record.source_dtype = backend.source_dtype(tensor);

    let shape = backend
        .shape(tensor)
        .map_err(CaptureExecutionError::Backend)?;
    geometry.validate_actual(point, &shape)?;
    if let Some(expected) = geometry.resolve(point)? {
        if expected != shape {
            return Err(CaptureError::Invalid(format!(
                "runtime shape for {path}: expected {expected:?}, got {shape:?}"
            ))
            .into());
        }
    }
    let slice = resolve_slice(point, selection, &shape)?;
    capture_resolved_value(backend, tensor, selection, record, shape, &slice, ledger)
}

/// The native reservation/transform sequence shared by ordinary observations,
/// intervention evidence and globally identified partition fragments.
pub(super) fn capture_resolved_value<B: CaptureBackend>(
    backend: &mut B,
    tensor: &B::Tensor,
    selection: &CaptureSelection,
    record: &mut CaptureRecord,
    shape: Vec<u64>,
    slice: &ResolvedCaptureSlice,
    ledger: &mut dyn CaptureReservation,
) -> Result<(), CaptureExecutionError<B::Error>> {
    let usage = backend.estimate(tensor, selection, &slice)?;
    record.source_shape = Some(shape);
    record.selected_shape = Some(slice.shape.clone());
    if let Some(reason) = ledger.reserve(usage)? {
        record.outcome = CaptureOutcome::Skipped { reason };
        return Ok(());
    }
    record.charged = record.charged.checked_add(usage)?;
    let transformed = backend.transform(tensor, selection, &slice);
    // A deferred source becomes known only after its reserved factory runs.
    // Preserve that fact even if the subsequent transformation fails.
    record.source_dtype = backend.source_dtype(tensor);
    let mut payload = transformed.map_err(CaptureExecutionError::Backend)?;
    let source = if record.position == eredu_core::ObservationPosition::AfterIntervention {
        CandidateLogitsSource::Effective
    } else {
        CandidateLogitsSource::Original
    };
    match &mut payload {
        CapturePayload::Candidates(candidates) => candidates.source = source,
        CapturePayload::TokenScores(scores) => scores.source = source,
        _ => {}
    }
    let available = elements(&slice.shape)?;
    record.outcome = completed_capture_outcome(&selection.transform, available);
    record.payload = Some(payload);
    // Count through a bounded sink, without allocating a second JSON buffer.
    // This is a backend-contract check, not a substitute for the pre-copy estimate.
    let mut sink = CountingWriter {
        written: 0,
        limit: record.charged.encoded_bytes,
    };
    serde_json::to_writer(&mut sink, record)
        .map_err(|_| CaptureError::Invalid("backend underestimated encoded capture size".into()))?;
    Ok(())
}

pub(crate) fn completed_capture_outcome(transform: &CaptureTransform, available: u64) -> CaptureOutcome {
    match transform {
        CaptureTransform::Preview { max_elements } if *max_elements < available => {
            CaptureOutcome::Truncated {
                available_elements: available,
                emitted_elements: *max_elements,
            }
        }
        _ => CaptureOutcome::Captured,
    }
}

pub(crate) fn bounded_diagnostic(error: &impl std::fmt::Display) -> String {
    use std::fmt::Write;
    struct Message(String);
    impl std::fmt::Write for Message {
        fn write_str(&mut self, text: &str) -> std::fmt::Result {
            let mut end = text.len().min(256 - self.0.len());
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            self.0.push_str(&text[..end]);
            if end < text.len() {
                Err(std::fmt::Error)
            } else {
                Ok(())
            }
        }
    }
    let mut message = Message(String::with_capacity(256));
    let _ = write!(&mut message, "{error}");
    message.0
}

/// Conservative JSON envelope reservation: every UTF-8 byte can escape to at most
/// six bytes; each dimension can use twenty decimal digits. Numeric payloads are
/// reserved by the native estimator. Selection count and metadata are bounded before
/// generation begins, so missing/skipped points cannot form an unbounded queue.
pub fn metadata_reservation(
    selection: &CaptureSelection,
    point: &eredu_core::ObservationPoint,
) -> Result<CaptureUsage, CaptureError> {
    metadata_reservation_fields(
        &selection.id,
        &selection.path,
        &point.node_id,
        point.axes.as_ref().map(Vec::len),
    )
}

pub(crate) fn metadata_reservation_fields(
    id: &str,
    path: &str,
    node_id: &str,
    rank: Option<usize>,
) -> Result<CaptureUsage, CaptureError> {
    let strings = add(
        add(id.len() as u64, path.len() as u64)?,
        node_id.len() as u64,
    )?;
    let rank = rank.map_or(32, |rank| rank as u64);
    Ok(CaptureUsage {
        captures: 0,
        retained_bytes: 0,
        host_bytes: add(512, add(strings, mul(rank, 128)?)?)?,
        encoded_bytes: add(2048, add(mul(strings, 6)?, mul(rank, 64)?)?)?,
    })
}

/// Checks conservative known per-step and run-total costs before execution.
/// Backends supply only their side-effect-free storage/transfer estimator. Unknown
/// dimensions remain subject to exact observation-time reservation. Step estimates
/// conservatively allow every phase-enabled selection to coincide.
pub fn preflight(
    plan: &AdmittedCapturePlan,
    estimate: impl FnMut(
        &[u64],
        &CaptureSelection,
        &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError>,
) -> Result<(), CaptureError> {
    preflight_with_extra(plan, &[], CaptureUsage::default(), &[], estimate)
}

/// Revalidates admission against the loaded catalog, then checks known geometry
/// and budgets with backend estimates. Native execution-mode checks stay with the
/// caller; this helper never accesses a device or starts a submission.
pub fn validate_session(
    plan: &AdmittedCapturePlan,
    discovery: &CaptureDiscovery,
    estimate: impl FnMut(
        &[u64],
        &CaptureSelection,
        &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError>,
) -> Result<(), CaptureError> {
    validate_continuation(plan, discovery, 0, CaptureUsage::default(), estimate)
}

pub(crate) fn validate_continuation(
    plan: &AdmittedCapturePlan,
    discovery: &CaptureDiscovery,
    next_prediction: u64,
    inherited: CaptureUsage,
    estimate: impl FnMut(
        &[u64],
        &CaptureSelection,
        &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError>,
) -> Result<(), CaptureError> {
    plan.revalidate(discovery)?;
    preflight_continuation(
        plan,
        &[],
        CaptureUsage::default(),
        &[],
        next_prediction,
        inherited,
        estimate,
    )
}

pub(crate) fn preflight_with_extra(
    plan: &AdmittedCapturePlan,
    extra: &[(CaptureSelection, eredu_core::ObservationPoint)],
    base: CaptureUsage,
    scheduled_costs: &[(CaptureSchedule, [CaptureUsage; 2])],
    estimate: impl FnMut(
        &[u64],
        &CaptureSelection,
        &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError>,
) -> Result<(), CaptureError> {
    preflight_continuation(
        plan,
        extra,
        base,
        scheduled_costs,
        0,
        CaptureUsage::default(),
        estimate,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn preflight_continuation(
    plan: &AdmittedCapturePlan,
    extra: &[(CaptureSelection, eredu_core::ObservationPoint)],
    mut base: CaptureUsage,
    scheduled_costs: &[(CaptureSchedule, [CaptureUsage; 2])],
    next_prediction: u64,
    inherited: CaptureUsage,
    mut estimate: impl FnMut(
        &[u64],
        &CaptureSelection,
        &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError>,
) -> Result<(), CaptureError> {
    if plan.invocation_bounds().is_some() {
        return invocation::preflight_invocations(
            plan,
            extra,
            base,
            scheduled_costs,
            inherited,
            estimate,
        );
    }
    let entries = || {
        plan.plan()
            .selections
            .iter()
            .zip(plan.points())
            .chain(extra.iter().map(|(selection, point)| (selection, point)))
    };
    for (selection, point) in entries() {
        base = base.checked_add(metadata_reservation(selection, point)?)?;
    }
    let mut budget = PreflightBudget::new(plan, base, next_prediction, inherited)?;
    for phase in [CapturePhase::Prefill, CapturePhase::Decode] {
        if !budget.begin_phase(phase) {
            continue;
        }
        for (schedule, costs) in scheduled_costs {
            if let Some((count, _)) = schedule.count_and_last_from(
                phase,
                next_prediction,
                plan.request().max_predictions,
            )? {
                let cost = costs[if phase == CapturePhase::Prefill { 0 } else { 1 }];
                budget.add(cost, count)?;
            }
        }
        for (selection, point) in entries() {
            let Some((count, last)) = selection.schedule.count_and_last_from(
                phase,
                next_prediction,
                plan.request().max_predictions,
            )?
            else {
                continue;
            };
            if let Some(shape) = plan.estimate_shape(point, phase, last)? {
                let slice = resolve_slice(point, selection, &shape)?;
                let cost = estimate(&shape, selection, &slice)?;
                budget.add_selection(cost, count)?;
            }
        }
        budget.finish_phase()?;
    }
    budget.finish()
}

struct CountingWriter {
    written: u64,
    limit: u64,
}
impl std::io::Write for CountingWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let next = self
            .written
            .checked_add(bytes.len() as u64)
            .filter(|next| *next <= self.limit)
            .ok_or_else(|| std::io::Error::other("capture JSON budget exceeded"))?;
        self.written = next;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Failure before reservation or during the reserved native transformation.
#[derive(Debug, thiserror::Error)]
pub enum CaptureExecutionError<E: std::error::Error + 'static> {
    /// Invalid geometry or rejected budget/capability.
    #[error(transparent)]
    Admission(#[from] CaptureError),
    /// Native transformation failed under the backend's recovery owner.
    #[error("native capture failed: {0}")]
    Backend(#[source] E),
}

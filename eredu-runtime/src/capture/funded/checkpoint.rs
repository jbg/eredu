//! Closed fixed capture frontier copied under the saved-generation host plan.
use super::*;
use crate::working_memory::{CaptureRunHostPlan, PreparedCaptureRun};
use eredu_core::HostPreparationAuthority;

/// Fixed rejection by a capture checkpoint or its fresh continuation bank.
/// No formatting, copied admission, or erased backend failure is constructed.
#[derive(Debug, thiserror::Error)]
pub enum FundedCaptureCheckpointError {
    /// The existing capture transaction has not successfully drained.
    #[error("capture checkpoint requires a successful drained boundary")]
    Boundary,
    /// This producer cannot copy legacy partitioned/intervened or invocation capture.
    #[error("capture checkpoint source is not a plain shared capture session")]
    Source,
    /// Original host-bank health, geometry, or fresh destination refused.
    #[error(transparent)]
    Host(#[from] CaptureRunHostError),
    /// Current same-run cumulative source is busy or poisoned.
    #[error(transparent)]
    Protocol(#[from] CaptureProtocolError),
    /// Existing logical cumulative usage cannot fit the saved admission.
    #[error(transparent)]
    Admission(#[from] CaptureError),
}

#[derive(Clone, Copy, Debug)]
struct Frontier {
    prediction: u64,
    phase: CapturePhase,
    has_step: bool,
    usage: CaptureUsage,
}
impl Frontier {
    fn next_prediction(self) -> u64 {
        // The shared step worker requires prediction < max_predictions.
        if self.has_step {
            self.prediction + 1
        } else {
            0
        }
    }
}

/// Immutable host-only component of an original saved-generation transaction.
/// The exact original plan remains shared; no declaration String, Vec, or plan
/// payload is cloned. This owns no native handle, allocator, issuance cursor, or
/// mutable ledger. Copy and native completion admission remain the enclosing
/// saved-generation transaction's duties.
#[derive(Debug)]
pub struct FundedCaptureCheckpoint {
    partition_run: Option<partition::PreparedPartitionCaptureRunIdentity>,
    frontier: Frontier,
    source: SharedCapturePlan,
    interventions: Option<crate::working_memory::OriginalInterventionSource>,
    lineage: crate::working_memory::CaptureRunLedger,
    // Copied controls and the source alias retire before destination H.
    _host: HostPreparationAuthority,
}

/// Borrowed exact checkpoint producer. The saved-generation host provider must
/// include required_bytes before copying. The supplied H is lifetime custody,
/// never evidence of source identity or permission to execute native work.
#[derive(Debug)]
pub struct PreparedFundedCaptureCheckpoint<'a> {
    session: &'a FundedCaptureSession,
}
impl FundedCaptureSession {
    /// Inspect without copying or allocating, at the shared drained boundary.
    pub fn prepare_checkpoint(
        &self,
    ) -> Result<PreparedFundedCaptureCheckpoint<'_>, FundedCaptureCheckpointError> {
        validate(self)?;
        Ok(PreparedFundedCaptureCheckpoint { session: self })
    }
}
fn validate(session: &FundedCaptureSession) -> Result<(), FundedCaptureCheckpointError> {
    session
        .validate_factory()
        .map_err(CaptureRunHostError::from)?;
    let inner = &session.session;
    if session.has_pending_step() || !inner.checkpoint_boundary_ready() {
        return Err(FundedCaptureCheckpointError::Boundary);
    }
    if inner.interventions.is_some()
        || inner.partition.is_some()
        || inner.ordinary_prefill.is_some()
        || inner.invocation.is_some()
        || inner.invocation_window.is_some()
        || !inner
            .plan
            .shared()
            .is_some_and(|source| source.same_storage(session.source()))
    {
        return Err(FundedCaptureCheckpointError::Source);
    }
    Ok(())
}
impl PreparedFundedCaptureCheckpoint<'_> {
    /// Absolute source frontier before the destination owner is constructed.
    pub fn next_prediction(&self) -> u64 {
        let inner = &self.session.session;
        if inner.has_step {
            inner.prediction + 1
        } else {
            0
        }
    }

    /// Complete fixed destination, inspection, source-alias and result controls.
    /// Existing source C storage is retained separately rather than copied.
    pub fn required_bytes(&self) -> Result<u64, CaptureRunHostError> {
        checkpoint_control_bytes().map_err(Into::into)
    }
    /// Construct after the enclosing original host/copy admission succeeds.
    /// The borrow keeps source frontier and ledger fixed through the copy.
    pub fn construct(
        self,
        host: &HostPreparationAuthority,
    ) -> Result<FundedCaptureCheckpoint, FundedCaptureCheckpointError> {
        validate(self.session)?;
        let inner = &self.session.session;
        let current = self.session.lineage.borrow()?;
        Ok(FundedCaptureCheckpoint {
            partition_run: self.session.partition_run.clone(),
            frontier: Frontier {
                prediction: inner.prediction,
                phase: inner.phase,
                has_step: inner.has_step,
                usage: current.usage(),
            },
            source: self.session.source().clone(),
            interventions: self.session.run.intervention_source().cloned(),
            lineage: self.session.lineage.clone(),
            _host: host.clone(),
        })
    }
}
impl FundedCaptureCheckpoint {
    /// Absolute next coordinate, retaining the original admission's origin.
    pub fn next_prediction(&self) -> u64 {
        self.frontier.next_prediction()
    }
    /// Saved cumulative usage, including failed attempts before this boundary.
    pub fn inherited_usage(&self) -> CaptureUsage {
        self.frontier.usage
    }
    /// Exact physical source identity; no owning declaration export.
    pub fn source(&self) -> &SharedCapturePlan {
        &self.source
    }
    /// Original paid intervention declaration retained with this saved frontier.
    /// An identical semantic plan in another account is not the same source.
    pub fn intervention_source(&self) -> Option<&crate::working_memory::OriginalInterventionSource> {
        self.interventions.as_ref()
    }
    /// Current cumulative usage of the original run, including work after this
    /// snapshot and earlier same-run restorations. Independent forks have their
    /// own source; this does not merge their explicitly separate budgets.
    pub fn current_usage(&self) -> Result<CaptureUsage, FundedCaptureCheckpointError> {
        Ok(self.lineage.borrow()?.usage())
    }

    /// Validate the exact fresh input/frontier represented by this saved capture
    /// source. A saved initial prompt preserves its complete input; after any
    /// prediction the pending input is one token at the saved absolute frontier.
    /// A smaller output range is allowed, but no source coordinate is relabelled.
    pub fn validate_continuation_geometry(
        &self,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<(), FundedCaptureCheckpointError> {
        use crate::working_memory::WorkingMemoryError;
        geometry
            .validate()
            .map_err(|_| CaptureRunHostError::Memory(WorkingMemoryError::IdentityMismatch))?;
        let request = self.source.admission().request();
        let origin = self
            .source
            .admission()
            .text_origin()
            .ok_or(FundedCaptureCheckpointError::Source)?;
        let first = self.next_prediction();
        let input = if first == 0 { request.prompt_tokens } else { 1 };
        let cached = if first == 0 {
            Some(origin.cached_positions)
        } else {
            origin
                .cached_positions
                .checked_add(request.prompt_tokens)
                .and_then(|position| position.checked_add(first - 1))
        }
        .ok_or(CaptureRunHostError::Memory(WorkingMemoryError::Overflow))?;
        let end = first
            .checked_add(geometry.max_output_tokens)
            .ok_or(CaptureRunHostError::Memory(WorkingMemoryError::Overflow))?;
        if geometry.batch_size != request.batch
            || geometry.input_positions != input
            || geometry.cached_positions != cached
            || end > request.max_predictions
        {
            return Err(CaptureRunHostError::Memory(WorkingMemoryError::IdentityMismatch).into());
        }
        Ok(())
    }

    /// Price the same absolute schedule after authenticating its actual fresh
    /// geometry. This plan grants no native or host allocation permission.
    pub fn continuation_host_plan_for(
        &self,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<CaptureRunHostPlan<'_>, FundedCaptureCheckpointError> {
        self.validate_continuation_geometry(geometry)?;
        self.continuation_host_plan(self.next_prediction() + geometry.max_output_tokens)
            .map_err(Into::into)
    }

    /// Price a fresh independently admitted bank for the remaining interval.
    /// The exclusive upper bound may shorten but never extend the source plan.
    /// Every frame/tensor is priced by the existing absolute-coordinate worker.
    pub fn continuation_host_plan(
        &self,
        end_prediction: u64,
    ) -> Result<CaptureRunHostPlan<'_>, CaptureRunHostError> {
        let host = CaptureRunHostPlan::prepare_range(
            &self.source,
            self.next_prediction(),
            end_prediction,
            true,
        )?;
        match &self.interventions {
            Some(source) => host.with_interventions(source),
            None => Ok(host),
        }
    }
    /// Restore the saved frontier into a fresh physically admitted bank while
    /// retaining the original run's current cumulative authority. Later source
    /// spending is observed again before every capture operation. This never
    /// restores snapshot-time usage over the live ledger.
    pub fn into_restoration(
        &self,
        bank: PreparedCaptureRun,
    ) -> Result<FundedCaptureSession, FundedCaptureCheckpointError> {
        bank.validate_checkpoint_destination(&self.source, self.interventions.as_ref(), self.next_prediction())?;
        let current = self.lineage.borrow()?;
        let ledger = CaptureLedger::with_inherited_usage(self.source.admission(), current.usage())?;
        let mut session = FundedCaptureSession::from_run_with_lineage(bank, self.lineage.clone());
        session.partition_run = self.partition_run.clone();
        session.session.ledger = ledger;
        session.session.prediction = self.frontier.prediction;
        session.session.phase = self.frontier.phase;
        session.session.has_step = self.frontier.has_step;
        session
            .validate_factory()
            .map_err(CaptureRunHostError::from)?;
        Ok(session)
    }

    /// Populate only a fresh exact bank, retaining saved cumulative usage.
    /// This is an independent continuation, like CaptureCheckpoint::fork; it
    /// never rewinds a live session, resets its ledger, or refunds spent claims.
    pub fn into_continuation(
        &self,
        bank: PreparedCaptureRun,
    ) -> Result<FundedCaptureSession, FundedCaptureCheckpointError> {
        bank.validate_checkpoint_destination(&self.source, self.interventions.as_ref(), self.next_prediction())?;
        let ledger =
            CaptureLedger::with_inherited_usage(self.source.admission(), self.frontier.usage)?;
        // The fresh bank paid the same session/identity constructor and these
        // additional fixed handoff frames before its claim table was allocated.
        let lineage = bank.new_ledger(self.frontier.usage);
        let mut session = FundedCaptureSession::from_run_with_lineage(bank, lineage);
        session.session.ledger = ledger;
        session.session.prediction = self.frontier.prediction;
        session.session.phase = self.frontier.phase;
        session.session.has_step = self.frontier.has_step;
        session
            .validate_factory()
            .map_err(CaptureRunHostError::from)?;
        Ok(session)
    }
}
fn checkpoint_control_bytes() -> Result<u64, WorkingMemoryError> {
    [
        size_of::<FundedCaptureCheckpoint>(),
        size_of::<Option<FundedCaptureCheckpoint>>(),
        size_of::<PreparedFundedCaptureCheckpoint<'_>>(),
        size_of::<Frontier>(),
        size_of::<SharedCapturePlan>(),
        size_of::<Option<crate::working_memory::OriginalInterventionSource>>(),
        size_of::<HostPreparationAuthority>(),
        size_of::<FundedCaptureCheckpointError>(),
        size_of::<Result<PreparedFundedCaptureCheckpoint<'_>, FundedCaptureCheckpointError>>(),
        size_of::<Result<FundedCaptureCheckpoint, FundedCaptureCheckpointError>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .and_then(|n| u64::try_from(n).ok())
    .ok_or(WorkingMemoryError::Overflow)
}
pub(crate) fn continuation_control_bytes() -> Result<u64, WorkingMemoryError> {
    [
        size_of::<eredu_core::InferenceGeometry>(),
        size_of::<Result<(), FundedCaptureCheckpointError>>(),
        size_of::<Result<CaptureRunHostPlan<'_>, FundedCaptureCheckpointError>>(),
        size_of::<Frontier>(),
        size_of::<CaptureLedger>(),
        size_of::<&FundedCaptureCheckpoint>(),
        size_of::<FundedCaptureCheckpointError>(),
        size_of::<Result<FundedCaptureSession, FundedCaptureCheckpointError>>(),
        size_of::<Result<CaptureLedger, CaptureError>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .and_then(|n| u64::try_from(n).ok())
    .ok_or(WorkingMemoryError::Overflow)
}

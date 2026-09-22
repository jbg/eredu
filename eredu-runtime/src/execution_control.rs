//! Shared completed-token lifecycle and non-rewindable snapshot reservations.

use eredu_core::{execution_control::*, generation::FinishReason};
#[cfg(test)]
use std::rc::Rc;
use std::sync::{Arc, Mutex};

mod history;
pub use history::{ControllerHistoryError, ControllerHistorySuffix};
mod grammar;
pub use grammar::{PreparedGrammarBranch, PreparedGrammarBranchCause, PreparedGrammarBranchError};
mod choice;
mod sampling;
mod snapshot;
pub use choice::{
    PreparedControllerCause, PreparedControllerDecision, PreparedControllerSource,
    PreparedForbiddenDecision, PreparedGrammarChoice, PreparedGrammarChoiceCause,
    PreparedGrammarChoiceError, PreparedPlainDecision, PreparedTokenChoiceError,
    TokenChoiceController, TokenChoiceError,
};
pub use sampling::{
    apply_prepared_sampling_override, apply_sampling_override, prepare_sampling_override,
    validate_sampling_override, SamplingOverride, SamplingOverrideError, SamplingStateFacts,
    TextSamplingControlBackend, ValidatedSamplingOverride,
};

/// Logical owned storage of the audited serialized intervention request DTO.
/// Native estimators and admitted handles are deliberately excluded.
pub fn intervention_plan_storage_bytes(
    plan: &eredu_core::intervention::InterventionPlan,
) -> Option<u64> {
    (std::mem::size_of_val(plan) as u64).checked_add(storage::heap_bytes(plan)?)
}
pub(crate) mod storage;
pub use snapshot::{
    ManagedTextContinuation, PendingSnapshotResumeRetention, PreparedTextHostCopy,
    PreparedTextHostJournal, RetainedSnapshotBackendError, SamplingCopyPolicy,
    SnapshotTokenController, TextContinuationSnapshot, TextHostCopyError, TextSnapshotBackend,
    TextSnapshotError,
};

/// Quiescent, rewindable lifecycle component of a complete generation snapshot.
/// It is not native completion evidence or a substitute for a complete snapshot.
#[derive(Debug, Clone)]
pub struct GenerationBoundary {
    status: GenerationStatus,
    prediction: u64,
    finish_reason: Option<FinishReason>,
}

/// One portable lifecycle shared by all native implementations. A Running state
/// includes native completion and record delivery, not just host submission.
#[derive(Debug)]
pub struct GenerationLifecycle {
    boundary: GenerationBoundary,
    epoch: u64,
}

impl Default for GenerationLifecycle {
    fn default() -> Self {
        Self {
            boundary: GenerationBoundary {
                status: GenerationStatus::Prepared,
                prediction: 0,
                finish_reason: None,
            },
            epoch: 0,
        }
    }
}

impl GenerationLifecycle {
    /// Current state. Paused means all work and delivery at the boundary completed.
    pub fn status(&self) -> GenerationStatus {
        self.boundary.status
    }
    /// Next absolute prediction: zero is prefill, later values are decode.
    pub fn next_prediction(&self) -> u64 {
        self.boundary.prediction
    }
    /// Monotone restore epoch for consumer reconciliation. Never rewound.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
    /// Retained terminal outcome, including cancellation.
    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.boundary.finish_reason
    }

    fn invalid(&self, to: GenerationStatus) -> ExecutionControlError {
        ExecutionControlError::Transition {
            from: self.status(),
            to,
        }
    }

    /// Enters one prediction. Call only after observing remote pause/cancel requests.
    pub fn begin_prediction(&mut self) -> Result<(), ExecutionControlError> {
        if !matches!(
            self.status(),
            GenerationStatus::Prepared | GenerationStatus::Paused
        ) {
            return Err(self.invalid(GenerationStatus::Running));
        }
        self.boundary
            .prediction
            .checked_add(1)
            .ok_or(ExecutionControlError::Overflow)?;
        self.boundary.status = GenerationStatus::Running;
        Ok(())
    }

    /// Commits one prediction only after exact native completion, semantic
    /// commitment and associated capture/intervention record delivery succeed.
    pub fn complete_prediction(
        &mut self,
        reason: Option<FinishReason>,
    ) -> Result<(), ExecutionControlError> {
        if self.status() != GenerationStatus::Running {
            return Err(self.invalid(GenerationStatus::Paused));
        }
        self.boundary.prediction = self
            .boundary
            .prediction
            .checked_add(1)
            .ok_or(ExecutionControlError::Overflow)?;
        self.boundary.finish_reason = reason;
        self.boundary.status = match reason {
            Some(FinishReason::Cancelled) => GenerationStatus::Cancelled,
            Some(_) => GenerationStatus::Completed,
            None => GenerationStatus::Paused,
        };
        Ok(())
    }

    /// Pauses an already quiescent session without advancing a prediction or RNG.
    pub fn pause(&mut self) -> Result<(), ExecutionControlError> {
        if !matches!(
            self.status(),
            GenerationStatus::Prepared | GenerationStatus::Paused
        ) {
            return Err(self.invalid(GenerationStatus::Paused));
        }
        self.boundary.status = GenerationStatus::Paused;
        Ok(())
    }

    /// Cancels before the next prediction, after native work is known quiescent.
    pub fn cancel(&mut self) -> Result<(), ExecutionControlError> {
        if !matches!(
            self.status(),
            GenerationStatus::Prepared | GenerationStatus::Paused
        ) {
            return Err(self.invalid(GenerationStatus::Cancelled));
        }
        self.boundary.status = GenerationStatus::Cancelled;
        self.boundary.finish_reason = Some(FinishReason::Cancelled);
        Ok(())
    }

    /// Completes termination without a newly committed prediction. Cancellation
    /// and an already complete semantic constraint use the same transition.
    /// Call only after verifying a completed, drained native boundary.
    pub fn finish_without_prediction(
        &mut self,
        reason: FinishReason,
    ) -> Result<(), ExecutionControlError> {
        if self.status() != GenerationStatus::Running {
            return Err(self.invalid(GenerationStatus::Completed));
        }
        self.boundary.status = if reason == FinishReason::Cancelled {
            GenerationStatus::Cancelled
        } else {
            GenerationStatus::Completed
        };
        self.boundary.finish_reason = Some(reason);
        Ok(())
    }

    /// Fences generation after an unresolved native or semantic/delivery failure.
    /// Failure does not imply native completion; the existing native owner survives.
    pub fn fail(&mut self) {
        self.boundary.status = GenerationStatus::Failed;
    }

    /// Saves a quiescent initial, paused or normally completed boundary.
    pub fn checkpoint(&self) -> Result<GenerationBoundary, ExecutionControlError> {
        if !matches!(
            self.status(),
            GenerationStatus::Prepared | GenerationStatus::Paused | GenerationStatus::Completed
        ) {
            return Err(self.invalid(GenerationStatus::Paused));
        }
        Ok(self.boundary.clone())
    }

    /// Allows complete state placement after cancellation without reviving the
    /// cancelled branch. The native owner must separately prove quiescence.
    /// Copying or restoring a cancelled branch remains disallowed.
    pub fn validate_placement(&self) -> Result<(), ExecutionControlError> {
        if matches!(
            self.status(),
            GenerationStatus::Running | GenerationStatus::Failed
        ) {
            return Err(self.invalid(GenerationStatus::Paused));
        }
        Ok(())
    }

    /// Validates the portable restore transition before native copying begins.
    pub fn validate_restore(&self) -> Result<(), ExecutionControlError> {
        self.checkpoint()?;
        self.epoch
            .checked_add(1)
            .ok_or(ExecutionControlError::Overflow)?;
        Ok(())
    }

    /// Installs the saved boundary after compatibility checks and atomic native
    /// restoration. A completed snapshot remains terminal; an earlier snapshot
    /// can resume an otherwise completed run. Cancellation and failure stay fenced.
    pub fn restore(&mut self, saved: &GenerationBoundary) -> Result<(), ExecutionControlError> {
        self.validate_restore()?;
        self.epoch += 1;
        self.boundary = saved.clone();
        Ok(())
    }

    /// Creates child lifecycle state at the same absolute position with epoch zero.
    /// Native state and immutable admissions must be independently forked/rebound.
    pub fn fork(saved: &GenerationBoundary) -> Self {
        Self {
            boundary: saved.clone(),
            epoch: 0,
        }
    }
}

#[derive(Debug)]
struct BudgetState {
    limits: SnapshotLimits,
    usage: SnapshotUsage,
}

/// Shared logical reservation owner for retained snapshots, child states and
/// provisional restore copies. It deliberately lives outside rewindable state.
#[derive(Debug)]
pub struct SnapshotBudget(Option<Arc<BudgetOwner>>);
#[derive(Debug)]
struct BudgetOwner {
    state: Mutex<BudgetState>,
    // The shared shell and initialized Mutex PAL retire before this custody.
    _host: eredu_core::HostPreparationAuthority,
}
impl Clone for SnapshotBudget {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.inner())))
    }
}
impl Drop for SnapshotBudget {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}

impl SnapshotBudget {
    /// Constructs the logical copy budget under the caller's actual original
    /// metadata account, including its initialized synchronization owner.
    pub fn prepare(
        limits: SnapshotLimits,
        funding: &eredu_core::HostMetadataFunding,
    ) -> Result<Self, eredu_core::HostMetadataFundingError> {
        use eredu_core::{HostMetadataFundingError as E, HostPreparationAuthority};
        let bytes = Self::construction_bytes()
            .ok_or(E::Unavailable)?
            .checked_add(
                HostPreparationAuthority::retention_bytes::<eredu_core::HostMetadataFunding>()
                    .ok_or(E::Overflow)?,
            )
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Result<Self, E>>()))
            .and_then(|bytes| {
                bytes.checked_add(eredu_core::HostMetadataFunding::reservation_control_bytes())
            })
            .ok_or(E::Overflow)?;
        funding.reserve_metadata(bytes)?;
        Ok(Self::new_with_host(
            limits,
            HostPreparationAuthority::retain(funding.clone()),
        ))
    }
    fn inner(&self) -> &Arc<BudgetOwner> {
        self.0.as_ref().expect("live snapshot budget")
    }
    /// Creates an explicitly bounded resource owner; zero limits disable retention.
    pub fn new(limits: SnapshotLimits) -> Self {
        Self::new_with_host(limits, eredu_core::HostPreparationAuthority::unmanaged())
    }
    pub(crate) fn new_with_host(
        limits: SnapshotLimits,
        host: eredu_core::HostPreparationAuthority,
    ) -> Self {
        let budget = Self(Some(Arc::new(BudgetOwner {
            state: Mutex::new(BudgetState {
                limits,
                usage: SnapshotUsage::default(),
            }),
            _host: host,
        })));
        // The selected host Mutex may allocate its PAL lazily. Materialize it in
        // this caller-owned constructor, before any allocation-free reservation.
        drop(
            budget
                .inner()
                .state
                .lock()
                .unwrap_or_else(|p| p.into_inner()),
        );
        budget
    }
    /// Exact shared owner, initialized PAL and fixed constructor transports.
    /// Unknown platform storage is accepted only by an ordinary caller, whose
    /// metadata hook does not claim managed funding.
    pub(crate) fn construction_bytes() -> Option<usize> {
        use std::mem::size_of;
        let pal = usize::try_from(
            crate::working_memory::OriginalHostMetadataCustody::initialized_mutex_bytes().ok()?,
        )
        .ok()?;
        let shared =
            usize::try_from(crate::working_memory::qualified_shared_bytes::<BudgetOwner>().ok()?)
                .ok()?;
        [
            shared,
            pal,
            size_of::<Self>(),
            size_of::<BudgetOwner>(),
            size_of::<Option<BudgetOwner>>(),
            size_of::<(SnapshotLimits, eredu_core::HostPreparationAuthority)>(),
            size_of::<std::sync::LockResult<std::sync::MutexGuard<'_, BudgetState>>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
    /// Observes retained counts and cumulative copying without native side effects.
    pub fn usage(&self) -> SnapshotUsage {
        self.inner()
            .state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .usage
    }
    /// Reserves known costs before copying or retaining any state. Unknown costs,
    /// overflows and limits fail without changing accounting. After admission,
    /// dropping a failed attempt releases retention but never refunds copying.
    pub fn reserve(
        &self,
        kind: SnapshotResourceKind,
        estimate: Option<SnapshotEstimate>,
    ) -> Result<SnapshotReservation, ExecutionControlError> {
        self.reserve_pending(kind, estimate)
            .map(PendingSnapshotReservation::publish)
    }

    // Changes only logical usage. The inline lease refunds retention on failure,
    // preserving consumed copy work; publishing its Arc requires host authority.
    pub(crate) fn reserve_pending(
        &self,
        kind: SnapshotResourceKind,
        estimate: Option<SnapshotEstimate>,
    ) -> Result<PendingSnapshotReservation, ExecutionControlError> {
        let estimate = estimate.ok_or(ExecutionControlError::UnknownEstimate)?;
        let mut state = self.inner().state.lock().unwrap_or_else(|p| p.into_inner());
        let add = |a: u64, b: u64| a.checked_add(b).ok_or(ExecutionControlError::Overflow);
        let next = SnapshotUsage {
            snapshots: add(
                state.usage.snapshots,
                u64::from(kind == SnapshotResourceKind::Snapshot),
            )?,
            branches: add(
                state.usage.branches,
                u64::from(kind == SnapshotResourceKind::Branch),
            )?,
            retained_bytes: add(state.usage.retained_bytes, estimate.retained_bytes)?,
            cumulative_copy_bytes: add(state.usage.cumulative_copy_bytes, estimate.copy_bytes)?,
        };
        for (exceeded, name) in [
            (
                next.snapshots > state.limits.max_snapshots,
                "snapshot count",
            ),
            (next.branches > state.limits.max_branches, "branch count"),
            (
                next.retained_bytes > state.limits.retained_bytes,
                "retained bytes",
            ),
            (
                next.cumulative_copy_bytes > state.limits.cumulative_copy_bytes,
                "cumulative copy bytes",
            ),
        ] {
            if exceeded {
                return Err(ExecutionControlError::Limit(name));
            }
        }
        state.usage = next;
        Ok(PendingSnapshotReservation(ReservationLease {
            budget: self.clone(),
            kind,
            retained_bytes: estimate.retained_bytes,
            host: eredu_core::HostPreparationAuthority::unmanaged(),
        }))
    }
}

// The original capture path publishes only after destination host admission.
pub(crate) struct PendingSnapshotReservation(ReservationLease);
impl PendingSnapshotReservation {
    fn publish(self) -> SnapshotReservation {
        self.publish_with_host(eredu_core::HostPreparationAuthority::unmanaged())
    }
    pub(crate) fn publish_with_host(
        mut self,
        host: eredu_core::HostPreparationAuthority,
    ) -> SnapshotReservation {
        self.0.host = host;
        SnapshotReservation {
            lease: Some(Arc::new(self.0)),
        }
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        use std::{alloc::Layout, mem::size_of, sync::atomic::AtomicUsize};
        let shared = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<ReservationLease>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        [
            shared,
            size_of::<Self>(),
            size_of::<ReservationLease>(),
            size_of::<SnapshotReservation>(),
            size_of::<eredu_core::HostPreparationAuthority>(),
            size_of::<Option<snapshot::PendingSnapshotResumeRetention>>(),
            size_of::<Option<Arc<ReservationLease>>>(),
            size_of::<Option<ReservationLease>>(),
            size_of::<Result<Self, ExecutionControlError>>(),
            size_of::<SnapshotUsage>(),
            size_of::<SnapshotEstimate>(),
            size_of::<std::sync::MutexGuard<'_, BudgetState>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
}

/// Retention lease held alongside an opaque snapshot or child state. Cloning the
/// lease is only for handles sharing that same object; copying native state needs
/// another reservation. Final drop releases retained resources, not cumulative work.
#[derive(Debug)]
pub struct SnapshotReservation {
    lease: Option<Arc<ReservationLease>>,
}
impl Clone for SnapshotReservation {
    fn clone(&self) -> Self {
        Self {
            lease: Some(Arc::clone(
                self.lease.as_ref().expect("live snapshot lease"),
            )),
        }
    }
}
impl Drop for SnapshotReservation {
    fn drop(&mut self) {
        if let Some(owner) = self.lease.take() {
            drop(Arc::into_inner(owner));
        }
    }
}

impl SnapshotReservation {
    fn lease(&self) -> &ReservationLease {
        self.lease.as_deref().expect("live snapshot lease")
    }
    /// Logical storage charged while any handle to this exact object is retained.
    pub fn retained_bytes(&self) -> u64 {
        self.lease().retained_bytes
    }
}

#[derive(Debug)]
struct ReservationLease {
    budget: SnapshotBudget,
    kind: SnapshotResourceKind,
    retained_bytes: u64,
    // The lease shell, refund operation and budget alias retire before custody.
    host: eredu_core::HostPreparationAuthority,
}

impl Drop for ReservationLease {
    fn drop(&mut self) {
        let mut state = self
            .budget
            .inner()
            .state
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        state.usage.snapshots -= u64::from(self.kind == SnapshotResourceKind::Snapshot);
        state.usage.branches -= u64::from(self.kind == SnapshotResourceKind::Branch);
        state.usage.retained_bytes -= self.retained_bytes;
    }
}

#[cfg(test)]
mod tests;

mod trace;
pub use trace::{TraceBudget, TraceLimits};

/// Complete logical host storage of the supported immutable intervention DTO.
pub fn admitted_intervention_storage_bytes(
    plan: &eredu_core::intervention::AdmittedInterventionPlan,
) -> Option<u64> {
    storage::heap_bytes(plan.plan())?
        .checked_add(storage::heap_bytes(plan.points())?)?
        .checked_add(plan.identity().len() as u64)?
        .checked_add(plan.artifact_identity().len() as u64)?
        .checked_add(plan.session_id().len() as u64)?
        .checked_add(std::mem::size_of_val(plan) as u64)
}

pub(crate) use choice::ControllerChoiceIdentity;

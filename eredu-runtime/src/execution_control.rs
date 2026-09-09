//! Shared completed-token lifecycle and non-rewindable snapshot reservations.

use eredu_core::{execution_control::*, generation::FinishReason};
use std::{cell::RefCell, rc::Rc};

mod choice;
mod sampling;
mod snapshot;
pub use choice::{TokenChoiceController, TokenChoiceError};
pub use sampling::{
    apply_prepared_sampling_override, apply_sampling_override, SamplingOverride,
    SamplingOverrideError, SamplingStateFacts, TextSamplingControlBackend,
    ValidatedSamplingOverride,
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
    ManagedTextContinuation, SnapshotTokenController, TextBranchRequest, TextContinuationBranch,
    TextContinuationSnapshot, TextSnapshotBackend, TextSnapshotError,
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

    /// Completes a cancellation observed after entering Running but before the
    /// ordinary cursor submitted or committed any prediction. Call only after
    /// verifying a quiescent native boundary; this never increments position.
    pub fn cancel_without_prediction(&mut self) -> Result<(), ExecutionControlError> {
        if self.status() != GenerationStatus::Running {
            return Err(self.invalid(GenerationStatus::Cancelled));
        }
        self.boundary.status = GenerationStatus::Cancelled;
        self.boundary.finish_reason = Some(FinishReason::Cancelled);
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

struct BudgetState {
    limits: SnapshotLimits,
    usage: SnapshotUsage,
}

/// Shared logical reservation owner for retained snapshots, child states and
/// provisional restore copies. It deliberately lives outside rewindable state.
#[derive(Clone)]
pub struct SnapshotBudget(Rc<RefCell<BudgetState>>);

impl SnapshotBudget {
    /// Creates an explicitly bounded resource owner; zero limits disable retention.
    pub fn new(limits: SnapshotLimits) -> Self {
        Self(Rc::new(RefCell::new(BudgetState {
            limits,
            usage: SnapshotUsage::default(),
        })))
    }
    /// Observes retained counts and cumulative copying without native side effects.
    pub fn usage(&self) -> SnapshotUsage {
        self.0.borrow().usage
    }
    /// Reserves known costs before copying or retaining any state. Unknown costs,
    /// overflows and limits fail without changing accounting. After admission,
    /// dropping a failed attempt releases retention but never refunds copying.
    pub fn reserve(
        &self,
        kind: SnapshotResourceKind,
        estimate: Option<SnapshotEstimate>,
    ) -> Result<SnapshotReservation, ExecutionControlError> {
        let estimate = estimate.ok_or(ExecutionControlError::UnknownEstimate)?;
        let mut state = self.0.borrow_mut();
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
        Ok(SnapshotReservation {
            lease: Rc::new(ReservationLease {
                budget: self.clone(),
                kind,
                retained_bytes: estimate.retained_bytes,
            }),
        })
    }
}

/// Retention lease held alongside an opaque snapshot or child state. Cloning the
/// lease is only for handles sharing that same object; copying native state needs
/// another reservation. Final drop releases retained resources, not cumulative work.
#[derive(Clone)]
pub struct SnapshotReservation {
    lease: Rc<ReservationLease>,
}

impl SnapshotReservation {
    /// Logical storage charged while any handle to this exact object is retained.
    pub fn retained_bytes(&self) -> u64 {
        self.lease.retained_bytes
    }
}

struct ReservationLease {
    budget: SnapshotBudget,
    kind: SnapshotResourceKind,
    retained_bytes: u64,
}

impl Drop for ReservationLease {
    fn drop(&mut self) {
        let mut state = self.budget.0.borrow_mut();
        state.usage.snapshots -= u64::from(self.kind == SnapshotResourceKind::Snapshot);
        state.usage.branches -= u64::from(self.kind == SnapshotResourceKind::Branch);
        state.usage.retained_bytes -= self.retained_bytes;
    }
}

#[cfg(test)]
mod tests;

mod trace;
pub use trace::{TraceBudget, TraceLimits};

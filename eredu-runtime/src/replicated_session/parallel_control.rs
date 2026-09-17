//! Actual shared-session control occurrences, outside all saved model state.
use crate::working_memory::WorkingMemoryError;
use crate::DistributedExecutionPhase;
use std::{
    mem::{size_of, size_of_val},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_IDENTITY: AtomicU64 = AtomicU64::new(1);

/// One control operation reached by the ordinary shared lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParallelControlEvent {
    /// Conjunction of the selected members' local phase status.
    Phase(DistributedExecutionPhase),
    /// Final decision after native completion and observation delivery.
    Commit,
}

/// Process-local identity of one request's control issuance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParallelControlIdentity(u64);

/// Monotonic issuance retained by the admitted request, outside snapshots.
/// This cursor describes actual work; it grants no memory or native authority.
#[derive(Debug)]
pub struct ParallelControlCursor {
    identity: ParallelControlIdentity,
    attempted: u64,
}

/// Move-only description consumed before a native control operation is admitted.
#[derive(Debug)]
pub struct ParallelControlClaim {
    identity: ParallelControlIdentity,
    event: ParallelControlEvent,
    ordinal: u64,
}
impl ParallelControlCursor {
    /// Creates allocation-free request identity without resetting any old cursor.
    pub fn new() -> Result<Self, WorkingMemoryError> {
        let identity = NEXT_IDENTITY
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| WorkingMemoryError::Overflow)?;
        Ok(Self {
            identity: ParallelControlIdentity(identity),
            attempted: 0,
        })
    }
    /// Exact fixed construction and claim transports, for source host admission.
    pub fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<ParallelControlClaim>(),
            size_of::<Result<Self, WorkingMemoryError>>(),
            size_of::<Result<ParallelControlClaim, WorkingMemoryError>>(),
            size_of::<(&mut Self, ParallelControlEvent)>(),
            size_of::<(u64, u64, ParallelControlIdentity)>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    /// Claims before any fallible source inspection, account or native producer.
    /// Failure and dropping a claim never return its ordinal.
    pub fn claim(
        &mut self,
        event: ParallelControlEvent,
    ) -> Result<ParallelControlClaim, WorkingMemoryError> {
        let ordinal = self.attempted;
        self.attempted = ordinal.checked_add(1).ok_or(WorkingMemoryError::Overflow)?;
        Ok(ParallelControlClaim {
            identity: self.identity,
            event,
            ordinal,
        })
    }
    /// Identity checked by the same request's source binding.
    pub const fn identity(&self) -> ParallelControlIdentity {
        self.identity
    }
    /// Attempted work, including failed admission and discarded proposals.
    pub const fn attempted(&self) -> u64 {
        self.attempted
    }
}
impl ParallelControlClaim {
    /// Exact issuer, independently of equal event and ordinal values.
    pub const fn identity(&self) -> ParallelControlIdentity {
        self.identity
    }
    /// Actual shared lifecycle event.
    pub const fn event(&self) -> ParallelControlEvent {
        self.event
    }
    /// Monotonic request-local attempt.
    pub const fn ordinal(&self) -> u64 {
        self.ordinal
    }
}

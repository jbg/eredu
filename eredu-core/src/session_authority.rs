//! Exact session admission and backend-independent mutable-submission ownership.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use crate::SessionCapabilities;

/// Exact capabilities admitted before materialization, not a subset requirement.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct SessionAdmission {
    capabilities: SessionCapabilities,
}

impl SessionAdmission {
    /// Retains the complete pre-materialization report.
    pub const fn new(capabilities: SessionCapabilities) -> Self {
        Self { capabilities }
    }

    /// Validates the complete realized report before a session is published.
    pub fn validate(self, realized: SessionCapabilities) -> Result<(), SessionAdmissionError> {
        if self.capabilities == realized {
            Ok(())
        } else {
            Err(SessionAdmissionError {
                admitted: self.capabilities,
                realized,
            })
        }
    }
}

/// A realized session differs from the exact admitted capability report.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
#[error(
    "realized session capabilities {realized:?} do not match pre-materialization admission {admitted:?}"
)]
pub struct SessionAdmissionError {
    admitted: SessionCapabilities,
    realized: SessionCapabilities,
}

impl SessionAdmissionError {
    /// Returns the complete admitted report.
    pub const fn admitted(self) -> SessionCapabilities {
        self.admitted
    }

    /// Returns the complete realized report.
    pub const fn realized(self) -> SessionCapabilities {
        self.realized
    }
}

/// Unique authority for a mutable session and its unresolved submission.
///
/// Beginning work requires exclusive access to this non-cloneable owner. The
/// active ticket is atomic so a completion may resolve on another thread; this
/// does not make a backend's native session or resources thread-safe. Each
/// authority allocates once, and each lease shares that allocation.
#[derive(Debug)]
pub struct SessionAuthority {
    active: Arc<AtomicU64>,
    next_ticket: u64,
}

impl Default for SessionAuthority {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionAuthority {
    /// Creates idle authority for one session.
    pub fn new() -> Self {
        Self {
            active: Arc::new(AtomicU64::new(0)),
            next_ticket: 1,
        }
    }

    /// Rejects mutation while a submission completion owns this session.
    pub fn require_idle(&self) -> Result<(), SessionAuthorityError> {
        if self.active.load(Ordering::Acquire) == 0 {
            Ok(())
        } else {
            Err(SessionAuthorityError::Busy)
        }
    }

    /// Begins one submission before native allocation or mutable execution.
    ///
    /// An aborted submission releases automatically when its lease is dropped.
    /// Successful submission moves the lease into its native completion; no
    /// second independently releasable copy can be created.
    pub fn begin_submission(&mut self) -> Result<SubmissionLease, SessionAuthorityError> {
        self.require_idle()?;
        let ticket = self.next_ticket;
        self.next_ticket = ticket
            .checked_add(1)
            .ok_or(SessionAuthorityError::TicketExhausted)?;
        self.active.store(ticket, Ordering::Release);
        Ok(SubmissionLease {
            owner: Arc::clone(&self.active),
            ticket,
        })
    }
}

/// Submission exclusion or exhaustion, with no backend error dependency.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum SessionAuthorityError {
    /// Another unresolved completion still owns mutable session state.
    #[error("model session already owns an unresolved submission completion")]
    Busy,
    /// The non-repeating ticket sequence cannot allocate another submission.
    #[error("model session submission ticket space exhausted")]
    TicketExhausted,
}

/// Native retirement and health facts used by a portable submission owner.
pub trait SessionRetirement {
    /// Error reported by the native resource owner.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Progresses native retirement and checks whether the session can accept work.
    fn refresh_native_session(&self) -> Result<(), Self::Error>;

    /// Reports whether native work has fenced the session.
    fn native_session_fenced(&self) -> bool;

    /// Preserves the typed rejection when combining native and portable errors.
    fn submission_authority_error(error: SessionAuthorityError) -> Self::Error;
}

/// Native operator selection shared by target-only and assistant-assisted execution.
pub trait SpeculativeTargetNative: crate::BackendProvider {
    /// Native operators for the retained target.
    type TargetOperators;
}

/// Native storage and operator types used to compose a selected draft.
pub trait DraftNativeResources: SpeculativeTargetNative {
    /// Native queue retained for secondary execution.
    type DraftQueue;
    /// Native checkpoint binding resources for one assistant.
    type AssistantBinder;
    /// Opaque dispatch chosen by the portable caller.
    type AssistantDispatch: 'static;
    /// Native operators supplied to a bound assistant.
    type AssistantOperators;
    /// Retires native work while retaining its exact opaque payload.
    type DraftStorage<P: 'static>: RetainedExecutionPayload<
        Payload = P,
        Queue = Self::DraftQueue,
        Error = Self::Error,
    >;
}

/// Native retention of an installed payload and its exact submission context.
pub trait RetainedExecutionPayload {
    /// Opaque value whose allocations may outlive the calling operation.
    type Payload: 'static;
    /// Exact native queue or execution context.
    type Queue;
    /// Failure establishing native authority or completion.
    type Error;
    /// Exclusive mutation scope that settles work or retains the payload on drop.
    type Operation<'a>: RetainedPayloadOperation<Self::Error, Payload = Self::Payload>
    where
        Self: 'a;

    /// Observes the payload without beginning native work.
    fn payload(&self) -> &Self::Payload;
    /// Retains an owned reference to the exact queue.
    fn retain_queue(&self) -> Self::Queue;
    /// Acquires exclusive mutation authority over the retained payload.
    fn begin_operation(&mut self) -> Result<Self::Operation<'_>, Self::Error>;
}

/// One native operation with exclusive mutation authority over an opaque payload.
pub trait RetainedPayloadOperation<E>: Sized {
    /// Opaque value guarded by the operation.
    type Payload;
    /// Borrows the value after the native owner establishes exclusivity.
    fn payload_mut(&mut self) -> &mut Self::Payload;
    /// Settles native work or retains its resources until safe retirement.
    fn settle<T>(self, result: Result<T, E>) -> Result<T, E>;
}

/// Move-only ownership of one unresolved session submission.
///
/// The backend must retain this value until exact native completion, terminal
/// failure, or safe cancellation/teardown. A pending observation must not
/// resolve it. Native resources must be drained or safely retained before
/// dropping it: this portable type neither waits nor cancels native work.
#[derive(Debug)]
#[must_use = "the submission lease must be retained until native work is safely resolved"]
pub struct SubmissionLease {
    owner: Arc<AtomicU64>,
    ticket: u64,
}

impl SubmissionLease {
    /// Releases only this ticket after the backend establishes safe resolution.
    ///
    /// Returns whether this call released it. Repeated resolution and later
    /// destruction cannot clear any newer submission, even across threads.
    pub fn resolve(&self) -> bool {
        self.owner
            .compare_exchange(self.ticket, 0, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }
}

impl Drop for SubmissionLease {
    fn drop(&mut self) {
        self.resolve();
    }
}

#[cfg(test)]
mod tests;

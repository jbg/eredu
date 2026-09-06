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
#[error("realized session capabilities {realized:?} do not match pre-materialization admission {admitted:?}")]
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

//! Shared failure fencing for one opaque communication session.

use std::sync::{Arc, Mutex};

use super::{
    CommunicationCompletionPolicy, CommunicationManifest, CommunicationOperation,
    CommunicationRouteId,
};
use crate::{
    BackendFailure, BoundedCompletion, BoundedSubmissionOutcome, CompletionCancellationMode,
    Submission,
};

/// First failed communication operation recorded for a native world.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
#[error(
    "communication is poisoned by prior {operation:?} (route {route:?}, disposition {cancellation:?})"
)]
pub struct CommunicationPoisoned {
    /// Operation that fenced the session.
    pub operation: CommunicationOperation,
    /// Opaque route, if this was a directed transfer.
    pub route: Option<CommunicationRouteId>,
    /// Disposition selected or reported for work live at failure.
    pub cancellation: CompletionCancellationMode,
}

/// Failure of an exact native communication submission or wait.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CommunicationError {
    /// A previously recorded failure forbids further ordinary submission.
    #[error(transparent)]
    Poisoned(#[from] CommunicationPoisoned),
    /// This communication requires an explicit wait and cancellation bound.
    #[error("communication manifest has no bounded completion policy")]
    MissingCompletionPolicy,
    /// The native backend failed to submit the selected operation.
    #[error("communication {operation:?} submission failed (route {route:?}): {source}")]
    Submission {
        /// Requested operation.
        operation: CommunicationOperation,
        /// Opaque route, when applicable.
        route: Option<CommunicationRouteId>,
        /// Original mechanism failure.
        #[source]
        source: BackendFailure,
    },
    /// The native backend failed to establish exact completion.
    #[error("communication {operation:?} completion failed (route {route:?}): {source}")]
    Completion {
        /// Requested operation.
        operation: CommunicationOperation,
        /// Opaque route, when applicable.
        route: Option<CommunicationRouteId>,
        /// Original mechanism failure.
        #[source]
        source: BackendFailure,
    },
    /// The exact submission exceeded the selected wait.
    #[error("communication {operation:?} exceeded its deadline (route {route:?}, disposition {cancellation:?})")]
    Deadline {
        /// Requested operation.
        operation: CommunicationOperation,
        /// Opaque route, when applicable.
        route: Option<CommunicationRouteId>,
        /// Safe disposition reported by the completion owner.
        cancellation: CompletionCancellationMode,
    },
}

/// Bounded completion authority shared by native handles and the portable driver.
///
/// The fence retains the first failed operation. It grants no release of native
/// resources and does not schedule work or classify model execution phases.
#[derive(Debug, Clone)]
pub struct CommunicationAuthority {
    policy: Option<CommunicationCompletionPolicy>,
    failure: Arc<Mutex<Option<CommunicationPoisoned>>>,
}

impl CommunicationAuthority {
    /// Creates one authority from the exact selected opaque manifest.
    pub fn from_manifest(manifest: &CommunicationManifest) -> Result<Self, CommunicationError> {
        let policy = manifest.completion_policy();
        if policy.is_none() && (!manifest.groups().is_empty() || !manifest.routes().is_empty()) {
            return Err(CommunicationError::MissingCompletionPolicy);
        }
        Ok(Self {
            policy,
            failure: Arc::new(Mutex::new(None)),
        })
    }

    fn failure_guard(&self) -> std::sync::MutexGuard<'_, Option<CommunicationPoisoned>> {
        self.failure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Rejects ordinary work once this shared authority has fenced the session.
    pub fn ensure_active(&self) -> Result<(), CommunicationPoisoned> {
        match *self.failure_guard() {
            Some(failure) => Err(failure),
            None => Ok(()),
        }
    }

    /// The exact bounded completion selected by the manifest.
    pub const fn completion_policy(&self) -> Option<CommunicationCompletionPolicy> {
        self.policy
    }

    /// Fences the session with the observed disposition, preserving the first failure.
    pub fn record_failure(
        &self,
        operation: CommunicationOperation,
        route: Option<CommunicationRouteId>,
        cancellation: CompletionCancellationMode,
    ) {
        self.failure_guard().get_or_insert(CommunicationPoisoned {
            operation,
            route,
            cancellation,
        });
    }

    /// Fences this operation under the selected cancellation policy.
    pub fn fence(&self, operation: CommunicationOperation, route: Option<CommunicationRouteId>) {
        let policy = self
            .policy
            .expect("communication operation requires bounded completion");
        self.record_failure(operation, route, policy.cancellation());
    }

    /// Records a native submission failure while preserving its typed cause.
    pub fn submission_failure(
        &self,
        error: impl std::error::Error + Send + Sync + 'static,
        operation: CommunicationOperation,
        route: Option<CommunicationRouteId>,
    ) -> CommunicationError {
        self.fence(operation, route);
        CommunicationError::Submission {
            operation,
            route,
            source: BackendFailure::from_error(error),
        }
    }

    /// Records failure to establish completion while preserving its typed cause.
    pub fn completion_failure(
        &self,
        error: impl std::error::Error + Send + Sync + 'static,
        operation: CommunicationOperation,
        route: Option<CommunicationRouteId>,
    ) -> CommunicationError {
        self.fence(operation, route);
        CommunicationError::Completion {
            operation,
            route,
            source: BackendFailure::from_error(error),
        }
    }

    /// Waits exactly for an ordinary native operation and fences on failure.
    pub fn wait<T, C>(
        &self,
        submission: Submission<T, C>,
        operation: CommunicationOperation,
        route: Option<CommunicationRouteId>,
    ) -> Result<T, CommunicationError>
    where
        C: BoundedCompletion,
    {
        self.ensure_active()?;
        let policy = self
            .policy
            .ok_or(CommunicationError::MissingCompletionPolicy)?
            .bounded_wait();
        match submission
            .wait_bounded(policy)
            .map_err(|error| self.completion_failure(error, operation, route))?
        {
            BoundedSubmissionOutcome::Completed(output) => Ok(output),
            BoundedSubmissionOutcome::DeadlineExceeded { cancellation } => {
                self.record_failure(operation, route, cancellation);
                Err(CommunicationError::Deadline {
                    operation,
                    route,
                    cancellation,
                })
            }
        }
    }
}

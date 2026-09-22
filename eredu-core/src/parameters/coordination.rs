//! Shared public outcomes and accounting for live parameter operations.
use super::{CaptureUsage, ParameterError};
use serde::{Deserialize, Serialize};

/// A typed coordination cause and the host account paying for its owned boxes.
/// The cause retires before its final accounting owner.
#[derive(Debug)]
pub struct ParameterCoordinationFailure {
    cause: Option<Box<ParameterCoordinationError>>,
    funding: Option<crate::HostMetadataFunding>,
}
impl ParameterCoordinationFailure {
    pub(super) fn new(cause: ParameterCoordinationError) -> Self {
        Self {
            cause: Some(Box::new(cause)),
            funding: None,
        }
    }
    /// Retains the account whose caller has prepaid this protocol's error
    /// representations. This attachment grants no execution or source authority.
    pub fn retain_prepaid_metadata(&mut self, funding: &crate::HostMetadataFunding) {
        if self.funding.is_none() {
            self.funding = Some(funding.clone());
        }
    }
}
impl std::ops::Deref for ParameterCoordinationFailure {
    type Target = ParameterCoordinationError;
    fn deref(&self) -> &Self::Target {
        self.cause.as_deref().expect("live coordination cause")
    }
}
impl std::fmt::Display for ParameterCoordinationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&**self, f)
    }
}
impl std::error::Error for ParameterCoordinationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&**self)
    }
}
impl Drop for ParameterCoordinationFailure {
    fn drop(&mut self) {
        if let Some(cause) = self.cause.take() {
            drop(*cause);
        }
    }
}

/// Monotone reservations for mandatory session-control exchanges. Parameter
/// tensors, payload exchange and result storage have separate caller limits.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterCoordinationUsage {
    /// Attempts begun, including rejected and terminally failed operations.
    pub attempts: u64,
    /// Reserved control-frame/native-completion/host-decoding storage, summed
    /// over attempts. This is not simultaneous allocation or allocator telemetry.
    pub reserved: CaptureUsage,
}

/// Ordered boundary in a live parameter read or reversible transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParameterCoordinationStage {
    /// Exact common operation and each rank's local authority and budget checks.
    Admission,
    /// Native reads or completed replacement tensors, before global publication.
    Preparation,
    /// Completed result decoding/assembly before releasing any public result.
    Delivery,
    /// Local parameter handles and compatible cache/version state were published.
    Publication,
    /// Every rank restored original values and compatible state after rejection.
    Restoration,
}

/// Failures preserve local policy errors and original native causes. An
/// incomplete exchange never yields a successful or partially measured result.
#[derive(Debug, thiserror::Error)]
pub enum ParameterCoordinationError {
    /// The supplied native facts cannot support finite bounded coordination.
    #[error("invalid parameter coordination admission: {0}")]
    Admission(&'static str),
    /// Shared operation authority is busy or poisoned by unwinding.
    #[error("parameter operation owner is busy or poisoned")]
    Busy,
    /// An unagreed failure permanently fenced this operation owner.
    #[error("parameter operation owner is fenced")]
    Fenced,
    /// Invalid peer identity, ordering, shape or completed decision.
    #[error("parameter operation protocol mismatch: {0}")]
    Protocol(&'static str),
    /// A peer rejected a completed operation boundary.
    #[error("parameter operation rank {rank} rejected {stage:?}")]
    PeerRejected {
        /// First rejecting rank in retained world order.
        rank: usize,
        /// Boundary rejected before advancement.
        stage: ParameterCoordinationStage,
    },
    /// This rank retains its own error while agreeing the global rejection.
    #[error("parameter operation rank {rank} rejected {stage:?}: {source}")]
    LocalRejected {
        /// Rank retaining the original cause.
        rank: usize,
        /// Rejected boundary.
        stage: ParameterCoordinationStage,
        /// Local validation, reservation, native work or assembly failure.
        #[source]
        source: Box<ParameterError>,
    },
    /// Native submission, exact completion or host resolution failed.
    #[error(transparent)]
    Backend(#[from] crate::BackendFailure),
    /// The completion owner applied the selected safe terminal disposition.
    #[error("parameter operation deadline exceeded ({cancellation:?})")]
    Deadline {
        /// Actual completion disposition, not an assertion of physical completion.
        cancellation: crate::CompletionCancellationMode,
    },
}

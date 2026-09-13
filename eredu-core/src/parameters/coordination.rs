//! Shared public outcomes and accounting for live parameter operations.
use super::{CaptureUsage, ParameterError};
use serde::{Deserialize, Serialize};

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

//! Readiness and bounded accounting for distributed text setup and prediction.

use serde::{Deserialize, Serialize};

/// Ordered application/backend readiness boundary around run setup and token steps.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TextPreparationStage {
    /// Portable request, tokenizer, semantic policy and initial cancellation.
    Request,
    /// Native prompt construction.
    Prompt,
    /// Native sampling-state construction.
    Sampling,
    /// Capture and intervention installation.
    Instrumentation,
    /// Host record delivery and cancellation at setup or a committed-token boundary.
    Delivery,
    /// Exact request reservation before native prompt and sampling construction.
    Admission,
    /// Current prediction authority before controller decisions or native input.
    Prediction,
    /// Controller decision readiness before native prediction submission.
    Decision,
    /// Controlled token observation, controller commit and permit finalization.
    Commitment,
    /// Originally funded empty-state preparation, before any rank installs it.
    SessionReset,
    /// All ranks report the outcome of the prepared empty-state installation.
    SessionResetPublication,
}

/// Local disposition supplied even when preparation failed or was cancelled.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TextPreparationStatus {
    /// Local preparation succeeded and the next stage may run.
    Ready,
    /// The caller cancelled before model work.
    Cancelled,
    /// The caller retains the original local preparation failure.
    Failed,
}

/// Complete agreement; an error or an incomplete exchange yields no outcome.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TextPreparationOutcome {
    /// Every participant is ready for the next preparation stage.
    Ready,
    /// At least one participant cancelled and none reported a local failure.
    Cancelled,
    /// At least one participant failed. Local callers retain their own cause.
    Rejected {
        /// First failed participant in the retained rank order.
        rank: usize,
    },
}

/// Cumulative logical reservations for session-owned preparation agreements.
///
/// These are charged before submission, never reset by model snapshot/restore,
/// and include failed attempts. Retained bytes sum reservations over attempts;
/// they are not simultaneous allocation or a physical allocator measurement.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TextPreparationUsage {
    /// Number of preparation attempts, each including its agreement exchanges.
    pub attempts: u64,
    /// Native values, staging, completion ownership and host framing retained.
    pub retained_bytes: u64,
    /// Host materialization, including protocol decoding storage.
    pub host_bytes: u64,
}

/// Portable rejection when another participant cannot start the next stage.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
#[error("text preparation at {stage:?} was rejected by rank {rank}")]
pub struct TextPreparationRejected {
    /// Boundary rejected before model work.
    pub stage: TextPreparationStage,
    /// First failed participant.
    pub rank: usize,
}

/// Cancellation encountered at a boundary that requires a ready value.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
#[error("text preparation cancelled at {stage:?}")]
pub struct TextPreparationCancelled {
    /// Cancelled preparation boundary.
    pub stage: TextPreparationStage,
}

/// Capture installation retains portable policy errors and agreement failures.
#[derive(Debug, thiserror::Error)]
pub enum TextCaptureSetupError {
    /// Local immutable-plan or accounting rejection.
    #[error(transparent)]
    Capture(#[from] crate::capture::CaptureError),
    /// Peer rejection or failed native agreement, with its original source.
    #[error(transparent)]
    Preparation(crate::BackendFailure),
}

/// Resolves local preparation through a supplied session agreement. Adapters may
/// use this inside an existing native operation whose authority is already held.
/// Agreement runs even on local failure; the local cause is preserved, and a
/// failed agreement must independently fence unresolved native authority.
pub fn finish_preparation<T, E>(
    stage: TextPreparationStage,
    local: Result<T, E>,
    agree: impl FnOnce(TextPreparationStatus) -> Result<TextPreparationOutcome, crate::BackendFailure>,
    map_backend: impl FnOnce(crate::BackendFailure) -> E,
) -> Result<T, E> {
    use crate::run_preparation::{
        TextPreparationOutcome as Outcome, TextPreparationStatus as Status,
    };
    let agreement = agree(if local.is_ok() {
        Status::Ready
    } else {
        Status::Failed
    });
    let value = local?;
    match agreement {
        Ok(Outcome::Ready) => Ok(value),
        Ok(Outcome::Rejected { rank }) => Err(map_backend(crate::BackendFailure::new(
            crate::BackendFailureKind::InvalidInput,
            crate::run_preparation::TextPreparationRejected { stage, rank },
        ))),
        Ok(Outcome::Cancelled) => Err(map_backend(crate::BackendFailure::new(
            crate::BackendFailureKind::InvalidInput,
            crate::run_preparation::TextPreparationCancelled { stage },
        ))),
        Err(error) => Err(map_backend(error)),
    }
}

/// Cancellable form of [`finish_preparation`]. `Ok(None)` votes cancellation;
/// local failure remains an error even if agreement also fails.
pub fn finish_preparation_cancellable<T, E>(
    stage: TextPreparationStage,
    local: Result<Option<T>, E>,
    agree: impl FnOnce(TextPreparationStatus) -> Result<TextPreparationOutcome, crate::BackendFailure>,
    map_backend: impl FnOnce(crate::BackendFailure) -> E,
) -> Result<Option<T>, E> {
    use crate::run_preparation::{
        TextPreparationOutcome as Outcome, TextPreparationStatus as Status,
    };
    let status = match &local {
        Ok(Some(_)) => Status::Ready,
        Ok(None) => Status::Cancelled,
        Err(_) => Status::Failed,
    };
    let agreement = agree(status);
    let value = local?;
    match agreement {
        Ok(Outcome::Ready) => Ok(value),
        Ok(Outcome::Cancelled) => Ok(None),
        Ok(Outcome::Rejected { rank }) => Err(map_backend(crate::BackendFailure::new(
            crate::BackendFailureKind::InvalidInput,
            crate::run_preparation::TextPreparationRejected { stage, rank },
        ))),
        Err(error) => Err(map_backend(error)),
    }
}

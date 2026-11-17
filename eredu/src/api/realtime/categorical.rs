//! Host-facing model and scheduler operations for categorical frames.

use std::time::Instant;

use eredu_core::{
    scheduler::{SchedulerError, SchedulerProgress, WorkId},
    BackendFailure,
};

use super::*;

/// Bound categorical model with request-local state allocation.
pub trait CategoricalRealtimeModel {
    /// The model's admitted categorical frame layout and timing.
    fn frame_schedule(&self) -> &RealtimeSpeechConfig;

    /// Creates a fresh controlled request over the already bound model.
    fn start(
        &self,
        request: RequestId,
        sampling: RealtimeSampling,
        limits: SchedulerLimits,
    ) -> Result<Box<dyn CategoricalRealtimeSession>, CategoricalRealtimeStartError>;
}

/// Failure to create a categorical request.
#[derive(Debug, thiserror::Error)]
pub enum CategoricalRealtimeStartError {
    /// The request or its scheduling policy could not be admitted.
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    /// The selected model rejected request-local state admission.
    #[error(transparent)]
    Admission(#[from] eredu_runtime::RealtimeSessionError),
    /// The selected mechanism could not realize request-local resources.
    #[error(transparent)]
    Backend(#[from] BackendFailure),
}

/// One scheduling turn with host-owned frame outputs.
pub type CategoricalRealtimeProgress =
    SchedulerProgress<RealtimeInputFrame, Result<RealtimeOutputFrame, BackendFailure>>;

/// Controlled categorical request over the shared realtime scheduler.
pub trait CategoricalRealtimeSession {
    /// Admits one host frame to the bounded queue.
    fn enqueue(&mut self, frame: RealtimeInputFrame) -> Result<WorkId, SchedulerError>;
    /// Polls and advances at most one newly submitted transition.
    fn advance(&mut self, now: Instant) -> Result<CategoricalRealtimeProgress, SchedulerError>;
    /// Current status of this request.
    fn status(&self) -> Option<RequestStatus>;
    /// Finishes an idle request.
    fn finish(&mut self) -> Result<(), SchedulerError>;
    /// Cancels the request through the shared scheduler.
    fn cancel(&mut self) -> Result<(), SchedulerError>;
}

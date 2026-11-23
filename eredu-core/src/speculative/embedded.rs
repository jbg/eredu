//! Exact native tensor operations used by the portable embedded predictor.

use crate::{
    capture::SpeculativeCaptureTypes, execution_control::SnapshotEstimate,
    speculative::SpeculativeQueueMechanisms, BackendFailure, BoundedCompletion,
};
use eredu_nn::Tensor;

type Queue<M> = <<M as EmbeddedPredictionOperators>::Tensor as Tensor>::Context;

/// Native tensor and completion operations for embedded prediction.
pub trait EmbeddedPredictionOperators: Sized + 'static {
    /// Native retained tensor.
    type Tensor: Tensor + Clone;
    /// Native scores for sampling.
    type Logits: Clone;
    /// Native capture, transport and queues.
    type Capture: SpeculativeCaptureTypes<
        Queues: SpeculativeQueueMechanisms<Queue = Queue<Self>, Error = Self::Error>,
    >;
    /// Completion retaining verification tensors.
    type Completion: BoundedCompletion<Error = Self::Error>;
    /// Native failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Preserves an owned failure.
    fn failure(error: BackendFailure) -> Self::Error;
    /// Reports rejected geometry in the numerical error domain.
    fn invalid(message: impl Into<String>) -> Self::Error;
    /// Holds a native validation scope until its deferred checks settle.
    fn validate<T>(operation: impl FnOnce() -> Result<T, Self::Error>) -> Result<T, Self::Error>;
    /// Materializes a single exact prediction token.
    fn token(token: u32, queue: &Queue<Self>) -> Result<Self::Tensor, Self::Error>;
    /// Observes retained storage for a native score value.
    fn logits_estimate(value: &Self::Logits) -> Option<SnapshotEstimate>;
    /// Observes retained and copied tensor storage.
    fn tensor_estimate(value: &Self::Tensor) -> Option<SnapshotEstimate>;
    /// Copies a tensor with native completion settled.
    fn copy_tensor(value: &Self::Tensor, queue: &Queue<Self>) -> Result<Self::Tensor, Self::Error>;
    /// Selects one score row on the specified queue.
    fn logits_row(
        value: &Self::Tensor,
        row: usize,
        queue: &Queue<Self>,
    ) -> Result<Self::Logits, Self::Error>;
    /// Materializes exact tokens for the target queue.
    fn target_tokens(
        tokens: &[u32],
        destination: &Queue<Self>,
        crosses_devices: bool,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Submits the exact verification roots.
    fn submit_verification(roots: &[&Self::Tensor]) -> Result<Self::Completion, Self::Error>;
}

//! Exact native tensor operations used by the portable embedded predictor.

use crate::{
    capture::SpeculativeCapturePrimitives, execution_control::SnapshotEstimate,
    speculative::SpeculativeQueueMechanisms, BackendFailure, BoundedCompletion,
    SpeculativeTelemetry,
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
    type Capture: SpeculativeCapturePrimitives<
        Tensor = Self::Tensor,
        InvocationError = Self::Error,
        Queues: SpeculativeQueueMechanisms<Queue = Queue<Self>>,
    >;
    /// Completion retaining verification tensors.
    type Completion: BoundedCompletion<Error = Self::Error>;
    /// Optional component telemetry.
    type Telemetry: SpeculativeTelemetry;
    /// Native failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Preserves an owned failure.
    fn failure(error: BackendFailure) -> Self::Error;
    /// Reports rejected geometry in the numerical error domain.
    fn invalid(message: impl Into<String>) -> Self::Error;
    /// Drains native component telemetry.
    fn take_telemetry() -> Result<Self::Telemetry, Self::Error>;
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
    /// Reads the sequence extent.
    fn sequence_len(value: &Self::Tensor) -> Result<usize, Self::Error>;
    /// Selects one score row on the specified queue.
    fn logits_row(
        value: &Self::Tensor,
        row: usize,
        queue: &Queue<Self>,
    ) -> Result<Self::Logits, Self::Error>;
    /// Selects one sequence row, preserving the sequence dimension.
    fn tensor_row(
        value: &Self::Tensor,
        row: usize,
        queue: &Queue<Self>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Selects an exact sequence prefix.
    fn tensor_prefix(
        value: &Self::Tensor,
        end: usize,
        queue: &Queue<Self>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Selects an exact token span.
    fn token_range(
        value: &Self::Tensor,
        start: usize,
        end: usize,
        queue: &Queue<Self>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Materializes exact tokens, with optional transfer to a nominated queue.
    fn target_tokens(
        tokens: &[u32],
        destination: Option<&Queue<Self>>,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Submits the exact verification roots.
    fn submit_verification(roots: &[&Self::Tensor]) -> Result<Self::Completion, Self::Error>;
}

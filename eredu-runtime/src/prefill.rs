//! Request cursors advanced through ordinary session transactions.
use crate::{LayeredArchitecture, RuntimeState};
use eredu_nn::NeuralBackend;
use std::{ops::Range, sync::Arc};

/// Architecture-owned input construction and immutable context between prefill steps.
pub trait ChunkedPrefillRequest<A, B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S> + 'static,
{
    /// Completed request values reused by subsequent chunks, such as encoder outputs.
    type Continuation: Clone;
    /// Complete admitted decoder extent.
    fn token_count(&self) -> usize;
    /// Largest target invocation admitted by this request.
    fn maximum_chunk_tokens(&self) -> usize;
    /// Borrows one exact input without transferring traversal or state ownership.
    /// Preparation errors return before invoking `operation`; success invokes it
    /// exactly once and returns its result unchanged.
    fn with_chunk<R>(
        &self,
        range: Range<usize>,
        continuation: Option<&Self::Continuation>,
        operation: impl for<'a> FnOnce(A::Input<'a>) -> R,
    ) -> Result<R, A::Error>;
    /// Selects immutable values from the executed graph. The driver publishes this
    /// continuation only after native completion and commitment.
    fn continuation(&self, completed: &A::ForwardContext) -> Self::Continuation;
    /// Immutable tensor roots retained by the prepared request, for snapshot retention.
    fn request_values(&self) -> Vec<&B::Tensor>;
    /// Native values that must complete before the continuation may be reused or
    /// copied into a cursor snapshot. This list is bounded by the request contract.
    fn retained_values<'a>(&self, continuation: &'a Self::Continuation) -> Vec<&'a B::Tensor>;
}

/// Exact request extent and selected prefill branch, agreed before execution.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PrefillRequestDescriptor {
    /// Complete decoder request extent.
    pub tokens: usize,
    /// Selected chunk size, or ordinary single-invocation execution.
    pub chunk_tokens: Option<usize>,
}

/// Request progress owned by a caller performing controlled or uninterrupted prefill.
/// Clones share immutable request values; snapshot users must pair this cursor with
/// the corresponding session state. Observation and execution budgets stay in the session.
#[derive(Clone)]
pub struct PrefillCursor<R, C> {
    pub(crate) request: R,
    pub(crate) continuation: Option<C>,
    pub(crate) position: usize,
    pub(crate) tokens: usize,
    pub(crate) chunk_tokens: usize,
    pub(crate) session: Arc<()>,
    pub(crate) input_identity: Option<crate::PreparedInputCacheIdentity>,
}
impl<R, C> PrefillCursor<R, C> {
    /// Number of successfully committed request positions.
    pub fn position(&self) -> usize {
        self.position
    }
    /// Complete admitted request extent.
    pub fn token_count(&self) -> usize {
        self.tokens
    }
    /// Absolute span of the next invocation (empty after completion).
    pub fn next_range(&self) -> Range<usize> {
        self.position
            ..self
                .position
                .saturating_add(self.chunk_tokens)
                .min(self.tokens)
    }
    /// Immutable prepared request shared by cursor snapshots.
    pub fn request(&self) -> &R {
        &self.request
    }
    /// Completed immutable products retained for subsequent chunks.
    pub fn continuation(&self) -> Option<&C> {
        self.continuation.as_ref()
    }
    /// Whether every admitted position has committed.
    pub fn is_complete(&self) -> bool {
        self.position == self.tokens
    }
}

/// One committed step. Intermediate output is available to shared generation policy.
pub struct PrefillAdvance<T> {
    /// Architecture-selected logits for this chunk.
    pub output: T,
    /// Absolute request positions committed by the step.
    pub range: Range<usize>,
    /// Whether this is the final chunk.
    pub complete: bool,
}

/// An invocation context cannot yet represent multi-chunk request advancement.
/// These are integration gaps, not restrictions of a model's equations.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
pub enum ChunkedPrefillContextError {
    /// Prediction target capture is not carried across request chunks.
    #[error("chunked composite prefill requires prediction capture across request chunks")]
    Prediction,
    /// Capture records need attribution to the request's absolute chunk spans.
    #[error("chunked composite prefill requires capture attribution to absolute request spans")]
    CaptureGeometry,
}

//! One prepared request, shared encoder output and ordinary target chunk inputs.
use super::*;
use crate::{
    composite_execution::{PreparedCompositeArchitecture, PreparedCompositeInput},
    media_plan::AdmittedCompositeInput,
    qwen4_exp::media::{admission::MediaPrefillChunk, MediaInputError, MediaInputPartPlan},
};
use eredu_runtime::prefill::ChunkedPrefillRequest;
use std::{ops::Range, sync::Arc};

/// Named tensor roots owned by one media prefill request and its continuation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaPrefillTensorRole {
    /// Original processor payload or metadata, still owned by the admission proof.
    Original(eredu_runtime::input::PreparedInputTensorRole),
    /// Prepared ingress and whole-request ID/rotary products.
    Prepared(crate::qwen4_exp::media::PreparedMediaTensorRole),
    /// Encoder output reused across decoder chunks.
    EncoderOutput,
}

/// Immutable prepared request shared by uninterrupted and controlled prefill cursors.
#[derive(Clone)]
pub struct MediaPrefillRequest<T> {
    admitted: Arc<AdmittedCompositeInput<MediaInputPartPlan<T>>>,
    prepared: Arc<PreparedMediaInput<T>>,
    maximum_chunk: usize,
}
impl<T: Tensor> MediaPrefillRequest<T> {
    /// Constructs native request products once from the retained host proof.
    pub fn new(
        admitted: AdmittedCompositeInput<MediaInputPartPlan<T>>,
        context: &T::Context,
    ) -> Result<Self, MediaInputError> {
        let request = MediaInputPartPlan::request(admitted.parts())?;
        let maximum_chunk = request.maximum_chunk_tokens();
        let prepared = Arc::new(request.prepare(context)?);
        // Public callers can retain an admission borrowed from `with_chunk`.
        // A fresh request prepares its own full products and must not retain
        // hidden old chunk roots in the part plans as well.
        let admitted = admitted.map_parts(|_, part| part.without_chunk());
        Ok(Self {
            admitted: Arc::new(admitted),
            prepared,
            maximum_chunk,
        })
    }
    /// Observes every owned request tensor and the supplied encoder continuation
    /// in one survey, preserving aliases across original and prepared inputs.
    ///
    /// Call with the continuation belonging to this request (or `None` before
    /// encoder execution). This neither evaluates lazy values nor proves that a
    /// continuation has completed. The report retains no tensors; indices are
    /// local to this call. Parameters, graph-internal values, session state,
    /// logits, host allocations and native scratch are outside this observation.
    pub fn inspect_storage(
        &self,
        continuation: Option<&T>,
    ) -> Result<eredu_nn::tensor_storage::TensorStorageSnapshot<MediaPrefillTensorRole>, Error>
    {
        let original =
            MediaInputPartPlan::request(self.admitted.parts()).map_err(Error::backend_source)?;
        eredu_nn::tensor_storage::TensorStorageSnapshot::inspect(
            original
                .input()
                .storage_values()
                .map(|(role, value)| (MediaPrefillTensorRole::Original(role), value))
                .chain(
                    self.prepared
                        .storage_values()
                        .map(|(role, value)| (MediaPrefillTensorRole::Prepared(role), value)),
                )
                .chain(continuation.map(|value| (MediaPrefillTensorRole::EncoderOutput, value))),
        )
    }
    fn checked_range(&self, range: Range<usize>) -> Result<Range<i32>, Error> {
        if range.start >= range.end
            || range.end > self.prepared.token_ids().dim(1) as usize
            || range.len() > self.maximum_chunk
        {
            return Err(Error::backend_source(MediaInputError::Geometry(
                "prefill span exceeds request bounds",
            )));
        }
        Ok(range.start as i32..range.end as i32)
    }
}
impl<B, S> ChunkedPrefillRequest<ConditionalModel<B>, B, S> for MediaPrefillRequest<B::Tensor>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
{
    type Continuation = Option<B::Tensor>;
    fn token_count(&self) -> usize {
        self.prepared.token_ids().dim(1) as usize
    }
    fn maximum_chunk_tokens(&self) -> usize {
        self.maximum_chunk
    }
    fn with_chunk<R>(
        &self,
        range: Range<usize>,
        continuation: Option<&Self::Continuation>,
        operation: impl for<'a> FnOnce(ConditionalInput<'a, B::Tensor>) -> R,
    ) -> Result<R, Error> {
        let range = self.checked_range(range)?;
        Ok(operation(ConditionalInput::MediaChunk {
            prepared: &self.prepared,
            range,
            projected: continuation.and_then(Option::as_ref),
        }))
    }
    fn request_values(&self) -> Vec<&B::Tensor> {
        let request = MediaInputPartPlan::request(self.admitted.parts())
            .expect("prepared request retains its admitted input");
        request
            .input()
            .storage_values()
            .map(|(_, value)| value)
            .chain(self.prepared.storage_values().map(|(_, value)| value))
            .collect()
    }
    fn continuation(&self, completed: &ConditionalForward<B::Tensor>) -> Self::Continuation {
        completed.projected.clone()
    }
    fn retained_values<'a>(&self, continuation: &'a Self::Continuation) -> Vec<&'a B::Tensor> {
        continuation.iter().collect()
    }
}
impl<B, S> ChunkedPrefillRequest<PreparedCompositeArchitecture<ConditionalModel<B>>, B, S>
    for MediaPrefillRequest<B::Tensor>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
{
    type Continuation = Option<B::Tensor>;
    fn token_count(&self) -> usize {
        self.prepared.token_ids().dim(1) as usize
    }
    fn maximum_chunk_tokens(&self) -> usize {
        self.maximum_chunk
    }
    fn with_chunk<R>(
        &self,
        range: Range<usize>,
        continuation: Option<&Self::Continuation>,
        operation: impl for<'a> FnOnce(
            PreparedCompositeInput<'a, B::Tensor, MediaInputPartPlan<B::Tensor>>,
        ) -> R,
    ) -> Result<R, Error> {
        let range = self.checked_range(range)?;
        let chunk = Arc::new(MediaPrefillChunk {
            prepared: self.prepared.clone(),
            range,
            projected: continuation.and_then(Option::as_ref).cloned(),
        });
        let admitted = self
            .admitted
            .as_ref()
            .clone()
            .map_parts(|_, part| part.with_chunk(chunk.clone()));
        let request =
            MediaInputPartPlan::request(admitted.parts()).map_err(Error::backend_source)?;
        let paired =
            PreparedCompositeInput::new(request.input(), &admitted).map_err(Error::backend)?;
        Ok(operation(paired))
    }
    fn request_values(&self) -> Vec<&B::Tensor> {
        let request = MediaInputPartPlan::request(self.admitted.parts())
            .expect("prepared request retains its admitted input");
        request
            .input()
            .storage_values()
            .map(|(_, value)| value)
            .chain(self.prepared.storage_values().map(|(_, value)| value))
            .collect()
    }
    fn continuation(&self, completed: &ConditionalForward<B::Tensor>) -> Self::Continuation {
        completed.projected.clone()
    }
    fn retained_values<'a>(&self, continuation: &'a Self::Continuation) -> Vec<&'a B::Tensor> {
        continuation.iter().collect()
    }
}

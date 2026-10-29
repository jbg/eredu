//! Immutable assembled decoder input retained across ordinary prefill transactions.
use super::*;
use eredu_runtime::prefill::ChunkedPrefillRequest;
use std::{ops::Range, sync::Arc};

/// A decoder span over the same exact admitted composite request.
pub(crate) struct DecoderPrefillChunk<'a, T> {
    pub range: Range<usize>,
    pub assembled: Option<&'a T>,
}

/// Shared request storage for architectures that assemble encoder products once
/// and retain the complete decoder ingress between bounded decoder invocations.
#[derive(Clone)]
pub struct DecoderPrefillRequest<T, P> {
    input: Arc<(
        eredu_runtime::PreparedModelInput<T>,
        AdmittedCompositeInput<P>,
    )>,
    tokens: usize,
}

impl<T, P> DecoderPrefillRequest<T, P> {
    /// Retains exact prepared tensors and their architecture admission.
    pub fn new(
        prepared: eredu_runtime::PreparedModelInput<T>,
        admitted: AdmittedCompositeInput<P>,
    ) -> Result<Self, eredu_nn::Error> {
        PreparedCompositeInput::new(&prepared, &admitted).map_err(eredu_nn::Error::backend)?;
        let tokens =
            i32::try_from(admitted.decoder_positions()).map_err(eredu_nn::Error::backend)? as usize;
        if tokens == 0 {
            return Err(eredu_nn::Error::backend(
                "composite prefill request is empty",
            ));
        }
        Ok(Self {
            input: Arc::new((prepared, admitted)),
            tokens,
        })
    }
}

impl<A, B, S> ChunkedPrefillRequest<A, B, S> for DecoderPrefillRequest<B::Tensor, A::InputPartPlan>
where
    B: NeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: CompositeArchitecture<B, S>,
    A::Error: From<eredu_nn::Error>,
{
    // Progress is committed on every rank. Only decoder-ingress owners retain
    // the assembled input; downstream/encoder-only stages receive activations.
    type Continuation = Option<B::Tensor>;
    fn token_count(&self) -> usize {
        self.tokens
    }
    fn maximum_chunk_tokens(&self) -> usize {
        self.tokens
    }
    fn with_chunk<R>(
        &self,
        range: Range<usize>,
        continuation: Option<&Self::Continuation>,
        operation: impl for<'a> FnOnce(A::Input<'a>) -> R,
    ) -> Result<R, A::Error> {
        let assembled = continuation.and_then(Option::as_ref);
        if range.start >= range.end
            || range.end > self.tokens
            || (range.start > 0 && continuation.is_none())
            || assembled.is_some_and(|value| {
                value.shape().len() != 3 || value.dim(1) as usize != self.tokens
            })
        {
            return Err(eredu_nn::Error::backend(
                "invalid retained composite prefill span or continuation",
            )
            .into());
        }
        let mut input = PreparedCompositeInput::new(&self.input.0, &self.input.1)
            .map_err(eredu_nn::Error::backend)?;
        input.decoder_chunk = Some((range.start, range.end, assembled));
        Ok(operation(A::prepared_input(input)))
    }
    fn continuation(&self, completed: &A::ForwardContext) -> Self::Continuation {
        A::prefill_decoder_continuation(completed).cloned()
    }
    fn request_values(&self) -> Vec<&B::Tensor> {
        self.input
            .0
            .storage_values()
            .map(|(_, value)| value)
            .collect()
    }
    fn retained_values<'a>(&self, continuation: &'a Self::Continuation) -> Vec<&'a B::Tensor> {
        continuation.iter().collect()
    }
}

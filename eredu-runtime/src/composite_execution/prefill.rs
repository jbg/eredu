//! Immutable assembled decoder input retained across ordinary prefill transactions.
use super::*;
use crate::{ComponentAdapter, ComponentDefinition, ComponentResourceAccess};
use std::{borrow::Borrow, ops::Range, sync::Arc};

/// Shared request storage for architectures that assemble encoder products once
/// and retain the complete decoder ingress between bounded decoder invocations.
#[derive(Clone)]
pub struct DecoderPrefillRequest<T, P> {
    input: Arc<(eredu_core::PreparedModelInput<T>, AdmittedCompositeInput<P>)>,
    tokens: usize,
}

impl<T, P> DecoderPrefillRequest<T, P> {
    /// Retains exact prepared tensors and their architecture admission.
    pub fn new(
        prepared: eredu_core::PreparedModelInput<T>,
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

impl<B, D, K, S, R> ChunkedPrefillRequest<ComponentAdapter<D, K, R>, B, S>
    for DecoderPrefillRequest<B::Tensor, K::InputPartPlan>
where
    B: NeuralBackend,
    D: Borrow<ComponentDefinition> + 'static,
    R: ComponentResourceAccess<B, K> + 'static,
    S: RuntimeState<B>,
    K: CompositeComponentKernel<B, S>,
    K::Error: From<eredu_nn::Error>,
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
    fn with_chunk<O>(
        &self,
        range: Range<usize>,
        continuation: Option<&Self::Continuation>,
        operation: impl for<'a> FnOnce(K::Input<'a>) -> O,
    ) -> Result<O, K::Error> {
        let input = PreparedCompositeInput::new(&self.input.0, &self.input.1)
            .and_then(|input| input.with_decoder_chunk(range, continuation))
            .map_err(eredu_nn::Error::backend)?;
        Ok(operation(
            <K as CompositeComponentKernel<B, S>>::prepared_input(input),
        ))
    }
    fn continuation(&self, completed: &K::ForwardContext) -> Self::Continuation {
        <K as CompositeComponentKernel<B, S>>::prefill_decoder_continuation(completed).cloned()
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

/// One indivisible prepared composite invocation. Architectures that have no
/// chunk continuation contract retain their existing whole-request semantics.
/// Clones share the immutable prepared tensors and admission rather than copying
/// payloads; observation and state-copy budgets remain with the session.
#[derive(Clone)]
pub struct SingleCompositePrefillRequest<T, P> {
    input: std::sync::Arc<(eredu_core::PreparedModelInput<T>, AdmittedCompositeInput<P>)>,
    tokens: usize,
}

impl<T, P> SingleCompositePrefillRequest<T, P> {
    /// Couples owned prepared tensors to their exact admission.
    pub fn new(
        prepared: eredu_core::PreparedModelInput<T>,
        admitted: AdmittedCompositeInput<P>,
    ) -> Result<Self, eredu_nn::Error> {
        PreparedCompositeInput::new(&prepared, &admitted).map_err(eredu_nn::Error::backend)?;
        let tokens =
            usize::try_from(admitted.decoder_positions()).map_err(eredu_nn::Error::backend)?;
        Ok(Self {
            input: std::sync::Arc::new((prepared, admitted)),
            tokens,
        })
    }
}

impl<B, D, K, S, R> ChunkedPrefillRequest<ComponentAdapter<D, K, R>, B, S>
    for SingleCompositePrefillRequest<B::Tensor, K::InputPartPlan>
where
    B: NeuralBackend,
    D: Borrow<ComponentDefinition> + 'static,
    R: ComponentResourceAccess<B, K> + 'static,
    S: RuntimeState<B>,
    K: CompositeComponentKernel<B, S>,
    K::Error: From<eredu_nn::Error>,
{
    type Continuation = ();
    fn token_count(&self) -> usize {
        self.tokens
    }
    fn maximum_chunk_tokens(&self) -> usize {
        self.tokens
    }
    fn with_chunk<O>(
        &self,
        range: std::ops::Range<usize>,
        _continuation: Option<&Self::Continuation>,
        operation: impl for<'a> FnOnce(K::Input<'a>) -> O,
    ) -> Result<O, K::Error> {
        if range != (0..self.tokens) {
            return Err(eredu_nn::Error::backend(
                "composite request has no partial invocation contract",
            )
            .into());
        }
        let paired = PreparedCompositeInput::new(&self.input.0, &self.input.1)
            .map_err(eredu_nn::Error::backend)?;
        Ok(operation(
            <K as CompositeComponentKernel<B, S>>::prepared_input(paired),
        ))
    }
    fn request_values(&self) -> Vec<&B::Tensor> {
        self.input
            .0
            .storage_values()
            .map(|(_, value)| value)
            .collect()
    }
    fn continuation(&self, _completed: &K::ForwardContext) {}
    fn retained_values<'a>(&self, _continuation: &'a ()) -> Vec<&'a B::Tensor> {
        Vec::new()
    }
}

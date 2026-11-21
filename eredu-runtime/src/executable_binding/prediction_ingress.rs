//! Portable input admission and chunk advancement for embedded prediction.

use eredu_core::{observation::ObserverTensorMapping, BackendFailure, BackendFailureKind};
use eredu_nn::{NeuralBackend, Tensor};
use std::marker::PhantomData;

use super::{
    embedded_erasure::EmbeddedTextBinding,
    text_invocation::{prepare_composite_prompt, text_tokens, token_part},
    CompositeSessionIngress,
};
use crate::{
    composite_execution::{CompositeArchitecture, PreparedCompositeInput},
    prefill::ChunkedPrefillRequest,
    speculative::{ReplicatedPredictionInput, SpeculativeTensorMechanisms},
    text_session_driver::{TextPrefillInput, TextPromptView},
    PreparedInputCacheIdentity, ReplicatedTextArchitecture, RuntimeState,
};

type NativeError<M> =
    <<M as EmbeddedTextBinding>::Mechanisms as SpeculativeTensorMechanisms>::Error;

pub(super) struct PredictionIngress<'a, I, M> {
    ingress: &'a I,
    representations: PhantomData<fn() -> M>,
}

impl<'a, I, M> PredictionIngress<'a, I, M> {
    pub(super) fn new(ingress: &'a I) -> Self {
        Self {
            ingress,
            representations: PhantomData,
        }
    }
}

fn invalid<M: EmbeddedTextBinding>(message: &'static str) -> NativeError<M> {
    M::input_failure(BackendFailure::new(
        BackendFailureKind::InvalidInput,
        std::io::Error::other(message),
    ))
}

fn sequence<M: EmbeddedTextBinding>(tokens: &M::Tensor) -> Result<usize, NativeError<M>> {
    let [_, sequence] = tokens.shape() else {
        return Err(invalid::<M>("prediction token ids must have rank two"));
    };
    usize::try_from(*sequence)
        .map_err(|_| invalid::<M>("prediction input has a negative sequence length"))
}

impl<A, B, S, M> ReplicatedPredictionInput<A, B, S, NativeError<M>> for PredictionIngress<'_, (), M>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>,
    M: EmbeddedTextBinding<Tensor = B::Tensor, Context = <B::Tensor as Tensor>::Context>,
{
    type Input = crate::text_session_driver::PreparedTextPrompt<M::InputTensor>;

    fn with_prefill_chunks(
        &mut self,
        input: Self::Input,
        maximum_chunk_tokens: usize,
        context: &M::Context,
        mut operation: impl for<'a> FnMut(
            A::Input<'a>,
            M::Tensor,
            Option<&PreparedInputCacheIdentity>,
        ) -> Result<A::ForwardContext, NativeError<M>>,
    ) -> Result<(), NativeError<M>> {
        if maximum_chunk_tokens == 0 {
            return Err(invalid::<M>(
                "prediction prefill chunk size must be positive",
            ));
        }
        let input = input.borrowed();
        let tokens = text_tokens::<M>(input, context)
            .map_err(|error| M::input_failure(BackendFailure::from_error(error)))?;
        let sequence = sequence::<M>(&tokens)?;
        if sequence > maximum_chunk_tokens && !A::supports_chunked_prefill() {
            return Err(invalid::<M>(
                "selected architecture cannot split prediction prefill input",
            ));
        }
        for start in (0..sequence).step_by(maximum_chunk_tokens) {
            let end = start.saturating_add(maximum_chunk_tokens).min(sequence);
            let chunk = if start == 0 && end == sequence {
                tokens.clone()
            } else {
                M::slice_tokens(&tokens, start..end, context)?
            };
            operation(
                A::text_input(&chunk, None),
                chunk.clone(),
                input.cache_identity(),
            )?;
        }
        Ok(())
    }

    fn with_decode<R>(
        &mut self,
        tokens: &M::Tensor,
        _: &M::Context,
        operation: impl for<'a> FnOnce(A::Input<'a>) -> Result<R, NativeError<M>>,
    ) -> Result<R, NativeError<M>> {
        operation(A::text_input(tokens, None))
    }
}

impl<A, B, S, M> ReplicatedPredictionInput<A, B, S, NativeError<M>>
    for PredictionIngress<'_, CompositeSessionIngress<A::AdmissionConfig>, M>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: CompositeArchitecture<B, S, Error = eredu_nn::Error>,
    M: EmbeddedTextBinding<Tensor = B::Tensor, Context = <B::Tensor as Tensor>::Context>,
{
    type Input = crate::text_session_driver::PreparedTextPrompt<M::InputTensor>;

    fn with_prefill_chunks(
        &mut self,
        input: Self::Input,
        maximum_chunk_tokens: usize,
        context: &M::Context,
        mut operation: impl for<'a> FnMut(
            A::Input<'a>,
            M::Tensor,
            Option<&PreparedInputCacheIdentity>,
        ) -> Result<A::ForwardContext, NativeError<M>>,
    ) -> Result<(), NativeError<M>> {
        let (prepared, admitted, identity) = prepare_composite_prompt::<A, B, S, M>(
            &self.ingress.admission,
            &self.ingress.processor,
            input.borrowed(),
            true,
        )
        .map_err(|error| M::input_failure(BackendFailure::from_error(error)))?;
        let paired = PreparedCompositeInput::new(&prepared, &admitted).map_err(|error| {
            M::input_failure(BackendFailure::from_error(std::io::Error::other(
                error.to_string(),
            )))
        })?;
        let tokens = A::prepared_prediction_token_ids(paired, context)
            .map_err(|error| M::input_failure(BackendFailure::from_error(error)))?;
        let sequence = sequence::<M>(&tokens)?;
        if maximum_chunk_tokens == 0 {
            return Err(invalid::<M>(
                "prediction prefill chunk size must be positive",
            ));
        }
        let request = A::prepare_prefill_request(prepared, admitted, context)
            .map_err(|error| M::input_failure(BackendFailure::from_error(error)))?;
        if request.token_count() != sequence {
            return Err(invalid::<M>(
                "prediction token count differs from prepared request",
            ));
        }
        let chunk_tokens = maximum_chunk_tokens.min(request.maximum_chunk_tokens());
        if chunk_tokens == 0 {
            return Err(invalid::<M>(
                "prepared prediction request has no chunk capacity",
            ));
        }
        let mut continuation = None;
        for start in (0..sequence).step_by(chunk_tokens) {
            let end = start.saturating_add(chunk_tokens).min(sequence);
            let chunk = M::slice_tokens(&tokens, start..end, context)?;
            let forward = request
                .with_chunk(start..end, continuation.as_ref(), |paired| {
                    operation(paired, chunk, identity.as_ref())
                })
                .map_err(|error| M::input_failure(BackendFailure::from_error(error)))??;
            let next = request.continuation(&forward);
            for value in request.retained_values(&next) {
                M::settle_input(value)?;
            }
            continuation = Some(next);
        }
        Ok(())
    }

    fn with_decode<R>(
        &mut self,
        tokens: &M::Tensor,
        _: &M::Context,
        operation: impl for<'a> FnOnce(A::Input<'a>) -> Result<R, NativeError<M>>,
    ) -> Result<R, NativeError<M>> {
        let part = token_part(M::prompt_tokens(
            M::ObserverMapping::borrow_invocation_tensor(tokens),
        ))
        .map_err(M::input_failure)?;
        let (prepared, admitted, _) = prepare_composite_prompt::<A, B, S, M>(
            &self.ingress.admission,
            &self.ingress.processor,
            TextPromptView::new(std::slice::from_ref(&part)),
            true,
        )
        .map_err(|error| M::input_failure(BackendFailure::from_error(error)))?;
        let paired = PreparedCompositeInput::new(&prepared, &admitted).map_err(|error| {
            M::input_failure(BackendFailure::from_error(std::io::Error::other(
                error.to_string(),
            )))
        })?;
        operation(A::prepared_input(paired))
    }
}

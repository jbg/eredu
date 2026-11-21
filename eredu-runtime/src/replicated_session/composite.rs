//! Invocation of admitted composite prompts on the installed text session.

use super::*;
use crate::composite_execution::{
    AdmittedCompositeInput, CompositeArchitecture, PreparedCompositeInput,
};
use eredu_core::capture::CaptureRequestShape;
use eredu_core::PreparedModelInput;

/// Prepared tensors, exact ingress plan, and optional reusable prefix identity.
pub type CompositePrompt<T, P> = (
    PreparedModelInput<T>,
    AdmittedCompositeInput<P>,
    Option<PreparedInputCacheIdentity>,
);

/// A physical invocation of an installed composite decoder.
pub enum CompositeTextInvocation<T, P> {
    /// An unobserved prefill with a final-position readout.
    FinalPositionPrefill {
        /// Exact prepared input or a local preparation failure.
        prompt: Result<CompositePrompt<T, P>, eredu_nn::Error>,
        /// An internal fragment overrides the input identity, including when nonfinal.
        fragment_identity: Option<Option<PreparedInputCacheIdentity>>,
    },
    /// Prefill with ordinary score rows, optional capture, and architecture chunking.
    Prefill {
        /// Exact prepared input or a local preparation failure.
        prompt: Result<CompositePrompt<T, P>, eredu_nn::Error>,
        /// Selected invocation geometry for bounded capture.
        capture: Option<CaptureRequestShape>,
        /// Whether a prediction extension participates in this invocation.
        prediction: bool,
    },
    /// Cached decode for the prepared input positions.
    Decode(Result<CompositePrompt<T, P>, eredu_nn::Error>),
}

impl<A, B, M, D> ReplicatedTextSession<A, B, M, D>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: CompositeArchitecture<B, M::State, Error = eredu_nn::Error>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    /// Invokes the composite architecture with an explicit attention visibility.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_composite_attention<O>(
        &mut self,
        config: &A::AdmissionConfig,
        prompt: &CompositePrompt<B::Tensor, A::InputPartPlan>,
        feedback: Option<&B::Tensor>,
        invocation: eredu_core::decoding::AttentionInvocation,
        pass: ExpertPass,
        context: &B::Executor,
        observer: &mut O,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        use eredu_core::decoding::AttentionInvocation;

        if !A::supports_attention_invocation(config, invocation)
            || (invocation == AttentionInvocation::CAUSAL && feedback.is_some())
        {
            return Err(ReplicatedTextSessionError::Contract(
                "selected composite rejects attention invocation".into(),
            ));
        }
        let input = Self::composite_attention_input(prompt, feedback, invocation)?;
        self.sequence_output_with_observer(input, pass, context, observer)
    }

    /// Materializes an admitted context and reports its exact decoder extent.
    pub fn execute_composite_context<O>(
        &mut self,
        prompt: &CompositePrompt<B::Tensor, A::InputPartPlan>,
        context: &B::Executor,
        observer: &mut O,
    ) -> Result<(B::Tensor, usize), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let count = usize::try_from(prompt.1.decoder_positions()).map_err(|_| {
            ReplicatedTextSessionError::Contract("prepared context position count overflows".into())
        })?;
        let input = Self::composite_attention_input(
            prompt,
            None,
            eredu_core::decoding::AttentionInvocation::CAUSAL,
        )?;
        self.sequence_output_with_observer(input, ExpertPass::Prefill, context, observer)
            .map(|output| (output, count))
    }

    fn composite_attention_input<'a>(
        prompt: &'a CompositePrompt<B::Tensor, A::InputPartPlan>,
        feedback: Option<&'a B::Tensor>,
        invocation: eredu_core::decoding::AttentionInvocation,
    ) -> Result<A::Input<'a>, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        let mut input = PreparedCompositeInput::new(&prompt.0, &prompt.1)
            .map_err(ReplicatedTextSessionError::Contract)?
            .with_attention_invocation(invocation);
        if invocation.visibility == eredu_core::decoding::AttentionVisibility::PrefixAndBlock {
            input = input.with_refinement(feedback);
        }
        Ok(A::prepared_input(input))
    }

    /// Advances the installed model's previously admitted composite cursor.
    pub fn advance_composite_prefill<O>(
        &mut self,
        cursor: &mut crate::prefill::ErasedPrefillCursor,
        context: &B::Executor,
        observer: &mut O,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
        A::PrefillRequest: 'static,
        <A::PrefillRequest as crate::prefill::ChunkedPrefillRequest<A, B, M::State>>::Continuation:
            'static,
    {
        let cursor = cursor.chunked_mut::<A, B, M::State>().map_err(|error| {
            ReplicatedTextSessionError::BeforeStateMutation(Box::new(
                ReplicatedTextSessionError::Architecture(eredu_nn::Error::backend_source(error)),
            ))
        })?;
        self.advance_prefill(cursor, context, observer)
            .map(|step| step.output)
    }
}

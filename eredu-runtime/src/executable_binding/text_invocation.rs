//! Portable invocation of bound replicated and composite sessions.

use eredu_core::{BackendFailure, BackendFailureKind, SubmissionBackend};
use eredu_nn::{
    BlockwiseAttentionBackend, DistributedNeuralBackend, GroupedNeuralBackend, HyperNeuralBackend,
    Tensor,
};

use crate::{
    composite_execution::CompositeArchitecture,
    prefill::ErasedPrefillCursor,
    replicated_session::{
        CompositePrompt, CompositeTextInvocation, PromptCacheInput, TextPromptCache,
    },
    speculative::{
        MaterializedPredictionExecutor, PredictionExtensionMaterializer,
        SpeculativeActivationExecution,
    },
    text_session_driver::{TextExecutable, TextPromptView},
    ReplicatedTextArchitecture, ReplicatedTextExecutionStrategy, ReplicatedTextInvocation,
    ReplicatedTextSession, ReplicatedTextSessionError, ReplicatedTextSessionMechanisms,
};

use super::{
    retained_executable::TextTensorMechanisms, BoundTextSession, CompositeSessionIngress,
    NoSelectedPrediction, SelectedPrediction,
};

/// Native tensors used by replicated and composite text invocations.
pub trait TextInvocationMechanisms: TextTensorMechanisms {
    /// Materializer used by installed prediction extensions, if supported.
    type PredictionMaterializer;

    /// Combines admitted native token parts for text-only input.
    fn concatenate_token_parts(
        parts: &[&Self::InputTensor],
        context: &Self::Context,
    ) -> Result<Self::Tensor, Self::Error>;

    /// Clones the native token handle for prepared composite ingress.
    fn prompt_tokens(tokens: &Self::InvocationTensor) -> Self::InputTensor;

    /// Clones a prepared decode handle for composite ingress.
    fn decode_prompt_tokens(tokens: &Self::Decode) -> Self::InputTensor;

    /// Reads decoder token IDs for exact persisted-prefix provenance.
    fn decoder_token_ids(value: &Self::Tensor) -> Result<Vec<u32>, Self::Error>;
}

/// Native queue and communication retained by a bound text session.
pub trait TextInvocationResources {
    /// Representation of the installed resource bundle.
    type Types: super::TextExecutableTypes;
    /// Native transfer and allocation mechanisms for retained weight pools.
    type BankStorage: crate::dense::DenseResidencyMechanisms
        + eredu_core::residency::NativeBankMechanisms<
            Backend: eredu_nn::GroupedNeuralBackend<
                Tensor = <Self::Types as super::TextExecutableTypes>::Tensor,
            >,
            BankError = <Self::Types as super::TextExecutableTypes>::Error,
        >;
    /// Exact queue used for session-level operations.
    fn context(&self) -> &<Self::Types as super::TextExecutableTypes>::Context;
    /// Whether the rank retains the selected partition control.
    fn has_partition_control(&self) -> bool;
}

/// Transfers typed state out of the backend's erased lane representation.
pub trait TextPredictionLane<S> {
    /// Failure of the erased state transfer.
    type Error;
    /// Stores an independently prepared typed state.
    fn from_state(state: S) -> Self;
    /// Takes the matching typed state.
    fn take_state(&mut self) -> Result<S, Self::Error>;
    /// Restores ownership of the typed state.
    fn restore_state(&mut self, state: S);
}

/// Exact installed prediction facts used in ordinary text invocation.
pub trait TextPredictionStatus<A, B, X> {
    /// Whether the route installs a prediction extension.
    const INSTALLED: bool;
    /// Reports the active extension's realized observation path.
    fn activation(&self) -> Option<SpeculativeActivationExecution>;
}

impl<A, B, X> TextPredictionStatus<A, B, X> for NoSelectedPrediction {
    const INSTALLED: bool = false;
    fn activation(&self) -> Option<SpeculativeActivationExecution> {
        None
    }
}

impl<A, B, X, P> TextPredictionStatus<A, B, X> for SelectedPrediction<P>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    X: PredictionExtensionMaterializer<B>,
    P: MaterializedPredictionExecutor<A, B, X>,
{
    const INSTALLED: bool = true;
    fn activation(&self) -> Option<SpeculativeActivationExecution> {
        self.extension.activation_execution(&self.selected)
    }
}

fn invocation_tensor<M: TextTensorMechanisms>(value: &M::InvocationTensor) -> M::Tensor {
    <M::ObserverMapping as eredu_core::observation::ObserverTensorMapping>::borrow_tensor(value)
        .clone()
}

pub(super) fn token_part<T>(tokens: T) -> Result<eredu_core::PreparedInputPart<T>, BackendFailure> {
    eredu_core::PreparedInputPart::new_with_extents(
        eredu_core::InputModality::Text,
        eredu_core::PreparedInputPayload::TokenIds(tokens),
        [],
        [],
    )
    .map_err(BackendFailure::from_error)
}

pub(super) fn text_tokens<M: TextInvocationMechanisms>(
    input: TextPromptView<'_, M::InputTensor>,
    context: &M::Context,
) -> Result<M::Tensor, M::Error> {
    use eredu_core::InputModality;
    use eredu_core::PreparedInputPayload;
    let parts = input
        .parts
        .iter()
        .map(|part| match (part.modality(), part.payload()) {
            (InputModality::Text, PreparedInputPayload::TokenIds(tokens)) => Ok(tokens),
            (InputModality::Text, PreparedInputPayload::Embeddings(_)) => {
                Err("text embeddings are not supported by this model".to_owned())
            }
            _ => Err(format!(
                "{} input is not supported by this model",
                part.modality().as_str()
            )),
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|reason| {
            BackendFailure::new(
                BackendFailureKind::InvalidInput,
                std::io::Error::other(reason),
            )
        })?;
    if parts.is_empty() {
        return Err(BackendFailure::new(
            BackendFailureKind::InvalidInput,
            std::io::Error::other("model input must contain at least one part"),
        )
        .into());
    }
    M::concatenate_token_parts(&parts, context)
}

fn session_failure<M, P, E>(error: ReplicatedTextSessionError<eredu_nn::Error, P, E>) -> M::Error
where
    M: TextTensorMechanisms,
    P: std::error::Error + Send + Sync + 'static,
    E: std::error::Error + Send + Sync + 'static,
{
    let preserved = error.state_preserved();
    let failure = BackendFailure::from_error(error);
    if preserved {
        failure.with_model_state_preserved(true).into()
    } else {
        failure.into()
    }
}

fn publish<M: TextInvocationMechanisms>(scores: M::Tensor) -> M::Scores {
    M::scores(scores)
}

/// Admits prepared composite input and its exact semantic identity before invocation.
pub fn prepare_composite_prompt<A, B, S, M>(
    admission: &A::AdmissionConfig,
    processor: &crate::SelectedProcessorExecution,
    input: TextPromptView<'_, M::InputTensor>,
    fingerprint_text: bool,
) -> Result<CompositePrompt<B::Tensor, A::InputPartPlan>, M::Error>
where
    M: TextTensorMechanisms<Tensor = B::Tensor>,
    B: eredu_nn::NeuralBackend,
    S: crate::RuntimeState<B>,
    A: CompositeArchitecture<B, S>,
{
    let prepared = M::prepared_input(input)?;
    let (admitted, identity) =
        crate::composite_execution::admit_selected_composite_input::<A, B, S>(
            admission,
            processor,
            &prepared,
            input.cache_identity(),
            fingerprint_text,
            &M::tensor_input_inspector(),
        )
        .map_err(BackendFailure::from_error)?;
    Ok((prepared, admitted, identity))
}

/// Projects composite geometry without requiring a cache fingerprint or executing the prompt.
pub fn composite_input_plans<A, B, S, M>(
    ingress: &CompositeSessionIngress<A::AdmissionConfig>,
    model_type: &str,
    input: TextPromptView<'_, M::InputTensor>,
) -> Result<Vec<super::retained_session::PreparedInputPartPlan>, eredu_core::CapabilityError>
where
    M: TextTensorMechanisms<Tensor = B::Tensor>,
    B: eredu_nn::NeuralBackend,
    S: crate::RuntimeState<B>,
    A: CompositeArchitecture<B, S>,
{
    let invalid = |reason: String| eredu_core::CapabilityError::UnsupportedInput {
        architecture: model_type.into(),
        reason,
    };
    let prepared = M::prepared_input(input).map_err(|error| invalid(error.to_string()))?;
    crate::composite_execution::validate_selected_composite_input(
        &ingress.processor,
        prepared.parts(),
    )
    .map_err(invalid)?;
    let admitted =
        A::admit_prepared_input(&ingress.admission, &prepared, &M::tensor_input_inspector())?;
    Ok(admitted.parts().iter().cloned().map(Into::into).collect())
}

impl<A, B, S, R, D, I, N, P> BoundTextSession<ReplicatedTextSession<A, B, R, D>, I, N, P>
where
    B: SubmissionBackend<Executor = <<B as eredu_nn::NeuralBackend>::Tensor as Tensor>::Context>,
    S: crate::RuntimeState<B>,
    R: ReplicatedTextSessionMechanisms<A, B, State = S>,
    R::PolicyError: std::error::Error + Send + Sync + 'static,
    R::Error: std::error::Error + Send + Sync + 'static,
    A: crate::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
    D: ReplicatedTextExecutionStrategy<A, B, S, R::ResidentPolicy, R::BoundedPolicy>,
    N: TextInvocationResources,
{
    fn prepare_causal_state<M>(&mut self) -> Result<M::PredictionState, M::Error>
    where
        M: TextTensorMechanisms<Tensor = B::Tensor, Context = B::Executor>,
        N: TextInvocationResources<Types = M>,
        M::PredictionState: TextPredictionLane<S, Error = M::Error>,
    {
        self.session
            .prepare_prediction_target_state(self.native.context())
            .map(M::PredictionState::from_state)
            .map_err(|error| BackendFailure::from_error(error).into())
    }

    fn exchange_causal_state<M>(
        &mut self,
        cache: &mut M::PredictionState,
        context: &M::Context,
    ) -> Result<(), M::Error>
    where
        M: TextTensorMechanisms<Tensor = B::Tensor, Context = B::Executor>,
        M::PredictionState: TextPredictionLane<S, Error = M::Error>,
    {
        let mut state = cache.take_state()?;
        let result = self
            .session
            .exchange_prediction_target_state(&mut state, context)
            .map_err(|error| BackendFailure::from_error(error).into());
        cache.restore_state(state);
        result
    }
}

impl<A, B, S, R, D, N, P, M> TextExecutable<M>
    for BoundTextSession<ReplicatedTextSession<A, B, R, D>, (), N, P>
where
    B: SubmissionBackend<Executor = <<B as eredu_nn::NeuralBackend>::Tensor as Tensor>::Context>
        + 'static,
    B::Executor: Sized,
    S: crate::RuntimeState<B>,
    R: ReplicatedTextSessionMechanisms<A, B, State = S>,
    R::PolicyError: std::error::Error + Send + Sync + 'static,
    R::Error: std::error::Error + Send + Sync + 'static,
    A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>,
    D: ReplicatedTextExecutionStrategy<A, B, S, R::ResidentPolicy, R::BoundedPolicy>,
    M: TextInvocationMechanisms<Tensor = B::Tensor, Context = B::Executor>,
    M::PredictionState: TextPredictionLane<S, Error = M::Error>,
    N: TextInvocationResources<Types = M>,
    P: TextPredictionStatus<A, B, M::PredictionMaterializer>,
{
    type Decode = M::Decode;
    type ObservationTensor = M::Tensor;
    type ObserverMapping = M::ObserverMapping;
    type Scores = M::Scores;
    type Context = M::Context;
    type Error = M::Error;
    type AttentionState = M::PredictionState;

    fn partition_activation_execution(
        &self,
    ) -> Option<crate::speculative::SpeculativeActivationExecution> {
        self.prediction.activation().or_else(|| {
            self.facts
                .capability()
                .shared_backbone_activation_execution()
        })
    }

    fn prepared_input_positions(&self, input: TextPromptView<'_, M::InputTensor>) -> Option<u64> {
        input.parts.iter().try_fold(0_u64, |total, part| {
            let plan = super::input_admission::text_only_input_part(
                self.facts.effective_model_type(),
                part,
                &M::input_inspector(),
            )
            .ok()?;
            total.checked_add(plan.decoder_positions())
        })
    }

    fn accepts_split_prefill(&self) -> bool {
        !P::INSTALLED
            && A::supports_chunked_prefill()
            && self.session.prefill_chunking_support().is_ok()
    }

    fn block_refinement_geometry(
        &self,
    ) -> Option<crate::generation::refinement::architecture::BlockRefinementGeometry> {
        self.session.block_refinement_geometry()
    }

    fn accepts_final_position_prefill(&self) -> bool {
        !P::INSTALLED
    }

    fn prefill_final_position(
        &mut self,
        input: TextPromptView<'_, M::InputTensor>,
        stream: &M::Context,
    ) -> Result<M::Scores, M::Error> {
        let tokens = text_tokens::<M>(input, stream).map_err(eredu_nn::Error::backend_source);
        let committed = self.session.tracks_local_tokens().then(|| {
            tokens
                .as_ref()
                .ok()
                .and_then(|tokens| M::decoder_token_ids(tokens).ok())
        });
        let output = self
            .execute_text(
                ReplicatedTextInvocation::FinalPositionPrefill(tokens),
                stream,
                &mut crate::NoopObserver,
            )
            .map_err(session_failure::<M, _, _>)?;
        if let Some(committed) = committed {
            self.session.record_local_tokens(committed.as_deref());
        }
        Ok(publish::<M>(output))
    }

    fn prepare_autoregressive_cache(&mut self) -> Result<M::PredictionState, M::Error> {
        self.prepare_causal_state::<M>()
    }

    fn exchange_autoregressive_cache(
        &mut self,
        cache: &mut M::PredictionState,
        context: &M::Context,
    ) -> Result<(), M::Error> {
        self.exchange_causal_state::<M>(cache, context)
    }

    fn projects_final_prefill_position(&self) -> bool {
        !P::INSTALLED && D::SELECTS_FINAL_TEXT_OUTPUT && A::projects_final_text_position()
    }

    fn prepared_context_forward(
        &mut self,
        input: TextPromptView<'_, M::InputTensor>,
        cache: &mut M::PredictionState,
        context: &M::Context,
        observer: &mut dyn crate::ActivationObserver<M::InvocationTensor, M::Error>,
    ) -> Result<(M::Scores, usize), M::Error> {
        use eredu_core::observation::ObserverTensorMapping;
        let tokens = text_tokens::<M>(input, context)?;
        let [_, length] = tokens.shape() else {
            return Err(BackendFailure::new(
                BackendFailureKind::InvalidInput,
                std::io::Error::other("prepared causal context requires batch and sequence axes"),
            )
            .into());
        };
        let count = usize::try_from(*length).map_err(BackendFailure::from_error)?;
        let output = self.attention_forward_with_observer(
            M::ObserverMapping::borrow_invocation_tensor(&tokens),
            None,
            eredu_core::decoding::AttentionInvocation::CAUSAL,
            Some(cache),
            true,
            context,
            observer,
        )?;
        Ok((output, count))
    }

    fn attention_forward_with_observer(
        &mut self,
        tokens: &M::InvocationTensor,
        feedback: Option<&M::InvocationTensor>,
        invocation: eredu_core::decoding::AttentionInvocation,
        cache: Option<&mut M::PredictionState>,
        prefill: bool,
        stream: &M::Context,
        observer: &mut dyn crate::ActivationObserver<M::InvocationTensor, M::Error>,
    ) -> Result<M::Scores, M::Error> {
        let tokens = invocation_tensor::<M>(tokens);
        let feedback = feedback.map(invocation_tensor::<M>);
        let pass = if prefill {
            crate::ExpertPass::Prefill
        } else {
            crate::ExpertPass::Decode
        };
        let mut observer =
            crate::inspection::MappedActivationObserver::<_, M::TensorObserver>::new(observer);
        if let Some(cache) = cache {
            let mut lane = cache.take_state()?;
            let output = self.session.with_prediction_target_state::<_, M::Error>(
                &mut lane,
                stream,
                |session| {
                    session
                        .execute_text_attention(
                            &tokens,
                            feedback.as_ref(),
                            invocation,
                            pass,
                            stream,
                            &mut observer,
                        )
                        .map_err(|error| BackendFailure::from_error(error).into())
                },
                |error| BackendFailure::from_error(error).into(),
            );
            cache.restore_state(lane);
            let output = output?;
            Ok(publish::<M>(output))
        } else {
            self.session
                .execute_text_attention(
                    &tokens,
                    feedback.as_ref(),
                    invocation,
                    pass,
                    stream,
                    &mut observer,
                )
                .map(M::scores)
                .map_err(session_failure::<M, _, _>)
        }
    }

    fn effective_model_type(&self) -> &str {
        self.facts.effective_model_type()
    }

    fn capability_estimate(&self) -> &crate::CapabilityEstimate {
        self.facts.capability()
    }

    fn has_partition_control(&self) -> bool {
        self.native.has_partition_control()
    }

    fn partition_public_output(&self) -> bool {
        self.publishes_scores
    }

    fn has_embedded_prediction(&self) -> bool {
        P::INSTALLED
    }

    fn prompt_cache_model_identity(&self) -> Option<&eredu_core::cache::PromptCacheModelIdentity> {
        self.session.prompt_cache_model_identity()
    }

    fn prompt_cache_prefix_identity(
        &self,
        prefix: &[u32],
        input: Option<&crate::PreparedInputCacheIdentity>,
    ) -> String {
        if self.session.tracks_local_tokens() {
            eredu_core::cache::prompt_cache_token_fingerprint(prefix)
        } else {
            input.map_or_else(
                || eredu_core::cache::prompt_cache_token_fingerprint(prefix),
                |identity| identity.prefix_content_fingerprint().to_owned(),
            )
        }
    }

    fn prompt_cache<'a>(
        &'a mut self,
        _: &'a M::Context,
    ) -> Option<TextPromptCache<'a, M::Context>> {
        Some(
            self.session
                .prompt_cache(self.native.context(), PromptCacheInput::Tokens),
        )
    }

    fn reset_local(&mut self) -> Result<(), M::Error> {
        self.session
            .reset(self.native.context())
            .map_err(|error| BackendFailure::from_error(error).into())
    }

    fn reset_partitioned(&mut self) -> Result<(), M::Error> {
        self.session
            .reset_distributed(self.native.context())
            .map_err(|error| BackendFailure::from_error(error).into())
    }

    fn prefill_result_with_observer(
        &mut self,
        input: Result<TextPromptView<'_, M::InputTensor>, M::Error>,
        mask: Option<&M::InvocationTensor>,
        capture_geometry: Option<eredu_core::capture::CaptureRequestShape>,
        stream: &M::Context,
        observer: &mut dyn crate::ActivationObserver<M::InvocationTensor, M::Error>,
    ) -> Result<M::Scores, M::Error> {
        let tokens = input
            .and_then(|input| text_tokens::<M>(input, stream))
            .map_err(eredu_nn::Error::backend_source);
        let committed = self.session.tracks_local_tokens().then(|| {
            tokens
                .as_ref()
                .ok()
                .and_then(|tokens| M::decoder_token_ids(tokens).ok())
        });
        let mask = mask.map(invocation_tensor::<M>);
        let mut observer =
            crate::inspection::MappedActivationObserver::<_, M::TensorObserver>::new(observer);
        let output = self
            .execute_text(
                ReplicatedTextInvocation::Prefill {
                    tokens,
                    mask,
                    capture: capture_geometry,
                },
                stream,
                &mut observer,
            )
            .map_err(session_failure::<M, _, _>)?;
        if let Some(committed) = committed {
            self.session.record_local_tokens(committed.as_deref());
        }
        Ok(publish::<M>(output))
    }

    fn decode_result_with_observer(
        &mut self,
        tokens: Result<&M::Decode, M::Error>,
        stream: &M::Context,
        observer: &mut dyn crate::ActivationObserver<M::InvocationTensor, M::Error>,
    ) -> Result<M::Scores, M::Error> {
        let tokens = tokens.map(M::decode_tensor);
        let committed = self.session.tracks_local_tokens().then(|| {
            tokens
                .as_ref()
                .ok()
                .and_then(|tokens| M::decoder_token_ids(tokens).ok())
        });
        let mut observer =
            crate::inspection::MappedActivationObserver::<_, M::TensorObserver>::new(observer);
        let output = self
            .execute_text(
                ReplicatedTextInvocation::Decode(tokens.map_err(eredu_nn::Error::backend_source)),
                stream,
                &mut observer,
            )
            .map_err(session_failure::<M, _, _>)?;
        if let Some(committed) = committed {
            self.session.record_local_tokens(committed.as_deref());
        }
        Ok(publish::<M>(output))
    }
}

impl<A, B, S, R, D, N, P, M> TextExecutable<M>
    for BoundTextSession<
        ReplicatedTextSession<A, B, R, D>,
        CompositeSessionIngress<A::AdmissionConfig>,
        N,
        P,
    >
where
    B: SubmissionBackend<Executor = <<B as eredu_nn::NeuralBackend>::Tensor as Tensor>::Context>
        + 'static,
    B::Executor: Sized,
    S: crate::RuntimeState<B> + 'static,
    R: ReplicatedTextSessionMechanisms<A, B, State = S>,
    R::PolicyError: std::error::Error + Send + Sync + 'static,
    R::Error: std::error::Error + Send + Sync + 'static,
    A: CompositeArchitecture<B, S, Error = eredu_nn::Error>,
    D: ReplicatedTextExecutionStrategy<A, B, S, R::ResidentPolicy, R::BoundedPolicy>,
    M: TextInvocationMechanisms<Tensor = B::Tensor, Context = B::Executor>,
    M::PredictionState: TextPredictionLane<S, Error = M::Error>,
    N: TextInvocationResources<Types = M>,
    P: TextPredictionStatus<A, B, M::PredictionMaterializer>,
    A::InputPartPlan: 'static,
{
    type Decode = M::Decode;
    type ObservationTensor = M::Tensor;
    type ObserverMapping = M::ObserverMapping;
    type Scores = M::Scores;
    type Context = M::Context;
    type Error = M::Error;
    type AttentionState = M::PredictionState;

    fn partition_activation_execution(
        &self,
    ) -> Option<crate::speculative::SpeculativeActivationExecution> {
        self.prediction.activation().or_else(|| {
            self.facts
                .capability()
                .shared_backbone_activation_execution()
        })
    }

    fn prepared_input_plans(
        &self,
        input: TextPromptView<'_, M::InputTensor>,
    ) -> Result<Vec<super::retained_session::PreparedInputPartPlan>, eredu_core::CapabilityError>
    {
        composite_input_plans::<A, B, S, M>(&self.ingress, self.facts.effective_model_type(), input)
    }

    fn prepared_input_positions(&self, input: TextPromptView<'_, M::InputTensor>) -> Option<u64> {
        composite_input_plans::<A, B, S, M>(&self.ingress, self.facts.effective_model_type(), input)
            .ok()?
            .iter()
            .try_fold(0_u64, |total, part| {
                total.checked_add(part.decoder_positions())
            })
    }

    fn accepts_split_prefill(&self) -> bool {
        !P::INSTALLED
            && self.session.prefill_chunking_support().is_ok()
            && (A::supports_chunked_text_prefill()
                || <Self as TextExecutable<M>>::retains_prefill_cursor(self))
    }

    fn retains_prefill_cursor(&self) -> bool {
        A::supports_retained_prefill() && self.session.prefill_chunking_support().is_ok()
    }

    fn block_refinement_geometry(
        &self,
    ) -> Option<crate::generation::refinement::architecture::BlockRefinementGeometry> {
        A::block_refinement_geometry(&self.ingress.admission)
    }

    fn accepts_final_position_prefill(&self) -> bool {
        !P::INSTALLED && D::SELECTS_FINAL_TEXT_OUTPUT && A::projects_final_text_position()
    }

    fn prefill_final_position(
        &mut self,
        input: TextPromptView<'_, M::InputTensor>,
        stream: &M::Context,
    ) -> Result<M::Scores, M::Error> {
        // Internal fragments carry whole-request identity separately from the
        // suffix tensor geometry. Ordinary caller-supplied identities still go
        // through exact prepared-input validation.
        let prepared = prepare_composite_prompt::<A, B, S, M>(
            &self.ingress.admission,
            &self.ingress.processor,
            if input.is_prefill_fragment() {
                TextPromptView::new(input.parts)
            } else {
                input
            },
            true,
        )
        .map_err(eredu_nn::Error::backend_source);
        let fragment_identity = input
            .is_prefill_fragment()
            .then(|| input.cache_identity().cloned());
        let output = self
            .execute_composite_text(
                CompositeTextInvocation::FinalPositionPrefill {
                    prompt: prepared,
                    fragment_identity,
                },
                stream,
                &mut crate::NoopObserver,
            )
            .map_err(session_failure::<M, _, _>)?;
        Ok(publish::<M>(output))
    }

    fn start_prefill_cursor(
        &mut self,
        input: TextPromptView<'_, M::InputTensor>,
        max_tokens: std::num::NonZeroUsize,
        capture_geometry: Option<eredu_core::capture::CaptureRequestShape>,
        stream: &M::Context,
    ) -> Result<Option<ErasedPrefillCursor>, M::Error> {
        let prepared = prepare_composite_prompt::<A, B, S, M>(
            &self.ingress.admission,
            &self.ingress.processor,
            input,
            true,
        );
        self.session
            .start_composite_prefill(
                prepared.map_err(eredu_nn::Error::backend_source),
                max_tokens.get(),
                capture_geometry,
                stream,
            )
            .map(Some)
            .map_err(|error| {
                M::Error::from(BackendFailure::from_error(error).with_model_state_preserved(true))
            })
    }

    fn advance_prefill_cursor(
        &mut self,
        cursor: &mut ErasedPrefillCursor,
        stream: &M::Context,
        observer: &mut dyn crate::ActivationObserver<M::InvocationTensor, M::Error>,
    ) -> Result<M::Scores, M::Error> {
        let mut observer =
            crate::inspection::MappedActivationObserver::<_, M::TensorObserver>::new(observer);
        let output = self
            .session
            .advance_composite_prefill(cursor, stream, &mut observer)
            .map_err(session_failure::<M, _, _>)?;
        Ok(publish::<M>(output))
    }

    fn prepare_autoregressive_cache(&mut self) -> Result<M::PredictionState, M::Error> {
        self.prepare_causal_state::<M>()
    }

    fn exchange_autoregressive_cache(
        &mut self,
        cache: &mut M::PredictionState,
        context: &M::Context,
    ) -> Result<(), M::Error> {
        self.exchange_causal_state::<M>(cache, context)
    }

    fn projects_final_prefill_position(&self) -> bool {
        !P::INSTALLED && D::SELECTS_FINAL_TEXT_OUTPUT && A::projects_final_text_position()
    }

    fn prepared_context_forward(
        &mut self,
        input: TextPromptView<'_, M::InputTensor>,
        cache: &mut M::PredictionState,
        context: &M::Context,
        observer: &mut dyn crate::ActivationObserver<M::InvocationTensor, M::Error>,
    ) -> Result<(M::Scores, usize), M::Error> {
        let prepared = prepare_composite_prompt::<A, B, S, M>(
            &self.ingress.admission,
            &self.ingress.processor,
            input,
            false,
        )?;
        let mut lane = cache.take_state()?;
        let result = self.session.with_prediction_target_state::<_, M::Error>(
            &mut lane,
            self.native.context(),
            |session| session.execute_composite_context(
                &prepared,
                context,
                &mut crate::inspection::MappedActivationObserver::<_, M::TensorObserver>::new(observer),
            ).map_err(|error| BackendFailure::from_error(error).into()),
            |error| BackendFailure::from_error(error).into(),
        );
        cache.restore_state(lane);
        let (output, count) = result?;
        Ok((publish::<M>(output), count))
    }

    fn attention_forward_with_observer(
        &mut self,
        tokens: &M::InvocationTensor,
        feedback: Option<&M::InvocationTensor>,
        invocation: eredu_core::decoding::AttentionInvocation,
        cache: Option<&mut M::PredictionState>,
        prefill: bool,
        stream: &M::Context,
        observer: &mut dyn crate::ActivationObserver<M::InvocationTensor, M::Error>,
    ) -> Result<M::Scores, M::Error> {
        let part = token_part(M::prompt_tokens(tokens))?;
        let prepared = prepare_composite_prompt::<A, B, S, M>(
            &self.ingress.admission,
            &self.ingress.processor,
            TextPromptView::new(&[part]),
            true,
        )?;
        let feedback = feedback.map(invocation_tensor::<M>);
        let pass = if prefill {
            crate::ExpertPass::Prefill
        } else {
            crate::ExpertPass::Decode
        };
        let mut observer =
            crate::inspection::MappedActivationObserver::<_, M::TensorObserver>::new(observer);
        let mut execute = |session: &mut ReplicatedTextSession<_, _, _, _>,
                           admission: &A::AdmissionConfig| {
            session
                .execute_composite_attention(
                    admission,
                    &prepared,
                    feedback.as_ref(),
                    invocation,
                    pass,
                    stream,
                    &mut observer,
                )
                .map_err(session_failure::<M, _, _>)
        };
        let output = match cache {
            Some(cache) => {
                let mut lane = cache.take_state()?;
                let result = self.session.with_prediction_target_state::<_, M::Error>(
                    &mut lane,
                    self.native.context(),
                    |session| execute(session, &self.ingress.admission),
                    |error| BackendFailure::from_error(error).into(),
                );
                cache.restore_state(lane);
                result?
            }
            None => execute(&mut self.session, &self.ingress.admission)?,
        };
        Ok(publish::<M>(output))
    }

    fn effective_model_type(&self) -> &str {
        self.facts.effective_model_type()
    }

    fn capability_estimate(&self) -> &crate::CapabilityEstimate {
        self.facts.capability()
    }

    fn has_embedded_prediction(&self) -> bool {
        P::INSTALLED
    }

    fn has_partition_control(&self) -> bool {
        self.native.has_partition_control()
    }

    fn partition_public_output(&self) -> bool {
        self.publishes_scores
    }

    fn prompt_cache_model_identity(&self) -> Option<&eredu_core::cache::PromptCacheModelIdentity> {
        self.session.prompt_cache_model_identity()
    }

    fn prompt_cache<'a>(
        &'a mut self,
        _: &'a M::Context,
    ) -> Option<TextPromptCache<'a, M::Context>> {
        Some(
            self.session
                .prompt_cache(self.native.context(), PromptCacheInput::PreparedInput),
        )
    }

    fn reset_local(&mut self) -> Result<(), M::Error> {
        self.session
            .reset(self.native.context())
            .map_err(|error| BackendFailure::from_error(error).into())
    }

    fn reset_partitioned(&mut self) -> Result<(), M::Error> {
        self.session
            .reset_distributed(self.native.context())
            .map_err(|error| BackendFailure::from_error(error).into())
    }

    fn prefill_result_with_observer(
        &mut self,
        input: Result<TextPromptView<'_, M::InputTensor>, M::Error>,
        mask: Option<&M::InvocationTensor>,
        capture_geometry: Option<eredu_core::capture::CaptureRequestShape>,
        stream: &M::Context,
        observer: &mut dyn crate::ActivationObserver<M::InvocationTensor, M::Error>,
    ) -> Result<M::Scores, M::Error> {
        let prepared = input.and_then(|input| {
            if mask.is_some() {
                return Err(BackendFailure::new(
                    BackendFailureKind::InvalidInput,
                    std::io::Error::other(
                        "explicit composite decoder masks require prepared-input metadata",
                    ),
                )
                .into());
            }
            prepare_composite_prompt::<A, B, S, M>(
                &self.ingress.admission,
                &self.ingress.processor,
                input,
                true,
            )
        });
        let mut observer =
            crate::inspection::MappedActivationObserver::<_, M::TensorObserver>::new(observer);
        let output = self
            .execute_composite_text(
                CompositeTextInvocation::Prefill {
                    prompt: prepared.map_err(eredu_nn::Error::backend_source),
                    capture: capture_geometry,
                    prediction: P::INSTALLED,
                },
                stream,
                &mut observer,
            )
            .map_err(session_failure::<M, _, _>)?;
        Ok(publish::<M>(output))
    }

    fn decode_result_with_observer(
        &mut self,
        tokens: Result<&M::Decode, M::Error>,
        stream: &M::Context,
        observer: &mut dyn crate::ActivationObserver<M::InvocationTensor, M::Error>,
    ) -> Result<M::Scores, M::Error> {
        let prepared = tokens.and_then(|tokens| {
            prepare_composite_prompt::<A, B, S, M>(
                &self.ingress.admission,
                &self.ingress.processor,
                TextPromptView::new(&[token_part(M::decode_prompt_tokens(tokens))?]),
                true,
            )
        });
        let mut observer =
            crate::inspection::MappedActivationObserver::<_, M::TensorObserver>::new(observer);
        let output = self
            .execute_composite_text(
                CompositeTextInvocation::Decode(prepared.map_err(eredu_nn::Error::backend_source)),
                stream,
                &mut observer,
            )
            .map_err(session_failure::<M, _, _>)?;
        Ok(publish::<M>(output))
    }
}

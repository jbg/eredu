//! Portable ownership of resident shared-backbone proposal and commitment state.

use crate::speculative::autoregressive::{
    MappedSpeculativeTargetObserver, SpeculativeTargetObserver,
};
use crate::text_session_driver::TextSessionResourcesFor;
use eredu_core::{
    backend::{SharedTextNative, TextExecutableTypes},
    capture::SpeculativeCapturePrimitives,
    observation::ObserverTensorMapping,
    speculative::SpeculativeQueueMechanisms,
};

use crate::run_preparation::TextPreparationAgreementError;
use std::ops::DerefMut;

use eredu_core::{
    capture::CapturePhase,
    decoding::{DecodingCoordination, DecodingMode},
    Completion, GenerationError, ModelRuntime, PendingTextInput, Submission, TextOperationError,
    TextOperationOutput, TextOperationPolicyError, TextProgress, TokenSamplingDecision,
};
use eredu_nn::Error as NeuralError;

use crate::{
    capture::{partition::PartitionCaptureHookTransport, CaptureSession},
    speculative::{
        autoregressive::AutoregressiveMechanisms,
        shared_backbone::{SharedBackboneDraftOptions, SharedBackboneMechanisms},
    },
    text_session_driver::{
        capture::TextCaptureMechanisms, causal::CausalTextMechanisms, TextDecodeInput,
        TextExecutable, TextSessionDriver,
    },
    NoopObserver,
};

use super::{Input, SpecializedTextSubmission, State, TextModeBinding, TextModeState};

type Shared<B> = <B as TextModeBinding>::Shared;
type Scores<B> = <Shared<B> as AutoregressiveMechanisms>::Logits;
type Randomness<B> = <Shared<B> as SharedBackboneMechanisms>::ProposalRandomness;

/// Request-owned state borrowed by the shared-backbone driver.
struct SharedTextState<'a, B: TextModeBinding> {
    /// Canonical mode and tentative verification transaction.
    pub decoding: &'a mut TextModeState<B::Refinement, B::Shared>,
    /// Request-owned proposal stream, when initialized.
    pub randomness: &'a mut Option<Randomness<B>>,
    /// Seed used to initialize the request's random streams.
    pub seed: u64,
    /// Absolute position of the next decision.
    pub prediction: &'a mut u64,
    /// Whether to report tentative proposal tokens.
    pub draft_events: bool,
    /// Bounded capture retained by this request.
    pub capture: Option<&'a mut CaptureSession>,
}

fn shared_state<B: TextModeBinding>(state: &mut State<B>) -> SharedTextState<'_, B> {
    SharedTextState {
        decoding: &mut state.sampling.decoding,
        randomness: &mut state.sampling.proposal_random,
        seed: state.sampling.refinement_seed,
        prediction: &mut state.sampling.next_prediction,
        draft_events: state.sampling.draft_events,
        capture: state.capture.as_mut(),
    }
}

/// Portable shared-backbone invocation over an already selected native queue.
pub(super) trait SharedTextMechanisms: TextModeBinding + SharedTextNative {
    /// Binds one portable target invocation and its optional observer.
    fn shared_context<'a, 'capture: 'a>(
        backend: &'a Self,
        observer: Option<&'a MappedSpeculativeTargetObserver<'capture, Mapping<Self>>>,
    ) -> <Shared<Self> as AutoregressiveMechanisms>::Context<'a>;
}

type NativeTypes<B> =
    crate::text_session_driver::SessionTextTypes<B, <B as SharedTextNative>::Native>;
type Mapping<B> = <NativeTypes<B> as TextExecutableTypes>::ObserverMapping;
type Capture<B> = <Shared<B> as SharedBackboneMechanisms>::Capture;
type Queues<B> = <Capture<B> as SpeculativeCapturePrimitives>::Queues;
type InvocationObserver<'a, B> = dyn SpeculativeTargetObserver<
        <Mapping<B> as ObserverTensorMapping>::InvocationTensor,
        <Mapping<B> as ObserverTensorMapping>::InvocationError,
        Scores<B>,
    > + 'a;

impl<B> SharedTextMechanisms for B
where
    B: TextModeBinding + SharedTextNative,
    Capture<B>: SpeculativeCapturePrimitives<
        Queues: SpeculativeQueueMechanisms<
            Queue = <NativeTypes<B> as TextExecutableTypes>::Context,
        >,
    >,
    Shared<B>: for<'a> AutoregressiveMechanisms<
        Context<'a> = crate::speculative::SpeculativeExecutionContext<
            'a,
            Queues<B>,
            <Capture<B> as SpeculativeCapturePrimitives>::Transport,
            InvocationObserver<'a, B>,
        >,
    >,
{
    fn shared_context<'a, 'capture: 'a>(
        backend: &'a B,
        observer: Option<&'a MappedSpeculativeTargetObserver<'capture, Mapping<B>>>,
    ) -> <Shared<B> as AutoregressiveMechanisms>::Context<'a> {
        let context =
            crate::speculative::SpeculativeExecutionContext::single(backend.shared_queue());
        match observer {
            Some(observer) => context.with_target_observer(observer),
            None => context,
        }
    }
}

/// Admitted next operation for resident shared-backbone drafting.
pub enum SharedTextProposal {
    /// Commit the final causal input when no proposal fits the remaining budget.
    CausalTail,
    /// Propose and verify within the exact portable admission.
    Verify {
        /// Maximum proposals within the remaining request budget.
        capacity: std::num::NonZeroUsize,
        /// Numerical options checked against the retained predictor.
        draft_options: SharedBackboneDraftOptions,
        /// Exact completion limit selected for verification.
        completion_wait: eredu_core::BoundedCompletionWait,
    },
}

/// Portable phase execution for a text session with a retained shared backbone.
pub(super) trait SharedTextDriver<B: TextModeBinding>:
    eredu_core::BackendSession<B> + Sized
{
    /// Numerical policy retained by the selected shared-backbone executable.
    fn shared_policy(
        &self,
    ) -> Option<crate::speculative::shared_backbone::SharedBackboneExecutionPolicy>;

    fn propose(
        runtime: &mut ModelRuntime<B, Self>,
        committed: B::Token,
        state: &mut State<B>,
        proposal: SharedTextProposal,
    ) -> Result<SpecializedTextSubmission<B>, B::Error>;

    fn select(
        runtime: &mut ModelRuntime<B, Self>,
        decision: &TokenSamplingDecision<'_>,
        state: &mut State<B>,
    ) -> Result<SpecializedTextSubmission<B>, B::Error>;

    fn coordinate_mode(
        runtime: &ModelRuntime<B, Self>,
        local: DecodingCoordination,
    ) -> Result<DecodingCoordination, TextOperationError<B::Error>>;

    fn settle_mode(
        runtime: &mut ModelRuntime<B, Self>,
        pending: &mut Option<Input<B>>,
        state: &mut State<B>,
        requested: DecodingMode,
    ) -> Result<bool, TextOperationError<B::Error>>;

    fn commit(
        runtime: &mut ModelRuntime<B, Self>,
        token: u32,
        state: &mut State<B>,
    ) -> Result<(), B::Error>;
}

impl<B, R, N> SharedTextDriver<B> for TextSessionDriver<R, N>
where
    B: SharedTextMechanisms<
        Native = N,
        Shared: SharedBackboneMechanisms<Model = N::Executable, Output: Clone>,
    >,
    Shared<B>: eredu_core::backend::CommittedTokenReadback<
        B::Token,
        <B::Refinement as eredu_core::refinement::RefinementBackend>::Token,
        <Shared<B> as SharedBackboneMechanisms>::Queue,
        <Shared<B> as AutoregressiveMechanisms>::Error,
    >,
    crate::text_session_driver::TextPromptFor<B>: crate::text_session_driver::TextPrefillInput,
    B::Error: From<<Shared<B> as AutoregressiveMechanisms>::Error>
        + From<eredu_core::BackendFailure>
        + From<NeuralError>
        + From<GenerationError>
        + From<eredu_core::SpeculativeDriverError<<Shared<B> as AutoregressiveMechanisms>::Error>>,
    N: CausalTextMechanisms<
            B,
            Resources = R,
            Token = B::Token,
            SamplingCompletion = B::TextCompletion,
            TextValues: eredu_core::backend::NativeTextValues<Scores = Scores<B>>,
            Executable: TextExecutable<
                crate::text_session_driver::SessionTextTypes<B, N>,
                Scores = Scores<B>,
            >,
            Prefill = crate::text_session_driver::TextPromptFor<B>,
        > + TextCaptureMechanisms<
            N::Retained,
            Transport: PartitionCaptureHookTransport<
                Error: Send + Sync + 'static,
                Completion: Completion<Error: Send + Sync + 'static>,
            >,
        >,
    R: DerefMut<Target = TextSessionResourcesFor<B, N>> + 'static,
{
    fn shared_policy(
        &self,
    ) -> Option<crate::speculative::shared_backbone::SharedBackboneExecutionPolicy> {
        self.shared_backbone_policy::<B>()
    }

    fn propose(
        runtime: &mut ModelRuntime<B, Self>,
        committed: B::Token,
        state: &mut State<B>,
        proposal: SharedTextProposal,
    ) -> Result<SpecializedTextSubmission<B>, B::Error> {
        let state = shared_state::<B>(state);
        let (backend, session) = runtime.parts_mut();
        let last = B::shared_token_id(backend, &committed)?;
        let root = match proposal {
            SharedTextProposal::CausalTail => {
                let submission = session.advance_observed_decode(
                    backend,
                    TextDecodeInput::TokenId(last),
                    &mut NoopObserver,
                )?;
                submission.completion.wait()?;
                state.decoding.next_logits = Some(submission.output.clone());
                state.decoding.committed_token_cached = true;
                submission.output
            }
            SharedTextProposal::Verify {
                capacity,
                draft_options,
                completion_wait,
            } => {
                let random = match state.randomness.take() {
                    Some(random) => random,
                    None => {
                        let root = Shared::<B>::model_randomness_root(state.seed)?;
                        Shared::<B>::split_model_randomness(
                            &root,
                            2,
                            B::shared_context(backend, None),
                        )?
                    }
                };
                let prepared;
                let mut collector;
                let mut noop = NoopObserver;
                let observer = if let Some(capture) = state.capture {
                    let invocation = session
                        .text_capture_shape((capacity.get() + 1) as u64)
                        .map(|shape| (CapturePhase::Decode, shape));
                    prepared =
                        session.prepare_capture_invocation(capture, *state.prediction, invocation);
                    collector = prepared.open::<N::Collector>(
                        capture,
                        N::context(backend),
                        None,
                        *state.prediction,
                    );
                    &mut *collector
                } else {
                    &mut noop
                };
                let transaction = {
                    let observer =
                        crate::speculative::autoregressive::MappedSpeculativeTargetObserver::<
                            Mapping<B>,
                        >::new(observer);
                    session.propose_shared::<B, Shared<B>>(
                        backend,
                        capacity,
                        last,
                        draft_options,
                        random,
                        completion_wait,
                        B::shared_context(backend, Some(&observer)),
                    )?
                };
                observer.finish()?;
                *state.randomness = Some(transaction.proposal_randomness().clone());
                state.decoding.speculative = Some(Box::new(transaction));
                B::shared_progress_value()
            }
        };
        let completion = B::finish_shared_progress(session.retain_scores(backend, root)?)?;
        Ok(Submission {
            output: TextOperationOutput::Progress {
                pending: PendingTextInput::Continue,
                progress: TextProgress {
                    phase: eredu_core::refinement::GenerationPhase::Refinement,
                    block: *state.prediction,
                    iteration: 1,
                    tentative_token_ids: state.draft_events.then(|| {
                        state
                            .decoding
                            .speculative
                            .as_ref()
                            .map_or_else(Vec::new, |txn| txn.proposals().to_vec())
                    }),
                },
            },
            completion,
        })
    }

    fn select(
        runtime: &mut ModelRuntime<B, Self>,
        decision: &TokenSamplingDecision<'_>,
        state: &mut State<B>,
    ) -> Result<SpecializedTextSubmission<B>, B::Error> {
        let txn = state
            .sampling
            .decoding
            .speculative
            .as_ref()
            .expect("active verification");
        let (backend, session) = runtime.parts_mut();
        let scores = session.shared_target_scores::<B, Shared<B>>(
            backend,
            txn,
            B::shared_context(backend, None),
        )?;
        let submission = session.retain_scores(backend, scores)?;
        let sampled = crate::text_session_driver::causal::sample_causal_scores::<B, N>(
            &session.resources().model,
            backend,
            Submission {
                output: N::publish_scores(submission.output, true),
                completion: submission.completion,
            },
            decision.filter(),
            &mut state.sampling.sampler,
            state.sampling.prng.as_mut(),
            state.sampling.temperature,
        )?;
        let frontier = shared_state::<B>(state);
        crate::text_session_driver::causal::advance_sample(frontier.prediction, frontier.capture)
            .map_err(eredu_core::BackendFailure::from_error)?;
        let token = B::shared_token_id(backend, &sampled.output)?;
        shared_state::<B>(state)
            .decoding
            .speculative
            .as_mut()
            .expect("active verification")
            .stage_target_token(token)
            .map_err(NeuralError::backend_source)?;
        Ok(Submission {
            output: TextOperationOutput::Token(sampled.output),
            completion: sampled.completion,
        })
    }

    fn coordinate_mode(
        runtime: &ModelRuntime<B, Self>,
        local: DecodingCoordination,
    ) -> Result<DecodingCoordination, TextOperationError<B::Error>> {
        if runtime.session().shared_policy().is_none() {
            return Ok(local);
        }
        runtime
            .session()
            .coordinate_decoding_control::<B>(local)
            .map_err(|error| match error {
                TextPreparationAgreementError::Decoding(error) => {
                    TextOperationPolicyError::DecodingMode(error).into()
                }
                error => TextOperationError::Backend(N::preparation_error(
                    eredu_core::BackendFailure::from_error(error),
                )),
            })
    }

    fn settle_mode(
        runtime: &mut ModelRuntime<B, Self>,
        pending: &mut Option<Input<B>>,
        state: &mut State<B>,
        requested: DecodingMode,
    ) -> Result<bool, TextOperationError<B::Error>> {
        let policy = runtime.session().shared_policy();
        let temperature = state.sampling.temperature;
        let (backend, session) = runtime.parts_mut();
        session.settle_mode(
            backend,
            pending,
            shared_state::<B>(state).decoding,
            requested,
            policy,
            temperature,
            B::shared_context(backend, None),
        )
    }

    fn commit(
        runtime: &mut ModelRuntime<B, Self>,
        token: u32,
        state: &mut State<B>,
    ) -> Result<(), B::Error> {
        let state = shared_state::<B>(state);
        let Some(mut txn) = state.decoding.speculative.take() else {
            return Ok(());
        };
        let (backend, session) = runtime.parts_mut();
        let retained = session.commit_shared_target::<B, Shared<B>>(
            backend,
            &mut txn,
            token,
            B::shared_context(backend, None),
        )?;
        state.decoding.committed_token_cached = retained;
        if retained {
            state.decoding.speculative = Some(txn);
        }
        Ok(())
    }
}

//! Portable ownership of resident block-refinement advancement.

use crate::text_session_driver::TextSessionResourcesFor;
use eredu_core::refinement::RefinementBackend;
use eredu_core::refinement::RefinementFinalizationBackend;

use eredu_core::backend::TextTransactionMechanisms;
use std::{ops::DerefMut, sync::Arc};

use eredu_core::{
    capture::{CaptureError, CapturePhase},
    refinement::{GenerationPhase, RefinementCanvas},
    BackendError, Completion, ModelRuntime, PendingTextInput, TextOperationOutput,
    TokenSamplingDecision,
};
use eredu_nn::Error as NeuralError;

use crate::SpeculativeSampler;
use crate::{
    capture::{partition::PartitionCaptureHookTransport, CaptureSession},
    generation::refinement::block::BlockRefinementState,
    speculative::{
        autoregressive::AutoregressiveMechanisms, shared_backbone::SharedBackboneMechanisms,
    },
    text_session_driver::{
        capture::TextCaptureMechanisms, causal::CausalTextMechanisms, TextExecutable,
        TextSessionDriver, TextSessionMechanisms,
    },
    NoopObserver,
};
use eredu_core::{
    input::CommittedTokenTensors,
    refinement::TextRefinementNative,
    sampling::{SamplingBackend, TokenDomain},
    BackendFailure,
};

use crate::text_session_driver::TextPrefillInput as _;

use super::{
    shared::SharedTextMechanisms, Input, SpecializedTextSubmission, State, TextModeBinding,
    TextModeState,
};

type Refinement<B> = <B as TextModeBinding>::Refinement;
type BlockToken<B> = <Refinement<B> as RefinementBackend>::Token;
type BlockLogits<B> = <Refinement<B> as RefinementBackend>::Logits;
type Native<B> = <B as eredu_core::backend::SharedTextNative>::Native;
type Executable<B> = <Native<B> as TextSessionMechanisms<B>>::Executable;
type ExecutableTypes<B> = crate::text_session_driver::SessionTextTypes<B, Native<B>>;
/// Request-owned state used to advance a retained refinement block.
struct BlockTextState<'a, B: TextModeBinding> {
    /// Canonical mode and active numerical state.
    pub decoding: &'a mut TextModeState<B::Refinement, B::Shared>,
    /// Sampler shared with the ordinary causal route.
    pub sampler: &'a mut crate::generation::TextSampler,
    /// The parameter version bound to this request.
    pub parameter_epoch: &'a mut Option<u64>,
    /// Absolute position of the next commitment.
    pub prediction: &'a mut u64,
    /// Seed for numerical refinement when a stream was not already retained.
    pub seed: u64,
    /// Default request temperature.
    pub temperature: f32,
    /// Whether tentative tokens should be emitted.
    pub draft_events: bool,
    /// Caller-supplied partial canvas.
    pub canvas: Option<Arc<RefinementCanvas>>,
    /// Selected readback for one refinement forward.
    readout_selection: &'a mut Option<Vec<eredu_core::refinement::RefinementScoreSelection>>,
    readout_rows: &'a mut Option<eredu_core::refinement::RefinementScoreRows>,
    /// Optional request observation.
    pub capture: Option<&'a mut CaptureSession>,
}

fn block_state<B: TextModeBinding>(state: &mut State<B>) -> BlockTextState<'_, B> {
    BlockTextState {
        decoding: &mut state.sampling.decoding,
        sampler: &mut state.sampling.sampler,
        parameter_epoch: &mut state.sampling.parameter_epoch,
        prediction: &mut state.sampling.next_prediction,
        seed: state.sampling.refinement_seed,
        temperature: state.sampling.temperature,
        draft_events: state.sampling.draft_events,
        canvas: state.sampling.canvas.clone(),
        readout_selection: &mut state.readout_selection,
        readout_rows: &mut state.readout_rows,
        capture: state.capture.as_mut(),
    }
}

pub(super) trait BlockTextDriver<B: super::TextModeBinding>:
    eredu_core::BackendSession<B> + Sized
{
    fn advance_block(
        runtime: &mut ModelRuntime<B, Self>,
        phase: GenerationPhase,
        input: Input<B>,
        decision: Option<&TokenSamplingDecision<'_>>,
        state: &mut State<B>,
    ) -> Result<SpecializedTextSubmission<B>, B::Error>;
}

fn block_error(message: &'static str) -> BackendError {
    BackendError::Execution {
        session: "block-refinement".into(),
        operation: "advance text".into(),
        message: message.into(),
    }
}

impl<B, R, N, I: Clone + 'static> BlockTextDriver<B> for TextSessionDriver<R, N>
where
    B: SharedTextMechanisms<
            Native = N,
            InputTensor = I,
            Refinement: RefinementBackend<
                Token = <Executable<B> as TextExecutable<ExecutableTypes<B>>>::ObservationTensor,
                Context = <Executable<B> as TextExecutable<ExecutableTypes<B>>>::Context,
                Error: std::error::Error + Send + Sync + 'static,
            > + RefinementFinalizationBackend<
                CanvasTokens = BlockToken<B>,
                CanvasScores = BlockLogits<B>,
                PositionScores = BlockLogits<B>,
                FinalizationContext = <Executable<B> as TextExecutable<ExecutableTypes<B>>>::Context,
                FinalizationError = <Refinement<B> as RefinementBackend>::Error,
            >,
        > + CommittedTokenTensors<TokenTensor = I, BlockToken = BlockToken<B>>
        + TextRefinementNative<
            BlockSampling: SamplingBackend<
                Token = BlockToken<B>,
                Logits = BlockLogits<B>,
                Context = <Refinement<B> as RefinementBackend>::Context,
                Error = <Refinement<B> as RefinementBackend>::Error,
            >,
            ScoreTensor = <<B as TextModeBinding>::Shared as AutoregressiveMechanisms>::Logits,
            InvocationCompletion = <N as eredu_core::backend::TextSessionNative<B>>::Completion,
            ConditioningFormat = <Refinement<B> as RefinementBackend>::ConditioningFormat,
        >,
    BlockLogits<B>:
        eredu_nn::Tensor<Context = <Executable<B> as TextExecutable<ExecutableTypes<B>>>::Context>,
    B::Error: From<BackendError>
        + From<BackendFailure>
        + From<NeuralError>
        + From<<B::Shared as AutoregressiveMechanisms>::Error>
        + From<<Refinement<B> as RefinementBackend>::Error>,
    N: CausalTextMechanisms<
            B,
            Resources = R,
            Prefill = crate::text_session_driver::TextPromptFor<B>,
            TextValues: eredu_core::backend::NativeTextValues<
                InputTensor = I,
                Scores = <<B as TextModeBinding>::Shared as AutoregressiveMechanisms>::Logits,
            >,
            Executable: TextExecutable<
                ExecutableTypes<B>,
                Scores = <B::Shared as AutoregressiveMechanisms>::Logits,
            >,
        > + TextCaptureMechanisms<
            N::Retained,
            Transport: PartitionCaptureHookTransport<
                Error: Send + Sync + 'static,
                Completion: Completion<Error: Send + Sync + 'static>,
            >,
        > + TextTransactionMechanisms<Error = B::Error>,
    R: DerefMut<Target = TextSessionResourcesFor<B, N>> + 'static,
{
    fn advance_block(
        runtime: &mut ModelRuntime<B, Self>,
        phase: GenerationPhase,
        input: Input<B>,
        decision: Option<&TokenSamplingDecision<'_>>,
        state: &mut State<B>,
    ) -> Result<SpecializedTextSubmission<B>, B::Error> {
        let geometry = runtime
            .session()
            .resources()
            .model
            .as_ref()
            .block_refinement_geometry()
            .ok_or_else(|| block_error("missing selected refinement geometry"))?;
        let shared_policy = runtime.session().shared_backbone_policy::<B>();
        let mut state = block_state::<B>(state);
        runtime
            .session()
            .validate_parameter_epoch(state.parameter_epoch)?;
        if state.decoding.refinement.is_none()
            && matches!(
                state.decoding.mode,
                eredu_core::decoding::DecodingMode::Diffusion { policy: None }
            )
        {
            if let Some(policy) = shared_policy {
                state.decoding.refinement_policy = policy
                    .diffusion_policy(state.temperature)
                    .map_err(NeuralError::backend_source)?;
            }
        }

        let (backend, session) = runtime.parts_mut();
        let context = N::context(backend);
        let submission = match phase {
            GenerationPhase::Context => {
                let cached = if state.decoding.refinement.is_none()
                    && matches!(input, PendingTextInput::Continue)
                {
                    state.decoding.next_logits.take()
                } else {
                    None
                };
                if let Some(scores) = cached {
                    session.retain_scores(backend, scores)?
                } else {
                    let prompt = match (&state.decoding.refinement, input) {
                        (None, PendingTextInput::Prefill(prompt)) => prompt,
                        (None, PendingTextInput::Decode(token)) => {
                            crate::text_session_driver::PreparedTextPrompt::from_committed_tokens(
                                backend.decode_token_tensor(token)?,
                            )?
                        }
                        (Some(block), _) => {
                            let tokens = block.authoritative_tokens();
                            if tokens.len() != geometry.canvas_length.get() as usize {
                                return Err(block_error(
                                    "next context requires a complete authoritative block",
                                )
                                .into());
                            }
                            crate::text_session_driver::PreparedTextPrompt::from_committed_tokens(
                                backend.concatenate_committed_tokens(tokens)?,
                            )?
                        }
                        _ => {
                            return Err(block_error(
                                "initial refinement context requires a prepared prompt",
                            )
                            .into())
                        }
                    };
                    let prepared;
                    let mut collector;
                    let mut noop = NoopObserver;
                    let observer = if let Some(capture) = state.capture.as_deref_mut() {
                        let invocation = session
                            .payload
                            .model
                            .as_ref()
                            .prepared_input_plans(prompt.borrowed())
                            .map_err(|error| CaptureError::Invalid(error.to_string()))
                            .and_then(|parts| {
                                parts.iter().try_fold(0u64, |total, part| {
                                    total
                                        .checked_add(part.decoder_positions())
                                        .ok_or(CaptureError::Overflow)
                                })
                            })
                            .and_then(|sequence| session.text_capture_shape(sequence))
                            .map(|shape| (CapturePhase::Prefill, shape));
                        prepared = session.prepare_capture_invocation(
                            capture,
                            *state.prediction,
                            invocation,
                        );
                        collector = prepared.open::<N::Collector>(
                            capture,
                            context,
                            None,
                            *state.prediction,
                        );
                        &mut *collector
                    } else {
                        &mut noop
                    };
                    session.advance_observed_prefill(backend, Ok(prompt), None, observer)?
                }
            }
            GenerationPhase::Refinement => {
                let refinement = state
                    .decoding
                    .refinement
                    .as_ref()
                    .and_then(|block| block.refinement())
                    .ok_or_else(|| block_error("missing tentative canvas"))?;
                let feedback_shape = geometry.feedback_shape();
                let (canvas, feedback) = B::block_canvas(
                    backend,
                    refinement.canvas(),
                    &geometry.canvas_shape(),
                    refinement
                        .self_conditioning()
                        .map(|scores| (scores, feedback_shape.as_slice())),
                )?;
                let prepared;
                let mut collector;
                let mut noop = NoopObserver;
                let observer = if let Some(capture) = state.capture.as_deref_mut() {
                    let invocation = session
                        .text_capture_shape(u64::from(geometry.canvas_length.get()))
                        .map(|shape| (CapturePhase::Decode, shape));
                    prepared =
                        session.prepare_capture_invocation(capture, *state.prediction, invocation);
                    collector =
                        prepared.open::<N::Collector>(capture, context, None, *state.prediction);
                    &mut *collector
                } else {
                    &mut noop
                };
                session.advance_refinement(backend, &canvas, feedback.as_ref(), observer)?
            }
            GenerationPhase::Finalization => {
                session.retain_scores(backend, B::shared_progress_value())?
            }
        };
        let scope = B::open_block_scope(&submission)?;
        let completed = (|| {
            if state.decoding.refinement.is_none() {
                let random = match state.decoding.refinement_random.take() {
                    Some(random) => random,
                    None => {
                        let root = B::Shared::model_randomness_root(state.seed)?;
                        if shared_policy.is_some() {
                            B::Shared::split_model_randomness(
                                &root,
                                1,
                                B::shared_context(backend, None),
                            )?
                        } else {
                            root
                        }
                    }
                };
                state.decoding.refinement = Some(BlockRefinementState::new(
                    state.decoding.refinement_policy,
                    geometry.canvas_length,
                    TokenDomain::new(geometry.vocabulary_size.get() as usize),
                    B::block_conditioning_format(&submission.output),
                    random,
                    state.canvas.clone(),
                ));
            }
            let block = state
                .decoding
                .refinement
                .as_mut()
                .expect("active refinement block");
            let output = match phase {
                GenerationPhase::Context => {
                    if decision.is_some() {
                        return Err(block_error(
                            "context progress cannot consume a commitment decision",
                        )
                        .into());
                    }
                    let scores = block
                        .needs_context_seed(shared_policy.is_some_and(|policy| policy.seed_first()))
                        .then(|| B::block_causal_scores(backend, &submission.output))
                        .transpose()?;
                    let progress = block
                        .accept_context(scores.as_ref(), context)
                        .map_err(NeuralError::backend_source)?;
                    TextOperationOutput::Progress {
                        pending: PendingTextInput::Continue,
                        progress,
                    }
                }
                GenerationPhase::Refinement => {
                    if decision.is_some() {
                        return Err(block_error(
                            "refinement progress cannot consume a commitment decision",
                        )
                        .into());
                    }
                    let scores = B::block_refinement_scores(
                        backend,
                        &submission.output,
                        &geometry.position_score_shape(),
                    )?;
                    if let Some(selections) = state.readout_selection.take() {
                        *state.readout_rows = Some(B::read_refinement_scores(
                            backend,
                            &submission.output,
                            selections,
                        )?);
                    }
                    let mut progress = block
                        .accept_refinement(&scores, context)
                        .map_err(NeuralError::backend_source)?;
                    if state.draft_events {
                        let draft = block
                            .refinement()
                            .and_then(|refinement| refinement.draft())
                            .ok_or_else(|| {
                                block_error("completed refinement has no tentative IDs")
                            })?;
                        progress.tentative_token_ids = Some(B::block_token_ids(backend, draft)?);
                    }
                    TextOperationOutput::Progress {
                        pending: PendingTextInput::Continue,
                        progress,
                    }
                }
                GenerationPhase::Finalization => {
                    let decision = decision.ok_or_else(|| {
                        block_error("finalization requires an evolving commitment decision")
                    })?;
                    let finalized = block
                        .commit_candidate(decision.filter(), context)
                        .map_err(NeuralError::backend_source)?;
                    if shared_policy.is_some() {
                        let token = B::BlockSampling::token_id(&finalized.token, context)?;
                        SpeculativeSampler::<B::BlockSampling>::commit_token(
                            state.sampler,
                            &finalized.scores,
                            token,
                            context,
                        )?;
                    }
                    *state.prediction = state
                        .prediction
                        .checked_add(1)
                        .ok_or_else(|| block_error("text prediction overflow"))?;
                    TextOperationOutput::Token(B::block_output_token(
                        backend,
                        finalized.token,
                        &submission,
                    ))
                }
            };
            let root = match &output {
                TextOperationOutput::Token(token) => B::block_token_root(token),
                _ => block
                    .refinement()
                    .map(|refinement| B::block_canvas_root(refinement.canvas()))
                    .unwrap_or_else(|| submission.output.clone()),
            };
            if let Some(capture) = &mut state.capture {
                capture
                    .set_commitment_boundary(*state.prediction)
                    .map_err(|error| {
                        BackendFailure::from_error(error)
                            .with_operation("publish refinement capture boundary")
                    })?;
            }
            Ok((output, root))
        })();
        B::finish_block_scope(scope, submission, completed)
    }
}

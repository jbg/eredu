//! Binds the portable block state to local text forwards and synchronous scores.

use super::*;
#[cfg(feature = "tensor-capture")]
use crate::text_session_driver::TextPrefillInput;
use crate::{
    generation::refinement::{
        architecture::{BlockRefinementGeometry, RefinementExecutionPolicy},
        block::BlockRefinementState,
        host::{add_retained_value, scalar, HostRefinement, ScoreMatrix},
    },
    text_session_driver::TextExecutable,
    NoopObserver,
};
use eredu_core::{
    decoding::DecodingMode,
    refinement::{
        GenerationPhase, PredictionScoreKind, RefinementPolicy, RefinementReadoutError,
        RefinementScoreRows, RefinementScoreSelection,
    },
    sampling::TokenDomain,
    PendingTextInput, TextOperationOutput, TextOperationPolicyError, TokenSamplingDecision,
};

#[derive(Clone)]
pub(super) struct LocalRefinementBlock {
    block: BlockRefinementState<HostRefinement>,
    declaration: RefinementExecutionPolicy,
    parameter_epoch: Option<u64>,
    feedback_element: Option<eredu_nn::TensorElementType>,
}

impl LocalRefinementBlock {
    pub(super) fn operation_kind(&self) -> TextOperationKind {
        match self.block.phase() {
            GenerationPhase::Finalization => {
                TextOperationKind::Decision(PredictionScoreKind::ConstrainedFinalization)
            }
            phase => TextOperationKind::Progress(phase),
        }
    }

    pub(super) fn estimate_snapshot(
        &self,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        let copied = self.block.snapshot_host_bytes()?;
        let mut bytes = Some(copied);
        self.block
            .visit_retained_values(|value| bytes = add_retained_value(bytes, value));
        Some(eredu_core::execution_control::SnapshotEstimate {
            retained_bytes: bytes?,
            copy_bytes: copied,
        })
    }

    pub(super) fn estimate_growth(&self) -> Option<u64> {
        let geometry = self.declaration.geometry;
        let canvas = u64::from(geometry.canvas_length.get());
        let width = u64::from(geometry.vocabulary_size.get());
        let history = u64::from(self.declaration.policy.stability_history_length());
        let history_row_bytes = canvas
            .checked_mul(4)?
            .checked_add(64 + size_of::<std::sync::Arc<[u32]>>() as u64)?;
        let matrices = match self.declaration.policy {
            RefinementPolicy::EntropyFeedback(_) => 2,
            RefinementPolicy::MaskedUnmasking { .. } => 1,
        };
        canvas
            .checked_mul(width)?
            .checked_mul(4 * matrices)?
            .checked_add(canvas.checked_mul(128)?)?
            .checked_add(history.checked_mul(history_row_bytes)?)?
            .checked_add(4096)
    }

    pub(super) fn continuation_context(&self, predictions: u64) -> Option<u64> {
        self.block.continuation_context_tokens(predictions)
    }

    pub(super) fn replace_random(&mut self, seed: u64) {
        self.block.replace_random(HostRandomState::with_stream(
            seed,
            self.declaration.randomness_substream().unwrap_or(0) as u64,
        ));
    }
}

#[derive(Default)]
pub(super) struct Readout {
    selections: Option<Vec<RefinementScoreSelection>>,
    rows: Option<RefinementScoreRows>,
}

impl Readout {
    pub(super) fn configure(
        &mut self,
        geometry: Option<BlockRefinementGeometry>,
        selections: Vec<RefinementScoreSelection>,
    ) -> Result<(), RefinementReadoutError> {
        let geometry = geometry.ok_or(RefinementReadoutError::Unsupported)?;
        geometry.validate_readout(&selections)?;
        self.rows = None;
        self.selections = Some(selections);
        Ok(())
    }

    pub(super) fn take(&mut self) -> Option<RefinementScoreRows> {
        self.selections = None;
        self.rows.take()
    }
}

impl<R: 'static, N> TextSessionDriver<R, N, CausalTextGeneration> {
    pub(super) fn local_refinement_policy<B>(&self) -> Option<RefinementExecutionPolicy>
    where
        B: TextGenerationBackend<Error = BackendFailure>,
        N: TextSessionMechanisms<B, Resources = R>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        if !self.native().supports_host_refinement() {
            return None;
        }
        let model = self.resources().model.as_ref();
        let declaration = model.capability_estimate().ordinary_refinement()?;
        (Some(declaration.geometry) == model.block_refinement_geometry()).then_some(declaration)
    }

    pub(super) fn start_local_refinement<B>(
        &self,
        config: &TextGenerationConfig,
    ) -> Result<Option<LocalRefinementBlock>, BackendFailure>
    where
        B: TextGenerationBackend<Error = BackendFailure>,
        N: TextSessionMechanisms<B, Resources = R>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        let Some(DecodingMode::Diffusion { policy }) = config.decoding_mode() else {
            return Ok(None);
        };
        let declaration = self.local_refinement_policy::<B>().ok_or_else(|| {
            BackendFailure::from_error(
                TextOperationPolicyError::RequiresBlockRefinementArchitecture,
            )
        })?;
        let policy = declaration
            .prepare(policy, config.sampling().temperature)
            .map_err(BackendFailure::from_error)?;
        let random = HostRandomState::with_stream(
            config.seed(),
            declaration.randomness_substream().unwrap_or(0) as u64,
        );
        Ok(Some(LocalRefinementBlock {
            declaration: RefinementExecutionPolicy {
                policy,
                ..declaration
            },
            parameter_epoch: None,
            feedback_element: None,
            block: BlockRefinementState::new(
                policy,
                declaration.geometry.canvas_length,
                TokenDomain::new(declaration.geometry.vocabulary_size.get() as usize),
                (),
                random,
                config.refinement_canvas().cloned(),
            ),
        }))
    }

    pub(super) fn advance_local_refinement<B>(
        runtime: &mut ModelRuntime<B, Self>,
        input: PendingTextInput<TextPromptFor<B>, <B as TextGenerationBackend>::Token>,
        decision: Option<&TokenSamplingDecision<'_>>,
        state: &mut CausalRequest<B>,
    ) -> Result<
        Submission<
            TextOperationOutput<TextPromptFor<B>, <B as TextGenerationBackend>::Token>,
            B::TextCompletion,
        >,
        BackendFailure,
    >
    where
        B: LocalCausalBackend,
        N: TextSessionMechanisms<B, Resources = R, Prefill = TextPromptFor<B>>
            + TextSamplingNative<
                B,
                Token = <B as TextGenerationBackend>::Token,
                SamplingCompletion = B::TextCompletion,
                ScoreRow = Vec<f32>,
                Sample = u32,
                TextValues: NativeTextValues<InputTensor = B::InputTensor>,
            > + eredu_core::backend::TextTransactionMechanisms<Error = BackendFailure>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        let local =
            state.sampling.refinement.as_mut().ok_or_else(|| {
                BackendFailure::from_error(TextOperationPolicyError::OutcomeMismatch)
            })?;
        let (backend, session) = runtime.parts_mut();
        let context = N::context(backend);
        session.validate_parameter_epoch(&mut local.parameter_epoch)?;
        let geometry = local.declaration.geometry;
        let block = &mut local.block;
        let prediction = state.sampling.next_prediction;
        #[cfg(not(feature = "tensor-capture"))]
        let _ = prediction;
        let progress = match block.phase() {
            GenerationPhase::Context => {
                let prompt = if block.block() == 0 {
                    let PendingTextInput::Prefill(prompt) = input else {
                        return Err(BackendFailure::from_error(
                            TextOperationPolicyError::OutcomeMismatch,
                        ));
                    };
                    prompt
                } else {
                    let tokens = block
                        .authoritative_tokens()
                        .iter()
                        .map(|token| scalar(token))
                        .collect::<Result<Vec<_>, _>>()?;
                    if tokens.len() != geometry.canvas_length.get() as usize {
                        return Err(BackendFailure::from_error(
                            TextOperationPolicyError::OutcomeMismatch,
                        ));
                    }
                    prepare_plain_text_prompt(backend, tokens)?
                };
                #[cfg(feature = "tensor-capture")]
                let submission = if let Some(capture) = state.capture.as_mut() {
                    let parts = session
                        .resources()
                        .model
                        .as_ref()
                        .prepared_input_plans(prompt.borrowed())
                        .map_err(BackendFailure::from_error)?;
                    let (sequence, media) =
                        crate::text_session_driver::capture::prepared_prompt_capture_shape(&parts)
                            .map_err(BackendFailure::from_error)?;
                    let mut shape = session
                        .text_capture_shape(sequence)
                        .map_err(BackendFailure::from_error)?;
                    shape.media = media;
                    let mut observer = Self::refinement_observer::<B>(
                        context,
                        capture,
                        prediction,
                        eredu_core::capture::CapturePhase::Prefill,
                        shape,
                    )?;
                    session.advance_observed_prefill(backend, Ok(prompt), None, &mut observer)?
                } else {
                    session.advance_observed_prefill(
                        backend,
                        Ok(prompt),
                        None,
                        &mut NoopObserver,
                    )?
                };
                #[cfg(not(feature = "tensor-capture"))]
                let submission = session.advance_observed_prefill(
                    backend,
                    Ok(prompt),
                    None,
                    &mut NoopObserver,
                )?;
                let submission = submission.wait()?;
                if matches!(
                    local.declaration.policy,
                    RefinementPolicy::EntropyFeedback(_)
                ) {
                    local.feedback_element =
                        Some(N::host_refinement_format(&submission).ok_or_else(|| {
                            BackendFailure::from_error(TextOperationPolicyError::OutcomeMismatch)
                        })?);
                }
                let scores = if block.needs_context_seed(local.declaration.seed_first) {
                    let published = N::publish_scores(submission, true);
                    let row = N::last_score_row(&published, context)?.ok_or_else(|| {
                        BackendFailure::from_error(TextOperationPolicyError::OutcomeMismatch)
                    })?;
                    Some(ScoreMatrix::new(
                        row,
                        [1, geometry.vocabulary_size.get() as usize],
                    )?)
                } else {
                    None
                };
                let progress = block
                    .accept_context(scores.as_ref(), &())
                    .map_err(BackendFailure::from_error)?;
                progress
            }
            GenerationPhase::Refinement => {
                let current = block.refinement().ok_or_else(|| {
                    BackendFailure::from_error(TextOperationPolicyError::OutcomeMismatch)
                })?;
                let canvas = N::host_refinement_canvas(current.canvas(), context)?;
                let feedback = current
                    .self_conditioning()
                    .map(|scores| {
                        let element = local.feedback_element.ok_or_else(|| {
                            BackendFailure::from_error(TextOperationPolicyError::OutcomeMismatch)
                        })?;
                        N::host_refinement_feedback(
                            scores.values(),
                            &geometry.feedback_shape(),
                            element,
                            context,
                        )
                    })
                    .transpose()?;
                #[cfg(feature = "tensor-capture")]
                let submission = if let Some(capture) = state.capture.as_mut() {
                    let shape = session
                        .text_capture_shape(u64::from(geometry.canvas_length.get()))
                        .map_err(BackendFailure::from_error)?;
                    let mut observer = Self::refinement_observer::<B>(
                        context,
                        capture,
                        prediction,
                        eredu_core::capture::CapturePhase::Decode,
                        shape,
                    )?;
                    session.advance_refinement(
                        backend,
                        &canvas,
                        feedback.as_ref(),
                        &mut observer,
                    )?
                } else {
                    session.advance_refinement(
                        backend,
                        &canvas,
                        feedback.as_ref(),
                        &mut NoopObserver,
                    )?
                };
                #[cfg(not(feature = "tensor-capture"))]
                let submission = session.advance_refinement(
                    backend,
                    &canvas,
                    feedback.as_ref(),
                    &mut NoopObserver,
                )?;
                let raw = submission.wait()?;
                let shape = [
                    geometry.canvas_length.get() as usize,
                    geometry.vocabulary_size.get() as usize,
                ];
                let scores =
                    ScoreMatrix::new(N::host_refinement_scores(&raw, shape, context)?, shape)?;
                if let Some(selections) = state.readout.selections.take() {
                    state.readout.rows = Some(scores.read_selected(selections)?);
                }
                let mut progress = block
                    .accept_refinement(&scores, &())
                    .map_err(BackendFailure::from_error)?;
                if state.sampling.refinement_drafts {
                    progress.tentative_token_ids = block
                        .refinement()
                        .and_then(|current| current.draft())
                        .map(|tokens| tokens.to_vec());
                }
                progress
            }
            GenerationPhase::Finalization => {
                let decision = decision.ok_or_else(|| {
                    BackendFailure::from_error(TextOperationPolicyError::MissingDecision)
                })?;
                let lease = session.acquire_submission()?;
                let finalized = block
                    .commit_candidate(decision.filter(), &())
                    .map_err(BackendFailure::from_error)?;
                let result = N::complete_host_token(scalar(&finalized.token)?, lease)?;
                return Ok(Submission {
                    output: TextOperationOutput::Token(result.output),
                    completion: result.completion,
                });
            }
        };
        Ok(Submission {
            output: TextOperationOutput::Progress {
                pending: PendingTextInput::Continue,
                progress,
            },
            completion: N::complete_host_progress(session.acquire_submission()?)?,
        })
    }

    #[cfg(feature = "tensor-capture")]
    fn refinement_observer<'a, B>(
        context: &'a <N::TextValues as NativeTextValues>::Context,
        capture: &'a mut crate::capture::CaptureSession,
        prediction: u64,
        phase: eredu_core::capture::CapturePhase,
        shape: eredu_core::capture::CaptureInvocationShape,
    ) -> Result<
        impl crate::ActivationObserver<<N::TextValues as NativeTextValues>::Tensor, BackendFailure> + 'a,
        BackendFailure,
    >
    where
        B: TextGenerationBackend<Error = BackendFailure> + 'a,
        N: TextSessionMechanisms<B, Resources = R> + 'a,
    {
        let collector = N::local_capture(context, None).ok_or_else(|| {
            BackendFailure::from_error(eredu_core::capture::CaptureError::Unsupported(
                "no local refinement collector".into(),
            ))
        })?;
        super::capture::prepare_local_invocation(capture, prediction, phase, shape)?;
        Ok(crate::text_session_driver::capture::local_text_capture(
            capture, collector, prediction,
        ))
    }
}

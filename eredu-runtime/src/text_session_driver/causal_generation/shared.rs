//! Binds host sampling to the shared-backbone verification transaction.

use super::*;
use crate::{
    capture::SpeculativeContextFor,
    speculative::{
        autoregressive::local_causal::LocalCausalMechanisms,
        shared_backbone::{SharedBackboneExecutionPolicy, SharedBackboneTransaction},
    },
};
use eredu_core::{
    decoding::{DecodingMode, DecodingModeError, SpeculationOptions},
    refinement::{GenerationPhase, PredictionScoreKind},
    sampling::host::HostScores,
    PendingTextInput, TextOperationOutput, TextOperationPolicyError, TextProgress, TokenOutput,
    TokenSamplingDecision,
};

type Shared<B> = LocalCausalMechanisms<B>;
type Context<'a, B> = SpeculativeContextFor<
    'a,
    <B as eredu_core::backend::local_text::HostCausalVerification>::ContextBinding,
>;

pub(super) struct LocalSharedState<B: LocalCausalBackend> {
    options: SpeculationOptions,
    randomness: HostRandomState,
    transaction: Option<Box<SharedBackboneTransaction<Shared<B>>>>,
    tail_scores: Option<HostScores>,
}

impl<B: LocalCausalBackend> Clone for LocalSharedState<B> {
    fn clone(&self) -> Self {
        Self {
            options: self.options,
            randomness: self.randomness.clone(),
            transaction: self.transaction.clone(),
            tail_scores: self.tail_scores.clone(),
        }
    }
}

impl<B: LocalCausalBackend> LocalSharedState<B> {
    pub(super) fn operation_kind(
        &self,
        input: &PendingTextInput<TextPromptFor<B>, <B as TextGenerationBackend>::Token>,
    ) -> TextOperationKind {
        match input {
            PendingTextInput::Prefill(_) => TextOperationKind::CausalDecision,
            _ if self.transaction.is_some() || self.tail_scores.is_some() => {
                TextOperationKind::Decision(PredictionScoreKind::CausalConditional)
            }
            _ => TextOperationKind::Progress(GenerationPhase::Refinement),
        }
    }

    pub(super) fn retained_bytes(&self) -> Option<u64> {
        let sampling = size_of::<Self>() as u64;
        let scores = self
            .tail_scores
            .as_ref()
            .map_or(Some(0), HostScores::retained_bytes)?;
        let verification = self
            .transaction
            .as_ref()
            .map_or(Some(0), |txn| txn.retained_bytes())?;
        sampling.checked_add(scores)?.checked_add(verification)
    }

    pub(super) fn proposal_capacity(&self) -> u64 {
        self.options.request().max_draft_tokens.get() as u64
    }

    pub(super) fn reseed(&mut self, seed: u64) {
        self.randomness = HostRandomState::with_stream(seed, 2);
    }
}

impl<R: 'static, N> TextSessionDriver<R, N, CausalTextGeneration> {
    pub(super) fn local_shared_policy<B>(&self) -> Option<SharedBackboneExecutionPolicy>
    where
        B: TextGenerationBackend<Error = BackendFailure>,
        N: TextSessionMechanisms<B, Resources = R>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        self.native()
            .supports_host_refinement()
            .then(|| {
                self.resources()
                    .model
                    .as_ref()
                    .capability_estimate()
                    .shared_backbone_draft()
            })
            .flatten()
    }

    pub(super) fn start_local_shared<B>(
        &self,
        config: &TextGenerationConfig,
    ) -> Result<Option<LocalSharedState<B>>, BackendFailure>
    where
        B: LocalCausalBackend,
        N: TextSessionMechanisms<B, Resources = R>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        let Some(DecodingMode::SelfSpeculation(options)) = config.decoding_mode() else {
            return Ok(None);
        };
        let policy = self.local_shared_policy::<B>().ok_or_else(unavailable)?;
        policy
            .prepare(options, config.sampling().temperature)
            .map_err(BackendFailure::from_error)?;
        Ok(Some(LocalSharedState {
            options,
            randomness: HostRandomState::with_stream(config.seed(), 2),
            transaction: None,
            tail_scores: None,
        }))
    }
}

impl<B: LocalCausalBackend> LocalCausalSession<B> {
    pub(super) fn advance_local_shared(
        runtime: &mut ModelRuntime<B, Self>,
        input: PendingTextInput<TextPromptFor<B>, <B as TextGenerationBackend>::Token>,
        decision: Option<&TokenSamplingDecision<'_>>,
        state: &mut CausalRequest<B>,
        remaining: Option<usize>,
    ) -> Result<
        Submission<
            TextOperationOutput<TextPromptFor<B>, <B as TextGenerationBackend>::Token>,
            B::TextCompletion,
        >,
        BackendFailure,
    > {
        let (backend, session) = runtime.parts_mut();
        let shared = state.sampling.shared.as_mut().ok_or_else(outcome)?;
        let queue = B::context(backend);
        if let Some(decision) = decision {
            let logits = match &mut shared.transaction {
                Some(txn) => session.shared_target_scores::<B, Shared<B>>(
                    backend,
                    txn,
                    Context::<B>::single(queue),
                )?,
                None => shared.tail_scores.take().ok_or_else(outcome)?,
            };
            let mut random = state.sampling.random.clone();
            let sample = HostSample::select(
                &state.sampling.sampler,
                logits.values().to_vec(),
                decision.filter(),
                state.sampling.temperature,
                random.as_mut(),
            )?;
            let token = sample.token;
            if let Some(txn) = &mut shared.transaction {
                txn.stage_target_token(token)
                    .map_err(BackendFailure::from_error)?;
            }
            let result = B::complete_host_token(token, session.acquire_submission()?)?;
            state.pending = Some(PendingSample { sample, random });
            return Ok(Submission {
                output: TextOperationOutput::Token(result.output),
                completion: result.completion,
            });
        }

        let PendingTextInput::Decode(last) = input else {
            return Err(outcome());
        };
        let last = last.token_id()?;
        let request = shared.options.request();
        let capacity = request
            .max_draft_tokens
            .get()
            .min(remaining.map_or(usize::MAX, |n| n.saturating_sub(1)));
        if let Some(capacity) = std::num::NonZeroUsize::new(capacity) {
            let policy = session.local_shared_policy::<B>().ok_or_else(unavailable)?;
            let options = policy
                .prepare(shared.options, state.sampling.temperature)
                .map_err(BackendFailure::from_error)?;
            let wait = request
                .scheduler
                .completion_wait()
                .map_err(BackendFailure::from_error)?;
            let context = Context::<B>::single(queue);
            #[cfg(feature = "tensor-capture")]
            let mut observer;
            #[cfg(feature = "tensor-capture")]
            let mapped;
            #[cfg(feature = "tensor-capture")]
            let context = if let Some(capture) = state.capture.as_mut() {
                let collector = B::local_capture(queue, None).ok_or_else(|| {
                    BackendFailure::from_error(eredu_core::capture::CaptureError::Unsupported(
                        "no local verification collector".into(),
                    ))
                })?;
                let shape = session
                    .text_capture_shape((capacity.get() + 1) as u64)
                    .map_err(BackendFailure::from_error)?;
                super::capture::prepare_local_invocation(
                    capture,
                    state.sampling.next_prediction,
                    eredu_core::capture::CapturePhase::Decode,
                    shape,
                )?;
                observer = crate::text_session_driver::capture::local_text_capture(
                    capture,
                    collector,
                    state.sampling.next_prediction,
                );
                mapped = crate::speculative::autoregressive::MappedSpeculativeTargetObserver::<
                    <B::Values as NativeTextValues>::ObserverMapping,
                >::new(&mut observer);
                context.with_target_observer(&mapped)
            } else {
                context
            };
            let transaction = session.propose_shared::<B, Shared<B>>(
                backend,
                capacity,
                last,
                options,
                shared.randomness.clone(),
                wait,
                context,
            )?;
            shared.randomness = transaction.proposal_randomness().clone();
            shared.transaction = Some(Box::new(transaction));
        } else {
            let input = TextDecodeInput::TokenId(last);
            #[cfg(feature = "tensor-capture")]
            let submission = if let Some(capture) = state.capture.as_mut() {
                session
                    .submit_local_capture_decode(
                        backend,
                        input,
                        None,
                        state.sampling.next_prediction,
                        capture,
                    )
                    .map_err(|error| match error {
                        TextOperationError::Backend(error) => error,
                        TextOperationError::Policy(error) => BackendFailure::from_error(error),
                    })?
            } else {
                session.advance_decode(backend, input)?
            };
            #[cfg(not(feature = "tensor-capture"))]
            let submission = session.advance_decode(backend, input)?;
            let output = submission.wait()?;
            let row = B::last_score_row(&output, queue)?.ok_or_else(outcome)?;
            shared.tail_scores = Some(HostScores::from_row(row)?);
        }
        Ok(Submission {
            output: TextOperationOutput::Progress {
                pending: PendingTextInput::Continue,
                progress: TextProgress {
                    phase: GenerationPhase::Refinement,
                    block: state.sampling.next_prediction,
                    iteration: 1,
                    tentative_token_ids: state.sampling.refinement_drafts.then(|| {
                        shared
                            .transaction
                            .as_ref()
                            .map_or_else(Vec::new, |txn| txn.proposals().to_vec())
                    }),
                },
            },
            completion: B::complete_host_progress(session.acquire_submission()?)?,
        })
    }

    pub(super) fn commit_local_shared(
        runtime: &mut ModelRuntime<B, Self>,
        shared: &mut LocalSharedState<B>,
        token: u32,
    ) -> Result<(), BackendFailure> {
        if let Some(mut transaction) = shared.transaction.take() {
            let (backend, session) = runtime.parts_mut();
            if session.commit_shared_target::<B, Shared<B>>(
                backend,
                &mut transaction,
                token,
                Context::<B>::single(B::context(backend)),
            )? {
                shared.transaction = Some(transaction);
            }
        }
        Ok(())
    }
}

fn unavailable() -> BackendFailure {
    BackendFailure::from_error(DecodingModeError::Unsupported(
        "shared-backbone drafting is unavailable for the selected execution",
    ))
}

fn outcome() -> BackendFailure {
    BackendFailure::from_error(TextOperationPolicyError::OutcomeMismatch)
}

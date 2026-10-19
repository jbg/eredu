//! Shared-executor transactions inside the ordinary physical generation driver.
use super::*;
use crate::composition::mlx::speculative::{
    autoregressive::MlxAutoregressiveMechanisms, SpeculativeExecutionStreams,
};
use eredu_core::{PendingTextInput, TextOperationOutput, TextProgress};
use eredu_runtime::speculative::{
    autoregressive::{AutoregressiveMechanisms, CausalExecutor},
    shared_backbone::{SharedBackboneExecutor, SharedBackboneTransaction},
};

type Executor<'a> = SharedBackboneExecutor<'a, MlxAutoregressiveMechanisms>;

pub(in crate::composition::mlx::session) fn retained_transaction_bytes(
    transaction: &SharedBackboneTransaction<MlxAutoregressiveMechanisms>,
) -> Option<u64> {
    MlxAutoregressiveMechanisms::checkpoint_retained_bytes(transaction.baseline())?
        .checked_add(MlxAutoregressiveMechanisms::checkpoint_retained_bytes(
            transaction.verified(),
        )?)?
        .checked_add(transaction.verification().output().retained_bytes()?)?
        .checked_add(transaction.host_retained_bytes()?)?
        .checked_add(transaction.proposal_randomness().as_array().nbytes() as u64)?
        .checked_add(4096)
}

impl MlxBackend<'_> {
    pub(super) fn append_shared_committed_suffix(
        runtime: &mut ModelRuntime<Self>,
        state: &MlxTextGenerationState,
        tokens: &[u32],
    ) -> Result<Array, Error> {
        let stream = runtime.backend().stream().clone();
        let wait = match state.sampling.mode {
            eredu_core::decoding::DecodingMode::SelfSpeculation(options) => {
                options.request().scheduler
            }
            _ => eredu_core::generation::SpeculativeSchedulerOptions::default(),
        }
        .completion_wait()
        .map_err(|error| Error::Exception(Exception::from_source(error)))?;
        runtime.session_mut().with_model_operation(|model| {
            let context = SpeculativeExecutionStreams::single(&stream);
            let mut executor = CausalExecutor::<
                MlxAutoregressiveMechanisms,
                eredu_runtime::speculative::autoregressive::NoTargetRefinement,
            >::new(
                model.erased_mut(),
                eredu_runtime::speculative::autoregressive::NoTargetRefinement,
            );
            let mut cache = executor.new_cache(context)?;
            let settled = eredu_core::speculative::settle_committed_suffix(
                &mut executor,
                &mut cache,
                &eredu_runtime::speculative::autoregressive::CausalSeed::new(None, ()),
                tokens,
                wait,
                |_| Ok(()),
                context,
            )
            .map_err(|error| Error::Exception(Exception::from_source(error)))?;
            drop(executor);
            cache.publish(model.erased_mut())?;
            Ok(settled.logits)
        })
    }

    pub(super) fn submit_shared_speculation(
        runtime: &mut ModelRuntime<Self>,
        input: PendingTextInput<MlxModelInput, MlxTextToken>,
        decision: Option<&eredu_core::TokenSamplingDecision<'_>>,
        state: &mut MlxTextGenerationState,
        remaining: Option<usize>,
    ) -> Result<
        Submission<TextOperationOutput<MlxModelInput, MlxTextToken>, MlxTextCompletion>,
        Error,
    > {
        let stream = runtime.backend().stream().clone();
        if state.sampling.speculative.is_none() {
            let PendingTextInput::Decode(last) = input else {
                return Err(Error::ArchitectureModel(
                    "shared proposal needs a pending committed token".into(),
                ));
            };
            let last = last.value.clone().try_item::<u32>(&stream)?;
            let eredu_core::decoding::DecodingMode::SelfSpeculation(options) = state.sampling.mode
            else {
                unreachable!()
            };
            let policy = runtime
                .session()
                .payload
                .model
                .erased()
                .capability_estimate()
                .shared_backbone_draft()
                .ok_or_else(|| {
                    Error::ArchitectureModel("shared draft source was not retained".into())
                })?;
            let draft_options = policy
                .prepare(options, state.sampling.temperature)
                .map_err(|error| Error::Exception(Exception::from_source(error)))?;
            let capacity = options
                .request()
                .max_draft_tokens
                .get()
                .min(remaining.map_or(usize::MAX, |n| n.saturating_sub(1)));
            if capacity == 0 {
                let (backend, session) = runtime.parts_mut();
                let tokens = Array::from_slice(&[last], &[1, 1]);
                let submission = session.submit_decode_with_observer(
                    backend,
                    || Ok(tokens),
                    &mut eredu_runtime::NoopObserver,
                )?;
                submission.completion.wait()?;
                state.sampling.next_logits = Some(submission.output.clone());
                state.sampling.committed_token_cached = true;
                return progress_submission(session, backend, submission.output, state);
            }
            let capacity = std::num::NonZeroUsize::new(capacity).unwrap();
            let random = state.sampling.proposal_random.take().map_or_else(
                || {
                    crate::backend::random::split_key_at(
                        &safemlx::random::key(state.sampling.refinement_seed)?,
                        2,
                        &stream,
                    )
                    .map(RandomState::from_key)
                },
                Ok,
            )?;
            let prediction = state.sampling.next_prediction;
            let session = runtime.session_mut();
            let invocation = session
                .refinement_capture_shape((capacity.get() + 1) as u64)
                .map(|shape| (eredu_core::capture::CapturePhase::Decode, shape));
            let execute = |session: &mut MlxModelSession,
                           observer: &mut dyn RuntimeActivationObserver<MlxTensor, Error>|
             -> Result<
                SharedBackboneTransaction<MlxAutoregressiveMechanisms>,
                Error,
            > {
                let mut array_observer = ArrayObserverAdapter {
                    inner: observer,
                    routed_path: None,
                    routed_invocation_active: false,
                };
                let observer_ref: &mut dyn eredu_runtime::ActivationObserver<Array, Error> =
                    &mut array_observer;
                let observer_cell = crate::composition::mlx::speculative::BorrowedTargetObserver(
                    RefCell::new(observer_ref),
                );
                let transaction = session.with_model_operation(|model| {
                    let context = SpeculativeExecutionStreams::single(&stream)
                        .with_target_observer(&observer_cell);
                    let mut executor = Executor::new(model.erased_mut(), capacity);
                    let mut cache = executor.new_cache(context)?;
                    let completion_wait = options
                        .request()
                        .scheduler
                        .completion_wait()
                        .map_err(|error| Error::Exception(Exception::from_source(error)))?;
                    SharedBackboneTransaction::submit(
                        &mut executor,
                        &mut cache,
                        last,
                        draft_options,
                        random,
                        completion_wait,
                        context,
                    )
                    .map_err(|error| Error::Exception(Exception::from_source(error)))
                })?;
                observer.finish()?;
                Ok(transaction)
            };
            let transaction = match &mut state.capture {
                Some(capture) => partition_capture::with_invocation_observer(
                    session, capture, &stream, prediction, invocation, execute,
                )?,
                None => execute(session, &mut eredu_runtime::NoopObserver)?,
            };
            state.sampling.proposal_random = Some(transaction.proposal_randomness().clone());
            state.sampling.speculative = Some(Box::new(transaction));
            let (backend, session) = runtime.parts_mut();
            return progress_submission(session, backend, Array::from_f32(0.0), state);
        }
        let decision = decision.ok_or_else(|| {
            Error::ArchitectureModel("verification needs an evolving target decision".into())
        })?;
        let transaction = state
            .sampling
            .speculative
            .as_ref()
            .expect("active transaction");
        let scores = runtime.session_mut().with_model_operation(|model| {
            let executor = Executor::new(model.erased_mut(), transaction.capacity());
            transaction
                .logits(&executor, SpeculativeExecutionStreams::single(&stream))
                .map_err(|error| Error::Exception(Exception::from_source(error)))
        })?;
        let (backend, session) = runtime.parts_mut();
        let submission = session.submit_settled_logits(backend, scores)?;
        let submission = Submission {
            output: MlxModelOutput::new(Some(MlxTensor::from_array(submission.output))),
            completion: submission.completion,
        };
        let sampled = sample_text_submission(
            session,
            submission,
            decision.filter(),
            state,
            stream.clone(),
        )?;
        let token = sampled.output.value.clone().try_item::<u32>(&stream)?;
        state
            .sampling
            .speculative
            .as_mut()
            .expect("active transaction")
            .stage_target_token(token)
            .map_err(|error| Error::Exception(Exception::from_source(error)))?;
        Ok(Submission {
            output: TextOperationOutput::Token(sampled.output),
            completion: sampled.completion,
        })
    }

    pub(super) fn commit_shared_token(
        runtime: &mut ModelRuntime<Self>,
        token: u32,
        state: &mut MlxTextGenerationState,
    ) -> Result<(), Error> {
        let Some(mut transaction) = state.sampling.speculative.take() else {
            return Ok(());
        };
        let stream = runtime.backend().stream().clone();
        let matching = runtime.session_mut().with_model_operation(|model| {
            let context = SpeculativeExecutionStreams::single(&stream);
            let mut executor = Executor::new(model.erased_mut(), transaction.capacity());
            let (matching, mut cache) = transaction
                .commit_target_token(&mut executor, token, context)
                .map_err(|error| Error::Exception(Exception::from_source(error)))?;
            drop(executor);
            cache.publish(model.erased_mut())?;
            Ok(matching)
        })?;
        state.sampling.committed_token_cached = matching;
        if matching {
            state.sampling.speculative = Some(transaction);
        }
        Ok(())
    }
}

fn progress_submission(
    session: &mut MlxModelSession,
    backend: &MlxBackend<'_>,
    root: Array,
    state: &MlxTextGenerationState,
) -> Result<Submission<TextOperationOutput<MlxModelInput, MlxTextToken>, MlxTextCompletion>, Error>
{
    let submission = session.submit_settled_logits(backend, root)?;
    let completion = MlxCompletion::submission(submission.output)?;
    Ok(Submission {
        output: TextOperationOutput::Progress {
            pending: PendingTextInput::Continue,
            progress: TextProgress {
                phase: eredu_core::refinement::GenerationPhase::Refinement,
                block: state.sampling.next_prediction,
                iteration: 1,
                tentative_token_ids: state.sampling.draft_events.then(|| {
                    state
                        .sampling
                        .speculative
                        .as_ref()
                        .map_or_else(Vec::new, |transaction| transaction.proposals().to_vec())
                }),
            },
        },
        completion: MlxTextCompletion {
            model: submission.completion,
            token: completion.completion,
            recovery: RefCell::new(None),
        },
    })
}

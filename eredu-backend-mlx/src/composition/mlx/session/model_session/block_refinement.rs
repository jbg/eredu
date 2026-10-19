//! Native realization of the portable alternating block state machine.

use super::*;
use eredu_core::capture::{CaptureError, CaptureInvocationShape, CapturePhase};
use eredu_core::refinement::GenerationPhase;
use eredu_core::{PendingTextInput, TextOperationOutput, TextProgress};
use eredu_runtime::{
    generation::refinement::{
        block::BlockRefinementState, finalization::finalize_refinement_position,
    },
    TokenDomain,
};

fn causal_next_logits(scores: Array, stream: &Stream) -> Result<Array, Error> {
    let shape = scores.shape();
    match shape {
        [1, sequence, vocabulary] if *sequence > 0 && *vocabulary > 0 => {
            Ok(scores.try_index_device((.., -1, ..), stream)?)
        }
        [1, vocabulary] if *vocabulary > 0 => Ok(scores),
        _ => Err(Error::ArchitectureModel(
            "causal context has invalid next-token score geometry".into(),
        )),
    }
}

impl MlxModelSession {
    pub(super) fn refinement_capture_shape(
        &self,
        sequence: u64,
    ) -> Result<CaptureInvocationShape, CaptureError> {
        let frontier = self
            .payload
            .model
            .erased()
            .forecast_state_offset()
            .map_err(|error| CaptureError::Invalid(error.to_string()))?
            .ok_or_else(|| {
                CaptureError::Unsupported("execution has no exact context frontier".into())
            })?;
        let frontier = u64::try_from(frontier)
            .map_err(|_| CaptureError::Invalid("negative context frontier".into()))?;
        Ok(CaptureInvocationShape {
            media: None,
            batch: 1,
            sequence,
            context: Some(
                frontier
                    .checked_add(sequence)
                    .ok_or(CaptureError::Overflow)?,
            ),
        })
    }
    fn submit_refinement_with_observer(
        &mut self,
        backend: &MlxBackend<'_>,
        canvas: &Array,
        feedback: Option<&Array>,
        observer: &mut dyn RuntimeActivationObserver<MlxTensor, Error>,
    ) -> Result<Submission<Array, MlxSessionCompletion>, Error> {
        let mut operation = self.begin_submission(backend)?;
        let validation = TokenValidationScope::begin()?;
        let result = operation
            .model()
            .erased_mut()
            .attention_forward_with_observer(
                canvas,
                feedback,
                eredu_core::decoding::AttentionInvocation::DIFFUSION,
                None,
                false,
                backend.stream(),
                &mut ArrayObserverAdapter {
                    inner: observer,
                    routed_path: None,
                    routed_invocation_active: false,
                },
            )
            .and_then(|output| {
                observer.finish()?;
                Ok(output)
            });
        let (output, owner, recovery) = operation.finish_execution(result)?;
        Ok(model_array_submission(
            output,
            validation.finish(),
            owner,
            recovery,
        ))
    }

    pub(super) fn submit_settled_logits(
        &mut self,
        backend: &MlxBackend<'_>,
        scores: Array,
    ) -> Result<Submission<Array, MlxSessionCompletion>, Error> {
        let operation = self.begin_submission(backend)?;
        let validation = TokenValidationScope::begin()?;
        let (output, owner, recovery) = operation.finish(Ok(scores))?;
        Ok(model_array_submission(
            output,
            validation.finish(),
            owner,
            recovery,
        ))
    }

    fn submit_finalization_boundary(
        &mut self,
        backend: &MlxBackend<'_>,
    ) -> Result<Submission<Array, MlxSessionCompletion>, Error> {
        let operation = self.begin_submission(backend)?;
        let validation = TokenValidationScope::begin()?;
        let (output, owner, recovery) = operation.finish(Ok(Array::from_f32(0.0)))?;
        Ok(model_array_submission(
            output,
            validation.finish(),
            owner,
            recovery,
        ))
    }
}

impl MlxBackend<'_> {
    pub(super) fn settle_shared_decoding_mode(
        runtime: &mut ModelRuntime<Self>,
        input: &mut Option<PendingTextInput<MlxModelInput, MlxTextToken>>,
        state: &mut MlxTextGenerationState,
        previous: eredu_core::decoding::DecodingMode,
        requested: eredu_core::decoding::DecodingMode,
    ) -> Result<bool, Error> {
        if state.sampling.speculative.is_some() {
            return Ok(false);
        }
        if matches!(
            previous,
            eredu_core::decoding::DecodingMode::SelfSpeculation(_)
        ) {
            if let Some(PendingTextInput::Decode(token)) = input.as_ref() {
                let stream = runtime.backend().stream().clone();
                let token = token.value.clone().try_item::<u32>(&stream)?;
                state.sampling.next_logits = Some(Self::append_shared_committed_suffix(
                    runtime,
                    state,
                    &[token],
                )?);
                state.sampling.committed_token_cached = true;
                *input = Some(PendingTextInput::Continue);
            }
        }
        if matches!(
            previous,
            eredu_core::decoding::DecodingMode::Diffusion { .. }
        ) {
            if let Some(block) = &state.sampling.refinement {
                if block.phase() != GenerationPhase::Context {
                    return Ok(false);
                }
                let arrays = block
                    .authoritative_tokens()
                    .iter()
                    .map(|token| token.as_array())
                    .collect::<Vec<_>>();
                if !arrays.is_empty() {
                    let stream = runtime.backend().stream().clone();
                    let tokens = safemlx::ops::concatenate(&arrays, &stream)?
                        .as_dtype(Dtype::Uint32, &stream)?
                        .into_evaluated()?
                        .try_to_vec::<u32>()
                        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                    state.sampling.next_logits = Some(Self::append_shared_committed_suffix(
                        runtime, state, &tokens,
                    )?);
                    *input = Some(PendingTextInput::Continue);
                }
            }
            state.sampling.refinement_random = state
                .sampling
                .refinement
                .take()
                .map(BlockRefinementState::into_random)
                .or_else(|| state.sampling.refinement_random.take());
        }
        state.sampling.mode = requested;
        if let eredu_core::decoding::DecodingMode::Diffusion { policy } = requested {
            state.sampling.refinement_policy = match policy {
                Some(policy) => policy,
                None => runtime
                    .session()
                    .payload
                    .model
                    .erased()
                    .capability_estimate()
                    .shared_backbone_draft()
                    .ok_or_else(|| {
                        Error::ArchitectureModel("no retained masked generation policy".into())
                    })?
                    .diffusion_policy(state.sampling.temperature)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
            };
        }
        Ok(true)
    }

    pub(super) fn submit_block_refinement_operation(
        runtime: &mut ModelRuntime<Self>,
        input: PendingTextInput<MlxModelInput, MlxTextToken>,
        decision: Option<&eredu_core::TokenSamplingDecision<'_>>,
        state: &mut MlxTextGenerationState,
    ) -> Result<
        Submission<TextOperationOutput<MlxModelInput, MlxTextToken>, MlxTextCompletion>,
        Error,
    > {
        let geometry = runtime
            .session()
            .payload
            .model
            .erased()
            .block_refinement_geometry()
            .ok_or_else(|| {
                Error::ArchitectureModel("missing selected refinement geometry".into())
            })?;
        runtime
            .session()
            .validate_parameter_epoch(&mut state.sampling.parameter_epoch)?;
        let shared_policy = runtime
            .session()
            .payload
            .model
            .erased()
            .capability_estimate()
            .shared_backbone_draft();
        if state.sampling.refinement.is_none()
            && matches!(
                state.sampling.mode,
                eredu_core::decoding::DecodingMode::Diffusion { policy: None }
            )
        {
            if let Some(policy) = shared_policy {
                state.sampling.refinement_policy = policy
                    .diffusion_policy(state.sampling.temperature)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            }
        }

        let phase = state
            .sampling
            .refinement
            .as_ref()
            .map_or(GenerationPhase::Context, |state| state.phase());
        let stream = runtime.backend().stream().clone();
        let domain = TokenDomain::new(geometry.vocabulary_size.get() as usize);
        let submission = match phase {
            GenerationPhase::Context => {
                let cached = if state.sampling.refinement.is_none()
                    && matches!(input, PendingTextInput::Continue)
                {
                    state.sampling.next_logits.take()
                } else {
                    None
                };
                if let Some(scores) = cached {
                    let (backend, session) = runtime.parts_mut();
                    session.submit_settled_logits(backend, scores)?
                } else {
                    let prompt = match (&state.sampling.refinement, input) {
                        (None, PendingTextInput::Prefill(prompt)) => prompt,
                        (None, PendingTextInput::Decode(token)) => {
                            let tokens = token.value.reshape(&[1, 1], &stream)?;
                            let parts = [input::token_ids_part(&tokens)?];
                            MlxModelInput::from(input::ModelInput::new(&parts))
                        }
                        (Some(block), _) => {
                            let arrays = block
                                .authoritative_tokens()
                                .iter()
                                .map(|token| token.as_array())
                                .collect::<Vec<_>>();
                            if arrays.len() != geometry.canvas_length.get() as usize {
                                return Err(Error::ArchitectureModel(
                                    "next context requires a complete authoritative block".into(),
                                ));
                            }
                            let tokens = safemlx::ops::concatenate(&arrays, &stream)?
                                .reshape(&[1, arrays.len() as i32], &stream)?;
                            let parts = [input::input_part(
                                InputModality::Text,
                                input::InputPayload::TokenIds(tokens),
                                [],
                                [],
                            )?];
                            MlxModelInput::from(input::ModelInput::new(&parts))
                        }
                        _ => {
                            return Err(Error::ArchitectureModel(
                                "initial refinement context requires a prepared prompt".into(),
                            ))
                        }
                    };
                    let (backend, session) = runtime.parts_mut();
                    let invocation = prompt.with_borrowed(|input| {
                        let plans = session
                            .payload
                            .model
                            .erased()
                            .prepared_input_plans(input)
                            .map_err(|error| CaptureError::Invalid(error.to_string()))?;
                        let sequence = plans
                            .iter()
                            .try_fold(0u64, |total, part| {
                                total.checked_add(part.decoder_positions())
                            })
                            .ok_or(CaptureError::Overflow)?;
                        session
                            .refinement_capture_shape(sequence)
                            .map(|shape| (CapturePhase::Prefill, shape))
                    });
                    match &mut state.capture {
                        Some(capture) => partition_capture::with_invocation_observer(
                            session,
                            capture,
                            &stream,
                            state.sampling.next_prediction,
                            invocation,
                            |session, observer| {
                                session.submit_prefill_result_with_observer(
                                    backend,
                                    Ok(prompt),
                                    None,
                                    &mut eredu_runtime::BorrowedActivationObserver(observer),
                                )
                            },
                        )?,
                        None => session.submit_prefill_result_with_observer(
                            backend,
                            Ok(prompt),
                            None,
                            &mut eredu_runtime::NoopObserver,
                        )?,
                    }
                }
            }
            GenerationPhase::Refinement => {
                let block = state.sampling.refinement.as_ref().ok_or_else(|| {
                    Error::ArchitectureModel("missing active refinement block".into())
                })?;
                let refinement = block
                    .refinement()
                    .ok_or_else(|| Error::ArchitectureModel("missing tentative canvas".into()))?;
                let canvas = refinement
                    .canvas()
                    .as_array()
                    .reshape(&[1, geometry.canvas_length.get() as i32], &stream)?;
                let feedback = refinement
                    .self_conditioning()
                    .map(|scores| {
                        scores.as_array().reshape(
                            &[
                                1,
                                geometry.canvas_length.get() as i32,
                                geometry.vocabulary_size.get() as i32,
                            ],
                            &stream,
                        )
                    })
                    .transpose()?;
                let (backend, session) = runtime.parts_mut();
                let invocation = session
                    .refinement_capture_shape(u64::from(geometry.canvas_length.get()))
                    .map(|shape| (CapturePhase::Decode, shape));
                match &mut state.capture {
                    Some(capture) => partition_capture::with_invocation_observer(
                        session,
                        capture,
                        &stream,
                        state.sampling.next_prediction,
                        invocation,
                        |session, observer| {
                            session.submit_refinement_with_observer(
                                backend,
                                &canvas,
                                feedback.as_ref(),
                                observer,
                            )
                        },
                    )?,
                    None => session.submit_refinement_with_observer(
                        backend,
                        &canvas,
                        feedback.as_ref(),
                        &mut eredu_runtime::NoopObserver,
                    )?,
                }
            }
            GenerationPhase::Finalization => {
                let (backend, session) = runtime.parts_mut();
                session.submit_finalization_boundary(backend)?
            }
        };
        let operation = ResourceOperation::begin(submission.completion.owner())?;
        let completed = (|| {
            if state.sampling.refinement.is_none() {
                state.sampling.refinement = Some(BlockRefinementState::new(
                    state.sampling.refinement_policy,
                    geometry.canvas_length,
                    domain,
                    submission.output.dtype(),
                    state.sampling.refinement_random.take().map_or_else(
                        || {
                            if shared_policy.is_some() {
                                crate::backend::random::split_key_at(
                                    &safemlx::random::key(state.sampling.refinement_seed)?,
                                    1,
                                    &stream,
                                )
                                .map(RandomState::from_key)
                            } else {
                                RandomState::with_seed(state.sampling.refinement_seed)
                            }
                        },
                        Ok,
                    )?,
                    state.sampling.canvas.clone(),
                ));
            }
            let block = state.sampling.refinement.as_mut().expect("active block");
            let output = match phase {
                GenerationPhase::Context => {
                    if decision.is_some() {
                        return Err(Error::ArchitectureModel(
                            "context progress cannot consume a commitment decision".into(),
                        ));
                    }
                    let block_index = block.block();
                    let mut random = block.random().clone();
                    let seeds = if shared_policy.is_some_and(|policy| policy.seed_first())
                        && state.sampling.canvas.is_none()
                    {
                        use eredu_runtime::generation::refinement::RefinementBackend;
                        let eredu_core::refinement::RefinementPolicy::MaskedUnmasking {
                            config,
                            mask_token_id,
                        } = state.sampling.refinement_policy
                        else {
                            return Err(Error::ArchitectureModel(
                                "seeded masked context requires a masked policy".into(),
                            ));
                        };
                        let scores = MlxTensor::from_array(causal_next_logits(
                            submission.output.clone(),
                            &stream,
                        )?);
                        let mask =
                            MlxTensor::from_array(Array::from_slice(&[mask_token_id as i32], &[1]));
                        let sampled = MlxSamplingBackend::masked_refinement_step(
                            &mask,
                            None,
                            &scores,
                            mask_token_id,
                            config,
                            1,
                            &mut random,
                            &stream,
                        )?
                        .ok_or_else(|| {
                            Error::ArchitectureModel("masked seed sampling is unavailable".into())
                        })?;
                        let seed = sampled
                            .canvas
                            .as_array()
                            .as_dtype(Dtype::Uint32, &stream)?
                            .into_evaluated()?
                            .try_to_vec::<u32>()
                            .map_err(|error| Error::ArchitectureModel(error.to_string()))?[0];
                        let mut seeds = vec![None; geometry.canvas_length.get() as usize];
                        seeds[0] = Some(seed);
                        Some(eredu_core::refinement::RefinementCanvas {
                            seed_tokens: seeds,
                            pinned_positions: if seed == mask_token_id {
                                vec![]
                            } else {
                                vec![0]
                            },
                        })
                    } else {
                        None
                    };
                    block.replace_random(random);
                    block
                        .context_encoded(&stream, seeds.as_ref())
                        .map_err(eredu_nn::Error::backend_source)?;
                    TextOperationOutput::Progress {
                        pending: PendingTextInput::Continue,
                        progress: TextProgress {
                            phase,
                            block: block_index,
                            iteration: 0,
                            tentative_token_ids: None,
                        },
                    }
                }
                GenerationPhase::Refinement => {
                    if decision.is_some() {
                        return Err(Error::ArchitectureModel(
                            "refinement progress cannot consume a commitment decision".into(),
                        ));
                    }
                    let scores = MlxTensor::from_array(submission.output.reshape(
                        &[
                            geometry.canvas_length.get() as i32,
                            geometry.vocabulary_size.get() as i32,
                        ],
                        &stream,
                    )?);
                    block
                        .refine(&scores, &stream)
                        .map_err(eredu_nn::Error::backend_source)?;
                    let tentative_token_ids = if state.sampling.draft_events {
                        Some(
                            block
                                .refinement()
                                .and_then(|refinement| refinement.draft())
                                .ok_or_else(|| {
                                    Error::ArchitectureModel(
                                        "completed refinement has no tentative IDs".into(),
                                    )
                                })?
                                .as_array()
                                .as_dtype(Dtype::Uint32, &stream)?
                                .into_evaluated()?
                                .try_to_vec::<u32>()
                                .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
                        )
                    } else {
                        None
                    };
                    TextOperationOutput::Progress {
                        pending: PendingTextInput::Continue,
                        progress: TextProgress {
                            phase,
                            block: block.block(),
                            iteration: block.iteration(),
                            tentative_token_ids,
                        },
                    }
                }
                GenerationPhase::Finalization => {
                    let decision = decision.ok_or_else(|| {
                        Error::ArchitectureModel(
                            "finalization requires an evolving commitment decision".into(),
                        )
                    })?;
                    let (scores, position) = block
                        .finalization_input()
                        .map_err(eredu_nn::Error::backend_source)?;
                    let finalized = finalize_refinement_position::<MlxSamplingBackend>(
                        scores,
                        position,
                        block
                            .refinement()
                            .and_then(|state| state.selected_candidates()),
                        domain,
                        decision.filter(),
                        &stream,
                    )
                    .map_err(eredu_nn::Error::backend_source)?;
                    block
                        .finalize(
                            finalized.token.clone(),
                            finalized.candidate_replaced,
                            &stream,
                        )
                        .map_err(eredu_nn::Error::backend_source)?;
                    if shared_policy.is_some() {
                        let token_id =
                            <MlxSamplingBackend as eredu_runtime::SamplingBackend>::token_id(
                                &finalized.token,
                                &stream,
                            )?;
                        <MlxTextSampler as eredu_runtime::SpeculativeSampler<
                            MlxSamplingBackend,
                        >>::commit_token(
                            &mut state.sampling.sampler,
                            &finalized.scores,
                            token_id,
                            &stream,
                        )?;
                    }
                    state.sampling.next_prediction = state
                        .sampling
                        .next_prediction
                        .checked_add(1)
                        .ok_or_else(|| {
                            Error::ArchitectureModel("text prediction overflow".into())
                        })?;
                    TextOperationOutput::Token(MlxTextToken {
                        value: finalized.token.into_array(),
                        stream: stream.clone(),
                        owner: Rc::clone(submission.completion.owner()),
                    })
                }
            };
            let root = match &output {
                TextOperationOutput::Token(token) => token.value.clone(),
                _ => block
                    .refinement()
                    .map(|refinement| refinement.canvas().as_array().clone())
                    .unwrap_or_else(|| submission.output.clone()),
            };
            if let Some(capture) = &mut state.capture {
                capture
                    .set_commitment_boundary(state.sampling.next_prediction)
                    .map_err(Error::observation)?;
            }
            Ok((output, MlxCompletion::submission(root)?))
        })();
        let ((output, tail), recovery) = operation.finish(completed)?;
        Ok(Submission {
            output,
            completion: MlxTextCompletion {
                token: tail.completion,
                model: submission.completion,
                recovery: RefCell::new(Some(recovery)),
            },
        })
    }
}

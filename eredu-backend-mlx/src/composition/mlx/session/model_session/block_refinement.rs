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

impl MlxModelSession {
    fn refinement_capture_shape(
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
            .refine_block_with_observer(
                canvas,
                feedback,
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
        let phase = state
            .sampling
            .refinement
            .as_ref()
            .map_or(GenerationPhase::Context, |state| state.phase());
        let stream = runtime.backend().stream().clone();
        let domain = TokenDomain::new(geometry.vocabulary_size.get() as usize);
        let submission = match phase {
            GenerationPhase::Context => {
                let prompt = match (&state.sampling.refinement, input) {
                    (None, PendingTextInput::Prefill(prompt)) => prompt,
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
                    RandomState::with_seed(state.sampling.refinement_seed)?,
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
                    block
                        .context_encoded(&stream)
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
                        domain,
                        decision.filter(),
                        &stream,
                    )
                    .map_err(eredu_nn::Error::backend_source)?;
                    block
                        .finalize(finalized.token.clone())
                        .map_err(eredu_nn::Error::backend_source)?;
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

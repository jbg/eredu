//! Native copying for the shared ordinary-continuation snapshot driver.

use super::*;
use eredu_core::{execution_control::SnapshotEstimate, PendingTextInput};
use eredu_runtime::{capture::CaptureSession, execution_control::TextSnapshotBackend};
use generation::MlxTextSamplingState;

fn array_storage(array: &Array) -> Option<u64> {
    u64::try_from(array.nbytes())
        .ok()?
        .checked_mul(2)?
        .checked_add(4096)?
        .checked_add(u64::try_from(array.shape().len()).ok()?.checked_mul(16)?)
}

fn copy_array(array: &Array, stream: &Stream) -> Result<Array, Error> {
    Ok(array.contiguous(false, stream)?.deep_clone()?)
}

impl eredu_runtime::execution_control::TextSamplingControlBackend for MlxBackend<'_> {
    fn sampling_score_kind(
        _: &ModelRuntime<Self>,
        state: &Self::TextGenerationState,
    ) -> eredu_core::refinement::PredictionScoreKind {
        if matches!(
            state.sampling.mode,
            eredu_core::decoding::DecodingMode::Diffusion { .. }
        ) {
            eredu_core::refinement::PredictionScoreKind::ConstrainedFinalization
        } else {
            eredu_core::refinement::PredictionScoreKind::CausalConditional
        }
    }
    fn sampling_control_facts(
        state: &MlxTextGenerationState,
    ) -> eredu_runtime::execution_control::SamplingStateFacts {
        eredu_runtime::execution_control::SamplingStateFacts {
            temperature: state.sampling.temperature,
            requires_positive_temperature: matches!(
                state.sampling.sampler,
                MlxTextSampler::MirostatV2(_)
            ),
            has_rng: state.sampling.prng.is_some()
                || (matches!(
                    state.sampling.mode,
                    eredu_core::decoding::DecodingMode::Diffusion { .. }
                ) && state.sampling.refinement.is_some()),
        }
    }
    fn install_sampling_override(
        runtime: &mut ModelRuntime<Self>,
        state: &mut MlxTextGenerationState,
        request: eredu_runtime::execution_control::ValidatedSamplingOverride,
    ) -> Result<(), Error> {
        let replacement = runtime.session_mut().with_model_operation(|_| {
            Ok(request.reseed().map(RandomState::with_seed).transpose())
        })??;
        if let Some(random) = replacement {
            if let Some(refinement) = &mut state.sampling.refinement {
                refinement.replace_random(random.clone());
            }
            if let Some(seed) = request.reseed() {
                state.sampling.refinement_seed = seed;
                if runtime
                    .session()
                    .capability_estimate()
                    .map_err(|error| Error::Exception(Exception::from_source(error)))?
                    .shared_backbone_draft()
                    .is_some()
                {
                    let stream = runtime.backend().stream();
                    state.sampling.refinement_random = Some(RandomState::from_key(
                        crate::backend::random::split_key_at(random.as_array(), 1, stream)?,
                    ));
                    state.sampling.proposal_random = Some(RandomState::from_key(
                        crate::backend::random::split_key_at(random.as_array(), 2, stream)?,
                    ));
                    if let Some(refinement) = &mut state.sampling.refinement {
                        refinement.replace_random(
                            state.sampling.refinement_random.as_ref().unwrap().clone(),
                        );
                    }
                }
            }
            state.sampling.prng = Some(random);
        }
        state.sampling.temperature = request.temperature();
        Ok(())
    }
}

fn with_text_prompt_array<T>(
    prompt: &MlxModelInput,
    inspect: impl FnOnce(&Array) -> T,
) -> Option<T> {
    prompt.with_borrowed(|input| {
        let [part] = input.parts else {
            return None;
        };
        if part.modality() != InputModality::Text
            || !part.metadata().is_empty()
            || !part.extents().is_empty()
        {
            return None;
        }
        let input::InputPayload::TokenIds(tokens) = part.payload() else {
            return None;
        };
        if tokens.shape().len() != 2 || tokens.shape()[0] != 1 || tokens.shape()[1] <= 0 {
            return None;
        }
        if !matches!(tokens.dtype(), Dtype::Int32 | Dtype::Uint32) {
            return None;
        }
        Some(inspect(tokens))
    })
}

impl TextSnapshotBackend for MlxBackend<'_> {
    type SamplingState = MlxTextSamplingState;

    fn continuation_input_tokens(
        runtime: &ModelRuntime<Self>,
        sampling: &MlxTextSamplingState,
        pending: Option<PendingTextInput<&MlxModelInput, &MlxTextToken>>,
        predictions: u64,
    ) -> Option<u64> {
        #[cfg(test)]
        if std::env::var_os("EREDU_TEST_SNAPSHOT_GROWTH_TRACE").is_some() {
            eprintln!("snapshot growth input: predictions={predictions}, mode={:?}, transaction={}, logits={}, cached={}, pending={}", sampling.mode,
                sampling.speculative.is_some(), sampling.next_logits.is_some(), sampling.committed_token_cached,
                match pending { None => "none", Some(PendingTextInput::Continue) => "continue", Some(PendingTextInput::Decode(_)) => "decode", Some(PendingTextInput::Prefill(_)) => "prefill" });
        }
        if predictions == 0 {
            return Some(0);
        }
        if let Some(geometry) = runtime.session().block_refinement_geometry().filter(|_| {
            matches!(
                sampling.mode,
                eredu_core::decoding::DecodingMode::Diffusion { .. }
            )
        }) {
            if let Some(block) = &sampling.refinement {
                return block.continuation_context_tokens(predictions);
            }
            let PendingTextInput::Prefill(prompt) = pending? else {
                return None;
            };
            let prompt =
                with_text_prompt_array(prompt, |array| u64::try_from(array.shape()[1]).ok())
                    .flatten()?;
            let subsequent = predictions
                .checked_sub(1)?
                .checked_div(u64::from(geometry.canvas_length.get()))?
                .checked_mul(u64::from(geometry.canvas_length.get()))?;
            return prompt.checked_add(subsequent);
        }
        if let eredu_core::decoding::DecodingMode::SelfSpeculation(options) = sampling.mode {
            if sampling.speculative.is_some()
                && matches!(
                    pending,
                    Some(PendingTextInput::Continue) | Some(PendingTextInput::Decode(_))
                )
            {
                // A retained transaction may finish or reject before the requested
                // horizon. The final transaction can verify beyond that horizon.
                return predictions.checked_add(options.request().max_draft_tokens.get() as u64);
            }
        }
        let inputs = match pending? {
            PendingTextInput::Decode(token) if token.value.size() == 1 => Some(predictions),
            PendingTextInput::Prefill(prompt) if prompt.prefill_cursor().is_some() => {
                let cursor = prompt.prefill_cursor()?;
                u64::try_from(
                    cursor
                        .token_count()
                        .checked_sub(cursor.next_range().start)?,
                )
                .ok()?
                .checked_add(predictions - 1)
            }
            PendingTextInput::Prefill(prompt) => with_text_prompt_array(prompt, |array| {
                u64::try_from(array.shape()[1])
                    .ok()?
                    .checked_add(predictions - 1)
            })
            .flatten(),
            PendingTextInput::Continue if sampling.next_logits.is_some() => {
                predictions.checked_sub(1)
            }
            _ => None,
        };
        match sampling.mode {
            eredu_core::decoding::DecodingMode::SelfSpeculation(options) => {
                inputs?.checked_add(options.request().max_draft_tokens.get() as u64)
            }
            _ => inputs,
        }
    }

    fn decoding_settlement_input_tokens(
        runtime: &ModelRuntime<Self>,
        sampling: &MlxTextSamplingState,
        pending: Option<PendingTextInput<&MlxModelInput, &MlxTextToken>>,
    ) -> Option<u64> {
        match sampling.mode {
            eredu_core::decoding::DecodingMode::Diffusion { .. } => {
                let block = sampling.refinement.as_ref()?;
                if block.phase() == eredu_core::refinement::GenerationPhase::Context {
                    Some(block.authoritative_tokens().len() as u64)
                } else {
                    Some(u64::from(
                        runtime
                            .session()
                            .block_refinement_geometry()?
                            .canvas_length
                            .get(),
                    ))
                }
            }
            eredu_core::decoding::DecodingMode::SelfSpeculation(_)
                if sampling.speculative.is_some() =>
            {
                Some(1)
            }
            eredu_core::decoding::DecodingMode::SelfSpeculation(_) => Some(u64::from(
                matches!(pending, Some(PendingTextInput::Decode(_)))
                    && !sampling.committed_token_cached,
            )),
            _ => Some(0),
        }
    }

    fn estimate_sampling_growth(
        runtime: &ModelRuntime<Self>,
        sampling: &MlxTextSamplingState,
        predictions: u64,
    ) -> Result<Option<u64>, Error> {
        // Both ordinary samplers retain one u32 history entry per decision.
        // Reserve descriptors/materialization for a scalar pending token and a
        // two-u32 RNG key, including a key explicitly introduced by reseeding.
        // Stream/completion owners are shared; no array is allocated to estimate.
        let causal = predictions
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(2 * (4096 + 16 + 2 * 8)));
        let Some(geometry) = runtime.session().block_refinement_geometry() else {
            return Ok(causal);
        };
        let canvas = u64::from(geometry.canvas_length.get());
        let vocabulary = u64::from(geometry.vocabulary_size.get());
        // Every block retains at most two full score matrices, one canvas,
        // one argmax, bounded stability history and finalized scalar handles.
        // These are immutable snapshot values; the next block replaces them.
        let shared_state =
            if let eredu_core::decoding::DecodingMode::SelfSpeculation(options) = sampling.mode {
                use eredu_core::execution_control::NativeTextStateBackend;
                let native = Self::estimate_native_text_state(runtime, None)?;
                let additional =
                    predictions.checked_add(options.request().max_draft_tokens.get() as u64);
                match (native, additional) {
                    (Some(native), Some(additional)) => runtime
                        .session()
                        .installed_control_growth(additional)?
                        .and_then(|growth| native.retained_bytes.checked_add(growth))
                        // Baseline, full verification, accepted restoration and
                        // transactional rollback retain independent mutable cache storage.
                        .and_then(|bytes| bytes.checked_mul(4)),
                    _ => None,
                }
            } else {
                Some(0)
            };
        let estimate = (|| {
            let score_bytes = canvas
                .checked_mul(vocabulary)?
                .checked_mul(16)?
                .checked_add(2 * (4096 + 128))?;
            let history = u64::from(sampling.refinement_policy.stability_history_length())
                .checked_add(if sampling.canvas.is_some() { 5 } else { 3 })?;
            let tokens = canvas.checked_mul(8)?.checked_mul(history)?;
            let scalar_handles = canvas.checked_mul(4096 + 128)?;
            causal?
                .checked_add(score_bytes)?
                .checked_add(tokens)?
                .checked_add(scalar_handles)?
                .checked_add(history.checked_mul(4096 + 128)?)?
                .checked_add(shared_state?)?
                .checked_add(4096)
        })();
        #[cfg(test)]
        if std::env::var_os("EREDU_TEST_SNAPSHOT_GROWTH_TRACE").is_some() {
            eprintln!("snapshot sampling growth: {estimate:?}; shared state={shared_state:?}");
        }
        Ok(estimate)
    }

    fn sampling_state(state: &MlxTextGenerationState) -> &MlxTextSamplingState {
        &state.sampling
    }
    fn install_sampling_state(state: &mut MlxTextGenerationState, sampling: MlxTextSamplingState) {
        state.sampling = sampling;
    }
    fn assemble_generation_state(
        sampling: MlxTextSamplingState,
        capture: Option<CaptureSession>,
    ) -> MlxTextGenerationState {
        MlxTextGenerationState { sampling, capture }
    }
    fn sampling_prediction(sampling: &MlxTextSamplingState) -> u64 {
        sampling.next_prediction
    }
    fn capture_run(state: &MlxTextGenerationState) -> Option<&CaptureSession> {
        state.capture.as_ref()
    }
    fn capture_run_mut(state: &mut MlxTextGenerationState) -> Option<&mut CaptureSession> {
        state.capture.as_mut()
    }

    fn estimate_child_capture(
        _: &ModelRuntime<Self>,
        shape: &[u64],
        selection: &eredu_core::capture::CaptureSelection,
        slice: &eredu_core::capture::ResolvedCaptureSlice,
    ) -> Result<eredu_core::capture::CaptureUsage, eredu_core::capture::CaptureError> {
        super::bounded_capture::estimate_shape(shape, selection, slice)
    }
    fn child_intervention_estimator(
        _: &ModelRuntime<Self>,
    ) -> Result<
        std::sync::Arc<dyn eredu_core::intervention::InterventionEstimator>,
        eredu_core::capture::CaptureError,
    > {
        Ok(std::sync::Arc::new(
            super::intervention::NativeInterventionEstimator,
        ))
    }

    fn estimate_sampling_state(
        _: &ModelRuntime<Self>,
        sampling: &MlxTextSamplingState,
    ) -> Result<Option<SnapshotEstimate>, Error> {
        let history = match &sampling.sampler {
            MlxTextSampler::Standard(sampler) => sampler.generated_tokens(),
            MlxTextSampler::MirostatV2(sampler) => sampler.generated_tokens(),
        };
        let estimate = (|| {
            let mut bytes = u64::try_from(std::mem::size_of::<MlxTextSamplingState>())
                .ok()?
                .checked_add(u64::try_from(history.len()).ok()?.checked_mul(4)?)?;
            if let Some(random) = &sampling.prng {
                bytes = bytes.checked_add(array_storage(random.as_array())?)?;
            }
            if let Some(scores) = &sampling.next_logits {
                bytes = bytes.checked_add(array_storage(scores)?)?;
            }
            if let Some(random) = &sampling.refinement_random {
                bytes = bytes.checked_add(array_storage(random.as_array())?)?;
            }
            if let Some(random) = &sampling.proposal_random {
                bytes = bytes.checked_add(array_storage(random.as_array())?)?;
            }
            if let Some(transaction) = &sampling.speculative {
                bytes = bytes.checked_add(
                    super::model_session::shared_speculation::retained_transaction_bytes(
                        transaction,
                    )?,
                )?;
            }
            bytes = bytes.checked_add(sampling.canvas.as_ref().map_or(0, |c| c.host_bytes()))?;
            let mut copy_bytes = bytes;
            if let Some(refinement) = &sampling.refinement {
                let host = refinement.snapshot_host_bytes()?;
                bytes = bytes.checked_add(host)?;
                copy_bytes = copy_bytes.checked_add(host)?;
                let mut retained = Some(bytes);
                refinement.visit_retained_values(|value| {
                    let storage = match value {
                        eredu_runtime::generation::refinement::RefinementValue::Tokens(value)
                        | eredu_runtime::generation::refinement::RefinementValue::Scores(value)
                        | eredu_runtime::generation::refinement::RefinementValue::Mask(value) => {
                            array_storage(value.as_array())
                        }
                        eredu_runtime::generation::refinement::RefinementValue::Random(value) => {
                            array_storage(value.as_array())
                        }
                    };
                    retained = retained.and_then(|bytes| bytes.checked_add(storage?));
                });
                bytes = retained?;
            }
            Some(SnapshotEstimate {
                retained_bytes: bytes,
                copy_bytes,
            })
        })();
        Ok(estimate)
    }

    fn copy_sampling_state(
        runtime: &mut ModelRuntime<Self>,
        sampling: &MlxTextSamplingState,
    ) -> Result<MlxTextSamplingState, Error> {
        let stream = runtime.backend().stream().clone();
        runtime.session_mut().with_model_operation(|_| {
            Ok((|| {
                let prng = sampling
                    .prng
                    .as_ref()
                    .map(|random| copy_array(random.as_array(), &stream).map(RandomState::from_key))
                    .transpose()?;
                Ok(MlxTextSamplingState {
                    mode: sampling.mode,
                    temperature: sampling.temperature,
                    prng,
                    sampler: sampling.sampler.clone(),
                    next_prediction: sampling.next_prediction,
                    parameter_epoch: sampling.parameter_epoch,
                    refinement_seed: sampling.refinement_seed,
                    proposal_random: sampling
                        .proposal_random
                        .as_ref()
                        .map(|random| {
                            copy_array(random.as_array(), &stream).map(RandomState::from_key)
                        })
                        .transpose()?,
                    speculative: sampling.speculative.clone(),
                    committed_token_cached: sampling.committed_token_cached,
                    next_logits: sampling
                        .next_logits
                        .as_ref()
                        .map(|scores| copy_array(scores, &stream))
                        .transpose()?,
                    refinement_random: sampling
                        .refinement_random
                        .as_ref()
                        .map(|random| {
                            copy_array(random.as_array(), &stream).map(RandomState::from_key)
                        })
                        .transpose()?,
                    refinement_policy: sampling.refinement_policy,
                    draft_events: sampling.draft_events,
                    canvas: sampling.canvas.clone(),
                    refinement: sampling.refinement.clone(),
                })
            })())
        })?
    }

    fn estimate_pending_input(
        _: &ModelRuntime<Self>,
        pending: Option<PendingTextInput<&MlxModelInput, &MlxTextToken>>,
    ) -> Result<Option<SnapshotEstimate>, Error> {
        let bytes = match pending {
            None | Some(PendingTextInput::Continue) => Some(0),
            Some(PendingTextInput::Decode(token)) if token.value.size() == 1 => {
                array_storage(&token.value)
            }
            Some(PendingTextInput::Decode(_)) => None,
            Some(PendingTextInput::Prefill(prompt)) if prompt.prefill_cursor().is_some() => {
                let cursor = prompt.prefill_cursor().expect("retained cursor");
                u64::try_from(cursor.retained_bytes()?)
                    .ok()
                    .and_then(|bytes| {
                        bytes.checked_add(std::mem::size_of::<MlxModelInput>() as u64)
                    })
                    .and_then(|bytes| {
                        bytes.checked_add(match prompt.cache_identity() {
                            Some(identity) => identity.logical_metadata_bytes()?,
                            None => 0,
                        })
                    })
            }
            Some(PendingTextInput::Prefill(prompt)) => prompt.with_borrowed(|input| {
                let mut bytes = u64::try_from(std::mem::size_of::<MlxModelInput>()).ok()?;
                for part in input.parts {
                    bytes = bytes
                        .checked_add(std::mem::size_of::<input::InputPart>() as u64)?
                        .checked_add(array_storage(part.payload().value())?)?;
                    for array in part.metadata().values() {
                        bytes = bytes.checked_add(array_storage(array)?)?;
                    }
                }
                bytes.checked_add(match prompt.cache_identity() {
                    Some(identity) => identity.logical_metadata_bytes()?,
                    None => 0,
                })
            }),
        };
        Ok(bytes.map(|bytes| SnapshotEstimate {
            retained_bytes: bytes,
            copy_bytes: bytes,
        }))
    }

    fn copy_pending_input(
        runtime: &mut ModelRuntime<Self>,
        pending: Option<PendingTextInput<&MlxModelInput, &MlxTextToken>>,
    ) -> Result<Option<PendingTextInput<MlxModelInput, MlxTextToken>>, Error> {
        if Self::estimate_pending_input(
            runtime,
            pending.as_ref().map(|input| match input {
                PendingTextInput::Prefill(prompt) => PendingTextInput::Prefill(*prompt),
                PendingTextInput::Decode(token) => PendingTextInput::Decode(*token),
                PendingTextInput::Continue => PendingTextInput::Continue,
            }),
        )?
        .is_none()
        {
            return Err(Error::ArchitectureModel(
                "pending input cannot be represented by the snapshot contract".into(),
            ));
        }
        let stream = runtime.backend().stream().clone();
        runtime.session_mut().with_model_operation(|_| {
            Ok((|| {
                Ok(match pending {
                    None => None,
                    Some(PendingTextInput::Continue) => Some(PendingTextInput::Continue),
                    Some(PendingTextInput::Decode(token)) => {
                        Some(PendingTextInput::Decode(MlxTextToken {
                            value: copy_array(&token.value, &stream)?,
                            stream: stream.clone(),
                            owner: std::rc::Rc::clone(&token.owner),
                        }))
                    }
                    Some(PendingTextInput::Prefill(prompt))
                        if prompt.prefill_cursor().is_some() =>
                    {
                        // The cursor owns immutable prepared tensors and a separate
                        // mutable continuation position. Native state is snapshotted
                        // by the existing session contract, not by this input clone.
                        Some(PendingTextInput::Prefill(prompt.clone()))
                    }
                    Some(PendingTextInput::Prefill(prompt))
                        if with_text_prompt_array(prompt, |_| ()).is_none() =>
                    {
                        // Prepared inputs expose immutable tensor handles. Evaluate
                        // their roots before sharing storage with a snapshot so no
                        // pending native request work outlives this operation guard.
                        prompt.with_borrowed(|input| {
                            let roots = input
                                .parts
                                .iter()
                                .flat_map(|part| {
                                    std::iter::once(part.payload().value())
                                        .chain(part.metadata().values())
                                })
                                .collect::<Vec<_>>();
                            safemlx::transforms::eval(roots)
                        })?;
                        Some(PendingTextInput::Prefill(prompt.clone()))
                    }
                    Some(PendingTextInput::Prefill(prompt)) => {
                        let tokens =
                            with_text_prompt_array(prompt, |tokens| copy_array(tokens, &stream))
                                .ok_or_else(|| {
                                Error::ArchitectureModel("unsupported snapshot prompt".into())
                            })??;
                        let parts = [input::input_part(
                            InputModality::Text,
                            input::InputPayload::TokenIds(tokens),
                            [],
                            [],
                        )?];
                        let input = match prompt.cache_identity() {
                            Some(identity) => {
                                input::ModelInput::with_cache_identity(&parts, identity)
                            }
                            None => input::ModelInput::new(&parts),
                        };
                        Some(PendingTextInput::Prefill(MlxModelInput::from(input)))
                    }
                })
            })())
        })?
    }
}

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
    fn sampling_control_facts(
        state: &MlxTextGenerationState,
    ) -> eredu_runtime::execution_control::SamplingStateFacts {
        eredu_runtime::execution_control::SamplingStateFacts {
            temperature: state.sampling.temperature,
            requires_positive_temperature: matches!(
                state.sampling.sampler,
                MlxTextSampler::MirostatV2(_)
            ),
            has_rng: state.sampling.prng.is_some(),
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
        pending: Option<PendingTextInput<&MlxModelInput, &MlxTextToken>>,
        predictions: u64,
    ) -> Option<u64> {
        if predictions == 0 {
            return Some(0);
        }
        match pending? {
            PendingTextInput::Decode(token) if token.value.size() == 1 => Some(predictions),
            PendingTextInput::Prefill(prompt) => with_text_prompt_array(prompt, |array| {
                u64::try_from(array.shape()[1])
                    .ok()?
                    .checked_add(predictions - 1)
            })
            .flatten(),
            _ => None,
        }
    }

    fn estimate_sampling_growth(
        _: &ModelRuntime<Self>,
        _: &MlxTextSamplingState,
        predictions: u64,
    ) -> Result<Option<u64>, Error> {
        // Both ordinary samplers retain one u32 history entry per decision.
        // Reserve descriptors/materialization for a scalar pending token and a
        // two-u32 RNG key, including a key explicitly introduced by reseeding.
        // Stream/completion owners are shared; no array is allocated to estimate.
        Ok(predictions
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(2 * (4096 + 16 + 2 * 8))))
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
            Some(SnapshotEstimate {
                retained_bytes: bytes,
                copy_bytes: bytes,
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
                    temperature: sampling.temperature,
                    prng,
                    sampler: sampling.sampler.clone(),
                    next_prediction: sampling.next_prediction,
                })
            })())
        })?
    }

    fn estimate_pending_input(
        _: &ModelRuntime<Self>,
        pending: Option<PendingTextInput<&MlxModelInput, &MlxTextToken>>,
    ) -> Result<Option<SnapshotEstimate>, Error> {
        let bytes = match pending {
            None => Some(0),
            Some(PendingTextInput::Decode(token)) if token.value.size() == 1 => {
                array_storage(&token.value)
            }
            Some(PendingTextInput::Decode(_)) => None,
            Some(PendingTextInput::Prefill(prompt)) => with_text_prompt_array(prompt, |array| {
                array_storage(array)?
                    .checked_add(u64::try_from(std::mem::size_of::<MlxModelInput>()).ok()?)?
                    .checked_add(u64::try_from(std::mem::size_of::<input::InputPart>()).ok()?)?
                    .checked_add(match prompt.cache_identity() {
                        Some(identity) => identity.logical_metadata_bytes()?,
                        None => 0,
                    })
            })
            .flatten(),
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
            }),
        )?
        .is_none()
        {
            return Err(Error::ArchitectureModel(
                "snapshots require a single-sequence ordinary token-ID input".into(),
            ));
        }
        let stream = runtime.backend().stream().clone();
        runtime.session_mut().with_model_operation(|_| {
            Ok((|| {
                Ok(match pending {
                    None => None,
                    Some(PendingTextInput::Decode(token)) => {
                        Some(PendingTextInput::Decode(MlxTextToken {
                            value: copy_array(&token.value, &stream)?,
                            stream: stream.clone(),
                            owner: std::rc::Rc::clone(&token.owner),
                        }))
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

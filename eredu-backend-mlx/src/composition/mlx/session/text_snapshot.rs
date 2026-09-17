//! Native copying for the shared ordinary-continuation snapshot driver.

use super::*;
use crate::backend::managed_memory::{NativeMemoryOwner, NativeMemoryRetention};
use eredu_core::{PendingTextInput, execution_control::SnapshotEstimate};
use eredu_runtime::{capture::CaptureSession, execution_control::TextSnapshotBackend};
use generation::{MlxOrdinarySampler, MlxTextSamplingState, TextInferenceRetention};

mod cold_estimate;
mod pending_input;
mod saved_components;
pub use saved_components::MlxSavedTextComponents;
mod saved_sampling;
pub use saved_sampling::MlxSavedSamplingState;

fn sampling_growth_estimate(predictions: u64) -> Option<u64> {
    predictions
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(2 * (4096 + 16 + 2 * 8)))
}

fn array_storage(array: &Array) -> Option<u64> {
    crate::backend::runtime::cache::state::snapshot_estimate::array_bytes(
        array.nbytes(),
        array.shape().len(),
    )
}

// Shared logical sampler equation. The original caller supplies a borrowed
// descriptor reader; ordinary estimation preserves its existing native getters.
fn sampling_storage(
    sampling: &MlxTextSamplingState,
    array_bytes: impl Fn(&Array) -> Option<u64>,
) -> Option<SnapshotEstimate> {
    // The isolated worker copies the complete fixed history box,
    // including cleared or spare slots. This remains a logical
    // snapshot estimate for the enclosing native component; its array
    // and ownership metadata allowances are not physical admission.
    let history = sampling.sampler.prepare_copy().ok()?;
    sampling_storage_parts(
        history.history_bytes(),
        sampling.inference_retention.logical_metadata_bytes()?,
        sampling.memory_retention.logical_metadata_bytes()?,
        sampling.prng.as_ref().map(|random| random.as_array()),
        array_bytes,
    )
}

fn sampling_storage_parts(
    history_bytes: u64,
    inference_metadata: u64,
    memory_metadata: u64,
    key: Option<&Array>,
    array_bytes: impl Fn(&Array) -> Option<u64>,
) -> Option<SnapshotEstimate> {
    let mut bytes = u64::try_from(std::mem::size_of::<MlxTextSamplingState>())
        .ok()?
        .checked_add(history_bytes)?
        .checked_add(inference_metadata)?
        .checked_add(memory_metadata.max(NativeMemoryRetention::singleton_metadata_bytes()))?;
    if let Some(key) = key {
        bytes = bytes.checked_add(array_bytes(key)?)?;
    }
    Some(SnapshotEstimate {
        retained_bytes: bytes,
        copy_bytes: bytes,
    })
}

fn copy_array(array: &Array, stream: &Stream, owner: &NativeMemoryOwner) -> Result<Array, Error> {
    let copy = crate::backend::array_copy::IsolatedArrayCopy::new(array).copy(stream)?;
    // A copied prompt can expose this backing through its borrowed input view.
    // Complete the copy inside recovery before attaching physical ownership.
    copy.evaluated()?;
    owner.retain_array(&copy)?;
    Ok(copy)
}

#[cfg(test)]
mod sampling_copy_tests;

impl eredu_runtime::execution_control::TextSamplingControlBackend for MlxBackend<'_> {
    fn sampling_control_facts(
        state: &MlxTextGenerationState,
    ) -> eredu_runtime::execution_control::SamplingStateFacts {
        eredu_runtime::execution_control::SamplingStateFacts {
            temperature: state.sampling.temperature,
            requires_positive_temperature: matches!(
                state.sampling.sampler.as_sampler(),
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
        if state.sampling.quote.is_some() || state.sampling.sampler.is_funded() {
            // Semantic validation alone does not reprice replacement overlap
            // or the future native sampling trace of this retained run.
            return Err(Error::Other(Box::new(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            )));
        }
        let Some(seed) = request.reseed() else {
            runtime.session().ensure_no_submission_in_flight()?;
            state.sampling.temperature = request.temperature();
            return Ok(());
        };
        let owner = NativeMemoryOwner::acquire(runtime.backend().memory_pool())?;
        let mut memory = state.sampling.memory_retention.clone();
        memory.retain(&owner);
        let replacement = super::recovery::detached_retained(
            TextInferenceRetention::new(state.sampling.inference_retention.clone(), memory.clone()),
            || {
                runtime.session_mut().with_model_operation(|_| {
                    Ok((|| {
                        let random = RandomState::with_seed(seed)?;
                        random.as_array().evaluated()?;
                        owner.retain_array(random.as_array())?;
                        Ok::<_, Error>(random)
                    })())
                })?
            },
        )?;
        state.sampling.prng = Some(replacement);
        state.sampling.memory_retention = memory;
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
    type SavedSamplingState = MlxSavedSamplingState;
    type SavedTextComponents = MlxSavedTextComponents;

    fn original_snapshot_estimates(
        runtime: &ModelRuntime<Self>,
        sampling: &MlxTextSamplingState,
        pending: Option<PendingTextInput<&MlxModelInput, &MlxTextToken>>,
    ) -> Option<[SnapshotEstimate; 3]> {
        cold_estimate::inspect(runtime, sampling, pending)
    }

    fn original_generation_snapshot_estimates(
        runtime: &ModelRuntime<Self>,
        state: &Self::TextGenerationState,
        pending: Option<PendingTextInput<&Self::Prompt, &Self::Token>>,
    ) -> Option<[SnapshotEstimate; 3]> {
        let mut estimates = cold_estimate::inspect(runtime, &state.sampling, pending)?;
        let capture =
            model_session::saved_array_copy::capture::PreparedCaptureCopy::inspect(state).ok()?;
        if let Some(capture) = capture {
            let bytes = capture.logical_bytes()?;
            estimates[1].retained_bytes = estimates[1].retained_bytes.checked_add(bytes)?;
            estimates[1].copy_bytes = estimates[1].copy_bytes.checked_add(bytes)?;
        }
        Some(estimates)
    }

    fn original_saved_generation_preparation_bytes(
        runtime: &ModelRuntime<Self>,
        state: &Self::TextGenerationState,
        pending: Option<PendingTextInput<&Self::Prompt, &Self::Token>>,
    ) -> Result<Option<u64>, eredu_runtime::working_memory::WorkingMemoryError> {
        use eredu_runtime::working_memory::WorkingMemoryError;
        use model_session::saved_array_copy::{capture, decoder::PreparedTextComponentsCopy};
        let capture = capture::PreparedCaptureCopy::inspect(state).map_err(capture::into_memory)?;
        let bytes = PreparedTextComponentsCopy::known_input_preparation_with_capture_bytes(
            runtime,
            &state.sampling,
            pending,
            capture.as_ref().map(|capture| capture.source()),
        )?;
        let capture_bytes = capture
            .as_ref()
            .map(|capture| capture.required_bytes())
            .transpose()
            .map_err(capture::into_memory)?
            .unwrap_or(0);
        let bytes = bytes
            .checked_add(capture_bytes)
            .ok_or(WorkingMemoryError::Overflow)?;
        u64::try_from(bytes)
            .map(Some)
            .map_err(|_| WorkingMemoryError::Overflow)
    }

    fn original_saved_components_resume_preparation_bytes<C: eredu_core::TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        saved: &Self::SavedTextComponents,
        _config: eredu_core::TextGenerationConfig,
        _controller: &C,
    ) -> Result<Option<u64>, eredu_runtime::working_memory::WorkingMemoryError> {
        // H0 follows the immutable saved source and its actual pending geometry.
        // Config/controller-dependent equations use the separately funded planning
        // context and fresh request; the shared provider owns controller copying.
        saved.original_resume_preparation_bytes(runtime).map(Some)
    }

    fn original_saved_components_resume_estimate(
        runtime: &ModelRuntime<Self>,
        saved: &Self::SavedTextComponents,
        config: eredu_core::TextGenerationConfig,
    ) -> Option<SnapshotEstimate> {
        saved.original_resume_estimate(runtime, config)
    }

    fn original_saved_components_preparation_bytes(
        runtime: &ModelRuntime<Self>,
        sampling: &Self::SamplingState,
        pending: Option<PendingTextInput<&Self::Prompt, &Self::Token>>,
    ) -> Result<Option<u64>, eredu_runtime::working_memory::WorkingMemoryError> {
        let bytes = model_session::saved_array_copy::decoder::PreparedTextComponentsCopy::
            known_input_preparation_component_bytes(runtime, sampling, pending)?;
        u64::try_from(bytes)
            .map(Some)
            .map_err(|_| eredu_runtime::working_memory::WorkingMemoryError::Overflow)
    }

    fn original_snapshot_host_pool(
        runtime: &ModelRuntime<Self>,
    ) -> Option<&eredu_runtime::working_memory::WorkingMemoryPool> {
        Some(runtime.backend().memory_pool())
    }

    fn capture_saved_components(
        runtime: &mut ModelRuntime<Self>,
        sampling: &Self::SamplingState,
        pending: Option<PendingTextInput<&Self::Prompt, &Self::Token>>,
        policy: eredu_runtime::execution_control::SamplingCopyPolicy,
    ) -> Result<Self::SavedTextComponents, Error> {
        MlxSavedTextComponents::capture(runtime, sampling, pending, policy, None)
    }

    fn capture_saved_components_with_host(
        runtime: &mut ModelRuntime<Self>,
        sampling: &Self::SamplingState,
        pending: Option<PendingTextInput<&Self::Prompt, &Self::Token>>,
        policy: eredu_runtime::execution_control::SamplingCopyPolicy,
        host: &eredu_core::HostPreparationAuthority,
    ) -> Result<Self::SavedTextComponents, Error> {
        MlxSavedTextComponents::capture(runtime, sampling, pending, policy, Some(host))
    }

    fn capture_saved_generation_components_with_host(
        runtime: &mut ModelRuntime<Self>,
        state: &Self::TextGenerationState,
        pending: Option<PendingTextInput<&Self::Prompt, &Self::Token>>,
        policy: eredu_runtime::execution_control::SamplingCopyPolicy,
        host: &eredu_core::HostPreparationAuthority,
    ) -> Result<Self::SavedTextComponents, Error> {
        MlxSavedTextComponents::capture_generation(runtime, state, pending, policy, host)
    }

    fn copy_saved_components(
        runtime: &mut ModelRuntime<Self>,
        saved: &Self::SavedTextComponents,
        policy: eredu_runtime::execution_control::SamplingCopyPolicy,
    ) -> Result<Self::SavedTextComponents, Error> {
        saved.copy(runtime, policy)
    }

    fn saved_sampling(saved: &Self::SavedTextComponents) -> &Self::SavedSamplingState {
        saved.sampling()
    }

    fn validate_saved_components(
        runtime: &ModelRuntime<Self>,
        saved: &Self::SavedTextComponents,
    ) -> Result<(), Error> {
        saved.validate(runtime)
    }

    fn estimate_saved_components(
        runtime: &ModelRuntime<Self>,
        saved: &Self::SavedTextComponents,
    ) -> Result<Option<SnapshotEstimate>, Error> {
        saved.estimate(runtime)
    }

    fn estimate_saved_native_growth(
        runtime: &ModelRuntime<Self>,
        saved: &Self::SavedTextComponents,
        input_tokens: u64,
    ) -> Result<Option<u64>, Error> {
        saved.native_growth(runtime, input_tokens)
    }

    fn prepare_saved_components_resume(
        runtime: &mut ModelRuntime<Self>,
        saved: &Self::SavedTextComponents,
    ) -> Result<
        (
            Self::NativeTextState,
            Self::SamplingState,
            Option<PendingTextInput<Self::Prompt, Self::Token>>,
        ),
        Error,
    > {
        saved.prepare_resume(runtime)
    }

    fn capture_saved_sampling(
        runtime: &mut ModelRuntime<Self>,
        sampling: &Self::SamplingState,
        pending: Option<PendingTextInput<&Self::Prompt, &Self::Token>>,
        policy: eredu_runtime::execution_control::SamplingCopyPolicy,
    ) -> Result<Self::SavedSamplingState, Error> {
        MlxSavedSamplingState::capture(runtime, sampling, pending, policy)
    }

    fn copy_saved_sampling(
        runtime: &mut ModelRuntime<Self>,
        saved: &Self::SavedSamplingState,
        policy: eredu_runtime::execution_control::SamplingCopyPolicy,
    ) -> Result<Self::SavedSamplingState, Error> {
        saved.copy(runtime, policy)
    }

    fn saved_sampling_prediction(saved: &Self::SavedSamplingState) -> u64 {
        saved.prediction()
    }

    fn estimate_saved_sampling(
        runtime: &ModelRuntime<Self>,
        saved: &Self::SavedSamplingState,
    ) -> Result<Option<SnapshotEstimate>, Error> {
        saved.estimate(runtime)
    }

    fn saved_input_tokens(saved: &Self::SavedSamplingState, predictions: u64) -> Option<u64> {
        saved.input_tokens(predictions)
    }

    fn estimate_saved_sampling_growth(
        _: &ModelRuntime<Self>,
        _: &Self::SavedSamplingState,
        predictions: u64,
    ) -> Result<Option<u64>, Error> {
        // Logical continuation growth only. Fresh physical resume admission is
        // a separate fallible step, never authorized by this diagnostic.
        Ok(sampling_growth_estimate(predictions))
    }

    fn prepare_saved_sampling_resume(
        runtime: &mut ModelRuntime<Self>,
        saved: &Self::SavedSamplingState,
    ) -> Result<
        (
            Self::SamplingState,
            Option<PendingTextInput<Self::Prompt, Self::Token>>,
        ),
        Error,
    > {
        saved.prepare_resume(runtime)
    }

    fn continuation_input_tokens(
        pending: Option<PendingTextInput<&MlxModelInput, &MlxTextToken>>,
        predictions: u64,
    ) -> Option<u64> {
        if predictions == 0 {
            return Some(0);
        }
        match pending? {
            PendingTextInput::Decode(token) if token.value.size() == 1 => Some(predictions),
            PendingTextInput::Prefill(prompt) => {
                pending_input::decoder_positions(prompt)?.checked_add(predictions - 1)
            }
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
        Ok(sampling_growth_estimate(predictions))
    }

    fn sampling_state(state: &MlxTextGenerationState) -> &MlxTextSamplingState {
        &state.sampling
    }
    fn install_sampling_state(
        state: &mut MlxTextGenerationState,
        mut sampling: MlxTextSamplingState,
    ) {
        sampling
            .inference_retention
            .extend_from(&state.sampling.inference_retention);
        sampling
            .memory_retention
            .extend_from(&state.sampling.memory_retention);
        state.sampling = sampling;
    }
    fn assemble_generation_state(
        sampling: MlxTextSamplingState,
        capture: Option<CaptureSession>,
    ) -> MlxTextGenerationState {
        MlxTextGenerationState {
            sampling,
            capture,
            funded_capture: None,
            funding: None,
        }
    }
    fn rebind_pending_capture(
        runtime: &ModelRuntime<Self>,
        saved: Option<&eredu_runtime::capture::CaptureCheckpoint>,
        capture: Option<&CaptureSession>,
        pending: &mut Option<PendingTextInput<MlxModelInput, MlxTextToken>>,
    ) -> Result<(), Error> {
        let Some(PendingTextInput::Prefill(prompt)) = pending else {
            return Ok(());
        };
        prompt.rebind_ordinary_capture(runtime, saved, capture)
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
        Ok(sampling_storage(sampling, array_storage))
    }

    fn copy_sampling_state(
        runtime: &mut ModelRuntime<Self>,
        sampling: &MlxTextSamplingState,
    ) -> Result<MlxTextSamplingState, Error> {
        if sampling.quote.is_some() || sampling.sampler.is_funded() {
            return Err(Error::Other(Box::new(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            )));
        }
        // The plan borrows the exact history and controls through native copy
        // preparation. It allocates nothing and grants no destination funding.
        let sampler = sampling
            .sampler
            .prepare_copy()
            .map_err(|error| Error::Other(Box::new(error)))?;
        let owner = NativeMemoryOwner::acquire(runtime.backend().memory_pool())?;
        let memory = NativeMemoryRetention::from_owner(&owner);
        let mut recovery_memory = sampling.memory_retention.clone();
        recovery_memory.extend_from(&memory);
        let stream = runtime.backend().stream().clone();
        super::recovery::detached_retained(
            TextInferenceRetention::new(sampling.inference_retention.clone(), recovery_memory),
            || {
                runtime.session_mut().with_model_operation(|_| {
                    Ok((|| {
                        let prng = sampling
                            .prng
                            .as_ref()
                            .map(|random| {
                                copy_array(random.as_array(), &stream, &owner)
                                    .map(RandomState::from_key)
                            })
                            .transpose()?;
                        Ok(MlxTextSamplingState {
                            temperature: sampling.temperature,
                            prng,
                            sampler: MlxOrdinarySampler::Unquoted(sampler.copy()),
                            next_prediction: sampling.next_prediction,
                            parameter_epoch: sampling.parameter_epoch,
                            inference_retention: sampling.inference_retention.clone(),
                            memory_retention: memory,
                            quote: sampling.quote.clone(),
                        })
                    })())
                })?
            },
        )
    }

    fn estimate_pending_input(
        _: &ModelRuntime<Self>,
        pending: Option<PendingTextInput<&MlxModelInput, &MlxTextToken>>,
    ) -> Result<Option<SnapshotEstimate>, Error> {
        let bytes = match pending {
            None => Some(0),
            Some(PendingTextInput::Decode(token)) if token.value.size() == 1 => {
                // The descriptor allowance includes the fresh authority and its
                // retention in the backing and shared token submission owner.
                array_storage(&token.value)
            }
            Some(PendingTextInput::Decode(_)) => None,
            Some(PendingTextInput::Prefill(prompt)) => {
                return Ok(
                    pending_input::PromptCopyPlan::prepare(prompt).map(|plan| plan.estimate())
                );
            }
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
        let pending = match pending {
            None => return Ok(None),
            Some(PendingTextInput::Prefill(prompt)) => {
                if prompt.has_original_input_custody() {
                    return Err(pending_input::unknown());
                }
                let plan = pending_input::PromptCopyPlan::prepare(prompt).ok_or_else(|| {
                    Error::ArchitectureModel("ordinary snapshot input needs consumed source attribution or legacy text IDs".into())
                })?;
                return plan
                    .copy(runtime)
                    .map(|prompt| Some(PendingTextInput::Prefill(prompt)));
            }
            pending => pending,
        };
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
        let owner = NativeMemoryOwner::acquire(runtime.backend().memory_pool())?;
        let memory = NativeMemoryRetention::from_owner(&owner);
        let stream = runtime.backend().stream().clone();
        let mut retention = eredu_runtime::working_memory::InferenceRetention::new();
        match pending.as_ref() {
            Some(PendingTextInput::Decode(token)) => {
                retention.extend_from(&token.owner.inference_retention())
            }
            Some(PendingTextInput::Prefill(prompt)) => prompt.with_borrowed(|input| {
                if let Some(request) = input.inference_request() {
                    retention.retain(request);
                }
            }),
            None => {}
        }
        super::recovery::detached_retained(
            TextInferenceRetention::new(retention, memory.clone()),
            || {
                runtime.session_mut().with_model_operation(|_| {
                    Ok((|| {
                        Ok(match pending {
                            None => None,
                            Some(PendingTextInput::Decode(token)) => {
                                let value = copy_array(&token.value, &stream, &owner)?;
                                token.owner.retain_memory(&memory);
                                let mut copied = MlxTextToken::new_with_scalar_scope_and_capture(
                                    value,
                                    stream.clone(),
                                    token.owner.clone(),
                                    None,
                                    token.ordinary_error_custody().cloned(),
                                );
                                if let Some(receipt) = token.step_receipt() {
                                    copied
                                        .attach_step_receipt(receipt.clone())
                                        .map_err(|error| Error::Other(Box::new(error)))?;
                                }
                                Some(PendingTextInput::Decode(copied))
                            }
                            Some(PendingTextInput::Prefill(_)) => {
                                unreachable!("prefill copy uses its exact borrowed plan")
                            }
                        })
                    })())
                })?
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_runtime::working_memory::WorkingMemoryPool;

    #[test]
    fn copied_input_preserves_values_and_retains_escaped_backing_until_final_alias() {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
        let owner = NativeMemoryOwner::acquire(&pool).unwrap();
        let source = Array::from_slice(&[1_u32, 3, 5, 7, 9], &[1, 5]);
        let cropped = source.try_index_device((.., 1..4), &stream).unwrap();
        assert_eq!(cropped.allocation_info().unwrap(), None);
        let copied = super::super::recovery::detached_retained(owner.clone(), || {
            copy_array(&cropped, &stream, &owner)
        })
        .unwrap();
        assert_eq!(copied.shape(), &[1, 3]);
        assert_eq!(copied.evaluated().unwrap().as_slice::<u32>(), &[3, 5, 7]);
        assert_ne!(
            copied.allocation_info().unwrap().unwrap().identity(),
            source.allocation_info().unwrap().unwrap().identity(),
        );
        let escaped = copied.try_index_device((.., ..2), &stream).unwrap();
        drop((copied, cropped, source, owner));
        safemlx::reclaim_allocation_owners();
        assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
        assert_eq!(escaped.evaluated().unwrap().as_slice::<u32>(), &[3, 5]);
        drop(escaped);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while pool.unquoted_owner_count().unwrap() != 0 {
            stream.synchronize().unwrap();
            safemlx::reclaim_allocation_owners();
            assert!(std::time::Instant::now() < deadline);
            std::thread::yield_now();
        }
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

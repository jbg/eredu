//! Native copying for the shared ordinary-continuation snapshot driver.

use super::*;
use crate::backend::managed_memory::{NativeMemoryOwner, NativeMemoryRetention};
use eredu_core::{execution_control::SnapshotEstimate, PendingTextInput};
use eredu_runtime::{capture::CaptureSession, execution_control::TextSnapshotBackend};
use generation::{MlxTextSamplingState, TextInferenceRetention};

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

impl eredu_core::TextSamplingControlBackend for MlxBackend<'_> {
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
    fn apply_sampling_override(
        runtime: &mut ModelRuntime<Self>,
        state: &mut MlxTextGenerationState,
        context: Option<&eredu_core::TextStepContext>,
        request: eredu_core::SamplingOverride,
    ) -> Result<eredu_core::SamplingStateFacts, eredu_core::SamplingOverrideError<Error>> {
        let action = eredu_runtime::execution_control::prepare_sampling_override::<Self>(
            runtime, state, request,
        )?;
        install_sampling_override(runtime, state, action, context)
            .map_err(eredu_core::SamplingOverrideError::Backend)?;
        Ok(Self::sampling_control_facts(state))
    }
}

fn install_sampling_override(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    state: &mut MlxTextGenerationState,
    request: eredu_runtime::execution_control::ValidatedSamplingOverride,
    context: Option<&eredu_core::TextStepContext>,
) -> Result<(), Error> {
    if state.sampling.quote.is_some() || state.sampling.sampler.is_funded() {
        if context.is_none() {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        let context = context.ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
        ))?;
        let quote = state.sampling.quote.clone().ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
        ))?;
        return quote.replace_sampling(runtime, &mut state.sampling, context, request);
    }
    let Some(seed) = request.reseed() else {
        runtime.session().ensure_no_submission_in_flight()?;
        state.sampling.temperature = request.temperature();
        return Ok(());
    };
    let owner = NativeMemoryOwner::acquire(runtime.backend().memory_ledger())?;
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
        config: eredu_core::TextGenerationConfig,
        _controller: &C,
        options: &eredu_core::OriginalTextResumeOptions<'_>,
    ) -> Result<Option<u64>, eredu_runtime::working_memory::WorkingMemoryError> {
        // H0 follows the immutable saved source and its actual pending geometry.
        // Config/controller-dependent equations use the separately funded planning
        // context and fresh request; the shared provider owns controller copying.
        saved
            .original_resume_preparation_bytes(runtime, config, options)
            .map(Some)
    }

    fn original_saved_components_resume_estimate(
        runtime: &ModelRuntime<Self>,
        saved: &Self::SavedTextComponents,
        config: eredu_core::TextGenerationConfig,
        options: &eredu_core::OriginalTextResumeOptions<'_>,
    ) -> Option<SnapshotEstimate> {
        saved.original_resume_estimate(runtime, config, options)
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
    ) -> Option<&eredu_runtime::working_memory::MemoryLedger> {
        Some(runtime.backend().memory_ledger())
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

    fn saved_sampling(saved: &Self::SavedTextComponents) -> &Self::SavedSamplingState {
        saved.sampling()
    }

    fn saved_capture_checkpoint(
        saved: &Self::SavedTextComponents,
    ) -> Option<&eredu_runtime::capture::FundedCaptureCheckpoint> {
        saved.funded_source_fixed()?.capture_checkpoint()
    }

    fn validate_saved_components(
        runtime: &ModelRuntime<Self>,
        saved: &Self::SavedTextComponents,
    ) -> Result<(), Error> {
        saved.validate(runtime)
    }

    fn estimate_saved_native_growth(
        runtime: &ModelRuntime<Self>,
        saved: &Self::SavedTextComponents,
        input_tokens: u64,
    ) -> Result<Option<u64>, Error> {
        saved.native_growth(runtime, input_tokens)
    }

    fn saved_sampling_prediction(saved: &Self::SavedSamplingState) -> u64 {
        saved.prediction()
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
    fn sampling_prediction(sampling: &MlxTextSamplingState) -> u64 {
        sampling.next_prediction
    }
    fn capture_run(state: &MlxTextGenerationState) -> Option<&CaptureSession> {
        state.capture.as_ref()
    }
    fn capture_usage(state: &MlxTextGenerationState) -> eredu_core::capture::CaptureUsage {
        state
            .funded_capture
            .as_ref()
            .map(|capture| capture.collector().usage())
            .or_else(|| state.capture.as_ref().map(CaptureSession::cumulative_usage))
            .unwrap_or_default()
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
}

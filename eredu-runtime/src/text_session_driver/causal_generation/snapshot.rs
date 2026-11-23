use super::*;
use crate::executable_binding::ErasedTextExecutable;
use crate::text_session_driver::{TextExecutable, TextPrefillInput};
use crate::{
    execution_control::{
        SamplingStateFacts, TextSamplingControlBackend, TextSnapshotBackend,
        ValidatedSamplingOverride,
    },
    text_session_driver::snapshot::{
        copy_pending_input, estimate_pending_input, pending_geometry, shared_verification_growth,
        SnapshotPendingGeometry,
    },
};
use eredu_core::{
    backend::TextTransactionMechanisms,
    execution_control::{NativeTextStateSession, SnapshotEstimate, TextSnapshotNative},
    PendingTextInput,
};

type Driver<R, N> = TextSessionDriver<R, N, CausalTextGeneration>;

impl<B, R, N> TextSnapshotBackend<Driver<R, N>> for B
where
    B: TextSnapshotNative<Error = BackendFailure> + LocalCausalBackend,
    N: TextSessionMechanisms<
            B,
            Resources = R,
            TextValues: NativeTextValues<InputTensor = B::InputTensor>,
        > + TextTransactionMechanisms<Error = BackendFailure>,
    R: DerefMut<Target = TextSessionResourcesFor<B, N>> + 'static,
    Driver<R, N>: NativeTextStateSession<B>
        + TextOperationDriver<B, Prompt = TextPromptFor<B>, TextGenerationState = CausalRequest<B>>,
{
    type SamplingState = HostCausalSamplingState<B>;

    fn sampling_state(state: &CausalRequest<B>) -> &Self::SamplingState {
        &state.sampling
    }

    fn install_sampling_state(state: &mut CausalRequest<B>, sampling: Self::SamplingState) {
        state.sampling = sampling;
        state.pending = None;
    }

    fn assemble_generation_state(
        sampling: Self::SamplingState,
        #[cfg(feature = "tensor-capture")] capture: Option<crate::capture::CaptureSession>,
    ) -> CausalRequest<B> {
        CausalRequest {
            sampling,
            pending: None,
            readout: Default::default(),
            #[cfg(feature = "tensor-capture")]
            capture,
        }
    }

    fn sampling_prediction(sampling: &Self::SamplingState) -> u64 {
        sampling.next_prediction
    }

    fn estimate_sampling_state(
        _: &ModelRuntime<B, Driver<R, N>>,
        state: &Self::SamplingState,
    ) -> Result<Option<SnapshotEstimate>, BackendFailure> {
        Ok(state.sampler.history_bytes().and_then(|bytes| {
            let bytes = bytes.checked_add(std::mem::size_of_val(state) as u64)?;
            let shared = state
                .shared
                .as_ref()
                .map_or(Some(0), |shared| shared.retained_bytes())?;
            let bytes = bytes.checked_add(shared)?;
            let block = state.refinement.as_ref().map_or(
                Some(SnapshotEstimate {
                    retained_bytes: 0,
                    copy_bytes: 0,
                }),
                LocalRefinementBlock::estimate_snapshot,
            )?;
            Some(SnapshotEstimate {
                retained_bytes: bytes.checked_add(block.retained_bytes)?,
                copy_bytes: bytes.checked_add(block.copy_bytes)?,
            })
        }))
    }

    fn copy_sampling_state(
        _: &mut ModelRuntime<B, Driver<R, N>>,
        state: &Self::SamplingState,
    ) -> Result<Self::SamplingState, BackendFailure> {
        Ok(state.clone())
    }

    fn continuation_input_tokens(
        runtime: &ModelRuntime<B, Driver<R, N>>,
        sampling: &Self::SamplingState,
        input: Option<PendingTextInput<&TextPromptFor<B>, &<B as TextGenerationBackend>::Token>>,
        predictions: u64,
    ) -> Option<u64> {
        if predictions == 0 {
            return Some(0);
        }
        let input = input?;
        let growth = if let Some(refinement) = &sampling.refinement {
            refinement.continuation_context(predictions)?
        } else {
            predictions - 1
        };
        let prompt = match &input {
            PendingTextInput::Prefill(prompt) => Some(*prompt),
            _ => None,
        };
        let inputs = match pending_geometry::<B>(input)? {
            SnapshotPendingGeometry::Decode if sampling.refinement.is_some() => Some(growth),
            SnapshotPendingGeometry::Decode => Some(predictions),
            SnapshotPendingGeometry::PrefillCursor(positions)
            | SnapshotPendingGeometry::TextPrompt(positions) => positions.checked_add(growth),
            SnapshotPendingGeometry::PreparedPrompt => runtime
                .session()
                .resources()
                .model
                .as_ref()
                .prepared_input_positions(prompt?.borrowed())?
                .checked_add(growth),
            SnapshotPendingGeometry::Continue => sampling
                .refinement
                .as_ref()
                .map(|_| growth)
                .or_else(|| sampling.shared.as_ref().map(|_| growth)),
        }?;
        inputs.checked_add(
            sampling
                .shared
                .as_ref()
                .map_or(0, |shared| shared.proposal_capacity()),
        )
    }

    fn decoding_settlement_input_tokens(
        _: &ModelRuntime<B, Driver<R, N>>,
        _: &Self::SamplingState,
        _: Option<PendingTextInput<&TextPromptFor<B>, &<B as TextGenerationBackend>::Token>>,
    ) -> Option<u64> {
        Some(0)
    }

    fn estimate_sampling_growth(
        runtime: &ModelRuntime<B, Driver<R, N>>,
        state: &Self::SamplingState,
        predictions: u64,
    ) -> Result<Option<u64>, BackendFailure> {
        let mut growth = state
            .refinement
            .as_ref()
            .map_or(Some(0), LocalRefinementBlock::estimate_growth);
        if let Some(shared) = state.shared.as_ref().filter(|_| predictions > 0) {
            let session = runtime.session();
            session.ensure_no_submission_in_flight()?;
            let native = Driver::<R, N>::estimate_native_text_state(runtime, None)?;
            let model = &session.resources().model;
            growth = predictions
                .checked_add(shared.proposal_capacity())
                .and_then(|inputs| {
                    let geometry = model.as_ref().block_refinement_geometry()?;
                    let verification =
                        shared_verification_growth(model.as_ref().control_state(), native, inputs)?;
                    let canvas = u64::from(geometry.canvas_length.get());
                    let vocab = u64::from(geometry.vocabulary_size.get());
                    let scores = canvas
                        .max(shared.proposal_capacity().checked_add(1)?)
                        .checked_mul(vocab)?
                        .checked_mul(8)?;
                    verification
                        .checked_add(scores)?
                        .checked_add(canvas.checked_mul(128)?)?
                        .checked_add(4096)
                });
        }
        Ok(predictions.checked_mul(4).and_then(|bytes| {
            if predictions == 0 {
                return Some(bytes);
            }
            bytes.checked_add(growth?)
        }))
    }

    fn estimate_pending_input(
        _: &ModelRuntime<B, Driver<R, N>>,
        input: Option<PendingTextInput<&TextPromptFor<B>, &<B as TextGenerationBackend>::Token>>,
    ) -> Result<Option<SnapshotEstimate>, BackendFailure> {
        estimate_pending_input::<B>(input)
    }

    fn copy_pending_input(
        runtime: &mut ModelRuntime<B, Driver<R, N>>,
        input: Option<PendingTextInput<&TextPromptFor<B>, &<B as TextGenerationBackend>::Token>>,
    ) -> Result<
        Option<PendingTextInput<TextPromptFor<B>, <B as TextGenerationBackend>::Token>>,
        BackendFailure,
    > {
        let (backend, session) = runtime.parts_mut();
        session.with_model_transaction(|_| Ok(copy_pending_input::<B>(backend, input)))?
    }

    #[cfg(feature = "tensor-capture")]
    fn capture_run(state: &CausalRequest<B>) -> Option<&crate::capture::CaptureSession> {
        state.capture.as_ref()
    }

    #[cfg(feature = "tensor-capture")]
    fn capture_run_mut(
        state: &mut CausalRequest<B>,
    ) -> Option<&mut crate::capture::CaptureSession> {
        state.capture.as_mut()
    }

    #[cfg(feature = "tensor-capture")]
    fn estimate_child_capture(
        _: &ModelRuntime<B, Driver<R, N>>,
        shape: &[u64],
        selection: &eredu_core::capture::CaptureSelection,
        slice: &eredu_core::capture::ResolvedCaptureSlice,
    ) -> Result<eredu_core::capture::CaptureUsage, eredu_core::capture::CaptureError> {
        N::estimate_local_capture(shape, selection, slice)
    }

    #[cfg(feature = "tensor-capture")]
    fn child_intervention_estimator(
        runtime: &ModelRuntime<B, Driver<R, N>>,
    ) -> Result<
        std::sync::Arc<dyn eredu_core::intervention::InterventionEstimator>,
        eredu_core::capture::CaptureError,
    > {
        runtime
            .session()
            .native()
            .local_intervention()
            .map(|(_, estimator)| estimator)
            .ok_or_else(super::capture::no_local_intervention)
    }
}

impl<B, R, N> TextSamplingControlBackend<Driver<R, N>> for B
where
    B: LocalCausalBackend,
    N: TextSessionMechanisms<B, Resources = R> + TextTransactionMechanisms<Error = BackendFailure>,
    R: DerefMut<Target = TextSessionResourcesFor<B, N>> + 'static,
    Driver<R, N>: TextOperationDriver<B, TextGenerationState = CausalRequest<B>>,
{
    fn sampling_control_facts(state: &CausalRequest<B>) -> SamplingStateFacts {
        SamplingStateFacts {
            temperature: state.sampling.temperature,
            requires_positive_temperature: state.sampling.sampler.requires_positive_temperature(),
            has_rng: state.sampling.random.is_some(),
        }
    }

    fn install_sampling_override(
        _: &mut ModelRuntime<B, Driver<R, N>>,
        state: &mut CausalRequest<B>,
        request: ValidatedSamplingOverride,
    ) -> Result<(), BackendFailure> {
        if state.pending.is_some() {
            return Err(invalid_sample_state(
                "cannot replace sampling with a pending commitment",
            ));
        }
        if let Some(seed) = request.reseed() {
            state.sampling.random = Some(HostRandomState::with_seed(seed));
            if let Some(refinement) = &mut state.sampling.refinement {
                refinement.replace_random(seed);
            }
            if let Some(shared) = &mut state.sampling.shared {
                shared.reseed(seed);
            }
        }
        state.sampling.temperature = request.temperature();
        Ok(())
    }
}

//! Local text generation using shared host sampling and optional masked refinement.

use eredu_core::sampling::host_random::HostRandomState;
use std::ops::DerefMut;

mod refinement;
mod shared;
use refinement::LocalRefinementBlock;

use crate::{
    generation::{host_sampling::HostSample, TextSampler},
    speculative::autoregressive::{LocalCausalBackend, LocalCausalSession},
};
use eredu_core::{
    backend::{
        CausalTextGeneration, CausalTextSession, NativeTextValues, TextOperationDriver,
        TextSamplingNative,
    },
    capture::CaptureTokenDomain,
    execution_control::ControlSupport,
    BackendFailure, BackendFailureKind, ModelRuntime, Submission, TextGenerationBackend,
    TextGenerationConfig, TextOperationError, TextOperationKind,
};

use super::{
    prepare_plain_text_prompt, PlainTextPrefill, TextDecodeInput, TextExecutable, TextPromptFor,
    TextSessionDriver, TextSessionMechanisms, TextSessionResourcesFor,
};

/// Request state for local host generation on the prepared driver.
pub struct CausalRequest<B: LocalCausalBackend> {
    sampling: HostCausalSamplingState<B>,
    pending: Option<PendingSample>,
    readout: refinement::Readout,
    #[cfg(feature = "tensor-capture")]
    capture: Option<crate::capture::CaptureSession>,
}

/// Exact host sampler and optional iterative state at a completed physical boundary.
pub struct HostCausalSamplingState<B: LocalCausalBackend> {
    next_prediction: u64,
    sampler: TextSampler,
    temperature: f32,
    random: Option<HostRandomState>,
    refinement: Option<LocalRefinementBlock>,
    refinement_drafts: bool,
    shared: Option<shared::LocalSharedState<B>>,
}

impl<B: LocalCausalBackend> CausalRequest<B> {
    fn require_committed(&self) -> Result<(), TextOperationError<BackendFailure>> {
        if self.pending.is_some() {
            return Err(TextOperationError::Backend(invalid_sample_state(
                "the previous host sample is awaiting commitment",
            )));
        }
        Ok(())
    }
}

impl<B: LocalCausalBackend> Clone for HostCausalSamplingState<B> {
    fn clone(&self) -> Self {
        Self {
            next_prediction: self.next_prediction,
            sampler: self.sampler.clone(),
            temperature: self.temperature,
            random: self.random.clone(),
            refinement: self.refinement.clone(),
            refinement_drafts: self.refinement_drafts,
            shared: self.shared.clone(),
        }
    }
}

struct PendingSample {
    sample: HostSample,
    random: Option<HostRandomState>,
}

type Output<B, N> = eredu_core::backend::SessionSubmission<
    B,
    TextSessionDriver<<N as TextSessionMechanisms<B>>::Resources, N, CausalTextGeneration>,
>;

impl<B: LocalCausalBackend> TextOperationDriver<B> for LocalCausalSession<B> {
    type Prompt = TextPromptFor<B>;
    type TextGenerationState = CausalRequest<B>;

    fn plain_text_prefill_tokens(prompt: &Self::Prompt) -> Option<usize> {
        PlainTextPrefill::<B>::plain_token_count(prompt)
    }

    fn prepare_text_prompt(backend: &B, ids: Vec<u32>) -> Result<Self::Prompt, BackendFailure> {
        prepare_plain_text_prompt(backend, ids)
    }

    fn start_text_generation(
        &self,
        _: &B,
        config: TextGenerationConfig,
    ) -> Result<CausalRequest<B>, BackendFailure> {
        let temperature = config.sampling().temperature;
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(BackendFailure::new(
                BackendFailureKind::InvalidInput,
                std::io::Error::other("sampling temperature must be finite and nonnegative"),
            ));
        }
        let sampler = TextSampler::from_config(&config).map_err(BackendFailure::from_error)?;
        if temperature == 0.0 && sampler.requires_positive_temperature() {
            return Err(BackendFailure::from_error(
                eredu_core::GenerationError::InvalidMirostatTemperature(temperature),
            ));
        }
        Ok(CausalRequest {
            sampling: HostCausalSamplingState {
                next_prediction: 0,
                sampler,
                temperature,
                random: (temperature > 0.0).then(|| HostRandomState::with_seed(config.seed())),
                refinement: self.start_local_refinement::<B>(&config)?,
                refinement_drafts: config.draft_events(),
                shared: self.start_local_shared::<B>(&config)?,
            },
            pending: None,
            readout: Default::default(),
            #[cfg(feature = "tensor-capture")]
            capture: None,
        })
    }

    fn sample_causal_submission(
        &self,
        backend: &B,
        submission: Output<B, B>,
        filter: &eredu_core::TokenFilter,
        state: &mut CausalRequest<B>,
    ) -> Result<
        Submission<<B as TextGenerationBackend>::Token, B::TextCompletion>,
        TextOperationError<BackendFailure>,
    > {
        state.require_committed()?;
        let mut random = state.sampling.random.clone();
        let mut selected = None;
        let submission = super::causal::sample_causal_with::<B, B>(
            &self.resources().model,
            backend,
            submission,
            |scores, _| {
                let sample = HostSample::select(
                    &state.sampling.sampler,
                    scores,
                    filter,
                    state.sampling.temperature,
                    random.as_mut(),
                )?;
                let token = sample.token;
                selected = Some(sample);
                Ok(token)
            },
        )
        .map_err(TextOperationError::Backend)?;
        let sample = selected.ok_or_else(|| {
            TextOperationError::Backend(invalid_sample_state(
                "host sample was not produced locally",
            ))
        })?;
        state.pending = Some(PendingSample { sample, random });
        Ok(submission)
    }

    fn expert_cache_telemetry(
        &self,
    ) -> Result<Option<eredu_core::ExpertCacheTelemetry>, BackendFailure> {
        self.text_expert_cache_telemetry()
    }

    fn text_prefill_chunking_support(&self) -> Result<(), &'static str> {
        self.prefill_chunking_support::<B>()
    }

    fn prefill_text_prefix(
        runtime: &mut ModelRuntime<B, Self>,
        prompt: &mut Self::Prompt,
        max_tokens: std::num::NonZeroUsize,
        state: &mut CausalRequest<B>,
    ) -> Result<bool, BackendFailure> {
        if runtime.session().text_prefill_chunking_support().is_err()
            || !PlainTextPrefill::<B>::plain_token_count(prompt)
                .is_some_and(|count| count > max_tokens.get())
        {
            return Ok(false);
        }
        #[cfg(feature = "tensor-capture")]
        if state.capture.is_some() {
            return Err(BackendFailure::from_error(
                eredu_core::capture::CaptureError::Unsupported(
                    "local observed prefill requires one complete invocation".into(),
                ),
            ));
        }
        if state.pending.is_some() {
            return Err(invalid_sample_state(
                "cannot advance prefill with pending commitment",
            ));
        }
        let (prefix, suffix) = prompt.split_plain_prompt(runtime.backend(), max_tokens)?;
        runtime.prefill(prefix)?.wait()?;
        *prompt = suffix;
        Ok(true)
    }

    fn commit_text_token(
        runtime: &mut ModelRuntime<B, Self>,
        token: u32,
        _: eredu_core::refinement::GenerationPosition,
        state: &mut CausalRequest<B>,
    ) -> Result<(), BackendFailure> {
        if state.sampling.refinement.is_some() {
            super::causal::advance_sample(
                &mut state.sampling.next_prediction,
                #[cfg(feature = "tensor-capture")]
                state.capture.as_mut(),
                #[cfg(not(feature = "tensor-capture"))]
                None,
            )
            .map_err(BackendFailure::from_error)?;
            return Ok(());
        }
        let pending = state
            .pending
            .take()
            .filter(|pending| pending.sample.token == token)
            .ok_or_else(|| {
                invalid_sample_state("host sample does not match the committed token")
            })?;
        let next = state
            .sampling
            .next_prediction
            .checked_add(1)
            .ok_or_else(|| invalid_sample_state("prediction position overflowed"))?;
        if let Some(shared) = &mut state.sampling.shared {
            Self::commit_local_shared(runtime, shared, token)?;
        }
        pending.sample.commit(&mut state.sampling.sampler)?;
        state.sampling.random = pending.random;
        state.sampling.next_prediction = next;
        Ok(())
    }

    fn text_execution_control_support(&self, backend: &B) -> ControlSupport {
        match self
            .native()
            .validate_native_target(&self.resources().retained, backend)
        {
            Ok(()) => ControlSupport::Supported,
            Err(error) => ControlSupport::Unsupported {
                reason: error.to_string(),
            },
        }
    }

    fn refinement_canvas_length(&self) -> Option<std::num::NonZeroU32> {
        self.local_refinement_policy::<B>()
            .map(|policy| policy.geometry.canvas_length)
    }

    fn refinement_vocabulary_size(&self) -> Option<std::num::NonZeroU32> {
        self.local_refinement_policy::<B>()
            .map(|policy| policy.geometry.vocabulary_size)
    }

    fn supports_block_refinement(&self) -> bool {
        self.local_refinement_policy::<B>().is_some()
    }

    fn decoding_capabilities(&self) -> eredu_core::decoding::DecodingCapabilities {
        let policy = self.local_refinement_policy::<B>();
        let shared = self.local_shared_policy::<B>();
        eredu_core::decoding::DecodingCapabilities {
            speculation: shared.is_some(),
            draft_adapter: shared.is_some_and(|policy| policy.adapter_available()),
            max_draft_tokens: shared.map(|policy| policy.proposal_capacity()),
            autoregressive: self
                .resources()
                .model
                .as_ref()
                .capability_estimate()
                .ordinary_refinement()
                .is_none_or(|policy| policy.causal_sampling),
            diffusion: policy.is_some(),
            entropy_feedback: policy.is_some_and(|retained| {
                matches!(
                    retained.policy,
                    eredu_core::refinement::RefinementPolicy::EntropyFeedback(_)
                )
            }),
            masked_unmasking: policy.is_some_and(|retained| {
                matches!(
                    retained.policy,
                    eredu_core::refinement::RefinementPolicy::MaskedUnmasking { .. }
                )
            }),
            mask_token_id: policy.and_then(|policy| match policy.policy {
                eredu_core::refinement::RefinementPolicy::MaskedUnmasking {
                    mask_token_id, ..
                } => Some(mask_token_id),
                _ => None,
            }),
            ..Default::default()
        }
    }

    fn supports_refinement_policy(&self, policy: eredu_core::refinement::RefinementPolicy) -> bool {
        self.local_refinement_policy::<B>()
            .is_some_and(|retained| retained.prepare(Some(policy), 0.0).is_ok())
    }

    fn generation_score_kind(&self) -> eredu_core::refinement::PredictionScoreKind {
        use eredu_core::refinement::PredictionScoreKind;
        if self.local_shared_policy::<B>().is_none()
            && self
                .local_refinement_policy::<B>()
                .is_some_and(|policy| !policy.causal_sampling)
        {
            PredictionScoreKind::RefinementPosition
        } else {
            PredictionScoreKind::CausalConditional
        }
    }

    fn configure_refinement_readout(
        &self,
        state: &mut CausalRequest<B>,
        selections: Vec<eredu_core::refinement::RefinementScoreSelection>,
    ) -> Result<(), eredu_core::refinement::RefinementReadoutError> {
        state.readout.configure(
            self.local_refinement_policy::<B>()
                .map(|policy| policy.geometry),
            selections,
        )
    }

    fn take_refinement_readout(
        state: &mut CausalRequest<B>,
    ) -> Option<eredu_core::refinement::RefinementScoreRows> {
        state.readout.take()
    }

    fn submit_specialized_text_operation(
        runtime: &mut ModelRuntime<B, Self>,
        input: eredu_core::PendingTextInput<Self::Prompt, <B as TextGenerationBackend>::Token>,
        decision: Option<&eredu_core::TokenSamplingDecision<'_>>,
        state: &mut CausalRequest<B>,
        remaining: Option<usize>,
    ) -> Result<
        Submission<
            eredu_core::TextOperationOutput<Self::Prompt, <B as TextGenerationBackend>::Token>,
            B::TextCompletion,
        >,
        TextOperationError<BackendFailure>,
    > {
        if state.sampling.shared.is_some() {
            state.require_committed()?;
            return Self::advance_local_shared(runtime, input, decision, state, remaining)
                .map_err(TextOperationError::Backend);
        }
        Self::advance_local_refinement(runtime, input, decision, state)
            .map_err(TextOperationError::Backend)
    }

    #[cfg(feature = "tensor-capture")]
    fn text_capture_invocation_bounds(
        &self,
        request: eredu_core::capture::CaptureRequestShape,
    ) -> Result<
        Option<eredu_core::capture::CaptureInvocationBounds>,
        eredu_core::capture::CaptureError,
    > {
        self.text_capture_invocation_bounds(request)
    }

    #[cfg(feature = "tensor-capture")]
    fn intervention_discovery(
        &self,
    ) -> Result<eredu_core::intervention::InterventionDiscovery, eredu_core::capture::CaptureError>
    {
        self.local_intervention_discovery::<B>()
    }

    #[cfg(feature = "tensor-capture")]
    fn validate_text_interventions(
        &self,
        capture: &eredu_core::capture::AdmittedCapturePlan,
        plan: &eredu_core::intervention::AdmittedInterventionPlan,
    ) -> Result<(), eredu_core::capture::CaptureError> {
        self.validate_local_interventions::<B>(capture, plan)
    }

    #[cfg(feature = "tensor-capture")]
    fn configure_text_interventions(
        &self,
        state: &mut CausalRequest<B>,
        capture: eredu_core::capture::AdmittedCapturePlan,
        plan: eredu_core::intervention::AdmittedInterventionPlan,
    ) -> Result<(), eredu_core::capture::CaptureError> {
        self.install_local_capture::<B>(&mut state.capture, capture, Some(plan))
    }

    #[cfg(feature = "tensor-capture")]
    fn capture_declarations(
        &self,
    ) -> Result<eredu_core::capture::CaptureDeclarations, eredu_core::capture::CaptureError> {
        self.local_capture_declarations()
    }

    #[cfg(feature = "tensor-capture")]
    fn capture_discovery(
        &self,
    ) -> Result<eredu_core::capture::CaptureDiscovery, eredu_core::capture::CaptureError> {
        self.local_capture_discovery()
    }

    #[cfg(feature = "tensor-capture")]
    fn configure_text_capture(
        &self,
        state: &mut CausalRequest<B>,
        plan: eredu_core::capture::AdmittedCapturePlan,
    ) -> Result<(), eredu_core::capture::CaptureError> {
        self.install_local_capture::<B>(&mut state.capture, plan, None)
    }

    #[cfg(feature = "tensor-capture")]
    fn validate_text_capture(
        &self,
        plan: &eredu_core::capture::AdmittedCapturePlan,
    ) -> Result<(), eredu_core::capture::CaptureError> {
        self.validate_local_capture::<B>(plan)
    }

    #[cfg(feature = "tensor-capture")]
    fn take_text_capture(
        state: &mut CausalRequest<B>,
    ) -> Option<eredu_core::capture::CapturedStep> {
        state
            .capture
            .as_mut()
            .and_then(crate::capture::CaptureSession::take_step)
    }

    fn text_operation_kind(
        _: &ModelRuntime<B, Self>,
        input: &eredu_core::PendingTextInput<Self::Prompt, <B as TextGenerationBackend>::Token>,
        state: &CausalRequest<B>,
    ) -> TextOperationKind {
        if let Some(shared) = &state.sampling.shared {
            return shared.operation_kind(input);
        }
        state.sampling.refinement.as_ref().map_or(
            TextOperationKind::CausalDecision,
            LocalRefinementBlock::operation_kind,
        )
    }
}

fn invalid_sample_state(message: &'static str) -> BackendFailure {
    BackendFailure::new(
        BackendFailureKind::InvalidSession,
        std::io::Error::other(message),
    )
}

impl<B: LocalCausalBackend>
    CausalTextSession<B, TextPromptFor<B>, <B as TextGenerationBackend>::Token, CausalRequest<B>>
    for LocalCausalSession<B>
{
    fn submit_causal_prefill(
        &mut self,
        backend: &B,
        prompt: TextPromptFor<B>,
        domain: Option<CaptureTokenDomain<'_>>,
        state: &mut CausalRequest<B>,
    ) -> Result<Output<B, B>, TextOperationError<BackendFailure>> {
        state.require_committed()?;
        #[cfg(feature = "tensor-capture")]
        if let Some(capture) = state.capture.as_mut() {
            return self.submit_local_capture_prefill(backend, prompt, domain, capture);
        }
        let _ = domain;
        self.advance_prefill(backend, Ok(prompt))
            .map_err(TextOperationError::Backend)
    }

    fn submit_causal_decode(
        &mut self,
        backend: &B,
        token: <B as TextGenerationBackend>::Token,
        domain: Option<CaptureTokenDomain<'_>>,
        state: &mut CausalRequest<B>,
    ) -> Result<Output<B, B>, TextOperationError<BackendFailure>> {
        state.require_committed()?;
        let input = TextDecodeInput::Sampled(Ok(B::sampled_token_value(token)));
        #[cfg(feature = "tensor-capture")]
        if let Some(capture) = state.capture.as_mut() {
            return self.submit_local_capture_decode(
                backend,
                input,
                domain,
                state.sampling.next_prediction,
                capture,
            );
        }
        let _ = domain;
        self.advance_decode(backend, input)
            .map_err(TextOperationError::Backend)
    }
}

#[cfg(feature = "tensor-capture")]
mod capture;
mod snapshot;

//! Portable dispatch of admitted specialized text operations.

use crate::text_session_driver::TextSessionResourcesFor;
use eredu_core::refinement::RefinementBackend;

use std::{
    num::{NonZeroU32, NonZeroUsize},
    sync::Arc,
};

mod block;
use block::BlockTextDriver;
mod shared;
use shared::{SharedTextDriver, SharedTextProposal};

use eredu_core::{
    BackendSession, ModelRuntime, PendingTextInput, Submission, TextGenerationBackend,
    TextOperationError, TextOperationOutput, TextOperationPolicyError, TokenSamplingDecision,
};

use super::{
    causal::{CausalPrefillDriver, CausalTextMechanisms},
    TextAdvanceRoute, TextExecutable, TextModeState, TextSessionDriver,
};
use crate::{capture::CaptureSession, speculative::shared_backbone::SharedBackboneMechanisms};
use eredu_core::{
    backend::TextOperationDriver,
    capture::{
        AdmittedCapturePlan, CaptureDeclarations, CaptureDiscovery, CaptureError,
        CaptureInvocationBounds, CaptureRequestShape, CapturedStep,
    },
    decoding::{DecodingCoordination, DecodingMode, DecodingModeError},
    intervention::{AdmittedInterventionPlan, InterventionDiscovery},
    BackendFailure, ExpertCacheTelemetry, GenerationError, ResidencyTelemetry, TextOperationKind,
};

type Input<B> = PendingTextInput<
    crate::text_session_driver::TextPromptFor<B>,
    <B as TextGenerationBackend>::Token,
>;
type State<B> = super::TextRequestFor<B>;
/// Published outcome and exact completion of a specialized text operation.
pub type SpecializedTextSubmission<B> = Submission<
    TextOperationOutput<
        crate::text_session_driver::TextPromptFor<B>,
        <B as TextGenerationBackend>::Token,
    >,
    <B as TextGenerationBackend>::TextCompletion,
>;

/// Exposes the retained mode state to portable operation selection.
pub trait TextModeBinding:
    TextGenerationBackend
    + eredu_core::sampling::SeededRandomKey<
        Key = <<Self as TextModeBinding>::Refinement as RefinementBackend>::RandomState,
    >
{
    /// Native tensor operations for a refinement block.
    type Refinement: RefinementBackend<RandomState: Clone>;
    /// Native operations for shared-backbone verification.
    type Shared: SharedBackboneMechanisms<
            ProposalRandomness = <Self::Refinement as RefinementBackend>::RandomState,
        > + 'static;
}

impl<B> TextModeBinding for B
where
    B: eredu_core::refinement::TextRefinementNative<
            BlockSampling: RefinementBackend<RandomState: Clone>,
        > + eredu_core::session_authority::SpeculativeTargetNative<
            TargetOperators: SharedBackboneMechanisms<
                ProposalRandomness = <B::BlockSampling as RefinementBackend>::RandomState,
            > + 'static,
        > + eredu_core::sampling::SeededRandomKey<
            Key = <B::BlockSampling as RefinementBackend>::RandomState,
        >,
{
    type Refinement = B::BlockSampling;
    type Shared = B::TargetOperators;
}

impl<B, R: 'static, N, T> TextOperationDriver<B> for TextSessionDriver<R, N>
where
    B: TextModeBinding<InputTensor = T> + eredu_core::input::PlainTextTokens<T>,
    T: Clone + 'static,
    Self: SharedTextDriver<B>
        + BlockTextDriver<B>
        + CausalPrefillDriver<B>
        + BackendSession<B, Output = N::Output, Completion = N::Completion>,
    B::Error: From<GenerationError> + From<eredu_core::BackendError> + From<BackendFailure>,
    N: CausalTextMechanisms<
        B,
        Resources = R,
        Token = B::Token,
        SamplingCompletion = B::TextCompletion,
        Model: super::InstalledTextState,
    >,
    R: std::ops::Deref<Target = TextSessionResourcesFor<B, N>>,
{
    type Prompt = crate::text_session_driver::TextPromptFor<B>;
    type TextGenerationState = super::TextRequestFor<B>;

    fn prepare_text_prompt(
        backend: &B,
        ids: Vec<u32>,
    ) -> Result<crate::text_session_driver::TextPromptFor<B>, B::Error> {
        super::prepare_plain_text_prompt(backend, ids)
    }

    fn start_text_generation(
        &self,
        _: &B,
        config: eredu_core::TextGenerationConfig,
    ) -> Result<super::TextRequestFor<B>, B::Error> {
        let seed = (config.sampling().temperature != 0.0).then(|| config.seed());
        let prng = seed.map(B::seeded_random).transpose()?;
        let sampler = crate::generation::TextSampler::from_config(&config).map_err(|error| {
            eredu_core::BackendError::Execution {
                session: "text-generation".into(),
                operation: "configure Mirostat V2".into(),
                message: error.to_string(),
            }
        })?;
        Ok(super::TextRequestState::start(config, sampler, prng))
    }

    fn sample_causal_submission(
        &self,
        backend: &B,
        submission: eredu_core::backend::SessionSubmission<B, Self>,
        filter: &eredu_core::TokenFilter,
        state: &mut super::TextRequestFor<B>,
    ) -> Result<Submission<B::Token, B::TextCompletion>, TextOperationError<B::Error>> {
        let sampled = super::causal::sample_causal_scores::<B, N>(
            &self.resources().model,
            backend,
            submission,
            filter,
            &mut state.sampling.sampler,
            state.sampling.prng.as_mut(),
            state.sampling.temperature,
        )
        .map_err(TextOperationError::Backend)?;
        let state = super::causal::causal_state::<B>(state);
        super::causal::advance_sample(state.prediction, state.capture).map_err(|error| {
            TextOperationError::Backend(BackendFailure::from_error(error).into())
        })?;
        *state.committed_token_cached = false;
        Ok(sampled)
    }

    fn text_execution_control_support(
        &self,
        backend: &B,
    ) -> eredu_core::execution_control::ControlSupport {
        if let Err(error) = self
            .native()
            .validate_native_target(&self.resources().retained, backend)
        {
            return eredu_core::execution_control::ControlSupport::Unsupported {
                reason: error.to_string(),
            };
        }
        super::erased_state::state_support(&self.resources().model)
    }

    fn text_sampling_control_support(
        &self,
        backend: &B,
    ) -> eredu_core::execution_control::ControlSupport {
        self.text_execution_control_support(backend)
    }

    fn active_parameter_overlay(&self) -> Option<&str> {
        self.parameter_overlay_identity()
    }

    fn configure_refinement_readout(
        &self,
        state: &mut Self::TextGenerationState,
        selections: Vec<eredu_core::refinement::RefinementScoreSelection>,
    ) -> Result<(), eredu_core::refinement::RefinementReadoutError> {
        use eredu_core::refinement::RefinementReadoutError;
        let geometry = self
            .resources()
            .model
            .as_ref()
            .block_refinement_geometry()
            .ok_or(RefinementReadoutError::Unsupported)?;
        if selections.is_empty()
            || selections.iter().any(|selection| {
                selection.position >= geometry.canvas_length.get()
                    || selection.token_ids.is_empty()
                    || selection
                        .token_ids
                        .iter()
                        .any(|&id| id >= geometry.vocabulary_size.get())
            })
        {
            return Err(RefinementReadoutError::Geometry);
        }
        state.readout_selection = Some(selections);
        state.readout_rows = None;
        Ok(())
    }

    fn take_refinement_readout(
        state: &mut Self::TextGenerationState,
    ) -> Option<eredu_core::refinement::RefinementScoreRows> {
        state.readout_selection = None;
        state.readout_rows.take()
    }

    fn refinement_vocabulary_size(&self) -> Option<NonZeroU32> {
        self.resources()
            .model
            .as_ref()
            .block_refinement_geometry()
            .map(|geometry| geometry.vocabulary_size)
    }

    fn refinement_canvas_length(&self) -> Option<NonZeroU32> {
        self.resources()
            .model
            .as_ref()
            .block_refinement_geometry()
            .map(|geometry| geometry.canvas_length)
    }

    fn supports_block_refinement(&self) -> bool {
        self.resources()
            .model
            .as_ref()
            .block_refinement_geometry()
            .is_some()
    }

    fn decoding_capabilities(&self) -> eredu_core::decoding::DecodingCapabilities {
        use eredu_core::refinement::RefinementPolicy;
        let estimate = self.resources().model.as_ref().capability_estimate();
        let shared = estimate.shared_backbone_draft();
        let ordinary = estimate
            .ordinary_refinement()
            .filter(|_| self.supports_block_refinement());
        eredu_core::decoding::DecodingCapabilities {
            speculation_limits: None,
            autoregressive: ordinary.is_none_or(|policy| policy.causal_sampling),
            diffusion: ordinary.is_some(),
            entropy_feedback: ordinary.is_some_and(|policy| {
                matches!(policy.policy, RefinementPolicy::EntropyFeedback(_))
            }),
            masked_unmasking: ordinary.is_some_and(|policy| {
                matches!(policy.policy, RefinementPolicy::MaskedUnmasking { .. })
            }),
            mask_token_id: ordinary.and_then(|policy| match policy.policy {
                RefinementPolicy::MaskedUnmasking { mask_token_id, .. } => Some(mask_token_id),
                _ => None,
            }),
            mode_changes: shared.is_some(),
            speculation: shared.is_some(),
            draft_adapter: shared.is_some_and(|policy| policy.adapter_available()),
            max_draft_tokens: shared.map(|policy| policy.proposal_capacity()),
            completion_cancellation: N::completion_cancellation_mode(),
        }
    }

    fn supports_refinement_policy(&self, policy: eredu_core::refinement::RefinementPolicy) -> bool {
        use eredu_core::refinement::RefinementPolicy;
        let capability = self.decoding_capabilities();
        match policy {
            RefinementPolicy::EntropyFeedback(_) => capability.entropy_feedback,
            RefinementPolicy::MaskedUnmasking { mask_token_id, .. } => {
                capability.masked_unmasking && capability.mask_token_id == Some(mask_token_id)
            }
        }
    }

    fn generation_score_kind(&self) -> eredu_core::refinement::PredictionScoreKind {
        use eredu_core::refinement::PredictionScoreKind;
        let estimate = self.resources().model.as_ref().capability_estimate();
        if estimate.shared_backbone_draft().is_some() || !self.supports_block_refinement() {
            PredictionScoreKind::CausalConditional
        } else {
            PredictionScoreKind::RefinementPosition
        }
    }

    fn text_prefill_chunking_support(&self) -> Result<(), &'static str> {
        self.prefill_chunking_support::<B>()
    }

    fn residency_telemetry(&self) -> Result<Option<ResidencyTelemetry>, BackendFailure> {
        N::residency_report(self.resources())
            .map(|report| report.as_ref().map(crate::residency_telemetry))
            .map_err(|error| {
                BackendFailure::from_error(error).with_operation("residency telemetry")
            })
    }

    fn expert_cache_telemetry(&self) -> Result<Option<ExpertCacheTelemetry>, BackendFailure> {
        self.parameter_bank_report()
            .map(|report| report.map(super::operations::expert_cache_telemetry))
            .map_err(|error| {
                BackendFailure::from_error(error).with_operation("expert-cache telemetry")
            })
    }

    #[cfg(feature = "tensor-capture")]
    fn intervention_discovery(&self) -> Result<InterventionDiscovery, CaptureError> {
        self.text_intervention_discovery()
    }

    #[cfg(feature = "tensor-capture")]
    fn validate_text_interventions(
        &self,
        capture: &AdmittedCapturePlan,
        plan: &AdmittedInterventionPlan,
    ) -> Result<(), CaptureError> {
        TextSessionDriver::validate_text_interventions(
            self,
            capture,
            plan,
            &N::intervention_estimator(),
        )
    }

    #[cfg(feature = "tensor-capture")]
    fn configure_text_interventions(
        &self,
        state: &mut State<B>,
        capture: AdmittedCapturePlan,
        plan: AdmittedInterventionPlan,
    ) -> Result<(), CaptureError> {
        self.configure_text_capture(
            &mut state.capture,
            capture,
            Some((plan, Arc::new(N::intervention_estimator()))),
        )
    }

    #[cfg(feature = "tensor-capture")]
    fn capture_declarations(&self) -> Result<CaptureDeclarations, CaptureError> {
        self.text_capture_declarations()
    }

    #[cfg(feature = "tensor-capture")]
    fn capture_discovery(&self) -> Result<CaptureDiscovery, CaptureError> {
        self.text_capture_discovery()
    }

    #[cfg(feature = "tensor-capture")]
    fn text_capture_invocation_bounds(
        &self,
        request: CaptureRequestShape,
    ) -> Result<Option<CaptureInvocationBounds>, CaptureError> {
        TextSessionDriver::text_capture_invocation_bounds(self, request)
    }

    #[cfg(feature = "tensor-capture")]
    fn configure_text_capture(
        &self,
        state: &mut State<B>,
        plan: AdmittedCapturePlan,
    ) -> Result<(), CaptureError> {
        TextSessionDriver::configure_text_capture(self, &mut state.capture, plan, None)
    }

    #[cfg(feature = "tensor-capture")]
    fn validate_text_capture(&self, plan: &AdmittedCapturePlan) -> Result<(), CaptureError> {
        TextSessionDriver::validate_text_capture(self, plan)
    }

    #[cfg(feature = "tensor-capture")]
    fn take_text_capture(state: &mut State<B>) -> Option<CapturedStep> {
        state.capture.as_mut().and_then(CaptureSession::take_step)
    }

    fn text_operation_kind(
        _: &ModelRuntime<B, Self>,
        input: &Input<B>,
        state: &State<B>,
    ) -> TextOperationKind {
        state
            .sampling
            .decoding
            .advance_route(input)
            .operation_kind()
    }

    fn submit_specialized_text_operation(
        runtime: &mut ModelRuntime<B, Self>,
        input: Input<B>,
        decision: Option<&TokenSamplingDecision<'_>>,
        state: &mut State<B>,
        remaining: Option<usize>,
    ) -> Result<SpecializedTextSubmission<B>, TextOperationError<B::Error>> {
        let route = state.sampling.decoding.advance_route(&input);
        dispatch_specialized_text(route, runtime, input, decision, state, remaining)
    }

    fn prefill_text_prefix(
        runtime: &mut ModelRuntime<B, Self>,
        prompt: &mut crate::text_session_driver::TextPromptFor<B>,
        max_tokens: NonZeroUsize,
        state: &mut State<B>,
    ) -> Result<bool, B::Error> {
        <Self as CausalPrefillDriver<B>>::prefill_prefix(runtime, prompt, max_tokens, state)
    }

    fn complete_text_prefill(
        runtime: &mut ModelRuntime<B, Self>,
        prompt: crate::text_session_driver::TextPromptFor<B>,
        state: &mut State<B>,
    ) -> Result<bool, B::Error> {
        <Self as CausalPrefillDriver<B>>::complete_prefill(runtime, prompt, state)
    }

    fn coordinate_decoding_control(
        runtime: &mut ModelRuntime<B, Self>,
        local: DecodingCoordination,
    ) -> Result<DecodingCoordination, TextOperationError<B::Error>> {
        Self::coordinate_mode(runtime, local)
    }

    fn settle_decoding_mode(
        runtime: &mut ModelRuntime<B, Self>,
        pending: &mut Option<Input<B>>,
        state: &mut State<B>,
        _previous: DecodingMode,
        requested: DecodingMode,
    ) -> Result<bool, TextOperationError<B::Error>> {
        <Self as SharedTextDriver<B>>::settle_mode(runtime, pending, state, requested)
    }

    fn commit_text_token(
        runtime: &mut ModelRuntime<B, Self>,
        token: u32,
        _: eredu_core::refinement::GenerationPosition,
        state: &mut State<B>,
    ) -> Result<(), B::Error> {
        Self::commit(runtime, token, state)
    }
}

/// Dispatches the selected route, checking the required input and controller decision.
fn dispatch_specialized_text<B, S>(
    route: TextAdvanceRoute,
    runtime: &mut ModelRuntime<B, S>,
    input: Input<B>,
    decision: Option<&TokenSamplingDecision<'_>>,
    state: &mut State<B>,
    remaining: Option<usize>,
) -> Result<SpecializedTextSubmission<B>, TextOperationError<B::Error>>
where
    B: TextModeBinding,
    S: SharedTextDriver<B> + BlockTextDriver<B>,
    B::Error: From<GenerationError>,
{
    match (route, input) {
        (TextAdvanceRoute::Block(phase), input) => {
            S::advance_block(runtime, phase, input, decision, state)
        }
        (TextAdvanceRoute::SharedProposal, PendingTextInput::Decode(committed)) => {
            let DecodingMode::SelfSpeculation(options) = state.sampling.decoding.mode else {
                return Err(TextOperationPolicyError::OutcomeMismatch.into());
            };
            let policy =
                runtime
                    .session()
                    .shared_policy()
                    .ok_or(TextOperationPolicyError::DecodingMode(
                        DecodingModeError::Unsupported("shared draft source was not retained"),
                    ))?;
            let draft_options = policy
                .prepare(options, state.sampling.temperature)
                .map_err(TextOperationPolicyError::DecodingMode)?;
            let request = options.request();
            let capacity = request
                .max_draft_tokens
                .get()
                .min(remaining.map_or(usize::MAX, |n| n.saturating_sub(1)));
            let proposal = match NonZeroUsize::new(capacity) {
                Some(capacity) => SharedTextProposal::Verify {
                    capacity,
                    draft_options,
                    completion_wait: request
                        .scheduler
                        .completion_wait()
                        .map_err(B::Error::from)
                        .map_err(TextOperationError::Backend)?,
                },
                None => SharedTextProposal::CausalTail,
            };
            S::propose(runtime, committed, state, proposal)
        }
        (TextAdvanceRoute::SharedTarget, _) => {
            let decision = decision.ok_or(TextOperationPolicyError::OutcomeMismatch)?;
            S::select(runtime, decision, state)
        }
        _ => return Err(TextOperationPolicyError::OutcomeMismatch.into()),
    }
    .map_err(TextOperationError::Backend)
}

//! Controlled speculation shares facade preparation and the ordinary scheduler.

use super::request::PreparedGenerationMode;
use super::{LoadedModel, PreparedChatSpeculativeError, PreparedChatSpeculativeGenerationRequest};
pub use eredu_core::speculative::{
    AdmittedSpeculativeActivations, SpeculativeActivationCapture, SpeculativeActivationDiscovery,
    SpeculativeActivationOrigin, SpeculativeActivationPhase, SpeculativeActivationPlan,
    SpeculativeCaptureBinding, SpeculativeCaptureRole, SpeculativeCaptureScope,
    SpeculativeControlError, SpeculativeInterventionPlan, SpeculativePredictionCapture,
    SpeculativeProposalView, SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
};
use eredu_core::{
    generation::SemanticEvent, SpeculativeGenerationBackend, SpeculativeGenerationOutput,
};
pub use eredu_runtime::memory_forecast::{
    EmbeddedContinuationMemoryPlan, SpeculativeContinuationForecast,
    SpeculativeContinuationMemoryPlan,
};
pub use eredu_runtime::speculative::{
    ControlledSpeculativeActivation, ControlledSpeculativeOptions, ControlledSpeculativeSession,
    ControlledSpeculativeStep, SpeculativeBranchHandle, SpeculativeBranchInfo,
    SpeculativeProposalDisposition, SpeculativeSnapshotHandle, SpeculativeVerificationRecord,
};

/// Preparation or controlled speculative execution failure, with neutral causes.
#[derive(Debug, thiserror::Error)]
pub enum ControlledSpeculativeGenerationError {
    /// Existing speculative admission, tokenizer, constraint or backend failure.
    #[error(transparent)]
    Prepared(#[from] PreparedChatSpeculativeError),
    /// Controlled advancement, snapshot or delivery failure.
    #[error(transparent)]
    Control(#[from] SpeculativeControlError),
}

impl<B: SpeculativeGenerationBackend> LoadedModel<B> {
    /// Internal forward capture and edit support for the actually loaded
    /// speculative executor. Ordinary sampler capture uses a separate report.
    pub fn speculative_activation_discovery(
        &self,
    ) -> Result<SpeculativeActivationDiscovery, eredu_core::capture::CaptureError> {
        B::speculative_activation_discovery(&self.runtime)
    }

    /// Admits capture and optional edits together, bound to the loaded session
    /// and effective parameters. The bounds describe physical forward rows;
    /// prediction schedules remain coordinates in the generated token prefix.
    pub fn prepare_speculative_activations(
        &self,
        plan: SpeculativeActivationPlan,
    ) -> Result<AdmittedSpeculativeActivations, eredu_core::capture::CaptureError> {
        let admitted = plan.admit(&self.speculative_activation_discovery()?)?;
        B::validate_speculative_activations(&self.runtime, &admitted)?;
        Ok(admitted)
    }

    /// Admits bounded speculative prediction captures against this loaded model.
    /// Each raw-logit observation is one vocabulary row, including target prefill;
    /// sequence slices therefore address a single row rather than the full prompt.
    /// MLX currently accepts model.logits selections for both target and draft.
    pub fn prepare_speculative_capture(
        &self,
        settings: super::PreparedChatGenerationSettings,
        plan: eredu_core::capture::CapturePlan,
    ) -> Result<eredu_core::capture::AdmittedCapturePlan, super::PreparedChatError> {
        let (_, maximum) = self.resolve_text_generation_settings(settings)?;
        let (plan, _) = self.admit_capture(
            plan,
            eredu_core::capture::CaptureRequestShape {
                batch: 1,
                prompt_tokens: 1,
                max_predictions: maximum.get() as u64,
            },
        )?;
        B::validate_speculative_capture(&self.runtime, &plan)?;
        Ok(plan)
    }

    /// Genuine mutable points with speculative phase attribution for this session.
    pub fn speculative_intervention_discovery(
        &self,
    ) -> Result<eredu_core::intervention::InterventionDiscovery, eredu_core::capture::CaptureError>
    {
        B::speculative_intervention_discovery(&self.runtime)
    }

    /// Admits a target- or draft-specific edit under the existing capture budget.
    /// Supply the same capture plan in controlled options, even for evidence-only runs.
    pub fn prepare_speculative_intervention(
        &self,
        capture: &eredu_core::capture::AdmittedCapturePlan,
        role: SpeculativeCaptureRole,
        plan: eredu_core::intervention::InterventionPlan,
    ) -> Result<SpeculativeInterventionPlan, eredu_core::capture::CaptureError> {
        let discovery = self.speculative_intervention_discovery()?;
        let plan = plan.admit(&discovery, capture.request(), &self.session_identity)?;
        B::validate_speculative_interventions(&self.runtime, capture, &plan)?;
        Ok(SpeculativeInterventionPlan { role, plan })
    }
}

impl<B> LoadedModel<B>
where
    B: SpeculativeGenerationBackend
        + eredu_runtime::memory_forecast::SpeculativeForecastBackend<B::Drafter>,
{
    /// Runs the shared controlled driver continuously, delivering each bounded
    /// speculative step immediately. Breaking the callback cancels and settles
    /// execution through the same scheduler. Use the scoped controlled API when
    /// aborted invocation evidence must be drained after a failed action.
    pub fn generate_observed_chat_speculative<'a, F, D>(
        &mut self,
        request: PreparedChatSpeculativeGenerationRequest<'a, B, B::Drafter, F>,
        options: ControlledSpeculativeOptions,
        on_step: D,
    ) -> Result<SpeculativeGenerationOutput, ControlledSpeculativeGenerationError>
    where
        F: FnMut(SemanticEvent),
        D: FnMut(ControlledSpeculativeStep) -> std::ops::ControlFlow<()>,
    {
        self.with_controlled_chat_speculative(request, options, continuous_steps(on_step))
    }

    /// Literal-text equivalent of `generate_observed_chat_speculative`, sharing
    /// the ordinary decoding, termination and controlled advancement policy.
    pub fn generate_observed_text_speculative<'a, F, D>(
        &mut self,
        request: PreparedChatSpeculativeGenerationRequest<'a, B, B::Drafter, F>,
        options: ControlledSpeculativeOptions,
        on_step: D,
    ) -> Result<SpeculativeGenerationOutput, ControlledSpeculativeGenerationError>
    where
        F: FnMut(SemanticEvent),
        D: FnMut(ControlledSpeculativeStep) -> std::ops::ControlFlow<()>,
    {
        self.with_controlled_text_speculative(request, options, continuous_steps(on_step))
    }

    /// Lends a controlled speculative chat to an Inspector or worker command loop.
    /// Uses the same drafting, acceptance, stop, decoding and cancellation path as
    /// `generate_prepared_chat_speculative`. No token is generated before `step`.
    /// Pausing means returning to the command loop without calling `step`.
    /// Native resources stay borrowed for the closure; unfinished work is cancelled
    /// and safely settled when it returns. The returned output includes active TTFT.
    pub fn with_controlled_chat_speculative<'a, F, D>(
        &mut self,
        request: PreparedChatSpeculativeGenerationRequest<'a, B, B::Drafter, F>,
        options: ControlledSpeculativeOptions,
        drive: D,
    ) -> Result<SpeculativeGenerationOutput, ControlledSpeculativeGenerationError>
    where
        F: FnMut(SemanticEvent),
        D: FnOnce(&mut dyn ControlledSpeculativeSession) -> Result<(), SpeculativeControlError>,
    {
        self.with_controlled_speculative(request, options, drive, PreparedGenerationMode::Semantic)
    }

    /// Literal-text equivalent, including templates without semantic recognition.
    ///
    /// ```no_run
    /// # use eredu::api::*;
    /// # use eredu::runtime::chat::PreparedChat;
    /// # use eredu_core::{SpeculativeGenerationBackend, SpeculativeDraft, SpeculativeGenerationOutput};
    /// # fn inspect<B: SpeculativeGenerationBackend + eredu_runtime::memory_forecast::SpeculativeForecastBackend<B::Drafter>>(model: &mut LoadedModel<B>, chat: &PreparedChat,
    /// #     drafting: SpeculativeDraft<'_, B::Drafter>) -> Result<SpeculativeGenerationOutput, Box<dyn std::error::Error>> {
    /// let output = model.with_controlled_text_speculative(
    ///     PreparedChatSpeculativeGenerationRequest {
    ///         input: PreparedChatInput::rendered_prompt(chat), drafting,
    ///         settings: Default::default(), options: Default::default(),
    ///         caller_stop_sequences: &[], cancellation: Default::default(),
    ///         on_event: |event| println!("committed: {event:?}"),
    ///     },
    ///     ControlledSpeculativeOptions::default(),
    ///     |session| {
    ///         // A worker may wait for the next Inspector command before each step.
    ///         while let Some(step) = session.step()? {
    ///             println!("draft: {:?}, verification: {:?}", step.drafted, step.verification);
    ///         }
    ///         Ok(())
    ///     },
    /// )?;
    /// # Ok(output)
    /// # }
    /// ```
    pub fn with_controlled_text_speculative<'a, F, D>(
        &mut self,
        request: PreparedChatSpeculativeGenerationRequest<'a, B, B::Drafter, F>,
        options: ControlledSpeculativeOptions,
        drive: D,
    ) -> Result<SpeculativeGenerationOutput, ControlledSpeculativeGenerationError>
    where
        F: FnMut(SemanticEvent),
        D: FnOnce(&mut dyn ControlledSpeculativeSession) -> Result<(), SpeculativeControlError>,
    {
        self.with_controlled_speculative(request, options, drive, PreparedGenerationMode::Text)
    }

    fn with_controlled_speculative<'a, F, D>(
        &mut self,
        request: PreparedChatSpeculativeGenerationRequest<'a, B, B::Drafter, F>,
        options: ControlledSpeculativeOptions,
        drive: D,
        mode: PreparedGenerationMode,
    ) -> Result<SpeculativeGenerationOutput, ControlledSpeculativeGenerationError>
    where
        F: FnMut(SemanticEvent),
        D: FnOnce(&mut dyn ControlledSpeculativeSession) -> Result<(), SpeculativeControlError>,
    {
        let validation = (|| -> Result<(), ControlledSpeculativeGenerationError> {
            if let Some(plan) = options.activations.as_ref() {
                B::validate_speculative_activations(&self.runtime, plan)
                    .map_err(SpeculativeControlError::from)?;
                let (_, maximum) = self
                    .resolve_text_generation_settings(request.settings)
                    .map_err(PreparedChatSpeculativeError::from)?;
                if plan.captures().invocation_bounds().is_none_or(|bounds| {
                    bounds.batch != 1 || bounds.max_predictions < maximum.get() as u64
                }) {
                    return Err(SpeculativeControlError::Capture(
                        eredu_core::capture::CaptureError::Invalid(
                            "internal activation admission does not cover this speculative request"
                                .into(),
                        ),
                    )
                    .into());
                }
            }
            if let Some(plan) = options.capture.as_ref() {
                if plan.request().prompt_tokens != 1 || plan.request().batch != 1 {
                    return Err(SpeculativeControlError::Capture(eredu_core::capture::CaptureError::Invalid(
                    "speculative captures require one-row admission from prepare_speculative_capture".into())).into());
                }
                B::validate_speculative_capture(&self.runtime, plan)
                    .map_err(SpeculativeControlError::from)?;
                let (_, maximum) = self
                    .resolve_text_generation_settings(request.settings)
                    .map_err(PreparedChatSpeculativeError::from)?;
                if plan.request().max_predictions < maximum.get() as u64 {
                    return Err(SpeculativeControlError::Capture(
                    eredu_core::capture::CaptureError::Invalid(
                        "capture admission does not cover the requested speculative token limit"
                            .into(),
                    ),
                )
                .into());
                }
            }
            Ok(())
        })();
        self.runtime.finish_text_preparation(
            eredu_core::run_preparation::TextPreparationStage::Instrumentation,
            validation,
            |error| PreparedChatSpeculativeError::Backend(error).into(),
        )?;
        // Retain selection geometry before the backend lends its mutable execution
        // resources. Unsupported forecasts must not prevent ordinary execution.
        let forecast_profiles = B::speculative_target_memory_profile(&self.runtime)
            .ok()
            .and_then(|target| {
                B::speculative_memory_profile(&self.runtime, &request.drafting)
                    .ok()
                    .flatten()
                    .map(|draft| (target, draft))
            });
        let mut failure = None;
        let driver = eredu_runtime::speculative::DriveControlledSpeculation::new(
            request.options.scheduler,
            options,
            drive,
            &mut failure,
        )
        .with_forecast_profiles(forecast_profiles)
        .with_vocabulary(
            self.tokenizer
                .get_vocab(true)
                .values()
                .copied()
                .max()
                .map_or(0, |id| id as usize + 1),
        )
        .with_intervention_discovery(B::speculative_intervention_discovery(&self.runtime).ok())
        .with_activation_discovery(B::speculative_activation_discovery(&self.runtime).ok());
        let result = self.generate_prepared_speculative_with(request, mode, driver);
        if let Some(error) = failure {
            return Err(error.into());
        }
        result.map_err(Into::into)
    }
}

fn continuous_steps(
    mut on_step: impl FnMut(ControlledSpeculativeStep) -> std::ops::ControlFlow<()>,
) -> impl FnOnce(&mut dyn ControlledSpeculativeSession) -> Result<(), SpeculativeControlError> {
    move |session| {
        while let Some(step) = session.step()? {
            if on_step(step).is_break() {
                session.cancel()?;
                break;
            }
        }
        Ok(())
    }
}

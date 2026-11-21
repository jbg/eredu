//! Controlled speculation shares facade preparation and the ordinary scheduler.

use super::request::OutputMode;
use super::request::SourceRequest;
use super::{LoadedModel, PreparedChatError};
use crate::runtime::speculative::SpeculativeBatchBackend;
#[cfg(feature = "tensor-capture")]
pub use eredu_core::speculative::{
    AdmittedSpeculativeActivations, SpeculativeActivationCapture, SpeculativeActivationDiscovery,
    SpeculativeActivationOrigin, SpeculativeActivationPhase, SpeculativeActivationPlan,
    SpeculativeCaptureBinding, SpeculativeCaptureRole, SpeculativeCaptureScope,
    SpeculativeInterventionPlan, SpeculativePredictionCapture,
    SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
};
pub use eredu_core::speculative::{SpeculativeControlError, SpeculativeProposalView};
use eredu_core::{generation::SemanticEvent, SpeculativeGenerationOutput};
pub use eredu_runtime::memory_forecast::{
    EmbeddedContinuationMemoryPlan, SpeculativeContinuationForecast,
    SpeculativeContinuationMemoryPlan,
};
#[cfg(feature = "tensor-capture")]
pub use eredu_runtime::speculative::ControlledSpeculativeActivation;
use eredu_runtime::speculative::ControlledSpeculativeOptions;
pub use eredu_runtime::speculative::{
    ControlledSpeculativeSession, ControlledSpeculativeStep, SpeculativeBranchHandle,
    SpeculativeBranchInfo, SpeculativeProposalDisposition, SpeculativeSnapshotHandle,
    SpeculativeVerificationRecord,
};

/// Preparation or controlled speculative execution failure, with neutral causes.
#[derive(Debug, thiserror::Error)]
pub(super) enum ControlledSpeculativeGenerationError {
    /// Existing speculative admission, tokenizer, constraint or backend failure.
    #[error(transparent)]
    Prepared(#[from] PreparedChatError),
    /// Controlled advancement, snapshot or delivery failure.
    #[error(transparent)]
    Control(#[from] SpeculativeControlError),
}

#[cfg(feature = "tensor-capture")]
impl<B: SpeculativeBatchBackend<S>, S: eredu_core::TextExecutionSession<B>> LoadedModel<B, S> {
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

impl<B, S> LoadedModel<B, S>
where
    B: SpeculativeBatchBackend<S>,
    S: eredu_core::TextExecutionSession<B>
        + eredu_runtime::memory_forecast::SpeculativeForecastSession<B, B::Drafter>,
{
    pub(super) fn with_controlled_speculative<'a, F, D>(
        &mut self,
        request: SourceRequest<'a, S::Prompt, B::Drafter, F>,
        options: ControlledSpeculativeOptions,
        drive: D,
        mode: OutputMode,
    ) -> Result<SpeculativeGenerationOutput, ControlledSpeculativeGenerationError>
    where
        F: FnMut(SemanticEvent),
        D: FnOnce(&mut dyn ControlledSpeculativeSession) -> Result<(), SpeculativeControlError>,
    {
        let validation = (|| -> Result<(), ControlledSpeculativeGenerationError> {
            #[cfg(feature = "tensor-capture")]
            if let Some(plan) = options.activations.as_ref() {
                B::validate_speculative_activations(&self.runtime, plan)
                    .map_err(SpeculativeControlError::from)?;
                let (_, maximum) = self
                    .resolve_text_generation_settings(request.request.settings)
                    .map_err(PreparedChatError::from)?;
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
            #[cfg(feature = "tensor-capture")]
            if let Some(plan) = options.capture.as_ref() {
                if plan.request().prompt_tokens != 1 || plan.request().batch != 1 {
                    return Err(SpeculativeControlError::Capture(eredu_core::capture::CaptureError::Invalid(
                    "speculative captures require one-row admission from prepare_speculative_capture".into())).into());
                }
                B::validate_speculative_capture(&self.runtime, plan)
                    .map_err(SpeculativeControlError::from)?;
                let (_, maximum) = self
                    .resolve_text_generation_settings(request.request.settings)
                    .map_err(PreparedChatError::from)?;
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
            |error| PreparedChatError::Backend(error).into(),
        )?;
        // Retain selection geometry before the backend lends its mutable execution
        // resources. Unsupported forecasts must not prevent ordinary execution.
        let forecast_profiles = S::speculative_target_memory_profile(&self.runtime)
            .ok()
            .and_then(|target| {
                S::speculative_memory_profile(&self.runtime, &request.drafting)
                    .ok()
                    .flatten()
                    .map(|draft| (target, draft))
            });
        let mut failure = None;
        let driver = eredu_runtime::speculative::DriveControlledSpeculation::new(
            super::loaded::source_request_scheduler(&request),
            options,
            drive,
            &mut failure,
        )
        .with_preparation_started(request.started)
        .with_speculation_limits(request.limits.map(Into::into))
        .with_forecast_profiles(forecast_profiles)
        .with_vocabulary(
            self.tokenizer
                .get_vocab(true)
                .values()
                .copied()
                .max()
                .map_or(0, |id| id as usize + 1),
        );
        #[cfg(feature = "tensor-capture")]
        let driver = driver
            .with_intervention_discovery(B::speculative_intervention_discovery(&self.runtime).ok())
            .with_activation_discovery(B::speculative_activation_discovery(&self.runtime).ok());
        let result = self.generate_prepared_speculative_with(request, mode, driver);
        if let Some(error) = failure {
            return Err(error.into());
        }
        result.map_err(Into::into)
    }
}

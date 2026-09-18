//! Controlled speculation shares facade preparation and the ordinary scheduler.

use super::{LoadedModel, PreparedChatSpeculativeError};
use eredu_core::SpeculativeGenerationBackend;
pub use eredu_core::speculative::{
    AdmittedSpeculativeActivations, SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
    SpeculativeActivationCapture, SpeculativeActivationDiscovery, SpeculativeActivationOrigin,
    SpeculativeActivationPhase, SpeculativeActivationPlan, SpeculativeCaptureBinding,
    SpeculativeCaptureRole, SpeculativeCaptureScope, SpeculativeControlError,
    SpeculativeInterventionPlan, SpeculativePredictionCapture, SpeculativeProposalView,
};
pub use eredu_runtime::speculative::{
    ControlledSpeculativeActivation, ControlledSpeculativeOptions, ControlledSpeculativeSession,
    ControlledSpeculativeStep, SpeculativeBranchHandle, SpeculativeBranchInfo,
    SpeculativeProposalDisposition, SpeculativeSnapshotHandle, SpeculativeVerificationRecord,
};

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
    ) -> Result<eredu_core::capture::AdmittedCapturePlan, PreparedChatSpeculativeError> {
        let (_, maximum) = self
            .resolve_text_generation_settings(settings)
            .map_err(PreparedChatSpeculativeError::from_generation)?;
        let plan = self
            .admit_capture(
                plan,
                eredu_core::capture::CaptureRequestShape {
                    batch: 1,
                    prompt_tokens: 1,
                    max_predictions: maximum.get() as u64,
                },
            )
            .map_err(PreparedChatSpeculativeError::from_capture)?;
        B::validate_speculative_capture(&self.runtime, &plan)
            .map_err(PreparedChatSpeculativeError::from_capture)?;
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

pub(super) fn continuous_steps(
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

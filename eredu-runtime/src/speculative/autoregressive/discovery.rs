//! Admission and discovery for the installed speculative target.

use super::{BatchSession, ExternalBatchMechanisms, SpeculativeBatchBinding};
use crate::{
    capture::partition::PartitionCaptureHookTransport,
    text_session_driver::{
        capture::TextCaptureMechanisms, speculative_partition_binding, TextExecutable,
        TextSessionDriver, TextSessionMechanisms,
    },
};
use eredu_core::{
    capture::{AdmittedCapturePlan, CaptureError, TextCaptureNative},
    intervention::{AdmittedInterventionPlan, InterventionDiscovery},
    speculative::{AdmittedSpeculativeActivations, SpeculativeActivationDiscovery},
    ModelRuntime, MODEL_LOGITS_OBSERVATION_PATH,
};

/// Reports observation support on the installed session and names its draft source type.
pub trait SpeculativeGenerationBackend<S: eredu_core::BackendSession<Self>>:
    eredu_core::TextGenerationBackend
{
    /// Backend-owned separately prepared draft model.
    type Drafter;

    /// Exact internal activation/edit support for the selected speculative path.
    /// Ordinary sampler-logit capture has a separate admission contract.
    #[cfg(feature = "tensor-capture")]
    fn speculative_activation_discovery(
        _runtime: &ModelRuntime<Self, S>,
    ) -> Result<SpeculativeActivationDiscovery, eredu_core::capture::CaptureError> {
        Err(eredu_core::capture::CaptureError::Unsupported(
            "backend has no scoped speculative activation admission".into(),
        ))
    }

    /// Rechecks immutable internal authority against the actual loaded session.
    #[cfg(feature = "tensor-capture")]
    fn validate_speculative_activations(
        runtime: &ModelRuntime<Self, S>,
        plan: &AdmittedSpeculativeActivations,
    ) -> Result<(), eredu_core::capture::CaptureError> {
        plan.validate(&Self::speculative_activation_discovery(runtime)?)
    }

    /// Genuine mutable hooks with explicit speculative attribution.
    #[cfg(feature = "tensor-capture")]
    fn speculative_intervention_discovery(
        _runtime: &ModelRuntime<Self, S>,
    ) -> Result<eredu_core::intervention::InterventionDiscovery, eredu_core::capture::CaptureError>
    {
        Err(eredu_core::capture::CaptureError::Unsupported(
            "backend has no speculative tensor interventions".into(),
        ))
    }
    /// Validates edit geometry, exact loaded capabilities and evidence budgets.
    #[cfg(feature = "tensor-capture")]
    fn validate_speculative_interventions(
        _runtime: &ModelRuntime<Self, S>,
        _capture: &eredu_core::capture::AdmittedCapturePlan,
        _plan: &eredu_core::intervention::AdmittedInterventionPlan,
    ) -> Result<(), eredu_core::capture::CaptureError> {
        Err(eredu_core::capture::CaptureError::Unsupported(
            "backend has no speculative tensor interventions".into(),
        ))
    }

    /// Validates speculative observation support without allocating or executing.
    /// Ordinary capture support does not imply attribution for speculative phases.
    #[cfg(feature = "tensor-capture")]
    fn validate_speculative_capture(
        _runtime: &ModelRuntime<Self, S>,
        plan: &eredu_core::capture::AdmittedCapturePlan,
    ) -> Result<(), eredu_core::capture::CaptureError> {
        if plan.is_empty() {
            Ok(())
        } else {
            Err(eredu_core::capture::CaptureError::Unsupported(
                "backend has no bounded speculative capture".into(),
            ))
        }
    }
}

/// Validates ordinary score capture attributed to a speculative decision.
pub fn validate_control_capture(plan: &AdmittedCapturePlan) -> Result<(), CaptureError> {
    if plan.is_empty() {
        return Ok(());
    }
    if plan.request().batch != 1 || plan.request().prompt_tokens != 1 {
        return Err(CaptureError::Invalid(
            "speculative logits require one-row capture admission".into(),
        ));
    }
    if plan
        .plan()
        .selections
        .iter()
        .any(|point| point.path != MODEL_LOGITS_OBSERVATION_PATH)
    {
        return Err(CaptureError::Unsupported(
            "speculative layer capture requires draft or verification phase attribution".into(),
        ));
    }
    Ok(())
}

#[cfg(feature = "tensor-capture")]
fn discover<B, N>(
    session: &TextSessionDriver<N::Resources, N>,
) -> Result<SpeculativeActivationDiscovery, CaptureError>
where
    B: eredu_core::BackendProvider,
    N: TextSessionMechanisms<B> + TextCaptureMechanisms<N::Retained>,
    N::ControlTransport: PartitionCaptureHookTransport,
{
    let mechanisms = N::intervention_mechanisms();
    if let Some(binding) = speculative_partition_binding::<B, N>(session) {
        return binding.discovery(&mechanisms);
    }
    let prepared = session.model_discovery().ok_or_else(|| {
        CaptureError::Unsupported("session has no retained activation catalog".into())
    })?;
    let execution = session
        .resources()
        .model
        .as_ref()
        .partition_activation_execution()
        .ok_or_else(|| {
            CaptureError::Unsupported(
                "selected prediction executor has no complete internal activation path".into(),
            )
        })?;
    prepared.speculative_activations(
        &execution,
        &mechanisms,
        session.intervention_identity(),
        session.parameter_overlay_identity(),
    )
}

impl<B, M> SpeculativeGenerationBackend<BatchSession<B, M>> for B
where
    B: SpeculativeBatchBinding<Mechanisms = M>,
    M: ExternalBatchMechanisms<B>,
    M::SessionMechanisms: TextCaptureMechanisms<
        <M::SessionMechanisms as eredu_core::backend::TextSessionNative<B>>::Retained,
    >,
    <M::SessionMechanisms as eredu_core::backend::TextSessionNative<B>>::ControlTransport:
        PartitionCaptureHookTransport,
    BatchSession<B, M>: eredu_core::BackendSession<B>,
{
    type Drafter = super::BatchDraft<B, M>;

    #[cfg(feature = "tensor-capture")]
    fn speculative_activation_discovery(
        runtime: &ModelRuntime<B, BatchSession<B, M>>,
    ) -> Result<SpeculativeActivationDiscovery, CaptureError> {
        discover::<B, M::SessionMechanisms>(runtime.session())
    }

    #[cfg(feature = "tensor-capture")]
    fn validate_speculative_activations(
        runtime: &ModelRuntime<B, BatchSession<B, M>>,
        plan: &AdmittedSpeculativeActivations,
    ) -> Result<(), CaptureError> {
        plan.validate(&Self::speculative_activation_discovery(runtime)?)?;
        crate::intervention::preflight(
            plan.captures(),
            plan.interventions(),
            &M::SessionMechanisms::intervention_estimator(),
        )
    }

    #[cfg(feature = "tensor-capture")]
    fn validate_speculative_capture(
        runtime: &ModelRuntime<B, BatchSession<B, M>>,
        plan: &AdmittedCapturePlan,
    ) -> Result<(), CaptureError> {
        validate_control_capture(plan)?;
        runtime.session().validate_text_capture(plan)
    }

    #[cfg(feature = "tensor-capture")]
    fn speculative_intervention_discovery(
        runtime: &ModelRuntime<B, BatchSession<B, M>>,
    ) -> Result<InterventionDiscovery, CaptureError> {
        let mut discovery = runtime.session().text_intervention_discovery()?;
        discovery
            .points
            .retain(|point| point.path == MODEL_LOGITS_OBSERVATION_PATH);
        Ok(discovery)
    }

    #[cfg(feature = "tensor-capture")]
    fn validate_speculative_interventions(
        runtime: &ModelRuntime<B, BatchSession<B, M>>,
        capture: &AdmittedCapturePlan,
        plan: &AdmittedInterventionPlan,
    ) -> Result<(), CaptureError> {
        Self::validate_speculative_capture(runtime, capture)?;
        crate::intervention::validate_session(
            capture,
            plan,
            &Self::speculative_intervention_discovery(runtime)?,
            &M::SessionMechanisms::intervention_estimator(),
        )
    }
}

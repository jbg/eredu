//! Local capture and intervention on the shared causal submission boundary.

use eredu_core::{
    capture::{AdmittedCapturePlan, CaptureDeclarations, CaptureDiscovery, CaptureError},
    intervention::{AdmittedInterventionPlan, InterventionDiscovery, InterventionEstimator},
};

use crate::{
    capture::CaptureSession,
    text_session_driver::{TextExecutable, TextPrefillInput},
};

use super::*;

impl<R: 'static, N> TextSessionDriver<R, N, CausalTextGeneration> {
    pub(super) fn local_capture_declarations(&self) -> Result<CaptureDeclarations, CaptureError> {
        Ok(self
            .model_discovery()
            .ok_or_else(|| CaptureError::Unsupported("no retained capture catalog".into()))?
            .capture_declarations())
    }

    pub(super) fn local_capture_discovery(&self) -> Result<CaptureDiscovery, CaptureError> {
        self.model_discovery()
            .ok_or_else(|| CaptureError::Unsupported("no retained capture catalog".into()))?
            .capture()
    }

    pub(crate) fn validate_local_capture<B>(
        &self,
        plan: &AdmittedCapturePlan,
    ) -> Result<(), CaptureError>
    where
        B: TextGenerationBackend<Error = BackendFailure>,
        N: TextSessionMechanisms<B, Resources = R>,
    {
        if plan.is_empty() {
            return Ok(());
        }
        if plan.request().batch != 1 {
            return Err(CaptureError::Unsupported(
                "local capture requires single-sequence causal geometry".into(),
            ));
        }
        crate::capture::validate_session(
            plan,
            &self.local_capture_declarations()?,
            N::estimate_local_capture,
        )
    }

    pub(super) fn local_intervention_discovery<B>(
        &self,
    ) -> Result<InterventionDiscovery, CaptureError>
    where
        B: TextGenerationBackend<Error = BackendFailure>,
        N: TextSessionMechanisms<B, Resources = R>,
    {
        let (mechanisms, _) = self
            .native()
            .local_intervention()
            .ok_or_else(no_local_intervention)?;
        self.retained_intervention_discovery(&mechanisms)
    }

    pub(super) fn validate_local_interventions<B>(
        &self,
        capture: &AdmittedCapturePlan,
        plan: &AdmittedInterventionPlan,
    ) -> Result<(), CaptureError>
    where
        B: TextGenerationBackend<Error = BackendFailure>,
        N: TextSessionMechanisms<B, Resources = R>,
    {
        self.validate_local_capture::<B>(capture)?;
        if plan.is_empty() {
            return Ok(());
        }
        self.local_intervention_estimator::<B>(capture, plan)
            .map(|_| ())
    }

    fn local_intervention_estimator<B>(
        &self,
        capture: &AdmittedCapturePlan,
        plan: &AdmittedInterventionPlan,
    ) -> Result<std::sync::Arc<dyn InterventionEstimator>, CaptureError>
    where
        B: TextGenerationBackend<Error = BackendFailure>,
        N: TextSessionMechanisms<B, Resources = R>,
    {
        let (mechanisms, estimator) = self
            .native()
            .local_intervention()
            .ok_or_else(no_local_intervention)?;
        crate::intervention::validate_session(
            capture,
            plan,
            &self.retained_intervention_discovery(&mechanisms)?,
            estimator.as_ref(),
        )?;
        Ok(estimator)
    }

    pub(super) fn install_local_capture<B>(
        &self,
        slot: &mut Option<CaptureSession>,
        capture: AdmittedCapturePlan,
        intervention: Option<AdmittedInterventionPlan>,
    ) -> Result<(), CaptureError>
    where
        B: TextGenerationBackend<Error = BackendFailure>,
        N: TextSessionMechanisms<B, Resources = R>,
    {
        self.validate_local_capture::<B>(&capture)?;
        let intervention = intervention
            .filter(|plan| !plan.is_empty())
            .map(|plan| {
                let estimator = self.local_intervention_estimator::<B>(&capture, &plan)?;
                Ok::<_, CaptureError>((plan, estimator))
            })
            .transpose()?;
        if capture.plan().provenance == eredu_core::capture::CaptureProvenance::ContentExact {
            self.local_capture_discovery()?;
        }
        crate::intervention::install_session(slot, capture, intervention)
    }

    pub(super) fn submit_local_capture_prefill<B>(
        &mut self,
        backend: &B,
        prompt: N::Prefill,
        domain: Option<CaptureTokenDomain<'_>>,
        capture: &mut CaptureSession,
    ) -> Result<Output<B, N>, TextOperationError<BackendFailure>>
    where
        B: TextGenerationBackend<Error = BackendFailure>,
        N: TextSessionMechanisms<B, Resources = R>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        let request = capture.plan().request();
        let geometry = self
            .resources()
            .model
            .as_ref()
            .prepared_input_plans(prompt.borrowed())
            .map_err(|error| TextOperationError::Backend(BackendFailure::from_error(error)))?;
        let (count, media) =
            crate::text_session_driver::capture::prepared_prompt_capture_shape(&geometry)
                .map_err(|error| TextOperationError::Backend(BackendFailure::from_error(error)))?;
        if capture.plan().invocation_bounds().is_some() {
            let mut shape = self
                .text_capture_shape(count)
                .map_err(|error| TextOperationError::Backend(BackendFailure::from_error(error)))?;
            shape.media = media;
            prepare_local_invocation(
                capture,
                0,
                eredu_core::capture::CapturePhase::Prefill,
                shape,
            )
            .map_err(TextOperationError::Backend)?;
        }
        if capture.plan().invocation_bounds().is_none() && count != request.prompt_tokens {
            return Err(TextOperationError::Backend(
                BackendFailure::from_error(CaptureError::Invalid(
                    "actual decoder prompt positions differ from admitted capture geometry".into(),
                ))
                .with_model_state_preserved(true),
            ));
        }
        self.submit_local_observed(backend, domain, capture, 0, |session, observer| {
            session.run_prefill(backend, Ok(prompt), Some(request), Some(observer))
        })
    }

    pub(super) fn submit_local_capture_decode<B>(
        &mut self,
        backend: &B,
        input: TextDecodeInput<
            super::super::TextDecodeValue<B, N>,
            N::SampledDecode,
            BackendFailure,
        >,
        domain: Option<CaptureTokenDomain<'_>>,
        prediction: u64,
        capture: &mut CaptureSession,
    ) -> Result<Output<B, N>, TextOperationError<BackendFailure>>
    where
        B: TextGenerationBackend<Error = BackendFailure>,
        N: TextSessionMechanisms<B, Resources = R>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        if capture.plan().invocation_bounds().is_some() {
            let shape = self
                .text_capture_shape(1)
                .map_err(|error| TextOperationError::Backend(BackendFailure::from_error(error)))?;
            prepare_local_invocation(
                capture,
                prediction,
                eredu_core::capture::CapturePhase::Decode,
                shape,
            )
            .map_err(TextOperationError::Backend)?;
        }
        self.submit_local_observed(backend, domain, capture, prediction, |session, observer| {
            session.run_decode(backend, input, Some(observer))
        })
    }

    fn submit_local_observed<B>(
        &mut self,
        backend: &B,
        domain: Option<CaptureTokenDomain<'_>>,
        capture: &mut CaptureSession,
        prediction: u64,
        execute: impl FnOnce(
            &mut Self,
            &mut super::super::TextObserver<'_, B, N>,
        )
            -> Result<(super::super::RawTextSubmission<B, N>, bool), BackendFailure>,
    ) -> Result<Output<B, N>, TextOperationError<BackendFailure>>
    where
        B: TextGenerationBackend<Error = BackendFailure>,
        N: TextSessionMechanisms<B, Resources = R>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        let collector = N::local_capture(N::context(backend), domain)
            .ok_or_else(|| {
                BackendFailure::from_error(CaptureError::Unsupported(
                    "no local tensor collector".into(),
                ))
                .with_model_state_preserved(true)
            })
            .map_err(TextOperationError::Backend)?;
        let mut observer =
            crate::text_session_driver::capture::local_text_capture(capture, collector, prediction);
        let (submission, publish) =
            execute(self, &mut observer).map_err(TextOperationError::Backend)?;
        Ok(Submission {
            output: N::publish_scores(submission.output, publish),
            completion: submission.completion,
        })
    }
}

pub(super) fn no_local_intervention() -> CaptureError {
    CaptureError::Unsupported("no local intervention mechanisms".into())
}

pub(super) fn prepare_local_invocation(
    capture: &mut CaptureSession,
    prediction: u64,
    phase: eredu_core::capture::CapturePhase,
    shape: eredu_core::capture::CaptureInvocationShape,
) -> Result<(), BackendFailure> {
    capture
        .set_commitment_boundary(prediction)
        .map_err(BackendFailure::from_error)?;
    capture
        .begin_invocation(phase, prediction, shape, Default::default())
        .map_err(BackendFailure::from_error)
}

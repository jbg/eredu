//! Shared admission, immutable installation and bounded observer forwarding.
use super::*;
use std::sync::Arc;

/// Revalidates exact retained session capabilities before shared geometry/budget
/// traversal. Callers first validate native execution-mode facts and capture support.
pub fn validate_session(
    capture: &AdmittedCapturePlan,
    plan: &AdmittedInterventionPlan,
    discovery: &InterventionDiscovery,
    estimator: &dyn InterventionEstimator,
) -> Result<(), CaptureError> {
    validate_continuation(
        capture,
        plan,
        discovery,
        estimator,
        0,
        CaptureUsage::default(),
    )
}

pub(crate) fn validate_continuation(
    capture: &AdmittedCapturePlan,
    plan: &AdmittedInterventionPlan,
    discovery: &InterventionDiscovery,
    estimator: &dyn InterventionEstimator,
    next_prediction: u64,
    inherited: CaptureUsage,
) -> Result<(), CaptureError> {
    if plan.request().batch != 1 {
        return Err(CaptureError::Unsupported(
            "interventions require single-sequence text generation".into(),
        ));
    }
    let checked = plan
        .plan()
        .clone()
        .admit(discovery, plan.request(), plan.session_id())?;
    if checked.identity() != plan.identity() {
        return Err(CaptureError::Invalid(
            "intervention admission differs from loaded source/session capabilities".into(),
        ));
    }
    preflight_continuation(capture, &checked, estimator, next_prediction, inherited)
}

/// Installs one immutable run after validation. Both capture-only and combined
/// plans obey the same replacement rule; empty runs keep the ordinary fast path.
pub fn install_session(
    slot: &mut Option<CaptureSession>,
    capture: AdmittedCapturePlan,
    intervention: Option<(AdmittedInterventionPlan, Arc<dyn InterventionEstimator>)>,
) -> Result<(), CaptureError> {
    if slot.is_some() {
        return Err(CaptureError::Invalid(
            "capture/intervention plan already installed".into(),
        ));
    }
    if capture.is_empty()
        && intervention
            .as_ref()
            .is_none_or(|(plan, _)| plan.is_empty())
    {
        return Ok(());
    }
    let mut session = CaptureSession::new(capture);
    if let Some((plan, estimator)) = intervention {
        session.enable_interventions(plan, estimator)?;
    }
    *slot = Some(session);
    Ok(())
}

/// Borrows the existing capture owner. Backends provide native primitives and a
/// typed error conversion; this adapter owns no completion or native recovery state.
pub struct CaptureObserver<'a, B, F> {
    session: &'a mut CaptureSession,
    backend: B,
    map_error: F,
}
impl<'a, B, F> CaptureObserver<'a, B, F> {
    /// Creates an observer that forwards all value/control hooks to the shared run.
    pub fn new(session: &'a mut CaptureSession, backend: B, map_error: F) -> Self {
        Self {
            session,
            backend,
            map_error,
        }
    }
}
impl<B, E, F> crate::ActivationObserver<B::Tensor, E> for CaptureObserver<'_, B, F>
where
    B: InterventionBackend,
    F: Fn(CaptureExecutionError<B::Error>) -> E,
{
    fn observe(&mut self, path: &str, value: &B::Tensor) -> Result<(), E> {
        self.session
            .observe(&mut self.backend, path, value)
            .map_err(&self.map_error)
    }
    fn intervene(&mut self, path: &str, value: &B::Tensor) -> Result<Option<B::Tensor>, E> {
        self.session
            .intervene(&mut self.backend, path, value)
            .map_err(&self.map_error)
    }
    fn routing_control(
        &mut self,
        path: &str,
        rows: u64,
    ) -> Result<Option<eredu_nn::routing_intervention::GroupSelectionControl>, E> {
        self.session
            .routing_control(path, rows)
            .map_err(|error| (self.map_error)(error.into()))
    }
    fn routing_applied(
        &mut self,
        path: &str,
        original: Option<crate::RoutingDecision<'_, B::Tensor>>,
        effective: crate::RoutingDecision<'_, B::Tensor>,
    ) -> Result<(), E> {
        self.session
            .routing_applied(&mut self.backend, path, original, effective)
            .map_err(&self.map_error)
    }
    fn routing_failed(&mut self, path: &str, message: &str) {
        self.session.routing_failed(path, message);
    }
    fn finish(&mut self) -> Result<(), E> {
        self.session
            .finish_interventions()
            .map_err(|error| (self.map_error)(error.into()))
    }
    fn observe_routing(
        &mut self,
        routing: crate::RoutingObservation<'_, B::Tensor>,
    ) -> Result<(), E> {
        let mut result = Ok(());
        routing.for_each_tensor(|path, value| {
            if result.is_ok() {
                result = self.observe(&path, value);
            }
        });
        result
    }
}

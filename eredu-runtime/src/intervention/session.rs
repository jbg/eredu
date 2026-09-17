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
    let checked = plan.readmit(discovery)?;
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
    next_prediction: Option<u64>,
    routed_path: Option<std::borrow::Cow<'a, str>>,
    routed_error: Option<&'a dyn Fn(eredu_nn::Error) -> eredu_nn::Error>,
}
impl<'a, B, F> CaptureObserver<'a, B, F> {
    /// Creates an observer that forwards all value/control hooks to the shared run.
    pub fn new(session: &'a mut CaptureSession, backend: B, map_error: F) -> Self {
        Self {
            session,
            backend,
            map_error,
            next_prediction: None,
            routed_path: None,
            routed_error: None,
        }
    }

    /// Records routed-provider failures before crossing the neural error domain.
    /// The handler must preserve the original cause and performs no native work.
    pub fn with_routed_error_handler(
        mut self,
        handler: &'a dyn Fn(eredu_nn::Error) -> eredu_nn::Error,
    ) -> Self {
        self.routed_error = Some(handler);
        self
    }

    fn routed_result<R>(&self, result: Result<R, eredu_nn::Error>) -> Result<R, eredu_nn::Error> {
        result.map_err(|error| match self.routed_error {
            Some(handler) => handler(error),
            None => error,
        })
    }

    pub(crate) fn with_routed_path(mut self, path: &'a str) -> Self {
        self.routed_path = Some(std::borrow::Cow::Borrowed(path));
        self
    }

    /// Defers step admission to the shared forward's local preparation phase.
    /// Construction performs no reservation or native work. The actual forward
    /// supplies its phase; a local admission failure is agreed with peers before
    /// any observer coordination or model work.
    pub fn for_step(
        session: &'a mut CaptureSession,
        backend: B,
        prediction: u64,
        map_error: F,
    ) -> Self {
        Self {
            session,
            backend,
            map_error,
            next_prediction: Some(prediction),
            routed_path: None,
            routed_error: None,
        }
    }
}
impl<B, E, F> crate::ActivationObserver<B::Tensor, E> for CaptureObserver<'_, B, F>
where
    B: InterventionBackend,
    B::Error: Send + Sync,
    F: Fn(CaptureExecutionError<B::Error>) -> E,
{
    fn routed_unit_observer(
        &mut self,
        path: &str,
    ) -> Result<Option<&mut dyn crate::RoutedUnitObserver<B::Tensor>>, E> {
        if !self.session.wants_routed_units(path) && !self.session.wants_routed_interventions(path)
        {
            return Ok(None);
        }
        if self.routed_path.as_deref() != Some(path) {
            self.routed_path = Some(std::borrow::Cow::Owned(path.into()));
        }
        Ok(Some(self))
    }
    fn requires_prepared_traversal(&self) -> bool {
        // Decode still uses the installed source. The p0-only predicate would
        // select the custom-callback path and invalidate its prepared binding.
        self.session.ordinary_prefill_source_binding().is_some()
    }
    fn requires_sequence_readout(&self) -> bool {
        self.session
            .ordinary_prefill_capture()
            .map_or(true, |capture| capture.requires_sequence())
    }
    fn supports_prefill_spans(&self) -> bool {
        self.session.ordinary_prefill_capture().is_some()
    }
    fn ordinary_prefill_capture(&self) -> Option<&crate::capture::OrdinaryPrefillCapture> {
        self.session.ordinary_prefill_capture()
    }
    fn begin_prefill_chunk(&mut self, chunk: &crate::prefill::PrefillChunk) -> Result<(), E> {
        if self.session.ordinary_prefill_capture().is_some() {
            self.session
                .begin_ordinary_prefill(chunk)
                .map_err(|error| (self.map_error)(error.into()))?;
        }
        Ok(())
    }
    fn finish_prefill(&mut self, committed: bool) {
        self.session.finish_ordinary_prefill(committed);
    }
    fn transactional(&self) -> bool {
        true
    }
    fn prepare_transaction(
        &mut self,
        epoch: eredu_core::DistributedCommitEpoch,
        pass: crate::ExpertPass,
    ) -> Result<(), E> {
        if self.session.ordinary_prefill_capture().is_some() {
            self.next_prediction.take();
            return self
                .session
                .prepare_ordinary_prefill(epoch, pass)
                .map_err(|error| (self.map_error)(error.into()));
        }
        match self.next_prediction.take() {
            Some(prediction) => self
                .session
                .prepare_step_transaction(epoch, pass, prediction),
            None => self.session.prepare_transaction(epoch, pass),
        }
        .map_err(|error| (self.map_error)(error.into()))
    }
    fn complete_transaction(&mut self, epoch: eredu_core::DistributedCommitEpoch) -> Result<(), E> {
        if self.session.ordinary_prefill_capture().is_some() {
            return self
                .session
                .complete_ordinary_prefill(epoch)
                .map_err(|error| (self.map_error)(error.into()));
        }
        self.session
            .complete_transaction(epoch)
            .map_err(|error| (self.map_error)(error.into()))
    }
    fn finish_transaction(&mut self, epoch: eredu_core::DistributedCommitEpoch, committed: bool) {
        if self.session.ordinary_prefill_capture().is_some() {
            self.session.finish_ordinary_chunk(epoch, committed);
        } else {
            self.session.finish_transaction(epoch, committed)
        }
    }
    fn observe(&mut self, path: &str, value: &B::Tensor) -> Result<(), E> {
        self.session
            .observe(&mut self.backend, path, value)
            .map_err(&self.map_error)
    }
    fn observe_generated(
        &mut self,
        path: &str,
        prototype: &B::Tensor,
        source: &eredu_core::capture::GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<B::Tensor, E>,
    ) -> Result<(), E> {
        self.session.observe_generated(
            &mut self.backend,
            path,
            prototype,
            source,
            generate,
            &self.map_error,
        )
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
    fn routing_unmodified_interest(&self, path: &str) -> crate::RoutingUnmodifiedInterest {
        self.session.routing_unmodified_interest(path)
    }
    fn routing_unmodified(
        &mut self,
        path: &str,
        effective: crate::RoutingDecision<'_, B::Tensor>,
    ) -> Result<(), E> {
        self.session
            .routing_unmodified(&mut self.backend, path, effective)
            .map_err(&self.map_error)
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
            .finish_routed_captures()
            .map_err(|error| (self.map_error)(error.into()))?;
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

impl<B: InterventionBackend, F> crate::RoutedUnitObserver<B::Tensor> for CaptureObserver<'_, B, F>
where
    B::Error: Send + Sync,
{
    fn intervene(
        &mut self,
        batch: &crate::RoutedUnitBatch<'_, B::Tensor>,
    ) -> Result<Option<B::Tensor>, eredu_nn::Error> {
        let result = (|| {
            if batch.origins.is_some() || batch.unit_coordinates.is_some() {
                return Err(eredu_nn::Error::backend(
                    "partitioned sparse unit edits require committed distributed receipts",
                ));
            }
            let path = self.routed_path.as_deref().ok_or_else(|| {
                eredu_nn::Error::backend("missing sparse intervention invocation")
            })?;
            let token_offset = batch
                .source_token(0)
                .and_then(|n| u64::try_from(n).ok())
                .ok_or_else(|| {
                    eredu_nn::Error::backend("sparse intervention token offset overflow")
                })?;
            self.session
                .intervene_routed_units(
                    &mut self.backend,
                    path,
                    &RoutedUnitCaptureSource {
                        values: batch.units.values,
                        token_indices: batch.units.token_indices,
                        selection_indices: batch.units.selection_indices,
                        coefficients: batch.units.coefficients,
                        source_groups: batch.source_groups,
                        token_offset,
                        global_groups: batch.global_groups,
                    },
                )
                .map_err(eredu_nn::Error::backend_source)
        })();
        self.routed_result(result)
    }
    fn observe(
        &mut self,
        batch: &crate::RoutedUnitBatch<'_, B::Tensor>,
    ) -> Result<(), eredu_nn::Error> {
        self.capture_units(batch, false)
    }
    fn observe_effective(
        &mut self,
        batch: &crate::RoutedUnitBatch<'_, B::Tensor>,
    ) -> Result<(), eredu_nn::Error> {
        self.capture_units(batch, true)
    }
}
impl<B: InterventionBackend, F> CaptureObserver<'_, B, F>
where
    B::Error: Send + Sync,
{
    fn capture_units(
        &mut self,
        batch: &crate::RoutedUnitBatch<'_, B::Tensor>,
        effective: bool,
    ) -> Result<(), eredu_nn::Error> {
        let result = self.capture_units_inner(batch, effective);
        self.routed_result(result)
    }

    fn capture_units_inner(
        &mut self,
        batch: &crate::RoutedUnitBatch<'_, B::Tensor>,
        effective: bool,
    ) -> Result<(), eredu_nn::Error> {
        if batch.origins.is_some() || batch.unit_coordinates.is_some() {
            return Err(eredu_nn::Error::backend(
                "partitioned routed-unit capture requires committed distributed receipts",
            ));
        }
        let path = self
            .routed_path
            .as_deref()
            .ok_or_else(|| eredu_nn::Error::backend("missing routed capture invocation"))?;
        let token_offset = batch
            .source_token(0)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| eredu_nn::Error::backend("routed capture token offset overflow"))?;
        self.session
            .observe_routed_units(
                &mut self.backend,
                path,
                effective,
                &RoutedUnitCaptureSource {
                    values: batch.units.values,
                    token_indices: batch.units.token_indices,
                    selection_indices: batch.units.selection_indices,
                    coefficients: batch.units.coefficients,
                    source_groups: batch.source_groups,
                    token_offset,
                    global_groups: batch.global_groups,
                },
            )
            .map_err(eredu_nn::Error::backend_source)
    }
}

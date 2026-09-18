use super::*;
use crate::{intervention::CaptureObserver, ActivationObserver, RoutedUnitObserver};

impl<P: CaptureBackendProvider, F> SpeculativeCaptureObserver<P, F> {
    fn observer<E>(
        &mut self,
    ) -> CaptureObserver<'_, P::Backend<'_>, impl Fn(CaptureExecutionError<P::Error>) -> E + '_>
    where
        F: SpeculativeCaptureErrorTransport<P::Error, E>,
    {
        let failure = &self.failure;
        let map = &self.map_error;
        let observer =
            CaptureObserver::new(&mut self.session, self.provider.backend(), move |error| {
                signal(failure, map, error)
            });
        match self.routed_path.as_deref() {
            Some(path) => observer.with_routed_path(path),
            None => observer,
        }
    }
}

impl<P, F, E> ActivationObserver<P::Tensor, E> for SpeculativeCaptureObserver<P, F>
where
    P: CaptureBackendProvider,
    F: SpeculativeCaptureErrorTransport<P::Error, E>,
{
    fn supports_prefill_spans(&self) -> bool {
        true
    }
    fn transactional(&self) -> bool {
        true
    }
    fn prepare_transaction(
        &mut self,
        epoch: eredu_core::DistributedCommitEpoch,
        pass: crate::ExpertPass,
    ) -> Result<(), E> {
        self.observer().prepare_transaction(epoch, pass)
    }
    fn complete_transaction(&mut self, epoch: eredu_core::DistributedCommitEpoch) -> Result<(), E> {
        self.observer().complete_transaction(epoch)
    }
    fn finish_transaction(&mut self, epoch: eredu_core::DistributedCommitEpoch, committed: bool) {
        self.session.finish_transaction(epoch, committed);
    }
    fn observe(&mut self, path: &str, value: &P::Tensor) -> Result<(), E> {
        ActivationObserver::observe(&mut self.observer(), path, value)
    }
    fn observe_generated(
        &mut self,
        path: &str,
        prototype: &P::Tensor,
        source: &GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<P::Tensor, E>,
    ) -> Result<(), E> {
        self.observer()
            .observe_generated(path, prototype, source, generate)
    }
    fn intervene(&mut self, path: &str, value: &P::Tensor) -> Result<Option<P::Tensor>, E> {
        ActivationObserver::intervene(&mut self.observer(), path, value)
    }
    fn routing_control(
        &mut self,
        path: &str,
        rows: u64,
    ) -> Result<Option<eredu_nn::routing_intervention::GroupSelectionControl>, E> {
        self.observer().routing_control(path, rows)
    }
    fn routing_unmodified_interest(&self, path: &str) -> crate::RoutingUnmodifiedInterest {
        self.session.routing_unmodified_interest(path)
    }
    fn routing_unmodified(
        &mut self,
        path: &str,
        effective: crate::RoutingDecision<'_, P::Tensor>,
    ) -> Result<(), E> {
        self.observer().routing_unmodified(path, effective)
    }

    fn routing_applied(
        &mut self,
        path: &str,
        original: Option<crate::RoutingDecision<'_, P::Tensor>>,
        effective: crate::RoutingDecision<'_, P::Tensor>,
    ) -> Result<(), E> {
        self.observer().routing_applied(path, original, effective)
    }
    fn routing_failed(&mut self, path: &str, message: &str) {
        self.session.routing_failed(path, message);
    }
    fn observe_routing(
        &mut self,
        routing: crate::RoutingObservation<'_, P::Tensor>,
    ) -> Result<(), E> {
        self.observer().observe_routing(routing)
    }
    fn routed_unit_observer(
        &mut self,
        path: &str,
    ) -> Result<Option<&mut dyn RoutedUnitObserver<P::Tensor>>, E> {
        if !self.session.wants_routed_units(path) && !self.session.wants_routed_interventions(path)
        {
            return Ok(None);
        }
        if self.routed_path.as_deref() != Some(path) {
            self.routed_path = Some(path.into());
        }
        Ok(Some(self))
    }
    fn finish(&mut self) -> Result<(), E> {
        self.observer().finish()
    }
}

// Sparse providers retain their neutral neural error domain. Preserve portable
// admission failures from its source chain before returning that original error.
pub(super) fn retain_sparse<E: std::error::Error + 'static>(
    failure: &RefCell<Option<SpeculativeControlError>>,
    error: eredu_nn::Error,
) -> eredu_nn::Error {
    let mut portable = None;
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(&error);
    while let Some(cause) = current {
        if let Some(CaptureExecutionError::<E>::Admission(error)) =
            cause.downcast_ref::<CaptureExecutionError<E>>()
        {
            portable = Some(error.clone());
            break;
        }
        let partition_admission = cause
            .downcast_ref::<partition::PartitionCaptureObserverError<E>>()
            .and_then(|error| match error {
                partition::PartitionCaptureObserverError::Capture(
                    CaptureExecutionError::Admission(error),
                ) => Some(error),
                partition::PartitionCaptureObserverError::Exchange(error) => {
                    exchange_admission(error)
                }
                _ => None,
            })
            .or_else(|| {
                cause
                    .downcast_ref::<partition::PartitionCaptureExchangeError>()
                    .and_then(exchange_admission)
            });
        if let Some(error) = partition_admission {
            portable = Some(error.clone());
            break;
        }
        if let Some(error) = cause.downcast_ref::<CaptureError>() {
            portable = Some(error.clone());
            break;
        }
        current = cause.source();
    }
    let shared = std::sync::Arc::new(error);
    let mut stored = failure.borrow_mut();
    if stored.is_none() {
        *stored = Some(match portable {
            Some(error) => SpeculativeControlError::Capture(error),
            None => SpeculativeControlError::backend(std::sync::Arc::clone(&shared)),
        });
    }
    eredu_nn::Error::backend_retained_source(shared)
}

// Transparent error domains can skip inner variants in Error::source. Follow
// typed local-rejection ownership before returning to the generic source walk.
fn exchange_admission(error: &partition::PartitionCaptureExchangeError) -> Option<&CaptureError> {
    match error {
        partition::PartitionCaptureExchangeError::Capture(error) => Some(error),
        partition::PartitionCaptureExchangeError::LocalRejected { source, .. } => {
            exchange_admission(source)
        }
        _ => None,
    }
}

impl<P: CaptureBackendProvider, F> RoutedUnitObserver<P::Tensor>
    for SpeculativeCaptureObserver<P, F>
{
    fn intervene(
        &mut self,
        batch: &crate::RoutedUnitBatch<'_, P::Tensor>,
    ) -> Result<Option<P::Tensor>, eredu_nn::Error> {
        let mut observer = CaptureObserver::new(&mut self.session, self.provider.backend(), ());
        if let Some(path) = self.routed_path.as_deref() {
            observer = observer.with_routed_path(path);
        }
        let result = RoutedUnitObserver::intervene(&mut observer, batch);
        result.map_err(|error| retain_sparse::<P::Error>(&self.failure, error))
    }
    fn observe(
        &mut self,
        batch: &crate::RoutedUnitBatch<'_, P::Tensor>,
    ) -> Result<(), eredu_nn::Error> {
        let mut observer = CaptureObserver::new(&mut self.session, self.provider.backend(), ());
        if let Some(path) = self.routed_path.as_deref() {
            observer = observer.with_routed_path(path);
        }
        let result = RoutedUnitObserver::observe(&mut observer, batch);
        result.map_err(|error| retain_sparse::<P::Error>(&self.failure, error))
    }
    fn observe_effective(
        &mut self,
        batch: &crate::RoutedUnitBatch<'_, P::Tensor>,
    ) -> Result<(), eredu_nn::Error> {
        let mut observer = CaptureObserver::new(&mut self.session, self.provider.backend(), ());
        if let Some(path) = self.routed_path.as_deref() {
            observer = observer.with_routed_path(path);
        }
        let result = RoutedUnitObserver::observe_effective(&mut observer, batch);
        result.map_err(|error| retain_sparse::<P::Error>(&self.failure, error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sparse_error_chain_preserves_portable_limits_and_native_sources() {
        use std::error::Error;
        let failure = RefCell::new(None);
        let limit = CaptureError::Limit {
            budget: CaptureBudget::Retention,
            cumulative: true,
        };
        let signal = retain_sparse::<std::io::Error>(
            &failure,
            eredu_nn::Error::backend_retained_source(CaptureExecutionError::<std::io::Error>::Admission(
                limit.clone(),
            )),
        );
        assert!(
            matches!(failure.borrow_mut().take(), Some(SpeculativeControlError::Capture(actual)) if actual == limit)
        );
        for error in [
            partition::PartitionCaptureObserverError::<std::io::Error>::Capture(
                CaptureExecutionError::Admission(limit.clone()),
            ),
            partition::PartitionCaptureObserverError::Exchange(
                partition::PartitionCaptureExchangeError::Capture(limit.clone()),
            ),
            partition::PartitionCaptureObserverError::Exchange(
                partition::PartitionCaptureExchangeError::LocalRejected {
                    rank: 2,
                    stage: partition::PartitionCaptureExchangeStage::Preparation,
                    source: Box::new(partition::PartitionCaptureExchangeError::Capture(
                        limit.clone(),
                    )),
                },
            ),
        ] {
            let partition =
                retain_sparse::<std::io::Error>(&failure, eredu_nn::Error::backend_retained_source(error));
            assert!(
                matches!(failure.borrow_mut().take(), Some(SpeculativeControlError::Capture(actual)) if actual == limit)
            );
            assert!(partition.source().is_some());
        }
        assert!(signal.source().is_some());
        let signal = retain_sparse::<std::io::Error>(
            &failure,
            eredu_nn::Error::backend_retained_source(CaptureExecutionError::Backend(std::io::Error::other(
                "native sparse failure",
            ))),
        );
        let Some(SpeculativeControlError::Backend(error)) = failure.borrow_mut().take() else {
            panic!("native source missing");
        };
        let mut cause = error.source();
        let mut found = false;
        while let Some(error) = cause {
            found |= error.is::<std::io::Error>();
            cause = error.source();
        }
        assert!(found);
        assert!(signal.to_string().contains("native sparse failure"));
    }
}

//! Observation forwarding across transparent backend tensor representations.

use super::{ActivationObserver, RoutingDecision, RoutingObservation};
use crate::{ExpertPass, RoutedUnitBatch, RoutedUnitInvocation, RoutedUnitObserver};
use eredu_core::{
    capture::GeneratedCaptureSource, observation::ObserverTensorMapping, DistributedCommitEpoch,
};
use eredu_nn::{routing_intervention::GroupSelectionControl, Error};

/// Portable forwarding of an activation observer through native tensor conversions.
pub struct MappedActivationObserver<'a, O: ?Sized, M> {
    inner: &'a mut O,
    routed_path: Option<String>,
    routed_invocation_active: bool,
    mapping: std::marker::PhantomData<fn() -> M>,
}

impl<'a, O: ?Sized, M> MappedActivationObserver<'a, O, M> {
    /// Borrows the admitted observer for one execution.
    pub fn new(observer: &'a mut O) -> Self {
        Self {
            inner: observer,
            routed_path: None,
            routed_invocation_active: false,
            mapping: std::marker::PhantomData,
        }
    }
}

impl<O, M> ActivationObserver<M::InvocationTensor, M::InvocationError>
    for MappedActivationObserver<'_, O, M>
where
    M: ObserverTensorMapping,
    O: ActivationObserver<M::ObserverTensor, M::ObserverError> + ?Sized,
{
    fn is_noop(&self) -> bool {
        self.inner.is_noop()
    }

    fn routed_unit_observer(
        &mut self,
        path: &str,
    ) -> Result<Option<&mut dyn RoutedUnitObserver<M::InvocationTensor>>, M::InvocationError> {
        let Some(observer) = self
            .inner
            .routed_unit_observer(path)
            .map_err(M::invocation_error)?
        else {
            self.routed_path = None;
            self.routed_invocation_active = false;
            return Ok(None);
        };
        self.routed_invocation_active = observer.invocation_active();
        if self.routed_path.as_deref() != Some(path) {
            self.routed_path = Some(path.into());
        }
        Ok(Some(self))
    }

    fn transactional(&self) -> bool {
        self.inner.transactional()
    }

    fn prepare_transaction(
        &mut self,
        epoch: DistributedCommitEpoch,
        pass: ExpertPass,
    ) -> Result<(), M::InvocationError> {
        self.inner
            .prepare_transaction(epoch, pass)
            .map_err(M::invocation_error)
    }

    fn coordinate_transaction(
        &mut self,
        epoch: DistributedCommitEpoch,
    ) -> Result<(), M::InvocationError> {
        self.inner
            .coordinate_transaction(epoch)
            .map_err(M::invocation_error)
    }

    fn complete_transaction(
        &mut self,
        epoch: DistributedCommitEpoch,
    ) -> Result<(), M::InvocationError> {
        self.inner
            .complete_transaction(epoch)
            .map_err(M::invocation_error)
    }

    fn finish_transaction(&mut self, epoch: DistributedCommitEpoch, committed: bool) {
        self.inner.finish_transaction(epoch, committed);
    }

    fn routing_control(
        &mut self,
        path: &str,
        rows: u64,
    ) -> Result<Option<GroupSelectionControl>, M::InvocationError> {
        self.inner
            .routing_control(path, rows)
            .map_err(M::invocation_error)
    }

    fn routing_applied(
        &mut self,
        path: &str,
        original: Option<RoutingDecision<'_, M::InvocationTensor>>,
        effective: RoutingDecision<'_, M::InvocationTensor>,
    ) -> Result<(), M::InvocationError> {
        fn map<M: ObserverTensorMapping>(
            decision: RoutingDecision<'_, M::InvocationTensor>,
        ) -> RoutingDecision<'_, M::ObserverTensor> {
            RoutingDecision {
                ids: M::borrow_tensor(decision.ids),
                coefficients: M::borrow_tensor(decision.coefficients),
            }
        }
        self.inner
            .routing_applied(path, original.map(map::<M>), map::<M>(effective))
            .map_err(M::invocation_error)
    }

    fn routing_failed(&mut self, path: &str, message: &str) {
        self.inner.routing_failed(path, message);
    }

    fn observe(
        &mut self,
        path: &str,
        value: &M::InvocationTensor,
    ) -> Result<(), M::InvocationError> {
        self.inner
            .observe(path, M::borrow_tensor(value))
            .map_err(M::invocation_error)
    }

    fn observe_replica(
        &mut self,
        path: &str,
        value: &M::InvocationTensor,
    ) -> Result<(), M::InvocationError> {
        self.inner
            .observe_replica(path, M::borrow_tensor(value))
            .map_err(M::invocation_error)
    }

    fn observe_generated(
        &mut self,
        path: &str,
        prototype: &M::InvocationTensor,
        source: &GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<M::InvocationTensor, M::InvocationError>,
    ) -> Result<(), M::InvocationError> {
        self.inner
            .observe_generated(path, M::borrow_tensor(prototype), source, &mut || {
                generate()
                    .map(M::generated_tensor)
                    .map_err(M::observer_error)
            })
            .map_err(M::invocation_error)
    }

    fn intervene(
        &mut self,
        path: &str,
        value: &M::InvocationTensor,
    ) -> Result<Option<M::InvocationTensor>, M::InvocationError> {
        self.inner
            .intervene(path, M::borrow_tensor(value))
            .map(|replacement| replacement.map(M::replacement_tensor))
            .map_err(M::invocation_error)
    }

    fn observe_routing(
        &mut self,
        routing: RoutingObservation<'_, M::InvocationTensor>,
    ) -> Result<(), M::InvocationError> {
        self.inner
            .observe_routing(RoutingObservation {
                path: routing.path,
                selected_experts: M::borrow_tensor(routing.selected_experts),
                selected_scores: M::borrow_tensor(routing.selected_scores),
                coefficients: M::borrow_tensor(routing.coefficients),
                routed_output: M::borrow_tensor(routing.routed_output),
                local_routed_output: routing.local_routed_output.map(M::borrow_tensor),
                reduced_routed_output: routing.reduced_routed_output.map(M::borrow_tensor),
                shared_output: routing.shared_output.map(M::borrow_tensor),
                combined_output: routing.combined_output.map(M::borrow_tensor),
                expert_count: routing.expert_count,
            })
            .map_err(M::invocation_error)
    }
}

impl<O, M> MappedActivationObserver<'_, O, M>
where
    M: ObserverTensorMapping,
    O: ActivationObserver<M::ObserverTensor, M::ObserverError> + ?Sized,
{
    fn routed(&mut self) -> Result<Option<&mut dyn RoutedUnitObserver<M::ObserverTensor>>, Error> {
        let path = self
            .routed_path
            .as_deref()
            .ok_or_else(|| Error::backend("missing routed observer path"))?;
        self.inner
            .routed_unit_observer(path)
            .map_err(Error::backend_source)
    }
}

impl<O, M> RoutedUnitObserver<M::InvocationTensor> for MappedActivationObserver<'_, O, M>
where
    M: ObserverTensorMapping,
    O: ActivationObserver<M::ObserverTensor, M::ObserverError> + ?Sized,
{
    fn begin_invocation(
        &mut self,
        invocation: &RoutedUnitInvocation<'_, M::InvocationTensor>,
    ) -> Result<(), Error> {
        self.routed_invocation_active = true;
        match self.routed()? {
            Some(observer) => observer.begin_invocation(&RoutedUnitInvocation {
                input: M::borrow_tensor(invocation.input),
                origins: invocation.origins,
                unit_coordinates: invocation.unit_coordinates,
            }),
            None => Ok(()),
        }
    }

    fn finish_invocation(&mut self, success: bool) -> Result<(), Error> {
        self.routed_invocation_active = false;
        match self.routed()? {
            Some(observer) => observer.finish_invocation(success),
            None => Ok(()),
        }
    }

    fn invocation_active(&self) -> bool {
        self.routed_invocation_active
    }

    fn observe(&mut self, batch: &RoutedUnitBatch<'_, M::InvocationTensor>) -> Result<(), Error> {
        match self.routed()? {
            Some(observer) => observer.observe(&batch.map_tensors(M::borrow_tensor)),
            None => Ok(()),
        }
    }

    fn intervene(
        &mut self,
        batch: &RoutedUnitBatch<'_, M::InvocationTensor>,
    ) -> Result<Option<M::InvocationTensor>, Error> {
        match self.routed()? {
            Some(observer) => observer
                .intervene(&batch.map_tensors(M::borrow_tensor))
                .map(|value| value.map(M::replacement_tensor)),
            None => Ok(None),
        }
    }

    fn observe_effective(
        &mut self,
        batch: &RoutedUnitBatch<'_, M::InvocationTensor>,
    ) -> Result<(), Error> {
        match self.routed()? {
            Some(observer) => observer.observe_effective(&batch.map_tensors(M::borrow_tensor)),
            None => Ok(()),
        }
    }
}

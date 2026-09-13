//! Lossless observer failure ownership across an execution error boundary.
use super::*;

/// Borrows an observer across two error domains without replacing its first
/// failure with an execution-layer string. All reservation, routing and
/// transaction callbacks retain their original observer implementation.
pub struct ObserverErrorBridge<'a, O: ?Sized, E, ToObserver, ToExecution> {
    observer: &'a mut O,
    failure: Option<E>,
    to_observer: ToObserver,
    to_execution: ToExecution,
}

impl<'a, O: ?Sized, E, ToObserver, ToExecution>
    ObserverErrorBridge<'a, O, E, ToObserver, ToExecution>
{
    /// Creates a borrowed bridge. The execution error is only a propagation
    /// signal; `resolve` returns the retained original observer failure.
    pub fn new(observer: &'a mut O, to_observer: ToObserver, to_execution: ToExecution) -> Self {
        Self {
            observer,
            failure: None,
            to_observer,
            to_execution,
        }
    }

    /// Resolves the enclosing operation without losing a recorded observer
    /// failure, even if an intermediate adapter swallowed its propagation signal.
    /// Native completion and state recovery remain the enclosing owner's duty.
    pub fn resolve<V>(self, result: Result<V, E>) -> Result<V, E> {
        match self.failure {
            Some(error) => Err(error),
            None => result,
        }
    }
}

fn result<V, E, X>(
    failure: &mut Option<E>,
    convert: &mut impl FnMut(&E) -> X,
    value: Result<V, E>,
) -> Result<V, X> {
    value.map_err(|error| {
        let signal = convert(&error);
        if failure.is_none() {
            *failure = Some(error);
        }
        signal
    })
}

impl<T, X, E, O, ToObserver, ToExecution> ActivationObserver<T, X>
    for ObserverErrorBridge<'_, O, E, ToObserver, ToExecution>
where
    O: ActivationObserver<T, E> + ?Sized,
    ToObserver: FnMut(X) -> E,
    ToExecution: FnMut(&E) -> X,
{
    fn transactional(&self) -> bool {
        self.observer.transactional()
    }
    fn prepare_transaction(
        &mut self,
        epoch: eredu_core::DistributedCommitEpoch,
        pass: crate::ExpertPass,
    ) -> Result<(), X> {
        let value = self.observer.prepare_transaction(epoch, pass);
        result(&mut self.failure, &mut self.to_execution, value)
    }
    fn coordinate_transaction(
        &mut self,
        epoch: eredu_core::DistributedCommitEpoch,
    ) -> Result<(), X> {
        let value = self.observer.coordinate_transaction(epoch);
        result(&mut self.failure, &mut self.to_execution, value)
    }
    fn complete_transaction(&mut self, epoch: eredu_core::DistributedCommitEpoch) -> Result<(), X> {
        let value = self.observer.complete_transaction(epoch);
        result(&mut self.failure, &mut self.to_execution, value)
    }
    fn finish_transaction(&mut self, epoch: eredu_core::DistributedCommitEpoch, committed: bool) {
        self.observer.finish_transaction(epoch, committed);
    }
    fn routing_control(
        &mut self,
        path: &str,
        rows: u64,
    ) -> Result<Option<eredu_nn::routing_intervention::GroupSelectionControl>, X> {
        let value = self.observer.routing_control(path, rows);
        result(&mut self.failure, &mut self.to_execution, value)
    }
    fn routing_applied(
        &mut self,
        path: &str,
        original: Option<RoutingDecision<'_, T>>,
        effective: RoutingDecision<'_, T>,
    ) -> Result<(), X> {
        let value = self.observer.routing_applied(path, original, effective);
        result(&mut self.failure, &mut self.to_execution, value)
    }
    fn routing_failed(&mut self, path: &str, message: &str) {
        self.observer.routing_failed(path, message);
    }
    fn routed_unit_observer(
        &mut self,
        path: &str,
    ) -> Result<Option<&mut dyn crate::RoutedUnitObserver<T>>, X> {
        let value = self.observer.routed_unit_observer(path);
        result(&mut self.failure, &mut self.to_execution, value)
    }
    fn finish(&mut self) -> Result<(), X> {
        let value = self.observer.finish();
        result(&mut self.failure, &mut self.to_execution, value)
    }
    fn observe(&mut self, path: &str, tensor: &T) -> Result<(), X> {
        let value = self.observer.observe(path, tensor);
        result(&mut self.failure, &mut self.to_execution, value)
    }
    fn observe_replica(&mut self, path: &str, tensor: &T) -> Result<(), X> {
        let value = self.observer.observe_replica(path, tensor);
        result(&mut self.failure, &mut self.to_execution, value)
    }
    fn observe_generated(
        &mut self,
        path: &str,
        prototype: &T,
        source: &eredu_core::capture::GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<T, X>,
    ) -> Result<(), X> {
        let convert = &mut self.to_observer;
        let value = self
            .observer
            .observe_generated(path, prototype, source, &mut || {
                generate().map_err(&mut *convert)
            });
        result(&mut self.failure, &mut self.to_execution, value)
    }
    fn intervene(&mut self, path: &str, tensor: &T) -> Result<Option<T>, X> {
        let value = self.observer.intervene(path, tensor);
        result(&mut self.failure, &mut self.to_execution, value)
    }
    fn observe_routing(&mut self, event: RoutingObservation<'_, T>) -> Result<(), X> {
        let value = self.observer.observe_routing(event);
        result(&mut self.failure, &mut self.to_execution, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, rc::Rc};
    #[derive(Debug)]
    struct Failure(Rc<u32>);
    struct Observer {
        remaining: u64,
        failure: Rc<u32>,
        values: Vec<i32>,
        events: Vec<&'static str>,
    }
    impl crate::RoutedUnitObserver<i32> for Observer {
        fn observe(&mut self, _: &crate::RoutedUnitBatch<'_, i32>) -> Result<(), eredu_nn::Error> {
            Ok(())
        }
        fn invocation_active(&self) -> bool {
            true
        }
    }
    impl ActivationObserver<i32, Failure> for Observer {
        fn transactional(&self) -> bool {
            true
        }
        fn prepare_transaction(
            &mut self,
            _: eredu_core::DistributedCommitEpoch,
            _: crate::ExpertPass,
        ) -> Result<(), Failure> {
            self.events.push("prepare");
            Ok(())
        }
        fn coordinate_transaction(
            &mut self,
            _: eredu_core::DistributedCommitEpoch,
        ) -> Result<(), Failure> {
            self.events.push("coordinate");
            Ok(())
        }
        fn complete_transaction(
            &mut self,
            _: eredu_core::DistributedCommitEpoch,
        ) -> Result<(), Failure> {
            self.events.push("complete");
            Ok(())
        }
        fn finish_transaction(&mut self, _: eredu_core::DistributedCommitEpoch, committed: bool) {
            self.events.push(if committed { "commit" } else { "abort" });
        }
        fn observe(&mut self, _: &str, value: &i32) -> Result<(), Failure> {
            self.values.push(*value);
            Ok(())
        }
        fn observe_replica(&mut self, _: &str, _: &i32) -> Result<(), Failure> {
            self.events.push("replica");
            Ok(())
        }
        fn observe_generated(
            &mut self,
            path: &str,
            _: &i32,
            source: &eredu_core::capture::GeneratedCaptureSource,
            generate: &mut dyn FnMut() -> Result<i32, Failure>,
        ) -> Result<(), Failure> {
            self.remaining = self
                .remaining
                .checked_sub(source.creation_bytes)
                .ok_or_else(|| Failure(Rc::clone(&self.failure)))?;
            self.observe(path, &generate()?)
        }
        fn intervene(&mut self, _: &str, value: &i32) -> Result<Option<i32>, Failure> {
            Ok(Some(value + 7))
        }
        fn routed_unit_observer(
            &mut self,
            _: &str,
        ) -> Result<Option<&mut dyn crate::RoutedUnitObserver<i32>>, Failure> {
            self.events.push("units");
            Ok(Some(self))
        }
        fn routing_failed(&mut self, _: &str, _: &str) {
            self.events.push("routing_failed");
        }
        fn finish(&mut self) -> Result<(), Failure> {
            self.events.push("finish");
            Ok(())
        }
    }
    #[test]
    fn reservations_are_forwarded_before_generation_and_original_failure_survives() {
        let identity = Rc::new(41);
        let mut observer = Observer {
            remaining: 16,
            failure: Rc::clone(&identity),
            values: vec![],
            events: vec![],
        };
        let generated = Cell::new(0);
        let source = eredu_core::capture::GeneratedCaptureSource {
            creation_bytes: 16,
            source_dtype: None,
        };
        let mut bridge = ObserverErrorBridge::new(
            &mut observer,
            |n: u32| Failure(Rc::new(n)),
            |e: &Failure| *e.0,
        );
        bridge
            .observe_generated("actual", &0, &source, &mut || {
                generated.set(generated.get() + 1);
                Ok(13)
            })
            .unwrap();
        assert_eq!(bridge.intervene("actual", &13).unwrap(), Some(20));
        assert_eq!(
            bridge.observe_generated("denied", &0, &source, &mut || {
                generated.set(generated.get() + 1);
                Ok(99)
            }),
            Err(41)
        );
        // An enclosing adapter's string or even a swallowed signal cannot replace
        // the observer's original non-Clone error object.
        let error = bridge.resolve(Ok(())).unwrap_err();
        assert!(Rc::ptr_eq(&error.0, &identity));
        assert_eq!(generated.get(), 1);
        assert_eq!(observer.remaining, 0);
        assert_eq!(observer.values, [13]);
    }
    #[test]
    fn generated_failures_and_transactional_sparse_callbacks_keep_their_owner() {
        let mut observer = Observer {
            remaining: 16,
            failure: Rc::new(1),
            values: vec![],
            events: vec![],
        };
        let mut bridge = ObserverErrorBridge::new(
            &mut observer,
            |n: u32| Failure(Rc::new(n)),
            |e: &Failure| *e.0,
        );
        let epoch = eredu_core::DistributedCommitEpoch::new(7).unwrap();
        assert!(bridge.transactional());
        bridge
            .prepare_transaction(epoch, crate::ExpertPass::Prefill)
            .unwrap();
        bridge.coordinate_transaction(epoch).unwrap();
        assert!(bridge
            .routed_unit_observer("expert")
            .unwrap()
            .unwrap()
            .invocation_active());
        bridge.observe_replica("replica", &0).unwrap();
        let source = eredu_core::capture::GeneratedCaptureSource {
            creation_bytes: 16,
            source_dtype: None,
        };
        assert_eq!(
            bridge.observe_generated("failed", &0, &source, &mut || Err(73)),
            Err(73)
        );
        bridge.routing_failed("expert", "failure");
        bridge.finish().unwrap();
        bridge.complete_transaction(epoch).unwrap();
        bridge.finish_transaction(epoch, false);
        let error = bridge
            .resolve(Err::<(), _>(Failure(Rc::new(99))))
            .unwrap_err();
        assert_eq!(*error.0, 73);
        assert_eq!(observer.remaining, 0);
        assert_eq!(
            observer.events,
            [
                "prepare",
                "coordinate",
                "units",
                "replica",
                "routing_failed",
                "finish",
                "complete",
                "abort"
            ]
        );
    }
}

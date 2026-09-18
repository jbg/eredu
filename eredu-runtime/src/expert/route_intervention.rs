//! Selection remains in the architecture-configured selector. These adapters
//! only connect portable control/evidence hooks before provider execution.
use super::RoutedExpertProvider;
use eredu_nn::{Error, GroupSelection, GroupSelectionOperator, GroupedNeuralBackend, Tensor};

fn token_rows<T: Tensor>(input: &T) -> Result<u64, Error> {
    let shape = input.shape();
    let rows = shape
        .get(..shape.len().saturating_sub(1))
        .ok_or_else(|| Error::backend("router input has no token axes"))?;
    rows.iter()
        .try_fold(1u64, |n, value| {
            u64::try_from(*value)
                .ok()
                .and_then(|value| n.checked_mul(value))
        })
        .ok_or_else(|| Error::backend("routing token-row shape overflow"))
}

/// Selects through an architecture-owned selector before an observed provider's
/// expert call. Ordinary providers take the original lightweight select path.
pub fn select_routes_with_provider<B, P>(
    selector: &mut B::Selector,
    input: &B::Tensor,
    context: &<B::Tensor as Tensor>::Context,
    provider: &mut P,
    bank: crate::RoutedBankId,
) -> Result<GroupSelection<B::Tensor>, Error>
where
    B: GroupedNeuralBackend,
    P: RoutedExpertProvider<B>,
    P::Error: std::fmt::Display,
{
    let Some(control) = provider
        .routing_control(bank, token_rows(input)?)
        .map_err(Error::backend_retained_source)?
    else {
        if provider.routing_unmodified_interest(bank) == crate::RoutingUnmodifiedInterest::None {
            return selector.select(input, context);
        }
        let result: Result<_, Error> = (|| {
            let selection = selector.select(input, context)?;
            provider
                .routing_unmodified(bank, (&selection).into())
                .map_err(Error::backend_retained_source)?;
            Ok(selection)
        })();
        if let Err(error) = &result {
            provider.routing_failed(bank, &error.to_string());
        }
        return result;
    };
    let result: Result<_, Error> = (|| {
        let selection = selector.select_intervened(input, &control, context)?;
        provider
            .routing_applied(
                bank,
                selection.original.as_ref().map(Into::into),
                (&selection.effective).into(),
            )
            .map_err(Error::backend_retained_source)?;
        Ok(selection.effective)
    })();
    if let Err(error) = &result {
        provider.routing_failed(bank, &error.to_string());
    }
    result
}

/// Selects with a direct architecture observation hook, preserving shared-expert
/// work outside routed control. No unmodified model forward pass is performed.
pub fn select_routes_with_observer<T, S, O>(
    selector: &mut S,
    input: &T,
    context: &T::Context,
    path: &str,
    observer: &mut O,
) -> Result<GroupSelection<T>, Error>
where
    T: Tensor,
    S: GroupSelectionOperator<T>,
    O: crate::ActivationObserver<T, Error> + ?Sized,
{
    let Some(control) = observer.routing_control(path, token_rows(input)?)? else {
        if observer.routing_unmodified_interest(path) == crate::RoutingUnmodifiedInterest::None {
            return selector.select(input, context);
        }
        let result: Result<_, Error> = (|| {
            let selection = selector.select(input, context)?;
            observer.routing_unmodified(path, (&selection).into())?;
            Ok(selection)
        })();
        if let Err(error) = &result {
            observer.routing_failed(path, &error.to_string());
        }
        return result;
    };
    let result: Result<_, Error> = (|| {
        let selection = selector.select_intervened(input, &control, context)?;
        observer.routing_applied(
            path,
            selection.original.as_ref().map(Into::into),
            (&selection.effective).into(),
        )?;
        Ok(selection.effective)
    })();
    if let Err(error) = &result {
        observer.routing_failed(path, &error.to_string());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_nn::{workspace::*, ParameterVisitor, ParameterVisitorMut, Parameterized};
    use std::{cell::RefCell, rc::Rc};
    #[derive(Debug)]
    struct Unknown;
    impl WorkspaceMechanisms for Unknown {
        fn operation_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, Error> {
            Ok(None)
        }
    }
    #[derive(Clone, Debug)]
    struct Selector {
        log: Rc<RefCell<Vec<&'static str>>>,
        fail: bool,
    }
    impl Parameterized<WorkspaceTensor> for Selector {
        fn visit_parameter_sources<'a, V: eredu_nn::ParameterSourceVisitor<'a, WorkspaceTensor>>(&'a self, _: &mut V) -> Result<(), eredu_nn::ParameterSourceError> {
 let mut __source_result = Ok(());

 __source_result
}
        fn visit_parameters_mut<'a, V: ParameterVisitorMut<'a, WorkspaceTensor>>(
            &'a mut self,
            _: &mut V,
        ) {
        }
        fn set_trainable(&mut self, _: bool) {}
    }
    impl GroupSelectionOperator<WorkspaceTensor> for Selector {
        fn select(
            &mut self,
            input: &WorkspaceTensor,
            _: &WorkspaceContext,
        ) -> Result<GroupSelection<WorkspaceTensor>, Error> {
            self.log.borrow_mut().push("select");
            if self.fail {
                return Err(Error::backend("actual selector failure"));
            }
            Ok(GroupSelection::new(
                input.clone(),
                input.clone(),
                input.clone(),
            ))
        }
        fn select_indices(
            &mut self,
            _: &WorkspaceTensor,
            _: &WorkspaceTensor,
            _: &WorkspaceContext,
        ) -> Result<GroupSelection<WorkspaceTensor>, Error> {
            panic!("ordinary selection must not be replayed")
        }
    }
    struct Observer {
        log: Rc<RefCell<Vec<&'static str>>>,
        fail: bool,
        interest: crate::RoutingUnmodifiedInterest,
        retained: Option<WorkspaceTensor>,
    }
    impl crate::ActivationObserver<WorkspaceTensor, Error> for Observer {
        fn routing_unmodified_interest(&self, _: &str) -> crate::RoutingUnmodifiedInterest {
            self.interest
        }
        fn routing_unmodified(
            &mut self,
            path: &str,
            value: crate::RoutingDecision<'_, WorkspaceTensor>,
        ) -> Result<(), Error> {
            assert_eq!(path, "router");
            assert_eq!(value.ids.shape(), [2, 2]);
            assert_ne!(self.interest, crate::RoutingUnmodifiedInterest::None);
            if self.interest == crate::RoutingUnmodifiedInterest::Values {
                self.retained = Some(value.ids.clone());
            }
            self.log.borrow_mut().push("unmodified");
            if self.fail {
                return Err(Error::backend("notification failure"));
            }
            Ok(())
        }
        fn routing_applied(
            &mut self,
            _: &str,
            _: Option<crate::RoutingDecision<'_, WorkspaceTensor>>,
            _: crate::RoutingDecision<'_, WorkspaceTensor>,
        ) -> Result<(), Error> {
            panic!("no control was requested")
        }
        fn routing_failed(&mut self, _: &str, message: &str) {
            assert!(message.contains("failure"));
            self.log.borrow_mut().push("failed");
        }
        fn observe(&mut self, _: &str, _: &WorkspaceTensor) -> Result<(), Error> {
            Ok(())
        }
    }
    #[test]
    fn ordinary_selector_notifies_once_before_dispatch_and_preserves_failure_order() {
        let context = WorkspaceContext::new(Unknown);
        let input = WorkspaceTensor::existing(
            WorkspaceLayout::new(&[2, 2], WorkspaceDtype::Float32).unwrap(),
            &context,
        )
        .unwrap();
        for mode in 0..3 {
            let log = Rc::new(RefCell::new(Vec::new()));
            let mut selector = Selector {
                log: log.clone(),
                fail: mode == 1,
            };
            let mut observer = Observer {
                log: log.clone(),
                fail: mode == 2,
                interest: crate::RoutingUnmodifiedInterest::Values,
                retained: None,
            };
            let result = select_routes_with_observer(
                &mut selector,
                &input,
                &context,
                "router",
                &mut observer,
            );
            if result.is_ok() {
                log.borrow_mut().push("dispatch");
            }
            assert_eq!(
                *log.borrow(),
                match mode {
                    0 => vec!["select", "unmodified", "dispatch"],
                    1 => vec!["select", "failed"],
                    _ => vec!["select", "unmodified", "failed"],
                }
            );
        }
    }
    #[derive(Debug)]
    struct LoggedMechanisms(Rc<RefCell<Vec<&'static str>>>);
    impl WorkspaceMechanisms for LoggedMechanisms {
        fn operation_bound(
            &self,
            op: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, Error> {
            if matches!(op.kind, WorkspaceOperationKind::GroupSelection { .. }) {
                self.0.borrow_mut().push("select");
            }
            Ok(None)
        }
    }
    struct Provider {
        log: Rc<RefCell<Vec<&'static str>>>,
        fail: bool,
        interest: crate::RoutingUnmodifiedInterest,
        retained: Option<WorkspaceTensor>,
    }
    impl RoutedExpertProvider<WorkspaceBackend> for Provider {
        type Error = Error;
        fn routing_unmodified_interest(
            &self,
            _: crate::RoutedBankId,
        ) -> crate::RoutingUnmodifiedInterest {
            self.interest
        }
        fn routing_unmodified(
            &mut self,
            bank: crate::RoutedBankId,
            value: crate::RoutingDecision<'_, WorkspaceTensor>,
        ) -> Result<(), Error> {
            assert_eq!(bank.value(), 7);
            assert_eq!(value.ids.shape(), [2, 2]);
            assert_eq!(value.ids.layout().dtype(), WorkspaceDtype::Uint32);
            assert_ne!(self.interest, crate::RoutingUnmodifiedInterest::None);
            if self.interest == crate::RoutingUnmodifiedInterest::Values {
                self.retained = Some(value.ids.clone());
            }
            self.log.borrow_mut().push("unmodified");
            if self.fail {
                Err(Error::backend("notification failure"))
            } else {
                Ok(())
            }
        }
        fn routing_failed(&mut self, _: crate::RoutedBankId, _: &str) {
            self.log.borrow_mut().push("failed");
        }
        fn forward_grouped(
            &mut self,
            _: &mut <WorkspaceBackend as GroupedNeuralBackend>::GatedProductGroups,
            _: crate::RoutedExpertRequest<'_, '_, WorkspaceTensor>,
            _: &WorkspaceContext,
        ) -> Result<WorkspaceTensor, Error> {
            panic!("dispatch belongs to caller")
        }
        fn forward_linear_routed(
            &mut self,
            _: &mut <WorkspaceBackend as GroupedNeuralBackend>::LinearGroups,
            _: crate::RoutedExpertRequest<'_, '_, WorkspaceTensor>,
            _: &WorkspaceContext,
        ) -> Result<WorkspaceTensor, Error> {
            panic!("dispatch belongs to caller")
        }
        fn forward_relu2_routed(
            &mut self,
            _: &mut <WorkspaceBackend as GroupedNeuralBackend>::Relu2Groups,
            _: crate::RoutedExpertRequest<'_, '_, WorkspaceTensor>,
            _: &WorkspaceContext,
        ) -> Result<WorkspaceTensor, Error> {
            panic!("dispatch belongs to caller")
        }
    }
    #[test]
    fn provider_selector_forwards_one_actual_decision_before_dispatch() {
        use eredu_nn::{
            GroupScoring, LinearFormatSpec, ParameterSpec, TopKGroupSelectionSpec,
            TopKGroupSelectorSpec,
        };
        for (interest, fail) in [
            (crate::RoutingUnmodifiedInterest::None, false),
            (crate::RoutingUnmodifiedInterest::None, true),
            (crate::RoutingUnmodifiedInterest::Metadata, false),
            (crate::RoutingUnmodifiedInterest::Values, false),
            (crate::RoutingUnmodifiedInterest::Values, true),
        ] {
            let log = Rc::new(RefCell::new(Vec::new()));
            let context = WorkspaceContext::new(LoggedMechanisms(log.clone()));
            let spec = TopKGroupSelectorSpec::new(
                2,
                ParameterSpec::trainable("router.weight").unwrap(),
                LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense).unwrap(),
                TopKGroupSelectionSpec::new(4, 2, GroupScoring::Softmax, true).unwrap(),
            )
            .unwrap();
            let mut selector = WorkspaceBackend::top_k_group_selector(spec, &context).unwrap();
            let input = WorkspaceTensor::existing(
                WorkspaceLayout::new(&[2, 2], WorkspaceDtype::Float32).unwrap(),
                &context,
            )
            .unwrap();
            context.begin_span();
            log.borrow_mut().clear();
            let mut provider = Provider {
                log: log.clone(),
                fail,
                interest,
                retained: None,
            };
            let selected = select_routes_with_provider::<WorkspaceBackend, _>(
                &mut selector,
                &input,
                &context,
                &mut provider,
                crate::RoutedBankId::new(7),
            );
            if let Ok(selected) = selected {
                assert_eq!(selected.group_indices().shape(), [2, 2]);
                log.borrow_mut().push("dispatch");
            }
            assert_eq!(
                *log.borrow(),
                if interest == crate::RoutingUnmodifiedInterest::None {
                    vec!["select", "dispatch"]
                } else if fail {
                    vec!["select", "unmodified", "failed"]
                } else {
                    vec!["select", "unmodified", "dispatch"]
                }
            );
            assert_eq!(
                provider.retained.is_some(),
                interest == crate::RoutingUnmodifiedInterest::Values
            );
        }
    }

    #[test]
    fn direct_ordinary_interest_distinguishes_noop_metadata_and_retained_values() {
        let context = WorkspaceContext::new(Unknown);
        let input = WorkspaceTensor::existing(
            WorkspaceLayout::new(&[2, 2], WorkspaceDtype::Float32).unwrap(),
            &context,
        )
        .unwrap();
        for interest in [
            crate::RoutingUnmodifiedInterest::None,
            crate::RoutingUnmodifiedInterest::Metadata,
            crate::RoutingUnmodifiedInterest::Values,
        ] {
            let log = Rc::new(RefCell::new(Vec::new()));
            let mut selector = Selector {
                log: log.clone(),
                fail: false,
            };
            let mut observer = Observer {
                log: log.clone(),
                fail: false,
                interest,
                retained: None,
            };
            select_routes_with_observer(&mut selector, &input, &context, "router", &mut observer)
                .unwrap();
            assert_eq!(
                *log.borrow(),
                if interest == crate::RoutingUnmodifiedInterest::None {
                    vec!["select"]
                } else {
                    vec!["select", "unmodified"]
                }
            );
            assert_eq!(
                observer.retained.is_some(),
                interest == crate::RoutingUnmodifiedInterest::Values
            );
        }
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut selector = Selector {
            log: log.clone(),
            fail: true,
        };
        let mut ordinary = crate::NoopObserver;
        assert!(select_routes_with_observer(
            &mut selector,
            &input,
            &context,
            "router",
            &mut ordinary
        )
        .is_err());
        assert_eq!(*log.borrow(), vec!["select"]);
    }
}

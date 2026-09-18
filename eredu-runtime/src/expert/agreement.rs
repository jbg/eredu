//! Execution votes around whole provider calls, including idle addressable owners.
use super::*;
use eredu_nn::Error;

/// A peer rejected its local provider work before the next model collective.
#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("a peer rejected provider work before the next model collective")]
pub struct ProviderAgreementRejected;

fn finish<T, E: std::error::Error + Send + Sync + 'static>(
    local: Result<T, E>,
    agree: &mut impl FnMut(bool) -> Result<bool, Error>,
) -> Result<T, Error> {
    let agreement = agree(local.is_ok());
    match (local, agreement) {
        (Err(error), _) => Err(Error::backend_retained_source(error)),
        (Ok(_), Err(error)) => Err(error),
        (Ok(_), Ok(false)) => Err(Error::backend_retained_source(ProviderAgreementRejected)),
        (Ok(output), Ok(true)) => Ok(output),
    }
}

/// Borrows a provider and a retained execution-group vote. This adds no tensor
/// observation; the vote runs even when local execution returns an error.
pub struct AgreeingRoutedExpertProvider<'a, P, F> {
    provider: &'a mut P,
    agree: F,
}
impl<'a, P, F: FnMut(bool) -> Result<bool, Error>> AgreeingRoutedExpertProvider<'a, P, F> {
    /// Binds the exact participant agreement selected for these provider calls.
    pub fn new(provider: &'a mut P, agree: F) -> Self {
        Self { provider, agree }
    }
}
impl<B, P, F> RoutedExpertProvider<B> for AgreeingRoutedExpertProvider<'_, P, F>
where
    B: GroupedNeuralBackend,
    P: RoutedExpertProvider<B>,
    F: FnMut(bool) -> Result<bool, Error>,
{
    type Error = Error;
    fn routing_control(
        &mut self,
        bank: RoutedBankId,
        rows: u64,
    ) -> Result<Option<eredu_nn::routing_intervention::GroupSelectionControl>, Error> {
        self.provider
            .routing_control(bank, rows)
            .map_err(Error::backend_retained_source)
    }
    fn routing_unmodified_interest(&self, bank: RoutedBankId) -> crate::RoutingUnmodifiedInterest {
        self.provider.routing_unmodified_interest(bank)
    }
    fn routing_unmodified(
        &mut self,
        bank: RoutedBankId,
        effective: crate::RoutingDecision<'_, B::Tensor>,
    ) -> Result<(), Error> {
        self.provider
            .routing_unmodified(bank, effective)
            .map_err(Error::backend_retained_source)
    }

    fn routing_applied(
        &mut self,
        bank: RoutedBankId,
        original: Option<crate::RoutingDecision<'_, B::Tensor>>,
        effective: crate::RoutingDecision<'_, B::Tensor>,
    ) -> Result<(), Error> {
        self.provider
            .routing_applied(bank, original, effective)
            .map_err(Error::backend_retained_source)
    }
    fn routing_failed(&mut self, bank: RoutedBankId, message: &str) {
        self.provider.routing_failed(bank, message);
    }
    fn forward_grouped(
        &mut self,
        bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let local = self.provider.forward_grouped(bank, request, context);
        finish(local, &mut self.agree)
    }
    fn forward_compact_grouped(
        &mut self,
        bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let local = self
            .provider
            .forward_compact_grouped(bank, request, context);
        finish(local, &mut self.agree)
    }
    fn forward_linear_routed(
        &mut self,
        bank: &mut B::LinearGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let local = self.provider.forward_linear_routed(bank, request, context);
        finish(local, &mut self.agree)
    }
    fn forward_relu2_routed(
        &mut self,
        bank: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let local = self.provider.forward_relu2_routed(bank, request, context);
        finish(local, &mut self.agree)
    }
}
impl<B, P, F> TensorParallelRoutedExpertProvider<B> for AgreeingRoutedExpertProvider<'_, P, F>
where
    B: GroupedNeuralBackend,
    P: TensorParallelRoutedExpertProvider<B>,
    F: FnMut(bool) -> Result<bool, Error>,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Error> {
        let local = self
            .provider
            .forward_grouped_tensor_parallel(bank, request, partitions, context);
        finish(local, &mut self.agree)
    }
    fn forward_compact_grouped_tensor_parallel(
        &mut self,
        bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Error> {
        let local = self
            .provider
            .forward_compact_grouped_tensor_parallel(bank, request, partitions, context);
        finish(local, &mut self.agree)
    }
    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        bank: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Error> {
        let local = self
            .provider
            .forward_relu2_routed_tensor_parallel(bank, request, partitions, context);
        finish(local, &mut self.agree)
    }
}

/// Agrees an entire owner-local invocation, including validation and zero-row work.
/// Use this around an exchanged provider so idle TP owners also enter its vote.
pub struct AgreeingAddressableExpertProvider<'a, P, F> {
    provider: &'a mut P,
    agree: F,
}
impl<'a, P, F: FnMut(bool) -> Result<bool, Error>> AgreeingAddressableExpertProvider<'a, P, F> {
    /// Binds the exact participant agreement selected for this owner invocation.
    pub fn new(provider: &'a mut P, agree: F) -> Self {
        Self { provider, agree }
    }
}
impl<T, P, F> AddressableExpertRouteProvider<T> for AgreeingAddressableExpertProvider<'_, P, F>
where
    P: AddressableExpertRouteProvider<T>,
    F: FnMut(bool) -> Result<bool, Error>,
{
    type Error = Error;
    fn execute_addressable_routes(
        &mut self,
        request: AddressableExpertRouteRequest<'_, T>,
    ) -> Result<T, Error> {
        let local = self.provider.execute_addressable_routes(request);
        finish(local, &mut self.agree)
    }
    fn execute_addressable_routes_tensor_parallel(
        &mut self,
        request: AddressableExpertRouteRequest<'_, T>,
    ) -> Result<RoutedExpertTensorParallelOutput<T>, Error> {
        let local = self
            .provider
            .execute_addressable_routes_tensor_parallel(request);
        finish(local, &mut self.agree)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Barrier,
    };

    #[derive(Debug, thiserror::Error)]
    #[error("provider fault on owner {0}")]
    struct ProviderFault(usize);
    #[derive(Debug, thiserror::Error)]
    #[error("agreement failed")]
    struct AgreementFault;

    struct Provider {
        owner: usize,
        fail: bool,
        calls: usize,
    }
    impl AddressableExpertRouteProvider<Vec<f32>> for Provider {
        type Error = ProviderFault;
        fn execute_addressable_routes(
            &mut self,
            request: AddressableExpertRouteRequest<'_, Vec<f32>>,
        ) -> Result<Vec<f32>, Self::Error> {
            self.calls += 1;
            if self.fail {
                return Err(ProviderFault(self.owner));
            }
            Ok(request.input.iter().map(|value| value * 2.).collect())
        }
        fn execute_addressable_routes_tensor_parallel(
            &mut self,
            request: AddressableExpertRouteRequest<'_, Vec<f32>>,
        ) -> Result<RoutedExpertTensorParallelOutput<Vec<f32>>, Self::Error> {
            let bias = vec![0.25; request.input.len()];
            self.execute_addressable_routes(request).map(|value| {
                RoutedExpertTensorParallelOutput::Partial(TensorParallelGroupedOutput::new(
                    value,
                    Some(bias),
                ))
            })
        }
    }
    fn has_source<E: std::error::Error + 'static>(
        mut error: &(dyn std::error::Error + 'static),
    ) -> bool {
        loop {
            if error.is::<E>() {
                return true;
            }
            let Some(next) = error.source() else {
                return false;
            };
            error = next;
        }
    }

    #[test]
    fn owner_agreement_includes_idle_work_preserves_causes_and_stops_downstream_collectives() {
        for failing_owner in [None, Some(0), Some(2)] {
            for agreement_fails in [false, true] {
                for tensor_parallel in [false, true] {
                    let ready = Barrier::new(3);
                    let failed = AtomicBool::new(false);
                    let votes = AtomicUsize::new(0);
                    let downstream = AtomicUsize::new(0);
                    std::thread::scope(|scope| {
                        for owner in 0..3 {
                            let (ready, failed, votes, downstream) =
                                (&ready, &failed, &votes, &downstream);
                            scope.spawn(move || {
                                let input = if owner == 2 {
                                    vec![]
                                } else {
                                    vec![owner as f32 + 2., -3.]
                                };
                                let coefficients = vec![1.; input.len()];
                                let groups = vec![owner; input.len()];
                                let local_groups = vec![0; input.len()];
                                let tags = (0..input.len()).collect::<Vec<_>>();
                                let counts = [input.len()];
                                let mut provider = Provider {
                                    owner,
                                    fail: failing_owner == Some(owner),
                                    calls: 0,
                                };
                                let mut agreed = AgreeingAddressableExpertProvider::new(
                                    &mut provider,
                                    |success| {
                                        if !success {
                                            failed.store(true, Ordering::SeqCst);
                                        }
                                        votes.fetch_add(1, Ordering::SeqCst);
                                        ready.wait();
                                        if agreement_fails {
                                            Err(Error::backend_retained_source(AgreementFault))
                                        } else {
                                            Ok(!failed.load(Ordering::SeqCst))
                                        }
                                    },
                                );
                                let request = AddressableExpertRouteRequest {
                                    unit_origins: RoutedUnitOrigins::new(&counts, &tags, 1)
                                        .unwrap(),
                                    bank: RoutedBankId::new(0),
                                    unit: 0,
                                    input: &input,
                                    global_experts: &groups,
                                    owner_local_experts: &local_groups,
                                    selected_scores: &coefficients,
                                    coefficients: &coefficients,
                                    pass: ExpertPass::Prefill,
                                    access: ExpertPass::Prefill.parameter_bank_access(),
                                    combination: ExpertRouteCombination::CoefficientWeightedSum,
                                };
                                let result = if tensor_parallel {
                                    agreed.execute_addressable_routes_tensor_parallel(request)
                                } else {
                                    agreed
                                        .execute_addressable_routes(request)
                                        .map(RoutedExpertTensorParallelOutput::Complete)
                                };
                                assert_eq!(
                                    provider.calls, 1,
                                    "idle owners also execute their scoped provider"
                                );
                                if failing_owner.is_some() || agreement_fails {
                                    let error = result
                                        .err()
                                        .expect("no result released after a local or peer failure");
                                    if failing_owner == Some(owner) {
                                        assert!(has_source::<ProviderFault>(&error));
                                    } else if agreement_fails {
                                        assert!(has_source::<AgreementFault>(&error));
                                    } else {
                                        assert!(has_source::<ProviderAgreementRejected>(&error));
                                    }
                                } else {
                                    assert_eq!(votes.load(Ordering::SeqCst), 3);
                                    downstream.fetch_add(1, Ordering::SeqCst);
                                    let expected =
                                        input.iter().map(|value| value * 2.).collect::<Vec<_>>();
                                    match result.unwrap() {
                                        RoutedExpertTensorParallelOutput::Complete(value) => {
                                            assert!(!tensor_parallel);
                                            assert_eq!(value, expected);
                                        }
                                        RoutedExpertTensorParallelOutput::Partial(value) => {
                                            assert!(tensor_parallel);
                                            assert_eq!(value.reducible(), &expected);
                                            assert_eq!(
                                                value.post_reduce(),
                                                Some(&vec![0.25; input.len()])
                                            );
                                        }
                                    }
                                }
                            });
                        }
                    });
                    assert_eq!(votes.load(Ordering::SeqCst), 3);
                    assert_eq!(
                        downstream.load(Ordering::SeqCst),
                        if failing_owner.is_some() || agreement_fails {
                            0
                        } else {
                            3
                        }
                    );
                }
            }
        }
    }
}

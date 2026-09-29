//! Independent routed-bank dispatch within a logical layer.
use super::*;
use std::collections::BTreeMap;

/// Failure before or during dispatch to a declared bank.
#[derive(Debug, thiserror::Error)]
pub enum RoutedBankProviderError<E> {
    /// No provider was prepared for this architecture bank identity.
    #[error("no provider for routed bank {0:?}")]
    Missing(RoutedBankId),
    /// The selected bank's acquisition, execution, or completion failed.
    #[error("routed bank failed: {0}")]
    Provider(#[source] E),
}

/// Independently identified providers may share a bounded residency pool.
#[derive(Debug)]
pub struct RoutedBankProviders<P> {
    banks: BTreeMap<RoutedBankId, P>,
}
impl<P> RoutedBankProviders<P> {
    /// Retains exactly one provider per architecture bank identity.
    pub fn new(
        banks: impl IntoIterator<Item = (RoutedBankId, P)>,
    ) -> Result<Self, eredu_nn::Error> {
        let mut result = BTreeMap::new();
        for (id, provider) in banks {
            if result.insert(id, provider).is_some() {
                return Err(eredu_nn::Error::backend("duplicate routed provider bank"));
            }
        }
        if result.is_empty() {
            return Err(eredu_nn::Error::backend("empty routed provider collection"));
        }
        Ok(Self { banks: result })
    }
    /// Iterates providers without losing their architecture bank identity.
    pub fn banks(&self) -> &BTreeMap<RoutedBankId, P> {
        &self.banks
    }
    /// Reads one independently retained bank, including its telemetry.
    pub fn bank(&self, id: RoutedBankId) -> Option<&P> {
        self.banks.get(&id)
    }
    /// Mutably borrows one retained bank without changing identity.
    pub fn bank_mut(&mut self, id: RoutedBankId) -> Option<&mut P> {
        self.banks.get_mut(&id)
    }
    /// Moves selected auxiliary banks without cloning providers or residency owners.
    /// Empty requests preserve the collection; invalid requests leave it unchanged.
    pub fn split_off(&mut self, ids: &[RoutedBankId]) -> Result<Option<Self>, eredu_nn::Error> {
        if ids.is_empty() {
            return Ok(None);
        }
        let requested: std::collections::BTreeSet<_> = ids.iter().copied().collect();
        if requested.len() != ids.len()
            || requested.len() >= self.banks.len()
            || requested.iter().any(|id| !self.banks.contains_key(id))
        {
            return Err(eredu_nn::Error::backend(
                "auxiliary banks must be unique, present and leave a primary provider",
            ));
        }
        let banks = requested
            .into_iter()
            .map(|id| {
                (
                    id,
                    self.banks.remove(&id).expect("validated auxiliary bank"),
                )
            })
            .collect();
        Ok(Some(Self { banks }))
    }
    fn selected<E>(&mut self, id: RoutedBankId) -> Result<&mut P, RoutedBankProviderError<E>> {
        self.banks
            .get_mut(&id)
            .ok_or(RoutedBankProviderError::Missing(id))
    }
}

impl<B: GroupedNeuralBackend, P: ParameterProvider<B>> ParameterProvider<B>
    for RoutedBankProviders<P>
{
    type Error = RoutedBankProviderError<P::Error>;
    fn has_row_parameter(&self, parameter: &eredu_nn::ParameterId) -> bool {
        self.banks
            .values()
            .any(|provider| provider.has_row_parameter(parameter))
    }
    fn lookup_rows(
        &mut self,
        spec: &crate::RowLookupSpec,
        rows: &[u64],
        access: crate::ParameterBankAccess,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, crate::RowLookupError> {
        let mut matching = self
            .banks
            .values_mut()
            .filter(|provider| provider.has_row_parameter(&spec.parameter));
        let provider = matching
            .next()
            .ok_or_else(|| crate::RowLookupError::Missing(spec.parameter.clone()))?;
        if matching.next().is_some() {
            return Err(crate::RowLookupError::Specification(spec.parameter.clone()));
        }
        provider.lookup_rows(spec, rows, access, context)
    }
    fn routing_control(
        &mut self,
        bank: RoutedBankId,
        rows: u64,
    ) -> Result<Option<eredu_nn::routing_intervention::GroupSelectionControl>, Self::Error> {
        self.selected(bank)?
            .routing_control(bank, rows)
            .map_err(RoutedBankProviderError::Provider)
    }
    fn routing_applied(
        &mut self,
        bank: RoutedBankId,
        original: Option<crate::RoutingDecision<'_, B::Tensor>>,
        effective: crate::RoutingDecision<'_, B::Tensor>,
    ) -> Result<(), Self::Error> {
        self.selected(bank)?
            .routing_applied(bank, original, effective)
            .map_err(RoutedBankProviderError::Provider)
    }
    fn routing_failed(&mut self, bank: RoutedBankId, message: &str) {
        if let Some(provider) = self.banks.get_mut(&bank) {
            provider.routing_failed(bank, message);
        }
    }
    fn forward_grouped(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.selected(request.bank)?
            .forward_grouped(resident, request, context)
            .map_err(RoutedBankProviderError::Provider)
    }
    fn forward_compact_grouped(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.selected(request.bank)?
            .forward_compact_grouped(resident, request, context)
            .map_err(RoutedBankProviderError::Provider)
    }
    fn forward_linear_routed(
        &mut self,
        resident: &mut B::LinearGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.selected(request.bank)?
            .forward_linear_routed(resident, request, context)
            .map_err(RoutedBankProviderError::Provider)
    }
    fn forward_relu2_routed(
        &mut self,
        resident: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.selected(request.bank)?
            .forward_relu2_routed(resident, request, context)
            .map_err(RoutedBankProviderError::Provider)
    }
}
impl<B: GroupedNeuralBackend, P: TensorParallelParameterProvider<B>>
    TensorParallelParameterProvider<B> for RoutedBankProviders<P>
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        parts: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        self.selected(request.bank)?
            .forward_grouped_tensor_parallel(resident, request, parts, context)
            .map_err(RoutedBankProviderError::Provider)
    }
    fn forward_compact_grouped_tensor_parallel(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        parts: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        self.selected(request.bank)?
            .forward_compact_grouped_tensor_parallel(resident, request, parts, context)
            .map_err(RoutedBankProviderError::Provider)
    }
    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        resident: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        parts: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        self.selected(request.bank)?
            .forward_relu2_routed_tensor_parallel(resident, request, parts, context)
            .map_err(RoutedBankProviderError::Provider)
    }
}

// Erasure happens after portable preparation has selected each bank's equation
// and resource owner. The box retains that provider and its completion state.
impl<B: GroupedNeuralBackend, P: ParameterProvider<B> + ?Sized> ParameterProvider<B> for Box<P> {
    type Error = P::Error;
    fn has_row_parameter(&self, parameter: &eredu_nn::ParameterId) -> bool {
        self.as_ref().has_row_parameter(parameter)
    }
    fn lookup_rows(
        &mut self,
        spec: &crate::RowLookupSpec,
        rows: &[u64],
        access: crate::ParameterBankAccess,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, crate::RowLookupError> {
        self.as_mut().lookup_rows(spec, rows, access, context)
    }
    fn routing_control(
        &mut self,
        bank: RoutedBankId,
        rows: u64,
    ) -> Result<Option<eredu_nn::routing_intervention::GroupSelectionControl>, Self::Error> {
        self.as_mut().routing_control(bank, rows)
    }
    fn routing_applied(
        &mut self,
        bank: RoutedBankId,
        original: Option<crate::RoutingDecision<'_, B::Tensor>>,
        effective: crate::RoutingDecision<'_, B::Tensor>,
    ) -> Result<(), Self::Error> {
        self.as_mut().routing_applied(bank, original, effective)
    }
    fn routing_failed(&mut self, bank: RoutedBankId, message: &str) {
        self.as_mut().routing_failed(bank, message)
    }
    fn forward_grouped(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.as_mut().forward_grouped(resident, request, context)
    }
    fn forward_compact_grouped(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.as_mut()
            .forward_compact_grouped(resident, request, context)
    }
    fn forward_linear_routed(
        &mut self,
        resident: &mut B::LinearGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.as_mut()
            .forward_linear_routed(resident, request, context)
    }
    fn forward_relu2_routed(
        &mut self,
        resident: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.as_mut()
            .forward_relu2_routed(resident, request, context)
    }
}
impl<B: GroupedNeuralBackend, P: TensorParallelParameterProvider<B> + ?Sized>
    TensorParallelParameterProvider<B> for Box<P>
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        self.as_mut()
            .forward_grouped_tensor_parallel(resident, request, partitions, context)
    }
    fn forward_compact_grouped_tensor_parallel(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        self.as_mut()
            .forward_compact_grouped_tensor_parallel(resident, request, partitions, context)
    }
    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        resident: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        self.as_mut()
            .forward_relu2_routed_tensor_parallel(resident, request, partitions, context)
    }
}

#[cfg(test)]
mod split_tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    struct Owner(Arc<AtomicUsize>);
    impl Drop for Owner {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    #[test]
    fn auxiliary_split_moves_owners_and_preserves_failed_requests() {
        let drops = Arc::new(AtomicUsize::new(0));
        let ids = [
            RoutedBankId::new(0),
            RoutedBankId::new(2),
            RoutedBankId::new(5),
        ];
        let mut providers =
            RoutedBankProviders::new(ids.map(|id| (id, Owner(drops.clone())))).unwrap();
        for invalid in [
            vec![ids[1], ids[1]],
            vec![RoutedBankId::new(7)],
            ids.to_vec(),
        ] {
            assert!(providers.split_off(&invalid).is_err());
            assert_eq!(providers.banks().len(), 3);
            assert_eq!(drops.load(Ordering::SeqCst), 0);
        }
        assert!(providers.split_off(&[]).unwrap().is_none());
        let auxiliary = providers.split_off(&ids[1..]).unwrap().unwrap();
        assert_eq!(
            providers.banks().keys().copied().collect::<Vec<_>>(),
            ids[..1]
        );
        assert_eq!(
            auxiliary.banks().keys().copied().collect::<Vec<_>>(),
            ids[1..]
        );
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(providers);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        drop(auxiliary);
        assert_eq!(drops.load(Ordering::SeqCst), 3);
    }
}

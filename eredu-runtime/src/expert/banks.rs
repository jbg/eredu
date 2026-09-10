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
    Provider(E),
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
    fn selected<E>(&mut self, id: RoutedBankId) -> Result<&mut P, RoutedBankProviderError<E>> {
        self.banks
            .get_mut(&id)
            .ok_or(RoutedBankProviderError::Missing(id))
    }
}

impl<B: GroupedNeuralBackend, P: RoutedExpertProvider<B>> RoutedExpertProvider<B>
    for RoutedBankProviders<P>
{
    type Error = RoutedBankProviderError<P::Error>;
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
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.selected(request.bank)?
            .forward_grouped(resident, request, context)
            .map_err(RoutedBankProviderError::Provider)
    }
    fn forward_compact_grouped(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.selected(request.bank)?
            .forward_compact_grouped(resident, request, context)
            .map_err(RoutedBankProviderError::Provider)
    }
    fn forward_linear_routed(
        &mut self,
        resident: &mut B::LinearGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.selected(request.bank)?
            .forward_linear_routed(resident, request, context)
            .map_err(RoutedBankProviderError::Provider)
    }
    fn forward_relu2_routed(
        &mut self,
        resident: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.selected(request.bank)?
            .forward_relu2_routed(resident, request, context)
            .map_err(RoutedBankProviderError::Provider)
    }
}
impl<B: GroupedNeuralBackend, P: TensorParallelRoutedExpertProvider<B>>
    TensorParallelRoutedExpertProvider<B> for RoutedBankProviders<P>
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
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
        request: RoutedExpertRequest<'_, B::Tensor>,
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
        request: RoutedExpertRequest<'_, B::Tensor>,
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
impl<B: GroupedNeuralBackend, P: RoutedExpertProvider<B> + ?Sized> RoutedExpertProvider<B>
    for Box<P>
{
    type Error = P::Error;
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
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.as_mut().forward_grouped(resident, request, context)
    }
    fn forward_compact_grouped(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.as_mut()
            .forward_compact_grouped(resident, request, context)
    }
    fn forward_linear_routed(
        &mut self,
        resident: &mut B::LinearGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.as_mut()
            .forward_linear_routed(resident, request, context)
    }
    fn forward_relu2_routed(
        &mut self,
        resident: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.as_mut()
            .forward_relu2_routed(resident, request, context)
    }
}
impl<B: GroupedNeuralBackend, P: TensorParallelRoutedExpertProvider<B> + ?Sized>
    TensorParallelRoutedExpertProvider<B> for Box<P>
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        self.as_mut()
            .forward_grouped_tensor_parallel(resident, request, partitions, context)
    }
    fn forward_compact_grouped_tensor_parallel(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        self.as_mut()
            .forward_compact_grouped_tensor_parallel(resident, request, partitions, context)
    }
    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        resident: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        self.as_mut()
            .forward_relu2_routed_tensor_parallel(resident, request, partitions, context)
    }
}

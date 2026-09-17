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
    fn selected<E>(&mut self, id: RoutedBankId) -> Result<&mut P, RoutedBankProviderError<E>> {
        self.banks
            .get_mut(&id)
            .ok_or(RoutedBankProviderError::Missing(id))
    }
}

/// Allocation-free loan of an immutable resident provider table.
///
/// Each provider must itself implement the existing expert contract through an
/// immutable reference. Stateful addressable providers cannot acquire this loan.
#[derive(Debug)]
pub struct BorrowedRoutedBankProviders<'a, P> {
    banks: &'a BTreeMap<RoutedBankId, P>,
}
impl<P> RoutedBankProviders<P> {
    /// Borrows exact provider identities without copying the table or plans.
    pub fn borrowed(&self) -> BorrowedRoutedBankProviders<'_, P> {
        BorrowedRoutedBankProviders { banks: &self.banks }
    }
    fn lookup(&self, id: RoutedBankId) -> Option<&P> { self.banks.get(&id) }
    fn lookup_mut(&mut self, id: RoutedBankId) -> Option<&mut P> { self.banks.get_mut(&id) }
}
impl<'a, P> BorrowedRoutedBankProviders<'a, P> {
    fn selected<E>(&mut self, id: RoutedBankId) -> Result<&'a P, RoutedBankProviderError<E>> {
        self.banks.get(&id).ok_or(RoutedBankProviderError::Missing(id))
    }
    fn lookup(&self, id: RoutedBankId) -> Option<&'a P> { self.banks.get(&id) }
    fn lookup_mut(&mut self, id: RoutedBankId) -> Option<&'a P> { self.banks.get(&id) }
}

macro_rules! routed_bank_ordinary_dispatch { ($provider:ty $(, $mutable:tt)?) => {
    fn resident_unit_equations() -> bool {
        <$provider as RoutedExpertProvider<B>>::resident_unit_equations()
    }

    fn routing_control(
        &mut self,
        bank: RoutedBankId,
        rows: u64,
    ) -> Result<Option<eredu_nn::routing_intervention::GroupSelectionControl>, Self::Error> {
        { let $($mutable)? provider = self.selected(bank)?; provider.routing_control(bank, rows).map_err(RoutedBankProviderError::Provider) }
    }
    fn routing_unmodified_interest(&self, bank: RoutedBankId) -> crate::RoutingUnmodifiedInterest {
        self.lookup(bank)
            .map_or(crate::RoutingUnmodifiedInterest::None, |provider| {
                provider.routing_unmodified_interest(bank)
            })
    }
    fn routing_unmodified(
        &mut self,
        bank: RoutedBankId,
        effective: crate::RoutingDecision<'_, B::Tensor>,
    ) -> Result<(), Self::Error> {
        { let $($mutable)? provider = self.selected(bank)?; provider.routing_unmodified(bank, effective).map_err(RoutedBankProviderError::Provider) }
    }

    fn routing_applied(
        &mut self,
        bank: RoutedBankId,
        original: Option<crate::RoutingDecision<'_, B::Tensor>>,
        effective: crate::RoutingDecision<'_, B::Tensor>,
    ) -> Result<(), Self::Error> {
        { let $($mutable)? provider = self.selected(bank)?; provider.routing_applied(bank, original, effective).map_err(RoutedBankProviderError::Provider) }
    }
    fn routing_failed(&mut self, bank: RoutedBankId, message: &str) {
        if let Some($($mutable)? provider) = self.lookup_mut(bank) {
            provider.routing_failed(bank, message);
        }
    }
    fn forward_grouped(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        { let $($mutable)? provider = self.selected(request.bank)?; provider.forward_grouped(resident, request, context).map_err(RoutedBankProviderError::Provider) }
    }
    fn forward_compact_grouped(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        { let $($mutable)? provider = self.selected(request.bank)?; provider.forward_compact_grouped(resident, request, context).map_err(RoutedBankProviderError::Provider) }
    }
    fn forward_linear_routed(
        &mut self,
        resident: &mut B::LinearGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        { let $($mutable)? provider = self.selected(request.bank)?; provider.forward_linear_routed(resident, request, context).map_err(RoutedBankProviderError::Provider) }
    }
    fn forward_relu2_routed(
        &mut self,
        resident: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        { let $($mutable)? provider = self.selected(request.bank)?; provider.forward_relu2_routed(resident, request, context).map_err(RoutedBankProviderError::Provider) }
    }

}; }

impl<B: GroupedNeuralBackend, P: RoutedExpertProvider<B>> RoutedExpertProvider<B>
    for RoutedBankProviders<P>
{
    type Error = RoutedBankProviderError<P::Error>;
    routed_bank_ordinary_dispatch!(P);
}

impl<'a, B: GroupedNeuralBackend, P> RoutedExpertProvider<B> for BorrowedRoutedBankProviders<'a, P>
where &'a P: RoutedExpertProvider<B>,
{
    type Error = RoutedBankProviderError<<&'a P as RoutedExpertProvider<B>>::Error>;
    routed_bank_ordinary_dispatch!(&'a P, mut);
}
macro_rules! routed_bank_parallel_dispatch { ($provider:ty $(, $mutable:tt)?) => {
    fn forward_grouped_tensor_parallel(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        parts: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        { let $($mutable)? provider = self.selected(request.bank)?; provider.forward_grouped_tensor_parallel(resident, request, parts, context).map_err(RoutedBankProviderError::Provider) }
    }
    fn forward_compact_grouped_tensor_parallel(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        parts: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        { let $($mutable)? provider = self.selected(request.bank)?; provider.forward_compact_grouped_tensor_parallel(resident, request, parts, context).map_err(RoutedBankProviderError::Provider) }
    }
    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        resident: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        parts: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        { let $($mutable)? provider = self.selected(request.bank)?; provider.forward_relu2_routed_tensor_parallel(resident, request, parts, context).map_err(RoutedBankProviderError::Provider) }
    }

}; }

impl<B: GroupedNeuralBackend, P: TensorParallelRoutedExpertProvider<B>>
    TensorParallelRoutedExpertProvider<B> for RoutedBankProviders<P>
{
    routed_bank_parallel_dispatch!(P);
}

impl<'a, B: GroupedNeuralBackend, P> TensorParallelRoutedExpertProvider<B> for BorrowedRoutedBankProviders<'a, P>
where &'a P: TensorParallelRoutedExpertProvider<B>,
{
    routed_bank_parallel_dispatch!(&'a P, mut);
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
    fn routing_unmodified_interest(&self, bank: RoutedBankId) -> crate::RoutingUnmodifiedInterest {
        self.as_ref().routing_unmodified_interest(bank)
    }
    fn routing_unmodified(
        &mut self,
        bank: RoutedBankId,
        effective: crate::RoutingDecision<'_, B::Tensor>,
    ) -> Result<(), Self::Error> {
        self.as_mut().routing_unmodified(bank, effective)
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
impl<B: GroupedNeuralBackend, P: TensorParallelRoutedExpertProvider<B> + ?Sized>
    TensorParallelRoutedExpertProvider<B> for Box<P>
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

//! Runtime ownership boundary for routed expert acquisition and residency.

use eredu_nn::{RoutedNeuralBackend, RoutingResult, SwiGluExpertBankOperator, Tensor};

use crate::ExpertPass;

/// One architecture route batch submitted to a runtime expert provider.
pub struct RoutedExpertRequest<'a, T> {
    /// Global decoder layer requesting experts.
    pub layer: usize,
    /// Flattened token rows submitted to the selected experts.
    pub input: &'a T,
    /// Backend-native selected expert IDs, scores, and weights.
    pub routes: &'a RoutingResult<T>,
    /// Whether this route batch belongs to prefill or decode.
    pub pass: ExpertPass,
}

/// Runtime boundary for resident or independently cached routed experts.
///
/// Implementations own identity ordering, acquisition, leases, chunking,
/// budgets, and residency reports. They keep every lease alive until the
/// backend-native routed result is safe to return. The backend retains tensor
/// storage, transfers, compact-bank construction, and execution kernels.
pub trait RoutedExpertProvider<B>
where
    B: RoutedNeuralBackend,
{
    /// Provider-specific acquisition or execution failure.
    type Error;

    /// Executes one typed route batch while retaining its acquired resources.
    fn forward_routed(
        &mut self,
        resident_bank: &mut B::SwiGluExpertBank,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;
}

/// Provider for a fully resident expert bank.
#[derive(Debug, Default, Clone, Copy)]
pub struct ResidentExpertProvider;

impl<B> RoutedExpertProvider<B> for ResidentExpertProvider
where
    B: RoutedNeuralBackend,
{
    type Error = eredu_nn::Error;

    fn forward_routed(
        &mut self,
        resident_bank: &mut B::SwiGluExpertBank,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        resident_bank.forward_routed(request.input, request.routes, context)
    }
}

//! Prepared composition of grouped and row-addressable parameter mechanisms.
use crate::{
    ParameterProvider, RoutedBankId, RoutedExpertRequest, RoutedExpertTensorParallelOutput,
    RowLookupProvider, TensorParallelParameterProvider,
};
use eredu_nn::{GroupedNeuralBackend, Tensor};

/// One provider passed through resident, layered, partitioned and prediction
/// drivers. The mechanisms may share the same underlying residency manager.
pub struct ParameterProviders<P, L> {
    /// Selected grouped-operation provider.
    pub grouped: P,
    /// Selected row-lookup provider.
    pub rows: L,
}
impl<B: GroupedNeuralBackend, P: ParameterProvider<B>, L: RowLookupProvider<B>> ParameterProvider<B>
    for ParameterProviders<P, L>
{
    type Error = P::Error;
    fn has_row_parameter(&self, parameter: &eredu_nn::ParameterId) -> bool {
        self.rows.has_row_parameter(parameter)
    }
    fn lookup_rows(
        &mut self,
        spec: &crate::RowLookupSpec,
        rows: &[u64],
        access: crate::ParameterBankAccess,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, crate::RowLookupError> {
        self.rows.lookup_rows(spec, rows, access, context)
    }
    fn routing_control(
        &mut self,
        bank: RoutedBankId,
        rows: u64,
    ) -> Result<Option<eredu_nn::routing_intervention::GroupSelectionControl>, Self::Error> {
        self.grouped.routing_control(bank, rows)
    }
    fn routing_applied(
        &mut self,
        bank: RoutedBankId,
        original: Option<crate::RoutingDecision<'_, B::Tensor>>,
        effective: crate::RoutingDecision<'_, B::Tensor>,
    ) -> Result<(), Self::Error> {
        self.grouped.routing_applied(bank, original, effective)
    }
    fn routing_failed(&mut self, bank: RoutedBankId, message: &str) {
        self.grouped.routing_failed(bank, message)
    }
    fn forward_grouped(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.grouped.forward_grouped(resident, request, context)
    }
    fn forward_compact_grouped(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.grouped
            .forward_compact_grouped(resident, request, context)
    }
    fn forward_linear_routed(
        &mut self,
        resident: &mut B::LinearGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.grouped
            .forward_linear_routed(resident, request, context)
    }
    fn forward_relu2_routed(
        &mut self,
        resident: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.grouped
            .forward_relu2_routed(resident, request, context)
    }
}
impl<B: GroupedNeuralBackend, P: TensorParallelParameterProvider<B>, L: RowLookupProvider<B>>
    TensorParallelParameterProvider<B> for ParameterProviders<P, L>
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        self.grouped
            .forward_grouped_tensor_parallel(resident, request, partitions, context)
    }
    fn forward_compact_grouped_tensor_parallel(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        self.grouped
            .forward_compact_grouped_tensor_parallel(resident, request, partitions, context)
    }
    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        resident: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        self.grouped
            .forward_relu2_routed_tensor_parallel(resident, request, partitions, context)
    }
}

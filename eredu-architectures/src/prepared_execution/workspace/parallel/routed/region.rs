//! Cold boundary adapter around the ordinary routed architecture equations.
use super::*;
use eredu_nn::{GroupedGatedProductOperator, GroupedLinearOperator, GroupedRelu2Operator};
use eredu_runtime::{RoutedExpertProvider, TensorParallelRoutedExpertProvider};

pub(super) struct RegionProvider<'a, P> {
    execution: &'a crate::partitioned_execution::PreparedRoutedExecutionHandoff,
    provider: &'a mut P,
}
impl<'a, P> RegionProvider<'a, P> {
    pub(super) fn new(execution: &'a crate::partitioned_execution::PreparedRoutedExecutionHandoff,
        provider: &'a mut P) -> Self { Self { execution, provider } }
}

macro_rules! ordinary {
    ($method:ident, $module:ty, $variant:ident, $projection:ident) => {
        fn $method(&mut self, bank: &mut $module,
            request: eredu_runtime::RoutedExpertRequest<'_, '_, WorkspaceTensor>,
            context: &WorkspaceContext) -> Result<WorkspaceTensor, Self::Error> {
            let (realizations, group) = self.execution.expert_region_source();
            let Some(group) = group else {
                return self.provider.$method(bank, request, context)
                    .map_err(crate::RoutedTextExecutionError::from_error);
            };
            let realization = realizations.get(&request.bank).and_then(|value| value.$projection())
                .ok_or("cold region differs from its retained grouped equation")?;
            if realization.unit_is_replicated(request.layer) {
                let selected_bank=request.bank;let unit=request.layer;
                let output=self.provider.$method(bank,request,context)
                    .map_err(crate::RoutedTextExecutionError::from_error)?;
                let (tensor_group,wave_group)=self.execution.expert_provider_groups()?;
                eredu_nn::workspace::record_expert_provider_wave(eredu_nn::workspace::WorkspaceExpertProviderWave{
                    bank:selected_bank.value(),unit,wave:None,tensor_group,expert_group:group,wave_group},context)
                    .map_err(crate::RoutedTextExecutionError::from_error)?;
                return Ok(output);
            }
            let (tensor_group,wave_group)=self.execution.expert_provider_groups()?;
            let source = crate::partitioned_execution::partition_region_declaration::<WorkspaceBackend, _>(
                realization, &request, group, None, tensor_group, wave_group, eredu_nn::workspace::WorkspaceExpertKernel::$variant)?;
            let interested=request.unit_observer.is_some();
            let mut observer=request.unit_observer;
            let mut observe=|source:eredu_nn::workspace::WorkspaceExpertObservationView<'_>| -> Result<eredu_nn::workspace::WorkspaceExpertObservationSource,eredu_nn::Error> {
                observer.as_deref_mut().ok_or_else(||context.metadata_source(eredu_nn::GroupedUnitError::Unavailable))?.observe_region_source(source)
            };
            context.charge_metadata(std::mem::size_of_val(&observe)
                .checked_add(std::mem::size_of::<(bool,Option<&mut dyn eredu_runtime::RoutedUnitObserver<WorkspaceTensor>>,
                    Result<eredu_nn::workspace::WorkspaceExpertObservationSource,eredu_nn::Error>)>())
                .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)
                .map_err(crate::RoutedTextExecutionError::from_error)?)
                .map_err(crate::RoutedTextExecutionError::from_error)?;
            let (output, bias) = eredu_nn::workspace::record_expert_region_with_observation(source, bank,
                request.input, request.routes, context, interested.then_some(&mut observe))
                .map_err(crate::RoutedTextExecutionError::from_error)?.into_parts();
            if bias.is_some() { return Err("complete cold region returned a tensor partial".into()); }
            Ok(output)
        }
    };
}
macro_rules! parallel {
    ($method:ident, $module:ty, $variant:ident, $projection:ident) => {
        fn $method(&mut self, bank: &mut $module,
            request: eredu_runtime::RoutedExpertRequest<'_, '_, WorkspaceTensor>,
            partitions: usize, context: &WorkspaceContext)
            -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<WorkspaceTensor>, Self::Error> {
            let (realizations, group) = self.execution.expert_region_source();
            let Some(group) = group else {
                return self.provider.$method(bank, request, partitions, context)
                    .map_err(crate::RoutedTextExecutionError::from_error);
            };
            let realization = realizations.get(&request.bank).and_then(|value| value.$projection())
                .ok_or("cold region differs from its retained grouped equation")?;
            if realization.unit_is_replicated(request.layer) {
                let selected_bank=request.bank;let unit=request.layer;
                let output=self.provider.$method(bank,request,partitions,context)
                    .map_err(crate::RoutedTextExecutionError::from_error)?;
                let (tensor_group,wave_group)=self.execution.expert_provider_groups()?;
                eredu_nn::workspace::record_expert_provider_wave(eredu_nn::workspace::WorkspaceExpertProviderWave{
                    bank:selected_bank.value(),unit,wave:None,tensor_group,expert_group:group,wave_group},context)
                    .map_err(crate::RoutedTextExecutionError::from_error)?;
                return Ok(output);
            }
            let (tensor_group,wave_group)=self.execution.expert_provider_groups()?;
            let source = crate::partitioned_execution::partition_region_declaration::<WorkspaceBackend, _>(
                realization, &request, group, Some(partitions), tensor_group, wave_group, eredu_nn::workspace::WorkspaceExpertKernel::$variant)?;
            let interested=request.unit_observer.is_some();
            let mut observer=request.unit_observer;
            let mut observe=|source:eredu_nn::workspace::WorkspaceExpertObservationView<'_>| -> Result<eredu_nn::workspace::WorkspaceExpertObservationSource,eredu_nn::Error> {
                observer.as_deref_mut().ok_or_else(||context.metadata_source(eredu_nn::GroupedUnitError::Unavailable))?.observe_region_source(source)
            };
            context.charge_metadata(std::mem::size_of_val(&observe)
                .checked_add(std::mem::size_of::<(bool,Option<&mut dyn eredu_runtime::RoutedUnitObserver<WorkspaceTensor>>,
                    Result<eredu_nn::workspace::WorkspaceExpertObservationSource,eredu_nn::Error>)>())
                .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)
                .map_err(crate::RoutedTextExecutionError::from_error)?)
                .map_err(crate::RoutedTextExecutionError::from_error)?;
            eredu_nn::workspace::record_expert_region_with_observation(source, bank, request.input, request.routes, context, interested.then_some(&mut observe))
                .map(eredu_runtime::RoutedExpertTensorParallelOutput::Partial)
                .map_err(crate::RoutedTextExecutionError::from_error)
        }
    };
}
impl<P> RoutedExpertProvider<WorkspaceBackend> for RegionProvider<'_, P>
where P: TensorParallelRoutedExpertProvider<WorkspaceBackend> {
    type Error = crate::RoutedTextExecutionError;
    fn routing_control(&mut self, bank: eredu_runtime::RoutedBankId, rows: u64)
        -> Result<Option<eredu_nn::routing_intervention::GroupSelectionControl>, Self::Error> {
        self.provider.routing_control(bank, rows).map_err(crate::RoutedTextExecutionError::from_error)
    }
    fn routing_unmodified_interest(&self, bank: eredu_runtime::RoutedBankId) -> eredu_runtime::RoutingUnmodifiedInterest {
        self.provider.routing_unmodified_interest(bank)
    }
    fn routing_unmodified(&mut self, bank: eredu_runtime::RoutedBankId,
        effective: eredu_runtime::RoutingDecision<'_, WorkspaceTensor>) -> Result<(), Self::Error> {
        self.provider.routing_unmodified(bank, effective).map_err(crate::RoutedTextExecutionError::from_error)
    }
    fn routing_applied(&mut self, bank: eredu_runtime::RoutedBankId,
        original: Option<eredu_runtime::RoutingDecision<'_, WorkspaceTensor>>,
        effective: eredu_runtime::RoutingDecision<'_, WorkspaceTensor>) -> Result<(), Self::Error> {
        self.provider.routing_applied(bank, original, effective).map_err(crate::RoutedTextExecutionError::from_error)
    }
    fn routing_failed(&mut self, bank: eredu_runtime::RoutedBankId, message: &str) {
        self.provider.routing_failed(bank, message);
    }
    ordinary!(forward_grouped, <WorkspaceBackend as eredu_nn::GroupedNeuralBackend>::GatedProductGroups, Gated, gated);
    ordinary!(forward_linear_routed, <WorkspaceBackend as eredu_nn::GroupedNeuralBackend>::LinearGroups, Linear, linear);
    ordinary!(forward_relu2_routed, <WorkspaceBackend as eredu_nn::GroupedNeuralBackend>::Relu2Groups, Relu2, relu2);
}
impl<P> TensorParallelRoutedExpertProvider<WorkspaceBackend> for RegionProvider<'_, P>
where P: TensorParallelRoutedExpertProvider<WorkspaceBackend> {
    parallel!(forward_grouped_tensor_parallel, <WorkspaceBackend as eredu_nn::GroupedNeuralBackend>::GatedProductGroups, Gated, gated);
    parallel!(forward_relu2_routed_tensor_parallel, <WorkspaceBackend as eredu_nn::GroupedNeuralBackend>::Relu2Groups, Relu2, relu2);
}

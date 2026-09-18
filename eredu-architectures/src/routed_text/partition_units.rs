//! Borrow selected scalar coordinates through every partition bank provider path.
use super::*;
use eredu_runtime::{RoutedExpertTensorParallelOutput, TensorParallelRoutedExpertProvider};

pub(crate) struct PartitionUnitProvider<P, C = BTreeMap<usize, eredu_core::component::RoutedComponentCoordinateMap>> {
    inner: P,
    coordinates: C,
}
impl<P> PartitionUnitProvider<P> {
    pub(crate) fn new(bank: &SelectedRoutedBank, inner: P) -> Self {
        Self {
            inner,
            coordinates: bank.partition_unit_coordinates.clone(),
        }
    }
}

pub(crate) fn with_optional_coordinates<T: Tensor, R>(
    coordinates: Option<&BTreeMap<usize, eredu_core::component::RoutedComponentCoordinateMap>>,
    request: RoutedExpertRequest<'_, '_, T>,
    execute: impl FnOnce(RoutedExpertRequest<'_, '_, T>) -> Result<R, RoutedTextExecutionError>,
) -> Result<R, RoutedTextExecutionError> {
    match coordinates {
        Some(coordinates) => with_coordinates(coordinates, request, execute),
        None => {
            // A selected resident bank still has one logical invocation. Its
            // provider batches must not bypass begin/finish merely because no
            // partition-coordinate adapter is needed.
            let RoutedExpertRequest { bank, layer, input, routes, pass, unit_observer } = request;
            eredu_runtime::with_routed_unit_invocation(
                unit_observer,
                eredu_runtime::RoutedUnitInvocation {
                    input, origins: None, unit_coordinates: None,
                },
                |unit_observer| execute(RoutedExpertRequest {
                    bank, layer, input, routes, pass, unit_observer,
                }),
                RoutedTextExecutionError::from_error,
            )
        },
    }
}

fn with_coordinates<T: Tensor, R>(
    coordinates: &BTreeMap<usize, eredu_core::component::RoutedComponentCoordinateMap>,
    request: RoutedExpertRequest<'_, '_, T>,
    execute: impl FnOnce(RoutedExpertRequest<'_, '_, T>) -> Result<R, RoutedTextExecutionError>,
) -> Result<R, RoutedTextExecutionError> {
    if request.unit_observer.is_none() {
        return execute(request);
    }
    let coordinates = coordinates.get(&request.layer);
    let RoutedExpertRequest {
        bank,
        layer,
        input,
        routes,
        pass,
        unit_observer,
    } = request;
    eredu_runtime::with_routed_unit_invocation(
        unit_observer,
        eredu_runtime::RoutedUnitInvocation {
            input,
            origins: None,
            unit_coordinates: coordinates.map(|coordinates| coordinates.units()),
        },
        |mut observer| {
            let coordinates = coordinates.ok_or_else(|| {
                RoutedTextExecutionError::Contract(
                    "selected partition bank has no scalar coordinates for this invocation".into(),
                )
            })?;
            eredu_runtime::with_partition_unit_observer(
                &mut observer,
                coordinates.units(),
                |observer| {
                    execute(RoutedExpertRequest {
                        bank,
                        layer,
                        input,
                        routes,
                        pass,
                        unit_observer: observer,
                    })
                },
            )
        },
        RoutedTextExecutionError::from_error,
    )
}

impl<B, P, C> RoutedExpertProvider<B> for PartitionUnitProvider<P, C>
where
    B: GroupedNeuralBackend,
    C: std::borrow::Borrow<BTreeMap<usize, eredu_core::component::RoutedComponentCoordinateMap>>,
    P: RoutedExpertProvider<B, Error = RoutedTextExecutionError>,
{
    type Error = RoutedTextExecutionError;
    fn routing_control(
        &mut self,
        bank: RoutedBankId,
        rows: u64,
    ) -> Result<Option<eredu_nn::routing_intervention::GroupSelectionControl>, Self::Error> {
        self.inner.routing_control(bank, rows)
    }
    fn routing_unmodified_interest(
        &self,
        bank: RoutedBankId,
    ) -> eredu_runtime::RoutingUnmodifiedInterest {
        self.inner.routing_unmodified_interest(bank)
    }
    fn routing_unmodified(
        &mut self,
        bank: RoutedBankId,
        effective: eredu_runtime::RoutingDecision<'_, B::Tensor>,
    ) -> Result<(), Self::Error> {
        self.inner.routing_unmodified(bank, effective)
    }

    fn routing_applied(
        &mut self,
        bank: RoutedBankId,
        original: Option<eredu_runtime::RoutingDecision<'_, B::Tensor>>,
        effective: eredu_runtime::RoutingDecision<'_, B::Tensor>,
    ) -> Result<(), Self::Error> {
        self.inner.routing_applied(bank, original, effective)
    }
    fn routing_failed(&mut self, bank: RoutedBankId, message: &str) {
        self.inner.routing_failed(bank, message);
    }
    fn forward_grouped(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        with_coordinates(self.coordinates.borrow(), request, |request| {
            self.inner.forward_grouped(resident, request, context)
        })
    }
    fn forward_compact_grouped(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        with_coordinates(self.coordinates.borrow(), request, |request| {
            self.inner
                .forward_compact_grouped(resident, request, context)
        })
    }
    fn forward_relu2_routed(
        &mut self,
        resident: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        with_coordinates(self.coordinates.borrow(), request, |request| {
            self.inner.forward_relu2_routed(resident, request, context)
        })
    }
    fn forward_linear_routed(
        &mut self,
        resident: &mut B::LinearGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        // A linear bank has no FFN unit map; preserve its existing unsupported hook.
        self.inner.forward_linear_routed(resident, request, context)
    }
}
impl<B, P, C> TensorParallelRoutedExpertProvider<B> for PartitionUnitProvider<P, C>
where
    B: GroupedNeuralBackend,
    C: std::borrow::Borrow<BTreeMap<usize, eredu_core::component::RoutedComponentCoordinateMap>>,
    P: TensorParallelRoutedExpertProvider<B, Error = RoutedTextExecutionError>,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        with_coordinates(self.coordinates.borrow(), request, |request| {
            self.inner
                .forward_grouped_tensor_parallel(resident, request, partitions, context)
        })
    }
    fn forward_compact_grouped_tensor_parallel(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        with_coordinates(self.coordinates.borrow(), request, |request| {
            self.inner
                .forward_compact_grouped_tensor_parallel(resident, request, partitions, context)
        })
    }
    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        resident: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        with_coordinates(self.coordinates.borrow(), request, |request| {
            self.inner
                .forward_relu2_routed_tensor_parallel(resident, request, partitions, context)
        })
    }
}

impl<P> PartitionUnitProvider<P> {
    fn borrowed(&self) -> PartitionUnitProvider<&P, &BTreeMap<usize, eredu_core::component::RoutedComponentCoordinateMap>> {
        PartitionUnitProvider { inner: &self.inner, coordinates: &self.coordinates }
    }
}

impl<'a, B, P> RoutedExpertProvider<B> for &'a PartitionUnitProvider<P>
where
    B: GroupedNeuralBackend,
    &'a P: RoutedExpertProvider<B, Error = RoutedTextExecutionError>,
{
    type Error = RoutedTextExecutionError;
    fn routing_control(
        &mut self,
        bank: RoutedBankId,
        rows: u64,
    ) -> Result<Option<eredu_nn::routing_intervention::GroupSelectionControl>, Self::Error> {
        RoutedExpertProvider::<B>::routing_control(&mut self.borrowed(), bank, rows)
    }

    fn routing_unmodified_interest(
        &self,
        bank: RoutedBankId,
    ) -> eredu_runtime::RoutingUnmodifiedInterest {
        RoutedExpertProvider::<B>::routing_unmodified_interest(&mut self.borrowed(), bank)
    }

    fn routing_unmodified(
        &mut self,
        bank: RoutedBankId,
        effective: eredu_runtime::RoutingDecision<'_, B::Tensor>,
    ) -> Result<(), Self::Error> {
        RoutedExpertProvider::<B>::routing_unmodified(&mut self.borrowed(), bank, effective)
    }

    fn routing_applied(
        &mut self,
        bank: RoutedBankId,
        original: Option<eredu_runtime::RoutingDecision<'_, B::Tensor>>,
        effective: eredu_runtime::RoutingDecision<'_, B::Tensor>,
    ) -> Result<(), Self::Error> {
        RoutedExpertProvider::<B>::routing_applied(&mut self.borrowed(), bank, original, effective)
    }

    fn routing_failed(&mut self, bank: RoutedBankId, message: &str) {
        RoutedExpertProvider::<B>::routing_failed(&mut self.borrowed(), bank, message)
    }

    fn forward_grouped(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if let Some(metadata) = B::construction_metadata(context).filter(|source| source.uses_checked_metadata()) {
            let bytes = [std::mem::size_of::<(&mut Self, &Self)>(), std::mem::size_of_val(&resident), std::mem::size_of_val(&request), std::mem::size_of_val(&context), std::mem::size_of::<Result<B::Tensor, Self::Error>>()].into_iter().try_fold(0usize, usize::checked_add)
                .ok_or_else(|| RoutedTextExecutionError::from_error(eredu_nn::workspace::WorkspaceMetadataError::Overflow))?;
            metadata.charge_metadata(bytes).map_err(RoutedTextExecutionError::from_error)?;
        }
        RoutedExpertProvider::<B>::forward_grouped(&mut self.borrowed(), resident, request, context)
    }

    fn forward_compact_grouped(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if let Some(metadata) = B::construction_metadata(context).filter(|source| source.uses_checked_metadata()) {
            let bytes = [std::mem::size_of::<(&mut Self, &Self)>(), std::mem::size_of_val(&resident), std::mem::size_of_val(&request), std::mem::size_of_val(&context), std::mem::size_of::<Result<B::Tensor, Self::Error>>()].into_iter().try_fold(0usize, usize::checked_add)
                .ok_or_else(|| RoutedTextExecutionError::from_error(eredu_nn::workspace::WorkspaceMetadataError::Overflow))?;
            metadata.charge_metadata(bytes).map_err(RoutedTextExecutionError::from_error)?;
        }
        RoutedExpertProvider::<B>::forward_compact_grouped(&mut self.borrowed(), resident, request, context)
    }

    fn forward_relu2_routed(
        &mut self,
        resident: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if let Some(metadata) = B::construction_metadata(context).filter(|source| source.uses_checked_metadata()) {
            let bytes = [std::mem::size_of::<(&mut Self, &Self)>(), std::mem::size_of_val(&resident), std::mem::size_of_val(&request), std::mem::size_of_val(&context), std::mem::size_of::<Result<B::Tensor, Self::Error>>()].into_iter().try_fold(0usize, usize::checked_add)
                .ok_or_else(|| RoutedTextExecutionError::from_error(eredu_nn::workspace::WorkspaceMetadataError::Overflow))?;
            metadata.charge_metadata(bytes).map_err(RoutedTextExecutionError::from_error)?;
        }
        RoutedExpertProvider::<B>::forward_relu2_routed(&mut self.borrowed(), resident, request, context)
    }

    fn forward_linear_routed(
        &mut self,
        resident: &mut B::LinearGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if let Some(metadata) = B::construction_metadata(context).filter(|source| source.uses_checked_metadata()) {
            let bytes = [std::mem::size_of::<(&mut Self, &Self)>(), std::mem::size_of_val(&resident), std::mem::size_of_val(&request), std::mem::size_of_val(&context), std::mem::size_of::<Result<B::Tensor, Self::Error>>()].into_iter().try_fold(0usize, usize::checked_add)
                .ok_or_else(|| RoutedTextExecutionError::from_error(eredu_nn::workspace::WorkspaceMetadataError::Overflow))?;
            metadata.charge_metadata(bytes).map_err(RoutedTextExecutionError::from_error)?;
        }
        RoutedExpertProvider::<B>::forward_linear_routed(&mut self.borrowed(), resident, request, context)
    }
}

impl<'a, B, P> TensorParallelRoutedExpertProvider<B> for &'a PartitionUnitProvider<P>
where
    B: GroupedNeuralBackend,
    &'a P: TensorParallelRoutedExpertProvider<B, Error = RoutedTextExecutionError>,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        if let Some(metadata) = B::construction_metadata(context).filter(|source| source.uses_checked_metadata()) {
            let bytes = [std::mem::size_of::<(&mut Self, &Self)>(), std::mem::size_of_val(&resident), std::mem::size_of_val(&request), std::mem::size_of_val(&partitions), std::mem::size_of_val(&context), std::mem::size_of::<Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error>>()].into_iter().try_fold(0usize, usize::checked_add)
                .ok_or_else(|| RoutedTextExecutionError::from_error(eredu_nn::workspace::WorkspaceMetadataError::Overflow))?;
            metadata.charge_metadata(bytes).map_err(RoutedTextExecutionError::from_error)?;
        }
        TensorParallelRoutedExpertProvider::<B>::forward_grouped_tensor_parallel(&mut self.borrowed(), resident, request, partitions, context)
    }

    fn forward_compact_grouped_tensor_parallel(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        if let Some(metadata) = B::construction_metadata(context).filter(|source| source.uses_checked_metadata()) {
            let bytes = [std::mem::size_of::<(&mut Self, &Self)>(), std::mem::size_of_val(&resident), std::mem::size_of_val(&request), std::mem::size_of_val(&partitions), std::mem::size_of_val(&context), std::mem::size_of::<Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error>>()].into_iter().try_fold(0usize, usize::checked_add)
                .ok_or_else(|| RoutedTextExecutionError::from_error(eredu_nn::workspace::WorkspaceMetadataError::Overflow))?;
            metadata.charge_metadata(bytes).map_err(RoutedTextExecutionError::from_error)?;
        }
        TensorParallelRoutedExpertProvider::<B>::forward_compact_grouped_tensor_parallel(&mut self.borrowed(), resident, request, partitions, context)
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        resident: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        if let Some(metadata) = B::construction_metadata(context).filter(|source| source.uses_checked_metadata()) {
            let bytes = [std::mem::size_of::<(&mut Self, &Self)>(), std::mem::size_of_val(&resident), std::mem::size_of_val(&request), std::mem::size_of_val(&partitions), std::mem::size_of_val(&context), std::mem::size_of::<Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error>>()].into_iter().try_fold(0usize, usize::checked_add)
                .ok_or_else(|| RoutedTextExecutionError::from_error(eredu_nn::workspace::WorkspaceMetadataError::Overflow))?;
            metadata.charge_metadata(bytes).map_err(RoutedTextExecutionError::from_error)?;
        }
        TensorParallelRoutedExpertProvider::<B>::forward_relu2_routed_tensor_parallel(&mut self.borrowed(), resident, request, partitions, context)
    }
}

//! Borrow selected scalar coordinates through every partition bank provider path.
use super::*;
use eredu_runtime::{RoutedExpertTensorParallelOutput, TensorParallelRoutedExpertProvider};

pub(crate) struct PartitionUnitProvider<P> {
    inner: P,
    coordinates: BTreeMap<usize, eredu_core::component::RoutedComponentCoordinateMap>,
}
impl<P> PartitionUnitProvider<P> {
    pub(crate) fn new(bank: &SelectedRoutedBank, inner: P) -> Self {
        Self {
            inner,
            coordinates: bank.partition_unit_coordinates.clone(),
        }
    }
}

pub(super) fn with_optional_coordinates<T: Tensor, R>(
    coordinates: Option<&BTreeMap<usize, eredu_core::component::RoutedComponentCoordinateMap>>,
    request: RoutedExpertRequest<'_, '_, T>,
    execute: impl FnOnce(RoutedExpertRequest<'_, '_, T>) -> Result<R, RoutedTextExecutionError>,
) -> Result<R, RoutedTextExecutionError> {
    match coordinates {
        Some(coordinates) => with_coordinates(coordinates, request, execute),
        None => execute(request),
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

impl<B, P> RoutedExpertProvider<B> for PartitionUnitProvider<P>
where
    B: GroupedNeuralBackend,
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
        with_coordinates(&self.coordinates, request, |request| {
            self.inner.forward_grouped(resident, request, context)
        })
    }
    fn forward_compact_grouped(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        with_coordinates(&self.coordinates, request, |request| {
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
        with_coordinates(&self.coordinates, request, |request| {
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
impl<B, P> TensorParallelRoutedExpertProvider<B> for PartitionUnitProvider<P>
where
    B: GroupedNeuralBackend,
    P: TensorParallelRoutedExpertProvider<B, Error = RoutedTextExecutionError>,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        resident: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, '_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        with_coordinates(&self.coordinates, request, |request| {
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
        with_coordinates(&self.coordinates, request, |request| {
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
        with_coordinates(&self.coordinates, request, |request| {
            self.inner
                .forward_relu2_routed_tensor_parallel(resident, request, partitions, context)
        })
    }
}

//! Portable ownership of admitted parameter unit storage.

use super::resident::execution_group_reports;
use super::*;
use crate::{
    bounded_execution::units::SelectedUnitResources, dense::DenseResidencyMechanisms,
    residency::ResidencyDriver, DenseDiskStreamReport, ParameterUnitLayout,
    ParameterUnitStorageResources, ResidencyReport, ResidentLayerGroupReport,
};
use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_core::residency::{
    ParameterConversionRetentionTrimReport, ParameterConversionTrimError, UnitRetirement,
};
use eredu_nn::NeuralBackend;

/// Exact prepared unit resources disagreed with the installed layout.
#[derive(Debug, thiserror::Error)]
pub enum UnitStorageError {
    /// The physical catalog or its lookahead does not match the selected layout.
    #[error("parameter units or transfer window differ from the admitted layout")]
    InvalidLayout,
    /// The portable bind has no transfer for resident modules.
    #[error("resident unit storage has no prepared transfer")]
    MissingResidentTransfer,
    /// The caller supplied a different number of physical units.
    #[error("resident modules differ from the admitted storage catalog")]
    ResidentUnitCount,
}

type StaticPin<N> = eredu_core::residency::ResidencyLease<
    <<N as eredu_core::residency::ResidencyStorageMechanisms>::Storage
        as eredu_core::residency::ResidencyStorage>::Copy,
>;

/// Static parameter pin and resident transfer protected by native retirement.
pub type ResidentUnitRetention<N, O> = (
    <O as UnitRetirement>::RetainedResource<Option<StaticPin<N>>>,
    <N as eredu_core::residency::ResidencyStorageMechanisms>::Transfer,
);

/// Portable owner of the residency ledger, selected unit layout and native operations.
pub struct BoundUnitStorage<
    N: DenseResidencyMechanisms + Clone + Send + Sync + 'static,
    O: NativeBoundedUnits,
> {
    residency: ResidencyDriver<N>,
    source: SharedCheckpointSource,
    layout: ParameterUnitLayout,
    resident_transfer: Option<N::Transfer>,
    pinned: BTreeSet<OffloadUnitId>,
    dense: Option<Arc<DenseStreamController<ResidencyDriver<N>>>>,
    native: O,
    population: Option<SelectiveUnitParameters<O::Tensor>>,
    static_pin: O::RetainedResource<Option<StaticPin<N>>>,
}

impl<N, O> BoundUnitStorage<N, O>
where
    N: DenseResidencyMechanisms + Clone + Send + Sync + 'static,
    O: NativeBoundedUnits,
{
    /// Installs the portable layout and ledger with the supplied native operations.
    pub(crate) fn new(
        residency: ResidencyDriver<N>,
        source: SharedCheckpointSource,
        layout: ParameterUnitLayout,
        resources: ParameterUnitStorageResources<ResidencyDriver<N>>,
        native: O,
        population: Option<SelectiveUnitParameters<O::Tensor>>,
        static_pin: Option<StaticPin<N>>,
    ) -> Result<Self, O::Error> {
        if layout.unit_ids.len() != layout.invocations.storage_layout().len()
            || layout.unit_ids.is_empty()
            || layout.metadata.device_layer_capacity() == 0
        {
            return Err(BackendFailure::from_error(UnitStorageError::InvalidLayout).into());
        }
        Ok(Self {
            residency,
            source,
            layout,
            resident_transfer: resources.resident_transfer,
            pinned: resources.pinned_units,
            dense: resources.dense,
            native,
            population,
            static_pin: O::retain_resource(static_pin),
        })
    }

    /// Borrows the existing driver shared with independently invoked modules.
    pub fn residency_driver(&self) -> &ResidencyDriver<N> {
        &self.residency
    }

    /// Binds permanently resident units and adopts their native handles.
    pub fn into_resident_units<B, U>(
        self,
        units: Vec<U>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<
        ResidentUnitPolicy<B, N, O::Module<U>, ResidentUnitRetention<N, O>, O::Error>,
        O::Error,
    >
    where
        B: NeuralBackend,
        O: NativeResidentUnits<B, N::Transfer> + NativeBoundedUnits<Tensor = B::Tensor>,
        U: Parameterized<B::Tensor> + 'static,
    {
        if units.len() != self.layout.unit_ids.len() {
            return Err(BackendFailure::from_error(UnitStorageError::ResidentUnitCount).into());
        }
        let transfer = self
            .resident_transfer
            .ok_or_else(|| BackendFailure::from_error(UnitStorageError::MissingResidentTransfer))?;
        let static_count = usize::from(self.static_pin.is_some());
        if O::order_resident_transfer(&transfer, context)? != units.len() {
            return Err(BackendFailure::from_error(UnitStorageError::ResidentUnitCount).into());
        }
        let modules = units
            .into_iter()
            .enumerate()
            .map(|(ordinal, mut unit)| {
                let weights = MaterializedUnit::<
                    <O as NativeResidentUnits<B, N::Transfer>>::Parameters,
                >::try_from_named_weights(O::resident_weights(
                    &transfer, ordinal,
                )?)
                .map_err(eredu_core::BackendFailure::from_error)?;
                super::population::install_unit::<
                    <O as NativeResidentUnits<B, N::Transfer>>::Parameters,
                    _,
                >(&mut unit, weights, self.population.as_ref())?;
                Ok::<_, O::Error>(O::retain_unit(unit))
            })
            .collect::<Result<_, _>>()?;
        let retained = (self.static_pin, transfer);
        self.residency
            .enable_resident_parameter_conversions()
            .map_err(BackendFailure::from_error)?;
        Ok(ResidentUnitPolicy::new(
            modules,
            self.residency,
            self.source,
            self.layout.unit_ids,
            self.layout.invocations,
            self.layout.metadata.device_layer_capacity(),
            static_count,
            retained,
        ))
    }
}

impl<N, O> UnitRetirement for BoundUnitStorage<N, O>
where
    N: DenseResidencyMechanisms + Clone + Send + Sync + 'static,
    O: NativeBoundedUnits,
{
    type Error = O::Error;
    type RetainedResource<T: 'static> = O::RetainedResource<T>;

    fn retain_resource<T: 'static>(resource: T) -> Self::RetainedResource<T> {
        O::retain_resource(resource)
    }
}

impl<N, O> NativeBoundedUnits for BoundUnitStorage<N, O>
where
    N: DenseResidencyMechanisms + Clone + Send + Sync + 'static,
    O: NativeBoundedUnits<Transfer = N::Transfer>,
{
    type Tensor = O::Tensor;
    type Context = O::Context;
    type Parameters = O::Parameters;
    type Lease<U: 'static> = O::Lease<U>;
    type Transfer = N::Transfer;

    fn reclaim(&mut self, inspection: bool) {
        self.native.reclaim(inspection);
    }
    fn reclaim_retired(&mut self) {
        self.native.reclaim_retired();
    }
    fn token_width(&self, input: &Self::Tensor) -> usize {
        self.native.token_width(input)
    }
    fn is_complete<U: 'static>(&self, lease: &Self::Lease<U>) -> Result<bool, Self::Error> {
        self.native.is_complete::<U>(lease)
    }
    fn retire<U: 'static>(&mut self, lease: Self::Lease<U>) -> Result<(), Self::Error> {
        self.native.retire::<U>(lease)
    }
    fn bind_lease<U: 'static>(
        &mut self,
        unit: U,
        transfer: N::Transfer,
        context: &Self::Context,
    ) -> Result<Self::Lease<U>, Self::Error> {
        self.native.bind_lease(unit, transfer, context)
    }
    fn unit_weights<U: 'static>(
        lease: &mut Self::Lease<U>,
    ) -> Result<
        Vec<(
            String,
            <Self::Parameters as CheckpointParameterBackend>::MaterializedWeight,
        )>,
        Self::Error,
    > {
        O::unit_weights::<U>(lease)
    }
    fn submit<'a, U: 'static>(
        &mut self,
        lease: &mut Self::Lease<U>,
        roots: impl Iterator<Item = &'a Self::Tensor>,
    ) -> Result<(), Self::Error>
    where
        Self::Tensor: 'a,
    {
        self.native.submit::<U>(lease, roots)
    }
    fn submit_inspection<U: 'static>(
        &mut self,
        lease: &mut Self::Lease<U>,
        context: &Self::Context,
    ) -> Result<(), Self::Error> {
        self.native.submit_inspection::<U>(lease, context)
    }
    fn finish_output(
        &mut self,
        output: &Self::Tensor,
        context: &Self::Context,
    ) -> Result<(), Self::Error> {
        self.native.finish_output(output, context)
    }
}

impl<N, O> BoundedUnitMechanisms for BoundUnitStorage<N, O>
where
    N: DenseResidencyMechanisms + Clone + Send + Sync + 'static,
    O: NativeBoundedUnits<Transfer = N::Transfer>,
{
    type Residency = ResidencyDriver<N>;

    fn population(&self) -> Option<&SelectiveUnitParameters<Self::Tensor>> {
        self.population.as_ref()
    }

    fn publish_replacements(
        &mut self,
        values: &std::collections::BTreeMap<String, Self::Tensor>,
        active: bool,
    ) -> bool {
        let Some(population) = &mut self.population else {
            return false;
        };
        population.publish(values, active);
        true
    }

    fn residency(&self) -> &Self::Residency {
        &self.residency
    }
    fn memory_sampling(&self) -> (bool, bool) {
        (
            self.layout.residency.sample_backend_memory(),
            self.layout.residency.sample_process_memory(),
        )
    }
}

impl<N, O> BoundedUnitBinding for BoundUnitStorage<N, O>
where
    N: DenseResidencyMechanisms + Clone + Send + Sync + 'static,
    O: NativeBoundedUnits<Transfer = N::Transfer>,
{
    type DenseOwner = O::RetainedResource<BoundedDenseRun<ResidencyDriver<N>>>;
    fn bounded_layout(&self) -> (Vec<OffloadUnitId>, BTreeSet<OffloadUnitId>, usize) {
        (
            self.layout.unit_ids.clone(),
            self.pinned.clone(),
            self.layout.metadata.device_layer_capacity(),
        )
    }
    fn dense_controller(&self) -> Option<&Arc<DenseStreamController<ResidencyDriver<N>>>> {
        self.dense.as_ref()
    }
    fn retain_dense_run(&self, run: BoundedDenseRun<ResidencyDriver<N>>) -> Self::DenseOwner {
        O::retain_resource(run)
    }
}

impl<N, O> SelectedUnitResources for BoundUnitStorage<N, O>
where
    N: DenseResidencyMechanisms + Clone + Send + Sync + 'static,
    O: NativeBoundedUnits,
{
    type Error = O::Error;
    fn addresses(&self) -> &ExecutionUnitAddressMap {
        &self.layout.invocations
    }
    fn residency_report(&self) -> Result<ResidencyReport, Self::Error> {
        self.residency
            .report()
            .map_err(BackendFailure::from_error)
            .map_err(Into::into)
    }
    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Self::Error> {
        self.dense
            .as_ref()
            .map(|dense| dense.report())
            .transpose()
            .map_err(BackendFailure::from_error)
            .map_err(Into::into)
    }
    fn execution_group_reports(&self) -> Result<Vec<ResidentLayerGroupReport>, Self::Error> {
        execution_group_reports(
            &self.layout.invocations,
            &self.layout.unit_ids,
            self.layout.metadata.device_layer_capacity(),
            &self.residency,
        )
    }
    fn trim_parameter_conversions(
        &self,
    ) -> Result<Vec<ParameterConversionRetentionTrimReport>, ParameterConversionTrimError> {
        self.residency.trim_parameter_conversions()
    }
}

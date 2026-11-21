//! Portable ownership and acquisition of fully resident execution units.

use super::SelectedUnitResources;
use crate::{
    dense::DenseResidencyMechanisms, residency::ResidencyDriver, DenseDiskStreamReport,
    ExecutionUnitAddress, ExecutionUnitAddressMap, ExecutionUnitLeaseAddress,
    LayerwiseAcquireError, LayerwisePolicy, ResidencyReport, ResidentLayerGroup,
    ResidentLayerGroupReport, UnitResidencyTelemetry,
};
use eredu_checkpoint::store::{CheckpointSource, SharedCheckpointSource};
use eredu_core::residency::{
    OffloadUnitId, ParameterConversionRetentionTrimReport, ParameterConversionTrimError,
};
use eredu_core::BackendFailure;
use eredu_nn::{NeuralBackend, Parameterized, Tensor};
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::Arc,
};

/// A rejected resident-unit acquisition or return.
#[derive(Debug, thiserror::Error)]
pub enum ResidentUnitError {
    /// A required physical unit is not idle.
    #[error("parameter unit {index} is not idle")]
    Unavailable {
        /// Physical storage ordinal.
        index: usize,
    },
    /// The returned unit's slot already contains an owner.
    #[error("resident unit slot {index} was unexpectedly occupied")]
    Occupied {
        /// Physical storage ordinal.
        index: usize,
    },
}

/// Portable ownership of populated units and their prepared residency.
pub struct ResidentUnitPolicy<B, N: DenseResidencyMechanisms, M, T, E> {
    units: Vec<Option<M>>,
    residency: ResidencyDriver<N>,
    store: SharedCheckpointSource,
    unit_ids: Vec<OffloadUnitId>,
    invocations: ExecutionUnitAddressMap,
    window_depth: usize,
    static_lease_count: usize,
    _retained: T,
    error: PhantomData<fn() -> (B, E)>,
}

/// Exclusive ownership of one populated unit for the exact invocation.
pub struct ResidentUnit<M> {
    index: usize,
    invocation: ExecutionUnitLeaseAddress,
    unit: M,
}

impl<M: Deref> Deref for ResidentUnit<M> {
    type Target = M::Target;

    fn deref(&self) -> &Self::Target {
        &self.unit
    }
}

impl<M: DerefMut> DerefMut for ResidentUnit<M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.unit
    }
}

impl<B, N, M, T, E> ResidentUnitPolicy<B, N, M, T, E>
where
    N: DenseResidencyMechanisms,
    E: From<BackendFailure>,
{
    /// Adopts native modules and storage guards after exact parameter population.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        units: Vec<M>,
        residency: ResidencyDriver<N>,
        store: SharedCheckpointSource,
        unit_ids: Vec<OffloadUnitId>,
        invocations: ExecutionUnitAddressMap,
        window_depth: usize,
        static_lease_count: usize,
        retained: T,
    ) -> Self {
        Self {
            units: units.into_iter().map(Some).collect(),
            residency,
            store,
            unit_ids,
            invocations,
            window_depth,
            static_lease_count,
            _retained: retained,
            error: PhantomData,
        }
    }

    /// Returns the checkpoint source backing this resident policy.
    pub fn checkpoint_store(&self) -> &dyn CheckpointSource {
        self.store.as_ref()
    }

    /// Clones the shared checkpoint source backing this resident policy.
    pub fn checkpoint_store_arc(&self) -> SharedCheckpointSource {
        Arc::clone(&self.store)
    }

    /// Releases optional conversion claims at the settled boundary.
    pub fn trim_parameter_conversions(
        &self,
    ) -> Result<
        Vec<eredu_core::residency::ParameterConversionRetentionTrimReport>,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        self.residency.trim_parameter_conversions()
    }

    /// Returns current weight-residency accounting.
    pub fn residency_report(&self) -> Result<ResidencyReport, E> {
        self.residency
            .report()
            .map_err(BackendFailure::from_error)
            .map_err(Into::into)
    }

    /// Returns the number of pinned static-weight leases.
    pub fn static_lease_count(&self) -> usize {
        self.static_lease_count
    }

    /// Returns residency reports for every semantic execution group.
    pub fn execution_group_reports(&self) -> Result<Vec<ResidentLayerGroupReport>, E> {
        execution_group_reports(
            &self.invocations,
            &self.unit_ids,
            self.window_depth,
            &self.residency,
        )
    }
}

/// Reports admitted execution groups against the exact residency owner.
pub(super) fn execution_group_reports<R, E>(
    invocations: &ExecutionUnitAddressMap,
    units: &[OffloadUnitId],
    window_depth: usize,
    residency: &R,
) -> Result<Vec<ResidentLayerGroupReport>, E>
where
    R: crate::residency::ResidencyWindowManager,
    R::Error: Send + Sync + 'static,
    E: From<BackendFailure>,
{
    let layout = invocations.storage_layout();
    (0..layout.group_count())
        .filter_map(|group| {
            let range = layout.group_range(group)?;
            (!range.is_empty()).then_some((group, range))
        })
        .map(|(group, range)| {
            let id = layout
                .group_id(group)
                .expect("validated execution group")
                .as_str();
            let depth = window_depth.min(range.len());
            let group = ResidentLayerGroup::new(id, units[range].iter().cloned(), depth)
                .map_err(BackendFailure::from_error)?;
            group
                .report(residency)
                .map_err(BackendFailure::from_error)
                .map_err(E::from)
        })
        .collect()
}

impl<B, U, N, M, T, E> LayerwisePolicy<B, U> for ResidentUnitPolicy<B, N, M, T, E>
where
    B: NeuralBackend,
    U: Parameterized<B::Tensor>,
    N: DenseResidencyMechanisms,
    M: DerefMut<Target = U>,
    E: From<BackendFailure>,
{
    type Lease = ResidentUnit<M>;
    type Error = E;

    fn permits_deferred_serial_groups(&self) -> bool {
        true
    }

    fn publish_parameter_replacements(
        &mut self,
        values: &std::collections::BTreeMap<String, B::Tensor>,
        _active: bool,
    ) -> Result<bool, E> {
        Ok(<Self as LayerwisePolicy<B, U>>::visit_resident_units(
            self,
            &mut |unit| eredu_nn::publish_parameter_replacements(unit, values),
        ))
    }

    fn resident_parameters_available(&self) -> bool {
        self.units.iter().all(Option::is_some)
    }

    fn visit_resident_units(&mut self, visitor: &mut impl FnMut(&mut U)) -> bool {
        if !<Self as LayerwisePolicy<B, U>>::resident_parameters_available(self) {
            return false;
        }
        for unit in &mut self.units {
            visitor(unit.as_deref_mut().expect("idle resident unit"));
        }
        true
    }

    fn begin(
        &mut self,
        _initial: &B::Tensor,
        _stream: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn inspect_unit<BuildError, F, V>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        _build: F,
        operation: V,
        _stream: &<B::Tensor as Tensor>::Context,
    ) -> Result<bool, LayerwiseAcquireError<BuildError, Self::Error>>
    where
        F: FnOnce(&<B::Tensor as Tensor>::Context) -> Result<U, BuildError>,
        V: FnOnce(&mut U) -> Result<(), Self::Error>,
    {
        let invocation = self
            .invocations
            .resolve(ordinal, address)
            .map_err(|error| {
                LayerwiseAcquireError::Policy(BackendFailure::from_error(error).into())
            })?;
        let unit = self
            .units
            .get_mut(invocation.storage_location().ordinal())
            .and_then(Option::as_mut)
            .ok_or_else(|| {
                LayerwiseAcquireError::Policy(
                    BackendFailure::from_error(ResidentUnitError::Unavailable {
                        index: invocation.storage_location().ordinal(),
                    })
                    .into(),
                )
            })?;
        operation(unit).map_err(LayerwiseAcquireError::Policy)?;
        // Resident ownership remains in the outer session's completion scope.
        Ok(true)
    }

    fn abort(
        &mut self,
        active: Option<(usize, ExecutionUnitAddress, Self::Lease)>,
        _stream: &<B::Tensor as Tensor>::Context,
    ) {
        let Some((_, _, lease)) = active else {
            return;
        };
        if let Some(slot) = self.units.get_mut(lease.index) {
            debug_assert!(slot.is_none());
            if slot.is_none() {
                *slot = Some(lease.unit);
            }
        }
    }

    fn acquire<BuildError, BF>(
        &mut self,
        index: usize,
        address: ExecutionUnitAddress,
        _build: BF,
        _stream: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Lease, LayerwiseAcquireError<BuildError, Self::Error>>
    where
        BF: FnOnce(&<B::Tensor as Tensor>::Context) -> Result<U, BuildError>,
    {
        let invocation = self.invocations.resolve(index, address).map_err(|error| {
            LayerwiseAcquireError::Policy(BackendFailure::from_error(error).into())
        })?;
        let index = invocation.storage_location().ordinal();
        let unit = self
            .units
            .get_mut(index)
            .and_then(Option::take)
            .ok_or_else(|| {
                LayerwiseAcquireError::Policy(
                    BackendFailure::from_error(ResidentUnitError::Unavailable { index }).into(),
                )
            })?;
        Ok(ResidentUnit {
            index,
            invocation,
            unit,
        })
    }

    fn complete<'a, StateValues, ContextValues>(
        &mut self,
        index: usize,
        address: ExecutionUnitAddress,
        lease: Self::Lease,
        _output: &'a B::Tensor,
        _state_values: StateValues,
        _context_values: ContextValues,
        _stream: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Self::Error>
    where
        B::Tensor: 'a,
        StateValues: Iterator<Item = &'a B::Tensor>,
        ContextValues: Iterator<Item = &'a B::Tensor>,
    {
        let returned = lease
            .invocation
            .validate_return(index, address)
            .map_err(BackendFailure::from_error)
            .map_err(E::from);
        let slot = &mut self.units[lease.index];
        if slot.is_some() {
            return Err(BackendFailure::from_error(ResidentUnitError::Occupied {
                index: lease.index,
            })
            .into());
        }
        *slot = Some(lease.unit);
        returned
    }

    fn finish(
        &mut self,
        _output: &B::Tensor,
        _stream: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl<B, N, M, T, E> SelectedUnitResources for ResidentUnitPolicy<B, N, M, T, E>
where
    N: DenseResidencyMechanisms,
    E: std::error::Error + Send + Sync + 'static + From<BackendFailure>,
{
    type Error = E;

    fn addresses(&self) -> &ExecutionUnitAddressMap {
        &self.invocations
    }

    fn residency_report(&self) -> Result<ResidencyReport, E> {
        self.residency_report()
    }

    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, E> {
        Ok(None)
    }

    fn execution_group_reports(&self) -> Result<Vec<ResidentLayerGroupReport>, E> {
        self.execution_group_reports()
    }

    fn trim_parameter_conversions(
        &self,
    ) -> Result<Vec<ParameterConversionRetentionTrimReport>, ParameterConversionTrimError> {
        self.trim_parameter_conversions()
    }
}

impl<B, N, M, T, E> UnitResidencyTelemetry for ResidentUnitPolicy<B, N, M, T, E>
where
    N: DenseResidencyMechanisms,
    E: From<BackendFailure>,
{
    type Error = E;

    fn residency_report(&self) -> Result<ResidencyReport, E> {
        self.residency_report()
    }

    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, E> {
        Ok(None)
    }

    fn execution_group_reports(&self) -> Result<Vec<ResidentLayerGroupReport>, E> {
        self.execution_group_reports()
    }
}

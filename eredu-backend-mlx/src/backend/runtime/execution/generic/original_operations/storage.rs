//! Concrete complete slot types. Population authority remains in the selected
//! policy/source traversal, not this storage-only construction worker.
use super::*;
use eredu_runtime::working_memory::OriginalOperationMetadataCustody;
use crate::backend::runtime::{
    checkpoint::store::{
        MaterializationPayloadShape, OriginalMaterializationSlots,
        PreparedMaterializationObservation, PreparedPendingWeight, PreparedWeightMaterialization,
    },
    residency::manager::{
        NameCatalogOwner, NamedPreparationSource, NamedStorageLayout, OriginalResidencySlots,
        PreparedHostMaterialization, PreparedResidentTransfer, PreparedTransferDestinationCause,
        PreparedTransferObservation, TransferPayloadShape,
    },
};

pub(super) mod foreground_disk;
pub(in crate::backend::runtime::execution::generic) use foreground_disk::{
    ForegroundDiskRequestPlan, PreparedSpeculativeForegroundSource,
};
mod population;
pub(super) use population::{ResidencyPopulation, TRANSFERS_PER_NONEMPTY_WINDOW};

pub(super) enum ForegroundDiskSourceTicket {
    Text(eredu_runtime::working_memory::OriginalHostDestinationBank),
    Source {
        bank: eredu_runtime::working_memory::OriginalHostSourceBank,
        custody: eredu_runtime::working_memory::OriginalHostSourceCustody,
    },
}

pub(super) fn unit_factory<U: 'static>(
    controls: Option<OriginalOperationMetadataCustody>,
) -> impl FnMut(usize) -> Result<PreparedUnit<U>, Infallible> {
    move |_| {
        Ok(PreparedUnit::new(
            controls
                .as_ref()
                .expect("accepted original unit custody")
                .clone(),
        ))
    }
}
pub(super) fn unit_layout<U: 'static, F: FnMut(usize) -> Result<PreparedUnit<U>, Infallible>>(
    count: usize,
    _factory: &F,
) -> Option<u64> {
    Some(
        PreparedOperationBank::<PreparedUnit<U>>::layout::<F, Infallible>(
            count,
            PreparedUnit::<U>::control_bytes()?,
        )?
        .total_control_bytes,
    )
}
macro_rules! slot_factory {
    ($factory:ident,$layout:ident,$slot:ty) => {
        slot_factory!($factory, $layout, $slot, OriginalTextControlGuard, |value| value);
    };
    ($factory:ident,$layout:ident,$slot:ty,$custody:ty,$project:expr) => {
        fn $factory(
            controls: Option<$custody>,
        ) -> impl FnMut(usize) -> Result<$slot, Infallible> {
            move |_| {
                Ok(<$slot>::new(($project)(
                    controls
                        .as_ref()
                        .expect("accepted original slot custody")
                        .clone(),
                )))
            }
        }
        fn $layout(count: usize) -> Option<u64> {
            fn measured<F: FnMut(usize) -> Result<$slot, Infallible>>(
                count: usize,
                _factory: &F,
            ) -> Option<u64> {
                Some(<$slot>::bank_layout::<F, Infallible>(count)?.total_control_bytes)
            }
            measured(count, &$factory(None))
        }
    };
}
// The error owns the partially reserved final slot. Measure this exact factory
// and error, rather than the infallible fixed-node constructor's storage.
macro_rules! payload_slot_factory {
    ($factory:ident,$layout:ident,$slot:ty,$shape:ty) => {
        fn $factory(
            controls: Option<OriginalTextControlGuard>,
            shape: $shape,
        ) -> impl FnMut(usize) -> Result<$slot, ($slot, TryReserveError)> {
            move |_| {
                <$slot>::try_new(
                    controls
                        .as_ref()
                        .expect("accepted original payload custody")
                        .clone(),
                    shape,
                )
            }
        }
        fn $layout(count: usize, shape: $shape) -> Option<u64> {
            fn measured<F: FnMut(usize) -> Result<$slot, ($slot, TryReserveError)>>(
                count: usize,
                shape: $shape,
                _factory: &F,
            ) -> Option<u64> {
                Some(
                    <$slot>::bank_layout_with_payload::<F, ($slot, TryReserveError)>(count, shape)?
                        .total_control_bytes,
                )
            }
            measured(count, shape, &$factory(None, shape))
        }
    };
}

fn payload_bank_error<S, F>(
    error: BankPreparationError<S, F, (S, TryReserveError)>,
    controls: &OriginalOperationMetadataCustody,
) -> Error {
    let (cause, prefix, factory) = error.into_parts();
    let error = match cause {
        BankPreparationCause::Overflow => overflow(),
        BankPreparationCause::Reserve(cause) => reserve_error(cause, controls),
        BankPreparationCause::Slot {
            cause: (partial, cause),
            ..
        } => {
            // The escaping error obtains its Q guard before any final slot or
            // earlier successful slot retires. These slots have never activated.
            let error = reserve_error(cause, controls);
            drop(partial);
            error
        }
    };
    drop(prefix);
    drop(factory);
    error
}

fn transfer_factory<'a>(
    tier: MemoryTier,
    controls: Option<OriginalOperationMetadataCustody>,
    shape: TransferPayloadShape,
    manager: Option<&'a ResidencyManager>,
    catalog: Option<&'a NameCatalogOwner>,
    roots: &'a [OffloadUnitId],
    scratch: &'a mut [eredu_runtime::residency::ResidencyClosureSlot],
) -> impl FnMut(
    usize,
) -> Result<
    PreparedResidentTransfer,
    (PreparedResidentTransfer, PreparedTransferDestinationCause),
> + 'a {
    move |_| {
        PreparedResidentTransfer::try_new_for_window_in_tier(
            controls
                .as_ref()
                .expect("accepted named transfer custody")
                .clone(),
            shape,
            manager.expect("actual manager loan"),
            catalog.expect("actual name catalog"),
            roots,
            scratch,
            tier,
        )
    }
}
fn transfer_layout(
    count: usize,
    shape: TransferPayloadShape,
    names: NamedStorageLayout,
) -> Option<u64> {
    fn measured<
        F: FnMut(
            usize,
        ) -> Result<
            PreparedResidentTransfer,
            (PreparedResidentTransfer, PreparedTransferDestinationCause),
        >,
    >(
        count: usize,
        shape: TransferPayloadShape,
        names: NamedStorageLayout,
        _factory: &F,
    ) -> Option<u64> {
        Some(
            PreparedResidentTransfer::bank_layout_with_named_payload::<
                F,
                (PreparedResidentTransfer, PreparedTransferDestinationCause),
            >(count, shape, names)?
            .total_control_bytes,
        )
    }
    measured(
        count,
        shape,
        names,
        &transfer_factory(MemoryTier::Device, None, shape, None, None, &[], &mut []),
    )
}

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct NamedPreparationFailure {
    #[source]
    cause: NamedPreparationSource,
    controls: OriginalOperationMetadataCustody,
}
fn named_error(cause: NamedPreparationSource, controls: &OriginalOperationMetadataCustody) -> Error {
    Error::with_original_control_source(
        eredu_core::BackendFailure::from_error(NamedPreparationFailure {
            cause,
            controls: controls.clone(),
        }),
        false,
    )
}
pub(super) fn named_error_control_bytes() -> Option<u64> {
    u64::try_from(eredu_core::BackendFailure::source_retention_peak_bytes::<
        NamedPreparationFailure,
    >()?.checked_add(size_of::<eredu_runtime::working_memory::OriginalOperationMetadataCustody>())?
        .checked_add(size_of::<eredu_runtime::working_memory::OriginalTextMetadataCustody>())?)
    .ok()
}
fn named_bank_error<F>(
    error: BankPreparationError<
        PreparedResidentTransfer,
        F,
        (PreparedResidentTransfer, PreparedTransferDestinationCause),
    >,
    controls: &OriginalOperationMetadataCustody,
) -> Error {
    let (cause, prefix, factory) = error.into_parts();
    let error = match cause {
        BankPreparationCause::Overflow => overflow(),
        BankPreparationCause::Reserve(cause) => reserve_error(cause, controls),
        BankPreparationCause::Slot {
            cause: (partial, cause),
            ..
        } => {
            let error = named_error(cause.into_source(), controls);
            drop(partial);
            error
        }
    };
    drop(prefix);
    drop(factory);
    error
}
slot_factory!(
    observation_factory,
    observation_layout,
    PreparedTransferObservation,
    OriginalOperationMetadataCustody,
    |value| value
);
slot_factory!(host_factory, host_layout, PreparedHostMaterialization,
    OriginalOperationMetadataCustody,
    |value| value);
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct SourceBankFailure {
    #[source]
    cause: eredu_runtime::working_memory::HostDestinationCause,
    _controls: OriginalTextControlGuard,
}
fn source_bank_error(
    cause: eredu_runtime::working_memory::HostDestinationCause,
    controls: &OriginalTextControlGuard,
) -> Error {
    Error::with_original_control_source(
        eredu_core::BackendFailure::from_error(SourceBankFailure {
            cause,
            _controls: controls.clone(),
        }),
        false,
    )
}
fn source_pending_error<S, F>(
    error: BankPreparationError<S, F, Error>,
    controls: &OriginalOperationMetadataCustody,
) -> Error {
    let (cause, prefix, factory) = error.into_parts();
    let error = match cause {
        BankPreparationCause::Overflow => overflow(),
        BankPreparationCause::Reserve(cause) => reserve_error(cause, controls),
        BankPreparationCause::Slot { cause, .. } => cause,
    };
    drop(prefix);
    drop(factory);
    error
}
fn pending_factory<'a>(
    controls: Option<OriginalTextControlGuard>,
    runtime: Option<Rc<safemlx::PreparedInputRuntime>>,
    mut sources: Option<(
        super::gguf_host::source_arenas::SourceArenaWindow,
        &'a mut eredu_runtime::working_memory::OriginalHostDestinationBank,
    )>,
) -> impl FnMut(usize) -> Result<PreparedPendingWeight, Error> + 'a {
    move |_| {
        let controls = controls.as_ref().expect("accepted pending custody");
        let runtime = runtime.as_ref().expect("actual prepared host-copy runtime");
        if let Some((bound, bank)) = sources.as_mut() {
            let child = bank
                .split(
                    bound.destination_bytes,
                    bound.destination_attempts,
                    Some((bound.bytes_per_pending, bound.attempts_per_pending)),
                )
                .map_err(|cause| source_bank_error(cause, controls))?;
            PreparedPendingWeight::new_with_gguf_host_destinations(
                controls.clone(),
                Rc::clone(runtime),
                child,
            )
            .map_err(|(bank, cause)| {
                let error = source_bank_error(
                    eredu_runtime::working_memory::HostDestinationCause::Memory(cause),
                    controls,
                );
                drop(bank);
                error
            })
        } else {
            Ok(PreparedPendingWeight::new_with_gguf_host_copy(
                controls.clone(),
                Rc::clone(runtime),
            ))
        }
    }
}
fn pending_layout(count: usize) -> Option<u64> {
    fn measured<F: FnMut(usize) -> Result<PreparedPendingWeight, Error>>(
        count: usize,
        _: &F,
    ) -> Option<u64> {
        let failure = u64::try_from(eredu_core::BackendFailure::source_retention_peak_bytes::<
            SourceBankFailure,
        >()?)
        .ok()?;
        PreparedPendingWeight::bank_layout::<F, Error>(count)?
            .total_control_bytes
            .checked_add(failure.checked_mul(u64::try_from(count).ok()?)?)
    }
    measured(count, &pending_factory(None, None, None))
}
payload_slot_factory!(
    weight_factory,
    weight_layout,
    PreparedWeightMaterialization,
    MaterializationPayloadShape
);
slot_factory!(
    materialization_factory,
    materialization_layout,
    PreparedMaterializationObservation
);

mod windows;
pub(super) use windows::{PreparedResidencyAttempt, PreparedSelectedResidency};

pub(super) struct PreparedStorage<U: 'static> {
    closure: Vec<eredu_runtime::residency::ResidencyClosureSlot>,
    pub(super) units: PreparedOperationBank<PreparedUnit<U>>,
    pub(super) neural: PreparedOperationBank<super::neural::PreparedNeuralSubmission>,
    windows: Vec<PreparedOperationBank<PreparedResidencyAttempt>>,
    host: Option<HostAcquisitions>,
    pub(super) background: Option<crate::backend::runtime::residency::dense_stream::BackgroundHostCoordinator>,
}
struct HostAcquisitions {
    closure: Vec<eredu_runtime::residency::ResidencyClosureSlot>,
    windows: Vec<PreparedOperationBank<PreparedResidencyAttempt>>,
}
pub(super) fn background_acquisition_control_bytes(sources: &[crate::backend::runtime::residency::manager::WindowPopulation], forwards: usize, controller_units: usize) -> Option<u64> {
    let fixed = [size_of::<HostAcquisitions>(), size_of::<Option<HostAcquisitions>>(),
        Layout::array::<eredu_runtime::residency::ResidencyClosureSlot>(controller_units).ok()?.size(),
        size_of::<Vec<eredu_runtime::residency::ResidencyClosureSlot>>(),
        size_of::<Option<crate::backend::runtime::residency::dense_stream::BackgroundHostCoordinator>>(),
        size_of::<std::cell::RefMut<'_, Option<crate::backend::runtime::residency::dense_stream::BackgroundHostCoordinator>>>(),
        size_of::<Result<(), Error>>()];
    windows::layout(sources, forwards)?.checked_add(u64::try_from(fixed.into_iter().try_fold(size_of::<[usize;7]>(), usize::checked_add)?).ok()?)
}
impl<U: 'static> PreparedStorage<U> {
    pub(super) fn new(
        units: usize,
        population: ResidencyPopulation,
        neural: NeuralPopulation,
        gguf_host_runtime: Option<&Rc<safemlx::PreparedInputRuntime>>,
        sources: &[crate::backend::runtime::residency::manager::WindowPopulation],
        manager: &ResidencyManager,
        ids: &[OffloadUnitId],
        controls: &OriginalTextControlGuard,
        disk: Option<&ForegroundDiskRequestPlan>,
        reservation: &eredu_runtime::working_memory::WorkingMemoryReservation,
        source: Option<(
            &super::gguf_host::source_arenas::SourceArenaPlan,
            eredu_runtime::working_memory::OriginalHostDestinationBank,
        )>,
        disk_source: Option<eredu_runtime::working_memory::OriginalHostDestinationBank>,
    ) -> Result<Self, Error> {
        let custody = controls.metadata_custody().into();
        Self::new_with_custody(units, population, neural, gguf_host_runtime, sources,
            manager, ids, &custody, Some(controls), disk, Some(reservation), source, disk_source.map(ForegroundDiskSourceTicket::Text))
    }

    pub(super) fn new_with_custody(
        units: usize,
        population: ResidencyPopulation,
        neural: NeuralPopulation,
        gguf_host_runtime: Option<&Rc<safemlx::PreparedInputRuntime>>,
        sources: &[crate::backend::runtime::residency::manager::WindowPopulation],
        manager: &ResidencyManager,
        ids: &[OffloadUnitId],
        custody: &OriginalOperationMetadataCustody,
        controls: Option<&OriginalTextControlGuard>,
        disk: Option<&ForegroundDiskRequestPlan>,
        reservation: Option<&eredu_runtime::working_memory::WorkingMemoryReservation>,
        mut source: Option<(
            &super::gguf_host::source_arenas::SourceArenaPlan,
            eredu_runtime::working_memory::OriginalHostDestinationBank,
        )>,
        disk_source: Option<ForegroundDiskSourceTicket>,
    ) -> Result<Self, Error> {
        if controls.is_none() && (reservation.is_some() || source.is_some()
            || matches!(disk_source.as_ref(), Some(ForegroundDiskSourceTicket::Text(_))) || gguf_host_runtime.is_some()) {
            return Err(identity());
        }
        let mut closure = Vec::new();
        closure
            .try_reserve_exact(population.controller_units)
            .map_err(|error| reserve_error(error, custody))?;
        closure.resize(
            population.controller_units,
            eredu_runtime::residency::ResidencyClosureSlot::default(),
        );
        let units = PreparedOperationBank::try_new(units, unit_factory(Some(custody.clone())))
            .map_err(|e| bank_error(e, custody))?;
        let neural = super::neural::prepare_with_custody(neural, custody.clone())?;
        let catalog = manager
            .prepare_name_catalog(ids, &mut closure, custody.clone())
            .map_err(|cause| named_error(cause.into_source(), custody))?;
        let (windows, background) = windows::prepare(
            sources,
            population.forwards,
            MemoryTier::Device,
            gguf_host_runtime,
            custody,
            controls,
            manager,
            &catalog,
            ids,
            &mut closure,
            source.as_mut().map(|(plan, bank)| (*plan, bank)),
            disk,
            reservation,
            disk_source,
        )?;
        let host = if background.is_some() {
            let mut host_closure = Vec::new();
            host_closure.try_reserve_exact(population.controller_units).map_err(|cause| reserve_error(cause, custody))?;
            host_closure.resize(population.controller_units, eredu_runtime::residency::ResidencyClosureSlot::default());
            let (host_windows, absent) = windows::prepare(disk.and_then(ForegroundDiskRequestPlan::background).ok_or_else(identity)?.host_acquisitions(), population.forwards, MemoryTier::Host, None, custody, controls, manager, &catalog,
                ids, &mut host_closure, None, None, reservation, None)?;
            debug_assert!(absent.is_none());
            Some(HostAcquisitions { closure: host_closure, windows: host_windows })
        } else { None };
        Ok(Self { closure, units, neural, windows, host, background })
    }
    pub(super) fn with_host_and_device<T>(
        &mut self, index: usize, device: &mut PreparedResidencyAttempt,
        reservation: Option<&eredu_runtime::working_memory::WorkingMemoryReservation>,
        execute: impl FnOnce(&mut OriginalResidencySlots<'_>, &mut OriginalResidencySlots<'_>) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let host = self.host.as_mut().ok_or_else(identity)?;
        let mut attempt = windows::checkout(&mut host.windows, index).map_err(|_| Error::PrefillScopeUnavailable)?;
        execute(&mut attempt.residency(&mut host.closure, reservation), &mut device.residency(&mut self.closure, reservation))
    }
    pub(super) fn checkout_residency(
        &mut self,
        index: usize,
    ) -> Result<PreparedResidencyAttempt, Error> {
        windows::checkout(&mut self.windows, index).map_err(|error| match error {
            windows::WindowCheckoutFailure::UnknownWindow => identity(),
            windows::WindowCheckoutFailure::Exhausted => Error::PrefillScopeUnavailable,
        })
    }
    pub(super) fn residency<'a>(
        &'a mut self,
        attempt: &'a mut PreparedResidencyAttempt,
        reservation: Option<&'a eredu_runtime::working_memory::WorkingMemoryReservation>,
    ) -> OriginalResidencySlots<'a> {
        attempt.residency(&mut self.closure, reservation)
    }
}

#[cfg(test)]
mod tests;

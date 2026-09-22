//! One finite bank per actual layout ordinal. Each checkout owns one complete
//! attempt, so skipped warm/miss work cannot move another window's cursor.
use super::*;
use crate::backend::runtime::residency::manager::{
    ClosurePreparationCause, ClosurePreparationError, ControllerPreparationCause,
    ControllerPreparationError, ForegroundDiskSourceCapacity, ForegroundDiskWindowPlan,
    PreparedClosureIds, PreparedControllerAttempt, PreparedForegroundDiskSlots, WindowPopulation,
};

mod selected;
pub(in crate::backend::runtime::execution::generic::original_operations) use selected::PreparedSelectedResidency;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WindowCheckoutFailure {
    UnknownWindow,
    Exhausted,
}
// Shared cursor transition: the index selects a bank before its one-way
// checkout. No omission or exhaustion can consume another bank's capacity.
pub(super) fn checkout<S>(
    windows: &mut [PreparedOperationBank<S>],
    index: usize,
) -> Result<S, WindowCheckoutFailure> {
    windows
        .get_mut(index)
        .ok_or(WindowCheckoutFailure::UnknownWindow)?
        .checkout()
        .map_err(|_| WindowCheckoutFailure::Exhausted)
}

pub(crate) struct PreparedResidencyAttempt {
    acquisitions: crate::backend::runtime::checkpoint::store::PreparedSourceAcquisitions,
    controller: Option<PreparedControllerAttempt>,
    closure_ids: Option<PreparedClosureIds>,
    transfers: PreparedOperationBank<PreparedResidentTransfer>,
    observations: PreparedOperationBank<PreparedTransferObservation>,
    host: PreparedOperationBank<PreparedHostMaterialization>,
    pending: PreparedOperationBank<PreparedPendingWeight>,
    weight: PreparedOperationBank<PreparedWeightMaterialization>,
    materialization: PreparedOperationBank<PreparedMaterializationObservation>,
    foreground_disk: PreparedForegroundDiskSlots,
}
impl PreparedResidencyAttempt {
    fn new(
        source: WindowPopulation,
        tier: MemoryTier,
        gguf_host_runtime: Option<&Rc<safemlx::PreparedInputRuntime>>,
        custody: &OriginalOperationMetadataCustody,
        controls: Option<&OriginalTextControlGuard>,
        manager: &ResidencyManager,
        catalog: &NameCatalogOwner,
        ids: &[OffloadUnitId],
        scratch: &mut [eredu_runtime::residency::ResidencyClosureSlot],
        source_bound: super::super::gguf_host::source_arenas::SourceArenaWindow,
        acquisitions: &[crate::backend::runtime::residency::manager::acquisition_destinations::SelectedSourceOccurrence],
        mut source_bank: Option<&mut eredu_runtime::working_memory::OriginalHostDestinationBank>,
        disk: Option<(
            &ForegroundDiskWindowPlan,
            &eredu_runtime::working_memory::MemoryLedger,
            &ForegroundDiskSourceCapacity,
        )>,
        reservation: Option<&eredu_runtime::working_memory::WorkingMemoryReservation>,
    ) -> Result<Self, Error> {
        if let Some(bank) = &source_bank {
            if gguf_host_runtime.is_none()
                || source_bound.pending != population::AttemptPopulation::new(source)?.pending
                || !bank.belongs_to(controls.ok_or_else(identity)?)
            {
                return Err(identity());
            }
        }
        let roots = ids
            .get(source.request_start..source.request_end)
            .ok_or_else(identity)?;
        if roots.len() != source.requested {
            return Err(identity());
        }
        let closure_ids = if roots.is_empty() {
            None
        } else {
            Some(
                manager
                    .prepare_closure_ids(
                        roots,
                        scratch,
                        source.units,
                        source.unit_id_bytes,
                        custody.clone(),
                    )
                    .map_err(|error| closure_error(error, custody))?,
            )
        };
        let controller = if roots.is_empty() {
            None
        } else {
            Some(
                manager
                    .prepare_controller_attempt(source, custody.clone())
                    .map_err(|error| controller_error(error, custody))?,
            )
        };
        if disk.is_some() && source_bank.is_some() {
            return Err(identity());
        }
        if let Some((_, _, capacity)) = disk {
            if !capacity.custody().metadata_custody().same_account(custody) {
                return Err(identity());
            }
        }
        let foreground_disk = match disk {
            Some((plan, pool, capacity)) => plan
                .prepare_source(pool, capacity.custody().clone(), reservation, capacity)
                .map_err(|cause| {
                    Error::with_original_control_source(
                        eredu_core::BackendFailure::from_error(cause),
                        false,
                    )
                })?,
            None => PreparedForegroundDiskSlots::unavailable(),
        };
        let retained_route = source_bank.is_none();
        let acquisitions = if let Some(bank) = source_bank.as_deref_mut() {
            let controls = controls.ok_or_else(identity)?;
            let mut acquisition_bank = bank
                .split(0, 0, Some((source_bound.acquisition_bytes, 1)))
                .map_err(|cause| source_bank_error(cause, controls))?;
            crate::backend::runtime::checkpoint::store::PreparedSourceAcquisitions::prepare_selected(
            acquisitions, source_bound.pending, source_bound.acquisition_bytes,
            acquisition_bank.take_source_constructions().expect("paired source component"), controls.clone(),
        ).map_err(|cause| acquisition_error(
            crate::backend::runtime::checkpoint::store::CheckpointMaterializationError::from(cause).into(), custody))?
        } else {
            crate::backend::runtime::checkpoint::store::PreparedSourceAcquisitions::unavailable(custody.clone())
                .map_err(|cause| acquisition_error(
                    crate::backend::runtime::checkpoint::store::CheckpointMaterializationError::from(cause).into(), custody))?
        };
        let population = population::AttemptPopulation::new(source)?;
        let transfers = PreparedOperationBank::try_new(
            population.transfers,
            transfer_factory(
                tier,
                Some(custody.clone()),
                population.transfer,
                Some(manager),
                Some(catalog),
                roots,
                scratch,
            ),
        )
        .map_err(|e| named_bank_error(e, custody))?;
        let observations = PreparedOperationBank::try_new(
            population.observations,
            observation_factory(Some(custody.clone())),
        )
        .map_err(|e| bank_error(e, custody))?;
        let host = PreparedOperationBank::try_new(0, host_factory(Some(custody.clone())))
            .map_err(|e| bank_error(e, custody))?;
        let pending = PreparedOperationBank::try_new(
            if retained_route {
                0
            } else {
                population.pending
            },
            pending_factory(
                controls.cloned(),
                gguf_host_runtime.cloned(),
                source_bank.map(|bank| (source_bound, bank)),
            ),
        )
        .map_err(|e| source_pending_error(e, custody))?;
        let weight = PreparedOperationBank::try_new(
            if retained_route { 0 } else { population.weight },
            weight_factory(controls.cloned(), population.materialization),
        )
        .map_err(|e| payload_bank_error(e, custody))?;
        let materialization =
            PreparedOperationBank::try_new(0, materialization_factory(controls.cloned()))
                .map_err(|e| bank_error(e, custody))?;
        Ok(Self {
            acquisitions,
            controller,
            closure_ids,
            transfers,
            observations,
            host,
            pending,
            weight,
            materialization,
            foreground_disk,
        })
    }
    fn control_bytes(source: WindowPopulation) -> Option<u64> {
        let p = population::AttemptPopulation::new(source).ok()?;
        [
            0, // G4 bank allocation/failure layouts are in the paired source component.
            acquisition_error_control_bytes()?,
            if source.requested == 0 {
                0
            } else {
                PreparedControllerAttempt::control_bytes(
                    source.controller_units,
                    source.requested,
                    source.units,
                    source.maximum_id_bytes,
                )?
            },
            controller_error_control_bytes()?,
            PreparedClosureIds::control_bytes(source.units, source.unit_id_bytes)?,
            closure_error_control_bytes()?,
            transfer_layout(p.transfers, p.transfer, source.named)?,
            observation_layout(p.observations)?,
            host_layout(0)?,
            pending_layout(p.pending)?,
            weight_layout(p.weight, p.materialization)?,
            materialization_layout(0)?,
        ]
        .into_iter()
        .try_fold(0u64, u64::checked_add)
    }
    pub(super) fn residency<'a>(
        &'a mut self,
        closure: &'a mut [eredu_runtime::residency::ResidencyClosureSlot],
        reservation: Option<&'a eredu_runtime::working_memory::WorkingMemoryReservation>,
    ) -> OriginalResidencySlots<'a> {
        OriginalResidencySlots {
            materialized_recipe: None,
            background_host: None,
            controller: &mut self.controller,
            closure_ids: &mut self.closure_ids,
            closure,
            transfers: &mut self.transfers,
            observations: &mut self.observations,
            host_materializations: &mut self.host,
            reservation,
            foreground_disk: &mut self.foreground_disk,
            materialization: OriginalMaterializationSlots {
                acquisitions: &mut self.acquisitions,
                pending_weights: &mut self.pending,
                weight_materializations: &mut self.weight,
                observations: &mut self.materialization,
            },
        }
    }
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct ControllerPreparationFailure {
    #[source]
    cause: ControllerPreparationCause,
    _controls: OriginalOperationMetadataCustody,
}
fn controller_error(
    error: ControllerPreparationError,
    controls: &OriginalOperationMetadataCustody,
) -> Error {
    let ControllerPreparationError { cause, prefix } = error;
    let error = Error::with_original_control_source(
        eredu_core::BackendFailure::from_error(ControllerPreparationFailure {
            cause,
            _controls: controls.clone(),
        }),
        false,
    );
    drop(prefix);
    error
}
fn controller_error_control_bytes() -> Option<u64> {
    u64::try_from(eredu_core::BackendFailure::source_retention_peak_bytes::<
        ControllerPreparationFailure,
    >()?)
    .ok()
}

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct ClosurePreparationFailure {
    #[source]
    cause: ClosurePreparationCause,
    _controls: OriginalOperationMetadataCustody,
}
fn closure_error(
    error: ClosurePreparationError,
    controls: &OriginalOperationMetadataCustody,
) -> Error {
    let ClosurePreparationError { cause, prefix } = error;
    let error = Error::with_original_control_source(
        eredu_core::BackendFailure::from_error(ClosurePreparationFailure {
            cause,
            _controls: controls.clone(),
        }),
        false,
    );
    // The escaping owner retains the accepted controls before the unactivated
    // prefix's IDs, Vec allocation and Weak header are destroyed.
    drop(prefix);
    error
}
fn closure_error_control_bytes() -> Option<u64> {
    u64::try_from(eredu_core::BackendFailure::source_retention_peak_bytes::<
        ClosurePreparationFailure,
    >()?)
    .ok()
}

fn factory<'a>(
    source: WindowPopulation,
    tier: MemoryTier,
    gguf_host_runtime: Option<Rc<safemlx::PreparedInputRuntime>>,
    custody: Option<OriginalOperationMetadataCustody>,
    controls: Option<&'a OriginalTextControlGuard>,
    manager: Option<&'a ResidencyManager>,
    catalog: Option<&'a NameCatalogOwner>,
    ids: &'a [OffloadUnitId],
    scratch: &'a mut [eredu_runtime::residency::ResidencyClosureSlot],
    source_bound: super::super::gguf_host::source_arenas::SourceArenaWindow,
    acquisitions: &'a [crate::backend::runtime::residency::manager::acquisition_destinations::SelectedSourceOccurrence],
    mut source_bank: Option<&'a mut eredu_runtime::working_memory::OriginalHostDestinationBank>,
    disk: Option<(
        &'a ForegroundDiskWindowPlan,
        &'a eredu_runtime::working_memory::MemoryLedger,
        &'a ForegroundDiskSourceCapacity,
    )>,
    reservation: Option<&'a eredu_runtime::working_memory::WorkingMemoryReservation>,
) -> impl FnMut(usize) -> Result<PreparedResidencyAttempt, Error> + 'a {
    move |_| {
        PreparedResidencyAttempt::new(
            source,
            tier,
            gguf_host_runtime.as_ref(),
            custody
                .as_ref()
                .expect("accepted window preparation custody"),
            controls,
            manager.expect("actual manager loan"),
            catalog.expect("actual name catalog"),
            ids,
            scratch,
            source_bound,
            acquisitions,
            source_bank.as_deref_mut(),
            disk,
            reservation,
        )
    }
}

fn window_layout(source: WindowPopulation, forwards: usize) -> Option<u64> {
    fn measured<F: FnMut(usize) -> Result<PreparedResidencyAttempt, Error>>(
        source: WindowPopulation,
        forwards: usize,
        _factory: &F,
    ) -> Option<u64> {
        Some(
            PreparedOperationBank::<PreparedResidencyAttempt>::layout::<F, Error>(
                forwards,
                PreparedResidencyAttempt::control_bytes(source)?,
            )?
            .total_control_bytes,
        )
    }
    measured(
        source,
        forwards,
        &factory(
            source,
            MemoryTier::Device,
            None,
            None,
            None,
            None,
            None,
            &[],
            &mut [],
            Default::default(),
            &[],
            None,
            None,
            None,
        ),
    )
}
pub(super) fn layout(sources: &[WindowPopulation], forwards: usize) -> Option<u64> {
    let arrays = Layout::array::<PreparedOperationBank<PreparedResidencyAttempt>>(sources.len())
        .ok()?
        .size();
    let controls = [
        size_of::<Vec<PreparedOperationBank<PreparedResidencyAttempt>>>(),
        size_of::<Result<Vec<PreparedOperationBank<PreparedResidencyAttempt>>, Error>>(),
        size_of::<population::AttemptPopulation>(),
        size_of::<Result<population::AttemptPopulation, Error>>(),
        size_of::<WindowPopulation>(),
        size_of::<WindowCheckoutFailure>(),
        size_of::<ForegroundDiskSourceTicket>(),
        size_of::<Option<ForegroundDiskSourceTicket>>(),
        size_of::<Result<PreparedResidencyAttempt, WindowCheckoutFailure>>(),
        size_of::<usize>(),
        size_of::<usize>(),
    ]
    .into_iter()
    .try_fold(arrays, usize::checked_add)?;
    sources
        .iter()
        .try_fold(u64::try_from(controls).ok()?, |sum, source| {
            sum.checked_add(window_layout(*source, forwards)?)
        })
}
fn preparation_error<F>(
    error: BankPreparationError<PreparedResidencyAttempt, F, Error>,
    controls: &OriginalOperationMetadataCustody,
) -> Error {
    let (cause, prefix, factory) = error.into_parts();
    let error = match cause {
        BankPreparationCause::Overflow => overflow(),
        BankPreparationCause::Reserve(cause) => reserve_error(cause, controls),
        BankPreparationCause::Slot { cause, .. } => cause,
    };
    // Slot failures already retain the accepted guard. Retire successful prior
    // attempts and the preparation factory while that escaping owner remains.
    drop(prefix);
    drop(factory);
    error
}
pub(super) fn prepare(
    sources: &[WindowPopulation],
    forwards: usize,
    tier: MemoryTier,
    gguf_host_runtime: Option<&Rc<safemlx::PreparedInputRuntime>>,
    custody: &OriginalOperationMetadataCustody,
    controls: Option<&OriginalTextControlGuard>,
    manager: &ResidencyManager,
    catalog: &NameCatalogOwner,
    ids: &[OffloadUnitId],
    scratch: &mut [eredu_runtime::residency::ResidencyClosureSlot],
    mut source_owner: Option<(
        &super::super::gguf_host::source_arenas::SourceArenaPlan,
        &mut eredu_runtime::working_memory::OriginalHostDestinationBank,
    )>,
    disk: Option<&ForegroundDiskRequestPlan>,
    reservation: Option<&eredu_runtime::working_memory::WorkingMemoryReservation>,
    disk_source: Option<ForegroundDiskSourceTicket>,
) -> Result<
    (
        Vec<PreparedOperationBank<PreparedResidencyAttempt>>,
        Option<crate::backend::runtime::residency::dense_stream::BackgroundHostCoordinator>,
    ),
    Error,
> {
    if !matches!(tier, MemoryTier::Host | MemoryTier::Device)
        || (tier == MemoryTier::Host
            && (disk.is_some() || source_owner.is_some() || gguf_host_runtime.is_some()))
    {
        return Err(identity());
    }
    if disk.is_some_and(|plan| !plan.matches(sources.len(), forwards)) {
        return Err(identity());
    }
    // One native retirement counter for the entire request, created only
    // after exact reservation/domain authentication. Slots retain cheap clones;
    // they allocate final source buffers only when their read is consumed.
    let capacity = match (disk, disk_source) {
        (Some(plan), Some(ForegroundDiskSourceTicket::Text(bank))) => Some(plan.prepare_capacity(
            controls.ok_or_else(identity)?.clone(),
            reservation.ok_or_else(identity)?,
            bank,
        )?),
        (
            Some(plan),
            Some(ForegroundDiskSourceTicket::Source {
                bank,
                custody: source,
            }),
        ) => {
            if controls.is_some()
                || reservation.is_some()
                || !source.metadata_custody().same_account(custody)
            {
                return Err(identity());
            }
            Some(plan.prepare_source_capacity(source, None, bank)?)
        }
        (None, None) => None,
        _ => return Err(identity()),
    };
    let background = match disk.and_then(ForegroundDiskRequestPlan::background) {
        Some(background) => {
            let capacity = capacity.as_ref().ok_or_else(identity)?;
            Some(background.prepare(
                manager,
                disk.expect("paired plan").pool(),
                ids,
                disk.expect("paired plan").windows(),
                capacity.custody().clone(),
                reservation,
                capacity,
            )?)
        }
        None => None,
    };
    let mut windows = Vec::new();
    windows
        .try_reserve_exact(sources.len())
        .map_err(|error| reserve_error(error, custody))?;
    for (ordinal, source) in sources.iter().enumerate() {
        let (bound, acquisitions, source_bank) = match source_owner.as_mut() {
            Some((plan, bank)) => {
                let plan = *plan;
                (
                    plan.window(ordinal).ok_or_else(identity)?,
                    plan.acquisition_plans(ordinal).ok_or_else(identity)?,
                    Some(&mut **bank),
                )
            }
            None => (Default::default(), &[][..], None),
        };
        let bank = PreparedOperationBank::try_new(
            forwards,
            factory(
                *source,
                tier,
                gguf_host_runtime.cloned(),
                Some(custody.clone()),
                controls,
                Some(manager),
                Some(catalog),
                ids,
                scratch,
                bound,
                acquisitions,
                source_bank,
                disk.map(|plan| {
                    (
                        match plan.background() {
                            Some(background) => background
                                .direct(ordinal)
                                .expect("validated direct window count"),
                            None => plan.window(ordinal).expect("validated window count"),
                        },
                        plan.pool(),
                        capacity.as_ref().expect("paired source capacity"),
                    )
                }),
                reservation,
            ),
        )
        .map_err(|error| preparation_error(error, custody))?;
        windows.push(bank);
    }
    Ok((windows, background))
}

#[cfg(test)]
mod tests;

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct AcquisitionPreparationFailure {
    #[source]
    cause: crate::backend::runtime::residency::manager::ResidencyError,
    _controls: OriginalOperationMetadataCustody,
}
fn acquisition_error(
    cause: crate::backend::runtime::residency::manager::ResidencyError,
    controls: &OriginalOperationMetadataCustody,
) -> Error {
    Error::with_original_control_source(
        eredu_core::BackendFailure::from_error(AcquisitionPreparationFailure {
            cause,
            _controls: controls.clone(),
        }),
        false,
    )
}
fn acquisition_error_control_bytes() -> Option<u64> {
    u64::try_from(eredu_core::BackendFailure::source_retention_peak_bytes::<
        AcquisitionPreparationFailure,
    >()?)
    .ok()
}

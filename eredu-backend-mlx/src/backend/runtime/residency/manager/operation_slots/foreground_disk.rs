//! Keyed, once-only read metadata for one actual canonical window.
use super::super::{
    ForegroundDiskDescriptors, ForegroundDiskReadError, ForegroundDiskReadLayout,
    PreparedForegroundDiskRead, ResidencyError, ResidencyManager, WindowPopulation,
};
use crate::backend::{
    Error,
    submission_recovery::observed::bank::{BankPreparationCause, PreparedOperationBank},
};
use eredu_core::residency::OffloadUnitId;
use eredu_runtime::{
    residency::ResidencyClosureSlot,
    working_memory::{
        MemoryLedger, OriginalHostSourceCustody, WorkingMemoryError, WorkingMemoryReservation,
    },
};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    mem::{size_of, size_of_val},
    sync::TryLockError,
};

mod capacity;
mod series;
pub(crate) use capacity::ForegroundDiskSourceCapacity;
pub(crate) use series::ForegroundDiskSourceSeries;

/// The shared unit worker permits its initial attempt plus one whole-unit retry.
/// Warm transfers and per-binding recipe retries do not execute this batch reader.
pub(crate) const FOREGROUND_DISK_ATTEMPTS_PER_UNIT: usize = 2;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ForegroundDiskPopulation {
    pub(crate) attempts: usize,
    pub(crate) outputs: usize,
    pub(crate) sources: usize,
    pub(crate) materialized: super::super::construction::ForegroundMaterializationPopulation,
    pub(crate) output_logical_bytes: usize,
    pub(crate) source_backing_bytes: usize,
    pub(crate) maximum_rank: usize,
}
impl ForegroundDiskPopulation {
    pub(crate) fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            attempts: self.attempts.checked_add(other.attempts)?,
            outputs: self.outputs.checked_add(other.outputs)?,
            sources: self.sources.checked_add(other.sources)?,
            materialized: self.materialized.checked_add(other.materialized)?,
            output_logical_bytes: self
                .output_logical_bytes
                .checked_add(other.output_logical_bytes)?,
            source_backing_bytes: self
                .source_backing_bytes
                .checked_add(other.source_backing_bytes)?,
            maximum_rank: self.maximum_rank.max(other.maximum_rank),
        })
    }
    pub(crate) fn checked_mul(self, count: usize) -> Option<Self> {
        Some(Self {
            attempts: self.attempts.checked_mul(count)?,
            outputs: self.outputs.checked_mul(count)?,
            sources: self.sources.checked_mul(count)?,
            materialized: self.materialized.checked_mul(count)?,
            output_logical_bytes: self.output_logical_bytes.checked_mul(count)?,
            source_backing_bytes: self.source_backing_bytes.checked_mul(count)?,
            maximum_rank: if count == 0 { 0 } else { self.maximum_rank },
        })
    }
}
#[derive(Clone, Copy)]
struct UnitRead {
    ordinal: usize,
    layout: ForegroundDiskReadLayout,
    attempts: usize,
}
/// Cold planning inputs. Finite slot metadata is prepared after the request
/// accepts its controls; final buffers are deferred until an attempt is read.
pub(crate) struct ForegroundDiskWindowPlan {
    units: Vec<UnitRead>,
    source: ForegroundDiskDescriptors,
    background_repeated: bool,
}
struct UnitSlots {
    ordinal: usize,
    reads: PreparedOperationBank<PreparedForegroundDiskRead>,
}
/// Each unit has its own cursor; skipping a warm unit cannot spend another
/// unit's attempt. The outer custody follows every row/slot and its Vec backing.
pub(crate) struct PreparedForegroundDiskSlots {
    units: Vec<UnitSlots>,
    source: Option<ForegroundDiskDescriptors>,
    _custody: Option<OriginalHostSourceCustody>,
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("foreground disk slot memory: {0}")]
    Memory(#[from] WorkingMemoryError),
    #[error("foreground disk slot reservation: {0}")]
    Reserve(#[from] TryReserveError),
    #[error("foreground disk slot construction: {0}")]
    Read(#[from] ForegroundDiskReadError),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(crate) struct ForegroundDiskSlotError {
    #[source]
    cause: Cause,
    _custody: OriginalHostSourceCustody,
}
fn memory(cause: WorkingMemoryError) -> Error {
    Error::PrefillControl(cause)
}
fn identity() -> Error {
    memory(WorkingMemoryError::IdentityMismatch)
}
fn overflow() -> Error {
    memory(WorkingMemoryError::Overflow)
}
fn same_layout(a: ForegroundDiskReadLayout, b: ForegroundDiskReadLayout) -> bool {
    a.host_bytes == b.host_bytes
        && a.source_control_bytes == b.source_control_bytes
        && a.source_backing_bytes == b.source_backing_bytes
        && a.output_logical_bytes == b.output_logical_bytes
        && a.outputs == b.outputs
        && a.sources == b.sources
        && a.materialized == b.materialized
        && a.maximum_rank == b.maximum_rank
}

impl ForegroundDiskWindowPlan {
    /// Exact descriptor/vector constructor used by `new_with_metadata` before
    /// the selected read destinations are prepared.
    pub(crate) fn construction_control_bytes(units: usize) -> Option<usize> {
        let bytes = [
            Layout::array::<UnitRead>(units).ok()?.size(),
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Vec<UnitRead>>(),
            size_of::<UnitRead>(),
            size_of::<ForegroundDiskReadLayout>(),
            size_of::<Result<ForegroundDiskReadLayout, WorkingMemoryError>>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<Box<std::collections::TryReserveError>>(),
            size_of::<WindowPopulation>(),
            size_of::<Result<(), ResidencyError>>(),
        ];
        bytes
            .into_iter()
            .try_fold(size_of_val(&bytes), usize::checked_add)
    }
    pub(crate) fn new(
        manager: &ResidencyManager,
        pool: &MemoryLedger,
        window: WindowPopulation,
        ids: &[OffloadUnitId],
        scratch: &mut [ResidencyClosureSlot],
    ) -> Result<Self, Error> {
        Self::new_with_metadata(manager, pool, window, ids, scratch, None)
    }
    pub(crate) fn new_with_metadata(
        manager: &ResidencyManager,
        pool: &MemoryLedger,
        window: WindowPopulation,
        ids: &[OffloadUnitId],
        scratch: &mut [ResidencyClosureSlot],
        funding: Option<&eredu_nn::workspace::HostMetadataFunding>,
    ) -> Result<Self, Error> {
        if let Some(funding) = funding {
            funding
                .reserve_metadata(
                    Self::construction_control_bytes(window.units).ok_or_else(overflow)?,
                )
                .map_err(Error::WorkspacePlanning)?;
        }
        let source = manager
            .original_foreground_disk_descriptors()
            .ok_or_else(identity)?;
        let roots = ids
            .get(window.request_start..window.request_end)
            .ok_or_else(identity)?;
        if roots.len() != window.requested {
            return Err(identity());
        }
        let mut units = Vec::new();
        units
            .try_reserve_exact(window.units)
            .map_err(|cause| Error::Other(Box::new(cause)))?;
        {
            // Only immutable declaration ordinals cross this lock. Allocator
            // queries, source validation and native calls follow its release.
            let state = manager
                .inner
                .state
                .try_lock()
                .map_err(|error| match error {
                    TryLockError::WouldBlock => ResidencyError::OriginalManagerBusy,
                    TryLockError::Poisoned(_) => ResidencyError::StatePoisoned,
                })?;
            let closure = state
                .control
                .operation_closure(roots, scratch)
                .map_err(ResidencyError::OperationClosure)?;
            if closure.len() != window.units {
                return Err(identity());
            }
            for unit in closure.units() {
                if manager.inner.sources.prepared_host(unit.id()).is_some() {
                    continue;
                }
                let ordinal = source
                    .units()
                    .position(|retained| retained == unit)
                    .ok_or_else(identity)?;
                units.push(UnitRead {
                    ordinal,
                    attempts: FOREGROUND_DISK_ATTEMPTS_PER_UNIT,
                    layout: ForegroundDiskReadLayout {
                        host_bytes: 0,
                        source_control_bytes: 0,
                        source_backing_bytes: 0,
                        output_logical_bytes: 0,
                        outputs: 0,
                        sources: 0,
                        materialized: Default::default(),
                        maximum_rank: 0,
                    },
                });
            }
        }
        for row in &mut units {
            let unit = source.units().nth(row.ordinal).ok_or_else(identity)?;
            row.layout = source.read_plan(unit.id(), pool).map_err(memory)?.layout();
        }
        Ok(Self {
            units,
            source: source.clone(),
            background_repeated: false,
        })
    }
    /// Union of the actual selected forward's canonical source rows. Each
    /// background unit is read at most once: source lifecycle has no retry/reset.
    /// The same per-unit read constructor and source accountant remain in use.
    pub(crate) fn background_union(
        windows: &[Self],
        funding: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<Self, Error> {
        Self::background_population(windows, funding, false)
    }
    /// One read destination for each actual selected Host-window occurrence of
    /// a canonical unit. Units retain separate cursors; a warm omission cannot
    /// spend another unit's read. The source counter still follows every actual
    /// backing through publication, eviction and escaped transfer/failure.
    pub(crate) fn background_occurrences(
        windows: &[Self],
        funding: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<Self, Error> {
        Self::background_population(windows, funding, true)
    }
    fn background_population(
        windows: &[Self],
        funding: &eredu_nn::workspace::HostMetadataFunding,
        repeated: bool,
    ) -> Result<Self, Error> {
        let source = &windows.first().ok_or_else(identity)?.source;
        let mut count = 0usize;
        for (index, window) in windows.iter().enumerate() {
            if !source.same_source(&window.source) {
                return Err(identity());
            }
            for row in &window.units {
                let previous = windows[..index]
                    .iter()
                    .flat_map(|part| &part.units)
                    .find(|old| old.ordinal == row.ordinal);
                if let Some(old) = previous {
                    if !same_layout(old.layout, row.layout) {
                        return Err(identity());
                    }
                } else {
                    count = count.checked_add(1).ok_or_else(overflow)?;
                }
            }
        }
        let controls = [
            Layout::array::<UnitRead>(count)
                .map_err(|_| overflow())?
                .size(),
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Vec<UnitRead>>(),
            size_of::<UnitRead>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<std::slice::Iter<'_, Self>>(),
            size_of::<std::slice::Iter<'_, UnitRead>>(),
            size_of::<(&[Self], &eredu_nn::workspace::HostMetadataFunding, bool)>(),
            size_of::<bool>(),
            size_of::<std::slice::IterMut<'_, UnitRead>>(),
            size_of::<Option<usize>>(),
            size_of::<Result<usize, Error>>(),
            size_of::<(usize, &Self, &UnitRead)>(),
        ];
        funding
            .reserve_metadata(
                controls
                    .into_iter()
                    .try_fold(size_of_val(&controls), usize::checked_add)
                    .ok_or_else(overflow)?,
            )
            .map_err(Error::WorkspacePlanning)?;
        let mut units = Vec::new();
        units
            .try_reserve_exact(count)
            .map_err(|_| Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
        for window in windows {
            for row in &window.units {
                if !units
                    .iter()
                    .any(|old: &UnitRead| old.ordinal == row.ordinal)
                {
                    units.push(*row);
                }
            }
        }
        units.sort_unstable_by_key(|row| row.ordinal);
        for row in &mut units {
            row.attempts = if repeated {
                windows
                    .iter()
                    .try_fold(0usize, |count, window| {
                        count.checked_add(usize::from(
                            window
                                .units
                                .iter()
                                .any(|candidate| candidate.ordinal == row.ordinal),
                        ))
                    })
                    .ok_or_else(overflow)?
            } else {
                1
            };
        }
        Ok(Self {
            units,
            source: source.clone(),
            background_repeated: repeated,
        })
    }
    /// Exact Device source rows not already supplied by the selected Host
    /// promotion. Keeps the existing direct reader's actual retry population.
    pub(crate) fn difference(
        &self,
        covered: &Self,
        funding: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<Self, Error> {
        if !self.source.same_source(&covered.source) {
            return Err(identity());
        }
        let selected = |row: &&UnitRead| {
            !covered
                .units
                .iter()
                .any(|other| other.ordinal == row.ordinal)
        };
        let count = self.units.iter().filter(selected).count();
        let controls = [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<(&Self, &Self, &eredu_nn::workspace::HostMetadataFunding)>(),
            size_of::<UnitRead>(),
            size_of::<std::slice::Iter<'_, UnitRead>>(),
            size_of_val(&selected),
        ];
        funding
            .reserve_metadata(
                controls
                    .into_iter()
                    .try_fold(size_of_val(&controls), usize::checked_add)
                    .ok_or_else(overflow)?,
            )
            .map_err(Error::WorkspacePlanning)?;
        let mut units = funding.metadata_vec::<UnitRead>(count)?;
        for row in self.units.iter().filter(selected) {
            units.push(*row);
        }
        Ok(Self {
            units,
            source: self.source.clone(),
            background_repeated: false,
        })
    }
    pub(in crate::backend::runtime::residency::manager) fn permits_window_rearm(&self) -> bool {
        self.background_repeated
    }
    /// Physical source population for one read of each canonical row, regardless
    /// of the finite cumulative constructor slots retained by this plan.
    pub(crate) fn single_read_population(&self) -> Option<ForegroundDiskPopulation> {
        self.units
            .iter()
            .try_fold(ForegroundDiskPopulation::default(), |sum, row| {
                sum.checked_add(ForegroundDiskPopulation {
                    attempts: 1,
                    outputs: row.layout.outputs,
                    sources: row.layout.sources,
                    materialized: row.layout.materialized,
                    output_logical_bytes: row.layout.output_logical_bytes,
                    source_backing_bytes: row.layout.source_backing_bytes,
                    maximum_rank: row.layout.maximum_rank,
                })
            })
    }
    /// Same immutable canonical read rows used by this window's finite banks.
    /// Warm source-owned host rows were excluded by the shared planner.
    pub(in crate::backend::runtime::residency::manager) fn read_layout(
        &self,
        id: &OffloadUnitId,
    ) -> Option<ForegroundDiskReadLayout> {
        self.units
            .iter()
            .find(|row| {
                self.source
                    .units()
                    .nth(row.ordinal)
                    .is_some_and(|unit| unit.id() == id)
            })
            .map(|row| row.layout)
    }
    pub(in crate::backend::runtime::residency::manager) fn source(
        &self,
    ) -> &ForegroundDiskDescriptors {
        &self.source
    }
    pub(crate) fn read_units(&self) -> impl ExactSizeIterator<Item = &OffloadUnitId> {
        self.units.iter().map(|row| {
            self.source
                .units()
                .nth(row.ordinal)
                .expect("validated source ordinal")
                .id()
        })
    }
    pub(crate) fn matches_manager(&self, manager: &ResidencyManager) -> bool {
        manager
            .original_foreground_disk_descriptors()
            .is_some_and(|source| source.same_source(&self.source))
    }
    pub(crate) fn population(&self) -> Option<ForegroundDiskPopulation> {
        self.units
            .iter()
            .try_fold(ForegroundDiskPopulation::default(), |sum, row| {
                sum.checked_add(
                    ForegroundDiskPopulation {
                        attempts: 1,
                        outputs: row.layout.outputs,
                        sources: row.layout.sources,
                        materialized: row.layout.materialized,
                        output_logical_bytes: row.layout.output_logical_bytes,
                        source_backing_bytes: row.layout.source_backing_bytes,
                        maximum_rank: row.layout.maximum_rank,
                    }
                    .checked_mul(row.attempts)?,
                )
            })
    }
    pub(crate) fn source_control_bytes(&self) -> Option<u64> {
        self.units.iter().try_fold(0u64, |sum, row| {
            sum.checked_add(
                u64::try_from(row.layout.source_control_bytes)
                    .ok()?
                    .checked_mul(u64::try_from(row.attempts).ok()?)?,
            )
        })
    }
    pub(crate) fn peak_selection(
        &self,
        bytes: u64,
    ) -> eredu_runtime::working_memory::HostSourcePeakSelection {
        self.source.peak_selection(bytes)
    }
    pub(crate) fn retained_control_bytes(&self) -> Option<u64> {
        let fixed = [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<UnitRead>(),
            size_of::<ForegroundDiskReadLayout>(),
            size_of::<ForegroundDiskDescriptors>(),
            size_of::<&MemoryLedger>(),
            size_of::<std::sync::MutexGuard<'static, super::super::ManagerState>>(),
            size_of::<
                Result<
                    std::sync::MutexGuard<'static, super::super::ManagerState>,
                    TryLockError<std::sync::MutexGuard<'static, super::super::ManagerState>>,
                >,
            >(),
            size_of::<eredu_runtime::residency::ResidencyClosure<'_>>(),
            size_of::<
                Result<
                    eredu_runtime::residency::ResidencyClosure<'_>,
                    eredu_runtime::residency::ResidencyClosureError,
                >,
            >(),
            size_of::<std::slice::IterMut<'_, UnitRead>>(),
            size_of::<Option<usize>>(),
            size_of::<TryReserveError>(),
            size_of::<Box<TryReserveError>>(),
        ];
        let bytes = fixed.into_iter().try_fold(
            Layout::array::<UnitRead>(self.units.capacity())
                .ok()?
                .size()
                .checked_add(size_of_val(&fixed))?,
            usize::checked_add,
        )?;
        u64::try_from(bytes).ok()
    }
    /// Includes all final bank rows and every deferred read-constructor control.
    /// Cumulative source backing is descriptive in population(); the request
    /// admits one bounded live capacity across these finite attempts.
    fn attempt_fixed_control_bytes(units: usize) -> Option<u64> {
        let fixed = [
            size_of::<PreparedForegroundDiskSlots>(),
            size_of::<Result<PreparedForegroundDiskSlots, ForegroundDiskSlotError>>(),
            size_of::<Option<PreparedForegroundDiskRead>>(),
            size_of::<Result<Option<PreparedForegroundDiskRead>, ResidencyError>>(),
            size_of::<UnitRead>(),
            size_of::<UnitSlots>(),
            size_of::<Option<UnitSlots>>(),
            size_of::<&MemoryLedger>(),
            size_of::<&WorkingMemoryReservation>(),
            size_of::<Option<&WorkingMemoryReservation>>(),
            size_of::<&ForegroundDiskSourceCapacity>(),
            size_of::<Option<&ForegroundDiskSourceCapacity>>(),
            size_of::<Option<OriginalHostSourceCustody>>(),
            size_of::<Cause>(),
            size_of::<Result<(), Cause>>(),
            size_of::<ForegroundDiskSlotError>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<Result<super::super::ForegroundDiskReadPlan<'_>, WorkingMemoryError>>(),
        ];
        let bytes = fixed.into_iter().try_fold(
            Layout::array::<UnitSlots>(units)
                .ok()?
                .size()
                .checked_add(size_of_val(&fixed))?,
            usize::checked_add,
        )?;
        let error =
            eredu_core::BackendFailure::source_retention_peak_bytes::<ForegroundDiskSlotError>()?;
        u64::try_from(bytes.checked_add(error)?).ok()
    }
    pub(crate) fn attempt_control_bytes(&self) -> Option<u64> {
        self.units.iter().try_fold(
            Self::attempt_fixed_control_bytes(self.units.len())?,
            |sum, row| sum.checked_add(read_bank_layout(*row, row.attempts)?),
        )
    }
    pub(crate) fn prepare(
        &self,
        pool: &MemoryLedger,
        custody: eredu_runtime::working_memory::OriginalTextControlGuard,
        reservation: &WorkingMemoryReservation,
        capacity: &ForegroundDiskSourceCapacity,
    ) -> Result<PreparedForegroundDiskSlots, ForegroundDiskSlotError> {
        self.prepare_source(pool, custody.into(), Some(reservation), capacity)
    }
    pub(crate) fn prepare_source(
        &self,
        pool: &MemoryLedger,
        custody: OriginalHostSourceCustody,
        reservation: Option<&WorkingMemoryReservation>,
        capacity: &ForegroundDiskSourceCapacity,
    ) -> Result<PreparedForegroundDiskSlots, ForegroundDiskSlotError> {
        // The enclosing request compares the complete cold amount before this
        // constructor. This validates identity; it grants no additional bytes.
        if !custody
            .metadata_custody()
            .matches_accounting_owner(pool.shared_storage_accounting_id())
        {
            return Err(ForegroundDiskSlotError {
                cause: Cause::Memory(WorkingMemoryError::IdentityMismatch),
                _custody: custody,
            });
        }
        custody
            .validate_account(reservation)
            .map_err(|cause| ForegroundDiskSlotError {
                cause: Cause::Memory(cause),
                _custody: custody.clone(),
            })?;
        if !custody.same_source(capacity.custody()) {
            return Err(ForegroundDiskSlotError {
                cause: Cause::Memory(WorkingMemoryError::IdentityMismatch),
                _custody: custody,
            });
        }
        capacity
            .validate_account(reservation)
            .map_err(|cause| ForegroundDiskSlotError {
                cause: Cause::Memory(cause),
                _custody: custody.clone(),
            })?;
        let mut out = PreparedForegroundDiskSlots {
            units: Vec::new(),
            source: Some(self.source.clone()),
            _custody: Some(custody),
        };
        let result = (|| -> Result<(), Cause> {
            out.units.try_reserve_exact(self.units.len())?;
            for row in &self.units {
                let bank = PreparedOperationBank::try_new(
                    row.attempts,
                    read_factory(
                        Some(&self.source),
                        Some(pool),
                        *row,
                        out._custody.as_ref(),
                        reservation,
                        Some(capacity),
                    ),
                );
                let reads = match bank {
                    Ok(reads) => reads,
                    Err(error) => {
                        let (cause, prefix, factory) = error.into_parts();
                        let cause = match cause {
                            BankPreparationCause::Overflow => {
                                Cause::Memory(WorkingMemoryError::Overflow)
                            }
                            BankPreparationCause::Reserve(cause) => Cause::Reserve(cause),
                            BankPreparationCause::Slot { cause, .. } => cause,
                        };
                        // out retains custody through destruction of every
                        // successful prefix and the unused factory transports.
                        drop(prefix);
                        drop(factory);
                        return Err(cause);
                    }
                };
                out.units.push(UnitSlots {
                    ordinal: row.ordinal,
                    reads,
                });
            }
            Ok(())
        })();
        match result {
            Ok(()) => Ok(out),
            Err(cause) => Err(ForegroundDiskSlotError {
                cause,
                _custody: out._custody.as_ref().expect("accepted custody").clone(),
            }),
        }
    }
}
fn read_factory<'a>(
    source: Option<&'a ForegroundDiskDescriptors>,
    pool: Option<&'a MemoryLedger>,
    row: UnitRead,
    custody: Option<&'a OriginalHostSourceCustody>,
    reservation: Option<&'a WorkingMemoryReservation>,
    capacity: Option<&'a ForegroundDiskSourceCapacity>,
) -> impl FnMut(usize) -> Result<PreparedForegroundDiskRead, Cause> + 'a {
    move |_| {
        let source = source.expect("actual retained disk source");
        let unit = source
            .units()
            .nth(row.ordinal)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let plan = source.read_plan(unit.id(), pool.expect("actual source pool"))?;
        if !same_layout(plan.layout(), row.layout) {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        Ok(plan.prepare_source(
            custody.expect("accepted custody").clone(),
            reservation,
            capacity.expect("accepted source capacity"),
        )?)
    }
}
fn read_bank_layout(row: UnitRead, count: usize) -> Option<u64> {
    fn measured<F: FnMut(usize) -> Result<PreparedForegroundDiskRead, Cause>>(
        row: UnitRead,
        count: usize,
        _: &F,
    ) -> Option<u64> {
        Some(
            PreparedOperationBank::<PreparedForegroundDiskRead>::layout::<F, Cause>(
                count,
                u64::try_from(row.layout.host_bytes).ok()?,
            )?
            .total_control_bytes,
        )
    }
    measured(row, count, &read_factory(None, None, row, None, None, None))
}
impl PreparedForegroundDiskSlots {
    pub(crate) fn unavailable() -> Self {
        Self {
            units: Vec::new(),
            source: None,
            _custody: None,
        }
    }
    /// None identifies a route which has no foreground reader. In a selected
    /// reader, a foreign unit or exhausted cursor is an error, never fallback.
    pub(crate) fn checkout(
        &mut self,
        id: &OffloadUnitId,
    ) -> Result<Option<PreparedForegroundDiskRead>, ResidencyError> {
        let Some(source) = &self.source else {
            return Ok(None);
        };
        let row = self
            .units
            .iter_mut()
            .find(|row| {
                source
                    .units()
                    .nth(row.ordinal)
                    .is_some_and(|unit| unit.id() == id)
            })
            .ok_or(ResidencyError::OriginalOperationDomain)?;
        row.reads
            .checkout()
            .map(Some)
            .map_err(|cause| ResidencyError::OriginalOperationCapacity {
                family: "foreground disk unit read",
                prepared: cause.prepared,
            })
    }
}

/// Finite read envelope from the actual eligible canonical source rows. It is
/// descriptive until the matching source bank is accepted; it does not select
/// actual IDs, construct their slots, or authorize an ordinary fallback.
pub(crate) struct ForegroundDiskSubsetCeiling {
    source: ForegroundDiskDescriptors,
    population: ForegroundDiskPopulation,
    source_controls: u64,
    attempt_controls: u64,
}
impl ForegroundDiskWindowPlan {
    /// Any selected closure has at most this many actual read units. Each
    /// component is bounded by both the full source sum and selected-count
    /// times the largest actual row. No demand subsets are enumerated.
    pub(crate) fn subset_ceiling(
        &self,
        maximum_units: usize,
    ) -> Option<ForegroundDiskSubsetCeiling> {
        let count = maximum_units.min(self.units.len());
        let full = self.population()?;
        let mut max = ForegroundDiskPopulation::default();
        let mut maximum_source = 0u64;
        let mut maximum_bank = 0u64;
        let mut full_bank = 0u64;
        for row in &self.units {
            let one = ForegroundDiskPopulation {
                attempts: 1,
                outputs: row.layout.outputs,
                sources: row.layout.sources,
                materialized: row.layout.materialized,
                output_logical_bytes: row.layout.output_logical_bytes,
                source_backing_bytes: row.layout.source_backing_bytes,
                maximum_rank: row.layout.maximum_rank,
            }
            .checked_mul(row.attempts)?;
            max.attempts = max.attempts.max(one.attempts);
            max.outputs = max.outputs.max(one.outputs);
            max.sources = max.sources.max(one.sources);
            max.materialized = max.materialized.maximum(one.materialized);
            max.output_logical_bytes = max.output_logical_bytes.max(one.output_logical_bytes);
            max.source_backing_bytes = max.source_backing_bytes.max(one.source_backing_bytes);
            max.maximum_rank = max.maximum_rank.max(one.maximum_rank);
            maximum_source = maximum_source.max(
                u64::try_from(row.layout.source_control_bytes)
                    .ok()?
                    .checked_mul(u64::try_from(row.attempts).ok()?)?,
            );
            let bank = read_bank_layout(*row, row.attempts)?;
            maximum_bank = maximum_bank.max(bank);
            full_bank = full_bank.checked_add(bank)?;
        }
        let upper = max.checked_mul(count)?;
        let population = ForegroundDiskPopulation {
            attempts: full.attempts.min(upper.attempts),
            outputs: full.outputs.min(upper.outputs),
            sources: full.sources.min(upper.sources),
            materialized: full.materialized.minimum(upper.materialized),
            output_logical_bytes: full.output_logical_bytes.min(upper.output_logical_bytes),
            source_backing_bytes: full.source_backing_bytes.min(upper.source_backing_bytes),
            maximum_rank: upper.maximum_rank,
        };
        let count = u64::try_from(count).ok()?;
        Some(ForegroundDiskSubsetCeiling {
            source: self.source.clone(),
            population,
            source_controls: self
                .source_control_bytes()?
                .min(maximum_source.checked_mul(count)?),
            attempt_controls: Self::attempt_fixed_control_bytes(usize::try_from(count).ok()?)?
                .checked_add(full_bank.min(maximum_bank.checked_mul(count)?))?,
        })
    }
}
impl ForegroundDiskSubsetCeiling {
    pub(crate) fn clone_control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<ForegroundDiskDescriptors>(),
            size_of::<(&Self, &eredu_nn::workspace::HostMetadataFunding)>(),
            size_of::<Result<Self, eredu_nn::workspace::HostMetadataFundingError>>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    /// Paid descriptive alias only: the actual read counters, bank and native
    /// capacity remain outside this immutable source envelope.
    pub(crate) fn clone_for_invocation(
        &self,
        funding: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<Self, eredu_nn::workspace::HostMetadataFundingError> {
        funding.reserve_metadata(
            Self::clone_control_bytes()
                .ok_or(eredu_nn::workspace::HostMetadataFundingError::Overflow)?,
        )?;
        Ok(Self {
            source: self.source.clone(),
            population: self.population,
            source_controls: self.source_controls,
            attempt_controls: self.attempt_controls,
        })
    }
    /// Closed stack/descriptor transports for the one source query and the
    /// accepted native capacity factory. Per-window/read destinations remain
    /// in attempt_control_bytes and the exact read-source layout.
    pub(crate) fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<(&ForegroundDiskWindowPlan, usize)>(),
            size_of::<[ForegroundDiskPopulation; 4]>(),
            size_of::<[u64; 5]>(),
            size_of::<std::slice::Iter<'_, UnitRead>>(),
            size_of::<UnitRead>(),
            size_of::<eredu_runtime::working_memory::HostSourceConstructionFacts>(),
            size_of::<Option<eredu_runtime::working_memory::HostSourceConstructionFacts>>(),
            size_of::<eredu_runtime::working_memory::OriginalHostSourceBank>(),
            size_of::<OriginalHostSourceCustody>(),
            size_of::<Option<&WorkingMemoryReservation>>(),
            size_of::<Result<ForegroundDiskSourceCapacity, WorkingMemoryError>>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            ForegroundDiskSourceCapacity::control_bytes()?,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    pub(crate) fn population(&self) -> ForegroundDiskPopulation {
        self.population
    }
    pub(crate) fn attempt_control_bytes(&self) -> u64 {
        self.attempt_controls
    }
    pub(crate) fn matches_manager(&self, manager: &ResidencyManager) -> bool {
        manager
            .original_foreground_disk_descriptors()
            .is_some_and(|source| self.source.same_source(source))
    }
    /// The same native source counter is retained across these calls; source
    /// bytes are cumulative, while backing may retire between completed chunks.
    pub(crate) fn source_facts(
        &self,
        calls: usize,
    ) -> Option<eredu_runtime::working_memory::HostSourceConstructionFacts> {
        let population = self.population.checked_mul(calls)?;
        let controls = self
            .source_controls
            .checked_mul(u64::try_from(calls).ok()?)?;
        let backing = if calls == 0 {
            0
        } else {
            self.population.source_backing_bytes
        };
        eredu_runtime::working_memory::HostSourceConstructionFacts::new(
            controls,
            population.sources,
            population.attempts,
        )
        .ok()?
        .with_peak_backing(self.source.peak_selection(u64::try_from(backing).ok()?))
        .ok()
    }
    /// Consumes a bank already admitted for this exact descriptor source and
    /// finite population. The returned owner is the existing shared read worker.
    pub(crate) fn prepare_capacity(
        &self,
        calls: usize,
        bank: eredu_runtime::working_memory::OriginalHostSourceBank,
        custody: OriginalHostSourceCustody,
        reservation: Option<&WorkingMemoryReservation>,
    ) -> Result<ForegroundDiskSourceCapacity, WorkingMemoryError> {
        let facts = self
            .source_facts(calls)
            .ok_or(WorkingMemoryError::Overflow)?;
        if !bank.belongs_to_source(&custody) || !bank.matches_facts(facts) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let backing = if calls == 0 {
            0
        } else {
            self.population.source_backing_bytes
        };
        ForegroundDiskSourceCapacity::with_source_account(
            self.source
                .peak_selection(u64::try_from(backing).map_err(|_| WorkingMemoryError::Overflow)?),
            bank,
            custody,
            reservation,
        )
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;

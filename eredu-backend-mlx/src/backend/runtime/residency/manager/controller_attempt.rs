//! Final neutral admission destinations belonging to one exact window attempt.
use super::*;
use eredu_core::residency::{
    PreparedResidencyAdmissionFailure, ResidencyAdmissionStorage, ResidencyReservationRow,
};
use eredu_runtime::working_memory::OriginalOperationMetadataCustody;
use std::{alloc::Layout, collections::TryReserveError, sync::Weak};

#[derive(Debug)]
pub(crate) struct PreparedControllerAttempt {
    pub(super) missing: Vec<bool>,
    pub(super) selected: Vec<bool>,
    pub(super) reservations: Vec<ResidencyReservationRow>,
    pub(super) admission: Option<ResidencyAdmissionStorage>,
    manager: ManagerWeak,
    // All final allocations and source/manager aliases precede their custody.
    _custody: ResidencyControlCustody,
}
#[derive(Debug, thiserror::Error)]
pub(crate) enum ControllerPreparationCause {
    #[error("actual controller preparation source mismatch")]
    Source,
    #[error("actual controller preparation lock unavailable")]
    Busy,
    #[error("actual controller preparation state is poisoned")]
    Poisoned,
    #[error("controller destination allocation failed: {0}")]
    Reserve(#[source] TryReserveError),
}
#[derive(Debug)]
pub(crate) struct ControllerPreparationError {
    pub(crate) cause: ControllerPreparationCause,
    pub(crate) prefix: Option<PreparedControllerAttempt>,
}
/// This is an inline public backend error payload, not a native observer owner.
/// Its neutral error/source and all scratch allocations drop before custody.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct PreparedAdmissionFailure {
    #[source]
    cause: PreparedResidencyAdmissionFailure,
    _owner: PreparedControllerAttempt,
}
impl PreparedAdmissionFailure {
    pub(crate) fn capacity(&self) -> Option<eredu_core::residency::ResidencyCapacityRef<'_>> {
        self.cause.capacity()
    }
}
impl PreparedControllerAttempt {
    pub(super) fn take(
        slot: &mut Option<Self>,
        manager: &ManagerOwner,
        ledger: &eredu_core::residency::ResidencyLedger,
        requested: usize,
    ) -> Result<Self, ResidencyError> {
        let value = slot
            .as_ref()
            .ok_or(ResidencyError::OriginalOperationCapacity {
                family: "controller acquisition",
                prepared: 1,
            })?;
        if !std::ptr::eq(value.manager.as_ptr(), manager.as_ptr())
            || value.selected.len() != requested
            || !value
                .admission
                .as_ref()
                .is_some_and(|storage| ledger.matches_plan_source(storage.source()))
        {
            return Err(ResidencyError::OriginalOperationDomain);
        }
        Ok(slot
            .take()
            .expect("validated one-use controller destination"))
    }
    pub(super) fn validate(
        mut self,
        ledger: &eredu_core::residency::ResidencyLedger,
        ids: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<Self, ResidencyError> {
        let storage = self
            .admission
            .take()
            .expect("unconsumed admission destination");
        match storage.validate_batch(ledger, ids, tier) {
            Ok(storage) => {
                self.admission = Some(storage);
                Ok(self)
            }
            Err(cause) => Err(self.fail(cause)),
        }
    }
    pub(super) fn fail(self, cause: PreparedResidencyAdmissionFailure) -> ResidencyError {
        ResidencyError::OriginalAdmission(PreparedAdmissionFailure {
            cause,
            _owner: self,
        })
    }
    pub(crate) fn control_bytes(
        units: usize,
        requested: usize,
        closure: usize,
        maximum_id: usize,
    ) -> Option<u64> {
        let payload = [
            Layout::array::<bool>(closure).ok()?.size(),
            Layout::array::<bool>(requested).ok()?.size(),
            Layout::array::<ResidencyReservationRow>(closure)
                .ok()?
                .size(),
            ResidencyAdmissionStorage::requested_bytes(units, requested.max(closure), maximum_id)?,
        ];
        let fixed = [
            size_of::<OriginalOperationMetadataCustody>(),
            size_of::<ResidencyControlCustody>(),
            size_of::<eredu_runtime::working_memory::OriginalTextMetadataCustody>(),
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<PreparedAdmissionFailure>(),
            size_of::<Result<Self, ResidencyError>>(),
            size_of::<ControllerPreparationError>(),
            size_of::<Result<Self, ControllerPreparationError>>(),
            size_of::<eredu_runtime::residency::ResidencyAcquisitionRef<'static, 'static>>(),
            size_of::<AcquisitionResult>(),
            size_of::<WorkFailure>(),
            size_of::<Result<Option<SubmittedResidentTransfer>, WorkFailure>>(),
        ];
        u64::try_from(
            payload
                .into_iter()
                .chain(fixed)
                .try_fold(0usize, usize::checked_add)?,
        )
        .ok()
    }
}
impl ResidencyManager {
    /// Borrows the exact ledger only during cold destination construction.
    /// U/R and N are checked against the retained window/source population.
    pub(crate) fn prepare_controller_attempt(
        &self,
        source: WindowPopulation,
        custody: OriginalOperationMetadataCustody,
    ) -> Result<PreparedControllerAttempt, ControllerPreparationError> {
        self.prepare_controller_with_custody(source, custody.into())
    }

    pub(super) fn prepare_controller_with_custody(
        &self,
        source: WindowPopulation,
        custody: ResidencyControlCustody,
    ) -> Result<PreparedControllerAttempt, ControllerPreparationError> {
        let state = self
            .inner
            .state
            .try_lock()
            .map_err(|cause| ControllerPreparationError {
                cause: match cause {
                    std::sync::TryLockError::WouldBlock => ControllerPreparationCause::Busy,
                    std::sync::TryLockError::Poisoned(_) => ControllerPreparationCause::Poisoned,
                },
                prefix: None,
            })?;
        let plan = state.control.ledger().plan_source();
        if plan.len() != source.controller_units
            || plan.maximum_id_bytes() != source.maximum_id_bytes
            || source.units > plan.len()
        {
            return Err(ControllerPreparationError {
                cause: ControllerPreparationCause::Source,
                prefix: None,
            });
        }
        drop(state);
        let mut value = PreparedControllerAttempt {
            missing: Vec::new(),
            selected: Vec::new(),
            reservations: Vec::new(),
            admission: None,
            manager: self.inner.downgrade(),
            _custody: custody,
        };
        let result = (|| {
            value.missing.try_reserve_exact(source.units)?;
            value.missing.resize(source.units, false);
            value.selected.try_reserve_exact(source.requested)?;
            value.selected.resize(source.requested, false);
            value.reservations.try_reserve_exact(source.units)?;
            Ok::<_, TryReserveError>(())
        })();
        if let Err(cause) = result {
            return Err(ControllerPreparationError {
                cause: ControllerPreparationCause::Reserve(cause),
                prefix: Some(value),
            });
        }
        match ResidencyAdmissionStorage::try_new(
            plan,
            source.units.max(source.requested),
            source.maximum_id_bytes,
        ) {
            Ok(storage) => {
                value.admission = Some(storage);
                Ok(value)
            }
            Err(error) => {
                value.admission = Some(error.prefix);
                Err(ControllerPreparationError {
                    cause: ControllerPreparationCause::Reserve(error.cause),
                    prefix: Some(value),
                })
            }
        }
    }
}

pub(super) enum AcquisitionResult {
    Ordinary(Vec<bool>),
    Prepared(PreparedControllerAttempt),
}
impl AcquisitionResult {
    pub(super) fn select(self, roots: &[OffloadUnitId], closure: &[OffloadUnitId]) -> Self {
        match self {
            Self::Ordinary(flags) => Self::Ordinary(
                roots
                    .iter()
                    .map(|id| {
                        flags[closure
                            .binary_search(id)
                            .expect("validated root in closure")]
                    })
                    .collect(),
            ),
            Self::Prepared(mut owner) => {
                assert_eq!(owner.selected.len(), roots.len(), "actual root destination");
                for (id, flag) in roots.iter().zip(&mut owner.selected) {
                    *flag = owner.missing[closure
                        .binary_search(id)
                        .expect("validated root in closure")];
                }
                Self::Prepared(owner)
            }
        }
    }
    pub(super) fn into_ordinary(self) -> Vec<bool> {
        match self {
            Self::Ordinary(flags) => flags,
            Self::Prepared(_) => unreachable!("ordinary entry has no original storage"),
        }
    }
    #[cfg(test)]
    pub(super) fn flags(&self) -> &[bool] {
        match self {
            Self::Ordinary(flags) => flags,
            Self::Prepared(owner) => &owner.selected,
        }
    }
}
// Native/recipe failures retain their original typed owner. Only the neutral
// admission refusal takes final admission storage out of the same attempt.
pub(super) enum WorkFailure {
    Residency(ResidencyError),
    Admission(PreparedResidencyAdmissionFailure),
}
impl From<ResidencyError> for WorkFailure {
    fn from(value: ResidencyError) -> Self {
        Self::Residency(value)
    }
}
impl From<ResidencyLedgerError> for WorkFailure {
    fn from(value: ResidencyLedgerError) -> Self {
        Self::Residency(value.into())
    }
}

#[cfg(test)]
impl ResidencyManager {
    pub(super) fn prepare_controller_for_test(
        &self,
        roots: &[OffloadUnitId],
        scratch: &mut [eredu_runtime::residency::ResidencyClosureSlot],
        custody: eredu_runtime::working_memory::OriginalTextControlGuard,
    ) -> PreparedControllerAttempt {
        let source = {
            let state = self
                .inner
                .state
                .try_lock()
                .expect("cold fixture source loan");
            super::operation_source::WindowPopulation::collect(&state.control, roots, scratch)
                .expect("actual fixture window")
        };
        self.prepare_controller_attempt(source, custody.metadata_custody().into())
            .expect("actual controller destination")
    }
}

#[cfg(test)]
std::thread_local! { static LAST_USE:std::cell::Cell<Option<(usize,usize,usize,usize,usize,usize)>>=const {std::cell::Cell::new(None)}; }
#[cfg(test)]
pub(super) fn last_use() -> Option<(usize, usize, usize, usize, usize, usize)> {
    LAST_USE.with(|slot| slot.take())
}
#[cfg(test)]
pub(super) fn record_use(result: &AcquisitionResult) {
    if let AcquisitionResult::Prepared(owner) = result {
        let (a, b, c, d) = owner.source_addresses();
        LAST_USE.with(|slot| {
            slot.set(Some((
                a,
                b,
                c,
                d,
                owner.selected.len(),
                owner.selected.iter().filter(|flag| **flag).count(),
            )))
        });
    }
}
#[cfg(test)]
impl PreparedControllerAttempt {
    pub(super) fn source_addresses(&self) -> (usize, usize, usize, usize) {
        (
            self.missing.as_ptr() as usize,
            self.selected.as_ptr() as usize,
            self.reservations.as_ptr() as usize,
            self.admission
                .as_ref()
                .expect("actual admission source")
                .order()
                .as_ptr() as usize,
        )
    }
}

#[cfg(test)]
impl ResidencyManager {
    pub(super) fn exercise_controller_destination_checks(
        &self,
        foreign: &Self,
        controls: &eredu_runtime::working_memory::OriginalTextControlGuard,
    ) -> PreparedAdmissionFailure {
        let ids = [
            OffloadUnitId::new("first").unwrap(),
            OffloadUnitId::new("second").unwrap(),
        ];
        let mut scratch = [eredu_runtime::residency::ResidencyClosureSlot::default(); 2];
        let mut slot = Some(self.prepare_controller_for_test(&ids, &mut scratch, controls.clone()));
        let addresses = slot.as_ref().unwrap().source_addresses();
        {
            let state = foreign.inner.state.try_lock().unwrap();
            assert!(matches!(
                PreparedControllerAttempt::take(
                    &mut slot,
                    &foreign.inner,
                    state.control.ledger(),
                    2
                ),
                Err(ResidencyError::OriginalOperationDomain)
            ));
            assert_eq!(slot.as_ref().unwrap().source_addresses(), addresses);
        }
        let state = self.inner.state.try_lock().unwrap();
        let owner =
            PreparedControllerAttempt::take(&mut slot, &self.inner, state.control.ledger(), 2)
                .unwrap();
        assert!(slot.is_none());
        assert_eq!(owner.source_addresses(), addresses);
        let before = state.control.ledger().telemetry();
        let unknown = [OffloadUnitId::new("alien").unwrap()];
        let failure = owner
            .validate(state.control.ledger(), &unknown, MemoryTier::Device)
            .unwrap_err();
        assert_eq!(state.control.ledger().telemetry(), before);
        assert!(state.storage.values().all(|unit| unit.device.is_none()));
        drop(state);
        drop(unknown);
        let ResidencyError::OriginalAdmission(failure) = failure else {
            panic!("owning prepared validation failure");
        };
        assert_eq!(failure.to_string(), "unknown residency unit: alien");
        failure
    }
}

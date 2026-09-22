//! Source-bound synchronous IDs crossing the controller's mutable loan.
use super::{
    ManagerOwner, ManagerWeak, OffloadUnitId, ResidencyControlCustody, ResidencyError,
    ResidencyManager,
};
use eredu_runtime::{
    residency::{ResidencyClosure, ResidencyClosureSlot},
    working_memory::OriginalOperationMetadataCustody,
};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    sync::{Arc, TryLockError, Weak},
};

pub(crate) struct PreparedClosureIds {
    ids: Vec<OffloadUnitId>,
    manager: ManagerWeak,
    complete: bool,
    // Includes the Vec/String allocations and Weak header alias above.
    _custody: ResidencyControlCustody,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum ClosurePreparationCause {
    #[error("closure-ID destination layout overflow")]
    Overflow,
    #[error("closure-ID destination reserve failed: {0}")]
    Reserve(#[source] TryReserveError),
    #[error("closure-ID source refused: {0}")]
    Source(#[source] ResidencyError),
}

pub(crate) struct ClosurePreparationError {
    pub(crate) cause: ClosurePreparationCause,
    pub(crate) prefix: PreparedClosureIds,
}

impl ResidencyManager {
    pub(crate) fn prepare_closure_ids(
        &self,
        roots: &[OffloadUnitId],
        scratch: &mut [ResidencyClosureSlot],
        units: usize,
        id_bytes: usize,
        custody: OriginalOperationMetadataCustody,
    ) -> Result<PreparedClosureIds, ClosurePreparationError> {
        self.prepare_closure_ids_with_custody(roots, scratch, units, id_bytes, custody.into())
    }

    pub(super) fn prepare_closure_ids_with_custody(
        &self,
        roots: &[OffloadUnitId],
        scratch: &mut [ResidencyClosureSlot],
        units: usize,
        id_bytes: usize,
        custody: ResidencyControlCustody,
    ) -> Result<PreparedClosureIds, ClosurePreparationError> {
        let mut value = PreparedClosureIds {
            ids: Vec::new(),
            manager: self.inner.downgrade(),
            complete: false,
            _custody: custody,
        };
        let result = (|| {
            let state = self.inner.state.try_lock().map_err(|error| {
                ClosurePreparationCause::Source(match error {
                    TryLockError::WouldBlock => ResidencyError::OriginalManagerBusy,
                    TryLockError::Poisoned(_) => ResidencyError::StatePoisoned,
                })
            })?;
            let closure = state
                .control
                .operation_closure(roots, scratch)
                .map_err(|e| {
                    ClosurePreparationCause::Source(ResidencyError::OperationClosure(e))
                })?;
            let actual_bytes = closure
                .units()
                .try_fold(0usize, |n, unit| n.checked_add(unit.id().as_str().len()))
                .ok_or(ClosurePreparationCause::Overflow)?;
            if closure.is_empty() || closure.len() != units || actual_bytes != id_bytes {
                return Err(ClosurePreparationCause::Source(
                    ResidencyError::OriginalOperationDomain,
                ));
            }
            Layout::array::<OffloadUnitId>(units).map_err(|_| ClosurePreparationCause::Overflow)?;
            value
                .ids
                .try_reserve_exact(units)
                .map_err(ClosurePreparationCause::Reserve)?;
            for unit in closure.units() {
                // Actual immutable source IDs, in the same controller order.
                // String::clone retains its existing abort-on-OOM contract.
                value.ids.push(unit.id().clone());
            }
            value.complete = true;
            Ok(())
        })();
        match result {
            Ok(()) => Ok(value),
            Err(cause) => Err(ClosurePreparationError {
                cause,
                prefix: value,
            }),
        }
    }
}

impl PreparedClosureIds {
    /// Validate before the one-use take. The caller has the actual manager lock
    /// and closure; pointer equality compares a still-retained Weak header.
    pub(super) fn take(
        slot: &mut Option<Self>,
        manager: &ManagerOwner,
        closure: &ResidencyClosure<'_>,
    ) -> Result<ClosureIds, ResidencyError> {
        if closure.is_empty() && slot.is_none() {
            return Ok(ClosureIds::Empty);
        }
        let owner = slot
            .as_ref()
            .ok_or(ResidencyError::OriginalOperationCapacity {
                family: "closure IDs",
                prepared: 1,
            })?;
        if !owner.complete || !std::ptr::eq(owner.manager.as_ptr(), manager.as_ptr()) {
            return Err(ResidencyError::OriginalOperationDomain);
        }
        if owner.ids.len() != closure.len()
            || !owner.ids.iter().eq(closure.units().map(|unit| unit.id()))
        {
            return Err(ResidencyError::OperationClosure(
                eredu_runtime::residency::ResidencyClosureError::InvalidOwner,
            ));
        }
        Ok(ClosureIds::Prepared(
            slot.take().expect("validated closure IDs"),
        ))
    }

    pub(crate) fn control_bytes(units: usize, id_bytes: usize) -> Option<u64> {
        let heap = Layout::array::<OffloadUnitId>(units)
            .ok()?
            .size()
            .checked_add(id_bytes)?;
        let controls = [
            size_of::<OriginalOperationMetadataCustody>(),
            size_of::<ResidencyControlCustody>(),
            size_of::<eredu_runtime::working_memory::OriginalTextMetadataCustody>(),
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<ClosureIds>(),
            size_of::<ClosurePreparationCause>(),
            size_of::<ClosurePreparationError>(),
            size_of::<Result<Self, ClosurePreparationError>>(),
            size_of::<Result<ClosureIds, ResidencyError>>(),
            size_of::<&[OffloadUnitId]>(),
            size_of::<ResidencyClosure<'static>>(),
        ]
        .into_iter()
        .try_fold(heap, usize::checked_add)?;
        u64::try_from(controls).ok()
    }
}

/// Only the original arm carries a prepared owner. Ordinary execution retains
/// its existing collect, and all acquisition/rollback logic borrows one slice.
pub(super) enum ClosureIds {
    Ordinary(Vec<OffloadUnitId>),
    Prepared(PreparedClosureIds),
    Empty,
}
impl ClosureIds {
    pub(super) fn as_slice(&self) -> &[OffloadUnitId] {
        match self {
            Self::Ordinary(ids) => ids,
            Self::Prepared(owner) => &owner.ids,
            Self::Empty => &[],
        }
    }
}

#[cfg(test)]
thread_local! {
    static LAST_USE: std::cell::Cell<Option<(usize, usize, usize, usize)>> = const { std::cell::Cell::new(None) };
}
#[cfg(test)]
pub(super) fn addresses(ids: &[OffloadUnitId]) -> (usize, usize, usize, usize) {
    (
        ids.as_ptr() as usize,
        ids.len(),
        ids.first().map_or(0, |id| id.as_str().as_ptr() as usize),
        ids.last().map_or(0, |id| id.as_str().as_ptr() as usize),
    )
}
#[cfg(test)]
pub(super) fn record_use(ids: &[OffloadUnitId]) {
    LAST_USE.set(Some(addresses(ids)));
}
#[cfg(test)]
pub(super) fn last_use() -> Option<(usize, usize, usize, usize)> {
    LAST_USE.take()
}
#[cfg(test)]
impl PreparedClosureIds {
    pub(super) fn source_addresses(&self) -> (usize, usize, usize, usize) {
        addresses(&self.ids)
    }
}

#[cfg(test)]
impl ResidencyManager {
    pub(super) fn exercise_closure_destination_checks(
        &self,
        foreign: &ResidencyManager,
        controls: &eredu_runtime::working_memory::OriginalTextControlGuard,
    ) {
        let first = OffloadUnitId::new("first").unwrap();
        let second = OffloadUnitId::new("second").unwrap();
        let mut scratch = [ResidencyClosureSlot::default(); 2];
        let mut slot = Some(
            self.prepare_closure_ids(
                std::slice::from_ref(&first),
                &mut scratch,
                1,
                5,
                controls.metadata_custody().into(),
            )
            .unwrap_or_else(|_| panic!("prepare actual one-unit destination")),
        );
        let pointers = slot.as_ref().unwrap().source_addresses();
        {
            let state = self.inner.state.try_lock().unwrap();
            let closure = state
                .control
                .operation_closure(std::slice::from_ref(&first), &mut scratch)
                .unwrap();
            assert!(matches!(
                PreparedClosureIds::take(&mut slot, &foreign.inner, &closure),
                Err(ResidencyError::OriginalOperationDomain)
            ));
            assert_eq!(slot.as_ref().unwrap().source_addresses(), pointers);
            slot.as_mut().unwrap().complete = false;
            assert!(matches!(
                PreparedClosureIds::take(&mut slot, &self.inner, &closure),
                Err(ResidencyError::OriginalOperationDomain)
            ));
            slot.as_mut().unwrap().complete = true;
        }
        {
            let state = self.inner.state.try_lock().unwrap();
            let other = state
                .control
                .operation_closure(&[second], &mut scratch)
                .unwrap();
            assert!(matches!(
                PreparedClosureIds::take(&mut slot, &self.inner, &other),
                Err(ResidencyError::OperationClosure(_))
            ));
            assert_eq!(slot.as_ref().unwrap().source_addresses(), pointers);
        }
        {
            let state = self.inner.state.try_lock().unwrap();
            let closure = state
                .control
                .operation_closure(std::slice::from_ref(&first), &mut scratch)
                .unwrap();
            let taken = PreparedClosureIds::take(&mut slot, &self.inner, &closure)
                .unwrap_or_else(|e| panic!("take real closure: {e}"));
            assert_eq!(addresses(taken.as_slice()), pointers);
            assert!(matches!(
                PreparedClosureIds::take(&mut slot, &self.inner, &closure),
                Err(ResidencyError::OriginalOperationCapacity {
                    family: "closure IDs",
                    prepared: 1
                })
            ));
            drop(taken);
        }
        let state = self.inner.state.try_lock().unwrap();
        let empty = state.control.operation_closure(&[], &mut scratch).unwrap();
        assert!(PreparedClosureIds::take(&mut slot, &self.inner, &empty)
            .unwrap_or_else(|e| panic!("empty closure: {e}"))
            .as_slice()
            .is_empty());
        drop(state);
        let error = self
            .prepare_closure_ids(
                &[first],
                &mut scratch,
                1,
                99,
                controls.metadata_custody().into(),
            )
            .err()
            .expect("source byte mismatch");
        assert!(error.prefix.ids.is_empty());
        assert!(!error.prefix.complete);
        assert!(matches!(
            error.cause,
            ClosurePreparationCause::Source(ResidencyError::OriginalOperationDomain)
        ));
    }
}

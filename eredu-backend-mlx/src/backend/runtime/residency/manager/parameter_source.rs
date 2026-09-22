//! Immutable device-source loan; no residency acquisition, wait or reaping.
use super::*;
use std::{
    mem::{size_of, size_of_val},
    sync::{atomic::Ordering, TryLockError},
};

#[derive(Clone, Copy, Debug, thiserror::Error)]
pub(crate) enum ResidentParameterSourceError {
    #[error("resident parameter source manager is busy")]
    Busy,
    #[error("resident parameter source manager is poisoned")]
    Poisoned,
    #[error("resident parameter source is not settled")]
    Unsettled,
    #[error("resident parameter source declaration differs")]
    Mismatch,
    #[error("resident parameter device source is absent")]
    Missing,
}

/// Array table aliases retain their existing source/operation custody. This is
/// an immutable source snapshot, not a residency lease or execution permission.
pub(crate) struct ResidentParameterSource {
    arrays: ResidentArraysOwner,
}
impl ResidentParameterSource {
    pub(crate) fn value(&self, name: &str) -> Option<&Array> {
        self.arrays.arrays.get(name)
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<Result<Self, ResidentParameterSourceError>>(),
            size_of::<MutexGuard<'_, transfer::ManagerState>>(),
            size_of::<TryLockError<MutexGuard<'_, transfer::ManagerState>>>(),
            size_of::<(&ResidencyManager, &OffloadUnitId, &[WeightBinding])>(),
            size_of::<Option<&OffloadUnit>>(),
            size_of::<Option<&ResidentArraysOwner>>(),
            size_of::<std::slice::Iter<'_, WeightBinding>>(),
            size_of::<named_arrays::NamedIter<'_>>(),
            size_of::<Option<&Array>>(),
            size_of::<bool>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}
impl ResidencyManager {
    /// Clones only the actual source-owning table alias under a fixed try-lock.
    /// The caller prices this control before entry; native descriptor inspection
    /// happens after this method releases the manager mutex.
    pub(crate) fn resident_parameter_source(
        &self,
        id: &OffloadUnitId,
        bindings: &[WeightBinding],
    ) -> Result<ResidentParameterSource, ResidentParameterSourceError> {
        use ResidentParameterSourceError as E;
        let state = self.inner.state.try_lock().map_err(|cause| match cause {
            TryLockError::WouldBlock => E::Busy,
            TryLockError::Poisoned(_) => E::Poisoned,
        })?;
        if self.inner.failed_transfer.load(Ordering::Acquire) {
            return Err(E::Unsettled);
        }
        let unit = state.control.unit(id).ok_or(E::Missing)?;
        if unit.bindings() != bindings {
            return Err(E::Mismatch);
        }
        let status = state
            .control
            .ledger()
            .copy_status(id, MemoryTier::Device)
            .map_err(|_| E::Mismatch)?
            .ok_or(E::Missing)?;
        if status.in_flight().is_some() {
            return Err(E::Unsettled);
        }
        let arrays = state
            .storage
            .get(id)
            .and_then(|row| row.device.as_ref())
            .ok_or(E::Missing)?;
        if arrays.arrays.len() != bindings.len()
            || bindings
                .iter()
                .any(|binding| !arrays.arrays.contains_key(binding.name()))
        {
            return Err(E::Mismatch);
        }
        if self.inner.failed_transfer.load(Ordering::Acquire) {
            return Err(E::Unsettled);
        }
        Ok(ResidentParameterSource {
            arrays: arrays.clone(),
        })
    }
}

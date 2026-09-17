//! Explicit eviction at the consuming bank's proven completion boundary.
use super::*;
use eredu_core::residency::{ResidencyEvictionError, ResidencyLedger};
use eredu_runtime::working_memory::OriginalOperationMetadataCustody;
use std::{
    mem::{size_of, size_of_val},
    sync::TryLockError,
};

impl SupplementaryResidencySource {
    /// Called during the actual call census, before the role/bank is admitted.
    /// The caller adds its own result/error wrapper and retained call controls.
    pub(crate) fn original_eviction_control_bytes(&self) -> Option<usize> {
        let controls = [
            size_of::<(
                &ResidencyManager,
                &OffloadUnitId,
                MemoryTier,
                &Self,
                &OriginalOperationMetadataCustody,
                &safemlx::OriginalScopeObserver,
            )>(),
            size_of::<Result<bool, ResidencyError>>(),
            size_of::<Result<(), eredu_runtime::working_memory::WorkingMemoryError>>(),
            size_of::<Result<(), OperationSourceFailure>>(),
            size_of::<MutexGuard<'_, ManagerState>>(),
            size_of::<TryLockError<MutexGuard<'_, ManagerState>>>(),
            size_of::<Option<usize>>(),
            size_of::<Option<&mut transfer::UnitStorage>>(),
            size_of::<Option<&transfer::UnitStorage>>(),
            size_of::<(&mut transfer::UnitStorage, MemoryTier)>(),
            size_of::<Option<ResidentArraysOwner>>(),
            size_of::<Option<ResidentHostOwner>>(),
            size_of::<std::sync::Weak<disk_workspace::DiskRouteActivation>>(),
            size_of::<ResidencyEvictionError>(),
            size_of::<bool>(),
        ];
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)?
            .checked_add(self.projection_control_bytes()?)?
            .checked_add(safemlx::OriginalScopeObserver::control_bytes()?)?
            .checked_add(ResidencyLedger::borrowed_eviction_control_bytes()?)
    }
}
impl ResidencyManager {
    /// The consumed module call invokes this only after its lease's exact
    /// native completion succeeds. Its bank authenticates role/order and owns
    /// every returned error under the prepaid source/operation custody.
    /// No global reaping, transfer progress, ID clone or fresh owner is created.
    pub(crate) fn evict_original_supplementary(
        &self,
        id: &OffloadUnitId,
        tier: MemoryTier,
        source: &SupplementaryResidencySource,
        custody: &OriginalOperationMetadataCustody,
        observer: &safemlx::OriginalScopeObserver,
    ) -> Result<bool, ResidencyError> {
        self.validate_supplementary_source(source)
            .map_err(|_| ResidencyError::OriginalOperationDomain)?;
        if source.ordinal(id).is_none() {
            return Err(ResidencyError::OriginalOperationDomain);
        }
        self.inner
            .validate_operation_custody(custody)
            .map_err(ResidencyError::OriginalInventory)?;
        let current = safemlx::OriginalScopeObserver::require_current()
            .map_err(ResidencyError::OriginalNative)?;
        if !current.same_scope(observer) {
            return Err(ResidencyError::OriginalOperationDomain);
        }
        if let Some(cause) = observer.retained_failure() {
            return Err(ResidencyError::OriginalNative(cause));
        }
        let mut state = self.lock_original(observer)?;
        if tier == MemoryTier::Disk {
            return Err(ResidencyError::OriginalEviction {
                cause: ResidencyEvictionError::InvalidTargetTier,
                tier,
            });
        }
        // Membership was authenticated by the immutable manager source above;
        // unknown/corrupt state stays a fixed refusal without cloning the ID.
        let Some(_) = state
            .control
            .ledger_mut()
            .evict_settled_borrowed(id, tier)
            .map_err(|cause| ResidencyError::OriginalEviction { cause, tier })?
        else {
            return Ok(false);
        };
        if !state
            .storage
            .get_mut(id)
            .is_some_and(|unit| unit.remove_storage(tier))
        {
            return Err(ResidencyError::StatePoisoned);
        }
        Ok(true)
    }
}

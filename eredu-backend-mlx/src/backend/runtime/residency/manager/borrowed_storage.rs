use super::*;
use crate::backend::runtime::residency::storage::{
    visit_checkpoint_storage, RetainedStorageInspectionError, RetainedStorageRef,
    RetainedStorageVisitFailure,
};
use std::sync::{atomic::Ordering, TryLockError};

impl ResidencyManager {
    /// Borrows actual storage fields under a nonblocking state-mutex loan.
    /// The callback may retain an Array only with its nonblocking inspection
    /// clone, or retain an Arc/source owner. It must not reenter this manager,
    /// execute native work, use ordinary Array::clone, or drop retained owners.
    /// It owns retention/capacity checks/errors; any
    /// acquired prefix must outlive this call on error or unwind. No source map,
    /// array handle, name, payload or inventory is constructed by this visitor.
    ///
    /// Completeness covers only this manager at observation time. Failed/pending
    /// transfers may own resources elsewhere and make it false. No ledger pin,
    /// source-health, capacity bound, publication or execution grant is provided.
    pub(crate) fn try_visit_retained_storage<E>(
        &self,
        visitor: &mut dyn FnMut(RetainedStorageRef<'_>) -> Result<(), E>,
    ) -> Result<bool, E>
    where
        E: From<RetainedStorageInspectionError>,
    {
        let mut source_callback_failure = None;
        let result = (|| {
            let state = match self.inner.state.try_lock() {
                Ok(state) => state,
                Err(TryLockError::WouldBlock) => {
                    return Err(RetainedStorageVisitFailure::Inspection(
                        RetainedStorageInspectionError::Busy,
                    ));
                }
                Err(TryLockError::Poisoned(_)) => {
                    return Err(RetainedStorageVisitFailure::inspection(
                        ResidencyError::StatePoisoned,
                    ));
                }
            };
            let mut complete = self
                .borrowed_storage_complete(&state)
                .map_err(RetainedStorageVisitFailure::inspection)?;
            for source in self.inner.sources.ordinary_sources() {
                complete &= visit_checkpoint_storage(
                    source.as_ref(),
                    visitor,
                    &mut source_callback_failure,
                )?;
            }
            if let ResidencySources::Original(source) = &self.inner.sources {
                for host in source.hosts() {
                    for buffer in host.buffers.values() {
                        visitor(RetainedStorageRef::RetainedHost(buffer))
                            .map_err(RetainedStorageVisitFailure::callback)?;
                    }
                }
            }
            for storage in state.storage.values() {
                if let Some(host) = &storage.host {
                    for buffer in host.buffers.values() {
                        visitor(RetainedStorageRef::RetainedHost(buffer))
                            .map_err(RetainedStorageVisitFailure::callback)?;
                    }
                }
                if let Some(device) = &storage.device {
                    // Actual source-backed device cells (including canonical
                    // aliases) retain these owners until device eviction. They
                    // are not a Host cache/tier publication.
                    for buffer in device.arrays.host_sources() {
                        visitor(RetainedStorageRef::RetainedHost(buffer))
                            .map_err(RetainedStorageVisitFailure::callback)?;
                    }
                    for array in device.arrays.retained_values() {
                        visitor(array)
                            .map_err(RetainedStorageVisitFailure::callback)?;
                    }
                }
            }
            // Worker failure can become visible without acquiring this mutex.
            complete &= self
                .borrowed_storage_complete(&state)
                .map_err(RetainedStorageVisitFailure::inspection)?;
            Ok(complete)
        })();
        // No state lock survives caller-defined conversion or secondary Drop.
        result.map_err(RetainedStorageVisitFailure::into_error)
    }

    fn borrowed_storage_complete(&self, state: &ManagerState) -> Result<bool, ResidencyError> {
        let mut complete = !self.inner.failed_transfer.load(Ordering::Acquire);
        // Pending acquisition can precede its first UnitStorage insertion.
        // Enumerate the selected ledger's full unit set, not storage map keys.
        for unit in state.control.units() {
            for tier in [MemoryTier::Host, MemoryTier::Device] {
                if state
                    .control
                    .ledger()
                    .copy_status(unit.id(), tier)?
                    .is_some_and(|copy| copy.in_flight().is_some())
                {
                    complete = false;
                }
            }
        }
        Ok(complete)
    }
}

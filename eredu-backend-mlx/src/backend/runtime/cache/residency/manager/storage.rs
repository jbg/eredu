//! Cold physical ownership of sealed cache blocks and buffered shard payloads.

use super::*;
use crate::backend::runtime::residency::{manager::ResidencyError, storage::RetainedStorage};

impl CacheResidencyManager {
    /// Retains all physical payloads owned by this manager, across logical tiers.
    /// Mutable tails belong to the layer states and must be included separately.
    /// Pending workers can own additional payloads, so their bound stays unknown.
    /// This query never polls workers, reads a shard, or changes cache residency.
    pub fn retained_storage(&self) -> Result<RetainedStorage, ResidencyError> {
        let mut storage = crate::backend::runtime::residency::storage::RetainedStorage::default();
        self.collect_retained_storage(&mut storage)?;
        Ok(storage)
    }

    /// Fills caller-owned storage without allocating an intermediate inventory.
    pub fn collect_retained_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), ResidencyError> {
        if storage.is_snapshot_census() {
            use crate::backend::runtime::residency::storage::{
                RetainedStorageInspectionError, RetainedStorageRef,
            };
            let complete = self
                .try_visit_retained_storage(&mut |entry| {
                    match entry {
                        RetainedStorageRef::Array(array) => storage.include_array(array),
                        RetainedStorageRef::CanonicalArray(cell) => storage.include_canonical_array(cell),
                        RetainedStorageRef::Host(host) => storage.include_host(Arc::clone(host)),
                        RetainedStorageRef::RetainedHost(host) => {
                            storage.include_retained_host(host.clone())
                        }
                        RetainedStorageRef::Bytes(bytes) => {
                            storage.include_bytes(Arc::clone(bytes));
                            Ok(())
                        }
                        RetainedStorageRef::Source(_) => {
                            storage.mark_incomplete();
                            Ok(())
                        }
                    }
                    .map_err(RetainedStorageInspectionError::Residency)
                })
                .map_err(|cause| match cause {
                    RetainedStorageInspectionError::Busy => ResidencyError::OriginalManagerBusy,
                    RetainedStorageInspectionError::Residency(cause) => cause,
                })?;
            if !complete {
                storage.mark_incomplete();
            }
            return Ok(());
        }
        let state = self
            .inner
            .state
            .lock()
            .map_err(|_| ResidencyError::StatePoisoned)?;
        if self
            .inner
            .host_demotion_worker
            .active_payload
            .load(Ordering::Acquire)
            || self
                .inner
                .disk_worker
                .as_ref()
                .is_some_and(|worker| worker.inner.has_retained_work().unwrap_or(true))
        {
            storage.mark_incomplete();
        }
        if state.background_disk_error.is_some()
            || !state.host_write_reservations.is_empty()
            || !state.retiring_host_demotions.is_empty()
            || !state.retiring_disk_reads.is_empty()
        {
            storage.mark_incomplete();
        }
        for record in state.blocks.values() {
            if record.pending_disk().is_some() || record.host_demotion_ticket().is_some() {
                storage.mark_incomplete();
            }
            if let Some(arrays) = record.physical.device_resource() {
                for array in arrays.arrays() {
                    storage.include_array(array)?;
                }
            }
            if let Some(host) = record.host_block() {
                let buffers = match host {
                    HostCacheBlock::KeyValue { keys, values } => [keys, values],
                    HostCacheBlock::CompressedLatentRotary { latent, rotary_key } => {
                        [latent, rotary_key]
                    }
                };
                for buffer in buffers {
                    storage.include_host(Arc::clone(buffer))?;
                }
            }
            if let Some(buffered) = record
                .disk()
                .and_then(|location| location.buffered.as_ref())
            {
                storage.include_bytes(Arc::clone(buffered));
            }
        }
        Ok(())
    }
}

impl CacheResidencyManager {
    /// Borrows actual manager entries while holding only a nonblocking state
    /// loan. The callback may retain an Array only with its nonblocking
    /// inspection clone, or retain an Arc. It must not reenter this manager,
    /// execute native work, use ordinary Array::clone, or drop retained owners.
    /// It owns storage, capacity checks and errors;
    /// acquired prefix owners must survive this call on error or unwind.
    ///
    /// This creates no inventory, array handle, source map or payload copy. All
    /// pending/failed worker evidence stays incomplete; no worker is polled or
    /// retired. Mutable tails and other outer owners remain separate.
    pub(crate) fn try_visit_retained_storage<E>(
        &self,
        visitor: &mut dyn FnMut(
            crate::backend::runtime::residency::storage::RetainedStorageRef<'_>,
        ) -> Result<(), E>,
    ) -> Result<bool, E>
    where
        E: From<crate::backend::runtime::residency::storage::RetainedStorageInspectionError>,
    {
        use crate::backend::runtime::residency::storage::{
            RetainedStorageInspectionError, RetainedStorageRef, RetainedStorageVisitFailure,
        };
        use std::sync::TryLockError;
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
            let before = self.borrowed_storage_complete(&state);
            for record in state.blocks.values() {
                if let Some(arrays) = record.physical.device_resource() {
                    for array in arrays.arrays() {
                        visitor(RetainedStorageRef::Array(array))
                            .map_err(RetainedStorageVisitFailure::callback)?;
                    }
                }
                if let Some(host) = record.host_block() {
                    let buffers = match host {
                        HostCacheBlock::KeyValue { keys, values } => [keys, values],
                        HostCacheBlock::CompressedLatentRotary { latent, rotary_key } => {
                            [latent, rotary_key]
                        }
                    };
                    for buffer in buffers {
                        visitor(RetainedStorageRef::Host(buffer))
                            .map_err(RetainedStorageVisitFailure::callback)?;
                    }
                }
                if let Some(buffered) = record
                    .disk()
                    .and_then(|location| location.buffered.as_ref())
                {
                    visitor(RetainedStorageRef::Bytes(buffered))
                        .map_err(RetainedStorageVisitFailure::callback)?;
                }
            }
            // Active/registry evidence can change independently of this mutex.
            let after = self.borrowed_storage_complete(&state);
            Ok(before && after)
        })();
        result.map_err(RetainedStorageVisitFailure::into_error)
    }

    pub(super) fn borrowed_storage_complete(&self, state: &CacheManagerState) -> bool {
        let host_quiet = !self
            .inner
            .host_demotion_worker
            .active_payload
            .load(Ordering::Acquire);
        let disk_quiet =
            self.inner.disk_worker.as_ref().is_none_or(|worker| {
                matches!(worker.inner.try_has_retained_work(), Ok(Some(false)))
            });
        host_quiet
            && disk_quiet
            && state.background_disk_error.is_none()
            && state.host_write_reservations.is_empty()
            && state.retiring_host_demotions.is_empty()
            && state.retiring_disk_reads.is_empty()
            && state.blocks.values().all(|record| {
                record.pending_disk().is_none() && record.host_demotion_ticket().is_none()
            })
    }
}

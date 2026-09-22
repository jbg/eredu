//! Publication of the actual private loaded realtime numerical/source owners.

use super::{ErasedRealtimeExecutionContract, SelectedRealtimeResources};
use crate::backend::{
    error::Error,
    managed_memory::NativeMemoryOwner,
    runtime::residency::{
        manager::ResidencyManager,
        storage::{RetainedStorage, RetainedStoragePublication},
    },
};
use eredu_runtime::{
    working_memory::{InferenceExecutionIdentity, MemoryLedger, WorkingMemoryError},
    HostSlotTable,
};

/// The new fixed source record has its own closed slot allocation. Nested
/// native/source payloads are inventoried separately; its inline size is never
/// used as a bound for model declarations, checkpoint buffers or manager heaps.
struct LoadedSourceRecord {
    _manager: ResidencyManager,
    execution: InferenceExecutionIdentity,
    pool: MemoryLedger,
    _publication: RetainedStoragePublication,
}

pub(super) struct LoadedRealtimeSource {
    // Record payload and its actual source roots retire before this publication.
    _record: HostSlotTable<LoadedSourceRecord>,
    _host_publication: RetainedStoragePublication,
}

impl LoadedRealtimeSource {
    pub(super) fn validate(
        &self,
        execution: &InferenceExecutionIdentity,
        pool: &MemoryLedger,
    ) -> Result<(), Error> {
        let record = &self._record.slots()[0];
        if !record.execution.same_execution(execution) || !record.pool.same_ledger(pool) {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        Ok(())
    }

    pub(super) fn publish(
        execution: &dyn ErasedRealtimeExecutionContract,
        manager: ResidencyManager,
        resources: &SelectedRealtimeResources,
        loading: &NativeMemoryOwner,
    ) -> Result<Self, Error> {
        let numerical = {
            let _inspection = safemlx::OrdinaryArrayMetadataGuard::enter()?;
            let mut storage = RetainedStorage::default();
            execution.collect_retained_module_storage(&mut storage)?;
            storage
        };
        // Shared with ordinary text loading: every submitted descriptor is
        // retained before submission, and no inventory guard crosses the wait.
        crate::composition::mlx::settle_loaded_numerical_values(numerical, loading)?;
        let storage = {
            let _inspection = safemlx::OrdinaryArrayMetadataGuard::enter()?;
            let mut storage = RetainedStorage::default();
            execution.collect_retained_module_storage(&mut storage)?;
            manager.collect_retained_storage(&mut storage)?;
            storage.include_checkpoint_source(resources.inner._store.as_ref())?;
            if let Some(world) = &resources.inner.world {
                world.collect_idle_retained_storage(&mut storage)?;
            }
            storage
        };
        // The temporary loading exclusion is not attached to any array or host
        // metadata. Each actual backing receives its independent registration;
        // unknown coverage remains a typed refusal before model publication.
        let publication = storage.publish_unquoted(loading)?;
        let record = HostSlotTable::new(Box::new([LoadedSourceRecord {
            _manager: manager,
            execution: resources.execution_identity().clone(),
            pool: loading.pool().clone(),
            _publication: publication,
        }]));
        let mut host = RetainedStorage::default();
        host.include_slot_metadata(record.metadata().clone())?;
        let host_publication = host.publish_unquoted(loading)?;
        Ok(Self {
            _record: record,
            _host_publication: host_publication,
        })
    }
}

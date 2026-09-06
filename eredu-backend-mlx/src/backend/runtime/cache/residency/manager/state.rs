//! Manager-owned cache records, storage, and shared mutable state.

use super::*;

#[derive(Debug, Clone)]
pub(in super::super) struct MlxCacheIoOperation {
    pub(super) ticket: DiskTicket,
    pub(super) reserved_host_bytes: Option<u64>,
}

impl CacheIoOperation for MlxCacheIoOperation {
    fn key(&self) -> &CacheIoOperationKey {
        &self.ticket.key
    }
}

pub(in super::super) type MlxCacheBlockStorage = CacheBlockStorage<
    CacheBlockArrays,
    HostCacheBlock,
    DiskLocation,
    HostDemotionTicket,
    MlxCacheIoOperation,
>;

#[derive(Debug, Clone)]
pub(in super::super) struct CacheBlockRecord {
    pub(in super::super) physical: MlxCacheBlockStorage,
    pub(in super::super) bytes: u64,
    pub(in super::super) shapes: [Vec<i32>; 2],
    pub(in super::super) dtypes: [String; 2],
    pub(in super::super) imported: bool,
}

impl CacheBlockRecord {
    pub(in super::super) fn tier(&self) -> CacheTier {
        self.physical.tier()
    }

    pub(in super::super) fn disk(&self) -> Option<&DiskLocation> {
        self.physical.backing()
    }

    pub(in super::super) fn pending_disk(&self) -> Option<&MlxCacheIoOperation> {
        self.physical.io()
    }

    #[cfg(test)]
    pub(in super::super) fn device_arrays(&self) -> Option<&CacheBlockArrays> {
        self.physical.device_resource()
    }

    pub(in super::super) fn host_block(&self) -> Option<&HostCacheBlock> {
        self.physical.host_resource()
    }

    pub(in super::super) fn host_demotion_ticket(&self) -> Option<&HostDemotionTicket> {
        self.physical.host_demotion()
    }
}

#[derive(Debug)]
pub(in super::super) struct CacheManagerState {
    pub(in super::super) pool: CacheResidencyPool,
    pub(in super::super) pool_manager_id: u64,
    pub(in super::super) generation: u64,
    pub(in super::super) background_disk_error: Option<String>,
    pub(in super::super) lifecycle: CacheBlockLifecycle,
    pub(in super::super) blocks: BTreeMap<CacheBlockId, CacheBlockRecord>,
    pub(in super::super) host_write_reservations:
        HashMap<CacheIoOperationKey, HostWriteReservation>,
    pub(in super::super) retiring_host_demotions: HashMap<u64, RetiringHostDemotion>,
    pub(in super::super) retiring_disk_reads: HashMap<CacheIoOperationKey, (usize, u64)>,
    pub(in super::super) transfer_device: Option<CacheTransferDevice>,
    pub(in super::super) telemetry: CacheResidencyTelemetry,
    pub(in super::super) recent_device_blocks: usize,
    pub(in super::super) device_budget_bytes: u64,
    pub(in super::super) host_budget_bytes: u64,
    pub(in super::super) disk_budget_bytes: Option<u64>,
}

impl CacheManagerState {
    pub(in super::super) fn layer_activity_mut(
        &mut self,
        global_layer: usize,
    ) -> &mut CacheLayerResidencyStats {
        self.telemetry.layer_activity_mut(global_layer)
    }
}

impl Drop for CacheManagerState {
    fn drop(&mut self) {
        // The final state owner has exclusive access without taking its mutex.
        // Disk commits hold only a weak reference to avoid an ownership cycle.
        // A late commit removes its own ephemeral output if this state is gone.
        for record in self.blocks.values() {
            remove_ephemeral_file(record);
        }
    }
}

#[derive(Debug)]
pub(super) struct CacheResidencyManagerInner {
    pub(super) options: PagedCacheOptions,
    pub(super) state: Arc<Mutex<CacheManagerState>>,
    pub(super) host_demotion_worker: Arc<HostDemotionWorker>,
    pub(super) disk_worker: Option<Arc<DiskWorker>>,
    pub(super) pool_membership: Arc<CachePoolMembership>,
}

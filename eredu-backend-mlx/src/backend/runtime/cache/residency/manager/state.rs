//! Manager-owned cache records, storage, and shared mutable state.

use super::*;
use eredu_runtime::cache::CacheRecordTable;

#[derive(Debug, Clone)]
pub(in super::super) struct MlxCacheIoOperation {
    pub(super) ticket: DiskTicket,
    pub(super) reserved_host_bytes: Option<u64>,
    pub(super) prepared_read: Option<source::DiskReadOccupancy>,
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
    // Set only by a completed exact scan. Native/source owners retire first.
    pub(in super::super) original_discard: Option<super::original_discard::PendingOriginalDiscard>,
    pub(in super::super) _metadata_funding: Option<eredu_nn::workspace::WorkspaceMetadataFunding>,
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
    pub(in super::super) blocks: CacheRecordTable<CacheBlockId, CacheBlockRecord>,
    pub(in super::super) history_retentions: Vec<Weak<CacheHistoryRetention>>,
    pub(in super::super) host_write_reservations:
        CacheRecordTable<CacheIoOperationKey, HostWriteReservation>,
    pub(in super::super) retiring_host_demotions: HashMap<u64, RetiringHostDemotion>,
    pub(in super::super) retiring_disk_reads: HashMap<CacheIoOperationKey, (usize, u64)>,
    pub(in super::super) transfer_device: Option<CacheTransferDevice>,
    pub(in super::super) telemetry: CacheResidencyTelemetry,
    pub(in super::super) report_rows: Option<eredu_runtime::cache::CacheTelemetryRows>,
    pub(in super::super) recent_device_blocks: usize,
    pub(in super::super) device_budget_bytes: u64,
    pub(in super::super) host_budget_bytes: u64,
    pub(in super::super) disk_budget_bytes: Option<u64>,
    // Actual copied block/tail resources retire before their registered Q/H.
    pub(in super::super) original_copy_custody: Option<eredu_core::HostPreparationAuthority>,
}

impl CacheManagerState {
    /// Shared empty canonical state assembly. Workers, pool membership and
    /// destination funding are supplied by the actual manager constructor.
    pub(super) fn empty(
        options: &PagedCacheOptions,
        pool: CacheResidencyPool,
        session_id: u64,
        telemetry: CacheResidencyTelemetry,
    ) -> Self {
        Self {
            pool,
            pool_manager_id: session_id,
            generation: 0,
            original_copy_custody: None,
            background_disk_error: None,
            lifecycle: CacheBlockLifecycle::new(),
            blocks: eredu_runtime::cache::CacheRecordTable::new(),
            history_retentions: Vec::new(),
            host_write_reservations: CacheRecordTable::new(),
            retiring_host_demotions: HashMap::new(),
            retiring_disk_reads: HashMap::new(),
            transfer_device: None,
            telemetry,
            report_rows: None,
            recent_device_blocks: options.recent_device_blocks(),
            device_budget_bytes: options.device_budget_bytes(),
            host_budget_bytes: options.host_budget_bytes(),
            disk_budget_bytes: match options.live_disk_policy() {
                LiveCacheDiskPolicy::Disabled => None,
                LiveCacheDiskPolicy::Enabled { budget_bytes, .. } => Some(*budget_bytes),
            },
        }
    }
    pub(in super::super) fn layer_activity_mut(
        &mut self,
        global_layer: usize,
    ) -> &mut CacheLayerResidencyStats {
        self.telemetry.layer_activity_mut(global_layer)
    }
}

// Ephemeral file retirement belongs to the actual DiskLocation source token.
// Worker tasks/results can therefore outlive this canonical manager safely.

#[derive(Debug)]
pub(super) struct CacheResidencyManagerInner {
    pub(super) options: PagedCacheOptions,
    pub(super) state: Arc<Mutex<CacheManagerState>>,
    pub(super) host_demotion_worker: Arc<HostDemotionWorker>,
    pub(super) disk_worker: Option<Arc<DiskWorker>>,
    pub(super) pool_membership: Arc<CachePoolMembership>,
    // Both canonical state and worker aliases retire before construction H.
    pub(super) _metadata_funding: Option<eredu_nn::workspace::WorkspaceMetadataFunding>,
}

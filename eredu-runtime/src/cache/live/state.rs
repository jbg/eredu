//! Manager-owned cache records, storage, and shared mutable state.

use super::*;

#[derive(Debug, Clone)]
pub(super) struct CacheIoEntry<M: LiveCacheBackend> {
    pub(super) ticket: DiskTicket<M>,
    pub(super) reserved_host_bytes: Option<u64>,
}

impl<M: LiveCacheBackend> CacheIoOperation for CacheIoEntry<M> {
    fn key(&self) -> &CacheIoOperationKey {
        &self.ticket.key
    }
}

pub(super) type LiveCacheBlockStorage<M> = CacheBlockStorage<
    Page<M>,
    HostCacheBlock<M>,
    DiskLocation,
    HostDemotionTicket<M>,
    CacheIoEntry<M>,
>;

#[derive(Debug, Clone)]
pub(super) struct CacheBlockRecord<M: LiveCacheBackend> {
    pub(super) physical: LiveCacheBlockStorage<M>,
    pub(super) bytes: u64,
    pub(super) shapes: [Vec<i32>; 2],
    pub(super) dtypes: [String; 2],
    pub(super) imported: bool,
}

impl<M: LiveCacheBackend> CacheBlockRecord<M> {
    pub(super) fn tier(&self) -> CacheTier {
        self.physical.tier()
    }

    pub(super) fn disk(&self) -> Option<&DiskLocation> {
        self.physical.backing()
    }

    pub(super) fn pending_disk(&self) -> Option<&CacheIoEntry<M>> {
        self.physical.io()
    }

    pub(super) fn host_block(&self) -> Option<&HostCacheBlock<M>> {
        self.physical.host_resource()
    }

    pub(super) fn host_demotion_ticket(&self) -> Option<&HostDemotionTicket<M>> {
        self.physical.host_demotion()
    }
}

#[derive(Debug)]
pub(super) struct CacheManagerState<M: LiveCacheBackend> {
    pub(super) pool: CacheResidencyPool,
    pub(super) pool_manager_id: u64,
    pub(super) generation: u64,
    pub(super) background_disk_error: Option<Arc<CacheResidencyError>>,
    pub(super) lifecycle: CacheBlockLifecycle,
    pub(super) blocks: BTreeMap<CacheBlockId, CacheBlockRecord<M>>,
    pub(super) history_retentions: Vec<Weak<CacheHistoryRetention>>,
    pub(super) host_write_reservations: HashMap<CacheIoOperationKey, HostWriteReservation<M>>,
    pub(super) retiring_host_demotions: HashMap<u64, RetiringHostDemotion>,
    pub(super) retiring_disk_reads: HashMap<CacheIoOperationKey, (usize, u64)>,
    pub(super) transfer_device: Option<M::Device>,
    pub(super) telemetry: CacheResidencyTelemetry,
    pub(super) recent_device_blocks: usize,
    pub(super) device_budget_bytes: u64,
    pub(super) host_budget_bytes: u64,
    pub(super) disk_budget_bytes: Option<u64>,
}

impl<M: LiveCacheBackend> CacheManagerState<M> {
    /// A write retaining host capacity until its worker retires the owned bytes.
    pub(super) fn pending_spill(&self) -> Option<&DiskTicket<M>> {
        self.host_write_reservations
            .values()
            .min_by_key(|reservation| reservation.reservation_id)
            .map(|reservation| &reservation.ticket)
            .or_else(|| {
                self.blocks.values().find_map(|record| {
                    record
                        .pending_disk()
                        .filter(|pending| pending.ticket.key.kind == CacheIoOperationKind::Write)
                        .map(|pending| &pending.ticket)
                })
            })
    }

    pub(super) fn layer_activity_mut(
        &mut self,
        global_layer: usize,
    ) -> &mut CacheLayerResidencyStats {
        self.telemetry.layer_activity_mut(global_layer)
    }
}

impl<M: LiveCacheBackend> Drop for CacheManagerState<M> {
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
pub(super) struct CacheResidencyManagerInner<M: LiveCacheBackend> {
    pub(super) options: PagedCacheOptions,
    pub(super) state: Arc<Mutex<CacheManagerState<M>>>,
    pub(super) host_demotion_worker: Arc<HostDemotionWorker<M>>,
    pub(super) disk_worker: Option<Arc<DiskWorker<M>>>,
    pub(super) pool_membership: Arc<CachePoolMembership>,
}

//! Actual manager/worker storage installation before the first original task.
use super::*;
use eredu_runtime::cache::{
    CacheRecordTable, PreparedCacheIoQueue, PreparedCacheIoRegistry, PreparedCacheTable,
    RetiredCacheIoQueue, RetiredCacheIoRegistry,
};
type Writes = CacheRecordTable<CacheIoOperationKey, HostWriteReservation>;

pub(crate) struct PreparedDiskWorker {
    registry: Option<PreparedCacheIoRegistry<DiskResult>>,
    queue: Option<PreparedCacheIoQueue<DiskTask, DiskResult>>,
    writes: Option<PreparedCacheTable<CacheIoOperationKey, HostWriteReservation>>,
    retired_registry: Option<RetiredCacheIoRegistry<DiskResult>>,
    retired_queue: Option<RetiredCacheIoQueue<DiskTask, DiskResult>>,
    retired_writes: Option<Writes>,
    manager: CacheResidencyManager,
    worker: Arc<DiskWorker>,
    generation: u64,
    funding: HostMetadataFunding,
}
/// Same installed storage and exact manager generation, not a file/native grant.
pub(crate) struct InstalledDiskWorker {
    manager: CacheResidencyManager,
    worker: Arc<DiskWorker>,
    generation: u64,
    funding: HostMetadataFunding,
}
#[derive(thiserror::Error)]
#[error("{cause}")]
pub(crate) struct DiskWorkerInstallFailure {
    #[source]
    cause: CacheSourceFailure,
    retained: Mutex<PreparedDiskWorker>,
}
impl std::fmt::Debug for DiskWorkerInstallFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskWorkerInstallFailure")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl CacheBlockSourceLoan<'_> {
    /// Uses the configured worker's exact queue capacity for selected live
    /// writes or authenticated retained reads. Both authoritative registries
    /// and the manager's ordinary write-reservation table are reused.
    pub(crate) fn prepare_disk_worker(
        &self,
        maximum: usize,
        context: &WorkspaceContext,
    ) -> Result<PreparedDiskWorker, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        let funding = context.metadata_funding().ok_or_else(|| {
            CacheSourceFailure::metadata(WorkspaceMetadataError::Unqualified.into(), context)
        })?;
        context
            .charge_metadata(
                PreparedDiskWorker::fixed_controls()
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        if !matches!(
            self.manager.inner.options.live_disk_policy(),
            LiveCacheDiskPolicy::Enabled { .. }
        ) {
            // The queue also serves authenticated imported reads. A live-write
            // policy cannot authorize those files, and disabling writes does
            // not remove the actual canonical file source from this manager.
            let mut retained_read = false;
            for block in self.all_blocks() {
                if block.disk().is_some() {
                    retained_read |= block.retained_disk_types().map_err(fail)?.is_some();
                }
            }
            if !retained_read {
                return Err(fail(CacheSourceError::PromotionRequired));
            }
        }
        let worker = self
            .manager
            .inner
            .disk_worker
            .as_ref()
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        let registry = worker
            .inner
            .prepare_registry(maximum, context)
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        let queue = worker
            .inner
            .prepare_queue(context)
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        let writes = PreparedCacheTable::prepare(maximum, context)
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        Ok(PreparedDiskWorker {
            registry: Some(registry),
            queue: Some(queue),
            writes: Some(writes),
            retired_registry: None,
            retired_queue: None,
            retired_writes: None,
            manager: self.manager.clone(),
            worker: worker.clone(),
            generation: self.generation,
            funding,
        })
    }
}
impl PreparedDiskWorker {
    pub(crate) fn control_bytes(maximum: usize, queue_capacity: usize) -> Option<usize> {
        Self::fixed_controls()?
            .checked_add(PreparedCacheIoRegistry::<DiskResult>::control_bytes(
                maximum,
            )?)?
            .checked_add(PreparedCacheIoQueue::<DiskTask, DiskResult>::control_bytes(
                queue_capacity,
            )?)?
            .checked_add(PreparedCacheTable::<
                CacheIoOperationKey,
                HostWriteReservation,
            >::control_bytes(maximum)?)
    }
    fn fixed_controls() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<InstalledDiskWorker>(),
            size_of::<DiskWorkerInstallFailure>(),
            initialized_mutex_control_bytes::<PreparedDiskWorker>()?,
            size_of::<Result<Self, CacheSourceFailure>>(),
            size_of::<Result<InstalledDiskWorker, DiskWorkerInstallFailure>>(),
            size_of::<Result<(), CacheSourceFailure>>(),
            size_of::<MutexGuard<'_, CacheManagerState>>(),
            size_of::<
                Result<
                    MutexGuard<'_, CacheManagerState>,
                    TryLockError<MutexGuard<'_, CacheManagerState>>,
                >,
            >(),
            size_of::<Option<RetiredCacheIoRegistry<DiskResult>>>(),
            size_of::<Option<RetiredCacheIoQueue<DiskTask, DiskResult>>>(),
            size_of::<Option<Writes>>(),
            Writes::mutation_control_bytes()?,
            CacheBlockSource::retained_disk_control_bytes(),
            size_of::<bool>(),
            size_of::<CacheBlockSource<'_>>(),
            size_of::<eredu_runtime::cache::CacheRecordTableIter<'_, CacheBlockId, CacheBlockRecord>>(
            ),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    /// Exact idle checks precede each storage replacement. If a later check
    /// fails, all installed tables remain valid and the error retains the rest
    /// plus old empty storage. No task/canonical tier is started or rolled back.
    pub(crate) fn install(mut self) -> Result<InstalledDiskWorker, DiskWorkerInstallFailure> {
        let result = (|| {
            let fail = |cause| CacheSourceFailure {
                cause: CacheSourceFailureCause::Source(cause),
                _funding: Some(self.funding.clone()),
            };
            let manager = self.manager.clone();
            let mut state = manager.inner.state.try_lock().map_err(|cause| {
                fail(match cause {
                    TryLockError::WouldBlock => CacheSourceError::Busy,
                    TryLockError::Poisoned(_) => CacheSourceError::Poisoned,
                })
            })?;
            if state.generation != self.generation
                || !manager.borrowed_storage_complete(&state)
                || !state.host_write_reservations.is_empty()
                || manager
                    .inner
                    .disk_worker
                    .as_ref()
                    .is_none_or(|worker| !Arc::ptr_eq(worker, &self.worker))
            {
                return Err(fail(CacheSourceError::Identity));
            }
            if let Some(registry) = self.registry.take() {
                match registry.install(&self.worker.inner) {
                    Ok(retired) => self.retired_registry = Some(retired),
                    Err(error) => {
                        let (cause, retained) = error.into_parts();
                        self.registry = Some(retained);
                        return Err(fail(CacheSourceError::DiskWorker(cause)));
                    }
                }
            }
            if let Some(queue) = self.queue.take() {
                match queue.install(&self.worker.inner) {
                    Ok(retired) => self.retired_queue = Some(retired),
                    Err(error) => {
                        let (cause, retained) = error.into_parts();
                        self.queue = Some(retained);
                        return Err(fail(CacheSourceError::DiskWorker(cause)));
                    }
                }
            }
            let writes = self.writes.take().expect("one write-table installation");
            self.retired_writes = Some(
                state
                    .host_write_reservations
                    .install(writes)
                    .unwrap_or_else(|_| unreachable!("validated empty write table")),
            );
            Ok(())
        })();
        match result {
            Err(cause) => {
                let retained = Mutex::new(self);
                drop(retained.lock().expect("new unshared retained-worker mutex"));
                Err(DiskWorkerInstallFailure { cause, retained })
            }
            Ok(()) => {
                // All manager and worker borrows have ended before old metadata
                // or its account can retire.
                drop((
                    self.retired_registry.take(),
                    self.retired_queue.take(),
                    self.retired_writes.take(),
                ));
                Ok(InstalledDiskWorker {
                    manager: self.manager,
                    worker: self.worker,
                    generation: self.generation,
                    funding: self.funding,
                })
            }
        }
    }
}

#[path = "worker/operation.rs"]
mod operation;
pub(crate) use operation::{DiskWriteOperation, DiskWriteOperationFailure};

impl InstalledDiskWorker {
    pub(crate) fn prepare_read(
        &self,
        read: super::super::disk_read::PreparedDiskRead,
        context: &WorkspaceContext,
    ) -> Result<super::super::disk_read::DiskReadOperation, CacheSourceFailure> {
        super::super::disk_read::prepare_operation(
            read,
            &self.manager,
            &self.worker,
            self.generation,
            context,
        )
    }
}

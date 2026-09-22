//! Asynchronous host/disk residency operations.

use super::*;

#[derive(Debug, Clone)]
struct DiskLocation {
    inner: Arc<DiskLocationData>,
    // Every alias drops its actual immutable fields/file before their H.
    funding: Option<eredu_nn::workspace::HostMetadataFunding>,
}
#[derive(Debug)]
struct DiskLocationData {
    path: PathBuf,
    first_name: String,
    second_name: String,
    persistent: bool,
    buffered: Option<Arc<[u8]>>,
    payload_sha256: Option<String>,
    payload_verification: Arc<OnceLock<Result<(), String>>>,
    // Declared last: paths/verification and worker payloads retire before the
    // final source owner removes an ephemeral file. Persistent files have none.
    live_source: Option<LiveCacheBlockSource>,
    persistent_source: Option<eredu_runtime::cache::PersistentCacheBlockSource>,
}

impl std::ops::Deref for DiskLocation {
    type Target = DiskLocationData;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl DiskLocation {
    fn ordinary(data: DiskLocationData) -> Self {
        Self {
            inner: Arc::new(data),
            funding: None,
        }
    }
}
#[path = "io/file_source.rs"]
mod file_source;
#[path = "io/location.rs"]
mod location;
pub(crate) use file_source::{CacheFileReadFailure, CacheFileSource, PreparedCacheFileRead};

enum DiskTask {
    PreparedWrite(manager::PreparedDiskWrite),
    PreparedRead(manager::PreparedDiskRead),
    Write {
        directory: PathBuf,
        id: CacheBlockId,
        block: HostCacheBlock,
        commit: Option<DiskWriteCommit>,
    },
    Read {
        location: DiskLocation,
        representation: CacheRepresentation,
    },
    #[cfg(test)]
    Pause {
        started: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
    },
    #[cfg(test)]
    PauseWrite {
        started: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
        commit: Option<DiskWriteCommit>,
    },
    #[cfg(test)]
    Panic,
}

struct DiskWriteCommit {
    state: Weak<Mutex<CacheManagerState>>,
    key: CacheIoOperationKey,
    reservation_id: u64,
    armed: bool,
}

#[derive(Debug, Clone)]
struct HostWriteReservation {
    reservation_id: u64,
    global_layer: usize,
    logical_bytes: u64,
    host_capacity: u64,
    ticket: DiskTicket,
    // Present only when the exact original task independently owns occupancy.
    prepared: Option<manager::DiskWriteOccupancy>,
}

#[derive(Debug, Clone)]
struct RetiringHostDemotion {
    id: CacheBlockId,
    device_bytes: u64,
    host_bytes: u64,
}

#[derive(Debug, Clone)]
enum DiskResult {
    PreparedWrite(manager::PreparedDiskWriteOutput),
    PreparedRead(manager::PreparedDiskReadOutput),
    Write(DiskLocation),
    Read(HostCacheBlock),
    #[cfg(test)]
    Test,
}

impl DiskWriteCommit {
    fn reconcile(&self, result: &Result<DiskResult, CacheResidencyError>) {
        let Some(state) = self.state.upgrade() else {
            // The result owns its unpublished file until the worker's actual
            // completion/discard path releases the last source alias.
            return;
        };
        let Ok(mut state) = state.lock() else {
            return;
        };
        let stale = state.generation != self.key.generation;
        match result {
            Ok(DiskResult::Write(location)) if !stale => {
                let mut transitioned_to_disk = false;
                let mut bytes = 0;
                debug_assert_eq!(state.lifecycle.lease_count(&self.key.id).ok(), Some(0));
                if let Some(record) = state.blocks.get_mut(&self.key.id) {
                    if record.physical.io_matches(&self.key)
                        && record.physical.phase() == CacheStoragePhase::HostWriting
                    {
                        bytes = record.bytes;
                        record
                            .physical
                            .finish_write(&self.key, location.clone())
                            .expect("matching host write has a valid core transition");
                        transitioned_to_disk = true;
                    }
                }
                if bytes != 0 {
                    state.telemetry.report.transfer_bytes += bytes;
                    state
                        .layer_activity_mut(self.key.id.global_layer)
                        .transfer_bytes += bytes;
                }
                if transitioned_to_disk {
                    state.telemetry.report.disk_demotions += 1;
                    state
                        .layer_activity_mut(self.key.id.global_layer)
                        .disk_demotions += 1;
                }
            }
            Ok(DiskResult::Write(_)) => {
                // Stale generations retain the exact result owner until the
                // worker discards it; no path is unlinked through a borrow.
            }
            Ok(_) => {
                state.telemetry.report.failures += 1;
                state.layer_activity_mut(self.key.id.global_layer).failures += 1;
                state.background_disk_error =
                    Some("cache disk worker returned an unexpected write result".into());
            }
            Err(_) if stale => {}
            Err(error) => {
                if let Some(record) = state.blocks.get_mut(&self.key.id) {
                    record.physical.fail_io_if_matches(&self.key);
                }
                state.telemetry.report.failures += 1;
                state.layer_activity_mut(self.key.id.global_layer).failures += 1;
                state.background_disk_error = Some(error.to_string());
            }
        }
        update_report_totals(&mut state);
        drop(state);
    }
}

impl Drop for DiskWriteCommit {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let Some(state) = self.state.upgrade() else {
            return;
        };
        let Ok(mut state) = state.lock() else {
            return;
        };
        if state
            .host_write_reservations
            .get(&self.key)
            .is_none_or(|reservation| reservation.reservation_id != self.reservation_id)
        {
            return;
        }
        if let Some(record) = state.blocks.get_mut(&self.key.id) {
            record.physical.fail_io_if_matches(&self.key);
        }
        if state.host_write_reservations.remove(&self.key).is_some() {
            update_report_totals(&mut state);
        }
    }
}

type RuntimeDiskWorker = RuntimeCacheIoWorker<DiskTask, DiskResult>;
type RuntimeDiskSubmission = RuntimeCacheIoSubmission<DiskTask, DiskResult>;

#[derive(Debug, Clone)]
struct DiskTicket {
    inner: RuntimeCacheIoTicket<DiskResult>,
}

impl std::ops::Deref for DiskTicket {
    type Target = RuntimeCacheIoTicket<DiskResult>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DiskTicket {
    fn wait(&self) -> Result<DiskResult, CacheResidencyError> {
        self.inner.wait().map_err(disk_worker_error)
    }

    fn cancel(&self) -> bool {
        self.inner.cancel()
    }

    fn wait_for_task_resources(&self) -> Result<(), CacheResidencyError> {
        self.inner
            .wait_for_task_resources()
            .map_err(disk_worker_error)
    }

    #[cfg(test)]
    fn shares_completion_with(&self, other: &Self) -> bool {
        self.inner.shares_completion_with(&other.inner)
    }
}

struct DiskSubmission {
    inner: RuntimeDiskSubmission,
    ticket: DiskTicket,
    write_reservation_id: Option<u64>,
}

impl DiskSubmission {
    fn enqueue(self) -> Result<CacheIoSubmissionOutcome, CacheResidencyError> {
        self.inner.enqueue().map_err(disk_worker_error)
    }
}

#[derive(Debug)]
struct DiskWorker {
    inner: RuntimeDiskWorker,
}

impl DiskWorker {
    fn new(capacity: usize) -> Result<Self, CacheResidencyError> {
        Ok(Self {
            inner: RuntimeDiskWorker::new(
                capacity,
                "eredu-mlx-cache-disk",
                execute_disk_task,
                discard_disk_result,
            )
            .map_err(disk_worker_error)?
            .with_nonblocking_drop(),
        })
    }

    fn prepare(
        &self,
        key: CacheIoOperationKey,
        task: DiskTask,
    ) -> Result<DiskSubmission, CacheResidencyError> {
        self.prepare_with_write_reservation(key, task, None)
    }

    fn prepare_with_write_reservation(
        &self,
        key: CacheIoOperationKey,
        task: DiskTask,
        write_reservation_id: Option<u64>,
    ) -> Result<DiskSubmission, CacheResidencyError> {
        let mut inner = self.inner.prepare(key, task).map_err(disk_worker_error)?;
        let joined = inner.joined;
        if joined {
            if let Some(DiskTask::Write {
                commit: Some(commit),
                ..
            }) = inner.joined_task_mut()
            {
                commit.armed = false;
            }
        }
        let ticket = DiskTicket {
            inner: inner.ticket.clone(),
        };
        Ok(DiskSubmission {
            inner,
            ticket,
            write_reservation_id: if joined { None } else { write_reservation_id },
        })
    }

    fn prepare_write(
        &self,
        generation: u64,
        directory: &Path,
        id: &CacheBlockId,
        block: &HostCacheBlock,
        state: Weak<Mutex<CacheManagerState>>,
    ) -> Result<DiskSubmission, CacheResidencyError> {
        let reservation_id = NEXT_HOST_WRITE_RESERVATION_ID.fetch_add(1, Ordering::Relaxed);
        self.prepare_with_write_reservation(
            CacheIoOperationKey {
                generation,
                id: id.clone(),
                kind: CacheIoOperationKind::Write,
            },
            DiskTask::Write {
                directory: directory.to_path_buf(),
                id: id.clone(),
                block: block.clone(),
                commit: Some(DiskWriteCommit {
                    state,
                    key: CacheIoOperationKey {
                        generation,
                        id: id.clone(),
                        kind: CacheIoOperationKind::Write,
                    },
                    reservation_id,
                    armed: true,
                }),
            },
            Some(reservation_id),
        )
    }

    fn prepare_read(
        &self,
        generation: u64,
        id: &CacheBlockId,
        location: &DiskLocation,
        representation: CacheRepresentation,
    ) -> Result<DiskSubmission, CacheResidencyError> {
        self.prepare(
            CacheIoOperationKey {
                generation,
                id: id.clone(),
                kind: CacheIoOperationKind::Read,
            },
            DiskTask::Read {
                location: location.clone(),
                representation,
            },
        )
    }

    fn retire(&self, ticket: &DiskTicket) {
        self.inner.retire(&ticket.inner);
    }
}

fn execute_disk_task(task: DiskTask) -> Result<DiskResult, String> {
    // The actual prepared task publishes its typed result into its preallocated
    // owner. Generic completion means the task settled, not that its I/O succeeded.
    // No ordinary string conversion, manager callback or constructor runs here.
    let task = match task {
        DiskTask::PreparedWrite(task) => return Ok(DiskResult::PreparedWrite(task.run())),
        DiskTask::PreparedRead(task) => return Ok(DiskResult::PreparedRead(task.run())),
        other => other,
    };
    let mut write_commit = None;
    let result = catch_unwind(AssertUnwindSafe(|| match task {
        DiskTask::PreparedWrite(_) | DiskTask::PreparedRead(_) => {
            unreachable!("handled prepared task")
        }
        DiskTask::Write {
            directory,
            id,
            block,
            commit,
        } => {
            write_commit = commit;
            write_live_block(&directory, &id, &block).map(DiskResult::Write)
        }
        DiskTask::Read {
            location,
            representation,
        } => load_host_cache_block_direct(&location, representation).map(DiskResult::Read),
        #[cfg(test)]
        DiskTask::Pause { started, release } => {
            let _ = started.send(());
            let _ = release.recv();
            Ok(DiskResult::Test)
        }
        #[cfg(test)]
        DiskTask::PauseWrite {
            started,
            release,
            commit,
        } => {
            write_commit = commit;
            let _ = started.send(());
            let _ = release.recv();
            Err(CacheResidencyError::Runtime(
                "injected canceled cache write".into(),
            ))
        }
        #[cfg(test)]
        DiskTask::Panic => panic!("injected cache disk worker panic"),
    }))
    .unwrap_or_else(|_| {
        Err(CacheResidencyError::Runtime(
            "live cache disk worker operation panicked".into(),
        ))
    });
    if let Some(commit) = &write_commit {
        commit.reconcile(&result);
    }
    drop(write_commit);
    result.map_err(|error| error.to_string())
}

fn discard_disk_result(result: DiskResult) {
    // Dropping the actual location releases one shared file owner. A published
    // manager or a queued read may still retain the same immutable file.
    drop(result);
}

fn disk_worker_error(error: CacheIoWorkerError) -> CacheResidencyError {
    match error {
        CacheIoWorkerError::Poisoned => CacheResidencyError::ManagerPoisoned,
        CacheIoWorkerError::OperationFailed(message) => CacheResidencyError::Runtime(message),
        CacheIoWorkerError::Cancelled { generation } => {
            CacheResidencyError::DiskOperationCancelled { generation }
        }
        CacheIoWorkerError::Spawn {
            thread_name,
            source,
        } => CacheResidencyError::Io {
            action: "start live cache disk worker",
            path: PathBuf::from(thread_name),
            source,
        },
        CacheIoWorkerError::Execution(error) => error.into(),
    }
}

#[path = "manager.rs"]
mod manager;
pub use manager::{
    CacheBlockLease, CacheBlockPrefetch, CacheResidencyManager, LoadedPromptCacheStateTensor,
    PromptCacheStateArray,
};
use manager::{
    CacheManagerState, load_host_cache_block_direct, update_report_totals, write_live_block,
};
pub(crate) use manager::{CacheTransferStreamError, PreparedCacheTransferStream, PromptCacheTail};
pub(crate) use manager::{PromptCacheMaterialization, load_prompt_cache_state_tensors_funded};

pub(crate) use manager::{
    CacheBlockSource, CacheBlockSourceLoan, CacheDiskSource, CacheSourceError, CacheSourceFailure,
    CacheSourceFailureCause, IndependentCacheManagerPlan, PinnedCacheBlock, PinnedCacheBlockLease,
    PinnedCacheSource, PreparedIndependentCacheManager,
};

pub(crate) use manager::{
    CacheBlockMetadata, CatalogInstallFailure, InstalledManagerCatalog, PreparedCacheDiscard,
    PreparedFloatingBlockMetadata, PreparedManagerCatalog,
};

pub(crate) use manager::{PagedArrayCopyLayout, PreparedPagedArrayCopy};

pub(crate) use manager::{
    PreparedCacheHostPromotion, PreparedCacheHostPromotionSlots, PreparedHostPromotion,
    PreparedHostReturn, PreparedOrdinaryCacheHostPromotion,
};

pub(crate) use manager::{
    PreparedCacheHostDemotion, PreparedCacheTransferSource, PreparedHostEviction,
    PreparedOrdinaryCacheHostDemotion, StoredCacheHostSource,
};

pub(crate) use manager::PreparedDiskWrite;

pub(crate) use manager::{OrdinaryWrittenCacheHostSource, PreparedDiskWriteHostRetirement};
pub(crate) use manager::{PreparedCacheDiskWriteSource, PreparedDiskWriteDestination};

pub(crate) use manager::{
    DiskWriteOperation, DiskWriteOperationFailure, InstalledDiskWorker, PreparedDiskWorker,
};

pub(crate) use manager::{DiskReadBinding, PreparedDiskReadDestination, disk_read_source_facts};

pub(crate) use manager::{
    DiskReadFinishFailure, DiskReadOperation, DiskReadOperationFailure, PreparedDiskReadSource,
};

pub(crate) use manager::PreparedInitialDiskReturn;
pub(crate) use manager::{CacheHistoryOwner, PreparedCacheHistory};

pub(crate) use manager::{OrdinaryDiskReadSource, OrdinaryReadCacheHostSource};

#[cfg(test)]
pub use manager::{load_prompt_cache_state_tensors, open_prompt_cache};

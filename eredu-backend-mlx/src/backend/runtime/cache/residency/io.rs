//! Asynchronous host/disk residency operations.

use super::*;

#[derive(Debug, Clone)]
struct DiskLocation {
    path: PathBuf,
    first_name: String,
    second_name: String,
    persistent: bool,
    buffered: Option<Arc<[u8]>>,
    payload_sha256: Option<String>,
    payload_verification: Arc<OnceLock<Result<(), String>>>,
}

enum DiskTask {
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
}

#[derive(Debug, Clone)]
struct RetiringHostDemotion {
    id: CacheBlockId,
    device_bytes: u64,
    host_bytes: u64,
}

#[derive(Debug, Clone)]
enum DiskResult {
    Write(DiskLocation),
    Read(HostCacheBlock),
    #[cfg(test)]
    Test,
}

impl DiskWriteCommit {
    fn reconcile(&self, result: &Result<DiskResult, CacheResidencyError>) {
        let Some(state) = self.state.upgrade() else {
            if let Ok(DiskResult::Write(location)) = result {
                if !location.persistent {
                    let _ = fs::remove_file(&location.path);
                }
            }
            return;
        };
        let Ok(mut state) = state.lock() else {
            return;
        };
        let stale = state.generation != self.key.generation;
        let mut cleanup = None;
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
            Ok(DiskResult::Write(location)) => {
                if !location.persistent {
                    cleanup = Some(location.path.clone());
                }
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
        if let Some(path) = cleanup {
            let _ = fs::remove_file(path);
        }
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
            .map_err(disk_worker_error)?,
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
    let mut write_commit = None;
    let result = catch_unwind(AssertUnwindSafe(|| match task {
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
    if let DiskResult::Write(location) = result {
        if !location.persistent {
            let _ = fs::remove_file(location.path);
        }
    }
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
use manager::{
    load_host_cache_block_direct, update_report_totals, write_live_block, CacheManagerState,
};
pub use manager::{
    load_prompt_cache_state_tensors, open_prompt_cache, CacheBlockLease, CacheBlockPrefetch,
    CacheResidencyManager, LoadedPromptCacheStateTensor, PromptCacheStateArray,
};

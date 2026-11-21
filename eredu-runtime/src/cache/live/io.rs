//! Asynchronous host/disk residency operations.

use super::*;

#[derive(Debug, Clone)]
pub(super) struct DiskLocation {
    pub(super) path: PathBuf,
    pub(super) first_name: String,
    pub(super) second_name: String,
    pub(super) persistent: bool,
    pub(super) source: Option<(
        Arc<dyn eredu_core::cache::mechanisms::PromptCacheShardPayload>,
        PromptCacheBlock,
    )>,
    pub(super) logical_bytes: u64,
}

impl From<crate::cache::PublishedLiveCacheBlock> for DiskLocation {
    fn from(block: crate::cache::PublishedLiveCacheBlock) -> Self {
        Self {
            path: block.path,
            first_name: block.names.0.into(),
            second_name: block.names.1.into(),
            persistent: false,
            source: None,
            logical_bytes: block.logical_bytes,
        }
    }
}

pub(super) enum DiskTask<M: LiveCacheBackend> {
    Write {
        directory: PathBuf,
        block: HostCacheBlock<M>,
        commit: DiskWriteCommit<M>,
    },
    Read {
        location: DiskLocation,
        representation: CacheRepresentation,
        shapes: [Vec<i32>; 2],
        dtypes: [String; 2],
    },
    #[cfg(test)]
    PauseWrite {
        started: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
        commit: DiskWriteCommit<M>,
        staged: crate::cache::StagedLiveCacheBlock,
    },
}

pub(super) struct DiskWriteCommit<M: LiveCacheBackend> {
    pub(super) state: Weak<Mutex<CacheManagerState<M>>>,
    pub(super) key: CacheIoOperationKey,
    pub(super) reservation_id: u64,
    pub(super) armed: bool,
}

#[derive(Debug, Clone)]
pub(super) struct HostWriteReservation<M: LiveCacheBackend> {
    pub(super) reservation_id: u64,
    pub(super) logical_bytes: u64,
    pub(super) host_capacity: u64,
    pub(super) ticket: DiskTicket<M>,
}

#[derive(Debug, Clone)]
pub(super) struct RetiringHostDemotion {
    pub(super) id: CacheBlockId,
    pub(super) device_bytes: u64,
    pub(super) host_bytes: u64,
}

#[derive(Debug, Clone)]
pub(super) enum DiskResult<M: LiveCacheBackend> {
    Write,
    Read(HostCacheBlock<M>),
}

impl<M: LiveCacheBackend> DiskWriteCommit<M> {
    pub(super) fn complete(
        &self,
        staged: Result<crate::cache::StagedLiveCacheBlock, CacheResidencyError>,
    ) -> CacheTaskResult<DiskResult<M>> {
        let staged = staged.map_err(Arc::new);
        let Some(state) = self.state.upgrade() else {
            return staged.map(|_| DiskResult::Write);
        };
        let mut state = state
            .lock()
            .map_err(|_| Arc::new(CacheResidencyError::ManagerPoisoned))?;
        let generation = state.generation;
        if generation != self.key.generation {
            return staged.map(|_| DiskResult::Write);
        }
        let record = state.blocks.get_mut(&self.key.id);
        let bytes = record.as_ref().map_or(0, |record| record.bytes);
        let result = staged.and_then(|staged| {
            staged
                .publish(generation, record.map(|record| &mut record.physical))
                .map_err(|error| Arc::new(error.into()))
        });
        match &result {
            Ok(true) => {
                state.telemetry.report.transfer_bytes += bytes;
                state.telemetry.report.disk_demotions += 1;
                let activity = state.layer_activity_mut(self.key.id.global_layer);
                activity.transfer_bytes += bytes;
                activity.disk_demotions += 1;
            }
            Ok(false) => {}
            Err(error) => {
                if let Some(record) = state.blocks.get_mut(&self.key.id) {
                    record.physical.fail_io_if_matches(&self.key);
                }
                state.telemetry.report.failures += 1;
                state.layer_activity_mut(self.key.id.global_layer).failures += 1;
                state.background_disk_error = Some(Arc::clone(error));
            }
        }
        update_report_totals(&mut state);
        result.map(|_| DiskResult::Write)
    }
}

impl<M: LiveCacheBackend> Drop for DiskWriteCommit<M> {
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

type RuntimeDiskWorker<M> = CacheIoWorker<DiskTask<M>, CacheTaskResult<DiskResult<M>>>;
type RuntimeDiskSubmission<M> = CacheIoSubmission<DiskTask<M>, CacheTaskResult<DiskResult<M>>>;

#[derive(Debug, Clone)]
pub(super) struct DiskTicket<M: LiveCacheBackend> {
    pub(super) inner: CacheIoTicket<CacheTaskResult<DiskResult<M>>>,
}

impl<M: LiveCacheBackend> std::ops::Deref for DiskTicket<M> {
    type Target = CacheIoTicket<CacheTaskResult<DiskResult<M>>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<M: LiveCacheBackend> DiskTicket<M> {
    pub(super) fn wait(&self) -> Result<DiskResult<M>, CacheResidencyError> {
        self.inner
            .wait()
            .map_err(disk_worker_error)?
            .map_err(Into::into)
    }

    pub(super) fn cancel(&self) -> bool {
        self.inner.cancel()
    }

    pub(super) fn wait_for_task_resources(&self) -> Result<(), CacheResidencyError> {
        self.inner
            .wait_for_task_resources()
            .map_err(disk_worker_error)
    }
}

pub(super) struct DiskSubmission<M: LiveCacheBackend> {
    pub(super) inner: RuntimeDiskSubmission<M>,
    pub(super) ticket: DiskTicket<M>,
    pub(super) write_reservation_id: Option<u64>,
}

impl<M: LiveCacheBackend> DiskSubmission<M> {
    pub(super) fn enqueue(self) -> Result<CacheIoSubmissionOutcome, CacheResidencyError> {
        self.inner.enqueue().map_err(disk_worker_error)
    }
}

#[derive(Debug)]
pub(super) struct DiskWorker<M: LiveCacheBackend> {
    pub(super) inner: RuntimeDiskWorker<M>,
}

impl<M: LiveCacheBackend> DiskWorker<M> {
    pub(super) fn new(capacity: usize) -> Result<Self, CacheResidencyError> {
        Ok(Self {
            inner: RuntimeDiskWorker::new(capacity, "eredu-cache-disk", execute_disk_task, drop)
                .map_err(disk_worker_error)?
                .with_nonblocking_drop(),
        })
    }

    pub(super) fn prepare(
        &self,
        key: CacheIoOperationKey,
        task: DiskTask<M>,
    ) -> Result<DiskSubmission<M>, CacheResidencyError> {
        self.prepare_with_write_reservation(key, task, None)
    }

    pub(super) fn prepare_with_write_reservation(
        &self,
        key: CacheIoOperationKey,
        task: DiskTask<M>,
        write_reservation_id: Option<u64>,
    ) -> Result<DiskSubmission<M>, CacheResidencyError> {
        let mut inner = self.inner.prepare(key, task).map_err(disk_worker_error)?;
        let joined = inner.joined;
        if joined {
            if let Some(DiskTask::Write { commit, .. }) = inner.joined_task_mut() {
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

    pub(super) fn prepare_write(
        &self,
        generation: u64,
        directory: &Path,
        id: &CacheBlockId,
        block: &HostCacheBlock<M>,
        state: Weak<Mutex<CacheManagerState<M>>>,
    ) -> Result<DiskSubmission<M>, CacheResidencyError> {
        let reservation_id = NEXT_HOST_WRITE_RESERVATION_ID.fetch_add(1, Ordering::Relaxed);
        self.prepare_with_write_reservation(
            CacheIoOperationKey {
                generation,
                id: id.clone(),
                kind: CacheIoOperationKind::Write,
            },
            DiskTask::Write {
                directory: directory.to_path_buf(),
                block: block.clone(),
                commit: DiskWriteCommit {
                    state,
                    key: CacheIoOperationKey {
                        generation,
                        id: id.clone(),
                        kind: CacheIoOperationKind::Write,
                    },
                    reservation_id,
                    armed: true,
                },
            },
            Some(reservation_id),
        )
    }

    pub(super) fn prepare_read(
        &self,
        generation: u64,
        id: &CacheBlockId,
        location: &DiskLocation,
        record: &CacheBlockRecord<M>,
    ) -> Result<DiskSubmission<M>, CacheResidencyError> {
        self.prepare(
            CacheIoOperationKey {
                generation,
                id: id.clone(),
                kind: CacheIoOperationKind::Read,
            },
            DiskTask::Read {
                location: location.clone(),
                representation: id.representation,
                shapes: record.shapes.clone(),
                dtypes: record.dtypes.clone(),
            },
        )
    }

    pub(super) fn retire(&self, ticket: &DiskTicket<M>) {
        self.inner.retire(&ticket.inner);
    }
}

pub(super) fn execute_disk_task<M: LiveCacheBackend>(
    task: DiskTask<M>,
) -> Result<CacheTaskResult<DiskResult<M>>, String> {
    let result = match task {
        DiskTask::Write {
            directory,
            block,
            commit,
        } => {
            let staged = catch_unwind(AssertUnwindSafe(|| {
                let views = host_cache_tensors::<M>(&block)?;
                crate::cache::stage_live_cache_block(&directory, &commit.key, views)
                    .map_err(CacheResidencyError::from)
            }))
            .unwrap_or_else(|_| {
                Err(CacheResidencyError::Runtime(
                    "live cache disk worker operation panicked".into(),
                ))
            });
            drop(block);
            commit.complete(staged)
        }
        DiskTask::Read {
            location,
            representation,
            shapes,
            dtypes,
        } => load_host_cache_block_direct::<M>(&location, representation, &shapes, &dtypes)
            .map(DiskResult::Read)
            .map_err(Arc::new),
        #[cfg(test)]
        DiskTask::PauseWrite {
            started,
            release,
            commit,
            staged,
        } => {
            let _ = started.send(());
            let _ = release.recv();
            commit.complete(Ok(staged))
        }
    };
    Ok(result)
}

pub(super) fn disk_worker_error(error: CacheIoWorkerError) -> CacheResidencyError {
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

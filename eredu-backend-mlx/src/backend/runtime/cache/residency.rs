//! Block-addressable residency for mutable attention state.
//!
//! This module is deliberately independent from weight residency. Attention
//! blocks are mutable activation state until sealed, while checkpoint weights
//! are immutable inputs with a different ownership and persistence model.

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    fs::{self, File},
    panic::{catch_unwind, AssertUnwindSafe},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc, Arc, Condvar, Mutex, MutexGuard, OnceLock, Weak,
    },
    thread::{self, JoinHandle},
    time::Instant,
};

use memmap2::{Mmap, MmapOptions};
use safemlx::{
    host_transfer_capacity_upper_bound,
    transforms::{async_eval_with_event, eval},
    Array, Device, DeviceType, Dtype, Event, HostTransferBuffer, HostTransferPolicy,
    ImmutableHostTransferBuffer, Stream,
};
use safetensors::tensor::{serialize_to_file, Dtype as StoredDtype, TensorView};
use sha2::{Digest, Sha256};

use eredu_core::{
    cache::{
        prompt_cache_token_fingerprint, validate_prompt_cache_model_identity, CacheBlockId,
        CachePolicyError, CacheRankIdentity, CacheRepresentation, CacheTier, PromptCacheBlock,
        PromptCacheDescriptor, PromptCacheError, PromptCacheManifest, PromptCacheModelIdentity,
        PromptCacheOptions, PromptCacheStateTensor, StateTensorOwner, StateTensorRole,
        PROMPT_CACHE_SCHEMA_VERSION,
    },
    residency::CacheEvictionPolicy,
};
use eredu_runtime::{
    finalize_prompt_cache_shard, hash_prompt_cache_shard_payload, inspect_prompt_cache,
    resolve_prompt_cache_root, safe_prompt_cache_shard_path, CacheBlockLifecycle,
    CacheBlockStorage, CacheHostDemotionOperation, CacheIoExecutionStateError, CacheIoOperation,
    CacheIoOperationKey, CacheIoOperationKind, CacheIoSubmission as RuntimeCacheIoSubmission,
    CacheIoSubmissionOutcome, CacheIoTicket as RuntimeCacheIoTicket,
    CacheIoWorker as RuntimeCacheIoWorker, CacheIoWorkerError, CacheLayerResidencyStats,
    CacheLifecycleError, CachePoolError, CachePoolMembership, CachePoolReservation,
    CachePoolResource, CachePoolUsage, CacheResidencyConfigurationError, CacheResidencyPool,
    CacheResidencyReport, CacheResidencyTelemetry, CacheStorageError, CacheStoragePhase,
    LiveCacheBlockPublication, LiveCacheDiskPolicy, LiveCachePublicationError, MutableCacheTail,
    PagedCacheOptions, PromptCachePersistenceError, PromptCachePublication,
};

pub const PAGED_CACHE_PREFETCH_BLOCKS: usize = 2;
static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_HOST_WRITE_RESERVATION_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_HOST_DEMOTION_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub enum CacheBlockArrays {
    KeyValue { keys: Array, values: Array },
    CompressedLatentRotary { latent: Array, rotary_key: Array },
}

impl CacheBlockArrays {
    pub fn representation(&self) -> CacheRepresentation {
        match self {
            Self::KeyValue { .. } => CacheRepresentation::KeyValue,
            Self::CompressedLatentRotary { .. } => CacheRepresentation::CompressedLatentRotary,
        }
    }

    fn arrays(&self) -> [&Array; 2] {
        match self {
            Self::KeyValue { keys, values } => [keys, values],
            Self::CompressedLatentRotary { latent, rotary_key } => [latent, rotary_key],
        }
    }

    fn bytes(&self) -> u64 {
        self.arrays()
            .iter()
            .map(|array| array.nbytes() as u64)
            .sum()
    }

    fn shapes(&self) -> [Vec<i32>; 2] {
        let arrays = self.arrays();
        [arrays[0].shape().to_vec(), arrays[1].shape().to_vec()]
    }

    fn dtypes(&self) -> [String; 2] {
        let arrays = self.arrays();
        [dtype_name(arrays[0].dtype()), dtype_name(arrays[1].dtype())]
    }
}

fn host_cache_capacity_upper_bound(arrays: &CacheBlockArrays) -> Result<u64, CacheResidencyError> {
    arrays.arrays().into_iter().try_fold(0u64, |total, array| {
        let capacity =
            host_transfer_capacity_upper_bound(array.nbytes(), HostTransferPolicy::Transfer)
                .map_err(|source| transfer_error("query host cache capacity bound", source))?;
        let capacity = u64::try_from(capacity).map_err(|_| {
            CacheResidencyError::Runtime(
                "host cache capacity bound exceeds the u64 accounting range".into(),
            )
        })?;
        total.checked_add(capacity).ok_or_else(|| {
            CacheResidencyError::Runtime("host cache capacity bound overflowed".into())
        })
    })
}

fn host_cache_layout_capacity_upper_bound(
    shapes: &[Vec<i32>; 2],
    dtypes: &[String; 2],
) -> Result<u64, CacheResidencyError> {
    shapes
        .iter()
        .zip(dtypes)
        .try_fold(0u64, |total, (shape, dtype)| {
            let element_bytes = match dtype.as_str() {
                "Bool" | "Uint8" | "Int8" => 1u64,
                "Uint16" | "Int16" | "Float16" | "Bfloat16" => 2,
                "Uint32" | "Int32" | "Float32" => 4,
                "Uint64" | "Int64" | "Float64" | "Complex64" => 8,
                other => {
                    return Err(CacheResidencyError::Runtime(format!(
                        "unsupported host cache dtype {other} in capacity admission"
                    )))
                }
            };
            let logical_bytes = shape.iter().try_fold(element_bytes, |bytes, dimension| {
                let dimension = u64::try_from(*dimension).map_err(|_| {
                    CacheResidencyError::Runtime(
                        "host cache shape contains a negative dimension".into(),
                    )
                })?;
                bytes.checked_mul(dimension).ok_or_else(|| {
                    CacheResidencyError::Runtime("host cache logical byte length overflowed".into())
                })
            })?;
            let logical_bytes = usize::try_from(logical_bytes).map_err(|_| {
                CacheResidencyError::Runtime(
                    "host cache logical byte length exceeds the addressable range".into(),
                )
            })?;
            let capacity =
                host_transfer_capacity_upper_bound(logical_bytes, HostTransferPolicy::Transfer)
                    .map_err(|source| transfer_error("query host cache capacity bound", source))?;
            let capacity = u64::try_from(capacity).map_err(|_| {
                CacheResidencyError::Runtime(
                    "host cache capacity bound exceeds the u64 accounting range".into(),
                )
            })?;
            total.checked_add(capacity).ok_or_else(|| {
                CacheResidencyError::Runtime("host cache capacity bound overflowed".into())
            })
        })
}

#[derive(Debug, Clone)]
enum HostCacheBlock {
    KeyValue {
        keys: Arc<ImmutableHostTransferBuffer>,
        values: Arc<ImmutableHostTransferBuffer>,
    },
    CompressedLatentRotary {
        latent: Arc<ImmutableHostTransferBuffer>,
        rotary_key: Arc<ImmutableHostTransferBuffer>,
    },
}

impl HostCacheBlock {
    fn from_device_arrays(
        arrays: &CacheBlockArrays,
        stream: &Stream,
    ) -> Result<Self, CacheResidencyError> {
        let [first, second] = arrays.arrays();
        let first =
            HostTransferBuffer::copy_from_array(first, HostTransferPolicy::Transfer, stream)
                .map_err(|source| {
                    transfer_error("submit first cache block host transfer", source)
                })?;
        let second =
            HostTransferBuffer::copy_from_array(second, HostTransferPolicy::Transfer, stream)
                .map_err(|source| {
                    transfer_error("submit second cache block host transfer", source)
                })?;
        let first = Arc::new(
            first
                .synchronize()
                .map_err(|source| {
                    transfer_error("complete first cache block host transfer", source)
                })?
                .freeze(),
        );
        let second = Arc::new(
            second
                .synchronize()
                .map_err(|source| {
                    transfer_error("complete second cache block host transfer", source)
                })?
                .freeze(),
        );
        Ok(match arrays {
            CacheBlockArrays::KeyValue { .. } => Self::KeyValue {
                keys: first,
                values: second,
            },
            CacheBlockArrays::CompressedLatentRotary { .. } => Self::CompressedLatentRotary {
                latent: first,
                rotary_key: second,
            },
        })
    }

    fn from_buffers(
        representation: CacheRepresentation,
        first: ImmutableHostTransferBuffer,
        second: ImmutableHostTransferBuffer,
    ) -> Self {
        let first = Arc::new(first);
        let second = Arc::new(second);
        match representation {
            CacheRepresentation::KeyValue => Self::KeyValue {
                keys: first,
                values: second,
            },
            CacheRepresentation::CompressedLatentRotary => Self::CompressedLatentRotary {
                latent: first,
                rotary_key: second,
            },
        }
    }

    fn representation(&self) -> CacheRepresentation {
        match self {
            Self::KeyValue { .. } => CacheRepresentation::KeyValue,
            Self::CompressedLatentRotary { .. } => CacheRepresentation::CompressedLatentRotary,
        }
    }

    fn buffers(&self) -> [&ImmutableHostTransferBuffer; 2] {
        match self {
            Self::KeyValue { keys, values } => [keys, values],
            Self::CompressedLatentRotary { latent, rotary_key } => [latent, rotary_key],
        }
    }

    fn shapes(&self) -> Result<[Vec<i32>; 2], CacheResidencyError> {
        let [first, second] = self.buffers();
        Ok([
            first
                .shape()
                .map_err(|source| transfer_error("inspect first host cache shape", source))?,
            second
                .shape()
                .map_err(|source| transfer_error("inspect second host cache shape", source))?,
        ])
    }

    fn dtypes(&self) -> Result<[String; 2], CacheResidencyError> {
        let [first, second] = self.buffers();
        Ok([
            dtype_name(
                first
                    .dtype()
                    .map_err(|source| transfer_error("inspect first host cache dtype", source))?,
            ),
            dtype_name(
                second
                    .dtype()
                    .map_err(|source| transfer_error("inspect second host cache dtype", source))?,
            ),
        ])
    }

    fn bytes(&self) -> Result<u64, CacheResidencyError> {
        self.buffers().into_iter().try_fold(0u64, |total, buffer| {
            let bytes = buffer
                .nbytes()
                .map_err(|source| transfer_error("inspect host cache byte length", source))?;
            Ok(total.saturating_add(bytes as u64))
        })
    }

    fn capacity(&self) -> Result<u64, CacheResidencyError> {
        self.buffers().into_iter().try_fold(0u64, |total, buffer| {
            let capacity = buffer
                .capacity()
                .map_err(|source| transfer_error("inspect host cache capacity", source))?;
            let capacity = u64::try_from(capacity).map_err(|_| {
                CacheResidencyError::Runtime(
                    "host cache capacity exceeds the u64 accounting range".into(),
                )
            })?;
            total.checked_add(capacity).ok_or_else(|| {
                CacheResidencyError::Runtime("host cache capacity total overflowed".into())
            })
        })
    }

    fn copy_to_device(
        &self,
        stream: &Stream,
    ) -> Result<(CacheBlockArrays, Vec<Event>), CacheResidencyError> {
        let [first, second] = self.buffers();
        let first = first
            .copy_to_array(stream)
            .map_err(|source| transfer_error("submit first cache block promotion", source))?;
        let second = second
            .copy_to_array(stream)
            .map_err(|source| transfer_error("submit second cache block promotion", source))?;
        let (first, first_completion) = first.into_parts();
        let (second, second_completion) = second.into_parts();
        let arrays = match self {
            Self::KeyValue { .. } => CacheBlockArrays::KeyValue {
                keys: first,
                values: second,
            },
            Self::CompressedLatentRotary { .. } => CacheBlockArrays::CompressedLatentRotary {
                latent: first,
                rotary_key: second,
            },
        };
        Ok((arrays, vec![first_completion, second_completion]))
    }
}

#[derive(Debug, Default)]
struct HostDemotionCompletion {
    result: Mutex<Option<Result<HostCacheBlock, String>>>,
    ready: Condvar,
}

impl HostDemotionCompletion {
    fn finish(&self, result: Result<HostCacheBlock, CacheResidencyError>) {
        if let Ok(mut slot) = self.result.lock() {
            if slot.is_none() {
                *slot = Some(result.map_err(|error| error.to_string()));
                self.ready.notify_all();
            }
        }
    }

    fn wait(&self) -> Result<HostCacheBlock, CacheResidencyError> {
        let mut slot = self
            .result
            .lock()
            .map_err(|_| CacheResidencyError::ManagerPoisoned)?;
        while slot.is_none() {
            slot = self
                .ready
                .wait(slot)
                .map_err(|_| CacheResidencyError::ManagerPoisoned)?;
        }
        match slot.as_ref().expect("host demotion completion is ready") {
            Ok(block) => Ok(block.clone()),
            Err(error) => Err(CacheResidencyError::Runtime(error.clone())),
        }
    }
}

#[derive(Debug, Clone)]
struct HostDemotionTicket {
    operation_id: u64,
    id: CacheBlockId,
    reserved_host_bytes: u64,
    completion: Arc<HostDemotionCompletion>,
}

impl HostDemotionTicket {
    fn wait(&self) -> Result<HostCacheBlock, CacheResidencyError> {
        self.completion.wait()
    }
}

impl CacheHostDemotionOperation for HostDemotionTicket {
    fn block_id(&self) -> &CacheBlockId {
        &self.id
    }

    fn operation_id(&self) -> u64 {
        self.operation_id
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct CacheTransferDevice {
    device_type: DeviceType,
    index: i32,
}

impl CacheTransferDevice {
    const CPU: Self = Self {
        device_type: DeviceType::Cpu,
        index: 0,
    };

    fn from_stream(stream: &Stream) -> Result<Self, CacheResidencyError> {
        let device = stream
            .get_device()
            .map_err(|source| transfer_error("inspect cache transfer device", source))?;
        Ok(Self {
            device_type: device
                .get_type()
                .map_err(|source| transfer_error("inspect cache transfer device type", source))?,
            index: device
                .get_index()
                .map_err(|source| transfer_error("inspect cache transfer device index", source))?,
        })
    }
}

enum HostDemotionRequest {
    Demote {
        arrays: CacheBlockArrays,
        device: CacheTransferDevice,
        completion: Arc<HostDemotionCompletion>,
    },
    Stop,
}

struct HostDemotionWorker {
    sender: mpsc::Sender<HostDemotionRequest>,
    handle: Mutex<Option<JoinHandle<()>>>,
}

impl std::fmt::Debug for HostDemotionWorker {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("HostDemotionWorker").finish()
    }
}

impl HostDemotionWorker {
    fn new() -> Result<Self, CacheResidencyError> {
        let (sender, receiver) = mpsc::channel();
        let handle = thread::Builder::new()
            .name("eredu-mlx-cache-host-demotion".into())
            .spawn(move || {
                while let Ok(request) = receiver.recv() {
                    match request {
                        HostDemotionRequest::Demote {
                            arrays,
                            device,
                            completion,
                        } => {
                            let result = catch_unwind(AssertUnwindSafe(|| {
                                let device = Device::new(device.device_type, device.index);
                                let stream = Stream::new_with_device(&device);
                                HostCacheBlock::from_device_arrays(&arrays, &stream)
                            }))
                            .unwrap_or_else(|_| {
                                Err(CacheResidencyError::Runtime(
                                    "cache host demotion worker operation panicked".into(),
                                ))
                            });
                            completion.finish(result);
                        }
                        HostDemotionRequest::Stop => break,
                    }
                }
            })
            .map_err(|source| CacheResidencyError::Io {
                action: "start cache host demotion worker",
                path: PathBuf::from("eredu-mlx-cache-host-demotion"),
                source,
            })?;
        Ok(Self {
            sender,
            handle: Mutex::new(Some(handle)),
        })
    }

    fn submit(
        &self,
        id: &CacheBlockId,
        arrays: CacheBlockArrays,
        device: CacheTransferDevice,
        reserved_host_bytes: u64,
    ) -> Result<HostDemotionTicket, CacheResidencyError> {
        let completion = Arc::new(HostDemotionCompletion::default());
        let ticket = HostDemotionTicket {
            operation_id: NEXT_HOST_DEMOTION_ID.fetch_add(1, Ordering::Relaxed),
            id: id.clone(),
            reserved_host_bytes,
            completion: Arc::clone(&completion),
        };
        self.sender
            .send(HostDemotionRequest::Demote {
                arrays,
                device,
                completion,
            })
            .map_err(|_| {
                CacheResidencyError::Runtime("cache host demotion worker stopped".into())
            })?;
        Ok(ticket)
    }
}

impl Drop for HostDemotionWorker {
    fn drop(&mut self) {
        let _ = self.sender.send(HostDemotionRequest::Stop);
        if let Ok(handle) = self.handle.get_mut() {
            if let Some(handle) = handle.take() {
                let _ = handle.join();
            }
        }
    }
}

fn transfer_error(
    operation: &'static str,
    source: safemlx::error::Exception,
) -> CacheResidencyError {
    CacheResidencyError::Runtime(format!("{operation}: {source}"))
}

#[derive(Debug, Clone)]
struct DiskLocation {
    path: PathBuf,
    first_name: String,
    second_name: String,
    persistent: bool,
    mapped: Option<Arc<Mmap>>,
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

#[derive(Debug, Clone)]
struct MlxCacheIoOperation {
    ticket: DiskTicket,
    reserved_host_bytes: Option<u64>,
}

impl CacheIoOperation for MlxCacheIoOperation {
    fn key(&self) -> &CacheIoOperationKey {
        &self.ticket.key
    }
}

type MlxCacheBlockStorage = CacheBlockStorage<
    CacheBlockArrays,
    HostCacheBlock,
    DiskLocation,
    HostDemotionTicket,
    MlxCacheIoOperation,
>;

#[derive(Debug, Clone)]
struct CacheBlockRecord {
    physical: MlxCacheBlockStorage,
    bytes: u64,
    shapes: [Vec<i32>; 2],
    dtypes: [String; 2],
    imported: bool,
}

impl CacheBlockRecord {
    fn tier(&self) -> CacheTier {
        self.physical.tier()
    }

    fn disk(&self) -> Option<&DiskLocation> {
        self.physical.backing()
    }

    fn pending_disk(&self) -> Option<&MlxCacheIoOperation> {
        self.physical.io()
    }

    #[cfg(test)]
    fn device_arrays(&self) -> Option<&CacheBlockArrays> {
        self.physical.device_resource()
    }

    fn host_block(&self) -> Option<&HostCacheBlock> {
        self.physical.host_resource()
    }

    fn host_demotion_ticket(&self) -> Option<&HostDemotionTicket> {
        self.physical.host_demotion()
    }
}

#[derive(Debug)]
struct CacheManagerState {
    pool: CacheResidencyPool,
    pool_manager_id: u64,
    generation: u64,
    background_disk_error: Option<String>,
    lifecycle: CacheBlockLifecycle,
    blocks: BTreeMap<CacheBlockId, CacheBlockRecord>,
    host_write_reservations: HashMap<CacheIoOperationKey, HostWriteReservation>,
    retiring_host_demotions: HashMap<u64, RetiringHostDemotion>,
    retiring_disk_reads: HashMap<CacheIoOperationKey, (usize, u64)>,
    transfer_device: Option<CacheTransferDevice>,
    telemetry: CacheResidencyTelemetry,
    recent_device_blocks: usize,
    device_budget_bytes: u64,
    host_budget_bytes: u64,
    disk_budget_bytes: Option<u64>,
}

impl CacheManagerState {
    fn layer_activity_mut(&mut self, global_layer: usize) -> &mut CacheLayerResidencyStats {
        self.telemetry.layer_activity_mut(global_layer)
    }
}

#[derive(Debug)]
struct CacheResidencyManagerInner {
    options: PagedCacheOptions,
    state: Arc<Mutex<CacheManagerState>>,
    host_demotion_worker: Arc<HostDemotionWorker>,
    disk_worker: Option<Arc<DiskWorker>>,
    pool_membership: Arc<CachePoolMembership>,
}

/// Shared architecture-independent manager enforcing budgets across all layers.
///
/// Clones retain one compact shared inner allocation containing the catalog,
/// workers, options, and process-pool membership.
#[derive(Debug, Clone)]
pub struct CacheResidencyManager {
    session_id: u64,
    inner: Arc<CacheResidencyManagerInner>,
}

enum HostDemotionProgress {
    Retry,
    Freed,
    Pending(DiskTicket),
}

enum PendingCacheOperation {
    Disk(DiskTicket),
    HostDemotion(HostDemotionTicket),
}

impl CacheResidencyManager {
    /// Creates an empty manager with globally shared finite limits.
    pub fn new(options: PagedCacheOptions) -> Result<Self, CacheResidencyError> {
        if let LiveCacheDiskPolicy::Enabled { directory, .. } = options.live_disk_policy() {
            fs::create_dir_all(directory).map_err(|source| CacheResidencyError::Io {
                action: "create live cache directory",
                path: directory.clone(),
                source,
            })?;
        }
        let queue_capacity = match options.live_disk_policy() {
            LiveCacheDiskPolicy::Disabled => 0,
            LiveCacheDiskPolicy::Enabled { queue_capacity, .. } => *queue_capacity,
        };
        let effective_queue_capacity = queue_capacity.max(1);
        let telemetry = CacheResidencyTelemetry::new(effective_queue_capacity);
        let disk_worker = Some(Arc::new(DiskWorker::new(effective_queue_capacity)?));
        let host_demotion_worker = Arc::new(HostDemotionWorker::new()?);
        let recent_device_blocks = options.recent_device_blocks();
        let device_budget_bytes = options.device_budget_bytes();
        let host_budget_bytes = options.host_budget_bytes();
        let disk_budget_bytes = match options.live_disk_policy() {
            LiveCacheDiskPolicy::Disabled => None,
            LiveCacheDiskPolicy::Enabled { budget_bytes, .. } => Some(*budget_bytes),
        };
        let session_id = NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed);
        let pool = match options.pool().cloned() {
            Some(pool) => pool,
            None => options.create_pool()?,
        };
        let pool_membership = Arc::new(pool.register_manager(session_id)?);
        Ok(Self {
            session_id,
            inner: Arc::new(CacheResidencyManagerInner {
                options,
                state: Arc::new(Mutex::new(CacheManagerState {
                    pool,
                    pool_manager_id: session_id,
                    generation: 0,
                    background_disk_error: None,
                    lifecycle: CacheBlockLifecycle::new(),
                    blocks: BTreeMap::new(),
                    host_write_reservations: HashMap::new(),
                    retiring_host_demotions: HashMap::new(),
                    retiring_disk_reads: HashMap::new(),
                    transfer_device: None,
                    telemetry,
                    recent_device_blocks,
                    device_budget_bytes,
                    host_budget_bytes,
                    disk_budget_bytes,
                })),
                host_demotion_worker,
                disk_worker,
                pool_membership,
            }),
        })
    }

    /// Returns the live cache identity included in every block id.
    pub const fn session_id(&self) -> u64 {
        self.session_id
    }

    /// Returns validated paged-cache options.
    pub fn options(&self) -> &PagedCacheOptions {
        &self.inner.options
    }

    /// Returns the aggregate process pool accounting for this manager.
    pub fn pool(&self) -> &CacheResidencyPool {
        self.inner.pool_membership.pool()
    }

    fn lock(&self) -> Result<MutexGuard<'_, CacheManagerState>, CacheResidencyError> {
        self.inner
            .state
            .lock()
            .map_err(|_| CacheResidencyError::ManagerPoisoned)
    }

    pub fn bind_transfer_device(&self, stream: &Stream) -> Result<(), CacheResidencyError> {
        let device = CacheTransferDevice::from_stream(stream)?;
        let mut state = self.lock()?;
        match state.transfer_device {
            Some(bound) if bound != device => Err(CacheResidencyError::Runtime(format!(
                "paged cache is bound to {:?} device {} but received {:?} device {}",
                bound.device_type, bound.index, device.device_type, device.index
            ))),
            Some(_) => Ok(()),
            None => {
                state.transfer_device = Some(device);
                Ok(())
            }
        }
    }

    pub fn set_tail_state(
        &self,
        layer: usize,
        bytes: u64,
        end: i64,
    ) -> Result<(), CacheResidencyError> {
        let mut state = self.lock()?;
        let previous = state
            .lifecycle
            .set_tail(layer, MutableCacheTail { bytes, end });
        let allocated = previous.is_none_or(|tail| tail.bytes == 0) && bytes > 0;
        if allocated {
            state.telemetry.report.tail_allocations += 1;
        }
        drop(state);
        if let Err(error) = self.rebalance(None, false) {
            let mut state = self.lock()?;
            state.lifecycle.restore_tail(layer, previous);
            if allocated {
                state.telemetry.report.tail_allocations =
                    state.telemetry.report.tail_allocations.saturating_sub(1);
            }
            update_report_totals(&mut state);
            return Err(error);
        }
        Ok(())
    }

    pub fn seal_block(
        &self,
        global_layer: usize,
        start: i64,
        end: i64,
        rank: Option<CacheRankIdentity>,
        arrays: CacheBlockArrays,
        protected_prefix: bool,
    ) -> Result<CacheBlockId, CacheResidencyError> {
        if start < 0 || end <= start {
            return Err(CacheResidencyError::InvalidTokenRange { start, end });
        }
        let representation = arrays.representation();
        validate_block_arrays(&arrays, end - start)?;
        eval(arrays.arrays()).map_err(|source| CacheResidencyError::Runtime(source.to_string()))?;
        let id = CacheBlockId {
            session_id: self.session_id,
            global_layer,
            representation,
            start,
            end,
            rank,
        };
        let mut state = self.lock()?;
        if state.blocks.contains_key(&id) {
            return Err(CacheLifecycleError::DuplicateBlock(id).into());
        }
        let bytes = arrays.bytes();
        let record = CacheBlockRecord {
            shapes: arrays.shapes(),
            dtypes: arrays.dtypes(),
            physical: MlxCacheBlockStorage::device(id.clone(), arrays, None),
            bytes,
            imported: false,
        };
        state
            .lifecycle
            .insert(id.clone(), protected_prefix)
            .map_err(CacheResidencyError::from)?;
        state.blocks.insert(id.clone(), record);
        drop(state);
        if let Err(error) = self.rebalance(Some(&id), false) {
            let mut state = self.lock()?;
            if let Some(record) = state.blocks.remove(&id) {
                state.lifecycle.remove(&id)?;
                cancel_record_operation(&record, &mut state.telemetry.report);
                remove_ephemeral_file(&record);
            }
            update_report_totals(&mut state);
            return Err(error);
        }
        let mut state = self.lock()?;
        state.telemetry.report.block_seals += 1;
        Ok(id)
    }

    pub fn layer_block_ids(
        &self,
        layer: usize,
        representation: CacheRepresentation,
        visible_start: i64,
        visible_end: i64,
        prefix_tokens: i64,
    ) -> Result<Vec<CacheBlockId>, CacheResidencyError> {
        let state = self.lock()?;
        Ok(state
            .blocks
            .keys()
            .filter(|id| {
                id.global_layer == layer
                    && id.representation == representation
                    && id.start < visible_end
                    && (id.end > visible_start || id.start < prefix_tokens)
            })
            .cloned()
            .collect())
    }

    pub fn layer_end(
        &self,
        layer: usize,
        representation: CacheRepresentation,
    ) -> Result<i64, CacheResidencyError> {
        let state = self.lock()?;
        Ok(state
            .blocks
            .keys()
            .filter(|id| id.global_layer == layer && id.representation == representation)
            .map(|id| id.end)
            .max()
            .unwrap_or(0))
    }

    pub fn remove_block(&self, id: &CacheBlockId) -> Result<(), CacheResidencyError> {
        let mut state = self.lock()?;
        if !state.blocks.contains_key(id) {
            return Err(CacheLifecycleError::MissingBlock(id.clone()).into());
        }
        state.lifecycle.remove(id)?;
        let tickets = advance_generation_locked(&mut state);
        let record = state
            .blocks
            .remove(id)
            .expect("validated cache block still present");
        remove_ephemeral_file(&record);
        update_report_totals(&mut state);
        drop(state);
        self.retire_tickets(&tickets)?;
        Ok(())
    }

    pub fn truncate_layer_transaction(
        &self,
        global_layer: usize,
        representation: CacheRepresentation,
        end: i64,
        replacement: Option<(CacheBlockLease, CacheBlockArrays)>,
        protected_prefix_tokens: i64,
    ) -> Result<(), CacheResidencyError> {
        if end < 0 {
            return Err(CacheResidencyError::InvalidTokenRange { start: 0, end });
        }
        if let Some((lease, arrays)) = &replacement {
            let old_id = lease.id();
            if old_id.global_layer != global_layer
                || old_id.representation != representation
                || old_id.start >= end
                || old_id.end <= end
                || arrays.representation() != representation
            {
                return Err(CacheResidencyError::ArrayMismatch(
                    "trailing cache replacement does not match the truncated layer".into(),
                ));
            }
            validate_block_arrays(arrays, end - old_id.start)?;
            eval(arrays.arrays())
                .map_err(|source| CacheResidencyError::Runtime(source.to_string()))?;
        }

        let mut state = self.lock()?;
        if let Some(error) = state.background_disk_error.take() {
            return Err(CacheResidencyError::Runtime(format!(
                "background cache disk write failed: {error}"
            )));
        }
        let affected = state
            .blocks
            .keys()
            .filter(|id| {
                id.global_layer == global_layer
                    && id.representation == representation
                    && id.end > end
            })
            .cloned()
            .collect::<Vec<_>>();
        let crossing = affected.iter().find(|id| id.start < end);
        match (crossing, replacement.as_ref()) {
            (Some(crossing), Some((lease, _))) if crossing == lease.id() => {}
            (None, None) => {}
            _ => {
                return Err(CacheResidencyError::ArrayMismatch(
                    "trailing cache replacement does not match the block crossing the truncation boundary"
                        .into(),
                ))
            }
        }
        for id in &affected {
            let record = state
                .blocks
                .get(id)
                .ok_or_else(|| CacheLifecycleError::MissingBlock(id.clone()))?;
            let owned_replacement_lease = replacement
                .as_ref()
                .is_some_and(|(lease, _)| lease.id() == id);
            if owned_replacement_lease && record.tier() != CacheTier::Device {
                return Err(CacheResidencyError::Runtime(
                    "truncated cache replacement lease is not device resident".into(),
                ));
            }
        }

        let replacement_id = replacement.as_ref().map(|(lease, _)| CacheBlockId {
            session_id: self.session_id,
            global_layer,
            representation,
            start: lease.id().start,
            end,
            rank: lease.id().rank,
        });
        if let Some(id) = &replacement_id {
            if state.blocks.contains_key(id) && !affected.contains(id) {
                return Err(CacheLifecycleError::DuplicateBlock(id.clone()).into());
            }
        }

        let removals = affected
            .iter()
            .map(|id| {
                let owned_replacement_lease = replacement
                    .as_ref()
                    .is_some_and(|(lease, _)| lease.id() == id);
                (id.clone(), usize::from(owned_replacement_lease))
            })
            .collect::<Vec<_>>();
        state.lifecycle.replace(
            &removals,
            replacement_id
                .clone()
                .map(|id| (id, end <= protected_prefix_tokens)),
            global_layer,
            MutableCacheTail { bytes: 0, end },
        )?;

        let tickets = advance_generation_locked(&mut state);
        let mut removed = Vec::with_capacity(affected.len());
        for id in &affected {
            if let Some(record) = state.blocks.remove(id) {
                removed.push(record);
            }
        }
        if let Some((mut lease, arrays)) = replacement {
            let old_id = lease.id().clone();
            lease.released = true;
            let id = replacement_id.expect("validated replacement id is available");
            debug_assert_eq!(id.start, old_id.start);
            let record = CacheBlockRecord {
                shapes: arrays.shapes(),
                dtypes: arrays.dtypes(),
                bytes: arrays.bytes(),
                physical: MlxCacheBlockStorage::device(id.clone(), arrays, None),
                imported: false,
            };
            let previous = state.blocks.insert(id, record);
            debug_assert!(previous.is_none());
            state.telemetry.report.block_seals += 1;
        }
        update_report_totals(&mut state);
        drop(state);

        for record in &removed {
            remove_ephemeral_file(record);
        }
        self.retire_tickets(&tickets)?;
        Ok(())
    }

    pub fn lease_block(
        &self,
        id: &CacheBlockId,
        stream: &Stream,
    ) -> Result<CacheBlockLease, CacheResidencyError> {
        let lease = self.prepare_block_transfer(id, stream)?;
        lease.wait_on(stream)?;
        Ok(lease)
    }

    /// Creates a fixed two-block promotion window on a dedicated stream for
    /// the execution stream's device.
    pub fn prefetch_blocks(
        &self,
        ids: Vec<CacheBlockId>,
        execution_stream: &Stream,
    ) -> Result<CacheBlockPrefetch, CacheResidencyError> {
        CacheBlockPrefetch::new(self.clone(), ids, execution_stream)
    }

    fn prepare_block_transfer(
        &self,
        id: &CacheBlockId,
        transfer_stream: &Stream,
    ) -> Result<CacheBlockLease, CacheResidencyError> {
        self.bind_transfer_device(transfer_stream)?;
        let started = Instant::now();
        let mut loaded_from_disk = false;
        loop {
            let mut state = self.lock()?;
            let generation = state.generation;
            let physical = state
                .blocks
                .get(id)
                .ok_or_else(|| CacheLifecycleError::MissingBlock(id.clone()))?
                .physical
                .clone();

            match physical.phase() {
                CacheStoragePhase::DemotingToHost => {
                    let ticket = physical
                        .host_demotion()
                        .expect("demoting phase owns its exact completion")
                        .clone();
                    drop(state);
                    self.finish_device_demotion(&ticket)?;
                }
                CacheStoragePhase::DiskReady | CacheStoragePhase::DiskReading => {
                    let location = physical
                        .backing()
                        .expect("disk phase owns its backing")
                        .clone();
                    let worker = self.inner.disk_worker.as_ref().ok_or_else(|| {
                        CacheResidencyError::Runtime("cache disk worker is unavailable".into())
                    })?;
                    let (ticket, submission, joined, _transfer_reservation, reserved_host_bytes) =
                        match physical.phase() {
                            CacheStoragePhase::DiskReading => {
                                let pending = physical
                                    .io()
                                    .expect("disk-reading phase owns its exact completion");
                                (
                                    pending.ticket.clone(),
                                    None,
                                    true,
                                    None,
                                    pending
                                        .reserved_host_bytes
                                        .expect("disk-read completion owns its host reservation"),
                                )
                            }
                            CacheStoragePhase::DiskReady => {
                                let record = state
                                    .blocks
                                    .get(id)
                                    .ok_or_else(|| CacheLifecycleError::MissingBlock(id.clone()))?;
                                let bytes = record.bytes;
                                let reserved_host_bytes = host_cache_layout_capacity_upper_bound(
                                    &record.shapes,
                                    &record.dtypes,
                                )?;
                                let required_host_bytes = state
                                    .telemetry
                                    .report
                                    .current_host_bytes
                                    .saturating_add(reserved_host_bytes);
                                if required_host_bytes > self.options().host_budget_bytes() {
                                    let candidate = eviction_candidate(
                                        &state,
                                        CacheTier::Host,
                                        Some(id),
                                        0,
                                        self.options().eviction_policy(),
                                    );
                                    drop(state);
                                    if let Some(candidate) = candidate {
                                        match self.begin_host_demotion(&candidate)? {
                                            HostDemotionProgress::Retry
                                            | HostDemotionProgress::Freed => continue,
                                            HostDemotionProgress::Pending(ticket) => {
                                                self.wait_for_host_release(&ticket)?;
                                                continue;
                                            }
                                        }
                                    }
                                    return Err(CacheResidencyError::BudgetExceeded {
                                        tier: CacheTier::Host,
                                        required: required_host_bytes,
                                        budget: self.options().host_budget_bytes(),
                                    });
                                }
                                let host_admission = state.pool.reserve(CachePoolUsage {
                                    host_bytes: reserved_host_bytes,
                                    ..CachePoolUsage::default()
                                })?;
                                let transfer_reservation =
                                    Some(state.pool.reserve_transfer(bytes)?);
                                let submission = worker.prepare_read(
                                    generation,
                                    id,
                                    &location,
                                    id.representation,
                                )?;
                                let ticket = submission.ticket.clone();
                                let record = state
                                    .blocks
                                    .get_mut(id)
                                    .ok_or_else(|| CacheLifecycleError::MissingBlock(id.clone()))?;
                                record.physical.begin_read(MlxCacheIoOperation {
                                    ticket: ticket.clone(),
                                    reserved_host_bytes: Some(reserved_host_bytes),
                                })?;
                                update_report_totals(&mut state);
                                drop(host_admission);
                                (
                                    ticket,
                                    Some(submission),
                                    false,
                                    transfer_reservation,
                                    reserved_host_bytes,
                                )
                            }
                            _ => unreachable!("disk phase was matched above"),
                        };
                    drop(state);

                    let outcome = match submission {
                        Some(submission) => Some(submission.enqueue()?),
                        None => None,
                    };
                    let result = ticket.wait();
                    let mut state = self.lock()?;
                    if joined || outcome.as_ref().is_some_and(|outcome| outcome.joined) {
                        state.telemetry.report.in_flight_waits += 1;
                        state.layer_activity_mut(id.global_layer).in_flight_waits += 1;
                    }
                    if let Some(outcome) = &outcome {
                        state.telemetry.report.queue_peak_occupancy = state
                            .telemetry
                            .report
                            .queue_peak_occupancy
                            .max(outcome.peak_occupancy);
                        state.telemetry.report.queue_backpressure +=
                            u64::from(outcome.backpressure);
                    }
                    let stale = state.generation != ticket.key.generation;
                    match result {
                        Ok(DiskResult::Read(block)) if !stale => {
                            let shapes = block.shapes()?;
                            let dtypes = block.dtypes()?;
                            let bytes = block.bytes()?;
                            let capacity = block.capacity()?;
                            let record = state
                                .blocks
                                .get_mut(id)
                                .ok_or_else(|| CacheLifecycleError::MissingBlock(id.clone()))?;
                            if shapes != record.shapes
                                || dtypes != record.dtypes
                                || bytes != record.bytes
                            {
                                record.physical.fail_io_if_matches(&ticket.key);
                                drop(state);
                                worker.retire(&ticket);
                                return Err(CacheResidencyError::MalformedShard {
                                    path: location.path,
                                    reason: "array shape or dtype does not match the manifest"
                                        .into(),
                                });
                            }
                            if capacity > reserved_host_bytes {
                                record.physical.fail_io_if_matches(&ticket.key);
                                drop(state);
                                worker.retire(&ticket);
                                return Err(CacheResidencyError::Runtime(format!(
                                    "host cache allocation capacity {capacity} exceeded its pre-admitted bound {reserved_host_bytes}"
                                )));
                            }
                            if record.physical.io_matches(&ticket.key) {
                                record.physical.finish_read(&ticket.key, block)?;
                            }
                            loaded_from_disk = true;
                            update_report_totals(&mut state);
                        }
                        Ok(DiskResult::Read(_))
                        | Err(CacheResidencyError::DiskOperationCancelled { .. })
                            if stale =>
                        {
                            drop(state);
                            worker.retire(&ticket);
                            return Err(CacheResidencyError::DiskOperationCancelled {
                                generation: ticket.key.generation,
                            });
                        }
                        Ok(_) => {
                            drop(state);
                            worker.retire(&ticket);
                            return Err(CacheResidencyError::Runtime(
                                "cache disk worker returned an unexpected operation result".into(),
                            ));
                        }
                        Err(error) => {
                            if let Some(record) = state.blocks.get_mut(id) {
                                record.physical.fail_io_if_matches(&ticket.key);
                            }
                            state.telemetry.report.failures += 1;
                            state.layer_activity_mut(id.global_layer).failures += 1;
                            drop(state);
                            worker.retire(&ticket);
                            return Err(error);
                        }
                    }
                    drop(state);
                    worker.retire(&ticket);
                }
                CacheStoragePhase::HostUnbacked
                | CacheStoragePhase::HostBacked
                | CacheStoragePhase::HostWriting => {
                    if physical.phase() == CacheStoragePhase::HostWriting {
                        let pending = physical
                            .io()
                            .expect("host-writing phase owns its exact completion");
                        drop(state);
                        self.wait_for_host_release(&pending.ticket)?;
                        continue;
                    }
                    let block = physical
                        .host_resource()
                        .expect("stable host phase owns host resources")
                        .clone();
                    let transfer_reservation = state.pool.reserve_transfer(block.capacity()?)?;
                    let (device_arrays, completions) = block.copy_to_device(transfer_stream)?;
                    if state.generation != generation {
                        return Err(CacheResidencyError::DiskOperationCancelled { generation });
                    }
                    state.lifecycle.acquire(id)?;
                    let (bytes, promotion) = {
                        let record = state
                            .blocks
                            .get_mut(id)
                            .ok_or_else(|| CacheLifecycleError::MissingBlock(id.clone()))?;
                        let promotion = record.physical.promote_host(device_arrays.clone())?;
                        (record.bytes, promotion)
                    };
                    state.telemetry.report.demand_misses += 1;
                    if loaded_from_disk {
                        state.telemetry.report.disk_promotions += 1;
                    } else {
                        state.telemetry.report.host_promotions += 1;
                    }
                    state.telemetry.report.transfer_bytes += bytes;
                    let transfer_wait = started.elapsed();
                    state.telemetry.report.transfer_wait += transfer_wait;
                    let activity = state.layer_activity_mut(id.global_layer);
                    activity.demand_misses += 1;
                    if loaded_from_disk {
                        activity.disk_promotions += 1;
                    } else {
                        activity.host_promotions += 1;
                    }
                    activity.transfer_bytes += bytes;
                    activity.transfer_wait += transfer_wait;
                    drop(state);
                    if let Err(error) = self.rebalance(Some(id), true) {
                        let mut state = self.lock()?;
                        if state.generation == generation {
                            state.lifecycle.release(id)?;
                            if let Some(record) = state.blocks.get_mut(id) {
                                record.physical.restore_host(promotion)?;
                            }
                            update_report_totals(&mut state);
                        }
                        return Err(error);
                    }
                    return Ok(CacheBlockLease {
                        id: id.clone(),
                        arrays: device_arrays,
                        manager: self.clone(),
                        completions,
                        _transfer_reservation: Some(transfer_reservation),
                        released: false,
                    });
                }
                CacheStoragePhase::Device => {
                    let arrays = physical
                        .device_resource()
                        .expect("device phase owns device arrays")
                        .clone();
                    state.lifecycle.acquire(id)?;
                    drop(state);
                    let completion = match async_eval_with_event(arrays.arrays()) {
                        Ok(completion) => completion,
                        Err(source) => {
                            if let Ok(mut state) = self.lock() {
                                let _ = state.lifecycle.release(id);
                                update_report_totals(&mut state);
                            }
                            return Err(transfer_error(
                                "submit resident cache block completion",
                                source,
                            ));
                        }
                    };
                    let mut state = self.lock()?;
                    if state.generation != generation {
                        state.lifecycle.release(id)?;
                        return Err(CacheResidencyError::DiskOperationCancelled { generation });
                    }
                    state.telemetry.report.demand_hits += 1;
                    let transfer_wait = started.elapsed();
                    state.telemetry.report.transfer_wait += transfer_wait;
                    let activity = state.layer_activity_mut(id.global_layer);
                    activity.demand_hits += 1;
                    activity.transfer_wait += transfer_wait;
                    drop(state);
                    if let Err(error) = self.rebalance(Some(id), true) {
                        if let Ok(mut state) = self.lock() {
                            let _ = state.lifecycle.release(id);
                            update_report_totals(&mut state);
                        }
                        return Err(error);
                    }
                    return Ok(CacheBlockLease {
                        id: id.clone(),
                        arrays,
                        manager: self.clone(),
                        completions: vec![completion],
                        _transfer_reservation: None,
                        released: false,
                    });
                }
            }
        }
    }

    pub fn discard_before(
        &self,
        layer: usize,
        representation: CacheRepresentation,
        visible_start: i64,
        prefix_tokens: i64,
    ) -> Result<(), CacheResidencyError> {
        if self.options().retains_discarded_for_persistence() {
            return Ok(());
        }
        let mut state = self.lock()?;
        let ids = state
            .blocks
            .iter()
            .filter(|(id, record)| {
                id.global_layer == layer
                    && id.representation == representation
                    && id.end <= visible_start
                    && id.end > prefix_tokens
                    && state.lifecycle.lease_count(id).ok() == Some(0)
                    && !record.imported
            })
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        let tickets = if ids.is_empty() {
            Vec::new()
        } else {
            advance_generation_locked(&mut state)
        };
        for id in ids {
            if let Some(record) = state.blocks.remove(&id) {
                state.lifecycle.remove(&id)?;
                remove_ephemeral_file(&record);
                state.telemetry.report.discarded_sliding_blocks += 1;
            }
        }
        update_report_totals(&mut state);
        drop(state);
        self.retire_tickets(&tickets)?;
        Ok(())
    }

    /// Clears every live block and advances the manager generation.
    pub fn clear(&self) -> Result<(), CacheResidencyError> {
        let mut state = self.lock()?;
        state.lifecycle.clear()?;
        let tickets = advance_generation_locked(&mut state);
        for record in state.blocks.values() {
            remove_ephemeral_file(record);
        }
        state.blocks.clear();
        update_report_totals(&mut state);
        drop(state);
        self.retire_tickets(&tickets)?;
        Ok(())
    }

    /// Returns a bounded aggregate snapshot without retaining per-block history.
    pub fn report(&self) -> Result<CacheResidencyReport, CacheResidencyError> {
        let mut state = self.lock()?;
        update_report_totals(&mut state);
        if self.options().process_sampling_enabled() {
            sample_process(&mut state.telemetry.report);
        }
        Ok(state.telemetry.report.clone())
    }

    fn retire_tickets(&self, tickets: &[PendingCacheOperation]) -> Result<(), CacheResidencyError> {
        let mut first_error = None;
        for ticket in tickets {
            match ticket {
                PendingCacheOperation::Disk(ticket) => {
                    let result = ticket.wait_for_task_resources();
                    if result.is_ok() {
                        if let Some(worker) = &self.inner.disk_worker {
                            worker.retire(ticket);
                        }
                    }
                    let mut state = self.lock()?;
                    state.retiring_disk_reads.remove(&ticket.key);
                    update_report_totals(&mut state);
                    drop(state);
                    if let Err(error) = result {
                        if first_error.is_none() {
                            first_error = Some(error);
                        }
                    }
                }
                PendingCacheOperation::HostDemotion(ticket) => {
                    let result = self.finish_device_demotion(ticket);
                    let mut state = self.lock()?;
                    state.retiring_host_demotions.remove(&ticket.operation_id);
                    update_report_totals(&mut state);
                    drop(state);
                    if let Err(error) = result {
                        if first_error.is_none() {
                            first_error = Some(error);
                        }
                    }
                }
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    pub fn record_attention_scan(
        &self,
        global_layer: usize,
        prefill: bool,
        blocks: u64,
        bytes: u64,
        scratch_bytes: u64,
    ) -> Result<(), CacheResidencyError> {
        let mut state = self.lock()?;
        if prefill {
            state.telemetry.report.prefill_full_attention_blocks += blocks;
            state.telemetry.report.prefill_full_attention_bytes += bytes;
            let activity = state.layer_activity_mut(global_layer);
            activity.prefill_full_attention_blocks += blocks;
            activity.prefill_full_attention_bytes += bytes;
        } else {
            state.telemetry.report.decode_full_attention_blocks += blocks;
            state.telemetry.report.decode_full_attention_bytes += bytes;
            let activity = state.layer_activity_mut(global_layer);
            activity.decode_full_attention_blocks += blocks;
            activity.decode_full_attention_bytes += bytes;
        }
        state.telemetry.report.attention_scratch_peak_bytes = state
            .telemetry
            .report
            .attention_scratch_peak_bytes
            .max(scratch_bytes);
        let activity = state.layer_activity_mut(global_layer);
        activity.attention_scratch_peak_bytes =
            activity.attention_scratch_peak_bytes.max(scratch_bytes);
        Ok(())
    }

    fn release_lease(&self, id: &CacheBlockId) {
        let mut background_failed = false;
        if let Ok(mut state) = self.inner.state.lock() {
            let released = state.lifecycle.release(id);
            debug_assert!(
                released.is_ok(),
                "cache lease release lost ownership: {released:?}"
            );
            background_failed = released.is_err() || state.background_disk_error.is_some();
        }
        if !background_failed {
            let _ = self.rebalance(None, false);
        }
    }

    fn begin_host_demotion(
        &self,
        id: &CacheBlockId,
    ) -> Result<HostDemotionProgress, CacheResidencyError> {
        let mut state = self.lock()?;
        let Some(record) = state.blocks.get(id) else {
            return Ok(HostDemotionProgress::Retry);
        };
        if record.tier() != CacheTier::Host
            || state.lifecycle.is_leased(id)?
            || record.pending_disk().is_some()
        {
            return Ok(HostDemotionProgress::Retry);
        }

        // Persistent prompt-cache blocks and completed live-cache writes can be
        // released immediately; they do not require live writeback to be enabled.
        if record.disk().is_some() {
            let record = state.blocks.get_mut(id).expect("host block exists");
            record.physical.release_host_to_disk()?;
            state.telemetry.report.disk_demotions += 1;
            state.layer_activity_mut(id.global_layer).disk_demotions += 1;
            update_report_totals(&mut state);
            return Ok(HostDemotionProgress::Freed);
        }

        let (directory, budget_bytes) = match self.options().live_disk_policy() {
            LiveCacheDiskPolicy::Disabled => {
                state.telemetry.report.failures += 1;
                state.layer_activity_mut(id.global_layer).failures += 1;
                return Err(CacheResidencyError::LiveDiskRequired {
                    required: state.telemetry.report.current_host_bytes,
                    budget: self.options().host_budget_bytes(),
                });
            }
            LiveCacheDiskPolicy::Enabled {
                directory,
                budget_bytes,
                ..
            } => (directory.clone(), *budget_bytes),
        };
        let worker = self.inner.disk_worker.as_ref().ok_or_else(|| {
            CacheResidencyError::Runtime("live cache disk worker is unavailable".into())
        })?;
        let record = state.blocks.get(id).expect("host block exists");
        let live_disk_bytes = state
            .blocks
            .values()
            .filter(|record| record.disk().is_some_and(|location| !location.persistent))
            .map(|record| record.bytes)
            .sum::<u64>()
            .saturating_add(
                state
                    .host_write_reservations
                    .iter()
                    .filter(|(key, _)| {
                        !state.blocks.get(&key.id).is_some_and(|record| {
                            record.disk().is_some_and(|location| !location.persistent)
                        })
                    })
                    .map(|(_, reservation)| reservation.logical_bytes)
                    .sum(),
            );
        let projected = live_disk_bytes.saturating_add(record.bytes);
        if projected > budget_bytes {
            state.telemetry.report.failures += 1;
            state.layer_activity_mut(id.global_layer).failures += 1;
            return Err(CacheResidencyError::BudgetExceeded {
                tier: CacheTier::Disk,
                required: projected,
                budget: budget_bytes,
            });
        }
        let block = record
            .host_block()
            .ok_or_else(|| CacheResidencyError::MissingResidentArrays(id.clone()))?
            .clone();
        let host_capacity = block.capacity()?;
        let pool_admission = state.pool.reserve(CachePoolUsage {
            transfer_in_flight_bytes: host_capacity,
            disk_bytes: record.bytes,
            ..CachePoolUsage::default()
        })?;
        let submission = worker.prepare_write(
            state.generation,
            &directory,
            id,
            &block,
            Arc::downgrade(&self.inner.state),
        )?;
        let ticket = submission.ticket.clone();
        let record_bytes = record.bytes;
        if let Some(reservation_id) = submission.write_reservation_id {
            state.host_write_reservations.insert(
                ticket.key.clone(),
                HostWriteReservation {
                    reservation_id,
                    global_layer: id.global_layer,
                    logical_bytes: record_bytes,
                    host_capacity,
                    ticket: ticket.clone(),
                },
            );
        }
        let record = state.blocks.get_mut(id).expect("host block exists");
        record.physical.begin_write(MlxCacheIoOperation {
            ticket: ticket.clone(),
            reserved_host_bytes: None,
        })?;
        update_report_totals(&mut state);
        drop(pool_admission);
        drop(state);

        let enqueue_started = Instant::now();
        let outcome = match submission.enqueue() {
            Ok(outcome) => outcome,
            Err(error) => {
                let mut state = self.lock()?;
                if let Some(record) = state.blocks.get_mut(&ticket.key.id) {
                    record.physical.fail_io_if_matches(&ticket.key);
                }
                state.telemetry.report.failures += 1;
                state
                    .layer_activity_mut(ticket.key.id.global_layer)
                    .failures += 1;
                update_report_totals(&mut state);
                drop(state);
                worker.retire(&ticket);
                return Err(error);
            }
        };
        let enqueue_wait = enqueue_started.elapsed();
        let mut state = self.lock()?;
        if outcome.joined {
            state.telemetry.report.in_flight_waits += 1;
            state
                .layer_activity_mut(ticket.key.id.global_layer)
                .in_flight_waits += 1;
        }
        state.telemetry.report.queue_peak_occupancy = state
            .telemetry
            .report
            .queue_peak_occupancy
            .max(outcome.peak_occupancy);
        state.telemetry.report.queue_backpressure += u64::from(outcome.backpressure);
        if outcome.backpressure {
            state.telemetry.report.transfer_wait += enqueue_wait;
            state
                .layer_activity_mut(ticket.key.id.global_layer)
                .transfer_wait += enqueue_wait;
        }
        update_report_totals(&mut state);
        Ok(HostDemotionProgress::Pending(ticket))
    }

    fn wait_for_host_release(&self, ticket: &DiskTicket) -> Result<(), CacheResidencyError> {
        let started = Instant::now();
        let result = ticket.wait();
        ticket.wait_for_task_resources()?;
        let elapsed = started.elapsed();
        let mut state = self.lock()?;
        state.telemetry.report.in_flight_waits += 1;
        state.telemetry.report.transfer_wait += elapsed;
        let activity = state.layer_activity_mut(ticket.key.id.global_layer);
        activity.in_flight_waits += 1;
        activity.transfer_wait += elapsed;
        if result.is_err() {
            // The write commit records its error for asynchronous callers. This
            // caller observed it directly, so do not surface the same failure twice.
            state.background_disk_error = None;
        }
        update_report_totals(&mut state);
        drop(state);
        match result {
            Ok(DiskResult::Write(_)) => Ok(()),
            Ok(_) => Err(CacheResidencyError::Runtime(
                "cache disk worker returned an unexpected write result".into(),
            )),
            Err(error) => Err(error),
        }
    }

    fn begin_device_demotion(
        &self,
        id: &CacheBlockId,
    ) -> Result<HostDemotionTicket, CacheResidencyError> {
        let mut state = self.lock()?;
        let record = state
            .blocks
            .get(id)
            .ok_or_else(|| CacheLifecycleError::MissingBlock(id.clone()))?;
        if state.lifecycle.is_leased(id)? {
            return Err(CacheLifecycleError::BlockLeased(id.clone()).into());
        }
        if record.physical.phase() != CacheStoragePhase::Device
            || record.physical.backing().is_some()
        {
            return Err(CacheResidencyError::MissingResidentArrays(id.clone()));
        }
        let arrays = record
            .physical
            .device_resource()
            .ok_or_else(|| CacheResidencyError::MissingResidentArrays(id.clone()))?
            .clone();
        let reserved_host_bytes = host_cache_capacity_upper_bound(&arrays)?;
        let required_host_bytes = state
            .telemetry
            .report
            .current_host_bytes
            .saturating_add(reserved_host_bytes);
        if required_host_bytes > self.options().host_budget_bytes() {
            return Err(CacheResidencyError::BudgetExceeded {
                tier: CacheTier::Host,
                required: required_host_bytes,
                budget: self.options().host_budget_bytes(),
            });
        }
        let pool_admission = state.pool.reserve(CachePoolUsage {
            host_bytes: reserved_host_bytes,
            transfer_in_flight_bytes: reserved_host_bytes,
            ..CachePoolUsage::default()
        })?;
        let device = state.transfer_device.unwrap_or(CacheTransferDevice::CPU);
        let ticket = self.inner.host_demotion_worker.submit(
            id,
            arrays.clone(),
            device,
            reserved_host_bytes,
        )?;
        let record = state.blocks.get_mut(id).expect("demotion block exists");
        record.physical.begin_host_demotion(ticket.clone())?;
        update_report_totals(&mut state);
        drop(pool_admission);
        Ok(ticket)
    }

    fn finish_device_demotion(
        &self,
        ticket: &HostDemotionTicket,
    ) -> Result<(), CacheResidencyError> {
        let started = Instant::now();
        let result = ticket.wait();
        let elapsed = started.elapsed();
        let mut state = self.lock()?;
        let matches = state
            .blocks
            .get(&ticket.id)
            .and_then(CacheBlockRecord::host_demotion_ticket)
            .is_some_and(|pending| pending.operation_id == ticket.operation_id);
        if !matches {
            return Ok(());
        }
        state.telemetry.report.in_flight_waits += 1;
        state.telemetry.report.transfer_wait += elapsed;
        let activity = state.layer_activity_mut(ticket.id.global_layer);
        activity.in_flight_waits += 1;
        activity.transfer_wait += elapsed;
        match result {
            Ok(block) => {
                let capacity = block.capacity()?;
                if capacity > ticket.reserved_host_bytes {
                    let record = state
                        .blocks
                        .get_mut(&ticket.id)
                        .expect("demotion block exists");
                    record.physical.fail_host_demotion(ticket.operation_id)?;
                    state.telemetry.report.failures += 1;
                    state.layer_activity_mut(ticket.id.global_layer).failures += 1;
                    update_report_totals(&mut state);
                    return Err(CacheResidencyError::Runtime(format!(
                        "host cache allocation used {capacity} bytes, exceeding reserved upper bound {}",
                        ticket.reserved_host_bytes
                    )));
                }
                let bytes = state
                    .blocks
                    .get(&ticket.id)
                    .expect("demotion block exists")
                    .bytes;
                let record = state
                    .blocks
                    .get_mut(&ticket.id)
                    .expect("demotion block exists");
                record
                    .physical
                    .finish_host_demotion(ticket.operation_id, block)?;
                state.telemetry.report.host_demotions += 1;
                state.telemetry.report.transfer_bytes += bytes;
                let activity = state.layer_activity_mut(ticket.id.global_layer);
                activity.host_demotions += 1;
                activity.transfer_bytes += bytes;
                update_report_totals(&mut state);
                Ok(())
            }
            Err(error) => {
                let record = state
                    .blocks
                    .get_mut(&ticket.id)
                    .expect("demotion block exists");
                record.physical.fail_host_demotion(ticket.operation_id)?;
                state.telemetry.report.failures += 1;
                state.layer_activity_mut(ticket.id.global_layer).failures += 1;
                update_report_totals(&mut state);
                Err(error)
            }
        }
    }

    fn rebalance(
        &self,
        required: Option<&CacheBlockId>,
        allow_recent_eviction: bool,
    ) -> Result<(), CacheResidencyError> {
        loop {
            let mut state = self.lock()?;
            if let Some(error) = state.background_disk_error.take() {
                return Err(CacheResidencyError::Runtime(format!(
                    "background cache disk write failed: {error}"
                )));
            }
            update_report_totals(&mut state);
            let pool_report = state.pool.report()?;
            let pool_device_over =
                pool_report.current_device_bytes > pool_report.limits.device_bytes();
            let local_device_over =
                state.telemetry.report.current_device_bytes > self.options().device_budget_bytes();
            if local_device_over || pool_device_over {
                if let Some(ticket) = state
                    .blocks
                    .values()
                    .find_map(CacheBlockRecord::host_demotion_ticket)
                    .cloned()
                {
                    drop(state);
                    self.finish_device_demotion(&ticket)?;
                    continue;
                }
                let candidate = eviction_candidate(
                    &state,
                    CacheTier::Device,
                    required,
                    self.options().recent_device_blocks(),
                    self.options().eviction_policy(),
                )
                .or_else(|| {
                    // Recent protection remains strict for mutation capacity,
                    // but may yield to an existing block demanded by attention.
                    if allow_recent_eviction {
                        eviction_candidate(
                            &state,
                            CacheTier::Device,
                            required,
                            0,
                            self.options().eviction_policy(),
                        )
                    } else {
                        None
                    }
                });
                let Some(id) = candidate else {
                    state.telemetry.report.failures += 1;
                    if let Some(required) = required {
                        state.layer_activity_mut(required.global_layer).failures += 1;
                    } else {
                        state.telemetry.unassigned_activity_mut().failures += 1;
                    }
                    return Err(if pool_device_over && !local_device_over {
                        CachePoolError::BudgetExceeded {
                            resource: CachePoolResource::Device,
                            required: pool_report.current_device_bytes,
                            budget: pool_report.limits.device_bytes(),
                        }
                        .into()
                    } else {
                        CacheResidencyError::BudgetExceeded {
                            tier: CacheTier::Device,
                            required: state.telemetry.report.current_device_bytes,
                            budget: self.options().device_budget_bytes(),
                        }
                    });
                };
                if state
                    .blocks
                    .get(&id)
                    .and_then(CacheBlockRecord::disk)
                    .is_some()
                {
                    let record = state.blocks.get_mut(&id).expect("candidate exists");
                    record.physical.release_device_to_disk()?;
                    state.telemetry.report.disk_demotions += 1;
                    state.layer_activity_mut(id.global_layer).disk_demotions += 1;
                    continue;
                }
                let candidate_host_bytes = state
                    .blocks
                    .get(&id)
                    .expect("candidate exists")
                    .physical
                    .device_resource()
                    .map(host_cache_capacity_upper_bound)
                    .transpose()?
                    .ok_or_else(|| CacheResidencyError::MissingResidentArrays(id.clone()))?;
                let required_host_bytes = state
                    .telemetry
                    .report
                    .current_host_bytes
                    .saturating_add(candidate_host_bytes);
                if required_host_bytes > self.options().host_budget_bytes() {
                    if candidate_host_bytes > self.options().host_budget_bytes() {
                        state.telemetry.report.failures += 1;
                        state.layer_activity_mut(id.global_layer).failures += 1;
                        return Err(CacheResidencyError::BudgetExceeded {
                            tier: CacheTier::Host,
                            required: candidate_host_bytes,
                            budget: self.options().host_budget_bytes(),
                        });
                    }
                    let host_candidate = eviction_candidate(
                        &state,
                        CacheTier::Host,
                        required,
                        0,
                        self.options().eviction_policy(),
                    );
                    let pending = state
                        .host_write_reservations
                        .values()
                        .next()
                        .map(|reservation| reservation.ticket.clone())
                        .or_else(|| {
                            state.blocks.values().find_map(|record| {
                                record
                                    .pending_disk()
                                    .filter(|pending| {
                                        pending.ticket.key.kind == CacheIoOperationKind::Write
                                    })
                                    .map(|pending| pending.ticket.clone())
                            })
                        });
                    drop(state);
                    if let Some(id) = host_candidate {
                        match self.begin_host_demotion(&id)? {
                            HostDemotionProgress::Retry | HostDemotionProgress::Freed => continue,
                            HostDemotionProgress::Pending(ticket) => {
                                self.wait_for_host_release(&ticket)?;
                                continue;
                            }
                        }
                    }
                    if let Some(ticket) = pending {
                        self.wait_for_host_release(&ticket)?;
                        continue;
                    }
                    let mut state = self.lock()?;
                    state.telemetry.report.failures += 1;
                    state.layer_activity_mut(id.global_layer).failures += 1;
                    return Err(match self.options().live_disk_policy() {
                        LiveCacheDiskPolicy::Disabled => CacheResidencyError::LiveDiskRequired {
                            required: required_host_bytes,
                            budget: self.options().host_budget_bytes(),
                        },
                        LiveCacheDiskPolicy::Enabled { .. } => {
                            CacheResidencyError::BudgetExceeded {
                                tier: CacheTier::Host,
                                required: required_host_bytes,
                                budget: self.options().host_budget_bytes(),
                            }
                        }
                    });
                }
                drop(state);
                self.begin_device_demotion(&id)?;
                continue;
            }

            let pool_report = state.pool.report()?;
            let pool_host_over = pool_report.current_host_bytes > pool_report.limits.host_bytes();
            let local_host_over =
                state.telemetry.report.current_host_bytes > self.options().host_budget_bytes();
            if local_host_over || pool_host_over {
                let candidate = eviction_candidate(
                    &state,
                    CacheTier::Host,
                    required,
                    0,
                    self.options().eviction_policy(),
                );
                let pending = state
                    .host_write_reservations
                    .values()
                    .next()
                    .map(|reservation| reservation.ticket.clone())
                    .or_else(|| {
                        state.blocks.values().find_map(|record| {
                            record
                                .pending_disk()
                                .filter(|pending| {
                                    pending.ticket.key.kind == CacheIoOperationKind::Write
                                })
                                .map(|pending| pending.ticket.clone())
                        })
                    });
                let required_host_bytes = state.telemetry.report.current_host_bytes;
                drop(state);
                if let Some(id) = candidate {
                    match self.begin_host_demotion(&id)? {
                        HostDemotionProgress::Retry | HostDemotionProgress::Freed => continue,
                        HostDemotionProgress::Pending(ticket) => {
                            self.wait_for_host_release(&ticket)?;
                            continue;
                        }
                    }
                }
                if let Some(ticket) = pending {
                    self.wait_for_host_release(&ticket)?;
                    continue;
                }
                let mut state = self.lock()?;
                state.telemetry.report.failures += 1;
                if let Some(required) = required {
                    state.layer_activity_mut(required.global_layer).failures += 1;
                } else {
                    state.telemetry.unassigned_activity_mut().failures += 1;
                }
                return Err(if pool_host_over && !local_host_over {
                    CachePoolError::BudgetExceeded {
                        resource: CachePoolResource::Host,
                        required: pool_report.current_host_bytes,
                        budget: pool_report.limits.host_bytes(),
                    }
                    .into()
                } else {
                    CacheResidencyError::BudgetExceeded {
                        tier: CacheTier::Host,
                        required: required_host_bytes,
                        budget: self.options().host_budget_bytes(),
                    }
                });
            }

            // Start one background write as soon as the finite host tier fills.
            // It remains charged to host memory until the worker commits and
            // releases its buffers; a later demotion waits only if it needs space.
            let proactive = matches!(
                self.options().live_disk_policy(),
                LiveCacheDiskPolicy::Enabled { .. }
            ) && state.telemetry.report.current_host_bytes != 0
                && (state.telemetry.report.current_host_bytes
                    >= self.options().host_budget_bytes()
                    || pool_report.current_host_bytes >= pool_report.limits.host_bytes());
            let candidate = if proactive {
                eviction_candidate(
                    &state,
                    CacheTier::Host,
                    required,
                    0,
                    self.options().eviction_policy(),
                )
            } else {
                None
            };
            drop(state);
            if let Some(id) = candidate {
                let _ = self.begin_host_demotion(&id)?;
            }
            return Ok(());
        }
    }

    /// Writes a completed immutable prefix atomically to a persistent directory.
    pub fn save_prompt_cache(
        &self,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        state_arrays: &[PromptCacheStateArray<'_>],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, CacheResidencyError> {
        let destination = destination.as_ref();
        descriptor.validate()?;
        let publication = PromptCachePublication::begin(destination, options.replace_existing)?;
        let temporary = publication.staging_directory().to_path_buf();

        let result = (|| {
            let records = {
                let state = self.lock()?;
                if let Some(id) = state.lifecycle.first_leased() {
                    return Err(CacheLifecycleError::BlockLeased(id.clone()).into());
                }
                state.blocks.values().cloned().collect::<Vec<_>>()
            };
            let mut manifest_blocks = Vec::with_capacity(records.len());
            let mut manifest_state = Vec::with_capacity(state_arrays.len());
            let mut logical_bytes = 0u64;
            for (index, record) in records.iter().enumerate() {
                let shard = format!("block-{index:08}.safetensors");
                let shard_path = temporary.join(&shard);
                if let Some(arrays) = record.physical.device_resource() {
                    save_block_arrays(&shard_path, arrays)?;
                } else if let Some(block) = record.physical.host_resource() {
                    save_host_cache_block(&shard_path, block)?;
                } else {
                    let location = record
                        .physical
                        .backing()
                        .expect("backing-only phase owns its disk location");
                    let block = load_host_cache_block_direct(
                        location,
                        record.physical.id().representation,
                    )?;
                    save_host_cache_block(&shard_path, &block)?;
                }
                let payload_sha256 = finalize_prompt_cache_shard(&shard_path)?;
                logical_bytes += record.bytes;
                let names = array_names(record.physical.id().representation);
                manifest_blocks.push(PromptCacheBlock {
                    global_layer: record.physical.id().global_layer,
                    representation: record.physical.id().representation,
                    start: record.physical.id().start,
                    end: record.physical.id().end,
                    rank: record.physical.id().rank,
                    shard,
                    first_array: names.0.into(),
                    second_array: names.1.into(),
                    first_shape: record.shapes[0].clone(),
                    second_shape: record.shapes[1].clone(),
                    first_dtype: record.dtypes[0].clone(),
                    second_dtype: record.dtypes[1].clone(),
                    logical_bytes: record.bytes,
                    payload_sha256,
                });
            }
            for (index, state) in state_arrays.iter().enumerate() {
                let shard = format!("state-{index:08}.safetensors");
                let shard_path = temporary.join(&shard);
                let array_name = "state";
                Array::save_safetensors([(array_name, state.array)], None, &shard_path).map_err(
                    |source| {
                        CacheResidencyError::Runtime(format!(
                            "save {}: {source}",
                            shard_path.display()
                        ))
                    },
                )?;
                let payload_sha256 = finalize_prompt_cache_shard(&shard_path)?;
                let state_bytes = state.array.nbytes() as u64;
                logical_bytes = logical_bytes.checked_add(state_bytes).ok_or_else(|| {
                    CacheResidencyError::PromptCache(PromptCacheError::Malformed(
                        "prompt-cache state byte count overflow".into(),
                    ))
                })?;
                manifest_state.push(PromptCacheStateTensor {
                    owner: state.owner,
                    role: state.role,
                    shard,
                    array: array_name.into(),
                    shape: state.array.shape().to_vec(),
                    dtype: dtype_name(state.array.dtype()),
                    logical_bytes: state_bytes,
                    payload_sha256,
                });
            }
            let manifest = PromptCacheManifest {
                schema_version: PROMPT_CACHE_SCHEMA_VERSION,
                model_family: descriptor.model_family,
                effective_model_type: descriptor.effective_model_type,
                checkpoint_fingerprint: descriptor.checkpoint_fingerprint,
                prefix_content_fingerprint: descriptor.prefix_content_fingerprint,
                architecture_fingerprint: descriptor.architecture_fingerprint,
                layer_count: descriptor.layer_count,
                global_layer_start: descriptor.global_layer_start,
                global_layer_end: descriptor.global_layer_end,
                block_size_tokens: self.options().block_size_tokens(),
                batch_size: descriptor.batch_size,
                total_prefix_tokens: prefix_token_ids.len(),
                prefix_sha256: prompt_cache_token_fingerprint(prefix_token_ids),
                layer_layout: descriptor.layer_layout,
                layer_prefix_offsets: descriptor.layer_prefix_offsets,
                state_segments: descriptor.state_segments,
                sink_tokens: descriptor.sink_tokens,
                topology: descriptor.topology,
                application_namespace: options.application_namespace.clone(),
                blocks: manifest_blocks,
                state_tensors: manifest_state,
            };
            publication.commit(&manifest)?;
            let mut state = self.lock()?;
            state.telemetry.report.prompt_cache_saves += 1;
            state.telemetry.report.prompt_cache_bytes += logical_bytes;
            Ok(manifest)
        })();

        result
    }
}

impl Drop for CacheResidencyManager {
    fn drop(&mut self) {
        if Arc::strong_count(&self.inner) != 1 {
            return;
        }
        if let Ok(state) = self.inner.state.lock() {
            for record in state.blocks.values() {
                remove_ephemeral_file(record);
            }
        }
    }
}

/// Fixed current-plus-next cache-block promotion window.
pub struct CacheBlockPrefetch {
    manager: CacheResidencyManager,
    ids: VecDeque<CacheBlockId>,
    pending: VecDeque<CacheBlockLease>,
    transfer_stream: Stream,
    execution_stream: Stream,
}

impl CacheBlockPrefetch {
    fn new(
        manager: CacheResidencyManager,
        ids: Vec<CacheBlockId>,
        execution_stream: &Stream,
    ) -> Result<Self, CacheResidencyError> {
        let device = execution_stream
            .get_device()
            .map_err(|source| transfer_error("inspect cache execution stream", source))?;
        Ok(Self {
            manager,
            ids: ids.into(),
            pending: VecDeque::with_capacity(PAGED_CACHE_PREFETCH_BLOCKS),
            transfer_stream: Stream::new_with_device(&device),
            execution_stream: execution_stream.clone(),
        })
    }

    /// Returns the next ordered block. At most the current and following block
    /// hold leases; a one-block device budget falls back to demand promotion.
    pub fn next_block(&mut self) -> Result<Option<CacheBlockLease>, CacheResidencyError> {
        while self.pending.len() < PAGED_CACHE_PREFETCH_BLOCKS {
            let Some(id) = self.ids.front() else {
                break;
            };
            if !self.pending.is_empty() && !self.window_has_capacity_for(id)? {
                break;
            }
            match self
                .manager
                .prepare_block_transfer(id, &self.transfer_stream)
            {
                Ok(lease) => {
                    self.ids.pop_front();
                    self.pending.push_back(lease);
                }
                Err(CacheResidencyError::BudgetExceeded {
                    tier: CacheTier::Device,
                    ..
                }) if !self.pending.is_empty() => break,
                Err(error) => return Err(error),
            }
        }
        let Some(lease) = self.pending.pop_front() else {
            return Ok(None);
        };
        lease.wait_on(&self.execution_stream)?;
        Ok(Some(lease))
    }

    fn window_has_capacity_for(&self, id: &CacheBlockId) -> Result<bool, CacheResidencyError> {
        let pending_bytes = self
            .pending
            .iter()
            .fold(0u64, |total, lease| total.saturating_add(lease.bytes()));
        let state = self.manager.lock()?;
        let next_bytes = state
            .blocks
            .get(id)
            .ok_or_else(|| CacheLifecycleError::MissingBlock(id.clone()))?
            .bytes;
        Ok(
            pending_bytes.saturating_add(next_bytes)
                <= self.manager.options().device_budget_bytes(),
        )
    }

    #[cfg(test)]
    fn pending_len(&self) -> usize {
        self.pending.len()
    }

    #[cfg(test)]
    fn stream_indices(&self) -> Result<(i32, i32), CacheResidencyError> {
        Ok((
            self.execution_stream
                .get_index()
                .map_err(|source| transfer_error("inspect cache execution stream", source))?,
            self.transfer_stream
                .get_index()
                .map_err(|source| transfer_error("inspect cache transfer stream", source))?,
        ))
    }
}

/// One device-cataloged block and its single-shot promotion completion.
///
/// The lease inserts its event dependency before exposing arrays to a consumer.
/// Dropping it after that wait is safe: MLX retains the event for queued work,
/// while lazy consumer graphs retain the arrays they reference. Asynchronous
/// promotion failures poison the consumer stream and surface when that work is
/// evaluated. The lease is intentionally neither `Send` nor `Sync` because it
/// owns `safemlx`'s thread-affine [`Event`].
pub struct CacheBlockLease {
    id: CacheBlockId,
    arrays: CacheBlockArrays,
    manager: CacheResidencyManager,
    completions: Vec<Event>,
    _transfer_reservation: Option<CachePoolReservation>,
    released: bool,
}

impl CacheBlockLease {
    pub fn id(&self) -> &CacheBlockId {
        &self.id
    }

    pub fn arrays(&self) -> &CacheBlockArrays {
        &self.arrays
    }

    pub fn bytes(&self) -> u64 {
        self.arrays.bytes()
    }

    fn wait_on(&self, stream: &Stream) -> Result<(), CacheResidencyError> {
        for completion in &self.completions {
            completion
                .wait_on(stream)
                .map_err(|source| transfer_error("wait for cache block promotion", source))?;
        }
        Ok(())
    }
}

impl Drop for CacheBlockLease {
    fn drop(&mut self) {
        if !self.released {
            self.manager.release_lease(&self.id);
            self.released = true;
        }
    }
}

/// In-memory fixed state supplied when saving a cache snapshot.
pub struct PromptCacheStateArray<'a> {
    /// Layer or model-global owner.
    pub owner: StateTensorOwner,
    /// Semantic role declared by the canonical layout.
    pub role: StateTensorRole,
    /// Array to persist.
    pub array: &'a Array,
}

/// Catalogs a compatible prompt prefix lazily as read-only disk-backed blocks.
pub fn open_prompt_cache(
    directory: impl AsRef<Path>,
    expected: &PromptCacheDescriptor,
    model: &PromptCacheModelIdentity,
    prefix_token_ids: &[u32],
    options: PagedCacheOptions,
) -> Result<(CacheResidencyManager, PromptCacheManifest), CacheResidencyError> {
    validate_prompt_cache_model_identity(expected, model)?;
    let directory = directory.as_ref();
    let cache_root = resolve_prompt_cache_root(directory)?;
    let manifest = inspect_prompt_cache(directory)?;
    manifest.validate_compatibility(expected, prefix_token_ids)?;
    if manifest.block_size_tokens != options.block_size_tokens() {
        return Err(CacheResidencyError::PromptCache(
            PromptCacheError::Incompatible(format!(
                "block size {} does not match requested {}",
                manifest.block_size_tokens,
                options.block_size_tokens()
            )),
        ));
    }
    let manager = CacheResidencyManager::new(options)?;
    {
        let mut state = manager.lock()?;
        for block in &manifest.blocks {
            let id = CacheBlockId {
                session_id: manager.session_id,
                global_layer: block.global_layer,
                representation: block.representation,
                start: block.start,
                end: block.end,
                rank: block.rank,
            };
            let shard = safe_prompt_cache_shard_path(&cache_root, &block.shard)?;
            let mapped = map_prompt_cache_shard(&shard)?;
            let record = CacheBlockRecord {
                physical: MlxCacheBlockStorage::disk(
                    id.clone(),
                    DiskLocation {
                        path: shard,
                        first_name: block.first_array.clone(),
                        second_name: block.second_array.clone(),
                        persistent: true,
                        mapped: Some(mapped),
                        payload_sha256: Some(block.payload_sha256.clone()),
                        payload_verification: Arc::new(OnceLock::new()),
                    },
                ),
                bytes: block.logical_bytes,
                shapes: [block.first_shape.clone(), block.second_shape.clone()],
                dtypes: [block.first_dtype.clone(), block.second_dtype.clone()],
                imported: true,
            };
            state
                .lifecycle
                .insert(id.clone(), block.end <= manifest.sink_tokens as i64)?;
            if state.blocks.insert(id.clone(), record).is_some() {
                return Err(CacheLifecycleError::DuplicateBlock(id).into());
            }
        }
        state.telemetry.report.prompt_cache_loads += 1;
        state.telemetry.report.prompt_cache_bytes += manifest
            .blocks
            .iter()
            .map(|block| block.logical_bytes)
            .sum::<u64>();
        state.telemetry.report.imported_mapped_shards += manifest.blocks.len() as u64;
        update_report_totals(&mut state);
    }
    Ok((manager, manifest))
}

/// Materialized non-attention state tensor from a validated prompt cache.
pub struct LoadedPromptCacheStateTensor {
    pub owner: StateTensorOwner,
    pub role: StateTensorRole,
    pub array: Array,
}

/// Loads all fixed-state tensors after manifest and model compatibility validation.
pub fn load_prompt_cache_state_tensors(
    directory: impl AsRef<Path>,
    manifest: &PromptCacheManifest,
    stream: &Stream,
) -> Result<Vec<LoadedPromptCacheStateTensor>, CacheResidencyError> {
    let root = resolve_prompt_cache_root(directory.as_ref())?;
    let host_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut loaded = Vec::with_capacity(manifest.state_tensors.len());
    for state in &manifest.state_tensors {
        let path = safe_prompt_cache_shard_path(&root, &state.shard)?;
        let actual_hash = hash_prompt_cache_shard_payload(&path)?;
        if actual_hash != state.payload_sha256 {
            return Err(CacheResidencyError::MalformedShard {
                path,
                reason: "fixed-state payload digest does not match the manifest".into(),
            });
        }
        let mut arrays = Array::load_safetensors(&path, &host_stream).map_err(|source| {
            CacheResidencyError::Runtime(format!("load {}: {source}", path.display()))
        })?;
        let array =
            arrays
                .remove(&state.array)
                .ok_or_else(|| CacheResidencyError::MalformedShard {
                    path: path.clone(),
                    reason: format!("missing state array {}", state.array),
                })?;
        if !arrays.is_empty() {
            return Err(CacheResidencyError::MalformedShard {
                path,
                reason: "fixed-state shard contains unexpected arrays".into(),
            });
        }
        let array = array.copy(stream).map_err(|source| {
            CacheResidencyError::Runtime(format!(
                "copy {} to execution stream: {source}",
                path.display()
            ))
        })?;
        eval([&array]).map_err(|source| {
            CacheResidencyError::Runtime(format!(
                "evaluate {} on execution stream: {source}",
                path.display()
            ))
        })?;
        loaded.push(LoadedPromptCacheStateTensor {
            owner: state.owner,
            role: state.role,
            array,
        });
    }
    Ok(loaded)
}

fn validate_block_arrays(
    arrays: &CacheBlockArrays,
    token_count: i64,
) -> Result<(), CacheResidencyError> {
    let [first, second] = arrays.arrays();
    if first.dtype() != second.dtype() {
        return Err(CacheResidencyError::ArrayMismatch(
            "both arrays in a cache block must share a dtype".into(),
        ));
    }
    let sequence_axis = match arrays {
        CacheBlockArrays::KeyValue { .. } => {
            if first.ndim() < 2 || second.ndim() < 2 {
                return Err(CacheResidencyError::ArrayMismatch(
                    "key/value blocks must have a sequence axis".into(),
                ));
            }
            first.ndim() - 2
        }
        CacheBlockArrays::CompressedLatentRotary { .. } => {
            if first.ndim() != 3 || second.ndim() != 3 {
                return Err(CacheResidencyError::ArrayMismatch(
                    "compressed latent blocks must be rank-3".into(),
                ));
            }
            1
        }
    };
    if first.dim(sequence_axis as i32) as i64 != token_count
        || second.dim(sequence_axis as i32) as i64 != token_count
    {
        return Err(CacheResidencyError::ArrayMismatch(
            "cache block range does not match its sequence dimensions".into(),
        ));
    }
    match arrays {
        CacheBlockArrays::KeyValue { .. } => {
            if first.ndim() != second.ndim()
                || first.shape()[..first.ndim() - 2] != second.shape()[..second.ndim() - 2]
            {
                return Err(CacheResidencyError::ArrayMismatch(
                    "key and value blocks must share leading dimensions".into(),
                ));
            }
        }
        CacheBlockArrays::CompressedLatentRotary { .. } => {
            if first.dim(0) != second.dim(0) || first.dim(1) != second.dim(1) {
                return Err(CacheResidencyError::ArrayMismatch(
                    "compressed latent and rotary blocks must share batch and sequence dimensions"
                        .into(),
                ));
            }
        }
    }
    Ok(())
}

fn cancel_record_operation(record: &CacheBlockRecord, report: &mut CacheResidencyReport) {
    if let Some(pending) = record.pending_disk() {
        if pending.ticket.cancel() {
            report.cancellations += 1;
        }
    }
}

fn advance_generation_locked(state: &mut CacheManagerState) -> Vec<PendingCacheOperation> {
    state.generation = state.generation.wrapping_add(1);
    state.background_disk_error = None;
    let mut tickets = Vec::new();
    let mut demotion_reservations = Vec::new();
    let mut read_reservations = Vec::new();
    for record in state.blocks.values_mut() {
        if let Some(ticket) = record.host_demotion_ticket().cloned() {
            demotion_reservations.push((
                ticket.operation_id,
                RetiringHostDemotion {
                    id: record.physical.id().clone(),
                    device_bytes: record.bytes,
                    host_bytes: ticket.reserved_host_bytes,
                },
            ));
            tickets.push(PendingCacheOperation::HostDemotion(ticket));
        }
        if let Some(pending) = record.pending_disk().cloned() {
            if pending.ticket.cancel() {
                state.telemetry.report.cancellations += 1;
            }
            tickets.push(PendingCacheOperation::Disk(pending.ticket.clone()));
            if pending.ticket.key.kind != CacheIoOperationKind::Write {
                if let Some(reserved_host_bytes) = pending.reserved_host_bytes {
                    read_reservations.push((
                        pending.ticket.key.clone(),
                        (record.physical.id().global_layer, reserved_host_bytes),
                    ));
                }
                record.physical.fail_io_if_matches(&pending.ticket.key);
            }
        }
    }
    state.retiring_host_demotions.extend(demotion_reservations);
    state.retiring_disk_reads.extend(read_reservations);
    tickets
}

fn update_report_totals(state: &mut CacheManagerState) {
    let device_budget_bytes = state.device_budget_bytes;
    let host_budget_bytes = state.host_budget_bytes;
    let disk_budget_bytes = state.disk_budget_bytes;
    let pool_disk_bytes = state
        .blocks
        .values()
        .filter(|record| record.disk().is_some_and(|location| !location.persistent))
        .map(|record| record.bytes)
        .sum::<u64>()
        .saturating_add(
            state
                .host_write_reservations
                .iter()
                .filter(|(key, _)| {
                    !state.blocks.get(&key.id).is_some_and(|record| {
                        record.disk().is_some_and(|location| !location.persistent)
                    })
                })
                .map(|(_, reservation)| reservation.logical_bytes)
                .sum(),
        );
    let report = &mut state.telemetry.report;
    report.key_value_blocks = 0;
    report.compressed_latent_blocks = 0;
    report.device_blocks = 0;
    report.host_blocks = 0;
    report.disk_blocks = 0;
    let tails = state.lifecycle.tails().collect::<Vec<_>>();
    report.current_device_bytes = tails.iter().map(|(_, tail)| tail.bytes).sum();
    report.mutable_tail_bytes = report.current_device_bytes;
    report.current_host_bytes = 0;
    report.current_disk_bytes = 0;
    report.in_flight_write_blocks = 0;
    report.in_flight_write_bytes = 0;
    report.in_flight_host_demotion_blocks = 0;
    report.in_flight_host_demotion_bytes = 0;
    report.protected_prefix_blocks = 0;
    report.protected_recent_blocks = 0;
    report.logical_cached_tokens = 0;
    let mut per_layer = BTreeMap::<usize, CacheLayerResidencyStats>::new();
    let mut layer_ends: HashMap<usize, i64> = HashMap::new();
    for (layer, tail) in tails {
        layer_ends.insert(layer, tail.end);
        let layer_report = per_layer.entry(layer).or_default();
        layer_report.current_device_bytes += tail.bytes;
        layer_report.mutable_tail_bytes += tail.bytes;
        layer_report.logical_cached_tokens = tail.end.max(0) as u64;
    }
    for record in state.blocks.values() {
        let layer_report = per_layer
            .entry(record.physical.id().global_layer)
            .or_default();
        match record.physical.id().representation {
            CacheRepresentation::KeyValue => {
                report.key_value_blocks += 1;
                layer_report.key_value_blocks += 1;
            }
            CacheRepresentation::CompressedLatentRotary => {
                report.compressed_latent_blocks += 1;
                layer_report.compressed_latent_blocks += 1;
            }
        }
        let pending_write = record
            .pending_disk()
            .is_some_and(|pending| pending.ticket.key.kind == CacheIoOperationKind::Write);
        let pending_write_is_reserved = record.pending_disk().is_some_and(|pending| {
            state
                .host_write_reservations
                .contains_key(&pending.ticket.key)
        });
        if pending_write && !pending_write_is_reserved {
            let host_capacity = record
                .host_block()
                .and_then(|block| block.capacity().ok())
                .expect("pending cache write retains inspectable host-transfer storage");
            report.in_flight_write_blocks += 1;
            report.in_flight_write_bytes += host_capacity;
            layer_report.in_flight_write_blocks += 1;
            layer_report.in_flight_write_bytes += host_capacity;
        }
        match record.physical.phase() {
            CacheStoragePhase::Device => {
                report.device_blocks += 1;
                report.current_device_bytes += record.bytes;
                layer_report.device_blocks += 1;
                layer_report.current_device_bytes += record.bytes;
            }
            CacheStoragePhase::DemotingToHost => {
                let ticket = record
                    .physical
                    .host_demotion()
                    .expect("demoting phase owns its exact completion");
                let host_bytes = ticket.reserved_host_bytes;
                report.device_blocks += 1;
                report.host_blocks += 1;
                report.current_device_bytes += record.bytes;
                report.current_host_bytes += host_bytes;
                report.in_flight_host_demotion_blocks += 1;
                report.in_flight_host_demotion_bytes += host_bytes;
                layer_report.device_blocks += 1;
                layer_report.host_blocks += 1;
                layer_report.current_device_bytes += record.bytes;
                layer_report.current_host_bytes += host_bytes;
                layer_report.in_flight_host_demotion_blocks += 1;
                layer_report.in_flight_host_demotion_bytes += host_bytes;
            }
            CacheStoragePhase::HostUnbacked
            | CacheStoragePhase::HostWriting
            | CacheStoragePhase::HostBacked => {
                let block = record
                    .physical
                    .host_resource()
                    .expect("host phase owns host resources");
                let host_bytes = block
                    .capacity()
                    .expect("validated host cache block has inspectable capacity");
                report.host_blocks += 1;
                report.current_host_bytes += host_bytes;
                layer_report.host_blocks += 1;
                layer_report.current_host_bytes += host_bytes;
            }
            CacheStoragePhase::DiskReady | CacheStoragePhase::DiskReading => {
                report.disk_blocks += 1;
                report.current_disk_bytes += record.bytes;
                layer_report.disk_blocks += 1;
                layer_report.current_disk_bytes += record.bytes;
                if let Some(reserved_host_bytes) = record
                    .physical
                    .io()
                    .and_then(|pending| pending.reserved_host_bytes)
                {
                    report.current_host_bytes += reserved_host_bytes;
                    layer_report.current_host_bytes += reserved_host_bytes;
                }
            }
        }
        if state
            .lifecycle
            .is_protected_prefix(record.physical.id())
            .expect("storage block has lifecycle state")
        {
            report.protected_prefix_blocks += 1;
            layer_report.protected_prefix_blocks += 1;
        }
        layer_report.logical_cached_tokens = layer_report
            .logical_cached_tokens
            .max(record.physical.id().end.max(0) as u64);
        layer_ends
            .entry(record.physical.id().global_layer)
            .and_modify(|end| *end = (*end).max(record.physical.id().end))
            .or_insert(record.physical.id().end);
    }
    for (operation_id, reservation) in &state.retiring_host_demotions {
        let covered_by_demoting_record = state.blocks.get(&reservation.id).is_some_and(|record| {
            record
                .host_demotion_ticket()
                .is_some_and(|ticket| ticket.operation_id == *operation_id)
        });
        if covered_by_demoting_record {
            continue;
        }
        report.device_blocks += 1;
        report.host_blocks += 1;
        report.current_device_bytes += reservation.device_bytes;
        report.current_host_bytes += reservation.host_bytes;
        report.in_flight_host_demotion_blocks += 1;
        report.in_flight_host_demotion_bytes += reservation.host_bytes;
        let layer_report = per_layer.entry(reservation.id.global_layer).or_default();
        layer_report.device_blocks += 1;
        layer_report.host_blocks += 1;
        layer_report.current_device_bytes += reservation.device_bytes;
        layer_report.current_host_bytes += reservation.host_bytes;
        layer_report.in_flight_host_demotion_blocks += 1;
        layer_report.in_flight_host_demotion_bytes += reservation.host_bytes;
    }
    for (key, reservation) in &state.host_write_reservations {
        report.in_flight_write_blocks += 1;
        report.in_flight_write_bytes += reservation.host_capacity;
        let layer_report = per_layer.entry(reservation.global_layer).or_default();
        layer_report.in_flight_write_blocks += 1;
        layer_report.in_flight_write_bytes += reservation.host_capacity;
        let covered_by_host_record = state.blocks.get(&key.id).is_some_and(|record| {
            record.tier() == CacheTier::Host
                && record
                    .pending_disk()
                    .is_some_and(|pending| pending.ticket.key == *key)
        });
        if !covered_by_host_record {
            report.current_host_bytes += reservation.host_capacity;
            layer_report.current_host_bytes += reservation.host_capacity;
        }
    }
    for (key, (global_layer, reserved_host_bytes)) in &state.retiring_disk_reads {
        let covered_by_pending_record = state.blocks.get(&key.id).is_some_and(|record| {
            record.physical.phase() == CacheStoragePhase::DiskReading
                && record.physical.io_matches(key)
        });
        if !covered_by_pending_record {
            report.current_host_bytes += reserved_host_bytes;
            per_layer
                .entry(*global_layer)
                .or_default()
                .current_host_bytes += reserved_host_bytes;
        }
    }
    let device_ids = state
        .blocks
        .values()
        .filter(|record| record.tier() == CacheTier::Device)
        .map(|record| record.physical.id().clone())
        .collect::<Vec<_>>();
    let recent = state
        .lifecycle
        .recent_protection_counts(device_ids, state.recent_device_blocks)
        .expect("storage blocks have lifecycle state");
    report.protected_recent_blocks = recent.values().sum();
    for (layer, count) in recent {
        per_layer.entry(layer).or_default().protected_recent_blocks = count;
    }
    report.logical_cached_tokens = layer_ends.values().copied().max().unwrap_or(0).max(0) as u64;
    state.telemetry.finalize_snapshot(
        per_layer,
        device_budget_bytes,
        host_budget_bytes,
        disk_budget_bytes,
    );
    let report = &state.telemetry.report;
    let pool_usage = CachePoolUsage {
        device_bytes: report.current_device_bytes,
        host_bytes: report.current_host_bytes,
        transfer_in_flight_bytes: report
            .in_flight_write_bytes
            .saturating_add(report.in_flight_host_demotion_bytes),
        // Pending writes reserve their eventual disk extent before the worker
        // owns the request, so aggregate disk admission cannot overcommit.
        disk_bytes: pool_disk_bytes,
    };
    let _ = state.pool.update_manager(state.pool_manager_id, pool_usage);
}

fn eviction_candidate(
    state: &CacheManagerState,
    tier: CacheTier,
    required: Option<&CacheBlockId>,
    recent_per_layer: usize,
    policy: CacheEvictionPolicy,
) -> Option<CacheBlockId> {
    let candidates = state
        .blocks
        .values()
        .filter(|record| {
            physical_state_is_tier_candidate(&record.physical, tier)
                && record.pending_disk().is_none()
        })
        .map(|record| record.physical.id().clone());
    state
        .lifecycle
        .eviction_candidate(candidates, required, recent_per_layer, policy)
        .expect("storage blocks have lifecycle state")
}

fn physical_state_is_tier_candidate(physical: &MlxCacheBlockStorage, tier: CacheTier) -> bool {
    matches!(
        (physical.phase(), tier),
        (CacheStoragePhase::Device, CacheTier::Device)
            | (
                CacheStoragePhase::HostUnbacked | CacheStoragePhase::HostBacked,
                CacheTier::Host
            )
            | (CacheStoragePhase::DiskReady, CacheTier::Disk)
    )
}

fn write_live_block(
    directory: &Path,
    id: &CacheBlockId,
    block: &HostCacheBlock,
) -> Result<DiskLocation, CacheResidencyError> {
    let publication = LiveCacheBlockPublication::begin(directory, id);
    save_host_cache_block(publication.staging_path(), block)?;
    sync_file(publication.staging_path())?;
    let path = publication.commit()?;
    let names = array_names(id.representation);
    Ok(DiskLocation {
        path,
        first_name: names.0.into(),
        second_name: names.1.into(),
        persistent: false,
        mapped: None,
        payload_sha256: None,
        payload_verification: Arc::new(OnceLock::new()),
    })
}

fn save_block_arrays(path: &Path, arrays: &CacheBlockArrays) -> Result<(), CacheResidencyError> {
    let names = array_names(arrays.representation());
    let values = arrays.arrays();
    Array::save_safetensors([(names.0, values[0]), (names.1, values[1])], None, path).map_err(
        |source| CacheResidencyError::Runtime(format!("save {}: {source}", path.display())),
    )
}

fn save_host_cache_block(path: &Path, block: &HostCacheBlock) -> Result<(), CacheResidencyError> {
    let names = array_names(block.representation());
    let [first, second] = block.buffers();
    let first_shape = host_shape_to_stored(first)?;
    let second_shape = host_shape_to_stored(second)?;
    let first_dtype = host_dtype_to_stored(first)?;
    let second_dtype = host_dtype_to_stored(second)?;
    let first_bytes = first
        .as_bytes()
        .map_err(|source| transfer_error("read first host cache payload", source))?;
    let second_bytes = second
        .as_bytes()
        .map_err(|source| transfer_error("read second host cache payload", source))?;
    let first_view = TensorView::new(first_dtype, first_shape, first_bytes).map_err(|source| {
        CacheResidencyError::Runtime(format!("create first host cache tensor view: {source}"))
    })?;
    let second_view =
        TensorView::new(second_dtype, second_shape, second_bytes).map_err(|source| {
            CacheResidencyError::Runtime(format!("create second host cache tensor view: {source}"))
        })?;
    serialize_to_file([(names.0, first_view), (names.1, second_view)], None, path).map_err(
        |source| CacheResidencyError::Runtime(format!("save {}: {source}", path.display())),
    )
}

fn host_shape_to_stored(
    buffer: &ImmutableHostTransferBuffer,
) -> Result<Vec<usize>, CacheResidencyError> {
    buffer
        .shape()
        .map_err(|source| transfer_error("inspect host cache shape", source))?
        .into_iter()
        .map(|dimension| {
            usize::try_from(dimension).map_err(|_| {
                CacheResidencyError::Runtime(
                    "host cache shape contains a negative dimension".into(),
                )
            })
        })
        .collect()
}

fn host_dtype_to_stored(
    buffer: &ImmutableHostTransferBuffer,
) -> Result<StoredDtype, CacheResidencyError> {
    let dtype = buffer
        .dtype()
        .map_err(|source| transfer_error("inspect host cache dtype", source))?;
    Ok(match dtype {
        Dtype::Bool => StoredDtype::BOOL,
        Dtype::Uint8 => StoredDtype::U8,
        Dtype::Uint16 => StoredDtype::U16,
        Dtype::Uint32 => StoredDtype::U32,
        Dtype::Uint64 => StoredDtype::U64,
        Dtype::Int8 => StoredDtype::I8,
        Dtype::Int16 => StoredDtype::I16,
        Dtype::Int32 => StoredDtype::I32,
        Dtype::Int64 => StoredDtype::I64,
        Dtype::Float16 => StoredDtype::F16,
        Dtype::Float32 => StoredDtype::F32,
        Dtype::Float64 => StoredDtype::F64,
        Dtype::Bfloat16 => StoredDtype::BF16,
        Dtype::Complex64 => StoredDtype::C64,
    })
}

fn stored_dtype_to_host(dtype: StoredDtype) -> Result<Dtype, CacheResidencyError> {
    match dtype {
        StoredDtype::BOOL => Ok(Dtype::Bool),
        StoredDtype::U8 => Ok(Dtype::Uint8),
        StoredDtype::U16 => Ok(Dtype::Uint16),
        StoredDtype::U32 => Ok(Dtype::Uint32),
        StoredDtype::U64 => Ok(Dtype::Uint64),
        StoredDtype::I8 => Ok(Dtype::Int8),
        StoredDtype::I16 => Ok(Dtype::Int16),
        StoredDtype::I32 => Ok(Dtype::Int32),
        StoredDtype::I64 => Ok(Dtype::Int64),
        StoredDtype::F16 => Ok(Dtype::Float16),
        StoredDtype::F32 => Ok(Dtype::Float32),
        StoredDtype::F64 => Ok(Dtype::Float64),
        StoredDtype::BF16 => Ok(Dtype::Bfloat16),
        StoredDtype::C64 => Ok(Dtype::Complex64),
        other => Err(CacheResidencyError::ArrayMismatch(format!(
            "unsupported host cache dtype {other:?}"
        ))),
    }
}

fn verify_disk_payload(location: &DiskLocation) -> Result<(), CacheResidencyError> {
    let Some(expected) = &location.payload_sha256 else {
        return Ok(());
    };
    let verification = location.payload_verification.get_or_init(|| {
        let actual = if let Some(mapped) = &location.mapped {
            if mapped.len() < 8 {
                return Err("file is too short for a safetensors header".into());
            }
            let mut length_bytes = [0u8; 8];
            length_bytes.copy_from_slice(&mapped[..8]);
            let header_len = usize::try_from(u64::from_le_bytes(length_bytes))
                .map_err(|_| "safetensors header length exceeds addressable memory".to_string())?;
            let data_start = 8usize
                .checked_add(header_len)
                .filter(|start| *start <= mapped.len())
                .ok_or_else(|| "safetensors header extends beyond the mapped shard".to_string())?;
            sha256_hex(Sha256::digest(&mapped[data_start..]))
        } else {
            hash_prompt_cache_shard_payload(&location.path).map_err(|error| error.to_string())?
        };
        if &actual == expected {
            Ok(())
        } else {
            Err(format!(
                "payload SHA-256 mismatch: expected {expected}, computed {actual}"
            ))
        }
    });
    verification
        .as_ref()
        .map_err(|reason| CacheResidencyError::MalformedShard {
            path: location.path.clone(),
            reason: reason.clone(),
        })
        .copied()
}

fn load_host_cache_block_direct(
    location: &DiskLocation,
    representation: CacheRepresentation,
) -> Result<HostCacheBlock, CacheResidencyError> {
    verify_disk_payload(location)?;
    let owned;
    let bytes = if let Some(mapped) = &location.mapped {
        mapped.as_ref()
    } else {
        owned = fs::read(&location.path).map_err(|source| CacheResidencyError::Io {
            action: "read cache block shard",
            path: location.path.clone(),
            source,
        })?;
        owned.as_slice()
    };
    let tensors = safetensors::SafeTensors::deserialize(bytes).map_err(|error| {
        CacheResidencyError::MalformedShard {
            path: location.path.clone(),
            reason: error.to_string(),
        }
    })?;
    if tensors.names().len() != 2 {
        return Err(CacheResidencyError::MalformedShard {
            path: location.path.clone(),
            reason: "unexpected extra arrays".into(),
        });
    }
    let first = tensors.tensor(&location.first_name).map_err(|error| {
        CacheResidencyError::MalformedShard {
            path: location.path.clone(),
            reason: error.to_string(),
        }
    })?;
    let second = tensors.tensor(&location.second_name).map_err(|error| {
        CacheResidencyError::MalformedShard {
            path: location.path.clone(),
            reason: error.to_string(),
        }
    })?;
    Ok(HostCacheBlock::from_buffers(
        representation,
        host_buffer_from_view(first)?,
        host_buffer_from_view(second)?,
    ))
}

fn host_buffer_from_view(
    view: safetensors::tensor::TensorView<'_>,
) -> Result<ImmutableHostTransferBuffer, CacheResidencyError> {
    let shape = view
        .shape()
        .iter()
        .map(|dimension| {
            i32::try_from(*dimension).map_err(|_| {
                CacheResidencyError::ArrayMismatch(
                    "cache block dimension exceeds the MLX i32 shape range".into(),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let dtype = stored_dtype_to_host(view.dtype())?;
    let mut buffer = HostTransferBuffer::new(&shape, dtype, HostTransferPolicy::Transfer)
        .map_err(|source| transfer_error("allocate disk-loaded host cache buffer", source))?;
    let destination = buffer
        .as_bytes_mut()
        .map_err(|source| transfer_error("access disk-loaded host cache buffer", source))?;
    if destination.len() != view.data().len() {
        return Err(CacheResidencyError::ArrayMismatch(
            "cache block payload length does not match its shape and dtype".into(),
        ));
    }
    destination.copy_from_slice(view.data());
    Ok(buffer.freeze())
}

fn remove_ephemeral_file(record: &CacheBlockRecord) {
    if let Some(location) = record.disk() {
        if !location.persistent {
            let _ = fs::remove_file(&location.path);
        }
    }
}

fn array_names(representation: CacheRepresentation) -> (&'static str, &'static str) {
    match representation {
        CacheRepresentation::KeyValue => ("keys", "values"),
        CacheRepresentation::CompressedLatentRotary => ("latent", "rotary_key"),
    }
}

fn dtype_name(dtype: Dtype) -> String {
    format!("{dtype:?}")
}

fn sha256_hex(digest: impl AsRef<[u8]>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    let digest = digest.as_ref();
    let mut encoded = String::with_capacity(digest.len() * 2);
    for &byte in digest {
        encoded.push(HEX[usize::from(byte >> 4)] as char);
        encoded.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    encoded
}

fn map_prompt_cache_shard(path: &Path) -> Result<Arc<Mmap>, CacheResidencyError> {
    let file = File::open(path).map_err(|source| CacheResidencyError::Io {
        action: "open prompt cache shard for mapping",
        path: path.to_path_buf(),
        source,
    })?;
    // SAFETY: prompt-cache shards are immutable after publication and the Mmap
    // is retained by every DiskLocation that can create an MLX view from it.
    let mapped =
        unsafe { MmapOptions::new().map(&file) }.map_err(|source| CacheResidencyError::Io {
            action: "map prompt cache shard",
            path: path.to_path_buf(),
            source,
        })?;
    safetensors::SafeTensors::deserialize(&mapped).map_err(|error| {
        CacheResidencyError::MalformedShard {
            path: path.to_path_buf(),
            reason: error.to_string(),
        }
    })?;
    Ok(Arc::new(mapped))
}

#[cfg(test)]
fn cpu_stream() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
}

fn sync_file(path: &Path) -> Result<(), CacheResidencyError> {
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|source| CacheResidencyError::Io {
            action: "synchronize cache file",
            path: path.to_path_buf(),
            source,
        })
}

#[cfg(unix)]
fn sample_process(report: &mut CacheResidencyReport) {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `usage` points to writable storage of the exact type required by
    // `getrusage`; the value is read only after a successful return.
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if status != 0 {
        return;
    }
    // SAFETY: a successful `getrusage` call initialized the structure.
    let usage = unsafe { usage.assume_init() };
    #[cfg(target_os = "macos")]
    let rss_bytes = usage.ru_maxrss.max(0) as u64;
    #[cfg(not(target_os = "macos"))]
    let rss_bytes = (usage.ru_maxrss.max(0) as u64).saturating_mul(1024);
    report.process_rss_bytes = Some(rss_bytes);
    report.process_minor_page_faults = Some(usage.ru_minflt.max(0) as u64);
    report.process_major_page_faults = Some(usage.ru_majflt.max(0) as u64);
}

#[cfg(not(unix))]
fn sample_process(_report: &mut CacheResidencyReport) {}

/// Structured cache residency and persistence failures.
#[derive(Debug, thiserror::Error)]
pub enum CacheResidencyError {
    /// Backend-neutral cache identity, geometry, or state policy is invalid.
    #[error(transparent)]
    Policy(#[from] CachePolicyError),
    /// Backend-neutral mutable-cache residency configuration is invalid.
    #[error(transparent)]
    Configuration(#[from] CacheResidencyConfigurationError),
    /// Backend-neutral aggregate cache ownership or admission failed.
    #[error(transparent)]
    Pool(#[from] CachePoolError),
    /// Backend-neutral block, lease, access, or mutable-tail lifecycle failed.
    #[error(transparent)]
    Lifecycle(#[from] CacheLifecycleError),
    /// Backend-neutral physical storage transition failed.
    #[error(transparent)]
    Storage(#[from] CacheStorageError),
    /// Backend-neutral cache I/O admission or completion lifecycle failed.
    #[error(transparent)]
    IoExecution(#[from] CacheIoExecutionStateError),
    /// Backend-neutral prompt-cache identity or catalog validation failed.
    #[error(transparent)]
    PromptCache(#[from] PromptCacheError),
    /// Backend-neutral prompt-cache filesystem or publication operation failed.
    #[error(transparent)]
    Persistence(#[from] PromptCachePersistenceError),
    /// Backend-neutral live-cache file publication failed.
    #[error(transparent)]
    LivePublication(#[from] LiveCachePublicationError),
    /// Paged options were contradictory or unbounded.
    #[error("invalid paged cache options: {0}")]
    InvalidOptions(String),
    /// A sealed block used an invalid absolute token range.
    #[error("invalid cache block token range {start}..{end}")]
    InvalidTokenRange {
        /// Inclusive absolute token position.
        start: i64,
        /// Exclusive absolute token position.
        end: i64,
    },
    /// Both arrays in a block did not describe the same token range.
    #[error("invalid cache block arrays: {0}")]
    ArrayMismatch(String),
    /// A disk-backed block had no safe location.
    #[error("cache block has no disk location: {0:?}")]
    MissingDiskLocation(CacheBlockId),
    /// A host or device block had no evaluated arrays.
    #[error("cache block has no resident arrays: {0:?}")]
    MissingResidentArrays(CacheBlockId),
    /// A finite tier budget could not admit required state.
    #[error("{tier:?} cache budget exceeded: requires {required} bytes, budget is {budget}")]
    BudgetExceeded {
        /// Tier that could not admit required state.
        tier: CacheTier,
        /// Bytes required by the operation (physical capacity for the host tier).
        required: u64,
        /// Configured finite tier budget.
        budget: u64,
    },
    /// Full-context history exceeded host memory without explicit disk backing.
    #[error(
        "host cache requires {required} bytes but budget is {budget}; enable live disk backing or use a larger finite budget"
    )]
    LiveDiskRequired {
        /// Physical host allocation capacity required by retained history.
        required: u64,
        /// Configured finite host budget.
        budget: u64,
    },
    /// The manager lock was poisoned by a panic.
    #[error("cache residency manager lock was poisoned")]
    ManagerPoisoned,
    /// A queued or in-flight disk operation belonged to an invalidated generation.
    #[error("cache disk operation from generation {generation} was cancelled")]
    DiskOperationCancelled {
        /// Generation invalidated by reset or truncation.
        generation: u64,
    },
    /// MLX evaluation or array I/O failed.
    #[error("cache runtime failure: {0}")]
    Runtime(String),
    /// A filesystem operation failed.
    #[error("failed to {action} at {path}: {source}")]
    Io {
        /// Filesystem action that failed.
        action: &'static str,
        /// Path involved in the failed action.
        path: PathBuf,
        /// Underlying filesystem failure.
        #[source]
        source: std::io::Error,
    },
    /// A safetensors block had missing, extra, or corrupt arrays.
    #[error("malformed prompt cache shard {path}: {reason}")]
    MalformedShard {
        /// Invalid shard path.
        path: PathBuf,
        /// Structural or data validation failure.
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::{
        cpu_stream, hash_prompt_cache_shard_payload, host_cache_capacity_upper_bound,
        inspect_prompt_cache, map_prompt_cache_shard, open_prompt_cache, verify_disk_payload,
        CacheBlockArrays, CacheBlockId, CacheBlockRecord, CacheIoOperationKey,
        CacheIoOperationKind, CacheLayerResidencyStats, CacheManagerState, CachePoolError,
        CachePoolResource, CacheRankIdentity, CacheRepresentation, CacheResidencyError,
        CacheResidencyManager, CacheResidencyPool, CacheStoragePhase, CacheTier, DiskLocation,
        DiskResult, DiskTask, DiskWorker, DiskWriteCommit, HostCacheBlock, HostDemotionCompletion,
        HostDemotionTicket, HostWriteReservation, MlxCacheBlockStorage, MlxCacheIoOperation,
        PagedCacheOptions, StateTensorOwner, StateTensorRole,
    };
    use eredu_core::cache::{
        prompt_cache_token_fingerprint, validate_prompt_cache_model_identity, LayerCachePolicy,
        MutableStateResidency, PromptCacheBlock, PromptCacheDescriptor, PromptCacheError,
        PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions, PromptCacheStateTensor,
        PromptCacheTopology, StateResidencyClass, StateTensorDimension, StateTensorDtype,
        StateTensorPolicy, PROMPT_CACHE_SCHEMA_VERSION,
    };
    use eredu_core::{AttentionPolicy, LayerSchedule};
    use eredu_runtime::{
        resolve_prompt_cache_root, CachePoolLimits, CacheResidencyConfigurationError,
        MutableCacheTail, PromptCachePersistenceError, CACHE_RESIDENCY_LAYER_REPORT_LIMIT,
        PROMPT_CACHE_CURRENT_FILE, PROMPT_CACHE_GENERATIONS_DIRECTORY,
    };
    use safemlx::{
        host_transfer_capacity_upper_bound, transforms::async_eval_with_event, Array, Device,
        DeviceType, HostTransferPolicy, HostTransferStorageKind, Stream,
    };
    use safetensors::tensor::{serialize_to_file, Dtype as StoredDtype, TensorView};
    use std::{
        fs,
        hash::{DefaultHasher, Hash, Hasher},
        path::{Path, PathBuf},
        sync::{mpsc, Arc, OnceLock},
        thread,
        time::Duration,
    };

    fn disk_test_id(start: i64) -> CacheBlockId {
        CacheBlockId {
            session_id: 7,
            global_layer: 0,
            representation: CacheRepresentation::KeyValue,
            start,
            end: start + 1,
            rank: None,
        }
    }

    #[test]
    fn layer_truncation_clears_only_the_selected_pages_and_mutable_tail() {
        let manager = CacheResidencyManager::new(
            PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
                .unwrap()
                .with_full_attention(true),
        )
        .unwrap();
        let block_id = |global_layer| CacheBlockId {
            session_id: manager.session_id,
            global_layer,
            representation: CacheRepresentation::KeyValue,
            start: 0,
            end: 4,
            rank: None,
        };
        {
            let mut state = manager.lock().unwrap();
            for global_layer in 0..=1 {
                let id = block_id(global_layer);
                let mut location = missing_location(
                    Path::new("/nonexistent/eredu-cache-test"),
                    &format!("layer-{global_layer}.safetensors"),
                );
                location.persistent = true;
                insert_test_record(
                    &mut state,
                    CacheBlockRecord {
                        physical: MlxCacheBlockStorage::disk(id, location),
                        bytes: 0,
                        shapes: [vec![1, 1, 1, 1], vec![1, 1, 1, 1]],
                        dtypes: ["Float32".into(), "Float32".into()],
                        imported: false,
                    },
                    false,
                    0,
                );
            }
        }
        manager.set_tail_state(0, 40, 5).unwrap();
        manager.set_tail_state(1, 56, 7).unwrap();
        let generation_before = manager.lock().unwrap().generation;

        manager
            .truncate_layer_transaction(1, CacheRepresentation::KeyValue, 0, None, 0)
            .unwrap();

        let state = manager.lock().unwrap();
        assert_eq!(state.generation, generation_before + 1);
        assert!(state.blocks.contains_key(&block_id(0)));
        assert!(!state.blocks.contains_key(&block_id(1)));
        assert_eq!(
            state.lifecycle.tail(0),
            Some(MutableCacheTail { bytes: 40, end: 5 })
        );
        assert_eq!(
            state.lifecycle.tail(1),
            Some(MutableCacheTail { bytes: 0, end: 0 })
        );
    }

    fn missing_location(root: &Path, name: &str) -> DiskLocation {
        DiskLocation {
            path: root.join(name),
            first_name: "keys".into(),
            second_name: "values".into(),
            persistent: false,
            mapped: None,
            payload_sha256: None,
            payload_verification: Arc::new(OnceLock::new()),
        }
    }

    fn test_device_block() -> CacheBlockArrays {
        CacheBlockArrays::KeyValue {
            keys: Array::from_slice(&[0.0f32], &[1]),
            values: Array::from_slice(&[0.0f32], &[1]),
        }
    }

    fn test_host_block() -> HostCacheBlock {
        HostCacheBlock::from_device_arrays(&test_device_block(), &super::cpu_stream()).unwrap()
    }

    fn test_host_writing(block: HostCacheBlock, ticket: super::DiskTicket) -> MlxCacheBlockStorage {
        let mut physical = MlxCacheBlockStorage::host(ticket.key.id.clone(), block, None);
        physical
            .begin_write(MlxCacheIoOperation {
                ticket,
                reserved_host_bytes: None,
            })
            .unwrap();
        physical
    }

    fn test_demoting(arrays: CacheBlockArrays, ticket: HostDemotionTicket) -> MlxCacheBlockStorage {
        let mut physical = MlxCacheBlockStorage::device(ticket.id.clone(), arrays, None);
        physical.begin_host_demotion(ticket).unwrap();
        physical
    }

    fn insert_test_record(
        state: &mut CacheManagerState,
        record: CacheBlockRecord,
        protected_prefix: bool,
        leases: usize,
    ) {
        let id = record.physical.id().clone();
        state
            .lifecycle
            .insert(id.clone(), protected_prefix)
            .unwrap();
        for _ in 0..leases {
            state.lifecycle.acquire(&id).unwrap();
        }
        assert!(state.blocks.insert(id, record).is_none());
    }

    fn manager_with_leased_block() -> CacheResidencyManager {
        let manager = CacheResidencyManager::new(
            PagedCacheOptions::new(1, 64, 64, 1)
                .unwrap()
                .with_full_attention(true),
        )
        .unwrap();
        let id = CacheBlockId {
            session_id: manager.session_id,
            global_layer: 0,
            representation: CacheRepresentation::KeyValue,
            start: 0,
            end: 1,
            rank: None,
        };
        insert_test_record(
            &mut manager.lock().unwrap(),
            CacheBlockRecord {
                physical: MlxCacheBlockStorage::host(id.clone(), test_host_block(), None),
                bytes: 0,
                shapes: [vec![1, 1, 1, 1], vec![1, 1, 1, 1]],
                dtypes: ["Float32".into(), "Float32".into()],
                imported: false,
            },
            false,
            1,
        );
        manager
    }

    #[test]
    fn process_pool_enforces_aggregate_device_budget_and_releases_membership() {
        let pool = CacheResidencyPool::new(CachePoolLimits::new(20, 20, 20, 0).unwrap());
        let options = PagedCacheOptions::new(1, 20, 20, 1)
            .unwrap()
            .with_pool(pool.clone())
            .unwrap();
        let first = CacheResidencyManager::new(options.clone()).unwrap();
        let second = CacheResidencyManager::new(options).unwrap();
        let first_layer_handle = first.clone();

        first.set_tail_state(0, 12, 1).unwrap();
        let error = second.set_tail_state(0, 12, 1).unwrap_err();
        assert!(matches!(
            error,
            CacheResidencyError::Pool(CachePoolError::BudgetExceeded {
                resource: CachePoolResource::Device,
                required: 24,
                budget: 20,
            })
        ));
        let report = pool.report().unwrap();
        assert_eq!(report.managers, 2);
        assert_eq!(report.current_device_bytes, 12);
        assert_eq!(report.peak_device_bytes, 12);

        drop(first);
        let report = pool.report().unwrap();
        assert_eq!(report.managers, 2);
        assert_eq!(report.current_device_bytes, 12);
        drop(first_layer_handle);
        let report = pool.report().unwrap();
        assert_eq!(report.managers, 1);
        assert_eq!(report.current_device_bytes, 0);
        drop(second);
        assert_eq!(pool.report().unwrap().managers, 0);
    }

    #[test]
    fn per_cache_limits_cannot_exceed_their_process_pool() {
        let pool = CacheResidencyPool::new(CachePoolLimits::new(8, 4, 8, 0).unwrap());
        let error = PagedCacheOptions::new(1, 16, 4, 1)
            .unwrap()
            .with_pool(pool)
            .unwrap_err();
        assert!(matches!(
            error,
            CacheResidencyConfigurationError::InvalidOptions(_)
        ));
        assert!(error.to_string().contains("per-cache device budget 16"));
    }

    fn prompt_descriptor() -> PromptCacheDescriptor {
        PromptCacheDescriptor {
            model_family: "decoder".into(),
            effective_model_type: "decoder".into(),
            checkpoint_fingerprint: "checkpoint".into(),
            prefix_content_fingerprint: "text:prefix".into(),
            architecture_fingerprint: "architecture".into(),
            layer_count: 1,
            global_layer_start: 0,
            global_layer_end: 1,
            batch_size: 1,
            layer_layout: PromptCacheModelIdentity::key_value_layouts([None], 1, 1).unwrap(),
            sink_tokens: 0,
            layer_prefix_offsets: vec![0],
            state_segments: vec![
                eredu_core::cache::PromptCacheStateSegment::new("state", 0..1).unwrap(),
            ],
            topology: PromptCacheTopology::default(),
        }
    }

    fn key_value_layout(
        windows: impl IntoIterator<Item = Option<i32>>,
    ) -> LayerSchedule<LayerCachePolicy> {
        PromptCacheModelIdentity::key_value_layouts(windows, 1, 1).unwrap()
    }

    fn stable_hash(value: &impl Hash) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    fn prompt_model_identity() -> PromptCacheModelIdentity {
        let descriptor = prompt_descriptor();
        PromptCacheModelIdentity {
            model_family: descriptor.model_family,
            effective_model_type: descriptor.effective_model_type,
            architecture_fingerprint: descriptor.architecture_fingerprint,
            layer_count: descriptor.layer_count,
            global_layer_start: descriptor.global_layer_start,
            global_layer_end: descriptor.global_layer_end,
            sink_tokens: descriptor.sink_tokens,
            layer_prefix_offsets: descriptor.layer_prefix_offsets,
            state_segments: descriptor.state_segments,
            topology: descriptor.topology,
            layer_layout: descriptor.layer_layout,
        }
    }

    const TEST_PROMPT_CACHE_GENERATION: &str = "generation-test";

    fn create_prompt_fixture_generation(root: &Path) -> PathBuf {
        let generation = root
            .join(PROMPT_CACHE_GENERATIONS_DIRECTORY)
            .join(TEST_PROMPT_CACHE_GENERATION);
        fs::create_dir_all(&generation).unwrap();
        fs::write(
            root.join(PROMPT_CACHE_CURRENT_FILE),
            format!("{TEST_PROMPT_CACHE_GENERATION}\n"),
        )
        .unwrap();
        generation
    }

    fn prompt_fixture_root(root: &Path) -> PathBuf {
        resolve_prompt_cache_root(root).unwrap()
    }

    fn prompt_fixture_manifest_path(root: &Path) -> PathBuf {
        prompt_fixture_root(root).join("manifest.json")
    }

    fn write_prompt_fixture(root: &Path, namespace: &str) -> PromptCacheManifest {
        let generation = create_prompt_fixture_generation(root);
        let keys = 1.0f32.to_le_bytes();
        let values = 2.0f32.to_le_bytes();
        let key_view = TensorView::new(StoredDtype::F32, vec![1, 1, 1, 1], &keys).unwrap();
        let value_view = TensorView::new(StoredDtype::F32, vec![1, 1, 1, 1], &values).unwrap();
        serialize_to_file(
            [("keys", key_view), ("values", value_view)],
            None,
            &generation.join("block.safetensors"),
        )
        .unwrap();
        let descriptor = prompt_descriptor();
        let manifest = PromptCacheManifest {
            schema_version: PROMPT_CACHE_SCHEMA_VERSION,
            model_family: descriptor.model_family,
            effective_model_type: descriptor.effective_model_type,
            checkpoint_fingerprint: descriptor.checkpoint_fingerprint,
            prefix_content_fingerprint: descriptor.prefix_content_fingerprint,
            architecture_fingerprint: descriptor.architecture_fingerprint,
            layer_count: 1,
            global_layer_start: 0,
            global_layer_end: 1,
            block_size_tokens: 1,
            batch_size: 1,
            total_prefix_tokens: 1,
            prefix_sha256: prompt_cache_token_fingerprint(&[7]),
            layer_layout: descriptor.layer_layout,
            sink_tokens: 0,
            layer_prefix_offsets: vec![0],
            state_segments: descriptor.state_segments,
            topology: PromptCacheTopology::default(),
            application_namespace: Some(namespace.into()),
            blocks: vec![PromptCacheBlock {
                global_layer: 0,
                representation: CacheRepresentation::KeyValue,
                start: 0,
                end: 1,
                rank: None,
                shard: "block.safetensors".into(),
                first_array: "keys".into(),
                second_array: "values".into(),
                first_shape: vec![1, 1, 1, 1],
                second_shape: vec![1, 1, 1, 1],
                first_dtype: "Float32".into(),
                second_dtype: "Float32".into(),
                logical_bytes: 8,
                payload_sha256: hash_prompt_cache_shard_payload(
                    &generation.join("block.safetensors"),
                )
                .unwrap(),
            }],
            state_tensors: Vec::new(),
        };
        fs::write(
            generation.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        manifest
    }

    #[test]
    fn paged_options_require_finite_nonzero_limits() {
        assert!(PagedCacheOptions::new(0, 1, 1, 1).is_err());
        assert!(PagedCacheOptions::new(16, 0, 1, 1).is_err());
        assert!(PagedCacheOptions::new(16, 1, 1, 0).is_err());
        assert!(PagedCacheOptions::new(16, 1, 0, 1).is_ok());
    }

    #[test]
    fn prefix_hash_is_stable_and_order_sensitive() {
        assert_eq!(
            prompt_cache_token_fingerprint(&[1, 2, 3]),
            prompt_cache_token_fingerprint(&[1, 2, 3])
        );
        assert_ne!(
            prompt_cache_token_fingerprint(&[1, 2, 3]),
            prompt_cache_token_fingerprint(&[3, 2, 1])
        );
    }

    #[test]
    fn prompt_cache_topology_preserves_parallel_coordinates_and_rank_identity() {
        use crate::backend::{DeviceAssignment, MlxParallelContext};

        let topology =
            MlxParallelContext::for_rank(5, 2, 2, 2, DeviceAssignment::new(DeviceType::Cpu, 0))
                .unwrap();
        let cache_topology = crate::backend::cache::prompt_cache_topology(topology);

        assert_eq!(cache_topology.pipeline, Some((2, 1)));
        assert_eq!(cache_topology.tensor_parallel, Some((2, 0)));
        assert_eq!(cache_topology.expert_parallel, Some((2, 1)));
        assert_eq!(
            cache_topology.cache_rank_identity(),
            Some(CacheRankIdentity {
                pipeline_rank: Some(1),
                tensor_parallel_rank: Some(0),
                expert_parallel_rank: Some(1),
            })
        );

        let replicated =
            MlxParallelContext::for_rank(0, 1, 1, 1, DeviceAssignment::new(DeviceType::Cpu, 0))
                .unwrap();
        let replicated = crate::backend::cache::prompt_cache_topology(replicated);
        assert_eq!(replicated, PromptCacheTopology::default());
        assert_eq!(replicated.cache_rank_identity(), None);
    }

    #[test]
    fn ordered_layer_layout_round_trips_all_attention_patterns() {
        let schedules = [
            vec![None, None, None, None],
            vec![Some(4), Some(4), Some(4), Some(4)],
            vec![None, Some(4), None, Some(4)],
            vec![Some(3), None, None, Some(9)],
            vec![Some(2), Some(5), Some(11), None],
        ];
        for windows in schedules {
            let layout = key_value_layout(windows);
            let json = serde_json::to_string(&layout).unwrap();
            let restored: LayerSchedule<LayerCachePolicy> = serde_json::from_str(&json).unwrap();
            assert_eq!(restored, layout);
        }
    }

    #[test]
    fn attention_windows_reject_zero_negative_and_overflowing_sources() {
        assert!(AttentionPolicy::from_sliding_window(Some(0)).is_err());
        assert!(AttentionPolicy::from_sliding_window(Some(-1)).is_err());
        for json in [
            r#"{"sliding":{"window":0}}"#,
            r#"{"sliding":{"window":-1}}"#,
            r#"{"sliding":{"window":4294967296}}"#,
        ] {
            assert!(serde_json::from_str::<AttentionPolicy>(json).is_err());
        }
    }

    #[test]
    fn cache_identity_hashes_the_complete_ordered_layout() {
        let base = prompt_descriptor();
        let variants = [
            key_value_layout([Some(4)]),
            key_value_layout([Some(5)]),
            key_value_layout([None]),
            key_value_layout([None, Some(4)]),
            LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
            PromptCacheModelIdentity::compressed_layouts(1, 1, 1).unwrap(),
        ];
        let hashes = variants
            .into_iter()
            .map(|layer_layout| {
                stable_hash(&PromptCacheDescriptor {
                    layer_count: layer_layout.len(),
                    global_layer_end: layer_layout.len(),
                    state_segments: vec![eredu_core::cache::PromptCacheStateSegment::new(
                        "state",
                        0..layer_layout.len(),
                    )
                    .unwrap()],
                    layer_layout,
                    ..base.clone()
                })
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(hashes.len(), 6);

        let first = key_value_layout([None, Some(4)]);
        let reordered = key_value_layout([Some(4), None]);
        assert_ne!(stable_hash(&first), stable_hash(&reordered));
    }

    #[test]
    fn schema_v3_is_rejected_before_v4_fields_are_decoded() {
        let directory = tempfile::tempdir().unwrap();
        let generation = create_prompt_fixture_generation(directory.path());
        fs::write(
            generation.join("manifest.json"),
            br#"{"schema_version":3,"layer_layout":[]}"#,
        )
        .unwrap();
        assert!(matches!(
            inspect_prompt_cache(directory.path()),
            Err(PromptCachePersistenceError::PromptCache(
                PromptCacheError::UnsupportedSchema(3)
            ))
        ));
    }

    #[test]
    fn v7_layer_frontiers_validate_speculative_cache_coverage() {
        let directory = tempfile::tempdir().unwrap();
        let mut manifest = write_prompt_fixture(directory.path(), "speculative-frontier");
        manifest.layer_count = 2;
        manifest.global_layer_end = 2;
        manifest.state_segments =
            vec![eredu_core::cache::PromptCacheStateSegment::new("state", 0..2).unwrap()];
        manifest.total_prefix_tokens = 2;
        manifest.prefix_sha256 = prompt_cache_token_fingerprint(&[7, 8]);
        manifest.layer_layout = key_value_layout([None, None]);
        manifest.layer_prefix_offsets = vec![0, -1];

        let first = manifest.blocks[0].clone();
        let mut target_tail = first.clone();
        target_tail.start = 1;
        target_tail.end = 2;
        let mut draft = first;
        draft.global_layer = 1;
        manifest.blocks = vec![manifest.blocks[0].clone(), target_tail, draft];
        fs::write(
            prompt_fixture_manifest_path(directory.path()),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        assert_eq!(
            inspect_prompt_cache(directory.path())
                .unwrap()
                .layer_prefix_offsets,
            [0, -1]
        );

        manifest.layer_prefix_offsets = vec![0, 0];
        fs::write(
            prompt_fixture_manifest_path(directory.path()),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        assert!(inspect_prompt_cache(directory.path())
            .unwrap_err()
            .to_string()
            .contains("ends at 1, expected 2"));
    }

    #[test]
    fn v5_behavioral_state_layout_round_trips_and_changes_identity() {
        let incoherent = StateTensorPolicy::new(
            StateTensorRole::Recurrent,
            vec![StateTensorDimension::Scalar],
            StateTensorDtype::Float32,
            MutableStateResidency::AlwaysDeviceMutable,
        )
        .expect_err("large recurrent state cannot use the rolling-state lifecycle");
        assert!(incoherent
            .to_string()
            .contains("requires LayerScopedOffloadable"));
        let convolution = StateTensorPolicy::new(
            StateTensorRole::Convolution { slot: 0 },
            vec![
                StateTensorDimension::Batch,
                StateTensorDimension::fixed(3).unwrap(),
                StateTensorDimension::fixed(4).unwrap(),
            ],
            StateTensorDtype::Floating,
            MutableStateResidency::AlwaysDeviceMutable,
        )
        .unwrap();
        let recurrent = StateTensorPolicy::new(
            StateTensorRole::Recurrent,
            vec![
                StateTensorDimension::Batch,
                StateTensorDimension::fixed(4).unwrap(),
            ],
            StateTensorDtype::Float32,
            MutableStateResidency::LayerScopedOffloadable,
        )
        .unwrap();
        let layouts = [
            LayerSchedule::new(
                1,
                vec![LayerCachePolicy::fixed_only(vec![convolution.clone()]).unwrap()],
            )
            .unwrap(),
            LayerSchedule::new(
                1,
                vec![LayerCachePolicy::fixed_only(vec![recurrent.clone()]).unwrap()],
            )
            .unwrap(),
            LayerSchedule::new(
                1,
                vec![LayerCachePolicy::key_value_with_fixed_state(
                    AttentionPolicy::Full,
                    1,
                    1,
                    vec![convolution.clone()],
                )
                .unwrap()],
            )
            .unwrap(),
        ];
        let hashes = layouts
            .iter()
            .map(|layout| {
                let json = serde_json::to_string(layout).unwrap();
                let restored: LayerSchedule<LayerCachePolicy> =
                    serde_json::from_str(&json).unwrap();
                assert_eq!(&restored, layout);
                stable_hash(layout)
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(hashes.len(), layouts.len());
        assert_eq!(
            convolution.residency_class(),
            StateResidencyClass::AlwaysDeviceMutable
        );
        assert_eq!(
            recurrent.residency_class(),
            StateResidencyClass::LayerScopedOffloadable
        );
        assert_eq!(
            layouts[2].get(0).unwrap().attention_residency_class(),
            Some(StateResidencyClass::SealablePaged)
        );
    }

    fn write_fixed_state_fixture(root: &Path) -> PromptCacheManifest {
        let generation = create_prompt_fixture_generation(root);
        let values = (0..12)
            .flat_map(|value| (value as f32).to_le_bytes())
            .collect::<Vec<_>>();
        let view = TensorView::new(StoredDtype::F32, vec![1, 3, 4], &values).unwrap();
        let shard = generation.join("state.safetensors");
        serialize_to_file([("state", view)], None, &shard).unwrap();
        let policy = StateTensorPolicy::new(
            StateTensorRole::Convolution { slot: 0 },
            vec![
                StateTensorDimension::Batch,
                StateTensorDimension::fixed(3).unwrap(),
                StateTensorDimension::fixed(4).unwrap(),
            ],
            StateTensorDtype::Floating,
            MutableStateResidency::AlwaysDeviceMutable,
        )
        .unwrap();
        let manifest = PromptCacheManifest {
            schema_version: PROMPT_CACHE_SCHEMA_VERSION,
            model_family: "fixed".into(),
            effective_model_type: "fixed".into(),
            checkpoint_fingerprint: "checkpoint".into(),
            prefix_content_fingerprint: "state:prefix".into(),
            architecture_fingerprint: "architecture".into(),
            layer_count: 1,
            global_layer_start: 0,
            global_layer_end: 1,
            block_size_tokens: 1,
            batch_size: 1,
            total_prefix_tokens: 1,
            prefix_sha256: prompt_cache_token_fingerprint(&[7]),
            layer_layout: LayerSchedule::new(
                1,
                vec![LayerCachePolicy::fixed_only(vec![policy]).unwrap()],
            )
            .unwrap(),
            layer_prefix_offsets: vec![0],
            state_segments: vec![
                eredu_core::cache::PromptCacheStateSegment::new("state", 0..1).unwrap(),
            ],
            sink_tokens: 0,
            topology: PromptCacheTopology::default(),
            application_namespace: None,
            blocks: Vec::new(),
            state_tensors: vec![PromptCacheStateTensor {
                owner: StateTensorOwner::Layer(0),
                role: StateTensorRole::Convolution { slot: 0 },
                shard: "state.safetensors".into(),
                array: "state".into(),
                shape: vec![1, 3, 4],
                dtype: "Float32".into(),
                logical_bytes: 48,
                payload_sha256: hash_prompt_cache_shard_payload(&shard).unwrap(),
            }],
        };
        fs::write(
            generation.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        manifest
    }

    #[test]
    fn v4_fixed_state_validation_rejects_missing_reordered_kind_and_geometry() {
        let directory = tempfile::tempdir().unwrap();
        let base = write_fixed_state_fixture(directory.path());
        inspect_prompt_cache(directory.path()).unwrap();
        let write = |manifest: &PromptCacheManifest| {
            fs::write(
                prompt_fixture_manifest_path(directory.path()),
                serde_json::to_vec(manifest).unwrap(),
            )
            .unwrap();
            inspect_prompt_cache(directory.path())
                .unwrap_err()
                .to_string()
        };

        let mut missing = base.clone();
        missing.state_tensors.clear();
        assert!(write(&missing).contains("count"));

        let mut unexpected = base.clone();
        unexpected.state_tensors[0].role = StateTensorRole::Recurrent;
        assert!(write(&unexpected).contains("does not match its policy"));

        let mut geometry = base.clone();
        geometry.state_tensors[0].shape = vec![1, 2, 4];
        assert!(write(&geometry).contains("does not match its policy"));

        let mut dtype = base;
        dtype.state_tensors[0].dtype = "Int32".into();
        assert!(write(&dtype).contains("does not match its policy"));
    }

    #[test]
    fn manifest_rejects_reordered_duplicate_missing_and_unexpected_layers() {
        let directory = tempfile::tempdir().unwrap();
        let base = write_prompt_fixture(directory.path(), "ordered-layout");
        let mut two_layers = base.clone();
        two_layers.layer_count = 2;
        two_layers.global_layer_end = 2;
        two_layers.state_segments =
            vec![eredu_core::cache::PromptCacheStateSegment::new("state", 0..2).unwrap()];
        two_layers.layer_layout = key_value_layout([None, Some(7)]);
        two_layers.layer_prefix_offsets = vec![0, 0];
        let mut second = two_layers.blocks[0].clone();
        second.global_layer = 1;
        two_layers.blocks.push(second);

        let write = |manifest: &PromptCacheManifest| {
            fs::write(
                prompt_fixture_manifest_path(directory.path()),
                serde_json::to_vec(manifest).unwrap(),
            )
            .unwrap();
            inspect_prompt_cache(directory.path()).unwrap_err()
        };

        let mut reordered = two_layers.clone();
        reordered.blocks.reverse();
        assert!(write(&reordered).to_string().contains("reordered"));

        let mut duplicate = two_layers.clone();
        duplicate.blocks.insert(1, duplicate.blocks[0].clone());
        assert!(write(&duplicate).to_string().contains("duplicated"));

        let mut missing = two_layers.clone();
        missing.blocks.pop();
        assert!(write(&missing).to_string().contains("missing blocks"));

        let mut unexpected = two_layers.clone();
        unexpected.blocks[1].global_layer = 2;
        assert!(write(&unexpected)
            .to_string()
            .contains("outside the owned range"));
    }

    #[test]
    fn manifest_rejects_policy_payload_kind_and_geometry_mismatches() {
        let directory = tempfile::tempdir().unwrap();
        let base = write_prompt_fixture(directory.path(), "policy-mismatch");
        let mut kind = base.clone();
        kind.layer_layout = PromptCacheModelIdentity::compressed_layouts(1, 1, 1).unwrap();
        fs::write(
            prompt_fixture_manifest_path(directory.path()),
            serde_json::to_vec(&kind).unwrap(),
        )
        .unwrap();
        assert!(inspect_prompt_cache(directory.path())
            .unwrap_err()
            .to_string()
            .contains("does not match its policy"));

        let mut geometry = base;
        geometry.layer_layout = PromptCacheModelIdentity::key_value_layouts([None], 2, 1).unwrap();
        fs::write(
            prompt_fixture_manifest_path(directory.path()),
            serde_json::to_vec(&geometry).unwrap(),
        )
        .unwrap();
        assert!(inspect_prompt_cache(directory.path())
            .unwrap_err()
            .to_string()
            .contains("does not match its policy"));
    }

    #[test]
    fn same_length_prompt_payload_corruption_is_rejected_before_array_conversion() {
        let directory = tempfile::tempdir().unwrap();
        let manifest = write_prompt_fixture(directory.path(), "payload-checksum");
        let shard = prompt_fixture_root(directory.path()).join(&manifest.blocks[0].shard);
        let mut bytes = fs::read(&shard).unwrap();
        let final_byte = bytes.last_mut().expect("fixture shard has a payload");
        *final_byte ^= 0x01;
        fs::write(&shard, &bytes).unwrap();

        // Header-only inspection remains valid because metadata and length did
        // not change. The mapped payload gate must still reject the shard.
        inspect_prompt_cache(directory.path()).unwrap();
        let location = DiskLocation {
            path: shard.clone(),
            first_name: "keys".into(),
            second_name: "values".into(),
            persistent: true,
            mapped: Some(map_prompt_cache_shard(&shard).unwrap()),
            payload_sha256: Some(manifest.blocks[0].payload_sha256.clone()),
            payload_verification: Arc::new(OnceLock::new()),
        };
        let error = verify_disk_payload(&location).unwrap_err();
        assert!(error.to_string().contains("payload SHA-256 mismatch"));
    }

    #[test]
    fn imported_prompt_shards_are_actually_mapped_and_retained() {
        let directory = tempfile::tempdir().unwrap();
        write_prompt_fixture(directory.path(), "mapped");
        let options = PagedCacheOptions::new(1, 64, 64, 1).unwrap();
        let (manager, _) = open_prompt_cache(
            directory.path(),
            &prompt_descriptor(),
            &prompt_model_identity(),
            &[7],
            options,
        )
        .unwrap();
        let state = manager.lock().unwrap();
        assert_eq!(state.telemetry.report.imported_mapped_shards, 1);
        assert!(state.blocks.values().all(|record| record
            .disk()
            .and_then(|location| location.mapped.as_ref())
            .is_some()));
        for record in state.blocks.values() {
            verify_disk_payload(record.disk().unwrap()).unwrap();
        }
    }

    #[test]
    fn loaded_model_identity_rejects_a_forged_caller_descriptor() {
        let mut descriptor = prompt_descriptor();
        descriptor.layer_count = 2;
        descriptor.global_layer_end = 2;
        let loaded_model = PromptCacheModelIdentity {
            model_family: "decoder".into(),
            effective_model_type: "decoder".into(),
            architecture_fingerprint: descriptor.architecture_fingerprint.clone(),
            layer_count: 1,
            global_layer_start: 0,
            global_layer_end: 1,
            sink_tokens: 0,
            layer_prefix_offsets: vec![0],
            state_segments: vec![
                eredu_core::cache::PromptCacheStateSegment::new("state", 0..1).unwrap(),
            ],
            topology: PromptCacheTopology::default(),
            layer_layout: PromptCacheModelIdentity::key_value_layouts([None], 1, 1).unwrap(),
        };
        assert!(matches!(
            validate_prompt_cache_model_identity(&descriptor, &loaded_model),
            Err(PromptCacheError::Incompatible(_))
        ));
    }

    #[test]
    fn loaded_model_identity_rejects_a_forged_architecture_fingerprint() {
        let mut descriptor = prompt_descriptor();
        let loaded_model = PromptCacheModelIdentity {
            model_family: descriptor.model_family.clone(),
            effective_model_type: descriptor.effective_model_type.clone(),
            architecture_fingerprint: "sha256:derived-from-loaded-model".into(),
            layer_count: descriptor.layer_count,
            global_layer_start: descriptor.global_layer_start,
            global_layer_end: descriptor.global_layer_end,
            sink_tokens: descriptor.sink_tokens,
            layer_prefix_offsets: descriptor.layer_prefix_offsets.clone(),
            state_segments: descriptor.state_segments.clone(),
            topology: descriptor.topology.clone(),
            layer_layout: descriptor.layer_layout.clone(),
        };
        descriptor.architecture_fingerprint = "sha256:caller-repeated-stale-value".into();
        let error = validate_prompt_cache_model_identity(&descriptor, &loaded_model).unwrap_err();
        assert!(error.to_string().contains("architecture_fingerprint"));
    }

    #[test]
    fn loaded_model_identity_rejects_a_forged_layer_frontier() {
        let mut descriptor = prompt_descriptor();
        let loaded_model = prompt_model_identity();
        descriptor.layer_prefix_offsets = vec![-1];
        let error = validate_prompt_cache_model_identity(&descriptor, &loaded_model).unwrap_err();
        assert!(error.to_string().contains("layer_prefix_offsets"));
    }

    #[test]
    fn prompt_load_rejects_model_incompatible_key_value_dimensions() {
        let directory = tempfile::tempdir().unwrap();
        let mut manifest = write_prompt_fixture(directory.path(), "wrong-kv-dimensions");
        let keys = vec![0u8; 32];
        let values = vec![0u8; 32];
        let key_view = TensorView::new(StoredDtype::F32, vec![1, 2, 1, 4], &keys).unwrap();
        let value_view = TensorView::new(StoredDtype::F32, vec![1, 2, 1, 4], &values).unwrap();
        serialize_to_file(
            [("keys", key_view), ("values", value_view)],
            None,
            &prompt_fixture_root(directory.path()).join("block.safetensors"),
        )
        .unwrap();
        manifest.blocks[0].first_shape = vec![1, 2, 1, 4];
        manifest.blocks[0].second_shape = vec![1, 2, 1, 4];
        manifest.blocks[0].logical_bytes = 64;
        fs::write(
            prompt_fixture_manifest_path(directory.path()),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        let error = open_prompt_cache(
            directory.path(),
            &prompt_descriptor(),
            &prompt_model_identity(),
            &[7],
            PagedCacheOptions::new(1, 64, 64, 1).unwrap(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("does not match its policy"));
    }

    #[test]
    fn prompt_load_rejects_model_incompatible_layer_representation() {
        let directory = tempfile::tempdir().unwrap();
        let mut manifest = write_prompt_fixture(directory.path(), "wrong-representation");
        let latent = vec![0u8; 16];
        let rotary = vec![0u8; 8];
        let latent_view = TensorView::new(StoredDtype::F32, vec![1, 1, 4], &latent).unwrap();
        let rotary_view = TensorView::new(StoredDtype::F32, vec![1, 1, 2], &rotary).unwrap();
        serialize_to_file(
            [("latent", latent_view), ("rotary_key", rotary_view)],
            None,
            &prompt_fixture_root(directory.path()).join("block.safetensors"),
        )
        .unwrap();
        manifest.blocks[0].representation = CacheRepresentation::CompressedLatentRotary;
        manifest.blocks[0].first_array = "latent".into();
        manifest.blocks[0].second_array = "rotary_key".into();
        manifest.blocks[0].first_shape = vec![1, 1, 4];
        manifest.blocks[0].second_shape = vec![1, 1, 2];
        manifest.blocks[0].logical_bytes = 24;
        fs::write(
            prompt_fixture_manifest_path(directory.path()),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        let error = open_prompt_cache(
            directory.path(),
            &prompt_descriptor(),
            &prompt_model_identity(),
            &[7],
            PagedCacheOptions::new(1, 64, 64, 1).unwrap(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("does not match its policy"));
    }

    #[test]
    fn model_reset_surfaces_propagate_paged_clear_failures() {
        use crate::{
            backend::runtime::cache::{state::MlxKeyValueState, PagedKeyValueCache},
            composition::mlx::distributed::pipeline::{
                PipelineCache, PipelineKeyValueCache, PipelineLayerCache,
            },
        };

        let manager = manager_with_leased_block();
        let layout = eredu_runtime::StateLayout::new(
            eredu_core::LayerSchedule::new(
                1,
                vec![eredu_core::cache::LayerCachePolicy::key_value(
                    eredu_core::AttentionPolicy::Full,
                    1,
                    1,
                )
                .unwrap()],
            )
            .unwrap(),
        )
        .unwrap();
        let mut state = MlxKeyValueState::paged(layout.clone(), manager.clone(), None).unwrap();
        assert!(state.clear().is_err());
        assert_eq!(manager.lock().unwrap().blocks.len(), 1);

        let manager = manager_with_leased_block();
        let mut pipeline = PipelineCache::new(
            eredu_architectures::ModelKind::Llama,
            vec![PipelineLayerCache::KeyValue {
                global_layer: 0,
                cache: PipelineKeyValueCache::Paged(
                    PagedKeyValueCache::new(manager.clone(), 0, None).unwrap(),
                ),
                slots: Vec::new(),
            }],
        );
        assert!(pipeline.reset().is_err());
        assert_eq!(manager.lock().unwrap().blocks.len(), 1);
    }

    #[test]
    fn disk_worker_coalesces_duplicate_in_flight_reads() {
        let directory = tempfile::tempdir().unwrap();
        let worker = DiskWorker::new(1).unwrap();
        let id = disk_test_id(0);
        let location = missing_location(directory.path(), "missing.safetensors");
        let first = worker
            .prepare_read(3, &id, &location, CacheRepresentation::KeyValue)
            .unwrap();
        let ticket = first.ticket.clone();
        let second = worker
            .prepare_read(3, &id, &location, CacheRepresentation::KeyValue)
            .unwrap();
        assert!(second.inner.joined);
        let second_ticket = second.ticket.clone();
        first.enqueue().unwrap();
        second.enqueue().unwrap();
        assert!(ticket.wait().is_err());
        assert!(second_ticket.wait().is_err());
        assert!(ticket.shares_completion_with(&second_ticket));
        worker.retire(&ticket);
    }

    #[test]
    fn disk_worker_applies_backpressure_only_outside_submission() {
        let directory = tempfile::tempdir().unwrap();
        let worker = DiskWorker::new(1).unwrap();
        let (first_started_tx, first_started_rx) = mpsc::channel();
        let (first_release_tx, first_release_rx) = mpsc::channel();
        let first = worker
            .prepare(
                CacheIoOperationKey {
                    generation: 0,
                    id: disk_test_id(0),
                    kind: CacheIoOperationKind::Read,
                },
                DiskTask::Pause {
                    started: first_started_tx,
                    release: first_release_rx,
                },
            )
            .unwrap();
        let first_ticket = first.ticket.clone();
        first.enqueue().unwrap();
        first_started_rx.recv().unwrap();

        let (second_started_tx, second_started_rx) = mpsc::channel();
        let (second_release_tx, second_release_rx) = mpsc::channel();
        let second = worker
            .prepare(
                CacheIoOperationKey {
                    generation: 0,
                    id: disk_test_id(1),
                    kind: CacheIoOperationKind::Read,
                },
                DiskTask::Pause {
                    started: second_started_tx,
                    release: second_release_rx,
                },
            )
            .unwrap();
        let second_ticket = second.ticket.clone();
        second.enqueue().unwrap();

        let third = worker
            .prepare_read(
                0,
                &disk_test_id(2),
                &missing_location(directory.path(), "third.safetensors"),
                CacheRepresentation::KeyValue,
            )
            .unwrap();
        let third_ticket = third.ticket.clone();
        let (outcome_tx, outcome_rx) = mpsc::channel();
        let enqueue_thread = thread::spawn(move || outcome_tx.send(third.enqueue()).unwrap());
        assert!(outcome_rx.recv_timeout(Duration::from_millis(20)).is_err());

        first_release_tx.send(()).unwrap();
        second_started_rx.recv().unwrap();
        let outcome = outcome_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap();
        assert!(outcome.backpressure);
        assert_eq!(outcome.peak_occupancy, 1);
        second_release_tx.send(()).unwrap();
        enqueue_thread.join().unwrap();
        assert!(matches!(first_ticket.wait().unwrap(), DiskResult::Test));
        assert!(matches!(second_ticket.wait().unwrap(), DiskResult::Test));
        assert!(third_ticket.wait().is_err());
        worker.retire(&first_ticket);
        worker.retire(&second_ticket);
        worker.retire(&third_ticket);
    }

    #[test]
    fn disk_worker_cancels_queued_generation_work() {
        let directory = tempfile::tempdir().unwrap();
        let worker = DiskWorker::new(1).unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let blocker = worker
            .prepare(
                CacheIoOperationKey {
                    generation: 8,
                    id: disk_test_id(0),
                    kind: CacheIoOperationKind::Read,
                },
                DiskTask::Pause {
                    started: started_tx,
                    release: release_rx,
                },
            )
            .unwrap();
        let blocker_ticket = blocker.ticket.clone();
        blocker.enqueue().unwrap();
        started_rx.recv().unwrap();

        let cancelled = worker
            .prepare_read(
                8,
                &disk_test_id(1),
                &missing_location(directory.path(), "cancelled.safetensors"),
                CacheRepresentation::KeyValue,
            )
            .unwrap();
        let cancelled_ticket = cancelled.ticket.clone();
        cancelled.enqueue().unwrap();
        assert!(cancelled_ticket.cancel());
        assert!(matches!(
            cancelled_ticket.wait(),
            Err(CacheResidencyError::DiskOperationCancelled { generation: 8 })
        ));
        release_tx.send(()).unwrap();
        assert!(matches!(blocker_ticket.wait().unwrap(), DiskResult::Test));
        worker.retire(&blocker_ticket);
        worker.retire(&cancelled_ticket);
    }

    #[test]
    fn cancellation_wakes_a_backpressured_submitter() {
        let directory = tempfile::tempdir().unwrap();
        let worker = DiskWorker::new(1).unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let blocker = worker
            .prepare(
                CacheIoOperationKey {
                    generation: 4,
                    id: disk_test_id(0),
                    kind: CacheIoOperationKind::Read,
                },
                DiskTask::Pause {
                    started: started_tx,
                    release: release_rx,
                },
            )
            .unwrap();
        let blocker_ticket = blocker.ticket.clone();
        blocker.enqueue().unwrap();
        started_rx.recv().unwrap();

        let queued = worker
            .prepare_read(
                4,
                &disk_test_id(1),
                &missing_location(directory.path(), "queued.safetensors"),
                CacheRepresentation::KeyValue,
            )
            .unwrap();
        let queued_ticket = queued.ticket.clone();
        queued.enqueue().unwrap();
        let blocked = worker
            .prepare_read(
                4,
                &disk_test_id(2),
                &missing_location(directory.path(), "blocked.safetensors"),
                CacheRepresentation::KeyValue,
            )
            .unwrap();
        let blocked_ticket = blocked.ticket.clone();
        let (outcome_tx, outcome_rx) = mpsc::channel();
        let enqueue_thread = thread::spawn(move || outcome_tx.send(blocked.enqueue()).unwrap());
        assert!(outcome_rx.recv_timeout(Duration::from_millis(20)).is_err());

        assert!(blocked_ticket.cancel());
        let outcome = outcome_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap();
        assert!(outcome.backpressure);
        enqueue_thread.join().unwrap();
        assert!(matches!(
            blocked_ticket.wait(),
            Err(CacheResidencyError::DiskOperationCancelled { generation: 4 })
        ));
        release_tx.send(()).unwrap();
        assert!(matches!(blocker_ticket.wait().unwrap(), DiskResult::Test));
        assert!(queued_ticket.wait().is_err());
        worker.retire(&blocker_ticket);
        worker.retire(&queued_ticket);
        worker.retire(&blocked_ticket);
    }

    #[test]
    fn disk_worker_reports_operation_panics_and_keeps_running() {
        let directory = tempfile::tempdir().unwrap();
        let worker = DiskWorker::new(1).unwrap();
        let panicking = worker
            .prepare(
                CacheIoOperationKey {
                    generation: 2,
                    id: disk_test_id(0),
                    kind: CacheIoOperationKind::Read,
                },
                DiskTask::Panic,
            )
            .unwrap();
        let panicking_ticket = panicking.ticket.clone();
        panicking.enqueue().unwrap();
        assert!(matches!(
            panicking_ticket.wait(),
            Err(CacheResidencyError::Runtime(message))
                if message.contains("operation panicked")
        ));
        worker.retire(&panicking_ticket);

        let following = worker
            .prepare_read(
                2,
                &disk_test_id(1),
                &missing_location(directory.path(), "following.safetensors"),
                CacheRepresentation::KeyValue,
            )
            .unwrap();
        let following_ticket = following.ticket.clone();
        following.enqueue().unwrap();
        assert!(following_ticket.wait().is_err());
        worker.retire(&following_ticket);
    }

    #[test]
    fn background_write_failures_surface_on_the_next_foreground_operation() {
        let manager = CacheResidencyManager::new(
            PagedCacheOptions::new(1, 64, 64, 1)
                .unwrap()
                .with_full_attention(true),
        )
        .unwrap();
        let worker = manager.inner.disk_worker.as_ref().unwrap();
        let id = disk_test_id(0);
        let key = CacheIoOperationKey {
            generation: 0,
            id: id.clone(),
            kind: CacheIoOperationKind::Write,
        };
        let submission = worker.prepare(key.clone(), DiskTask::Panic).unwrap();
        let ticket = submission.ticket.clone();
        insert_test_record(
            &mut manager.lock().unwrap(),
            CacheBlockRecord {
                physical: test_host_writing(test_host_block(), ticket.clone()),
                bytes: 0,
                shapes: [vec![1], vec![1]],
                dtypes: ["Float32".into(), "Float32".into()],
                imported: false,
            },
            false,
            0,
        );
        DiskWriteCommit {
            state: Arc::downgrade(&manager.inner.state),
            key,
            reservation_id: 0,
            armed: true,
        }
        .reconcile(&Err(CacheResidencyError::Runtime(
            "injected asynchronous write failure".into(),
        )));

        let error = manager.set_tail_state(0, 0, 0).unwrap_err();
        assert!(error
            .to_string()
            .contains("injected asynchronous write failure"));
        let report = manager.report().unwrap();
        assert_eq!(report.failures, 1);
        assert_eq!(report.per_layer.len(), 1);
        assert_eq!(report.per_layer[0].global_layer, 0);
        assert_eq!(report.per_layer[0].stats.failures, report.failures);
        worker.retire(&ticket);
    }

    #[test]
    fn promoted_and_cancelled_writes_retain_host_reservations_until_release() {
        let directory = tempfile::tempdir().unwrap();
        let host_block = test_host_block();
        let host_capacity = host_block.capacity().unwrap();
        let pool = CacheResidencyPool::new(
            CachePoolLimits::new(16, host_capacity, host_capacity, 1024).unwrap(),
        );
        let options = PagedCacheOptions::new(1, 16, host_capacity, 1)
            .unwrap()
            .with_live_disk(directory.path(), 1024, 1)
            .unwrap()
            .with_pool(pool.clone())
            .unwrap();
        let manager = CacheResidencyManager::new(options).unwrap();
        let worker = manager.inner.disk_worker.as_ref().unwrap();
        let id = disk_test_id(0);
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let key = CacheIoOperationKey {
            generation: 0,
            id: id.clone(),
            kind: CacheIoOperationKind::Write,
        };
        let submission = worker
            .prepare(
                key.clone(),
                DiskTask::PauseWrite {
                    started: started_tx,
                    release: release_rx,
                    commit: Some(DiskWriteCommit {
                        state: Arc::downgrade(&manager.inner.state),
                        key: key.clone(),
                        reservation_id: 7,
                        armed: true,
                    }),
                },
            )
            .unwrap();
        let ticket = submission.ticket.clone();
        submission.enqueue().unwrap();
        started_rx.recv().unwrap();
        {
            let mut state = manager.lock().unwrap();
            state.host_write_reservations.insert(
                key,
                HostWriteReservation {
                    reservation_id: 7,
                    global_layer: id.global_layer,
                    logical_bytes: 16,
                    host_capacity,
                    ticket: ticket.clone(),
                },
            );
            insert_test_record(
                &mut state,
                CacheBlockRecord {
                    physical: test_host_writing(host_block, ticket.clone()),
                    bytes: 16,
                    shapes: [vec![1], vec![1]],
                    dtypes: ["Float32".into(), "Float32".into()],
                    imported: false,
                },
                false,
                0,
            );
        }

        let report = manager.report().unwrap();
        assert_eq!(report.current_host_bytes, host_capacity);
        assert_eq!(report.in_flight_write_bytes, host_capacity);
        assert_eq!(report.host_blocks, 1);
        let aggregate = pool.report().unwrap();
        assert_eq!(aggregate.current_host_bytes, host_capacity);
        assert_eq!(aggregate.current_transfer_in_flight_bytes, host_capacity);
        assert_eq!(aggregate.current_disk_bytes, 16);

        // A pending host write is encoded only by the Host variant. Promotion
        // waits for that write rather than creating an incoherent device block
        // with an attached host-write operation.
        assert_eq!(
            manager
                .lock()
                .unwrap()
                .blocks
                .get(&id)
                .unwrap()
                .physical
                .phase(),
            CacheStoragePhase::HostWriting
        );

        let clear_manager = manager.clone();
        let (cleared_tx, cleared_rx) = mpsc::channel();
        let clear_thread = thread::spawn(move || {
            cleared_tx.send(clear_manager.clear()).unwrap();
        });
        assert!(matches!(
            ticket.wait(),
            Err(CacheResidencyError::DiskOperationCancelled { generation: 0 })
        ));
        assert!(matches!(
            cleared_rx.recv_timeout(Duration::from_millis(20)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        let report = manager.report().unwrap();
        assert_eq!(report.cancellations, 1);
        assert_eq!(report.host_blocks, 0);
        assert_eq!(report.current_host_bytes, host_capacity);
        assert_eq!(report.in_flight_write_bytes, host_capacity);
        let aggregate = pool.report().unwrap();
        assert_eq!(aggregate.current_host_bytes, host_capacity);
        assert_eq!(aggregate.current_transfer_in_flight_bytes, host_capacity);
        assert_eq!(aggregate.current_disk_bytes, 16);
        release_tx.send(()).unwrap();
        cleared_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("clear did not finish after the write released its arrays")
            .unwrap();
        clear_thread.join().unwrap();
        let report = manager.report().unwrap();
        assert_eq!(report.current_host_bytes, 0);
        assert_eq!(report.in_flight_write_bytes, 0);
        let aggregate = pool.report().unwrap();
        assert_eq!(aggregate.current_host_bytes, 0);
        assert_eq!(aggregate.current_transfer_in_flight_bytes, 0);
        assert_eq!(aggregate.current_disk_bytes, 0);
    }

    #[test]
    fn disk_backed_device_blocks_bypass_a_zero_host_budget() {
        let directory = tempfile::tempdir().unwrap();
        let manager =
            CacheResidencyManager::new(PagedCacheOptions::new(1, 16, 0, 1).unwrap()).unwrap();
        let older = disk_test_id(0);
        let recent = disk_test_id(2);
        {
            let mut state = manager.lock().unwrap();
            for (id, disk) in [
                (
                    older.clone(),
                    Some(missing_location(directory.path(), "older.safetensors")),
                ),
                (recent.clone(), None),
            ] {
                insert_test_record(
                    &mut state,
                    CacheBlockRecord {
                        physical: MlxCacheBlockStorage::device(
                            id.clone(),
                            test_device_block(),
                            disk,
                        ),
                        bytes: 16,
                        shapes: [vec![1], vec![1]],
                        dtypes: ["Float32".into(), "Float32".into()],
                        imported: false,
                    },
                    false,
                    0,
                );
            }
        }

        manager.rebalance(None, false).unwrap();
        let state = manager.lock().unwrap();
        assert_eq!(state.blocks.get(&older).unwrap().tier(), CacheTier::Disk);
        assert_eq!(state.blocks.get(&recent).unwrap().tier(), CacheTier::Device);
        assert_eq!(state.telemetry.report.current_host_bytes, 0);
        assert_eq!(state.telemetry.report.current_device_bytes, 16);
        assert_eq!(state.telemetry.report.current_disk_bytes, 16);
    }

    #[test]
    fn per_layer_residency_report_is_bounded_and_losslessly_aggregated() {
        let manager =
            CacheResidencyManager::new(PagedCacheOptions::new(1, u64::MAX, u64::MAX, 1).unwrap())
                .unwrap();
        let layer_count = CACHE_RESIDENCY_LAYER_REPORT_LIMIT + 3;
        {
            let mut state = manager.lock().unwrap();
            for global_layer in 0..layer_count {
                let representation = if global_layer % 2 == 0 {
                    CacheRepresentation::KeyValue
                } else {
                    CacheRepresentation::CompressedLatentRotary
                };
                let tier = match global_layer % 3 {
                    0 => CacheTier::Device,
                    1 => CacheTier::Host,
                    _ => CacheTier::Disk,
                };
                let id = CacheBlockId {
                    session_id: manager.session_id,
                    global_layer,
                    representation,
                    start: 0,
                    end: global_layer as i64 + 1,
                    rank: None,
                };
                let physical = match tier {
                    CacheTier::Device => {
                        MlxCacheBlockStorage::device(id.clone(), test_device_block(), None)
                    }
                    CacheTier::Host => {
                        MlxCacheBlockStorage::host(id.clone(), test_host_block(), None)
                    }
                    CacheTier::Disk => MlxCacheBlockStorage::disk(
                        id.clone(),
                        missing_location(
                            Path::new("/tmp/eredu-mlx-cache-report-test"),
                            &format!("layer-{global_layer}.safetensors"),
                        ),
                    ),
                };
                insert_test_record(
                    &mut state,
                    CacheBlockRecord {
                        physical,
                        bytes: global_layer as u64 + 1,
                        shapes: [vec![1], vec![1]],
                        dtypes: ["Float32".into(), "Float32".into()],
                        imported: false,
                    },
                    global_layer % 5 == 0,
                    0,
                );
                state.lifecycle.set_tail(
                    global_layer,
                    MutableCacheTail {
                        bytes: 2,
                        end: global_layer as i64 + 1,
                    },
                );
            }
        }

        let report = manager.report().unwrap();
        assert_eq!(report.per_layer.len(), CACHE_RESIDENCY_LAYER_REPORT_LIMIT);
        assert_eq!(report.per_layer_overflow_layers, 3);
        assert_eq!(
            report
                .per_layer
                .iter()
                .map(|layer| layer.global_layer)
                .collect::<Vec<_>>(),
            (0..CACHE_RESIDENCY_LAYER_REPORT_LIMIT).collect::<Vec<_>>()
        );

        let mut aggregate = CacheLayerResidencyStats::default();
        for layer in &report.per_layer {
            aggregate.accumulate(&layer.stats);
        }
        aggregate.accumulate(&report.per_layer_overflow);
        assert_eq!(aggregate.key_value_blocks, report.key_value_blocks);
        assert_eq!(
            aggregate.compressed_latent_blocks,
            report.compressed_latent_blocks
        );
        assert_eq!(aggregate.device_blocks, report.device_blocks);
        assert_eq!(aggregate.host_blocks, report.host_blocks);
        assert_eq!(aggregate.disk_blocks, report.disk_blocks);
        assert_eq!(aggregate.current_device_bytes, report.current_device_bytes);
        assert_eq!(aggregate.current_host_bytes, report.current_host_bytes);
        assert_eq!(aggregate.current_disk_bytes, report.current_disk_bytes);
        assert_eq!(aggregate.mutable_tail_bytes, report.mutable_tail_bytes);
        assert_eq!(
            aggregate.protected_recent_blocks,
            report.protected_recent_blocks
        );
        assert_eq!(
            aggregate.protected_prefix_blocks,
            report.protected_prefix_blocks
        );
        assert_eq!(report.logical_cached_tokens, layer_count as u64);
        assert_eq!(
            report.per_layer_overflow.logical_cached_tokens,
            ((CACHE_RESIDENCY_LAYER_REPORT_LIMIT + 1)..=layer_count)
                .map(|tokens| tokens as u64)
                .sum::<u64>()
        );
    }

    #[test]
    fn per_layer_cumulative_attention_is_bounded_and_survives_clear() {
        let manager =
            CacheResidencyManager::new(PagedCacheOptions::new(1, u64::MAX, u64::MAX, 1).unwrap())
                .unwrap();
        let layer_count = CACHE_RESIDENCY_LAYER_REPORT_LIMIT + 3;
        for global_layer in 0..layer_count {
            manager
                .record_attention_scan(
                    global_layer,
                    global_layer % 2 == 0,
                    1,
                    global_layer as u64 + 1,
                    global_layer as u64 + 7,
                )
                .unwrap();
        }

        let report = manager.report().unwrap();
        assert_eq!(report.per_layer.len(), CACHE_RESIDENCY_LAYER_REPORT_LIMIT);
        assert_eq!(report.per_layer_overflow_layers, 0);
        assert_eq!(
            report
                .per_layer
                .iter()
                .map(|layer| layer.global_layer)
                .collect::<Vec<_>>(),
            (0..CACHE_RESIDENCY_LAYER_REPORT_LIMIT).collect::<Vec<_>>()
        );
        let mut aggregate = CacheLayerResidencyStats::default();
        for layer in &report.per_layer {
            aggregate.accumulate(&layer.stats);
        }
        aggregate.accumulate(&report.per_layer_overflow);
        assert_eq!(
            aggregate.prefill_full_attention_blocks,
            report.prefill_full_attention_blocks
        );
        assert_eq!(
            aggregate.prefill_full_attention_bytes,
            report.prefill_full_attention_bytes
        );
        assert_eq!(
            aggregate.decode_full_attention_blocks,
            report.decode_full_attention_blocks
        );
        assert_eq!(
            aggregate.decode_full_attention_bytes,
            report.decode_full_attention_bytes
        );
        assert_eq!(
            aggregate.attention_scratch_peak_bytes,
            report.attention_scratch_peak_bytes
        );
        assert_eq!(report.per_layer_overflow.prefill_full_attention_blocks, 2);
        assert_eq!(report.per_layer_overflow.decode_full_attention_blocks, 1);

        manager.clear().unwrap();
        let after_clear = manager.report().unwrap();
        assert_eq!(
            after_clear.per_layer.len(),
            CACHE_RESIDENCY_LAYER_REPORT_LIMIT
        );
        assert_eq!(
            after_clear.prefill_full_attention_blocks,
            report.prefill_full_attention_blocks
        );
        assert_eq!(
            after_clear.per_layer_overflow.decode_full_attention_bytes,
            report.per_layer_overflow.decode_full_attention_bytes
        );
        assert!(after_clear
            .per_layer
            .iter()
            .all(|layer| layer.stats.current_device_bytes == 0
                && layer.stats.current_host_bytes == 0
                && layer.stats.current_disk_bytes == 0));
    }

    fn execution_key_value_block(stream: &Stream) -> CacheBlockArrays {
        let keys = Array::zeros::<f32>(&[1, 1, 2, 1], stream).unwrap();
        let values = Array::ones::<f32>(&[1, 1, 2, 1], stream).unwrap();
        async_eval_with_event([&keys, &values])
            .unwrap()
            .synchronize()
            .unwrap();
        CacheBlockArrays::KeyValue { keys, values }
    }

    fn two_buffer_host_capacity(logical_bytes_each: usize) -> u64 {
        let capacity =
            host_transfer_capacity_upper_bound(logical_bytes_each, HostTransferPolicy::Transfer)
                .unwrap() as u64;
        capacity.checked_mul(2).unwrap()
    }

    fn f32_storage_pointers(arrays: &CacheBlockArrays) -> [usize; 2] {
        arrays
            .arrays()
            .map(|array| array.evaluated().unwrap().as_slice::<f32>().as_ptr() as usize)
    }

    fn backend_key_value_block(stream: &Stream) -> CacheBlockArrays {
        let keys = Array::ones::<f32>(&[1, 1, 1, 1], stream).unwrap();
        let values = Array::ones::<f32>(&[1, 1, 1, 1], stream)
            .unwrap()
            .multiply(Array::from(2.0f32), stream)
            .unwrap();
        async_eval_with_event([&keys, &values])
            .unwrap()
            .synchronize()
            .unwrap();
        CacheBlockArrays::KeyValue { keys, values }
    }

    fn assert_backend_key_value_block(arrays: &CacheBlockArrays) {
        let CacheBlockArrays::KeyValue { keys, values } = arrays else {
            panic!("expected key/value cache arrays");
        };
        assert_eq!(
            keys.evaluated().unwrap().try_as_slice::<f32>().unwrap(),
            &[1.0]
        );
        assert_eq!(
            values.evaluated().unwrap().try_as_slice::<f32>().unwrap(),
            &[2.0]
        );
    }

    fn exercise_backend_cache_lifecycle(device: Device, expected_storage: HostTransferStorageKind) {
        let stream = Stream::new_with_device(&device);
        let consumer = Stream::new_with_device(&device);
        let host_capacity = two_buffer_host_capacity(size_of::<f32>());
        let pool = CacheResidencyPool::new(
            CachePoolLimits::new(16, host_capacity * 2, host_capacity * 2, 16).unwrap(),
        );
        let options = || {
            PagedCacheOptions::new(1, 16, host_capacity, 1)
                .unwrap()
                .with_full_attention(true)
                .with_pool(pool.clone())
                .unwrap()
        };
        let manager = CacheResidencyManager::new(options()).unwrap();
        manager.bind_transfer_device(&stream).unwrap();
        let id = manager
            .seal_block(0, 0, 1, None, backend_key_value_block(&stream), false)
            .unwrap();
        let competing = CacheResidencyManager::new(options()).unwrap();
        competing.set_tail_state(0, 8, 1).unwrap();
        let aggregate_error = competing.set_tail_state(0, 12, 1).unwrap_err();
        assert!(matches!(
            aggregate_error,
            CacheResidencyError::Pool(CachePoolError::BudgetExceeded {
                resource: CachePoolResource::Device,
                required: 20,
                budget: 16,
            })
        ));

        let demotion = manager.begin_device_demotion(&id).unwrap();
        let in_flight = pool.report().unwrap();
        assert_eq!(in_flight.current_device_bytes, 16);
        assert_eq!(in_flight.current_host_bytes, host_capacity);
        assert_eq!(in_flight.current_transfer_in_flight_bytes, host_capacity);
        manager.finish_device_demotion(&demotion).unwrap();
        {
            let state = manager.lock().unwrap();
            let block = state.blocks.get(&id).unwrap().host_block().unwrap();
            for buffer in block.buffers() {
                assert_eq!(buffer.storage_kind().unwrap(), expected_storage);
            }
        }
        let demoted = pool.report().unwrap();
        assert_eq!(demoted.current_device_bytes, 8);
        assert_eq!(demoted.current_host_bytes, host_capacity);
        assert_eq!(demoted.current_transfer_in_flight_bytes, 0);

        let destination = tempfile::tempdir().unwrap();
        let cache_path = destination.path().join("backend-cache");
        manager
            .save_prompt_cache(
                &cache_path,
                prompt_descriptor(),
                &[7],
                &[],
                &PromptCacheOptions {
                    application_namespace: Some("backend-verification".into()),
                    replace_existing: false,
                },
            )
            .unwrap();
        let (restored, manifest) = open_prompt_cache(
            &cache_path,
            &prompt_descriptor(),
            &prompt_model_identity(),
            &[7],
            options(),
        )
        .unwrap();
        assert_eq!(
            manifest.application_namespace.as_deref(),
            Some("backend-verification")
        );
        restored.bind_transfer_device(&stream).unwrap();
        let restored_id = restored
            .lock()
            .unwrap()
            .blocks
            .keys()
            .next()
            .unwrap()
            .clone();
        let transfer = restored
            .prepare_block_transfer(&restored_id, &stream)
            .unwrap();
        transfer.wait_on(&consumer).unwrap();
        let consumed = transfer.arrays().arrays()[0].square(&consumer).unwrap();
        async_eval_with_event([&consumed])
            .unwrap()
            .synchronize()
            .unwrap();
        assert_backend_key_value_block(transfer.arrays());
        drop(transfer);

        let restored_report = restored.report().unwrap();
        assert_eq!(restored_report.prompt_cache_loads, 1);
        assert_eq!(restored_report.disk_promotions, 1);
        assert!(restored_report.transfer_bytes >= 8);
        let aggregate = pool.report().unwrap();
        assert_eq!(aggregate.managers, 3);
        assert!(aggregate.current_device_bytes <= aggregate.limits.device_bytes());
        assert!(aggregate.current_host_bytes <= aggregate.limits.host_bytes());
        assert!(
            aggregate.current_transfer_in_flight_bytes
                <= aggregate.limits.transfer_in_flight_bytes()
        );

        manager.clear().unwrap();
        competing.clear().unwrap();
        restored.clear().unwrap();
        let cleared = pool.report().unwrap();
        assert_eq!(cleared.current_device_bytes, 0);
        assert_eq!(cleared.current_host_bytes, 0);
        assert_eq!(cleared.current_transfer_in_flight_bytes, 0);
        drop((manager, competing, restored));
        assert_eq!(pool.report().unwrap().managers, 0);
    }

    #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
    #[test]
    fn cpu_cache_backend_preserves_ordering_persistence_and_aggregate_budgets() {
        exercise_backend_cache_lifecycle(
            Device::new(DeviceType::Cpu, 0),
            HostTransferStorageKind::Cpu,
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "explicit Metal cache-residency test; run outside the sandbox"]
    fn metal_cache_backend_preserves_ordering_persistence_and_aggregate_budgets() {
        exercise_backend_cache_lifecycle(
            Device::new(DeviceType::Gpu, 0),
            HostTransferStorageKind::MetalShared,
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "explicit CUDA cache-residency test; requires a CUDA-capable host"]
    fn cuda_cache_backend_preserves_ordering_persistence_and_aggregate_budgets() {
        assert!(safemlx::cuda::is_available().unwrap());
        exercise_backend_cache_lifecycle(
            Device::new(DeviceType::Gpu, 0),
            HostTransferStorageKind::CudaPinned,
        );
    }

    #[test]
    fn two_block_prefetch_uses_a_dedicated_cpu_stream_and_bounds_leases() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let host_capacity = two_buffer_host_capacity(2 * size_of::<f32>());
        let manager = CacheResidencyManager::new(
            PagedCacheOptions::new(2, 32, host_capacity * 2, 1).unwrap(),
        )
        .unwrap();
        let ids = (0..3)
            .map(|index| {
                manager
                    .seal_block(
                        0,
                        index * 2,
                        index * 2 + 2,
                        None,
                        execution_key_value_block(&stream),
                        false,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let ids = vec![ids[0].clone(), ids[2].clone(), ids[1].clone()];

        let mut blocks = manager.prefetch_blocks(ids, &stream).unwrap();
        let (execution_index, transfer_index) = blocks.stream_indices().unwrap();
        assert_ne!(execution_index, transfer_index);

        let first = blocks.next_block().unwrap().unwrap();
        assert_eq!(blocks.pending_len(), 1);
        let consumed = first.arrays().arrays()[0].square(&stream).unwrap();
        let consumed = async_eval_with_event([&consumed]).unwrap();
        drop(first);

        let second = blocks.next_block().unwrap().unwrap();
        assert_eq!(blocks.pending_len(), 1);
        drop(second);
        let third = blocks.next_block().unwrap().unwrap();
        assert_eq!(blocks.pending_len(), 0);
        drop(third);
        assert!(blocks.next_block().unwrap().is_none());

        // The submitted consumer and queued waits retain their cache arrays
        // after every public lease has been released.
        consumed.synchronize().unwrap();
    }

    #[test]
    fn two_block_prefetch_falls_back_under_a_one_block_device_budget() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let host_capacity = two_buffer_host_capacity(2 * size_of::<f32>());
        let manager = CacheResidencyManager::new(
            PagedCacheOptions::new(2, 16, host_capacity * 2, 1).unwrap(),
        )
        .unwrap();
        let ids = (0..3)
            .map(|index| {
                manager
                    .seal_block(
                        0,
                        index * 2,
                        index * 2 + 2,
                        None,
                        execution_key_value_block(&stream),
                        false,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let mut blocks = manager.prefetch_blocks(ids, &stream).unwrap();
        for _ in 0..3 {
            let lease = blocks.next_block().unwrap().unwrap();
            assert_eq!(blocks.pending_len(), 0);
            drop(lease);
        }
        assert!(blocks.next_block().unwrap().is_none());
        let report = manager.report().unwrap();
        assert!(report.current_device_bytes <= 16);
        assert_eq!(report.failures, 0);
    }

    #[test]
    #[ignore = "explicit Metal paged-cache prefetch test; run on a Metal host"]
    fn two_block_metal_prefetch_is_gpu_ordered_without_host_synchronization() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let block_bytes = 2 * 1024 * 1024 * std::mem::size_of::<f32>() as u64;
        let manager = CacheResidencyManager::new(
            PagedCacheOptions::new(1024, block_bytes * 3, block_bytes, 1).unwrap(),
        )
        .unwrap();
        let ids = (0..3)
            .map(|index| {
                let keys = Array::ones::<f32>(&[1024, 1024], &stream).unwrap();
                let values = Array::ones::<f32>(&[1024, 1024], &stream).unwrap();
                manager
                    .seal_block(
                        0,
                        index * 1024,
                        index * 1024 + 1024,
                        None,
                        CacheBlockArrays::KeyValue { keys, values },
                        false,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        // Stage one exact host block without creating device-budget pressure;
        // the promotion assertion below then observes only the async copy.
        let host_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let device_arrays = manager
            .lock()
            .unwrap()
            .blocks
            .get(&ids[0])
            .unwrap()
            .device_arrays()
            .unwrap()
            .clone();
        let host_block = HostCacheBlock::from_device_arrays(&device_arrays, &host_stream).unwrap();
        {
            let mut state = manager.lock().unwrap();
            let record = state.blocks.get_mut(&ids[0]).unwrap();
            record.physical = MlxCacheBlockStorage::host(ids[0].clone(), host_block, None);
            super::update_report_totals(&mut state);
        }
        let ids = vec![ids[0].clone(), ids[2].clone(), ids[1].clone()];

        let transfer = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let blocker_lhs = Array::ones::<f32>(&[4096, 4096], &transfer).unwrap();
        let blocker_rhs = Array::ones::<f32>(&[4096, 4096], &transfer).unwrap();
        let blocker = blocker_lhs.matmul(&blocker_rhs, &transfer).unwrap();
        safemlx::transforms::async_eval([&blocker]).unwrap();
        let direct = manager.prepare_block_transfer(&ids[0], &transfer).unwrap();
        assert!(
            direct
                .completions
                .iter()
                .any(|completion| !completion.is_complete().unwrap()),
            "paged-cache promotion blocked the host"
        );
        direct.wait_on(&stream).unwrap();
        let consumed = direct.arrays().arrays()[0].square(&stream).unwrap();
        let completion = async_eval_with_event([&consumed]).unwrap();
        drop(direct);
        completion.synchronize().unwrap();

        let mut blocks = manager.prefetch_blocks(ids, &stream).unwrap();
        let (execution_index, transfer_index) = blocks.stream_indices().unwrap();
        assert_ne!(execution_index, transfer_index);
        let first = blocks.next_block().unwrap().unwrap();
        assert_eq!(blocks.pending_len(), 1);
        drop(first);
        drop(blocks);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn promotion_waits_for_pending_write_without_overcommitting_host_storage() {
        let directory = tempfile::tempdir().unwrap();
        let host_capacity = two_buffer_host_capacity(2 * size_of::<f32>());
        let options = PagedCacheOptions::new(2, 32, host_capacity, 1)
            .unwrap()
            .with_live_disk(directory.path(), 1024, 1)
            .unwrap();
        let manager = CacheResidencyManager::new(options).unwrap();
        let worker = manager.inner.disk_worker.as_ref().unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let blocker = worker
            .prepare(
                CacheIoOperationKey {
                    generation: 0,
                    id: disk_test_id(99),
                    kind: CacheIoOperationKind::Read,
                },
                DiskTask::Pause {
                    started: started_tx,
                    release: release_rx,
                },
            )
            .unwrap();
        let blocker_ticket = blocker.ticket.clone();
        blocker.enqueue().unwrap();
        started_rx.recv().unwrap();

        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let first = manager
            .seal_block(0, 0, 2, None, execution_key_value_block(&stream), false)
            .unwrap();
        manager
            .seal_block(0, 2, 4, None, execution_key_value_block(&stream), false)
            .unwrap();
        manager
            .seal_block(0, 4, 6, None, execution_key_value_block(&stream), false)
            .unwrap();
        let report = manager.report().unwrap();
        assert_eq!(report.current_host_bytes, host_capacity);
        assert_eq!(report.in_flight_write_bytes, host_capacity);

        let promotion_manager = manager.clone();
        let (promoted_tx, promoted_rx) = mpsc::channel();
        let promotion_thread = thread::spawn(move || {
            let promotion_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
            let result = promotion_manager.lease_block(&first, &promotion_stream);
            promoted_tx.send(result.map(drop)).unwrap();
        });
        match promoted_rx.recv_timeout(Duration::from_millis(20)) {
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            result => panic!("promotion completed before host capacity was released: {result:?}"),
        }
        let report = manager.report().unwrap();
        assert_eq!(report.current_host_bytes, host_capacity);
        assert_eq!(report.in_flight_write_bytes, host_capacity);
        assert!(report.current_host_bytes <= manager.options().host_budget_bytes());

        release_tx.send(()).unwrap();
        assert!(matches!(blocker_ticket.wait().unwrap(), DiskResult::Test));
        promoted_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("promotion did not finish after writeback released host capacity")
            .unwrap();
        promotion_thread.join().unwrap();
        let report = manager.report().unwrap();
        assert!(report.current_host_bytes <= manager.options().host_budget_bytes());
        assert!(report.peak_host_bytes <= manager.options().host_budget_bytes());
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn host_to_disk_demotion_returns_before_background_write_completes() {
        let directory = tempfile::tempdir().unwrap();
        let host_capacity = two_buffer_host_capacity(2 * size_of::<f32>());
        let options = PagedCacheOptions::new(2, 16, host_capacity, 1)
            .unwrap()
            .with_live_disk(directory.path(), 1024, 1)
            .unwrap();
        let manager = CacheResidencyManager::new(options).unwrap();
        let worker = manager.inner.disk_worker.as_ref().unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let blocker = worker
            .prepare(
                CacheIoOperationKey {
                    generation: 0,
                    id: disk_test_id(99),
                    kind: CacheIoOperationKind::Read,
                },
                DiskTask::Pause {
                    started: started_tx,
                    release: release_rx,
                },
            )
            .unwrap();
        let blocker_ticket = blocker.ticket.clone();
        blocker.enqueue().unwrap();
        started_rx.recv().unwrap();

        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        manager
            .seal_block(0, 0, 2, None, execution_key_value_block(&stream), false)
            .unwrap();
        let second = execution_key_value_block(&stream);
        let background_manager = manager.clone();
        let (sealed_tx, sealed_rx) = mpsc::channel();
        let seal_thread = thread::spawn(move || {
            sealed_tx
                .send(background_manager.seal_block(0, 2, 4, None, second, false))
                .unwrap();
        });

        sealed_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("demotion waited for the blocked disk worker")
            .unwrap();
        let report = manager.report().unwrap();
        assert_eq!(report.in_flight_write_blocks, 1);
        assert_eq!(report.in_flight_write_bytes, host_capacity);
        assert_eq!(report.current_host_bytes, host_capacity);
        assert_eq!(report.host_blocks, 1);
        assert!(report.current_host_bytes <= manager.options().host_budget_bytes());
        assert_eq!(report.disk_demotions, 0);

        // A third block needs the same host slot. It must wait for the pending
        // write to commit instead of retaining another host allocation beyond
        // the byte budget.
        let third = execution_key_value_block(&stream);
        let waiting_manager = manager.clone();
        let (third_tx, third_rx) = mpsc::channel();
        let third_thread = thread::spawn(move || {
            third_tx
                .send(waiting_manager.seal_block(0, 4, 6, None, third, false))
                .unwrap();
        });
        assert!(matches!(
            third_rx.recv_timeout(Duration::from_millis(20)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        let report = manager.report().unwrap();
        assert_eq!(report.current_host_bytes, host_capacity);
        assert_eq!(report.in_flight_write_bytes, host_capacity);

        release_tx.send(()).unwrap();
        assert!(matches!(blocker_ticket.wait().unwrap(), DiskResult::Test));
        seal_thread.join().unwrap();
        third_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("host-capacity wait did not finish after writeback")
            .unwrap();
        third_thread.join().unwrap();
        for _ in 0..100 {
            let report = manager.report().unwrap();
            if report.disk_demotions >= 2 {
                assert_eq!(report.in_flight_write_blocks, 0);
                assert_eq!(report.disk_blocks, 2);
                assert!(report.current_host_bytes <= manager.options().host_budget_bytes());
                assert!(report.peak_host_bytes <= manager.options().host_budget_bytes());
                assert!(report.in_flight_waits >= 1);
                let layer = report
                    .per_layer
                    .iter()
                    .find(|layer| layer.global_layer == 0)
                    .unwrap();
                assert_eq!(layer.stats.disk_demotions, report.disk_demotions);
                assert_eq!(layer.stats.in_flight_waits, report.in_flight_waits);
                return;
            }
            thread::sleep(Duration::from_millis(10));
        }
        panic!("background disk write did not commit");
    }

    #[test]
    fn failed_asynchronous_host_demotion_restores_the_device_state() {
        let manager =
            CacheResidencyManager::new(PagedCacheOptions::new(1, 16, 16, 1).unwrap()).unwrap();
        let id = disk_test_id(0);
        let completion = Arc::new(HostDemotionCompletion::default());
        completion.finish(Err(CacheResidencyError::Runtime(
            "injected host demotion failure".into(),
        )));
        let ticket = HostDemotionTicket {
            operation_id: 71,
            id: id.clone(),
            reserved_host_bytes: 8,
            completion,
        };
        insert_test_record(
            &mut manager.lock().unwrap(),
            CacheBlockRecord {
                physical: test_demoting(test_device_block(), ticket.clone()),
                bytes: 8,
                shapes: [vec![1], vec![1]],
                dtypes: ["Float32".into(), "Float32".into()],
                imported: false,
            },
            false,
            0,
        );

        let error = manager.finish_device_demotion(&ticket).unwrap_err();
        assert!(error.to_string().contains("injected host demotion failure"));
        let state = manager.lock().unwrap();
        assert_eq!(
            state.blocks.get(&id).unwrap().physical.phase(),
            CacheStoragePhase::Device
        );
        drop(state);
        let report = manager.report().unwrap();
        assert_eq!(report.current_device_bytes, 8);
        assert_eq!(report.current_host_bytes, 0);
        assert_eq!(report.in_flight_host_demotion_blocks, 0);
        assert_eq!(report.failures, 1);
    }

    #[test]
    fn retiring_host_demotion_remains_charged_during_generation_reset() {
        let manager =
            CacheResidencyManager::new(PagedCacheOptions::new(1, 16, 16, 1).unwrap()).unwrap();
        let id = disk_test_id(0);
        let completion = Arc::new(HostDemotionCompletion::default());
        let ticket = HostDemotionTicket {
            operation_id: 72,
            id: id.clone(),
            reserved_host_bytes: 8,
            completion: Arc::clone(&completion),
        };
        insert_test_record(
            &mut manager.lock().unwrap(),
            CacheBlockRecord {
                physical: test_demoting(test_device_block(), ticket),
                bytes: 8,
                shapes: [vec![1], vec![1]],
                dtypes: ["Float32".into(), "Float32".into()],
                imported: false,
            },
            false,
            0,
        );

        let clearing_manager = manager.clone();
        let clearing = thread::spawn(move || clearing_manager.clear());
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        loop {
            let report = manager.report().unwrap();
            if report.in_flight_host_demotion_blocks == 1
                && manager.lock().unwrap().blocks.is_empty()
            {
                assert_eq!(report.current_device_bytes, 8);
                assert_eq!(report.current_host_bytes, 8);
                assert_eq!(report.in_flight_host_demotion_bytes, 8);
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "clear did not publish its retiring demotion reservation"
            );
            thread::yield_now();
        }

        completion.finish(Err(CacheResidencyError::Runtime(
            "discarded generation transfer failed".into(),
        )));
        clearing.join().unwrap().unwrap();
        let report = manager.report().unwrap();
        assert_eq!(report.current_device_bytes, 0);
        assert_eq!(report.current_host_bytes, 0);
        assert_eq!(report.in_flight_host_demotion_blocks, 0);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn asynchronous_host_demotion_charges_both_allocations_until_reconciled() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let host_capacity = two_buffer_host_capacity(2 * size_of::<f32>());
        let manager =
            CacheResidencyManager::new(PagedCacheOptions::new(2, 32, host_capacity, 1).unwrap())
                .unwrap();
        manager.bind_transfer_device(&stream).unwrap();
        let arrays = execution_key_value_block(&stream);
        let device_pointers = f32_storage_pointers(&arrays);
        let id = manager.seal_block(0, 0, 2, None, arrays, false).unwrap();

        let ticket = manager.begin_device_demotion(&id).unwrap();
        {
            let state = manager.lock().unwrap();
            let record = state.blocks.get(&id).unwrap();
            assert_eq!(record.physical.phase(), CacheStoragePhase::DemotingToHost);
            let arrays = record.physical.device_resource().unwrap();
            let live = record.physical.host_demotion().unwrap();
            assert_eq!(live.operation_id, ticket.operation_id);
            assert_eq!(f32_storage_pointers(arrays), device_pointers);
        }
        let report = manager.report().unwrap();
        assert_eq!(report.device_blocks, 1);
        assert_eq!(report.host_blocks, 1);
        assert_eq!(report.current_device_bytes, 16);
        assert_eq!(report.current_host_bytes, host_capacity);
        assert_eq!(report.in_flight_host_demotion_blocks, 1);
        assert_eq!(report.in_flight_host_demotion_bytes, host_capacity);
        assert_eq!(report.peak_in_flight_host_demotion_bytes, host_capacity);
        let layer = report
            .per_layer
            .iter()
            .find(|layer| layer.global_layer == 0)
            .unwrap();
        assert_eq!(layer.stats.in_flight_host_demotion_blocks, 1);
        assert_eq!(layer.stats.in_flight_host_demotion_bytes, host_capacity);

        manager.finish_device_demotion(&ticket).unwrap();
        let report = manager.report().unwrap();
        assert_eq!(report.device_blocks, 0);
        assert_eq!(report.host_blocks, 1);
        assert_eq!(report.current_device_bytes, 0);
        assert_eq!(report.current_host_bytes, host_capacity);
        assert_eq!(report.in_flight_host_demotion_blocks, 0);
        assert_eq!(report.in_flight_host_demotion_bytes, 0);
        assert_eq!(report.host_demotions, 1);
        assert_eq!(report.transfer_bytes, 16);
    }

    #[test]
    fn cache_pool_charges_physical_host_allocation_capacity() {
        let stream = cpu_stream();
        let arrays = CacheBlockArrays::KeyValue {
            keys: Array::zeros::<f32>(&[1, 1, 1, 5000], &stream).unwrap(),
            values: Array::zeros::<f32>(&[1, 1, 1, 5000], &stream).unwrap(),
        };
        let logical = arrays.bytes();
        let capacity = host_cache_capacity_upper_bound(&arrays).unwrap();
        let manager =
            CacheResidencyManager::new(PagedCacheOptions::new(1, logical, capacity, 1).unwrap())
                .unwrap();
        manager.bind_transfer_device(&stream).unwrap();
        let id = manager.seal_block(0, 0, 1, None, arrays, false).unwrap();
        let ticket = manager.begin_device_demotion(&id).unwrap();
        manager.finish_device_demotion(&ticket).unwrap();

        let report = manager.report().unwrap();
        assert_eq!(report.current_host_bytes, capacity);
        assert_eq!(report.peak_host_bytes, capacity);
        assert_eq!(
            manager.pool().report().unwrap().current_host_bytes,
            capacity
        );
        let state = manager.lock().unwrap();
        let block = state.blocks.get(&id).unwrap().host_block().unwrap();
        assert_eq!(block.bytes().unwrap(), logical);
        assert_eq!(block.capacity().unwrap(), capacity);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn clear_waits_for_asynchronous_host_demotion_resources() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let host_capacity = two_buffer_host_capacity(2 * size_of::<f32>());
        let manager =
            CacheResidencyManager::new(PagedCacheOptions::new(2, 32, host_capacity, 1).unwrap())
                .unwrap();
        manager.bind_transfer_device(&stream).unwrap();
        let id = manager
            .seal_block(0, 0, 2, None, execution_key_value_block(&stream), false)
            .unwrap();
        let ticket = manager.begin_device_demotion(&id).unwrap();

        manager.clear().unwrap();
        assert!(ticket.wait().is_ok());
        let report = manager.report().unwrap();
        assert_eq!(report.device_blocks, 0);
        assert_eq!(report.host_blocks, 0);
        assert_eq!(report.current_device_bytes, 0);
        assert_eq!(report.current_host_bytes, 0);
        assert_eq!(report.in_flight_host_demotion_blocks, 0);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn host_demotion_uses_typed_buffers_and_promotion_rebuilds_device_arrays() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        // Two 16-byte blocks fit on the device. A third block forces the oldest
        // one into backend-selected host-transfer storage while retaining one
        // recent block.
        let host_capacity = two_buffer_host_capacity(2 * size_of::<f32>());
        let options = PagedCacheOptions::new(2, 32, host_capacity, 1).unwrap();
        let manager = CacheResidencyManager::new(options).unwrap();
        manager.bind_transfer_device(&stream).unwrap();
        let first_arrays = execution_key_value_block(&stream);
        let first = manager
            .seal_block(0, 0, 2, None, first_arrays, false)
            .unwrap();
        manager
            .seal_block(0, 2, 4, None, execution_key_value_block(&stream), false)
            .unwrap();
        manager
            .seal_block(0, 4, 6, None, execution_key_value_block(&stream), false)
            .unwrap();

        let first_host_capacity = {
            let state = manager.lock().unwrap();
            let record = state.blocks.get(&first).unwrap();
            assert_eq!(record.tier(), CacheTier::Host);
            let block = record.host_block().unwrap();
            let [first, second] = block.buffers();
            first.capacity().unwrap() + second.capacity().unwrap()
        };
        assert!(first_host_capacity >= 16);
        let report = manager.report().unwrap();
        assert_eq!(report.device_blocks, 2);
        assert_eq!(report.host_blocks, 1);
        assert_eq!(report.current_device_bytes, 32);
        assert_eq!(report.current_host_bytes, host_capacity);

        let lease = manager.lease_block(&first, &stream).unwrap();
        let promoted_pointers = f32_storage_pointers(lease.arrays());
        // A demoted allocation may reuse the virtual address of its released
        // source on unified-memory systems, so pointer inequality is not a
        // storage-identity invariant. The typed buffer capacity is verified
        // above; promotion is verified by values and the live device record.
        match lease.arrays() {
            CacheBlockArrays::KeyValue { keys, values } => {
                assert_eq!(keys.evaluated().unwrap().as_slice::<f32>(), &[0.0, 0.0]);
                assert_eq!(values.evaluated().unwrap().as_slice::<f32>(), &[1.0, 1.0]);
            }
            CacheBlockArrays::CompressedLatentRotary { .. } => unreachable!(),
        }
        {
            let state = manager.lock().unwrap();
            let record = state.blocks.get(&first).unwrap();
            assert_eq!(record.tier(), CacheTier::Device);
            assert_eq!(
                f32_storage_pointers(record.device_arrays().unwrap()),
                promoted_pointers
            );
        }
        drop(lease);

        let report = manager.report().unwrap();
        assert_eq!(report.device_blocks, 2);
        assert_eq!(report.host_blocks, 1);
        assert_eq!(report.current_device_bytes, 32);
        assert_eq!(report.current_host_bytes, host_capacity);
        assert_eq!(report.host_promotions, 1);
        assert_eq!(report.host_demotions, 2);
        let layer = report
            .per_layer
            .iter()
            .find(|layer| layer.global_layer == 0)
            .unwrap();
        assert_eq!(layer.stats.host_promotions, report.host_promotions);
        assert_eq!(layer.stats.host_demotions, report.host_demotions);
        assert_eq!(layer.stats.transfer_bytes, report.transfer_bytes);
        assert_eq!(layer.stats.demand_misses, report.demand_misses);
        assert_eq!(first.representation, CacheRepresentation::KeyValue);
    }
}

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

/// Maximum number of cache blocks scheduled ahead by paged-cache prefetch.
pub const PAGED_CACHE_PREFETCH_BLOCKS: usize = 2;
static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_HOST_WRITE_RESERVATION_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_HOST_DEMOTION_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
/// Device arrays held by one resident attention-cache block.
pub enum CacheBlockArrays {
    /// Conventional key/value attention state.
    KeyValue {
        /// Key-cache values.
        keys: Array,
        /// Value-cache values.
        values: Array,
    },
    /// Compressed latent attention state and its decoupled rotary keys.
    CompressedLatentRotary {
        /// Compressed latent values.
        latent: Array,
        /// Decoupled rotary-key values.
        rotary_key: Array,
    },
}

impl CacheBlockArrays {
    /// Returns the portable representation encoded by these arrays.
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

fn dtype_name(dtype: Dtype) -> String {
    format!("{dtype:?}")
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
    #[cfg(test)]
    Pause {
        started: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
        retained: Arc<()>,
        finished: mpsc::Sender<()>,
    },
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
                        #[cfg(test)]
                        HostDemotionRequest::Pause {
                            started,
                            release,
                            retained,
                            finished,
                        } => {
                            let _ = started.send(());
                            let _ = release.recv();
                            drop(retained);
                            let _ = finished.send(());
                        }
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
        // The receiver owns every queued array/device/completion. Disconnecting
        // the sender lets it finish those messages without joining under a
        // native retirement lock; dropping JoinHandle only detaches it.
        if let Ok(handle) = self.handle.get_mut() {
            handle.take();
        }
    }
}

fn transfer_error(
    operation: &'static str,
    source: safemlx::error::Exception,
) -> CacheResidencyError {
    CacheResidencyError::Runtime(format!("{operation}: {source}"))
}

mod io;
pub use io::{
    load_prompt_cache_state_tensors, open_prompt_cache, CacheBlockLease, CacheBlockPrefetch,
    CacheResidencyManager, LoadedPromptCacheStateTensor, PromptCacheStateArray,
};

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

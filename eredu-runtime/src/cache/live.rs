//! Portable live cache catalog, leases, promotion, and spill publication.

use super::*;
use eredu_checkpoint::StoredDtype;
use eredu_core::{
    cache::{
        mechanisms::{
            CacheHostDemotionOperation, CacheHostTensor, CacheIoOperation, CacheIoOperationKey,
            CacheIoOperationKind, CacheTransferLane, LiveCacheBackend,
        },
        CacheBlockId, CachePolicyError, CacheRankIdentity, CacheRepresentation, CacheStreamId,
        CacheTier, PromptCacheBlock, PromptCacheDescriptor, PromptCacheError, PromptCacheManifest,
        PromptCacheStateTensor, PromptCacheStoreError, StateTensorOwner, StateTensorRole,
    },
    residency::CacheEvictionPolicy,
    BackendFailure,
};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    panic::{catch_unwind, AssertUnwindSafe},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc, Arc, Condvar, Mutex, MutexGuard, Weak,
    },
    thread::{self, JoinHandle},
    time::Instant,
};

static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_HOST_WRITE_RESERVATION_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_HOST_DEMOTION_ID: AtomicU64 = AtomicU64::new(1);

/// Keeps immutable sliding-history catalog entries available to a transient
/// checkpoint. It does not lease device arrays or prevent ordinary tier movement.
#[derive(Debug)]
pub struct CacheHistoryRetention {
    global_layer: usize,
    representation: CacheRepresentation,
    start: i64,
    end: i64,
}

impl CacheHistoryRetention {
    fn contains(&self, id: &CacheBlockId) -> bool {
        id.global_layer == self.global_layer
            && id.representation == self.representation
            && id.start < self.end
            && id.end > self.start
    }
}

#[derive(Debug, Clone)]
/// Device arrays held by one resident attention-cache block.
pub enum CacheBlockArrays<T> {
    /// Named records. The zero-width reserved array preserves the shared two-array
    /// paging format without allocating a second payload.
    AppendStream {
        /// Architecture-declared stream slot.
        slot: u32,
        /// Independent batch lane.
        lane: u32,
        /// Exact records, shaped `[1, 1, entries, width]`.
        records: T,
        /// Same scalar type, shaped `[1, 1, entries, 0]`.
        reserved: T,
    },
    /// Conventional key/value attention state.
    KeyValue {
        /// Key-cache values.
        keys: T,
        /// Value-cache values.
        values: T,
    },
    /// Compressed latent attention state and its decoupled rotary keys.
    CompressedLatentRotary {
        /// Compressed latent values.
        latent: T,
        /// Decoupled rotary-key values.
        rotary_key: T,
    },
}

impl<T> CacheBlockArrays<T> {
    fn from_parts(representation: CacheRepresentation, [first, second]: [T; 2]) -> Self {
        match representation {
            CacheRepresentation::AppendStream { slot, lane } => Self::AppendStream {
                slot,
                lane,
                records: first,
                reserved: second,
            },
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
    /// Returns the portable representation encoded by these arrays.
    pub fn representation(&self) -> CacheRepresentation {
        match self {
            Self::AppendStream { slot, lane, .. } => CacheRepresentation::AppendStream {
                slot: *slot,
                lane: *lane,
            },
            Self::KeyValue { .. } => CacheRepresentation::KeyValue,
            Self::CompressedLatentRotary { .. } => CacheRepresentation::CompressedLatentRotary,
        }
    }

    fn arrays(&self) -> [&T; 2] {
        match self {
            Self::AppendStream {
                records, reserved, ..
            } => [records, reserved],
            Self::KeyValue { keys, values } => [keys, values],
            Self::CompressedLatentRotary { latent, rotary_key } => [latent, rotary_key],
        }
    }
}

type Page<M> = CacheBlockArrays<<M as LiveCacheBackend>::Tensor>;
type Queue<M> = <M as LiveCacheBackend>::Queue;
type TransferEvents<M> = Vec<<M as LiveCacheBackend>::Event>;

fn page_bytes<M: LiveCacheBackend>(page: &Page<M>) -> u64 {
    page.arrays().into_iter().map(M::bytes).sum()
}

fn page_shapes<M: LiveCacheBackend>(page: &Page<M>) -> [Vec<i32>; 2] {
    page.arrays().map(|value| M::shape(value).to_vec())
}

fn page_dtypes<M: LiveCacheBackend>(page: &Page<M>) -> Result<[String; 2], CacheResidencyError> {
    let [first, second] = page.arrays();
    Ok([
        dtype_name(M::dtype(first))?.into(),
        dtype_name(M::dtype(second))?.into(),
    ])
}

fn host_cache_capacity_upper_bound<M: LiveCacheBackend>(
    arrays: &Page<M>,
) -> Result<u64, CacheResidencyError> {
    arrays.arrays().into_iter().try_fold(0u64, |total, array| {
        checked_capacity(total, M::host_capacity_bound(M::bytes(array))?)
    })
}

fn checked_capacity(total: u64, bytes: u64) -> Result<u64, CacheResidencyError> {
    total
        .checked_add(bytes)
        .ok_or_else(|| CacheResidencyError::Runtime("host cache capacity overflowed".into()))
}
fn host_cache_layout_capacity_upper_bound<M: LiveCacheBackend>(
    shapes: &[Vec<i32>; 2],
    dtypes: &[String; 2],
) -> Result<u64, CacheResidencyError> {
    shapes
        .iter()
        .zip(dtypes)
        .try_fold(0u64, |total, (shape, dtype)| {
            let (_, _, element_bytes) = cache_dtype(dtype)?;
            let logical_bytes = shape.iter().try_fold(*element_bytes, |bytes, dimension| {
                let dimension = u64::try_from(*dimension).map_err(|_| {
                    CacheResidencyError::Runtime(
                        "host cache shape contains a negative dimension".into(),
                    )
                })?;
                bytes.checked_mul(dimension).ok_or_else(|| {
                    CacheResidencyError::Runtime("host cache logical byte length overflowed".into())
                })
            })?;
            checked_capacity(total, M::host_capacity_bound(logical_bytes)?)
        })
}

// Stable names used in the prompt-cache wire format.
const CACHE_DTYPES: [(StoredDtype, &str, u64); 14] = [
    (StoredDtype::Bool, "Bool", 1),
    (StoredDtype::U8, "Uint8", 1),
    (StoredDtype::U16, "Uint16", 2),
    (StoredDtype::U32, "Uint32", 4),
    (StoredDtype::U64, "Uint64", 8),
    (StoredDtype::I8, "Int8", 1),
    (StoredDtype::I16, "Int16", 2),
    (StoredDtype::I32, "Int32", 4),
    (StoredDtype::I64, "Int64", 8),
    (StoredDtype::F16, "Float16", 2),
    (StoredDtype::F32, "Float32", 4),
    (StoredDtype::F64, "Float64", 8),
    (StoredDtype::BF16, "Bfloat16", 2),
    (StoredDtype::C64, "Complex64", 8),
];

fn dtype_name(dtype: StoredDtype) -> Result<&'static str, CacheResidencyError> {
    CACHE_DTYPES
        .iter()
        .find_map(|(encoding, name, _)| (*encoding == dtype).then_some(*name))
        .ok_or_else(|| {
            CacheResidencyError::ArrayMismatch(format!("unsupported cache dtype {dtype:?}"))
        })
}

fn cache_dtype(
    name: &str,
) -> Result<&'static (StoredDtype, &'static str, u64), CacheResidencyError> {
    CACHE_DTYPES
        .iter()
        .find(|(_, wire_name, _)| *wire_name == name)
        .ok_or_else(|| {
            CacheResidencyError::ArrayMismatch(format!("unsupported cache dtype {name}"))
        })
}

fn stored_dtype(name: &str) -> Result<StoredDtype, CacheResidencyError> {
    Ok(cache_dtype(name)?.0.clone())
}

#[derive(Debug, Clone)]
struct HostCacheBlock<M: LiveCacheBackend> {
    representation: CacheRepresentation,
    buffers: [M::Host; 2],
}

impl<M: LiveCacheBackend> HostCacheBlock<M> {
    fn from_device_arrays(arrays: &Page<M>, queue: &Queue<M>) -> Result<Self, CacheResidencyError> {
        Ok(Self {
            representation: arrays.representation(),
            buffers: M::copy_to_host(arrays.arrays(), queue)?,
        })
    }

    fn buffers(&self) -> [&M::Host; 2] {
        [&self.buffers[0], &self.buffers[1]]
    }

    fn shapes(&self) -> Result<[Vec<i32>; 2], CacheResidencyError> {
        let convert = |host: &M::Host| {
            M::host_payload(host)?
                .shape
                .into_iter()
                .map(|dim| i32::try_from(dim).map_err(BackendFailure::from_error))
                .collect::<Result<Vec<_>, _>>()
        };
        let [first, second] = self.buffers();
        Ok([convert(first)?, convert(second)?])
    }

    fn dtypes(&self) -> Result<[String; 2], CacheResidencyError> {
        let [first, second] = self.buffers();
        Ok([
            dtype_name(M::host_payload(first)?.dtype)?.to_owned(),
            dtype_name(M::host_payload(second)?.dtype)?.to_owned(),
        ])
    }

    fn bytes(&self) -> Result<u64, CacheResidencyError> {
        let [first, second] = self.buffers();
        Ok(
            M::host_payload(first)?.bytes.len() as u64
                + M::host_payload(second)?.bytes.len() as u64,
        )
    }

    fn capacity(&self) -> Result<u64, CacheResidencyError> {
        self.buffers().into_iter().try_fold(0, |total, host| {
            checked_capacity(total, M::host_capacity(host)?)
        })
    }

    fn copy_to_device(
        &self,
        queue: &Queue<M>,
    ) -> Result<(Page<M>, TransferEvents<M>), CacheResidencyError> {
        let [(first, first_ready), (second, second_ready)] =
            M::copy_to_device(self.buffers(), queue)?;
        Ok((
            CacheBlockArrays::from_parts(self.representation, [first, second]),
            vec![first_ready, second_ready],
        ))
    }
}
type CacheTaskResult<T> = Result<T, Arc<CacheResidencyError>>;

#[derive(Debug)]
struct HostDemotionCompletion<M: LiveCacheBackend> {
    result: Mutex<Option<CacheTaskResult<HostCacheBlock<M>>>>,
    ready: Condvar,
}

impl<M: LiveCacheBackend> HostDemotionCompletion<M> {
    fn finish(&self, result: Result<HostCacheBlock<M>, CacheResidencyError>) {
        if let Ok(mut slot) = self.result.lock() {
            if slot.is_none() {
                *slot = Some(result.map_err(Arc::new));
                self.ready.notify_all();
            }
        }
    }

    fn wait(&self) -> Result<HostCacheBlock<M>, CacheResidencyError> {
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
            Err(error) => Err(Arc::clone(error).into()),
        }
    }
}

#[derive(Debug, Clone)]
struct HostDemotionTicket<M: LiveCacheBackend> {
    operation_id: u64,
    id: CacheBlockId,
    reserved_host_bytes: u64,
    completion: Arc<HostDemotionCompletion<M>>,
}

impl<M: LiveCacheBackend> HostDemotionTicket<M> {
    fn wait(&self) -> Result<HostCacheBlock<M>, CacheResidencyError> {
        self.completion.wait()
    }
}

impl<M: LiveCacheBackend> CacheHostDemotionOperation for HostDemotionTicket<M> {
    fn block_id(&self) -> &CacheBlockId {
        &self.id
    }

    fn operation_id(&self) -> u64 {
        self.operation_id
    }
}

enum HostDemotionRequest<M: LiveCacheBackend> {
    Demote {
        arrays: Page<M>,
        device: M::Device,
        completion: Arc<HostDemotionCompletion<M>>,
    },
}

struct HostDemotionWorker<M: LiveCacheBackend> {
    sender: mpsc::Sender<HostDemotionRequest<M>>,
    handle: Mutex<Option<JoinHandle<()>>>,
}

impl<M: LiveCacheBackend> std::fmt::Debug for HostDemotionWorker<M> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("HostDemotionWorker").finish()
    }
}

impl<M: LiveCacheBackend> HostDemotionWorker<M> {
    fn new() -> Result<Self, CacheResidencyError> {
        let (sender, receiver) = mpsc::channel();
        let handle = thread::Builder::new()
            .name("eredu-cache-host-demotion".into())
            .spawn(move || {
                while let Ok(request) = receiver.recv() {
                    match request {
                        HostDemotionRequest::Demote {
                            arrays,
                            device,
                            completion,
                        } => {
                            let result = catch_unwind(AssertUnwindSafe(|| {
                                let stream =
                                    M::transfer_queue(device, CacheTransferLane::Demotion)?;
                                HostCacheBlock::from_device_arrays(&arrays, &stream)
                            }))
                            .unwrap_or_else(|_| {
                                Err(CacheResidencyError::Runtime(
                                    "cache host demotion worker operation panicked".into(),
                                ))
                            });
                            completion.finish(result);
                        }
                    }
                }
            })
            .map_err(|source| CacheResidencyError::Io {
                action: "start cache host demotion worker",
                path: PathBuf::from("eredu-cache-host-demotion"),
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
        arrays: Page<M>,
        device: M::Device,
        reserved_host_bytes: u64,
    ) -> Result<HostDemotionTicket<M>, CacheResidencyError> {
        let completion = Arc::new(HostDemotionCompletion {
            result: Mutex::new(None),
            ready: Condvar::new(),
        });
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

impl<M: LiveCacheBackend> Drop for HostDemotionWorker<M> {
    fn drop(&mut self) {
        // The receiver owns every queued array/device/completion. Disconnecting
        // the sender lets it finish those messages without joining under a
        // native retirement lock; dropping JoinHandle only detaches it.
        if let Ok(handle) = self.handle.get_mut() {
            handle.take();
        }
    }
}

mod acquisition;
mod io;
mod lifecycle;
mod persistence;
mod prompt_cache;
mod reporting;
mod state;
mod stream;
mod transitions;

pub use acquisition::{CacheBlockLease, CacheBlockPrefetch};
use io::*;
use lifecycle::validate_block_arrays;
use persistence::*;
pub use prompt_cache::{
    CacheExport, CacheRestore, LoadedPromptCacheStateTensor, PromptCacheStateArray, RestoreCache,
    RestoredCache,
};
use reporting::update_report_totals;
use state::*;
pub use stream::{CacheAttentionRead, CacheStreamCheckpoint, CacheStreamState, LiveCacheStream};
use transitions::{eviction_candidate, HostDemotionProgress, PendingCacheOperation};

/// Storage that the portable driver can open for an admitted mutable state.
pub trait SelectedLiveCache: Sized {
    /// Creates the cache owner for the selected paging policy.
    fn open(selected: &crate::SelectedStateRealization) -> Result<Option<Self>, BackendFailure>;
}

impl<M: LiveCacheBackend> SelectedLiveCache for CacheResidencyManager<M> {
    fn open(selected: &crate::SelectedStateRealization) -> Result<Option<Self>, BackendFailure> {
        selected.create_live_cache()
    }
}

impl SelectedLiveCache for () {
    fn open(selected: &crate::SelectedStateRealization) -> Result<Option<Self>, BackendFailure> {
        if selected
            .paging_options()
            .map_err(BackendFailure::from_error)?
            .is_some()
        {
            return Err(BackendFailure::from_error(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "state storage cannot bind a live page cache",
            )));
        }
        Ok(None)
    }
}

/// Owner of a live cache catalog, its transfer workers, and tier budgets.
///
/// Clones share one catalog; fork and snapshot operations make a new catalog
/// with independent leases and publication authority.
#[derive(Debug, Clone)]
pub struct CacheResidencyManager<M: LiveCacheBackend> {
    session_id: u64,
    inner: Arc<CacheResidencyManagerInner<M>>,
}

/// Structured cache residency and persistence failures.
#[derive(Debug, thiserror::Error)]
pub enum CacheResidencyError {
    /// Store publication, format or capture allowance failure.
    #[error(transparent)]
    PromptStore(#[from] PromptCacheStoreError),
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
    /// A native storage or transfer operation failed.
    #[error(transparent)]
    Payload(#[from] BackendFailure),
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
    /// An indexed attention request addressed past the retained frontier.
    #[error("indexed cache position {position} exceeds retained range 0..{offset}")]
    IndexedPositionOutOfRange {
        /// Requested absolute position.
        position: i32,
        /// Exclusive retained frontier.
        offset: i64,
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
    /// Cache state could not be restored after a failed update.
    #[error("{operation}; cache rollback also failed: {rollback}")]
    Rollback {
        /// Original failure of the cache operation.
        #[source]
        operation: Box<CacheResidencyError>,
        /// Failure restoring the prior accounting and pages.
        rollback: Box<CacheResidencyError>,
    },
    /// A background cache task failed. Keep its typed cause for every waiter.
    #[error(transparent)]
    Background(#[from] Arc<CacheResidencyError>),
    /// Cache geometry, accounting or worker operation failed.
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
mod tests;

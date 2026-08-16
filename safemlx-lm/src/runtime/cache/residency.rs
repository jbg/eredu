//! Block-addressable residency for mutable attention state.
//!
//! This module is deliberately independent from weight residency. Attention
//! blocks are mutable activation state until sealed, while checkpoint weights
//! are immutable inputs with a different ownership and persistence model.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    num::NonZeroU32,
    panic::{catch_unwind, AssertUnwindSafe},
    path::{Component, Path, PathBuf},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc::{self, SyncSender, TrySendError},
        Arc, Condvar, Mutex, MutexGuard, OnceLock, Weak,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use memmap2::{Mmap, MmapOptions};
use safemlx::{
    host_transfer_capacity_upper_bound,
    transforms::{async_eval_with_event, eval},
    Array, Device, DeviceType, Dtype, Event, HostTransferBuffer, HostTransferPolicy,
    ImmutableHostTransferBuffer, Stream,
};
use safetensors::tensor::{serialize_to_file, Dtype as StoredDtype, TensorView};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    core::residency::CacheEvictionPolicy,
    runtime::attention::{AttentionPolicy, LayerSchedule},
};

const PROMPT_CACHE_SCHEMA_VERSION: u32 = 7;
const MAX_PROMPT_CACHE_SHARD_HEADER_BYTES: u64 = 1024 * 1024;
const PROMPT_CACHE_GENERATIONS_DIRECTORY: &str = ".generations";
const PROMPT_CACHE_CURRENT_FILE: &str = "CURRENT";
pub(crate) const PAGED_CACHE_PREFETCH_BLOCKS: usize = 2;
static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_CACHE_POOL_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_CACHE_POOL_RESERVATION_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_LIVE_SHARD_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_HOST_WRITE_RESERVATION_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_HOST_DEMOTION_ID: AtomicU64 = AtomicU64::new(1);
static LIVE_PROCESS_NAMESPACE: OnceLock<String> = OnceLock::new();

/// Selects the existing device-resident cache or bounded paged residency.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub enum CacheResidencyPolicy {
    /// Keep the existing cache representation entirely device resident.
    #[default]
    Device,
    /// Store sealed state in token-addressable blocks under finite budgets.
    Paged(PagedCacheOptions),
}

/// Process-wide finite limits shared by independently owned live caches.
///
/// Per-cache limits remain in [`PagedCacheOptions`] so one request cannot
/// monopolize a pool. These limits are the additional aggregate admission
/// boundary across every manager attached to the pool.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct CachePoolLimits {
    device_bytes: u64,
    host_bytes: u64,
    transfer_in_flight_bytes: u64,
    disk_bytes: u64,
}

impl CachePoolLimits {
    /// Creates finite aggregate cache limits. Device and transfer capacity
    /// must be nonzero; zero host or disk capacity disables that tier.
    pub fn new(
        device_bytes: u64,
        host_bytes: u64,
        transfer_in_flight_bytes: u64,
        disk_bytes: u64,
    ) -> Result<Self, CacheResidencyError> {
        if device_bytes == 0 {
            return Err(CacheResidencyError::InvalidOptions(
                "cache pool device budget must be nonzero".into(),
            ));
        }
        if transfer_in_flight_bytes == 0 {
            return Err(CacheResidencyError::InvalidOptions(
                "cache pool transfer-in-flight budget must be nonzero".into(),
            ));
        }
        Ok(Self {
            device_bytes,
            host_bytes,
            transfer_in_flight_bytes,
            disk_bytes,
        })
    }

    /// Aggregate device-cache capacity.
    pub const fn device_bytes(self) -> u64 {
        self.device_bytes
    }

    /// Aggregate physical host-transfer allocation capacity.
    pub const fn host_bytes(self) -> u64 {
        self.host_bytes
    }

    /// Aggregate bytes retained by asynchronous transfers.
    pub const fn transfer_in_flight_bytes(self) -> u64 {
        self.transfer_in_flight_bytes
    }

    /// Aggregate live-cache disk capacity. Zero disables live disk storage.
    pub const fn disk_bytes(self) -> u64 {
        self.disk_bytes
    }
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
struct CachePoolUsage {
    device_bytes: u64,
    host_bytes: u64,
    transfer_in_flight_bytes: u64,
    disk_bytes: u64,
}

impl CachePoolUsage {
    fn saturating_add(self, other: Self) -> Self {
        Self {
            device_bytes: self.device_bytes.saturating_add(other.device_bytes),
            host_bytes: self.host_bytes.saturating_add(other.host_bytes),
            transfer_in_flight_bytes: self
                .transfer_in_flight_bytes
                .saturating_add(other.transfer_in_flight_bytes),
            disk_bytes: self.disk_bytes.saturating_add(other.disk_bytes),
        }
    }

    fn saturating_sub(self, other: Self) -> Self {
        Self {
            device_bytes: self.device_bytes.saturating_sub(other.device_bytes),
            host_bytes: self.host_bytes.saturating_sub(other.host_bytes),
            transfer_in_flight_bytes: self
                .transfer_in_flight_bytes
                .saturating_sub(other.transfer_in_flight_bytes),
            disk_bytes: self.disk_bytes.saturating_sub(other.disk_bytes),
        }
    }
}

/// Aggregate process-pool occupancy and high-water marks.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct CachePoolReport {
    /// Stable process-local pool identity.
    pub pool_id: u64,
    /// Number of live cache managers contributing occupancy.
    pub managers: usize,
    /// Current aggregate device bytes.
    pub current_device_bytes: u64,
    /// Peak aggregate device bytes successfully admitted.
    pub peak_device_bytes: u64,
    /// Current aggregate physical host-transfer allocation capacity.
    pub current_host_bytes: u64,
    /// Peak aggregate physical host-transfer capacity successfully admitted.
    pub peak_host_bytes: u64,
    /// Current bytes retained by in-flight transfers.
    pub current_transfer_in_flight_bytes: u64,
    /// Peak bytes retained by in-flight transfers.
    pub peak_transfer_in_flight_bytes: u64,
    /// Current disk bytes, including pending write reservations.
    pub current_disk_bytes: u64,
    /// Peak disk bytes successfully admitted.
    pub peak_disk_bytes: u64,
    /// Aggregate finite limits.
    pub limits: CachePoolLimits,
}

#[derive(Debug)]
struct CachePoolState {
    managers: HashMap<u64, CachePoolUsage>,
    reservations: HashMap<u64, CachePoolUsage>,
    current: CachePoolUsage,
    peak: CachePoolUsage,
}

/// Shareable aggregate accounting pool for scheduler-owned and standalone caches.
#[derive(Clone)]
pub struct CacheResidencyPool {
    id: u64,
    limits: CachePoolLimits,
    state: Arc<Mutex<CachePoolState>>,
}

impl std::fmt::Debug for CacheResidencyPool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CacheResidencyPool")
            .field("id", &self.id)
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

impl PartialEq for CacheResidencyPool {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.state, &other.state)
    }
}

impl Eq for CacheResidencyPool {}

impl CacheResidencyPool {
    /// Creates an empty process pool under aggregate finite limits.
    pub fn new(limits: CachePoolLimits) -> Self {
        Self {
            id: NEXT_CACHE_POOL_ID.fetch_add(1, Ordering::Relaxed),
            limits,
            state: Arc::new(Mutex::new(CachePoolState {
                managers: HashMap::new(),
                reservations: HashMap::new(),
                current: CachePoolUsage::default(),
                peak: CachePoolUsage::default(),
            })),
        }
    }

    /// Creates a pool whose aggregate limits match one paged-cache policy.
    /// This is useful for a scheduler that learns its cache policy on the first
    /// request and then binds subsequent requests to the same pool.
    pub fn for_paged_options(options: &PagedCacheOptions) -> Result<Self, CacheResidencyError> {
        let disk_bytes = match options.live_disk_policy() {
            LiveCacheDiskPolicy::Disabled => 0,
            LiveCacheDiskPolicy::Enabled { budget_bytes, .. } => *budget_bytes,
        };
        // A transition can retain one source allocation while a concurrent
        // promotion or demotion owns another transfer reservation. Once host
        // backing is page-rounded, twice the larger tier is the smallest safe
        // implicit bound; transfer bytes are a throttle, not extra residency.
        let transfer_bytes = options
            .device_budget_bytes()
            .max(options.host_budget_bytes())
            .saturating_mul(2)
            .max(1);
        Ok(Self::new(CachePoolLimits::new(
            options.device_budget_bytes(),
            options.host_budget_bytes(),
            transfer_bytes,
            disk_bytes,
        )?))
    }

    /// Returns this pool's stable process-local identity.
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// Returns aggregate finite limits.
    pub const fn limits(&self) -> CachePoolLimits {
        self.limits
    }

    /// Returns aggregate current occupancy and high-water marks.
    pub fn report(&self) -> Result<CachePoolReport, CacheResidencyError> {
        let state = self
            .state
            .lock()
            .map_err(|_| CacheResidencyError::PoolPoisoned)?;
        Ok(CachePoolReport {
            pool_id: self.id,
            managers: state.managers.len(),
            current_device_bytes: state.current.device_bytes,
            peak_device_bytes: state.peak.device_bytes,
            current_host_bytes: state.current.host_bytes,
            peak_host_bytes: state.peak.host_bytes,
            current_transfer_in_flight_bytes: state.current.transfer_in_flight_bytes,
            peak_transfer_in_flight_bytes: state.peak.transfer_in_flight_bytes,
            current_disk_bytes: state.current.disk_bytes,
            peak_disk_bytes: state.peak.disk_bytes,
            limits: self.limits,
        })
    }

    fn update_manager(
        &self,
        manager: u64,
        usage: CachePoolUsage,
    ) -> Result<CachePoolUsage, CacheResidencyError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| CacheResidencyError::PoolPoisoned)?;
        let previous = state.managers.insert(manager, usage).unwrap_or_default();
        state.current = state.current.saturating_sub(previous).saturating_add(usage);
        if state.current.device_bytes <= self.limits.device_bytes {
            state.peak.device_bytes = state.peak.device_bytes.max(state.current.device_bytes);
        }
        if state.current.host_bytes <= self.limits.host_bytes {
            state.peak.host_bytes = state.peak.host_bytes.max(state.current.host_bytes);
        }
        if state.current.transfer_in_flight_bytes <= self.limits.transfer_in_flight_bytes {
            state.peak.transfer_in_flight_bytes = state
                .peak
                .transfer_in_flight_bytes
                .max(state.current.transfer_in_flight_bytes);
        }
        if state.current.disk_bytes <= self.limits.disk_bytes {
            state.peak.disk_bytes = state.peak.disk_bytes.max(state.current.disk_bytes);
        }
        Ok(state.current)
    }

    fn remove_manager(&self, manager: u64) {
        if let Ok(mut state) = self.state.lock() {
            if let Some(previous) = state.managers.remove(&manager) {
                state.current = state.current.saturating_sub(previous);
            }
        }
    }

    fn reserve_transfer(&self, bytes: u64) -> Result<CachePoolReservation, CacheResidencyError> {
        self.reserve_additional(CachePoolUsage {
            transfer_in_flight_bytes: bytes,
            ..CachePoolUsage::default()
        })
    }

    fn reserve_additional(
        &self,
        usage: CachePoolUsage,
    ) -> Result<CachePoolReservation, CacheResidencyError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| CacheResidencyError::PoolPoisoned)?;
        Self::validate_additional(state.current, usage, self.limits)?;
        let required = state.current.saturating_add(usage);
        let reservation = NEXT_CACHE_POOL_RESERVATION_ID.fetch_add(1, Ordering::Relaxed);
        state.reservations.insert(reservation, usage);
        state.current = required;
        if required.device_bytes <= self.limits.device_bytes {
            state.peak.device_bytes = state.peak.device_bytes.max(required.device_bytes);
        }
        if required.host_bytes <= self.limits.host_bytes {
            state.peak.host_bytes = state.peak.host_bytes.max(required.host_bytes);
        }
        if required.transfer_in_flight_bytes <= self.limits.transfer_in_flight_bytes {
            state.peak.transfer_in_flight_bytes = state
                .peak
                .transfer_in_flight_bytes
                .max(required.transfer_in_flight_bytes);
        }
        if required.disk_bytes <= self.limits.disk_bytes {
            state.peak.disk_bytes = state.peak.disk_bytes.max(required.disk_bytes);
        }
        Ok(CachePoolReservation {
            reservation,
            pool: self.clone(),
        })
    }

    fn validate_additional(
        current: CachePoolUsage,
        usage: CachePoolUsage,
        limits: CachePoolLimits,
    ) -> Result<(), CacheResidencyError> {
        let required = current.saturating_add(usage);
        for (resource, additional, required, budget) in [
            (
                CachePoolResource::Device,
                usage.device_bytes,
                required.device_bytes,
                limits.device_bytes,
            ),
            (
                CachePoolResource::Host,
                usage.host_bytes,
                required.host_bytes,
                limits.host_bytes,
            ),
            (
                CachePoolResource::TransferInFlight,
                usage.transfer_in_flight_bytes,
                required.transfer_in_flight_bytes,
                limits.transfer_in_flight_bytes,
            ),
            (
                CachePoolResource::Disk,
                usage.disk_bytes,
                required.disk_bytes,
                limits.disk_bytes,
            ),
        ] {
            if additional != 0 && required > budget {
                return Err(CacheResidencyError::PoolBudgetExceeded {
                    resource,
                    required,
                    budget,
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
struct CachePoolReservation {
    reservation: u64,
    pool: CacheResidencyPool,
}

impl Drop for CachePoolReservation {
    fn drop(&mut self) {
        if let Ok(mut state) = self.pool.state.lock() {
            if let Some(usage) = state.reservations.remove(&self.reservation) {
                state.current = state.current.saturating_sub(usage);
            }
        }
    }
}

#[derive(Debug)]
struct CachePoolMembership {
    manager: u64,
    pool: CacheResidencyPool,
}

impl Drop for CachePoolMembership {
    fn drop(&mut self) {
        self.pool.remove_manager(self.manager);
    }
}

/// Resource axis named by an aggregate pool admission error.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum CachePoolResource {
    /// Device-resident cache allocations.
    Device,
    /// Host-transfer cache allocations.
    Host,
    /// Buffers retained until asynchronous transfer completion.
    TransferInFlight,
    /// Live-cache disk files and pending write reservations.
    Disk,
}

/// Controls optional disk backing for a live inference cache.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub enum LiveCacheDiskPolicy {
    /// Do not write live attention state to disk.
    #[default]
    Disabled,
    /// Retain demoted sealed blocks in an explicit ephemeral directory.
    Enabled {
        /// Directory dedicated to this live cache.
        directory: PathBuf,
        /// Finite logical byte limit for live cache files.
        budget_bytes: u64,
        /// Bound on pending reader or writer requests.
        queue_capacity: usize,
    },
}

/// Validated finite limits for a paged attention cache.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PagedCacheOptions {
    block_size_tokens: i32,
    device_budget_bytes: u64,
    host_budget_bytes: u64,
    recent_device_blocks: usize,
    eviction_policy: CacheEvictionPolicy,
    full_attention: bool,
    retain_discarded_for_persistence: bool,
    live_disk: LiveCacheDiskPolicy,
    sample_process: bool,
    pool: Option<Arc<CacheResidencyPool>>,
}

impl PagedCacheOptions {
    /// Creates paged-cache limits. Every memory limit is finite and explicit.
    pub fn new(
        block_size_tokens: i32,
        device_budget_bytes: u64,
        host_budget_bytes: u64,
        recent_device_blocks: usize,
    ) -> Result<Self, CacheResidencyError> {
        if block_size_tokens <= 0 {
            return Err(CacheResidencyError::InvalidOptions(
                "cache block size must be positive".into(),
            ));
        }
        if device_budget_bytes == 0 {
            return Err(CacheResidencyError::InvalidOptions(
                "paged cache device budget must be nonzero".into(),
            ));
        }
        if recent_device_blocks == 0 {
            return Err(CacheResidencyError::InvalidOptions(
                "paged cache must protect at least one recent device block".into(),
            ));
        }
        Ok(Self {
            block_size_tokens,
            device_budget_bytes,
            host_budget_bytes,
            recent_device_blocks,
            eviction_policy: CacheEvictionPolicy::LeastRecentlyUsed,
            full_attention: false,
            retain_discarded_for_persistence: false,
            live_disk: LiveCacheDiskPolicy::Disabled,
            sample_process: false,
            pool: None,
        })
    }

    /// Enables exact blockwise full-context attention.
    pub const fn with_full_attention(mut self, enabled: bool) -> Self {
        self.full_attention = enabled;
        self
    }

    /// Retains blocks older than a sliding window solely for later persistence.
    pub const fn with_persistence_retention(mut self, enabled: bool) -> Self {
        self.retain_discarded_for_persistence = enabled;
        self
    }

    /// Selects deterministic block eviction ordering.
    pub const fn with_eviction_policy(mut self, policy: CacheEvictionPolicy) -> Self {
        self.eviction_policy = policy;
        self
    }

    /// Configures explicit live disk backing.
    pub fn with_live_disk(
        mut self,
        directory: impl Into<PathBuf>,
        budget_bytes: u64,
        queue_capacity: usize,
    ) -> Result<Self, CacheResidencyError> {
        if budget_bytes == 0 {
            return Err(CacheResidencyError::InvalidOptions(
                "live cache disk budget must be nonzero".into(),
            ));
        }
        if queue_capacity == 0 {
            return Err(CacheResidencyError::InvalidOptions(
                "live cache disk queue capacity must be nonzero".into(),
            ));
        }
        let directory = directory.into();
        if directory.as_os_str().is_empty() {
            return Err(CacheResidencyError::InvalidOptions(
                "live cache disk directory must not be empty".into(),
            ));
        }
        self.live_disk = LiveCacheDiskPolicy::Enabled {
            directory,
            budget_bytes,
            queue_capacity,
        };
        Ok(self)
    }

    /// Enables optional process-memory sampling in reports.
    pub const fn with_process_sampling(mut self, enabled: bool) -> Self {
        self.sample_process = enabled;
        self
    }

    /// Attaches this per-cache policy to an aggregate process pool.
    ///
    /// Clones retain the same pool identity, allowing high-level, tensor-
    /// parallel, expert-parallel, and pipeline constructors to share one
    /// scheduler-owned accounting boundary without architecture dispatch.
    pub fn with_pool(mut self, pool: CacheResidencyPool) -> Result<Self, CacheResidencyError> {
        if self.device_budget_bytes > pool.limits().device_bytes() {
            return Err(CacheResidencyError::InvalidOptions(format!(
                "per-cache device budget {} exceeds cache pool budget {}",
                self.device_budget_bytes,
                pool.limits().device_bytes()
            )));
        }
        if self.host_budget_bytes > pool.limits().host_bytes() {
            return Err(CacheResidencyError::InvalidOptions(format!(
                "per-cache host budget {} exceeds cache pool budget {}",
                self.host_budget_bytes,
                pool.limits().host_bytes()
            )));
        }
        if let LiveCacheDiskPolicy::Enabled { budget_bytes, .. } = &self.live_disk {
            if *budget_bytes > pool.limits().disk_bytes() {
                return Err(CacheResidencyError::InvalidOptions(format!(
                    "per-cache disk budget {budget_bytes} exceeds cache pool budget {}",
                    pool.limits().disk_bytes()
                )));
            }
        }
        self.pool = Some(Arc::new(pool));
        Ok(self)
    }

    /// Returns the block size in tokens.
    pub const fn block_size_tokens(&self) -> i32 {
        self.block_size_tokens
    }

    /// Returns the finite logical device-cache budget.
    pub const fn device_budget_bytes(&self) -> u64 {
        self.device_budget_bytes
    }

    /// Returns the finite physical host-transfer allocation budget.
    pub const fn host_budget_bytes(&self) -> u64 {
        self.host_budget_bytes
    }

    /// Returns the recent block count protected on the execution device per layer.
    pub const fn recent_device_blocks(&self) -> usize {
        self.recent_device_blocks
    }

    /// Returns whether exact blockwise full attention is enabled.
    pub const fn full_attention_enabled(&self) -> bool {
        self.full_attention
    }

    /// Returns whether discarded sliding state is retained for persistence.
    pub const fn retains_discarded_for_persistence(&self) -> bool {
        self.retain_discarded_for_persistence
    }

    /// Returns the live disk policy.
    pub const fn live_disk_policy(&self) -> &LiveCacheDiskPolicy {
        &self.live_disk
    }

    /// Returns the explicitly attached aggregate pool, if any.
    pub fn pool(&self) -> Option<&CacheResidencyPool> {
        self.pool.as_deref()
    }
}

/// Representation stored atomically in one cache block.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheRepresentation {
    /// Standard attention keys and values.
    KeyValue,
    /// DeepSeek compressed latent state and rotary keys.
    CompressedLatentRotary,
}

/// Optional rank identity included in a stable cache block identifier.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct CacheRankIdentity {
    /// Pipeline rank, when pipeline partitioning is active.
    pub pipeline_rank: Option<usize>,
    /// Tensor-parallel rank, when cache heads are sharded.
    pub tensor_parallel_rank: Option<usize>,
    /// Expert-parallel rank for replicated attention state.
    pub expert_parallel_rank: Option<usize>,
}

/// Stable identity for one immutable sealed cache block.
#[derive(Debug, Clone, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct CacheBlockId {
    /// Identity shared by every block in one live cache.
    pub session_id: u64,
    /// Architecture-global decoder layer index.
    pub global_layer: usize,
    /// Stored attention representation.
    pub representation: CacheRepresentation,
    /// Inclusive absolute token position.
    pub start: i64,
    /// Exclusive absolute token position.
    pub end: i64,
    /// Rank-local ownership identity.
    pub rank: Option<CacheRankIdentity>,
}

/// Logical location of a sealed cache block.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheTier {
    /// Available to execution without a catalog load.
    Device,
    /// Evaluated CPU-resident state with no execution-device copy retained by the manager.
    Host,
    /// Stored in a live or persistent safetensors shard.
    Disk,
}

/// Lifecycle state visible through cache diagnostics.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum CacheBlockLifecycle {
    /// A layer-owned append target that has not been sealed.
    MutableDeviceTail,
    /// Immutable state kept on the execution device.
    SealedDevice,
    /// Immutable evaluated host state.
    SealedHost,
    /// Immutable disk-backed state.
    DiskBacked,
    /// A shared background disk transfer is queued or running.
    InFlight,
    /// State removed from live attention and persistence retention.
    Discarded,
    /// Read-only state cataloged from a prompt cache.
    ImportedReadOnly,
}

#[derive(Debug, Clone)]
pub(crate) enum CacheBlockArrays {
    KeyValue { keys: Array, values: Array },
    CompressedLatentRotary { latent: Array, rotary_key: Array },
}

/// One resident attention payload prepared for atomic prompt-cache publication.
pub(crate) struct PromptCacheSnapshotBlock {
    pub(crate) global_layer: usize,
    pub(crate) start: i64,
    pub(crate) end: i64,
    pub(crate) rank: Option<CacheRankIdentity>,
    pub(crate) arrays: CacheBlockArrays,
}

/// One materialized attention payload loaded from a prompt-cache snapshot.
pub(crate) struct LoadedPromptCacheBlock {
    pub(crate) global_layer: usize,
    pub(crate) start: i64,
    pub(crate) end: i64,
    pub(crate) arrays: CacheBlockArrays,
}

impl CacheBlockArrays {
    pub(crate) fn representation(&self) -> CacheRepresentation {
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
            .name("safemlx-cache-host-demotion".into())
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
                path: PathBuf::from("safemlx-cache-host-demotion"),
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

#[derive(Debug, Clone)]
enum CacheBlockPhysicalState {
    Device {
        arrays: CacheBlockArrays,
        disk: Option<DiskLocation>,
    },
    Demoting {
        arrays: CacheBlockArrays,
        ticket: HostDemotionTicket,
    },
    Host {
        block: HostCacheBlock,
        persistence: HostCachePersistence,
    },
    Disk {
        location: DiskLocation,
        read: DiskCacheReadState,
    },
}

/// Persistence state valid while a block is physically host-resident.
///
/// A completed backing and an in-flight write are mutually exclusive by
/// construction. This prevents a host block from simultaneously claiming an
/// authoritative disk copy and a write that has not published one yet.
#[derive(Debug, Clone)]
enum HostCachePersistence {
    Unbacked,
    Backed(DiskLocation),
    Writing(PendingDiskOperation),
}

/// Read state valid while a block is physically disk-resident.
#[derive(Debug, Clone)]
enum DiskCacheReadState {
    Ready,
    Reading {
        pending: PendingDiskOperation,
        reserved_host_bytes: u64,
    },
}

impl CacheBlockPhysicalState {
    fn tier(&self) -> CacheTier {
        match self {
            Self::Device { .. } | Self::Demoting { .. } => CacheTier::Device,
            Self::Host { .. } => CacheTier::Host,
            Self::Disk { .. } => CacheTier::Disk,
        }
    }

    fn disk(&self) -> Option<&DiskLocation> {
        match self {
            Self::Device { disk, .. } => disk.as_ref(),
            Self::Demoting { .. } => None,
            Self::Host {
                persistence: HostCachePersistence::Backed(location),
                ..
            } => Some(location),
            Self::Host {
                persistence: HostCachePersistence::Unbacked | HostCachePersistence::Writing(_),
                ..
            } => None,
            Self::Disk { location, .. } => Some(location),
        }
    }

    fn pending(&self) -> Option<&PendingDiskOperation> {
        match self {
            Self::Host {
                persistence: HostCachePersistence::Writing(pending),
                ..
            }
            | Self::Disk {
                read: DiskCacheReadState::Reading { pending, .. },
                ..
            } => Some(pending),
            Self::Device { .. }
            | Self::Demoting { .. }
            | Self::Host { .. }
            | Self::Disk {
                read: DiskCacheReadState::Ready,
                ..
            } => None,
        }
    }

    fn pending_matches(&self, key: &DiskOperationKey) -> bool {
        self.pending()
            .is_some_and(|pending| pending.ticket.key == *key)
    }

    fn clear_pending(&mut self, key: &DiskOperationKey) {
        match self.clone() {
            Self::Host {
                block,
                persistence: HostCachePersistence::Writing(pending),
            } if pending.ticket.key == *key => {
                *self = Self::Host {
                    block,
                    persistence: HostCachePersistence::Unbacked,
                };
            }
            Self::Disk {
                location,
                read: DiskCacheReadState::Reading { pending, .. },
            } if pending.ticket.key == *key => {
                *self = Self::Disk {
                    location,
                    read: DiskCacheReadState::Ready,
                };
            }
            Self::Device { .. } | Self::Demoting { .. } | Self::Host { .. } | Self::Disk { .. } => {
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

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
enum DiskOperationKind {
    Write,
    Read,
}

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
struct DiskOperationKey {
    generation: u64,
    id: CacheBlockId,
    kind: DiskOperationKind,
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

enum DiskRequest {
    Operation {
        key: DiskOperationKey,
        task: Box<DiskTask>,
        completion: Arc<DiskCompletion>,
    },
    Stop,
}

struct DiskWriteCommit {
    state: Weak<Mutex<CacheManagerState>>,
    key: DiskOperationKey,
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
                if let Some(record) = state.blocks.get_mut(&self.key.id) {
                    if record.physical.pending_matches(&self.key) {
                        bytes = record.bytes;
                        if matches!(record.physical, CacheBlockPhysicalState::Host { .. }) {
                            debug_assert_eq!(record.leases, 0);
                            record.physical = CacheBlockPhysicalState::Disk {
                                location: location.clone(),
                                read: DiskCacheReadState::Ready,
                            };
                            transitioned_to_disk = true;
                        }
                    }
                }
                if bytes != 0 {
                    state.counters.report.transfer_bytes += bytes;
                    state
                        .layer_activity_mut(self.key.id.global_layer)
                        .transfer_bytes += bytes;
                }
                if transitioned_to_disk {
                    state.counters.report.disk_demotions += 1;
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
                state.counters.report.failures += 1;
                state.layer_activity_mut(self.key.id.global_layer).failures += 1;
                state.background_disk_error =
                    Some("cache disk worker returned an unexpected write result".into());
            }
            Err(_) if stale => {}
            Err(error) => {
                if let Some(record) = state.blocks.get_mut(&self.key.id) {
                    record.physical.clear_pending(&self.key);
                }
                state.counters.report.failures += 1;
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
            record.physical.clear_pending(&self.key);
        }
        if state.host_write_reservations.remove(&self.key).is_some() {
            update_report_totals(&mut state);
        }
    }
}

#[derive(Debug, Clone)]
enum DiskCompletionState {
    Finished(Result<DiskResult, String>),
    Cancelled,
}

#[derive(Debug, Default)]
struct DiskCompletion {
    state: Mutex<Option<DiskCompletionState>>,
    ready: Condvar,
    released: Mutex<bool>,
    released_ready: Condvar,
}

impl DiskCompletion {
    fn finish(&self, result: Result<DiskResult, CacheResidencyError>) {
        if let Ok(mut state) = self.state.lock() {
            if state.is_none() {
                *state = Some(DiskCompletionState::Finished(
                    result.map_err(|error| error.to_string()),
                ));
                self.ready.notify_all();
            }
        }
    }

    fn cancel(&self) -> bool {
        let Ok(mut state) = self.state.lock() else {
            return false;
        };
        if state.is_some() {
            return false;
        }
        *state = Some(DiskCompletionState::Cancelled);
        self.ready.notify_all();
        true
    }

    fn is_ready(&self) -> bool {
        self.state.lock().map_or(true, |state| state.is_some())
    }

    fn wait(&self, generation: u64) -> Result<DiskResult, CacheResidencyError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| CacheResidencyError::ManagerPoisoned)?;
        while state.is_none() {
            state = self
                .ready
                .wait(state)
                .map_err(|_| CacheResidencyError::ManagerPoisoned)?;
        }
        match state.as_ref().expect("completion state was awaited") {
            DiskCompletionState::Finished(Ok(result)) => Ok(result.clone()),
            DiskCompletionState::Finished(Err(error)) => {
                Err(CacheResidencyError::Runtime(error.clone()))
            }
            DiskCompletionState::Cancelled => {
                Err(CacheResidencyError::DiskOperationCancelled { generation })
            }
        }
    }

    fn release_task_resources(&self) {
        if let Ok(mut released) = self.released.lock() {
            *released = true;
            self.released_ready.notify_all();
        }
    }

    fn wait_for_task_resources(&self) -> Result<(), CacheResidencyError> {
        let mut released = self
            .released
            .lock()
            .map_err(|_| CacheResidencyError::ManagerPoisoned)?;
        while !*released {
            released = self
                .released_ready
                .wait(released)
                .map_err(|_| CacheResidencyError::ManagerPoisoned)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct DiskTicket {
    key: DiskOperationKey,
    completion: Arc<DiskCompletion>,
    shared: Arc<DiskWorkerShared>,
}

impl DiskTicket {
    fn wait(&self) -> Result<DiskResult, CacheResidencyError> {
        self.completion.wait(self.key.generation)
    }

    fn cancel(&self) -> bool {
        let Ok(_space) = self.shared.space.lock() else {
            return false;
        };
        let cancelled = self.completion.cancel();
        self.shared.space_available.notify_all();
        cancelled
    }

    fn wait_for_task_resources(&self) -> Result<(), CacheResidencyError> {
        self.completion.wait_for_task_resources()
    }
}

struct DiskSubmission {
    ticket: DiskTicket,
    sender: SyncSender<DiskRequest>,
    shared: Arc<DiskWorkerShared>,
    unsent: Option<DiskRequest>,
    joined: bool,
    write_reservation_id: Option<u64>,
}

#[derive(Debug)]
struct DiskSubmissionOutcome {
    joined: bool,
    backpressure: bool,
    peak_occupancy: usize,
}

impl DiskSubmission {
    fn enqueue(mut self) -> Result<DiskSubmissionOutcome, CacheResidencyError> {
        let mut backpressure = false;
        if let Some(mut request) = self.unsent.take() {
            let mut space = match self.shared.space.lock() {
                Ok(space) => space,
                Err(_) => {
                    drop(request);
                    self.ticket.completion.release_task_resources();
                    return Err(CacheResidencyError::ManagerPoisoned);
                }
            };
            loop {
                if self.ticket.completion.is_ready() {
                    drop(request);
                    self.ticket.completion.release_task_resources();
                    break;
                }
                match self.sender.try_send(request) {
                    Ok(()) => {
                        let occupancy = self.shared.queued.fetch_add(1, Ordering::AcqRel) + 1;
                        update_atomic_max(
                            &self.shared.peak_occupancy,
                            occupancy.min(self.shared.capacity),
                        );
                        break;
                    }
                    Err(TrySendError::Full(returned)) => {
                        request = returned;
                        backpressure = true;
                        space = match self.shared.space_available.wait(space) {
                            Ok(space) => space,
                            Err(_) => {
                                drop(request);
                                self.ticket.completion.release_task_resources();
                                return Err(CacheResidencyError::ManagerPoisoned);
                            }
                        };
                    }
                    Err(TrySendError::Disconnected(_)) => {
                        self.ticket
                            .completion
                            .finish(Err(CacheResidencyError::Runtime(
                                "live cache disk worker stopped".into(),
                            )));
                        self.ticket.completion.release_task_resources();
                        break;
                    }
                }
            }
        }
        Ok(DiskSubmissionOutcome {
            joined: self.joined,
            backpressure,
            peak_occupancy: self.shared.peak_occupancy.load(Ordering::Acquire),
        })
    }
}

impl Drop for DiskSubmission {
    fn drop(&mut self) {
        let Some(request) = self.unsent.take() else {
            return;
        };
        self.ticket.cancel();
        drop(request);
        self.ticket.completion.release_task_resources();
        retire_disk_completion(&self.shared, &self.ticket.key, &self.ticket.completion);
    }
}

#[derive(Debug)]
struct DiskWorkerShared {
    in_flight: Mutex<HashMap<DiskOperationKey, Arc<DiskCompletion>>>,
    space: Mutex<()>,
    space_available: Condvar,
    queued: AtomicUsize,
    peak_occupancy: AtomicUsize,
    capacity: usize,
}

impl DiskWorkerShared {
    fn new(capacity: usize) -> Self {
        Self {
            in_flight: Mutex::new(HashMap::new()),
            space: Mutex::new(()),
            space_available: Condvar::new(),
            queued: AtomicUsize::new(0),
            peak_occupancy: AtomicUsize::new(0),
            capacity,
        }
    }
}

fn retire_disk_completion(
    shared: &DiskWorkerShared,
    key: &DiskOperationKey,
    completion: &Arc<DiskCompletion>,
) {
    if let Ok(mut in_flight) = shared.in_flight.lock() {
        if in_flight
            .get(key)
            .is_some_and(|current| Arc::ptr_eq(current, completion))
        {
            in_flight.remove(key);
        }
    }
}

#[derive(Debug)]
struct DiskWorker {
    sender: SyncSender<DiskRequest>,
    handle: Mutex<Option<JoinHandle<()>>>,
    shared: Arc<DiskWorkerShared>,
}

impl DiskWorker {
    fn new(capacity: usize) -> Result<Self, CacheResidencyError> {
        let (sender, receiver) = mpsc::sync_channel::<DiskRequest>(capacity);
        let shared = Arc::new(DiskWorkerShared::new(capacity));
        let worker_shared = Arc::clone(&shared);
        let handle = thread::Builder::new()
            .name("safemlx-cache-disk".into())
            .spawn(move || {
                while let Ok(request) = receiver.recv() {
                    match request {
                        DiskRequest::Operation {
                            key,
                            task,
                            completion,
                        } => {
                            if let Ok(_space) = worker_shared.space.lock() {
                                worker_shared.queued.fetch_sub(1, Ordering::AcqRel);
                                worker_shared.space_available.notify_all();
                            }
                            if completion.is_ready() {
                                drop(task);
                                completion.release_task_resources();
                                retire_disk_completion(&worker_shared, &key, &completion);
                                continue;
                            }
                            let mut write_commit = None;
                            let result = catch_unwind(AssertUnwindSafe(|| match *task {
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
                                } => load_host_cache_block_direct(&location, representation)
                                    .map(DiskResult::Read),
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
                            // The task-local arrays have been dropped by this
                            // point. Release their reservation before waking
                            // logical completion waiters.
                            drop(write_commit);
                            if completion.is_ready() {
                                if let Ok(DiskResult::Write(location)) = result {
                                    if !location.persistent {
                                        let _ = fs::remove_file(location.path);
                                    }
                                }
                            } else {
                                completion.finish(result);
                            }
                            completion.release_task_resources();
                            retire_disk_completion(&worker_shared, &key, &completion);
                        }
                        DiskRequest::Stop => break,
                    }
                }
            })
            .map_err(|source| CacheResidencyError::Io {
                action: "start live cache disk worker",
                path: PathBuf::from("safemlx-cache-disk"),
                source,
            })?;
        Ok(Self {
            sender,
            handle: Mutex::new(Some(handle)),
            shared,
        })
    }

    fn prepare(
        &self,
        key: DiskOperationKey,
        task: DiskTask,
    ) -> Result<DiskSubmission, CacheResidencyError> {
        self.prepare_with_write_reservation(key, task, None)
    }

    fn prepare_with_write_reservation(
        &self,
        key: DiskOperationKey,
        mut task: DiskTask,
        write_reservation_id: Option<u64>,
    ) -> Result<DiskSubmission, CacheResidencyError> {
        let mut in_flight = self
            .shared
            .in_flight
            .lock()
            .map_err(|_| CacheResidencyError::ManagerPoisoned)?;
        if let Some(completion) = in_flight.get(&key) {
            if let DiskTask::Write {
                commit: Some(commit),
                ..
            } = &mut task
            {
                commit.armed = false;
            }
            return Ok(DiskSubmission {
                ticket: DiskTicket {
                    key,
                    completion: Arc::clone(completion),
                    shared: Arc::clone(&self.shared),
                },
                sender: self.sender.clone(),
                shared: Arc::clone(&self.shared),
                unsent: None,
                joined: true,
                write_reservation_id: None,
            });
        }
        let completion = Arc::new(DiskCompletion::default());
        in_flight.insert(key.clone(), Arc::clone(&completion));
        drop(in_flight);
        let request = DiskRequest::Operation {
            key: key.clone(),
            task: Box::new(task),
            completion: Arc::clone(&completion),
        };
        Ok(DiskSubmission {
            ticket: DiskTicket {
                key,
                completion,
                shared: Arc::clone(&self.shared),
            },
            sender: self.sender.clone(),
            shared: Arc::clone(&self.shared),
            unsent: Some(request),
            joined: false,
            write_reservation_id,
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
            DiskOperationKey {
                generation,
                id: id.clone(),
                kind: DiskOperationKind::Write,
            },
            DiskTask::Write {
                directory: directory.to_path_buf(),
                id: id.clone(),
                block: block.clone(),
                commit: Some(DiskWriteCommit {
                    state,
                    key: DiskOperationKey {
                        generation,
                        id: id.clone(),
                        kind: DiskOperationKind::Write,
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
            DiskOperationKey {
                generation,
                id: id.clone(),
                kind: DiskOperationKind::Read,
            },
            DiskTask::Read {
                location: location.clone(),
                representation,
            },
        )
    }

    fn retire(&self, ticket: &DiskTicket) {
        retire_disk_completion(&self.shared, &ticket.key, &ticket.completion);
    }
}

fn update_atomic_max(target: &AtomicUsize, value: usize) {
    let mut current = target.load(Ordering::Acquire);
    while value > current {
        match target.compare_exchange_weak(current, value, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

impl Drop for DiskWorker {
    fn drop(&mut self) {
        let _ = self.sender.send(DiskRequest::Stop);
        if let Ok(handle) = self.handle.get_mut() {
            if let Some(handle) = handle.take() {
                let _ = handle.join();
            }
        }
    }
}

#[derive(Debug, Clone)]
struct PendingDiskOperation {
    ticket: DiskTicket,
}

#[derive(Debug, Clone)]
struct CacheBlockRecord {
    id: CacheBlockId,
    physical: CacheBlockPhysicalState,
    bytes: u64,
    shapes: [Vec<i32>; 2],
    dtypes: [String; 2],
    imported: bool,
    leases: usize,
    access_count: u64,
    last_access: u64,
    protected_prefix: bool,
}

impl CacheBlockRecord {
    fn tier(&self) -> CacheTier {
        self.physical.tier()
    }

    fn disk(&self) -> Option<&DiskLocation> {
        self.physical.disk()
    }

    fn pending_disk(&self) -> Option<&PendingDiskOperation> {
        self.physical.pending()
    }

    #[cfg(test)]
    fn device_arrays(&self) -> Option<&CacheBlockArrays> {
        match &self.physical {
            CacheBlockPhysicalState::Device { arrays, .. }
            | CacheBlockPhysicalState::Demoting { arrays, .. } => Some(arrays),
            CacheBlockPhysicalState::Host { .. } | CacheBlockPhysicalState::Disk { .. } => None,
        }
    }

    fn host_block(&self) -> Option<&HostCacheBlock> {
        match &self.physical {
            CacheBlockPhysicalState::Host { block, .. } => Some(block),
            CacheBlockPhysicalState::Device { .. }
            | CacheBlockPhysicalState::Demoting { .. }
            | CacheBlockPhysicalState::Disk { .. } => None,
        }
    }

    fn host_demotion_ticket(&self) -> Option<&HostDemotionTicket> {
        match &self.physical {
            CacheBlockPhysicalState::Demoting { ticket, .. } => Some(ticket),
            CacheBlockPhysicalState::Device { .. }
            | CacheBlockPhysicalState::Host { .. }
            | CacheBlockPhysicalState::Disk { .. } => None,
        }
    }
}

/// Maximum number of individually identified layers in a residency report.
///
/// Additional active layers are folded into
/// [`CacheResidencyReport::per_layer_overflow`], so report size is independent
/// of caller-provided layer identifiers and remains bounded.
pub const CACHE_RESIDENCY_LAYER_REPORT_LIMIT: usize = 128;

/// Current residency and cumulative activity attributable to one layer or a
/// bounded overflow group of layers.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct CacheLayerResidencyStats {
    /// Logical cached tokens. For an overflow aggregate this is the sum of the
    /// per-layer logical token counts rather than a shared sequence length.
    pub logical_cached_tokens: u64,
    /// Sealed key/value blocks.
    pub key_value_blocks: u64,
    /// Sealed compressed-latent/rotary blocks.
    pub compressed_latent_blocks: u64,
    /// Blocks cataloged on the execution device.
    pub device_blocks: u64,
    /// Blocks cataloged in host memory.
    pub host_blocks: u64,
    /// Blocks cataloged on disk.
    pub disk_blocks: u64,
    /// Current logical device bytes, including mutable tails.
    pub current_device_bytes: u64,
    /// Current physical host allocation capacity, including in-flight ownership.
    pub current_host_bytes: u64,
    /// Current logical disk bytes.
    pub current_disk_bytes: u64,
    /// Current bytes in mutable tails.
    pub mutable_tail_bytes: u64,
    /// Blocks whose host buffers are owned by background disk writes.
    pub in_flight_write_blocks: u64,
    /// Physical host allocation capacity owned by background disk writes.
    pub in_flight_write_bytes: u64,
    /// Blocks retaining both device and host allocations during demotion.
    pub in_flight_host_demotion_blocks: u64,
    /// Physical host allocation capacity charged during device demotion.
    pub in_flight_host_demotion_bytes: u64,
    /// Recent device blocks protected from demotion.
    pub protected_recent_blocks: u64,
    /// Prefix or sink blocks protected for attention.
    pub protected_prefix_blocks: u64,
    /// Cumulative host promotions submitted through completion events.
    pub host_promotions: u64,
    /// Cumulative disk promotions submitted after their host read completes.
    pub disk_promotions: u64,
    /// Cumulative device demotions completed through exact-output event waits.
    pub host_demotions: u64,
    /// Cumulative demotions to disk.
    pub disk_demotions: u64,
    /// Cumulative logical bytes transferred between tiers.
    pub transfer_bytes: u64,
    /// Cumulative host time at layer-attributable disk and CPU ownership boundaries.
    pub transfer_wait: Duration,
    /// Cumulative demand accesses already resident on the execution device.
    pub demand_hits: u64,
    /// Cumulative demand accesses that required promotion.
    pub demand_misses: u64,
    /// Cumulative waits that joined or awaited an in-flight disk operation.
    pub in_flight_waits: u64,
    /// Cumulative layer-attributable residency or transfer failures.
    pub failures: u64,
    /// Sealed and mutable blocks scanned by full attention during prefill.
    pub prefill_full_attention_blocks: u64,
    /// Logical bytes scanned by full attention during prefill.
    pub prefill_full_attention_bytes: u64,
    /// Sealed and mutable blocks scanned by full attention during decode.
    pub decode_full_attention_blocks: u64,
    /// Logical bytes scanned by full attention during decode.
    pub decode_full_attention_bytes: u64,
    /// Peak logical scratch bytes used by this layer's attention.
    pub attention_scratch_peak_bytes: u64,
}

impl CacheLayerResidencyStats {
    fn accumulate(&mut self, other: &Self) {
        self.logical_cached_tokens += other.logical_cached_tokens;
        self.key_value_blocks += other.key_value_blocks;
        self.compressed_latent_blocks += other.compressed_latent_blocks;
        self.device_blocks += other.device_blocks;
        self.host_blocks += other.host_blocks;
        self.disk_blocks += other.disk_blocks;
        self.current_device_bytes += other.current_device_bytes;
        self.current_host_bytes += other.current_host_bytes;
        self.current_disk_bytes += other.current_disk_bytes;
        self.mutable_tail_bytes += other.mutable_tail_bytes;
        self.in_flight_write_blocks += other.in_flight_write_blocks;
        self.in_flight_write_bytes += other.in_flight_write_bytes;
        self.in_flight_host_demotion_blocks += other.in_flight_host_demotion_blocks;
        self.in_flight_host_demotion_bytes += other.in_flight_host_demotion_bytes;
        self.protected_recent_blocks += other.protected_recent_blocks;
        self.protected_prefix_blocks += other.protected_prefix_blocks;
        self.host_promotions += other.host_promotions;
        self.disk_promotions += other.disk_promotions;
        self.host_demotions += other.host_demotions;
        self.disk_demotions += other.disk_demotions;
        self.transfer_bytes += other.transfer_bytes;
        self.transfer_wait += other.transfer_wait;
        self.demand_hits += other.demand_hits;
        self.demand_misses += other.demand_misses;
        self.in_flight_waits += other.in_flight_waits;
        self.failures += other.failures;
        self.prefill_full_attention_blocks += other.prefill_full_attention_blocks;
        self.prefill_full_attention_bytes += other.prefill_full_attention_bytes;
        self.decode_full_attention_blocks += other.decode_full_attention_blocks;
        self.decode_full_attention_bytes += other.decode_full_attention_bytes;
        self.attention_scratch_peak_bytes = self
            .attention_scratch_peak_bytes
            .max(other.attention_scratch_peak_bytes);
    }
}

#[derive(Debug, Clone, Default)]
struct CacheLayerActivityCounters {
    stats: CacheLayerResidencyStats,
}

impl CacheLayerActivityCounters {
    fn apply_to(&self, stats: &mut CacheLayerResidencyStats) {
        stats.host_promotions += self.stats.host_promotions;
        stats.disk_promotions += self.stats.disk_promotions;
        stats.host_demotions += self.stats.host_demotions;
        stats.disk_demotions += self.stats.disk_demotions;
        stats.transfer_bytes += self.stats.transfer_bytes;
        stats.transfer_wait += self.stats.transfer_wait;
        stats.demand_hits += self.stats.demand_hits;
        stats.demand_misses += self.stats.demand_misses;
        stats.in_flight_waits += self.stats.in_flight_waits;
        stats.failures += self.stats.failures;
        stats.prefill_full_attention_blocks += self.stats.prefill_full_attention_blocks;
        stats.prefill_full_attention_bytes += self.stats.prefill_full_attention_bytes;
        stats.decode_full_attention_blocks += self.stats.decode_full_attention_blocks;
        stats.decode_full_attention_bytes += self.stats.decode_full_attention_bytes;
        stats.attention_scratch_peak_bytes = stats
            .attention_scratch_peak_bytes
            .max(self.stats.attention_scratch_peak_bytes);
    }
}

/// Bounded, individually identified per-layer residency observations.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct CacheLayerResidencyReport {
    /// Global model layer identifier.
    pub global_layer: usize,
    /// Current residency and cumulative activity for this layer.
    pub stats: CacheLayerResidencyStats,
}

/// Aggregated logical device/disk and physical host residency observations.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct CacheResidencyReport {
    /// Absolute token count represented by the longest layer.
    pub logical_cached_tokens: u64,
    /// Sealed key/value blocks.
    pub key_value_blocks: u64,
    /// Sealed compressed-latent/rotary blocks.
    pub compressed_latent_blocks: u64,
    /// Blocks cataloged on the execution device.
    pub device_blocks: u64,
    /// Blocks cataloged in host memory.
    pub host_blocks: u64,
    /// Blocks cataloged on disk.
    pub disk_blocks: u64,
    /// Current logical device bytes, including mutable tails.
    pub current_device_bytes: u64,
    /// Peak successfully admitted logical device bytes.
    pub peak_device_bytes: u64,
    /// Current physical host allocation capacity, including in-flight ownership.
    pub current_host_bytes: u64,
    /// Peak successfully admitted physical host allocation capacity.
    pub peak_host_bytes: u64,
    /// Current logical disk bytes.
    pub current_disk_bytes: u64,
    /// Peak successfully admitted logical disk bytes.
    pub peak_disk_bytes: u64,
    /// Blocks whose host buffers are owned by background disk writes.
    pub in_flight_write_blocks: u64,
    /// Physical host capacity owned by disk writes (included in current host bytes).
    pub in_flight_write_bytes: u64,
    /// Peak physical host capacity owned by background disk writes.
    pub peak_in_flight_write_bytes: u64,
    /// Blocks retaining both device and host allocations during demotion.
    pub in_flight_host_demotion_blocks: u64,
    /// Physical host capacity charged during device demotion.
    pub in_flight_host_demotion_bytes: u64,
    /// Peak physical host capacity charged during device demotion.
    pub peak_in_flight_host_demotion_bytes: u64,
    /// Current bytes in mutable tails.
    pub mutable_tail_bytes: u64,
    /// Recent blocks protected from device demotion.
    pub protected_recent_blocks: u64,
    /// Prefix or sink blocks protected for attention.
    pub protected_prefix_blocks: u64,
    /// Current residency and cumulative per-layer activity, sorted by global
    /// layer and capped by [`CACHE_RESIDENCY_LAYER_REPORT_LIMIT`].
    pub per_layer: Vec<CacheLayerResidencyReport>,
    /// Number of active layers folded into `per_layer_overflow`.
    pub per_layer_overflow_layers: u64,
    /// Exact current aggregate of active omitted layers plus cumulative
    /// activity for every layer without an identified row.
    pub per_layer_overflow: CacheLayerResidencyStats,
    /// Host-to-device promotions submitted through completion events.
    pub host_promotions: u64,
    /// Disk-to-device promotions submitted after their host read completes.
    pub disk_promotions: u64,
    /// Device-to-host demotions completed through exact-output event waits.
    pub host_demotions: u64,
    /// Completed resident-to-disk demotions using existing or newly written backing.
    pub disk_demotions: u64,
    /// Logical bytes copied by promotion and demotion operations.
    pub transfer_bytes: u64,
    /// Host time spent on disk operations and CPU ownership boundaries.
    pub transfer_wait: Duration,
    /// Blocks evicted because all configured tiers were exhausted.
    pub evictions: u64,
    /// Sliding-window blocks discarded as semantically invisible.
    pub discarded_sliding_blocks: u64,
    /// Completed block seals.
    pub block_seals: u64,
    /// Mutable tail allocations.
    pub tail_allocations: u64,
    /// Requests served by an already device-cataloged block.
    pub demand_hits: u64,
    /// Requests requiring host or disk promotion.
    pub demand_misses: u64,
    /// Requests that joined an existing transfer.
    pub in_flight_waits: u64,
    /// Effective bounded disk request queue capacity.
    pub queue_capacity: usize,
    /// Peak observed queue occupancy.
    pub queue_peak_occupancy: usize,
    /// Requests delayed by queue capacity.
    pub queue_backpressure: u64,
    /// Requests canceled by reset or truncation.
    pub cancellations: u64,
    /// Cache transfer or persistence failures.
    pub failures: u64,
    /// Blocks scanned by full attention during prefill.
    pub prefill_full_attention_blocks: u64,
    /// Logical bytes scanned by full attention during prefill.
    pub prefill_full_attention_bytes: u64,
    /// Blocks scanned by full attention during decode.
    pub decode_full_attention_blocks: u64,
    /// Logical bytes scanned by full attention during decode.
    pub decode_full_attention_bytes: u64,
    /// Peak logical scratch bytes used by attention.
    pub attention_scratch_peak_bytes: u64,
    /// Successful prompt-cache saves.
    pub prompt_cache_saves: u64,
    /// Successful prompt-cache loads.
    pub prompt_cache_loads: u64,
    /// Logical bytes written or cataloged for prompt caches.
    pub prompt_cache_bytes: u64,
    /// Imported persistent shard count.
    pub imported_mapped_shards: u64,
    /// Optional peak process resident-set size sampled from the operating system.
    pub process_rss_bytes: Option<u64>,
    /// Optional cumulative minor page faults.
    pub process_minor_page_faults: Option<u64>,
    /// Optional cumulative major page faults.
    pub process_major_page_faults: Option<u64>,
}

#[derive(Debug, Default)]
struct CacheCounters {
    report: CacheResidencyReport,
}

#[derive(Debug)]
struct CacheManagerState {
    pool: CacheResidencyPool,
    pool_manager_id: u64,
    generation: u64,
    background_disk_error: Option<String>,
    access_clock: u64,
    blocks: BTreeMap<CacheBlockId, CacheBlockRecord>,
    tails: HashMap<usize, (u64, i64)>,
    host_write_reservations: HashMap<DiskOperationKey, HostWriteReservation>,
    retiring_host_demotions: HashMap<u64, RetiringHostDemotion>,
    retiring_disk_reads: HashMap<DiskOperationKey, (usize, u64)>,
    transfer_device: Option<CacheTransferDevice>,
    layer_activity: BTreeMap<usize, CacheLayerActivityCounters>,
    layer_activity_overflow: CacheLayerActivityCounters,
    counters: CacheCounters,
    recent_device_blocks: usize,
    device_budget_bytes: u64,
    host_budget_bytes: u64,
    disk_budget_bytes: Option<u64>,
}

impl CacheManagerState {
    fn layer_activity_mut(&mut self, global_layer: usize) -> &mut CacheLayerResidencyStats {
        if self.layer_activity.contains_key(&global_layer)
            || self.layer_activity.len() < CACHE_RESIDENCY_LAYER_REPORT_LIMIT
        {
            &mut self.layer_activity.entry(global_layer).or_default().stats
        } else {
            &mut self.layer_activity_overflow.stats
        }
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
        if let LiveCacheDiskPolicy::Enabled { directory, .. } = &options.live_disk {
            fs::create_dir_all(directory).map_err(|source| CacheResidencyError::Io {
                action: "create live cache directory",
                path: directory.clone(),
                source,
            })?;
        }
        let queue_capacity = match &options.live_disk {
            LiveCacheDiskPolicy::Disabled => 0,
            LiveCacheDiskPolicy::Enabled { queue_capacity, .. } => *queue_capacity,
        };
        let effective_queue_capacity = queue_capacity.max(1);
        let mut counters = CacheCounters::default();
        counters.report.queue_capacity = effective_queue_capacity;
        let disk_worker = Some(Arc::new(DiskWorker::new(effective_queue_capacity)?));
        let host_demotion_worker = Arc::new(HostDemotionWorker::new()?);
        let recent_device_blocks = options.recent_device_blocks;
        let device_budget_bytes = options.device_budget_bytes;
        let host_budget_bytes = options.host_budget_bytes;
        let disk_budget_bytes = match &options.live_disk {
            LiveCacheDiskPolicy::Disabled => None,
            LiveCacheDiskPolicy::Enabled { budget_bytes, .. } => Some(*budget_bytes),
        };
        let session_id = NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed);
        let pool = match options.pool.clone() {
            Some(pool) => pool.as_ref().clone(),
            None => CacheResidencyPool::for_paged_options(&options)?,
        };
        pool.update_manager(session_id, CachePoolUsage::default())?;
        let pool_membership = Arc::new(CachePoolMembership {
            manager: session_id,
            pool: pool.clone(),
        });
        Ok(Self {
            session_id,
            inner: Arc::new(CacheResidencyManagerInner {
                options,
                state: Arc::new(Mutex::new(CacheManagerState {
                    pool,
                    pool_manager_id: session_id,
                    generation: 0,
                    background_disk_error: None,
                    access_clock: 0,
                    blocks: BTreeMap::new(),
                    tails: HashMap::new(),
                    host_write_reservations: HashMap::new(),
                    retiring_host_demotions: HashMap::new(),
                    retiring_disk_reads: HashMap::new(),
                    transfer_device: None,
                    layer_activity: BTreeMap::new(),
                    layer_activity_overflow: CacheLayerActivityCounters::default(),
                    counters,
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
        &self.inner.pool_membership.pool
    }

    fn lock(&self) -> Result<MutexGuard<'_, CacheManagerState>, CacheResidencyError> {
        self.inner
            .state
            .lock()
            .map_err(|_| CacheResidencyError::ManagerPoisoned)
    }

    pub(crate) fn bind_transfer_device(&self, stream: &Stream) -> Result<(), CacheResidencyError> {
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

    pub(crate) fn set_tail_state(
        &self,
        layer: usize,
        bytes: u64,
        end: i64,
    ) -> Result<(), CacheResidencyError> {
        let mut state = self.lock()?;
        let previous = state.tails.insert(layer, (bytes, end));
        let allocated = previous.is_none_or(|tail| tail.0 == 0) && bytes > 0;
        if allocated {
            state.counters.report.tail_allocations += 1;
        }
        drop(state);
        if let Err(error) = self.rebalance(None, false) {
            let mut state = self.lock()?;
            match previous {
                Some(previous) => {
                    state.tails.insert(layer, previous);
                }
                None => {
                    state.tails.remove(&layer);
                }
            }
            if allocated {
                state.counters.report.tail_allocations =
                    state.counters.report.tail_allocations.saturating_sub(1);
            }
            update_report_totals(&mut state);
            return Err(error);
        }
        Ok(())
    }

    pub(crate) fn seal_block(
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
            return Err(CacheResidencyError::DuplicateBlock(id));
        }
        state.access_clock += 1;
        let bytes = arrays.bytes();
        let record = CacheBlockRecord {
            id: id.clone(),
            shapes: arrays.shapes(),
            dtypes: arrays.dtypes(),
            physical: CacheBlockPhysicalState::Device { arrays, disk: None },
            bytes,
            imported: false,
            leases: 0,
            access_count: 0,
            last_access: state.access_clock,
            protected_prefix,
        };
        state.blocks.insert(id.clone(), record);
        drop(state);
        if let Err(error) = self.rebalance(Some(&id), false) {
            let mut state = self.lock()?;
            if let Some(record) = state.blocks.remove(&id) {
                cancel_record_operation(&record, &mut state.counters.report);
                remove_ephemeral_file(&record);
            }
            update_report_totals(&mut state);
            return Err(error);
        }
        let mut state = self.lock()?;
        state.counters.report.block_seals += 1;
        Ok(id)
    }

    pub(crate) fn layer_block_ids(
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

    pub(crate) fn layer_end(
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

    pub(crate) fn remove_block(&self, id: &CacheBlockId) -> Result<(), CacheResidencyError> {
        let mut state = self.lock()?;
        let record = state
            .blocks
            .get(id)
            .ok_or_else(|| CacheResidencyError::MissingBlock(id.clone()))?;
        if record.leases != 0 {
            return Err(CacheResidencyError::BlockLeased(id.clone()));
        }
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

    pub(crate) fn truncate_layer_transaction(
        &self,
        global_layer: usize,
        representation: CacheRepresentation,
        end: i64,
        replacement: Option<(CacheBlockId, CacheBlockArrays)>,
        protected_prefix_tokens: i64,
    ) -> Result<(), CacheResidencyError> {
        if end < 0 {
            return Err(CacheResidencyError::InvalidTokenRange { start: 0, end });
        }
        if let Some((old_id, arrays)) = &replacement {
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
            (Some(crossing), Some((old_id, _))) if crossing == old_id => {}
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
                .ok_or_else(|| CacheResidencyError::MissingBlock(id.clone()))?;
            let owned_replacement_lease =
                replacement.as_ref().is_some_and(|(old_id, _)| old_id == id);
            let expected_leases = usize::from(owned_replacement_lease);
            if record.leases != expected_leases {
                return Err(CacheResidencyError::BlockLeased(id.clone()));
            }
            if owned_replacement_lease && record.tier() != CacheTier::Device {
                return Err(CacheResidencyError::Runtime(
                    "truncated cache replacement lease is not device resident".into(),
                ));
            }
        }

        let replacement_id = replacement.as_ref().map(|(old_id, _)| CacheBlockId {
            session_id: self.session_id,
            global_layer,
            representation,
            start: old_id.start,
            end,
            rank: old_id.rank,
        });
        if let Some(id) = &replacement_id {
            if state.blocks.contains_key(id) && !affected.contains(id) {
                return Err(CacheResidencyError::DuplicateBlock(id.clone()));
            }
        }

        let tickets = advance_generation_locked(&mut state);
        let mut removed = Vec::with_capacity(affected.len());
        for id in &affected {
            if let Some(record) = state.blocks.remove(id) {
                removed.push(record);
            }
        }
        state.tails.insert(global_layer, (0, end));
        if let Some((old_id, arrays)) = replacement {
            state.access_clock += 1;
            let id = replacement_id.expect("validated replacement id is available");
            debug_assert_eq!(id.start, old_id.start);
            let record = CacheBlockRecord {
                id: id.clone(),
                shapes: arrays.shapes(),
                dtypes: arrays.dtypes(),
                bytes: arrays.bytes(),
                physical: CacheBlockPhysicalState::Device { arrays, disk: None },
                imported: false,
                leases: 0,
                access_count: 0,
                last_access: state.access_clock,
                protected_prefix: end <= protected_prefix_tokens,
            };
            let previous = state.blocks.insert(id, record);
            debug_assert!(previous.is_none());
            state.counters.report.block_seals += 1;
        }
        update_report_totals(&mut state);
        drop(state);

        for record in &removed {
            remove_ephemeral_file(record);
        }
        self.retire_tickets(&tickets)?;
        Ok(())
    }

    pub(crate) fn lease_block(
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
    pub(crate) fn prefetch_blocks(
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
                .ok_or_else(|| CacheResidencyError::MissingBlock(id.clone()))?
                .physical
                .clone();

            match physical {
                CacheBlockPhysicalState::Demoting { ticket, .. } => {
                    drop(state);
                    self.finish_device_demotion(&ticket)?;
                }
                CacheBlockPhysicalState::Disk { location, read } => {
                    let worker = self.inner.disk_worker.as_ref().ok_or_else(|| {
                        CacheResidencyError::Runtime("cache disk worker is unavailable".into())
                    })?;
                    let (ticket, submission, joined, _transfer_reservation, reserved_host_bytes) =
                        match read {
                            DiskCacheReadState::Reading {
                                pending,
                                reserved_host_bytes,
                            } => (pending.ticket, None, true, None, reserved_host_bytes),
                            DiskCacheReadState::Ready => {
                                let record = state
                                    .blocks
                                    .get(id)
                                    .ok_or_else(|| CacheResidencyError::MissingBlock(id.clone()))?;
                                let bytes = record.bytes;
                                let reserved_host_bytes = host_cache_layout_capacity_upper_bound(
                                    &record.shapes,
                                    &record.dtypes,
                                )?;
                                let required_host_bytes = state
                                    .counters
                                    .report
                                    .current_host_bytes
                                    .saturating_add(reserved_host_bytes);
                                if required_host_bytes > self.options().host_budget_bytes {
                                    let candidate = eviction_candidate(
                                        &state,
                                        CacheTier::Host,
                                        Some(id),
                                        0,
                                        self.options().eviction_policy,
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
                                        budget: self.options().host_budget_bytes,
                                    });
                                }
                                let host_admission =
                                    state.pool.reserve_additional(CachePoolUsage {
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
                                    .ok_or_else(|| CacheResidencyError::MissingBlock(id.clone()))?;
                                record.physical = CacheBlockPhysicalState::Disk {
                                    location: location.clone(),
                                    read: DiskCacheReadState::Reading {
                                        pending: PendingDiskOperation {
                                            ticket: ticket.clone(),
                                        },
                                        reserved_host_bytes,
                                    },
                                };
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
                        };
                    drop(state);

                    let outcome = match submission {
                        Some(submission) => Some(submission.enqueue()?),
                        None => None,
                    };
                    let result = ticket.wait();
                    let mut state = self.lock()?;
                    if joined || outcome.as_ref().is_some_and(|outcome| outcome.joined) {
                        state.counters.report.in_flight_waits += 1;
                        state.layer_activity_mut(id.global_layer).in_flight_waits += 1;
                    }
                    if let Some(outcome) = &outcome {
                        state.counters.report.queue_peak_occupancy = state
                            .counters
                            .report
                            .queue_peak_occupancy
                            .max(outcome.peak_occupancy);
                        state.counters.report.queue_backpressure += u64::from(outcome.backpressure);
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
                                .ok_or_else(|| CacheResidencyError::MissingBlock(id.clone()))?;
                            if shapes != record.shapes
                                || dtypes != record.dtypes
                                || bytes != record.bytes
                            {
                                record.physical.clear_pending(&ticket.key);
                                drop(state);
                                worker.retire(&ticket);
                                return Err(CacheResidencyError::MalformedShard {
                                    path: location.path,
                                    reason: "array shape or dtype does not match the manifest"
                                        .into(),
                                });
                            }
                            if capacity > reserved_host_bytes {
                                record.physical.clear_pending(&ticket.key);
                                drop(state);
                                worker.retire(&ticket);
                                return Err(CacheResidencyError::Runtime(format!(
                                    "host cache allocation capacity {capacity} exceeded its pre-admitted bound {reserved_host_bytes}"
                                )));
                            }
                            if record.physical.pending_matches(&ticket.key) {
                                record.physical = CacheBlockPhysicalState::Host {
                                    block,
                                    persistence: HostCachePersistence::Backed(location),
                                };
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
                                record.physical.clear_pending(&ticket.key);
                            }
                            state.counters.report.failures += 1;
                            state.layer_activity_mut(id.global_layer).failures += 1;
                            drop(state);
                            worker.retire(&ticket);
                            return Err(error);
                        }
                    }
                    drop(state);
                    worker.retire(&ticket);
                }
                CacheBlockPhysicalState::Host { block, persistence } => {
                    let disk = match persistence {
                        HostCachePersistence::Unbacked => None,
                        HostCachePersistence::Backed(location) => Some(location),
                        HostCachePersistence::Writing(pending) => {
                            drop(state);
                            self.wait_for_host_release(&pending.ticket)?;
                            continue;
                        }
                    };
                    let transfer_reservation = state.pool.reserve_transfer(block.capacity()?)?;
                    let (device_arrays, completions) = block.copy_to_device(transfer_stream)?;
                    if state.generation != generation {
                        return Err(CacheResidencyError::DiskOperationCancelled { generation });
                    }
                    state.access_clock += 1;
                    let access_clock = state.access_clock;
                    let bytes = {
                        let record = state
                            .blocks
                            .get_mut(id)
                            .ok_or_else(|| CacheResidencyError::MissingBlock(id.clone()))?;
                        record.physical = CacheBlockPhysicalState::Device {
                            arrays: device_arrays.clone(),
                            disk: disk.clone(),
                        };
                        record.leases += 1;
                        record.access_count += 1;
                        record.last_access = access_clock;
                        record.bytes
                    };
                    state.counters.report.demand_misses += 1;
                    if loaded_from_disk {
                        state.counters.report.disk_promotions += 1;
                    } else {
                        state.counters.report.host_promotions += 1;
                    }
                    state.counters.report.transfer_bytes += bytes;
                    let transfer_wait = started.elapsed();
                    state.counters.report.transfer_wait += transfer_wait;
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
                            if let Some(record) = state.blocks.get_mut(id) {
                                record.leases = record.leases.saturating_sub(1);
                                record.physical = CacheBlockPhysicalState::Host {
                                    block,
                                    persistence: match disk {
                                        Some(location) => HostCachePersistence::Backed(location),
                                        None => HostCachePersistence::Unbacked,
                                    },
                                };
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
                CacheBlockPhysicalState::Device { arrays, .. } => {
                    state.access_clock += 1;
                    let access_clock = state.access_clock;
                    let record = state
                        .blocks
                        .get_mut(id)
                        .ok_or_else(|| CacheResidencyError::MissingBlock(id.clone()))?;
                    record.leases += 1;
                    record.access_count += 1;
                    record.last_access = access_clock;
                    drop(state);
                    let completion = match async_eval_with_event(arrays.arrays()) {
                        Ok(completion) => completion,
                        Err(source) => {
                            if let Ok(mut state) = self.lock() {
                                if let Some(record) = state.blocks.get_mut(id) {
                                    record.leases = record.leases.saturating_sub(1);
                                }
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
                        if let Some(record) = state.blocks.get_mut(id) {
                            record.leases = record.leases.saturating_sub(1);
                        }
                        return Err(CacheResidencyError::DiskOperationCancelled { generation });
                    }
                    state.counters.report.demand_hits += 1;
                    let transfer_wait = started.elapsed();
                    state.counters.report.transfer_wait += transfer_wait;
                    let activity = state.layer_activity_mut(id.global_layer);
                    activity.demand_hits += 1;
                    activity.transfer_wait += transfer_wait;
                    drop(state);
                    if let Err(error) = self.rebalance(Some(id), true) {
                        if let Ok(mut state) = self.lock() {
                            if let Some(record) = state.blocks.get_mut(id) {
                                record.leases = record.leases.saturating_sub(1);
                            }
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

    pub(crate) fn discard_before(
        &self,
        layer: usize,
        representation: CacheRepresentation,
        visible_start: i64,
        prefix_tokens: i64,
    ) -> Result<(), CacheResidencyError> {
        if self.options().retain_discarded_for_persistence {
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
                    && record.leases == 0
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
                remove_ephemeral_file(&record);
                state.counters.report.discarded_sliding_blocks += 1;
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
        if let Some(id) = state
            .blocks
            .values()
            .find(|record| record.leases != 0)
            .map(|record| record.id.clone())
        {
            return Err(CacheResidencyError::BlockLeased(id));
        }
        let tickets = advance_generation_locked(&mut state);
        for record in state.blocks.values() {
            remove_ephemeral_file(record);
        }
        state.blocks.clear();
        state.tails.clear();
        update_report_totals(&mut state);
        drop(state);
        self.retire_tickets(&tickets)?;
        Ok(())
    }

    /// Returns a bounded aggregate snapshot without retaining per-block history.
    pub fn report(&self) -> Result<CacheResidencyReport, CacheResidencyError> {
        let mut state = self.lock()?;
        update_report_totals(&mut state);
        if self.options().sample_process {
            sample_process(&mut state.counters.report);
        }
        Ok(state.counters.report.clone())
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

    pub(crate) fn record_attention_scan(
        &self,
        global_layer: usize,
        prefill: bool,
        blocks: u64,
        bytes: u64,
        scratch_bytes: u64,
    ) -> Result<(), CacheResidencyError> {
        let mut state = self.lock()?;
        if prefill {
            state.counters.report.prefill_full_attention_blocks += blocks;
            state.counters.report.prefill_full_attention_bytes += bytes;
            let activity = state.layer_activity_mut(global_layer);
            activity.prefill_full_attention_blocks += blocks;
            activity.prefill_full_attention_bytes += bytes;
        } else {
            state.counters.report.decode_full_attention_blocks += blocks;
            state.counters.report.decode_full_attention_bytes += bytes;
            let activity = state.layer_activity_mut(global_layer);
            activity.decode_full_attention_blocks += blocks;
            activity.decode_full_attention_bytes += bytes;
        }
        state.counters.report.attention_scratch_peak_bytes = state
            .counters
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
            if let Some(record) = state.blocks.get_mut(id) {
                record.leases = record.leases.saturating_sub(1);
            }
            background_failed = state.background_disk_error.is_some();
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
        if record.tier() != CacheTier::Host || record.leases != 0 || record.pending_disk().is_some()
        {
            return Ok(HostDemotionProgress::Retry);
        }

        // Persistent prompt-cache blocks and completed live-cache writes can be
        // released immediately; they do not require live writeback to be enabled.
        if let Some(location) = record.disk().cloned() {
            let record = state.blocks.get_mut(id).expect("host block exists");
            record.physical = CacheBlockPhysicalState::Disk {
                location,
                read: DiskCacheReadState::Ready,
            };
            state.counters.report.disk_demotions += 1;
            state.layer_activity_mut(id.global_layer).disk_demotions += 1;
            update_report_totals(&mut state);
            return Ok(HostDemotionProgress::Freed);
        }

        let (directory, budget_bytes) = match &self.options().live_disk {
            LiveCacheDiskPolicy::Disabled => {
                state.counters.report.failures += 1;
                state.layer_activity_mut(id.global_layer).failures += 1;
                return Err(CacheResidencyError::LiveDiskRequired {
                    required: state.counters.report.current_host_bytes,
                    budget: self.options().host_budget_bytes,
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
            state.counters.report.failures += 1;
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
        let pool_admission = state.pool.reserve_additional(CachePoolUsage {
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
        let CacheBlockPhysicalState::Host { persistence, .. } = &mut record.physical else {
            unreachable!("host demotion candidate changed while cache state was locked");
        };
        debug_assert!(matches!(persistence, HostCachePersistence::Unbacked));
        *persistence = HostCachePersistence::Writing(PendingDiskOperation {
            ticket: ticket.clone(),
        });
        update_report_totals(&mut state);
        drop(pool_admission);
        drop(state);

        let enqueue_started = Instant::now();
        let outcome = match submission.enqueue() {
            Ok(outcome) => outcome,
            Err(error) => {
                let mut state = self.lock()?;
                if let Some(record) = state.blocks.get_mut(&ticket.key.id) {
                    record.physical.clear_pending(&ticket.key);
                }
                state.counters.report.failures += 1;
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
            state.counters.report.in_flight_waits += 1;
            state
                .layer_activity_mut(ticket.key.id.global_layer)
                .in_flight_waits += 1;
        }
        state.counters.report.queue_peak_occupancy = state
            .counters
            .report
            .queue_peak_occupancy
            .max(outcome.peak_occupancy);
        state.counters.report.queue_backpressure += u64::from(outcome.backpressure);
        if outcome.backpressure {
            state.counters.report.transfer_wait += enqueue_wait;
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
        state.counters.report.in_flight_waits += 1;
        state.counters.report.transfer_wait += elapsed;
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
            .ok_or_else(|| CacheResidencyError::MissingBlock(id.clone()))?;
        if record.leases != 0 {
            return Err(CacheResidencyError::BlockLeased(id.clone()));
        }
        let arrays = match &record.physical {
            CacheBlockPhysicalState::Device { arrays, disk: None } => arrays.clone(),
            CacheBlockPhysicalState::Device { disk: Some(_), .. }
            | CacheBlockPhysicalState::Demoting { .. }
            | CacheBlockPhysicalState::Host { .. }
            | CacheBlockPhysicalState::Disk { .. } => {
                return Err(CacheResidencyError::MissingResidentArrays(id.clone()));
            }
        };
        let reserved_host_bytes = host_cache_capacity_upper_bound(&arrays)?;
        let required_host_bytes = state
            .counters
            .report
            .current_host_bytes
            .saturating_add(reserved_host_bytes);
        if required_host_bytes > self.options().host_budget_bytes {
            return Err(CacheResidencyError::BudgetExceeded {
                tier: CacheTier::Host,
                required: required_host_bytes,
                budget: self.options().host_budget_bytes,
            });
        }
        let pool_admission = state.pool.reserve_additional(CachePoolUsage {
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
        record.physical = CacheBlockPhysicalState::Demoting {
            arrays,
            ticket: ticket.clone(),
        };
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
        state.counters.report.in_flight_waits += 1;
        state.counters.report.transfer_wait += elapsed;
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
                    let CacheBlockPhysicalState::Demoting { arrays, .. } = &record.physical else {
                        unreachable!("matching host demotion changed while cache state was locked");
                    };
                    record.physical = CacheBlockPhysicalState::Device {
                        arrays: arrays.clone(),
                        disk: None,
                    };
                    state.counters.report.failures += 1;
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
                record.physical = CacheBlockPhysicalState::Host {
                    block,
                    persistence: HostCachePersistence::Unbacked,
                };
                state.counters.report.host_demotions += 1;
                state.counters.report.transfer_bytes += bytes;
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
                let CacheBlockPhysicalState::Demoting { arrays, .. } = &record.physical else {
                    unreachable!("matching host demotion changed while cache state was locked");
                };
                record.physical = CacheBlockPhysicalState::Device {
                    arrays: arrays.clone(),
                    disk: None,
                };
                state.counters.report.failures += 1;
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
                state.counters.report.current_device_bytes > self.options().device_budget_bytes;
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
                    self.options().recent_device_blocks,
                    self.options().eviction_policy,
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
                            self.options().eviction_policy,
                        )
                    } else {
                        None
                    }
                });
                let Some(id) = candidate else {
                    state.counters.report.failures += 1;
                    if let Some(required) = required {
                        state.layer_activity_mut(required.global_layer).failures += 1;
                    } else {
                        state.layer_activity_overflow.stats.failures += 1;
                    }
                    return Err(if pool_device_over && !local_device_over {
                        CacheResidencyError::PoolBudgetExceeded {
                            resource: CachePoolResource::Device,
                            required: pool_report.current_device_bytes,
                            budget: pool_report.limits.device_bytes(),
                        }
                    } else {
                        CacheResidencyError::BudgetExceeded {
                            tier: CacheTier::Device,
                            required: state.counters.report.current_device_bytes,
                            budget: self.options().device_budget_bytes,
                        }
                    });
                };
                if let Some(location) = state
                    .blocks
                    .get(&id)
                    .and_then(CacheBlockRecord::disk)
                    .cloned()
                {
                    let record = state.blocks.get_mut(&id).expect("candidate exists");
                    record.physical = CacheBlockPhysicalState::Disk {
                        location,
                        read: DiskCacheReadState::Ready,
                    };
                    state.counters.report.disk_demotions += 1;
                    state.layer_activity_mut(id.global_layer).disk_demotions += 1;
                    continue;
                }
                let candidate_host_bytes =
                    match &state.blocks.get(&id).expect("candidate exists").physical {
                        CacheBlockPhysicalState::Device { arrays, .. } => {
                            host_cache_capacity_upper_bound(arrays)?
                        }
                        _ => return Err(CacheResidencyError::MissingResidentArrays(id.clone())),
                    };
                let required_host_bytes = state
                    .counters
                    .report
                    .current_host_bytes
                    .saturating_add(candidate_host_bytes);
                if required_host_bytes > self.options().host_budget_bytes {
                    if candidate_host_bytes > self.options().host_budget_bytes {
                        state.counters.report.failures += 1;
                        state.layer_activity_mut(id.global_layer).failures += 1;
                        return Err(CacheResidencyError::BudgetExceeded {
                            tier: CacheTier::Host,
                            required: candidate_host_bytes,
                            budget: self.options().host_budget_bytes,
                        });
                    }
                    let host_candidate = eviction_candidate(
                        &state,
                        CacheTier::Host,
                        required,
                        0,
                        self.options().eviction_policy,
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
                                        pending.ticket.key.kind == DiskOperationKind::Write
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
                    state.counters.report.failures += 1;
                    state.layer_activity_mut(id.global_layer).failures += 1;
                    return Err(match &self.options().live_disk {
                        LiveCacheDiskPolicy::Disabled => CacheResidencyError::LiveDiskRequired {
                            required: required_host_bytes,
                            budget: self.options().host_budget_bytes,
                        },
                        LiveCacheDiskPolicy::Enabled { .. } => {
                            CacheResidencyError::BudgetExceeded {
                                tier: CacheTier::Host,
                                required: required_host_bytes,
                                budget: self.options().host_budget_bytes,
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
                state.counters.report.current_host_bytes > self.options().host_budget_bytes;
            if local_host_over || pool_host_over {
                let candidate = eviction_candidate(
                    &state,
                    CacheTier::Host,
                    required,
                    0,
                    self.options().eviction_policy,
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
                                    pending.ticket.key.kind == DiskOperationKind::Write
                                })
                                .map(|pending| pending.ticket.clone())
                        })
                    });
                let required_host_bytes = state.counters.report.current_host_bytes;
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
                state.counters.report.failures += 1;
                if let Some(required) = required {
                    state.layer_activity_mut(required.global_layer).failures += 1;
                } else {
                    state.layer_activity_overflow.stats.failures += 1;
                }
                return Err(if pool_host_over && !local_host_over {
                    CacheResidencyError::PoolBudgetExceeded {
                        resource: CachePoolResource::Host,
                        required: pool_report.current_host_bytes,
                        budget: pool_report.limits.host_bytes(),
                    }
                } else {
                    CacheResidencyError::BudgetExceeded {
                        tier: CacheTier::Host,
                        required: required_host_bytes,
                        budget: self.options().host_budget_bytes,
                    }
                });
            }

            // Start one background write as soon as the finite host tier fills.
            // It remains charged to host memory until the worker commits and
            // releases its buffers; a later demotion waits only if it needs space.
            let proactive = matches!(
                &self.options().live_disk,
                LiveCacheDiskPolicy::Enabled { .. }
            ) && state.counters.report.current_host_bytes != 0
                && (state.counters.report.current_host_bytes >= self.options().host_budget_bytes
                    || pool_report.current_host_bytes >= pool_report.limits.host_bytes());
            let candidate = if proactive {
                eviction_candidate(
                    &state,
                    CacheTier::Host,
                    required,
                    0,
                    self.options().eviction_policy,
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
        if descriptor.layer_count == 0
            || descriptor.global_layer_start >= descriptor.global_layer_end
            || descriptor.global_layer_end > descriptor.layer_count
            || descriptor.layer_layout.len()
                != descriptor.global_layer_end - descriptor.global_layer_start
            || descriptor.layer_prefix_offsets.len()
                != descriptor.global_layer_end - descriptor.global_layer_start
            || descriptor.batch_size == 0
            || descriptor.batch_size > i32::MAX as usize
        {
            return Err(CacheResidencyError::MalformedManifest(
                "invalid prompt-cache descriptor dimensions".into(),
            ));
        }
        for policy in descriptor.layer_layout.iter() {
            validate_layer_cache_policy(policy)?;
        }
        let parent = destination.parent().ok_or_else(|| {
            CacheResidencyError::InvalidPromptCachePath(destination.to_path_buf())
        })?;
        fs::create_dir_all(parent).map_err(|source| CacheResidencyError::Io {
            action: "create prompt cache parent",
            path: parent.to_path_buf(),
            source,
        })?;
        let replacing = destination.exists();
        if replacing && !options.replace_existing {
            return Err(CacheResidencyError::PromptCacheExists(
                destination.to_path_buf(),
            ));
        }
        if replacing && !destination.is_dir() {
            return Err(CacheResidencyError::InvalidPromptCachePath(
                destination.to_path_buf(),
            ));
        }
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let file_name = destination
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| CacheResidencyError::InvalidPromptCachePath(destination.into()))?;
        let generation_name = format!("generation-{nonce}");
        let generations = destination.join(PROMPT_CACHE_GENERATIONS_DIRECTORY);
        if replacing {
            fs::create_dir_all(&generations).map_err(|source| CacheResidencyError::Io {
                action: "create prompt cache generation directory",
                path: generations.clone(),
                source,
            })?;
        }
        let temporary = if replacing {
            generations.join(format!(".tmp-{nonce}"))
        } else {
            parent.join(format!(".{file_name}.tmp-{nonce}"))
        };
        fs::create_dir(&temporary).map_err(|source| CacheResidencyError::Io {
            action: "create temporary prompt cache",
            path: temporary.clone(),
            source,
        })?;

        let result = (|| {
            let records = {
                let state = self.lock()?;
                if let Some(id) = state
                    .blocks
                    .values()
                    .find(|record| record.leases != 0)
                    .map(|record| record.id.clone())
                {
                    return Err(CacheResidencyError::BlockLeased(id));
                }
                state.blocks.values().cloned().collect::<Vec<_>>()
            };
            validate_complete_prefix(&records, &descriptor, prefix_token_ids.len())?;
            let mut manifest_blocks = Vec::with_capacity(records.len());
            let mut manifest_state = Vec::with_capacity(state_arrays.len());
            let mut logical_bytes = 0u64;
            for (index, record) in records.iter().enumerate() {
                let shard = format!("block-{index:08}.safetensors");
                let shard_path = temporary.join(&shard);
                match &record.physical {
                    CacheBlockPhysicalState::Device { arrays, .. }
                    | CacheBlockPhysicalState::Demoting { arrays, .. } => {
                        save_block_arrays(&shard_path, arrays)?;
                    }
                    CacheBlockPhysicalState::Host { block, .. } => {
                        save_host_cache_block(&shard_path, block)?;
                    }
                    CacheBlockPhysicalState::Disk { location, .. } => {
                        let block =
                            load_host_cache_block_direct(location, record.id.representation)?;
                        save_host_cache_block(&shard_path, &block)?;
                    }
                }
                sync_file(&shard_path)?;
                let payload_sha256 = hash_shard_payload(&shard_path)?;
                logical_bytes += record.bytes;
                let names = array_names(record.id.representation);
                manifest_blocks.push(PromptCacheBlock {
                    global_layer: record.id.global_layer,
                    representation: record.id.representation,
                    start: record.id.start,
                    end: record.id.end,
                    rank: record.id.rank,
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
            validate_state_arrays(
                &descriptor.layer_layout,
                &descriptor.layer_prefix_offsets,
                descriptor.global_layer_start,
                descriptor.batch_size,
                prefix_token_ids.len(),
                state_arrays,
            )?;
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
                sync_file(&shard_path)?;
                let payload_sha256 = hash_shard_payload(&shard_path)?;
                let state_bytes = state.array.nbytes() as u64;
                logical_bytes = logical_bytes.checked_add(state_bytes).ok_or_else(|| {
                    CacheResidencyError::MalformedManifest(
                        "prompt-cache state byte count overflow".into(),
                    )
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
                block_size_tokens: self.options().block_size_tokens,
                batch_size: descriptor.batch_size,
                total_prefix_tokens: prefix_token_ids.len(),
                prefix_sha256: hash_token_ids(prefix_token_ids),
                layer_layout: descriptor.layer_layout,
                layer_prefix_offsets: descriptor.layer_prefix_offsets,
                sink_tokens: descriptor.sink_tokens,
                topology: descriptor.topology,
                application_namespace: options.application_namespace.clone(),
                blocks: manifest_blocks,
                state_tensors: manifest_state,
            };
            let manifest_path = temporary.join("manifest.json");
            let file = File::create(&manifest_path).map_err(|source| CacheResidencyError::Io {
                action: "create prompt cache manifest",
                path: manifest_path.clone(),
                source,
            })?;
            let mut writer = BufWriter::new(file);
            serde_json::to_writer_pretty(&mut writer, &manifest)
                .map_err(CacheResidencyError::ManifestJson)?;
            writer
                .write_all(b"\n")
                .map_err(|source| CacheResidencyError::Io {
                    action: "write prompt cache manifest",
                    path: manifest_path.clone(),
                    source,
                })?;
            writer.flush().map_err(|source| CacheResidencyError::Io {
                action: "flush prompt cache manifest",
                path: manifest_path.clone(),
                source,
            })?;
            sync_file(&manifest_path)?;
            validate_manifest(&temporary, &manifest)?;
            sync_directory(&temporary)?;

            if replacing {
                let generation = generations.join(&generation_name);
                durable_rename(&temporary, &generation, false).map_err(|source| {
                    CacheResidencyError::Io {
                        action: "publish prompt cache generation",
                        path: generation.clone(),
                        source,
                    }
                })?;
                sync_directory(&generations)?;
                publish_prompt_cache_generation(destination, &generation_name, nonce)?;
            } else {
                durable_rename(&temporary, destination, false).map_err(|source| {
                    CacheResidencyError::Io {
                        action: "publish prompt cache",
                        path: destination.to_path_buf(),
                        source,
                    }
                })?;
            }
            sync_directory(parent)?;
            let mut state = self.lock()?;
            state.counters.report.prompt_cache_saves += 1;
            state.counters.report.prompt_cache_bytes += logical_bytes;
            Ok(manifest)
        })();

        if result.is_err() && temporary.exists() {
            let _ = fs::remove_dir_all(&temporary);
        }
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
pub(crate) struct CacheBlockPrefetch {
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
    pub(crate) fn next_block(&mut self) -> Result<Option<CacheBlockLease>, CacheResidencyError> {
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
            .ok_or_else(|| CacheResidencyError::MissingBlock(id.clone()))?
            .bytes;
        Ok(pending_bytes.saturating_add(next_bytes) <= self.manager.options().device_budget_bytes)
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
/// owns SafeMLX's thread-affine [`Event`].
pub(crate) struct CacheBlockLease {
    id: CacheBlockId,
    arrays: CacheBlockArrays,
    manager: CacheResidencyManager,
    completions: Vec<Event>,
    _transfer_reservation: Option<CachePoolReservation>,
    released: bool,
}

impl CacheBlockLease {
    pub(crate) fn id(&self) -> &CacheBlockId {
        &self.id
    }

    pub(crate) fn arrays(&self) -> &CacheBlockArrays {
        &self.arrays
    }

    pub(crate) fn bytes(&self) -> u64 {
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

/// Exact state kind, attention policy, and tensor geometry for one decoder layer.
#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerCachePolicy {
    /// This layer contributes no independently persisted attention state.
    NoState,
    /// Ordinary attention keys and values.
    KeyValue {
        /// Exact full or sliding attention range for this layer.
        attention: AttentionPolicy,
        /// Rank-local key/value head count.
        num_key_value_heads: NonZeroU32,
        /// Per-head key/value dimension.
        head_dim: NonZeroU32,
    },
    /// Attention history whose value payload is intentionally empty.
    KeyOnly {
        /// Exact full or sliding attention range for this layer.
        attention: AttentionPolicy,
        /// Rank-local key head count.
        num_key_heads: NonZeroU32,
        /// Per-head key dimension.
        head_dim: NonZeroU32,
    },
    /// DeepSeek compressed latent state plus rotary keys.
    CompressedLatentRotary {
        /// Exact full or sliding attention range for this layer.
        attention: AttentionPolicy,
        /// Compressed latent width.
        latent_dim: NonZeroU32,
        /// Rotary-key width.
        rotary_dim: NonZeroU32,
    },
    /// Fixed-size recurrent or convolution state without an attention payload.
    FixedState {
        /// Ordered tensors required to resume this layer.
        tensors: Vec<StateTensorPolicy>,
    },
    /// Ordinary attention plus fixed-size recurrent or convolution state.
    KeyValueWithFixedState {
        /// Exact full or sliding attention range for this layer.
        attention: AttentionPolicy,
        /// Rank-local key/value head count.
        num_key_value_heads: NonZeroU32,
        /// Per-head key/value dimension.
        head_dim: NonZeroU32,
        /// Ordered tensors required in addition to keys and values.
        tensors: Vec<StateTensorPolicy>,
    },
    /// Key-only attention plus mutable pooling or recurrent state.
    KeyOnlyWithFixedState {
        /// Exact full or sliding attention range for this layer.
        attention: AttentionPolicy,
        /// Rank-local key head count.
        num_key_heads: NonZeroU32,
        /// Per-head key dimension.
        head_dim: NonZeroU32,
        /// Ordered tensors required in addition to keys.
        tensors: Vec<StateTensorPolicy>,
    },
}

/// Semantic role of one non-attention cache tensor.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateTensorRole {
    /// Bounded causal-convolution history. `slot` distinguishes parallel histories.
    Convolution {
        /// Stable slot within the layer's ordered convolution states.
        slot: u32,
    },
    /// Recurrent transition or linear-attention state.
    Recurrent,
    /// Prepared multimodal prefix embeddings needed for exact replay.
    PrefixEmbedding,
    /// Model-global multimodal position offset.
    PositionDelta,
    /// One tensor in an append-only token-pooling stream.
    Pooling {
        /// Stable stream slot within the owning layer.
        stream: u32,
        /// Exact component of the pooling state.
        component: PoolingStateComponent,
    },
}

/// Semantic component of one append-only pooling stream.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolingStateComponent {
    /// Source values waiting for a complete pooling group.
    PendingValues,
    /// Source gate logits waiting for a complete pooling group.
    PendingGates,
    /// Complete pooled output history.
    Pooled,
    /// Source values retained for an overlapping group.
    OverlapValues,
    /// Source gate logits retained for an overlapping group.
    OverlapGates,
}

/// Runtime ownership behavior for live model state.
///
/// The variants describe mutually exclusive physical lifecycles instead of a
/// tier plus independent flags. Mutable rolling state cannot accidentally be
/// admitted to the sealed pager, and layer-scoped state has an explicit
/// promotion/demotion boundary.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateResidencyClass {
    /// Small mutable state that remains on the execution device for its lifetime.
    AlwaysDeviceMutable,
    /// Append-only state that becomes immutable blocks before paging.
    SealablePaged,
    /// Mutable state used by one layer at a time and eligible for host offload
    /// between layer executions.
    LayerScopedOffloadable,
}

/// Residency behaviors valid for mutable fixed-state tensors.
///
/// `SealablePaged` is deliberately absent: fixed mutable state cannot be
/// constructed as an append-only paged payload.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MutableStateResidency {
    /// Small mutable state that remains on the execution device.
    AlwaysDeviceMutable,
    /// Mutable state promoted only for its owning layer.
    LayerScopedOffloadable,
}

impl From<MutableStateResidency> for StateResidencyClass {
    fn from(value: MutableStateResidency) -> Self {
        match value {
            MutableStateResidency::AlwaysDeviceMutable => Self::AlwaysDeviceMutable,
            MutableStateResidency::LayerScopedOffloadable => Self::LayerScopedOffloadable,
        }
    }
}

/// One dimension in a persisted fixed-state tensor.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateTensorDimension {
    /// Manifest batch size.
    Batch,
    /// Exact prompt token count.
    PrefixTokens,
    /// Integer quotient of the exact prompt token count and a positive divisor.
    PrefixTokensDiv(NonZeroU32),
    /// Integer remainder of the exact prompt token count and a positive divisor.
    PrefixTokensRem(NonZeroU32),
    /// Positive architecture-defined dimension.
    Fixed(NonZeroU32),
    /// Scalar dimension list marker; only valid as the sole entry.
    Scalar,
}

/// Exact condition under which a state tensor must be materialized.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateTensorPresence {
    /// Every persisted cache materializes the tensor.
    Required,
    /// The tensor may be present independently of prefix geometry.
    Optional,
    /// The tensor is present exactly when the prefix has a non-zero remainder.
    PrefixRemainderNonZero(NonZeroU32),
    /// The tensor is present exactly when the prefix contains one complete group.
    PrefixAtLeast(NonZeroU32),
}

/// Accepted dtype family for one fixed-state tensor.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateTensorDtype {
    /// Any MLX floating dtype; the exact stored dtype is still recorded and checked.
    Floating,
    /// Exactly IEEE F32.
    Float32,
    /// Signed 32-bit integer.
    Int32,
    /// Unsigned 32-bit integer.
    Uint32,
}

/// Exact semantic role, symbolic shape, and dtype contract for a state tensor.
#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct StateTensorPolicy {
    /// Meaning of this tensor within its owning layer or global cache state.
    pub role: StateTensorRole,
    /// Exact shape with batch/prefix dimensions resolved from the manifest.
    pub shape: Vec<StateTensorDimension>,
    /// Accepted dtype family.
    pub dtype: StateTensorDtype,
    /// Authoritative live-state residency behavior.
    pub residency: MutableStateResidency,
    /// Exact condition under which persisted caches materialize this tensor.
    pub presence: StateTensorPresence,
}

impl LayerCachePolicy {
    /// Returns the residency behavior of this layer's attention payload.
    pub const fn attention_residency_class(&self) -> Option<StateResidencyClass> {
        match self {
            Self::NoState | Self::FixedState { .. } => None,
            Self::KeyValue { .. }
            | Self::KeyOnly { .. }
            | Self::CompressedLatentRotary { .. }
            | Self::KeyValueWithFixedState { .. }
            | Self::KeyOnlyWithFixedState { .. } => Some(StateResidencyClass::SealablePaged),
        }
    }

    /// Constructs validated ordinary key/value state geometry.
    pub fn key_value(
        attention: AttentionPolicy,
        num_key_value_heads: i32,
        head_dim: i32,
    ) -> Result<Self, CacheResidencyError> {
        Ok(Self::KeyValue {
            attention,
            num_key_value_heads: positive_u32(num_key_value_heads, "key/value head count")?,
            head_dim: positive_u32(head_dim, "key/value head dimension")?,
        })
    }

    /// Constructs validated key-only attention state geometry.
    pub fn key_only(
        attention: AttentionPolicy,
        num_key_heads: i32,
        head_dim: i32,
    ) -> Result<Self, CacheResidencyError> {
        Ok(Self::KeyOnly {
            attention,
            num_key_heads: positive_u32(num_key_heads, "key head count")?,
            head_dim: positive_u32(head_dim, "key head dimension")?,
        })
    }

    /// Constructs validated compressed-latent state geometry.
    pub fn compressed_latent_rotary(
        attention: AttentionPolicy,
        latent_dim: i32,
        rotary_dim: i32,
    ) -> Result<Self, CacheResidencyError> {
        Ok(Self::CompressedLatentRotary {
            attention,
            latent_dim: positive_u32(latent_dim, "compressed latent dimension")?,
            rotary_dim: positive_u32(rotary_dim, "rotary-key dimension")?,
        })
    }

    /// Constructs a validated fixed-state-only layer policy.
    pub fn fixed_only(tensors: Vec<StateTensorPolicy>) -> Result<Self, CacheResidencyError> {
        let policy = Self::FixedState { tensors };
        validate_layer_cache_policy(&policy)?;
        Ok(policy)
    }

    /// Constructs validated ordinary attention plus fixed-state geometry.
    pub fn key_value_with_fixed_state(
        attention: AttentionPolicy,
        num_key_value_heads: i32,
        head_dim: i32,
        tensors: Vec<StateTensorPolicy>,
    ) -> Result<Self, CacheResidencyError> {
        let policy = Self::KeyValueWithFixedState {
            attention,
            num_key_value_heads: positive_u32(num_key_value_heads, "key/value head count")?,
            head_dim: positive_u32(head_dim, "key/value head dimension")?,
            tensors,
        };
        validate_layer_cache_policy(&policy)?;
        Ok(policy)
    }

    /// Constructs validated key-only attention plus mutable state geometry.
    pub fn key_only_with_fixed_state(
        attention: AttentionPolicy,
        num_key_heads: i32,
        head_dim: i32,
        tensors: Vec<StateTensorPolicy>,
    ) -> Result<Self, CacheResidencyError> {
        let policy = Self::KeyOnlyWithFixedState {
            attention,
            num_key_heads: positive_u32(num_key_heads, "key head count")?,
            head_dim: positive_u32(head_dim, "key head dimension")?,
            tensors,
        };
        validate_layer_cache_policy(&policy)?;
        Ok(policy)
    }

    /// Returns the exact attention policy when this layer owns attention state.
    pub const fn attention(&self) -> Option<AttentionPolicy> {
        match self {
            Self::NoState | Self::FixedState { .. } => None,
            Self::KeyValue { attention, .. }
            | Self::KeyOnly { attention, .. }
            | Self::CompressedLatentRotary { attention, .. }
            | Self::KeyValueWithFixedState { attention, .. }
            | Self::KeyOnlyWithFixedState { attention, .. } => Some(*attention),
        }
    }

    /// Returns the ordered non-attention tensor policies for this layer.
    pub fn fixed_state(&self) -> &[StateTensorPolicy] {
        match self {
            Self::FixedState { tensors }
            | Self::KeyValueWithFixedState { tensors, .. }
            | Self::KeyOnlyWithFixedState { tensors, .. } => tensors,
            _ => &[],
        }
    }
}

impl StateTensorDimension {
    /// Constructs a positive fixed dimension.
    pub fn fixed(value: i32) -> Result<Self, CacheResidencyError> {
        positive_u32(value, "fixed-state tensor dimension").map(Self::Fixed)
    }
}

impl StateTensorPolicy {
    /// Constructs a state-tensor policy after validating its symbolic shape.
    pub fn new(
        role: StateTensorRole,
        shape: Vec<StateTensorDimension>,
        dtype: StateTensorDtype,
        residency: MutableStateResidency,
    ) -> Result<Self, CacheResidencyError> {
        let policy = Self {
            role,
            shape,
            dtype,
            residency,
            presence: StateTensorPresence::Required,
        };
        validate_state_tensor_policies(std::slice::from_ref(&policy))?;
        Ok(policy)
    }

    /// Marks this tensor as absent for cache instances that do not use the state.
    pub const fn optional(mut self) -> Self {
        self.presence = StateTensorPresence::Optional;
        self
    }

    /// Requires this tensor exactly when `prefix_tokens % divisor != 0`.
    pub const fn when_prefix_remainder_nonzero(mut self, divisor: NonZeroU32) -> Self {
        self.presence = StateTensorPresence::PrefixRemainderNonZero(divisor);
        self
    }

    /// Requires this tensor exactly when `prefix_tokens >= divisor`.
    pub const fn when_prefix_at_least(mut self, divisor: NonZeroU32) -> Self {
        self.presence = StateTensorPresence::PrefixAtLeast(divisor);
        self
    }

    /// Returns whether this tensor must be present for the exact prompt length.
    pub fn is_required_for(&self, prefix_tokens: usize) -> bool {
        match self.presence {
            StateTensorPresence::Required => true,
            StateTensorPresence::Optional => false,
            StateTensorPresence::PrefixRemainderNonZero(divisor) => {
                !prefix_tokens.is_multiple_of(divisor.get() as usize)
            }
            StateTensorPresence::PrefixAtLeast(divisor) => prefix_tokens >= divisor.get() as usize,
        }
    }

    /// Returns the unified residency classification for this fixed state.
    pub fn residency_class(&self) -> StateResidencyClass {
        self.residency.into()
    }
}

fn positive_u32(value: i32, field: &str) -> Result<NonZeroU32, CacheResidencyError> {
    u32::try_from(value)
        .ok()
        .and_then(NonZeroU32::new)
        .ok_or_else(|| {
            CacheResidencyError::InvalidOptions(format!(
                "prompt-cache {field} must be positive and fit u32, got {value}"
            ))
        })
}

fn attention_policy_from_sliding_window(
    window: Option<i32>,
) -> Result<AttentionPolicy, CacheResidencyError> {
    match window {
        None => Ok(AttentionPolicy::Full),
        Some(window) => {
            let window = u32::try_from(window)
                .ok()
                .and_then(NonZeroU32::new)
                .ok_or_else(|| {
                    CacheResidencyError::InvalidOptions(format!(
                        "attention sliding window must be positive and fit u32, got {window}"
                    ))
                })?;
            Ok(AttentionPolicy::Sliding { window })
        }
    }
}

fn attention_policy_sliding_window_i32(
    attention: AttentionPolicy,
) -> Result<Option<i32>, CacheResidencyError> {
    attention
        .window()
        .map(|window| {
            i32::try_from(window.get()).map_err(|_| {
                CacheResidencyError::IncompatiblePromptCache(format!(
                    "attention sliding window {window} exceeds the runtime i32 range"
                ))
            })
        })
        .transpose()
}

fn validate_layer_cache_policy(policy: &LayerCachePolicy) -> Result<(), CacheResidencyError> {
    if let Some(attention) = policy.attention() {
        attention_policy_sliding_window_i32(attention)?;
    }
    let validate_dimension = |dimension: NonZeroU32| {
        if dimension.get() > i32::MAX as u32 {
            Err(CacheResidencyError::IncompatiblePromptCache(format!(
                "prompt-cache layer dimension {dimension} exceeds the runtime i32 range"
            )))
        } else {
            Ok(())
        }
    };
    match policy {
        LayerCachePolicy::NoState => {}
        LayerCachePolicy::KeyValue {
            num_key_value_heads,
            head_dim,
            ..
        }
        | LayerCachePolicy::KeyValueWithFixedState {
            num_key_value_heads,
            head_dim,
            ..
        } => {
            validate_dimension(*num_key_value_heads)?;
            validate_dimension(*head_dim)?;
        }
        LayerCachePolicy::KeyOnly {
            num_key_heads,
            head_dim,
            ..
        }
        | LayerCachePolicy::KeyOnlyWithFixedState {
            num_key_heads,
            head_dim,
            ..
        } => {
            validate_dimension(*num_key_heads)?;
            validate_dimension(*head_dim)?;
        }
        LayerCachePolicy::CompressedLatentRotary {
            latent_dim,
            rotary_dim,
            ..
        } => {
            validate_dimension(*latent_dim)?;
            validate_dimension(*rotary_dim)?;
        }
        LayerCachePolicy::FixedState { .. } => {}
    }
    let tensors = policy.fixed_state();
    if tensors.is_empty()
        && matches!(
            policy,
            LayerCachePolicy::FixedState { .. }
                | LayerCachePolicy::KeyValueWithFixedState { .. }
                | LayerCachePolicy::KeyOnlyWithFixedState { .. }
        )
    {
        return Err(CacheResidencyError::InvalidOptions(
            "fixed-state cache policy must contain at least one tensor".into(),
        ));
    }
    validate_state_tensor_policies(tensors)
}

fn validate_state_tensor_policies(
    tensors: &[StateTensorPolicy],
) -> Result<(), CacheResidencyError> {
    let mut roles = BTreeSet::new();
    for tensor in tensors {
        if !roles.insert(tensor.role) {
            return Err(CacheResidencyError::InvalidOptions(format!(
                "duplicate fixed-state tensor role {:?}",
                tensor.role
            )));
        }
        if tensor.shape.is_empty()
            || (tensor.shape.contains(&StateTensorDimension::Scalar)
                && tensor.shape.as_slice() != [StateTensorDimension::Scalar])
        {
            return Err(CacheResidencyError::InvalidOptions(format!(
                "invalid fixed-state tensor shape for role {:?}",
                tensor.role
            )));
        }
        let expected = match tensor.role {
            StateTensorRole::Recurrent => MutableStateResidency::LayerScopedOffloadable,
            StateTensorRole::Convolution { .. }
            | StateTensorRole::PrefixEmbedding
            | StateTensorRole::PositionDelta
            | StateTensorRole::Pooling { .. } => MutableStateResidency::AlwaysDeviceMutable,
        };
        if tensor.residency != expected {
            return Err(CacheResidencyError::InvalidOptions(format!(
                "fixed-state tensor role {:?} requires {:?} residency, got {:?}",
                tensor.role, expected, tensor.residency
            )));
        }
    }
    Ok(())
}

fn resolved_state_shape(
    policy: &StateTensorPolicy,
    batch_size: usize,
    prefix_tokens: usize,
) -> Result<Vec<i32>, CacheResidencyError> {
    policy
        .shape
        .iter()
        .map(|dimension| match dimension {
            StateTensorDimension::Batch => i32::try_from(batch_size),
            StateTensorDimension::PrefixTokens => i32::try_from(prefix_tokens),
            StateTensorDimension::PrefixTokensDiv(divisor) => {
                i32::try_from(prefix_tokens / divisor.get() as usize)
            }
            StateTensorDimension::PrefixTokensRem(divisor) => {
                i32::try_from(prefix_tokens % divisor.get() as usize)
            }
            StateTensorDimension::Fixed(value) => i32::try_from(value.get()),
            StateTensorDimension::Scalar => Ok(1),
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| {
            CacheResidencyError::InvalidOptions(
                "fixed-state tensor dimension exceeds runtime i32 range".into(),
            )
        })
}

fn state_policy(
    layers: &LayerSchedule<LayerCachePolicy>,
    global_layer_start: usize,
    owner: StateTensorOwner,
    role: StateTensorRole,
) -> Option<&StateTensorPolicy> {
    let StateTensorOwner::Layer(layer) = owner;
    let policies = layers
        .get(layer.checked_sub(global_layer_start)?)?
        .fixed_state();
    policies.iter().find(|policy| policy.role == role)
}

fn validate_state_arrays(
    layers: &LayerSchedule<LayerCachePolicy>,
    layer_prefix_offsets: &[i32],
    global_layer_start: usize,
    batch_size: usize,
    prefix_tokens: usize,
    arrays: &[PromptCacheStateArray<'_>],
) -> Result<(), CacheResidencyError> {
    let mut seen = BTreeSet::new();
    for state in arrays {
        if !seen.insert((state.owner, state.role)) {
            return Err(CacheResidencyError::MalformedManifest(format!(
                "duplicate fixed-state tensor {:?} for {:?}",
                state.role, state.owner
            )));
        }
        let policy =
            state_policy(layers, global_layer_start, state.owner, state.role).ok_or_else(|| {
                CacheResidencyError::MalformedManifest(format!(
                    "unexpected fixed-state tensor {:?} for {:?}",
                    state.role, state.owner
                ))
            })?;
        let StateTensorOwner::Layer(global_layer) = state.owner;
        let layer_index = global_layer
            .checked_sub(global_layer_start)
            .ok_or_else(|| {
                CacheResidencyError::MalformedManifest(format!(
                    "fixed-state tensor owner {global_layer} precedes the owned layer range"
                ))
            })?;
        let layer_tokens = layer_prefix_tokens(
            prefix_tokens,
            *layer_prefix_offsets.get(layer_index).ok_or_else(|| {
                CacheResidencyError::MalformedManifest(format!(
                    "missing prefix offset for global layer {global_layer}"
                ))
            })?,
        )?;
        let expected = resolved_state_shape(policy, batch_size, layer_tokens)?;
        if state.array.shape() != expected {
            return Err(CacheResidencyError::MalformedManifest(format!(
                "fixed-state tensor {:?} for {:?} has shape {:?}, expected {:?}",
                state.role,
                state.owner,
                state.array.shape(),
                expected
            )));
        }
        let dtype = dtype_name(state.array.dtype());
        if !state_dtype_matches(policy.dtype, &dtype) {
            return Err(CacheResidencyError::MalformedManifest(format!(
                "fixed-state tensor {:?} for {:?} has incompatible dtype {dtype}",
                state.role, state.owner
            )));
        }
    }
    let mut expected = Vec::new();
    for (index, layer) in layers.iter().enumerate() {
        let owner = StateTensorOwner::Layer(global_layer_start + index);
        let layer_tokens = layer_prefix_tokens(
            prefix_tokens,
            *layer_prefix_offsets.get(index).ok_or_else(|| {
                CacheResidencyError::MalformedManifest(format!(
                    "missing prefix offset for global layer {}",
                    global_layer_start + index
                ))
            })?,
        )?;
        for policy in layer.fixed_state() {
            if policy.is_required_for(layer_tokens) || seen.contains(&(owner, policy.role)) {
                expected.push((owner, policy.role));
            }
        }
    }
    let actual = arrays
        .iter()
        .map(|state| (state.owner, state.role))
        .collect::<Vec<_>>();
    if actual != expected {
        return Err(CacheResidencyError::MalformedManifest(format!(
            "fixed-state tensors are missing, reordered, or unexpected: found {actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn state_dtype_matches(policy: StateTensorDtype, dtype: &str) -> bool {
    match policy {
        StateTensorDtype::Floating => {
            matches!(dtype, "Float16" | "Bfloat16" | "Float32" | "Float64")
        }
        StateTensorDtype::Float32 => dtype == "Float32",
        StateTensorDtype::Int32 => dtype == "Int32",
        StateTensorDtype::Uint32 => dtype == "Uint32",
    }
}

/// Compatibility identity supplied by a model when persisting a prefix cache.
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct PromptCacheDescriptor {
    /// Stable architecture family, such as `llama` or `deepseek_v3`.
    pub model_family: String,
    /// Effective normalized model type.
    pub effective_model_type: String,
    /// Caller-verified checkpoint identity that is not based only on a path.
    pub checkpoint_fingerprint: String,
    /// Caller-supplied identity for the fully processed prefix content.
    ///
    /// Text callers normally hash token IDs. Multimodal callers must also hash
    /// media content and processor settings that contributed cached activations.
    pub prefix_content_fingerprint: String,
    /// Hash or canonical serialization of RoPE and cache-relevant architecture settings.
    pub architecture_fingerprint: String,
    /// Total model layer count.
    pub layer_count: usize,
    /// Inclusive global layer range stored by this rank.
    pub global_layer_start: usize,
    /// Exclusive global layer range stored by this rank.
    pub global_layer_end: usize,
    /// Prefix batch size.
    pub batch_size: usize,
    /// Exact ordered cache state and attention layout for the owned layer range.
    pub layer_layout: LayerSchedule<LayerCachePolicy>,
    /// Per-owned-layer processed-token delta relative to the persisted prefix.
    /// Ordinary decoder layers use zero; speculative layers may trail it.
    pub layer_prefix_offsets: Vec<i32>,
    /// Attention sink or pinned-prefix token count.
    pub sink_tokens: usize,
    /// Distributed rank-local layout.
    pub topology: PromptCacheTopology,
}

/// Cache-relevant structure derived from a loaded model instance.
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct PromptCacheModelIdentity {
    pub(crate) model_family: String,
    pub(crate) effective_model_type: String,
    pub(crate) architecture_fingerprint: String,
    pub(crate) layer_count: usize,
    pub(crate) global_layer_start: usize,
    pub(crate) global_layer_end: usize,
    pub(crate) sink_tokens: usize,
    pub(crate) topology: PromptCacheTopology,
    pub(crate) layer_layout: LayerSchedule<LayerCachePolicy>,
    pub(crate) layer_prefix_offsets: Vec<i32>,
}

impl PromptCacheModelIdentity {
    pub(crate) fn key_value_layouts(
        sliding_windows: impl IntoIterator<Item = Option<i32>>,
        num_key_value_heads: i32,
        head_dim: i32,
    ) -> Result<LayerSchedule<LayerCachePolicy>, CacheResidencyError> {
        let policies = sliding_windows
            .into_iter()
            .map(|window| {
                let attention = attention_policy_from_sliding_window(window)?;
                LayerCachePolicy::key_value(attention, num_key_value_heads, head_dim)
            })
            .collect::<Result<Vec<_>, _>>()?;
        LayerSchedule::new(policies.len(), policies)
            .map_err(|error| CacheResidencyError::InvalidOptions(error.to_string()))
    }

    pub(crate) fn compressed_layouts(
        layer_count: usize,
        latent_dim: i32,
        rotary_dim: i32,
    ) -> Result<LayerSchedule<LayerCachePolicy>, CacheResidencyError> {
        let policies = (0..layer_count)
            .map(|_| {
                LayerCachePolicy::compressed_latent_rotary(
                    AttentionPolicy::Full,
                    latent_dim,
                    rotary_dim,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        LayerSchedule::new(layer_count, policies)
            .map_err(|error| CacheResidencyError::InvalidOptions(error.to_string()))
    }
}

pub(crate) fn validate_prompt_cache_model_identity(
    expected: &PromptCacheDescriptor,
    model: &PromptCacheModelIdentity,
) -> Result<(), CacheResidencyError> {
    macro_rules! require_model_equal {
        ($field:ident) => {
            if expected.$field != model.$field {
                return Err(CacheResidencyError::IncompatiblePromptCache(format!(
                    "caller descriptor {} does not match the loaded model",
                    stringify!($field)
                )));
            }
        };
    }
    require_model_equal!(model_family);
    require_model_equal!(effective_model_type);
    require_model_equal!(architecture_fingerprint);
    require_model_equal!(layer_count);
    require_model_equal!(global_layer_start);
    require_model_equal!(global_layer_end);
    require_model_equal!(sink_tokens);
    require_model_equal!(topology);
    require_model_equal!(layer_layout);
    require_model_equal!(layer_prefix_offsets);
    let owned_layers = model
        .global_layer_end
        .checked_sub(model.global_layer_start)
        .ok_or_else(|| {
            CacheResidencyError::IncompatiblePromptCache(
                "loaded model has an invalid prompt-cache layer range".into(),
            )
        })?;
    if model.layer_layout.len() != owned_layers {
        return Err(CacheResidencyError::IncompatiblePromptCache(format!(
            "loaded model supplied {} cache layouts for {owned_layers} owned layers",
            model.layer_layout.len()
        )));
    }
    if model.layer_prefix_offsets.len() != owned_layers {
        return Err(CacheResidencyError::IncompatiblePromptCache(format!(
            "loaded model supplied {} layer prefix offsets for {owned_layers} owned layers",
            model.layer_prefix_offsets.len()
        )));
    }
    Ok(())
}

pub(crate) fn derive_prompt_cache_architecture_fingerprint<I, K, V>(
    model_family: &str,
    fields: I,
) -> String
where
    I: IntoIterator<Item = (K, V)>,
    K: Into<String>,
    V: Into<String>,
{
    let mut fields = fields
        .into_iter()
        .map(|(key, value)| (key.into(), value.into()))
        .collect::<Vec<_>>();
    fields.sort_unstable();
    let mut hasher = Sha256::new();
    hash_fingerprint_component(&mut hasher, b"safemlx-prompt-cache-architecture-v1");
    hash_fingerprint_component(&mut hasher, model_family.as_bytes());
    for (key, value) in fields {
        hash_fingerprint_component(&mut hasher, key.as_bytes());
        hash_fingerprint_component(&mut hasher, value.as_bytes());
    }
    format!("sha256:{}", sha256_hex(hasher.finalize()))
}

fn hash_fingerprint_component(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

/// Rank-local topology recorded in a prompt-cache manifest.
#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct PromptCacheTopology {
    /// Pipeline world size and rank.
    pub pipeline: Option<(usize, usize)>,
    /// Tensor-parallel world size and rank.
    pub tensor_parallel: Option<(usize, usize)>,
    /// Expert-parallel world size and rank.
    pub expert_parallel: Option<(usize, usize)>,
    /// Whether attention cache state is replicated on the expert-parallel axis.
    pub expert_parallel_cache_replicated: bool,
}

impl Default for PromptCacheTopology {
    fn default() -> Self {
        Self {
            pipeline: None,
            tensor_parallel: None,
            expert_parallel: None,
            expert_parallel_cache_replicated: true,
        }
    }
}

impl PromptCacheTopology {
    pub(crate) fn for_parallel_topology(
        topology: crate::runtime::distributed::topology::ParallelTopology,
    ) -> Self {
        Self {
            pipeline: (topology.pipeline_parallel_size > 1).then_some((
                topology.pipeline_parallel_size,
                topology.pipeline_parallel_rank,
            )),
            tensor_parallel: (topology.tensor_parallel_size > 1)
                .then_some((topology.tensor_parallel_size, topology.tensor_parallel_rank)),
            expert_parallel: (topology.expert_parallel_size > 1)
                .then_some((topology.expert_parallel_size, topology.expert_parallel_rank)),
            expert_parallel_cache_replicated: true,
        }
    }

    pub(crate) fn cache_rank_identity(&self) -> Option<CacheRankIdentity> {
        (self.pipeline.is_some()
            || self.tensor_parallel.is_some()
            || self.expert_parallel.is_some())
        .then(|| CacheRankIdentity {
            pipeline_rank: self.pipeline.map(|(_, rank)| rank),
            tensor_parallel_rank: self.tensor_parallel.map(|(_, rank)| rank),
            expert_parallel_rank: self.expert_parallel.map(|(_, rank)| rank),
        })
    }
}

/// Explicit persistence behavior for a reusable prefix cache.
#[derive(Debug, Clone, Default)]
pub struct PromptCacheOptions {
    /// Optional application grouping label; never used for compatibility checks.
    pub application_namespace: Option<String>,
    /// Allows atomically replacing an existing destination.
    pub replace_existing: bool,
}

/// Versioned metadata that can be inspected without loading block arrays.
#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct PromptCacheManifest {
    /// Persistence schema version.
    pub schema_version: u32,
    /// Model architecture family.
    pub model_family: String,
    /// Effective normalized model type.
    pub effective_model_type: String,
    /// Checkpoint identity contract selected by the caller.
    pub checkpoint_fingerprint: String,
    /// Identity of text and any processed media that produced this prefix.
    pub prefix_content_fingerprint: String,
    /// RoPE and cache-relevant architecture identity.
    pub architecture_fingerprint: String,
    /// Total model layer count.
    pub layer_count: usize,
    /// Inclusive first global layer represented locally.
    pub global_layer_start: usize,
    /// Exclusive global layer boundary represented locally.
    pub global_layer_end: usize,
    /// Block size used by the producer.
    pub block_size_tokens: i32,
    /// Prefix batch size.
    pub batch_size: usize,
    /// Exact prefix token count.
    pub total_prefix_tokens: usize,
    /// SHA-256 over little-endian prefix token ids.
    pub prefix_sha256: String,
    /// Exact ordered cache state, attention policy, and tensor geometry.
    pub layer_layout: LayerSchedule<LayerCachePolicy>,
    /// Per-owned-layer processed-token delta relative to `total_prefix_tokens`.
    pub layer_prefix_offsets: Vec<i32>,
    /// Pinned prefix or sink token count.
    pub sink_tokens: usize,
    /// Distributed rank-local representation.
    pub topology: PromptCacheTopology,
    /// Optional non-authoritative application grouping label.
    pub application_namespace: Option<String>,
    /// Ordered immutable cache blocks.
    pub blocks: Vec<PromptCacheBlock>,
    /// Ordered fixed-size layer and model-global state tensors.
    pub state_tensors: Vec<PromptCacheStateTensor>,
}

/// Owner of one persisted non-attention state tensor.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateTensorOwner {
    /// Architecture-global decoder layer index.
    Layer(usize),
}

/// One independently validated fixed-size state tensor.
#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct PromptCacheStateTensor {
    /// Layer or model-global owner.
    pub owner: StateTensorOwner,
    /// Semantic role declared by the canonical layout.
    pub role: StateTensorRole,
    /// Safe relative safetensors shard path.
    pub shard: String,
    /// Array name within the shard.
    pub array: String,
    /// Exact stored shape.
    pub shape: Vec<i32>,
    /// Exact stored dtype.
    pub dtype: String,
    /// Logical bytes in the array.
    pub logical_bytes: u64,
    /// SHA-256 of the exact safetensors payload bytes.
    pub payload_sha256: String,
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

/// One cache block catalog entry in a prompt-cache manifest.
#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct PromptCacheBlock {
    /// Architecture-global layer identity.
    pub global_layer: usize,
    /// Stored attention representation.
    pub representation: CacheRepresentation,
    /// Inclusive absolute token position.
    pub start: i64,
    /// Exclusive absolute token position.
    pub end: i64,
    /// Optional rank identity.
    pub rank: Option<CacheRankIdentity>,
    /// Safe relative safetensors shard path.
    pub shard: String,
    /// First array name.
    pub first_array: String,
    /// Second array name.
    pub second_array: String,
    /// First array shape.
    pub first_shape: Vec<i32>,
    /// Second array shape.
    pub second_shape: Vec<i32>,
    /// First array dtype.
    pub first_dtype: String,
    /// Second array dtype.
    pub second_dtype: String,
    /// Logical bytes in both arrays.
    pub logical_bytes: u64,
    /// SHA-256 of the exact safetensors payload bytes.
    pub payload_sha256: String,
}

/// Reads and validates a prompt-cache manifest without loading its arrays.
pub fn inspect_prompt_cache(
    directory: impl AsRef<Path>,
) -> Result<PromptCacheManifest, CacheResidencyError> {
    let directory = resolve_prompt_cache_root(directory.as_ref())?;
    let manifest_path = directory.join("manifest.json");
    let reader =
        BufReader::new(
            File::open(&manifest_path).map_err(|source| CacheResidencyError::Io {
                action: "open prompt cache manifest",
                path: manifest_path.clone(),
                source,
            })?,
        );
    let value: serde_json::Value =
        serde_json::from_reader(reader).map_err(CacheResidencyError::ManifestJson)?;
    let schema_version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .and_then(|version| u32::try_from(version).ok())
        .ok_or_else(|| {
            CacheResidencyError::MalformedManifest(
                "prompt-cache schema_version is missing or is not a u32".into(),
            )
        })?;
    if schema_version != PROMPT_CACHE_SCHEMA_VERSION {
        return Err(CacheResidencyError::UnsupportedSchema(schema_version));
    }
    let manifest: PromptCacheManifest =
        serde_json::from_value(value).map_err(CacheResidencyError::ManifestJson)?;
    validate_manifest(&directory, &manifest)?;
    Ok(manifest)
}

/// Catalogs a compatible prompt prefix lazily as read-only disk-backed blocks.
pub(crate) fn open_prompt_cache(
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
    validate_compatibility(&manifest, expected, prefix_token_ids)?;
    validate_prompt_cache_layer_layouts(&manifest, model)?;
    if manifest.block_size_tokens != options.block_size_tokens {
        return Err(CacheResidencyError::IncompatiblePromptCache(format!(
            "block size {} does not match requested {}",
            manifest.block_size_tokens, options.block_size_tokens
        )));
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
            let shard = safe_shard_path(&cache_root, &block.shard)?;
            let mapped = map_prompt_cache_shard(&shard)?;
            let record = CacheBlockRecord {
                id: id.clone(),
                physical: CacheBlockPhysicalState::Disk {
                    location: DiskLocation {
                        path: shard,
                        first_name: block.first_array.clone(),
                        second_name: block.second_array.clone(),
                        persistent: true,
                        mapped: Some(mapped),
                        payload_sha256: Some(block.payload_sha256.clone()),
                        payload_verification: Arc::new(OnceLock::new()),
                    },
                    read: DiskCacheReadState::Ready,
                },
                bytes: block.logical_bytes,
                shapes: [block.first_shape.clone(), block.second_shape.clone()],
                dtypes: [block.first_dtype.clone(), block.second_dtype.clone()],
                imported: true,
                leases: 0,
                access_count: 0,
                last_access: 0,
                protected_prefix: block.end <= manifest.sink_tokens as i64,
            };
            if state.blocks.insert(id.clone(), record).is_some() {
                return Err(CacheResidencyError::DuplicateBlock(id));
            }
        }
        state.counters.report.prompt_cache_loads += 1;
        state.counters.report.prompt_cache_bytes += manifest
            .blocks
            .iter()
            .map(|block| block.logical_bytes)
            .sum::<u64>();
        state.counters.report.imported_mapped_shards += manifest.blocks.len() as u64;
        update_report_totals(&mut state);
    }
    Ok((manager, manifest))
}

/// Atomically saves resident attention blocks and fixed state through the canonical manifest.
pub(crate) fn save_prompt_cache_snapshot(
    destination: impl AsRef<Path>,
    descriptor: PromptCacheDescriptor,
    prefix_token_ids: &[u32],
    blocks: Vec<PromptCacheSnapshotBlock>,
    state_arrays: &[PromptCacheStateArray<'_>],
    options: &PromptCacheOptions,
) -> Result<PromptCacheManifest, CacheResidencyError> {
    let block_size = i32::try_from(prefix_token_ids.len()).map_err(|_| {
        CacheResidencyError::InvalidOptions(
            "prompt-cache prefix length exceeds the runtime i32 range".into(),
        )
    })?;
    let bytes = blocks
        .iter()
        .map(|block| block.arrays.bytes())
        .try_fold(0u64, |total, bytes| total.checked_add(bytes))
        .ok_or_else(|| {
            CacheResidencyError::InvalidOptions("prompt-cache byte count overflow".into())
        })?
        .max(1);
    let manager = CacheResidencyManager::new(
        PagedCacheOptions::new(block_size.max(1), bytes, 1, 1)?
            .with_full_attention(true)
            .with_persistence_retention(true),
    )?;
    for block in blocks {
        manager.seal_block(
            block.global_layer,
            block.start,
            block.end,
            block.rank,
            block.arrays,
            false,
        )?;
    }
    manager.save_prompt_cache(
        destination,
        descriptor,
        prefix_token_ids,
        state_arrays,
        options,
    )
}

/// Validates and materializes a resident prompt-cache snapshot on `stream`.
pub(crate) fn open_prompt_cache_snapshot(
    directory: impl AsRef<Path>,
    expected: &PromptCacheDescriptor,
    model: &PromptCacheModelIdentity,
    prefix_token_ids: &[u32],
    stream: &Stream,
) -> Result<
    (
        Vec<LoadedPromptCacheBlock>,
        Vec<LoadedPromptCacheStateTensor>,
        PromptCacheManifest,
    ),
    CacheResidencyError,
> {
    validate_prompt_cache_model_identity(expected, model)?;
    let root = resolve_prompt_cache_root(directory.as_ref())?;
    let manifest = inspect_prompt_cache(directory.as_ref())?;
    validate_compatibility(&manifest, expected, prefix_token_ids)?;
    validate_prompt_cache_layer_layouts(&manifest, model)?;
    let host_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut blocks = Vec::with_capacity(manifest.blocks.len());
    for block in &manifest.blocks {
        let path = safe_shard_path(&root, &block.shard)?;
        if hash_shard_payload(&path)? != block.payload_sha256 {
            return Err(CacheResidencyError::MalformedShard {
                path,
                reason: "attention payload digest does not match the manifest".into(),
            });
        }
        let mut arrays = Array::load_safetensors(&path, &host_stream).map_err(|source| {
            CacheResidencyError::Runtime(format!("load {}: {source}", path.display()))
        })?;
        let first = arrays.remove(&block.first_array).ok_or_else(|| {
            CacheResidencyError::MalformedShard {
                path: path.clone(),
                reason: format!("missing array {}", block.first_array),
            }
        })?;
        let second = arrays.remove(&block.second_array).ok_or_else(|| {
            CacheResidencyError::MalformedShard {
                path: path.clone(),
                reason: format!("missing array {}", block.second_array),
            }
        })?;
        if !arrays.is_empty() {
            return Err(CacheResidencyError::MalformedShard {
                path,
                reason: "attention shard contains unexpected arrays".into(),
            });
        }
        let first = first.copy(stream).map_err(|source| {
            CacheResidencyError::Runtime(format!(
                "copy {} to execution stream: {source}",
                path.display()
            ))
        })?;
        let second = second.copy(stream).map_err(|source| {
            CacheResidencyError::Runtime(format!(
                "copy {} to execution stream: {source}",
                path.display()
            ))
        })?;
        eval([&first, &second]).map_err(|source| {
            CacheResidencyError::Runtime(format!(
                "evaluate {} on execution stream: {source}",
                path.display()
            ))
        })?;
        let arrays = match block.representation {
            CacheRepresentation::KeyValue => CacheBlockArrays::KeyValue {
                keys: first,
                values: second,
            },
            CacheRepresentation::CompressedLatentRotary => {
                CacheBlockArrays::CompressedLatentRotary {
                    latent: first,
                    rotary_key: second,
                }
            }
        };
        blocks.push(LoadedPromptCacheBlock {
            global_layer: block.global_layer,
            start: block.start,
            end: block.end,
            arrays,
        });
    }
    let state = load_prompt_cache_state_tensors(directory, &manifest, stream)?;
    Ok((blocks, state, manifest))
}

/// Materialized non-attention state tensor from a validated prompt cache.
pub(crate) struct LoadedPromptCacheStateTensor {
    pub(crate) owner: StateTensorOwner,
    pub(crate) role: StateTensorRole,
    pub(crate) array: Array,
}

/// Loads all fixed-state tensors after manifest and model compatibility validation.
pub(crate) fn load_prompt_cache_state_tensors(
    directory: impl AsRef<Path>,
    manifest: &PromptCacheManifest,
    stream: &Stream,
) -> Result<Vec<LoadedPromptCacheStateTensor>, CacheResidencyError> {
    let root = resolve_prompt_cache_root(directory.as_ref())?;
    let host_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut loaded = Vec::with_capacity(manifest.state_tensors.len());
    for state in &manifest.state_tensors {
        let path = safe_shard_path(&root, &state.shard)?;
        let actual_hash = hash_shard_payload(&path)?;
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

fn validate_prompt_cache_layer_layouts(
    manifest: &PromptCacheManifest,
    model: &PromptCacheModelIdentity,
) -> Result<(), CacheResidencyError> {
    let owned_layers = model
        .global_layer_end
        .checked_sub(model.global_layer_start)
        .ok_or_else(|| {
            CacheResidencyError::IncompatiblePromptCache(
                "loaded model has an invalid prompt-cache layer range".into(),
            )
        })?;
    if model.layer_layout.len() != owned_layers {
        return Err(CacheResidencyError::IncompatiblePromptCache(format!(
            "loaded model supplied {} cache layouts for {owned_layers} owned layers",
            model.layer_layout.len()
        )));
    }
    for block in &manifest.blocks {
        let layout_index = block
            .global_layer
            .checked_sub(model.global_layer_start)
            .filter(|index| *index < model.layer_layout.len())
            .ok_or_else(|| {
                CacheResidencyError::IncompatiblePromptCache(format!(
                    "cache block layer {} is not owned by the loaded model",
                    block.global_layer
                ))
            })?;
        let token_count = i32::try_from(block.end - block.start).map_err(|_| {
            CacheResidencyError::IncompatiblePromptCache(format!(
                "cache block layer {} token range exceeds runtime dimensions",
                block.global_layer
            ))
        })?;
        let batch = i32::try_from(manifest.batch_size).map_err(|_| {
            CacheResidencyError::IncompatiblePromptCache(
                "prompt-cache batch size exceeds runtime dimensions".into(),
            )
        })?;
        let (representation, first_shape, second_shape) = match model
            .layer_layout
            .get(layout_index)
            .expect("validated prompt-cache layout index")
        {
            LayerCachePolicy::NoState | LayerCachePolicy::FixedState { .. } => {
                return Err(CacheResidencyError::IncompatiblePromptCache(format!(
                    "cache block unexpectedly materializes state for stateless layer {}",
                    block.global_layer
                )))
            }
            LayerCachePolicy::KeyValue {
                num_key_value_heads,
                head_dim,
                ..
            }
            | LayerCachePolicy::KeyValueWithFixedState {
                num_key_value_heads,
                head_dim,
                ..
            } => (
                CacheRepresentation::KeyValue,
                vec![
                    batch,
                    num_key_value_heads.get() as i32,
                    token_count,
                    head_dim.get() as i32,
                ],
                vec![
                    batch,
                    num_key_value_heads.get() as i32,
                    token_count,
                    head_dim.get() as i32,
                ],
            ),
            LayerCachePolicy::KeyOnly {
                num_key_heads,
                head_dim,
                ..
            }
            | LayerCachePolicy::KeyOnlyWithFixedState {
                num_key_heads,
                head_dim,
                ..
            } => (
                CacheRepresentation::KeyValue,
                vec![
                    batch,
                    num_key_heads.get() as i32,
                    token_count,
                    head_dim.get() as i32,
                ],
                vec![batch, num_key_heads.get() as i32, token_count, 1],
            ),
            LayerCachePolicy::CompressedLatentRotary {
                latent_dim,
                rotary_dim,
                ..
            } => (
                CacheRepresentation::CompressedLatentRotary,
                vec![batch, token_count, latent_dim.get() as i32],
                vec![batch, token_count, rotary_dim.get() as i32],
            ),
        };
        if block.representation != representation {
            return Err(CacheResidencyError::IncompatiblePromptCache(format!(
                "cache block layer {} uses {:?}, but the loaded model expects {:?}",
                block.global_layer, block.representation, representation
            )));
        }
        if block.first_shape != first_shape || block.second_shape != second_shape {
            return Err(CacheResidencyError::IncompatiblePromptCache(format!(
                "cache block layer {} dimensions {:?}/{:?} do not match the loaded model's expected {:?}/{:?}",
                block.global_layer,
                block.first_shape,
                block.second_shape,
                first_shape,
                second_shape
            )));
        }
    }
    Ok(())
}

fn resolve_prompt_cache_root(directory: &Path) -> Result<PathBuf, CacheResidencyError> {
    let current_path = directory.join(PROMPT_CACHE_CURRENT_FILE);
    if !current_path.exists() {
        return Ok(directory.to_path_buf());
    }
    let length = current_path
        .metadata()
        .map_err(|source| CacheResidencyError::Io {
            action: "stat prompt cache generation pointer",
            path: current_path.clone(),
            source,
        })?
        .len();
    if length == 0 || length > 256 {
        return Err(CacheResidencyError::MalformedManifest(
            "prompt-cache generation pointer has an invalid length".into(),
        ));
    }
    let generation =
        fs::read_to_string(&current_path).map_err(|source| CacheResidencyError::Io {
            action: "read prompt cache generation pointer",
            path: current_path.clone(),
            source,
        })?;
    let generation = generation.trim();
    let generation_path = Path::new(generation);
    if generation.is_empty()
        || generation_path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
        || generation_path.components().count() != 1
    {
        return Err(CacheResidencyError::MalformedManifest(
            "prompt-cache generation pointer is unsafe".into(),
        ));
    }
    let root = directory
        .join(PROMPT_CACHE_GENERATIONS_DIRECTORY)
        .join(generation_path);
    if !root.is_dir() {
        return Err(CacheResidencyError::MalformedManifest(format!(
            "prompt-cache generation {generation:?} is missing"
        )));
    }
    Ok(root)
}

fn validate_compatibility(
    manifest: &PromptCacheManifest,
    expected: &PromptCacheDescriptor,
    prefix_token_ids: &[u32],
) -> Result<(), CacheResidencyError> {
    macro_rules! require_equal {
        ($field:ident) => {
            if manifest.$field != expected.$field {
                return Err(CacheResidencyError::IncompatiblePromptCache(format!(
                    "{} mismatch",
                    stringify!($field)
                )));
            }
        };
    }
    require_equal!(model_family);
    require_equal!(effective_model_type);
    require_equal!(checkpoint_fingerprint);
    require_equal!(prefix_content_fingerprint);
    require_equal!(architecture_fingerprint);
    require_equal!(layer_count);
    require_equal!(global_layer_start);
    require_equal!(global_layer_end);
    require_equal!(batch_size);
    require_equal!(layer_layout);
    require_equal!(layer_prefix_offsets);
    require_equal!(sink_tokens);
    require_equal!(topology);
    if manifest.total_prefix_tokens != prefix_token_ids.len()
        || manifest.prefix_sha256 != hash_token_ids(prefix_token_ids)
    {
        return Err(CacheResidencyError::PrefixIdentityMismatch);
    }
    Ok(())
}

fn validate_manifest(
    directory: &Path,
    manifest: &PromptCacheManifest,
) -> Result<(), CacheResidencyError> {
    if manifest.schema_version != PROMPT_CACHE_SCHEMA_VERSION {
        return Err(CacheResidencyError::UnsupportedSchema(
            manifest.schema_version,
        ));
    }
    if manifest.prefix_content_fingerprint.is_empty()
        || manifest.block_size_tokens <= 0
        || manifest.layer_count == 0
        || manifest.global_layer_start >= manifest.global_layer_end
        || manifest.global_layer_end > manifest.layer_count
        || manifest.layer_layout.len() != manifest.global_layer_end - manifest.global_layer_start
        || manifest.layer_prefix_offsets.len()
            != manifest.global_layer_end - manifest.global_layer_start
        || manifest.batch_size == 0
        || manifest.batch_size > i32::MAX as usize
        || manifest.total_prefix_tokens == 0
    {
        return Err(CacheResidencyError::MalformedManifest(
            "invalid global cache dimensions".into(),
        ));
    }
    for offset in &manifest.layer_prefix_offsets {
        layer_prefix_tokens(manifest.total_prefix_tokens, *offset)?;
    }
    for (index, policy) in manifest.layer_layout.iter().enumerate() {
        validate_layer_cache_policy(policy).map_err(|error| {
            CacheResidencyError::MalformedManifest(format!(
                "invalid policy for global layer {}: {error}",
                manifest.global_layer_start + index
            ))
        })?;
    }
    for (name, topology) in [
        ("pipeline", manifest.topology.pipeline),
        ("tensor parallel", manifest.topology.tensor_parallel),
        ("expert parallel", manifest.topology.expert_parallel),
    ] {
        if topology.is_some_and(|(world_size, rank)| world_size == 0 || rank >= world_size) {
            return Err(CacheResidencyError::MalformedManifest(format!(
                "invalid {name} topology"
            )));
        }
    }
    let mut by_layer: BTreeMap<(usize, CacheRepresentation), Vec<&PromptCacheBlock>> =
        BTreeMap::new();
    let mut previous_block = None;
    for block in &manifest.blocks {
        if block.global_layer < manifest.global_layer_start
            || block.global_layer >= manifest.global_layer_end
            || block.start < 0
            || block.end <= block.start
            || block.end
                > layer_prefix_tokens(
                    manifest.total_prefix_tokens,
                    manifest.layer_prefix_offsets[block.global_layer - manifest.global_layer_start],
                )? as i64
            || block.logical_bytes == 0
            || block.first_shape.is_empty()
            || block.second_shape.is_empty()
            || !is_sha256_hex(&block.payload_sha256)
        {
            return Err(CacheResidencyError::MalformedManifest(format!(
                "invalid block at layer {} range {}..{}",
                block.global_layer, block.start, block.end
            )));
        }
        let order = (block.global_layer, block.start, block.end);
        if previous_block.is_some_and(|previous| previous >= order) {
            return Err(CacheResidencyError::MalformedManifest(format!(
                "prompt-cache blocks are reordered or duplicated at layer {} range {}..{}",
                block.global_layer, block.start, block.end
            )));
        }
        previous_block = Some(order);
        let policy = manifest
            .layer_layout
            .get(block.global_layer - manifest.global_layer_start)
            .ok_or_else(|| {
                CacheResidencyError::MalformedManifest(format!(
                    "missing cache policy for global layer {}",
                    block.global_layer
                ))
            })?;
        let expected_representation = match policy {
            LayerCachePolicy::NoState | LayerCachePolicy::FixedState { .. } => {
                return Err(CacheResidencyError::MalformedManifest(format!(
                    "stateless global layer {} has an unexpected payload",
                    block.global_layer
                )))
            }
            LayerCachePolicy::KeyValue { .. }
            | LayerCachePolicy::KeyOnly { .. }
            | LayerCachePolicy::KeyValueWithFixedState { .. }
            | LayerCachePolicy::KeyOnlyWithFixedState { .. } => CacheRepresentation::KeyValue,
            LayerCachePolicy::CompressedLatentRotary { .. } => {
                CacheRepresentation::CompressedLatentRotary
            }
        };
        if block.representation != expected_representation {
            return Err(CacheResidencyError::MalformedManifest(format!(
                "global layer {} payload {:?} does not match policy {:?}",
                block.global_layer, block.representation, policy
            )));
        }
        let token_count = i32::try_from(block.end - block.start).map_err(|_| {
            CacheResidencyError::MalformedManifest(format!(
                "global layer {} block token count exceeds runtime dimensions",
                block.global_layer
            ))
        })?;
        let batch = i32::try_from(manifest.batch_size).expect("validated prompt-cache batch");
        let (expected_first_shape, expected_second_shape) = match policy {
            LayerCachePolicy::NoState | LayerCachePolicy::FixedState { .. } => unreachable!(),
            LayerCachePolicy::KeyValue {
                num_key_value_heads,
                head_dim,
                ..
            }
            | LayerCachePolicy::KeyValueWithFixedState {
                num_key_value_heads,
                head_dim,
                ..
            } => {
                let shape = vec![
                    batch,
                    num_key_value_heads.get() as i32,
                    token_count,
                    head_dim.get() as i32,
                ];
                (shape.clone(), shape)
            }
            LayerCachePolicy::KeyOnly {
                num_key_heads,
                head_dim,
                ..
            }
            | LayerCachePolicy::KeyOnlyWithFixedState {
                num_key_heads,
                head_dim,
                ..
            } => (
                vec![
                    batch,
                    num_key_heads.get() as i32,
                    token_count,
                    head_dim.get() as i32,
                ],
                vec![batch, num_key_heads.get() as i32, token_count, 1],
            ),
            LayerCachePolicy::CompressedLatentRotary {
                latent_dim,
                rotary_dim,
                ..
            } => (
                vec![batch, token_count, latent_dim.get() as i32],
                vec![batch, token_count, rotary_dim.get() as i32],
            ),
        };
        if block.first_shape != expected_first_shape || block.second_shape != expected_second_shape
        {
            return Err(CacheResidencyError::MalformedManifest(format!(
                "global layer {} payload geometry {:?}/{:?} does not match policy geometry {:?}/{:?}",
                block.global_layer,
                block.first_shape,
                block.second_shape,
                expected_first_shape,
                expected_second_shape
            )));
        }
        let expected_rank = CacheRankIdentity {
            pipeline_rank: manifest.topology.pipeline.map(|(_, rank)| rank),
            tensor_parallel_rank: manifest.topology.tensor_parallel.map(|(_, rank)| rank),
            expert_parallel_rank: manifest.topology.expert_parallel.map(|(_, rank)| rank),
        };
        let has_rank = expected_rank.pipeline_rank.is_some()
            || expected_rank.tensor_parallel_rank.is_some()
            || expected_rank.expert_parallel_rank.is_some();
        if block.rank != has_rank.then_some(expected_rank) {
            return Err(CacheResidencyError::MalformedManifest(
                "block rank identity does not match the recorded topology".into(),
            ));
        }
        if block.first_dtype != block.second_dtype
            || block.first_shape.first() != Some(&(manifest.batch_size as i32))
            || block.second_shape.first() != Some(&(manifest.batch_size as i32))
        {
            return Err(CacheResidencyError::MalformedManifest(
                "block batch dimension or dtype is inconsistent".into(),
            ));
        }
        let names = array_names(block.representation);
        if block.first_array != names.0 || block.second_array != names.1 {
            return Err(CacheResidencyError::MalformedManifest(
                "block array names do not match its representation".into(),
            ));
        }
        match block.representation {
            CacheRepresentation::KeyValue
                if block.first_shape.len() != 4
                    || block.second_shape.len() != 4
                    || block.first_shape[..3] != block.second_shape[..3] =>
            {
                return Err(CacheResidencyError::MalformedManifest(
                    "key/value blocks must share rank-4 batch, head, and token dimensions".into(),
                ));
            }
            CacheRepresentation::CompressedLatentRotary
                if block.first_shape.len() != 3 || block.second_shape.len() != 3 =>
            {
                return Err(CacheResidencyError::MalformedManifest(
                    "compressed latent/rotary blocks must use rank-3 shapes".into(),
                ));
            }
            _ => {}
        }
        let sequence_axis = match block.representation {
            CacheRepresentation::KeyValue => block.first_shape.len().checked_sub(2),
            CacheRepresentation::CompressedLatentRotary => Some(1),
        }
        .ok_or_else(|| CacheResidencyError::MalformedManifest("invalid block rank".into()))?;
        if block.first_shape.get(sequence_axis) != Some(&((block.end - block.start) as i32))
            || block.second_shape.get(sequence_axis) != Some(&((block.end - block.start) as i32))
        {
            return Err(CacheResidencyError::MalformedManifest(
                "block token range does not match array shapes".into(),
            ));
        }
        let shard = safe_shard_path(directory, &block.shard)?;
        if !shard.is_file() {
            return Err(CacheResidencyError::MissingShard(shard));
        }
        validate_shard_file(&shard, block)?;
        by_layer
            .entry((block.global_layer, block.representation))
            .or_default()
            .push(block);
    }
    let actual_state = manifest
        .state_tensors
        .iter()
        .map(|entry| (entry.owner, entry.role))
        .collect::<BTreeSet<_>>();
    if actual_state.len() != manifest.state_tensors.len() {
        return Err(CacheResidencyError::MalformedManifest(
            "fixed-state tensors contain duplicate owner/role entries".into(),
        ));
    }
    let mut expected_state = Vec::new();
    for (index, layer) in manifest.layer_layout.iter().enumerate() {
        let owner = StateTensorOwner::Layer(manifest.global_layer_start + index);
        let layer_prefix_tokens = layer_prefix_tokens(
            manifest.total_prefix_tokens,
            manifest.layer_prefix_offsets[index],
        )?;
        for policy in layer.fixed_state() {
            if policy.is_required_for(layer_prefix_tokens)
                || actual_state.contains(&(owner, policy.role))
            {
                expected_state.push((owner, policy));
            }
        }
    }
    if manifest.state_tensors.len() != expected_state.len() {
        return Err(CacheResidencyError::MalformedManifest(format!(
            "fixed-state tensor count {} does not match layout count {}",
            manifest.state_tensors.len(),
            expected_state.len()
        )));
    }
    for (entry, (owner, policy)) in manifest.state_tensors.iter().zip(expected_state) {
        if entry.owner != owner || entry.role != policy.role {
            return Err(CacheResidencyError::MalformedManifest(format!(
                "fixed-state tensors are missing, reordered, or unexpected at {:?} {:?}",
                entry.owner, entry.role
            )));
        }
        let layer = match owner {
            StateTensorOwner::Layer(layer) => layer,
        };
        let layer_prefix_tokens = layer_prefix_tokens(
            manifest.total_prefix_tokens,
            manifest.layer_prefix_offsets[layer - manifest.global_layer_start],
        )?;
        let expected_shape = resolved_state_shape(policy, manifest.batch_size, layer_prefix_tokens)
            .map_err(|error| CacheResidencyError::MalformedManifest(error.to_string()))?;
        if entry.shape != expected_shape
            || !state_dtype_matches(policy.dtype, &entry.dtype)
            || entry.logical_bytes == 0
            || !is_sha256_hex(&entry.payload_sha256)
            || entry.array != "state"
        {
            return Err(CacheResidencyError::MalformedManifest(format!(
                "fixed-state tensor {:?} for {:?} does not match its policy",
                entry.role, entry.owner
            )));
        }
        let shard = safe_shard_path(directory, &entry.shard)?;
        if !shard.is_file() {
            return Err(CacheResidencyError::MissingShard(shard));
        }
        validate_state_shard_file(&shard, entry)?;
    }
    for layer in manifest.global_layer_start..manifest.global_layer_end {
        let layer_prefix_tokens = layer_prefix_tokens(
            manifest.total_prefix_tokens,
            manifest.layer_prefix_offsets[layer - manifest.global_layer_start],
        )?;
        let policy = manifest
            .layer_layout
            .get(layer - manifest.global_layer_start)
            .expect("validated prompt-cache layout length");
        let entries = by_layer
            .iter()
            .filter(|((entry_layer, _), _)| *entry_layer == layer)
            .flat_map(|(_, blocks)| blocks.iter().copied())
            .collect::<Vec<_>>();
        if matches!(
            policy,
            LayerCachePolicy::NoState | LayerCachePolicy::FixedState { .. }
        ) {
            if !entries.is_empty() {
                return Err(CacheResidencyError::MalformedManifest(format!(
                    "stateless global layer {layer} has unexpected blocks"
                )));
            }
            continue;
        }
        if entries.is_empty() && layer_prefix_tokens != 0 {
            return Err(CacheResidencyError::MalformedManifest(format!(
                "missing blocks for global layer {layer}"
            )));
        }
        if layer_prefix_tokens == 0 {
            continue;
        }
        let mut entries = entries;
        entries.sort_by_key(|block| block.start);
        let required_start = required_persisted_start(policy, layer_prefix_tokens)?;
        let mut expected_start = entries[0].start;
        if expected_start > required_start
            || (matches!(policy.attention(), Some(AttentionPolicy::Full)) && expected_start != 0)
        {
            return Err(CacheResidencyError::MalformedManifest(format!(
                "global layer {layer} starts at {expected_start}, but its policy requires history from {required_start}"
            )));
        }
        for block in entries {
            if block.start != expected_start {
                return Err(CacheResidencyError::MalformedManifest(format!(
                    "gap or overlap at global layer {layer}: expected {expected_start}, found {}",
                    block.start
                )));
            }
            expected_start = block.end;
        }
        if expected_start != layer_prefix_tokens as i64 {
            return Err(CacheResidencyError::MalformedManifest(format!(
                "global layer {layer} ends at {expected_start}, expected {layer_prefix_tokens}",
            )));
        }
    }
    Ok(())
}

fn validate_complete_prefix(
    records: &[CacheBlockRecord],
    descriptor: &PromptCacheDescriptor,
    prefix_tokens: usize,
) -> Result<(), CacheResidencyError> {
    if prefix_tokens == 0 {
        return Err(CacheResidencyError::MalformedManifest(
            "cannot persist an empty prefix".into(),
        ));
    }
    if records.iter().any(|record| {
        record.id.global_layer < descriptor.global_layer_start
            || record.id.global_layer >= descriptor.global_layer_end
    }) {
        return Err(CacheResidencyError::MalformedManifest(
            "cache contains blocks outside the persisted global layer range".into(),
        ));
    }
    for layer in descriptor.global_layer_start..descriptor.global_layer_end {
        let layer_prefix_tokens = layer_prefix_tokens(
            prefix_tokens,
            descriptor.layer_prefix_offsets[layer - descriptor.global_layer_start],
        )?;
        let policy = descriptor
            .layer_layout
            .get(layer - descriptor.global_layer_start)
            .ok_or_else(|| {
                CacheResidencyError::MalformedManifest(format!(
                    "missing descriptor cache policy for global layer {layer}"
                ))
            })?;
        let mut blocks = records
            .iter()
            .filter(|record| record.id.global_layer == layer)
            .collect::<Vec<_>>();
        if matches!(
            policy,
            LayerCachePolicy::NoState | LayerCachePolicy::FixedState { .. }
        ) {
            if !blocks.is_empty() {
                return Err(CacheResidencyError::MalformedManifest(format!(
                    "stateless global layer {layer} has unexpected cache blocks"
                )));
            }
            continue;
        }
        blocks.sort_by_key(|record| record.id.start);
        if blocks.is_empty() && layer_prefix_tokens != 0 {
            return Err(CacheResidencyError::MalformedManifest(format!(
                "global layer {layer} has no persisted cache blocks"
            )));
        }
        if layer_prefix_tokens == 0 {
            continue;
        }
        let required_start = required_persisted_start(policy, layer_prefix_tokens)?;
        let mut end = blocks[0].id.start;
        if end > required_start
            || (matches!(policy.attention(), Some(AttentionPolicy::Full)) && end != 0)
        {
            return Err(CacheResidencyError::MalformedManifest(format!(
                "global layer {layer} starts at {end}, but its policy requires history from {required_start}"
            )));
        }
        for block in blocks {
            let expected_representation = match policy {
                LayerCachePolicy::NoState | LayerCachePolicy::FixedState { .. } => unreachable!(),
                LayerCachePolicy::KeyValue { .. }
                | LayerCachePolicy::KeyOnly { .. }
                | LayerCachePolicy::KeyValueWithFixedState { .. }
                | LayerCachePolicy::KeyOnlyWithFixedState { .. } => CacheRepresentation::KeyValue,
                LayerCachePolicy::CompressedLatentRotary { .. } => {
                    CacheRepresentation::CompressedLatentRotary
                }
            };
            if block.id.representation != expected_representation {
                return Err(CacheResidencyError::MalformedManifest(format!(
                    "global layer {layer} cache representation does not match its policy"
                )));
            }
            let token_count = i32::try_from(block.id.end - block.id.start).map_err(|_| {
                CacheResidencyError::MalformedManifest(format!(
                    "global layer {layer} block token count exceeds runtime dimensions"
                ))
            })?;
            let batch = descriptor.batch_size as i32;
            let (first_shape, second_shape) = match policy {
                LayerCachePolicy::NoState | LayerCachePolicy::FixedState { .. } => unreachable!(),
                LayerCachePolicy::KeyValue {
                    num_key_value_heads,
                    head_dim,
                    ..
                }
                | LayerCachePolicy::KeyValueWithFixedState {
                    num_key_value_heads,
                    head_dim,
                    ..
                } => {
                    let shape = vec![
                        batch,
                        num_key_value_heads.get() as i32,
                        token_count,
                        head_dim.get() as i32,
                    ];
                    (shape.clone(), shape)
                }
                LayerCachePolicy::KeyOnly {
                    num_key_heads,
                    head_dim,
                    ..
                }
                | LayerCachePolicy::KeyOnlyWithFixedState {
                    num_key_heads,
                    head_dim,
                    ..
                } => (
                    vec![
                        batch,
                        num_key_heads.get() as i32,
                        token_count,
                        head_dim.get() as i32,
                    ],
                    vec![batch, num_key_heads.get() as i32, token_count, 1],
                ),
                LayerCachePolicy::CompressedLatentRotary {
                    latent_dim,
                    rotary_dim,
                    ..
                } => (
                    vec![batch, token_count, latent_dim.get() as i32],
                    vec![batch, token_count, rotary_dim.get() as i32],
                ),
            };
            if block.shapes != [first_shape, second_shape] {
                return Err(CacheResidencyError::MalformedManifest(format!(
                    "global layer {layer} live cache geometry does not match its persistence policy"
                )));
            }
            if block.id.start != end {
                return Err(CacheResidencyError::MalformedManifest(format!(
                    "global layer {layer} has a gap or overlap at {end}"
                )));
            }
            end = block.id.end;
        }
        if end != layer_prefix_tokens as i64 {
            return Err(CacheResidencyError::MalformedManifest(format!(
                "global layer {layer} contains {end} tokens, expected {layer_prefix_tokens}"
            )));
        }
    }
    Ok(())
}

fn required_persisted_start(
    policy: &LayerCachePolicy,
    total_prefix_tokens: usize,
) -> Result<i64, CacheResidencyError> {
    let total = i64::try_from(total_prefix_tokens).map_err(|_| {
        CacheResidencyError::MalformedManifest(
            "prompt-cache prefix length exceeds the runtime position range".into(),
        )
    })?;
    match policy.attention() {
        None | Some(AttentionPolicy::Full) => Ok(0),
        Some(AttentionPolicy::Sliding { window }) => {
            let retained_past = i64::from(window.get() - 1);
            Ok((total - retained_past).max(0))
        }
    }
}

fn layer_prefix_tokens(
    total_prefix_tokens: usize,
    offset: i32,
) -> Result<usize, CacheResidencyError> {
    if offset > 0 {
        return Err(CacheResidencyError::MalformedManifest(
            "layer prefix offsets must not advance beyond the persisted prefix".into(),
        ));
    }
    total_prefix_tokens.checked_sub(offset.unsigned_abs() as usize).ok_or_else(|| {
        CacheResidencyError::MalformedManifest(format!(
            "layer prefix offset {offset} precedes the start of a {total_prefix_tokens}-token prefix"
        ))
    })
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
                    id: record.id.clone(),
                    device_bytes: record.bytes,
                    host_bytes: ticket.reserved_host_bytes,
                },
            ));
            tickets.push(PendingCacheOperation::HostDemotion(ticket));
        }
        if let Some(pending) = record.pending_disk().cloned() {
            if pending.ticket.cancel() {
                state.counters.report.cancellations += 1;
            }
            tickets.push(PendingCacheOperation::Disk(pending.ticket.clone()));
            if pending.ticket.key.kind != DiskOperationKind::Write {
                if let CacheBlockPhysicalState::Disk {
                    read:
                        DiskCacheReadState::Reading {
                            reserved_host_bytes,
                            ..
                        },
                    ..
                } = &record.physical
                {
                    read_reservations.push((
                        pending.ticket.key.clone(),
                        (record.id.global_layer, *reserved_host_bytes),
                    ));
                }
                record.physical.clear_pending(&pending.ticket.key);
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
    let report = &mut state.counters.report;
    report.key_value_blocks = 0;
    report.compressed_latent_blocks = 0;
    report.device_blocks = 0;
    report.host_blocks = 0;
    report.disk_blocks = 0;
    report.current_device_bytes = state.tails.values().map(|tail| tail.0).sum();
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
    report.per_layer.clear();
    report.per_layer_overflow_layers = 0;
    report.per_layer_overflow = CacheLayerResidencyStats::default();
    let mut per_layer = BTreeMap::<usize, CacheLayerResidencyStats>::new();
    let mut layer_ends: HashMap<usize, i64> = HashMap::new();
    for (layer, (bytes, end)) in &state.tails {
        layer_ends.insert(*layer, *end);
        let layer_report = per_layer.entry(*layer).or_default();
        layer_report.current_device_bytes += *bytes;
        layer_report.mutable_tail_bytes += *bytes;
        layer_report.logical_cached_tokens = (*end).max(0) as u64;
    }
    for record in state.blocks.values() {
        let layer_report = per_layer.entry(record.id.global_layer).or_default();
        match record.id.representation {
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
            .is_some_and(|pending| pending.ticket.key.kind == DiskOperationKind::Write);
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
        match &record.physical {
            CacheBlockPhysicalState::Device { .. } => {
                report.device_blocks += 1;
                report.current_device_bytes += record.bytes;
                layer_report.device_blocks += 1;
                layer_report.current_device_bytes += record.bytes;
            }
            CacheBlockPhysicalState::Demoting { ticket, .. } => {
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
            CacheBlockPhysicalState::Host { block, .. } => {
                let host_bytes = block
                    .capacity()
                    .expect("validated host cache block has inspectable capacity");
                report.host_blocks += 1;
                report.current_host_bytes += host_bytes;
                layer_report.host_blocks += 1;
                layer_report.current_host_bytes += host_bytes;
            }
            CacheBlockPhysicalState::Disk { read, .. } => {
                report.disk_blocks += 1;
                report.current_disk_bytes += record.bytes;
                layer_report.disk_blocks += 1;
                layer_report.current_disk_bytes += record.bytes;
                if let DiskCacheReadState::Reading {
                    reserved_host_bytes,
                    ..
                } = read
                {
                    report.current_host_bytes += reserved_host_bytes;
                    layer_report.current_host_bytes += reserved_host_bytes;
                }
            }
        }
        if record.protected_prefix {
            report.protected_prefix_blocks += 1;
            layer_report.protected_prefix_blocks += 1;
        }
        layer_report.logical_cached_tokens = layer_report
            .logical_cached_tokens
            .max(record.id.end.max(0) as u64);
        layer_ends
            .entry(record.id.global_layer)
            .and_modify(|end| *end = (*end).max(record.id.end))
            .or_insert(record.id.end);
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
            matches!(
                &record.physical,
                CacheBlockPhysicalState::Disk {
                    read: DiskCacheReadState::Reading { pending, .. },
                    ..
                } if pending.ticket.key == *key
            )
        });
        if !covered_by_pending_record {
            report.current_host_bytes += reserved_host_bytes;
            per_layer
                .entry(*global_layer)
                .or_default()
                .current_host_bytes += reserved_host_bytes;
        }
    }
    let mut device_starts = HashMap::<usize, Vec<i64>>::new();
    for record in state
        .blocks
        .values()
        .filter(|record| record.tier() == CacheTier::Device && !record.protected_prefix)
    {
        device_starts
            .entry(record.id.global_layer)
            .or_default()
            .push(record.id.start);
    }
    report.protected_recent_blocks = device_starts
        .values()
        .map(|starts| starts.len().min(state.recent_device_blocks) as u64)
        .sum();
    for (layer, starts) in &device_starts {
        per_layer.entry(*layer).or_default().protected_recent_blocks =
            starts.len().min(state.recent_device_blocks) as u64;
    }
    report.logical_cached_tokens = layer_ends.values().copied().max().unwrap_or(0).max(0) as u64;
    // Historical counters keep the first observed layer identities stable.
    // Fill any remaining bounded slots with currently active layers, and fold
    // both current and historical activity for all other layers into overflow.
    let mut selected_layers = state.layer_activity.keys().copied().collect::<Vec<_>>();
    for global_layer in per_layer.keys().copied() {
        if selected_layers.len() == CACHE_RESIDENCY_LAYER_REPORT_LIMIT {
            break;
        }
        if !state.layer_activity.contains_key(&global_layer) {
            selected_layers.push(global_layer);
        }
    }
    selected_layers.sort_unstable();
    for global_layer in selected_layers {
        let mut stats = per_layer.remove(&global_layer).unwrap_or_default();
        if let Some(activity) = state.layer_activity.get(&global_layer) {
            activity.apply_to(&mut stats);
        }
        report.per_layer.push(CacheLayerResidencyReport {
            global_layer,
            stats,
        });
    }
    for (_, stats) in per_layer {
        report.per_layer_overflow_layers += 1;
        report.per_layer_overflow.accumulate(&stats);
    }
    state
        .layer_activity_overflow
        .apply_to(&mut report.per_layer_overflow);
    if report.current_device_bytes <= device_budget_bytes {
        report.peak_device_bytes = report.peak_device_bytes.max(report.current_device_bytes);
    }
    if report.current_host_bytes <= host_budget_bytes {
        report.peak_host_bytes = report.peak_host_bytes.max(report.current_host_bytes);
    }
    if disk_budget_bytes.is_none_or(|budget| report.current_disk_bytes <= budget) {
        report.peak_disk_bytes = report.peak_disk_bytes.max(report.current_disk_bytes);
    }
    report.peak_in_flight_write_bytes = report
        .peak_in_flight_write_bytes
        .max(report.in_flight_write_bytes);
    report.peak_in_flight_host_demotion_bytes = report
        .peak_in_flight_host_demotion_bytes
        .max(report.in_flight_host_demotion_bytes);
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
    let mut recent = HashMap::<usize, Vec<i64>>::new();
    if tier == CacheTier::Device && recent_per_layer > 0 {
        for record in state
            .blocks
            .values()
            .filter(|record| matches!(record.physical, CacheBlockPhysicalState::Device { .. }))
        {
            recent
                .entry(record.id.global_layer)
                .or_default()
                .push(record.id.start);
        }
        for starts in recent.values_mut() {
            starts.sort_unstable_by(|a, b| b.cmp(a));
            starts.truncate(recent_per_layer);
        }
    }
    state
        .blocks
        .values()
        .filter(|record| {
            physical_state_is_tier_candidate(&record.physical, tier)
                && record.leases == 0
                && record.pending_disk().is_none()
                && required != Some(&record.id)
                && !record.protected_prefix
                && !recent
                    .get(&record.id.global_layer)
                    .is_some_and(|starts| starts.contains(&record.id.start))
        })
        .min_by_key(|record| match policy {
            CacheEvictionPolicy::LeastRecentlyUsed => {
                (record.last_access, record.access_count, record.id.clone())
            }
            CacheEvictionPolicy::LeastFrequentlyUsed => {
                (record.access_count, record.last_access, record.id.clone())
            }
        })
        .map(|record| record.id.clone())
}

fn physical_state_is_tier_candidate(physical: &CacheBlockPhysicalState, tier: CacheTier) -> bool {
    matches!(
        (physical, tier),
        (CacheBlockPhysicalState::Device { .. }, CacheTier::Device)
            | (CacheBlockPhysicalState::Host { .. }, CacheTier::Host)
            | (CacheBlockPhysicalState::Disk { .. }, CacheTier::Disk)
    )
}

fn write_live_block(
    directory: &Path,
    id: &CacheBlockId,
    block: &HostCacheBlock,
) -> Result<DiskLocation, CacheResidencyError> {
    let (path, temporary) = live_block_paths(directory, id);
    let mut temporary_guard = TemporaryFileGuard::new(temporary);
    save_host_cache_block(temporary_guard.path(), block)?;
    sync_file(temporary_guard.path())?;
    publish_live_block_file(temporary_guard.path(), &path)?;
    temporary_guard.disarm();
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

fn publish_live_block_file(
    temporary: &Path,
    destination: &Path,
) -> Result<(), CacheResidencyError> {
    // A hard-link publication is atomic and fails if a destination somehow
    // collides, whereas rename would silently replace another process's shard.
    fs::hard_link(temporary, destination).map_err(|source| CacheResidencyError::Io {
        action: "publish uniquely named live cache block",
        path: destination.to_path_buf(),
        source,
    })?;
    if let Err(source) = fs::remove_file(temporary) {
        let _ = fs::remove_file(destination);
        return Err(CacheResidencyError::Io {
            action: "remove published live cache temporary file",
            path: temporary.to_path_buf(),
            source,
        });
    }
    Ok(())
}

fn live_block_paths(directory: &Path, id: &CacheBlockId) -> (PathBuf, PathBuf) {
    let process_namespace = LIVE_PROCESS_NAMESPACE.get_or_init(|| {
        let started = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        format!("p{:08x}-t{started:032x}", std::process::id())
    });
    let write_id = NEXT_LIVE_SHARD_ID.fetch_add(1, Ordering::Relaxed);
    let representation = match id.representation {
        CacheRepresentation::KeyValue => "kv",
        CacheRepresentation::CompressedLatentRotary => "mla",
    };
    let rank_component =
        |rank: Option<usize>| rank.map_or_else(|| "x".to_string(), |rank| rank.to_string());
    let rank = id.rank.map_or_else(
        || "rank-px-tx-ex".to_string(),
        |rank| {
            format!(
                "rank-p{}-t{}-e{}",
                rank_component(rank.pipeline_rank),
                rank_component(rank.tensor_parallel_rank),
                rank_component(rank.expert_parallel_rank)
            )
        },
    );
    let base = format!(
        "live-{process_namespace}-w{write_id:016x}-s{:016x}-layer-{:05}-{representation}-{rank}-{}-{}",
        id.session_id, id.global_layer, id.start, id.end
    );
    (
        directory.join(format!("{base}.safetensors")),
        directory.join(format!(".{base}.tmp.safetensors")),
    )
}

struct TemporaryFileGuard {
    path: PathBuf,
    armed: bool,
}

impl TemporaryFileGuard {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TemporaryFileGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
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

fn hash_shard_payload(path: &Path) -> Result<String, CacheResidencyError> {
    let (_, _, data_start) = read_shard_metadata(path)?;
    let mut file = File::open(path).map_err(|source| CacheResidencyError::Io {
        action: "open prompt cache shard payload",
        path: path.to_path_buf(),
        source,
    })?;
    file.seek(SeekFrom::Start(data_start))
        .map_err(|source| CacheResidencyError::Io {
            action: "seek prompt cache shard payload",
            path: path.to_path_buf(),
            source,
        })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| CacheResidencyError::Io {
                action: "hash prompt cache shard payload",
                path: path.to_path_buf(),
                source,
            })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(sha256_hex(hasher.finalize()))
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
            hash_shard_payload(&location.path).map_err(|error| error.to_string())?
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

fn hash_token_ids(tokens: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    sha256_hex(hasher.finalize())
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

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn safe_shard_path(directory: &Path, relative: &str) -> Result<PathBuf, CacheResidencyError> {
    let path = Path::new(relative);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(CacheResidencyError::UnsafeShardPath(relative.into()));
    }
    let joined = directory.join(path);
    if joined.exists() {
        let root = fs::canonicalize(directory).map_err(|source| CacheResidencyError::Io {
            action: "canonicalize prompt cache directory",
            path: directory.to_path_buf(),
            source,
        })?;
        let canonical = fs::canonicalize(&joined).map_err(|source| CacheResidencyError::Io {
            action: "canonicalize prompt cache shard",
            path: joined.clone(),
            source,
        })?;
        if !canonical.starts_with(&root) {
            return Err(CacheResidencyError::UnsafeShardPath(relative.into()));
        }
    }
    Ok(joined)
}

fn validate_shard_file(path: &Path, block: &PromptCacheBlock) -> Result<(), CacheResidencyError> {
    let (metadata, file_len, data_start) = read_shard_metadata(path)?;
    let entries = metadata.tensors();
    if entries.len() != 2 {
        return Err(CacheResidencyError::MalformedShard {
            path: path.to_path_buf(),
            reason: format!("expected two arrays, found {}", entries.len()),
        });
    }
    let mut logical_bytes = 0u64;
    for (name, expected_shape, expected_dtype) in [
        (&block.first_array, &block.first_shape, &block.first_dtype),
        (
            &block.second_array,
            &block.second_shape,
            &block.second_dtype,
        ),
    ] {
        let tensor = metadata
            .info(name)
            .ok_or_else(|| CacheResidencyError::MalformedShard {
                path: path.to_path_buf(),
                reason: format!("missing array {name}"),
            })?;
        let shape = tensor
            .shape
            .iter()
            .map(|dimension| i32::try_from(*dimension))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| CacheResidencyError::MalformedShard {
                path: path.to_path_buf(),
                reason: "array dimension exceeds runtime range".into(),
            })?;
        if &shape != expected_shape || stored_dtype_name(tensor.dtype) != *expected_dtype {
            return Err(CacheResidencyError::MalformedShard {
                path: path.to_path_buf(),
                reason: format!("array {name} shape or dtype does not match the manifest"),
            });
        }
        logical_bytes = logical_bytes.saturating_add(
            u64::try_from(tensor.data_offsets.1.saturating_sub(tensor.data_offsets.0))
                .unwrap_or(u64::MAX),
        );
    }
    if logical_bytes != block.logical_bytes {
        return Err(CacheResidencyError::MalformedShard {
            path: path.to_path_buf(),
            reason: format!(
                "logical byte count {logical_bytes} does not match manifest value {}",
                block.logical_bytes
            ),
        });
    }
    let expected_file_len = data_start
        .checked_add(metadata.data_len() as u64)
        .ok_or_else(|| CacheResidencyError::MalformedShard {
            path: path.to_path_buf(),
            reason: "safetensors file length overflow".into(),
        })?;
    if expected_file_len != file_len {
        return Err(CacheResidencyError::MalformedShard {
            path: path.to_path_buf(),
            reason: format!(
                "safetensors payload boundary {expected_file_len} does not match file length {file_len}"
            ),
        });
    }
    Ok(())
}

fn validate_state_shard_file(
    path: &Path,
    state: &PromptCacheStateTensor,
) -> Result<(), CacheResidencyError> {
    let (metadata, file_len, data_start) = read_shard_metadata(path)?;
    let entries = metadata.tensors();
    if entries.len() != 1 {
        return Err(CacheResidencyError::MalformedShard {
            path: path.to_path_buf(),
            reason: format!("expected one state array, found {}", entries.len()),
        });
    }
    let tensor =
        metadata
            .info(&state.array)
            .ok_or_else(|| CacheResidencyError::MalformedShard {
                path: path.to_path_buf(),
                reason: format!("missing state array {}", state.array),
            })?;
    let shape = tensor
        .shape
        .iter()
        .map(|dimension| i32::try_from(*dimension))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| CacheResidencyError::MalformedShard {
            path: path.to_path_buf(),
            reason: "state array dimension exceeds runtime range".into(),
        })?;
    let logical_bytes = u64::try_from(tensor.data_offsets.1.saturating_sub(tensor.data_offsets.0))
        .unwrap_or(u64::MAX);
    if shape != state.shape
        || stored_dtype_name(tensor.dtype) != state.dtype
        || logical_bytes != state.logical_bytes
    {
        return Err(CacheResidencyError::MalformedShard {
            path: path.to_path_buf(),
            reason: "state array shape, dtype, or byte count does not match the manifest".into(),
        });
    }
    let expected_file_len = data_start
        .checked_add(metadata.data_len() as u64)
        .ok_or_else(|| CacheResidencyError::MalformedShard {
            path: path.to_path_buf(),
            reason: "safetensors file length overflow".into(),
        })?;
    if expected_file_len != file_len {
        return Err(CacheResidencyError::MalformedShard {
            path: path.to_path_buf(),
            reason: "state safetensors payload boundary does not match file length".into(),
        });
    }
    Ok(())
}

fn read_shard_metadata(
    path: &Path,
) -> Result<(safetensors::tensor::Metadata, u64, u64), CacheResidencyError> {
    let mut file = File::open(path).map_err(|source| CacheResidencyError::Io {
        action: "open prompt cache shard metadata",
        path: path.to_path_buf(),
        source,
    })?;
    let file_len = file
        .metadata()
        .map_err(|source| CacheResidencyError::Io {
            action: "stat prompt cache shard",
            path: path.to_path_buf(),
            source,
        })?
        .len();
    let mut length_bytes = [0u8; 8];
    file.read_exact(&mut length_bytes)
        .map_err(|source| CacheResidencyError::Io {
            action: "read prompt cache shard header length",
            path: path.to_path_buf(),
            source,
        })?;
    let header_len = u64::from_le_bytes(length_bytes);
    if header_len == 0 || header_len > MAX_PROMPT_CACHE_SHARD_HEADER_BYTES {
        return Err(CacheResidencyError::MalformedShard {
            path: path.to_path_buf(),
            reason: format!(
                "safetensors header length {header_len} exceeds the prompt-cache bound"
            ),
        });
    }
    let data_start =
        8u64.checked_add(header_len)
            .ok_or_else(|| CacheResidencyError::MalformedShard {
                path: path.to_path_buf(),
                reason: "safetensors header length overflow".into(),
            })?;
    if data_start > file_len {
        return Err(CacheResidencyError::MalformedShard {
            path: path.to_path_buf(),
            reason: "safetensors header extends beyond the file".into(),
        });
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)
        .map_err(|source| CacheResidencyError::Io {
            action: "read prompt cache shard header",
            path: path.to_path_buf(),
            source,
        })?;
    let metadata =
        serde_json::from_slice::<safetensors::tensor::Metadata>(&header).map_err(|error| {
            CacheResidencyError::MalformedShard {
                path: path.to_path_buf(),
                reason: error.to_string(),
            }
        })?;
    Ok((metadata, file_len, data_start))
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

fn publish_prompt_cache_generation(
    destination: &Path,
    generation_name: &str,
    nonce: u128,
) -> Result<(), CacheResidencyError> {
    let temporary = destination.join(format!(".{PROMPT_CACHE_CURRENT_FILE}.tmp-{nonce}"));
    let current = destination.join(PROMPT_CACHE_CURRENT_FILE);
    let mut file = File::create(&temporary).map_err(|source| CacheResidencyError::Io {
        action: "create prompt cache generation pointer",
        path: temporary.clone(),
        source,
    })?;
    writeln!(file, "{generation_name}").map_err(|source| CacheResidencyError::Io {
        action: "write prompt cache generation pointer",
        path: temporary.clone(),
        source,
    })?;
    file.sync_all().map_err(|source| CacheResidencyError::Io {
        action: "sync prompt cache generation pointer",
        path: temporary.clone(),
        source,
    })?;
    durable_rename(&temporary, &current, true).map_err(|source| CacheResidencyError::Io {
        action: "switch prompt cache generation",
        path: current,
        source,
    })?;
    sync_directory(destination)
}

fn stored_dtype_name(dtype: safetensors::Dtype) -> String {
    use safetensors::Dtype as Stored;
    match dtype {
        Stored::BOOL => "Bool",
        Stored::U8 => "Uint8",
        Stored::U16 => "Uint16",
        Stored::U32 => "Uint32",
        Stored::U64 => "Uint64",
        Stored::I8 => "Int8",
        Stored::I16 => "Int16",
        Stored::I32 => "Int32",
        Stored::I64 => "Int64",
        Stored::F16 => "Float16",
        Stored::BF16 => "Bfloat16",
        Stored::F32 => "Float32",
        Stored::F64 => "Float64",
        dtype => return format!("{dtype:?}"),
    }
    .into()
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
fn sync_directory(path: &Path) -> Result<(), CacheResidencyError> {
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|source| CacheResidencyError::Io {
            action: "synchronize cache directory",
            path: path.to_path_buf(),
            source,
        })
}

#[cfg(windows)]
fn sync_directory(path: &Path) -> Result<(), CacheResidencyError> {
    // Windows has no POSIX-style directory fsync. Prompt-cache metadata is
    // published with MOVEFILE_WRITE_THROUGH in `durable_rename`; validate the
    // expected directory here without trying to open it as an ordinary file.
    if path.is_dir() {
        Ok(())
    } else {
        Err(CacheResidencyError::Io {
            action: "validate cache directory before durable publication",
            path: path.to_path_buf(),
            source: std::io::Error::new(
                std::io::ErrorKind::NotADirectory,
                "cache publication path is not a directory",
            ),
        })
    }
}

#[cfg(not(any(unix, windows)))]
fn sync_directory(path: &Path) -> Result<(), CacheResidencyError> {
    if path.is_dir() {
        Ok(())
    } else {
        Err(CacheResidencyError::Io {
            action: "validate cache directory before publication",
            path: path.to_path_buf(),
            source: std::io::Error::new(
                std::io::ErrorKind::NotADirectory,
                "cache publication path is not a directory",
            ),
        })
    }
}

#[cfg(not(windows))]
fn durable_rename(source: &Path, destination: &Path, _replace: bool) -> std::io::Result<()> {
    fs::rename(source, destination)
}

#[cfg(windows)]
fn durable_rename(source: &Path, destination: &Path, replace: bool) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    fn wide_path(path: &Path) -> std::io::Result<Vec<u16>> {
        let mut wide = path.as_os_str().encode_wide().collect::<Vec<_>>();
        if wide.contains(&0) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Windows cache publication path contains an embedded NUL",
            ));
        }
        wide.push(0);
        Ok(wide)
    }

    let source = wide_path(source)?;
    let destination = wide_path(destination)?;
    let mut flags = MOVEFILE_WRITE_THROUGH;
    if replace {
        flags |= MOVEFILE_REPLACE_EXISTING;
    }
    // SAFETY: both UTF-16 buffers are NUL-terminated and remain alive for the
    // duration of the call. The source and destination are sibling paths, so
    // publication cannot become a copy across volumes.
    let moved = unsafe { MoveFileExW(source.as_ptr(), destination.as_ptr(), flags) };
    if moved == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
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
    /// A stable block identity was published more than once.
    #[error("duplicate cache block {0:?}")]
    DuplicateBlock(CacheBlockId),
    /// A requested block was not cataloged.
    #[error("missing cache block {0:?}")]
    MissingBlock(CacheBlockId),
    /// A disk-backed block had no safe location.
    #[error("cache block has no disk location: {0:?}")]
    MissingDiskLocation(CacheBlockId),
    /// A host or device block had no evaluated arrays.
    #[error("cache block has no resident arrays: {0:?}")]
    MissingResidentArrays(CacheBlockId),
    /// Active attention prevented mutation or eviction.
    #[error("cache block is leased by active attention: {0:?}")]
    BlockLeased(CacheBlockId),
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
    /// Aggregate process-pool capacity could not admit an operation.
    #[error(
        "process cache pool {resource:?} budget exceeded: requires {required} bytes, budget is {budget}"
    )]
    PoolBudgetExceeded {
        /// Aggregate resource that could not admit the operation.
        resource: CachePoolResource,
        /// Aggregate bytes required by the operation.
        required: u64,
        /// Configured aggregate budget.
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
    /// The aggregate process-pool lock was poisoned by a panic.
    #[error("cache residency process-pool lock was poisoned")]
    PoolPoisoned,
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
    /// A manifest could not be encoded or decoded.
    #[error("invalid prompt cache manifest JSON: {0}")]
    ManifestJson(#[source] serde_json::Error),
    /// A manifest did not satisfy structural invariants.
    #[error("malformed prompt cache manifest: {0}")]
    MalformedManifest(String),
    /// The manifest schema is not supported by this runtime.
    #[error("unsupported prompt cache schema version {0}")]
    UnsupportedSchema(u32),
    /// A shard path could escape the prompt-cache directory.
    #[error("unsafe prompt cache shard path {0:?}")]
    UnsafeShardPath(String),
    /// A manifest referenced a missing shard.
    #[error("missing prompt cache shard {0}")]
    MissingShard(PathBuf),
    /// A safetensors block had missing, extra, or corrupt arrays.
    #[error("malformed prompt cache shard {path}: {reason}")]
    MalformedShard {
        /// Invalid shard path.
        path: PathBuf,
        /// Structural or data validation failure.
        reason: String,
    },
    /// The supplied model or topology differs from the producer.
    #[error("incompatible prompt cache: {0}")]
    IncompatiblePromptCache(String),
    /// Caller-provided prefix ids did not match the persisted prefix.
    #[error("prompt cache prefix token identity does not match")]
    PrefixIdentityMismatch,
    /// The target path cannot be published atomically.
    #[error("invalid prompt cache path {0}")]
    InvalidPromptCachePath(PathBuf),
    /// The destination exists and explicit replacement was not requested.
    #[error("prompt cache destination already exists: {0}")]
    PromptCacheExists(PathBuf),
}

#[cfg(test)]
mod tests {
    use super::{
        cpu_stream, durable_rename, hash_shard_payload, hash_token_ids,
        host_cache_capacity_upper_bound, inspect_prompt_cache, live_block_paths,
        map_prompt_cache_shard, open_prompt_cache, publish_live_block_file,
        publish_prompt_cache_generation, safe_shard_path, validate_prompt_cache_model_identity,
        verify_disk_payload, AttentionPolicy, CacheBlockArrays, CacheBlockId,
        CacheBlockPhysicalState, CacheBlockRecord, CacheLayerResidencyStats, CachePoolLimits,
        CachePoolResource, CacheRankIdentity, CacheRepresentation, CacheResidencyError,
        CacheResidencyManager, CacheResidencyPool, CacheTier, DiskCacheReadState, DiskLocation,
        DiskOperationKey, DiskOperationKind, DiskResult, DiskTask, DiskWorker, DiskWriteCommit,
        HostCacheBlock, HostCachePersistence, HostDemotionCompletion, HostDemotionTicket,
        HostWriteReservation, LayerCachePolicy, LayerSchedule, MutableStateResidency,
        PagedCacheOptions, PendingDiskOperation, PromptCacheBlock, PromptCacheDescriptor,
        PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions, PromptCacheStateTensor,
        PromptCacheTopology, StateTensorDimension, StateTensorDtype, StateTensorOwner,
        StateTensorPolicy, StateTensorRole, TemporaryFileGuard, CACHE_RESIDENCY_LAYER_REPORT_LIMIT,
        MAX_PROMPT_CACHE_SHARD_HEADER_BYTES, PROMPT_CACHE_GENERATIONS_DIRECTORY,
        PROMPT_CACHE_SCHEMA_VERSION,
    };
    use safemlx::{
        host_transfer_capacity_upper_bound, transforms::async_eval_with_event, Array, Device,
        DeviceType, HostTransferPolicy, HostTransferStorageKind, Stream,
    };
    use safetensors::tensor::{serialize_to_file, Dtype as StoredDtype, TensorView};
    use std::{
        fs,
        fs::OpenOptions,
        hash::{DefaultHasher, Hash, Hasher},
        io::Write as _,
        path::Path,
        sync::{mpsc, Arc, Barrier, OnceLock},
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
        manager.lock().unwrap().blocks.insert(
            id.clone(),
            CacheBlockRecord {
                id,
                physical: CacheBlockPhysicalState::Host {
                    block: test_host_block(),
                    persistence: HostCachePersistence::Unbacked,
                },
                bytes: 0,
                shapes: [vec![1, 1, 1, 1], vec![1, 1, 1, 1]],
                dtypes: ["Float32".into(), "Float32".into()],
                imported: false,
                leases: 1,
                access_count: 0,
                last_access: 0,
                protected_prefix: false,
            },
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
            CacheResidencyError::PoolBudgetExceeded {
                resource: CachePoolResource::Device,
                required: 24,
                budget: 20,
            }
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
    fn process_pool_bounds_and_releases_transient_transfer_capacity() {
        let pool = CacheResidencyPool::new(CachePoolLimits::new(64, 64, 10, 0).unwrap());
        let first = pool.reserve_transfer(6).unwrap();
        let error = pool.reserve_transfer(5).unwrap_err();
        assert!(matches!(
            error,
            CacheResidencyError::PoolBudgetExceeded {
                resource: CachePoolResource::TransferInFlight,
                required: 11,
                budget: 10,
            }
        ));
        assert_eq!(pool.report().unwrap().current_transfer_in_flight_bytes, 6);
        drop(first);
        assert_eq!(pool.report().unwrap().current_transfer_in_flight_bytes, 0);
        assert_eq!(pool.report().unwrap().peak_transfer_in_flight_bytes, 6);
    }

    #[test]
    fn process_pool_enforces_aggregate_host_and_disk_reservations() {
        let pool = CacheResidencyPool::new(CachePoolLimits::new(64, 10, 64, 10).unwrap());
        pool.update_manager(
            7,
            super::CachePoolUsage {
                host_bytes: 8,
                disk_bytes: 8,
                ..super::CachePoolUsage::default()
            },
        )
        .unwrap();
        for (usage, expected) in [
            (
                super::CachePoolUsage {
                    host_bytes: 3,
                    ..super::CachePoolUsage::default()
                },
                CachePoolResource::Host,
            ),
            (
                super::CachePoolUsage {
                    disk_bytes: 3,
                    ..super::CachePoolUsage::default()
                },
                CachePoolResource::Disk,
            ),
        ] {
            assert!(matches!(
                pool.reserve_additional(usage),
                Err(CacheResidencyError::PoolBudgetExceeded {
                    resource,
                    required: 11,
                    budget: 10,
                }) if resource == expected
            ));
        }
        let report = pool.report().unwrap();
        assert_eq!(report.current_host_bytes, 8);
        assert_eq!(report.current_disk_bytes, 8);
        pool.remove_manager(7);
        assert_eq!(pool.report().unwrap().current_host_bytes, 0);
        assert_eq!(pool.report().unwrap().current_disk_bytes, 0);
    }

    #[test]
    fn process_pool_admission_is_atomic_across_concurrent_managers() {
        let pool = CacheResidencyPool::new(CachePoolLimits::new(64, 10, 64, 0).unwrap());
        let start = Arc::new(Barrier::new(3));
        let finish = Arc::new(Barrier::new(3));
        let (sender, receiver) = mpsc::channel();
        let handles = (0..2)
            .map(|_| {
                let pool = pool.clone();
                let start = Arc::clone(&start);
                let finish = Arc::clone(&finish);
                let sender = sender.clone();
                thread::spawn(move || {
                    start.wait();
                    let admission = pool.reserve_additional(super::CachePoolUsage {
                        host_bytes: 6,
                        ..super::CachePoolUsage::default()
                    });
                    sender.send(admission.is_ok()).unwrap();
                    finish.wait();
                    drop(admission);
                })
            })
            .collect::<Vec<_>>();
        drop(sender);
        start.wait();
        let admitted = [receiver.recv().unwrap(), receiver.recv().unwrap()];
        assert_eq!(admitted.into_iter().filter(|admitted| *admitted).count(), 1);
        assert_eq!(pool.report().unwrap().current_host_bytes, 6);
        finish.wait();
        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(pool.report().unwrap().current_host_bytes, 0);
    }

    #[test]
    fn per_cache_limits_cannot_exceed_their_process_pool() {
        let pool = CacheResidencyPool::new(CachePoolLimits::new(8, 4, 8, 0).unwrap());
        let error = PagedCacheOptions::new(1, 16, 4, 1)
            .unwrap()
            .with_pool(pool)
            .unwrap_err();
        assert!(matches!(error, CacheResidencyError::InvalidOptions(_)));
        assert!(error.to_string().contains("per-cache device budget 16"));
    }

    fn prompt_descriptor() -> PromptCacheDescriptor {
        PromptCacheDescriptor {
            model_family: "llama".into(),
            effective_model_type: "llama".into(),
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
            topology: descriptor.topology,
            layer_layout: descriptor.layer_layout,
        }
    }

    fn write_prompt_fixture(root: &Path, namespace: &str) -> PromptCacheManifest {
        fs::create_dir_all(root).unwrap();
        let keys = 1.0f32.to_le_bytes();
        let values = 2.0f32.to_le_bytes();
        let key_view = TensorView::new(StoredDtype::F32, vec![1, 1, 1, 1], &keys).unwrap();
        let value_view = TensorView::new(StoredDtype::F32, vec![1, 1, 1, 1], &values).unwrap();
        serialize_to_file(
            [("keys", key_view), ("values", value_view)],
            None,
            &root.join("block.safetensors"),
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
            prefix_sha256: hash_token_ids(&[7]),
            layer_layout: descriptor.layer_layout,
            sink_tokens: 0,
            layer_prefix_offsets: vec![0],
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
                payload_sha256: hash_shard_payload(&root.join("block.safetensors")).unwrap(),
            }],
            state_tensors: Vec::new(),
        };
        fs::write(
            root.join("manifest.json"),
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
        assert_eq!(hash_token_ids(&[1, 2, 3]), hash_token_ids(&[1, 2, 3]));
        assert_ne!(hash_token_ids(&[1, 2, 3]), hash_token_ids(&[3, 2, 1]));
    }

    #[test]
    fn prompt_cache_topology_preserves_parallel_coordinates_and_rank_identity() {
        use crate::runtime::distributed::topology::{DeviceAssignment, ParallelTopology};

        let topology =
            ParallelTopology::from_rank(8, 5, 2, 2, 2, DeviceAssignment::new(DeviceType::Cpu, 0))
                .unwrap();
        let cache_topology = PromptCacheTopology::for_parallel_topology(topology);

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
            ParallelTopology::from_rank(1, 0, 1, 1, 1, DeviceAssignment::new(DeviceType::Cpu, 0))
                .unwrap();
        let replicated = PromptCacheTopology::for_parallel_topology(replicated);
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
        assert!(super::attention_policy_from_sliding_window(Some(0)).is_err());
        assert!(super::attention_policy_from_sliding_window(Some(-1)).is_err());
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
        fs::write(
            directory.path().join("manifest.json"),
            br#"{"schema_version":3,"layer_layout":[]}"#,
        )
        .unwrap();
        assert!(matches!(
            inspect_prompt_cache(directory.path()),
            Err(CacheResidencyError::UnsupportedSchema(3))
        ));
    }

    #[test]
    fn v7_layer_frontiers_validate_speculative_cache_coverage() {
        let directory = tempfile::tempdir().unwrap();
        let mut manifest = write_prompt_fixture(directory.path(), "speculative-frontier");
        manifest.layer_count = 2;
        manifest.global_layer_end = 2;
        manifest.total_prefix_tokens = 2;
        manifest.prefix_sha256 = hash_token_ids(&[7, 8]);
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
            directory.path().join("manifest.json"),
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
            directory.path().join("manifest.json"),
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
            super::StateResidencyClass::AlwaysDeviceMutable
        );
        assert_eq!(
            recurrent.residency_class(),
            super::StateResidencyClass::LayerScopedOffloadable
        );
        assert_eq!(
            layouts[2].get(0).unwrap().attention_residency_class(),
            Some(super::StateResidencyClass::SealablePaged)
        );
    }

    fn write_fixed_state_fixture(root: &Path) -> PromptCacheManifest {
        fs::create_dir_all(root).unwrap();
        let values = (0..12)
            .flat_map(|value| (value as f32).to_le_bytes())
            .collect::<Vec<_>>();
        let view = TensorView::new(StoredDtype::F32, vec![1, 3, 4], &values).unwrap();
        let shard = root.join("state.safetensors");
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
            prefix_sha256: hash_token_ids(&[7]),
            layer_layout: LayerSchedule::new(
                1,
                vec![LayerCachePolicy::fixed_only(vec![policy]).unwrap()],
            )
            .unwrap(),
            layer_prefix_offsets: vec![0],
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
                payload_sha256: hash_shard_payload(&shard).unwrap(),
            }],
        };
        fs::write(
            root.join("manifest.json"),
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
                directory.path().join("manifest.json"),
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
        assert!(write(&unexpected).contains("missing, reordered, or unexpected"));

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
        two_layers.layer_layout = key_value_layout([None, Some(7)]);
        two_layers.layer_prefix_offsets = vec![0, 0];
        let mut second = two_layers.blocks[0].clone();
        second.global_layer = 1;
        two_layers.blocks.push(second);

        let write = |manifest: &PromptCacheManifest| {
            fs::write(
                directory.path().join("manifest.json"),
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
        assert!(write(&unexpected).to_string().contains("invalid block"));
    }

    #[test]
    fn manifest_rejects_policy_payload_kind_and_geometry_mismatches() {
        let directory = tempfile::tempdir().unwrap();
        let base = write_prompt_fixture(directory.path(), "policy-mismatch");
        let mut kind = base.clone();
        kind.layer_layout = PromptCacheModelIdentity::compressed_layouts(1, 1, 1).unwrap();
        fs::write(
            directory.path().join("manifest.json"),
            serde_json::to_vec(&kind).unwrap(),
        )
        .unwrap();
        assert!(inspect_prompt_cache(directory.path())
            .unwrap_err()
            .to_string()
            .contains("does not match policy"));

        let mut geometry = base;
        geometry.layer_layout = PromptCacheModelIdentity::key_value_layouts([None], 2, 1).unwrap();
        fs::write(
            directory.path().join("manifest.json"),
            serde_json::to_vec(&geometry).unwrap(),
        )
        .unwrap();
        assert!(inspect_prompt_cache(directory.path())
            .unwrap_err()
            .to_string()
            .contains("policy geometry"));
    }

    #[test]
    fn live_shard_paths_include_process_representation_rank_and_unique_write_identity() {
        let directory = tempfile::tempdir().unwrap();
        let id = disk_test_id(0);
        let (first, first_temporary) = live_block_paths(directory.path(), &id);
        let (second, _) = live_block_paths(directory.path(), &id);
        assert_ne!(first, second);
        let first_name = first.file_name().unwrap().to_string_lossy();
        assert!(first_name.contains("live-p"));
        assert!(first_name.contains("-kv-rank-px-tx-ex-"));
        assert!(first_temporary
            .file_name()
            .unwrap()
            .to_string_lossy()
            .ends_with(".tmp.safetensors"));

        let mut ranked = id;
        ranked.representation = CacheRepresentation::CompressedLatentRotary;
        ranked.rank = Some(CacheRankIdentity {
            pipeline_rank: Some(1),
            tensor_parallel_rank: Some(2),
            expert_parallel_rank: Some(3),
        });
        let (ranked_path, _) = live_block_paths(directory.path(), &ranked);
        assert!(ranked_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .contains("-mla-rank-p1-t2-e3-"));
    }

    #[test]
    fn temporary_file_guard_removes_failed_and_panicking_writes() {
        let directory = tempfile::tempdir().unwrap();
        let failed = directory.path().join("failed.tmp.safetensors");
        {
            let _guard = TemporaryFileGuard::new(failed.clone());
            fs::write(&failed, b"partial").unwrap();
        }
        assert!(!failed.exists());

        let panicking = directory.path().join("panicking.tmp.safetensors");
        let _ = std::panic::catch_unwind(|| {
            let _guard = TemporaryFileGuard::new(panicking.clone());
            fs::write(&panicking, b"partial").unwrap();
            panic!("injected write panic");
        });
        assert!(!panicking.exists());
    }

    #[test]
    fn live_shard_publication_never_clobbers_an_existing_destination() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("live.safetensors");
        let temporary = directory.path().join(".live.tmp.safetensors");
        fs::write(&destination, b"first process").unwrap();
        {
            let _guard = TemporaryFileGuard::new(temporary.clone());
            fs::write(&temporary, b"second process").unwrap();
            assert!(publish_live_block_file(&temporary, &destination).is_err());
        }
        assert_eq!(fs::read(&destination).unwrap(), b"first process");
        assert!(!temporary.exists());
    }

    #[test]
    fn shard_paths_cannot_escape_cache_directory() {
        let root = Path::new("/tmp/cache");
        assert_eq!(
            safe_shard_path(root, "block-0001.safetensors").unwrap(),
            root.join("block-0001.safetensors")
        );
        assert!(matches!(
            safe_shard_path(root, "../outside.safetensors"),
            Err(CacheResidencyError::UnsafeShardPath(_))
        ));
        assert!(safe_shard_path(root, "/outside.safetensors").is_err());
    }

    #[test]
    fn malformed_manifest_is_rejected_without_loading_arrays() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("manifest.json"), b"{not-json").unwrap();
        assert!(matches!(
            inspect_prompt_cache(directory.path()),
            Err(CacheResidencyError::ManifestJson(_))
        ));
    }

    #[test]
    fn same_length_prompt_payload_corruption_is_rejected_before_array_conversion() {
        let directory = tempfile::tempdir().unwrap();
        let manifest = write_prompt_fixture(directory.path(), "payload-checksum");
        let shard = directory.path().join(&manifest.blocks[0].shard);
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
    fn shard_inspection_enforces_a_bounded_header_read() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("oversized.safetensors");
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.write_all(&(MAX_PROMPT_CACHE_SHARD_HEADER_BYTES + 1).to_le_bytes())
            .unwrap();
        let block = PromptCacheBlock {
            global_layer: 0,
            representation: CacheRepresentation::KeyValue,
            start: 0,
            end: 1,
            rank: None,
            shard: "oversized.safetensors".into(),
            first_array: "keys".into(),
            second_array: "values".into(),
            first_shape: vec![1, 1, 1, 1],
            second_shape: vec![1, 1, 1, 1],
            first_dtype: "Float32".into(),
            second_dtype: "Float32".into(),
            logical_bytes: 8,
            payload_sha256: "0".repeat(64),
        };
        assert!(matches!(
            super::validate_shard_file(&path, &block),
            Err(CacheResidencyError::MalformedShard { .. })
        ));
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
        assert_eq!(state.counters.report.imported_mapped_shards, 1);
        assert!(state.blocks.values().all(|record| record
            .disk()
            .and_then(|location| location.mapped.as_ref())
            .is_some()));
        for record in state.blocks.values() {
            verify_disk_payload(record.disk().unwrap()).unwrap();
        }
    }

    #[test]
    fn generation_switch_keeps_the_previous_cache_canonical_until_commit() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("prompt-cache");
        write_prompt_fixture(&destination, "old");
        let generation_name = "generation-test";
        let generation = destination
            .join(PROMPT_CACHE_GENERATIONS_DIRECTORY)
            .join(generation_name);
        write_prompt_fixture(&generation, "new");

        assert_eq!(
            inspect_prompt_cache(&destination)
                .unwrap()
                .application_namespace
                .as_deref(),
            Some("old")
        );
        publish_prompt_cache_generation(&destination, generation_name, 1).unwrap();
        assert_eq!(
            inspect_prompt_cache(&destination)
                .unwrap()
                .application_namespace
                .as_deref(),
            Some("new")
        );
    }

    #[test]
    fn durable_rename_publishes_directories_and_replaces_pointer_files() {
        let root = tempfile::tempdir().unwrap();
        let pointer = root.path().join("CURRENT");
        let temporary_pointer = root.path().join(".CURRENT.tmp");
        fs::write(&pointer, b"old\n").unwrap();
        fs::write(&temporary_pointer, b"new\n").unwrap();
        durable_rename(&temporary_pointer, &pointer, true).unwrap();
        assert_eq!(fs::read(&pointer).unwrap(), b"new\n");
        assert!(!temporary_pointer.exists());

        let temporary_generation = root.path().join(".generation.tmp");
        let generation = root.path().join("generation-1");
        fs::create_dir(&temporary_generation).unwrap();
        fs::write(temporary_generation.join("manifest.json"), b"{}").unwrap();
        durable_rename(&temporary_generation, &generation, false).unwrap();
        assert_eq!(fs::read(generation.join("manifest.json")).unwrap(), b"{}");
        assert!(!temporary_generation.exists());
    }

    #[test]
    fn loaded_model_identity_rejects_a_forged_caller_descriptor() {
        let mut descriptor = prompt_descriptor();
        descriptor.layer_count = 2;
        descriptor.global_layer_end = 2;
        let loaded_model = PromptCacheModelIdentity {
            model_family: "llama".into(),
            effective_model_type: "llama".into(),
            architecture_fingerprint: descriptor.architecture_fingerprint.clone(),
            layer_count: 1,
            global_layer_start: 0,
            global_layer_end: 1,
            sink_tokens: 0,
            layer_prefix_offsets: vec![0],
            topology: PromptCacheTopology::default(),
            layer_layout: PromptCacheModelIdentity::key_value_layouts([None], 1, 1).unwrap(),
        };
        assert!(matches!(
            validate_prompt_cache_model_identity(&descriptor, &loaded_model),
            Err(CacheResidencyError::IncompatiblePromptCache(_))
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
            &directory.path().join("block.safetensors"),
        )
        .unwrap();
        manifest.blocks[0].first_shape = vec![1, 2, 1, 4];
        manifest.blocks[0].second_shape = vec![1, 2, 1, 4];
        manifest.blocks[0].logical_bytes = 64;
        fs::write(
            directory.path().join("manifest.json"),
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
        assert!(error.to_string().contains("policy geometry"));
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
            &directory.path().join("block.safetensors"),
        )
        .unwrap();
        manifest.blocks[0].representation = CacheRepresentation::CompressedLatentRotary;
        manifest.blocks[0].first_array = "latent".into();
        manifest.blocks[0].second_array = "rotary_key".into();
        manifest.blocks[0].first_shape = vec![1, 1, 4];
        manifest.blocks[0].second_shape = vec![1, 1, 2];
        manifest.blocks[0].logical_bytes = 24;
        fs::write(
            directory.path().join("manifest.json"),
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
        assert!(error.to_string().contains("does not match policy"));
    }

    #[test]
    fn model_reset_surfaces_propagate_paged_clear_failures() {
        use crate::{
            architectures::gpt_oss::model::{Cache as GptOssCache, LayerCache as GptOssLayerCache},
            architectures::{
                distributed::{
                    expert::ExpertParallelCache,
                    pipeline::{PipelineCache, PipelineKeyValueCache, PipelineLayerCache},
                },
                llama::layerwise::LlamaCache,
            },
            runtime::cache::PagedKeyValueCache,
        };

        let manager = manager_with_leased_block();
        let mut llama = LlamaCache::Paged(vec![Some(
            PagedKeyValueCache::new(manager.clone(), 0, None).unwrap(),
        )]);
        assert!(llama.clear().is_err());
        assert_eq!(manager.lock().unwrap().blocks.len(), 1);

        let manager = manager_with_leased_block();
        let mut pipeline = PipelineCache::new(
            crate::api::ModelKind::Llama,
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

        let manager = manager_with_leased_block();
        let gpt_cache = || GptOssCache {
            layers: vec![GptOssLayerCache::Paged(
                PagedKeyValueCache::new(manager.clone(), 0, None).unwrap(),
            )],
        };
        let mut gpt_oss = gpt_cache();
        assert!(gpt_oss.reset().is_err());
        assert_eq!(manager.lock().unwrap().blocks.len(), 1);

        let mut expert_parallel = ExpertParallelCache::GptOss(gpt_cache());
        assert!(expert_parallel.reset().is_err());
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
        assert!(second.joined);
        let second_ticket = second.ticket.clone();
        first.enqueue().unwrap();
        second.enqueue().unwrap();
        assert!(ticket.wait().is_err());
        assert!(second_ticket.wait().is_err());
        assert!(std::sync::Arc::ptr_eq(
            &ticket.completion,
            &second_ticket.completion
        ));
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
                DiskOperationKey {
                    generation: 0,
                    id: disk_test_id(0),
                    kind: DiskOperationKind::Read,
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
                DiskOperationKey {
                    generation: 0,
                    id: disk_test_id(1),
                    kind: DiskOperationKind::Read,
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
                DiskOperationKey {
                    generation: 8,
                    id: disk_test_id(0),
                    kind: DiskOperationKind::Read,
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
                DiskOperationKey {
                    generation: 4,
                    id: disk_test_id(0),
                    kind: DiskOperationKind::Read,
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
                DiskOperationKey {
                    generation: 2,
                    id: disk_test_id(0),
                    kind: DiskOperationKind::Read,
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
        let key = DiskOperationKey {
            generation: 0,
            id: id.clone(),
            kind: DiskOperationKind::Write,
        };
        let submission = worker.prepare(key.clone(), DiskTask::Panic).unwrap();
        let ticket = submission.ticket.clone();
        manager.lock().unwrap().blocks.insert(
            id.clone(),
            CacheBlockRecord {
                id,
                physical: CacheBlockPhysicalState::Host {
                    block: test_host_block(),
                    persistence: HostCachePersistence::Writing(PendingDiskOperation {
                        ticket: ticket.clone(),
                    }),
                },
                bytes: 0,
                shapes: [vec![1], vec![1]],
                dtypes: ["Float32".into(), "Float32".into()],
                imported: false,
                leases: 0,
                access_count: 0,
                last_access: 0,
                protected_prefix: false,
            },
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
        let key = DiskOperationKey {
            generation: 0,
            id: id.clone(),
            kind: DiskOperationKind::Write,
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
            state.blocks.insert(
                id.clone(),
                CacheBlockRecord {
                    id: id.clone(),
                    physical: CacheBlockPhysicalState::Host {
                        block: host_block,
                        persistence: HostCachePersistence::Writing(PendingDiskOperation {
                            ticket: ticket.clone(),
                        }),
                    },
                    bytes: 16,
                    shapes: [vec![1], vec![1]],
                    dtypes: ["Float32".into(), "Float32".into()],
                    imported: false,
                    leases: 0,
                    access_count: 0,
                    last_access: 0,
                    protected_prefix: false,
                },
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
        assert!(matches!(
            manager.lock().unwrap().blocks.get(&id).unwrap().physical,
            CacheBlockPhysicalState::Host {
                persistence: HostCachePersistence::Writing(_),
                ..
            }
        ));

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
                state.blocks.insert(
                    id.clone(),
                    CacheBlockRecord {
                        id,
                        physical: CacheBlockPhysicalState::Device {
                            arrays: test_device_block(),
                            disk,
                        },
                        bytes: 16,
                        shapes: [vec![1], vec![1]],
                        dtypes: ["Float32".into(), "Float32".into()],
                        imported: false,
                        leases: 0,
                        access_count: 0,
                        last_access: 0,
                        protected_prefix: false,
                    },
                );
            }
        }

        manager.rebalance(None, false).unwrap();
        let state = manager.lock().unwrap();
        assert_eq!(state.blocks.get(&older).unwrap().tier(), CacheTier::Disk);
        assert_eq!(state.blocks.get(&recent).unwrap().tier(), CacheTier::Device);
        assert_eq!(state.counters.report.current_host_bytes, 0);
        assert_eq!(state.counters.report.current_device_bytes, 16);
        assert_eq!(state.counters.report.current_disk_bytes, 16);
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
                let physical = match tier {
                    CacheTier::Device => CacheBlockPhysicalState::Device {
                        arrays: test_device_block(),
                        disk: None,
                    },
                    CacheTier::Host => CacheBlockPhysicalState::Host {
                        block: test_host_block(),
                        persistence: HostCachePersistence::Unbacked,
                    },
                    CacheTier::Disk => CacheBlockPhysicalState::Disk {
                        location: missing_location(
                            Path::new("/tmp/safemlx-cache-report-test"),
                            &format!("layer-{global_layer}.safetensors"),
                        ),
                        read: DiskCacheReadState::Ready,
                    },
                };
                let id = CacheBlockId {
                    session_id: manager.session_id,
                    global_layer,
                    representation,
                    start: 0,
                    end: global_layer as i64 + 1,
                    rank: None,
                };
                state.blocks.insert(
                    id.clone(),
                    CacheBlockRecord {
                        id,
                        physical,
                        bytes: global_layer as u64 + 1,
                        shapes: [vec![1], vec![1]],
                        dtypes: ["Float32".into(), "Float32".into()],
                        imported: false,
                        leases: 0,
                        access_count: 0,
                        last_access: 0,
                        protected_prefix: global_layer % 5 == 0,
                    },
                );
                state
                    .tails
                    .insert(global_layer, (2, global_layer as i64 + 1));
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
            CacheResidencyError::PoolBudgetExceeded {
                resource: CachePoolResource::Device,
                required: 20,
                budget: 16,
            }
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
            record.physical = CacheBlockPhysicalState::Host {
                block: host_block,
                persistence: HostCachePersistence::Unbacked,
            };
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
                DiskOperationKey {
                    generation: 0,
                    id: disk_test_id(99),
                    kind: DiskOperationKind::Read,
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
                DiskOperationKey {
                    generation: 0,
                    id: disk_test_id(99),
                    kind: DiskOperationKind::Read,
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
        manager.lock().unwrap().blocks.insert(
            id.clone(),
            CacheBlockRecord {
                id: id.clone(),
                physical: CacheBlockPhysicalState::Demoting {
                    arrays: test_device_block(),
                    ticket: ticket.clone(),
                },
                bytes: 8,
                shapes: [vec![1], vec![1]],
                dtypes: ["Float32".into(), "Float32".into()],
                imported: false,
                leases: 0,
                access_count: 0,
                last_access: 0,
                protected_prefix: false,
            },
        );

        let error = manager.finish_device_demotion(&ticket).unwrap_err();
        assert!(error.to_string().contains("injected host demotion failure"));
        let state = manager.lock().unwrap();
        assert!(matches!(
            state.blocks.get(&id).unwrap().physical,
            CacheBlockPhysicalState::Device { .. }
        ));
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
        manager.lock().unwrap().blocks.insert(
            id.clone(),
            CacheBlockRecord {
                id,
                physical: CacheBlockPhysicalState::Demoting {
                    arrays: test_device_block(),
                    ticket,
                },
                bytes: 8,
                shapes: [vec![1], vec![1]],
                dtypes: ["Float32".into(), "Float32".into()],
                imported: false,
                leases: 0,
                access_count: 0,
                last_access: 0,
                protected_prefix: false,
            },
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
            let CacheBlockPhysicalState::Demoting {
                arrays,
                ticket: live,
            } = &record.physical
            else {
                panic!("device demotion was not represented as in flight");
            };
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

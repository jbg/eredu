//! Exact native transfers for portable prompt-cache publication and restoration.

use super::{
    CacheBlockId, PromptCacheBlock, PromptCacheError, PromptCacheStateTensor, StateTensorOwner,
    StateTensorRole,
};
use crate::BackendFailure;
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, sync::Arc};

/// Immutable authorized byte source for an exact cache shard.
pub trait PromptCacheShardPayload: std::fmt::Debug + Send + Sync {
    /// Fills every exact named destination and validates the retained payload digest.
    fn read_into(&self, destinations: &mut [(&str, &mut [u8])]) -> Result<(), BackendFailure>;
}

/// Native host allocation and writable transfer bytes for one exact cache tensor.
pub trait CacheHostBuffer {
    /// Backend allocation retained until a transfer takes ownership.
    type Buffer;

    /// Allocates a native host buffer for the supplied storage layout.
    fn allocate(
        shape: &[usize],
        dtype: eredu_checkpoint::StoredDtype,
    ) -> Result<Self::Buffer, BackendFailure>;

    /// Borrows the exact writable payload, excluding physical capacity padding.
    fn bytes_mut(buffer: &mut Self::Buffer) -> Result<&mut [u8], BackendFailure>;
}

/// Exact completed host payload for a cache tensor, excluding capacity padding.
pub struct CacheHostTensor<'a> {
    /// Logical tensor dimensions.
    pub shape: Vec<usize>,
    /// Physical storage encoding of the supplied bytes.
    pub dtype: eredu_checkpoint::StoredDtype,
    /// Contiguous completed payload.
    pub bytes: &'a [u8],
}

/// Exact tensor geometry and payload extent reported by a native state binding.
#[derive(Debug, Clone)]
pub struct PromptCacheTensorLayout {
    /// Exact logical dimensions; no native capacity padding.
    pub shape: Vec<i32>,
    /// Canonical backend-neutral dtype name used by the cache schema.
    pub dtype: String,
    /// Exact payload extent in bytes.
    pub bytes: u64,
}
impl PromptCacheTensorLayout {
    /// Validates a logical tensor layout and computes its exact byte extent.
    pub fn new(shape: Vec<i32>, dtype: String) -> Result<Self, PromptCacheError> {
        let width = match dtype.as_str() {
            "Bool" | "Uint8" | "Int8" => 1u64,
            "Uint16" | "Int16" | "Float16" | "Bfloat16" => 2,
            "Uint32" | "Int32" | "Float32" => 4,
            "Uint64" | "Int64" | "Float64" => 8,
            _ => {
                return Err(PromptCacheError::Malformed(
                    "unsupported prompt-cache dtype".into(),
                ))
            }
        };
        let bytes = shape
            .iter()
            .try_fold(width, |n, d| {
                u64::try_from(*d).ok().and_then(|d| n.checked_mul(d))
            })
            .ok_or_else(|| {
                PromptCacheError::Malformed("invalid prompt-cache tensor extent".into())
            })?;
        Ok(Self {
            shape,
            dtype,
            bytes,
        })
    }
}
/// A native block, independent of serialized shard naming or storage medium.
#[derive(Debug, Clone)]
pub struct PromptCacheExportBlock {
    /// Logical block identity and architecture-declared representation.
    pub id: CacheBlockId,
    /// Ordered first and second tensor layouts for this representation.
    pub tensors: [PromptCacheTensorLayout; 2],
}
/// One architecture-declared fixed or recurrent tensor.
#[derive(Debug, Clone)]
pub struct PromptCacheExportState {
    /// Architecture-declared layer or model-global owner.
    pub owner: StateTensorOwner,
    /// Architecture-declared mutable-state tensor role.
    pub role: StateTensorRole,
    /// Exact native tensor layout.
    pub tensor: PromptCacheTensorLayout,
}
/// State inventory obtained without copying native payloads.
#[derive(Debug, Clone)]
pub struct PromptCacheExportInventory {
    /// Selected physical block extent in token positions.
    pub block_size_tokens: i32,
    /// Stable ordered native block inventory.
    pub blocks: Vec<PromptCacheExportBlock>,
    /// Stable ordered fixed and recurrent state inventory.
    pub states: Vec<PromptCacheExportState>,
    /// Native streams whose mutable tails have not been sealed.
    pub unsealed_streams: Vec<super::CacheStreamId>,
    /// Peak simultaneous native host-transfer capacity, excluding encoded bytes.
    pub transfer_bytes: u64,
}
/// Completed host transfers. Their owner keeps exact bytes alive during encoding.
pub trait PromptCachePayload {
    /// Borrows completed native transfer buffers in inventory order.
    fn bytes(&self) -> Result<Vec<&[u8]>, BackendFailure>;
}
/// Native operations only: inventory and completed byte transfers.
pub trait PromptCacheExport {
    /// Owner of completed host-transfer buffers.
    type Payload: PromptCachePayload;
    /// Returns the stable inventory without copying native state.
    fn inventory(&self) -> &PromptCacheExportInventory;
    /// Completes the indexed native block transfer.
    fn transfer_block(&self, index: usize) -> Result<Self::Payload, BackendFailure>;
    /// Completes the indexed fixed/recurrent tensor transfer.
    fn transfer_state(&self, index: usize) -> Result<Self::Payload, BackendFailure>;
    /// Records native telemetry after the portable publication commits.
    fn record_publication(&self, _logical_bytes: u64) {}
}
/// Native allocation and transfer binding used by portable restore orchestration.
pub trait PromptCacheBlockRestore {
    /// Retains an immutable file handle for the existing lazy paging mechanism.
    fn retain_block(
        &mut self,
        block: &PromptCacheBlock,
        pinned: bool,
        source: Arc<dyn PromptCacheShardPayload>,
        path: PathBuf,
    ) -> Result<(), BackendFailure>;
    /// Completes host-to-native restoration of a bounded, validated block.
    fn transfer_block(
        &mut self,
        block: &PromptCacheBlock,
        pinned: bool,
        source: &dyn PromptCacheShardPayload,
    ) -> Result<(), BackendFailure>;
}
/// Binds a fixed/recurrent tensor to native allocation and completion operations.
pub trait PromptCacheStateRestore {
    /// Allocates native fixed state and completes its exact byte transfer.
    fn transfer_state(
        &mut self,
        state: &PromptCacheStateTensor,
        source: &dyn PromptCacheShardPayload,
    ) -> Result<(), BackendFailure>;
}

/// A transfer failure classified for portable lookahead.
pub enum CachePrefetchFailure<E> {
    /// Optional lookahead can yield to a currently leased page.
    DevicePressure(E),
    /// A transfer failure must reach the caller.
    Transfer(E),
}

/// Exact page preparation and byte observations used by the prefetch window.
pub trait CachePrefetchMechanisms {
    /// Exact lease that retains each prepared page.
    type Lease;
    /// Preparation or observation failure.
    type Error;
    /// Native transfer context.
    type Context;

    /// Returns the exact logical size of one catalog page.
    fn page_bytes(&self, id: &CacheBlockId) -> Result<u64, Self::Error>;
    /// Returns the size retained by a lease.
    fn leased_bytes(lease: &Self::Lease) -> u64;
    /// Prepares the selected page on the supplied context.
    fn prepare_page(
        &self,
        id: &CacheBlockId,
        context: &Self::Context,
    ) -> Result<Self::Lease, CachePrefetchFailure<Self::Error>>;
}

/// Kind of asynchronous backing-store operation.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheIoOperationKind {
    /// Publish host resources to a durable backing.
    Write,
    /// Reconstruct host resources from a durable backing.
    Read,
}

/// Exact identity of one backing-store operation.
#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct CacheIoOperationKey {
    /// Cache generation in which the operation was submitted.
    pub generation: u64,
    /// Block whose resources the operation owns.
    pub id: CacheBlockId,
    /// Direction of the operation.
    pub kind: CacheIoOperationKind,
}

/// Opaque backend host-demotion completion with stable identity.
pub trait CacheHostDemotionOperation {
    /// Block whose device resources the operation owns.
    fn block_id(&self) -> &CacheBlockId;

    /// Monotonic operation identity within the backend cache session.
    fn operation_id(&self) -> u64;
}

/// Opaque backend I/O completion with an exact neutral key.
pub trait CacheIoOperation {
    /// Returns the operation identity owned by this completion.
    fn key(&self) -> &CacheIoOperationKey;
}
/// Separate native queues prevent a transfer waiting on the reverse direction.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CacheTransferLane {
    /// Host-to-device payload transfer.
    Promotion,
    /// Device-to-host payload transfer.
    Demotion,
    /// Transfer for durable cache storage.
    Persistence,
}

/// Process counters sampled at a portable reporting boundary.
pub struct CacheProcessUsage {
    /// Peak resident process bytes.
    pub peak_rss: u64,
    /// Minor page faults.
    pub minor_faults: u64,
    /// Major page faults.
    pub major_faults: u64,
}

/// Native memory and submission operations for the portable cache.
///
/// The cache owns every page and decides when to call these operations. Native
/// implementations retain queued resources until completion or safe retirement,
/// including when only part of a batch was submitted successfully.
pub trait LiveCacheBackend:
    CacheHostBuffer + std::fmt::Debug + Clone + Send + Sync + 'static
{
    /// Device tensor shared with the cache's transfer worker.
    type Tensor: Clone + std::fmt::Debug + Send + Sync + 'static;
    /// Immutable completed host transfer storage.
    type Host: Clone + std::fmt::Debug + Send + Sync + 'static;
    /// Device binding retained by an admitted cache, including native context identity when needed.
    type Device: Clone + Eq + std::fmt::Debug + Send + Sync + 'static;
    /// Exact native execution context.
    type Queue: Clone;
    /// Native dependency retained until the consumer is ordered after it.
    type Event;

    /// Reads tensor geometry without submitting work.
    fn shape(tensor: &Self::Tensor) -> &[i32];
    /// Reads the physical storage encoding.
    fn dtype(tensor: &Self::Tensor) -> eredu_checkpoint::StoredDtype;
    /// Reads the logical byte extent.
    fn bytes(tensor: &Self::Tensor) -> u64;
    /// Copies the tensor into independent native storage.
    fn copy_tensor(
        tensor: &Self::Tensor,
        queue: &Self::Queue,
    ) -> Result<Self::Tensor, BackendFailure>;
    /// Completes work on the exact retained tensors before publishing cache state.
    fn settle_tensors(tensors: &[&Self::Tensor]) -> Result<(), BackendFailure>;
    /// Captures native completion for an existing device page.
    fn device_ready(tensors: [&Self::Tensor; 2]) -> Result<Self::Event, BackendFailure>;

    /// Determines the exact device selected by a queue.
    fn device(queue: &Self::Queue) -> Result<Self::Device, BackendFailure>;
    /// Host device used when the caller has not bound an accelerator.
    fn host_device() -> Self::Device;
    /// Opens or obtains the requested exact transfer lane.
    fn transfer_queue(
        device: Self::Device,
        lane: CacheTransferLane,
    ) -> Result<Self::Queue, BackendFailure>;
    /// Conservative transfer allocation size for one tensor payload.
    fn host_capacity_bound(bytes: u64) -> Result<u64, BackendFailure>;
    /// Actual native allocation size, including transfer padding.
    fn host_capacity(host: &Self::Host) -> Result<u64, BackendFailure>;
    /// Borrows completed bytes and their exact logical layout.
    fn host_payload(host: &Self::Host) -> Result<CacheHostTensor<'_>, BackendFailure>;
    /// Freezes a populated transfer allocation.
    fn freeze_host(buffer: Self::Buffer) -> Self::Host;
    /// Completes the requested native device-to-host transfers in order.
    fn copy_to_host<const N: usize>(
        tensors: [&Self::Tensor; N],
        queue: &Self::Queue,
    ) -> Result<[Self::Host; N], BackendFailure>;
    /// Submits native copies of the given immutable buffers to the exact device.
    fn copy_to_device<const N: usize>(
        buffers: [&Self::Host; N],
        queue: &Self::Queue,
    ) -> Result<[(Self::Tensor, Self::Event); N], BackendFailure>;
    /// Orders a consumer behind a native copy.
    fn wait_on(event: &Self::Event, queue: &Self::Queue) -> Result<(), BackendFailure>;
    /// Waits for native completion before publishing restored state.
    fn synchronize(event: Self::Event) -> Result<(), BackendFailure>;
    /// Reports process memory and page faults when available.
    fn process_usage() -> Option<CacheProcessUsage> {
        None
    }
}

/// Tensor operations used to turn an append into immutable cache pages.
pub trait LiveCacheAppendBackend: LiveCacheBackend {
    /// Returns the exact interval on the supplied tensor axis.
    fn slice_tokens(
        tensor: &Self::Tensor,
        axis: usize,
        range: std::ops::Range<i32>,
        queue: &Self::Queue,
    ) -> Result<Self::Tensor, BackendFailure>;

    /// Concatenates nonempty exact tensor slices on the supplied axis.
    fn concatenate_tokens(
        slices: &[&Self::Tensor],
        axis: usize,
        queue: &Self::Queue,
    ) -> Result<Self::Tensor, BackendFailure>;

    /// Retains a compact, settled tensor without keeping a larger input allocation.
    fn compact_tokens(
        tensor: &Self::Tensor,
        queue: &Self::Queue,
    ) -> Result<Self::Tensor, BackendFailure>;
}

/// Exact tensor storage selected for one layer's attention history.
#[derive(Clone, Copy, Debug)]
pub enum AttentionStorageKind {
    /// Ordinary keys and values or a key-only history.
    KeyValue {
        /// Sliding window in positions, when applicable.
        window: Option<i32>,
        /// Only keys carry model values.
        key_only: bool,
    },
    /// Compressed latent plus rotary key history.
    CompressedLatent,
}

/// Exact allocation request for one layer. The portable planner selects the
/// backing and validates that a shared attention cache can hold all components.
#[derive(Debug)]
pub struct LayerStateAllocation {
    /// Attention representation and whether it is backed by live pages.
    pub attention: Option<(AttentionStorageKind, bool)>,
    /// Initially empty device slots for fixed state.
    pub fixed: Vec<crate::cache::StateTensorRole>,
    /// Realized append-stream backing for this layer.
    pub streams: Vec<(
        crate::cache::append_stream::AppendStreamBinding,
        crate::cache::append_stream::AppendStreamBacking,
    )>,
}

/// Independent native copies of an exact selection from a leased cache page.
pub trait LiveCacheSelectedReadBackend: LiveCacheAppendBackend {
    /// Copies a contiguous slice into compact storage.
    /// Retains the supplied source owner through native completion or teardown.
    fn copy_token_slice(
        tensor: &Self::Tensor,
        axis: usize,
        range: std::ops::Range<i32>,
        queue: &Self::Queue,
        retention: impl Sized + 'static,
    ) -> Result<Self::Tensor, BackendFailure>;

    /// Copies the selected token rows of one batch lane into compact storage.
    /// Retains the supplied source owner through native completion or teardown.
    fn gather_token_rows(
        tensors: [&Self::Tensor; 2],
        batch: i32,
        rows: &[i32],
        queue: &Self::Queue,
        retention: impl Sized + 'static,
    ) -> Result<[Self::Tensor; 2], BackendFailure>;
}

/// One block supplied to a native streaming attention kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheAttentionInput {
    /// Immutable page whose owner must remain live through native completion.
    SealedPage,
    /// Unsealed retained state for the admitted stream.
    RetainedTail,
    /// Invocation-local values that do not advance the cache.
    Transient,
}

/// Ordered sweeps required by the selected numerical kernel.
#[derive(Clone, Copy)]
pub enum AttentionScan {
    /// Merge unnormalized values and row masses online.
    Online,
    /// Merge normalizers online, then accumulate rounded probabilities.
    OnlineThenValues,
    /// Compute global maxima, then masses, then normalized value products.
    MaximumThenMassThenValues,
}

impl AttentionScan {
    /// Number of scans over the same admitted page ranges.
    pub const fn passes(self) -> usize {
        match self {
            Self::Online => 1,
            Self::OnlineThenValues => 2,
            Self::MaximumThenMassThenValues => 3,
        }
    }
}

/// Native numerical operations over the exact blocks supplied by a cache scan.
pub trait CacheAttentionAccumulator<T, Q> {
    /// Returns the fixed numerical sweep plan.
    fn scan(&self) -> AttentionScan;

    /// Prepares the next sweep of the same page ranges.
    fn begin_pass(&mut self, index: usize, queue: &Q) -> Result<(), BackendFailure>;

    /// Accumulates one selected block and reports its native scratch usage.
    /// Takes ownership of any source lease until native work settles.
    fn accumulate(
        &mut self,
        source: CacheAttentionInput,
        positions: std::ops::Range<i64>,
        tensors: [&T; 2],
        queue: &Q,
        retention: impl Sized + 'static,
    ) -> Result<u64, BackendFailure>;

    /// Establishes completion and returns the computed result.
    fn finish(self, queue: &Q) -> Result<T, BackendFailure>;
}

/// Native scope protecting the source of a cache read until completion or teardown.
pub trait CacheReadSubmission: Sized {
    /// Settles work that reads the retained source.
    fn settle(self) -> Result<(), BackendFailure>;
}

/// Opens a native scope around cache reads whose computation spans multiple operators.
pub trait LiveCacheReadSubmissionBackend: LiveCacheBackend {
    /// Retains the opaque source owner through native completion or teardown.
    fn begin_cache_read(
        queue: &Self::Queue,
        retention: impl Sized + 'static,
    ) -> Result<impl CacheReadSubmission, BackendFailure>;
}

/// Conservative native storage facts for copying retained cache tensors.
pub struct CacheSnapshotTensorCosts {
    /// Descriptor bytes independent of tensor rank.
    pub tensor_metadata: u64,
    /// Descriptor bytes per dimension.
    pub dimension_metadata: u64,
    /// Concurrent source, destination, and working buffers per tensor.
    pub copy_buffers: u64,
    /// Maximum byte width of the backend's selected floating types.
    pub floating_width: u64,
}

impl CacheSnapshotTensorCosts {
    /// Conservative bytes needed to retain and independently copy the tensor.
    pub fn bytes(&self, payload: u64, rank: usize) -> Option<u64> {
        payload
            .checked_mul(self.copy_buffers)?
            .checked_add(self.tensor_metadata)?
            .checked_add(
                u64::try_from(rank)
                    .ok()?
                    .checked_mul(self.dimension_metadata)?,
            )
    }
}

/// Native tensor storage shared by resident attention cache owners.
pub trait CacheTensorStorage: LiveCacheAppendBackend {
    /// The tensor handle exposed to neural operators.
    type Value: eredu_nn::Tensor<Context = Self::Queue>;

    /// Adopts a native handle for use by neural operators.
    fn into_value(tensor: Self::Tensor) -> Self::Value;
    /// Adopts the native handle backing a neural tensor.
    fn into_storage(value: Self::Value) -> Self::Tensor;
    /// Borrows native storage for an existing neural tensor.
    fn storage_ref(tensor: &Self::Value) -> &Self::Tensor;
    /// Borrows the neural tensor handle without retaining another native value.
    fn borrow_value(tensor: &Self::Tensor) -> &Self::Value;

    /// Allocates exact zeroed dimensions using the supplied tensor's element type.
    fn zeroed_cache_tensor(
        shape: &[i32],
        like: &Self::Tensor,
        queue: &Self::Queue,
    ) -> Result<Self::Tensor, BackendFailure>;

    /// Allocates zeroed storage with the supplied token capacity on the axis.
    fn allocate_tokens(
        like: &Self::Tensor,
        axis: usize,
        capacity: i32,
        queue: &Self::Queue,
    ) -> Result<Self::Tensor, BackendFailure>;

    /// Returns new backing with the supplied interval replaced. Original
    /// storage values remain immutable.
    fn write_tokens(
        target: Self::Tensor,
        axis: usize,
        positions: std::ops::Range<i32>,
        source: &Self::Tensor,
        queue: &Self::Queue,
    ) -> Result<Self::Tensor, BackendFailure>;

    /// Reports tensor storage costs when complete isolated-copy bounds are available.
    fn snapshot_tensor_costs() -> Option<CacheSnapshotTensorCosts> {
        None
    }

    /// Forks the supplied backing tensor without changing its geometry.
    fn fork_tensor(tensor: &Self::Tensor) -> Result<Self::Tensor, BackendFailure>;
}

/// Settles exact tensor roots while retaining all enrolled resources and the copy authority.
pub trait CacheCopyScope<T, R> {
    /// Borrows the enrolled state and authority.
    fn resources(&self) -> &R;
    /// Retains intermediate copy results as soon as the caller creates them.
    fn resources_mut(&mut self) -> &mut R;
    /// Establishes completion for the supplied roots. An unresolved failure must
    /// keep the enrolled resources until safe native retirement.
    fn settle(&mut self, roots: &[T]) -> Result<(), BackendFailure>;
    /// Retires a successfully settled scope.
    fn finish(self);
}

/// Native queue and recovery for a portable cache copy transaction.
pub trait CacheCopySubmission: CacheTensorStorage {
    /// Queue scope that holds opaque Rust owners through native completion.
    type CopyScope<R: 'static>: CacheCopyScope<Self::Tensor, R>;

    /// Opens a native scope before the portable owner requests any copies.
    fn begin_copy<R: 'static>(
        queue: &Self::Queue,
        resources: R,
    ) -> Result<Self::CopyScope<R>, BackendFailure>;

    /// Reclaims terminal native owners that retain cache-copy authority.
    fn reclaim_copies();
}

/// Native storage and completion for a portable compressed attention cache.
pub trait CompressedCacheStorage:
    CacheTensorStorage + LiveCacheReadSubmissionBackend<Event: 'static>
{
    /// Native memory bound for making one independent snapshot tensor.
    fn snapshot_tensor_estimate(
        _tensor: &Self::Tensor,
    ) -> Option<crate::execution_control::SnapshotEstimate> {
        None
    }
}

/// Physical backing selected for the next resident cache append.
#[derive(Clone, Copy)]
pub enum ResidentAppendBacking {
    /// The cache can reuse the appended tensor handles.
    BorrowedInput,
    /// Native reservations cover the complete append.
    ReservedPrefix,
    /// The complete update fits the installed overwrite capacity.
    ExistingCapacity,
    /// Newly concatenated or padded backing is required.
    NewAllocation,
}

/// Exact resident backing selected for one append forecast.
pub struct ResidentAppendStorage<'a, T> {
    /// Token extent priced in the selected backing.
    pub positions: u64,
    /// Physical relationship between the input and selected backing.
    pub backing: ResidentAppendBacking,
    /// Whether keys also provide values.
    pub key_only: bool,
    /// Existing key storage to validate against the request.
    pub keys: Option<&'a T>,
    /// Existing value storage to validate against the request.
    pub values: Option<&'a T>,
}

/// Native immutable-prefix allocation for a resident key/value cache.
pub trait ResidentKeyValueStorage: CacheTensorStorage {
    /// Native reservation that owns spare token capacity beyond a visible prefix.
    type Prefix: Clone + std::fmt::Debug;

    /// Computes attention over the supplied, already selected tensors.
    fn resident_attention(
        request: eredu_nn::AttentionRequest<'_, Self::Value>,
        queue: &Self::Queue,
    ) -> Result<Self::Value, eredu_nn::Error>;

    /// Describes the native memory of the selected append backing.
    fn describe_resident_append(
        invocation: &eredu_nn::mechanism_memory::MechanismInvocation,
        storage: ResidentAppendStorage<'_, Self::Tensor>,
    ) -> Result<eredu_nn::mechanism_memory::MechanismMemoryContract, eredu_nn::Error>;

    /// Initializes private backing for an exact prefix and capacity.
    fn new_prefix(
        tensor: &Self::Tensor,
        capacity: i32,
        queue: &Self::Queue,
    ) -> Result<Self::Prefix, BackendFailure>;
    /// Appends in a reserved suffix or allocates an independent backing.
    fn append_prefix(
        prefix: &Self::Prefix,
        update: &Self::Tensor,
        capacity: i32,
        queue: &Self::Queue,
    ) -> Result<Self::Prefix, BackendFailure>;
    /// Borrows an immutable view of the committed logical prefix.
    fn prefix_view(
        prefix: &Self::Prefix,
        queue: &Self::Queue,
    ) -> Result<Self::Tensor, BackendFailure>;
    /// Preserves the reserved capacity while reducing visible length.
    fn truncate_prefix(prefix: &Self::Prefix, length: i32) -> Result<Self::Prefix, BackendFailure>;
    /// Exact physical capacity of the native prefix owner.
    fn prefix_capacity(prefix: &Self::Prefix) -> i32;
    /// Bytes currently reserved beyond the immutable visible range.
    fn prefix_spare_bytes(prefix: &Self::Prefix) -> u64;
    /// Observes whether the requested append can use the existing backing.
    fn can_append_prefix_without_copy(prefix: &Self::Prefix, count: i32) -> bool;
}

/// Exact geometry of one selected paged attention scan.
pub struct PagedAttentionGeometry {
    /// Position of the first query.
    pub query_start: i64,
    /// Exclusive end of the available context.
    pub context_end: i64,
    /// Sliding window in positions.
    pub window: Option<i32>,
    /// Always-visible prefix length in positions.
    pub prefix: i64,
    /// Start of retained history for a non-advancing invocation.
    pub transient_context_start: Option<i64>,
}

/// Numerical operands shared by advancing and non-advancing paged attention.
pub struct PagedAttentionOperands<'a, T> {
    /// Query tensor.
    pub queries: &'a T,
    /// Score scale.
    pub scale: f32,
    /// Explicit boolean or additive attention mask.
    pub mask: Option<&'a T>,
    /// Learned sink logits.
    pub sinks: Option<&'a T>,
    /// Optional score cap.
    pub softcap: Option<f32>,
    /// Score and normalization arithmetic.
    pub arithmetic: eredu_nn::AttentionArithmetic,
}

/// Numerical kernels and tensor operations used by portable paged attention.
pub trait PagedKeyValueStorage:
    CacheTensorStorage + LiveCacheSelectedReadBackend + LiveCacheReadSubmissionBackend<Event: 'static>
{
    /// Describes native memory for the selected paged append.
    fn describe_paged_append(
        invocation: &eredu_nn::mechanism_memory::MechanismInvocation,
    ) -> Result<eredu_nn::mechanism_memory::MechanismMemoryContract, eredu_nn::Error>;

    /// Prepares a numerical kernel for an already selected scan.
    fn paged_attention(
        invocation: PagedAttentionOperands<'_, Self::Tensor>,
        geometry: PagedAttentionGeometry,
        queue: &Self::Queue,
    ) -> Result<impl CacheAttentionAccumulator<Self::Tensor, Self::Queue>, BackendFailure>;

    /// Prepares relative attention for an already selected scan.
    fn paged_relative_attention(
        input: &eredu_nn::RelativeAttentionInput<'_, Self::Value>,
        geometry: PagedAttentionGeometry,
        queue: &Self::Queue,
    ) -> Result<impl CacheAttentionAccumulator<Self::Tensor, Self::Queue>, BackendFailure>;

    /// Reads one query's native index and validity tensors into host positions.
    fn indexed_positions(
        input: &eredu_nn::IndexedAttentionInput<'_, Self::Value>,
        batch: i32,
        query: i32,
        queue: &Self::Queue,
    ) -> Result<Vec<Option<i32>>, BackendFailure>;

    /// Computes and completes one indexed attention row with exact gathered values.
    fn attend_indexed_row(
        input: &eredu_nn::IndexedAttentionInput<'_, Self::Value>,
        batch: i32,
        query: i32,
        slots: &[Option<i32>],
        gathered: Option<[Self::Tensor; 2]>,
        queue: &Self::Queue,
    ) -> Result<Self::Tensor, BackendFailure>;
}

/// Native comparison kernels for exact cache visibility ranges.
pub trait PoolingCacheStorage: ResidentKeyValueStorage + PagedKeyValueStorage {
    /// Computes visibility for local keys within the admitted causal window.
    fn local_visibility(
        geometry: eredu_nn::attention_mask::LocalWindowVisibility,
        queue: &Self::Queue,
    ) -> Result<Self::Value, BackendFailure>;

    /// Compares pooled positions to their complete source-window frontier.
    fn pooled_visibility(
        geometry: eredu_nn::attention_mask::PooledWindowVisibility,
        queue: &Self::Queue,
    ) -> Result<Self::Tensor, BackendFailure>;
}

//! Portable prompt-cache publication and restoration over native state transfers.
use super::*;
use eredu_core::{
    cache::{
        PromptCacheDescriptor, PromptCacheOptions, PromptCacheRead, PromptCacheStore,
        PromptCacheStoreError, PromptCacheStreamFrontier, StateTensorOwner, StateTensorRole,
    },
    BackendFailure,
};
use std::sync::Arc;

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
    pub fn new(shape: Vec<i32>, dtype: String) -> Result<Self, PromptCacheStoreError> {
        let width = match dtype.as_str() {
            "Bool" | "Uint8" | "Int8" => 1u64,
            "Uint16" | "Int16" | "Float16" | "Bfloat16" => 2,
            "Uint32" | "Int32" | "Float32" => 4,
            "Uint64" | "Int64" | "Float64" => 8,
            _ => return Err(invalid("unsupported prompt-cache dtype")),
        };
        let bytes = shape
            .iter()
            .try_fold(width, |n, d| {
                u64::try_from(*d).ok().and_then(|d| n.checked_mul(d))
            })
            .ok_or_else(|| invalid("invalid prompt-cache tensor extent"))?;
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
    pub unsealed_streams: Vec<eredu_core::cache::CacheStreamId>,
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
}
fn invalid(message: &str) -> PromptCacheStoreError {
    PromptCacheError::Malformed(message.into()).into()
}
/// Canonical names for the neutral cache representation.
pub fn prompt_cache_array_names(
    representation: CacheRepresentation,
) -> (&'static str, &'static str) {
    match representation {
        CacheRepresentation::KeyValue => ("keys", "values"),
        CacheRepresentation::AppendStream { .. } => ("records", "reserved"),
        CacheRepresentation::CompressedLatentRotary => ("latent", "rotary_key"),
    }
}
fn encode(
    layouts: &[&PromptCacheTensorLayout],
    names: &[&str],
    payload: &impl PromptCachePayload,
) -> Result<Vec<u8>, PromptCacheStoreError> {
    use safetensors::{
        tensor::{serialize, TensorView},
        Dtype,
    };
    let bytes = payload.bytes().map_err(PromptCacheStoreError::Storage)?;
    if bytes.len() != layouts.len() {
        return Err(invalid(
            "native cache transfer tensor count differs from inventory",
        ));
    }
    let dtypes = [
        Dtype::BOOL,
        Dtype::U8,
        Dtype::U16,
        Dtype::U32,
        Dtype::U64,
        Dtype::I8,
        Dtype::I16,
        Dtype::I32,
        Dtype::I64,
        Dtype::F16,
        Dtype::BF16,
        Dtype::F32,
        Dtype::F64,
    ];
    let mut views = Vec::with_capacity(bytes.len());
    for ((layout, bytes), name) in layouts.iter().zip(bytes).zip(names) {
        if bytes.len() as u64 != layout.bytes {
            return Err(invalid(
                "native cache transfer length differs from inventory",
            ));
        }
        let dtype = dtypes
            .iter()
            .find(|d| stored_dtype_name(**d) == layout.dtype)
            .copied()
            .ok_or_else(|| invalid("unsupported cache tensor dtype"))?;
        let shape = layout
            .shape
            .iter()
            .map(|n| usize::try_from(*n))
            .collect::<Result<Vec<_>, _>>()
            .map_err(PromptCacheStoreError::from_error)?;
        let view =
            TensorView::new(dtype, shape, bytes).map_err(PromptCacheStoreError::from_error)?;
        views.push((*name, view));
    }
    serialize(views, None).map_err(PromptCacheStoreError::from_error)
}
/// Prices portable encoding and metadata for a native isolated-state bound.
/// The publisher separately admits its exact inventory-based staging reservation.
pub fn prompt_cache_export_allowance(native_state_bytes: u64) -> Option<u64> {
    native_state_bytes
        .checked_mul(3)
        .and_then(|n| n.checked_div(2))
        .and_then(|n| n.checked_add(1024 * 1024))
}
/// Reserves staging, serializes exact native bytes, validates and atomically publishes.
/// No backend chooses shard names, constructs a manifest or publishes a store.
pub fn publish_prompt_cache(
    source: &impl PromptCacheExport,
    store: &dyn PromptCacheStore,
    descriptor: PromptCacheDescriptor,
    prefix: &[u32],
    options: &PromptCacheOptions,
) -> Result<PromptCacheManifest, PromptCacheStoreError> {
    descriptor.validate()?;
    let inventory = source.inventory();
    let payload = inventory
        .blocks
        .iter()
        .flat_map(|b| b.tensors.iter())
        .chain(inventory.states.iter().map(|s| &s.tensor))
        .try_fold(0u64, |n, t| n.checked_add(t.bytes))
        .ok_or_else(|| invalid("cache payload byte overflow"))?;
    let mut stream_frontiers = Vec::new();
    for (index, policy) in descriptor.layer_layout().iter().enumerate() {
        let layer = descriptor.global_layer_start() + index;
        for stream in policy.append_streams() {
            for lane in 0..descriptor.batch_size() {
                let lane = u32::try_from(lane).map_err(PromptCacheStoreError::from_error)?;
                let representation = CacheRepresentation::AppendStream {
                    slot: stream.slot(),
                    lane,
                };
                if inventory
                    .unsealed_streams
                    .contains(&eredu_core::cache::CacheStreamId::new(
                        layer,
                        representation,
                    ))
                {
                    return Err(invalid(
                        "append-stream tail must be sealed before cache export",
                    ));
                }
                let records = inventory
                    .blocks
                    .iter()
                    .filter(|b| b.id.global_layer == layer && b.id.representation == representation)
                    .map(|b| usize::try_from(b.id.end))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(PromptCacheStoreError::from_error)?
                    .into_iter()
                    .max()
                    .unwrap_or(0);
                stream_frontiers.push(PromptCacheStreamFrontier {
                    global_layer: layer,
                    slot: stream.slot(),
                    lane,
                    records,
                });
            }
        }
    }
    stream_frontiers.sort_by_key(|f| (f.global_layer, f.slot, f.lane));
    let reserved = payload
        .checked_mul(2)
        .and_then(|n| n.checked_add(inventory.transfer_bytes))
        .and_then(|n| {
            n.checked_add((inventory.blocks.len() + inventory.states.len()) as u64 * 4096)
        })
        .ok_or_else(|| invalid("cache staging byte overflow"))?;
    let mut publication = store.begin(reserved, options)?;
    let mut blocks = Vec::with_capacity(inventory.blocks.len());
    for (index, block) in inventory.blocks.iter().enumerate() {
        let shard = format!("block-{index:08}.safetensors");
        let names = prompt_cache_array_names(block.id.representation);
        let encoded = encode(
            &[&block.tensors[0], &block.tensors[1]],
            &[names.0, names.1],
            &source
                .transfer_block(index)
                .map_err(PromptCacheStoreError::Storage)?,
        )?;
        let hash = publication.put_shard(&shard, encoded)?;
        blocks.push(PromptCacheBlock {
            global_layer: block.id.global_layer,
            representation: block.id.representation,
            start: block.id.start,
            end: block.id.end,
            rank: block.id.rank,
            shard,
            first_array: names.0.into(),
            second_array: names.1.into(),
            first_shape: block.tensors[0].shape.clone(),
            second_shape: block.tensors[1].shape.clone(),
            first_dtype: block.tensors[0].dtype.clone(),
            second_dtype: block.tensors[1].dtype.clone(),
            logical_bytes: block.tensors[0]
                .bytes
                .checked_add(block.tensors[1].bytes)
                .ok_or_else(|| invalid("block bytes overflow"))?,
            payload_sha256: hash,
        });
    }
    let mut states = Vec::with_capacity(inventory.states.len());
    for (index, state) in inventory.states.iter().enumerate() {
        let shard = format!("state-{index:08}.safetensors");
        let encoded = encode(
            &[&state.tensor],
            &["state"],
            &source
                .transfer_state(index)
                .map_err(PromptCacheStoreError::Storage)?,
        )?;
        let hash = publication.put_shard(&shard, encoded)?;
        states.push(PromptCacheStateTensor {
            owner: state.owner,
            role: state.role,
            shard,
            array: "state".into(),
            shape: state.tensor.shape.clone(),
            dtype: state.tensor.dtype.clone(),
            logical_bytes: state.tensor.bytes,
            payload_sha256: hash,
        });
    }
    let manifest = PromptCacheManifest {
        schema_version: PROMPT_CACHE_SCHEMA_VERSION,
        model_family: descriptor.model_family().into(),
        effective_model_type: descriptor.effective_model_type().into(),
        checkpoint_fingerprint: descriptor.checkpoint_fingerprint().into(),
        prefix_content_fingerprint: descriptor.prefix_content_fingerprint().into(),
        architecture_fingerprint: descriptor.architecture_fingerprint().into(),
        layer_count: descriptor.layer_count(),
        global_layer_start: descriptor.global_layer_start(),
        global_layer_end: descriptor.global_layer_end(),
        block_size_tokens: inventory.block_size_tokens,
        batch_size: descriptor.batch_size(),
        total_prefix_tokens: prefix.len(),
        prefix_sha256: eredu_core::cache::prompt_cache_token_fingerprint(prefix),
        layer_layout: descriptor.layer_layout().clone(),
        layer_prefix_offsets: descriptor.layer_prefix_offsets().to_vec(),
        state_segments: descriptor.state_segments().to_vec(),
        sink_tokens: descriptor.sink_tokens(),
        topology: descriptor.topology().clone(),
        distributed_commit: descriptor.distributed_commit(),
        application_namespace: options.application_namespace().map(str::to_owned),
        blocks,
        state_tensors: states,
        stream_frontiers,
    };
    manifest.validate_compatibility(&descriptor, prefix)?;
    publication.commit(&manifest)?;
    Ok(manifest)
}

/// Native allocation and transfer binding used by portable restore orchestration.
pub trait PromptCacheBlockRestore {
    /// Retains an immutable file handle for the existing lazy paging mechanism.
    fn retain_block(
        &mut self,
        block: &PromptCacheBlock,
        source: Arc<RetainedCacheShard>,
        path: PathBuf,
    ) -> Result<(), BackendFailure>;
    /// Completes host-to-native restoration of a bounded, validated block.
    fn transfer_block(
        &mut self,
        block: &PromptCacheBlock,
        source: &RetainedCacheShard,
    ) -> Result<(), BackendFailure>;
}
/// Binds a fixed/recurrent tensor to native allocation and completion operations.
pub trait PromptCacheStateRestore {
    /// Allocates native fixed state and completes its exact byte transfer.
    fn transfer_state(
        &mut self,
        state: &PromptCacheStateTensor,
        source: &RetainedCacheShard,
    ) -> Result<(), BackendFailure>;
}
/// A validated, pinned input generation. Store-medium policy stays in the runtime.
pub struct PromptCacheRestore<'a> {
    read: &'a dyn PromptCacheRead,
}
impl<'a> PromptCacheRestore<'a> {
    /// Validates compatibility and pins the admitted store generation.
    pub fn new(
        read: &'a dyn PromptCacheRead,
        expected: &PromptCacheDescriptor,
        model: &eredu_core::cache::PromptCacheModelIdentity,
        prefix: &[u32],
        block_size_tokens: i32,
    ) -> Result<Self, PromptCacheStoreError> {
        eredu_core::cache::validate_prompt_cache_model_identity(expected, model)?;
        read.manifest().validate_compatibility(expected, prefix)?;
        if read.manifest().block_size_tokens != block_size_tokens {
            return Err(PromptCacheError::Incompatible(
                "cache block size differs from selected storage".into(),
            )
            .into());
        }
        store::validate_artifact(read)?;
        Ok(Self { read })
    }
    /// Returns the admitted immutable manifest.
    pub fn manifest(&self) -> &PromptCacheManifest {
        self.read.manifest()
    }
    /// Drives native restoration using the store-independent block catalog.
    pub fn restore_blocks(
        &self,
        target: &mut impl PromptCacheBlockRestore,
    ) -> Result<(), PromptCacheStoreError> {
        for block in &self.manifest().blocks {
            let source = Arc::new(
                RetainedCacheShard::from_block_reader(
                    Path::new(&block.shard),
                    self.read.open_shard(&block.shard)?,
                    block,
                )
                .map_err(PromptCacheStoreError::from_error)?,
            );
            if let Some(path) = self.read.shard_path(&block.shard) {
                target.retain_block(block, source, path)
            } else {
                target.transfer_block(block, &source)
            }
            .map_err(PromptCacheStoreError::Storage)?;
        }
        Ok(())
    }
}
/// Restores fixed state from the same pinned generation; geometry and payload
/// validation apply identically to memory and directory stores.
pub fn restore_prompt_cache_states(
    read: &dyn PromptCacheRead,
    manifest: &PromptCacheManifest,
    target: &mut impl PromptCacheStateRestore,
) -> Result<(), PromptCacheStoreError> {
    if read.manifest() != manifest {
        return Err(invalid(
            "state restore generation differs from admitted manifest",
        ));
    }
    for state in &manifest.state_tensors {
        let source = RetainedCacheShard::from_state_reader(
            Path::new(&state.shard),
            read.open_shard(&state.shard)?,
            state,
        )
        .map_err(PromptCacheStoreError::from_error)?;
        target
            .transfer_state(state, &source)
            .map_err(PromptCacheStoreError::Storage)?;
    }
    Ok(())
}

/// Applies canonical rank scoping without exposing store-medium decisions to a
/// native binding. Process-local stores retain their existing ownership scope.
pub fn with_rank_prompt_cache_store<T>(
    store: &dyn PromptCacheStore,
    topology: &eredu_core::cache::PromptCacheTopology,
    operation: impl FnOnce(&dyn PromptCacheStore) -> T,
) -> T {
    let scoped = store.directory().map(|root| {
        super::DirectoryPromptCacheStore::new(
            super::prompt_cache_rank_path(root, topology),
            store.max_bytes(),
        )
    });
    operation(
        scoped
            .as_ref()
            .map_or(store, |store| store as &dyn PromptCacheStore),
    )
}

/// Portable staging/publication owner for the existing all-rank commit protocol.
/// Native mechanisms supply only a completed state export into the supplied store.
pub struct PreparedPromptCachePublication {
    publication: super::ReversiblePromptCachePublication,
    manifest: PromptCacheManifest,
}
impl PreparedPromptCachePublication {
    /// Stages one rank's export without replacing its active generation.
    pub fn stage(
        destination: &Path,
        topology: &eredu_core::cache::PromptCacheTopology,
        options: &PromptCacheOptions,
        export: impl FnOnce(
            &dyn PromptCacheStore,
            &PromptCacheOptions,
        ) -> Result<PromptCacheManifest, BackendFailure>,
    ) -> Result<Self, PromptCacheStoreError> {
        let destination = super::prompt_cache_rank_path(destination, topology);
        let publication = super::ReversiblePromptCachePublication::begin(
            &destination,
            options.replace_existing(),
        )
        .map_err(PromptCacheStoreError::from_error)?;
        let staging = super::DirectoryPromptCacheStore::new(
            publication.staging_destination(),
            options.capture_limit().unwrap_or(u64::MAX),
        );
        let staging_options =
            PromptCacheOptions::new(options.application_namespace().map(str::to_owned), false)?;
        let manifest =
            export(&staging, &staging_options).map_err(PromptCacheStoreError::Storage)?;
        Ok(Self {
            publication,
            manifest,
        })
    }
    /// Returns the staged manifest used by distributed compatibility agreement.
    pub fn manifest(&self) -> &PromptCacheManifest {
        &self.manifest
    }
    /// Provisionally publishes; the all-rank driver still owns commit/rollback.
    pub fn publish(&mut self) -> Result<(), PromptCacheStoreError> {
        self.publication
            .publish()
            .map_err(PromptCacheStoreError::from_error)
    }
    /// Finalizes an agreed publication. Failure violates the completed commit protocol.
    pub fn commit(self) {
        self.publication
            .commit()
            .unwrap_or_else(|e| panic!("committed prompt-cache publication cleanup failed: {e}"));
    }
    /// Restores the previous generation after an agreed abort.
    pub fn rollback(self) {
        self.publication
            .rollback()
            .unwrap_or_else(|e| panic!("prompt-cache publication rollback failed: {e}"));
    }
}

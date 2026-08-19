//! Backend-neutral checkpoint storage and encoded tensor leases.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    ops::Range,
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex, MutexGuard},
};

use memmap2::{Mmap, MmapOptions};
use safetensors::{
    tensor::{Dtype, Metadata, TensorInfo},
    SafeTensors,
};
use serde::{de::MapAccess, Deserialize, Deserializer};

use crate::StoredDtype;

/// Catalog metadata for one logical checkpoint tensor.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TensorMetadata {
    /// Stable logical checkpoint name.
    pub name: String,
    /// Logical tensor shape.
    pub logical_shape: Vec<usize>,
    /// Physical encoded shape when it differs from the logical tensor.
    pub physical_shape: Vec<usize>,
    /// On-disk scalar or packed encoding.
    pub stored_dtype: StoredDtype,
    /// Number of bytes in the complete encoded payload.
    pub encoded_byte_len: u64,
    /// Payload shard backing this tensor, when file-backed.
    pub backing_shard: Option<PathBuf>,
}

/// A requested logical subset of a checkpoint tensor.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum TensorSelection {
    /// Selects the complete tensor.
    Full,
    /// Selects a non-empty contiguous range on one axis.
    Range {
        /// Selected axis.
        axis: usize,
        /// Inclusive start coordinate.
        start: usize,
        /// Exclusive end coordinate.
        end: usize,
    },
    /// Selects ordered indices on one axis.
    Indices {
        /// Selected axis.
        axis: usize,
        /// Non-empty source indices in output order.
        indices: Vec<usize>,
    },
    /// Selects one physically contiguous row-major scalar span.
    Contiguous {
        /// Scalar offset from the logical tensor start.
        offset_elements: usize,
        /// Non-empty output geometry.
        shape: Vec<usize>,
    },
}

/// Whether a selected tensor may decode or read its complete source.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ReadPolicy {
    /// Acquisition must physically restrict payload I/O to the selection.
    RequireBounded,
    /// Explicit tooling may read the complete tensor before selection.
    AllowFullTensorRead,
}

/// One neutral tensor acquisition request.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TensorReadRequest {
    /// Logical checkpoint tensor name.
    pub key: String,
    /// Requested logical selection.
    pub selection: TensorSelection,
    /// Required physical I/O behavior.
    pub policy: ReadPolicy,
}

/// Proof recorded by a lease about the physical read it performed.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BoundedReadProof {
    /// Whether physical payload I/O was restricted to the requested selection.
    pub physically_bounded: bool,
    /// Physical payload byte offset relative to the tensor payload.
    pub offset_bytes: u64,
    /// Physical payload bytes read or mapped for this lease.
    pub length_bytes: u64,
}

/// Format-native encoded payload retained by a checkpoint lease.
pub trait EncodedTensorLease: Send + Sync + 'static {
    /// Returns complete catalog metadata.
    fn metadata(&self) -> &TensorMetadata;
    /// Returns the requested logical selection.
    fn selection(&self) -> &TensorSelection;
    /// Returns the logical output shape after selection.
    fn output_shape(&self) -> &[usize];
    /// Returns proof of bounded-read behavior.
    fn bounded_read_proof(&self) -> &BoundedReadProof;
    /// Returns the backing shard path, if the lease is file-backed.
    fn backing_path(&self) -> Option<&Path>;
    /// Returns the exact retained byte span when directly byte-addressable.
    fn encoded_bytes(&self) -> Option<&[u8]>;
}

/// Type-erased neutral lease covering the checkpoint formats supported by Eredu.
#[derive(Debug, Clone)]
pub enum CheckpointLease {
    /// Memory-mapped SafeTensors bytes.
    Safetensors(SafetensorsLease),
    /// Lazily read portable GGUF payload.
    Gguf(crate::gguf_store::GgufLease),
}

impl EncodedTensorLease for CheckpointLease {
    fn metadata(&self) -> &TensorMetadata {
        match self {
            Self::Safetensors(lease) => lease.metadata(),
            Self::Gguf(lease) => lease.metadata(),
        }
    }

    fn selection(&self) -> &TensorSelection {
        match self {
            Self::Safetensors(lease) => lease.selection(),
            Self::Gguf(lease) => lease.selection(),
        }
    }

    fn output_shape(&self) -> &[usize] {
        match self {
            Self::Safetensors(lease) => lease.output_shape(),
            Self::Gguf(lease) => lease.output_shape(),
        }
    }

    fn bounded_read_proof(&self) -> &BoundedReadProof {
        match self {
            Self::Safetensors(lease) => lease.bounded_read_proof(),
            Self::Gguf(lease) => lease.bounded_read_proof(),
        }
    }

    fn backing_path(&self) -> Option<&Path> {
        match self {
            Self::Safetensors(lease) => lease.backing_path(),
            Self::Gguf(lease) => lease.backing_path(),
        }
    }

    fn encoded_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Safetensors(lease) => lease.encoded_bytes(),
            Self::Gguf(lease) => lease.encoded_bytes(),
        }
    }
}

/// Object-safe cold-path checkpoint source used by generic materializers.
pub trait CheckpointSource: Send + Sync {
    /// Returns all logical catalog keys in deterministic order.
    fn source_keys(&self) -> Vec<String>;
    /// Returns metadata without reading tensor payloads.
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError>;
    /// Acquires a format-preserving encoded lease.
    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError>;
    /// Returns deterministic storage diagnostics.
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError>;

    /// Returns physical source keys consumed to synthesize overlay bindings.
    fn materialized_source_keys(&self) -> Vec<String> {
        Vec::new()
    }

    /// Returns catalog keys admitted but not claimed by a resolved contract.
    fn unclaimed_checkpoint_keys(&self) -> Vec<String> {
        Vec::new()
    }

    /// Returns whether an overlay key supersedes a source-side semantic recipe.
    fn is_authoritative_materialized_key(&self, _key: &str) -> bool {
        false
    }

    /// Returns whether this source is restricted by a resolved contract.
    fn is_checkpoint_contract_resolved(&self) -> bool {
        false
    }
}

/// Shared ownership of one backend-neutral checkpoint source.
pub type SharedCheckpointSource = Arc<dyn CheckpointSource>;

/// A checkpoint source restricted to one resolved architecture contract.
///
/// The wrapper is cold-path policy only: it filters catalog inspection and
/// rejects every lease request not selected by the resolved physical layout.
pub struct ResolvedCheckpointSource {
    source: Arc<dyn CheckpointSource>,
    contract: crate::validation::ResolvedCheckpointPlan,
}

impl ResolvedCheckpointSource {
    /// Restricts a source to the physical keys selected by a contract.
    pub fn new(
        source: Arc<dyn CheckpointSource>,
        contract: crate::validation::ResolvedCheckpointPlan,
    ) -> Self {
        Self { source, contract }
    }

    /// Returns the resolved contract identity.
    pub fn contract_identity(&self) -> &str {
        self.contract.identity()
    }

    /// Returns catalog keys admitted but not claimed by the selected layout.
    pub fn unclaimed_keys(&self) -> &BTreeSet<String> {
        self.contract.unclaimed_keys()
    }

    fn authorize(&self, key: &str) -> Result<(), StoreError> {
        if self.contract.source_keys().contains(key) {
            Ok(())
        } else {
            Err(StoreError::UnauthorizedTensor {
                contract: self.contract.identity().to_owned(),
                key: key.to_owned(),
            })
        }
    }
}

impl CheckpointSource for ResolvedCheckpointSource {
    fn source_keys(&self) -> Vec<String> {
        self.source
            .source_keys()
            .into_iter()
            .filter(|key| {
                self.contract.source_keys().contains(key)
                    || self.source.is_authoritative_materialized_key(key)
            })
            .collect()
    }

    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        if !self.source.is_authoritative_materialized_key(key) {
            self.authorize(key)?;
        }
        self.source.source_metadata(key)
    }

    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        if !self.source.is_authoritative_materialized_key(&request.key) {
            self.authorize(&request.key)?;
        }
        self.source.acquire_lease(request)
    }

    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        self.source.source_diagnostics()
    }

    fn materialized_source_keys(&self) -> Vec<String> {
        self.source
            .materialized_source_keys()
            .into_iter()
            .filter(|key| self.contract.source_keys().contains(key))
            .collect()
    }

    fn unclaimed_checkpoint_keys(&self) -> Vec<String> {
        self.contract.unclaimed_keys().iter().cloned().collect()
    }

    fn is_authoritative_materialized_key(&self, key: &str) -> bool {
        self.source.is_authoritative_materialized_key(key)
    }

    fn is_checkpoint_contract_resolved(&self) -> bool {
        true
    }
}

/// Persistent checkpoint storage contract with a concrete lease type.
pub trait WeightStore {
    /// Encoded lease retaining the source lifetime.
    type Lease: EncodedTensorLease;

    /// Returns all catalog keys in deterministic order.
    fn keys(&self) -> Vec<String>;
    /// Returns metadata without reading the tensor payload.
    fn metadata(&self, key: &str) -> Result<TensorMetadata, StoreError>;
    /// Acquires an encoded tensor lease under an explicit read policy.
    fn acquire(&self, request: TensorReadRequest) -> Result<Self::Lease, StoreError>;
    /// Returns a deterministic diagnostics snapshot.
    fn diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError>;
}

/// Storage format represented by a diagnostics snapshot.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum WeightStoreBackend {
    /// Memory-mapped SafeTensors payload shards.
    Safetensors,
    /// Seekable GGUF payload shards.
    Gguf,
    /// Immutable in-memory encoded data.
    Memory,
}

/// Deterministic checkpoint storage statistics.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct WeightStoreDiagnostics {
    /// Storage format.
    pub backend: WeightStoreBackend,
    /// Successful acquisitions reusing an existing mapping or reader.
    pub mapping_hits: u64,
    /// Acquisitions opening a new mapping or reader.
    pub mapping_misses: u64,
    /// Unleased mappings or readers removed to honor a bound.
    pub evictions: u64,
    /// Mappings or readers currently retained by the store.
    pub currently_mapped_shards: usize,
    /// Shard paths touched so far in stable order.
    pub touched_shard_paths: Vec<PathBuf>,
    /// Physical tensor or selected-region reads.
    pub physical_reads: u64,
    /// Encoded payload bytes requested by physical reads.
    pub physical_read_bytes: u64,
    /// Logical outputs served from a previously converted physical group.
    pub coalesced_group_hits: u64,
}

/// Structured neutral checkpoint store failures.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// The configured mapped-shard or reader limit was zero.
    #[error("maximum mapped-shard count must be nonzero")]
    InvalidMappedShardLimit,
    /// A requested tensor is absent.
    #[error("unknown checkpoint tensor {key:?}")]
    UnknownTensor {
        /// Requested logical name.
        key: String,
    },
    /// A resolved architecture contract did not authorize the requested key.
    #[error("checkpoint contract {contract:?} does not authorize tensor {key:?}")]
    UnauthorizedTensor {
        /// Resolved contract identity.
        contract: String,
        /// Rejected tensor key.
        key: String,
    },
    /// A checkpoint path or indexed payload shard is absent.
    #[error("checkpoint shard does not exist: {path}", path = .path.display())]
    MissingShard {
        /// Missing checkpoint or payload path.
        path: PathBuf,
    },
    /// A SafeTensors index could not be decoded or validated.
    #[error("malformed safetensors index {path}: {message}", path = .path.display())]
    MalformedIndex {
        /// Index path.
        path: PathBuf,
        /// Decoder or validation detail.
        message: String,
    },
    /// A SafeTensors payload header or contents are invalid.
    #[error("malformed safetensors shard {path}: {message}", path = .path.display())]
    MalformedSafetensors {
        /// Payload path.
        path: PathBuf,
        /// Parser detail.
        message: String,
    },
    /// An indexed shard path is absolute or escapes the checkpoint root.
    #[error("unsafe safetensors shard path {path}", path = .path.display())]
    UnsafeShardPath {
        /// Rejected path.
        path: PathBuf,
    },
    /// An index maps a tensor to a shard that does not contain it.
    #[error("index maps tensor {key:?} to {path}, but that shard does not contain it", path = .path.display())]
    ContradictoryIndexMapping {
        /// Tensor key from the index.
        key: String,
        /// Referenced payload shard.
        path: PathBuf,
    },
    /// A requested selection is invalid.
    #[error("invalid selection for tensor {key:?}: {message}")]
    InvalidSelection {
        /// Selected tensor name.
        key: String,
        /// Validation detail.
        message: String,
    },
    /// Required bounded physical I/O cannot be honored.
    #[error("bounded selection is unavailable for tensor {key:?}: {message}")]
    BoundedSelectionUnavailable {
        /// Selected tensor name.
        key: String,
        /// Backend planning detail.
        message: String,
    },
    /// Checked size arithmetic overflowed.
    #[error("checkpoint size overflow: {context}")]
    Overflow {
        /// Calculation that overflowed.
        context: String,
    },
    /// Every cache entry is pinned by a live lease.
    #[error("checkpoint mapping capacity {maximum} is exhausted; leased shards: {leased:?}")]
    CapacityExhausted {
        /// Configured mapping bound.
        maximum: usize,
        /// Deterministically ordered pinned paths.
        leased: Vec<PathBuf>,
    },
    /// Filesystem or container access failed.
    #[error("checkpoint I/O failed for {path}: {message}", path = .path.display())]
    Io {
        /// Affected path.
        path: PathBuf,
        /// Stable failure detail.
        message: String,
    },
    /// The catalog or mapping cache is internally unavailable.
    #[error("checkpoint store state is unavailable: {0}")]
    Internal(String),
    /// A GGUF catalog, selection, or payload-read operation failed.
    #[error("GGUF checkpoint operation failed for tensor {key:?}: {message}")]
    Gguf {
        /// Logical tensor involved, or an empty string for store-wide failures.
        key: String,
        /// Portable GGUF error detail.
        message: String,
    },
}

/// Default maximum number of simultaneously retained mapped shards.
pub const DEFAULT_MAX_MAPPED_SHARDS: usize = 4;

#[derive(Debug)]
struct MappedShard {
    path: PathBuf,
    mmap: Mmap,
    metadata: Metadata,
    payload_offset: usize,
}

#[derive(Debug)]
struct CacheEntry {
    shard: Arc<MappedShard>,
    last_used: u64,
}

#[derive(Debug, Default)]
struct CacheState {
    entries: BTreeMap<PathBuf, CacheEntry>,
    touched: BTreeSet<PathBuf>,
    tick: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

#[derive(Debug, Clone)]
struct CatalogEntry {
    shard: PathBuf,
    indexed: bool,
}

/// Encoded SafeTensors selection retaining its backing mmap.
#[derive(Debug, Clone)]
pub struct SafetensorsLease {
    metadata: TensorMetadata,
    selection: TensorSelection,
    output_shape: Vec<usize>,
    proof: BoundedReadProof,
    shard: Arc<MappedShard>,
    mapped_span: Range<usize>,
    selected_bytes: Option<Arc<[u8]>>,
}

impl EncodedTensorLease for SafetensorsLease {
    fn metadata(&self) -> &TensorMetadata {
        &self.metadata
    }

    fn selection(&self) -> &TensorSelection {
        &self.selection
    }

    fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    fn bounded_read_proof(&self) -> &BoundedReadProof {
        &self.proof
    }

    fn backing_path(&self) -> Option<&Path> {
        Some(&self.shard.path)
    }

    fn encoded_bytes(&self) -> Option<&[u8]> {
        match &self.selected_bytes {
            Some(bytes) => Some(bytes),
            None => self.shard.mmap.get(self.mapped_span.clone()),
        }
    }
}

/// Persistent neutral SafeTensors catalog with bounded mmap ownership.
#[derive(Debug)]
pub struct SafetensorsWeightStore {
    canonical_root: PathBuf,
    catalog: BTreeMap<String, CatalogEntry>,
    metadata: Mutex<BTreeMap<String, TensorMetadata>>,
    cache: Mutex<CacheState>,
    max_mapped_shards: usize,
}

impl SafetensorsWeightStore {
    /// Opens a file, indexed directory, or directory containing `model.safetensors`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        Self::open_with_max_mapped_shards(path, DEFAULT_MAX_MAPPED_SHARDS)
    }

    /// Opens a checkpoint with an explicit nonzero mapping bound.
    pub fn open_with_max_mapped_shards(
        path: impl AsRef<Path>,
        max_mapped_shards: usize,
    ) -> Result<Self, StoreError> {
        if max_mapped_shards == 0 {
            return Err(StoreError::InvalidMappedShardLimit);
        }
        let path = path.as_ref();
        if !path.exists() {
            return Err(StoreError::MissingShard {
                path: path.to_path_buf(),
            });
        }
        if path.is_dir() {
            let root = path.to_path_buf();
            let canonical_root = canonical_checkpoint_access_root(path)?;
            let index_path = root.join("model.safetensors.index.json");
            if index_path.exists() {
                let raw = std::fs::read_to_string(&index_path)
                    .map_err(|error| io_error(&index_path, error))?;
                let index: SafetensorsIndex =
                    serde_json::from_str(&raw).map_err(|error| StoreError::MalformedIndex {
                        path: index_path.clone(),
                        message: error.to_string(),
                    })?;
                if index.weight_map.0.is_empty() {
                    return Err(StoreError::MalformedIndex {
                        path: index_path,
                        message: "weight_map must not be empty".into(),
                    });
                }
                let mut catalog = BTreeMap::new();
                for (key, relative) in index.weight_map.0 {
                    if key.is_empty() {
                        return Err(StoreError::MalformedIndex {
                            path: index_path.clone(),
                            message: "tensor names must not be empty".into(),
                        });
                    }
                    let relative = validate_relative_shard_path(Path::new(&relative))?;
                    catalog.insert(
                        key,
                        CatalogEntry {
                            shard: root.join(relative),
                            indexed: true,
                        },
                    );
                }
                return Ok(Self {
                    canonical_root,
                    catalog,
                    metadata: Mutex::new(BTreeMap::new()),
                    cache: Mutex::new(CacheState::default()),
                    max_mapped_shards,
                });
            }
            return Self::from_single_file(
                root.join("model.safetensors"),
                canonical_root,
                max_mapped_shards,
            );
        }
        let file = path.to_path_buf();
        let root = file
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        Self::from_single_file(file, canonicalize(&root)?, max_mapped_shards)
    }

    fn from_single_file(
        file: PathBuf,
        canonical_root: PathBuf,
        max_mapped_shards: usize,
    ) -> Result<Self, StoreError> {
        let discovered = inspect_file(&file)?;
        let catalog = discovered
            .keys()
            .map(|key| {
                (
                    key.clone(),
                    CatalogEntry {
                        shard: file.clone(),
                        indexed: false,
                    },
                )
            })
            .collect();
        Ok(Self {
            canonical_root,
            catalog,
            metadata: Mutex::new(discovered),
            cache: Mutex::new(CacheState::default()),
            max_mapped_shards,
        })
    }

    fn lock_cache(&self) -> Result<MutexGuard<'_, CacheState>, StoreError> {
        self.cache
            .lock()
            .map_err(|_| StoreError::Internal("mapped-shard cache is poisoned".into()))
    }

    fn acquire_shard(&self, entry: &CatalogEntry) -> Result<Arc<MappedShard>, StoreError> {
        let canonical_path = self.validate_access_path(entry)?;
        let mut cache = self.lock_cache()?;
        cache.tick = cache.tick.saturating_add(1);
        let tick = cache.tick;
        if let Some(shard) = cache
            .entries
            .get(&canonical_path)
            .map(|entry| Arc::clone(&entry.shard))
        {
            cache.hits = cache.hits.saturating_add(1);
            cache.entries.get_mut(&canonical_path).unwrap().last_used = tick;
            return Ok(shard);
        }
        cache.misses = cache.misses.saturating_add(1);
        if cache.entries.len() >= self.max_mapped_shards {
            let victim = cache
                .entries
                .iter()
                .filter(|(_, candidate)| Arc::strong_count(&candidate.shard) == 1)
                .min_by(|(left_path, left), (right_path, right)| {
                    (left.last_used, *left_path).cmp(&(right.last_used, *right_path))
                })
                .map(|(path, _)| path.clone());
            if let Some(victim) = victim {
                cache.entries.remove(&victim);
                cache.evictions = cache.evictions.saturating_add(1);
            } else {
                return Err(StoreError::CapacityExhausted {
                    maximum: self.max_mapped_shards,
                    leased: cache
                        .entries
                        .values()
                        .map(|entry| entry.shard.path.clone())
                        .collect(),
                });
            }
        }
        let file = File::open(&canonical_path).map_err(|error| fs_error(&entry.shard, error))?;
        // SAFETY: leases retain the owning `MappedShard` for every exposed span.
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .map_err(|error| io_error(&entry.shard, error))?;
        let (header_len, metadata) = SafeTensors::read_metadata(&mmap).map_err(|error| {
            StoreError::MalformedSafetensors {
                path: entry.shard.clone(),
                message: error.to_string(),
            }
        })?;
        let payload_offset =
            8usize
                .checked_add(header_len)
                .ok_or_else(|| StoreError::Overflow {
                    context: format!("payload offset for {}", entry.shard.display()),
                })?;
        let shard = Arc::new(MappedShard {
            path: entry.shard.clone(),
            mmap,
            metadata,
            payload_offset,
        });
        cache.touched.insert(entry.shard.clone());
        cache.entries.insert(
            canonical_path,
            CacheEntry {
                shard: Arc::clone(&shard),
                last_used: tick,
            },
        );
        Ok(shard)
    }

    fn validate_access_path(&self, entry: &CatalogEntry) -> Result<PathBuf, StoreError> {
        let canonical = canonicalize(&entry.shard)?;
        if entry.indexed && !canonical.starts_with(&self.canonical_root) {
            return Err(StoreError::UnsafeShardPath {
                path: entry.shard.clone(),
            });
        }
        Ok(canonical)
    }

    fn metadata_from_shard(
        &self,
        key: &str,
        entry: &CatalogEntry,
        shard: &MappedShard,
    ) -> Result<TensorMetadata, StoreError> {
        if let Some(metadata) = self
            .metadata
            .lock()
            .map_err(|_| StoreError::Internal("metadata cache is poisoned".into()))?
            .get(key)
            .cloned()
        {
            return Ok(metadata);
        }
        shard
            .metadata
            .info(key)
            .ok_or_else(|| StoreError::ContradictoryIndexMapping {
                key: key.into(),
                path: entry.shard.clone(),
            })?;
        let mut discovered = BTreeMap::new();
        for name in shard.metadata.offset_keys() {
            if self
                .catalog
                .get(&name)
                .is_some_and(|candidate| candidate.shard == shard.path)
            {
                let info = shard.metadata.info(&name).expect("metadata key is present");
                discovered.insert(name.clone(), metadata_for_info(&name, &shard.path, info)?);
            }
        }
        let metadata = discovered
            .get(key)
            .cloned()
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })?;
        self.metadata
            .lock()
            .map_err(|_| StoreError::Internal("metadata cache is poisoned".into()))?
            .extend(discovered);
        Ok(metadata)
    }
}

impl WeightStore for SafetensorsWeightStore {
    type Lease = SafetensorsLease;

    fn keys(&self) -> Vec<String> {
        self.catalog.keys().cloned().collect()
    }

    fn metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        if let Some(metadata) = self
            .metadata
            .lock()
            .map_err(|_| StoreError::Internal("metadata cache is poisoned".into()))?
            .get(key)
            .cloned()
        {
            return Ok(metadata);
        }
        let entry = self
            .catalog
            .get(key)
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })?;
        let shard = self.acquire_shard(entry)?;
        self.metadata_from_shard(key, entry, &shard)
    }

    fn acquire(&self, request: TensorReadRequest) -> Result<Self::Lease, StoreError> {
        let entry = self
            .catalog
            .get(&request.key)
            .ok_or_else(|| StoreError::UnknownTensor {
                key: request.key.clone(),
            })?;
        let shard = self.acquire_shard(entry)?;
        let metadata = self.metadata_from_shard(&request.key, entry, &shard)?;
        let info = shard.metadata.info(&request.key).ok_or_else(|| {
            io_error(
                &entry.shard,
                format!("shard does not contain tensor {:?}", request.key),
            )
        })?;
        let output_shape =
            validate_selection(&request.key, &metadata.logical_shape, &request.selection)?;
        let payload_start = shard
            .payload_offset
            .checked_add(info.data_offsets.0)
            .ok_or_else(|| StoreError::Overflow {
                context: format!("payload start for {:?}", request.key),
            })?;
        let payload_end = shard
            .payload_offset
            .checked_add(info.data_offsets.1)
            .ok_or_else(|| StoreError::Overflow {
                context: format!("payload end for {:?}", request.key),
            })?;
        let payload = shard
            .mmap
            .get(payload_start..payload_end)
            .ok_or_else(|| io_error(&shard.path, "tensor payload is outside mapped shard"))?;
        let (relative_span, selected_bytes) = select_safetensors_bytes(
            &request.key,
            info.dtype,
            &info.shape,
            payload,
            &request.selection,
            &output_shape,
            request.policy,
        )?;
        let length = selected_bytes
            .as_ref()
            .map_or(relative_span.len(), |bytes| bytes.len());
        let mapped_span = payload_start + relative_span.start..payload_start + relative_span.end;
        let full_selection = matches!(request.selection, TensorSelection::Full);
        Ok(SafetensorsLease {
            metadata,
            selection: request.selection,
            output_shape,
            proof: BoundedReadProof {
                physically_bounded: matches!(request.policy, ReadPolicy::RequireBounded)
                    || full_selection,
                offset_bytes: u64::try_from(relative_span.start).map_err(|_| {
                    StoreError::Overflow {
                        context: "selection byte offset".into(),
                    }
                })?,
                length_bytes: u64::try_from(length).map_err(|_| StoreError::Overflow {
                    context: "selection byte length".into(),
                })?,
            },
            shard,
            mapped_span,
            selected_bytes: selected_bytes.map(Arc::from),
        })
    }

    fn diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        let cache = self.lock_cache()?;
        Ok(WeightStoreDiagnostics {
            backend: WeightStoreBackend::Safetensors,
            mapping_hits: cache.hits,
            mapping_misses: cache.misses,
            evictions: cache.evictions,
            currently_mapped_shards: cache.entries.len(),
            touched_shard_paths: cache.touched.iter().cloned().collect(),
            physical_reads: 0,
            physical_read_bytes: 0,
            coalesced_group_hits: 0,
        })
    }
}

impl CheckpointSource for SafetensorsWeightStore {
    fn source_keys(&self) -> Vec<String> {
        WeightStore::keys(self)
    }

    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        WeightStore::metadata(self, key)
    }

    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        WeightStore::acquire(self, request).map(CheckpointLease::Safetensors)
    }

    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        WeightStore::diagnostics(self)
    }
}

impl crate::validation::SafetensorsCatalog for SafetensorsWeightStore {
    fn keys(&self) -> Vec<String> {
        WeightStore::keys(self)
    }

    fn metadata(&self, key: &str) -> Result<crate::validation::CatalogTensorMetadata, String> {
        WeightStore::metadata(self, key)
            .map(|metadata| crate::validation::CatalogTensorMetadata {
                shape: metadata.logical_shape,
                stored_dtype: metadata.stored_dtype,
            })
            .map_err(|error| error.to_string())
    }
}

fn inspect_file(path: &Path) -> Result<BTreeMap<String, TensorMetadata>, StoreError> {
    let file = File::open(path).map_err(|error| fs_error(path, error))?;
    // SAFETY: all returned metadata is owned before this mapping is dropped.
    let mmap = unsafe { MmapOptions::new().map(&file) }.map_err(|error| io_error(path, error))?;
    let checkpoint =
        SafeTensors::deserialize(&mmap).map_err(|error| StoreError::MalformedSafetensors {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    checkpoint
        .iter()
        .map(|(key, view)| {
            metadata_for_parts(key, path, view.dtype(), view.shape(), view.data().len())
                .map(|metadata| (key.to_string(), metadata))
        })
        .collect()
}

fn metadata_for_info(
    key: &str,
    path: &Path,
    info: &TensorInfo,
) -> Result<TensorMetadata, StoreError> {
    let payload_len = info
        .data_offsets
        .1
        .checked_sub(info.data_offsets.0)
        .ok_or_else(|| io_error(path, format!("tensor {key:?} has descending offsets")))?;
    metadata_for_parts(key, path, info.dtype, &info.shape, payload_len)
}

fn metadata_for_parts(
    key: &str,
    path: &Path,
    dtype: Dtype,
    shape: &[usize],
    payload_len: usize,
) -> Result<TensorMetadata, StoreError> {
    let elements = checked_elements(key, shape)?;
    let bits = elements
        .checked_mul(dtype.bitsize())
        .ok_or_else(|| StoreError::Overflow {
            context: format!("encoded bit length for {key:?}"),
        })?;
    if !bits.is_multiple_of(8) || bits / 8 != payload_len {
        return Err(io_error(
            path,
            format!("tensor {key:?} payload contradicts metadata"),
        ));
    }
    Ok(TensorMetadata {
        name: key.into(),
        logical_shape: shape.to_vec(),
        physical_shape: shape.to_vec(),
        stored_dtype: stored_dtype_from_safetensors(dtype),
        encoded_byte_len: u64::try_from(payload_len).map_err(|_| StoreError::Overflow {
            context: format!("payload length for {key:?}"),
        })?,
        backing_shard: Some(path.to_path_buf()),
    })
}

pub(crate) fn validate_selection(
    key: &str,
    shape: &[usize],
    selection: &TensorSelection,
) -> Result<Vec<usize>, StoreError> {
    checked_elements(key, shape)?;
    let mut output = shape.to_vec();
    match selection {
        TensorSelection::Full => {}
        TensorSelection::Range { axis, start, end } => {
            let dimension = shape
                .get(*axis)
                .ok_or_else(|| invalid_selection(key, "axis outside rank"))?;
            if start >= end || *end > *dimension {
                return Err(invalid_selection(key, "range outside dimension"));
            }
            output[*axis] = end - start;
        }
        TensorSelection::Indices { axis, indices } => {
            let dimension = shape
                .get(*axis)
                .ok_or_else(|| invalid_selection(key, "axis outside rank"))?;
            if indices.is_empty() || indices.iter().any(|index| *index >= *dimension) {
                return Err(invalid_selection(
                    key,
                    "indices are empty or outside dimension",
                ));
            }
            output[*axis] = indices.len();
        }
        TensorSelection::Contiguous {
            offset_elements,
            shape: selected,
        } => {
            if selected.is_empty() || selected.contains(&0) {
                return Err(invalid_selection(key, "contiguous output shape is empty"));
            }
            let end = offset_elements
                .checked_add(checked_elements(key, selected)?)
                .ok_or_else(|| StoreError::Overflow {
                    context: format!("contiguous selection end for {key:?}"),
                })?;
            if end > checked_elements(key, shape)? {
                return Err(invalid_selection(key, "contiguous span outside tensor"));
            }
            output = selected.clone();
        }
    }
    checked_elements(key, &output)?;
    Ok(output)
}

fn select_safetensors_bytes(
    key: &str,
    dtype: Dtype,
    shape: &[usize],
    data: &[u8],
    selection: &TensorSelection,
    output_shape: &[usize],
    policy: ReadPolicy,
) -> Result<(Range<usize>, Option<Vec<u8>>), StoreError> {
    if matches!(selection, TensorSelection::Full) {
        return Ok((0..data.len(), None));
    }
    let bits = dtype.bitsize();
    let scalar_bytes = bits.checked_div(8).filter(|_| bits.is_multiple_of(8));
    if let (
        Some(scalar_bytes),
        TensorSelection::Contiguous {
            offset_elements,
            shape,
        },
    ) = (scalar_bytes, selection)
    {
        let start =
            offset_elements
                .checked_mul(scalar_bytes)
                .ok_or_else(|| StoreError::Overflow {
                    context: format!("contiguous byte start for {key:?}"),
                })?;
        let end = checked_elements(key, shape)?
            .checked_mul(scalar_bytes)
            .and_then(|length| start.checked_add(length))
            .ok_or_else(|| StoreError::Overflow {
                context: format!("contiguous byte end for {key:?}"),
            })?;
        return data
            .get(start..end)
            .map(|_| (start..end, None))
            .ok_or_else(|| invalid_selection(key, "contiguous byte span outside payload"));
    }
    if let (
        Some(_),
        TensorSelection::Range {
            axis: 0,
            start,
            end,
        },
    ) = (scalar_bytes, selection)
    {
        let row_bytes = data
            .len()
            .checked_div(shape[0])
            .filter(|_| data.len().is_multiple_of(shape[0]))
            .ok_or_else(|| invalid_selection(key, "payload is not row divisible"))?;
        let start = start * row_bytes;
        let end = end * row_bytes;
        return Ok((start..end, None));
    }
    if matches!(policy, ReadPolicy::AllowFullTensorRead) {
        return Ok((0..data.len(), None));
    }
    let (axis, indices): (usize, Vec<usize>) = match selection {
        TensorSelection::Range { axis, start, end } => (*axis, (*start..*end).collect()),
        TensorSelection::Indices { axis, indices } => (*axis, indices.clone()),
        TensorSelection::Contiguous { .. } => {
            return Err(StoreError::BoundedSelectionUnavailable {
                key: key.into(),
                message: "packed contiguous selection is not byte aligned".into(),
            })
        }
        TensorSelection::Full => unreachable!(),
    };
    let axis_len = shape[axis];
    let outer = shape[..axis].iter().product::<usize>();
    let inner = shape[axis + 1..].iter().product::<usize>();
    let output_bits = checked_elements(key, output_shape)?
        .checked_mul(bits)
        .ok_or_else(|| StoreError::Overflow {
            context: format!("selected bit length for {key:?}"),
        })?;
    if !output_bits.is_multiple_of(8) {
        return Err(StoreError::BoundedSelectionUnavailable {
            key: key.into(),
            message: "selected packed payload is not byte aligned".into(),
        });
    }
    let mut output = Vec::with_capacity(output_bits / 8);
    if bits == 4 {
        if !inner.is_multiple_of(2)
            || indices
                .iter()
                .any(|index| !(index * inner).is_multiple_of(2))
        {
            return Err(StoreError::BoundedSelectionUnavailable {
                key: key.into(),
                message: "FP4 selection crosses a nibble boundary".into(),
            });
        }
        let block_bytes = inner / 2;
        for outer_index in 0..outer {
            for index in &indices {
                let start = (outer_index * axis_len + index) * block_bytes;
                output.extend_from_slice(
                    data.get(start..start + block_bytes)
                        .ok_or_else(|| invalid_selection(key, "selection exceeds payload"))?,
                );
            }
        }
    } else {
        let scalar_bytes = scalar_bytes.ok_or_else(|| StoreError::BoundedSelectionUnavailable {
            key: key.into(),
            message: "stored scalar width is not byte aligned".into(),
        })?;
        let block_bytes = inner * scalar_bytes;
        for outer_index in 0..outer {
            for index in &indices {
                let start = (outer_index * axis_len + index) * block_bytes;
                output.extend_from_slice(
                    data.get(start..start + block_bytes)
                        .ok_or_else(|| invalid_selection(key, "selection exceeds payload"))?,
                );
            }
        }
    }
    Ok((0..output.len(), Some(output)))
}

fn checked_elements(key: &str, shape: &[usize]) -> Result<usize, StoreError> {
    shape.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| StoreError::Overflow {
                context: format!("element count for {key:?}"),
            })
    })
}

fn invalid_selection(key: &str, message: impl Into<String>) -> StoreError {
    StoreError::InvalidSelection {
        key: key.into(),
        message: message.into(),
    }
}

fn stored_dtype_from_safetensors(dtype: Dtype) -> StoredDtype {
    match dtype {
        Dtype::BOOL => StoredDtype::Bool,
        Dtype::U8 => StoredDtype::U8,
        Dtype::I8 => StoredDtype::I8,
        Dtype::I16 => StoredDtype::I16,
        Dtype::U16 => StoredDtype::U16,
        Dtype::F16 => StoredDtype::F16,
        Dtype::BF16 => StoredDtype::BF16,
        Dtype::I32 => StoredDtype::I32,
        Dtype::U32 => StoredDtype::U32,
        Dtype::F32 => StoredDtype::F32,
        Dtype::F64 => StoredDtype::F64,
        Dtype::I64 => StoredDtype::I64,
        Dtype::U64 => StoredDtype::U64,
        Dtype::C64 => StoredDtype::C64,
        Dtype::F8_E4M3 => StoredDtype::F8E4M3,
        Dtype::F4 => StoredDtype::F4,
        Dtype::F8_E8M0 => StoredDtype::F8E8M0,
        Dtype::F8_E5M2 => StoredDtype::F8E5M2,
        other => StoredDtype::Other(format!("{other:?}")),
    }
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: UniqueWeightMap,
}

#[derive(Debug)]
struct UniqueWeightMap(BTreeMap<String, String>);

impl<'de> Deserialize<'de> for UniqueWeightMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = UniqueWeightMap;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a tensor-to-shard object with unique names")
            }
            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut values = BTreeMap::new();
                while let Some((key, shard)) = map.next_entry::<String, String>()? {
                    if values.insert(key.clone(), shard).is_some() {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate tensor mapping for {key:?}"
                        )));
                    }
                }
                Ok(UniqueWeightMap(values))
            }
        }
        deserializer.deserialize_map(Visitor)
    }
}

fn validate_relative_shard_path(path: &Path) -> Result<PathBuf, StoreError> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(StoreError::UnsafeShardPath {
            path: path.to_path_buf(),
        });
    }
    Ok(path.to_path_buf())
}

fn canonicalize(path: &Path) -> Result<PathBuf, StoreError> {
    std::fs::canonicalize(path).map_err(|error| fs_error(path, error))
}

fn canonical_checkpoint_access_root(path: &Path) -> Result<PathBuf, StoreError> {
    let canonical_root = canonicalize(path)?;
    let Some(snapshots) = canonical_root.parent() else {
        return Ok(canonical_root);
    };
    if snapshots.file_name().and_then(|name| name.to_str()) != Some("snapshots") {
        return Ok(canonical_root);
    }
    let Some(repository_root) = snapshots.parent() else {
        return Ok(canonical_root);
    };
    if !repository_root.join("blobs").is_dir() {
        return Ok(canonical_root);
    }
    canonicalize(repository_root)
}

fn io_error(path: &Path, error: impl std::fmt::Display) -> StoreError {
    StoreError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    }
}

fn fs_error(path: &Path, error: std::io::Error) -> StoreError {
    if error.kind() == std::io::ErrorKind::NotFound {
        StoreError::MissingShard {
            path: path.to_path_buf(),
        }
    } else {
        io_error(path, error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize_to_file, TensorView};

    struct Lease {
        metadata: TensorMetadata,
        selection: TensorSelection,
        proof: BoundedReadProof,
        bytes: Vec<u8>,
    }

    impl EncodedTensorLease for Lease {
        fn metadata(&self) -> &TensorMetadata {
            &self.metadata
        }
        fn selection(&self) -> &TensorSelection {
            &self.selection
        }
        fn output_shape(&self) -> &[usize] {
            &self.metadata.logical_shape
        }
        fn bounded_read_proof(&self) -> &BoundedReadProof {
            &self.proof
        }
        fn backing_path(&self) -> Option<&Path> {
            None
        }
        fn encoded_bytes(&self) -> Option<&[u8]> {
            Some(&self.bytes)
        }
    }

    #[test]
    fn lease_exposes_encoding_selection_and_bounded_read_proof() {
        let lease = Lease {
            metadata: TensorMetadata {
                name: "model.weight".into(),
                logical_shape: vec![2, 2],
                physical_shape: vec![2, 2],
                stored_dtype: StoredDtype::F16,
                encoded_byte_len: 8,
                backing_shard: None,
            },
            selection: TensorSelection::Range {
                axis: 0,
                start: 1,
                end: 2,
            },
            proof: BoundedReadProof {
                physically_bounded: true,
                offset_bytes: 4,
                length_bytes: 4,
            },
            bytes: vec![0; 4],
        };
        assert_eq!(lease.metadata().stored_dtype, StoredDtype::F16);
        assert_eq!(lease.encoded_bytes().unwrap().len(), 4);
        assert!(lease.bounded_read_proof().physically_bounded);
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    #[test]
    fn safetensors_store_returns_exact_bounded_bytes_and_pins_mappings() {
        let directory = tempfile::tempdir().unwrap();
        let left = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let right = f32_bytes(&[5.0, 6.0, 7.0, 8.0]);
        let first = directory.path().join("model-00001-of-00002.safetensors");
        let second = directory.path().join("model-00002-of-00002.safetensors");
        serialize_to_file(
            [(
                "left",
                TensorView::new(Dtype::F32, vec![2, 2], &left).unwrap(),
            )],
            None,
            &first,
        )
        .unwrap();
        serialize_to_file(
            [(
                "right",
                TensorView::new(Dtype::F32, vec![2, 2], &right).unwrap(),
            )],
            None,
            &second,
        )
        .unwrap();
        std::fs::write(
            directory.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "weight_map": {
                    "left": first.file_name().unwrap().to_str().unwrap(),
                    "right": second.file_name().unwrap().to_str().unwrap()
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let store =
            SafetensorsWeightStore::open_with_max_mapped_shards(directory.path(), 1).unwrap();
        let lease = store
            .acquire(TensorReadRequest {
                key: "left".into(),
                selection: TensorSelection::Range {
                    axis: 0,
                    start: 1,
                    end: 2,
                },
                policy: ReadPolicy::RequireBounded,
            })
            .unwrap();
        assert_eq!(lease.output_shape(), &[1, 2]);
        assert_eq!(lease.encoded_bytes().unwrap(), &left[8..]);
        assert_eq!(lease.bounded_read_proof().length_bytes, 8);
        assert!(matches!(
            store.acquire(TensorReadRequest {
                key: "right".into(),
                selection: TensorSelection::Full,
                policy: ReadPolicy::RequireBounded,
            }),
            Err(StoreError::CapacityExhausted { maximum: 1, .. })
        ));
        drop(lease);
        assert!(store
            .acquire(TensorReadRequest {
                key: "right".into(),
                selection: TensorSelection::Full,
                policy: ReadPolicy::RequireBounded,
            })
            .is_ok());
    }

    #[test]
    fn resolved_source_rejects_unselected_physical_layouts() {
        let directory = tempfile::tempdir().unwrap();
        let bytes = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let file = directory.path().join("model.safetensors");
        serialize_to_file(
            [
                (
                    "selected",
                    TensorView::new(Dtype::F32, vec![2], &bytes[..8]).unwrap(),
                ),
                (
                    "unselected",
                    TensorView::new(Dtype::F32, vec![2], &bytes[8..]).unwrap(),
                ),
            ],
            None,
            &file,
        )
        .unwrap();
        let source: Arc<dyn CheckpointSource> =
            Arc::new(SafetensorsWeightStore::open(&file).unwrap());
        let contract =
            crate::validation::ResolvedCheckpointPlan::for_test("test architecture", ["selected"]);
        let source = ResolvedCheckpointSource::new(source, contract);

        assert_eq!(source.source_keys(), ["selected"]);
        assert!(source.source_metadata("selected").is_ok());
        assert!(matches!(
            source.source_metadata("unselected"),
            Err(StoreError::UnauthorizedTensor { .. })
        ));
        assert!(matches!(
            source.acquire_lease(TensorReadRequest {
                key: "unselected".into(),
                selection: TensorSelection::Full,
                policy: ReadPolicy::RequireBounded,
            }),
            Err(StoreError::UnauthorizedTensor { .. })
        ));
    }
}

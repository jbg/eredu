//! Persistent, lazy checkpoint tensor storage.
//!
//! A [`crate::runtime::checkpoint::store::WeightLease`] pins the bytes backing a safetensors
//! view. Materialization
//! never exposes that view as an MLX array: selection, copying, evaluation, and
//! conservative stream synchronization all finish before an owned array is
//! returned to the caller.

use std::{
    any::Any,
    collections::{BTreeMap, BTreeSet},
    fs::File,
    path::{Component, Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, MutexGuard, Weak,
    },
};

use memmap2::{Mmap, MmapOptions};
use safemlx::{
    ops::{
        indexing::TryIndexOp, GgufCheckpoint, GgufLogicalDtype, GgufMaterializer, GgufTensor,
        GgufTensorDescriptor, GgufTensorSelection, GgufTensorSelectionPlan,
    },
    transforms::eval,
    Array, Stream,
};
use safetensors::{
    tensor::{Dtype, Metadata, TensorInfo, TensorView},
    SafeTensors,
};
use serde::{de::MapAccess, Deserialize, Deserializer};

/// Default maximum number of simultaneously mapped payload shards.
pub const DEFAULT_MAX_MAPPED_SHARDS: usize = 4;

/// Backend-neutral description of a checkpoint's stored scalar encoding.
#[derive(Debug, Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum StoredDtype {
    /// Boolean values.
    Bool,
    /// Unsigned 8-bit integers.
    U8,
    /// Signed 8-bit integers.
    I8,
    /// Signed 16-bit integers.
    I16,
    /// Unsigned 16-bit integers.
    U16,
    /// IEEE half-precision floating point.
    F16,
    /// Brain floating point.
    BF16,
    /// Signed 32-bit integers.
    I32,
    /// Unsigned 32-bit integers.
    U32,
    /// IEEE single-precision floating point.
    F32,
    /// IEEE double-precision floating point.
    F64,
    /// Signed 64-bit integers.
    I64,
    /// Unsigned 64-bit integers.
    U64,
    /// Complex values with two 32-bit floating-point components.
    C64,
    /// Encoded FP8 E4M3 bytes. This is not an ordinary integer execution dtype.
    F8E4M3,
    /// Encoded FP8 E5M2 bytes.
    F8E5M2,
    /// Another safetensors encoding not represented by a named variant.
    Other(String),
}

impl From<Dtype> for StoredDtype {
    fn from(value: Dtype) -> Self {
        match value {
            Dtype::BOOL => Self::Bool,
            Dtype::U8 => Self::U8,
            Dtype::I8 => Self::I8,
            Dtype::I16 => Self::I16,
            Dtype::U16 => Self::U16,
            Dtype::F16 => Self::F16,
            Dtype::BF16 => Self::BF16,
            Dtype::I32 => Self::I32,
            Dtype::U32 => Self::U32,
            Dtype::F32 => Self::F32,
            Dtype::F64 => Self::F64,
            Dtype::I64 => Self::I64,
            Dtype::U64 => Self::U64,
            Dtype::C64 => Self::C64,
            Dtype::F8_E4M3 => Self::F8E4M3,
            Dtype::F8_E5M2 => Self::F8E5M2,
            other => Self::Other(format!("{other:?}")),
        }
    }
}

/// Catalog metadata for one logical checkpoint tensor.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct WeightMetadata {
    /// Stable logical checkpoint name.
    pub name: String,
    /// Logical tensor shape.
    pub shape: Vec<usize>,
    /// On-disk scalar encoding, distinct from an execution dtype.
    pub stored_dtype: StoredDtype,
    /// Number of bytes occupied by this tensor's encoded payload.
    pub logical_byte_len: usize,
    /// Payload shard that backs the tensor, when the backend is sharded.
    pub backing_shard: Option<PathBuf>,
}

/// A requested logical subset of a checkpoint tensor.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum TensorSelection {
    /// Select the complete tensor.
    Full,
    /// Select a non-empty contiguous range on one axis.
    Range {
        /// Axis to select.
        axis: usize,
        /// Inclusive start index.
        start: usize,
        /// Exclusive end index.
        end: usize,
    },
    /// Select indices on one axis in caller-supplied order.
    Indices {
        /// Axis to select.
        axis: usize,
        /// Non-empty ordered source indices.
        indices: Vec<usize>,
    },
    /// Select one physically contiguous row-major scalar span and expose it
    /// with an explicit logical shape.
    ///
    /// Semantic recipe pushdown constructs this form when independent axis
    /// ranges (for example one expert and a subset of its rows) collapse to a
    /// single storage interval. Callers do not need to calculate checkpoint
    /// byte offsets.
    Contiguous {
        /// Scalar offset from the start of the logical tensor.
        offset_elements: usize,
        /// Non-empty output geometry for the selected scalar span.
        shape: Vec<usize>,
    },
}

/// Whether a selected tensor may be obtained by decoding its complete source.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum WeightReadPolicy {
    /// Acquisition fails unless the backend can physically restrict payload I/O
    /// and decoding to the requested selection.
    RequireBounded,
    /// The backend may read and decode the complete tensor before selecting.
    ///
    /// This is intended for explicit tooling and diagnostics, not distributed
    /// or residency-managed execution.
    AllowFullTensorRead,
}

/// Deterministic mapped-shard cache statistics.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct WeightStoreDiagnostics {
    /// Storage backend represented by this snapshot.
    pub backend: WeightStoreBackend,
    /// Successful acquisitions that reused an existing mapping.
    pub mapping_hits: u64,
    /// Acquisition attempts that required a new mapping.
    pub mapping_misses: u64,
    /// Unleased mappings removed to honor the configured bound.
    pub evictions: u64,
    /// Number of mappings currently retained by the store cache.
    pub currently_mapped_shards: usize,
    /// Successfully mapped shard paths, in stable path order.
    pub touched_shard_paths: Vec<PathBuf>,
    /// Physical GGUF tensor or selected-region reads.
    pub physical_reads: u64,
    /// Encoded GGUF payload bytes requested by physical reads.
    pub physical_read_bytes: u64,
    /// Logical outputs served from an already converted physical group.
    pub coalesced_group_hits: u64,
}

/// Persistent checkpoint backend reported by [`WeightStoreDiagnostics`].
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum WeightStoreBackend {
    /// Memory-mapped SafeTensors payload shards.
    Safetensors,
    /// Seekable GGUF payload shards.
    Gguf,
    /// Immutable arrays produced by an explicit load-time transformation.
    Memory,
}

/// Structured failures from checkpoint catalog, mapping, and materialization.
#[derive(Debug, thiserror::Error)]
pub enum WeightStoreError {
    /// The configured mapping limit was zero.
    #[error("maximum mapped-shard count must be nonzero")]
    InvalidMappedShardLimit,
    /// A requested tensor is absent from the catalog.
    #[error("unknown checkpoint tensor {key:?}")]
    UnknownTensor {
        /// Requested logical key.
        key: String,
    },
    /// An indexed payload shard does not exist when accessed.
    #[error("checkpoint shard does not exist: {path}", path = .path.display())]
    MissingShard {
        /// Referenced shard path.
        path: PathBuf,
    },
    /// An index file could not be decoded.
    #[error("malformed safetensors index {path}: {message}", path = .path.display())]
    MalformedIndex {
        /// Index path.
        path: PathBuf,
        /// Decoder or validation detail.
        message: String,
    },
    /// A payload file has invalid safetensors metadata or contents.
    #[error("malformed safetensors shard {path}: {message}", path = .path.display())]
    MalformedSafetensors {
        /// Payload path.
        path: PathBuf,
        /// Parser detail.
        message: String,
    },
    /// An indexed shard path is absolute or escapes its model directory.
    #[error("unsafe safetensors shard path {path}", path = .path.display())]
    UnsafeShardPath {
        /// Rejected path.
        path: PathBuf,
    },
    /// The index maps a tensor to a shard that does not contain it.
    #[error("index maps tensor {key:?} to {path}, but that shard does not contain it", path = .path.display())]
    ContradictoryIndexMapping {
        /// Tensor key from the index.
        key: String,
        /// Referenced payload shard.
        path: PathBuf,
    },
    /// The requested subset is invalid for the cataloged tensor.
    #[error("invalid selection for tensor {key:?}: {message}")]
    InvalidSelection {
        /// Selected tensor key.
        key: String,
        /// Validation detail.
        message: String,
    },
    /// The backend cannot physically bound the requested selection.
    #[error("bounded selection is unavailable for tensor {key:?}: {message}")]
    BoundedSelectionUnavailable {
        /// Selected tensor key.
        key: String,
        /// Backend planning detail.
        message: String,
    },
    /// The stored encoding cannot be materialized by MLX.
    #[error("stored dtype {dtype:?} for tensor {key:?} is unsupported")]
    UnsupportedStoredDtype {
        /// Tensor key.
        key: String,
        /// Unsupported on-disk encoding.
        dtype: StoredDtype,
    },
    /// A shape, element count, byte size, or MLX dimension overflowed.
    #[error("checkpoint size overflow: {context}")]
    Overflow {
        /// Calculation that overflowed.
        context: String,
    },
    /// Every mapped shard is pinned by a live lease at the mapping bound.
    #[error(
        "mapped-shard capacity {max_mapped_shards} is exhausted; leased shards: {leased_shards:?}"
    )]
    CapacityExhausted {
        /// Configured simultaneous mapping bound.
        max_mapped_shards: usize,
        /// Deterministically ordered pinned shard paths.
        leased_shards: Vec<PathBuf>,
    },
    /// Filesystem access failed.
    #[error("I/O error for {path}: {source}", path = .path.display())]
    Io {
        /// Affected path.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// Memory mapping failed.
    #[error("failed to map checkpoint shard {path}: {source}", path = .path.display())]
    Mmap {
        /// Affected payload path.
        path: PathBuf,
        /// Mapping error.
        #[source]
        source: std::io::Error,
    },
    /// Safetensors-to-MLX conversion failed.
    #[error("failed to convert checkpoint tensor {key:?}: {source}")]
    MlxConversion {
        /// Tensor key.
        key: String,
        /// Conversion error.
        #[source]
        source: safemlx::error::ConversionError,
    },
    /// An MLX selection, copy, or evaluation operation failed.
    #[error("MLX {operation} failed for tensor {key:?}: {source}")]
    Mlx {
        /// Tensor key.
        key: String,
        /// Operation being performed.
        operation: &'static str,
        /// MLX exception.
        #[source]
        source: safemlx::error::Exception,
    },
    /// Conservative stream synchronization failed.
    #[error("stream synchronization failed for tensor {key:?} on {stream}: {source}")]
    Synchronization {
        /// Tensor key.
        key: String,
        /// Stream role.
        stream: &'static str,
        /// MLX exception.
        #[source]
        source: safemlx::error::Exception,
    },
    /// Internal cache state was poisoned by a prior panic.
    #[error("mapped-shard cache state is unavailable")]
    CachePoisoned,
    /// A GGUF catalog or materialization operation failed.
    #[error("GGUF weight store failed for tensor {key:?}: {message}")]
    Gguf {
        /// Requested logical tensor.
        key: String,
        /// Backend failure detail.
        message: String,
    },
}

/// Reusable checkpoint storage contract.
///
/// Implementations catalog keys without producing execution arrays. An
/// acquired lease owns the lifetime required for later safe materialization.
pub trait WeightStore: Any {
    /// Returns the concrete backend identity without consulting mutable diagnostics.
    fn backend(&self) -> WeightStoreBackend;

    /// Supports optional backend-specific inspection without making callers concrete.
    fn as_any(&self) -> &dyn Any;

    /// Returns all catalog keys in deterministic order.
    fn keys(&self) -> Vec<String>;

    /// Returns metadata, loading only the required backend metadata if needed.
    fn metadata(&self, key: &str) -> Result<WeightMetadata, WeightStoreError>;

    /// Acquires and validates a tensor selection under an explicit I/O policy
    /// while pinning its storage.
    fn acquire_with_policy(
        &self,
        key: &str,
        selection: TensorSelection,
        policy: WeightReadPolicy,
    ) -> Result<WeightLease, WeightStoreError>;

    /// Returns a deterministic snapshot of backend cache diagnostics.
    fn diagnostics(&self) -> Result<WeightStoreDiagnostics, WeightStoreError>;
}

/// Immutable array-backed store used to hand transformed weights to the
/// generalized execution engine without serializing an intermediate checkpoint.
#[derive(Clone)]
pub(crate) struct MemoryWeightStore {
    arrays: Arc<BTreeMap<String, Array>>,
    metadata: Arc<BTreeMap<String, WeightMetadata>>,
    backing: Arc<PathBuf>,
}

impl MemoryWeightStore {
    pub(crate) fn new(
        arrays: impl IntoIterator<Item = (String, Array)>,
    ) -> Result<Self, WeightStoreError> {
        let arrays = arrays.into_iter().collect::<BTreeMap<_, _>>();
        if arrays.is_empty() {
            return Err(WeightStoreError::Overflow {
                context: "in-memory checkpoint contains no tensors".into(),
            });
        }
        let backing = Arc::new(PathBuf::from("<load-time-transformed-checkpoint>"));
        let metadata = arrays
            .iter()
            .map(|(name, value)| {
                let shape = value
                    .shape()
                    .iter()
                    .map(|dimension| usize::try_from(*dimension))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| WeightStoreError::Overflow {
                        context: format!("in-memory shape for tensor {name:?}"),
                    })?;
                Ok((
                    name.clone(),
                    WeightMetadata {
                        name: name.clone(),
                        shape,
                        stored_dtype: stored_dtype_for_array(value),
                        logical_byte_len: value.nbytes(),
                        backing_shard: Some((*backing).clone()),
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>, WeightStoreError>>()?;
        Ok(Self {
            arrays: Arc::new(arrays),
            metadata: Arc::new(metadata),
            backing,
        })
    }
}

impl WeightStore for MemoryWeightStore {
    fn backend(&self) -> WeightStoreBackend {
        WeightStoreBackend::Memory
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn keys(&self) -> Vec<String> {
        self.arrays.keys().cloned().collect()
    }

    fn metadata(&self, key: &str) -> Result<WeightMetadata, WeightStoreError> {
        self.metadata
            .get(key)
            .cloned()
            .ok_or_else(|| WeightStoreError::UnknownTensor { key: key.into() })
    }

    fn acquire_with_policy(
        &self,
        key: &str,
        selection: TensorSelection,
        _policy: WeightReadPolicy,
    ) -> Result<WeightLease, WeightStoreError> {
        let metadata = self.metadata(key)?;
        let output_shape = validate_selection(key, &metadata.shape, &selection)?;
        let total_elements = metadata.shape.iter().product::<usize>();
        let selected_elements = output_shape.iter().product::<usize>();
        let selected_byte_len = metadata
            .logical_byte_len
            .checked_mul(selected_elements)
            .and_then(|bytes| bytes.checked_div(total_elements))
            .ok_or_else(|| WeightStoreError::Overflow {
                context: format!("in-memory selection size for tensor {key:?}"),
            })?;
        let value = self
            .arrays
            .get(key)
            .cloned()
            .ok_or_else(|| WeightStoreError::UnknownTensor { key: key.into() })?;
        Ok(WeightLease {
            key: key.into(),
            metadata,
            selection,
            output_shape,
            selected_byte_len,
            source: WeightLeaseSource::Memory(MemoryLeaseSource {
                value,
                backing: Arc::clone(&self.backing),
            }),
        })
    }

    fn diagnostics(&self) -> Result<WeightStoreDiagnostics, WeightStoreError> {
        Ok(WeightStoreDiagnostics {
            backend: WeightStoreBackend::Memory,
            mapping_hits: 0,
            mapping_misses: 0,
            evictions: 0,
            currently_mapped_shards: 0,
            touched_shard_paths: Vec::new(),
            physical_reads: 0,
            physical_read_bytes: 0,
            coalesced_group_hits: 0,
        })
    }
}

fn stored_dtype_for_array(value: &Array) -> StoredDtype {
    match value.dtype() {
        safemlx::Dtype::Bool => StoredDtype::Bool,
        safemlx::Dtype::Uint8 => StoredDtype::U8,
        safemlx::Dtype::Int8 => StoredDtype::I8,
        safemlx::Dtype::Int16 => StoredDtype::I16,
        safemlx::Dtype::Uint16 => StoredDtype::U16,
        safemlx::Dtype::Float16 => StoredDtype::F16,
        safemlx::Dtype::Bfloat16 => StoredDtype::BF16,
        safemlx::Dtype::Int32 => StoredDtype::I32,
        safemlx::Dtype::Uint32 => StoredDtype::U32,
        safemlx::Dtype::Float32 => StoredDtype::F32,
        other => StoredDtype::Other(format!("{other:?}")),
    }
}

#[derive(Debug, Clone)]
struct GgufCatalogEntry {
    checkpoint: usize,
    physical_name: String,
    original_name: String,
    metadata: WeightMetadata,
    physical_descriptor: GgufTensorDescriptor,
    logical_last_units_per_block: Option<usize>,
}

#[derive(Debug, Default)]
struct GgufStoreStatistics {
    physical_reads: AtomicU64,
    physical_read_bytes: AtomicU64,
    coalesced_group_hits: AtomicU64,
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
struct GgufGroupCacheKey {
    checkpoint: usize,
    physical_name: String,
    selection: Option<GgufTensorSelection>,
}

#[derive(Debug)]
struct CachedGgufGroup {
    arrays: Vec<(String, Array)>,
}

#[derive(Debug)]
struct GgufReaderCache {
    materializers: Vec<GgufMaterializer>,
    last_used: Vec<u64>,
    touched: BTreeSet<PathBuf>,
    tick: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

#[derive(Debug)]
struct GgufStoreInner {
    catalog: BTreeMap<String, GgufCatalogEntry>,
    readers: Mutex<GgufReaderCache>,
    max_cached_readers: usize,
    converted_groups: Mutex<BTreeMap<GgufGroupCacheKey, Weak<CachedGgufGroup>>>,
    statistics: GgufStoreStatistics,
}

/// Builder for a logical GGUF store backed by one or more checkpoints.
#[derive(Debug, Default)]
pub struct GgufWeightStoreBuilder {
    checkpoints: Vec<GgufCheckpoint>,
    catalog: BTreeMap<String, GgufCatalogEntry>,
    max_cached_readers: usize,
}

impl GgufWeightStoreBuilder {
    /// Sets the nonzero bound shared with mapped-shard loader controls.
    pub fn max_cached_readers(mut self, maximum: usize) -> Result<Self, WeightStoreError> {
        if maximum == 0 {
            return Err(WeightStoreError::InvalidMappedShardLimit);
        }
        self.max_cached_readers = maximum;
        Ok(self)
    }

    /// Adds one checkpoint and translates every converted logical output name.
    pub fn add_checkpoint<F>(
        mut self,
        checkpoint: GgufCheckpoint,
        mut translate: F,
    ) -> Result<Self, WeightStoreError>
    where
        F: FnMut(&str) -> String,
    {
        let checkpoint_index = self.checkpoints.len();
        for shard in checkpoint.catalog().shards() {
            for tensor in shard.tensors() {
                let physical_descriptor = tensor.descriptor().clone();
                for output in tensor.outputs() {
                    let name = translate(&output.name);
                    if self.catalog.contains_key(&name) {
                        return Err(WeightStoreError::Gguf {
                            key: name,
                            message: "translated logical tensor collides with an existing output"
                                .into(),
                        });
                    }
                    let shape = output
                        .shape
                        .iter()
                        .map(|dimension| {
                            usize::try_from(*dimension).map_err(|_| WeightStoreError::Overflow {
                                context: format!("GGUF logical shape for tensor {:?}", output.name),
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let width = logical_dtype_width(output.dtype);
                    let logical_byte_len = shape.iter().try_fold(width, |bytes, dimension| {
                        bytes
                            .checked_mul(*dimension)
                            .ok_or_else(|| WeightStoreError::Overflow {
                                context: format!(
                                    "GGUF logical byte length for tensor {:?}",
                                    output.name
                                ),
                            })
                    })?;
                    let logical_last_units_per_block =
                        gguf_logical_units_per_block(&output.name, &physical_descriptor, &shape)?;
                    let metadata = WeightMetadata {
                        name: name.clone(),
                        shape,
                        stored_dtype: stored_dtype_for_logical(output.dtype),
                        logical_byte_len,
                        backing_shard: Some(shard.path().to_path_buf()),
                    };
                    self.catalog.insert(
                        name,
                        GgufCatalogEntry {
                            checkpoint: checkpoint_index,
                            physical_name: tensor.descriptor().name.clone(),
                            original_name: output.name.clone(),
                            metadata,
                            physical_descriptor: physical_descriptor.clone(),
                            logical_last_units_per_block,
                        },
                    );
                }
            }
        }
        self.checkpoints.push(checkpoint);
        Ok(self)
    }

    /// Builds a non-empty immutable logical checkpoint store.
    pub fn build(self) -> Result<GgufWeightStore, WeightStoreError> {
        if self.catalog.is_empty() {
            return Err(WeightStoreError::Gguf {
                key: String::new(),
                message: "GGUF logical catalog is empty".into(),
            });
        }
        let materializers = self
            .checkpoints
            .iter()
            .map(GgufCheckpoint::materializer)
            .collect::<Vec<_>>();
        let materializer_count = materializers.len();
        Ok(GgufWeightStore {
            inner: Arc::new(GgufStoreInner {
                catalog: self.catalog,
                readers: Mutex::new(GgufReaderCache {
                    materializers,
                    last_used: vec![0; materializer_count],
                    touched: BTreeSet::new(),
                    tick: 0,
                    hits: 0,
                    misses: 0,
                    evictions: 0,
                }),
                max_cached_readers: if self.max_cached_readers == 0 {
                    DEFAULT_MAX_MAPPED_SHARDS
                } else {
                    self.max_cached_readers
                },
                converted_groups: Mutex::new(BTreeMap::new()),
                statistics: GgufStoreStatistics::default(),
            }),
        })
    }
}

/// Persistent logical tensor store backed by one or more GGUF checkpoints.
#[derive(Debug, Clone)]
pub struct GgufWeightStore {
    inner: Arc<GgufStoreInner>,
}

impl GgufWeightStore {
    /// Starts a multi-checkpoint GGUF store builder.
    pub fn builder() -> GgufWeightStoreBuilder {
        GgufWeightStoreBuilder::default()
    }

    /// Creates a single-checkpoint store with translated logical names.
    pub fn new<F>(checkpoint: GgufCheckpoint, translate: F) -> Result<Self, WeightStoreError>
    where
        F: FnMut(&str) -> String,
    {
        Self::builder()
            .add_checkpoint(checkpoint, translate)?
            .build()
    }

    /// Creates a single-checkpoint store with an explicit cached-reader bound.
    pub fn new_with_max_mapped_shards<F>(
        checkpoint: GgufCheckpoint,
        translate: F,
        max_mapped_shards: usize,
    ) -> Result<Self, WeightStoreError>
    where
        F: FnMut(&str) -> String,
    {
        Self::builder()
            .max_cached_readers(max_mapped_shards)?
            .add_checkpoint(checkpoint, translate)?
            .build()
    }
}

impl GgufReaderCache {
    fn materialize(
        &mut self,
        checkpoint: usize,
        physical_name: &str,
        selection: Option<&GgufTensorSelection>,
        max_cached_readers: usize,
        logical_key: &str,
    ) -> Result<GgufTensor, WeightStoreError> {
        let target_path = self
            .materializers
            .get(checkpoint)
            .ok_or_else(|| WeightStoreError::Gguf {
                key: logical_key.to_string(),
                message: "logical catalog references an unknown checkpoint".into(),
            })?
            .shard_path_for_tensor(physical_name)
            .map_err(|error| WeightStoreError::Gguf {
                key: logical_key.to_string(),
                message: error.to_string(),
            })?
            .to_path_buf();
        let reader_hit = self.materializers[checkpoint]
            .open_shard_path()
            .is_some_and(|path| path == target_path);
        self.tick = self.tick.saturating_add(1);
        if reader_hit {
            self.hits = self.hits.saturating_add(1);
        } else {
            self.misses = self.misses.saturating_add(1);
            if self.materializers[checkpoint].close_reader().is_some() {
                self.evictions = self.evictions.saturating_add(1);
            }
            if self
                .materializers
                .iter()
                .filter(|materializer| materializer.open_shard_path().is_some())
                .count()
                >= max_cached_readers
            {
                let victim = self
                    .materializers
                    .iter()
                    .enumerate()
                    .filter(|(_, materializer)| materializer.open_shard_path().is_some())
                    .min_by_key(|(index, _)| (self.last_used[*index], *index))
                    .map(|(index, _)| index)
                    .expect("a reader exists at the configured cache bound");
                self.materializers[victim].close_reader();
                self.evictions = self.evictions.saturating_add(1);
            }
        }
        self.last_used[checkpoint] = self.tick;
        let materializer = &mut self.materializers[checkpoint];
        let converted = match selection {
            Some(selection) => {
                materializer.converted_tensor_selected_host(physical_name, selection)
            }
            None => materializer.converted_tensor_host(physical_name),
        }
        .map_err(|error| WeightStoreError::Gguf {
            key: logical_key.to_string(),
            message: error.to_string(),
        })?;
        self.touched.insert(target_path);
        Ok(converted)
    }
}

impl WeightStore for GgufWeightStore {
    fn backend(&self) -> WeightStoreBackend {
        WeightStoreBackend::Gguf
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn keys(&self) -> Vec<String> {
        self.inner.catalog.keys().cloned().collect()
    }

    fn metadata(&self, key: &str) -> Result<WeightMetadata, WeightStoreError> {
        self.inner
            .catalog
            .get(key)
            .map(|entry| entry.metadata.clone())
            .ok_or_else(|| WeightStoreError::UnknownTensor {
                key: key.to_string(),
            })
    }

    fn acquire_with_policy(
        &self,
        key: &str,
        selection: TensorSelection,
        policy: WeightReadPolicy,
    ) -> Result<WeightLease, WeightStoreError> {
        let entry = self.inner.catalog.get(key).cloned().ok_or_else(|| {
            WeightStoreError::UnknownTensor {
                key: key.to_string(),
            }
        })?;
        let output_shape = validate_selection(key, &entry.metadata.shape, &selection)?;
        let selected_byte_len = selected_byte_len(key, &entry.metadata, &selection, &output_shape)?;
        let physical = match plan_bounded_gguf_selection(key, &entry, &selection) {
            Ok(plan) => plan,
            Err(WeightStoreError::BoundedSelectionUnavailable { .. })
                if policy == WeightReadPolicy::AllowFullTensorRead =>
            {
                GgufReadPlan {
                    physical_selection: None,
                    physical_byte_len: entry.physical_descriptor.byte_len,
                    selection_is_materialized: false,
                }
            }
            Err(error) => return Err(error),
        };
        Ok(WeightLease {
            key: key.to_string(),
            metadata: entry.metadata.clone(),
            selection,
            output_shape,
            selected_byte_len,
            source: WeightLeaseSource::Gguf(Box::new(GgufLeaseSource {
                store: Arc::clone(&self.inner),
                entry,
                physical_selection: physical.physical_selection,
                physical_byte_len: physical.physical_byte_len,
                selection_is_materialized: physical.selection_is_materialized,
            })),
        })
    }

    fn diagnostics(&self) -> Result<WeightStoreDiagnostics, WeightStoreError> {
        let readers = self
            .inner
            .readers
            .lock()
            .map_err(|_| WeightStoreError::CachePoisoned)?;
        Ok(WeightStoreDiagnostics {
            backend: WeightStoreBackend::Gguf,
            mapping_hits: readers.hits,
            mapping_misses: readers.misses,
            evictions: readers.evictions,
            currently_mapped_shards: readers
                .materializers
                .iter()
                .filter(|materializer| materializer.open_shard_path().is_some())
                .count(),
            touched_shard_paths: readers.touched.iter().cloned().collect(),
            physical_reads: self.inner.statistics.physical_reads.load(Ordering::Relaxed),
            physical_read_bytes: self
                .inner
                .statistics
                .physical_read_bytes
                .load(Ordering::Relaxed),
            coalesced_group_hits: self
                .inner
                .statistics
                .coalesced_group_hits
                .load(Ordering::Relaxed),
        })
    }
}

struct GgufReadPlan {
    physical_selection: Option<GgufTensorSelection>,
    physical_byte_len: u64,
    selection_is_materialized: bool,
}

fn plan_bounded_gguf_selection(
    key: &str,
    entry: &GgufCatalogEntry,
    selection: &TensorSelection,
) -> Result<GgufReadPlan, WeightStoreError> {
    if matches!(selection, TensorSelection::Full) {
        return Ok(GgufReadPlan {
            physical_selection: None,
            physical_byte_len: entry.physical_descriptor.byte_len,
            selection_is_materialized: true,
        });
    }
    if matches!(selection, TensorSelection::Contiguous { .. }) {
        return Err(WeightStoreError::BoundedSelectionUnavailable {
            key: key.to_string(),
            message: "GGUF readers cannot yet express a reshaped contiguous scalar span".into(),
        });
    }
    let rank = entry.metadata.shape.len();
    let logical_axis = match selection {
        TensorSelection::Full => unreachable!("full selections returned above"),
        TensorSelection::Range { axis, .. } | TensorSelection::Indices { axis, .. } => *axis,
        TensorSelection::Contiguous { .. } => unreachable!("rejected above"),
    };
    let physical_selection = if logical_axis + 1 != rank {
        match selection {
            TensorSelection::Full => unreachable!("full selections returned above"),
            TensorSelection::Range { axis, start, end } => GgufTensorSelection::Range {
                axis: *axis,
                start: *start,
                end: *end,
            },
            TensorSelection::Indices { axis, indices } => GgufTensorSelection::Indices {
                axis: *axis,
                indices: indices.clone(),
            },
            TensorSelection::Contiguous { .. } => unreachable!("rejected above"),
        }
    } else {
        map_gguf_innermost_selection(key, entry, selection)?
    };
    let plan = GgufTensorSelectionPlan::new(&entry.physical_descriptor, physical_selection.clone())
        .map_err(|error| WeightStoreError::BoundedSelectionUnavailable {
            key: key.to_string(),
            message: error.to_string(),
        })?;
    Ok(GgufReadPlan {
        physical_selection: Some(physical_selection),
        physical_byte_len: plan.encoded_byte_len(),
        selection_is_materialized: true,
    })
}

fn map_gguf_innermost_selection(
    key: &str,
    entry: &GgufCatalogEntry,
    selection: &TensorSelection,
) -> Result<GgufTensorSelection, WeightStoreError> {
    let logical_units = entry.logical_last_units_per_block.ok_or_else(|| {
        WeightStoreError::BoundedSelectionUnavailable {
            key: key.to_string(),
            message: "scalar GGUF output has no selectable innermost axis".into(),
        }
    })?;
    let (block_values, _) = entry
        .physical_descriptor
        .ggml_type
        .block_and_bytes()
        .map_err(|error| WeightStoreError::Gguf {
            key: key.to_string(),
            message: error.to_string(),
        })?;
    let block_values = usize::try_from(block_values).map_err(|_| WeightStoreError::Overflow {
        context: format!("GGUF block length for tensor {key:?}"),
    })?;
    let axis = entry.metadata.shape.len() - 1;
    match selection {
        TensorSelection::Range { start, end, .. } => {
            if start % logical_units != 0 || end % logical_units != 0 {
                return Err(WeightStoreError::BoundedSelectionUnavailable {
                    key: key.to_string(),
                    message: format!(
                        "logical innermost range {start}..{end} must align to {logical_units} converted units per native GGUF block"
                    ),
                });
            }
            let physical_start = (start / logical_units)
                .checked_mul(block_values)
                .ok_or_else(|| WeightStoreError::Overflow {
                    context: format!("GGUF physical selection start for tensor {key:?}"),
                })?;
            let physical_end =
                (end / logical_units)
                    .checked_mul(block_values)
                    .ok_or_else(|| WeightStoreError::Overflow {
                        context: format!("GGUF physical selection end for tensor {key:?}"),
                    })?;
            Ok(GgufTensorSelection::Range {
                axis,
                start: physical_start,
                end: physical_end,
            })
        }
        TensorSelection::Indices { indices, .. } => {
            if !indices.len().is_multiple_of(logical_units) {
                return Err(WeightStoreError::BoundedSelectionUnavailable {
                    key: key.to_string(),
                    message: format!(
                        "logical innermost indices must contain complete {logical_units}-unit converted GGUF blocks"
                    ),
                });
            }
            let block_count = indices.len() / logical_units;
            let physical_count = block_count.checked_mul(block_values).ok_or_else(|| {
                WeightStoreError::Overflow {
                    context: format!("GGUF physical index count for tensor {key:?}"),
                }
            })?;
            let mut physical_indices = Vec::new();
            physical_indices
                .try_reserve_exact(physical_count)
                .map_err(|_| WeightStoreError::Overflow {
                    context: format!("GGUF physical indices for tensor {key:?}"),
                })?;
            for logical_block in indices.chunks_exact(logical_units) {
                let logical_start = logical_block[0];
                if logical_start % logical_units != 0
                    || logical_block
                        .iter()
                        .copied()
                        .ne(logical_start..logical_start + logical_units)
                {
                    return Err(WeightStoreError::BoundedSelectionUnavailable {
                        key: key.to_string(),
                        message: format!(
                            "logical innermost indices must preserve every complete aligned {logical_units}-unit converted GGUF block"
                        ),
                    });
                }
                let physical_start = (logical_start / logical_units)
                    .checked_mul(block_values)
                    .ok_or_else(|| WeightStoreError::Overflow {
                        context: format!("GGUF physical index start for tensor {key:?}"),
                    })?;
                physical_indices.extend(physical_start..physical_start + block_values);
            }
            Ok(GgufTensorSelection::Indices {
                axis,
                indices: physical_indices,
            })
        }
        TensorSelection::Full => unreachable!("full selections are not mapped"),
        TensorSelection::Contiguous { .. } => {
            unreachable!("contiguous selections are rejected before innermost mapping")
        }
    }
}

fn logical_dtype_width(dtype: GgufLogicalDtype) -> usize {
    match dtype {
        GgufLogicalDtype::U8 | GgufLogicalDtype::I8 => 1,
        GgufLogicalDtype::F16 | GgufLogicalDtype::Bf16 | GgufLogicalDtype::I16 => 2,
        GgufLogicalDtype::F32 | GgufLogicalDtype::U32 | GgufLogicalDtype::I32 => 4,
        GgufLogicalDtype::I64 | GgufLogicalDtype::F64 => 8,
    }
}

fn gguf_logical_units_per_block(
    logical_name: &str,
    descriptor: &GgufTensorDescriptor,
    logical_shape: &[usize],
) -> Result<Option<usize>, WeightStoreError> {
    let physical_shape = descriptor
        .mlx_shape()
        .into_iter()
        .map(|dimension| {
            usize::try_from(dimension).map_err(|_| WeightStoreError::Overflow {
                context: format!("GGUF physical shape for tensor {logical_name:?}"),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if physical_shape.len() != logical_shape.len()
        || physical_shape
            .iter()
            .zip(logical_shape)
            .take(physical_shape.len().saturating_sub(1))
            .any(|(physical, logical)| physical != logical)
    {
        return Err(WeightStoreError::Gguf {
            key: logical_name.to_string(),
            message: format!(
                "logical shape {logical_shape:?} is not an innermost-axis projection of physical shape {physical_shape:?}"
            ),
        });
    }
    let (Some(&physical_last), Some(&logical_last)) = (physical_shape.last(), logical_shape.last())
    else {
        return Ok(None);
    };
    let (block_values, _) =
        descriptor
            .ggml_type
            .block_and_bytes()
            .map_err(|error| WeightStoreError::Gguf {
                key: logical_name.to_string(),
                message: error.to_string(),
            })?;
    let block_values = usize::try_from(block_values).map_err(|_| WeightStoreError::Overflow {
        context: format!("GGUF block length for tensor {logical_name:?}"),
    })?;
    if !physical_last.is_multiple_of(block_values) {
        return Err(WeightStoreError::Gguf {
            key: logical_name.to_string(),
            message: format!(
                "physical innermost dimension {physical_last} is not divisible by block length {block_values}"
            ),
        });
    }
    let physical_blocks = physical_last / block_values;
    if physical_blocks == 0 || !logical_last.is_multiple_of(physical_blocks) {
        return Err(WeightStoreError::Gguf {
            key: logical_name.to_string(),
            message: format!(
                "logical innermost dimension {logical_last} cannot be mapped across {physical_blocks} physical blocks"
            ),
        });
    }
    Ok(Some(logical_last / physical_blocks))
}

fn stored_dtype_for_logical(dtype: GgufLogicalDtype) -> StoredDtype {
    match dtype {
        GgufLogicalDtype::F32 => StoredDtype::F32,
        GgufLogicalDtype::F16 => StoredDtype::F16,
        GgufLogicalDtype::Bf16 => StoredDtype::BF16,
        GgufLogicalDtype::U8 => StoredDtype::U8,
        GgufLogicalDtype::I8 => StoredDtype::I8,
        GgufLogicalDtype::I16 => StoredDtype::I16,
        GgufLogicalDtype::U32 => StoredDtype::U32,
        GgufLogicalDtype::I32 => StoredDtype::I32,
        GgufLogicalDtype::I64 => StoredDtype::I64,
        GgufLogicalDtype::F64 => StoredDtype::F64,
    }
}

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

/// Safetensors-backed persistent checkpoint catalog and mapped-shard cache.
#[derive(Debug)]
pub struct SafetensorsWeightStore {
    canonical_root: PathBuf,
    catalog: BTreeMap<String, CatalogEntry>,
    metadata: Mutex<BTreeMap<String, WeightMetadata>>,
    cache: Mutex<CacheState>,
    max_mapped_shards: usize,
}

impl SafetensorsWeightStore {
    /// Opens a checkpoint with [`DEFAULT_MAX_MAPPED_SHARDS`].
    ///
    /// `path` may be a direct `.safetensors` file, an indexed Hugging Face
    /// directory, or a directory containing `model.safetensors`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, WeightStoreError> {
        Self::open_with_max_mapped_shards(path, DEFAULT_MAX_MAPPED_SHARDS)
    }

    /// Opens a checkpoint with an explicit nonzero per-store mapping bound.
    ///
    /// The bound counts cache-owned mappings. Because a live lease pins its
    /// cache entry, a new mapping returns [`WeightStoreError::CapacityExhausted`]
    /// when no unleased entry can be evicted.
    pub fn open_with_max_mapped_shards(
        path: impl AsRef<Path>,
        max_mapped_shards: usize,
    ) -> Result<Self, WeightStoreError> {
        if max_mapped_shards == 0 {
            return Err(WeightStoreError::InvalidMappedShardLimit);
        }
        let path = path.as_ref();
        if !path.exists() {
            return Err(WeightStoreError::MissingShard {
                path: path.to_path_buf(),
            });
        }

        if path.is_dir() {
            let root = path.to_path_buf();
            let canonical_root = canonical_checkpoint_access_root(path)?;
            let index_path = root.join("model.safetensors.index.json");
            if index_path.exists() {
                let raw = std::fs::read_to_string(&index_path).map_err(|source| {
                    WeightStoreError::Io {
                        path: index_path.clone(),
                        source,
                    }
                })?;
                let index: SafetensorsIndex = serde_json::from_str(&raw).map_err(|error| {
                    WeightStoreError::MalformedIndex {
                        path: index_path.clone(),
                        message: error.to_string(),
                    }
                })?;
                if index.weight_map.0.is_empty() {
                    return Err(WeightStoreError::MalformedIndex {
                        path: index_path,
                        message: "weight_map must not be empty".into(),
                    });
                }
                let mut catalog = BTreeMap::new();
                for (key, relative) in index.weight_map.0 {
                    if key.is_empty() {
                        return Err(WeightStoreError::MalformedIndex {
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
            .unwrap_or_else(|| Path::new("."));
        let canonical_root = canonicalize(root)?;
        Self::from_single_file(file, canonical_root, max_mapped_shards)
    }

    fn from_single_file(
        file: PathBuf,
        canonical_root: PathBuf,
        max_mapped_shards: usize,
    ) -> Result<Self, WeightStoreError> {
        if !file.exists() {
            return Err(WeightStoreError::MissingShard { path: file });
        }
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

    fn catalog_entry(&self, key: &str) -> Result<&CatalogEntry, WeightStoreError> {
        self.catalog
            .get(key)
            .ok_or_else(|| WeightStoreError::UnknownTensor {
                key: key.to_string(),
            })
    }

    fn lock_cache(&self) -> Result<MutexGuard<'_, CacheState>, WeightStoreError> {
        self.cache
            .lock()
            .map_err(|_| WeightStoreError::CachePoisoned)
    }

    fn acquire_shard(&self, entry: &CatalogEntry) -> Result<Arc<MappedShard>, WeightStoreError> {
        let canonical_path = self.validate_access_path(entry)?;
        let mut cache = self.lock_cache()?;
        cache.tick = cache.tick.saturating_add(1);
        let tick = cache.tick;
        if let Some(shard) = cache
            .entries
            .get(&canonical_path)
            .map(|existing| Arc::clone(&existing.shard))
        {
            cache.hits = cache.hits.saturating_add(1);
            if let Some(existing) = cache.entries.get_mut(&canonical_path) {
                existing.last_used = tick;
            }
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
                let leased_shards = cache
                    .entries
                    .values()
                    .map(|entry| entry.shard.path.clone())
                    .collect();
                return Err(WeightStoreError::CapacityExhausted {
                    max_mapped_shards: self.max_mapped_shards,
                    leased_shards,
                });
            }
        }

        let file = File::open(&canonical_path).map_err(|source| WeightStoreError::Io {
            path: entry.shard.clone(),
            source,
        })?;
        // SAFETY: MappedShard owns the Mmap, and every public data access is
        // mediated by a WeightLease holding an Arc<MappedShard>.
        let mmap =
            unsafe { MmapOptions::new().map(&file) }.map_err(|source| WeightStoreError::Mmap {
                path: entry.shard.clone(),
                source,
            })?;
        let (header_len, metadata) = SafeTensors::read_metadata(&mmap).map_err(|error| {
            WeightStoreError::MalformedSafetensors {
                path: entry.shard.clone(),
                message: error.to_string(),
            }
        })?;
        let payload_offset =
            8usize
                .checked_add(header_len)
                .ok_or_else(|| WeightStoreError::Overflow {
                    context: format!("payload offset for shard {}", entry.shard.display()),
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

    fn validate_access_path(&self, entry: &CatalogEntry) -> Result<PathBuf, WeightStoreError> {
        if !entry.shard.exists() {
            return Err(WeightStoreError::MissingShard {
                path: entry.shard.clone(),
            });
        }
        let canonical = canonicalize(&entry.shard)?;
        if entry.indexed && !canonical.starts_with(&self.canonical_root) {
            return Err(WeightStoreError::UnsafeShardPath {
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
    ) -> Result<WeightMetadata, WeightStoreError> {
        if let Some(metadata) = self
            .metadata
            .lock()
            .map_err(|_| WeightStoreError::CachePoisoned)?
            .get(key)
            .cloned()
        {
            return Ok(metadata);
        }
        shard.metadata.info(key).ok_or_else(|| {
            if entry.indexed {
                WeightStoreError::ContradictoryIndexMapping {
                    key: key.to_string(),
                    path: shard.path.clone(),
                }
            } else {
                WeightStoreError::UnknownTensor {
                    key: key.to_string(),
                }
            }
        })?;

        // Safetensors metadata is one JSON header for the whole shard. Large
        // split-expert checkpoints may ask for thousands of tensor records;
        // reparsing that header once per tensor dominates layerwise startup.
        // Cache every index-confirmed tensor from this parse in one pass.
        let mut discovered = BTreeMap::new();
        for name in shard.metadata.offset_keys() {
            if self
                .catalog
                .get(&name)
                .is_some_and(|candidate| candidate.shard == shard.path)
            {
                let info = shard
                    .metadata
                    .info(&name)
                    .expect("name came from the same safetensors metadata");
                discovered.insert(name.clone(), metadata_for_info(&name, &shard.path, info)?);
            }
        }
        let metadata = discovered.get(key).cloned().ok_or_else(|| {
            WeightStoreError::ContradictoryIndexMapping {
                key: key.to_string(),
                path: shard.path.clone(),
            }
        })?;
        self.metadata
            .lock()
            .map_err(|_| WeightStoreError::CachePoisoned)?
            .extend(discovered);
        Ok(metadata)
    }
}

impl WeightStore for SafetensorsWeightStore {
    fn backend(&self) -> WeightStoreBackend {
        WeightStoreBackend::Safetensors
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn keys(&self) -> Vec<String> {
        self.catalog.keys().cloned().collect()
    }

    fn metadata(&self, key: &str) -> Result<WeightMetadata, WeightStoreError> {
        if let Some(metadata) = self
            .metadata
            .lock()
            .map_err(|_| WeightStoreError::CachePoisoned)?
            .get(key)
            .cloned()
        {
            return Ok(metadata);
        }
        let entry = self.catalog_entry(key)?;
        let shard = self.acquire_shard(entry)?;
        self.metadata_from_shard(key, entry, &shard)
    }

    fn acquire_with_policy(
        &self,
        key: &str,
        selection: TensorSelection,
        _policy: WeightReadPolicy,
    ) -> Result<WeightLease, WeightStoreError> {
        let entry = self.catalog_entry(key)?;
        let shard = self.acquire_shard(entry)?;
        let metadata = self.metadata_from_shard(key, entry, &shard)?;
        let output_shape = validate_selection(key, &metadata.shape, &selection)?;
        let selected_byte_len = selected_byte_len(key, &metadata, &selection, &output_shape)?;
        Ok(WeightLease {
            key: key.to_string(),
            metadata,
            selection,
            output_shape,
            selected_byte_len,
            source: WeightLeaseSource::Safetensors(shard),
        })
    }

    fn diagnostics(&self) -> Result<WeightStoreDiagnostics, WeightStoreError> {
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

#[derive(Debug, Clone)]
enum WeightLeaseSource {
    Safetensors(Arc<MappedShard>),
    Gguf(Box<GgufLeaseSource>),
    Memory(MemoryLeaseSource),
}

#[derive(Debug, Clone)]
struct MemoryLeaseSource {
    value: Array,
    backing: Arc<PathBuf>,
}

#[derive(Debug, Clone)]
struct GgufLeaseSource {
    store: Arc<GgufStoreInner>,
    entry: GgufCatalogEntry,
    physical_selection: Option<GgufTensorSelection>,
    physical_byte_len: u64,
    selection_is_materialized: bool,
}

/// A validated selection that pins its mapped payload shard.
///
/// The lease deliberately has no method returning a borrowed or mmap-derived
/// MLX array. [`Self::materialize`] is the only array-producing operation.
#[derive(Debug, Clone)]
pub struct WeightLease {
    key: String,
    metadata: WeightMetadata,
    selection: TensorSelection,
    output_shape: Vec<usize>,
    selected_byte_len: usize,
    source: WeightLeaseSource,
}

impl WeightLease {
    /// Returns the logical key pinned by this lease.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Returns metadata captured when the lease was acquired.
    pub fn metadata(&self) -> &WeightMetadata {
        &self.metadata
    }

    /// Returns the validated selection.
    pub fn selection(&self) -> &TensorSelection {
        &self.selection
    }

    /// Returns the selected output shape.
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    /// Returns the logical encoded byte length of the validated selection.
    ///
    /// This is the selected tensor's checkpoint payload size. For execution
    /// dtypes supported by the store it also matches the materialized array's
    /// `nbytes()` value.
    pub const fn selected_byte_len(&self) -> usize {
        self.selected_byte_len
    }

    /// Returns the path of the pinned payload shard.
    pub fn backing_shard(&self) -> &Path {
        match &self.source {
            WeightLeaseSource::Safetensors(shard) => &shard.path,
            WeightLeaseSource::Gguf(source) => source
                .entry
                .metadata
                .backing_shard
                .as_deref()
                .expect("GGUF catalog entries always identify their shard"),
            WeightLeaseSource::Memory(source) => source.backing.as_path(),
        }
    }

    /// Safely materializes the selected tensor onto `execution_stream`.
    ///
    /// The returned copy is explicitly evaluated while this lease and its
    /// mmap-derived source array remain live. Batched residency callers use
    /// `prepare_materialization` internally evaluates several outputs together.
    /// An incomplete pending value synchronizes conservatively during drop.
    pub fn materialize(
        &self,
        source_stream: &Stream,
        execution_stream: &Stream,
    ) -> Result<Array, WeightStoreError> {
        self.clone()
            .prepare_materialization(source_stream, execution_stream)?
            .finish()
    }

    /// Schedules materialization while retaining every mmap-backed dependency.
    ///
    /// The returned value must be explicitly completed after its output is
    /// evaluated. Dropping it early conservatively synchronizes both streams.
    pub(crate) fn prepare_materialization(
        self,
        source_stream: &Stream,
        execution_stream: &Stream,
    ) -> Result<PendingWeightMaterialization, WeightStoreError> {
        match self.source.clone() {
            WeightLeaseSource::Safetensors(shard) => {
                self.prepare_safetensors(shard, source_stream, execution_stream)
            }
            WeightLeaseSource::Gguf(source) => {
                self.prepare_gguf(*source, source_stream, execution_stream)
            }
            WeightLeaseSource::Memory(source) => {
                self.prepare_memory(source, source_stream, execution_stream)
            }
        }
    }

    /// Schedules a host-only materialization that may borrow bounded
    /// SafeTensors bytes until a containing derived output is evaluated.
    ///
    /// Callers must not return `output()` after completing this pending value;
    /// use it only as an input to an evaluated dependent graph.
    pub(crate) fn prepare_borrowed_materialization(
        self,
        source_stream: &Stream,
    ) -> Result<PendingWeightMaterialization, WeightStoreError> {
        match self.source.clone() {
            WeightLeaseSource::Safetensors(shard) => {
                self.prepare_borrowed_safetensors(shard, source_stream)
            }
            WeightLeaseSource::Gguf(source) => {
                self.prepare_gguf(*source, source_stream, source_stream)
            }
            WeightLeaseSource::Memory(source) => {
                self.prepare_memory(source, source_stream, source_stream)
            }
        }
    }

    fn prepare_memory(
        self,
        source: MemoryLeaseSource,
        source_stream: &Stream,
        execution_stream: &Stream,
    ) -> Result<PendingWeightMaterialization, WeightStoreError> {
        let source_value = source.value;
        let materialized = match &self.selection {
            // Transformed stores are created on the execution stream. Keeping
            // the same immutable array handle avoids a second full model copy
            // while the residency manager retains its lease.
            TensorSelection::Full => source_value.clone(),
            TensorSelection::Range { axis, start, end } => materialize_range(
                &self.key,
                source_value.clone(),
                &self.metadata.shape,
                *axis,
                *start,
                *end,
                source_stream,
                execution_stream,
            )?,
            TensorSelection::Indices { axis, indices } => materialize_indices(
                &self.key,
                &source_value,
                *axis,
                indices,
                source_stream,
                execution_stream,
            )?,
            TensorSelection::Contiguous {
                offset_elements,
                shape,
            } => materialize_contiguous(
                &self.key,
                &source_value,
                *offset_elements,
                shape,
                source_stream,
                execution_stream,
            )?,
        };
        Ok(PendingWeightMaterialization {
            output: materialized,
            _source: source_value,
            _gguf_group: None,
            lease: Some(self),
            source_stream: source_stream.clone(),
            execution_stream: execution_stream.clone(),
            borrowed_source: false,
            completed: false,
        })
    }

    fn prepare_borrowed_safetensors(
        self,
        shard: Arc<MappedShard>,
        source_stream: &Stream,
    ) -> Result<PendingWeightMaterialization, WeightStoreError> {
        let info = shard.metadata.info(&self.key).ok_or_else(|| {
            WeightStoreError::ContradictoryIndexMapping {
                key: self.key.clone(),
                path: shard.path.clone(),
            }
        })?;
        if !is_supported_execution_dtype(info.dtype) {
            return Err(WeightStoreError::UnsupportedStoredDtype {
                key: self.key.clone(),
                dtype: info.dtype.into(),
            });
        }
        let payload_start = shard
            .payload_offset
            .checked_add(info.data_offsets.0)
            .ok_or_else(|| WeightStoreError::Overflow {
                context: format!("payload start for tensor {:?}", self.key),
            })?;
        let payload_end = shard
            .payload_offset
            .checked_add(info.data_offsets.1)
            .ok_or_else(|| WeightStoreError::Overflow {
                context: format!("payload end for tensor {:?}", self.key),
            })?;
        let data = shard.mmap.get(payload_start..payload_end).ok_or_else(|| {
            WeightStoreError::MalformedSafetensors {
                path: shard.path.clone(),
                message: format!("tensor {:?} payload is outside the mapped shard", self.key),
            }
        })?;
        let (shape, selected_data) = match &self.selection {
            TensorSelection::Full => (info.shape.clone(), data),
            TensorSelection::Range {
                axis: 0,
                start,
                end,
            } => {
                let outer = info.shape[0];
                let row_bytes = data
                    .len()
                    .checked_div(outer)
                    .filter(|_| data.len() % outer == 0)
                    .ok_or_else(|| WeightStoreError::MalformedSafetensors {
                        path: shard.path.clone(),
                        message: format!(
                            "tensor {:?} payload is not divisible by its outer dimension",
                            self.key
                        ),
                    })?;
                let start = start.checked_mul(row_bytes).ok_or_else(|| {
                    WeightStoreError::Overflow {
                        context: format!("borrowed byte start for tensor {:?}", self.key),
                    }
                })?;
                let end = end.checked_mul(row_bytes).ok_or_else(|| WeightStoreError::Overflow {
                    context: format!("borrowed byte end for tensor {:?}", self.key),
                })?;
                let selected = data.get(start..end).ok_or_else(|| {
                    WeightStoreError::MalformedSafetensors {
                        path: shard.path.clone(),
                        message: format!(
                            "borrowed selection for tensor {:?} is outside its payload",
                            self.key
                        ),
                    }
                })?;
                (self.output_shape.clone(), selected)
            }
            TensorSelection::Contiguous {
                offset_elements,
                shape,
            } => (
                shape.clone(),
                contiguous_safetensors_data(&self.key, &shard.path, &info.shape, data, *offset_elements, shape)?,
            ),
            selection => {
                return Err(WeightStoreError::BoundedSelectionUnavailable {
                    key: self.key.clone(),
                    message: format!(
                        "zero-copy conversion requires a full tensor, contiguous axis-zero range, or contiguous scalar span, got {selection:?}"
                    ),
                })
            }
        };
        let view = TensorView::new(info.dtype, shape, selected_data).map_err(|error| {
            WeightStoreError::MalformedSafetensors {
                path: shard.path.clone(),
                message: format!("tensor {:?}: {error}", self.key),
            }
        })?;
        let aligned = (selected_data.as_ptr() as usize)
            .is_multiple_of(safetensors_dtype_alignment(info.dtype));
        let source_value = if aligned {
            unsafe { Array::try_from_borrowed_safetensors(view) }
        } else {
            // SafeTensors permits tensor payloads whose offset is not aligned
            // for their scalar dtype. Keep conversion bounded to this selected
            // tile by using MLX's copying constructor in that case.
            Array::try_from(view)
        }
        .map_err(|source| WeightStoreError::MlxConversion {
            key: self.key.clone(),
            source,
        })?;
        Ok(PendingWeightMaterialization {
            output: source_value.clone(),
            _source: source_value,
            _gguf_group: None,
            lease: Some(self),
            source_stream: source_stream.clone(),
            execution_stream: source_stream.clone(),
            borrowed_source: aligned,
            completed: false,
        })
    }

    fn prepare_safetensors(
        self,
        shard: Arc<MappedShard>,
        source_stream: &Stream,
        execution_stream: &Stream,
    ) -> Result<PendingWeightMaterialization, WeightStoreError> {
        let info = shard.metadata.info(&self.key).ok_or_else(|| {
            WeightStoreError::ContradictoryIndexMapping {
                key: self.key.clone(),
                path: shard.path.clone(),
            }
        })?;
        if !is_supported_execution_dtype(info.dtype) {
            return Err(WeightStoreError::UnsupportedStoredDtype {
                key: self.key.clone(),
                dtype: info.dtype.into(),
            });
        }

        let start = shard
            .payload_offset
            .checked_add(info.data_offsets.0)
            .ok_or_else(|| WeightStoreError::Overflow {
                context: format!("payload start for tensor {:?}", self.key),
            })?;
        let end = shard
            .payload_offset
            .checked_add(info.data_offsets.1)
            .ok_or_else(|| WeightStoreError::Overflow {
                context: format!("payload end for tensor {:?}", self.key),
            })?;
        let data =
            shard
                .mmap
                .get(start..end)
                .ok_or_else(|| WeightStoreError::MalformedSafetensors {
                    path: shard.path.clone(),
                    message: format!("tensor {:?} payload is outside the mapped shard", self.key),
                })?;

        if let TensorSelection::Contiguous {
            offset_elements,
            shape,
        } = &self.selection
        {
            let selected_data = contiguous_safetensors_data(
                &self.key,
                &shard.path,
                &info.shape,
                data,
                *offset_elements,
                shape,
            )?;
            let view =
                TensorView::new(info.dtype, shape.clone(), selected_data).map_err(|error| {
                    WeightStoreError::MalformedSafetensors {
                        path: shard.path.clone(),
                        message: format!("tensor {:?}: {error}", self.key),
                    }
                })?;
            let source_value =
                Array::try_from(view).map_err(|source| WeightStoreError::MlxConversion {
                    key: self.key.clone(),
                    source,
                })?;
            let materialized = source_value
                .copy(execution_stream)
                .map_err(|source| self.mlx_error("copy", source))?;
            return Ok(PendingWeightMaterialization {
                output: materialized,
                _source: source_value,
                _gguf_group: None,
                lease: Some(self),
                source_stream: source_stream.clone(),
                execution_stream: execution_stream.clone(),
                borrowed_source: false,
                completed: false,
            });
        }

        // Axis-zero ranges are contiguous in safetensors storage. Slice the
        // mmap bytes before constructing an MLX array: `Array::try_from` copies
        // its complete `TensorView`, so selecting afterwards would transiently
        // materialize an entire packed expert bank for every requested expert.
        if let TensorSelection::Range {
            axis: 0,
            start,
            end,
        } = &self.selection
        {
            let outer = info.shape[0];
            let row_bytes = data
                .len()
                .checked_div(outer)
                .filter(|_| data.len() % outer == 0)
                .ok_or_else(|| WeightStoreError::MalformedSafetensors {
                    path: shard.path.clone(),
                    message: format!(
                        "tensor {:?} payload is not divisible by its outer dimension",
                        self.key
                    ),
                })?;
            let selected_start =
                start
                    .checked_mul(row_bytes)
                    .ok_or_else(|| WeightStoreError::Overflow {
                        context: format!("axis-zero byte start for tensor {:?}", self.key),
                    })?;
            let selected_end =
                end.checked_mul(row_bytes)
                    .ok_or_else(|| WeightStoreError::Overflow {
                        context: format!("axis-zero byte end for tensor {:?}", self.key),
                    })?;
            let selected_data = data.get(selected_start..selected_end).ok_or_else(|| {
                WeightStoreError::MalformedSafetensors {
                    path: shard.path.clone(),
                    message: format!(
                        "axis-zero selection for tensor {:?} is outside its payload",
                        self.key
                    ),
                }
            })?;
            let view = TensorView::new(info.dtype, self.output_shape.clone(), selected_data)
                .map_err(|error| WeightStoreError::MalformedSafetensors {
                    path: shard.path.clone(),
                    message: format!("tensor {:?}: {error}", self.key),
                })?;
            let source_value =
                Array::try_from(view).map_err(|source| WeightStoreError::MlxConversion {
                    key: self.key.clone(),
                    source,
                })?;
            let materialized = source_value
                .copy(execution_stream)
                .map_err(|source| self.mlx_error("copy", source))?;
            return Ok(PendingWeightMaterialization {
                output: materialized,
                _source: source_value,
                _gguf_group: None,
                lease: Some(self),
                source_stream: source_stream.clone(),
                execution_stream: execution_stream.clone(),
                borrowed_source: false,
                completed: false,
            });
        }

        let view = TensorView::new(info.dtype, info.shape.clone(), data).map_err(|error| {
            WeightStoreError::MalformedSafetensors {
                path: shard.path.clone(),
                message: format!("tensor {:?}: {error}", self.key),
            }
        })?;

        // This mmap-derived array never leaves this method. The lease pins the
        // mmap until selection, copy, evaluation, and synchronization finish.
        let source_value =
            Array::try_from(view).map_err(|source| WeightStoreError::MlxConversion {
                key: self.key.clone(),
                source,
            })?;
        let materialized = match &self.selection {
            TensorSelection::Full => source_value
                .copy(execution_stream)
                .map_err(|source| self.mlx_error("copy", source)),
            TensorSelection::Range { axis, start, end } => materialize_range(
                &self.key,
                source_value.clone(),
                &self.metadata.shape,
                *axis,
                *start,
                *end,
                source_stream,
                execution_stream,
            ),
            TensorSelection::Indices { axis, indices } => materialize_indices(
                &self.key,
                &source_value,
                *axis,
                indices,
                source_stream,
                execution_stream,
            ),
            TensorSelection::Contiguous {
                offset_elements,
                shape,
            } => materialize_contiguous(
                &self.key,
                &source_value,
                *offset_elements,
                shape,
                source_stream,
                execution_stream,
            ),
        }?;
        Ok(PendingWeightMaterialization {
            output: materialized,
            _source: source_value,
            _gguf_group: None,
            lease: Some(self),
            source_stream: source_stream.clone(),
            execution_stream: execution_stream.clone(),
            borrowed_source: false,
            completed: false,
        })
    }

    fn prepare_gguf(
        self,
        source: GgufLeaseSource,
        source_stream: &Stream,
        execution_stream: &Stream,
    ) -> Result<PendingWeightMaterialization, WeightStoreError> {
        let GgufLeaseSource {
            store,
            entry,
            physical_selection,
            physical_byte_len,
            selection_is_materialized,
        } = source;
        let cache_key = GgufGroupCacheKey {
            checkpoint: entry.checkpoint,
            physical_name: entry.physical_name.clone(),
            selection: physical_selection.clone(),
        };
        let mut groups = store
            .converted_groups
            .lock()
            .map_err(|_| WeightStoreError::CachePoisoned)?;
        groups.retain(|_, group| group.strong_count() > 0);
        let group = if let Some(cached) = groups.get(&cache_key).and_then(Weak::upgrade) {
            store
                .statistics
                .coalesced_group_hits
                .fetch_add(1, Ordering::Relaxed);
            cached
        } else {
            let converted = store
                .readers
                .lock()
                .map_err(|_| WeightStoreError::CachePoisoned)?
                .materialize(
                    entry.checkpoint,
                    &entry.physical_name,
                    physical_selection.as_ref(),
                    store.max_cached_readers,
                    &self.key,
                )?;
            store
                .statistics
                .physical_reads
                .fetch_add(1, Ordering::Relaxed);
            store
                .statistics
                .physical_read_bytes
                .fetch_add(physical_byte_len, Ordering::Relaxed);
            let cached = Arc::new(CachedGgufGroup {
                arrays: converted.into_arrays(),
            });
            groups.insert(cache_key, Arc::downgrade(&cached));
            cached
        };
        drop(groups);
        let source_value = group
            .arrays
            .iter()
            .find_map(|(name, value)| (name == &entry.original_name).then(|| value.clone()))
            .ok_or_else(|| WeightStoreError::Gguf {
                key: self.key.clone(),
                message: format!(
                    "physical tensor {:?} did not produce logical output {:?}",
                    entry.physical_name, entry.original_name
                ),
            })?;
        let materialized = if selection_is_materialized && source_stream == execution_stream {
            source_value.clone()
        } else if selection_is_materialized {
            source_value
                .copy(execution_stream)
                .map_err(|source| self.mlx_error("copy", source))?
        } else {
            match &self.selection {
                TensorSelection::Range { axis, start, end } => materialize_range(
                    &self.key,
                    source_value.clone(),
                    &self.metadata.shape,
                    *axis,
                    *start,
                    *end,
                    source_stream,
                    execution_stream,
                )?,
                TensorSelection::Indices { axis, indices } => materialize_indices(
                    &self.key,
                    &source_value,
                    *axis,
                    indices,
                    source_stream,
                    execution_stream,
                )?,
                TensorSelection::Full => unreachable!("handled above"),
                TensorSelection::Contiguous {
                    offset_elements,
                    shape,
                } => materialize_contiguous(
                    &self.key,
                    &source_value,
                    *offset_elements,
                    shape,
                    source_stream,
                    execution_stream,
                )?,
            }
        };
        Ok(PendingWeightMaterialization {
            output: materialized,
            _source: source_value,
            _gguf_group: Some(group),
            lease: Some(self),
            source_stream: source_stream.clone(),
            execution_stream: execution_stream.clone(),
            borrowed_source: false,
            completed: false,
        })
    }

    fn mlx_error(
        &self,
        operation: &'static str,
        source: safemlx::error::Exception,
    ) -> WeightStoreError {
        WeightStoreError::Mlx {
            key: self.key.clone(),
            operation,
            source,
        }
    }

    fn retain_mapping_after_sync_failure(&self) {
        // A failed synchronization leaves the runtime's dependency state
        // unknowable. Permanently retaining one Arc is conservative and avoids
        // releasing bytes that submitted MLX work may still reference.
        if let WeightLeaseSource::Safetensors(shard) = &self.source {
            std::mem::forget(Arc::clone(shard));
        }
    }
}

/// Scheduled tensor materialization that still pins its mmap-backed sources.
pub(crate) struct PendingWeightMaterialization {
    output: Array,
    _source: Array,
    _gguf_group: Option<Arc<CachedGgufGroup>>,
    lease: Option<WeightLease>,
    source_stream: Stream,
    execution_stream: Stream,
    borrowed_source: bool,
    completed: bool,
}

impl PendingWeightMaterialization {
    /// Returns the lazy materialized output.
    pub(crate) fn output(&self) -> &Array {
        &self.output
    }

    /// Evaluates this output and releases its source dependencies.
    pub(crate) fn finish(mut self) -> Result<Array, WeightStoreError> {
        let output = if self.borrowed_source {
            self.output.copy(&self.source_stream).map_err(|source| {
                self.lease
                    .as_ref()
                    .expect("pending materialization retains its lease")
                    .mlx_error("borrowed source copy", source)
            })?
        } else {
            self.output.clone()
        };
        eval([&output]).map_err(|source| {
            self.lease
                .as_ref()
                .expect("pending materialization retains its lease")
                .mlx_error("evaluation", source)
        })?;
        self.completed = true;
        self.lease.take();
        Ok(output)
    }

    /// Marks a batch member complete after a containing output was evaluated.
    pub(crate) fn complete(mut self) {
        self.completed = true;
        self.lease.take();
    }
}

impl Drop for PendingWeightMaterialization {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        let source = self.source_stream.synchronize();
        let execution = self.execution_stream.synchronize();
        if source.is_err() || execution.is_err() {
            if let Some(lease) = &self.lease {
                lease.retain_mapping_after_sync_failure();
            }
        }
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
                formatter.write_str("a tensor-to-shard object with unique tensor names")
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

fn validate_relative_shard_path(path: &Path) -> Result<PathBuf, WeightStoreError> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(WeightStoreError::UnsafeShardPath {
            path: path.to_path_buf(),
        });
    }
    Ok(path.to_path_buf())
}

fn canonicalize(path: &Path) -> Result<PathBuf, WeightStoreError> {
    std::fs::canonicalize(path).map_err(|source| WeightStoreError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn canonical_checkpoint_access_root(path: &Path) -> Result<PathBuf, WeightStoreError> {
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

    // Hugging Face snapshots store ordinary relative shard names in the index,
    // but materialize those names as symlinks into the repository-local blobs
    // directory. Treat that repository cache directory as the containment
    // boundary while preserving the model-directory boundary elsewhere.
    canonicalize(repository_root)
}

fn inspect_file(path: &Path) -> Result<BTreeMap<String, WeightMetadata>, WeightStoreError> {
    let file = File::open(path).map_err(|source| WeightStoreError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    // SAFETY: the mapping is retained until all metadata-only TensorViews have
    // been converted into owned WeightMetadata values below.
    let mmap =
        unsafe { MmapOptions::new().map(&file) }.map_err(|source| WeightStoreError::Mmap {
            path: path.to_path_buf(),
            source,
        })?;
    let checkpoint = SafeTensors::deserialize(&mmap).map_err(|error| {
        WeightStoreError::MalformedSafetensors {
            path: path.to_path_buf(),
            message: error.to_string(),
        }
    })?;
    let mut metadata = BTreeMap::new();
    for (key, view) in checkpoint.iter() {
        metadata.insert(key.to_string(), metadata_for_view(key, path, &view)?);
    }
    Ok(metadata)
}

fn metadata_for_view(
    key: &str,
    path: &Path,
    view: &safetensors::tensor::TensorView<'_>,
) -> Result<WeightMetadata, WeightStoreError> {
    metadata_for_parts(key, path, view.dtype(), view.shape(), view.data().len())
}

fn metadata_for_info(
    key: &str,
    path: &Path,
    info: &TensorInfo,
) -> Result<WeightMetadata, WeightStoreError> {
    let payload_len = info
        .data_offsets
        .1
        .checked_sub(info.data_offsets.0)
        .ok_or_else(|| WeightStoreError::MalformedSafetensors {
            path: path.to_path_buf(),
            message: format!("tensor {key:?} has descending payload offsets"),
        })?;
    metadata_for_parts(key, path, info.dtype, &info.shape, payload_len)
}

fn metadata_for_parts(
    key: &str,
    path: &Path,
    dtype: Dtype,
    shape: &[usize],
    payload_len: usize,
) -> Result<WeightMetadata, WeightStoreError> {
    let elements = shape.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| WeightStoreError::Overflow {
                context: format!("element count for tensor {key:?}"),
            })
    })?;
    let bits = elements
        .checked_mul(dtype.bitsize())
        .ok_or_else(|| WeightStoreError::Overflow {
            context: format!("encoded bit length for tensor {key:?}"),
        })?;
    if bits % 8 != 0 {
        return Err(WeightStoreError::MalformedSafetensors {
            path: path.to_path_buf(),
            message: format!("tensor {key:?} has a non-byte-aligned payload"),
        });
    }
    let logical_byte_len = bits / 8;
    if logical_byte_len != payload_len {
        return Err(WeightStoreError::MalformedSafetensors {
            path: path.to_path_buf(),
            message: format!("tensor {key:?} payload length contradicts its shape and dtype"),
        });
    }
    Ok(WeightMetadata {
        name: key.to_string(),
        shape: shape.to_vec(),
        stored_dtype: dtype.into(),
        logical_byte_len,
        backing_shard: Some(path.to_path_buf()),
    })
}

fn validate_selection(
    key: &str,
    shape: &[usize],
    selection: &TensorSelection,
) -> Result<Vec<usize>, WeightStoreError> {
    // Validate the complete shape with an explicit checked element count even
    // though the safetensors parser performs its own bounds validation.
    shape.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| WeightStoreError::Overflow {
                context: format!("shape for tensor {key:?}"),
            })
    })?;
    let mut output = shape.to_vec();
    match selection {
        TensorSelection::Full => {}
        TensorSelection::Range { axis, start, end } => {
            let Some(dimension) = shape.get(*axis) else {
                return Err(WeightStoreError::InvalidSelection {
                    key: key.to_string(),
                    message: format!(
                        "axis {axis} is outside rank {} shape {shape:?}",
                        shape.len()
                    ),
                });
            };
            if start >= end || *end > *dimension {
                return Err(WeightStoreError::InvalidSelection {
                    key: key.to_string(),
                    message: format!(
                        "range {start}..{end} is invalid for axis {axis} dimension {dimension}"
                    ),
                });
            }
            output[*axis] = end - start;
        }
        TensorSelection::Indices { axis, indices } => {
            let Some(dimension) = shape.get(*axis) else {
                return Err(WeightStoreError::InvalidSelection {
                    key: key.to_string(),
                    message: format!(
                        "axis {axis} is outside rank {} shape {shape:?}",
                        shape.len()
                    ),
                });
            };
            if indices.is_empty() {
                return Err(WeightStoreError::InvalidSelection {
                    key: key.to_string(),
                    message: "index selection must be non-empty".into(),
                });
            }
            if let Some(index) = indices.iter().find(|index| **index >= *dimension) {
                return Err(WeightStoreError::InvalidSelection {
                    key: key.to_string(),
                    message: format!("index {index} is outside axis {axis} dimension {dimension}"),
                });
            }
            output[*axis] = indices.len();
        }
        TensorSelection::Contiguous {
            offset_elements,
            shape: selected,
        } => {
            if selected.is_empty() || selected.contains(&0) {
                return Err(WeightStoreError::InvalidSelection {
                    key: key.to_string(),
                    message: "contiguous selection shape must be non-empty and nonzero".into(),
                });
            }
            let full_elements = shape.iter().try_fold(1usize, |count, dimension| {
                count
                    .checked_mul(*dimension)
                    .ok_or_else(|| WeightStoreError::Overflow {
                        context: format!("element count for tensor {key:?}"),
                    })
            })?;
            let selected_elements = selected.iter().try_fold(1usize, |count, dimension| {
                count
                    .checked_mul(*dimension)
                    .ok_or_else(|| WeightStoreError::Overflow {
                        context: format!("contiguous selection size for tensor {key:?}"),
                    })
            })?;
            let end = offset_elements
                .checked_add(selected_elements)
                .ok_or_else(|| WeightStoreError::Overflow {
                    context: format!("contiguous selection end for tensor {key:?}"),
                })?;
            if end > full_elements {
                return Err(WeightStoreError::InvalidSelection {
                    key: key.to_string(),
                    message: format!(
                        "contiguous scalar span {offset_elements}..{end} exceeds {full_elements} elements"
                    ),
                });
            }
            output = selected.clone();
        }
    }
    output.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| WeightStoreError::Overflow {
                context: format!("selected shape for tensor {key:?}"),
            })
    })?;
    Ok(output)
}

fn selected_byte_len(
    key: &str,
    metadata: &WeightMetadata,
    selection: &TensorSelection,
    output_shape: &[usize],
) -> Result<usize, WeightStoreError> {
    if matches!(selection, TensorSelection::Full) {
        return Ok(metadata.logical_byte_len);
    }
    let full_elements = metadata.shape.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| WeightStoreError::Overflow {
                context: format!("element count for tensor {key:?}"),
            })
    })?;
    let selected_elements = output_shape.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| WeightStoreError::Overflow {
                context: format!("selected element count for tensor {key:?}"),
            })
    })?;
    let scaled = metadata
        .logical_byte_len
        .checked_mul(selected_elements)
        .ok_or_else(|| WeightStoreError::Overflow {
            context: format!("selected byte length for tensor {key:?}"),
        })?;
    if full_elements == 0 || scaled % full_elements != 0 {
        return Err(WeightStoreError::InvalidSelection {
            key: key.to_string(),
            message: "selection does not have a whole-byte encoded length".into(),
        });
    }
    Ok(scaled / full_elements)
}

fn contiguous_safetensors_data<'a>(
    key: &str,
    path: &Path,
    full_shape: &[usize],
    data: &'a [u8],
    offset_elements: usize,
    selected_shape: &[usize],
) -> Result<&'a [u8], WeightStoreError> {
    let full_elements = full_shape.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| WeightStoreError::Overflow {
                context: format!("safetensors element count for tensor {key:?}"),
            })
    })?;
    let scalar_bytes = data
        .len()
        .checked_div(full_elements)
        .filter(|_| full_elements > 0 && data.len().is_multiple_of(full_elements))
        .ok_or_else(|| WeightStoreError::MalformedSafetensors {
            path: path.to_path_buf(),
            message: format!("tensor {key:?} payload has no integral scalar width"),
        })?;
    let selected_elements = selected_shape.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| WeightStoreError::Overflow {
                context: format!("contiguous selection size for tensor {key:?}"),
            })
    })?;
    let byte_start =
        offset_elements
            .checked_mul(scalar_bytes)
            .ok_or_else(|| WeightStoreError::Overflow {
                context: format!("contiguous byte start for tensor {key:?}"),
            })?;
    let byte_end = selected_elements
        .checked_mul(scalar_bytes)
        .and_then(|bytes| byte_start.checked_add(bytes))
        .ok_or_else(|| WeightStoreError::Overflow {
            context: format!("contiguous byte end for tensor {key:?}"),
        })?;
    data.get(byte_start..byte_end)
        .ok_or_else(|| WeightStoreError::MalformedSafetensors {
            path: path.to_path_buf(),
            message: format!("contiguous selection for tensor {key:?} is outside its payload"),
        })
}

fn materialize_contiguous(
    key: &str,
    source: &Array,
    offset_elements: usize,
    shape: &[usize],
    source_stream: &Stream,
    execution_stream: &Stream,
) -> Result<Array, WeightStoreError> {
    let elements = shape.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| WeightStoreError::Overflow {
                context: format!("contiguous materialization size for tensor {key:?}"),
            })
    })?;
    let end = offset_elements
        .checked_add(elements)
        .ok_or_else(|| WeightStoreError::Overflow {
            context: format!("contiguous materialization end for tensor {key:?}"),
        })?;
    let flattened =
        source
            .reshape(&[-1], source_stream)
            .map_err(|source| WeightStoreError::Mlx {
                key: key.to_string(),
                operation: "flatten contiguous selection",
                source,
            })?;
    let selected = materialize_range(
        key,
        flattened,
        &[source.size()],
        0,
        offset_elements,
        end,
        source_stream,
        execution_stream,
    )?;
    let shape = shape
        .iter()
        .map(|dimension| to_i32(key, "contiguous output dimension", *dimension))
        .collect::<Result<Vec<_>, _>>()?;
    selected
        .reshape(&shape, execution_stream)
        .map_err(|source| WeightStoreError::Mlx {
            key: key.to_string(),
            operation: "reshape contiguous selection",
            source,
        })
}

#[allow(clippy::too_many_arguments)]
fn materialize_range(
    key: &str,
    source: Array,
    source_shape: &[usize],
    axis: usize,
    start: usize,
    end: usize,
    source_stream: &Stream,
    execution_stream: &Stream,
) -> Result<Array, WeightStoreError> {
    let axis_i32 = to_i32(key, "axis", axis)?;
    let front = if axis == 0 {
        source
    } else {
        source
            .move_axis(axis_i32, 0, source_stream)
            .map_err(|source| mlx_error(key, "move range axis", source))?
    };
    let start = to_i32(key, "range start", start)?;
    let end = to_i32(key, "range end", end)?;
    let selected = front
        .try_index_device(start..end, source_stream)
        .map_err(|source| mlx_error(key, "range selection", source))?;
    let selected = if axis == 0 {
        selected
    } else {
        selected
            .move_axis(0, axis_i32, source_stream)
            .map_err(|source| mlx_error(key, "restore range axis", source))?
    };
    let selected = if axis == 0 {
        selected
    } else {
        // Inner-axis ranges are non-contiguous views. Compact only the selected
        // result, keeping the temporary bounded by the output shape.
        let mut output_shape = source_shape.to_vec();
        output_shape[axis] =
            usize::try_from(end - start).map_err(|_| WeightStoreError::Overflow {
                context: format!("selected range length for tensor {key:?}"),
            })?;
        let mlx_shape = output_shape
            .iter()
            .map(|dimension| to_i32(key, "selected dimension", *dimension))
            .collect::<Result<Vec<_>, _>>()?;
        selected
            .flatten(None, None, source_stream)
            .and_then(|value| value.reshape(&mlx_shape, source_stream))
            .map_err(|source| mlx_error(key, "range compaction", source))?
    };
    selected
        .copy(execution_stream)
        .map_err(|source| mlx_error(key, "copy", source))
}

fn materialize_indices(
    key: &str,
    source: &Array,
    axis: usize,
    indices: &[usize],
    source_stream: &Stream,
    execution_stream: &Stream,
) -> Result<Array, WeightStoreError> {
    let axis = to_i32(key, "axis", axis)?;
    let indices = indices
        .iter()
        .map(|index| to_i32(key, "tensor index", *index))
        .collect::<Result<Vec<_>, _>>()?;
    let count = to_i32(key, "index count", indices.len())?;
    let index_array = Array::from_slice(&indices, &[count])
        .copy(source_stream)
        .map_err(|source| mlx_error(key, "index upload", source))?;
    source
        .take_axis(&index_array, axis, source_stream)
        .and_then(|selected| selected.copy(execution_stream))
        .map_err(|source| mlx_error(key, "ordered index selection", source))
}

fn to_i32(key: &str, what: &'static str, value: usize) -> Result<i32, WeightStoreError> {
    i32::try_from(value).map_err(|_| WeightStoreError::Overflow {
        context: format!("{what} for tensor {key:?} does not fit in i32"),
    })
}

fn mlx_error(
    key: &str,
    operation: &'static str,
    source: safemlx::error::Exception,
) -> WeightStoreError {
    WeightStoreError::Mlx {
        key: key.to_string(),
        operation,
        source,
    }
}

fn is_supported_execution_dtype(dtype: Dtype) -> bool {
    matches!(
        dtype,
        Dtype::BOOL
            | Dtype::U8
            | Dtype::I8
            | Dtype::I16
            | Dtype::U16
            | Dtype::F16
            | Dtype::BF16
            | Dtype::I32
            | Dtype::U32
            | Dtype::F32
            | Dtype::F64
            | Dtype::I64
            | Dtype::U64
            | Dtype::F8_E4M3
    )
}

fn safetensors_dtype_alignment(dtype: Dtype) -> usize {
    match dtype {
        Dtype::I64 | Dtype::U64 | Dtype::F64 => 8,
        Dtype::I32 | Dtype::U32 | Dtype::F32 => 4,
        Dtype::I16 | Dtype::U16 | Dtype::F16 | Dtype::BF16 => 2,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safemlx::{Device, DeviceType, Dtype as MlxDtype};
    use safemlx_gguf::{GgmlType, TensorInput, Writer};
    use safetensors::tensor::{serialize_to_file, TensorView};

    trait AcquireBoundedForTest {
        fn acquire(
            &self,
            key: &str,
            selection: TensorSelection,
        ) -> Result<WeightLease, WeightStoreError>;
    }

    impl<T: WeightStore> AcquireBoundedForTest for T {
        fn acquire(
            &self,
            key: &str,
            selection: TensorSelection,
        ) -> Result<WeightLease, WeightStoreError> {
            WeightStore::acquire_with_policy(self, key, selection, WeightReadPolicy::RequireBounded)
        }
    }

    fn cpu_stream() -> Stream {
        Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
    }

    #[test]
    fn memory_store_preserves_catalog_and_materializes_bounded_ranges() {
        let stream = cpu_stream();
        let source = Array::from_slice(&[0f32, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
        let store = MemoryWeightStore::new([("matrix".to_string(), source)]).unwrap();

        assert_eq!(store.backend(), WeightStoreBackend::Memory);
        assert_eq!(store.keys(), ["matrix"]);
        assert_eq!(store.metadata("matrix").unwrap().shape, [2, 3]);
        let lease = store
            .acquire(
                "matrix",
                TensorSelection::Range {
                    axis: 1,
                    start: 1,
                    end: 3,
                },
            )
            .unwrap();
        assert_eq!(lease.output_shape(), [2, 2]);
        assert_eq!(lease.selected_byte_len(), 4 * std::mem::size_of::<f32>());
        assert_eq!(
            lease
                .materialize(&stream, &stream)
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<f32>(),
            &[1.0, 2.0, 4.0, 5.0]
        );
        let diagnostics = store.diagnostics().unwrap();
        assert_eq!(diagnostics.backend, WeightStoreBackend::Memory);
        assert_eq!(diagnostics.physical_reads, 0);
    }

    fn safetensors_shard(lease: &WeightLease) -> &Arc<MappedShard> {
        match &lease.source {
            WeightLeaseSource::Safetensors(shard) => shard,
            WeightLeaseSource::Gguf(_) => panic!("expected safetensors lease"),
            WeightLeaseSource::Memory(_) => panic!("expected safetensors lease"),
        }
    }

    fn write_index(dir: &Path, mappings: &[(&str, &str)]) {
        let weight_map = mappings
            .iter()
            .map(|(key, shard)| ((*key).to_string(), serde_json::json!(shard)))
            .collect::<serde_json::Map<_, _>>();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
        )
        .unwrap();
    }

    fn write_i32(path: &Path, name: &str, values: &[i32], shape: Vec<usize>) {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let view = TensorView::new(Dtype::I32, shape, &bytes).unwrap();
        serialize_to_file([(name, view)], None, path).unwrap();
    }

    fn write_two_i32(path: &Path) {
        let left_bytes = [1i32, 2]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let right_bytes = [3i32, 4]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let left = TensorView::new(Dtype::I32, vec![2], &left_bytes).unwrap();
        let right = TensorView::new(Dtype::I32, vec![2], &right_bytes).unwrap();
        serialize_to_file([("z_tensor", left), ("a_tensor", right)], None, path).unwrap();
    }

    fn write_affine_gguf(path: &Path) {
        let bytes = [0u8; 36];
        Writer::default()
            .write(
                std::fs::File::create(path).unwrap(),
                &BTreeMap::new(),
                &[TensorInput {
                    name: "bank.weight",
                    dimensions: &[32, 2],
                    ggml_type: GgmlType::Q4_0,
                    data: &bytes,
                }],
            )
            .unwrap();
    }

    fn write_wide_affine_gguf(path: &Path) {
        let blocks = [[0x00u8; 18], [0x11u8; 18], [0x22u8; 18], [0x33u8; 18]];
        let bytes = blocks.into_iter().flatten().collect::<Vec<_>>();
        Writer::default()
            .write(
                std::fs::File::create(path).unwrap(),
                &BTreeMap::new(),
                &[TensorInput {
                    name: "bank.weight",
                    dimensions: &[64, 2],
                    ggml_type: GgmlType::Q4_0,
                    data: &bytes,
                }],
            )
            .unwrap();
    }

    fn write_dense_gguf(path: &Path, name: &str, value: f32) {
        let bytes = value.to_le_bytes();
        Writer::default()
            .write(
                std::fs::File::create(path).unwrap(),
                &BTreeMap::new(),
                &[TensorInput {
                    name,
                    dimensions: &[1],
                    ggml_type: GgmlType::F32,
                    data: &bytes,
                }],
            )
            .unwrap();
    }

    fn write_block_gguf(path: &Path, ty: GgmlType, byte_len: usize) {
        let bytes = vec![0u8; byte_len];
        Writer::default()
            .write(
                std::fs::File::create(path).unwrap(),
                &BTreeMap::new(),
                &[TensorInput {
                    name: "bank.weight",
                    dimensions: &[64, 2],
                    ggml_type: ty,
                    data: &bytes,
                }],
            )
            .unwrap();
    }

    fn gguf_physical_selection(lease: &WeightLease) -> Option<GgufTensorSelection> {
        match &lease.source {
            WeightLeaseSource::Gguf(source) => source.physical_selection.clone(),
            WeightLeaseSource::Safetensors(_) => panic!("expected GGUF lease"),
            WeightLeaseSource::Memory(_) => panic!("expected GGUF lease"),
        }
    }

    #[test]
    fn gguf_store_rejects_translated_collisions_across_checkpoints() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("first.gguf");
        let second = dir.path().join("second.gguf");
        write_dense_gguf(&first, "text.weight", 1.0);
        write_dense_gguf(&second, "vision.weight", 2.0);
        let builder = GgufWeightStore::builder()
            .add_checkpoint(GgufCheckpoint::open(first).unwrap(), |_| {
                "shared.weight".into()
            })
            .unwrap();
        let error = builder
            .add_checkpoint(GgufCheckpoint::open(second).unwrap(), |_| {
                "shared.weight".into()
            })
            .unwrap_err();
        assert!(matches!(error, WeightStoreError::Gguf { .. }));
    }

    #[test]
    fn gguf_store_cataloging_does_not_touch_payload_readers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        write_dense_gguf(&path, "value.weight", 3.0);
        let store =
            GgufWeightStore::new(GgufCheckpoint::open(path).unwrap(), |name| name.to_string())
                .unwrap();
        assert_eq!(store.keys(), ["value.weight"]);
        assert_eq!(store.metadata("value.weight").unwrap().shape, [1]);
        let diagnostics = store.diagnostics().unwrap();
        assert_eq!(diagnostics.currently_mapped_shards, 0);
        assert!(diagnostics.touched_shard_paths.is_empty());
        assert_eq!(diagnostics.physical_reads, 0);
    }

    #[test]
    fn gguf_affine_companions_coalesce_selected_physical_reads() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        write_affine_gguf(&path);
        let store =
            GgufWeightStore::new(GgufCheckpoint::open(path).unwrap(), |name| name.to_string())
                .unwrap();
        let stream = cpu_stream();
        let selection = TensorSelection::Range {
            axis: 0,
            start: 1,
            end: 2,
        };
        let weight = store
            .acquire("bank.weight", selection.clone())
            .unwrap()
            .prepare_materialization(&stream, &stream)
            .unwrap();
        let scales = store
            .acquire("bank.scales", selection.clone())
            .unwrap()
            .prepare_materialization(&stream, &stream)
            .unwrap();
        let biases = store
            .acquire("bank.biases", selection)
            .unwrap()
            .prepare_materialization(&stream, &stream)
            .unwrap();
        weight.finish().unwrap();
        scales.finish().unwrap();
        biases.finish().unwrap();

        let diagnostics = store.diagnostics().unwrap();
        assert_eq!(diagnostics.backend, WeightStoreBackend::Gguf);
        assert_eq!(diagnostics.physical_reads, 1);
        assert_eq!(diagnostics.physical_read_bytes, 18);
        assert_eq!(diagnostics.coalesced_group_hits, 2);
    }

    #[test]
    fn gguf_inner_affine_companions_normalize_to_one_bounded_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        write_wide_affine_gguf(&path);
        let store =
            GgufWeightStore::new(GgufCheckpoint::open(path).unwrap(), |name| name.to_string())
                .unwrap();
        let stream = cpu_stream();
        let weight = store
            .acquire_with_policy(
                "bank.weight",
                TensorSelection::Range {
                    axis: 1,
                    start: 4,
                    end: 8,
                },
                WeightReadPolicy::RequireBounded,
            )
            .unwrap()
            .prepare_materialization(&stream, &stream)
            .unwrap();
        let scales = store
            .acquire_with_policy(
                "bank.scales",
                TensorSelection::Range {
                    axis: 1,
                    start: 1,
                    end: 2,
                },
                WeightReadPolicy::RequireBounded,
            )
            .unwrap()
            .prepare_materialization(&stream, &stream)
            .unwrap();
        let biases = store
            .acquire_with_policy(
                "bank.biases",
                TensorSelection::Range {
                    axis: 1,
                    start: 1,
                    end: 2,
                },
                WeightReadPolicy::RequireBounded,
            )
            .unwrap()
            .prepare_materialization(&stream, &stream)
            .unwrap();
        assert_eq!(weight.output().shape(), [2, 4]);
        assert_eq!(scales.output().shape(), [2, 1]);
        assert_eq!(biases.output().shape(), [2, 1]);
        weight.finish().unwrap();
        scales.finish().unwrap();
        biases.finish().unwrap();

        let diagnostics = store.diagnostics().unwrap();
        assert_eq!(diagnostics.physical_reads, 1);
        assert_eq!(diagnostics.physical_read_bytes, 36);
        assert_eq!(diagnostics.coalesced_group_hits, 2);
    }

    #[test]
    fn gguf_bounded_policy_rejects_misalignment_before_payload_io() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        write_wide_affine_gguf(&path);
        let store =
            GgufWeightStore::new(GgufCheckpoint::open(path).unwrap(), |name| name.to_string())
                .unwrap();
        let selection = TensorSelection::Range {
            axis: 1,
            start: 1,
            end: 5,
        };
        assert!(matches!(
            store.acquire_with_policy(
                "bank.weight",
                selection.clone(),
                WeightReadPolicy::RequireBounded,
            ),
            Err(WeightStoreError::BoundedSelectionUnavailable { .. })
        ));
        let diagnostics = store.diagnostics().unwrap();
        assert_eq!(diagnostics.physical_reads, 0);
        assert_eq!(diagnostics.physical_read_bytes, 0);
        assert!(diagnostics.touched_shard_paths.is_empty());

        let stream = cpu_stream();
        let value = store
            .acquire_with_policy(
                "bank.weight",
                selection,
                WeightReadPolicy::AllowFullTensorRead,
            )
            .unwrap()
            .materialize(&stream, &stream)
            .unwrap();
        assert_eq!(value.shape(), [2, 4]);
        let diagnostics = store.diagnostics().unwrap();
        assert_eq!(diagnostics.physical_reads, 1);
        assert_eq!(diagnostics.physical_read_bytes, 72);
    }

    #[test]
    fn gguf_mxfp4_and_iq_outputs_map_to_native_block_coordinates() {
        let expected = Some(GgufTensorSelection::Range {
            axis: 1,
            start: 32,
            end: 64,
        });
        for (ty, byte_len, outputs) in [
            (
                GgmlType::MxFp4,
                68,
                vec![("bank.weight", 4, 8), ("bank.scales", 1, 2)],
            ),
            (GgmlType::IQ4NL, 72, vec![("bank.weight", 18, 36)]),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("model.gguf");
            write_block_gguf(&path, ty, byte_len);
            let store =
                GgufWeightStore::new(GgufCheckpoint::open(path).unwrap(), |name| name.to_string())
                    .unwrap();
            for (key, start, end) in outputs {
                let lease = store
                    .acquire_with_policy(
                        key,
                        TensorSelection::Range {
                            axis: 1,
                            start,
                            end,
                        },
                        WeightReadPolicy::RequireBounded,
                    )
                    .unwrap();
                assert_eq!(gguf_physical_selection(&lease), expected);
            }
            assert_eq!(store.diagnostics().unwrap().physical_reads, 0);
        }
    }

    #[test]
    fn indexed_catalog_is_sorted_without_mapping_payloads() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("broken.safetensors"), b"not a checkpoint").unwrap();
        write_index(
            dir.path(),
            &[
                ("z.weight", "broken.safetensors"),
                ("a.weight", "missing.safetensors"),
            ],
        );

        let store = SafetensorsWeightStore::open(dir.path()).unwrap();
        assert_eq!(store.keys(), ["a.weight", "z.weight"]);
        assert_eq!(
            store.diagnostics().unwrap(),
            WeightStoreDiagnostics {
                backend: WeightStoreBackend::Safetensors,
                mapping_hits: 0,
                mapping_misses: 0,
                evictions: 0,
                currently_mapped_shards: 0,
                touched_shard_paths: vec![],
                physical_reads: 0,
                physical_read_bytes: 0,
                coalesced_group_hits: 0,
            }
        );
        assert!(matches!(
            store.acquire("a.weight", TensorSelection::Full),
            Err(WeightStoreError::MissingShard { .. })
        ));
        assert!(matches!(
            store.acquire("z.weight", TensorSelection::Full),
            Err(WeightStoreError::MalformedSafetensors { .. })
        ));
    }

    #[test]
    fn reports_contradictory_index_mapping_when_accessed() {
        let dir = tempfile::tempdir().unwrap();
        write_i32(
            &dir.path().join("payload.safetensors"),
            "actual",
            &[1],
            vec![1],
        );
        write_index(dir.path(), &[("claimed", "payload.safetensors")]);
        let store = SafetensorsWeightStore::open(dir.path()).unwrap();
        assert!(matches!(
            store.acquire("claimed", TensorSelection::Full),
            Err(WeightStoreError::ContradictoryIndexMapping { .. })
        ));
        assert_eq!(store.diagnostics().unwrap().currently_mapped_shards, 1);
    }

    #[test]
    fn discovers_direct_and_single_file_directory_catalogs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        write_two_i32(&path);

        let directory = SafetensorsWeightStore::open(dir.path()).unwrap();
        let direct = SafetensorsWeightStore::open(&path).unwrap();
        assert_eq!(directory.keys(), ["a_tensor", "z_tensor"]);
        assert_eq!(direct.keys(), directory.keys());
        assert_eq!(directory.diagnostics().unwrap().currently_mapped_shards, 0);
    }

    #[test]
    fn one_metadata_lookup_caches_the_complete_shard_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        write_two_i32(&path);
        let store = SafetensorsWeightStore::open(dir.path()).unwrap();

        store.metadata("a_tensor").unwrap();

        let cached = store.metadata.lock().unwrap();
        assert_eq!(cached.len(), 2);
        assert!(cached.contains_key("a_tensor"));
        assert!(cached.contains_key("z_tensor"));
    }

    #[test]
    fn rejects_malformed_indexes_and_unsafe_shard_paths() {
        let malformed = tempfile::tempdir().unwrap();
        std::fs::write(
            malformed.path().join("model.safetensors.index.json"),
            b"{invalid",
        )
        .unwrap();
        assert!(matches!(
            SafetensorsWeightStore::open(malformed.path()),
            Err(WeightStoreError::MalformedIndex { .. })
        ));

        let duplicate = tempfile::tempdir().unwrap();
        std::fs::write(
            duplicate.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"weight":"one.safetensors","weight":"two.safetensors"}}"#,
        )
        .unwrap();
        assert!(matches!(
            SafetensorsWeightStore::open(duplicate.path()),
            Err(WeightStoreError::MalformedIndex { .. })
        ));

        for shard in ["../escape.safetensors", "/absolute.safetensors"] {
            let dir = tempfile::tempdir().unwrap();
            write_index(dir.path(), &[("weight", shard)]);
            assert!(matches!(
                SafetensorsWeightStore::open(dir.path()),
                Err(WeightStoreError::UnsafeShardPath { .. })
            ));
        }
    }

    #[test]
    fn maps_only_acquired_shards_and_reuses_one_mapping() {
        let dir = tempfile::tempdir().unwrap();
        write_two_i32(&dir.path().join("local.safetensors"));
        write_i32(
            &dir.path().join("other.safetensors"),
            "other",
            &[5, 6],
            vec![2],
        );
        write_index(
            dir.path(),
            &[
                ("a_tensor", "local.safetensors"),
                ("z_tensor", "local.safetensors"),
                ("other", "other.safetensors"),
            ],
        );
        let store = SafetensorsWeightStore::open(dir.path()).unwrap();
        let first = store.acquire("a_tensor", TensorSelection::Full).unwrap();
        let second = store.acquire("z_tensor", TensorSelection::Full).unwrap();
        assert!(Arc::ptr_eq(
            safetensors_shard(&first),
            safetensors_shard(&second)
        ));
        let diagnostics = store.diagnostics().unwrap();
        assert_eq!(diagnostics.currently_mapped_shards, 1);
        assert_eq!(diagnostics.mapping_misses, 1);
        assert_eq!(diagnostics.mapping_hits, 1);
        assert_eq!(diagnostics.touched_shard_paths.len(), 1);
    }

    #[test]
    fn enforces_capacity_until_leases_drop_then_evicts_lru() {
        let dir = tempfile::tempdir().unwrap();
        write_i32(&dir.path().join("one.safetensors"), "one", &[1], vec![1]);
        write_i32(&dir.path().join("two.safetensors"), "two", &[2], vec![1]);
        write_index(
            dir.path(),
            &[("one", "one.safetensors"), ("two", "two.safetensors")],
        );
        let store = SafetensorsWeightStore::open_with_max_mapped_shards(dir.path(), 1).unwrap();
        let one = store.acquire("one", TensorSelection::Full).unwrap();
        let error = store.acquire("two", TensorSelection::Full).unwrap_err();
        assert!(matches!(
            error,
            WeightStoreError::CapacityExhausted {
                max_mapped_shards: 1,
                ..
            }
        ));
        assert_eq!(one.metadata().shape, [1]);
        let stream = cpu_stream();
        let pinned_value = one.materialize(&stream, &stream).unwrap();
        assert_eq!(pinned_value.evaluated().unwrap().as_slice::<i32>(), &[1]);
        drop(one);

        let two = store.acquire("two", TensorSelection::Full).unwrap();
        assert_eq!(two.metadata().shape, [1]);
        let diagnostics = store.diagnostics().unwrap();
        assert_eq!(diagnostics.currently_mapped_shards, 1);
        assert_eq!(diagnostics.evictions, 1);
        assert_eq!(diagnostics.touched_shard_paths.len(), 2);
    }

    #[test]
    fn materializes_full_ranges_and_ordered_indices() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        write_i32(&path, "matrix", &(0..12).collect::<Vec<_>>(), vec![3, 4]);
        let store = SafetensorsWeightStore::open(&path).unwrap();
        let stream = cpu_stream();

        let full = store
            .acquire("matrix", TensorSelection::Full)
            .unwrap()
            .materialize(&stream, &stream)
            .unwrap();
        let outer = store
            .acquire(
                "matrix",
                TensorSelection::Range {
                    axis: 0,
                    start: 1,
                    end: 3,
                },
            )
            .unwrap()
            .materialize(&stream, &stream)
            .unwrap();
        let inner = store
            .acquire(
                "matrix",
                TensorSelection::Range {
                    axis: 1,
                    start: 1,
                    end: 3,
                },
            )
            .unwrap()
            .materialize(&stream, &stream)
            .unwrap();
        let indexed = store
            .acquire(
                "matrix",
                TensorSelection::Indices {
                    axis: 0,
                    indices: vec![2, 0],
                },
            )
            .unwrap()
            .materialize(&stream, &stream)
            .unwrap();

        assert_eq!(
            full.evaluated().unwrap().as_slice::<i32>(),
            &(0..12).collect::<Vec<_>>()
        );
        assert_eq!(outer.shape(), [2, 4]);
        assert_eq!(
            outer.evaluated().unwrap().as_slice::<i32>(),
            &[4, 5, 6, 7, 8, 9, 10, 11]
        );
        assert_eq!(inner.shape(), [3, 2]);
        assert_eq!(
            inner.evaluated().unwrap().as_slice::<i32>(),
            &[1, 2, 5, 6, 9, 10]
        );
        assert_eq!(indexed.shape(), [2, 4]);
        assert_eq!(
            indexed.evaluated().unwrap().as_slice::<i32>(),
            &[8, 9, 10, 11, 0, 1, 2, 3]
        );
    }

    #[test]
    fn axis_zero_range_constructs_only_the_selected_safetensors_source() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        write_i32(&path, "bank", &(0..12).collect::<Vec<_>>(), vec![3, 4]);
        let store = SafetensorsWeightStore::open(&path).unwrap();
        let stream = cpu_stream();
        let pending = store
            .acquire(
                "bank",
                TensorSelection::Range {
                    axis: 0,
                    start: 1,
                    end: 2,
                },
            )
            .unwrap()
            .prepare_materialization(&stream, &stream)
            .unwrap();

        assert_eq!(pending._source.shape(), [1, 4]);
        assert_eq!(pending.output().shape(), [1, 4]);
        assert_eq!(
            pending
                .finish()
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[4, 5, 6, 7]
        );
    }

    #[test]
    fn validates_selection_and_selected_shapes() {
        let dir = tempfile::tempdir().unwrap();
        write_i32(
            &dir.path().join("model.safetensors"),
            "matrix",
            &[0, 1, 2, 3],
            vec![2, 2],
        );
        let store = SafetensorsWeightStore::open(dir.path()).unwrap();
        assert!(matches!(
            store.acquire("missing", TensorSelection::Full),
            Err(WeightStoreError::UnknownTensor { .. })
        ));
        for selection in [
            TensorSelection::Range {
                axis: 2,
                start: 0,
                end: 1,
            },
            TensorSelection::Range {
                axis: 0,
                start: 1,
                end: 1,
            },
            TensorSelection::Range {
                axis: 0,
                start: 0,
                end: 3,
            },
            TensorSelection::Indices {
                axis: 0,
                indices: vec![],
            },
            TensorSelection::Indices {
                axis: 1,
                indices: vec![2],
            },
        ] {
            assert!(matches!(
                store.acquire("matrix", selection),
                Err(WeightStoreError::InvalidSelection { .. })
            ));
        }
        let lease = store
            .acquire(
                "matrix",
                TensorSelection::Indices {
                    axis: 1,
                    indices: vec![1, 0, 1],
                },
            )
            .unwrap();
        assert_eq!(lease.output_shape(), [2, 3]);
        assert_eq!(lease.selected_byte_len(), 24);
        assert_eq!(
            store
                .acquire("matrix", TensorSelection::Full)
                .unwrap()
                .selected_byte_len(),
            16
        );
        assert!(matches!(
            validate_selection("overflow", &[usize::MAX, 2], &TensorSelection::Full),
            Err(WeightStoreError::Overflow { .. })
        ));
    }

    #[test]
    fn preserves_storage_encodings_and_supports_encoded_fp8() {
        let dir = tempfile::tempdir().unwrap();
        let f16_bytes = [0x00u8, 0x3c, 0x00, 0x40];
        let bf16_bytes = [0x80u8, 0x3f, 0x00, 0x40];
        let fp8_bytes = [0x38u8, 0x40];
        let f16 = TensorView::new(Dtype::F16, vec![2], &f16_bytes).unwrap();
        let bf16 = TensorView::new(Dtype::BF16, vec![2], &bf16_bytes).unwrap();
        let fp8 = TensorView::new(Dtype::F8_E4M3, vec![2], &fp8_bytes).unwrap();
        serialize_to_file(
            [("f16", f16), ("bf16", bf16), ("fp8", fp8)],
            None,
            &dir.path().join("model.safetensors"),
        )
        .unwrap();
        let store = SafetensorsWeightStore::open(dir.path()).unwrap();
        assert_eq!(
            store.metadata("f16").unwrap().stored_dtype,
            StoredDtype::F16
        );
        assert_eq!(
            store.metadata("bf16").unwrap().stored_dtype,
            StoredDtype::BF16
        );
        assert_eq!(
            store.metadata("fp8").unwrap().stored_dtype,
            StoredDtype::F8E4M3
        );
        let stream = cpu_stream();
        let f16 = store
            .acquire("f16", TensorSelection::Full)
            .unwrap()
            .materialize(&stream, &stream)
            .unwrap();
        let bf16 = store
            .acquire("bf16", TensorSelection::Full)
            .unwrap()
            .materialize(&stream, &stream)
            .unwrap();
        let fp8 = store
            .acquire("fp8", TensorSelection::Full)
            .unwrap()
            .materialize(&stream, &stream)
            .unwrap();
        assert_eq!(f16.dtype(), MlxDtype::Float16);
        assert_eq!(bf16.dtype(), MlxDtype::Bfloat16);
        assert_eq!(fp8.dtype(), MlxDtype::Uint8);
        assert_eq!(fp8.evaluated().unwrap().as_slice::<u8>(), &fp8_bytes);
    }

    #[test]
    fn rejects_unsupported_stored_dtype_during_materialization() {
        let dir = tempfile::tempdir().unwrap();
        let encoded = [0x3cu8, 0x40];
        let view = TensorView::new(Dtype::F8_E5M2, vec![2], &encoded).unwrap();
        serialize_to_file(
            [("unsupported", view)],
            None,
            &dir.path().join("model.safetensors"),
        )
        .unwrap();
        let store = SafetensorsWeightStore::open(dir.path()).unwrap();
        assert_eq!(
            store.metadata("unsupported").unwrap().stored_dtype,
            StoredDtype::F8E5M2
        );
        let stream = cpu_stream();
        assert!(matches!(
            store
                .acquire("unsupported", TensorSelection::Full)
                .unwrap()
                .materialize(&stream, &stream),
            Err(WeightStoreError::UnsupportedStoredDtype { .. })
        ));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_indexed_symlinks_that_escape_the_model_directory() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let outside_file = outside.path().join("outside.safetensors");
        write_i32(&outside_file, "weight", &[1], vec![1]);
        std::os::unix::fs::symlink(&outside_file, dir.path().join("linked.safetensors")).unwrap();
        write_index(dir.path(), &[("weight", "linked.safetensors")]);
        let store = SafetensorsWeightStore::open(dir.path()).unwrap();
        assert!(matches!(
            store.acquire("weight", TensorSelection::Full),
            Err(WeightStoreError::UnsafeShardPath { .. })
        ));
    }

    #[cfg(unix)]
    #[test]
    fn accepts_hugging_face_snapshot_symlinks_into_repository_blobs() {
        let cache = tempfile::tempdir().unwrap();
        let repository = cache.path().join("models--owner--model");
        let snapshot = repository.join("snapshots/revision");
        let blobs = repository.join("blobs");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::create_dir_all(&blobs).unwrap();
        write_i32(&blobs.join("payload"), "weight", &[7], vec![1]);
        std::os::unix::fs::symlink(
            "../../blobs/payload",
            snapshot.join("model-00001-of-00001.safetensors"),
        )
        .unwrap();
        write_index(&snapshot, &[("weight", "model-00001-of-00001.safetensors")]);

        let store = SafetensorsWeightStore::open(&snapshot).unwrap();
        let stream = cpu_stream();
        let materialized = store
            .acquire("weight", TensorSelection::Full)
            .unwrap()
            .materialize(&stream, &stream)
            .unwrap();
        let value = materialized.evaluated().unwrap();
        assert_eq!(value.as_slice::<i32>(), &[7]);
    }

    #[test]
    fn mappings_release_after_store_and_lease_drop() {
        let dir = tempfile::tempdir().unwrap();
        write_i32(
            &dir.path().join("model.safetensors"),
            "weight",
            &[1],
            vec![1],
        );
        let store = SafetensorsWeightStore::open(dir.path()).unwrap();
        let lease = store.acquire("weight", TensorSelection::Full).unwrap();
        let mapping = Arc::downgrade(safetensors_shard(&lease));
        drop(store);
        assert!(mapping.upgrade().is_some());
        drop(lease);
        assert!(mapping.upgrade().is_none());
    }

    #[test]
    fn returned_array_survives_lease_and_store_drop() {
        let dir = tempfile::tempdir().unwrap();
        write_i32(
            &dir.path().join("model.safetensors"),
            "weight",
            &[7, 8, 9],
            vec![3],
        );
        let stream = cpu_stream();
        let value = {
            let store = SafetensorsWeightStore::open(dir.path()).unwrap();
            let lease = store.acquire("weight", TensorSelection::Full).unwrap();
            lease.materialize(&stream, &stream).unwrap()
        };
        assert_eq!(value.evaluated().unwrap().as_slice::<i32>(), &[7, 8, 9]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn materializes_from_cpu_to_metal_execution_stream() {
        let dir = tempfile::tempdir().unwrap();
        write_i32(
            &dir.path().join("model.safetensors"),
            "weight",
            &[10, 20, 30],
            vec![3],
        );
        let store = SafetensorsWeightStore::open(dir.path()).unwrap();
        let source = cpu_stream();
        let execution = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let value = store
            .acquire("weight", TensorSelection::Full)
            .unwrap()
            .materialize(&source, &execution)
            .unwrap();
        assert_eq!(value.evaluated().unwrap().as_slice::<i32>(), &[10, 20, 30]);
    }
}

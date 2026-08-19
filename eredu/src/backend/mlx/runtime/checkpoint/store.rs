//! Persistent, lazy checkpoint tensor storage.
//!
//! A [`crate::backend::mlx::runtime::checkpoint::store::WeightLease`] pins the bytes backing a safetensors
//! view. Materialization returns an owning completion guard that retains the
//! view through asynchronous MLX evaluation. Callers may order a compatible
//! consumer stream without blocking the host, or synchronize the exact
//! materialization before taking its independently owned output.

pub use eredu_checkpoint::store::{
    ReadPolicy as WeightReadPolicy, SafetensorsWeightStore, TensorSelection, WeightStoreBackend,
    WeightStoreDiagnostics,
};
use eredu_checkpoint::{
    gguf_store::{
        GgufLease as NeutralGgufLease, GgufWeightStore as NeutralGgufWeightStore,
        GgufWeightStoreBuilder as NeutralGgufWeightStoreBuilder,
    },
    store::{
        CheckpointLease, CheckpointSource, EncodedTensorLease,
        SafetensorsLease as NeutralSafetensorsLease, TensorReadRequest,
    },
    StoredDtype,
};

use std::{
    any::Any,
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, Weak},
};

use safemlx::{
    ops::{indexing::TryIndexOp, GgufCheckpoint, GgufTensor},
    transforms::async_eval_with_event,
    Array, Event, Stream,
};
use safetensors::tensor::{Dtype, TensorView};

#[cfg(test)]
use eredu_checkpoint::gguf_store::GgufPhysicalSelection;
#[cfg(test)]
use safemlx::ops::GgufTensorSelection;

fn safetensors_dtype(key: &str, value: &StoredDtype) -> Result<Dtype, WeightStoreError> {
    match value {
        StoredDtype::Bool => Ok(Dtype::BOOL),
        StoredDtype::U8 => Ok(Dtype::U8),
        StoredDtype::I8 => Ok(Dtype::I8),
        StoredDtype::I16 => Ok(Dtype::I16),
        StoredDtype::U16 => Ok(Dtype::U16),
        StoredDtype::F16 => Ok(Dtype::F16),
        StoredDtype::BF16 => Ok(Dtype::BF16),
        StoredDtype::I32 => Ok(Dtype::I32),
        StoredDtype::U32 => Ok(Dtype::U32),
        StoredDtype::F32 => Ok(Dtype::F32),
        StoredDtype::F64 => Ok(Dtype::F64),
        StoredDtype::I64 => Ok(Dtype::I64),
        StoredDtype::U64 => Ok(Dtype::U64),
        StoredDtype::C64 => Err(WeightStoreError::UnsupportedStoredDtype {
            key: key.into(),
            dtype: value.clone(),
        }),
        StoredDtype::F8E4M3 => Ok(Dtype::F8_E4M3),
        StoredDtype::F4 => Ok(Dtype::F4),
        StoredDtype::F8E8M0 => Ok(Dtype::F8_E8M0),
        StoredDtype::F8E5M2 => Err(WeightStoreError::UnsupportedStoredDtype {
            key: key.into(),
            dtype: value.clone(),
        }),
        StoredDtype::Other(_) => Err(WeightStoreError::UnsupportedStoredDtype {
            key: key.into(),
            dtype: value.clone(),
        }),
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
    /// A tensor exists physically but was not selected by the resolved
    /// architecture checkpoint contract.
    #[error("checkpoint contract {contract:?} does not authorize tensor {key:?}")]
    UnauthorizedTensor {
        /// Resolved architecture checkpoint identity.
        contract: String,
        /// Rejected physical source key.
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
pub trait WeightStore {
    /// Returns the concrete backend identity without consulting mutable diagnostics.
    fn backend(&self) -> WeightStoreBackend;

    /// Supports optional backend-specific inspection without making callers concrete.
    fn as_any(&self) -> Option<&dyn Any>;

    /// Returns all catalog keys in deterministic order.
    fn keys(&self) -> Vec<String>;

    /// Returns source keys consumed to synthesize overlay bindings.
    fn materialized_source_keys(&self) -> Vec<String> {
        Vec::new()
    }

    /// Returns catalog keys admitted by a non-strict architecture policy but
    /// not claimed by its selected physical layout.
    fn unclaimed_checkpoint_keys(&self) -> Vec<String> {
        Vec::new()
    }

    /// Returns whether `key` is an authoritative output synthesized by this
    /// store and must supersede any source-side semantic recipe for the same
    /// logical parameter.
    fn is_authoritative_materialized_key(&self, _key: &str) -> bool {
        false
    }

    /// Whether this store is already restricted by a resolved architecture
    /// checkpoint contract.
    fn is_checkpoint_contract_resolved(&self) -> bool {
        false
    }

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

fn recipe_tensor_metadata(
    store: &dyn WeightStore,
    key: &str,
) -> Result<eredu_checkpoint::store::TensorMetadata, eredu_checkpoint::store::StoreError> {
    let metadata = store.metadata(key).map_err(|error| match error {
        WeightStoreError::UnknownTensor { key } => {
            eredu_checkpoint::store::StoreError::UnknownTensor { key }
        }
        other => eredu_checkpoint::store::StoreError::Internal(other.to_string()),
    })?;
    Ok(eredu_checkpoint::store::TensorMetadata {
        name: metadata.name,
        logical_shape: metadata.shape.clone(),
        physical_shape: metadata.shape,
        stored_dtype: metadata.stored_dtype,
        encoded_byte_len: u64::try_from(metadata.logical_byte_len).map_err(|_| {
            eredu_checkpoint::store::StoreError::Overflow {
                context: format!("recipe metadata byte length for {key:?}"),
            }
        })?,
        backing_shard: metadata.backing_shard,
    })
}

impl eredu_checkpoint::recipe::RecipeCatalog for dyn WeightStore + '_ {
    fn tensor_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, eredu_checkpoint::store::StoreError> {
        recipe_tensor_metadata(self, key)
    }
}

impl eredu_checkpoint::recipe::RecipeCatalog for dyn WeightStore + Send + Sync + '_ {
    fn tensor_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, eredu_checkpoint::store::StoreError> {
        recipe_tensor_metadata(self, key)
    }
}

/// A checkpoint store restricted to the exact physical layout selected by an
/// architecture contract.
///
/// This is the only store handed to binding recipes in production loaders.
/// Consequently validation and materialization cannot choose layouts
/// independently.
pub(crate) struct ContractWeightStore {
    source: Arc<dyn WeightStore + Send + Sync>,
    contract: eredu_checkpoint::validation::ResolvedCheckpointPlan,
}

impl ContractWeightStore {
    pub(crate) fn new(
        source: Arc<dyn WeightStore + Send + Sync>,
        contract: eredu_checkpoint::validation::ResolvedCheckpointPlan,
    ) -> Self {
        Self { source, contract }
    }

    fn authorize(&self, key: &str) -> Result<(), WeightStoreError> {
        if self.contract.source_keys().contains(key)
            || self.source.is_authoritative_materialized_key(key)
        {
            Ok(())
        } else {
            Err(WeightStoreError::UnauthorizedTensor {
                contract: self.contract.identity().into(),
                key: key.into(),
            })
        }
    }
}

impl WeightStore for ContractWeightStore {
    fn backend(&self) -> WeightStoreBackend {
        self.source.backend()
    }

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }

    fn keys(&self) -> Vec<String> {
        self.source
            .keys()
            .into_iter()
            .filter(|key| {
                self.contract.source_keys().contains(key)
                    || self.source.is_authoritative_materialized_key(key)
            })
            .collect()
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

    fn metadata(&self, key: &str) -> Result<WeightMetadata, WeightStoreError> {
        self.authorize(key)?;
        self.source.metadata(key)
    }

    fn acquire_with_policy(
        &self,
        key: &str,
        selection: TensorSelection,
        policy: WeightReadPolicy,
    ) -> Result<WeightLease, WeightStoreError> {
        self.authorize(key)?;
        self.source.acquire_with_policy(key, selection, policy)
    }

    fn diagnostics(&self) -> Result<WeightStoreDiagnostics, WeightStoreError> {
        self.source.diagnostics()
    }
}

/// Immutable array-backed store used to hand transformed weights to the
/// generalized execution engine without serializing an intermediate checkpoint.
#[derive(Clone)]
#[cfg(test)]
pub(crate) struct MemoryWeightStore {
    arrays: Arc<BTreeMap<String, Array>>,
    metadata: Arc<BTreeMap<String, WeightMetadata>>,
    backing: Arc<PathBuf>,
}

#[cfg(test)]
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

#[cfg(test)]
impl WeightStore for MemoryWeightStore {
    fn backend(&self) -> WeightStoreBackend {
        WeightStoreBackend::Memory
    }

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
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

#[cfg(test)]
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

#[derive(Debug)]
struct CachedGgufGroup {
    arrays: Vec<(String, Array)>,
}

/// MLX stream and lease-coalescing state used for neutral parameter realization.
#[derive(Debug, Clone)]
pub struct MlxParameterMaterializationContext {
    source_stream: Stream,
    execution_stream: Stream,
    converted_groups: Arc<
        Mutex<BTreeMap<eredu_checkpoint::gguf_store::GgufLeaseIdentity, Weak<CachedGgufGroup>>>,
    >,
}

impl MlxParameterMaterializationContext {
    /// Creates a reusable materialization context for one source/execution stream pair.
    pub fn new(source_stream: &Stream, execution_stream: &Stream) -> Self {
        Self {
            source_stream: source_stream.clone(),
            execution_stream: execution_stream.clone(),
            converted_groups: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    /// Host/source stream used to create and transform checkpoint arrays.
    pub const fn source_stream(&self) -> &Stream {
        &self.source_stream
    }

    /// Destination stream used for the final execution weight.
    pub const fn execution_stream(&self) -> &Stream {
        &self.execution_stream
    }

    pub(crate) fn weight_lease(
        &self,
        lease: CheckpointLease,
    ) -> Result<WeightLease, WeightStoreError> {
        WeightLease::from_checkpoint_lease(lease, Arc::clone(&self.converted_groups))
    }
}

pub(crate) struct NeutralCheckpointSourceAdapter<'a> {
    source: &'a dyn CheckpointSource,
    converted_groups: Arc<
        Mutex<BTreeMap<eredu_checkpoint::gguf_store::GgufLeaseIdentity, Weak<CachedGgufGroup>>>,
    >,
}

impl<'a> NeutralCheckpointSourceAdapter<'a> {
    pub(crate) fn new(
        source: &'a dyn CheckpointSource,
        context: &MlxParameterMaterializationContext,
    ) -> Self {
        Self {
            source,
            converted_groups: Arc::clone(&context.converted_groups),
        }
    }
}

impl WeightStore for NeutralCheckpointSourceAdapter<'_> {
    fn backend(&self) -> WeightStoreBackend {
        self.source
            .source_diagnostics()
            .map(|diagnostics| diagnostics.backend)
            .unwrap_or(WeightStoreBackend::Memory)
    }

    fn as_any(&self) -> Option<&dyn Any> {
        None
    }

    fn keys(&self) -> Vec<String> {
        self.source.source_keys()
    }

    fn metadata(&self, key: &str) -> Result<WeightMetadata, WeightStoreError> {
        self.source
            .source_metadata(key)
            .map(neutral_metadata)
            .map_err(neutral_store_error)
    }

    fn acquire_with_policy(
        &self,
        key: &str,
        selection: TensorSelection,
        policy: WeightReadPolicy,
    ) -> Result<WeightLease, WeightStoreError> {
        let lease = self
            .source
            .acquire_lease(TensorReadRequest {
                key: key.into(),
                selection,
                policy,
            })
            .map_err(neutral_store_error)?;
        WeightLease::from_checkpoint_lease(lease, Arc::clone(&self.converted_groups))
    }

    fn diagnostics(&self) -> Result<WeightStoreDiagnostics, WeightStoreError> {
        self.source
            .source_diagnostics()
            .map_err(neutral_store_error)
    }
}

/// Builder adapter preserving the MLX loader API while delegating GGUF
/// inspection and acquisition to the backend-neutral checkpoint crate.
#[derive(Debug, Default)]
pub(crate) struct GgufWeightStoreBuilder {
    inner: NeutralGgufWeightStoreBuilder,
}

impl GgufWeightStoreBuilder {
    pub(crate) fn max_cached_readers(mut self, maximum: usize) -> Result<Self, WeightStoreError> {
        self.inner = self
            .inner
            .max_cached_readers(maximum)
            .map_err(neutral_store_error)?;
        Ok(self)
    }

    pub(crate) fn add_checkpoint<F>(
        mut self,
        checkpoint: GgufCheckpoint,
        plan: &eredu_checkpoint::schema::GgufCheckpointPlan,
        translate: F,
    ) -> Result<Self, WeightStoreError>
    where
        F: FnMut(&str) -> String,
    {
        self.inner = self
            .inner
            .add_checkpoint(checkpoint.catalog().clone(), plan, translate)
            .map_err(neutral_store_error)?;
        Ok(self)
    }

    #[cfg(test)]
    fn add_checkpoint_for_test<F>(
        mut self,
        checkpoint: GgufCheckpoint,
        translate: F,
    ) -> Result<Self, WeightStoreError>
    where
        F: FnMut(&str) -> String,
    {
        let resolved = eredu_checkpoint::validation::ResolvedCheckpointPlan::for_test(
            "test GGUF catalog",
            checkpoint
                .catalog()
                .tensors()
                .map(|tensor| tensor.descriptor().name.clone()),
        );
        self.inner = self
            .inner
            .add_resolved_checkpoint(checkpoint.catalog().clone(), &resolved, translate)
            .map_err(neutral_store_error)?;
        Ok(self)
    }

    pub(crate) fn build(self) -> Result<GgufWeightStore, WeightStoreError> {
        Ok(GgufWeightStore {
            inner: self.inner.build().map_err(neutral_store_error)?,
            converted_groups: Arc::new(Mutex::new(BTreeMap::new())),
        })
    }
}

/// MLX materialization adapter over the backend-neutral GGUF store.
#[derive(Debug, Clone)]
pub(crate) struct GgufWeightStore {
    inner: NeutralGgufWeightStore,
    converted_groups: Arc<
        Mutex<BTreeMap<eredu_checkpoint::gguf_store::GgufLeaseIdentity, Weak<CachedGgufGroup>>>,
    >,
}

impl GgufWeightStore {
    pub(crate) fn builder() -> GgufWeightStoreBuilder {
        GgufWeightStoreBuilder::default()
    }

    pub(crate) fn new_with_max_mapped_shards<F>(
        checkpoint: GgufCheckpoint,
        plan: &eredu_checkpoint::schema::GgufCheckpointPlan,
        translate: F,
        max_mapped_shards: usize,
    ) -> Result<Self, WeightStoreError>
    where
        F: FnMut(&str) -> String,
    {
        Self::builder()
            .max_cached_readers(max_mapped_shards)?
            .add_checkpoint(checkpoint, plan, translate)?
            .build()
    }

    #[cfg(test)]
    pub(crate) fn new_for_test<F>(
        checkpoint: GgufCheckpoint,
        translate: F,
    ) -> Result<Self, WeightStoreError>
    where
        F: FnMut(&str) -> String,
    {
        Self::builder()
            .add_checkpoint_for_test(checkpoint, translate)?
            .build()
    }
}

impl WeightStore for GgufWeightStore {
    fn backend(&self) -> WeightStoreBackend {
        WeightStoreBackend::Gguf
    }

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }

    fn is_checkpoint_contract_resolved(&self) -> bool {
        true
    }

    fn keys(&self) -> Vec<String> {
        eredu_checkpoint::store::WeightStore::keys(&self.inner)
    }

    fn unclaimed_checkpoint_keys(&self) -> Vec<String> {
        self.inner.unclaimed_checkpoint_keys()
    }

    fn metadata(&self, key: &str) -> Result<WeightMetadata, WeightStoreError> {
        eredu_checkpoint::store::WeightStore::metadata(&self.inner, key)
            .map(neutral_metadata)
            .map_err(neutral_store_error)
    }

    fn acquire_with_policy(
        &self,
        key: &str,
        selection: TensorSelection,
        policy: WeightReadPolicy,
    ) -> Result<WeightLease, WeightStoreError> {
        let lease = eredu_checkpoint::store::WeightStore::acquire(
            &self.inner,
            TensorReadRequest {
                key: key.into(),
                selection: selection.clone(),
                policy,
            },
        )
        .map_err(neutral_store_error)?;
        let metadata = neutral_metadata(lease.metadata().clone());
        let selected_byte_len =
            selected_byte_len(key, &metadata, &selection, lease.output_shape())?;
        Ok(WeightLease {
            key: key.into(),
            metadata,
            selection,
            output_shape: lease.output_shape().to_vec(),
            selected_byte_len,
            source: WeightLeaseSource::Gguf(Box::new(GgufLeaseSource {
                lease,
                converted_groups: Arc::clone(&self.converted_groups),
            })),
        })
    }

    fn diagnostics(&self) -> Result<WeightStoreDiagnostics, WeightStoreError> {
        eredu_checkpoint::store::WeightStore::diagnostics(&self.inner).map_err(neutral_store_error)
    }
}

impl WeightStore for SafetensorsWeightStore {
    fn backend(&self) -> WeightStoreBackend {
        WeightStoreBackend::Safetensors
    }

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }

    fn keys(&self) -> Vec<String> {
        eredu_checkpoint::store::WeightStore::keys(self)
    }

    fn metadata(&self, key: &str) -> Result<WeightMetadata, WeightStoreError> {
        eredu_checkpoint::store::WeightStore::metadata(self, key)
            .map(neutral_metadata)
            .map_err(neutral_store_error)
    }

    fn acquire_with_policy(
        &self,
        key: &str,
        selection: TensorSelection,
        policy: WeightReadPolicy,
    ) -> Result<WeightLease, WeightStoreError> {
        let lease = eredu_checkpoint::store::WeightStore::acquire(
            self,
            TensorReadRequest {
                key: key.into(),
                selection: selection.clone(),
                policy,
            },
        )
        .map_err(neutral_store_error)?;
        let metadata = neutral_metadata(lease.metadata().clone());
        let selected_byte_len =
            usize::try_from(lease.bounded_read_proof().length_bytes).map_err(|_| {
                WeightStoreError::Overflow {
                    context: format!("selected byte length for tensor {key:?}"),
                }
            })?;
        Ok(WeightLease {
            key: key.into(),
            metadata,
            selection,
            output_shape: lease.output_shape().to_vec(),
            selected_byte_len,
            source: WeightLeaseSource::Safetensors(lease),
        })
    }

    fn diagnostics(&self) -> Result<WeightStoreDiagnostics, WeightStoreError> {
        eredu_checkpoint::store::WeightStore::diagnostics(self).map_err(neutral_store_error)
    }
}

#[cfg(test)]
fn validate_selection(
    key: &str,
    shape: &[usize],
    selection: &TensorSelection,
) -> Result<Vec<usize>, WeightStoreError> {
    let element_count = |dimensions: &[usize], context: &str| {
        dimensions.iter().try_fold(1usize, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or_else(|| WeightStoreError::Overflow {
                    context: format!("{context} for tensor {key:?}"),
                })
        })
    };
    let full_elements = element_count(shape, "element count")?;
    let mut output = shape.to_vec();
    match selection {
        TensorSelection::Full => {}
        TensorSelection::Range { axis, start, end } => {
            let dimension = shape
                .get(*axis)
                .ok_or_else(|| WeightStoreError::InvalidSelection {
                    key: key.into(),
                    message: format!("axis {axis} is outside rank {}", shape.len()),
                })?;
            if start >= end || *end > *dimension {
                return Err(WeightStoreError::InvalidSelection {
                    key: key.into(),
                    message: format!(
                        "range {start}..{end} is invalid for axis {axis} dimension {dimension}"
                    ),
                });
            }
            output[*axis] = end - start;
        }
        TensorSelection::Indices { axis, indices } => {
            let dimension = shape
                .get(*axis)
                .ok_or_else(|| WeightStoreError::InvalidSelection {
                    key: key.into(),
                    message: format!("axis {axis} is outside rank {}", shape.len()),
                })?;
            if indices.is_empty() || indices.iter().any(|index| *index >= *dimension) {
                return Err(WeightStoreError::InvalidSelection {
                    key: key.into(),
                    message: "indices are empty or outside the selected dimension".into(),
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
                    key: key.into(),
                    message: "contiguous selection shape must be non-empty and nonzero".into(),
                });
            }
            let selected_elements = element_count(selected, "contiguous selection size")?;
            let end = offset_elements
                .checked_add(selected_elements)
                .ok_or_else(|| WeightStoreError::Overflow {
                    context: format!("contiguous selection end for tensor {key:?}"),
                })?;
            if end > full_elements {
                return Err(WeightStoreError::InvalidSelection {
                    key: key.into(),
                    message: "contiguous selection exceeds the tensor".into(),
                });
            }
            output = selected.clone();
        }
    }
    element_count(&output, "selected element count")?;
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
    let count = |shape: &[usize], context: &str| {
        shape.iter().try_fold(1usize, |value, dimension| {
            value
                .checked_mul(*dimension)
                .ok_or_else(|| WeightStoreError::Overflow {
                    context: format!("{context} for tensor {key:?}"),
                })
        })
    };
    let full_elements = count(&metadata.shape, "element count")?;
    let selected_elements = count(output_shape, "selected element count")?;
    let scaled = metadata
        .logical_byte_len
        .checked_mul(selected_elements)
        .ok_or_else(|| WeightStoreError::Overflow {
            context: format!("selected byte length for tensor {key:?}"),
        })?;
    if full_elements == 0 || !scaled.is_multiple_of(full_elements) {
        return Err(WeightStoreError::InvalidSelection {
            key: key.into(),
            message: "selection does not have a whole-byte encoded length".into(),
        });
    }
    Ok(scaled / full_elements)
}

fn neutral_metadata(metadata: eredu_checkpoint::store::TensorMetadata) -> WeightMetadata {
    WeightMetadata {
        name: metadata.name,
        shape: metadata.logical_shape,
        stored_dtype: metadata.stored_dtype,
        logical_byte_len: usize::try_from(metadata.encoded_byte_len).unwrap_or(usize::MAX),
        backing_shard: metadata.backing_shard,
    }
}

pub(crate) fn neutral_store_error(error: eredu_checkpoint::store::StoreError) -> WeightStoreError {
    use eredu_checkpoint::store::StoreError;
    match error {
        StoreError::InvalidMappedShardLimit => WeightStoreError::InvalidMappedShardLimit,
        StoreError::UnknownTensor { key } => WeightStoreError::UnknownTensor { key },
        StoreError::UnauthorizedTensor { contract, key } => {
            WeightStoreError::UnauthorizedTensor { contract, key }
        }
        StoreError::MissingShard { path } => WeightStoreError::MissingShard { path },
        StoreError::MalformedIndex { path, message } => {
            WeightStoreError::MalformedIndex { path, message }
        }
        StoreError::MalformedSafetensors { path, message } => {
            WeightStoreError::MalformedSafetensors { path, message }
        }
        StoreError::UnsafeShardPath { path } => WeightStoreError::UnsafeShardPath { path },
        StoreError::ContradictoryIndexMapping { key, path } => {
            WeightStoreError::ContradictoryIndexMapping { key, path }
        }
        StoreError::InvalidSelection { key, message } => {
            WeightStoreError::InvalidSelection { key, message }
        }
        StoreError::BoundedSelectionUnavailable { key, message } => {
            WeightStoreError::BoundedSelectionUnavailable { key, message }
        }
        StoreError::Overflow { context } => WeightStoreError::Overflow { context },
        StoreError::CapacityExhausted { maximum, leased } => WeightStoreError::CapacityExhausted {
            max_mapped_shards: maximum,
            leased_shards: leased,
        },
        StoreError::Io { path, message } => {
            WeightStoreError::MalformedSafetensors { path, message }
        }
        StoreError::Internal(_) => WeightStoreError::CachePoisoned,
        StoreError::Gguf { key, message } => WeightStoreError::Gguf { key, message },
    }
}
#[derive(Debug, Clone)]
enum WeightLeaseSource {
    Safetensors(NeutralSafetensorsLease),
    Gguf(Box<GgufLeaseSource>),
    #[cfg(test)]
    Memory(MemoryLeaseSource),
}

#[derive(Debug, Clone)]
#[cfg(test)]
struct MemoryLeaseSource {
    value: Array,
    backing: Arc<PathBuf>,
}

#[derive(Debug, Clone)]
struct GgufLeaseSource {
    lease: NeutralGgufLease,
    converted_groups: Arc<
        Mutex<BTreeMap<eredu_checkpoint::gguf_store::GgufLeaseIdentity, Weak<CachedGgufGroup>>>,
    >,
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
    fn from_checkpoint_lease(
        lease: CheckpointLease,
        converted_groups: Arc<
            Mutex<BTreeMap<eredu_checkpoint::gguf_store::GgufLeaseIdentity, Weak<CachedGgufGroup>>>,
        >,
    ) -> Result<Self, WeightStoreError> {
        let key = lease.metadata().name.clone();
        let metadata = neutral_metadata(lease.metadata().clone());
        let selection = lease.selection().clone();
        let output_shape = lease.output_shape().to_vec();
        let selected_byte_len = match &lease {
            CheckpointLease::Safetensors(lease) => {
                usize::try_from(lease.bounded_read_proof().length_bytes).map_err(|_| {
                    WeightStoreError::Overflow {
                        context: format!("selected byte length for tensor {key:?}"),
                    }
                })?
            }
            CheckpointLease::Gguf(_) => {
                selected_byte_len(&key, &metadata, &selection, &output_shape)?
            }
        };
        let source = match lease {
            CheckpointLease::Safetensors(lease) => WeightLeaseSource::Safetensors(lease),
            CheckpointLease::Gguf(lease) => WeightLeaseSource::Gguf(Box::new(GgufLeaseSource {
                lease,
                converted_groups,
            })),
        };
        Ok(Self {
            key,
            metadata,
            selection,
            output_shape,
            selected_byte_len,
            source,
        })
    }

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
            WeightLeaseSource::Safetensors(lease) => lease
                .backing_path()
                .expect("SafeTensors leases are file-backed"),
            WeightLeaseSource::Gguf(source) => source
                .lease
                .backing_path()
                .expect("GGUF catalog entries always identify their shard"),
            #[cfg(test)]
            WeightLeaseSource::Memory(source) => source.backing.as_path(),
        }
    }

    /// Submits the selected tensor for materialization onto `execution_stream`.
    ///
    /// The returned guard owns this lease, every mmap-derived source, and the
    /// exact completion event. Call [`WeightMaterialization::wait_on`] before
    /// evaluating a dependent graph on another compatible stream, or call
    /// [`WeightMaterialization::synchronize`] to block for and take the output.
    /// MLX graph construction alone does not consume the materialization.
    pub fn materialize(
        &self,
        source_stream: &Stream,
        execution_stream: &Stream,
    ) -> Result<WeightMaterialization, WeightStoreError> {
        self.clone()
            .prepare_materialization(source_stream, execution_stream)?
            .submit()
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
            #[cfg(test)]
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
            #[cfg(test)]
            WeightLeaseSource::Memory(source) => {
                self.prepare_memory(source, source_stream, source_stream)
            }
        }
    }

    #[cfg(test)]
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
        source: NeutralSafetensorsLease,
        source_stream: &Stream,
    ) -> Result<PendingWeightMaterialization, WeightStoreError> {
        let dtype = safetensors_dtype(&self.key, &self.metadata.stored_dtype)?;
        let data =
            source
                .encoded_bytes()
                .ok_or_else(|| WeightStoreError::MalformedSafetensors {
                    path: source
                        .backing_path()
                        .unwrap_or_else(|| Path::new("<unknown>"))
                        .to_path_buf(),
                    message: format!("tensor {:?} lease has no encoded bytes", self.key),
                })?;
        let view = TensorView::new(dtype, self.output_shape.clone(), data).map_err(|error| {
            WeightStoreError::MalformedSafetensors {
                path: source
                    .backing_path()
                    .unwrap_or_else(|| Path::new("<unknown>"))
                    .to_path_buf(),
                message: format!("tensor {:?}: {error}", self.key),
            }
        })?;
        let aligned = (data.as_ptr() as usize).is_multiple_of(safetensors_dtype_alignment(dtype));
        let source_value = if aligned {
            unsafe { Array::try_from_borrowed_safetensors(view) }
        } else {
            Array::try_from(view)
        }
        .map_err(|conversion| WeightStoreError::MlxConversion {
            key: self.key.clone(),
            source: conversion,
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
        source: NeutralSafetensorsLease,
        source_stream: &Stream,
        execution_stream: &Stream,
    ) -> Result<PendingWeightMaterialization, WeightStoreError> {
        let dtype = safetensors_dtype(&self.key, &self.metadata.stored_dtype)?;
        let data =
            source
                .encoded_bytes()
                .ok_or_else(|| WeightStoreError::MalformedSafetensors {
                    path: source
                        .backing_path()
                        .unwrap_or_else(|| Path::new("<unknown>"))
                        .to_path_buf(),
                    message: format!("tensor {:?} lease has no encoded bytes", self.key),
                })?;
        let view = TensorView::new(dtype, self.output_shape.clone(), data).map_err(|error| {
            WeightStoreError::MalformedSafetensors {
                path: source
                    .backing_path()
                    .unwrap_or_else(|| Path::new("<unknown>"))
                    .to_path_buf(),
                message: format!("tensor {:?}: {error}", self.key),
            }
        })?;
        let source_value =
            Array::try_from(view).map_err(|conversion| WeightStoreError::MlxConversion {
                key: self.key.clone(),
                source: conversion,
            })?;
        let materialized = source_value
            .copy(execution_stream)
            .map_err(|error| self.mlx_error("copy", error))?;
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
            lease,
            converted_groups,
        } = source;
        let cache_key = lease.identity().clone();
        let selection_is_materialized = lease.selection_is_materialized();
        let logical_output_name = lease.logical_output_name().to_owned();
        let mut groups = converted_groups
            .lock()
            .map_err(|_| WeightStoreError::CachePoisoned)?;
        groups.retain(|_, group| group.strong_count() > 0);
        let group = if let Some(cached) = groups.get(&cache_key).and_then(Weak::upgrade) {
            lease.record_coalesced_group_hit();
            cached
        } else {
            let portable = lease.materialize_portable().map_err(neutral_store_error)?;
            let converted = GgufTensor::from_portable_host(portable).map_err(|error| {
                WeightStoreError::Gguf {
                    key: self.key.clone(),
                    message: error.to_string(),
                }
            })?;
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
            .find_map(|(name, value)| (name == &logical_output_name).then(|| value.clone()))
            .ok_or_else(|| WeightStoreError::Gguf {
                key: self.key.clone(),
                message: format!(
                    "portable GGUF group did not produce logical output {logical_output_name:?}"
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
            std::mem::forget(shard.clone());
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

    #[cfg(test)]
    fn finish(self) -> Result<Array, WeightStoreError> {
        self.submit()?.synchronize()
    }

    fn submit(mut self) -> Result<WeightMaterialization, WeightStoreError> {
        self.prepare_owned_output()?;
        let output = self.output.clone();
        WeightMaterialization::submit_retained(output, vec![self])
    }

    fn prepare_owned_output(&mut self) -> Result<(), WeightStoreError> {
        if !self.borrowed_source {
            return Ok(());
        }
        let output = self.output.copy(&self.source_stream).map_err(|source| {
            self.lease
                .as_ref()
                .expect("pending materialization retains its lease")
                .mlx_error("borrowed source copy", source)
        })?;
        self.output = output;
        self.borrowed_source = false;
        Ok(())
    }

    fn complete_in_place(&mut self) {
        self.completed = true;
        self.lease.take();
    }

    /// Marks a batch member complete after a containing output was evaluated.
    pub(crate) fn complete(mut self) {
        self.completed = true;
        self.lease.take();
    }
}

/// Owning completion for one checkpoint tensor materialization.
///
/// This single-shot guard retains checkpoint mappings and source arrays until
/// its exact MLX completion finishes. It may order multiple compatible
/// consumers. Dropping an unfinished guard blocks only for this event, never
/// for an entire stream. Asynchronous backend errors are returned by query or
/// synchronization. The type is intentionally neither `Send` nor `Sync`
/// because it owns `safemlx`'s thread-affine [`Event`].
#[must_use = "checkpoint mappings remain retained until this completion is consumed or dropped"]
pub struct WeightMaterialization {
    output: Array,
    sources: Vec<PendingWeightMaterialization>,
    event: Option<Event>,
}

impl WeightMaterialization {
    pub(crate) fn submit_retained(
        output: Array,
        sources: Vec<PendingWeightMaterialization>,
    ) -> Result<Self, WeightStoreError> {
        let key = sources
            .first()
            .and_then(|pending| pending.lease.as_ref())
            .map(|lease| lease.key.clone())
            .unwrap_or_else(|| "<derived checkpoint materialization>".into());
        let event = async_eval_with_event([&output]).map_err(|source| WeightStoreError::Mlx {
            key,
            operation: "evaluation submission",
            source,
        })?;
        Ok(Self {
            output,
            sources,
            event: Some(event),
        })
    }

    /// Returns the materialized output while this guard retains its sources.
    pub fn output(&self) -> &Array {
        &self.output
    }

    /// Orders subsequently submitted work on `stream` after this completion.
    ///
    /// This does not block the host. The stream must be backend/device
    /// compatible, and the consumer graph must be evaluated after this call.
    pub fn wait_on(&self, stream: &Stream) -> Result<(), WeightStoreError> {
        self.event
            .as_ref()
            .expect("unfinished materialization retains its event")
            .wait_on(stream)
            .map_err(|source| self.mlx_error("consumer stream wait", source))
    }

    /// Returns whether the exact materialization has completed without blocking.
    pub fn is_complete(&self) -> Result<bool, WeightStoreError> {
        self.event
            .as_ref()
            .expect("unfinished materialization retains its event")
            .is_complete()
            .map_err(|source| self.mlx_error("completion query", source))
    }

    /// Blocks for the exact completion and returns the independently owned output.
    pub fn synchronize(mut self) -> Result<Array, WeightStoreError> {
        let output = self.output().clone();
        self.finish(true)?;
        Ok(output)
    }

    fn finish(&mut self, report_error: bool) -> Result<(), WeightStoreError> {
        let Some(event) = self.event.take() else {
            return Ok(());
        };
        let key = self
            .sources
            .first()
            .and_then(|pending| pending.lease.as_ref())
            .map(|lease| lease.key.clone())
            .unwrap_or_else(|| "<completed checkpoint materialization>".into());
        let result = event.synchronize();
        for mut pending in self.sources.drain(..) {
            pending.complete_in_place();
        }
        match result {
            Ok(()) => Ok(()),
            Err(source) if report_error => Err(WeightStoreError::Mlx {
                key,
                operation: "completion",
                source,
            }),
            Err(_) => Ok(()),
        }
    }

    fn mlx_error(
        &self,
        operation: &'static str,
        source: safemlx::error::Exception,
    ) -> WeightStoreError {
        let key = self
            .sources
            .first()
            .and_then(|pending| pending.lease.as_ref())
            .map(|lease| lease.key.clone())
            .unwrap_or_else(|| "<completed checkpoint materialization>".into());
        WeightStoreError::Mlx {
            key,
            operation,
            source,
        }
    }
}

impl Drop for WeightMaterialization {
    fn drop(&mut self) {
        let _ = self.finish(false);
    }
}

impl Drop for PendingWeightMaterialization {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        // Submission creates the exact completion event and moves this value's
        // mapping lease into `WeightMaterialization`. If submission itself is
        // abandoned or fails before that event exists, draining both candidate
        // streams is the only conservative way to prove that no lazy copy still
        // references the mmap. This error-cleanup path is intentionally the sole
        // whole-stream wait in the eredu runtime.
        let source = self.source_stream.synchronize();
        let execution = self.execution_stream.synchronize();
        if source.is_err() || execution.is_err() {
            if let Some(lease) = &self.lease {
                lease.retain_mapping_after_sync_failure();
            }
        }
    }
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
    use eredu_gguf::{GgmlType, TensorInput, Writer};
    use safemlx::{
        transforms::{async_eval, async_eval_with_event},
        Device, DeviceType, Dtype as MlxDtype,
    };
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
                .synchronize()
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

    #[test]
    fn resolved_contract_store_rejects_unselected_physical_sources() {
        let selected = Array::from_slice(&[1f32], &[1]);
        let rejected = Array::from_slice(&[2f32], &[1]);
        let source: Arc<dyn WeightStore + Send + Sync> = Arc::new(
            MemoryWeightStore::new([
                ("packed".to_string(), selected),
                ("split".to_string(), rejected),
            ])
            .unwrap(),
        );
        let contract = eredu_checkpoint::validation::ResolvedCheckpointPlan::for_test(
            "test architecture",
            ["packed"],
        );
        let store = ContractWeightStore::new(source, contract);

        assert_eq!(store.keys(), ["packed"]);
        assert!(matches!(
            store.metadata("split"),
            Err(WeightStoreError::UnauthorizedTensor { key, .. }) if key == "split"
        ));
    }

    #[test]
    fn weight_materialization_event_orders_multiple_cpu_consumers() {
        let source_stream = cpu_stream();
        let execution_stream = cpu_stream();
        let consumer_a = cpu_stream();
        let consumer_b = cpu_stream();
        let source = Array::ones::<f32>(&[1024, 1024], &source_stream).unwrap();
        let store = MemoryWeightStore::new([("matrix".to_string(), source)]).unwrap();

        let blocker_lhs = Array::ones::<f32>(&[1024, 1024], &execution_stream).unwrap();
        let blocker_rhs = Array::ones::<f32>(&[1024, 1024], &execution_stream).unwrap();
        let blocker = blocker_lhs.matmul(&blocker_rhs, &execution_stream).unwrap();
        async_eval([&blocker]).unwrap();

        let materialization = store
            .acquire(
                "matrix",
                TensorSelection::Range {
                    axis: 0,
                    start: 0,
                    end: 1,
                },
            )
            .unwrap()
            .materialize(&source_stream, &execution_stream)
            .unwrap();
        materialization.wait_on(&consumer_a).unwrap();
        materialization.wait_on(&consumer_b).unwrap();
        let consumed_a = materialization.output().square(&consumer_a).unwrap();
        let consumed_b = materialization.output().square(&consumer_b).unwrap();
        let completion_a = async_eval_with_event([&consumed_a]).unwrap();
        let completion_b = async_eval_with_event([&consumed_b]).unwrap();

        let output = materialization.synchronize().unwrap();
        assert_eq!(output.shape(), [1, 1024]);
        completion_a.synchronize().unwrap();
        completion_b.synchronize().unwrap();
        assert_eq!(
            consumed_a.evaluated().unwrap().as_slice::<f32>(),
            &[1.0; 1024]
        );
        assert_eq!(
            consumed_b.evaluated().unwrap().as_slice::<f32>(),
            &[1.0; 1024]
        );
    }

    #[test]
    #[ignore = "explicit Metal checkpoint materialization test; run on a Metal host"]
    fn weight_materialization_metal_handoff_does_not_block_the_host() {
        let source_stream = cpu_stream();
        let transfer_stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let consumer_stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let source = Array::ones::<f32>(&[1024, 1024], &source_stream).unwrap();
        let store = MemoryWeightStore::new([("matrix".to_string(), source)]).unwrap();

        let blocker_lhs = Array::ones::<f32>(&[4096, 4096], &transfer_stream).unwrap();
        let blocker_rhs = Array::ones::<f32>(&[4096, 4096], &transfer_stream).unwrap();
        let blocker = blocker_lhs.matmul(&blocker_rhs, &transfer_stream).unwrap();
        async_eval([&blocker]).unwrap();
        let materialization = store
            .acquire(
                "matrix",
                TensorSelection::Range {
                    axis: 0,
                    start: 0,
                    end: 1,
                },
            )
            .unwrap()
            .materialize(&source_stream, &transfer_stream)
            .unwrap();
        assert!(!materialization.is_complete().unwrap());
        materialization.wait_on(&consumer_stream).unwrap();
        assert!(!materialization.is_complete().unwrap());
        let consumed = materialization.output().square(&consumer_stream).unwrap();
        let completion = async_eval_with_event([&consumed]).unwrap();
        drop(materialization);
        completion.synchronize().unwrap();
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

    fn write_dense_bank_gguf(path: &Path) -> Vec<f32> {
        let values = (0..24).map(|value| value as f32).collect::<Vec<_>>();
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        Writer::default()
            .write(
                std::fs::File::create(path).unwrap(),
                &BTreeMap::new(),
                &[TensorInput {
                    name: "bank.weight",
                    dimensions: &[4, 3, 2],
                    ggml_type: GgmlType::F32,
                    data: &bytes,
                }],
            )
            .unwrap();
        values
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

    fn write_two_dense_gguf(path: &Path) {
        let selected = 1.0f32.to_le_bytes();
        let unselected = 2.0f32.to_le_bytes();
        Writer::default()
            .write(
                std::fs::File::create(path).unwrap(),
                &BTreeMap::new(),
                &[
                    TensorInput {
                        name: "selected.weight",
                        dimensions: &[1],
                        ggml_type: GgmlType::F32,
                        data: &selected,
                    },
                    TensorInput {
                        name: "unselected.weight",
                        dimensions: &[1],
                        ggml_type: GgmlType::F32,
                        data: &unselected,
                    },
                ],
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
            WeightLeaseSource::Gguf(source) => {
                source
                    .lease
                    .identity()
                    .physical_selection()
                    .map(|selection| match selection {
                        GgufPhysicalSelection::Axis(selection) => selection.clone(),
                        GgufPhysicalSelection::DenseSpan(_) => {
                            panic!("expected single-axis GGUF selection")
                        }
                    })
            }
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
            .add_checkpoint_for_test(GgufCheckpoint::open(first).unwrap(), |_| {
                "shared.weight".into()
            })
            .unwrap();
        let error = builder
            .add_checkpoint_for_test(GgufCheckpoint::open(second).unwrap(), |_| {
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
        let store = GgufWeightStore::new_for_test(GgufCheckpoint::open(path).unwrap(), |name| {
            name.to_string()
        })
        .unwrap();
        assert_eq!(store.keys(), ["value.weight"]);
        assert_eq!(store.metadata("value.weight").unwrap().shape, [1]);
        let diagnostics = store.diagnostics().unwrap();
        assert_eq!(diagnostics.currently_mapped_shards, 0);
        assert!(diagnostics.touched_shard_paths.is_empty());
        assert_eq!(diagnostics.physical_reads, 0);
    }

    #[test]
    fn gguf_store_catalog_contains_only_contract_selected_sources() {
        use eredu_checkpoint::schema::{
            CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint,
            TensorOperation,
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        write_two_dense_gguf(&path);
        let plan = GgufCheckpointPlan::new(
            "selected layout",
            vec![GgufTensorConstraint::required(
                "selected.weight",
                vec![1],
                GgufTypeConstraint::OperationClass(TensorOperation::Dense),
            )],
            Vec::new(),
            CatalogPolicy::non_strict(),
        )
        .unwrap();
        let store = GgufWeightStore::builder()
            .add_checkpoint(GgufCheckpoint::open(path).unwrap(), &plan, str::to_string)
            .unwrap()
            .build()
            .unwrap();

        assert!(store.is_checkpoint_contract_resolved());
        assert_eq!(store.keys(), ["selected.weight"]);
        assert_eq!(store.unclaimed_checkpoint_keys(), ["unselected.weight"]);
        assert!(matches!(
            store.metadata("unselected.weight"),
            Err(WeightStoreError::UnknownTensor { key }) if key == "unselected.weight"
        ));
    }

    #[test]
    fn native_affine_store_bytes_equal_checkpoint_payload_bytes() {
        for ty in [GgmlType::Q4K, GgmlType::Q5_1, GgmlType::Q8_0] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("model.gguf");
            let (block_values, block_bytes) = ty.block_and_bytes().unwrap();
            let payload = vec![0; (2 * block_bytes) as usize];
            Writer::default()
                .write(
                    std::fs::File::create(&path).unwrap(),
                    &BTreeMap::new(),
                    &[TensorInput {
                        name: "bank.weight",
                        dimensions: &[block_values, 2],
                        ggml_type: ty,
                        data: &payload,
                    }],
                )
                .unwrap();
            let store =
                GgufWeightStore::new_for_test(GgufCheckpoint::open(path).unwrap(), str::to_string)
                    .unwrap();
            let metadata = store.metadata("bank.weight").unwrap();
            assert_eq!(metadata.shape, [2, block_bytes as usize], "{ty:?}");
            assert_eq!(metadata.stored_dtype, StoredDtype::U8, "{ty:?}");
            assert_eq!(metadata.logical_byte_len, payload.len(), "{ty:?}");
            assert_eq!(store.keys(), ["bank.weight"], "{ty:?}");
        }
    }

    #[test]
    fn gguf_dense_contiguous_span_reads_only_the_reshaped_interval() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        let values = write_dense_bank_gguf(&path);
        let store = GgufWeightStore::new_for_test(GgufCheckpoint::open(path).unwrap(), |name| {
            name.to_string()
        })
        .unwrap();
        let lease = store
            .acquire(
                "bank.weight",
                TensorSelection::Contiguous {
                    offset_elements: 8,
                    shape: vec![1, 2, 4],
                },
            )
            .unwrap();
        assert_eq!(lease.output_shape(), [1, 2, 4]);
        assert_eq!(lease.selected_byte_len(), 32);
        let WeightLeaseSource::Gguf(source) = &lease.source else {
            panic!("expected GGUF lease");
        };
        assert!(matches!(
            source.lease.identity().physical_selection(),
            Some(GgufPhysicalSelection::DenseSpan(_))
        ));

        let stream = cpu_stream();
        let selected = lease
            .materialize(&stream, &stream)
            .unwrap()
            .synchronize()
            .unwrap();
        assert_eq!(selected.shape(), [1, 2, 4]);
        assert_eq!(
            selected.evaluated().unwrap().as_slice::<f32>(),
            &values[8..16]
        );
        let diagnostics = store.diagnostics().unwrap();
        assert_eq!(diagnostics.physical_reads, 1);
        assert_eq!(diagnostics.physical_read_bytes, 32);
    }

    #[test]
    fn gguf_dense_contiguous_span_rejects_packed_input_before_payload_io() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        write_affine_gguf(&path);
        let store = GgufWeightStore::new_for_test(GgufCheckpoint::open(path).unwrap(), |name| {
            name.to_string()
        })
        .unwrap();
        let error = store
            .acquire(
                "bank.weight",
                TensorSelection::Contiguous {
                    offset_elements: 0,
                    shape: vec![1, 4],
                },
            )
            .unwrap_err();
        assert!(matches!(
            error,
            WeightStoreError::BoundedSelectionUnavailable { .. }
        ));
        assert!(error.to_string().contains("unquantized F32, F16, or BF16"));
        let diagnostics = store.diagnostics().unwrap();
        assert_eq!(diagnostics.physical_reads, 0);
        assert_eq!(diagnostics.physical_read_bytes, 0);
        assert!(diagnostics.touched_shard_paths.is_empty());
    }

    #[test]
    fn gguf_affine_companions_coalesce_selected_physical_reads() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        write_affine_gguf(&path);
        let store = GgufWeightStore::new_for_test(GgufCheckpoint::open(path).unwrap(), |name| {
            name.to_string()
        })
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
        let store = GgufWeightStore::new_for_test(GgufCheckpoint::open(path).unwrap(), |name| {
            name.to_string()
        })
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
        let store = GgufWeightStore::new_for_test(GgufCheckpoint::open(path).unwrap(), |name| {
            name.to_string()
        })
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
            .unwrap()
            .synchronize()
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
                GgufWeightStore::new_for_test(GgufCheckpoint::open(path).unwrap(), |name| {
                    name.to_string()
                })
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
    fn rejects_malformed_indexes_and_unsafe_shard_paths() {
        let malformed = tempfile::tempdir().unwrap();
        std::fs::write(
            malformed.path().join("model.safetensors.index.json"),
            b"{invalid",
        )
        .unwrap();
        assert!(matches!(
            SafetensorsWeightStore::open(malformed.path()),
            Err(eredu_checkpoint::store::StoreError::MalformedIndex { .. })
        ));

        let duplicate = tempfile::tempdir().unwrap();
        std::fs::write(
            duplicate.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"weight":"one.safetensors","weight":"two.safetensors"}}"#,
        )
        .unwrap();
        assert!(matches!(
            SafetensorsWeightStore::open(duplicate.path()),
            Err(eredu_checkpoint::store::StoreError::MalformedIndex { .. })
        ));

        for shard in ["../escape.safetensors", "/absolute.safetensors"] {
            let dir = tempfile::tempdir().unwrap();
            write_index(dir.path(), &[("weight", shard)]);
            assert!(matches!(
                SafetensorsWeightStore::open(dir.path()),
                Err(eredu_checkpoint::store::StoreError::UnsafeShardPath { .. })
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
        assert_eq!(first.backing_shard(), second.backing_shard());
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
        let pinned_value = one
            .materialize(&stream, &stream)
            .unwrap()
            .synchronize()
            .unwrap();
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
            .unwrap()
            .synchronize()
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
            .unwrap()
            .synchronize()
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
            .unwrap()
            .synchronize()
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
            .unwrap()
            .synchronize()
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
            .unwrap()
            .synchronize()
            .unwrap();
        let bf16 = store
            .acquire("bf16", TensorSelection::Full)
            .unwrap()
            .materialize(&stream, &stream)
            .unwrap()
            .synchronize()
            .unwrap();
        let fp8 = store
            .acquire("fp8", TensorSelection::Full)
            .unwrap()
            .materialize(&stream, &stream)
            .unwrap()
            .synchronize()
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
            .unwrap()
            .synchronize()
            .unwrap();
        let value = materialized.evaluated().unwrap();
        assert_eq!(value.as_slice::<i32>(), &[7]);
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
            lease
                .materialize(&stream, &stream)
                .unwrap()
                .synchronize()
                .unwrap()
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
            .unwrap()
            .synchronize()
            .unwrap();
        assert_eq!(value.evaluated().unwrap().as_slice::<i32>(), &[10, 20, 30]);
    }
}

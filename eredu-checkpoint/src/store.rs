//! Backend-neutral checkpoint storage and encoded tensor leases.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{Read, Seek, SeekFrom},
    ops::Range,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, MutexGuard,
    },
    time::SystemTime,
};

use crate::{
    safetensors::{SafetensorsShards, MAX_HEADER_BYTES},
    StoredDtype,
};
use safetensors::tensor::{Dtype, Metadata, TensorInfo};

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

/// Container-native provenance behind one logical checkpoint catalog key.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TensorSourceProvenance {
    /// Logical key accepted by [`CheckpointSource`].
    pub catalog_key: String,
    /// Physical tensor identity in the admitted container.
    pub physical_tensor: String,
    /// Exact logical output selected from the physical tensor.
    pub output: String,
    /// Payload shard backing the physical tensor, when file-backed.
    pub backing_shard: Option<PathBuf>,
    /// Exact physical container encoding.
    pub source_encoding: crate::SourceTensorEncoding,
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
    /// Physical payload bytes read for this lease.
    pub length_bytes: u64,
    /// Actual filesystem read operations, including exactness verification.
    pub physical_reads: u64,
    /// Actual filesystem bytes read, including exactness verification.
    pub physical_read_bytes: u64,
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
    /// Buffered SafeTensors bytes.
    Safetensors(SafetensorsLease),
    /// Lazily read portable GGUF payload.
    Gguf(Box<crate::gguf_store::GgufLease>),
    /// Immutable in-memory encoded bytes.
    Memory(MemoryLease),
}

impl EncodedTensorLease for CheckpointLease {
    fn metadata(&self) -> &TensorMetadata {
        match self {
            Self::Safetensors(lease) => lease.metadata(),
            Self::Gguf(lease) => lease.metadata(),
            Self::Memory(lease) => lease.metadata(),
        }
    }

    fn selection(&self) -> &TensorSelection {
        match self {
            Self::Safetensors(lease) => lease.selection(),
            Self::Gguf(lease) => lease.selection(),
            Self::Memory(lease) => lease.selection(),
        }
    }

    fn output_shape(&self) -> &[usize] {
        match self {
            Self::Safetensors(lease) => lease.output_shape(),
            Self::Gguf(lease) => lease.output_shape(),
            Self::Memory(lease) => lease.output_shape(),
        }
    }

    fn bounded_read_proof(&self) -> &BoundedReadProof {
        match self {
            Self::Safetensors(lease) => lease.bounded_read_proof(),
            Self::Gguf(lease) => lease.bounded_read_proof(),
            Self::Memory(lease) => lease.bounded_read_proof(),
        }
    }

    fn backing_path(&self) -> Option<&Path> {
        match self {
            Self::Safetensors(lease) => lease.backing_path(),
            Self::Gguf(lease) => lease.backing_path(),
            Self::Memory(lease) => lease.backing_path(),
        }
    }

    fn encoded_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Safetensors(lease) => lease.encoded_bytes(),
            Self::Gguf(lease) => lease.encoded_bytes(),
            Self::Memory(lease) => lease.encoded_bytes(),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct MemoryTensor {
    metadata: TensorMetadata,
    dtype: Dtype,
    bytes: Vec<u8>,
}

/// Encoded tensor selection retaining immutable in-memory storage.
#[derive(Debug, Clone)]
pub struct MemoryLease {
    tensor: Arc<MemoryTensor>,
    selection: TensorSelection,
    output_shape: Vec<usize>,
    proof: BoundedReadProof,
    span: Range<usize>,
    selected_bytes: Option<Arc<[u8]>>,
}

impl EncodedTensorLease for MemoryLease {
    fn metadata(&self) -> &TensorMetadata {
        &self.tensor.metadata
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
        None
    }

    fn encoded_bytes(&self) -> Option<&[u8]> {
        match &self.selected_bytes {
            Some(bytes) => Some(bytes.as_ref()),
            None => self.tensor.bytes.get(self.span.clone()),
        }
    }
}

/// Immutable in-memory SafeTensors-compatible encoded tensors.
#[derive(Debug, Default)]
pub struct MemoryWeightStore {
    tensors: BTreeMap<String, Arc<MemoryTensor>>,
}

impl MemoryWeightStore {
    /// Creates a store from owned encoded tensor payloads.
    pub fn from_safetensors(
        tensors: impl IntoIterator<Item = (String, Dtype, Vec<usize>, Vec<u8>)>,
    ) -> Result<Self, StoreError> {
        let mut catalog = BTreeMap::new();
        for (name, dtype, shape, bytes) in tensors {
            let mut metadata =
                metadata_for_parts(&name, Path::new("<memory>"), dtype, &shape, bytes.len())?;
            metadata.backing_shard = None;
            let tensor = Arc::new(MemoryTensor {
                metadata,
                dtype,
                bytes,
            });
            if catalog.insert(name.clone(), tensor).is_some() {
                return Err(StoreError::Internal(format!(
                    "duplicate in-memory tensor {name:?}"
                )));
            }
        }
        Ok(Self { tensors: catalog })
    }
}

impl WeightStore for MemoryWeightStore {
    type Lease = MemoryLease;

    fn keys(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    fn metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.tensors
            .get(key)
            .map(|tensor| tensor.metadata.clone())
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
    }

    fn acquire(&self, request: TensorReadRequest) -> Result<Self::Lease, StoreError> {
        let tensor =
            self.tensors
                .get(&request.key)
                .cloned()
                .ok_or_else(|| StoreError::UnknownTensor {
                    key: request.key.clone(),
                })?;
        let output_shape = validate_selection(
            &request.key,
            &tensor.metadata.logical_shape,
            &request.selection,
        )?;
        let (span, selected_bytes) = select_safetensors_bytes(
            &request.key,
            tensor.dtype,
            &tensor.metadata.logical_shape,
            &tensor.bytes,
            &request.selection,
            &output_shape,
            request.policy,
        )?;
        let length = selected_bytes
            .as_ref()
            .map_or(span.len(), |bytes| bytes.len());
        let full_selection = matches!(request.selection, TensorSelection::Full);
        Ok(MemoryLease {
            tensor,
            selection: request.selection,
            output_shape,
            proof: BoundedReadProof {
                physically_bounded: matches!(request.policy, ReadPolicy::RequireBounded)
                    || full_selection,
                offset_bytes: u64::try_from(span.start).map_err(|_| StoreError::Overflow {
                    context: "in-memory selection byte offset".into(),
                })?,
                length_bytes: u64::try_from(length).map_err(|_| StoreError::Overflow {
                    context: "in-memory selection byte length".into(),
                })?,
                physical_reads: 0,
                physical_read_bytes: 0,
            },
            span,
            selected_bytes: selected_bytes.map(Arc::from),
        })
    }

    fn diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        Ok(WeightStoreDiagnostics {
            backend: WeightStoreBackend::Memory,
            cache_hits: 0,
            cache_misses: 0,
            evictions: 0,
            currently_cached_shards: 0,
            touched_shard_paths: Vec::new(),
            payload_shard_paths: Vec::new(),
            physical_reads: 0,
            physical_read_bytes: 0,
            coalesced_group_hits: 0,
        })
    }
}

impl CheckpointSource for MemoryWeightStore {
    fn source_keys(&self) -> Vec<String> {
        WeightStore::keys(self)
    }

    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        WeightStore::metadata(self, key)
    }

    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        WeightStore::acquire(self, request).map(CheckpointLease::Memory)
    }

    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        WeightStore::diagnostics(self)
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

    /// Returns exact container provenance without opening tensor payloads.
    fn source_provenance(&self, key: &str) -> Result<TensorSourceProvenance, StoreError> {
        let metadata = self.source_metadata(key)?;
        Ok(TensorSourceProvenance {
            catalog_key: key.to_owned(),
            physical_tensor: key.to_owned(),
            output: key.to_owned(),
            backing_shard: metadata.backing_shard,
            source_encoding: crate::SourceTensorEncoding::Safetensors(metadata.stored_dtype),
        })
    }

    /// Returns physical source keys consumed to synthesize overlay bindings.
    fn materialized_source_keys(&self) -> Vec<String> {
        Vec::new()
    }

    /// Returns physical source shards whose payloads were consumed to build
    /// materialized overlay bindings.
    ///
    /// This is distinct from `touched_shard_paths`: catalog inspection may
    /// map shards solely to read tensor metadata, while this list records only
    /// the source payloads selected by an actual materialization plan.
    fn materialized_source_shards(&self) -> Vec<PathBuf> {
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

/// Immutable catalog entry retained across deferred payload acquisition.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PreparedTensorSource {
    /// Exact metadata admitted before payload access.
    pub metadata: TensorMetadata,
    /// Exact container identity and encoding admitted before payload access.
    pub provenance: TensorSourceProvenance,
}

/// Checkpoint source pinned to an exact metadata and provenance snapshot.
///
/// The wrapper revalidates the source before each acquisition and validates
/// the resulting lease before returning it. This closes the interval between
/// header-only preparation and deferred materialization without converting or
/// buffering payloads.
pub struct PreparedCheckpointSource {
    source: SharedCheckpointSource,
    catalog: BTreeMap<String, PreparedTensorSource>,
}

impl PreparedCheckpointSource {
    /// Opens an exact admitted SafeTensors shard set and pins it to metadata
    /// retained by header-only preparation.
    ///
    /// This performs no directory or index rediscovery and materializes no
    /// tensor payload. Deferred exact-range acquisition admits immutable bytes
    /// only after metadata, provenance, selection, and encoded-length checks.
    pub fn open_admitted_safetensors(
        shards: SafetensorsShards,
        catalog: BTreeMap<String, TensorMetadata>,
        max_cached_shards: usize,
    ) -> Result<Self, StoreError> {
        let source: SharedCheckpointSource = Arc::new(SafetensorsWeightStore::open_admitted(
            shards,
            max_cached_shards,
        )?);
        let catalog = catalog
            .into_iter()
            .map(|(key, metadata)| {
                let provenance = TensorSourceProvenance {
                    catalog_key: key.clone(),
                    physical_tensor: key.clone(),
                    output: key.clone(),
                    backing_shard: metadata.backing_shard.clone(),
                    source_encoding: crate::SourceTensorEncoding::Safetensors(
                        metadata.stored_dtype.clone(),
                    ),
                };
                (
                    key,
                    PreparedTensorSource {
                        metadata,
                        provenance,
                    },
                )
            })
            .collect();
        Self::new(source, catalog)
    }

    /// Pins a source to the supplied exact catalog.
    pub fn new(
        source: SharedCheckpointSource,
        catalog: BTreeMap<String, PreparedTensorSource>,
    ) -> Result<Self, StoreError> {
        let prepared = Self { source, catalog };
        let mut source_keys = prepared.source.source_keys();
        source_keys.sort();
        if source_keys != prepared.catalog.keys().cloned().collect::<Vec<_>>() {
            return Err(StoreError::PreparedCatalogMismatch {
                key: "<catalog>".into(),
            });
        }
        for key in prepared.catalog.keys() {
            prepared.validate_current(key)?;
        }
        Ok(prepared)
    }

    fn expected(&self, key: &str) -> Result<&PreparedTensorSource, StoreError> {
        self.catalog
            .get(key)
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
    }

    fn validate_current(&self, key: &str) -> Result<(), StoreError> {
        let expected = self.expected(key)?;
        if self.source.source_metadata(key)? != expected.metadata
            || self.source.source_provenance(key)? != expected.provenance
        {
            return Err(StoreError::PreparedCatalogMismatch { key: key.into() });
        }
        Ok(())
    }

    fn validate_lease(
        &self,
        request: &TensorReadRequest,
        lease: &CheckpointLease,
    ) -> Result<(), StoreError> {
        let expected = self.expected(&request.key)?;
        let proof = lease.bounded_read_proof();
        let full_selection = matches!(request.selection, TensorSelection::Full);
        let encoded_len = lease
            .encoded_bytes()
            .and_then(|bytes| u64::try_from(bytes.len()).ok());
        if lease.metadata() != &expected.metadata
            || lease.selection() != &request.selection
            || !proof.physically_bounded
            || (full_selection
                && (lease.output_shape() != expected.metadata.logical_shape
                    || proof.offset_bytes != 0
                    || proof.length_bytes != expected.metadata.encoded_byte_len))
            || lease.backing_path() != expected.metadata.backing_shard.as_deref()
            || encoded_len.is_some_and(|length| {
                length != proof.length_bytes
                    || (full_selection && length != expected.metadata.encoded_byte_len)
            })
        {
            return Err(StoreError::PreparedCatalogMismatch {
                key: request.key.clone(),
            });
        }
        self.validate_current(&request.key)
    }
}

impl CheckpointSource for PreparedCheckpointSource {
    fn source_keys(&self) -> Vec<String> {
        self.catalog.keys().cloned().collect()
    }

    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.validate_current(key)?;
        Ok(self.expected(key)?.metadata.clone())
    }

    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        self.validate_current(&request.key)?;
        let lease = self.source.acquire_lease(request.clone())?;
        self.validate_lease(&request, &lease)?;
        Ok(lease)
    }

    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        self.source.source_diagnostics()
    }

    fn source_provenance(&self, key: &str) -> Result<TensorSourceProvenance, StoreError> {
        self.validate_current(key)?;
        Ok(self.expected(key)?.provenance.clone())
    }

    fn materialized_source_keys(&self) -> Vec<String> {
        self.source.materialized_source_keys()
    }

    fn materialized_source_shards(&self) -> Vec<PathBuf> {
        self.source.materialized_source_shards()
    }

    fn unclaimed_checkpoint_keys(&self) -> Vec<String> {
        self.source.unclaimed_checkpoint_keys()
    }

    fn is_authoritative_materialized_key(&self, key: &str) -> bool {
        self.source.is_authoritative_materialized_key(key)
    }

    fn is_checkpoint_contract_resolved(&self) -> bool {
        self.source.is_checkpoint_contract_resolved()
    }
}

/// Disjoint logical union of independently opened checkpoint artifacts.
///
/// This is used by split model/projector artifacts while preserving each
/// source's native leases, bounded-read guarantees, and physical diagnostics.
pub struct CompositeCheckpointSource {
    sources: Vec<SharedCheckpointSource>,
    owners: BTreeMap<String, usize>,
}

impl CompositeCheckpointSource {
    /// Creates a deterministic union and rejects ambiguous logical keys.
    pub fn new(
        sources: impl IntoIterator<Item = SharedCheckpointSource>,
    ) -> Result<Self, StoreError> {
        let sources = sources.into_iter().collect::<Vec<_>>();
        if sources.is_empty() {
            return Err(StoreError::Internal(
                "composite checkpoint source requires at least one artifact".into(),
            ));
        }
        let mut owners = BTreeMap::new();
        for (owner, source) in sources.iter().enumerate() {
            for key in source.source_keys() {
                if let Some(previous) = owners.insert(key.clone(), owner) {
                    return Err(StoreError::Internal(format!(
                        "composite checkpoint key {key:?} is owned by sources {previous} and {owner}"
                    )));
                }
            }
        }
        Ok(Self { sources, owners })
    }

    fn source_for(&self, key: &str) -> Result<&dyn CheckpointSource, StoreError> {
        self.owners
            .get(key)
            .and_then(|owner| self.sources.get(*owner))
            .map(AsRef::as_ref)
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
    }
}

impl CheckpointSource for CompositeCheckpointSource {
    fn source_keys(&self) -> Vec<String> {
        self.owners.keys().cloned().collect()
    }

    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.source_for(key)?.source_metadata(key)
    }

    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        self.source_for(&request.key)?.acquire_lease(request)
    }

    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        let diagnostics = self
            .sources
            .iter()
            .map(|source| source.source_diagnostics())
            .collect::<Result<Vec<_>, _>>()?;
        let backend = diagnostics[0].backend;
        if diagnostics.iter().any(|value| value.backend != backend) {
            return Err(StoreError::Internal(
                "composite checkpoint sources use different physical backends".into(),
            ));
        }
        let mut touched = diagnostics
            .iter()
            .flat_map(|value| value.touched_shard_paths.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        touched.sort();
        let mut payloads = diagnostics
            .iter()
            .flat_map(|value| value.payload_shard_paths.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        payloads.sort();
        Ok(WeightStoreDiagnostics {
            backend,
            cache_hits: diagnostics.iter().map(|value| value.cache_hits).sum(),
            cache_misses: diagnostics.iter().map(|value| value.cache_misses).sum(),
            evictions: diagnostics.iter().map(|value| value.evictions).sum(),
            currently_cached_shards: diagnostics
                .iter()
                .map(|value| value.currently_cached_shards)
                .sum(),
            touched_shard_paths: touched,
            payload_shard_paths: payloads,
            physical_reads: diagnostics.iter().map(|value| value.physical_reads).sum(),
            physical_read_bytes: diagnostics
                .iter()
                .map(|value| value.physical_read_bytes)
                .sum(),
            coalesced_group_hits: diagnostics
                .iter()
                .map(|value| value.coalesced_group_hits)
                .sum(),
        })
    }

    fn source_provenance(&self, key: &str) -> Result<TensorSourceProvenance, StoreError> {
        self.source_for(key)?.source_provenance(key)
    }

    fn materialized_source_keys(&self) -> Vec<String> {
        self.sources
            .iter()
            .flat_map(|source| source.materialized_source_keys())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    fn materialized_source_shards(&self) -> Vec<PathBuf> {
        self.sources
            .iter()
            .flat_map(|source| source.materialized_source_shards())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    fn unclaimed_checkpoint_keys(&self) -> Vec<String> {
        self.sources
            .iter()
            .flat_map(|source| source.unclaimed_checkpoint_keys())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    fn is_authoritative_materialized_key(&self, key: &str) -> bool {
        self.source_for(key)
            .is_ok_and(|source| source.is_authoritative_materialized_key(key))
    }

    fn is_checkpoint_contract_resolved(&self) -> bool {
        self.sources
            .iter()
            .all(|source| source.is_checkpoint_contract_resolved())
    }
}

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

    fn source_provenance(&self, key: &str) -> Result<TensorSourceProvenance, StoreError> {
        if !self.source.is_authoritative_materialized_key(key) {
            self.authorize(key)?;
        }
        self.source.source_provenance(key)
    }

    fn materialized_source_keys(&self) -> Vec<String> {
        self.source
            .materialized_source_keys()
            .into_iter()
            .filter(|key| self.contract.source_keys().contains(key))
            .collect()
    }

    fn materialized_source_shards(&self) -> Vec<PathBuf> {
        self.source.materialized_source_shards()
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
    /// Buffered SafeTensors payload shards.
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
    /// Successful acquisitions reusing an existing shard buffer or reader.
    pub cache_hits: u64,
    /// Acquisitions loading a new shard buffer or opening a reader.
    pub cache_misses: u64,
    /// Unleased shard buffers or readers removed to honor a bound.
    pub evictions: u64,
    /// Shard buffers or readers currently retained by the store.
    pub currently_cached_shards: usize,
    /// Shard paths touched so far in stable order.
    pub touched_shard_paths: Vec<PathBuf>,
    /// Shard paths selected for tensor payload access in stable order.
    ///
    /// Unlike `touched_shard_paths`, metadata-only catalog validation does not
    /// add an entry here.
    pub payload_shard_paths: Vec<PathBuf>,
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
    /// The configured cached-shard or reader limit was zero.
    #[error("maximum cached-shard count must be nonzero")]
    InvalidShardCacheLimit,
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
    /// Canonical SafeTensors shard discovery or path admission failed.
    #[error(transparent)]
    SafetensorsShards(#[from] crate::safetensors::SafetensorsShardError),
    /// A SafeTensors payload header or contents are invalid.
    #[error("malformed safetensors shard {path}: {message}", path = .path.display())]
    MalformedSafetensors {
        /// Payload path.
        path: PathBuf,
        /// Parser detail.
        message: String,
    },
    /// An index maps a tensor to a shard that does not contain it.
    #[error("index maps tensor {key:?} to {path}, but that shard does not contain it", path = .path.display())]
    ContradictoryIndexMapping {
        /// Tensor key from the index.
        key: String,
        /// Referenced payload shard.
        path: PathBuf,
    },
    /// A payload shard contains a tensor absent from its index mappings.
    #[error("shard {path} contains tensor {key:?}, but the index does not map it to that shard", path = .path.display())]
    UnindexedShardTensor {
        /// Tensor key found in the shard header.
        key: String,
        /// Payload shard containing the unexpected tensor.
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
    #[error("checkpoint shard-cache capacity {maximum} is exhausted; leased shards: {leased:?}")]
    CapacityExhausted {
        /// Configured shard-cache bound.
        maximum: usize,
        /// Deterministically ordered pinned paths.
        leased: Vec<PathBuf>,
    },
    /// Physical checkpoint metadata no longer matches an admitted catalog.
    #[error("checkpoint tensor {key:?} no longer matches the prepared catalog")]
    PreparedCatalogMismatch {
        /// Logical tensor whose physical metadata changed.
        key: String,
    },
    /// An admitted filesystem object changed after its metadata snapshot.
    #[error("admitted checkpoint file changed after preparation: {path}", path = .path.display())]
    AdmittedFileChanged {
        /// Exact admitted path whose pinned object changed.
        path: PathBuf,
    },
    /// Filesystem or container access failed.
    #[error("checkpoint I/O failed for {path}: {message}", path = .path.display())]
    Io {
        /// Affected path.
        path: PathBuf,
        /// Stable failure detail.
        message: String,
    },
    /// The catalog or shard cache is internally unavailable.
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

/// Default maximum number of simultaneously retained shard buffers.
pub const DEFAULT_MAX_CACHED_SHARDS: usize = 4;

#[derive(Debug)]
struct CachedShard {
    path: PathBuf,
    admitted_file: Arc<AdmittedFile>,
    metadata: Metadata,
    payload_offset: usize,
    full_tensors: Mutex<BTreeMap<String, Arc<[u8]>>>,
}

#[derive(Debug)]
struct CacheEntry {
    shard: Arc<CachedShard>,
    last_used: u64,
}

#[derive(Debug, Default)]
struct CacheState {
    entries: BTreeMap<PathBuf, CacheEntry>,
    touched: BTreeSet<PathBuf>,
    payloads: BTreeSet<PathBuf>,
    tick: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

#[derive(Debug, Default)]
struct SafetensorsReadTelemetry {
    physical_reads: AtomicU64,
    physical_read_bytes: AtomicU64,
}

#[derive(Debug, Clone)]
struct CatalogEntry {
    shard: PathBuf,
}

#[derive(Debug, Eq, PartialEq)]
struct AdmittedFileIdentity {
    canonical_path: PathBuf,
    len: u64,
    modified: SystemTime,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    // Unix ctime is the strongest change-version metadata exposed by the
    // standard library: unlike mtime, normal timestamp APIs cannot restore it.
    #[cfg(unix)]
    change_time_seconds: i64,
    #[cfg(unix)]
    change_time_nanoseconds: i64,
    // Other targets do not expose an equivalent change counter through the
    // portable Metadata API. Retain creation time when the filesystem reports
    // it, in addition to length and modification time.
    #[cfg(not(unix))]
    created: Option<SystemTime>,
    #[cfg(windows)]
    file_attributes: u32,
    #[cfg(windows)]
    creation_time: u64,
}

impl AdmittedFileIdentity {
    fn from_metadata(path: &Path, metadata: &std::fs::Metadata) -> Result<Self, StoreError> {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt as _;

        Ok(Self {
            canonical_path: path.to_path_buf(),
            len: metadata.len(),
            modified: metadata.modified().map_err(|error| fs_error(path, error))?,
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
            #[cfg(unix)]
            change_time_seconds: metadata.ctime(),
            #[cfg(unix)]
            change_time_nanoseconds: metadata.ctime_nsec(),
            #[cfg(not(unix))]
            created: metadata.created().ok(),
            #[cfg(windows)]
            file_attributes: {
                use std::os::windows::fs::MetadataExt as _;
                metadata.file_attributes()
            },
            #[cfg(windows)]
            creation_time: {
                use std::os::windows::fs::MetadataExt as _;
                metadata.creation_time()
            },
        })
    }
}

#[derive(Debug)]
struct AdmittedFile {
    identity: AdmittedFileIdentity,
}

impl AdmittedFile {
    fn open(path: &Path) -> Result<Self, StoreError> {
        let file = File::open(path).map_err(|error| fs_error(path, error))?;
        let identity = AdmittedFileIdentity::from_metadata(
            path,
            &file.metadata().map_err(|error| fs_error(path, error))?,
        )?;
        Ok(Self { identity })
    }

    fn open_validated(&self, path: &Path) -> Result<File, StoreError> {
        if path != self.identity.canonical_path {
            return Err(StoreError::AdmittedFileChanged {
                path: path.to_path_buf(),
            });
        }
        let file = File::open(path).map_err(|error| fs_error(path, error))?;
        self.validate_file(path, &file)?;
        Ok(file)
    }

    fn validate_file(&self, path: &Path, file: &File) -> Result<(), StoreError> {
        let current = AdmittedFileIdentity::from_metadata(
            path,
            &file.metadata().map_err(|error| fs_error(path, error))?,
        )?;
        if current != self.identity {
            return Err(StoreError::AdmittedFileChanged {
                path: path.to_path_buf(),
            });
        }
        Ok(())
    }
}

/// Encoded SafeTensors selection retaining its cached shard metadata.
#[derive(Debug, Clone)]
pub struct SafetensorsLease {
    metadata: TensorMetadata,
    selection: TensorSelection,
    output_shape: Vec<usize>,
    proof: BoundedReadProof,
    shard: Arc<CachedShard>,
    bytes: Arc<[u8]>,
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
        Some(&self.bytes)
    }
}

/// Persistent neutral SafeTensors catalog with bounded shard-buffer ownership.
#[derive(Debug)]
pub struct SafetensorsWeightStore {
    catalog: BTreeMap<String, CatalogEntry>,
    indexed_shards: BTreeMap<PathBuf, BTreeSet<String>>,
    admitted_files: BTreeMap<PathBuf, Arc<AdmittedFile>>,
    metadata: Mutex<BTreeMap<String, TensorMetadata>>,
    cache: Mutex<CacheState>,
    read_telemetry: Arc<SafetensorsReadTelemetry>,
    max_cached_shards: usize,
}

impl SafetensorsWeightStore {
    /// Opens a file, indexed directory, or directory containing `model.safetensors`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        Self::open_with_max_cached_shards(path, DEFAULT_MAX_CACHED_SHARDS)
    }

    /// Opens a checkpoint with an explicit nonzero shard-cache bound.
    pub fn open_with_max_cached_shards(
        path: impl AsRef<Path>,
        max_cached_shards: usize,
    ) -> Result<Self, StoreError> {
        let shards = SafetensorsShards::discover_catalog(path)?;
        Self::open_admitted(shards, max_cached_shards)
    }

    /// Opens the exact shard set admitted by portable artifact inspection.
    ///
    /// This constructor performs no directory or index discovery. Indexed
    /// shards remain selectively opened and governed by `max_cached_shards`.
    pub fn open_admitted(
        shards: SafetensorsShards,
        max_cached_shards: usize,
    ) -> Result<Self, StoreError> {
        if max_cached_shards == 0 {
            return Err(StoreError::InvalidShardCacheLimit);
        }
        // Snapshot every admitted object without retaining one descriptor per
        // shard. A selected shard is reopened and validated against this exact
        // identity only after preflight.
        let admitted_files = shards
            .payload_paths()
            .iter()
            .map(|path| AdmittedFile::open(path).map(|file| (path.clone(), Arc::new(file))))
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        if let Some(locations) = shards.tensor_locations() {
            let mut indexed_shards = BTreeMap::<PathBuf, BTreeSet<String>>::new();
            for (key, shard) in locations {
                indexed_shards
                    .entry(shard.clone())
                    .or_default()
                    .insert(key.clone());
            }
            let catalog = locations
                .iter()
                .map(|(key, shard)| {
                    (
                        key.clone(),
                        CatalogEntry {
                            shard: shard.clone(),
                        },
                    )
                })
                .collect();
            return Ok(Self {
                catalog,
                indexed_shards,
                admitted_files,
                metadata: Mutex::new(BTreeMap::new()),
                cache: Mutex::new(CacheState::default()),
                read_telemetry: Arc::new(SafetensorsReadTelemetry::default()),
                max_cached_shards,
            });
        }
        let file = shards
            .payload_paths()
            .first()
            .expect("unindexed discovery returns one payload")
            .clone();
        let admitted_file = Arc::clone(
            admitted_files
                .get(&file)
                .expect("admitted payload has an identity snapshot"),
        );
        Self::from_single_file(file, admitted_file, admitted_files, max_cached_shards)
    }

    fn from_single_file(
        file: PathBuf,
        admitted_file: Arc<AdmittedFile>,
        admitted_files: BTreeMap<PathBuf, Arc<AdmittedFile>>,
        max_cached_shards: usize,
    ) -> Result<Self, StoreError> {
        let discovered = inspect_file(&file, &admitted_file)?;
        let catalog = discovered
            .keys()
            .map(|key| {
                (
                    key.clone(),
                    CatalogEntry {
                        shard: file.clone(),
                    },
                )
            })
            .collect();
        Ok(Self {
            catalog,
            indexed_shards: BTreeMap::new(),
            admitted_files,
            metadata: Mutex::new(discovered),
            cache: Mutex::new(CacheState::default()),
            read_telemetry: Arc::new(SafetensorsReadTelemetry::default()),
            max_cached_shards,
        })
    }

    fn lock_cache(&self) -> Result<MutexGuard<'_, CacheState>, StoreError> {
        self.cache
            .lock()
            .map_err(|_| StoreError::Internal("checkpoint shard cache is poisoned".into()))
    }

    fn acquire_shard(&self, entry: &CatalogEntry) -> Result<Arc<CachedShard>, StoreError> {
        let canonical_path = entry.shard.clone();
        let mut cache = self.lock_cache()?;
        cache.tick = cache.tick.saturating_add(1);
        let tick = cache.tick;
        if let Some(shard) = cache
            .entries
            .get(&canonical_path)
            .map(|entry| Arc::clone(&entry.shard))
        {
            drop(shard.admitted_file.open_validated(&canonical_path)?);
            cache.hits = cache.hits.saturating_add(1);
            cache.entries.get_mut(&canonical_path).unwrap().last_used = tick;
            return Ok(shard);
        }
        cache.misses = cache.misses.saturating_add(1);
        if cache.entries.len() >= self.max_cached_shards {
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
                    maximum: self.max_cached_shards,
                    leased: cache
                        .entries
                        .values()
                        .map(|entry| entry.shard.path.clone())
                        .collect(),
                });
            }
        }
        let admitted_file =
            Arc::clone(self.admitted_files.get(&canonical_path).ok_or_else(|| {
                StoreError::Internal(format!(
                    "admitted SafeTensors file is missing for {}",
                    canonical_path.display()
                ))
            })?);
        let (payload_offset, metadata) =
            read_safetensors_metadata(&canonical_path, &admitted_file)?;
        let shard = Arc::new(CachedShard {
            path: entry.shard.clone(),
            admitted_file,
            metadata,
            payload_offset,
            full_tensors: Mutex::new(BTreeMap::new()),
        });
        if let Some(expected) = self.indexed_shards.get(&shard.path) {
            let actual = shard
                .metadata
                .offset_keys()
                .into_iter()
                .collect::<BTreeSet<_>>();
            if let Some(key) = expected.difference(&actual).next() {
                return Err(StoreError::ContradictoryIndexMapping {
                    key: key.clone(),
                    path: shard.path.clone(),
                });
            }
            if let Some(key) = actual.difference(expected).next() {
                return Err(StoreError::UnindexedShardTensor {
                    key: key.clone(),
                    path: shard.path.clone(),
                });
            }
            let discovered = expected
                .iter()
                .map(|key| {
                    let info = shard
                        .metadata
                        .info(key)
                        .expect("exact shard validation established the tensor");
                    metadata_for_info(key, &shard.path, info)
                        .map(|metadata| (key.clone(), metadata))
                })
                .collect::<Result<BTreeMap<_, _>, _>>()?;
            self.metadata
                .lock()
                .map_err(|_| StoreError::Internal("metadata cache is poisoned".into()))?
                .extend(discovered);
        }
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

    fn cached_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.metadata
            .lock()
            .map_err(|_| StoreError::Internal("metadata cache is poisoned".into()))?
            .get(key)
            .cloned()
            .ok_or_else(|| {
                StoreError::Internal(format!(
                    "opened safetensors shard did not populate metadata for {key:?}"
                ))
            })
    }

    fn validate_admitted_path(&self, path: &Path) -> Result<(), StoreError> {
        let admitted = self.admitted_files.get(path).ok_or_else(|| {
            StoreError::Internal(format!(
                "admitted SafeTensors file is missing for {}",
                path.display()
            ))
        })?;
        drop(admitted.open_validated(path)?);
        Ok(())
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
            let entry = self
                .catalog
                .get(key)
                .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })?;
            self.validate_admitted_path(&entry.shard)?;
            return Ok(metadata);
        }
        let entry = self
            .catalog
            .get(key)
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })?;
        let shard = self.acquire_shard(entry)?;
        drop(shard);
        self.cached_metadata(key)
    }

    fn acquire(&self, request: TensorReadRequest) -> Result<Self::Lease, StoreError> {
        let entry = self
            .catalog
            .get(&request.key)
            .ok_or_else(|| StoreError::UnknownTensor {
                key: request.key.clone(),
            })?;
        let shard = self.acquire_shard(entry)?;
        let metadata = self.cached_metadata(&request.key)?;
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
        let tensor_len = info
            .data_offsets
            .1
            .checked_sub(info.data_offsets.0)
            .ok_or_else(|| io_error(&shard.path, "tensor payload offsets descend"))?;
        let read = plan_safetensors_reads(
            &request.key,
            info.dtype,
            &info.shape,
            tensor_len,
            &request.selection,
            &output_shape,
            request.policy,
        )?;
        let cached = shard
            .full_tensors
            .lock()
            .map_err(|_| StoreError::Internal("checkpoint tensor cache is poisoned".into()))?
            .get(&request.key)
            .cloned();
        let complete_tensor =
            read.ranges.len() == 1 && read.ranges[0].start == 0 && read.ranges[0].end == tensor_len;
        let cache_hit = cached.is_some();
        let bytes = match cached {
            Some(bytes) if complete_tensor => bytes,
            Some(bytes) => Arc::from(copy_safetensors_ranges(
                &request.key,
                bytes.as_ref(),
                &read.ranges,
            )?),
            None => {
                let bytes: Arc<[u8]> = Arc::from(read_safetensors_ranges(
                    &shard.path,
                    &shard.admitted_file,
                    payload_start,
                    &read.ranges,
                    self.read_telemetry.as_ref(),
                )?);
                if complete_tensor {
                    shard
                        .full_tensors
                        .lock()
                        .map_err(|_| {
                            StoreError::Internal("checkpoint tensor cache is poisoned".into())
                        })?
                        .insert(request.key.clone(), Arc::clone(&bytes));
                }
                bytes
            }
        };
        let length = u64::try_from(bytes.len()).map_err(|_| StoreError::Overflow {
            context: format!("physical read length for {:?}", request.key),
        })?;
        self.lock_cache()?.payloads.insert(shard.path.clone());
        Ok(SafetensorsLease {
            metadata,
            selection: request.selection,
            output_shape,
            proof: BoundedReadProof {
                physically_bounded: read.physically_bounded,
                offset_bytes: u64::try_from(read.ranges[0].start).map_err(|_| {
                    StoreError::Overflow {
                        context: "selection byte offset".into(),
                    }
                })?,
                length_bytes: length,
                physical_reads: if cache_hit {
                    0
                } else {
                    u64::try_from(read.ranges.len())
                        .unwrap_or(u64::MAX)
                        .saturating_mul(2)
                },
                physical_read_bytes: if cache_hit {
                    0
                } else {
                    length.saturating_mul(2)
                },
            },
            shard,
            bytes,
        })
    }

    fn diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        let cache = self.lock_cache()?;
        Ok(WeightStoreDiagnostics {
            backend: WeightStoreBackend::Safetensors,
            cache_hits: cache.hits,
            cache_misses: cache.misses,
            evictions: cache.evictions,
            currently_cached_shards: cache.entries.len(),
            touched_shard_paths: cache.touched.iter().cloned().collect(),
            payload_shard_paths: cache.payloads.iter().cloned().collect(),
            physical_reads: self.read_telemetry.physical_reads.load(Ordering::Relaxed),
            physical_read_bytes: self
                .read_telemetry
                .physical_read_bytes
                .load(Ordering::Relaxed),
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

fn inspect_file(
    path: &Path,
    admitted_file: &AdmittedFile,
) -> Result<BTreeMap<String, TensorMetadata>, StoreError> {
    let (_, metadata) = read_safetensors_metadata(path, admitted_file)?;
    metadata
        .tensors()
        .into_iter()
        .map(|(key, info)| metadata_for_info(&key, path, info).map(|metadata| (key, metadata)))
        .collect()
}

fn read_safetensors_metadata(
    path: &Path,
    admitted_file: &AdmittedFile,
) -> Result<(usize, Metadata), StoreError> {
    let mut file = admitted_file.open_validated(path)?;
    file.seek(SeekFrom::Start(0))
        .map_err(|error| io_error(path, error))?;
    let file_len = file
        .metadata()
        .map_err(|error| fs_error(path, error))?
        .len();
    let metadata = read_safetensors_metadata_from(path, &mut file, file_len)?;
    admitted_file.validate_file(path, &file)?;
    Ok(metadata)
}

fn read_safetensors_metadata_from(
    path: &Path,
    reader: &mut impl Read,
    file_len: u64,
) -> Result<(usize, Metadata), StoreError> {
    let mut encoded_header_len = [0u8; 8];
    reader
        .read_exact(&mut encoded_header_len)
        .map_err(|error| io_error(path, error))?;
    let header_len = u64::from_le_bytes(encoded_header_len);
    if header_len > MAX_HEADER_BYTES {
        return Err(StoreError::MalformedSafetensors {
            path: path.to_path_buf(),
            message: format!("header exceeds {MAX_HEADER_BYTES} bytes"),
        });
    }
    let payload_offset = 8u64
        .checked_add(header_len)
        .ok_or_else(|| StoreError::Overflow {
            context: format!("payload offset for {}", path.display()),
        })?;
    if payload_offset > file_len {
        return Err(StoreError::MalformedSafetensors {
            path: path.to_path_buf(),
            message: "header exceeds shard length".into(),
        });
    }
    let header_len = usize::try_from(header_len).map_err(|_| StoreError::Overflow {
        context: format!("header length for {}", path.display()),
    })?;
    let mut encoded_header = Vec::with_capacity(8 + header_len);
    encoded_header.extend_from_slice(&encoded_header_len);
    encoded_header.resize(8 + header_len, 0);
    reader
        .read_exact(&mut encoded_header[8..])
        .map_err(|error| io_error(path, error))?;
    let metadata = serde_json::from_slice::<Metadata>(&encoded_header[8..]).map_err(|error| {
        StoreError::MalformedSafetensors {
            path: path.to_path_buf(),
            message: error.to_string(),
        }
    })?;
    let described_payload =
        u64::try_from(metadata.data_len()).map_err(|_| StoreError::Overflow {
            context: format!("described payload length for {}", path.display()),
        })?;
    let described_file_len = payload_offset
        .checked_add(described_payload)
        .ok_or_else(|| StoreError::Overflow {
            context: format!("described shard length for {}", path.display()),
        })?;
    if described_file_len != file_len {
        return Err(StoreError::MalformedSafetensors {
            path: path.to_path_buf(),
            message: format!(
                "header describes {described_payload} payload bytes, but shard length provides {}",
                file_len - payload_offset
            ),
        });
    }
    let payload_offset = usize::try_from(payload_offset).map_err(|_| StoreError::Overflow {
        context: format!("payload offset for {}", path.display()),
    })?;
    Ok((payload_offset, metadata))
}

struct SafetensorsReadPlan {
    ranges: Vec<Range<usize>>,
    physically_bounded: bool,
}

impl SafetensorsReadPlan {
    fn single(range: Range<usize>, physically_bounded: bool) -> Self {
        Self {
            ranges: std::iter::once(range).collect(),
            physically_bounded,
        }
    }
}

fn push_coalesced_range(ranges: &mut Vec<Range<usize>>, range: Range<usize>) {
    if let Some(previous) = ranges.last_mut() {
        if previous.end == range.start {
            previous.end = range.end;
            return;
        }
    }
    ranges.push(range);
}

fn plan_safetensors_reads(
    key: &str,
    dtype: Dtype,
    shape: &[usize],
    payload_len: usize,
    selection: &TensorSelection,
    output_shape: &[usize],
    policy: ReadPolicy,
) -> Result<SafetensorsReadPlan, StoreError> {
    let bounded = matches!(policy, ReadPolicy::RequireBounded);
    if matches!(selection, TensorSelection::Full) {
        return Ok(SafetensorsReadPlan::single(0..payload_len, true));
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
        if end > payload_len {
            return Err(invalid_selection(
                key,
                "contiguous byte span outside payload",
            ));
        }
        return Ok(SafetensorsReadPlan::single(start..end, true));
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
        let row_bytes = payload_len
            .checked_div(shape[0])
            .filter(|_| payload_len.is_multiple_of(shape[0]))
            .ok_or_else(|| invalid_selection(key, "payload is not row divisible"))?;
        let byte_start = start
            .checked_mul(row_bytes)
            .ok_or_else(|| StoreError::Overflow {
                context: format!("row selection byte start for {key:?}"),
            })?;
        let byte_end = end
            .checked_mul(row_bytes)
            .ok_or_else(|| StoreError::Overflow {
                context: format!("row selection byte end for {key:?}"),
            })?;
        return Ok(SafetensorsReadPlan::single(byte_start..byte_end, true));
    }
    if !bounded {
        return Ok(SafetensorsReadPlan::single(0..payload_len, false));
    }
    let (axis, indices): (usize, Vec<usize>) = match selection {
        TensorSelection::Range { axis, start, end } => (*axis, (*start..*end).collect()),
        TensorSelection::Indices { axis, indices } => (*axis, indices.clone()),
        TensorSelection::Contiguous { .. } => {
            return Err(StoreError::BoundedSelectionUnavailable {
                key: key.into(),
                message: "packed contiguous selection is not byte aligned".into(),
            });
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
    let block_bytes = if bits == 4 {
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
        inner / 2
    } else {
        inner
            .checked_mul(
                scalar_bytes.ok_or_else(|| StoreError::BoundedSelectionUnavailable {
                    key: key.into(),
                    message: "stored scalar width is not byte aligned".into(),
                })?,
            )
            .ok_or_else(|| StoreError::Overflow {
                context: format!("selection block bytes for {key:?}"),
            })?
    };
    let mut ranges = Vec::new();
    for outer_index in 0..outer {
        for index in &indices {
            let start = outer_index
                .checked_mul(axis_len)
                .and_then(|value| value.checked_add(*index))
                .and_then(|value| value.checked_mul(block_bytes))
                .ok_or_else(|| StoreError::Overflow {
                    context: format!("selection byte start for {key:?}"),
                })?;
            let end = start
                .checked_add(block_bytes)
                .ok_or_else(|| StoreError::Overflow {
                    context: format!("selection byte end for {key:?}"),
                })?;
            if end > payload_len {
                return Err(invalid_selection(key, "selection exceeds payload"));
            }
            push_coalesced_range(&mut ranges, start..end);
        }
    }
    if ranges.is_empty() {
        return Err(invalid_selection(
            key,
            "selection produced no physical ranges",
        ));
    }
    Ok(SafetensorsReadPlan {
        ranges,
        physically_bounded: true,
    })
}

fn read_safetensors_ranges(
    path: &Path,
    admitted_file: &AdmittedFile,
    tensor_payload_start: usize,
    ranges: &[Range<usize>],
    telemetry: &SafetensorsReadTelemetry,
) -> Result<Vec<u8>, StoreError> {
    read_safetensors_ranges_with_hook(
        path,
        admitted_file,
        tensor_payload_start,
        ranges,
        telemetry,
        || {},
    )
}

fn read_safetensors_ranges_with_hook(
    path: &Path,
    admitted_file: &AdmittedFile,
    tensor_payload_start: usize,
    ranges: &[Range<usize>],
    telemetry: &SafetensorsReadTelemetry,
    between_passes: impl FnOnce(),
) -> Result<Vec<u8>, StoreError> {
    let capacity = ranges.iter().try_fold(0usize, |total, range| {
        total
            .checked_add(range.len())
            .ok_or_else(|| StoreError::Overflow {
                context: format!("selected payload length for {}", path.display()),
            })
    })?;
    let mut file = admitted_file.open_validated(path)?;
    let first = read_safetensors_range_pass(
        path,
        &mut file,
        tensor_payload_start,
        ranges,
        capacity,
        telemetry,
    )?;
    admitted_file.validate_file(path, &file)?;
    between_passes();
    admitted_file.validate_file(path, &file)?;
    let second = read_safetensors_range_pass(
        path,
        &mut file,
        tensor_payload_start,
        ranges,
        capacity,
        telemetry,
    )?;
    admitted_file.validate_file(path, &file)?;
    if first != second {
        return Err(StoreError::AdmittedFileChanged {
            path: path.to_path_buf(),
        });
    }
    Ok(first)
}

fn read_safetensors_range_pass(
    path: &Path,
    file: &mut File,
    tensor_payload_start: usize,
    ranges: &[Range<usize>],
    capacity: usize,
    telemetry: &SafetensorsReadTelemetry,
) -> Result<Vec<u8>, StoreError> {
    let mut output = Vec::with_capacity(capacity);
    for range in ranges {
        let absolute = tensor_payload_start
            .checked_add(range.start)
            .ok_or_else(|| StoreError::Overflow {
                context: format!("selected payload offset for {}", path.display()),
            })?;
        file.seek(SeekFrom::Start(u64::try_from(absolute).map_err(|_| {
            StoreError::Overflow {
                context: format!("selected payload offset for {}", path.display()),
            }
        })?))
        .map_err(|error| io_error(path, error))?;
        let start = output.len();
        output.resize(start + range.len(), 0);
        file.read_exact(&mut output[start..])
            .map_err(|error| io_error(path, error))?;
        telemetry.physical_reads.fetch_add(1, Ordering::Relaxed);
        telemetry
            .physical_read_bytes
            .fetch_add(range.len() as u64, Ordering::Relaxed);
    }
    Ok(output)
}

fn copy_safetensors_ranges(
    key: &str,
    payload: &[u8],
    ranges: &[Range<usize>],
) -> Result<Vec<u8>, StoreError> {
    let capacity = ranges.iter().try_fold(0usize, |total, range| {
        total
            .checked_add(range.len())
            .ok_or_else(|| StoreError::Overflow {
                context: format!("cached selected payload length for {key:?}"),
            })
    })?;
    let mut output = Vec::with_capacity(capacity);
    for range in ranges {
        output.extend_from_slice(
            payload
                .get(range.clone())
                .ok_or_else(|| invalid_selection(key, "cached selection exceeds payload"))?,
        );
    }
    Ok(output)
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
    use crate::{
        schema::{
            CatalogPolicy, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
            StoredDtypeConstraint,
        },
        validation::resolve_safetensors_plan,
    };
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
                physical_reads: 1,
                physical_read_bytes: 4,
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
    fn safetensors_metadata_parser_never_requests_payload_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.safetensors");
        let payload = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        serialize_to_file(
            [(
                "weight",
                TensorView::new(Dtype::F32, vec![2, 2], &payload).unwrap(),
            )],
            None,
            &path,
        )
        .unwrap();
        let encoded = std::fs::read(&path).unwrap();
        let header_len =
            usize::try_from(u64::from_le_bytes(encoded[..8].try_into().unwrap())).unwrap();
        let payload_offset = 8 + header_len;
        let mut header_only = std::io::Cursor::new(&encoded[..payload_offset]);

        let (actual_offset, metadata) = read_safetensors_metadata_from(
            &path,
            &mut header_only,
            u64::try_from(encoded.len()).unwrap(),
        )
        .unwrap();

        assert_eq!(actual_offset, payload_offset);
        assert_eq!(metadata.info("weight").unwrap().shape, [2, 2]);
        assert_eq!(
            header_only.position(),
            u64::try_from(payload_offset).unwrap()
        );
    }

    #[test]
    fn prepared_safetensors_open_uses_admitted_shards_without_payload_reads() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.safetensors");
        let payload = f32_bytes(&[1.0, 2.0]);
        serialize_to_file(
            [(
                "weight",
                TensorView::new(Dtype::F32, vec![2], &payload).unwrap(),
            )],
            None,
            &path,
        )
        .unwrap();
        let admitted =
            crate::safetensors::SafetensorsMetadataCatalog::discover(directory.path()).unwrap();
        let prepared = PreparedCheckpointSource::open_admitted_safetensors(
            admitted.admitted_shards(),
            admitted.tensors().clone(),
            1,
        )
        .unwrap();

        assert_eq!(prepared.source_keys(), vec!["weight"]);
        let diagnostics = prepared.source_diagnostics().unwrap();
        assert_eq!(diagnostics.physical_reads, 0);
        assert_eq!(diagnostics.physical_read_bytes, 0);
        assert!(diagnostics.payload_shard_paths.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn prepared_many_shard_catalog_retains_no_descriptor_per_shard() {
        fn open_descriptor_count() -> usize {
            let directory = if Path::new("/proc/self/fd").is_dir() {
                Path::new("/proc/self/fd")
            } else {
                Path::new("/dev/fd")
            };
            std::fs::read_dir(directory).unwrap().count()
        }

        let directory = tempfile::tempdir().unwrap();
        let mut weight_map = BTreeMap::new();
        for index in 0..96 {
            let key = format!("weight.{index}");
            let file_name = format!("model-{index:05}-of-00096.safetensors");
            let path = directory.path().join(&file_name);
            let payload = [u8::try_from(index).unwrap()];
            serialize_to_file(
                [(
                    key.as_str(),
                    TensorView::new(Dtype::U8, vec![1], &payload).unwrap(),
                )],
                None,
                &path,
            )
            .unwrap();
            weight_map.insert(key, file_name);
        }
        std::fs::write(
            directory.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
        )
        .unwrap();
        let admitted =
            crate::safetensors::SafetensorsMetadataCatalog::discover(directory.path()).unwrap();
        let before = open_descriptor_count();
        let prepared = PreparedCheckpointSource::open_admitted_safetensors(
            admitted.admitted_shards(),
            admitted.tensors().clone(),
            1,
        )
        .unwrap();
        let after = open_descriptor_count();

        // The descriptor-directory iterator itself and unrelated parallel
        // tests may account for a small fluctuation. A shard-scaled leak would
        // retain all 96 descriptors and exceed this fixed allowance.
        assert!(
            after <= before + 8,
            "descriptor count grew from {before} to {after}"
        );
        let diagnostics = prepared.source_diagnostics().unwrap();
        assert_eq!(diagnostics.physical_reads, 0);
        assert!(diagnostics.payload_shard_paths.is_empty());
    }

    #[test]
    fn prepared_safetensors_open_rejects_catalog_substitution_before_payload_reads() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.safetensors");
        let original = f32_bytes(&[1.0, 2.0]);
        serialize_to_file(
            [(
                "weight",
                TensorView::new(Dtype::F32, vec![2], &original).unwrap(),
            )],
            None,
            &path,
        )
        .unwrap();
        let admitted =
            crate::safetensors::SafetensorsMetadataCatalog::discover(directory.path()).unwrap();
        let changed = f32_bytes(&[1.0, 2.0, 3.0]);
        serialize_to_file(
            [(
                "weight",
                TensorView::new(Dtype::F32, vec![3], &changed).unwrap(),
            )],
            None,
            &path,
        )
        .unwrap();

        assert!(matches!(
            PreparedCheckpointSource::open_admitted_safetensors(
                admitted.admitted_shards(),
                admitted.tensors().clone(),
                1,
            ),
            Err(StoreError::PreparedCatalogMismatch { key }) if key == "weight"
        ));
    }

    #[cfg(unix)]
    #[test]
    fn prepared_safetensors_rejects_path_replacement_before_first_acquisition() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.safetensors");
        let original = f32_bytes(&[1.0, 2.0]);
        serialize_to_file(
            [(
                "weight",
                TensorView::new(Dtype::F32, vec![2], &original).unwrap(),
            )],
            None,
            &path,
        )
        .unwrap();
        let admitted =
            crate::safetensors::SafetensorsMetadataCatalog::discover(directory.path()).unwrap();
        let prepared = PreparedCheckpointSource::open_admitted_safetensors(
            admitted.admitted_shards(),
            admitted.tensors().clone(),
            1,
        )
        .unwrap();
        assert_eq!(prepared.source_diagnostics().unwrap().physical_reads, 0);

        let replacement = directory.path().join("replacement.safetensors");
        let substituted = f32_bytes(&[9.0, 10.0]);
        serialize_to_file(
            [(
                "weight",
                TensorView::new(Dtype::F32, vec![2], &substituted).unwrap(),
            )],
            None,
            &replacement,
        )
        .unwrap();
        std::fs::rename(&replacement, &path).unwrap();

        assert!(matches!(
            prepared.acquire_lease(TensorReadRequest {
                key: "weight".into(),
                selection: TensorSelection::Full,
                policy: ReadPolicy::RequireBounded,
            }),
            Err(StoreError::AdmittedFileChanged { .. })
        ));
        let diagnostics = prepared.source_diagnostics().unwrap();
        assert_eq!(diagnostics.physical_reads, 0);
        assert_eq!(diagnostics.physical_read_bytes, 0);
        assert!(diagnostics.payload_shard_paths.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn prepared_safetensors_rejects_restored_mtime_overwrite_before_first_acquisition() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.safetensors");
        let original = f32_bytes(&[1.0, 2.0]);
        serialize_to_file(
            [(
                "weight",
                TensorView::new(Dtype::F32, vec![2], &original).unwrap(),
            )],
            None,
            &path,
        )
        .unwrap();
        let admitted =
            crate::safetensors::SafetensorsMetadataCatalog::discover(directory.path()).unwrap();
        let prepared = PreparedCheckpointSource::open_admitted_safetensors(
            admitted.admitted_shards(),
            admitted.tensors().clone(),
            1,
        )
        .unwrap();
        let admitted_metadata = std::fs::metadata(&path).unwrap();
        let admitted_modified = admitted_metadata.modified().unwrap();

        // Ensure even filesystems with a coarser change-time clock observe a
        // distinct overwrite before the attacker restores mtime.
        std::thread::sleep(std::time::Duration::from_millis(10));
        let mut encoded = std::fs::read(&path).unwrap();
        let substituted = f32_bytes(&[9.0, 10.0]);
        let payload_start = encoded.len() - substituted.len();
        encoded[payload_start..].copy_from_slice(&substituted);
        std::fs::write(&path, encoded).unwrap();
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(admitted_modified))
            .unwrap();
        let attacked_metadata = std::fs::metadata(&path).unwrap();
        assert_eq!(attacked_metadata.len(), admitted_metadata.len());
        assert_eq!(attacked_metadata.modified().unwrap(), admitted_modified);

        assert!(matches!(
            prepared.acquire_lease(TensorReadRequest {
                key: "weight".into(),
                selection: TensorSelection::Full,
                policy: ReadPolicy::RequireBounded,
            }),
            Err(StoreError::AdmittedFileChanged { .. })
        ));
        let diagnostics = prepared.source_diagnostics().unwrap();
        assert_eq!(diagnostics.physical_reads, 0);
        assert_eq!(diagnostics.physical_read_bytes, 0);
        assert!(diagnostics.payload_shard_paths.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn acquired_bytes_survive_and_cached_source_rejects_later_in_place_substitution() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.safetensors");
        let original = f32_bytes(&[1.0, 2.0]);
        serialize_to_file(
            [(
                "weight",
                TensorView::new(Dtype::F32, vec![2], &original).unwrap(),
            )],
            None,
            &path,
        )
        .unwrap();
        let admitted =
            crate::safetensors::SafetensorsMetadataCatalog::discover(directory.path()).unwrap();
        let prepared = PreparedCheckpointSource::open_admitted_safetensors(
            admitted.admitted_shards(),
            admitted.tensors().clone(),
            1,
        )
        .unwrap();
        let request = TensorReadRequest {
            key: "weight".into(),
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        };
        let lease = prepared.acquire_lease(request.clone()).unwrap();
        assert_eq!(lease.encoded_bytes().unwrap(), original);
        let admitted_modified = std::fs::metadata(&path).unwrap().modified().unwrap();

        let mut encoded = std::fs::read(&path).unwrap();
        let substituted = f32_bytes(&[9.0, 10.0]);
        let payload_start = encoded.len() - substituted.len();
        encoded[payload_start..].copy_from_slice(&substituted);
        std::fs::write(&path, encoded).unwrap();
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(admitted_modified))
            .unwrap();

        assert_eq!(lease.encoded_bytes().unwrap(), original);
        assert!(matches!(
            prepared.acquire_lease(request),
            Err(StoreError::AdmittedFileChanged { .. })
        ));
        assert_eq!(prepared.source_diagnostics().unwrap().physical_reads, 2);
        assert_eq!(
            prepared.source_diagnostics().unwrap().physical_read_bytes,
            16
        );
    }

    #[test]
    fn safetensors_store_physically_reads_only_selected_noncontiguous_ranges() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.safetensors");
        let selected = f32_bytes(&(0..12).map(|value| value as f32).collect::<Vec<_>>());
        let unrelated = vec![0x5a; 8 * 1024];
        serialize_to_file(
            [
                (
                    "selected",
                    TensorView::new(Dtype::F32, vec![2, 3, 2], &selected).unwrap(),
                ),
                (
                    "unrelated",
                    TensorView::new(Dtype::U8, vec![unrelated.len()], &unrelated).unwrap(),
                ),
            ],
            None,
            &path,
        )
        .unwrap();
        let store = SafetensorsWeightStore::open(&path).unwrap();
        let before = store.diagnostics().unwrap();
        assert_eq!(before.physical_reads, 0);
        assert_eq!(before.physical_read_bytes, 0);
        assert!(before.payload_shard_paths.is_empty());

        let lease = store
            .acquire(TensorReadRequest {
                key: "selected".into(),
                selection: TensorSelection::Range {
                    axis: 1,
                    start: 1,
                    end: 2,
                },
                policy: ReadPolicy::RequireBounded,
            })
            .unwrap();

        let diagnostics = store.diagnostics().unwrap();
        assert_eq!(diagnostics.physical_reads, 4);
        assert_eq!(diagnostics.physical_read_bytes, 32);
        assert_eq!(lease.bounded_read_proof().offset_bytes, 8);
        assert_eq!(lease.bounded_read_proof().length_bytes, 16);
        assert_eq!(lease.bounded_read_proof().physical_reads, 4);
        assert_eq!(lease.bounded_read_proof().physical_read_bytes, 32);
        assert!(lease.bounded_read_proof().physically_bounded);
        let mut expected = selected[8..16].to_vec();
        expected.extend_from_slice(&selected[32..40]);
        assert_eq!(lease.encoded_bytes().unwrap(), expected);
        assert!(std::fs::metadata(&path).unwrap().len() > diagnostics.physical_read_bytes);
    }

    #[test]
    fn one_byte_selection_reads_exactly_that_byte_twice_for_admission() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.safetensors");
        let payload = (0..=255).collect::<Vec<u8>>();
        serialize_to_file(
            [(
                "bytes",
                TensorView::new(Dtype::U8, vec![payload.len()], &payload).unwrap(),
            )],
            None,
            &path,
        )
        .unwrap();
        let store = SafetensorsWeightStore::open(&path).unwrap();
        let lease = store
            .acquire(TensorReadRequest {
                key: "bytes".into(),
                selection: TensorSelection::Contiguous {
                    offset_elements: 137,
                    shape: vec![1],
                },
                policy: ReadPolicy::RequireBounded,
            })
            .unwrap();
        assert_eq!(lease.encoded_bytes().unwrap(), &[137]);
        assert_eq!(lease.bounded_read_proof().length_bytes, 1);
        assert_eq!(lease.bounded_read_proof().physical_reads, 2);
        assert_eq!(lease.bounded_read_proof().physical_read_bytes, 2);
        let diagnostics = store.diagnostics().unwrap();
        assert_eq!(diagnostics.physical_reads, 2);
        assert_eq!(diagnostics.physical_read_bytes, 2);
    }

    #[test]
    fn invalid_selection_reads_no_payload_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.safetensors");
        let payload = [1_u8, 2, 3, 4];
        serialize_to_file(
            [(
                "bytes",
                TensorView::new(Dtype::U8, vec![payload.len()], &payload).unwrap(),
            )],
            None,
            &path,
        )
        .unwrap();
        let store = SafetensorsWeightStore::open(&path).unwrap();
        assert!(store
            .acquire(TensorReadRequest {
                key: "bytes".into(),
                selection: TensorSelection::Range {
                    axis: 0,
                    start: 3,
                    end: 5,
                },
                policy: ReadPolicy::RequireBounded,
            })
            .is_err());
        let diagnostics = store.diagnostics().unwrap();
        assert_eq!(diagnostics.physical_reads, 0);
        assert_eq!(diagnostics.physical_read_bytes, 0);
        assert!(diagnostics.payload_shard_paths.is_empty());
    }

    #[cfg(unix)]
    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn exact_range_admission_rejects_restored_mtime_mutation_between_reads() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.safetensors");
        let payload = [1_u8, 2, 3, 4];
        serialize_to_file(
            [(
                "bytes",
                TensorView::new(Dtype::U8, vec![payload.len()], &payload).unwrap(),
            )],
            None,
            &path,
        )
        .unwrap();
        let admitted = AdmittedFile::open(&path).unwrap();
        let (payload_offset, _) = read_safetensors_metadata(&path, &admitted).unwrap();
        let modified = std::fs::metadata(&path).unwrap().modified().unwrap();
        let telemetry = SafetensorsReadTelemetry::default();
        let result = read_safetensors_ranges_with_hook(
            &path,
            &admitted,
            payload_offset,
            &[1..2],
            &telemetry,
            || {
                let mut bytes = std::fs::read(&path).unwrap();
                bytes[payload_offset + 1] = 9;
                std::fs::write(&path, bytes).unwrap();
                File::options()
                    .write(true)
                    .open(&path)
                    .unwrap()
                    .set_times(std::fs::FileTimes::new().set_modified(modified))
                    .unwrap();
            },
        );
        assert!(matches!(
            result,
            Err(StoreError::AdmittedFileChanged { .. })
        ));
        assert_eq!(telemetry.physical_reads.load(Ordering::Relaxed), 1);
        assert_eq!(telemetry.physical_read_bytes.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn safetensors_unbounded_selection_reports_complete_tensor_read() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.safetensors");
        let selected = f32_bytes(&(0..12).map(|value| value as f32).collect::<Vec<_>>());
        serialize_to_file(
            [(
                "selected",
                TensorView::new(Dtype::F32, vec![2, 3, 2], &selected).unwrap(),
            )],
            None,
            &path,
        )
        .unwrap();
        let store = SafetensorsWeightStore::open(&path).unwrap();

        let lease = store
            .acquire(TensorReadRequest {
                key: "selected".into(),
                selection: TensorSelection::Range {
                    axis: 1,
                    start: 1,
                    end: 2,
                },
                policy: ReadPolicy::AllowFullTensorRead,
            })
            .unwrap();

        let diagnostics = store.diagnostics().unwrap();
        assert_eq!(diagnostics.physical_reads, 2);
        assert_eq!(diagnostics.physical_read_bytes, 96);
        assert!(!lease.bounded_read_proof().physically_bounded);
        assert_eq!(lease.bounded_read_proof().offset_bytes, 0);
        assert_eq!(lease.bounded_read_proof().length_bytes, 48);
        assert_eq!(lease.bounded_read_proof().physical_reads, 2);
        assert_eq!(lease.bounded_read_proof().physical_read_bytes, 96);
        assert_eq!(lease.encoded_bytes().unwrap(), selected);

        let bounded = store
            .acquire(TensorReadRequest {
                key: "selected".into(),
                selection: TensorSelection::Range {
                    axis: 1,
                    start: 1,
                    end: 2,
                },
                policy: ReadPolicy::RequireBounded,
            })
            .unwrap();
        let cached_diagnostics = store.diagnostics().unwrap();
        assert_eq!(cached_diagnostics.physical_reads, 2);
        assert_eq!(cached_diagnostics.physical_read_bytes, 96);
        let mut expected = selected[8..16].to_vec();
        expected.extend_from_slice(&selected[32..40]);
        assert_eq!(bounded.encoded_bytes().unwrap(), expected);
        assert!(bounded.bounded_read_proof().physically_bounded);
        assert_eq!(bounded.bounded_read_proof().length_bytes, 16);
        assert_eq!(bounded.bounded_read_proof().physical_reads, 0);
        assert_eq!(bounded.bounded_read_proof().physical_read_bytes, 0);
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

        let admitted = SafetensorsShards::discover(directory.path()).unwrap();
        std::fs::remove_file(directory.path().join("model.safetensors.index.json")).unwrap();
        let store = SafetensorsWeightStore::open_admitted(admitted, 1).unwrap();
        let first = first.canonicalize().unwrap();
        store.metadata("left").unwrap();
        let metadata_diagnostics = store.diagnostics().unwrap();
        assert_eq!(
            metadata_diagnostics.touched_shard_paths,
            std::slice::from_ref(&first)
        );
        assert!(metadata_diagnostics.payload_shard_paths.is_empty());
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
        assert_eq!(lease.encoded_bytes().unwrap(), &left[8..]);
        assert_eq!(lease.bounded_read_proof().length_bytes, 8);
        let diagnostics = store.diagnostics().unwrap();
        assert_eq!(diagnostics.payload_shard_paths, [first]);
        assert_eq!(diagnostics.physical_reads, 2);
        assert_eq!(diagnostics.physical_read_bytes, 16);
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
    fn indexed_store_defers_validation_of_unrequested_shards() {
        let directory = tempfile::tempdir().unwrap();
        let local = directory.path().join("local.safetensors");
        let remote = directory.path().join("remote.safetensors");
        serialize_to_file(
            [(
                "local",
                TensorView::new(Dtype::F32, vec![1], &f32_bytes(&[1.0])).unwrap(),
            )],
            None,
            &local,
        )
        .unwrap();
        std::fs::write(&remote, b"not safetensors").unwrap();
        std::fs::write(
            directory.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "weight_map": {
                    "local": "local.safetensors",
                    "remote": "remote.safetensors"
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let store = SafetensorsWeightStore::open(directory.path()).unwrap();
        assert_eq!(store.keys(), ["local", "remote"]);
        assert_eq!(store.metadata("local").unwrap().logical_shape, [1]);
        assert_eq!(
            store.diagnostics().unwrap().touched_shard_paths,
            [local.canonicalize().unwrap()]
        );
        assert!(matches!(
            store.metadata("remote"),
            Err(StoreError::MalformedSafetensors { .. })
        ));
    }

    #[test]
    fn indexed_store_exactly_validates_every_opened_shard() {
        let missing = tempfile::tempdir().unwrap();
        let missing_shard = missing.path().join("payload.safetensors");
        serialize_to_file(
            [(
                "requested",
                TensorView::new(Dtype::F32, vec![1], &f32_bytes(&[1.0])).unwrap(),
            )],
            None,
            &missing_shard,
        )
        .unwrap();
        std::fs::write(
            missing.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "weight_map": {
                    "requested": "payload.safetensors",
                    "missing_sibling": "payload.safetensors"
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let store = SafetensorsWeightStore::open(missing.path()).unwrap();
        assert!(matches!(
            store.metadata("requested"),
            Err(StoreError::ContradictoryIndexMapping { key, .. })
                if key == "missing_sibling"
        ));

        let extra = tempfile::tempdir().unwrap();
        let extra_shard = extra.path().join("payload.safetensors");
        let requested = f32_bytes(&[1.0]);
        let unindexed = f32_bytes(&[2.0]);
        serialize_to_file(
            [
                (
                    "requested",
                    TensorView::new(Dtype::F32, vec![1], &requested).unwrap(),
                ),
                (
                    "unindexed",
                    TensorView::new(Dtype::F32, vec![1], &unindexed).unwrap(),
                ),
            ],
            None,
            &extra_shard,
        )
        .unwrap();
        std::fs::write(
            extra.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "weight_map": {"requested": "payload.safetensors"}
            }))
            .unwrap(),
        )
        .unwrap();

        let store = SafetensorsWeightStore::open(extra.path()).unwrap();
        assert!(matches!(
            store.metadata("requested"),
            Err(StoreError::UnindexedShardTensor { key, .. }) if key == "unindexed"
        ));
    }

    #[cfg(unix)]
    #[test]
    fn opening_rejects_symlinks_outside_the_checkpoint_root() {
        use std::os::unix::fs::symlink;

        let parent = tempfile::tempdir().unwrap();
        let checkpoint = parent.path().join("checkpoint");
        std::fs::create_dir(&checkpoint).unwrap();
        let outside = parent.path().join("outside.safetensors");
        serialize_to_file(
            [(
                "weight",
                TensorView::new(Dtype::F32, vec![1], &f32_bytes(&[1.0])).unwrap(),
            )],
            None,
            &outside,
        )
        .unwrap();
        symlink(&outside, checkpoint.join("model-00001.safetensors")).unwrap();
        std::fs::write(
            checkpoint.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "weight_map": {"weight": "model-00001.safetensors"}
            }))
            .unwrap(),
        )
        .unwrap();

        assert!(matches!(
            SafetensorsWeightStore::open(&checkpoint),
            Err(StoreError::SafetensorsShards(
                crate::safetensors::SafetensorsShardError::UnsafeShardPath { .. }
            ))
        ));
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
        let plan = SafetensorsCheckpointPlan::new(
            "test architecture",
            vec![SafetensorsTensorConstraint::required(
                "selected",
                vec![2],
                StoredDtypeConstraint::Exact(StoredDtype::F32),
            )],
            Vec::new(),
            CatalogPolicy::non_strict(),
        )
        .unwrap();
        let contract = resolve_safetensors_plan(source.as_ref(), &plan).unwrap();
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

    #[test]
    fn composite_source_routes_disjoint_leases_and_rejects_collisions() {
        let left: SharedCheckpointSource = Arc::new(
            MemoryWeightStore::from_safetensors([(
                "text.weight".into(),
                Dtype::F32,
                vec![1],
                f32_bytes(&[1.0]),
            )])
            .unwrap(),
        );
        let right: SharedCheckpointSource = Arc::new(
            MemoryWeightStore::from_safetensors([(
                "vision.weight".into(),
                Dtype::F32,
                vec![1],
                f32_bytes(&[2.0]),
            )])
            .unwrap(),
        );
        let source = CompositeCheckpointSource::new([left, right]).unwrap();
        assert_eq!(source.source_keys(), ["text.weight", "vision.weight"]);
        let lease = source
            .acquire_lease(TensorReadRequest {
                key: "vision.weight".into(),
                selection: TensorSelection::Full,
                policy: ReadPolicy::RequireBounded,
            })
            .unwrap();
        assert_eq!(lease.encoded_bytes().unwrap(), f32_bytes(&[2.0]));

        let first: SharedCheckpointSource = Arc::new(
            MemoryWeightStore::from_safetensors([(
                "collision".into(),
                Dtype::F32,
                vec![1],
                f32_bytes(&[1.0]),
            )])
            .unwrap(),
        );
        let second: SharedCheckpointSource = Arc::new(
            MemoryWeightStore::from_safetensors([(
                "collision".into(),
                Dtype::F32,
                vec![1],
                f32_bytes(&[2.0]),
            )])
            .unwrap(),
        );
        assert!(CompositeCheckpointSource::new([first, second]).is_err());
    }

    #[test]
    fn prepared_source_rejects_a_lease_that_differs_from_the_admitted_catalog() {
        struct LeaseSwapSource {
            prepared: TensorMetadata,
            payload: MemoryWeightStore,
        }

        impl CheckpointSource for LeaseSwapSource {
            fn source_keys(&self) -> Vec<String> {
                vec!["weight".into()]
            }

            fn source_metadata(&self, _: &str) -> Result<TensorMetadata, StoreError> {
                Ok(self.prepared.clone())
            }

            fn acquire_lease(
                &self,
                request: TensorReadRequest,
            ) -> Result<CheckpointLease, StoreError> {
                self.payload.acquire_lease(request)
            }

            fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
                self.payload.source_diagnostics()
            }
        }

        let prepared = TensorMetadata {
            name: "weight".into(),
            logical_shape: vec![1],
            physical_shape: vec![1],
            stored_dtype: StoredDtype::F32,
            encoded_byte_len: 4,
            backing_shard: None,
        };
        let source: SharedCheckpointSource = Arc::new(LeaseSwapSource {
            prepared: prepared.clone(),
            payload: MemoryWeightStore::from_safetensors([(
                "weight".into(),
                Dtype::I32,
                vec![1],
                7_i32.to_le_bytes().to_vec(),
            )])
            .unwrap(),
        });
        let provenance = source.source_provenance("weight").unwrap();
        let source = PreparedCheckpointSource::new(
            source,
            BTreeMap::from([(
                "weight".into(),
                PreparedTensorSource {
                    metadata: prepared,
                    provenance,
                },
            )]),
        )
        .unwrap();

        assert!(matches!(
            source.acquire_lease(TensorReadRequest {
                key: "weight".into(),
                selection: TensorSelection::Full,
                policy: ReadPolicy::RequireBounded,
            }),
            Err(StoreError::PreparedCatalogMismatch { key }) if key == "weight"
        ));
    }
}

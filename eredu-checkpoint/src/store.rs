//! Backend-neutral checkpoint storage and encoded tensor leases.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{Read, Seek, SeekFrom},
    ops::Range,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, MutexGuard, OnceLock, Weak,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::{
    StoredDtype,
    recipe::RecipeInferenceCache,
    safetensors::{
        MAX_HEADER_BYTES, SafetensorsDiscoveryLimits, SafetensorsHeaderAdmission,
        SafetensorsHeaderFailure, SafetensorsHeaderRequest, SafetensorsHeaderReservation,
        SafetensorsSourceAdmission,
        SafetensorsShards,
    },
};
use safetensors::tensor::{Dtype, Metadata};

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
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
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

mod encoded_owner;
pub use encoded_owner::PinnedEncodedBytes;

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

#[derive(Debug)]
struct MemoryTensor {
    metadata: TensorMetadata,
    dtype: Dtype,
    bytes: Vec<u8>,
}

/// Encoded tensor selection retaining immutable in-memory storage.
#[derive(Debug, Clone)]
pub struct MemoryLease {
    tensor: storage::SourceHandle<MemoryTensor>,
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
        encoded_owner::memory_bytes(&self.tensor, &self.selected_bytes, &self.span)
    }
}

/// Immutable in-memory SafeTensors-compatible encoded tensors.
#[derive(Debug, Default)]
pub struct MemoryWeightStore {
    recipes: RecipeInferenceCache,
    tensors: BTreeMap<String, storage::SourceHandle<MemoryTensor>>,
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
            let tensor = storage::SourceHandle::new(
                MemoryTensor {
                    metadata,
                    dtype,
                    bytes,
                },
                None,
            );
            if catalog.insert(name.clone(), tensor).is_some() {
                return Err(StoreError::Internal(format!(
                    "duplicate in-memory tensor {name:?}"
                )));
            }
        }
        Ok(Self {
            tensors: catalog,
            recipes: RecipeInferenceCache::default(),
        })
    }
}

impl WeightStore for MemoryWeightStore {
    fn recipe_cache(&self) -> Option<&RecipeInferenceCache> {
        Some(&self.recipes)
    }

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
    fn prepare_encoded_read(
        &self,
        keys: &[String],
    ) -> Result<Option<EncodedReadBatch>, StoreError> {
        bulk::prepare_memory(self, keys).map(Some)
    }

    fn prepared_acquisition_owner(self: Arc<Self>) -> Option<PreparedAcquisitionOwner> {
        Some(acquisition::retained_route::owner_memory(self))
    }

    fn prepared_acquisition_source(&self) -> PreparedAcquisitionSource<'_> {
        PreparedAcquisitionSource(acquisition::Route::Memory(self))
    }

    fn source_lease_controls<'a>(
        &'a self,
        key: &'a str,
    ) -> Result<SourceLeaseControls<'a>, LeaseControlBorrowError<'a>> {
        let metadata = self.source_metadata_borrowed(key)?;
        Ok(SourceLeaseControls::new(
            LeaseProvider::Memory,
            metadata,
            key,
        ))
    }

    fn source_metadata_borrowed(&self, key: &str) -> SourceMetadataLoan<'_> {
        self.tensors
            .get(key)
            .map(|tensor| &tensor.metadata)
            .ok_or(SourceMetadataBorrowError::UnknownTensor)
    }
    fn source_key_authority_borrowed(
        &self,
        _: &str,
    ) -> Result<SourceKeyAuthority, SourceMetadataBorrowError<'_>> {
        Ok(SourceKeyAuthority::Ordinary)
    }

    fn source_storage_slot_bound(&self) -> Result<Option<usize>, StoreError> {
        Ok(Some(self.tensors.len()))
    }

    fn visit_source_storage(
        &self,
        visitor: &mut dyn FnMut(SourceStorageRef<'_>),
    ) -> Result<bool, StoreError> {
        for tensor in self.tensors.values() {
            let bytes =
                u64::try_from(tensor.bytes.capacity()).map_err(|_| StoreError::Overflow {
                    context: "in-memory checkpoint payload capacity".into(),
                })?;
            visitor(tensor.storage_ref(bytes));
        }
        Ok(true)
    }

    fn source_storage(&self) -> Result<Option<SourceStorage>, StoreError> {
        let mut storage = SourceStorage::default();
        for tensor in self.tensors.values() {
            let bytes =
                u64::try_from(tensor.bytes.capacity()).map_err(|_| StoreError::Overflow {
                    context: "in-memory checkpoint payload capacity".into(),
                })?;
            storage.include_ref(tensor.storage_ref(bytes))?;
        }
        storage.bytes()?;
        Ok(Some(storage))
    }

    fn recipe_cache(&self) -> Option<&RecipeInferenceCache> {
        Some(&self.recipes)
    }

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
    /// Cold owning loan for the closed prepared GGUF route. Only concrete
    /// built-in implementations can construct an owner; forwarding a different
    /// owner cannot certify this root. The default leaves custom roots unknown.
    /// Selection retains this owner so accepted acquisition calls no virtual
    /// source methods. This loan grants no storage or submission authority.
    fn prepared_acquisition_owner(self: Arc<Self>) -> Option<PreparedAcquisitionOwner> {
        None
    }

    /// Lends this source's concrete acquisition route for final prepared storage.
    /// The default has no prepared destination and never invokes ordinary
    /// acquisition. Forwarding implementations must preserve their own complete
    /// authorization and validation route; this loan grants no budget authority.
    fn prepared_acquisition_source(&self) -> PreparedAcquisitionSource<'_> {
        PreparedAcquisitionSource::unavailable()
    }

    /// Borrows the concrete selected provider and actual request-forwarding
    /// population. Does not acquire a lease or fall back to owned metadata.
    /// Custom sources must explicitly forward their exact selected source;
    /// this description grants no complete storage/admission authority.
    fn source_lease_controls<'a>(
        &'a self,
        _key: &'a str,
    ) -> Result<SourceLeaseControls<'a>, LeaseControlBorrowError<'a>> {
        Err(SourceMetadataBorrowError::Unavailable.into())
    }

    /// Borrows the actual retained catalog entry without preparing headers,
    /// reading payloads, cloning metadata or invoking the ordinary fallback.
    /// Custom sources may implement this diagnostic; it grants no source or
    /// request custody. Authorization views preserve their actual selection.
    fn source_metadata_borrowed(&self, _key: &str) -> SourceMetadataLoan<'_> {
        Err(SourceMetadataBorrowError::Unavailable)
    }

    /// Allocation-free companion to materialized overlay membership. Unknown
    /// implementations remain unavailable instead of calling the ordinary API.
    fn source_key_authority_borrowed(
        &self,
        _key: &str,
    ) -> Result<SourceKeyAuthority, SourceMetadataBorrowError<'_>> {
        Err(SourceMetadataBorrowError::Unavailable)
    }

    /// Stable maximum visits from `visit_source_storage` for this exact source
    /// instance. Duplicate physical branches and zero-byte owners count. The
    /// bound includes every source-owned payload that may appear during ordinary
    /// use; mutable implementations without such a contract return None.
    /// No source visit, materialization, owner clone or accounting is performed.
    /// Callers must retain/bind this exact source rather than reuse the scalar
    /// for a replacement source; this diagnostic grants no storage authority.
    fn source_storage_slot_bound(&self) -> Result<Option<usize>, StoreError> {
        Ok(None)
    }

    /// Visits every physical source owner without building a storage map,
    /// reading payloads, or reopening files. `true` means complete visitation;
    /// `false` means unknown/incomplete even if no owner was visited. The default
    /// does not call the allocating legacy [`Self::source_storage`] method.
    ///
    /// Authorization views include hidden physical owners. Aliases may repeat:
    /// callers must deduplicate and check capacities/overflow, and preprice their
    /// own slots and pins. A caller may retain each borrowed owner immediately;
    /// later errors/unwind leave that caller-owned prefix intact. This contract
    /// covers source payload/reader storage only, not catalog/recipe metadata,
    /// external leases, operating-system caches or conversion workspace.
    fn visit_source_storage(
        &self,
        _visitor: &mut dyn FnMut(SourceStorageRef<'_>),
    ) -> Result<bool, StoreError> {
        Ok(false)
    }

    /// Complete source-owned host payload bound, or explicitly unknown.
    /// Inspect retained metadata only: do not read, convert or allocate payloads.
    /// Forwarding views must include their complete physical source even when
    /// they expose fewer tensor keys. Transient/external leases are separately
    /// charged by their operation owner; see [`SourceStorage`].
    fn source_storage(&self) -> Result<Option<SourceStorage>, StoreError> {
        Ok(None)
    }

    /// Inference cache bound to this immutable catalog and authorization view.
    /// Returning a cache promises metadata and provenance remain fixed, and
    /// every returned lease/read batch is checked against that same catalog.
    fn recipe_cache(&self) -> Option<&RecipeInferenceCache> {
        None
    }

    /// Returns all logical catalog keys in deterministic order.
    fn source_keys(&self) -> Vec<String>;
    /// Returns metadata without reading tensor payloads.
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError>;
    /// Acquires a format-preserving encoded lease.
    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError>;
    /// Returns deterministic storage diagnostics.
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError>;

    /// Prepares an ordered, bounded read of complete encoded tensors into
    /// caller-owned storage. Unsupported sources return `None` without reading
    /// payloads. The returned batch retains exact file admission or the existing
    /// immutable memory owner and its metadata. Execution validates file versions
    /// around reads; memory ranges copy directly into the supplied destination.
    fn prepare_encoded_read(
        &self,
        _keys: &[String],
    ) -> Result<Option<EncodedReadBatch>, StoreError> {
        Ok(None)
    }

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

mod retained_source;
pub use retained_source::{
    CheckpointSourceIdentity, RetainedCheckpointSource, SourceErasureStorageRequest,
};

pub(crate) mod acquisition;
mod memory_destination;
mod memory_buffer;
mod materialized;
pub use materialized::MaterializedCheckpointSource;
pub use memory_buffer::{MemoryTensorBuffer, MemoryTensorBufferError, MemoryWeightStoreBuildError};
pub use acquisition::{
    PreparedAcquisitionBank, PreparedAcquisitionBankError, PreparedAcquisitionFailure,
    PreparedAcquisitionOwner, PreparedAcquisitionRefusal, PreparedAcquisitionSource,
    PreparedAcquisitionStorage, PreparedCheckpointAcquisition, SelectedGgufAcquisitionStorage,
    SelectedGgufConversionPlan,
};
mod cache_policy;
mod lease_destination;
pub use lease_destination::{
    PreparedSafetensorsLease, PreparedSafetensorsLeaseFailure, SafetensorsLeaseSource,
};
mod read_bytes;
pub use read_bytes::{
    SafetensorsByteError, SafetensorsByteFailure, SafetensorsBytePlan, SafetensorsBytes,
};

mod read_plan;
pub(crate) use read_plan::encoded_selection_plan;
pub use read_plan::{
    SafetensorsReadDestinationPlan, SafetensorsReadError, SafetensorsReadRanges,
    SafetensorsReadSource, SelectionReadDestinationPlan, SelectionReadRanges,
};

mod lease_controls;
pub use lease_controls::{
    GgufLeaseCloneLayout, LeaseCloneLayout, LeaseControlBorrowError, LeaseProvider,
    RequestCloneLayout, SourceLeaseControls,
};
mod selection_validation;
pub use selection_validation::{SelectionValidationError, SelectionValidationPlan};

mod metadata_borrow;
pub use metadata_borrow::{
    MetadataCloneLayout, SelectedMetadataCloneLayout, SelectionCloneLayout, SourceKeyAuthority,
    SourceMetadataBorrowError, SourceMetadataLoan,
};

mod storage;
pub(crate) use storage::{SourceControl, SourceHandle};
pub use storage::{SourceStorage, SourceStorageIdentity, SourceStorageOwner, SourceStorageRef};

mod bulk;
pub(crate) use bulk::EncodedRange;
pub use bulk::{
    DetachedEncodedReadPlan, DetachedEncodedReadSlice, DetachedEncodedReads,
    DetachedReadBuildCause, DetachedReadBuildError, DetachedReadFailure, EncodedReadBatch,
    EncodedReadFailure, EncodedReadFailureCause, EncodedReadLayout,
    EncodedProjectionError, EncodedProjectionBuildError, EncodedReadProjectionPlan,
    MemoryEncodedReadBuildError, MemoryEncodedReadPlan, MemoryEncodedReadPlanError, MemoryEncodedReadRouteError,
    PreparedEncodedRead, SafetensorsEncodedReadPlan, SafetensorsEncodedReadPlanError, SafetensorsEncodedReadPlanErrorKind,
    SafetensorsEncodedReadBuildCause, SafetensorsEncodedReadBuildError,
};

/// Opens one exact admitted SafeTensors source and applies its retained resolution.
///
/// This is the singular backend-neutral source constructor shared by ordinary
/// and realtime architecture preparation. It never rediscovers an artifact and
/// does not acquire tensor payloads.
pub fn open_prepared_safetensors_source(
    shards: SafetensorsShards,
    catalog: BTreeMap<String, TensorMetadata>,
    resolution: crate::validation::ResolvedCheckpointPlan,
    max_cached_shards: usize,
) -> Result<SharedCheckpointSource, StoreError> {
    let prepared =
        PreparedCheckpointSource::open_admitted_safetensors(shards, catalog, max_cached_shards)?;
    Ok(Arc::new(ResolvedCheckpointSource::new(
        Arc::new(prepared),
        resolution,
    )))
}

/// Opens one admitted SafeTensors source and proves its retained schema resolution still holds.
///
/// Header and provenance validation happen before the resolved view is published. Payload bytes
/// remain lazy behind later leases.
pub fn open_validated_safetensors_source(
    shards: SafetensorsShards,
    catalog: BTreeMap<String, TensorMetadata>,
    checkpoint_plan: &crate::schema::SafetensorsCheckpointPlan,
    admitted_resolution: crate::validation::ResolvedCheckpointPlan,
    max_cached_shards: usize,
) -> Result<SharedCheckpointSource, StoreError> {
    let prepared =
        PreparedCheckpointSource::open_admitted_safetensors(shards, catalog, max_cached_shards)?;
    let current = crate::validation::resolve_safetensors_plan(
        &prepared as &dyn CheckpointSource,
        checkpoint_plan,
    )
    .map_err(|validation| {
        StoreError::Internal(format!(
            "admitted SafeTensors checkpoint contract no longer resolves: {validation:?}"
        ))
    })?;
    if current != admitted_resolution {
        return Err(StoreError::Internal(
            "admitted SafeTensors checkpoint resolution changed during source preparation".into(),
        ));
    }
    Ok(Arc::new(ResolvedCheckpointSource::new(
        Arc::new(prepared),
        current,
    )))
}

/// Immutable catalog entry retained across deferred payload acquisition.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PreparedTensorSource {
    /// Exact metadata admitted before payload access.
    pub metadata: TensorMetadata,
    /// Exact container identity and encoding admitted before payload access.
    pub provenance: TensorSourceProvenance,
}

impl PreparedTensorSource {
    // Encoded batches describe full scalar tensors. Compare the generated
    // provenance by borrowing its fields instead of cloning strings or paths.
    fn matches_encoded_metadata(&self, metadata: &TensorMetadata) -> bool {
        let provenance = &self.provenance;
        self.metadata == *metadata
            && provenance.catalog_key == metadata.name
            && provenance.physical_tensor == metadata.name
            && provenance.output == metadata.name
            && provenance.backing_shard == metadata.backing_shard
            && matches!(&provenance.source_encoding,
                crate::SourceTensorEncoding::Safetensors(dtype) if dtype == &metadata.stored_dtype)
    }
}

/// Checkpoint source pinned to an exact metadata and provenance snapshot.
///
/// The wrapper admits metadata and provenance once, then validates each
/// resulting lease before returning it. This closes the interval between
/// header-only preparation and deferred materialization without converting or
/// buffering payloads.
pub struct PreparedCheckpointSource {
    recipes: RecipeInferenceCache,
    source: RetainedCheckpointSource,
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
        let source = RetainedCheckpointSource::from_safetensors(
            SafetensorsWeightStore::open_admitted(shards, max_cached_shards)?,
        );
        Self::pin_safetensors(source, catalog)
    }

    /// Pins the exact typed SafeTensors source to inspected metadata. This shares
    /// ordinary preparation's comparison and provenance construction, without
    /// reopening a source or acquiring payloads. The caller retains input and
    /// admission custody separately if it needs them preserved on failure.
    pub fn pin_safetensors(
        source: RetainedCheckpointSource,
        catalog: BTreeMap<String, TensorMetadata>,
    ) -> Result<Self, StoreError> {
        let store = source.safetensors().ok_or_else(|| StoreError::PreparedCatalogMismatch {
            key: "<source>".into(),
        })?;
        if store.catalog.len() != catalog.len() {
            return Err(StoreError::PreparedCatalogMismatch {
                key: "<catalog>".into(),
            });
        }
        for (key, metadata) in &catalog {
            if store.prepare_metadata(key)? != metadata {
                return Err(StoreError::PreparedCatalogMismatch { key: key.clone() });
            }
        }
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
        Ok(Self {
            source: source.into(),
            catalog,
            recipes: RecipeInferenceCache::default(),
        })
    }

    /// Pins a source to the supplied exact catalog.
    pub fn new(
        source: impl Into<RetainedCheckpointSource>,
        catalog: BTreeMap<String, PreparedTensorSource>,
    ) -> Result<Self, StoreError> {
        let prepared = Self {
            source: source.into(),
            catalog,
            recipes: RecipeInferenceCache::default(),
        };
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

    fn validate_acquired(
        &self,
        request: &TensorReadRequest,
        lease: &CheckpointLease,
    ) -> Result<(), StoreError> {
        if self.source.recipe_cache().is_none() {
            self.validate_lease(request, lease)?;
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
        Ok(())
    }
}

impl CheckpointSource for PreparedCheckpointSource {
    fn prepared_acquisition_owner(self: Arc<Self>) -> Option<PreparedAcquisitionOwner> {
        Some(acquisition::retained_route::owner_prepared(self))
    }

    fn prepared_acquisition_source(&self) -> PreparedAcquisitionSource<'_> {
        PreparedAcquisitionSource(acquisition::Route::Prepared(self))
    }

    fn source_lease_controls<'a>(
        &'a self,
        key: &'a str,
    ) -> Result<SourceLeaseControls<'a>, LeaseControlBorrowError<'a>> {
        if !self.catalog.contains_key(key) {
            return Err(SourceMetadataBorrowError::UnknownTensor.into());
        }
        self.source.source_lease_controls(key)?.through_prepared()
    }

    fn source_metadata_borrowed(&self, key: &str) -> SourceMetadataLoan<'_> {
        self.catalog
            .get(key)
            .map(|entry| &entry.metadata)
            .ok_or(SourceMetadataBorrowError::UnknownTensor)
    }
    fn source_key_authority_borrowed(
        &self,
        key: &str,
    ) -> Result<SourceKeyAuthority, SourceMetadataBorrowError<'_>> {
        self.source.source_key_authority_borrowed(key)
    }

    fn source_storage_slot_bound(&self) -> Result<Option<usize>, StoreError> {
        self.source.source_storage_slot_bound()
    }

    fn visit_source_storage(
        &self,
        visitor: &mut dyn FnMut(SourceStorageRef<'_>),
    ) -> Result<bool, StoreError> {
        self.source.visit_source_storage(visitor)
    }

    fn source_storage(&self) -> Result<Option<SourceStorage>, StoreError> {
        self.source.source_storage()
    }

    fn recipe_cache(&self) -> Option<&RecipeInferenceCache> {
        Some(self.source.recipe_cache().unwrap_or(&self.recipes))
    }

    fn prepare_encoded_read(
        &self,
        keys: &[String],
    ) -> Result<Option<EncodedReadBatch>, StoreError> {
        // This child's fixed catalog was compared once at construction. Its
        // read contract already binds every batch to that same catalog.
        if self.source.recipe_cache().is_some() {
            return self.source.prepare_encoded_read(keys);
        }
        for key in keys {
            self.expected(key)?;
        }
        let Some(batch) = self.source.prepare_encoded_read(keys)? else {
            return Ok(None);
        };
        for metadata in batch.tensors() {
            let expected = self.expected(&metadata.name)?;
            if !expected.matches_encoded_metadata(metadata) {
                return Err(StoreError::PreparedCatalogMismatch {
                    key: metadata.name.clone(),
                });
            }
        }
        Ok(Some(batch))
    }

    fn source_keys(&self) -> Vec<String> {
        self.catalog.keys().cloned().collect()
    }

    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        Ok(self.expected(key)?.metadata.clone())
    }

    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        self.expected(&request.key)?;
        let lease = self.source.acquire_lease(request.clone())?;
        self.validate_acquired(&request, &lease)?;
        Ok(lease)
    }

    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        self.source.source_diagnostics()
    }

    fn source_provenance(&self, key: &str) -> Result<TensorSourceProvenance, StoreError> {
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

mod composite;
pub use composite::{
    GgufCompositeBuildFailure, GgufCompositeInputError, GgufCompositePlan,
    GgufCompositeStorageRequest,
};

/// Disjoint logical union of independently opened checkpoint artifacts.
///
/// This is used by split model/projector artifacts while preserving each
/// source's native leases, bounded-read guarantees, and physical diagnostics.
pub struct CompositeCheckpointSource {
    recipes: RecipeInferenceCache,
    sources: Vec<RetainedCheckpointSource>,
    owners: composite::OwnerRows,
    // Nested Vec/key/node/PAL storage retires before its constructor account.
    // The composite root and any ordinary child erasure remain separate owners.
    control: Option<storage::SourceControl>,
}

impl CompositeCheckpointSource {
    /// Borrow this exact built-in constructor custody without extracting it.
    /// Ordinary composites have none; a runtime recognizes only its private type.
    pub fn constructor_control_owner<C: std::any::Any>(&self) -> Option<&C> {
        self.control.as_ref()?.origin()
    }

    /// Creates a deterministic union and rejects ambiguous logical keys.
    pub fn new(
        sources: impl IntoIterator<Item = impl Into<RetainedCheckpointSource>>,
    ) -> Result<Self, StoreError> {
        let sources = sources
            .into_iter()
            .map(Into::into)
            .collect::<Vec<RetainedCheckpointSource>>();
        if sources.is_empty() {
            return Err(StoreError::Internal(
                "composite checkpoint source requires at least one artifact".into(),
            ));
        }
        let mut owners = composite::OwnerRows::default();
        for (owner, source) in sources.iter().enumerate() {
            for key in source.source_keys() {
                if let Some(previous) = owners.insert(key.clone(), owner) {
                    return Err(StoreError::Internal(format!(
                        "composite checkpoint key {key:?} is owned by sources {previous} and {owner}"
                    )));
                }
            }
        }
        Ok(Self {
            sources,
            owners,
            recipes: RecipeInferenceCache::default(),
            control: None,
        })
    }

    fn source_owner_for(&self, key: &str) -> Result<&RetainedCheckpointSource, StoreError> {
        self.owners
            .get(key)
            .and_then(|owner| self.sources.get(*owner))
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
    }
    fn source_for(&self, key: &str) -> Result<&dyn CheckpointSource, StoreError> {
        self.source_owner_for(key)
            .map(|source| source.as_ref() as &dyn CheckpointSource)
    }

    // Batch selection preserves the ordinary first-error/unsupported order.
    // An empty batch or the first change of child has no single encoded owner.
    fn encoded_read_owner(&self, keys: &[String]) -> Result<Option<&RetainedCheckpointSource>, usize> {
        let mut owner = None;
        for (index, key) in keys.iter().enumerate() {
            let current = *self.owners.get(key).ok_or(index)?;
            if owner.is_some_and(|previous| previous != current) {
                return Ok(None);
            }
            owner = Some(current);
        }
        Ok(owner.map(|index| &self.sources[index]))
    }
}

impl CheckpointSource for CompositeCheckpointSource {
    fn prepared_acquisition_owner(self: Arc<Self>) -> Option<PreparedAcquisitionOwner> {
        Some(acquisition::retained_route::owner_composite(self))
    }

    fn prepared_acquisition_source(&self) -> PreparedAcquisitionSource<'_> {
        PreparedAcquisitionSource(acquisition::Route::Composite(self))
    }

    fn source_lease_controls<'a>(
        &'a self,
        key: &'a str,
    ) -> Result<SourceLeaseControls<'a>, LeaseControlBorrowError<'a>> {
        let owner = *self
            .owners
            .get(key)
            .ok_or(SourceMetadataBorrowError::UnknownTensor)?;
        self.sources[owner].source_lease_controls(key)
    }

    fn source_metadata_borrowed(&self, key: &str) -> SourceMetadataLoan<'_> {
        let owner = *self
            .owners
            .get(key)
            .ok_or(SourceMetadataBorrowError::UnknownTensor)?;
        self.sources[owner].source_metadata_borrowed(key)
    }
    fn source_key_authority_borrowed(
        &self,
        key: &str,
    ) -> Result<SourceKeyAuthority, SourceMetadataBorrowError<'_>> {
        match self.owners.get(key) {
            Some(owner) => self.sources[*owner].source_key_authority_borrowed(key),
            None => Ok(SourceKeyAuthority::Ordinary),
        }
    }

    fn source_storage_slot_bound(&self) -> Result<Option<usize>, StoreError> {
        let mut count = 0usize;
        let mut complete = true;
        for source in &self.sources {
            match source.source_storage_slot_bound()? {
                Some(n) => {
                    count = count.checked_add(n).ok_or_else(|| StoreError::Overflow {
                        context: "composite source owner slots".into(),
                    })?
                }
                None => complete = false,
            }
        }
        Ok(complete.then_some(count))
    }

    fn visit_source_storage(
        &self,
        visitor: &mut dyn FnMut(SourceStorageRef<'_>),
    ) -> Result<bool, StoreError> {
        let mut complete = true;
        for source in &self.sources {
            // Inspect all physical sources, including hidden/repeated owners.
            // An incomplete child cannot short-circuit later known owners.
            complete &= source.visit_source_storage(visitor)?;
        }
        Ok(complete)
    }

    fn source_storage(&self) -> Result<Option<SourceStorage>, StoreError> {
        SourceStorage::collect(
            self.sources
                .iter()
                .map(|source| source.as_ref() as &dyn CheckpointSource),
        )
    }

    fn recipe_cache(&self) -> Option<&RecipeInferenceCache> {
        self.sources
            .iter()
            .all(|source| source.recipe_cache().is_some())
            .then_some(&self.recipes)
    }

    fn prepare_encoded_read(
        &self,
        keys: &[String],
    ) -> Result<Option<EncodedReadBatch>, StoreError> {
        match self.encoded_read_owner(keys).map_err(|index| StoreError::UnknownTensor {
            key: keys[index].clone(),
        })? {
            Some(source) => source.prepare_encoded_read(keys),
            None => Ok(None),
        }
    }

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

/// A named logical view that restricts the visible checkpoint catalog.
///
/// The view shares the underlying source, caches, leases, and diagnostics. It
/// changes only catalog authorization, so creating the view performs no
/// payload reads and acquiring an authorized lease preserves the source's
/// exact provenance and bounded-read guarantees.
pub struct RestrictedCheckpointSource {
    recipes: RecipeInferenceCache,
    source: RetainedCheckpointSource,
    contract: String,
    denied: BTreeSet<String>,
    allowed: Option<BTreeSet<String>>,
}

impl RestrictedCheckpointSource {
    /// Creates a source view excluding exactly the supplied catalog keys.
    ///
    /// Every denied key must exist in the source at construction time. This
    /// prevents a misspelled projection from silently widening the view.
    pub fn excluding(
        source: impl Into<RetainedCheckpointSource>,
        contract: impl Into<String>,
        denied: BTreeSet<String>,
    ) -> Result<Self, StoreError> {
        let source = source.into();
        let contract = contract.into();
        if contract.is_empty() {
            return Err(StoreError::Internal(
                "restricted checkpoint source requires a nonempty contract identity".into(),
            ));
        }
        let source_keys = source.source_keys().into_iter().collect::<BTreeSet<_>>();
        if let Some(key) = denied.iter().find(|key| !source_keys.contains(*key)) {
            return Err(StoreError::UnknownTensor { key: key.clone() });
        }
        Ok(Self {
            source,
            contract,
            recipes: RecipeInferenceCache::default(),
            denied,
            allowed: None,
        })
    }

    /// Creates a source view containing exactly the supplied catalog keys.
    ///
    /// Every allowed key must exist in the source. The explicit allow set is
    /// retained so callers can audit the projection without reconstructing it
    /// from the source catalog and an exclusion set.
    pub fn including(
        source: impl Into<RetainedCheckpointSource>,
        contract: impl Into<String>,
        allowed: BTreeSet<String>,
    ) -> Result<Self, StoreError> {
        let source = source.into();
        let contract = contract.into();
        if contract.is_empty() {
            return Err(StoreError::Internal(
                "restricted checkpoint source requires a nonempty contract identity".into(),
            ));
        }
        let source_keys = source.source_keys().into_iter().collect::<BTreeSet<_>>();
        if let Some(key) = allowed.iter().find(|key| !source_keys.contains(*key)) {
            return Err(StoreError::UnknownTensor { key: key.clone() });
        }
        let denied = source_keys.difference(&allowed).cloned().collect();
        Ok(Self {
            source,
            contract,
            recipes: RecipeInferenceCache::default(),
            denied,
            allowed: Some(allowed),
        })
    }

    /// Returns the stable identity used by authorization failures.
    pub fn contract_identity(&self) -> &str {
        &self.contract
    }

    /// Returns the exact keys denied by this view.
    pub fn denied_keys(&self) -> &BTreeSet<String> {
        &self.denied
    }

    /// Returns the exact allow set when this is an inclusion projection.
    pub fn allowed_keys(&self) -> Option<&BTreeSet<String>> {
        self.allowed.as_ref()
    }

    fn is_authorized(&self, key: &str) -> bool {
        self.allowed.as_ref().map_or_else(
            || !self.denied.contains(key),
            |allowed| allowed.contains(key),
        )
    }

    fn authorize(&self, key: &str) -> Result<(), StoreError> {
        if !self.is_authorized(key) {
            Err(StoreError::UnauthorizedTensor {
                contract: self.contract.clone(),
                key: key.to_owned(),
            })
        } else {
            Ok(())
        }
    }
}

impl CheckpointSource for RestrictedCheckpointSource {
    fn prepared_acquisition_owner(self: Arc<Self>) -> Option<PreparedAcquisitionOwner> {
        Some(acquisition::retained_route::owner_restricted(self))
    }

    fn prepared_acquisition_source(&self) -> PreparedAcquisitionSource<'_> {
        PreparedAcquisitionSource(acquisition::Route::Restricted(self))
    }

    fn source_lease_controls<'a>(
        &'a self,
        key: &'a str,
    ) -> Result<SourceLeaseControls<'a>, LeaseControlBorrowError<'a>> {
        if !self.is_authorized(key) {
            return Err(SourceMetadataBorrowError::UnauthorizedTensor.into());
        }
        self.source.source_lease_controls(key)
    }

    fn source_metadata_borrowed(&self, key: &str) -> SourceMetadataLoan<'_> {
        if !self.is_authorized(key) {
            return Err(SourceMetadataBorrowError::UnauthorizedTensor);
        }
        self.source.source_metadata_borrowed(key)
    }
    fn source_key_authority_borrowed(
        &self,
        key: &str,
    ) -> Result<SourceKeyAuthority, SourceMetadataBorrowError<'_>> {
        if !self.is_authorized(key) {
            return Ok(SourceKeyAuthority::Ordinary);
        }
        self.source.source_key_authority_borrowed(key)
    }

    fn source_storage_slot_bound(&self) -> Result<Option<usize>, StoreError> {
        self.source.source_storage_slot_bound()
    }

    fn visit_source_storage(
        &self,
        visitor: &mut dyn FnMut(SourceStorageRef<'_>),
    ) -> Result<bool, StoreError> {
        self.source.visit_source_storage(visitor)
    }

    fn source_storage(&self) -> Result<Option<SourceStorage>, StoreError> {
        self.source.source_storage()
    }

    fn recipe_cache(&self) -> Option<&RecipeInferenceCache> {
        self.source.recipe_cache().map(|_| &self.recipes)
    }

    fn prepare_encoded_read(
        &self,
        keys: &[String],
    ) -> Result<Option<EncodedReadBatch>, StoreError> {
        for key in keys {
            self.authorize(key)?;
        }
        self.source.prepare_encoded_read(keys)
    }

    fn source_keys(&self) -> Vec<String> {
        self.source
            .source_keys()
            .into_iter()
            .filter(|key| self.is_authorized(key))
            .collect()
    }

    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.authorize(key)?;
        self.source.source_metadata(key)
    }

    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        self.authorize(&request.key)?;
        self.source.acquire_lease(request)
    }

    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        self.source.source_diagnostics()
    }

    fn source_provenance(&self, key: &str) -> Result<TensorSourceProvenance, StoreError> {
        self.authorize(key)?;
        self.source.source_provenance(key)
    }

    fn materialized_source_keys(&self) -> Vec<String> {
        self.source
            .materialized_source_keys()
            .into_iter()
            .filter(|key| self.is_authorized(key))
            .collect()
    }

    fn materialized_source_shards(&self) -> Vec<PathBuf> {
        self.source.materialized_source_shards()
    }

    fn unclaimed_checkpoint_keys(&self) -> Vec<String> {
        self.source
            .unclaimed_checkpoint_keys()
            .into_iter()
            .filter(|key| self.is_authorized(key))
            .collect()
    }

    fn is_authoritative_materialized_key(&self, key: &str) -> bool {
        self.is_authorized(key) && self.source.is_authoritative_materialized_key(key)
    }

    fn is_checkpoint_contract_resolved(&self) -> bool {
        self.source.is_checkpoint_contract_resolved()
    }
}

/// A checkpoint source restricted to one resolved architecture contract.
///
/// The wrapper is cold-path policy only: it filters catalog inspection and
/// rejects every lease request not selected by the resolved physical layout.
pub struct ResolvedCheckpointSource {
    recipes: RecipeInferenceCache,
    source: RetainedCheckpointSource,
    contract: crate::validation::ResolvedCheckpointPlan,
}

impl ResolvedCheckpointSource {
    /// Restricts a source to the physical keys selected by a contract.
    pub fn new(
        source: impl Into<RetainedCheckpointSource>,
        contract: crate::validation::ResolvedCheckpointPlan,
    ) -> Self {
        Self {
            source: source.into(),
            contract,
            recipes: RecipeInferenceCache::default(),
        }
    }

    /// Returns the resolved contract identity.
    pub fn contract_identity(&self) -> &str {
        self.contract.identity()
    }

    /// Returns catalog keys admitted but not claimed by the selected layout.
    pub fn unclaimed_keys(&self) -> &BTreeSet<String> {
        self.contract.unclaimed_keys()
    }

    fn authorize_acquisition(&self, key: &str) -> Result<(), StoreError> {
        if !self.source.is_authoritative_materialized_key(key) {
            self.authorize(key)?;
        }
        Ok(())
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
    fn prepared_acquisition_owner(self: Arc<Self>) -> Option<PreparedAcquisitionOwner> {
        Some(acquisition::retained_route::owner_resolved(self))
    }

    fn prepared_acquisition_source(&self) -> PreparedAcquisitionSource<'_> {
        PreparedAcquisitionSource(acquisition::Route::Resolved(self))
    }

    fn source_lease_controls<'a>(
        &'a self,
        key: &'a str,
    ) -> Result<SourceLeaseControls<'a>, LeaseControlBorrowError<'a>> {
        if !self.contract.source_keys().contains(key)
            && self.source.source_key_authority_borrowed(key)? != SourceKeyAuthority::Materialized
        {
            return Err(SourceMetadataBorrowError::UnauthorizedTensor.into());
        }
        self.source.source_lease_controls(key)
    }

    fn source_metadata_borrowed(&self, key: &str) -> SourceMetadataLoan<'_> {
        // A contract member needs no overlay exception. Outside that contract,
        // use only the actual source's closed membership companion.
        if !self.contract.source_keys().contains(key)
            && self.source.source_key_authority_borrowed(key)? != SourceKeyAuthority::Materialized
        {
            return Err(SourceMetadataBorrowError::UnauthorizedTensor);
        }
        self.source.source_metadata_borrowed(key)
    }
    fn source_key_authority_borrowed(
        &self,
        key: &str,
    ) -> Result<SourceKeyAuthority, SourceMetadataBorrowError<'_>> {
        self.source.source_key_authority_borrowed(key)
    }

    fn source_storage_slot_bound(&self) -> Result<Option<usize>, StoreError> {
        self.source.source_storage_slot_bound()
    }

    fn visit_source_storage(
        &self,
        visitor: &mut dyn FnMut(SourceStorageRef<'_>),
    ) -> Result<bool, StoreError> {
        self.source.visit_source_storage(visitor)
    }

    fn source_storage(&self) -> Result<Option<SourceStorage>, StoreError> {
        self.source.source_storage()
    }

    fn recipe_cache(&self) -> Option<&RecipeInferenceCache> {
        self.source.recipe_cache().map(|_| &self.recipes)
    }

    fn prepare_encoded_read(
        &self,
        keys: &[String],
    ) -> Result<Option<EncodedReadBatch>, StoreError> {
        for key in keys {
            if !self.source.is_authoritative_materialized_key(key) {
                self.authorize(key)?;
            }
        }
        self.source.prepare_encoded_read(keys)
    }

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
        self.authorize_acquisition(&request.key)?;
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
    /// Inference cache bound to this immutable catalog and authorization view.
    /// Returning a cache promises metadata and provenance remain fixed, and
    /// every returned lease/read batch is checked against that same catalog.
    fn recipe_cache(&self) -> Option<&RecipeInferenceCache> {
        None
    }

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
#[derive(Debug, Clone, thiserror::Error)]
pub enum StoreError {
    /// Fixed geometry or allocation failure from encoded source projection.
    #[error(transparent)]
    EncodedProjection(#[from] EncodedProjectionError),
    /// Typed refusal from source discovery or store metadata admission.
    #[error("SafeTensors source admission failed: {0}")]
    SafetensorsSourceAdmission(#[source] Arc<dyn std::error::Error + Send + Sync>),
    /// Lazy SafeTensors header admission or construction failure, with custody.
    #[error("{0}")]
    SafetensorsHeader(#[source] Arc<SafetensorsHeaderFailure>),
    /// Explicit prepared GGUF reader storage could not supply its destination.
    #[error("GGUF prepared reader storage failed for tensor {key:?}: {source}")]
    GgufPreparedReaderStorage {
        /// Actual requested logical name.
        key: String,
        /// Original fixed cause and any shard context.
        #[source]
        source: Arc<eredu_gguf::Error>,
    },
    /// A prepared GGUF header cannot supply its exact retained destination.
    #[error("GGUF prepared header storage failed for tensor {key:?}: {source}")]
    GgufPreparedHeaderStorage {
        /// Actual requested logical name.
        key: String,
        /// Original typed cause and its shard context.
        #[source]
        source: Arc<eredu_gguf::Error>,
    },

    /// An explicitly prepared GGUF source observed different header bytes.
    #[error("GGUF prepared header changed for tensor {key:?}: {source}")]
    GgufPreparedHeaderChanged {
        /// Actual requested logical name.
        key: String,
        /// Original typed GGUF cause, retaining its shard context.
        #[source]
        source: Arc<eredu_gguf::Error>,
    },
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
    admission: Arc<AdmittedShard>,
    // Share overlapping leases without retaining a second resident checkpoint.
    // The Vec indirection releases the payload when its last lease ends, even
    // while a weak cache entry still retains the small Arc allocation.
    full_tensors: Mutex<BTreeMap<String, Weak<Vec<u8>>>>,
}

#[derive(Debug)]
struct CacheEntry {
    shard: Arc<CachedShard>,
    last_used: u64,
}

#[derive(Debug, Default)]
struct CacheState {
    entries: BTreeMap<PathBuf, CacheEntry>,
    paths: bulk::DiagnosticPaths,
    tick: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
    _source_admission: Option<Arc<dyn SafetensorsSourceAdmission>>,
}

#[derive(Debug, Default)]
struct SafetensorsReadTelemetry {
    physical_reads: AtomicU64,
    physical_read_bytes: AtomicU64,
    _source_admission: Option<Arc<dyn SafetensorsSourceAdmission>>,
}

#[derive(Debug, Clone)]
struct CatalogEntry {
    shard: PathBuf,
}

#[derive(Debug, Eq, PartialEq)]
struct AdmittedFileIdentity {
    canonical_path: PathBuf,
    version: crate::artifact::file::FileVersion,
}

impl AdmittedFileIdentity {
    fn from_metadata(path: &Path, metadata: &std::fs::Metadata) -> Result<Self, StoreError> {
        Ok(Self {
            // Keep the ordinary path allocation and error semantics unchanged.
            canonical_path: path.to_path_buf(),
            version: crate::artifact::file::FileVersion::from_metadata(metadata)
                .map_err(|error| fs_error(path, error))?,
        })
    }
}

#[derive(Debug)]
pub(crate) struct AdmittedFile {
    identity: AdmittedFileIdentity,
    source_admission: Option<Arc<dyn SafetensorsSourceAdmission>>,
}

impl AdmittedFile {
    pub(crate) fn open(path: &Path) -> Result<Self, StoreError> {
        let file = File::open(path).map_err(|error| fs_error(path, error))?;
        let identity = AdmittedFileIdentity::from_metadata(
            path,
            &file.metadata().map_err(|error| fs_error(path, error))?,
        )?;
        Ok(Self { identity, source_admission: None })
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
        read_bytes::ordinary_validate_file(self, path, file)
    }
}

/// Header admission survives payload-cache eviction and is shared by cloned shard sets.
#[derive(Debug)]
pub(crate) struct AdmittedShard {
    pub(crate) file: Arc<AdmittedFile>,
    expected: Option<BTreeSet<String>>,
    max_header_bytes: u64,
    header_admission: Option<Arc<dyn SafetensorsHeaderAdmission>>,
    header: OnceLock<Result<AdmittedHeader, StoreError>>,
    #[cfg(test)]
    header_reads: AtomicU64,
}

#[derive(Debug)]
pub(crate) struct AdmittedHeader {
    pub(crate) metadata: Metadata,
    pub(crate) payload_offset: usize,
    pub(crate) tensors: BTreeMap<String, TensorMetadata>,
    _reservation: Option<SafetensorsHeaderReservation>,
}

impl AdmittedShard {
    pub(crate) fn new(
        path: &Path,
        expected: Option<BTreeSet<String>>,
        max_header_bytes: u64,
        header_admission: Option<Arc<dyn SafetensorsHeaderAdmission>>,
        source_admission: Option<Arc<dyn SafetensorsSourceAdmission>>,
    ) -> Result<Self, StoreError> {
        let mut file = AdmittedFile::open(path)?;
        file.source_admission = source_admission;
        Ok(Self {
            file: Arc::new(file),
            expected,
            max_header_bytes: max_header_bytes.min(MAX_HEADER_BYTES),
            header_admission,
            header: OnceLock::new(),
            #[cfg(test)]
            header_reads: AtomicU64::new(0),
        })
    }

    pub(crate) fn header(&self, path: &Path) -> Result<&AdmittedHeader, StoreError> {
        self.header
            .get_or_init(|| {
                #[cfg(test)]
                self.header_reads.fetch_add(1, Ordering::Relaxed);
                let mut reservation = None;
                let result = (|| {
                    let (payload_offset, metadata) = read_safetensors_metadata_admitted(
                        path,
                        &self.file,
                        self.max_header_bytes,
                        self.header_admission.as_deref(),
                        &mut reservation,
                    )?;
                    if let Some(expected) = &self.expected {
                        let actual = metadata.offset_keys().into_iter().collect::<BTreeSet<_>>();
                        if let Some(key) = expected.difference(&actual).next() {
                            return Err(StoreError::ContradictoryIndexMapping {
                                key: key.clone(),
                                path: path.into(),
                            });
                        }
                        if let Some(key) = actual.difference(expected).next() {
                            return Err(StoreError::UnindexedShardTensor {
                                key: key.clone(),
                                path: path.into(),
                            });
                        }
                    }
                    let tensors = metadata
                        .tensors()
                        .into_iter()
                        .map(|(name, info)| {
                            // Zero dimensions are not model parameters; other geometry was
                            // established by Metadata's checked offset/shape validation.
                            if info.shape.contains(&0) {
                                return Err(io_error(
                                    path,
                                    format!("tensor {name:?} has a zero dimension"),
                                ));
                            }
                            let tensor = TensorMetadata {
                                name: name.clone(),
                                logical_shape: info.shape.clone(),
                                physical_shape: info.shape.clone(),
                                stored_dtype: stored_dtype_from_safetensors(info.dtype),
                                encoded_byte_len: (info.data_offsets.1 - info.data_offsets.0)
                                    as u64,
                                backing_shard: Some(path.into()),
                            };
                            Ok((name, tensor))
                        })
                        .collect::<Result<_, StoreError>>()?;
                    Ok(AdmittedHeader {
                        metadata,
                        payload_offset,
                        tensors,
                        _reservation: reservation.clone(),
                    })
                })();
                let completion = reservation.as_ref().and_then(|hold| hold.complete().err());
                match (result, reservation, completion) {
                    (Err(error), Some(reservation), completion) => {
                        Err(StoreError::SafetensorsHeader(Arc::new(
                            SafetensorsHeaderFailure::new(error, Some(reservation), completion),
                        )))
                    }
                    (Ok(header), Some(reservation), Some(cause)) => {
                        Err(StoreError::SafetensorsHeader(Arc::new(
                            SafetensorsHeaderFailure::completion_failed(cause, header, reservation),
                        )))
                    }
                    (result, _, None) => result,
                    (_, None, Some(_)) => unreachable!("completion requires a reservation"),
                }
            })
            .as_ref()
            .map_err(Clone::clone)
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
    bytes: Arc<Vec<u8>>,
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
    shards: SafetensorsShards,
    cache: Arc<Mutex<CacheState>>,
    read_telemetry: Arc<SafetensorsReadTelemetry>,
    max_cached_shards: usize,
    source_admission: Option<Arc<dyn SafetensorsSourceAdmission>>,
}

impl SafetensorsWeightStore {
    /// Borrows this source's actual discovery-policy value without extracting it.
    /// This is concrete type inspection, not a provider-reported admission claim.
    /// Ordinary sources have no policy; callers validate their own private type.
    pub fn source_admission_owner<C: std::any::Any>(&self) -> Option<&C> {
        let policy: &dyn std::any::Any = self.source_admission.as_ref()?.as_ref();
        policy.downcast_ref()
    }

    /// Opens a file, indexed directory, or directory containing `model.safetensors`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        Self::open_with_max_cached_shards(path, DEFAULT_MAX_CACHED_SHARDS)
    }

    /// Opens a checkpoint with an explicit nonzero shard-cache bound.
    pub fn open_with_max_cached_shards(
        path: impl AsRef<Path>,
        max_cached_shards: usize,
    ) -> Result<Self, StoreError> {
        Self::open_with_limits(
            path,
            max_cached_shards,
            SafetensorsDiscoveryLimits::default(),
        )
    }

    /// Opens with a nonzero shard-cache bound and encoded metadata input limits.
    /// Indexed headers remain lazy; each admission retains its header limit.
    pub fn open_with_limits(
        path: impl AsRef<Path>,
        max_cached_shards: usize,
        limits: SafetensorsDiscoveryLimits,
    ) -> Result<Self, StoreError> {
        let shards = SafetensorsShards::discover_catalog(path, limits)?;
        Self::open_admitted(shards, max_cached_shards)
    }

    /// Opens a source with prospective admission for each lazy header.
    /// Discovery/index/store metadata and payload storage need separate admission.
    /// The policy and each accepted reservation survive all shared header aliases.
    pub fn open_with_header_admission(
        path: impl AsRef<Path>,
        max_cached_shards: usize,
        limits: SafetensorsDiscoveryLimits,
        admission: Arc<dyn SafetensorsHeaderAdmission>,
    ) -> Result<Self, StoreError> {
        let shards = SafetensorsShards::discover_catalog_with_header_admission(
            path,
            limits,
            Some(admission),
        )?;
        Self::open_admitted(shards, max_cached_shards)
    }

    /// Opens under a policy funded before discovery. The caller retains it on
    /// failure and completes construction after this method returns. Independent
    /// metadata exports and read/payload construction need separate admission.
    pub fn open_with_source_admission(
        path: impl AsRef<Path>,
        max_cached_shards: usize,
        limits: SafetensorsDiscoveryLimits,
        admission: Arc<dyn SafetensorsSourceAdmission>,
    ) -> Result<Self, StoreError> {
        let shards = SafetensorsShards::discover_catalog_with_source_admission(
            path.as_ref(), limits, None, Some(admission.clone()),
        )?;
        Self::open_admitted_with_source_admission(shards, max_cached_shards, Some(admission))
    }

    /// Opens the exact shard set admitted by portable artifact inspection.
    ///
    /// This constructor performs no directory or index discovery. Indexed
    /// shards remain selectively opened and governed by `max_cached_shards`.
    pub fn open_admitted(
        shards: SafetensorsShards,
        max_cached_shards: usize,
    ) -> Result<Self, StoreError> {
        Self::open_admitted_with_source_admission(shards, max_cached_shards, None)
    }

    /// Constructs a fresh store/cache over exact retained shards. The supplied
    /// policy reserves this store's metadata independently of discovery custody.
    /// Its owner retains it on failure and completes construction on return.
    pub fn open_admitted_with_source_admission(
        shards: SafetensorsShards,
        max_cached_shards: usize,
        admission: Option<Arc<dyn SafetensorsSourceAdmission>>,
    ) -> Result<Self, StoreError> {
        if max_cached_shards == 0 {
            return Err(StoreError::InvalidShardCacheLimit);
        }
        if let Some(policy) = &admission {
            let mut input_bytes = Some(0usize);
            let mut add = |name: &str, path: &Path| {
                input_bytes = input_bytes.and_then(|n| n.checked_add(name.len()))
                    .and_then(|n| n.checked_add(path.as_os_str().len()));
            };
            if let Some(locations) = shards.tensor_locations() {
                for (name, path) in locations { add(name, path); }
            } else {
                let path = &shards.payload_paths()[0];
                for name in shards.admission(path).header(path)?.tensors.keys() { add(name, path); }
            }
            for path in shards.payload_paths() { add("", path); }
            let input_bytes = input_bytes.ok_or_else(|| StoreError::Overflow { context: "SafeTensors store metadata input".into() })?;
            policy.reserve_store(input_bytes).map_err(StoreError::SafetensorsSourceAdmission)?;
        }
        let catalog = if let Some(locations) = shards.tensor_locations() {
            locations
                .iter()
                .map(|(key, path)| {
                    (
                        key.clone(),
                        CatalogEntry {
                            shard: path.clone(),
                        },
                    )
                })
                .collect()
        } else {
            let path = &shards.payload_paths()[0];
            shards
                .admission(path)
                .header(path)?
                .tensors
                .keys()
                .map(|key| {
                    (
                        key.clone(),
                        CatalogEntry {
                            shard: path.clone(),
                        },
                    )
                })
                .collect()
        };
        let paths = bulk::DiagnosticPaths::new(shards.payload_paths());
        let source_admission = admission;
        Ok(Self {
            source_admission: source_admission.clone(),
            catalog,
            shards,
            cache: Arc::new(Mutex::new(CacheState {
                paths,
                _source_admission: source_admission.clone(),
                ..CacheState::default()
            })),
            read_telemetry: Arc::new(SafetensorsReadTelemetry {
                _source_admission: source_admission,
                ..SafetensorsReadTelemetry::default()
            }),
            max_cached_shards,
        })
    }

    fn lock_cache(&self) -> Result<MutexGuard<'_, CacheState>, StoreError> {
        cache_policy::lock(&self.cache)
    }

    fn acquire_shard(&self, entry: &CatalogEntry) -> Result<Arc<CachedShard>, StoreError> {
        cache_policy::acquire_shard(&self.cache, self.max_cached_shards, &entry.shard, || {
            Arc::clone(self.shards.admission(&entry.shard))
        })
    }

    /// Prepares the containing header through its retained admission policy and
    /// borrows the resulting tensor metadata. This reads no payload and creates
    /// no independent metadata copy. Repeated calls reuse the shared header.
    pub fn prepare_metadata(&self, key: &str) -> Result<&TensorMetadata, StoreError> {
        let entry = self
            .catalog
            .get(key)
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })?;
        let metadata = self
            .shards
            .admission(&entry.shard)
            .header(&entry.shard)?
            .tensors
            .get(key)
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })?;
        cache_policy::touch_metadata(&self.cache, &entry.shard)?;
        Ok(metadata)
    }
    fn cached_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.prepare_metadata(key).cloned()
    }

}

impl WeightStore for SafetensorsWeightStore {
    fn recipe_cache(&self) -> Option<&RecipeInferenceCache> {
        Some(self.shards.recipe_cache())
    }

    type Lease = SafetensorsLease;

    fn keys(&self) -> Vec<String> {
        self.catalog.keys().cloned().collect()
    }

    fn metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
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
        let header = shard.admission.header(&shard.path)?;
        let info = header.metadata.info(&request.key).ok_or_else(|| {
            io_error(
                &entry.shard,
                format!("shard does not contain tensor {:?}", request.key),
            )
        })?;
        let output_shape =
            validate_selection(&request.key, &metadata.logical_shape, &request.selection)?;
        let payload_start = header
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
        let cached = cache_policy::lookup(&shard, &request.key)?;
        let complete_tensor =
            read.ranges.len() == 1 && read.ranges[0].start == 0 && read.ranges[0].end == tensor_len;
        let cache_hit = cached.is_some();
        let cached = cached
            .map(cache_policy::CachedPayload::validate)
            .transpose()?;
        let bytes = match cached {
            Some(bytes) if complete_tensor => bytes.into_bytes(),
            Some(bytes) => Arc::new(copy_safetensors_ranges(
                &request.key,
                bytes.bytes(),
                &read.ranges,
            )?),
            None => {
                let bytes = Arc::new(read_safetensors_ranges(
                    &shard.path,
                    &shard.admitted_file,
                    payload_start,
                    &read.ranges,
                    self.read_telemetry.as_ref(),
                )?);
                if complete_tensor {
                    cache_policy::publish_full(&shard, &request.key, &bytes)?;
                }
                bytes
            }
        };
        let length = u64::try_from(bytes.len()).map_err(|_| StoreError::Overflow {
            context: format!("physical read length for {:?}", request.key),
        })?;
        cache_policy::publish_payload(&self.cache, &shard)?;
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
                    u64::try_from(read.ranges.len()).unwrap_or(u64::MAX)
                },
                physical_read_bytes: if cache_hit { 0 } else { length },
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
            touched_shard_paths: cache.paths.touched().cloned().collect(),
            payload_shard_paths: cache.paths.payloads().cloned().collect(),
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
    fn prepared_acquisition_owner(self: Arc<Self>) -> Option<PreparedAcquisitionOwner> {
        Some(acquisition::retained_route::owner_safetensors(self))
    }

    fn prepared_acquisition_source(&self) -> PreparedAcquisitionSource<'_> {
        PreparedAcquisitionSource(acquisition::Route::Safetensors(self))
    }

    fn source_lease_controls<'a>(
        &'a self,
        key: &'a str,
    ) -> Result<SourceLeaseControls<'a>, LeaseControlBorrowError<'a>> {
        let metadata = self.source_metadata_borrowed(key)?;
        let entry = &self.catalog[key];
        let admission = self.shards.admission(&entry.shard);
        let header = admission
            .header
            .get()
            .ok_or(SourceMetadataBorrowError::HeaderUnavailable)?
            .as_ref()
            .map_err(SourceMetadataBorrowError::Retained)?;
        let info = header
            .metadata
            .info(key)
            .ok_or(SourceMetadataBorrowError::UnknownTensor)?;
        Ok(
            SourceLeaseControls::new(LeaseProvider::Safetensors, metadata, key)
                .with_safetensors_lease(SafetensorsLeaseSource::new(self, key, metadata))
                .with_safetensors_reads(
                    SafetensorsReadSource::new(
                        key,
                        &metadata.logical_shape,
                        &info.shape,
                        info.dtype,
                        info.data_offsets,
                    )
                    .with_file(read_bytes::ReadFileSource {
                        path: &entry.shard,
                        admitted: &admission.file,
                        header_payload_start: header.payload_offset,
                        telemetry: &self.read_telemetry,
                    }),
                ),
        )
    }

    fn source_metadata_borrowed(&self, key: &str) -> SourceMetadataLoan<'_> {
        let entry = self
            .catalog
            .get(key)
            .ok_or(SourceMetadataBorrowError::UnknownTensor)?;
        let admitted = self.shards.admission(&entry.shard);
        let header = admitted
            .header
            .get()
            .ok_or(SourceMetadataBorrowError::HeaderUnavailable)?;
        let header = header
            .as_ref()
            .map_err(SourceMetadataBorrowError::Retained)?;
        header
            .tensors
            .get(key)
            .ok_or(SourceMetadataBorrowError::UnknownTensor)
    }
    fn source_key_authority_borrowed(
        &self,
        _: &str,
    ) -> Result<SourceKeyAuthority, SourceMetadataBorrowError<'_>> {
        Ok(SourceKeyAuthority::Ordinary)
    }

    fn source_storage_slot_bound(&self) -> Result<Option<usize>, StoreError> {
        Ok(Some(0))
    }

    fn visit_source_storage(
        &self,
        _visitor: &mut dyn FnMut(SourceStorageRef<'_>),
    ) -> Result<bool, StoreError> {
        // Header/file admission is separate metadata; payload cache entries are
        // Weak. Externally retained leases belong to their operation owners.
        Ok(true)
    }

    fn source_storage(&self) -> Result<Option<SourceStorage>, StoreError> {
        // Cached shards own exact file/header admission; payload cache entries
        // are Weak and the source does not keep encoded tensor bytes alive.
        Ok(Some(SourceStorage::default()))
    }

    fn recipe_cache(&self) -> Option<&RecipeInferenceCache> {
        Some(self.shards.recipe_cache())
    }

    fn prepare_encoded_read(
        &self,
        keys: &[String],
    ) -> Result<Option<EncodedReadBatch>, StoreError> {
        bulk::prepare(self, keys).map(Some)
    }

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

#[cfg(test)]
fn read_safetensors_metadata(
    path: &Path,
    admitted_file: &AdmittedFile,
    max_header_bytes: u64,
) -> Result<(usize, Metadata), StoreError> {
    read_safetensors_metadata_admitted(path, admitted_file, max_header_bytes, None, &mut None)
}

fn read_safetensors_metadata_admitted(
    path: &Path,
    admitted_file: &AdmittedFile,
    max_header_bytes: u64,
    admission: Option<&dyn SafetensorsHeaderAdmission>,
    reservation: &mut Option<SafetensorsHeaderReservation>,
) -> Result<(usize, Metadata), StoreError> {
    let mut file = admitted_file.open_validated(path)?;
    file.seek(SeekFrom::Start(0))
        .map_err(|error| io_error(path, error))?;
    let file_len = admitted_file.identity.version.length;
    let metadata = read_safetensors_metadata_from_admitted(
        path,
        &mut file,
        file_len,
        max_header_bytes,
        admission,
        reservation,
    )?;
    admitted_file.validate_file(path, &file)?;
    Ok(metadata)
}

#[cfg(test)]
fn read_safetensors_metadata_from(
    path: &Path,
    reader: &mut impl Read,
    file_len: u64,
    max_header_bytes: u64,
) -> Result<(usize, Metadata), StoreError> {
    read_safetensors_metadata_from_admitted(
        path,
        reader,
        file_len,
        max_header_bytes,
        None,
        &mut None,
    )
}

fn read_safetensors_metadata_from_admitted(
    path: &Path,
    reader: &mut impl Read,
    file_len: u64,
    max_header_bytes: u64,
    admission: Option<&dyn SafetensorsHeaderAdmission>,
    reservation: &mut Option<SafetensorsHeaderReservation>,
) -> Result<(usize, Metadata), StoreError> {
    let mut encoded_header_len = [0u8; 8];
    reader
        .read_exact(&mut encoded_header_len)
        .map_err(|error| io_error(path, error))?;
    let header_len = u64::from_le_bytes(encoded_header_len);
    let max_header_bytes = max_header_bytes.min(MAX_HEADER_BYTES);
    if header_len > max_header_bytes {
        return Err(StoreError::MalformedSafetensors {
            path: path.to_path_buf(),
            message: format!("header exceeds {max_header_bytes} bytes"),
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
    if let Some(admission) = admission {
        *reservation = Some(
            admission
                .reserve(SafetensorsHeaderRequest {
                    json_bytes: header_len,
                    buffer_bytes: 8 + header_len,
                    path_bytes: path.as_os_str().len(),
                })
                .map_err(StoreError::SafetensorsHeader)?,
        );
    }
    let mut encoded_header = Vec::with_capacity(8 + header_len);
    encoded_header.extend_from_slice(&encoded_header_len);
    encoded_header.resize(8 + header_len, 0);
    reader
        .read_exact(&mut encoded_header[8..])
        .map_err(|error| io_error(path, error))?;
    let metadata = crate::safetensors::parse_header(&encoded_header[8..]).map_err(|error| {
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

fn plan_safetensors_reads(
    key: &str,
    dtype: Dtype,
    shape: &[usize],
    payload_len: usize,
    selection: &TensorSelection,
    output_shape: &[usize],
    policy: ReadPolicy,
) -> Result<SafetensorsReadPlan, StoreError> {
    read_plan::ordinary(
        key,
        dtype,
        shape,
        payload_len,
        selection,
        output_shape,
        policy,
    )
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
    after_read: impl FnOnce(),
) -> Result<Vec<u8>, StoreError> {
    let capacity = ranges.iter().try_fold(0usize, |total, range| {
        total
            .checked_add(range.len())
            .ok_or_else(|| StoreError::Overflow {
                context: format!("selected payload length for {}", path.display()),
            })
    })?;
    let mut file = admitted_file.open_validated(path)?;
    let bytes = read_safetensors_range_pass(
        path,
        &mut file,
        tensor_payload_start,
        ranges,
        capacity,
        telemetry,
    )?;
    after_read();
    admitted_file.validate_file(path, &file)?;
    Ok(bytes)
}

fn read_safetensors_range_pass(
    path: &Path,
    file: &mut File,
    tensor_payload_start: usize,
    ranges: &[Range<usize>],
    capacity: usize,
    telemetry: &SafetensorsReadTelemetry,
) -> Result<Vec<u8>, StoreError> {
    read_bytes::ordinary_read_pass(
        path,
        file,
        tensor_payload_start,
        ranges,
        capacity,
        telemetry,
    )
}

fn copy_safetensors_ranges(
    key: &str,
    payload: &[u8],
    ranges: &[Range<usize>],
) -> Result<Vec<u8>, StoreError> {
    read_bytes::ordinary_copy(key, payload, ranges)
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
    selection_validation::ordinary(key, shape, selection)
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
    memory_destination::ordinary_selection(key, dtype, shape, data, selection, output_shape, policy)
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
    fn shard_admission_survives_views_stores_and_payload_cache_eviction() {
        let directory = tempfile::tempdir().unwrap();
        let mut locations = BTreeMap::new();
        for key in ["left", "right"] {
            let member = format!("{key}.safetensors");
            serialize_to_file(
                [(key, TensorView::new(Dtype::U8, vec![1], &[7]).unwrap())],
                None,
                &directory.path().join(&member),
            )
            .unwrap();
            locations.insert(key, member);
        }
        std::fs::write(
            directory.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map": locations})).unwrap(),
        )
        .unwrap();
        let shards = SafetensorsShards::discover(directory.path()).unwrap();
        let catalog =
            crate::safetensors::SafetensorsMetadataCatalog::from_admitted(shards.clone()).unwrap();
        let store = SafetensorsWeightStore::open_admitted(shards.clone(), 1).unwrap();
        for _ in 0..3 {
            for key in ["left", "right"] {
                assert_eq!(store.metadata(key).unwrap(), *catalog.tensor(key).unwrap());
                let before = shards
                    .admission(&store.catalog[key].shard)
                    .header_reads
                    .load(Ordering::Relaxed);
                let loan = store.source_metadata_borrowed(key).unwrap();
                let again = store.source_metadata_borrowed(key).unwrap();
                assert!(std::ptr::eq(loan, again));
                assert_eq!(loan, catalog.tensor(key).unwrap());
                assert_eq!(
                    shards
                        .admission(&store.catalog[key].shard)
                        .header_reads
                        .load(Ordering::Relaxed),
                    before
                );
                drop(
                    store
                        .acquire(TensorReadRequest {
                            key: key.into(),
                            selection: TensorSelection::Full,
                            policy: ReadPolicy::RequireBounded,
                        })
                        .unwrap(),
                );
            }
        }
        assert!(store.diagnostics().unwrap().evictions >= 4);
        let prepared = PreparedCheckpointSource::open_admitted_safetensors(
            shards.clone(),
            catalog.tensors().clone(),
            1,
        )
        .unwrap();
        let left = &locations["left"];
        std::fs::remove_file(directory.path().join(left)).unwrap();
        // Metadata and direct-read preparation are pure admitted-catalog operations.
        for _ in 0..4 {
            assert_eq!(
                prepared.source_metadata("left").unwrap(),
                *catalog.tensor("left").unwrap()
            );
            prepared.source_provenance("left").unwrap();
            prepared
                .prepare_encoded_read(&["left".into()])
                .unwrap()
                .unwrap();
        }
        assert!(prepared
            .acquire_lease(TensorReadRequest {
                key: "left".into(),
                selection: TensorSelection::Full,
                policy: ReadPolicy::RequireBounded
            })
            .is_err());
        for path in shards.payload_paths() {
            assert_eq!(
                shards.admission(path).header_reads.load(Ordering::Relaxed),
                1
            );
        }
    }

    #[test]
    fn header_input_limit_refuses_before_reading_the_body() {
        for (declared, limit) in [(17, 16), (MAX_HEADER_BYTES + 1, u64::MAX)] {
            let mut prefix = std::io::Cursor::new(declared.to_le_bytes());
            let error = read_safetensors_metadata_from(
                Path::new("header.safetensors"),
                &mut prefix,
                declared + 8,
                limit,
            )
            .unwrap_err();
            assert!(matches!(error, StoreError::MalformedSafetensors { .. }));
            assert!(error.to_string().contains("header exceeds"));
            assert_eq!(prefix.position(), 8);
        }
    }

    #[test]
    fn failed_header_admission_is_not_retried() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("broken.safetensors");
        std::fs::write(&path, b"broken").unwrap();
        std::fs::write(
            directory.path().join("model.safetensors.index.json"),
            br#"{"weight_map":{"weight":"broken.safetensors"}}"#,
        )
        .unwrap();
        let shards = SafetensorsShards::discover_catalog(
            directory.path(),
            SafetensorsDiscoveryLimits::default(),
        )
        .unwrap();
        let store = SafetensorsWeightStore::open_admitted(shards.clone(), 1).unwrap();
        assert!(matches!(
            store.source_metadata_borrowed("weight"),
            Err(SourceMetadataBorrowError::HeaderUnavailable)
        ));
        assert_eq!(
            shards
                .admission(&store.catalog["weight"].shard)
                .header_reads
                .load(Ordering::Relaxed),
            0
        );
        assert!(matches!(
            store.source_lease_controls("weight"),
            Err(LeaseControlBorrowError::Source(
                SourceMetadataBorrowError::HeaderUnavailable
            ))
        ));
        let first = store.metadata("weight").unwrap_err().to_string();
        let borrowed = store.source_metadata_borrowed("weight").unwrap_err();
        let SourceMetadataBorrowError::Retained(cause) = borrowed else {
            panic!("expected retained header failure")
        };
        let retained = shards
            .admission(&store.catalog["weight"].shard)
            .header
            .get()
            .unwrap()
            .as_ref()
            .unwrap_err();
        assert!(std::ptr::eq(cause, retained));
        let lease_error = store.source_lease_controls("weight").unwrap_err();
        assert_eq!(lease_error.to_string(), first);
        assert!(std::ptr::eq(
            std::error::Error::source(&lease_error)
                .unwrap()
                .downcast_ref::<StoreError>()
                .unwrap(),
            retained
        ));

        assert_eq!(borrowed.to_string(), first);
        assert!(std::ptr::eq(
            std::error::Error::source(&borrowed)
                .unwrap()
                .downcast_ref::<StoreError>()
                .unwrap(),
            retained
        ));
        std::fs::write(&path, b"different header").unwrap();
        assert_eq!(store.metadata("weight").unwrap_err().to_string(), first);
        assert_eq!(
            shards
                .admission(&shards.payload_paths()[0])
                .header_reads
                .load(Ordering::Relaxed),
            1
        );
    }

    #[test]
    fn prepared_catalog_substitution_is_rejected_without_rereading_headers() {
        let directory = tempfile::tempdir().unwrap();
        serialize_to_file(
            [(
                "weight",
                TensorView::new(Dtype::U8, vec![2], &[1, 2]).unwrap(),
            )],
            None,
            &directory.path().join("model.safetensors"),
        )
        .unwrap();
        let catalog =
            crate::safetensors::SafetensorsMetadataCatalog::discover(directory.path()).unwrap();
        let mut changed = catalog.tensors().clone();
        changed.get_mut("weight").unwrap().logical_shape = vec![1, 2];
        assert!(matches!(
            PreparedCheckpointSource::open_admitted_safetensors(
                catalog.admitted_shards(),
                changed,
                1
            ),
            Err(StoreError::PreparedCatalogMismatch { .. })
        ));
        let shards = catalog.shards();
        assert_eq!(
            shards
                .admission(&shards.payload_paths()[0])
                .header_reads
                .load(Ordering::Relaxed),
            1
        );
    }

    #[test]
    fn recipe_caches_preserve_catalog_and_restricted_view_identity() {
        use crate::recipe::DerivedWeightRecipe;
        let source: SharedCheckpointSource = Arc::new(
            MemoryWeightStore::from_safetensors([
                ("allowed".into(), Dtype::U8, vec![1], vec![1]),
                ("denied".into(), Dtype::U8, vec![2], vec![1, 2]),
            ])
            .unwrap(),
        );
        let recipe = DerivedWeightRecipe::source("denied", TensorSelection::Full);
        assert_eq!(recipe.infer(source.as_ref()).unwrap().shape, [2]);
        let view: SharedCheckpointSource = Arc::new(
            RestrictedCheckpointSource::including(
                source.clone(),
                "allowed only",
                BTreeSet::from(["allowed".into()]),
            )
            .unwrap(),
        );
        for _ in 0..2 {
            assert!(matches!(
                recipe.infer(view.as_ref()),
                Err(crate::recipe::RecipeError::Store(
                    StoreError::UnauthorizedTensor { .. }
                ))
            ));
        }
        let other: SharedCheckpointSource = Arc::new(
            MemoryWeightStore::from_safetensors([(
                "denied".into(),
                Dtype::U8,
                vec![3],
                vec![1, 2, 3],
            )])
            .unwrap(),
        );
        assert_eq!(recipe.infer(other.as_ref()).unwrap().shape, [3]);
        assert_eq!(recipe.infer(source.as_ref()).unwrap().shape, [2]);
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
            MAX_HEADER_BYTES,
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
    fn prepared_safetensors_rejects_changed_admitted_file_before_payload_reads() {
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

        let prepared = PreparedCheckpointSource::open_admitted_safetensors(
            admitted.admitted_shards(),
            admitted.tensors().clone(),
            1,
        )
        .unwrap();
        assert_eq!(
            prepared.source_metadata("weight").unwrap().logical_shape,
            [2]
        );
        assert!(matches!(
            prepared.acquire_lease(TensorReadRequest {
                key: "weight".into(),
                selection: TensorSelection::Full,
                policy: ReadPolicy::RequireBounded,
            }),
            Err(StoreError::AdmittedFileChanged { .. })
        ));
        assert_eq!(prepared.source_diagnostics().unwrap().physical_reads, 0);
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
        assert_eq!(prepared.source_diagnostics().unwrap().physical_reads, 1);
        assert_eq!(
            prepared.source_diagnostics().unwrap().physical_read_bytes,
            8
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
        assert_eq!(diagnostics.physical_reads, 2);
        assert_eq!(diagnostics.physical_read_bytes, 16);
        assert_eq!(lease.bounded_read_proof().offset_bytes, 8);
        assert_eq!(lease.bounded_read_proof().length_bytes, 16);
        assert_eq!(lease.bounded_read_proof().physical_reads, 2);
        assert_eq!(lease.bounded_read_proof().physical_read_bytes, 16);
        assert!(lease.bounded_read_proof().physically_bounded);
        let mut expected = selected[8..16].to_vec();
        expected.extend_from_slice(&selected[32..40]);
        assert_eq!(lease.encoded_bytes().unwrap(), expected);
        assert!(std::fs::metadata(&path).unwrap().len() > diagnostics.physical_read_bytes);
    }

    #[test]
    fn one_byte_selection_reads_exactly_once() {
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
        assert_eq!(lease.bounded_read_proof().physical_reads, 1);
        assert_eq!(lease.bounded_read_proof().physical_read_bytes, 1);
        let diagnostics = store.diagnostics().unwrap();
        assert_eq!(diagnostics.physical_reads, 1);
        assert_eq!(diagnostics.physical_read_bytes, 1);
    }

    #[test]
    fn full_tensor_buffers_are_shared_only_while_leased() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.safetensors");
        let payload = [1_u8, 2, 3, 4];
        serialize_to_file(
            [(
                "bytes",
                TensorView::new(Dtype::U8, vec![4], &payload).unwrap(),
            )],
            None,
            &path,
        )
        .unwrap();
        let store = SafetensorsWeightStore::open(&path).unwrap();
        let request = TensorReadRequest {
            key: "bytes".into(),
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        };
        let first = store.acquire(request.clone()).unwrap();
        let retained = Arc::downgrade(&first.bytes);
        let second = store.acquire(request.clone()).unwrap();
        assert!(Arc::ptr_eq(&first.bytes, &second.bytes));
        assert_eq!(second.bounded_read_proof().physical_reads, 0);
        drop(first);
        assert_eq!(second.encoded_bytes().unwrap(), payload);
        assert!(retained.upgrade().is_some());
        drop(second);
        assert!(
            retained.upgrade().is_none(),
            "cached metadata must not retain payloads"
        );
        assert_eq!(store.diagnostics().unwrap().currently_cached_shards, 1);

        let reloaded = store.acquire(request).unwrap();
        assert_eq!(reloaded.encoded_bytes().unwrap(), payload);
        assert_eq!(reloaded.bounded_read_proof().physical_reads, 1);
        let diagnostics = store.diagnostics().unwrap();
        assert_eq!(diagnostics.physical_reads, 2);
        assert_eq!(diagnostics.physical_read_bytes, 8);
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
    fn exact_range_admission_rejects_restored_mtime_mutation_during_read() {
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
        let (payload_offset, _) =
            read_safetensors_metadata(&path, &admitted, MAX_HEADER_BYTES).unwrap();
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
        assert_eq!(diagnostics.physical_reads, 1);
        assert_eq!(diagnostics.physical_read_bytes, 48);
        assert!(!lease.bounded_read_proof().physically_bounded);
        assert_eq!(lease.bounded_read_proof().offset_bytes, 0);
        assert_eq!(lease.bounded_read_proof().length_bytes, 48);
        assert_eq!(lease.bounded_read_proof().physical_reads, 1);
        assert_eq!(lease.bounded_read_proof().physical_read_bytes, 48);
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
        assert_eq!(cached_diagnostics.physical_reads, 1);
        assert_eq!(cached_diagnostics.physical_read_bytes, 48);
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
        assert_eq!(diagnostics.physical_reads, 1);
        assert_eq!(diagnostics.physical_read_bytes, 8);
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
        assert_eq!(
            source.source_metadata_borrowed("selected").unwrap(),
            &source.source_metadata("selected").unwrap()
        );
        assert!(matches!(
            source.source_metadata_borrowed("unselected"),
            Err(SourceMetadataBorrowError::UnauthorizedTensor)
        ));
        assert!(matches!(
            source.source_metadata_borrowed("missing"),
            Err(SourceMetadataBorrowError::UnauthorizedTensor)
        ));
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
    fn restricted_source_denies_exact_keys_without_rebuilding_storage() {
        let source: SharedCheckpointSource = Arc::new(
            MemoryWeightStore::from_safetensors([
                (
                    "target.weight".into(),
                    Dtype::F32,
                    vec![1],
                    f32_bytes(&[1.0]),
                ),
                (
                    "extension.weight".into(),
                    Dtype::F32,
                    vec![1],
                    f32_bytes(&[2.0]),
                ),
            ])
            .unwrap(),
        );
        let restricted = RestrictedCheckpointSource::excluding(
            Arc::clone(&source),
            "prediction-target",
            BTreeSet::from(["extension.weight".into()]),
        )
        .unwrap();

        assert_eq!(restricted.source_keys(), ["target.weight"]);
        assert_eq!(
            restricted.source_provenance("target.weight").unwrap(),
            source.source_provenance("target.weight").unwrap()
        );
        assert!(matches!(
            restricted.source_metadata("extension.weight"),
            Err(StoreError::UnauthorizedTensor { contract, key })
                if contract == "prediction-target" && key == "extension.weight"
        ));
        assert!(matches!(
            restricted.acquire_lease(TensorReadRequest {
                key: "extension.weight".into(),
                selection: TensorSelection::Full,
                policy: ReadPolicy::RequireBounded,
            }),
            Err(StoreError::UnauthorizedTensor { contract, key })
                if contract == "prediction-target" && key == "extension.weight"
        ));
        assert_eq!(
            restricted
                .acquire_lease(TensorReadRequest {
                    key: "target.weight".into(),
                    selection: TensorSelection::Full,
                    policy: ReadPolicy::RequireBounded,
                })
                .unwrap()
                .encoded_bytes()
                .unwrap(),
            f32_bytes(&[1.0])
        );
    }

    #[test]
    fn restricted_source_rejects_unknown_denied_keys() {
        let source: SharedCheckpointSource = Arc::new(
            MemoryWeightStore::from_safetensors([(
                "target.weight".into(),
                Dtype::F32,
                vec![1],
                f32_bytes(&[1.0]),
            )])
            .unwrap(),
        );

        assert!(matches!(
            RestrictedCheckpointSource::excluding(
                source,
                "prediction-target",
                BTreeSet::from(["missing.weight".into()]),
            ),
            Err(StoreError::UnknownTensor { key }) if key == "missing.weight"
        ));
    }

    #[test]
    fn restricted_source_includes_only_the_explicit_projection() {
        let source: SharedCheckpointSource = Arc::new(
            MemoryWeightStore::from_safetensors([
                (
                    "target.weight".into(),
                    Dtype::F32,
                    vec![1],
                    f32_bytes(&[1.0]),
                ),
                (
                    "extension.weight".into(),
                    Dtype::F32,
                    vec![1],
                    f32_bytes(&[2.0]),
                ),
            ])
            .unwrap(),
        );
        let allowed = BTreeSet::from(["extension.weight".into()]);
        let restricted = RestrictedCheckpointSource::including(
            Arc::clone(&source),
            "prediction-extension",
            allowed.clone(),
        )
        .unwrap();

        assert_eq!(restricted.allowed_keys(), Some(&allowed));
        assert_eq!(restricted.source_keys(), ["extension.weight"]);
        assert!(matches!(
            restricted.source_metadata("target.weight"),
            Err(StoreError::UnauthorizedTensor { contract, key })
                if contract == "prediction-extension" && key == "target.weight"
        ));
        assert_eq!(
            restricted
                .acquire_lease(TensorReadRequest {
                    key: "extension.weight".into(),
                    selection: TensorSelection::Full,
                    policy: ReadPolicy::RequireBounded,
                })
                .unwrap()
                .encoded_bytes()
                .unwrap(),
            f32_bytes(&[2.0])
        );
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

#[cfg(test)]
mod header_admission_tests;

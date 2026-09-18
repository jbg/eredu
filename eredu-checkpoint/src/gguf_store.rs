//! Backend-neutral logical GGUF storage and portable encoded leases.

use std::{
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
};

#[cfg(test)]
use std::path::PathBuf;
#[cfg(test)]
use std::collections::{BTreeMap, BTreeSet};

use eredu_gguf::{
    Checkpoint, ConvertedCheckpointTensor, DenseTensorSpan, DenseTensorSpanPlan, LogicalDtype,
    TensorDescriptor, TensorMaterializer, TensorSelection as GgufTensorSelection,
    TensorSelectionPlan,
};

use crate::{
    store::{
        validate_selection, BoundedReadProof, CheckpointLease, CheckpointSource,
        EncodedTensorLease, ReadPolicy, StoreError, TensorMetadata, TensorReadRequest,
        TensorSelection, WeightStore, WeightStoreBackend, WeightStoreDiagnostics,
        DEFAULT_MAX_CACHED_SHARDS,
    },
    validation::{resolve_gguf_plan, ResolvedCheckpointPlan},
    StoredDtype,
};

#[derive(Debug, Clone)]
struct CatalogEntry {
    checkpoint: usize,
    physical_name: String,
    original_name: String,
    metadata: TensorMetadata,
    physical_descriptor: TensorDescriptor,
    logical_last_units_per_block: Option<usize>,
    source_encoding: crate::SourceTensorEncoding,
}

#[derive(Debug, Default)]
struct StoreStatistics {
    physical_reads: AtomicU64,
    physical_read_bytes: AtomicU64,
    coalesced_group_hits: AtomicU64,
}

/// Stable identity of one physical GGUF read and selection in its actual source.
/// Store aliases share an identity; independently constructed stores do not,
/// even when their local shard indices, names, selections or paths coincide.
/// The weak source token does not retain the source value or its nested payload,
/// but keeps its inline Arc allocation reserved until the last identity retires.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct GgufLeaseIdentity {
    source: crate::store::SourceStorageIdentity,
    checkpoint: usize,
    physical_name: String,
    selection: Option<GgufPhysicalSelection>,
}

impl GgufLeaseIdentity {
    /// Inline source payload kept allocated by this identity's weak token.
    /// A qualified Arc layout must add its shared counters and padding. Nested
    /// catalog/recipe/reader allocations are separate source-owned storage.
    /// This is a retained layout fact, not a budget or complete source bound;
    /// cloning identities shares this allocation rather than allocating copies.
    pub const fn source_owner_payload_layout(&self) -> std::alloc::Layout {
        std::alloc::Layout::new::<StoreInner>()
    }

    /// Returns the bounded physical selection, if this is not a full read.
    pub fn physical_selection(&self) -> Option<&GgufPhysicalSelection> {
        self.selection.as_ref()
    }
}

/// Portable physical selection used by a GGUF lease.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub enum GgufPhysicalSelection {
    /// Selection along one logical tensor axis.
    Axis(GgufTensorSelection),
    /// Contiguous scalar span from an unquantized dense tensor.
    DenseSpan(DenseTensorSpan),
}

mod cache_identity;
pub use cache_identity::{
    GgufCacheIdentityView, StoredGgufCacheIdentity, StoredGgufCacheIdentityFailure,
};
mod conversion_plan;
mod reader_buffers;
pub use reader_buffers::{
    GgufReaderBuffers, GgufReaderBuffersCause, GgufReaderBuffersFailure,
    GgufSourcePreparationCause, GgufSourcePreparationFailure,
};
mod touched_paths;
pub use conversion_plan::{GgufConversionPlan, GgufConversionPlanError};
use touched_paths::{ShardCoordinate, TouchedPaths};

#[derive(Debug)]
struct ReaderCache {
    materializers: Vec<TensorMaterializer>,
    buffers: Option<GgufReaderBuffers>,
    last_used: Vec<u64>,
    touched: TouchedPaths,
    tick: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

mod catalog;
mod catalog_compile;
pub use catalog_compile::{
    GgufCatalogCompileFailure, GgufCatalogInput, GgufCatalogPlan, GgufCatalogStorageRequest,
    PreparedGgufCatalog,
};

#[derive(Debug)]
struct StoreInner {
    recipes: crate::recipe::RecipeInferenceCache,
    readers: Mutex<ReaderCache>,
    reader_slots: usize,
    touched_storage_bytes: u64,
    prepared_header_storage_bytes: u64,
    prepared_reader_storage_bytes: Option<u64>,
    prepared_materializer_storage_bytes: Option<u64>,
    max_cached_readers: usize,
    statistics: StoreStatistics,
    // The compiler's Vec<Checkpoint> may become materializer backing through
    // the consuming collect. All readers and that backing retire before its
    // original catalog account; this does not qualify reader construction.
    catalog: catalog::CatalogHandle,
}

/// Builder for a logical store backed by one or more validated GGUF checkpoints.
#[derive(Debug, Default)]
pub struct GgufWeightStoreBuilder {
    checkpoints: Vec<Checkpoint>,
    catalog: catalog::CatalogRows,
    unclaimed_keys: catalog::UnclaimedRows,
    sealed_catalog: Option<catalog::CatalogHandle>,
    max_cached_readers: usize,
}

impl GgufWeightStoreBuilder {
    /// Sets the nonzero maximum number of open GGUF shard readers.
    pub fn max_cached_readers(mut self, maximum: usize) -> Result<Self, StoreError> {
        if maximum == 0 {
            return Err(StoreError::InvalidShardCacheLimit);
        }
        self.max_cached_readers = maximum;
        Ok(self)
    }

    /// Resolves an architecture contract and adds only its selected layout
    /// under the canonical mapping retained by architecture admission.
    pub fn add_checkpoint(
        self,
        checkpoint: Checkpoint,
        plan: &crate::schema::GgufCheckpointPlan,
        tensor_mapping: &[eredu_gguf::TranslatedTensorLayout],
    ) -> Result<Self, StoreError> {
        let resolved =
            resolve_gguf_plan(&checkpoint, plan).map_err(|validation| StoreError::Gguf {
                key: String::new(),
                message: format!(
                    "checkpoint contract {:?} did not resolve: {validation:?}",
                    plan.identity
                ),
            })?;
        self.add_resolved_checkpoint(checkpoint, &resolved, tensor_mapping)
    }

    /// Adds a checkpoint using an already resolved, fail-closed contract and
    /// its admitted canonical tensor mapping.
    pub fn add_resolved_checkpoint(
        mut self,
        checkpoint: Checkpoint,
        resolved: &ResolvedCheckpointPlan,
        tensor_mapping: &[eredu_gguf::TranslatedTensorLayout],
    ) -> Result<Self, StoreError> {
        if let Err(cause) =
            catalog_compile::append_rows(&mut self, &checkpoint, resolved, tensor_mapping)
        {
            return Err(cause.into_store_error(&checkpoint, tensor_mapping));
        }
        self.checkpoints.push(checkpoint);
        Ok(self)
    }

    /// Builds a nonempty immutable logical GGUF store.
    pub fn build(self) -> Result<GgufWeightStore, StoreError> {
        self.build_inner(&mut None)
    }

    fn validate_build(&self) -> Result<(), StoreError> {
        if self.catalog.is_empty() && self.sealed_catalog.is_none() {
            return Err(gguf_error("", "GGUF logical catalog is empty"));
        }
        Ok(())
    }

    fn build_inner(
        self,
        buffers: &mut Option<GgufReaderBuffers>,
    ) -> Result<GgufWeightStore, StoreError> {
        self.validate_build()?;
        let materializers = self
            .checkpoints
            .into_iter()
            .map(|checkpoint| {
                if buffers.is_some() {
                    checkpoint
                        .into_shared_coordinate_materializer()
                        .with_pooled_reader_storage()
                        .expect("new unopened materializer")
                } else {
                    checkpoint.into_shared_materializer()
                }
            })
            .collect::<Vec<_>>();
        let count = materializers.len();
        let touched = TouchedPaths::new(&materializers)?;
        let touched_storage_bytes =
            u64::try_from(touched.capacity_layout()?.size()).map_err(|_| StoreError::Overflow {
                context: "GGUF touched coordinate bytes".into(),
            })?;
        let mut prepared_header_storage_bytes = 0u64;
        for (checkpoint, materializer) in materializers.iter().enumerate() {
            // Mutable scratch belongs to each materializer even when header Arcs alias.
            prepared_header_storage_bytes = materializer
                .prepared_header_scratch_layout()
                .and_then(|layout| u64::try_from(layout.size()).ok())
                .and_then(|bytes| prepared_header_storage_bytes.checked_add(bytes))
                .ok_or_else(|| StoreError::Overflow {
                    context: "GGUF prepared parser scratch".into(),
                })?;
            for (shard_index, shard) in materializer.shards().iter().enumerate() {
                let Some(header) = shard.prepared_header() else {
                    continue;
                };
                // Cloned checkpoints retain the same immutable header Arc. Compare
                // actual owners only while all catalog borrows are retained here.
                let earlier = materializers[..checkpoint]
                    .iter()
                    .flat_map(|m| m.shards())
                    .chain(materializer.shards()[..shard_index].iter());
                if earlier
                    .filter_map(|s| s.prepared_header())
                    .any(|other| std::ptr::eq(header, other))
                {
                    continue;
                }
                prepared_header_storage_bytes = header
                    .retained_payload_bytes()
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .and_then(|bytes| prepared_header_storage_bytes.checked_add(bytes))
                    .ok_or_else(|| StoreError::Overflow {
                        context: "GGUF prepared header buffers".into(),
                    })?;
            }
        }
        let prepared_reader_storage_bytes =
            buffers.as_ref().and_then(GgufReaderBuffers::storage_bytes);
        let mut prepared_materializer_storage_bytes = None;
        Ok(GgufWeightStore {
            inner: crate::store::SourceHandle::new(
                StoreInner {
                    recipes: Default::default(),
                    readers: Mutex::new({
                        let mut cache = ReaderCache {
                            materializers,
                            buffers: buffers.take(),
                            last_used: vec![0; count],
                            touched,
                            tick: 0,
                            hits: 0,
                            misses: 0,
                            evictions: 0,
                        };
                        if cache.buffers.is_some() {
                            match cache.prepared_materializer_storage_bytes() {
                                Ok(bytes) => prepared_materializer_storage_bytes = Some(bytes),
                                Err(error) => {
                                    *buffers = cache.buffers.take();
                                    return Err(error);
                                }
                            }
                        }
                        cache
                    }),
                    reader_slots: count,
                    touched_storage_bytes,
                    prepared_header_storage_bytes,
                    prepared_reader_storage_bytes,
                    prepared_materializer_storage_bytes,
                    max_cached_readers: if self.max_cached_readers == 0 {
                        DEFAULT_MAX_CACHED_SHARDS
                    } else {
                        self.max_cached_readers
                    },
                    statistics: StoreStatistics::default(),
                    catalog: self.sealed_catalog.unwrap_or_else(|| {
                        catalog::CatalogHandle::new(
                            catalog::CatalogData {
                                rows: self.catalog,
                                unclaimed: self.unclaimed_keys,
                            },
                            (),
                        )
                    }),
                },
                None,
            ),
        })
    }
}

/// Opens a backend-neutral GGUF source from an admitted checkpoint, exact
/// architecture contract, and canonical output mapping.
///
/// Construction validates only headers and mappings. Tensor payloads remain
/// lazy and are read through bounded leases after selection.
pub fn open_prepared_gguf_source(
    checkpoint: Checkpoint,
    plan: &crate::schema::GgufCheckpointPlan,
    tensor_mapping: &[eredu_gguf::TranslatedTensorLayout],
    max_cached_readers: usize,
) -> Result<GgufWeightStore, StoreError> {
    GgufWeightStoreBuilder::default()
        .max_cached_readers(max_cached_readers)?
        .add_checkpoint(checkpoint, plan, tensor_mapping)?
        .build()
}

/// Opens an admitted source with its own cold reader bank and physical index.
///
/// The same limit, contract and mapping validation precede reader preparation.
/// This selects existing storage for later requests; it does not authorize the
/// cold allocations or certify a complete source/request memory bound.
pub fn open_prepared_gguf_source_with_reader_buffers(
    checkpoint: Checkpoint,
    plan: &crate::schema::GgufCheckpointPlan,
    tensor_mapping: &[eredu_gguf::TranslatedTensorLayout],
    max_cached_readers: usize,
) -> Result<GgufWeightStore, GgufSourcePreparationFailure> {
    GgufWeightStoreBuilder::default()
        .max_cached_readers(max_cached_readers)
        .and_then(|builder| builder.add_checkpoint(checkpoint, plan, tensor_mapping))
        .map_err(GgufSourcePreparationFailure::validation)?
        .build_with_prepared_reader_buffers()
}

/// Uses the caller's retained resolved contract in the same cold source worker.
/// No semantic resolution is repeated. This ordinary convenience supplies no
/// accounting origin; callers with a catalog account use PreparedGgufCatalog.
pub fn open_resolved_gguf_source_with_reader_buffers(
    checkpoint: Checkpoint,
    resolved: &ResolvedCheckpointPlan,
    tensor_mapping: &[eredu_gguf::TranslatedTensorLayout],
    max_cached_readers: usize,
) -> Result<GgufWeightStore, GgufSourcePreparationFailure> {
    GgufWeightStoreBuilder::default()
        .max_cached_readers(max_cached_readers)
        .and_then(|builder| builder.add_resolved_checkpoint(checkpoint, resolved, tensor_mapping))
        .map_err(GgufSourcePreparationFailure::validation)?
        .build_with_prepared_reader_buffers()
}

/// Persistent backend-neutral logical tensor store for GGUF checkpoints.
#[derive(Debug, Clone)]
pub struct GgufWeightStore {
    inner: crate::store::SourceHandle<StoreInner>,
}

impl GgufWeightStore {
    pub(crate) fn catalog_keys(&self) -> impl Iterator<Item = &String> {
        self.inner.catalog.keys()
    }

    /// Borrow the private origin of this exact source constructor. It supplies
    /// no raw custody, amount or promotion of an ordinary source.
    pub fn source_control_owner<C: std::any::Any>(&self) -> Option<&C> {
        self.inner.origin()
    }

    // Named reader/touched buffers, actual immutable header String/Vec backing
    // and independent parser scratch. Map/Arc, allocator/OS and cold overlap remain separate.
    fn source_storage_bytes(&self) -> Result<u64, StoreError> {
        self.reader_storage_bytes()?
            .checked_add(self.inner.touched_storage_bytes)
            .and_then(|bytes| bytes.checked_add(self.inner.prepared_header_storage_bytes))
            .and_then(|bytes| {
                bytes.checked_add(self.inner.prepared_materializer_storage_bytes.unwrap_or(0))
            })
            .ok_or_else(|| StoreError::Overflow {
                context: "GGUF source buffer and coordinate bytes".into(),
            })
    }

    /// Actual owned reader bank and buffer capacities for the explicit prepared
    /// source, including active readers. Cold construction and other source
    /// controls remain separate qualification obligations.
    pub fn prepared_reader_storage_bytes(&self) -> Option<u64> {
        self.inner.prepared_reader_storage_bytes
    }

    /// Actual prepared coordinate index and materializer/LRU vector backing.
    /// Reader buffers and header scratch are reported separately, once; this
    /// does not qualify cold allocation or the remaining source controls.
    pub fn prepared_materializer_storage_bytes(&self) -> Option<u64> {
        self.inner.prepared_materializer_storage_bytes
    }

    fn reader_storage_bytes(&self) -> Result<u64, StoreError> {
        if let Some(bytes) = self.inner.prepared_reader_storage_bytes {
            return Ok(bytes);
        }
        // ReaderCache closes a victim before opening its replacement. Reserve
        // the entire configured ceiling even before any reader is opened.
        u64::try_from(self.inner.max_cached_readers.min(self.inner.reader_slots))
            .ok()
            .and_then(|count| {
                count.checked_mul(
                    eredu_gguf::Reader::<std::io::BufReader<std::fs::File>>::file_buffer_capacity()
                        as u64,
                )
            })
            .ok_or_else(|| StoreError::Overflow {
                context: "GGUF source reader buffers".into(),
            })
    }

    /// Borrow the actual constructor's concrete catalog custody, if present.
    /// This gives no storage amount, clone, mutable access or source authority.
    /// Ordinary catalogs retain `()`; callers must recognize their own closed
    /// accounting type and validate its original domain, never infer coverage
    /// merely from this store or a successfully borrowed catalog.
    pub fn catalog_control_owner<C: std::any::Any>(&self) -> Option<&C> {
        self.inner.catalog.origin()
    }

    /// Starts a multi-checkpoint store builder.
    pub fn builder() -> GgufWeightStoreBuilder {
        GgufWeightStoreBuilder::default()
    }

    /// Returns catalog keys admitted but unclaimed by a non-strict contract.
    pub fn unclaimed_checkpoint_keys(&self) -> Vec<String> {
        self.inner
            .catalog
            .data()
            .unclaimed
            .iter()
            .cloned()
            .collect()
    }
}

/// Encoded GGUF lease whose portable payload is realized only on request.
#[derive(Debug, Clone)]
pub struct GgufLease {
    store: crate::store::SourceHandle<StoreInner>,
    entry: CatalogEntry,
    selection: TensorSelection,
    output_shape: Vec<usize>,
    proof: BoundedReadProof,
    identity: GgufLeaseIdentity,
    selection_is_materialized: bool,
}

impl GgufLease {
    /// Stable physical group identity used for backend-side coalescing.
    pub fn identity(&self) -> &GgufLeaseIdentity {
        &self.identity
    }

    /// Logical output name within the converted physical group.
    pub fn logical_output_name(&self) -> &str {
        &self.entry.original_name
    }

    /// Whether the portable read already applies the requested selection.
    pub fn selection_is_materialized(&self) -> bool {
        self.selection_is_materialized
    }

    /// Records that a backend reused a previously converted physical group.
    pub fn record_coalesced_group_hit(&self) {
        self.store
            .statistics
            .coalesced_group_hits
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Reads and converts the selected physical payload into portable buffers.
    pub fn materialize_portable(&self) -> Result<ConvertedCheckpointTensor, StoreError> {
        read_destination::materialize(self, None).map_err(read_destination::Cause::ordinary)
    }
}

impl EncodedTensorLease for GgufLease {
    fn metadata(&self) -> &TensorMetadata {
        &self.entry.metadata
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
        self.entry.metadata.backing_shard.as_deref()
    }

    fn encoded_bytes(&self) -> Option<&[u8]> {
        None
    }
}

impl WeightStore for GgufWeightStore {
    fn recipe_cache(&self) -> Option<&crate::recipe::RecipeInferenceCache> {
        Some(&self.inner.recipes)
    }

    type Lease = GgufLease;

    fn keys(&self) -> Vec<String> {
        self.inner.catalog.keys().cloned().collect()
    }

    fn metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.inner
            .catalog
            .get(key)
            .map(|entry| entry.metadata.clone())
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
    }

    fn acquire(&self, request: TensorReadRequest) -> Result<GgufLease, StoreError> {
        lease_destination::acquire(self, request)
    }

    fn diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        let readers = self
            .inner
            .readers
            .lock()
            .map_err(|_| StoreError::Internal("GGUF reader cache is poisoned".into()))?;
        Ok(WeightStoreDiagnostics {
            backend: WeightStoreBackend::Gguf,
            cache_hits: readers.hits,
            cache_misses: readers.misses,
            evictions: readers.evictions,
            currently_cached_shards: readers
                .materializers
                .iter()
                .filter(|materializer| materializer.open_shard_path().is_some())
                .count(),
            touched_shard_paths: readers
                .touched
                .paths(&readers.materializers)
                .map(Path::to_path_buf)
                .collect(),
            payload_shard_paths: readers
                .touched
                .paths(&readers.materializers)
                .map(Path::to_path_buf)
                .collect(),
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

mod lease_controls;
mod lease_destination;
mod read_destination;
pub(crate) use lease_destination::Prepared as PreparedGgufLease;
pub use read_destination::{
    GgufRawStorage, GgufRawStorageProvider, GgufRawStorageRequest, GgufStorageProvider,
    PreparedGgufBoxedFailure, PreparedGgufConversion, PreparedGgufConversionFailure,
    PreparedGgufRead, PreparedGgufReadFailure, PreparedGgufSuppliedFailure, PreparedGgufTensor,
    PreparedGgufTensorFailure, StoredGgufFailure,
};

impl CheckpointSource for GgufWeightStore {
    fn prepared_acquisition_owner(
        self: Arc<Self>,
    ) -> Option<crate::store::PreparedAcquisitionOwner> {
        Some(crate::store::PreparedAcquisitionOwner::gguf(self))
    }

    fn prepared_acquisition_source(&self) -> crate::store::PreparedAcquisitionSource<'_> {
        crate::store::PreparedAcquisitionSource::gguf(self)
    }

    fn source_lease_controls<'a>(
        &'a self,
        key: &'a str,
    ) -> Result<crate::store::SourceLeaseControls<'a>, crate::store::LeaseControlBorrowError<'a>>
    {
        lease_controls::source(self, key)
    }

    fn source_metadata_borrowed(&self, key: &str) -> crate::store::SourceMetadataLoan<'_> {
        self.inner
            .catalog
            .get(key)
            .map(|entry| &entry.metadata)
            .ok_or(crate::store::SourceMetadataBorrowError::UnknownTensor)
    }
    fn source_key_authority_borrowed(
        &self,
        _: &str,
    ) -> Result<crate::store::SourceKeyAuthority, crate::store::SourceMetadataBorrowError<'_>> {
        Ok(crate::store::SourceKeyAuthority::Ordinary)
    }

    fn source_storage_slot_bound(&self) -> Result<Option<usize>, StoreError> {
        Ok(Some(1))
    }

    fn visit_source_storage(
        &self,
        visitor: &mut dyn FnMut(crate::store::SourceStorageRef<'_>),
    ) -> Result<bool, StoreError> {
        visitor(self.inner.storage_ref(self.source_storage_bytes()?));
        Ok(true)
    }

    fn source_storage(&self) -> Result<Option<crate::store::SourceStorage>, StoreError> {
        let bytes = self.source_storage_bytes()?;
        let mut storage = crate::store::SourceStorage::default();
        storage.insert_retained(self.inner.storage_ref(bytes).retain())?;
        Ok(Some(storage))
    }

    fn recipe_cache(&self) -> Option<&crate::recipe::RecipeInferenceCache> {
        Some(&self.inner.recipes)
    }

    fn source_keys(&self) -> Vec<String> {
        WeightStore::keys(self)
    }

    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        WeightStore::metadata(self, key)
    }

    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        WeightStore::acquire(self, request).map(|lease| CheckpointLease::Gguf(Box::new(lease)))
    }

    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        WeightStore::diagnostics(self)
    }

    fn source_provenance(
        &self,
        key: &str,
    ) -> Result<crate::store::TensorSourceProvenance, StoreError> {
        let entry = self
            .inner
            .catalog
            .get(key)
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })?;
        Ok(crate::store::TensorSourceProvenance {
            catalog_key: key.to_owned(),
            physical_tensor: entry.physical_name.clone(),
            output: entry.original_name.clone(),
            backing_shard: entry.metadata.backing_shard.clone(),
            source_encoding: entry.source_encoding.clone(),
        })
    }

    fn unclaimed_checkpoint_keys(&self) -> Vec<String> {
        self.unclaimed_checkpoint_keys()
    }

    fn is_checkpoint_contract_resolved(&self) -> bool {
        true
    }
}

impl ReaderCache {
    fn materialize_with_destination(
        &mut self,
        checkpoint: usize,
        physical_name: &str,
        selection: Option<&GgufPhysicalSelection>,
        maximum: usize,
        logical_key: &str,
        raw: Option<&mut [u8]>,
        conversion: Option<&mut eredu_gguf::PreparedConversion>,
        metadata: Option<&mut eredu_gguf::PreparedTensorMetadata>,
    ) -> Result<ConvertedCheckpointTensor, read_destination::Cause> {
        self.materialize_worker(
            checkpoint,
            physical_name,
            maximum,
            logical_key,
            |materializer| {
                if let Some(metadata) = metadata {
                    let raw = raw.expect("complete metadata is paired with its raw owner");
                    let conversion =
                        conversion.expect("complete metadata is paired with conversion storage");
                    let selection = match selection {
                        None => eredu_gguf::MetadataSelection::Full,
                        Some(GgufPhysicalSelection::Axis(s)) => {
                            eredu_gguf::MetadataSelection::Axis(s)
                        }
                        Some(GgufPhysicalSelection::DenseSpan(s)) => {
                            eredu_gguf::MetadataSelection::Span(s)
                        }
                    };
                    materializer.converted_tensor_with_metadata(
                        physical_name,
                        selection,
                        raw,
                        conversion,
                        metadata,
                    )
                } else if let Some(conversion) = conversion {
                    let raw = raw.expect("prepared conversion is paired with its raw owner");
                    match selection {
                        Some(GgufPhysicalSelection::Axis(selection)) => materializer
                            .converted_tensor_selected_with_destinations(
                                physical_name,
                                selection,
                                raw,
                                conversion,
                            ),
                        Some(GgufPhysicalSelection::DenseSpan(selection)) => materializer
                            .converted_dense_tensor_span_with_destinations(
                                physical_name,
                                selection,
                                raw,
                                conversion,
                            ),
                        None => materializer.converted_tensor_with_destinations(
                            physical_name,
                            raw,
                            conversion,
                        ),
                    }
                } else {
                    match (selection, raw) {
                        (Some(GgufPhysicalSelection::Axis(selection)), None) => materializer
                            .converted_tensor_selected(physical_name, selection)
                            .map_err(eredu_gguf::ReadDestinationError::from),
                        (Some(GgufPhysicalSelection::DenseSpan(selection)), None) => materializer
                            .converted_dense_tensor_span(physical_name, selection)
                            .map_err(eredu_gguf::ReadDestinationError::from),
                        (None, None) => materializer
                            .converted_tensor(physical_name)
                            .map_err(eredu_gguf::ReadDestinationError::from),
                        (Some(GgufPhysicalSelection::Axis(selection)), Some(raw)) => materializer
                            .converted_tensor_selected_with_raw_destination(
                                physical_name,
                                selection,
                                raw,
                            ),
                        (Some(GgufPhysicalSelection::DenseSpan(selection)), Some(raw)) => {
                            materializer.converted_dense_tensor_span_with_raw_destination(
                                physical_name,
                                selection,
                                raw,
                            )
                        }
                        (None, Some(raw)) => {
                            materializer.converted_tensor_with_raw_destination(physical_name, raw)
                        }
                    }
                }
            },
        )
    }
    fn materialize_worker<T>(
        &mut self,
        checkpoint: usize,
        physical_name: &str,
        maximum: usize,
        logical_key: &str,
        execute: impl FnOnce(&mut TensorMaterializer) -> Result<T, eredu_gguf::ReadDestinationError>,
    ) -> Result<T, read_destination::Cause> {
        let materializer = self
            .materializers
            .get(checkpoint)
            .ok_or_else(|| gguf_error(logical_key, "catalog references an unknown checkpoint"))?;
        let (shard, target_path) = materializer
            .shard_source_for_tensor(physical_name)
            .map_err(|error| gguf_error(logical_key, error))?;
        let touched_index = self.touched.find(&self.materializers, target_path);
        let coordinate = ShardCoordinate { checkpoint, shard };
        let reader_hit = materializer
            .open_shard_path()
            .is_some_and(|path| path == target_path);
        self.tick = self.tick.saturating_add(1);
        if reader_hit {
            self.hits = self.hits.saturating_add(1);
        } else {
            self.misses = self.misses.saturating_add(1);
            if self.materializers[checkpoint].close_reader_without_path() {
                self.evictions = self.evictions.saturating_add(1);
            }
            self.recycle_reader_buffer(checkpoint);
            if self
                .materializers
                .iter()
                .filter(|materializer| materializer.open_shard_path().is_some())
                .count()
                >= maximum
            {
                let victim = self
                    .materializers
                    .iter()
                    .enumerate()
                    .filter(|(_, materializer)| materializer.open_shard_path().is_some())
                    .min_by_key(|(index, _)| (self.last_used[*index], *index))
                    .map(|(index, _)| index)
                    .expect("an open reader exists at the configured bound");
                self.materializers[victim].close_reader_without_path();
                self.recycle_reader_buffer(victim);
                self.evictions = self.evictions.saturating_add(1);
            }
        }
        self.last_used[checkpoint] = self.tick;
        if !reader_hit {
            self.supply_reader_buffer(checkpoint, logical_key)?;
        }
        let converted = execute(&mut self.materializers[checkpoint]);
        // Open/parse refusal may have recycled its actual buffer. Return it
        // before error transport; successful open buffers stay with their File.
        self.recycle_reader_buffer(checkpoint);
        let converted =
            converted.map_err(|error| read_destination::Cause::read(error, logical_key))?;
        self.touched.mark(touched_index, coordinate);
        Ok(converted)
    }
}

struct ReadPlan {
    physical_selection: Option<GgufPhysicalSelection>,
    physical_offset: u64,
    physical_byte_len: u64,
    selection_is_materialized: bool,
}

fn plan_bounded_selection(
    key: &str,
    entry: &CatalogEntry,
    selection: &TensorSelection,
) -> Result<ReadPlan, StoreError> {
    if matches!(selection, TensorSelection::Full) {
        return Ok(ReadPlan {
            physical_selection: None,
            physical_offset: 0,
            physical_byte_len: entry.physical_descriptor.byte_len,
            selection_is_materialized: true,
        });
    }
    if let TensorSelection::Contiguous {
        offset_elements,
        shape,
    } = selection
    {
        let logical_offset = u64::try_from(*offset_elements).map_err(|_| StoreError::Overflow {
            context: format!("GGUF contiguous offset for tensor {key:?}"),
        })?;
        let mut physical_shape = shape
            .iter()
            .map(|dimension| {
                u64::try_from(*dimension).map_err(|_| StoreError::Overflow {
                    context: format!("GGUF contiguous shape for tensor {key:?}"),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let logical_units = u64::try_from(entry.logical_last_units_per_block.ok_or_else(|| {
            bounded_error(key, "scalar GGUF output has no contiguous block mapping")
        })?)
        .map_err(|_| StoreError::Overflow {
            context: format!("GGUF logical block units for tensor {key:?}"),
        })?;
        let (block_values, _) = entry
            .physical_descriptor
            .ggml_type
            .block_and_bytes()
            .map_err(|error| gguf_error(key, error))?;
        let logical_elements = physical_shape.iter().try_fold(1u64, |count, dimension| {
            count.checked_mul(*dimension).ok_or(StoreError::Overflow {
                context: format!("GGUF contiguous element count for tensor {key:?}"),
            })
        })?;
        let logical_end =
            logical_offset
                .checked_add(logical_elements)
                .ok_or(StoreError::Overflow {
                    context: format!("GGUF contiguous logical end for tensor {key:?}"),
                })?;
        if !logical_offset.is_multiple_of(logical_units)
            || !logical_elements.is_multiple_of(logical_units)
        {
            return Err(bounded_error(
                key,
                format!(
                    "contiguous logical span {logical_offset}..{logical_end} must align to {logical_units} converted units per native block"
                ),
            ));
        }
        let physical_last = physical_shape.last_mut().ok_or_else(|| {
            bounded_error(key, "contiguous GGUF selection has no packed input axis")
        })?;
        if !physical_last.is_multiple_of(logical_units) {
            return Err(bounded_error(
                key,
                "contiguous GGUF row must contain complete converted blocks",
            ));
        }
        *physical_last = physical_last
            .checked_div(logical_units)
            .and_then(|blocks| blocks.checked_mul(block_values))
            .ok_or(StoreError::Overflow {
                context: format!("GGUF contiguous physical shape for tensor {key:?}"),
            })?;
        let physical_offset = logical_offset
            .checked_div(logical_units)
            .and_then(|blocks| blocks.checked_mul(block_values))
            .ok_or(StoreError::Overflow {
                context: format!("GGUF contiguous physical offset for tensor {key:?}"),
            })?;
        let selection = DenseTensorSpan::new(physical_offset, physical_shape)
            .map_err(|error| bounded_error(key, error))?;
        let plan = DenseTensorSpanPlan::new(&entry.physical_descriptor, selection.clone())
            .map_err(|error| bounded_error(key, error))?;
        let physical_offset = plan
            .encoded_span()
            .offset()
            .checked_sub(entry.physical_descriptor.data_offset)
            .ok_or_else(|| StoreError::Overflow {
                context: format!("GGUF relative physical offset for tensor {key:?}"),
            })?;
        return Ok(ReadPlan {
            physical_selection: Some(GgufPhysicalSelection::DenseSpan(selection)),
            physical_offset,
            physical_byte_len: plan.encoded_byte_len(),
            selection_is_materialized: true,
        });
    }
    let rank = entry.metadata.logical_shape.len();
    let logical_axis = match selection {
        TensorSelection::Range { axis, .. } | TensorSelection::Indices { axis, .. } => *axis,
        TensorSelection::Full | TensorSelection::Contiguous { .. } => unreachable!(),
    };
    let physical_selection = if logical_axis + 1 != rank {
        match selection {
            TensorSelection::Range { axis, start, end } => GgufTensorSelection::Range {
                axis: *axis,
                start: *start,
                end: *end,
            },
            TensorSelection::Indices { axis, indices } => GgufTensorSelection::Indices {
                axis: *axis,
                indices: indices.clone(),
            },
            TensorSelection::Full | TensorSelection::Contiguous { .. } => unreachable!(),
        }
    } else {
        map_innermost_selection(key, entry, selection)?
    };
    let plan = TensorSelectionPlan::new(&entry.physical_descriptor, physical_selection.clone())
        .map_err(|error| bounded_error(key, error))?;
    let physical_offset = plan
        .encoded_spans()
        .next()
        .expect("nonempty GGUF selections have at least one encoded span")
        .offset()
        .checked_sub(entry.physical_descriptor.data_offset)
        .ok_or_else(|| StoreError::Overflow {
            context: format!("GGUF relative physical offset for tensor {key:?}"),
        })?;
    Ok(ReadPlan {
        physical_selection: Some(GgufPhysicalSelection::Axis(physical_selection)),
        physical_offset,
        physical_byte_len: plan.encoded_byte_len(),
        selection_is_materialized: true,
    })
}

fn map_innermost_selection(
    key: &str,
    entry: &CatalogEntry,
    selection: &TensorSelection,
) -> Result<GgufTensorSelection, StoreError> {
    let logical_units = entry
        .logical_last_units_per_block
        .ok_or_else(|| bounded_error(key, "scalar GGUF output has no selectable innermost axis"))?;
    let (block_values, _) = entry
        .physical_descriptor
        .ggml_type
        .block_and_bytes()
        .map_err(|error| gguf_error(key, error))?;
    let block_values = usize::try_from(block_values).map_err(|_| StoreError::Overflow {
        context: format!("GGUF block length for tensor {key:?}"),
    })?;
    let axis = entry.metadata.logical_shape.len() - 1;
    match selection {
        TensorSelection::Range { start, end, .. } => {
            if start % logical_units != 0 || end % logical_units != 0 {
                return Err(bounded_error(
                    key,
                    format!(
                        "logical innermost range {start}..{end} must align to {logical_units} converted units per native GGUF block"
                    ),
                ));
            }
            Ok(GgufTensorSelection::Range {
                axis,
                start: (start / logical_units)
                    .checked_mul(block_values)
                    .ok_or_else(|| StoreError::Overflow {
                        context: format!("GGUF physical selection start for tensor {key:?}"),
                    })?,
                end: (end / logical_units)
                    .checked_mul(block_values)
                    .ok_or_else(|| StoreError::Overflow {
                        context: format!("GGUF physical selection end for tensor {key:?}"),
                    })?,
            })
        }
        TensorSelection::Indices { indices, .. } => {
            if !indices.len().is_multiple_of(logical_units) {
                return Err(bounded_error(
                    key,
                    format!(
                        "logical innermost indices must contain complete {logical_units}-unit converted GGUF blocks"
                    ),
                ));
            }
            let physical_count = (indices.len() / logical_units)
                .checked_mul(block_values)
                .ok_or_else(|| StoreError::Overflow {
                    context: format!("GGUF physical index count for tensor {key:?}"),
                })?;
            let mut physical_indices = Vec::new();
            physical_indices
                .try_reserve_exact(physical_count)
                .map_err(|_| StoreError::Overflow {
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
                    return Err(bounded_error(
                        key,
                        format!(
                            "logical innermost indices must preserve every complete aligned {logical_units}-unit converted GGUF block"
                        ),
                    ));
                }
                let physical_start = (logical_start / logical_units)
                    .checked_mul(block_values)
                    .ok_or_else(|| StoreError::Overflow {
                        context: format!("GGUF physical index start for tensor {key:?}"),
                    })?;
                physical_indices.extend(physical_start..physical_start + block_values);
            }
            Ok(GgufTensorSelection::Indices {
                axis,
                indices: physical_indices,
            })
        }
        TensorSelection::Full | TensorSelection::Contiguous { .. } => unreachable!(),
    }
}

fn stored_dtype(dtype: LogicalDtype) -> StoredDtype {
    match dtype {
        LogicalDtype::F32 => StoredDtype::F32,
        LogicalDtype::F16 => StoredDtype::F16,
        LogicalDtype::Bf16 => StoredDtype::BF16,
        LogicalDtype::I8 => StoredDtype::I8,
        LogicalDtype::I16 => StoredDtype::I16,
        LogicalDtype::U8 => StoredDtype::U8,
        LogicalDtype::U32 => StoredDtype::U32,
        LogicalDtype::I32 => StoredDtype::I32,
        LogicalDtype::I64 => StoredDtype::I64,
        LogicalDtype::F64 => StoredDtype::F64,
    }
}

fn gguf_read_error(key: &str, error: eredu_gguf::Error) -> StoreError {
    if error.prepared_header_change().is_some() {
        StoreError::GgufPreparedHeaderChanged {
            key: key.into(),
            source: Arc::new(error),
        }
    } else if error.prepared_reader_storage().is_some() {
        StoreError::GgufPreparedReaderStorage {
            key: key.into(),
            source: Arc::new(error),
        }
    } else if error.prepared_header_storage().is_some() {
        StoreError::GgufPreparedHeaderStorage {
            key: key.into(),
            source: Arc::new(error),
        }
    } else {
        gguf_error(key, error)
    }
}

fn gguf_error(key: impl Into<String>, message: impl ToString) -> StoreError {
    StoreError::Gguf {
        key: key.into(),
        message: message.to_string(),
    }
}

fn bounded_error(key: &str, message: impl ToString) -> StoreError {
    StoreError::BoundedSelectionUnavailable {
        key: key.into(),
        message: message.to_string(),
    }
}

#[cfg(test)]
mod tests {
    mod cache_controls;
    mod cache_identity;
    mod conversion_plan;
    mod lease_controls;
    mod lease_destination;
    mod prepared_headers;
    mod read_destination;
    mod reference_cache;
    use reference_cache::{ReferenceCache, ReferenceLease, ReferenceStore};
    use std::{collections::BTreeMap, fs::File};

    use eredu_gguf::{ConvertedTensor, GgmlType, TensorInput, Writer};

    use super::*;
    use crate::recipe::DerivedWeightRecipe;
    use crate::schema::{
        CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint,
        TensorOperation,
    };

    fn write_tensor(path: &Path, name: &str, dimensions: &[u64], ty: GgmlType, data: &[u8]) {
        Writer::default()
            .write(
                File::create(path).unwrap(),
                &BTreeMap::new(),
                &[TensorInput {
                    name,
                    dimensions,
                    ggml_type: ty,
                    data,
                }],
            )
            .unwrap();
    }

    pub(super) fn test_plan(checkpoint: &Checkpoint) -> GgufCheckpointPlan {
        let constraints = checkpoint
            .tensors()
            .map(|tensor| {
                let descriptor = tensor.descriptor();
                let operation = match descriptor.ggml_type {
                    GgmlType::I32 => TensorOperation::I32,
                    GgmlType::MxFp4 => TensorOperation::MxFp4Matrix,
                    GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16 => TensorOperation::Dense,
                    _ => TensorOperation::Matrix,
                };
                GgufTensorConstraint::required(
                    descriptor.name.clone(),
                    descriptor
                        .row_major_shape()
                        .into_iter()
                        .map(|dimension| usize::try_from(dimension).unwrap())
                        .collect::<Vec<_>>(),
                    GgufTypeConstraint::OperationClass(operation),
                )
            })
            .collect();
        GgufCheckpointPlan::new(
            "test GGUF catalog",
            constraints,
            Vec::new(),
            CatalogPolicy::strict(),
        )
        .unwrap()
    }

    pub(super) fn test_store(path: &Path) -> GgufWeightStore {
        let checkpoint = Checkpoint::open(path).unwrap();
        let plan = test_plan(&checkpoint);
        let tensor_mapping = checkpoint.translated_outputs(str::to_owned).unwrap();
        GgufWeightStore::builder()
            .add_checkpoint(checkpoint, &plan, &tensor_mapping)
            .unwrap()
            .build()
            .unwrap()
    }

    #[test]
    fn borrowed_source_storage_visits_reader_ceiling_without_io_and_retains_exact_owner() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("borrowed.gguf");
        write_tensor(
            &path,
            "projection.weight",
            &[2],
            GgmlType::F32,
            &[1.25_f32.to_le_bytes(), (-3.5_f32).to_le_bytes()].concat(),
        );
        let source = test_store(&path);
        let metadata = source
            .source_metadata_borrowed("projection.weight")
            .unwrap();
        assert!(std::ptr::eq(
            metadata,
            &source.inner.catalog["projection.weight"].metadata
        ));
        let layout = crate::store::SelectedMetadataCloneLayout::for_selection(
            metadata,
            &TensorSelection::Full,
        )
        .unwrap();
        let lease = source
            .acquire(TensorReadRequest {
                key: "projection.weight".into(),
                selection: TensorSelection::Full,
                policy: ReadPolicy::RequireBounded,
            })
            .unwrap();
        assert_eq!(
            layout,
            crate::store::SelectedMetadataCloneLayout::from_lease(&lease).unwrap()
        );
        assert!(!std::ptr::eq(metadata, lease.metadata())); // GGUF owns its catalog clone
        assert_eq!(metadata, lease.metadata());
        drop(lease);
        let weak = source.inner.ordinary_weak();
        let legacy = source.source_storage().unwrap().unwrap();
        let (identity, expected) = legacy.capacities().next().unwrap();
        assert_eq!(expected, source.source_storage_bytes().unwrap());
        assert!(expected > 0);
        let before = source.source_diagnostics().unwrap();
        assert_eq!(before.physical_reads, 0);
        std::fs::remove_file(path).unwrap();
        assert!(std::ptr::eq(
            metadata,
            source
                .source_metadata_borrowed("projection.weight")
                .unwrap()
        ));
        assert!(matches!(
            source.source_metadata_borrowed("missing"),
            Err(crate::store::SourceMetadataBorrowError::UnknownTensor)
        ));
        let mut retained = None;
        let source_view: &dyn CheckpointSource = &source;
        assert!(source_view
            .visit_source_storage(&mut |owner| {
                assert!(retained.is_none());
                assert_eq!(owner.identity(), identity);
                assert_eq!(owner.bytes(), expected);
                retained = Some(owner.retain());
            })
            .unwrap());
        let after = source.source_diagnostics().unwrap();
        assert_eq!(after.physical_reads, before.physical_reads);
        assert_eq!(after.physical_read_bytes, before.physical_read_bytes);
        assert_eq!(after.cache_misses, before.cache_misses);
        assert_eq!(
            after.currently_cached_shards,
            before.currently_cached_shards
        );
        drop((legacy, source));
        assert!(weak.upgrade().is_some());
        assert_eq!(retained.as_ref().unwrap().identity(), identity);
        drop(retained);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn source_storage_reserves_reader_ceiling_without_reading_or_reopening_payloads() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("dense.gguf");
        write_tensor(
            &path,
            "projection.weight",
            &[1],
            GgmlType::F32,
            &1.0_f32.to_le_bytes(),
        );
        let source = test_store(&path);
        let storage = source.source_storage().unwrap().unwrap();
        let expected = eredu_gguf::Reader::<std::io::BufReader<File>>::file_buffer_capacity()
            as u64
            + source.inner.touched_storage_bytes;
        assert_eq!(storage.bytes().unwrap(), expected);
        assert_eq!(storage.owner_count(), 1);
        assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
        let lease = source
            .acquire_lease(TensorReadRequest {
                key: "projection.weight".into(),
                selection: TensorSelection::Full,
                policy: ReadPolicy::RequireBounded,
            })
            .unwrap();
        let CheckpointLease::Gguf(lease) = lease else {
            panic!("GGUF lease")
        };
        let converted = lease.materialize_portable().unwrap();
        let reads = source.source_diagnostics().unwrap().physical_reads;
        assert!(reads > 0);
        std::fs::remove_file(path).unwrap();
        let shared =
            crate::store::SourceStorage::collect([&source as &dyn CheckpointSource, &source])
                .unwrap()
                .unwrap();
        assert_eq!(shared.bytes().unwrap(), expected);
        assert_eq!(source.source_diagnostics().unwrap().physical_reads, reads);
        drop(converted);
        drop(lease);
    }

    #[test]
    fn store_requires_the_admitted_mapping_for_every_catalog_output() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("dense.gguf");
        write_tensor(
            &path,
            "projection.weight",
            &[1],
            GgmlType::F32,
            &1.0_f32.to_le_bytes(),
        );
        let checkpoint = Checkpoint::open(path).unwrap();
        let plan = test_plan(&checkpoint);

        let error = GgufWeightStore::builder()
            .add_checkpoint(checkpoint, &plan, &[])
            .unwrap_err();
        assert!(matches!(
            error,
            StoreError::Gguf { message, .. }
                if message.contains("mapping omits a catalog output")
        ));
    }

    #[test]
    fn catalog_and_acquisition_do_not_read_payloads() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("dense.gguf");
        let values = (0..8)
            .flat_map(|value| (value as f32).to_le_bytes())
            .collect::<Vec<_>>();
        write_tensor(&path, "matrix.weight", &[4, 2], GgmlType::F32, &values);
        let store = test_store(&path);

        assert_eq!(store.keys(), ["matrix.weight"]);
        assert_eq!(
            store.metadata("matrix.weight").unwrap().logical_shape,
            [2, 4]
        );
        let lease = store
            .acquire(TensorReadRequest {
                key: "matrix.weight".into(),
                selection: TensorSelection::Range {
                    axis: 0,
                    start: 0,
                    end: 1,
                },
                policy: ReadPolicy::RequireBounded,
            })
            .unwrap();
        assert!(lease.bounded_read_proof().physically_bounded);
        assert_eq!(lease.bounded_read_proof().length_bytes, 16);
        assert_eq!(store.diagnostics().unwrap().physical_reads, 0);

        let portable = lease.materialize_portable().unwrap();
        let ConvertedTensor::Dense(dense) = portable.into_converted() else {
            panic!("expected a dense portable tensor");
        };
        assert_eq!(dense.shape, [1, 4]);
        assert_eq!(dense.data.len(), 16);
        let diagnostics = store.diagnostics().unwrap();
        assert_eq!(diagnostics.physical_reads, 1);
        assert_eq!(diagnostics.physical_read_bytes, 16);
    }

    #[test]
    fn native_blocks_remain_packed_in_portable_lease() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("native.gguf");
        write_tensor(&path, "matrix.weight", &[32], GgmlType::Q8_0, &[0; 34]);
        let store = test_store(&path);
        let lease = store
            .acquire(TensorReadRequest {
                key: "matrix.weight".into(),
                selection: TensorSelection::Full,
                policy: ReadPolicy::RequireBounded,
            })
            .unwrap();

        let portable = lease.materialize_portable().unwrap();
        let ConvertedTensor::IQuant(native) = portable.into_converted() else {
            panic!("native GGUF blocks were expanded before backend materialization");
        };
        assert_eq!(native.ggml_type, GgmlType::Q8_0);
        assert_eq!(native.data.len(), 34);
    }

    #[test]
    fn native_byte_blocks_support_contiguous_expert_row_selection() {
        for ty in [GgmlType::IQ4NL, GgmlType::Q8_0] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("bank.gguf");
            let (_, block_bytes) = ty.block_and_bytes().unwrap();
            let block_bytes = block_bytes as usize;
            let raw = (0..16)
                .flat_map(|block| {
                    let mut bytes = vec![0, 60]; // finite f16 scale 1
                    bytes.extend((2..block_bytes).map(|i| ((i + block) % 127) as u8));
                    bytes
                })
                .collect::<Vec<_>>();
            write_tensor(&path, "bank.weight", &[64, 4, 2], ty, &raw);
            let store = test_store(&path);
            let offset = 8 * block_bytes;
            let lease = store
                .acquire(TensorReadRequest {
                    key: "bank.weight".into(),
                    selection: TensorSelection::Contiguous {
                        offset_elements: offset,
                        shape: vec![1, 2, 2 * block_bytes],
                    },
                    policy: ReadPolicy::RequireBounded,
                })
                .unwrap();
            assert_eq!(
                lease.bounded_read_proof().length_bytes,
                (4 * block_bytes) as u64
            );
            let ConvertedTensor::IQuant(selected) =
                lease.materialize_portable().unwrap().into_converted()
            else {
                panic!("expected native blocks")
            };
            assert_eq!(selected.shape, [1, 2, 64]);
            assert_eq!(selected.data, raw[offset..offset + 4 * block_bytes]);
            assert!(store
                .acquire(TensorReadRequest {
                    key: "bank.weight".into(),
                    selection: TensorSelection::Contiguous {
                        offset_elements: offset + 1,
                        shape: vec![1, 2, 2 * block_bytes]
                    },
                    policy: ReadPolicy::RequireBounded,
                })
                .is_err());
        }
    }

    #[test]
    fn mxfp4_expert_selection_pushes_through_component_concatenation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("experts.gguf");
        let gate = vec![0u8; 2 * 96 * 2 * 17];
        let up = vec![1u8; 2 * 96 * 2 * 17];
        let down = vec![2u8; 2 * 64 * 3 * 17];
        Writer::default()
            .write(
                File::create(&path).unwrap(),
                &BTreeMap::new(),
                &[
                    TensorInput {
                        name: "experts.gate.weight",
                        dimensions: &[64, 96, 2],
                        ggml_type: GgmlType::MxFp4,
                        data: &gate,
                    },
                    TensorInput {
                        name: "experts.up.weight",
                        dimensions: &[64, 96, 2],
                        ggml_type: GgmlType::MxFp4,
                        data: &up,
                    },
                    TensorInput {
                        name: "experts.down.weight",
                        dimensions: &[96, 64, 2],
                        ggml_type: GgmlType::MxFp4,
                        data: &down,
                    },
                ],
            )
            .unwrap();
        let checkpoint = Checkpoint::open(&path).unwrap();
        let plan = test_plan(&checkpoint);
        let tensor_mapping = checkpoint.translated_outputs(str::to_owned).unwrap();
        let store = GgufWeightStore::builder()
            .add_checkpoint(checkpoint, &plan, &tensor_mapping)
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(
            store.metadata("experts.gate.weight").unwrap().logical_shape,
            [2, 96, 8]
        );

        let fused = DerivedWeightRecipe::Concatenate {
            axis: 1,
            inputs: vec![
                DerivedWeightRecipe::source("experts.gate.weight", TensorSelection::Full),
                DerivedWeightRecipe::source("experts.up.weight", TensorSelection::Full),
            ],
        };
        let selected = fused
            .select_bounded(
                &store,
                TensorSelection::Range {
                    axis: 0,
                    start: 1,
                    end: 2,
                },
            )
            .unwrap();
        assert_eq!(selected.infer(&store).unwrap().shape(), &[1, 192, 8]);
        selected.preflight_bounded(&store).unwrap();
        let rank_zero_gate_up = selected
            .select_bounded(
                &store,
                TensorSelection::Indices {
                    axis: 1,
                    indices: (0..64).chain(96..160).collect(),
                },
            )
            .unwrap();
        assert_eq!(
            rank_zero_gate_up.infer(&store).unwrap().shape(),
            &[1, 128, 8]
        );
        rank_zero_gate_up.preflight_bounded(&store).unwrap();

        let down = DerivedWeightRecipe::source("experts.down.weight", TensorSelection::Full)
            .select_bounded(
                &store,
                TensorSelection::Range {
                    axis: 0,
                    start: 1,
                    end: 2,
                },
            )
            .unwrap();
        for (start, end, expected) in [(0, 8, 8), (8, 12, 4)] {
            let rank = down
                .select_bounded(
                    &store,
                    TensorSelection::Range {
                        axis: 2,
                        start,
                        end,
                    },
                )
                .unwrap();
            assert_eq!(rank.infer(&store).unwrap().shape(), &[1, 64, expected]);
            rank.preflight_bounded(&store).unwrap();
        }
        assert_eq!(store.diagnostics().unwrap().physical_reads, 0);

        let lease = store
            .acquire(TensorReadRequest {
                key: "experts.gate.weight".into(),
                selection: TensorSelection::Contiguous {
                    offset_elements: 96 * 8,
                    shape: vec![1, 64, 8],
                },
                policy: ReadPolicy::RequireBounded,
            })
            .unwrap();
        let converted = lease.materialize_portable().unwrap().into_converted();
        let ConvertedTensor::MxFp4(selected) = converted else {
            panic!("expected a native MXFP4 selected tensor")
        };
        assert_eq!(selected.weight_shape, [1, 64, 8]);
        assert_eq!(selected.scale_shape, [1, 64, 2]);
        assert_eq!(store.diagnostics().unwrap().physical_reads, 1);
    }
}

mod reader_compile;
pub use reader_compile::{GgufSourceStorageRequest, PreparedGgufSourceFailure};

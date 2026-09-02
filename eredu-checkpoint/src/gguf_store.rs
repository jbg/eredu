//! Backend-neutral logical GGUF storage and portable encoded leases.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
};

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

/// Stable identity of one physical GGUF read and selection.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct GgufLeaseIdentity {
    checkpoint: usize,
    physical_name: String,
    selection: Option<GgufPhysicalSelection>,
}

impl GgufLeaseIdentity {
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

#[derive(Debug)]
struct ReaderCache {
    materializers: Vec<TensorMaterializer>,
    last_used: Vec<u64>,
    touched: BTreeSet<PathBuf>,
    tick: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

#[derive(Debug)]
struct StoreInner {
    catalog: BTreeMap<String, CatalogEntry>,
    unclaimed_keys: BTreeSet<String>,
    readers: Mutex<ReaderCache>,
    max_cached_readers: usize,
    statistics: StoreStatistics,
}

/// Builder for a logical store backed by one or more validated GGUF checkpoints.
#[derive(Debug, Default)]
pub struct GgufWeightStoreBuilder {
    checkpoints: Vec<Checkpoint>,
    catalog: BTreeMap<String, CatalogEntry>,
    unclaimed_keys: BTreeSet<String>,
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
        let mut names = BTreeMap::new();
        for mapped in tensor_mapping {
            let key = (mapped.physical_name.as_str(), mapped.original_name.as_str());
            if names.insert(key, mapped.layout.name.as_str()).is_some() {
                return Err(gguf_error(
                    &mapped.layout.name,
                    "admitted GGUF tensor mapping contains a duplicate source output",
                ));
            }
        }
        let checkpoint_index = self.checkpoints.len();
        for shard in checkpoint.shards() {
            for tensor in shard.tensors() {
                let physical_name = &tensor.descriptor().name;
                let selected = resolved.source_keys().contains(physical_name);
                let unclaimed = resolved.unclaimed_keys().contains(physical_name);
                if !selected && !unclaimed {
                    continue;
                }
                let descriptor = tensor.descriptor().clone();
                let physical_shape = descriptor
                    .row_major_shape()
                    .into_iter()
                    .map(|dimension| {
                        usize::try_from(dimension).map_err(|_| StoreError::Overflow {
                            context: format!("GGUF physical shape for tensor {physical_name:?}"),
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                for output in tensor.outputs() {
                    let name = names
                        .get(&(physical_name.as_str(), output.name.as_str()))
                        .ok_or_else(|| {
                            gguf_error(
                                &output.name,
                                "admitted GGUF tensor mapping omits a catalog output",
                            )
                        })?
                        .to_string();
                    if unclaimed {
                        self.unclaimed_keys.insert(name);
                        continue;
                    }
                    if self.catalog.contains_key(&name) {
                        return Err(gguf_error(
                            name,
                            "translated logical tensor collides with an existing output",
                        ));
                    }
                    let shape = output
                        .shape
                        .iter()
                        .map(|dimension| {
                            usize::try_from(*dimension).map_err(|_| StoreError::Overflow {
                                context: format!("GGUF logical shape for tensor {:?}", output.name),
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let byte_len = logical_byte_len(&output.name, output.dtype, &shape)?;
                    let logical_last_units_per_block =
                        logical_units_per_block(&output.name, &descriptor, &shape)?;
                    let metadata = TensorMetadata {
                        name: name.clone(),
                        logical_shape: shape,
                        physical_shape: physical_shape.clone(),
                        stored_dtype: stored_dtype(output.dtype),
                        encoded_byte_len: byte_len,
                        backing_shard: Some(shard.path().to_path_buf()),
                    };
                    self.catalog.insert(
                        name,
                        CatalogEntry {
                            checkpoint: checkpoint_index,
                            physical_name: physical_name.clone(),
                            original_name: output.name.clone(),
                            metadata,
                            physical_descriptor: descriptor.clone(),
                            logical_last_units_per_block,
                            source_encoding: crate::SourceTensorEncoding::Gguf {
                                ggml_type: descriptor.ggml_type,
                                endian: shard.endian(),
                            },
                        },
                    );
                }
            }
        }
        self.checkpoints.push(checkpoint);
        Ok(self)
    }

    /// Builds a nonempty immutable logical GGUF store.
    pub fn build(self) -> Result<GgufWeightStore, StoreError> {
        if self.catalog.is_empty() {
            return Err(gguf_error("", "GGUF logical catalog is empty"));
        }
        let materializers = self
            .checkpoints
            .iter()
            .map(Checkpoint::materializer)
            .collect::<Vec<_>>();
        let count = materializers.len();
        Ok(GgufWeightStore {
            inner: Arc::new(StoreInner {
                catalog: self.catalog,
                unclaimed_keys: self.unclaimed_keys,
                readers: Mutex::new(ReaderCache {
                    materializers,
                    last_used: vec![0; count],
                    touched: BTreeSet::new(),
                    tick: 0,
                    hits: 0,
                    misses: 0,
                    evictions: 0,
                }),
                max_cached_readers: if self.max_cached_readers == 0 {
                    DEFAULT_MAX_CACHED_SHARDS
                } else {
                    self.max_cached_readers
                },
                statistics: StoreStatistics::default(),
            }),
        })
    }
}

/// Persistent backend-neutral logical tensor store for GGUF checkpoints.
#[derive(Debug, Clone)]
pub struct GgufWeightStore {
    inner: Arc<StoreInner>,
}

impl GgufWeightStore {
    /// Starts a multi-checkpoint store builder.
    pub fn builder() -> GgufWeightStoreBuilder {
        GgufWeightStoreBuilder::default()
    }

    /// Returns catalog keys admitted but unclaimed by a non-strict contract.
    pub fn unclaimed_checkpoint_keys(&self) -> Vec<String> {
        self.inner.unclaimed_keys.iter().cloned().collect()
    }
}

/// Encoded GGUF lease whose portable payload is realized only on request.
#[derive(Debug, Clone)]
pub struct GgufLease {
    store: Arc<StoreInner>,
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
        let converted = self
            .store
            .readers
            .lock()
            .map_err(|_| StoreError::Internal("GGUF reader cache is poisoned".into()))?
            .materialize(
                self.entry.checkpoint,
                &self.entry.physical_name,
                self.identity.selection.as_ref(),
                self.store.max_cached_readers,
                &self.entry.metadata.name,
            )?;
        self.store
            .statistics
            .physical_reads
            .fetch_add(1, Ordering::Relaxed);
        self.store
            .statistics
            .physical_read_bytes
            .fetch_add(self.proof.length_bytes, Ordering::Relaxed);
        Ok(converted)
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
        let entry = self
            .inner
            .catalog
            .get(&request.key)
            .cloned()
            .ok_or_else(|| StoreError::UnknownTensor {
                key: request.key.clone(),
            })?;
        let output_shape = validate_selection(
            &request.key,
            &entry.metadata.logical_shape,
            &request.selection,
        )?;
        let read = match plan_bounded_selection(&request.key, &entry, &request.selection) {
            Ok(plan) => plan,
            Err(StoreError::BoundedSelectionUnavailable { .. })
                if request.policy == ReadPolicy::AllowFullTensorRead =>
            {
                ReadPlan {
                    physical_selection: None,
                    physical_offset: 0,
                    physical_byte_len: entry.physical_descriptor.byte_len,
                    selection_is_materialized: false,
                }
            }
            Err(error) => return Err(error),
        };
        let physically_bounded =
            matches!(request.selection, TensorSelection::Full) || read.physical_selection.is_some();
        Ok(GgufLease {
            store: Arc::clone(&self.inner),
            identity: GgufLeaseIdentity {
                checkpoint: entry.checkpoint,
                physical_name: entry.physical_name.clone(),
                selection: read.physical_selection,
            },
            entry,
            selection: request.selection,
            output_shape,
            proof: BoundedReadProof {
                physically_bounded,
                offset_bytes: read.physical_offset,
                length_bytes: read.physical_byte_len,
            },
            selection_is_materialized: read.selection_is_materialized,
        })
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
            touched_shard_paths: readers.touched.iter().cloned().collect(),
            payload_shard_paths: readers.touched.iter().cloned().collect(),
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

impl CheckpointSource for GgufWeightStore {
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
    fn materialize(
        &mut self,
        checkpoint: usize,
        physical_name: &str,
        selection: Option<&GgufPhysicalSelection>,
        maximum: usize,
        logical_key: &str,
    ) -> Result<ConvertedCheckpointTensor, StoreError> {
        let target_path = self
            .materializers
            .get(checkpoint)
            .ok_or_else(|| gguf_error(logical_key, "catalog references an unknown checkpoint"))?
            .shard_path_for_tensor(physical_name)
            .map_err(|error| gguf_error(logical_key, error))?
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
                self.materializers[victim].close_reader();
                self.evictions = self.evictions.saturating_add(1);
            }
        }
        self.last_used[checkpoint] = self.tick;
        let materializer = &mut self.materializers[checkpoint];
        let converted = match selection {
            Some(GgufPhysicalSelection::Axis(selection)) => {
                materializer.converted_tensor_selected(physical_name, selection)
            }
            Some(GgufPhysicalSelection::DenseSpan(selection)) => {
                materializer.converted_dense_tensor_span(physical_name, selection)
            }
            None => materializer.converted_tensor(physical_name),
        }
        .map_err(|error| gguf_error(logical_key, error))?;
        self.touched.insert(target_path);
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
        if !block_values.is_multiple_of(logical_units) {
            return Err(bounded_error(
                key,
                format!(
                    "native block length {block_values} is not divisible by {logical_units} converted units"
                ),
            ));
        }
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
        let physical_units_per_logical = block_values / logical_units;
        let physical_last = physical_shape.last_mut().ok_or_else(|| {
            bounded_error(key, "contiguous GGUF selection has no packed input axis")
        })?;
        *physical_last = physical_last
            .checked_mul(physical_units_per_logical)
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

fn logical_units_per_block(
    logical_name: &str,
    descriptor: &TensorDescriptor,
    logical_shape: &[usize],
) -> Result<Option<usize>, StoreError> {
    let physical_shape = descriptor
        .row_major_shape()
        .into_iter()
        .map(|dimension| {
            usize::try_from(dimension).map_err(|_| StoreError::Overflow {
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
        return Err(gguf_error(
            logical_name,
            format!(
                "logical shape {logical_shape:?} is not an innermost-axis projection of physical shape {physical_shape:?}"
            ),
        ));
    }
    let (Some(&physical_last), Some(&logical_last)) = (physical_shape.last(), logical_shape.last())
    else {
        return Ok(None);
    };
    let (block_values, _) = descriptor
        .ggml_type
        .block_and_bytes()
        .map_err(|error| gguf_error(logical_name, error))?;
    let block_values = usize::try_from(block_values).map_err(|_| StoreError::Overflow {
        context: format!("GGUF block length for tensor {logical_name:?}"),
    })?;
    if !physical_last.is_multiple_of(block_values) {
        return Err(gguf_error(
            logical_name,
            format!(
                "physical innermost dimension {physical_last} is not divisible by block length {block_values}"
            ),
        ));
    }
    let blocks = physical_last / block_values;
    if blocks == 0 || !logical_last.is_multiple_of(blocks) {
        return Err(gguf_error(
            logical_name,
            format!(
                "logical innermost dimension {logical_last} is not divisible by {blocks} physical blocks"
            ),
        ));
    }
    Ok(Some(logical_last / blocks))
}

fn logical_byte_len(name: &str, dtype: LogicalDtype, shape: &[usize]) -> Result<u64, StoreError> {
    let width = match dtype {
        LogicalDtype::U8 | LogicalDtype::I8 => 1u64,
        LogicalDtype::F16 | LogicalDtype::Bf16 | LogicalDtype::I16 => 2,
        LogicalDtype::F32 | LogicalDtype::U32 | LogicalDtype::I32 => 4,
        LogicalDtype::I64 | LogicalDtype::F64 => 8,
    };
    shape.iter().try_fold(width, |bytes, dimension| {
        bytes
            .checked_mul(u64::try_from(*dimension).map_err(|_| StoreError::Overflow {
                context: format!("GGUF logical byte length for tensor {name:?}"),
            })?)
            .ok_or_else(|| StoreError::Overflow {
                context: format!("GGUF logical byte length for tensor {name:?}"),
            })
    })
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

    fn test_plan(checkpoint: &Checkpoint) -> GgufCheckpointPlan {
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

    fn test_store(path: &Path) -> GgufWeightStore {
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

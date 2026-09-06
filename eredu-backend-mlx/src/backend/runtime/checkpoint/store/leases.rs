use super::materialization::{materialize_contiguous, materialize_indices, materialize_range};
use super::*;

fn selected_byte_len(
    key: &str,
    metadata: &TensorMetadata,
    selection: &TensorSelection,
    output_shape: &[usize],
) -> Result<usize, CheckpointMaterializationError> {
    if matches!(selection, TensorSelection::Full) {
        return usize::try_from(metadata.encoded_byte_len).map_err(|_| {
            CheckpointMaterializationError::ArithmeticOverflow {
                context: format!("encoded byte length for tensor {key:?}"),
            }
        });
    }
    let count = |shape: &[usize], context: &str| {
        shape.iter().try_fold(1usize, |value, dimension| {
            value.checked_mul(*dimension).ok_or_else(|| {
                CheckpointMaterializationError::ArithmeticOverflow {
                    context: format!("{context} for tensor {key:?}"),
                }
            })
        })
    };
    let full_elements = count(&metadata.logical_shape, "element count")?;
    let selected_elements = count(output_shape, "selected element count")?;
    let encoded_byte_len = usize::try_from(metadata.encoded_byte_len).map_err(|_| {
        CheckpointMaterializationError::ArithmeticOverflow {
            context: format!("encoded byte length for tensor {key:?}"),
        }
    })?;
    let scaled = encoded_byte_len
        .checked_mul(selected_elements)
        .ok_or_else(|| CheckpointMaterializationError::ArithmeticOverflow {
            context: format!("selected byte length for tensor {key:?}"),
        })?;
    if full_elements == 0 || !scaled.is_multiple_of(full_elements) {
        return Err(StoreError::InvalidSelection {
            key: key.into(),
            message: "selection does not have a whole-byte encoded length".into(),
        }
        .into());
    }
    Ok(scaled / full_elements)
}

#[derive(Debug, Clone)]
pub(super) enum WeightLeaseSource {
    Safetensors(NeutralSafetensorsLease),
    Gguf(Box<GgufLeaseSource>),
    Memory(NeutralMemoryLease),
}

#[derive(Debug, Clone)]
pub(super) struct GgufLeaseSource {
    pub(super) lease: NeutralGgufLease,
    converted_groups: Arc<
        Mutex<BTreeMap<eredu_checkpoint::gguf_store::GgufLeaseIdentity, Weak<CachedGgufGroup>>>,
    >,
}

/// A validated selection that pins its buffered payload shard.
///
/// The lease deliberately has no method returning a borrowed or mmap-derived
/// MLX array. [`Self::materialize`] is the only array-producing operation.
#[derive(Debug, Clone)]
pub struct WeightLease {
    key: String,
    metadata: TensorMetadata,
    selection: TensorSelection,
    output_shape: Vec<usize>,
    selected_byte_len: usize,
    pub(super) source: WeightLeaseSource,
}

impl WeightLease {
    pub(super) fn from_checkpoint_lease(
        lease: CheckpointLease,
        converted_groups: Arc<
            Mutex<BTreeMap<eredu_checkpoint::gguf_store::GgufLeaseIdentity, Weak<CachedGgufGroup>>>,
        >,
    ) -> Result<Self, CheckpointMaterializationError> {
        let key = lease.metadata().name.clone();
        let metadata = lease.metadata().clone();
        let selection = lease.selection().clone();
        let output_shape = lease.output_shape().to_vec();
        let selected_byte_len = match &lease {
            CheckpointLease::Safetensors(lease) => {
                usize::try_from(lease.bounded_read_proof().length_bytes).map_err(|_| {
                    CheckpointMaterializationError::ArithmeticOverflow {
                        context: format!("selected byte length for tensor {key:?}"),
                    }
                })?
            }
            CheckpointLease::Gguf(_) => {
                selected_byte_len(&key, &metadata, &selection, &output_shape)?
            }
            CheckpointLease::Memory(lease) => {
                usize::try_from(lease.bounded_read_proof().length_bytes).map_err(|_| {
                    CheckpointMaterializationError::ArithmeticOverflow {
                        context: format!("selected byte length for tensor {key:?}"),
                    }
                })?
            }
        };
        let source = match lease {
            CheckpointLease::Safetensors(lease) => WeightLeaseSource::Safetensors(lease),
            CheckpointLease::Gguf(lease) => WeightLeaseSource::Gguf(Box::new(GgufLeaseSource {
                lease: *lease,
                converted_groups,
            })),
            CheckpointLease::Memory(lease) => WeightLeaseSource::Memory(lease),
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
    pub fn metadata(&self) -> &TensorMetadata {
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
            WeightLeaseSource::Memory(_) => Path::new("<memory>"),
        }
    }

    /// Submits the selected tensor for materialization onto `execution_stream`.
    ///
    /// The returned guard retains this checkpoint lease, independently owned
    /// native source bytes, and exact completion ownership. Call
    /// [`WeightMaterialization::wait_on`] before
    /// evaluating a dependent graph on another compatible stream, or call
    /// [`WeightMaterialization::synchronize`] to block for and take the output.
    /// MLX graph construction alone does not consume the materialization.
    pub fn materialize(
        &self,
        source_stream: &Stream,
        execution_stream: &Stream,
    ) -> Result<WeightMaterialization, CheckpointMaterializationError> {
        self.clone()
            .prepare_materialization(source_stream, execution_stream)?
            .submit()
    }

    /// Prepares materialization while retaining its exact checkpoint dependency.
    ///
    /// The returned value must remain owned by the output's evaluation scope.
    /// Early Drop never waits; unresolved preparation is retained independently.
    pub fn prepare_materialization(
        self,
        source_stream: &Stream,
        execution_stream: &Stream,
    ) -> Result<PendingWeightMaterialization, CheckpointMaterializationError> {
        match self.source.clone() {
            WeightLeaseSource::Safetensors(shard) => {
                self.prepare_safetensors(shard, source_stream, execution_stream)
            }
            WeightLeaseSource::Gguf(source) => {
                self.prepare_gguf(*source, source_stream, execution_stream)
            }
            WeightLeaseSource::Memory(source) => {
                self.prepare_encoded(source, source_stream, execution_stream)
            }
        }
    }

    /// Schedules a host-only materialization that may borrow bounded
    /// SafeTensors bytes until a containing derived output is evaluated.
    ///
    /// Callers must not return `output()` after completing this pending value;
    /// use it only as an input to an evaluated dependent graph.
    pub fn prepare_borrowed_materialization(
        self,
        source_stream: &Stream,
    ) -> Result<PendingWeightMaterialization, CheckpointMaterializationError> {
        match self.source.clone() {
            WeightLeaseSource::Safetensors(shard) => {
                self.prepare_borrowed_safetensors(shard, source_stream)
            }
            WeightLeaseSource::Gguf(source) => {
                self.prepare_gguf(*source, source_stream, source_stream)
            }
            WeightLeaseSource::Memory(source) => {
                self.prepare_borrowed_encoded(source, source_stream)
            }
        }
    }

    fn prepare_borrowed_safetensors(
        self,
        source: NeutralSafetensorsLease,
        source_stream: &Stream,
    ) -> Result<PendingWeightMaterialization, CheckpointMaterializationError> {
        self.prepare_borrowed_encoded(source, source_stream)
    }

    fn prepare_borrowed_encoded(
        self,
        source: impl EncodedTensorLease,
        source_stream: &Stream,
    ) -> Result<PendingWeightMaterialization, CheckpointMaterializationError> {
        let dtype = safetensors_dtype(&self.key, &self.metadata.stored_dtype)?;
        let data = source.encoded_bytes().ok_or_else(|| {
            CheckpointMaterializationError::InvalidEncodedTensor {
                key: self.key.clone(),
                path: source
                    .backing_path()
                    .unwrap_or_else(|| Path::new("<unknown>"))
                    .to_path_buf(),
                message: "lease has no encoded bytes".into(),
            }
        })?;
        let view = TensorView::new(dtype, self.output_shape.clone(), data).map_err(|error| {
            CheckpointMaterializationError::InvalidEncodedTensor {
                key: self.key.clone(),
                path: source
                    .backing_path()
                    .unwrap_or_else(|| Path::new("<unknown>"))
                    .to_path_buf(),
                message: error.to_string(),
            }
        })?;
        let mut pending =
            PendingWeightMaterialization::begin(self.clone(), source_stream, source_stream)?;
        let source_value = Array::try_from(view).map_err(|conversion| {
            CheckpointMaterializationError::MlxConversion {
                key: self.key.clone(),
                source: conversion,
            }
        })?;
        pending.set_source(source_value);
        let output = pending.source().clone();
        pending.prepared(output)
    }

    fn prepare_safetensors(
        self,
        source: NeutralSafetensorsLease,
        source_stream: &Stream,
        execution_stream: &Stream,
    ) -> Result<PendingWeightMaterialization, CheckpointMaterializationError> {
        self.prepare_encoded(source, source_stream, execution_stream)
    }

    fn prepare_encoded(
        self,
        source: impl EncodedTensorLease,
        source_stream: &Stream,
        execution_stream: &Stream,
    ) -> Result<PendingWeightMaterialization, CheckpointMaterializationError> {
        let dtype = safetensors_dtype(&self.key, &self.metadata.stored_dtype)?;
        let data = source.encoded_bytes().ok_or_else(|| {
            CheckpointMaterializationError::InvalidEncodedTensor {
                key: self.key.clone(),
                path: source
                    .backing_path()
                    .unwrap_or_else(|| Path::new("<unknown>"))
                    .to_path_buf(),
                message: "lease has no encoded bytes".into(),
            }
        })?;
        let view = TensorView::new(dtype, self.output_shape.clone(), data).map_err(|error| {
            CheckpointMaterializationError::InvalidEncodedTensor {
                key: self.key.clone(),
                path: source
                    .backing_path()
                    .unwrap_or_else(|| Path::new("<unknown>"))
                    .to_path_buf(),
                message: error.to_string(),
            }
        })?;
        let mut pending =
            PendingWeightMaterialization::begin(self.clone(), source_stream, execution_stream)?;
        let source_value = Array::try_from(view).map_err(|conversion| {
            CheckpointMaterializationError::MlxConversion {
                key: self.key.clone(),
                source: conversion,
            }
        })?;
        pending.set_source(source_value);
        let materialized = pending
            .source()
            .copy(execution_stream)
            .map_err(|error| self.mlx_error("copy", error))?;
        pending.prepared(materialized)
    }
    fn prepare_gguf(
        self,
        source: GgufLeaseSource,
        source_stream: &Stream,
        execution_stream: &Stream,
    ) -> Result<PendingWeightMaterialization, CheckpointMaterializationError> {
        let GgufLeaseSource {
            lease,
            converted_groups,
        } = source;
        let cache_key = lease.identity().clone();
        let selection_is_materialized = lease.selection_is_materialized();
        let logical_output_name = lease.logical_output_name().to_owned();
        let mut pending =
            PendingWeightMaterialization::begin(self.clone(), source_stream, execution_stream)?;
        let mut groups = converted_groups
            .lock()
            .map_err(|_| CheckpointMaterializationError::StatePoisoned)?;
        groups.retain(|_, group| group.strong_count() > 0);
        let group = if let Some(cached) = groups.get(&cache_key).and_then(Weak::upgrade) {
            lease.record_coalesced_group_hit();
            cached
        } else {
            let portable = lease
                .materialize_portable()
                .map_err(CheckpointMaterializationError::from)?;
            let converted = GgufTensor::from_portable_host(portable).map_err(|error| {
                CheckpointMaterializationError::GgufConversion {
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
        pending.set_group(Arc::clone(&group));
        let source_value = group
            .arrays
            .iter()
            .find_map(|(name, value)| (name == &logical_output_name).then(|| value.clone()))
            .ok_or_else(|| CheckpointMaterializationError::GgufConversion {
                key: self.key.clone(),
                message: format!(
                    "portable GGUF group did not produce logical output {logical_output_name:?}"
                ),
            })?;
        pending.set_source(source_value);
        let materialized = if selection_is_materialized && source_stream == execution_stream {
            pending.source().clone()
        } else if selection_is_materialized {
            pending
                .source()
                .copy(execution_stream)
                .map_err(|source| self.mlx_error("copy", source))?
        } else {
            match &self.selection {
                TensorSelection::Range { axis, start, end } => materialize_range(
                    &self.key,
                    pending.source().clone(),
                    &self.metadata.logical_shape,
                    *axis,
                    *start,
                    *end,
                    source_stream,
                    execution_stream,
                )?,
                TensorSelection::Indices { axis, indices } => materialize_indices(
                    &self.key,
                    pending.source(),
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
                    pending.source(),
                    *offset_elements,
                    shape,
                    source_stream,
                    execution_stream,
                )?,
            }
        };
        pending.prepared(materialized)
    }

    pub(super) fn mlx_error(
        &self,
        operation: &'static str,
        source: safemlx::error::Exception,
    ) -> CheckpointMaterializationError {
        CheckpointMaterializationError::Mlx {
            key: self.key.clone(),
            operation,
            source,
        }
    }
}

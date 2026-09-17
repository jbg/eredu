use super::materialization::{materialize_contiguous, materialize_indices, materialize_range};
use super::materialization::{
    validate_operation, OriginalMaterializationSlots, PreparedPendingWeight,
};
use super::*;
use eredu_checkpoint::store::PinnedEncodedBytes;

const ORIGINAL_SELECTION_MESSAGE: &str = "selection does not have a whole-byte encoded length";
pub(super) fn original_selection_error_bytes(key_bytes: usize) -> Option<usize> {
    // str Debug emits at most ten bytes for each input byte, plus quotes. All
    // contexts below are closed literals; this bounds their exact writer.
    let formatted = key_bytes
        .checked_mul(10)?
        .checked_add(2 + "selected element count for tensor ".len())?;
    Some(formatted.max(key_bytes.checked_add(ORIGINAL_SELECTION_MESSAGE.len())?))
}
fn selection_overflow(key: &str, context: &str, original: bool) -> CheckpointMaterializationError {
    let context = if original {
        use std::fmt::Write;
        let mut value = String::with_capacity(
            original_selection_error_bytes(key.len())
                .expect("selected prequalified diagnostic layout"),
        );
        write!(&mut value, "{context} for tensor {key:?}").expect("String writing is infallible");
        value
    } else {
        format!("{context} for tensor {key:?}")
    };
    CheckpointMaterializationError::ArithmeticOverflow { context }
}
fn selected_byte_len(
    key: &str,
    metadata: &TensorMetadata,
    selection: &TensorSelection,
    output_shape: &[usize],
    original: bool,
) -> Result<usize, CheckpointMaterializationError> {
    if matches!(selection, TensorSelection::Full) {
        return usize::try_from(metadata.encoded_byte_len)
            .map_err(|_| selection_overflow(key, "encoded byte length", original));
    }
    let count = |shape: &[usize], context: &str| {
        shape.iter().try_fold(1usize, |value, dimension| {
            value
                .checked_mul(*dimension)
                .ok_or_else(|| selection_overflow(key, context, original))
        })
    };
    let full_elements = count(&metadata.logical_shape, "element count")?;
    let selected_elements = count(output_shape, "selected element count")?;
    let encoded_byte_len = usize::try_from(metadata.encoded_byte_len)
        .map_err(|_| selection_overflow(key, "encoded byte length", original))?;
    let scaled = encoded_byte_len
        .checked_mul(selected_elements)
        .ok_or_else(|| selection_overflow(key, "selected byte length", original))?;
    if full_elements == 0 || !scaled.is_multiple_of(full_elements) {
        return Err(StoreError::InvalidSelection {
            key: key.into(),
            message: ORIGINAL_SELECTION_MESSAGE.into(),
        }
        .into());
    }
    Ok(scaled / full_elements)
}

#[derive(Debug, Clone)]
pub(super) enum WeightLeaseSource {
    Safetensors(NeutralSafetensorsLease),
    Gguf(GgufLeaseSource),
    Memory(NeutralMemoryLease),
}

#[derive(Debug, Clone)]
pub(super) struct GgufLeaseSource {
    pub(super) lease: GgufLeaseSlot,
    converted_groups: super::CacheHandle,
}

// Empty exists only in the synchronous pre-native take/restore transaction.
// Original copies then share this SAME Box with immutable native source custody.
#[derive(Debug, Clone)]
pub(super) enum GgufLeaseSlot {
    Owned(Option<Box<NeutralGgufLease>>),
    Original(super::materialization::gguf_host::SourceCustody),
}
impl GgufLeaseSlot {
    pub(super) fn as_ref(&self) -> &NeutralGgufLease {
        match self {
            Self::Owned(lease) => lease
                .as_deref()
                .expect("GGUF source restored before native access"),
            Self::Original(source) => source.lease(),
        }
    }
    pub(super) fn take(&mut self) -> Box<NeutralGgufLease> {
        let Self::Owned(lease) = self else {
            unreachable!("unpublished same-Box transaction")
        };
        lease.take().expect("one unpublished GGUF source")
    }
    pub(super) fn restore(&mut self, lease: Box<NeutralGgufLease>) {
        let Self::Owned(value) = self else {
            unreachable!("unpublished same-Box transaction")
        };
        assert!(
            value.is_none(),
            "unpublished GGUF source was already restored"
        );
        *value = Some(lease);
    }
    pub(super) fn bind_original(
        &mut self,
        source: super::materialization::gguf_host::SourceCustody,
    ) {
        assert!(
            matches!(self, Self::Owned(None)),
            "source already published"
        );
        *self = Self::Original(source);
    }
}

impl std::ops::Deref for GgufLeaseSlot {
    type Target = NeutralGgufLease;
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

/// A validated selection that pins its buffered payload shard.
///
/// The lease deliberately has no method returning a borrowed or mmap-derived
/// MLX array. [`Self::materialize`] is the only array-producing operation.
#[derive(Debug, Clone)]
pub struct WeightLease {
    selected_byte_len: usize,
    pub(super) source: WeightLeaseSource,
}

impl WeightLeaseSource {
    /// Borrow the metadata already retained by the exact neutral source lease.
    /// The trait-object reference carries no new allocation or source owner.
    fn encoded_lease(&self) -> &dyn EncodedTensorLease {
        match self {
            Self::Safetensors(lease) => lease,
            Self::Gguf(source) => source.lease.as_ref(),
            Self::Memory(lease) => lease,
        }
    }
}

// A public acquisition failure must never retain the cache/native Array tree.
#[derive(Debug)]
pub(super) enum RetainedAcquisitionSource {
    Neutral(CheckpointLease),
    Bound(super::materialization::gguf_host::SourceCustody),
}
impl WeightLease {
    pub(super) fn into_acquisition_source(self) -> RetainedAcquisitionSource {
        match self.source {
            WeightLeaseSource::Safetensors(lease) => {
                RetainedAcquisitionSource::Neutral(CheckpointLease::Safetensors(lease))
            }
            WeightLeaseSource::Memory(lease) => {
                RetainedAcquisitionSource::Neutral(CheckpointLease::Memory(lease))
            }
            WeightLeaseSource::Gguf(source) => match source.lease {
                GgufLeaseSlot::Owned(Some(lease)) => {
                    RetainedAcquisitionSource::Neutral(CheckpointLease::Gguf(lease))
                }
                GgufLeaseSlot::Original(source) => RetainedAcquisitionSource::Bound(source),
                GgufLeaseSlot::Owned(None) => {
                    unreachable!("unpublished transaction cannot enter pending construction")
                }
            },
        }
    }
    pub(super) fn from_checkpoint_lease(
        lease: CheckpointLease,
        converted_groups: super::CacheHandle,
    ) -> Result<Self, CheckpointMaterializationError> {
        let selected = Self::checkpoint_byte_len(&lease)?;
        Ok(Self::from_validated_checkpoint_lease(
            lease,
            converted_groups,
            selected,
        ))
    }
    pub(super) fn checkpoint_byte_len(
        lease: &CheckpointLease,
    ) -> Result<usize, CheckpointMaterializationError> {
        Self::checkpoint_byte_len_impl(lease, false)
    }
    pub(super) fn checkpoint_byte_len_original(
        lease: &CheckpointLease,
    ) -> Result<usize, CheckpointMaterializationError> {
        Self::checkpoint_byte_len_impl(lease, true)
    }
    fn checkpoint_byte_len_impl(
        lease: &CheckpointLease,
        original: bool,
    ) -> Result<usize, CheckpointMaterializationError> {
        let key = lease.metadata().name.as_str();
        let metadata = lease.metadata();
        let selection = lease.selection();
        let output_shape = lease.output_shape();
        let selected_byte_len = match lease {
            CheckpointLease::Safetensors(lease) => {
                usize::try_from(lease.bounded_read_proof().length_bytes).map_err(|_| {
                    CheckpointMaterializationError::ArithmeticOverflow {
                        context: format!("selected byte length for tensor {key:?}"),
                    }
                })?
            }
            CheckpointLease::Gguf(_) => {
                selected_byte_len(key, metadata, selection, output_shape, original)?
            }
            CheckpointLease::Memory(lease) => {
                usize::try_from(lease.bounded_read_proof().length_bytes).map_err(|_| {
                    CheckpointMaterializationError::ArithmeticOverflow {
                        context: format!("selected byte length for tensor {key:?}"),
                    }
                })?
            }
        };
        Ok(selected_byte_len)
    }
    pub(super) fn from_validated_checkpoint_lease(
        lease: CheckpointLease,
        converted_groups: super::CacheHandle,
        selected_byte_len: usize,
    ) -> Self {
        let source = match lease {
            CheckpointLease::Safetensors(lease) => WeightLeaseSource::Safetensors(lease),
            CheckpointLease::Gguf(lease) => WeightLeaseSource::Gguf(GgufLeaseSource {
                lease: GgufLeaseSlot::Owned(Some(lease)),
                converted_groups,
            }),
            CheckpointLease::Memory(lease) => WeightLeaseSource::Memory(lease),
        };
        Self {
            selected_byte_len,
            source,
        }
    }

    /// Returns the logical key pinned by this lease.
    pub fn key(&self) -> &str {
        &self.metadata().name
    }

    /// Returns metadata captured when the lease was acquired.
    pub fn metadata(&self) -> &TensorMetadata {
        self.source.encoded_lease().metadata()
    }

    /// Returns the validated selection.
    pub fn selection(&self) -> &TensorSelection {
        self.source.encoded_lease().selection()
    }

    /// Returns the selected output shape.
    pub fn output_shape(&self) -> &[usize] {
        self.source.encoded_lease().output_shape()
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
        self.prepare_materialization_impl(source_stream, execution_stream, None)
    }
    pub(crate) fn prepare_materialization_with_operations(
        self,
        source_stream: &Stream,
        execution_stream: &Stream,
        slots: &mut OriginalMaterializationSlots<'_>,
        observer: &safemlx::OriginalScopeObserver,
    ) -> Result<PendingWeightMaterialization, CheckpointMaterializationError> {
        validate_operation(observer)?;
        let ready = slots.pending_weights.checkout().map_err(|cause| {
            CheckpointMaterializationError::OriginalOperationCapacity {
                family: "pending weight",
                prepared: cause.prepared,
            }
        })?;
        self.prepare_materialization_impl(
            source_stream,
            execution_stream,
            Some((ready, observer.clone())),
        )
    }
    pub(super) fn prepare_materialization_impl(
        self,
        source_stream: &Stream,
        execution_stream: &Stream,
        original: Option<(PreparedPendingWeight, safemlx::OriginalScopeObserver)>,
    ) -> Result<PendingWeightMaterialization, CheckpointMaterializationError> {
        match &self.source {
            WeightLeaseSource::Safetensors(source) => {
                let bytes = source.pin_encoded_bytes();
                self.prepare_encoded(bytes, source_stream, execution_stream, original)
            }
            WeightLeaseSource::Gguf(_) => {
                self.prepare_gguf(source_stream, execution_stream, original)
            }
            WeightLeaseSource::Memory(source) => {
                let bytes = source.pin_encoded_bytes();
                self.prepare_encoded(bytes, source_stream, execution_stream, original)
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
        self.prepare_borrowed_materialization_impl(source_stream, None)
    }
    pub(crate) fn prepare_borrowed_materialization_with_operations(
        self,
        source_stream: &Stream,
        slots: &mut OriginalMaterializationSlots<'_>,
        observer: &safemlx::OriginalScopeObserver,
    ) -> Result<PendingWeightMaterialization, CheckpointMaterializationError> {
        validate_operation(observer)?;
        let ready = slots.pending_weights.checkout().map_err(|cause| {
            CheckpointMaterializationError::OriginalOperationCapacity {
                family: "pending weight",
                prepared: cause.prepared,
            }
        })?;
        self.prepare_borrowed_materialization_impl(source_stream, Some((ready, observer.clone())))
    }
    pub(super) fn prepare_borrowed_materialization_impl(
        self,
        source_stream: &Stream,
        original: Option<(PreparedPendingWeight, safemlx::OriginalScopeObserver)>,
    ) -> Result<PendingWeightMaterialization, CheckpointMaterializationError> {
        match &self.source {
            WeightLeaseSource::Safetensors(source) => {
                let bytes = source.pin_encoded_bytes();
                self.prepare_borrowed_encoded(bytes, source_stream, original)
            }
            WeightLeaseSource::Gguf(_) => self.prepare_gguf(source_stream, source_stream, original),
            WeightLeaseSource::Memory(source) => {
                let bytes = source.pin_encoded_bytes();
                self.prepare_borrowed_encoded(bytes, source_stream, original)
            }
        }
    }

    fn prepare_borrowed_encoded(
        self,
        source: PinnedEncodedBytes,
        source_stream: &Stream,
        original: Option<(PreparedPendingWeight, safemlx::OriginalScopeObserver)>,
    ) -> Result<PendingWeightMaterialization, CheckpointMaterializationError> {
        let dtype = safetensors_dtype(self.key(), &self.metadata().stored_dtype)?;
        let data = source.encoded_bytes().ok_or_else(|| {
            CheckpointMaterializationError::InvalidEncodedTensor {
                key: self.key().to_owned(),
                path: source
                    .backing_path()
                    .unwrap_or_else(|| Path::new("<unknown>"))
                    .to_path_buf(),
                message: "lease has no encoded bytes".into(),
            }
        })?;
        let view = TensorView::new(dtype, self.output_shape().to_vec(), data).map_err(|error| {
            CheckpointMaterializationError::InvalidEncodedTensor {
                key: self.key().to_owned(),
                path: source
                    .backing_path()
                    .unwrap_or_else(|| Path::new("<unknown>"))
                    .to_path_buf(),
                message: error.to_string(),
            }
        })?;
        let mut pending = PendingWeightMaterialization::begin_with_original(
            self,
            source_stream,
            source_stream,
            original,
        )?;
        let source_value = Array::try_from(view).map_err(|conversion| {
            CheckpointMaterializationError::MlxConversion {
                key: pending.key().to_owned(),
                source: conversion,
            }
        })?;
        pending.set_source(source_value);
        let output = pending.source().clone();
        pending.prepared(output)
    }

    fn prepare_encoded(
        self,
        source: PinnedEncodedBytes,
        source_stream: &Stream,
        execution_stream: &Stream,
        original: Option<(PreparedPendingWeight, safemlx::OriginalScopeObserver)>,
    ) -> Result<PendingWeightMaterialization, CheckpointMaterializationError> {
        let dtype = safetensors_dtype(self.key(), &self.metadata().stored_dtype)?;
        let data = source.encoded_bytes().ok_or_else(|| {
            CheckpointMaterializationError::InvalidEncodedTensor {
                key: self.key().to_owned(),
                path: source
                    .backing_path()
                    .unwrap_or_else(|| Path::new("<unknown>"))
                    .to_path_buf(),
                message: "lease has no encoded bytes".into(),
            }
        })?;
        let view = TensorView::new(dtype, self.output_shape().to_vec(), data).map_err(|error| {
            CheckpointMaterializationError::InvalidEncodedTensor {
                key: self.key().to_owned(),
                path: source
                    .backing_path()
                    .unwrap_or_else(|| Path::new("<unknown>"))
                    .to_path_buf(),
                message: error.to_string(),
            }
        })?;
        let mut pending = PendingWeightMaterialization::begin_with_original(
            self,
            source_stream,
            execution_stream,
            original,
        )?;
        let source_value = Array::try_from(view).map_err(|conversion| {
            CheckpointMaterializationError::MlxConversion {
                key: pending.key().to_owned(),
                source: conversion,
            }
        })?;
        pending.set_source(source_value);
        let materialized = pending
            .source()
            .copy(execution_stream)
            .map_err(|error| pending.lease().mlx_error("copy", error))?;
        pending.prepared(materialized)
    }
    #[cfg(test)]
    pub(super) fn prepare_original_gguf_fixture(
        self,
        stream: &Stream,
        ready: PreparedPendingWeight,
        observer: safemlx::OriginalScopeObserver,
    ) -> Result<PendingWeightMaterialization, CheckpointMaterializationError> {
        self.prepare_gguf(stream, stream, Some((ready, observer)))
    }

    fn prepare_gguf(
        self,
        source_stream: &Stream,
        execution_stream: &Stream,
        original: Option<(PreparedPendingWeight, safemlx::OriginalScopeObserver)>,
    ) -> Result<PendingWeightMaterialization, CheckpointMaterializationError> {
        let original_miss = original.is_some();
        let WeightLeaseSource::Gguf(source) = &self.source else {
            unreachable!("closed GGUF dispatch")
        };
        let selection_is_materialized = source.lease.selection_is_materialized();
        // Only the existing cache Arc is aliased; the real GGUF Box moves below.
        let converted_groups = source.converted_groups.clone();
        let mut pending = PendingWeightMaterialization::begin_with_original(
            self,
            source_stream,
            execution_stream,
            original,
        )?;
        // Declared before the loan: any early error releases the cache lock
        // before retired keys/weak blocks/raw custody can run destructors.
        let retired;
        let retired_replaced;
        let mut ordinary_node;
        let mut groups = converted_groups
            .lock()
            .map_err(|_| CheckpointMaterializationError::StatePoisoned)?;
        retired = groups.extract_if(|_, group| group.stale());
        #[cfg(test)]
        super::cache::after_sweep();
        let group = if let Some(cached) = groups
            .get_by(|key| {
                pending
                    .gguf_lease()
                    .identity()
                    .cache_view()
                    .cmp(&key.view())
            })
            .and_then(super::cache::WeakGroup::upgrade)
        {
            pending.gguf_lease().record_coalesced_group_hit();
            retired_replaced = None;
            cached
        } else {
            let admitted_conversion = original_miss && pending.has_admitted_gguf_storage();
            // Only the explicit partial Vec-only constructor lacks a source
            // bank. A present bank remains mandatory even when exhausted or
            // foreign; no preparation failure selects ordinary metadata.
            let admitted_cache = admitted_conversion && pending.has_gguf_cache_source_bank();
            if admitted_cache {
                pending.prepare_gguf_cache_key()?;
            }
            let arrays = if admitted_conversion {
                let portable = pending.materialize_gguf_admitted()?;
                pending.convert_original_stored_gguf(portable)?
            } else {
                let portable = if original_miss {
                    pending.materialize_gguf_prepared()?
                } else {
                    pending
                        .gguf_lease()
                        .materialize_portable()
                        .map_err(CheckpointMaterializationError::from)?
                };
                let converted = if original_miss {
                    pending.convert_original_gguf(portable)?
                } else {
                    GgufTensor::from_portable_host(portable).map_err(|error| {
                        CheckpointMaterializationError::GgufConversion {
                            key: pending.key().to_owned(),
                            message: error.to_string(),
                        }
                    })?
                };
                CachedGgufArrays::Ordinary(converted.into_arrays())
            };
            if admitted_cache {
                let (cached, replaced) = pending.complete_gguf_cache_entry(arrays, &mut groups)?;
                retired_replaced = replaced;
                cached
            } else {
                let cached = super::cache::Group::ordinary(arrays);
                ordinary_node = Some(super::cache::Node::new(
                    super::cache::Key::Ordinary(pending.gguf_lease().identity().clone()),
                    cached.downgrade(),
                    None,
                ));
                retired_replaced = groups
                    .insert_or_replace(&mut ordinary_node, |row| row.stale())
                    .map_err(|()| CheckpointMaterializationError::StatePoisoned)?;
                cached
            }
        };
        drop(groups);
        drop(retired_replaced);
        drop(retired);
        pending.set_group(group.clone());
        let logical_output_name = pending.gguf_lease().logical_output_name();
        let source_value = group
            .arrays
            .get(logical_output_name)
            .cloned()
            .ok_or_else(|| CheckpointMaterializationError::GgufConversion {
                key: pending.key().to_owned(),
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
                .map_err(|source| pending.lease().mlx_error("copy", source))?
        } else {
            match pending.lease().selection() {
                TensorSelection::Range { axis, start, end } => materialize_range(
                    pending.key(),
                    pending.source().clone(),
                    &pending.lease().metadata().logical_shape,
                    *axis,
                    *start,
                    *end,
                    source_stream,
                    execution_stream,
                )?,
                TensorSelection::Indices { axis, indices } => materialize_indices(
                    pending.key(),
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
                    pending.key(),
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
            key: self.key().to_owned(),
            operation,
            source,
        }
    }
}

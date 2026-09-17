//! Final boxed lazy lease, using the exact original acquisition worker.
use super::*;

pub(super) fn acquire(
    store: &GgufWeightStore,
    request: TensorReadRequest,
) -> Result<GgufLease, StoreError> {
    let entry = store
        .inner
        .catalog
        .get(&request.key)
        .cloned()
        .ok_or_else(|| StoreError::UnknownTensor {
            key: request.key.clone(),
        })?;
    let (output_shape, read) = plan_request(&entry, &request)?;
    let physically_bounded =
        matches!(request.selection, TensorSelection::Full) || read.physical_selection.is_some();
    Ok(GgufLease {
        store: store.inner.clone(),
        identity: GgufLeaseIdentity {
            source: store.inner.identity(),
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
            physical_reads: 1,
            physical_read_bytes: read.physical_byte_len,
        },
        selection_is_materialized: read.selection_is_materialized,
    })
}

/// Metadata-only final storage. Actual reader/conversion allocation remains
/// owned by the unchanged lazy materialization path, not prepaid by this box.
#[derive(Debug)]
pub(crate) struct Prepared {
    lease: Box<GgufLease>,
    key: String,
    policy: ReadPolicy,
}
impl Prepared {
    pub(crate) fn retained_payload_capacity(&self) -> Option<usize> {
        use crate::store::acquisition::storage::{metadata, selection, vector};
        let lease = &self.lease;
        let entry = &lease.entry;
        let physical_selection = match lease.identity.selection.as_ref() {
            Some(GgufPhysicalSelection::Axis(GgufTensorSelection::Indices { indices, .. })) => {
                vector::<usize>(indices.capacity())?
            }
            Some(GgufPhysicalSelection::DenseSpan(span)) => span.shape_storage_layout()?.size(),
            _ => 0,
        };
        let encoding = match &entry.source_encoding {
            crate::SourceTensorEncoding::Safetensors(StoredDtype::Other(name))
            | crate::SourceTensorEncoding::RecipeOutput(StoredDtype::Other(name)) => {
                name.capacity()
            }
            _ => 0,
        };
        [
            self.key.capacity(),
            std::mem::size_of::<GgufLease>(),
            metadata(&entry.metadata)?,
            selection(&lease.selection)?,
            vector::<usize>(lease.output_shape.capacity())?,
            entry.physical_name.capacity(),
            entry.original_name.capacity(),
            entry.physical_descriptor.name.capacity(),
            vector::<u64>(entry.physical_descriptor.dimensions.capacity())?,
            lease.identity.physical_name.capacity(),
            physical_selection,
            encoding,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }

    pub(crate) fn prepare(
        store: &GgufWeightStore,
        request: &TensorReadRequest,
    ) -> Result<Self, StoreError> {
        let lease = acquire(store, request.clone())?;
        Ok(Self {
            lease: Box::new(lease),
            key: request.key.clone(),
            policy: request.policy,
        })
    }
    pub(crate) fn prepare_plan(
        store: &GgufWeightStore,
        request: &TensorReadRequest,
        plan: &GgufConversionPlan,
    ) -> Option<Self> {
        let lease = plan.prepare_acquisition(store, request)?;
        Some(Self {
            lease: Box::new(lease),
            key: request.key.clone(),
            policy: request.policy,
        })
    }
    pub(crate) fn matches(&self, store: &GgufWeightStore, request: &TensorReadRequest) -> bool {
        self.lease.store.same(&store.inner)
            && self.key == request.key
            && self.lease.selection == request.selection
            && self.policy == request.policy
    }
    pub(crate) fn into_lease(self) -> CheckpointLease {
        CheckpointLease::Gguf(self.lease)
    }
    #[cfg(test)]
    pub(super) fn lease(&self) -> &GgufLease {
        &self.lease
    }
}

/// Actual request validation and physical selection, shared with cold planning.
pub(super) fn plan_request(
    entry: &CatalogEntry,
    request: &TensorReadRequest,
) -> Result<(Vec<usize>, ReadPlan), StoreError> {
    let output_shape = validate_selection(
        &request.key,
        &entry.metadata.logical_shape,
        &request.selection,
    )?;
    let read = match plan_bounded_selection(&request.key, entry, &request.selection) {
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
    Ok((output_shape, read))
}

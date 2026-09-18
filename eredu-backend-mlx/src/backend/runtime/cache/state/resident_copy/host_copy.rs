//! Dense state destinations paid by the same finite host metadata account.
use super::*;
use eredu_core::{BackendFailure, HostPreparationAuthority};
use eredu_nn::workspace::HostMetadataFunding;
use eredu_runtime::{DenseHostSlotInitialization, HostSlotTable, PreparedDenseHostCopyError};

/// Only host storage is authorized here. The caller must already have admitted
/// the numerical copy and keep its recovery owner through every callback.
pub(in crate::backend::runtime::cache::state) fn copy_slots<S, D, F>(
    source: DenseHostSlotInitialization<'_, S, D>,
    host: &HostPreparationAuthority,
    funding: &HostMetadataFunding,
    copy: F,
) -> Result<HostSlotTable<D>, Error>
where
    F: FnMut(usize, &S) -> Result<D, Error>,
{
    let bytes = source
        .prepared_copy_bytes::<Error, F>(&copy)
        .and_then(|n| usize::try_from(n).ok())
        .and_then(|n| {
            n.checked_add(BackendFailure::source_retention_peak_bytes::<
                PreparedDenseHostCopyError<Error>,
            >()?)
        })
        .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
    funding
        .reserve_metadata(bytes)
        .map_err(Error::WorkspacePlanning)?;
    source
        .copy_with_preparation(host, copy)
        .map_err(|cause| Error::StorageSource(BackendFailure::from_error(cause)))
}

/// Independently copied actual native representation, before outer erasure.
/// Numerical settlement/publication is required before the caller exposes it.
pub(crate) enum OriginalResidentState {
    KeyValue(MlxKeyValueState),
    Hybrid(MlxHybridState),
    Pooling(MlxPoolingAttentionState),
}
impl PreparedResidentDecoderCopy<'_> {
    pub(crate) fn copy_dense_with_preparation(
        self,
        host: &HostPreparationAuthority,
        funding: &HostMetadataFunding,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<OriginalResidentState, Error> {
        match self
            .into_dense_fixed()
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?
        {
            // This H-only table path lacks the independent paged manager
            // preparation. Paged callers use the existing state copy hook or
            // the fresh prompt's concrete construct_dense producer.
            dense::PreparedResidentDenseCopy::Paged(_) => Err(unknown()),
            dense::PreparedResidentDenseCopy::KeyValue(plan) => plan
                .copy_dense_with_preparation(host, funding, stream, roots)
                .map(OriginalResidentState::KeyValue),
            dense::PreparedResidentDenseCopy::HybridGrouped(plan) => plan
                .copy_dense_with_preparation(host, funding, stream, roots)
                .map(OriginalResidentState::Hybrid),
            dense::PreparedResidentDenseCopy::Pooling(plan) => plan
                .copy_dense_with_preparation(host, funding, stream, roots)
                .map(OriginalResidentState::Pooling),
        }
    }
}
impl OriginalResidentState {
    pub(crate) fn prepare_copy_fixed(
        &self,
    ) -> Result<PreparedResidentDecoderCopy<'_>, ResidentDecoderPreparationError> {
        match self {
            Self::KeyValue(state) => PreparedResidentDecoderCopy::key_value_fixed(state),
            Self::Hybrid(state) => PreparedResidentDecoderCopy::hybrid_fixed(state),
            Self::Pooling(state) => PreparedResidentDecoderCopy::pooling_fixed(state),
        }
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;

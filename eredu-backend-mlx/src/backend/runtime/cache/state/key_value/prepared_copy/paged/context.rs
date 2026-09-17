//! Finite partition of the enclosing accepted snapshot's original host owner.
use super::*;
use eredu_nn::workspace::{
    WorkspaceMetadataFunding, WorkspaceMetadataFundingError,
};
pub(super) fn control_bytes(bytes: usize) -> Option<usize> {
    let controls = [
        bytes,
        eredu_core::HostMetadataFunding::prepaid_control_bytes()?,
        WorkspaceContext::construction_bytes::<super::super::super::original_reset::HostMetadata>(
        )?,
        size_of::<HostFailure>(),
        eredu_core::BackendFailure::source_retention_peak_bytes::<HostFailure>()?,
        size_of::<WorkspaceContext>(),
        size_of::<Result<WorkspaceContext, Error>>(),
        eredu_core::BackendFailure::source_retention_peak_bytes::<WorkspaceMetadataFundingError>()?,
    ];
    controls
        .into_iter()
        .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
}
pub(super) fn prepare(
    bytes: usize,
    host: &HostPreparationAuthority,
) -> Result<WorkspaceContext, Error> {
    let limit = control_bytes(bytes).ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
    let funding = WorkspaceMetadataFunding::from(
        eredu_core::HostMetadataFunding::from_prepaid(limit, host.clone())
            .map_err(Error::WorkspacePlanning)?,
    );
    WorkspaceContext::new_with_metadata_funding(
        super::super::super::original_reset::HostMetadata,
        funding,
    )
    .map_err(|cause| Error::Neural(cause.into()))
}

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct HostFailure {
    #[source]
    cause: Error,
    _host: HostPreparationAuthority,
}
pub(super) fn failure(cause: Error, host: &HostPreparationAuthority) -> Error {
    Error::StorageSource(eredu_core::BackendFailure::from_error(HostFailure {
        cause,
        _host: host.clone(),
    }))
}

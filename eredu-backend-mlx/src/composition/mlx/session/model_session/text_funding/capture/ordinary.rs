//! One paid ordinary source inventory and its existing generic publication.
use super::*;
use crate::backend::runtime::residency::storage::StorageIdentity;
use eredu_runtime::working_memory::{MemoryLedger, StorageMetadataFunding};

fn collector_bytes() -> Result<usize, Error> {
    RetainedStorage::original_collector_control_bytes(1)
        .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))
}
/// One actual callback publishes one backing and its optional native-control row.
/// The source inventory itself retains exactly one Array, including empty values.
pub(in crate::composition::mlx::session::model_session) fn control_bytes() -> Result<u64, Error> {
    let overflow = || Error::PrefillControl(WorkingMemoryError::Overflow);
    let host = StorageMetadataFunding::host_owner_bytes(collector_bytes()?)
        .map_err(Error::WorkspacePlanning)?;
    let account = MemoryLedger::storage_metadata_control_bytes().map_err(Error::PrefillControl)?;
    let publication =
        crate::backend::runtime::residency::storage::generic_storage_publication_layout(2)
            .map_err(Error::PrefillControl)?
            .requested_bytes();
    let pin = eredu_runtime::working_memory::StoragePublicationLayout::<StorageIdentity>::new(2)
        .map_err(Error::PrefillControl)?
        .requested_bytes();
    u64::try_from(host)
        .map_err(|_| overflow())?
        .checked_add(account.checked_mul(3).ok_or_else(overflow)?)
        .and_then(|n| n.checked_add(publication))
        .and_then(|n| n.checked_add(pin))
        .ok_or_else(overflow)
}
pub(super) fn prepare(scope: &WorkingMemoryFundingScope) -> Result<RetainedStorage, Error> {
    let metadata = scope
        .prepare_storage_metadata()
        .map_err(Error::WorkspacePlanning)?;
    let host = metadata
        .prepare_host_owner(collector_bytes()?)
        .map_err(Error::WorkspacePlanning)?;
    RetainedStorage::prepare_ordinary(1, scope.pool(), &host).map_err(Into::into)
}

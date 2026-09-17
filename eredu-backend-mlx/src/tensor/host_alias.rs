//! Exact existing-descriptor alias storage under an explicit host source.
use super::MlxTensor;
use eredu_core::{BackendFailure,HostMetadataFunding,HostMetadataFundingError};
use safemlx::{Array,PreparedArrayClone};
use std::mem::{size_of,size_of_val};
#[derive(Debug,thiserror::Error)]
#[error("retained native tensor clone: {cause}")]
struct AliasFailure {#[source] cause:safemlx::PreparedArrayCloneCause,_funding:HostMetadataFunding}
impl MlxTensor {
    /// Same native clone slot, handle and fixed failure transport used below.
    /// No descriptor, backing, evaluation or allocation permission is created.
    pub(crate) fn host_clone_bytes()->Option<usize> {
        let parts=[PreparedArrayClone::control_bytes()?,Array::inspection_clone_handle_bytes(),
            size_of::<Self>(),size_of::<Result<Self,BackendFailure>>(),size_of::<AliasFailure>(),
            size_of::<(&Self,&HostMetadataFunding)>(),BackendFailure::source_retention_peak_bytes::<AliasFailure>()?];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
    /// Retains this exact immutable descriptor in an admitted final handle.
    /// The caller keeps the explicit funding after the returned tensor, while
    /// the descriptor continues to retain its original native backing custody.
    pub(crate) fn clone_with_host_source(&self,funding:&HostMetadataFunding)->Result<Self,BackendFailure> {
        funding.reserve_metadata(Self::host_clone_bytes().ok_or(HostMetadataFundingError::Overflow)?)?;
        let fail=|cause|BackendFailure::from_error(AliasFailure{cause,_funding:funding.clone()});
        let mut slot=PreparedArrayClone::try_prepare_for_inspection().map_err(fail)?;
        slot.fill_for_inspection(self.as_array()).map(Self::from_array).map_err(fail)
    }
}

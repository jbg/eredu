//! Concrete contiguous destinations used by portable source-funded workers.
use eredu_core::{HostMetadataFunding, HostMetadataFundingError};
use std::mem::{size_of, size_of_val};

pub(crate) fn funded_vec<T>(
    capacity: usize,
    funding: Option<&HostMetadataFunding>,
) -> Result<Vec<T>, HostMetadataFundingError> {
    if let Some(funding) = funding {
        let bytes = funded_vec_bytes::<T>(capacity).ok_or(HostMetadataFundingError::Overflow)?;
        funding.reserve_metadata(bytes)?;
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| HostMetadataFundingError::Unavailable)?;
    Ok(values)
}

pub(crate) fn funded_vec_bytes<T>(capacity: usize) -> Option<usize> {
        let parts = [
            size_of::<Vec<T>>(),
            size_of::<Result<Vec<T>, HostMetadataFundingError>>(),
            size_of::<std::collections::TryReserveError>(),
            HostMetadataFunding::reservation_control_bytes(),
        ];
        let backing = std::alloc::Layout::array::<T>(capacity).ok()?.size();
        let bytes = parts
            .into_iter()
            .try_fold(
                size_of_val(&parts).checked_add(backing)?,
                usize::checked_add,
            )
            ?;
    Some(bytes)
}

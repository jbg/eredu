//! A finite constructor partition of the same accepted reset account.
use super::*;
use eredu_core::BackendFailure;
use eredu_nn::workspace::{
    WorkspaceMetadataAccount, WorkspaceMetadataFunding, WorkspaceMetadataFundingError,
};
use std::sync::atomic::{AtomicUsize, Ordering};
#[derive(Debug)]
struct Account {
    spent: AtomicUsize,
    limit: usize,
    _custody: ResetCustody,
}
impl WorkspaceMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), WorkspaceMetadataFundingError> {
        let mut previous = self.spent.load(Ordering::Acquire);
        loop {
            let next = previous
                .checked_add(bytes)
                .ok_or(WorkspaceMetadataFundingError::Overflow)?;
            if next > self.limit {
                return Err(WorkspaceMetadataFundingError::Capacity {
                    required: u64::try_from(next)
                        .map_err(|_| WorkspaceMetadataFundingError::Overflow)?,
                    available: u64::try_from(self.limit)
                        .map_err(|_| WorkspaceMetadataFundingError::Overflow)?,
                });
            }
            match self.spent.compare_exchange_weak(
                previous,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(actual) => previous = actual,
            }
        }
    }
}
pub(super) fn control_bytes(bytes: usize) -> Option<usize> {
    if bytes == 0 {
        return Some(0);
    }
    let frames = [
        bytes,
        WorkspaceMetadataFunding::constructor_bytes::<Account>()?,
        size_of::<Account>(),
        size_of::<Option<WorkspaceMetadataFunding>>(),
        size_of::<Result<Option<WorkspaceMetadataFunding>, BackendFailure>>(),
        BackendFailure::source_retention_peak_bytes::<WorkspaceMetadataFundingError>()?,
        size_of::<(usize, usize, usize)>(),
        size_of::<Result<usize, usize>>(),
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}
pub(super) fn prepare(
    bytes: usize,
    custody: &ResetCustody,
) -> Result<Option<WorkspaceMetadataFunding>, BackendFailure> {
    if bytes == 0 {
        return Ok(None);
    }
    let limit = WorkspaceMetadataFunding::constructor_bytes::<Account>()
        .and_then(|n| n.checked_add(bytes))
        .ok_or_else(|| BackendFailure::from_error(WorkingMemoryError::Overflow))?;
    WorkspaceMetadataFunding::new(Account {
        spent: AtomicUsize::new(0),
        limit,
        _custody: custody.clone(),
    })
    .map(Some)
    .map_err(BackendFailure::from_error)
}

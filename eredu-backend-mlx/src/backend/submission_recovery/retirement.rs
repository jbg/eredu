//! Exact original owner cleanup after independent successful completion.
//! This observes no other request and grants no next operation or byte credit.
use crate::backend::error::Error;
use safemlx::{OriginalScopeObserver, SubmissionRetirement};

/// The caller has already established successful terminal completion and ended
/// its source/RefCell loans. A busy pass is not retirement evidence: the caller
/// must preserve its failure/stop disposition and cannot advance a reused bound.
pub(crate) fn complete(observer: &OriginalScopeObserver) -> Result<(), Error> {
    match observer.retire_completed_records()? {
        SubmissionRetirement::CompleteSnapshot => Ok(()),
        // Busy and any future unproven disposition cannot advance the bound.
        _ => Err(Error::PrefillScopeUnavailable),
    }
}

/// Actual fixed alias, query and failure transports. No new native owner or
/// allocation is constructed; native failure retains its original carrier.
pub(crate) fn control_bytes() -> Option<u64> {
    use std::mem::{size_of, size_of_val};
    let parts = [
        OriginalScopeObserver::control_bytes()?,
        size_of::<OriginalScopeObserver>(),
        size_of::<Option<OriginalScopeObserver>>(),
        size_of::<Result<OriginalScopeObserver, Error>>(),
        size_of::<Result<Option<OriginalScopeObserver>, Error>>(),
        size_of::<&OriginalScopeObserver>(),
        size_of::<SubmissionRetirement>(),
        size_of::<Result<SubmissionRetirement, safemlx::error::Exception>>(),
        size_of::<Result<(), Error>>(),
        size_of::<Result<super::Status, Error>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
        .and_then(|bytes| u64::try_from(bytes).ok())
}

//! One original shared arena, before Prompt or any prediction Scope.
use super::*;
use eredu_runtime::working_memory::{OriginalSubmissionTracking, SubmissionTrackingFacts};
use safemlx::{PreparedSubmissionRecordQuota, SubmissionRecordQuota, SubmissionRecordQuotaCause};
use std::{mem::size_of, num::NonZeroU64};

#[derive(Debug, thiserror::Error)]
#[error("original submission tracking construction failed: {cause}")]
struct Failure {
    #[source]
    cause: SubmissionRecordQuotaCause,
    // Last: no raw/native/quote backedge, and no erased preparation-node Box.
    original: OriginalSubmissionTracking,
}

pub(super) fn facts(
    capacity: Option<NonZeroU64>,
) -> Result<Option<SubmissionTrackingFacts>, Error> {
    selected_facts(capacity, capacity)
}

pub(super) fn selected_facts(
    requested: Option<NonZeroU64>,
    capacity: Option<NonZeroU64>,
) -> Result<Option<SubmissionTrackingFacts>, Error> {
    let (requested, capacity) = match (requested, capacity) {
        (None, None) => return Ok(None),
        (requested, Some(capacity)) if requested.is_none_or(|ceiling| capacity <= ceiling) => {
            (requested, capacity)
        }
        _ => return Err(memory(WorkingMemoryError::IdentityMismatch)),
    };
    let native_capacity =
        usize::try_from(capacity.get()).map_err(|_| memory(WorkingMemoryError::Overflow))?;
    let layout =
        PreparedSubmissionRecordQuota::<OriginalSubmissionTracking>::layout(native_capacity)
            .map_err(|cause| Error::Other(Box::new(cause)))?;
    let source = eredu_core::BackendFailure::source_retention_peak_bytes::<Failure>()
        .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    let controls = [
        size_of::<Option<OriginalSubmissionTracking>>(),
        size_of::<Result<Option<OriginalSubmissionTracking>, Error>>(),
        size_of::<Result<SubmissionRecordQuota, Error>>(),
        size_of::<Option<SubmissionRecordQuota>>(),
        size_of::<Failure>(),
        // Requested/selected policy transport and the complete-population query.
        size_of::<(Option<NonZeroU64>, Option<NonZeroU64>)>(),
        size_of::<(usize, bool)>(), // checked extent sum and completeness
        size_of::<Result<NonZeroU64, WorkingMemoryError>>(),
        source,
    ]
    .into_iter()
    .try_fold(
        layout
            .total_bytes()
            .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
        usize::checked_add,
    )
    .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    Ok(Some(
        SubmissionTrackingFacts::for_policy(
            requested,
            capacity,
            u64::try_from(controls).map_err(|_| memory(WorkingMemoryError::Overflow))?,
        )
        .map_err(memory)?,
    ))
}

pub(super) fn allocate(
    original: OriginalSubmissionTracking,
) -> Result<SubmissionRecordQuota, Error> {
    // The constructor consumes its capacity directly from the one accepted
    // capsule. No caller amount, replacement claim or additional hold is used.
    let capacity = original.facts().capacity().get() as usize;
    let prepared = PreparedSubmissionRecordQuota::try_new(capacity, original).map_err(|error| {
        let (cause, original) = error.into_parts();
        Error::with_original_control_source(
            eredu_core::BackendFailure::new(
                eredu_core::BackendFailureKind::InvalidSession,
                Failure { cause, original },
            ),
            false,
        )
    })?;
    prepared.try_allocate().map_err(|error| {
        let (cause, prepared) = error.into_parts();
        let original = prepared.into_owner(); // unbox before original custody
        Error::with_original_control_source(
            eredu_core::BackendFailure::new(
                eredu_core::BackendFailureKind::InvalidSession,
                Failure { cause, original },
            ),
            false,
        )
    })
}

#[cfg(all(
    test,
    feature = "metal",
    target_vendor = "apple",
    not(feature = "cuda")
))]
mod tests;

//! Actual ordered scalar readouts inside the same accepted model/prefill role.
use super::*;
use crate::backend::array_copy::{CaptureTensorNativeError, TokenScoreProgram};
use crate::backend::runtime::residency::storage::StorageIdentity;
use eredu_core::capture::{CandidateDomain, CapturePhase, CaptureTokenDomain};
use eredu_runtime::working_memory::{
    CaptureSourceSegment, CaptureTokenScoreClaim, ClaimedCaptureTokenScores, WorkingMemoryError,
};
fn error<E: std::error::Error + Send + Sync + 'static>(cause: E) -> Error {
    Error::Other(Box::new(cause))
}

impl FundedWork {
    pub(in crate::composition::mlx::session::model_session::text_funding) fn capture_token_scores_with_completion(
        &self,
        source: &Array,
        claim: CaptureTokenScoreClaim<'_, '_>,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
        domain: Option<CaptureTokenDomain<'_>>,
    ) -> Result<ClaimedCaptureTokenScores, Error> {
        if claim.geometry().phase() == CapturePhase::Prefill {
            return self.capture_prefill_token_scores(source, claim, stream, completion, domain);
        }
        // Successful decode readout roots are private scratch, not model/state.
        // Failure keeps the whole prefix in the existing original Work owner.
        let first = self.roots.try_borrow().map_err(error)?.len();
        let receipt = self.capture_token_scores_into(
            source,
            claim,
            stream,
            completion,
            domain,
            &self.roots,
            &self.publications,
            None,
        )?;
        self.retire_completed_readout_roots(first, completion)?;
        Ok(receipt)
    }
    pub(super) fn capture_token_scores_into(
        &self,
        source: &Array,
        claim: CaptureTokenScoreClaim<'_, '_>,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
        domain: Option<CaptureTokenDomain<'_>>,
        roots: &RefCell<Vec<Array>>,
        publications: &RefCell<Vec<RetainedStoragePublication>>,
        segment: Option<&mut CaptureSourceSegment>,
    ) -> Result<ClaimedCaptureTokenScores, Error> {
        let program = TokenScoreProgram::from_geometry(claim.geometry()).map_err(error)?;
        program.validate_source(source).map_err(error)?;
        let summary = domain.map(|domain| domain.summary(claim.geometry().vocabulary() as u32));
        let population = program.population().map_err(error)?;
        let roots_count = population
            .retained_outputs
            .checked_add(1)
            .ok_or_else(|| error(WorkingMemoryError::Overflow))?;
        self.with_capture_readout(
            source,
            claim,
            stream,
            completion,
            roots,
            publications,
            segment,
            roots_count,
            |claim, scope, segment, source_pin, retain| {
                let mut destination = match segment {
                    Some(segment) => {
                        claim.prepare_with_segment_source(scope, segment, source_pin, summary)
                    }
                    None => claim.prepare_with_source(scope, source_pin, summary),
                }
                .map_err(error)?;
                destination.validate().map_err(error)?;
                let partition = program
                    .execute(
                        source,
                        stream,
                        completion,
                        retain,
                        &mut |count| {
                            #[cfg(test)]
                            crate::backend::array_copy::CandidateExtraction::record_host_values(
                                count,
                            );
                            #[cfg(not(test))]
                            let _ = count;
                        },
                        |id| domain.is_none_or(|domain| domain.filter.allows(id)),
                        |score| {
                            destination
                                .push(score)
                                .map_err(CaptureTensorNativeError::from)
                        },
                    )
                    .map_err(error)?;
                destination.validate().map_err(error)?;
                destination.finish(partition).map_err(error)
            },
        )
    }
}

/// Concrete scalar-worker adapter and every possible typed Box shell. Existing
/// per-callback native population separately prices clones/frontiers/inspection;
/// this query is consumed before the enclosing original work owner is born.
pub(super) fn control_bytes() -> Option<usize> {
    use eredu_runtime::working_memory::{
        CaptureTokenScoreFailure, ScheduledCaptureTokenScoresTransfer, WorkingMemoryStorage,
    };
    use std::mem::{size_of, size_of_val};
    let frames = [
        size_of::<[Option<CaptureTokenDomain<'static>>; 5]>(),
        size_of::<[Option<CandidateDomain>; 2]>(),
        size_of::<TokenScoreProgram<'static>>(),
        size_of::<CaptureTokenScoreClaim<'static, 'static>>(),
        size_of::<ScheduledCaptureTokenScoresTransfer<'static, 'static, 'static, StorageIdentity>>(
        ),
        size_of::<Result<ClaimedCaptureTokenScores, Error>>(),
        size_of::<Result<WorkingMemoryStorage<StorageIdentity>, WorkingMemoryError>>(),
        size_of::<Option<&mut CaptureSourceSegment>>(),
        size_of::<Option<RetainedStoragePublication>>(),
        size_of::<std::cell::RefMut<'static, Vec<Array>>>(),
        size_of::<(CaptureCompletion<'static>, &Array, &Stream, &FundedWork)>(),
        size_of::<(
            &RefCell<Vec<Array>>,
            &RefCell<Vec<RetainedStoragePublication>>,
        )>(),
        size_of::<CaptureTokenScoreFailure>(),
        size_of::<Box<CaptureTokenScoreFailure>>(),
        size_of::<eredu_runtime::working_memory::CaptureRunHostError>(),
        size_of::<Box<eredu_runtime::working_memory::CaptureRunHostError>>(),
        size_of::<std::cell::BorrowError>(),
        size_of::<Box<std::cell::BorrowError>>(),
        size_of::<std::cell::BorrowMutError>(),
        size_of::<Box<std::cell::BorrowMutError>>(),
        size_of::<std::collections::TryReserveError>(),
        size_of::<Box<std::collections::TryReserveError>>(),
        size_of::<CaptureTensorNativeError>(),
        size_of::<Box<CaptureTensorNativeError>>(),
        size_of::<WorkingMemoryError>(),
        size_of::<Box<WorkingMemoryError>>(),
        size_of::<safemlx::error::Exception>(),
        size_of::<Box<safemlx::error::Exception>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)?
        .checked_mul(3)?
        .checked_add(publication_scope::control_bytes()?)
}

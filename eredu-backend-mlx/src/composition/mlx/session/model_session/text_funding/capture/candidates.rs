//! Existing sorted candidate worker under the same original source/readout owner.
use super::*;
use crate::backend::array_copy::{CandidateExtraction, CaptureTensorNativeError};
use eredu_core::capture::{CandidateDomain, CapturePhase, CaptureTokenDomain};
use eredu_runtime::working_memory::{
    CaptureCandidateClaim, CaptureSourceSegment, ClaimedCaptureCandidates,
};
fn error<E: std::error::Error + Send + Sync + 'static>(cause: E) -> Error {
    Error::Other(Box::new(cause))
}
impl FundedWork {
    pub(in crate::composition::mlx::session::model_session::text_funding) fn capture_candidates_with_completion(
        &self,
        source: &Array,
        claim: CaptureCandidateClaim<'_, '_>,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
        domain: Option<CaptureTokenDomain<'_>>,
    ) -> Result<ClaimedCaptureCandidates, Error> {
        if claim.geometry().phase() == CapturePhase::Prefill {
            return self.capture_prefill_candidates(source, claim, stream, completion, domain);
        }
        let first = self.roots.try_borrow().map_err(error)?.len();
        let receipt = self.capture_candidates_into(
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
    pub(super) fn capture_candidates_into(
        &self,
        source: &Array,
        claim: CaptureCandidateClaim<'_, '_>,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
        domain: Option<CaptureTokenDomain<'_>>,
        roots: &RefCell<Vec<Array>>,
        publications: &RefCell<Vec<RetainedStoragePublication>>,
        segment: Option<&mut CaptureSourceSegment>,
    ) -> Result<ClaimedCaptureCandidates, Error> {
        let program = CandidateExtraction::from_geometry(claim.geometry()).map_err(error)?;
        program.validate_source(source).map_err(error)?;
        let summary = domain.map(|domain| domain.summary(claim.geometry().vocabulary() as u32));
        self.with_capture_readout(
            source,
            claim,
            stream,
            completion,
            roots,
            publications,
            segment,
            1 + CandidateExtraction::ROOTS,
            |claim, scope, segment, source_pin, retain| {
                let mut destination = match segment {
                    Some(segment) => {
                        claim.prepare_with_segment_source(scope, segment, source_pin, summary)
                    }
                    None => claim.prepare_with_source(scope, source_pin, summary),
                }
                .map_err(error)?;
                destination.validate().map_err(error)?;
                let (ids, scores) = program
                    .execute_with_completion(source, stream, completion, retain)
                    .map_err(error)?;
                program
                    .read_with_completion(&ids, &scores, stream, completion, |id, score| {
                        destination
                            .push(
                                id,
                                score,
                                domain.is_none_or(|domain| domain.filter.allows(id)),
                            )
                            .map_err(CaptureTensorNativeError::from)
                    })
                    .map_err(error)?;
                destination.validate().map_err(error)?;
                destination.finish().map_err(error)
            },
        )
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use crate::backend::runtime::residency::storage::StorageIdentity;
    use eredu_runtime::working_memory::{
        CaptureCandidateFailure, CaptureRunHostError, ScheduledCaptureCandidatesTransfer,
    };
    use std::mem::{size_of, size_of_val};
    type Transfer = ScheduledCaptureCandidatesTransfer<'static, 'static, 'static, StorageIdentity>;
    let frames = [
        size_of::<[Option<CaptureTokenDomain<'static>>; 5]>(),
        size_of::<[Option<CandidateDomain>; 2]>(),
        size_of::<CandidateExtraction>(), // validated local survives through callback
        size_of::<(
            CandidateExtraction,
            &Array,
            &Stream,
            CaptureCompletion<'static>,
        )>(), // callback captures
        size_of::<CaptureCandidateClaim<'static, 'static>>(), // entry-to-inner moved argument
        size_of::<Transfer>(),            // fixed destination live alongside IDs/scores
        size_of::<Result<Transfer, CaptureRunHostError>>(), // destination construction return
        size_of::<(Array, Array)>(),
        size_of::<Result<(Array, Array), CaptureTensorNativeError>>(),
        size_of::<CaptureCandidateFailure>(),
        size_of::<Box<CaptureCandidateFailure>>(),
        // Three distinct nested call/result frames: typed callback returns into
        // with_capture_readout, that returns into capture_candidates_into, then
        // the model/carrier entry returns into the scheduled observer. Payloads
        // move once; reserving every frame does not allocate extra result owners.
        size_of::<Result<ClaimedCaptureCandidates, Error>>(),
        size_of::<Result<ClaimedCaptureCandidates, Error>>(),
        size_of::<Result<ClaimedCaptureCandidates, Error>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)?
        .checked_add(super::readout::control_bytes::<
            CaptureCandidateClaim<'static, 'static>,
            ClaimedCaptureCandidates,
        >()?)
}

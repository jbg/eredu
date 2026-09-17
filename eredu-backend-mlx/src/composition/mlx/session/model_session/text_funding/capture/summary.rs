//! Scalar summary through the shared source completion/publication and claim worker.
use super::*;
use crate::backend::array_copy::{CaptureTensorNativeError, PreparedCaptureSummary};
use eredu_core::capture::CapturePhase;
use eredu_runtime::working_memory::{
    CaptureSourceSegment, CaptureSummaryClaim, ClaimedCaptureSummary,
};
fn error<E: std::error::Error + Send + Sync + 'static>(cause: E) -> Error {
    Error::Other(Box::new(cause))
}
impl FundedWork {
    pub(in crate::composition::mlx::session::model_session::text_funding) fn capture_summary_with_completion(
        &self,
        source: &Array,
        claim: CaptureSummaryClaim<'_, '_>,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
    ) -> Result<ClaimedCaptureSummary, Error> {
        if claim.geometry().phase() == CapturePhase::Prefill {
            return self.capture_prefill_summary(source, claim, stream, completion);
        }
        let first = self.roots.try_borrow().map_err(error)?.len();
        let receipt = self.capture_summary_into(
            source,
            claim,
            stream,
            completion,
            &self.roots,
            &self.publications,
            None,
        )?;
        self.retire_completed_readout_roots(first, completion)?;
        Ok(receipt)
    }
    pub(super) fn capture_summary_into(
        &self,
        source: &Array,
        claim: CaptureSummaryClaim<'_, '_>,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
        roots: &RefCell<Vec<Array>>,
        publications: &RefCell<Vec<RetainedStoragePublication>>,
        segment: Option<&mut CaptureSourceSegment>,
    ) -> Result<ClaimedCaptureSummary, Error> {
        let program = PreparedCaptureSummary::from_geometry(claim.geometry()).map_err(error)?;
        program.validate_source(source).map_err(error)?;
        let population = program
            .population()
            .ok_or_else(|| error(eredu_runtime::working_memory::WorkingMemoryError::Overflow))?;
        self.with_capture_readout(
            source,
            claim,
            stream,
            completion,
            roots,
            publications,
            segment,
            population.retained_roots,
            |claim, scope, segment, pin, retain| {
                let destination = match segment {
                    Some(segment) => claim.prepare_with_segment_source(scope, segment, pin),
                    None => claim.prepare_with_source(scope, pin),
                }
                .map_err(error)?;
                destination.validate().map_err(error)?;
                let value = program
                    .execute(source, stream, completion, roots, retain)
                    .map_err(error)?;
                destination.finish(value).map_err(error)
            },
        )
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use crate::backend::runtime::residency::storage::StorageIdentity;
    use eredu_runtime::working_memory::{
        CaptureRunHostError, CaptureSummaryFailure, ScheduledCaptureSummaryTransfer,
    };
    use std::mem::{size_of, size_of_val};
    type Transfer = ScheduledCaptureSummaryTransfer<'static, 'static, 'static, StorageIdentity>;
    let frames = [
        size_of::<PreparedCaptureSummary>(),
        size_of::<(
            PreparedCaptureSummary,
            &Array,
            &Stream,
            CaptureCompletion<'static>,
            &RefCell<Vec<Array>>,
        )>(),
        size_of::<CaptureSummaryClaim<'static, 'static>>(),
        size_of::<Transfer>(),
        size_of::<Result<Transfer, CaptureRunHostError>>(),
        size_of::<eredu_core::capture::CaptureSummary>(),
        size_of::<Result<eredu_core::capture::CaptureSummary, CaptureTensorNativeError>>(),
        size_of::<CaptureSummaryFailure>(),
        size_of::<Box<CaptureSummaryFailure>>(),
        // Observer entry, into helper, and shared callback each hold one result frame.
        size_of::<Result<ClaimedCaptureSummary, Error>>(),
        size_of::<Result<ClaimedCaptureSummary, Error>>(),
        size_of::<Result<ClaimedCaptureSummary, Error>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)?
        .checked_add(super::readout::control_bytes::<
            CaptureSummaryClaim<'static, 'static>,
            ClaimedCaptureSummary,
        >()?)
}

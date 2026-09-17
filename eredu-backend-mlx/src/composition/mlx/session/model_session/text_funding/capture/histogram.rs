//! Fixed-edge histogram through the shared source completion/publication and claim worker.
use super::*;
use crate::backend::array_copy::{CaptureTensorNativeError, PreparedCaptureHistogram};
use eredu_core::capture::CapturePhase;
use eredu_runtime::working_memory::{
    CaptureHistogramClaim, CaptureSourceSegment, ClaimedCaptureHistogram,
};
fn error<E: std::error::Error + Send + Sync + 'static>(cause: E) -> Error {
    Error::Other(Box::new(cause))
}
impl FundedWork {
    pub(in crate::composition::mlx::session::model_session::text_funding) fn capture_histogram_with_completion(
        &self,
        source: &Array,
        claim: CaptureHistogramClaim<'_, '_>,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
    ) -> Result<ClaimedCaptureHistogram, Error> {
        if claim.geometry().phase() == CapturePhase::Prefill {
            return self.capture_prefill_histogram(source, claim, stream, completion);
        }
        let first = self.roots.try_borrow().map_err(error)?.len();
        let receipt = self.capture_histogram_into(
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
    pub(super) fn capture_histogram_into(
        &self,
        source: &Array,
        claim: CaptureHistogramClaim<'_, '_>,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
        roots: &RefCell<Vec<Array>>,
        publications: &RefCell<Vec<RetainedStoragePublication>>,
        segment: Option<&mut CaptureSourceSegment>,
    ) -> Result<ClaimedCaptureHistogram, Error> {
        let program = PreparedCaptureHistogram::from_geometry(claim.geometry()).map_err(error)?;
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
                let mut destination = match segment {
                    Some(segment) => claim.prepare_with_segment_source(scope, segment, pin),
                    None => claim.prepare_with_source(scope, pin),
                }
                .map_err(error)?;
                destination.validate().map_err(error)?;
                let value = program
                    .execute(
                        source,
                        stream,
                        completion,
                        roots,
                        retain,
                        &mut |index, count| {
                            destination
                                .add_bin(index, count)
                                .map_err(CaptureTensorNativeError::from)
                        },
                    )
                    .map_err(error)?;
                destination
                    .finish(value.below, value.above, value.non_finite)
                    .map_err(error)
            },
        )
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use crate::backend::runtime::residency::storage::StorageIdentity;
    use eredu_runtime::working_memory::{
        CaptureHistogramFailure, CaptureRunHostError, ScheduledCaptureHistogramTransfer,
    };
    use std::mem::{size_of, size_of_val};
    type Transfer = ScheduledCaptureHistogramTransfer<'static, 'static, 'static, StorageIdentity>;
    let frames = [
        size_of::<PreparedCaptureHistogram<'static>>(),
        size_of::<(
            PreparedCaptureHistogram<'static>,
            &Array,
            &Stream,
            CaptureCompletion<'static>,
            &RefCell<Vec<Array>>,
        )>(),
        size_of::<CaptureHistogramClaim<'static, 'static>>(),
        size_of::<Transfer>(),
        size_of::<Result<Transfer, CaptureRunHostError>>(),
        size_of::<eredu_core::capture::CaptureHistogram>(),
        size_of::<crate::backend::array_copy::HistogramTotals>(),
        size_of::<&mut Transfer>(),
        size_of::<&mut dyn FnMut(usize, u64) -> Result<(), CaptureTensorNativeError>>(),
        size_of::<Result<crate::backend::array_copy::HistogramTotals, CaptureTensorNativeError>>(),
        size_of::<CaptureHistogramFailure>(),
        size_of::<Box<CaptureHistogramFailure>>(),
        // Observer entry, into helper, and shared callback each hold one result frame.
        size_of::<Result<ClaimedCaptureHistogram, Error>>(),
        size_of::<Result<ClaimedCaptureHistogram, Error>>(),
        size_of::<Result<ClaimedCaptureHistogram, Error>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)?
        .checked_add(super::readout::control_bytes::<
            CaptureHistogramClaim<'static, 'static>,
            ClaimedCaptureHistogram,
        >()?)
}

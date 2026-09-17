//! Canonical per-chunk histogram keeps roots with its existing retirement ticket.
use super::*;
use crate::backend::array_copy::PreparedCaptureHistogram;
use eredu_runtime::working_memory::{CaptureHistogramClaim, ClaimedCaptureHistogram};
impl FundedWork {
    pub(in crate::composition::mlx::session::model_session::text_funding) fn capture_prefill_histogram(
        &self,
        source: &Array,
        claim: CaptureHistogramClaim<'_, '_>,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
    ) -> Result<ClaimedCaptureHistogram, Error> {
        let program = PreparedCaptureHistogram::from_geometry(claim.geometry()).map_err(error)?;
        let population = program
            .population()
            .ok_or_else(|| error(WorkingMemoryError::Overflow))?;
        let mut slot = self.capture.try_borrow_mut().map_err(error)?;
        let capture: &mut CaptureCarrier = slot
            .as_mut()
            .ok_or_else(|| error(CaptureCarrierError::Identity))?;
        capture.require_capacity(population.retained_roots)?;
        let segment = capture
            .segment
            .as_mut()
            .ok_or_else(|| error(CaptureCarrierError::Identity))?;
        let attempt = CaptureAttempt::new(&capture.progress)?;
        let receipt = self.capture_histogram_into(
            source,
            claim,
            stream,
            completion,
            &capture.roots,
            &capture.publications,
            Some(segment),
        )?;
        attempt.complete();
        Ok(receipt)
    }
}

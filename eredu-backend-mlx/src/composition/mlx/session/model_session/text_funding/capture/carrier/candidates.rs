//! Canonical terminal prefill carrier uses the shared typed candidate readout.
use super::*;
use crate::backend::array_copy::CandidateExtraction;
use eredu_core::capture::CaptureTokenDomain;
use eredu_runtime::working_memory::{CaptureCandidateClaim, ClaimedCaptureCandidates};

impl FundedWork {
    pub(in crate::composition::mlx::session::model_session::text_funding) fn capture_prefill_candidates(
        &self,
        source: &Array,
        claim: CaptureCandidateClaim<'_, '_>,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
        domain: Option<CaptureTokenDomain<'_>>,
    ) -> Result<ClaimedCaptureCandidates, Error> {
        let mut slot = self.capture.try_borrow_mut().map_err(error)?;
        let capture: &mut CaptureCarrier = slot
            .as_mut()
            .ok_or_else(|| error(CaptureCarrierError::Identity))?;
        capture.require_capacity(1 + CandidateExtraction::ROOTS)?;
        let segment = capture
            .segment
            .as_mut()
            .ok_or_else(|| error(CaptureCarrierError::Identity))?;
        let attempt = CaptureAttempt::new(&capture.progress)?;
        let receipt = self.capture_candidates_into(
            source,
            claim,
            stream,
            completion,
            domain,
            &capture.roots,
            &capture.publications,
            Some(segment),
        )?;
        attempt.complete();
        Ok(receipt)
    }
}

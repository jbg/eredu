//! Same terminal source carrier and scalar worker, with no replacement scope.
use super::*;
use crate::backend::array_copy::TokenScoreProgram;
use eredu_runtime::working_memory::{CaptureTokenScoreClaim, ClaimedCaptureTokenScores};
impl FundedWork {
    pub(in crate::composition::mlx::session::model_session::text_funding) fn capture_prefill_token_scores(
        &self,
        source: &Array,
        claim: CaptureTokenScoreClaim<'_, '_>,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
    ) -> Result<ClaimedCaptureTokenScores, Error> {
        let population = TokenScoreProgram::from_geometry(claim.geometry())
            .and_then(|program| program.population())
            .map_err(error)?;
        let mut capture_slot = self.capture.try_borrow_mut().map_err(error)?;
        let capture: &mut CaptureCarrier = capture_slot
            .as_mut()
            .ok_or_else(|| error(CaptureCarrierError::Identity))?;
        capture.require_capacity(
            population
                .retained_outputs
                .checked_add(1)
                .ok_or_else(|| error(WorkingMemoryError::Overflow))?,
        )?;
        let segment = capture
            .segment
            .as_mut()
            .ok_or_else(|| error(CaptureCarrierError::Identity))?;
        let attempt = CaptureAttempt::new(&capture.progress)?;
        let receipt = self.capture_token_scores_into(
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

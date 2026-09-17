use super::*;
use crate::working_memory::PreparedCaptureDelivery;

impl<'a> ScheduledCaptureStep<'a> {
    /// Validate a completed receipt's actual wire size. A mismatch preserves all
    /// charges and the spent claim while converting that record to Failed.
    pub fn validate_record_encoding(&mut self, index: usize) -> Result<(), CaptureRunHostError> {
        self.frame.validate_record_encoding(index)?;
        Ok(())
    }

    /// Complete storage and allocate the final shared owner before transaction
    /// agreement. The returned owner has no source/claim-row borrow; only its
    /// infallible outcome finalizer remains. No native completion is implied.
    pub fn prepare_delivery(
        self,
        step_usage: CaptureUsage,
        cumulative_usage: CaptureUsage,
        capture_seconds: f64,
    ) -> Result<PreparedCaptureDelivery, ScheduledCaptureStepFinishError<'a>> {
        let Self { frame, claim } = self;
        frame
            .prepare_delivery(step_usage, cumulative_usage, capture_seconds)
            .map_err(|error| ScheduledCaptureStepFinishError { error, claim })
    }
}

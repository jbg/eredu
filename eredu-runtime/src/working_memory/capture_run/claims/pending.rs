use super::*;

impl ScheduledCaptureStep<'_> {
    /// Detach an aborted frame without allocation, validation or native work.
    ///
    /// Releases the exclusive source/claim-row borrow while the exact partial
    /// frame and unused sidecars retain original whole-run custody. Already spent
    /// claims remain spent. This can run during unwind; subsequent fallible
    /// delivery is performed only after the enclosing native recovery settles.
    pub fn into_aborted_pending(
        self,
        step_usage: CaptureUsage,
        cumulative_usage: CaptureUsage,
        capture_seconds: f64,
    ) -> crate::working_memory::PendingCaptureDelivery {
        let Self { frame, claim } = self;
        let pending = frame.into_aborted_pending(step_usage, cumulative_usage, capture_seconds);
        drop(claim);
        pending
    }
}

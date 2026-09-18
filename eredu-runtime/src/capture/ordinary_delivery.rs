//! Preallocated retained frame delivery for every ordinary capture source.
use super::*;

impl CaptureSession {
    /// Includes active transactions and preallocated frames awaiting delivery.
    /// This read-only query never certifies native completion.
    pub fn has_pending_step(&self) -> bool {
        self.records.is_some()
            || self.ordinary_frame.is_some()
            || self.transaction.is_some()
            || self.ordinary_progress.is_some()
    }
    pub(super) fn prepare_frame(&mut self) -> Result<PreparedCapturedStep, CaptureError> {
        let host = self.owner.retained()?;
        policy::reserve_required(&mut self.ledger, policy::frame_usage()?)?;
        Ok(PreparedCapturedStep::retain(host))
    }
    /// Drains one retained ordinary frame only after the exact caller's native
    /// completion and the complete prefill/final-index transaction have settled.
    /// Pending probes do not consume custody. No allocation or host lock is
    /// required by the final owner move, including aborted/empty/skip frames.
    pub fn take_shared_step(&mut self) -> Option<SharedCapturedStep> {
        if self.ordinary_progress.is_some() {
            return None;
        }
        let step = self.take_step_inner()?;
        Some(self.publish_step(step))
    }
    // Called only while the enclosing runtime owns the unpublished frame. This
    // allows speculative abort attribution before the immutable owner escapes.
    pub(super) fn publish_step(&mut self, step: CapturedStep) -> SharedCapturedStep {
        // begin_step_inner installs these together before any later fallible
        // step work; no other ordinary constructor can install records.
        let owner = self
            .ordinary_frame
            .take()
            .expect("ordinary records have preallocated custody");
        owner.finish(step)
    }
}

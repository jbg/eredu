//! Retained delivery for the installed ordinary prepared-media source mode.
use super::*;
use eredu_core::HostPreparationAuthority;

impl CaptureSession {
    /// Whether this run requires retained ordinary frame delivery, including
    /// decode after the one whole-prefill collector has finished.
    pub fn uses_ordinary_shared_delivery(&self) -> bool {
        self.ordinary_prefill.is_some()
    }
    /// Actual pending/ready host delivery; an unstarted installed source is idle.
    /// This does not certify native completion.
    pub fn ordinary_shared_delivery_pending(&self) -> bool {
        self.uses_ordinary_shared_delivery()
            && (self.records.is_some()
                || self.ordinary_frame.is_some()
                || self.transaction.is_some()
                || self.ordinary_progress.is_some())
    }
    pub(super) fn prepare_ordinary_frame(
        &mut self,
    ) -> Result<Option<PreparedCapturedStep>, CaptureError> {
        if !self.uses_ordinary_shared_delivery() {
            return Ok(None);
        }
        let host = self.owner.retained()?;
        let bytes = PreparedCapturedStep::retained_control_bytes::<HostPreparationAuthority>()
            .ok_or(CaptureError::Overflow)?;
        policy::reserve_required(
            &mut self.ledger,
            CaptureUsage {
                host_bytes: bytes,
                ..CaptureUsage::default()
            },
        )?;
        Ok(Some(PreparedCapturedStep::retain(host)))
    }
    /// Drains one retained ordinary frame only after the exact caller's native
    /// completion and the complete prefill/final-index transaction have settled.
    /// Pending/raw probes do not consume custody. No allocation or host lock is
    /// required by the final owner move, including aborted/empty/skip frames.
    pub fn take_ordinary_shared_step(&mut self) -> Option<SharedCapturedStep> {
        if !self.uses_ordinary_shared_delivery() || self.ordinary_progress.is_some() {
            return None;
        }
        let step = self.take_step_inner()?;
        // begin_step_inner installs these together before any later fallible
        // step work; no other ordinary constructor can install records.
        let owner = self
            .ordinary_frame
            .take()
            .expect("ordinary records have preallocated custody");
        Some(owner.finish(step))
    }
}

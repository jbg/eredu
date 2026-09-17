//! Empty, preallocated frame storage for an already owned ordinary collector.
use super::*;

/// Unique preallocated frame control before record construction.
///
/// This storage bridge grants no original allowance, quota, source authority or
/// native completion. The caller reserves its concrete control before `retain`
/// and owns every later DTO allocation independently. No payload or alias can
/// escape before the consuming `finish` call.
#[doc(hidden)]
pub struct PreparedCapturedStep(StepOwner);
impl PreparedCapturedStep {
    /// Exact existing shared frame and concrete custody allocation layouts.
    pub fn retained_control_bytes<T: Send + Sync + 'static>() -> Option<u64> {
        UnpublishedCapturedStep::retained_control_bytes::<T>()
    }
    /// Allocates only the final Arc and concrete custody Box. All placeholder
    /// vectors are empty; its fields cannot hold arbitrary destructors.
    pub fn retain(custody: impl Send + Sync + 'static) -> Self {
        let placeholder = CapturedStep {
            outcome: CaptureStepOutcome::Untracked,
            phase: CapturePhase::Prefill,
            invocation: None,
            prediction_index: 0,
            records: Vec::new(),
            partitions: Vec::new(),
            interventions: Vec::new(),
            step_usage: CaptureUsage::default(),
            cumulative_usage: CaptureUsage::default(),
            capture_seconds: 0.0,
        };
        Self(UnpublishedCapturedStep::retain(placeholder, custody).0)
    }
    /// Moves the already constructed DTO into the same unique allocation.
    /// No allocation, callback, validation, locking or completion occurs here.
    /// The replaced value is always the closed empty constructor placeholder.
    pub fn finish(mut self, step: CapturedStep) -> SharedCapturedStep {
        *self.0.get_mut() = step;
        SharedCapturedStep(self.0)
    }
}
impl fmt::Debug for PreparedCapturedStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedCapturedStep")
            .finish_non_exhaustive()
    }
}
#[cfg(test)]
mod tests;

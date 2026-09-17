//! Owned aborted payload detached before a lexical observer unwinds.
use super::delivery::{OwnedCaptureDelivery, prepare_owned_delivery};
use super::*;

/// An aborted fixed frame awaiting the enclosing operation's settled delivery.
///
/// This contains no plan or claim-row borrow. Payload and unused sidecars retain
/// their original custody. Detachment performs no allocation, validation, quota
/// refund or native certification; the enclosing driver must settle native work
/// before attempting the fallible final delivery. No payload accessor, clone,
/// mutation, retry claim or committed-outcome conversion is exposed.
#[must_use = "finish after native recovery or retain/retire this protected aborted payload"]
pub struct PendingCaptureDelivery {
    payload: OwnedCaptureDelivery,
    step_usage: CaptureUsage,
    cumulative_usage: CaptureUsage,
    capture_seconds: f64,
}
impl PreparedCaptureStep<'_> {
    pub(in crate::working_memory) fn into_aborted_pending(
        mut self,
        step_usage: CaptureUsage,
        cumulative_usage: CaptureUsage,
        capture_seconds: f64,
    ) -> PendingCaptureDelivery {
        self.flush_intervention_evidence();
        let Self {
            frame,
            buffers,
            intervention_buffers,
            intervention_evidence,
            prefill,
            plan: _,
            partition_metadata,
            custody,
        } = self;
        drop(intervention_evidence);
        PendingCaptureDelivery {
            payload: OwnedCaptureDelivery {
                frame,
                buffers,
                intervention_buffers,
                prefill,
                partition_metadata,
                custody,
            },
            step_usage,
            cumulative_usage,
            capture_seconds,
        }
    }
}
impl PendingCaptureDelivery {
    #[cfg(test)]
    pub(crate) fn prefill_storage(
        &self,
        index: usize,
    ) -> Option<(*const f32, usize, usize, usize)> {
        let tensor = self
            .payload
            .prefill
            .as_ref()?
            .slots
            .get(index)?
            .tensor
            .as_ref()?;
        Some((
            tensor.data.as_ptr(),
            tensor.data.capacity(),
            tensor.data.len(),
            tensor.covered,
        ))
    }

    /// Validates and publishes this same already-protected payload as Aborted.
    /// This may allocate its final shared owner and reject a closed/quarantined
    /// account; it is never called from an infallible transaction finish or Drop.
    /// Successful host publication does not certify outstanding native work.
    pub fn finish(self) -> Result<SharedCapturedStep, PendingCaptureDeliveryError> {
        let Self {
            payload,
            step_usage,
            cumulative_usage,
            capture_seconds,
        } = self;
        match prepare_owned_delivery(payload, step_usage, cumulative_usage, capture_seconds) {
            Ok(prepared) => Ok(prepared.finish(CaptureStepOutcome::Aborted)),
            Err(error) => {
                let (payload, error) = error.into_parts();
                Err(PendingCaptureDeliveryError {
                    error,
                    pending: Self {
                        payload,
                        step_usage,
                        cumulative_usage,
                        capture_seconds,
                    },
                })
            }
        }
    }
}
impl fmt::Debug for PendingCaptureDelivery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingCaptureDelivery")
            .finish_non_exhaustive()
    }
}
/// Failed aborted delivery retains the same original payload and host custody.
/// Recovering this owner neither changes completion metadata nor issues a claim.
#[derive(Debug)]
pub struct PendingCaptureDeliveryError {
    error: CaptureStepError,
    pending: PendingCaptureDelivery,
}
impl PendingCaptureDeliveryError {
    /// Original typed delivery rejection.
    pub fn error(&self) -> &CaptureStepError {
        &self.error
    }
    pub(crate) fn into_parts(self) -> (PendingCaptureDelivery, CaptureStepError) {
        (self.pending, self.error)
    }
    /// Retain/retry the same pending payload, without any new constructor grant.
    pub fn into_pending(self) -> PendingCaptureDelivery {
        self.pending
    }
}
impl fmt::Display for PendingCaptureDeliveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, f)
    }
}
impl std::error::Error for PendingCaptureDeliveryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

//! Delivery preserves the actual ownership kind without allocating a DTO copy.
use super::{CapturedStep, SharedCapturedStep};

/// One capture frame delivered by a legacy or retained collector.
///
/// The carrier neither copies payload nor establishes completion or funding.
/// It deliberately has no Clone implementation or implicit owning raw export.
/// Its inline slot/control size is a separate preparation contribution; it is
/// not included in an independently measured frame payload by this type.
#[derive(Debug, PartialEq)]
pub enum CapturedStepDelivery {
    /// Existing caller-owned raw frame, preserving legacy backend behavior.
    Legacy(CapturedStep),
    /// Shared immutable frame with its original retained custody.
    Shared(SharedCapturedStep),
}
impl CapturedStepDelivery {
    /// Borrows the record. Explicit raw cloning is separate caller-owned work.
    pub fn as_step(&self) -> &CapturedStep {
        match self {
            Self::Legacy(step) => step,
            Self::Shared(step) => step.as_step(),
        }
    }
    /// Borrows retained ownership without detaching it from the frame.
    pub fn shared(&self) -> Option<&SharedCapturedStep> {
        match self {
            Self::Shared(step) => Some(step),
            Self::Legacy(_) => None,
        }
    }
    /// Moves a legacy frame only. A retained frame returns unchanged, still
    /// owning its custody; no allocation, payload clone or refund occurs.
    pub fn into_legacy(self) -> Result<CapturedStep, Self> {
        match self {
            Self::Legacy(step) => Ok(step),
            retained => Err(retained),
        }
    }
}
impl AsRef<CapturedStep> for CapturedStepDelivery {
    fn as_ref(&self) -> &CapturedStep {
        self.as_step()
    }
}

/// Another prediction requires draining the preceding capture transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("text capture has undrained records or a pending transaction")]
pub struct CaptureDeliveryPending;

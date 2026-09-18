//! Closed source origin: ordinary requests and speculative occurrence accounts
//! cannot substitute for one another or reset a span ordinal.
use crate::working_memory::{
    InferenceRequest, OriginalSpeculativePrefillSpan, OriginalSpeculativeRole, WorkingMemoryError,
};

pub(crate) enum MediaPrefillOrigin {
    Inference(InferenceRequest),
    Equation,
    Speculative {
        role: OriginalSpeculativeRole,
        next: usize,
    },
}
impl MediaPrefillOrigin {
    pub(crate) fn speculative(
        role: OriginalSpeculativeRole,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<Self, WorkingMemoryError> {
        role.validate_prefill_geometry(geometry)?;
        Ok(Self::Speculative { role, next: 0 })
    }
    pub(crate) fn request(&self) -> Result<&InferenceRequest, WorkingMemoryError> {
        match self {
            Self::Inference(request) => Ok(request),
            Self::Speculative { .. } | Self::Equation => Err(WorkingMemoryError::IdentityMismatch),
        }
    }
    pub(crate) fn validate_span(
        &self,
        span: &OriginalSpeculativePrefillSpan,
    ) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Speculative { role, next }
                if *next == span.ordinal() && role.same_role(span.role()) =>
            {
                Ok(())
            }
            _ => Err(WorkingMemoryError::IdentityMismatch),
        }
    }
    pub(crate) fn committed(&mut self) -> Result<(), WorkingMemoryError> {
        if let Self::Speculative { next, .. } = self {
            *next = next.checked_add(1).ok_or(WorkingMemoryError::Overflow)?;
        }
        Ok(())
    }
}

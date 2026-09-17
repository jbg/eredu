//! Carrier adaptation only. No envelope allocation or physical admission proof.
use super::*;

impl ObservedGenerationEvent {
    /// Borrows captures from either ownership representation without copying.
    /// Explicit cloning of the raw DTO is separate caller-owned work.
    pub fn captures(&self) -> Option<&CapturedStep> {
        match self {
            Self::Token { captures, .. } => captures.as_ref(),
            Self::CaptureFailure { captures, .. } => Some(captures),
            Self::SharedToken { captures, .. } | Self::SharedCaptureFailure { captures, .. } => {
                Some(captures.as_step())
            }
            _ => None,
        }
    }

    /// Borrows the original shared frame and its custody, if present.
    /// This does not account for or protect the surrounding facade envelope.
    pub fn shared_captures(&self) -> Option<&SharedCapturedStep> {
        match self {
            Self::SharedToken { captures, .. } | Self::SharedCaptureFailure { captures, .. } => {
                Some(captures)
            }
            _ => None,
        }
    }

    pub(in crate::api) fn from_token_delivery(
        token_id: u32,
        forced: bool,
        prediction_index: u64,
        input_range: [u64; 2],
        captures: Option<CapturedStepDelivery>,
        step_seconds: f64,
    ) -> Self {
        let captures = match captures {
            Some(CapturedStepDelivery::Shared(captures)) => {
                return Self::SharedToken {
                    token_id,
                    forced,
                    prediction_index,
                    input_range,
                    committed: true,
                    rank: 0,
                    captures,
                    step_seconds,
                }
            }
            Some(CapturedStepDelivery::Legacy(captures)) => Some(captures),
            None => None,
        };
        Self::Token {
            token_id,
            forced,
            prediction_index,
            input_range,
            committed: true,
            rank: 0,
            captures,
            step_seconds,
        }
    }

    pub(in crate::api) fn from_failed_delivery(
        prediction_index: u64,
        input_range: [u64; 2],
        captures: CapturedStepDelivery,
        step_seconds: f64,
    ) -> Self {
        match captures {
            CapturedStepDelivery::Legacy(captures) => Self::CaptureFailure {
                prediction_index,
                input_range,
                captures,
                step_seconds,
            },
            CapturedStepDelivery::Shared(captures) => Self::SharedCaptureFailure {
                prediction_index,
                input_range,
                captures,
                step_seconds,
            },
        }
    }
}

#[cfg(test)]
mod ordinary_frame_tests;

//! One retained event representation for ordinary and admitted capture.
use super::*;

impl ObservedGenerationEvent {
    /// Borrows diagnostics without detaching their retained custody.
    pub fn captures(&self) -> Option<&CapturedStep> {
        self.shared_captures().map(SharedCapturedStep::as_step)
    }
    /// Borrows the original frame owner, for any instrumented event.
    pub fn shared_captures(&self) -> Option<&SharedCapturedStep> {
        match self {
            Self::Token { captures, .. } => captures.as_ref(),
            Self::CaptureFailure { captures, .. } => Some(captures),
            _ => None,
        }
    }
    pub(in crate::api) fn from_token_delivery(
        token_id: u32,
        forced: bool,
        prediction_index: u64,
        input_range: [u64; 2],
        captures: Option<SharedCapturedStep>,
        step_seconds: f64,
    ) -> Self {
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
        captures: SharedCapturedStep,
        step_seconds: f64,
    ) -> Self {
        Self::CaptureFailure {
            prediction_index,
            input_range,
            captures,
            step_seconds,
        }
    }
}
#[cfg(test)]
mod ordinary_frame_tests;

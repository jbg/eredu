//! Cached opening geometry for ordinary request-local capture coordinates.
use super::*;

/// Positions already cached before this request's new prompt is processed.
///
/// This changes only ordinary Context axes: cached + new prompt + request-local
/// prediction. Prefill prediction is zero and the first ordinary decode is one.
/// It does not rebase observation schedules, carry an absolute prediction origin,
/// prove actual cached state, or grant execution/capture/funding authority.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureTextOrigin {
    /// Existing decoder positions at opening, excluding this request's prompt.
    pub cached_positions: u64,
}
impl CaptureTextOrigin {
    /// Allocation-free scalar geometry with checked context addition. Sequence
    /// still describes only the new prompt (prefill) or one token (decode).
    /// Use the retained admitted plan at execution so origin cannot be replaced.
    pub fn invocation_shape(
        self,
        request: CaptureRequestShape,
        phase: CapturePhase,
        prediction: u64,
    ) -> Result<CaptureInvocationShape, CaptureError> {
        let mut shape = request.invocation_shape(phase, prediction)?;
        shape.context = Some(add(
            self.cached_positions,
            shape.context.expect("ordinary context"),
        )?);
        Ok(shape)
    }
    pub(super) fn validate_request(self, request: CaptureRequestShape) -> Result<(), CaptureError> {
        let last = request
            .max_predictions
            .checked_sub(1)
            .ok_or_else(|| CaptureError::Invalid("empty capture prediction range".into()))?;
        self.invocation_shape(request, CapturePhase::Decode, last)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;

//! Private arithmetic binding between absolute observations and one new run.

use eredu_core::InferenceGeometry;
use eredu_runtime::working_memory::WorkingMemoryError;

/// Immutable coordinate metadata only. This cannot create a request, step,
/// receipt or storage permission. Ordinary native quotes use origin zero;
/// a future closed resume constructor must supply its validated saved origin.
#[derive(Debug)]
pub(super) struct PredictionCoordinates {
    origin: u64,
    cached_positions: u64,
    input_positions: u64,
    max_predictions: u64,
}

impl PredictionCoordinates {
    /// The caller separately validates the exact inference geometry. Check all
    /// coordinate endpoints before admission so a later query cannot wrap.
    pub(super) fn new(
        origin: u64,
        geometry: InferenceGeometry,
    ) -> Result<Self, WorkingMemoryError> {
        origin
            .checked_add(geometry.max_output_tokens)
            .ok_or(WorkingMemoryError::Overflow)?;
        geometry
            .cached_positions
            .checked_add(geometry.input_positions)
            .and_then(|position| position.checked_add(geometry.max_output_tokens))
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self {
            origin,
            cached_positions: geometry.cached_positions,
            input_positions: geometry.input_positions,
            max_predictions: geometry.max_output_tokens,
        })
    }

    /// Completed local predictions, including the final exhausted boundary for
    /// immutable capture. A step still needs its own one-use core permission.
    pub(super) fn local(&self, absolute: u64) -> Result<u64, WorkingMemoryError> {
        let local = absolute
            .checked_sub(self.origin)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if local > self.max_predictions {
            return Err(WorkingMemoryError::TextOutputAllowanceExceeded {
                issued: local,
                limit: self.max_predictions,
            });
        }
        Ok(local)
    }

    /// No cache array is consulted: stateless branches retain the same checked
    /// absolute frontier as stateful branches under the admitted geometry.
    pub(super) fn frontier(&self, absolute: u64) -> Result<u64, WorkingMemoryError> {
        let local = self.local(absolute)?;
        if local == 0 {
            return Ok(self.cached_positions);
        }
        self.cached_positions
            .checked_add(self.input_positions)
            .and_then(|position| position.checked_add(local - 1))
            .ok_or(WorkingMemoryError::Overflow)
    }
}

#[cfg(test)]
mod tests;

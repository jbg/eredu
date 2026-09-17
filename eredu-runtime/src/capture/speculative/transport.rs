//! Optional owned transport through the existing dense capture signal worker.
use super::{CaptureExecutionError, SpeculativeControlError};

/// Converts one dense capture failure into an execution signal and a separately
/// retained control error. Neither this transport nor its outputs grant memory
/// authority or prove native completion.
///
/// Existing borrowed mapper closures retain their ordinary ordering and error
/// representation through the blanket implementation below.
pub trait SpeculativeCaptureErrorTransport<E, X>
where
    E: std::error::Error + Send + Sync + 'static,
{
    /// Optionally splits an already-retained source into two owning aliases.
    /// Success must preserve its existing kind, operation, source identity and
    /// custody; the execution signal must retain its independent owner. No new
    /// producer, source allocation, classification or guard may be inferred.
    /// Refusal returns the identical, unmodified error for ordinary conversion.
    fn take_retained(
        &self,
        error: CaptureExecutionError<E>,
    ) -> Result<(X, SpeculativeControlError), CaptureExecutionError<E>> {
        Err(error)
    }

    /// Maps the ordinary borrowed error before the shared failure cell is lent.
    /// The worker calls this exactly once after retained transfer is refused.
    fn ordinary(&self, error: &CaptureExecutionError<E>) -> X;
}

impl<E, X, F> SpeculativeCaptureErrorTransport<E, X> for F
where
    E: std::error::Error + Send + Sync + 'static,
    F: Fn(&CaptureExecutionError<E>) -> X,
{
    fn ordinary(&self, error: &CaptureExecutionError<E>) -> X {
        self(error)
    }
}

#[cfg(test)]
mod tests;

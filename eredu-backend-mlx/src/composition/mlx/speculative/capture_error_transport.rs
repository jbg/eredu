//! Original-only error ownership component; current Exception observers retain
//! their ordinary mapper. A typed original observer/producers need separate fit.
use crate::backend::error::Error;
use eredu_core::speculative::SpeculativeControlError;
use eredu_runtime::capture::{CaptureExecutionError, SpeculativeCaptureErrorTransport};

/// The explicit ordinary fallback keeps native or admission variants outside
/// RetainedOriginal on the existing caller-selected path.
pub(in crate::composition::mlx) struct RetainedCaptureTransport<F>(F);
impl<F> RetainedCaptureTransport<F> {
    pub(in crate::composition::mlx) fn new(ordinary: F) -> Self {
        Self(ordinary)
    }
}
impl<F> SpeculativeCaptureErrorTransport<Error, Error> for RetainedCaptureTransport<F>
where
    F: Fn(&CaptureExecutionError<Error>) -> Error,
{
    fn take_retained(
        &self,
        error: CaptureExecutionError<Error>,
    ) -> Result<(Error, SpeculativeControlError), CaptureExecutionError<Error>> {
        match error {
            CaptureExecutionError::Backend(error) => error
                .split_retained_original()
                .map(|(signal, control)| (signal, SpeculativeControlError::Backend(control)))
                .map_err(CaptureExecutionError::Backend),
            error => Err(error),
        }
    }

    fn ordinary(&self, error: &CaptureExecutionError<Error>) -> Error {
        (self.0)(error)
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
pub(in crate::composition::mlx) use tests::fixture_pair;

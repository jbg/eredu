//! One complete error per actual original operation or consumed installation.
use super::*;
use eredu_runtime::working_memory::OriginalTextControlGuard;
use std::mem::size_of;

/// Constructed only after successful native claim or original installation take.
/// No Clone, guard extraction, refill, or arbitrary count/guard constructor.
pub(super) struct OriginalErrorAllowance {
    _controls: OriginalTextControlGuard,
}
#[derive(Debug)]
struct OriginalFailureSource {
    cause: Error,
    // Last: the selected nested error wrappers retire before this guard.
    // Escaped NN clones/dynamic native owners retain separate obligations.
    _allowance: OriginalErrorAllowance,
}
impl std::fmt::Debug for OriginalErrorAllowance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginalErrorAllowance")
            .finish_non_exhaustive()
    }
}
impl std::fmt::Display for OriginalFailureSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for OriginalFailureSource {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
impl OriginalErrorAllowance {
    pub(super) fn claimed(quote: &text_quote::TextExecutionQuote) -> Option<Self> {
        quote.original_controls().map(|controls| Self {
            _controls: controls,
        })
    }
    pub(super) fn installation_payload(capture: &text_capture::InstallationPayload) -> Self {
        Self {
            _controls: capture.original_controls(),
        }
    }
    #[cfg(test)]
    pub(super) fn for_rows_fixture(
        accepted: &eredu_runtime::working_memory::OwnedTextSpanWorkspace,
        rows: &crate::composition::mlx::replicated_text::NativeOpeningRows,
        source: &eredu_core::capture::SharedCapturePlan,
    ) -> Result<Self, Error> {
        rows.claim_fixture_error_controls(accepted, source)
            .map(|controls| Self {
                _controls: controls,
            })
    }
    fn retain(self, cause: Error) -> BackendFailure {
        BackendFailure::new(
            BackendFailureKind::Other,
            OriginalFailureSource {
                cause,
                _allowance: self,
            },
        )
    }
    pub(super) fn finish<T>(self, result: Result<T, Error>) -> Result<T, Error> {
        match result {
            Ok(value) => Ok(value),
            Err(cause) => {
                let preserved = cause.model_state_preserved();
                Err(Error::with_original_control_source(
                    self.retain(cause),
                    preserved,
                ))
            }
        }
    }
    pub(super) fn finish_backend<T>(self, result: Result<T, Error>) -> Result<T, BackendFailure> {
        result.map_err(|cause| self.retain(cause))
    }
}
pub(super) fn finish<T>(
    allowance: Option<OriginalErrorAllowance>,
    result: Result<T, Error>,
) -> Result<T, Error> {
    match allowance {
        Some(allowance) => allowance.finish(result),
        None => result,
    }
}

/// Introduced fixed owner, native/core envelope and named operation controls.
/// Nested diagnostic payloads have separate facts/limits; this is not a complete
/// error-memory certificate or permission to activate native capture.
pub(super) fn control_peak_bytes() -> Option<u64> {
    let parts = [
        size_of::<OriginalErrorAllowance>(),
        size_of::<Option<OriginalErrorAllowance>>(),
        size_of::<OriginalFailureSource>(),
        size_of::<crate::backend::error::OriginalControlFailure>(),
        size_of::<text_step::TextOperation<'static>>(),
        size_of::<Option<text_step::TextOperation<'static>>>(),
        size_of::<Result<(), Error>>(),
        size_of::<text_capture::InstalledCapture>(),
        size_of::<Result<text_capture::InstalledCapture, Error>>(),
        size_of::<(text_capture::InstalledCapture, OriginalErrorAllowance)>(),
        size_of::<Result<(text_capture::InstalledCapture, OriginalErrorAllowance), Error>>(),
        size_of::<Result<(), BackendFailure>>(),
        BackendFailure::source_retention_peak_bytes::<OriginalFailureSource>()?,
        // Ordinary core conversion may receive the final native error by value.
        BackendFailure::source_retention_peak_bytes::<Error>()?,
    ];
    type RuntimeError = eredu_runtime::ReplicatedTextSessionError<eredu_nn::Error, Error, Error>;
    // One runtime error allocation and, on agreed pre-state rejection, its one
    // BeforeStateMutation Box. Both use final Error layout. No recursive factor.
    let runtime = [
        size_of::<RuntimeError>(),
        size_of::<Box<RuntimeError>>(),
        size_of::<RuntimeError>(),
        size_of::<Box<RuntimeError>>(),
        size_of::<Result<(), RuntimeError>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)?;
    let leaf = crate::composition::mlx::replicated_text::NativeOpeningRows::error_control_bytes()?;
    let callback = super::text_funding::observer_error_control_bytes()?;
    let neural = crate::composition::retained_observation_error_control_bytes()?;
    // Mechanism errors carry the native leaf directly. Observer errors additionally
    // carry the funded capture Box and retained NN Arc. This checked maximum names
    // the two real paths; runtime and final owner costs occur only once.
    let nested = leaf.max(leaf.checked_add(callback)?.checked_add(neural)?);
    type NativeSubmission = Submission<MlxTextToken, MlxTextCompletion>;
    let result = size_of::<Result<Option<NativeSubmission>, Error>>()
        .max(size_of::<Result<NativeSubmission, Error>>());
    // Execution result, caller result after receipt attachment, free finish
    // argument, consuming method argument and outward return. These are named
    // fixed move controls, not source allocations or an error-depth multiplier.
    let operation_returns = result
        .checked_add(result)?
        .checked_add(result)?
        .checked_add(result)?
        .checked_add(result)?;
    let bytes = parts
        .into_iter()
        .try_fold(0usize, usize::checked_add)?
        .checked_add(runtime)?
        .checked_add(nested)?
        .checked_add(operation_returns)?;
    u64::try_from(bytes).ok()
}

use super::Error;
use eredu_core::{BackendFailure, BackendFailureKind, SharedBackendFailure};
use std::fmt;
#[cfg(test)]
use std::sync::Arc;

/// The exact first output-observation failure, shared by later observations.
///
/// There is no public constructor, owning extraction, Clone or Weak access.
/// This retains the original error and its existing custody; it grants neither
/// completion nor an additional original error allowance.
pub struct OutputObservationFailure {
    source: Option<SharedBackendFailure>,
    state_preserved: bool,
}
impl OutputObservationFailure {
    pub(crate) fn control_bytes() -> Option<u64> {
        let bytes = SharedBackendFailure::control_bytes::<Error>()?
            .checked_add(std::mem::size_of::<Self>())?
            .checked_add(std::mem::size_of::<Option<SharedBackendFailure>>())?;
        u64::try_from(bytes).ok()
    }
    pub(crate) fn into_backend_failure(mut self) -> BackendFailure {
        self.source
            .take()
            .expect("live observation error")
            .into_failure()
    }
    pub(crate) fn new(error: Error) -> Self {
        if let Error::OutputObservation(error) = error {
            return error;
        }
        let state_preserved = error.model_state_preserved();
        Self {
            source: Some(SharedBackendFailure::new(BackendFailureKind::Other, error)),
            state_preserved,
        }
    }
    pub(crate) fn retained(&self) -> Self {
        Self {
            source: Some(
                self.source
                    .as_ref()
                    .expect("live observation error")
                    .retained(),
            ),
            state_preserved: self.state_preserved,
        }
    }
    pub(super) const fn state_preserved(&self) -> bool {
        self.state_preserved
    }
    fn original(&self) -> &Error {
        self.source
            .as_ref()
            .expect("live observation error")
            .source_error()
            .downcast_ref::<Error>()
            .expect("closed concrete Error")
    }
}
impl fmt::Debug for OutputObservationFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("OutputObservationFailure")
            .field(self.original())
            .finish()
    }
}
impl fmt::Display for OutputObservationFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.original(), f)
    }
}
impl std::error::Error for OutputObservationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.original())
    }
}
impl Drop for OutputObservationFailure {
    fn drop(&mut self) {
        if let Some(source) = self.source.take() {
            // Core's closed shared owner performs concrete Arc::into_inner:
            // its allocation disappears before Error and its custody fields.
            drop(source);
        }
    }
}

#[cfg(test)]
mod tests;

//! Ordinary error ownership is separate from frame and original allowances.
use super::Error;
use eredu_core::{BackendFailure, HostPreparationAuthority};
use std::fmt;
/// Complete native diagnostic under its existing ordinary host exclusion.
#[derive(Debug)]
pub struct OrdinaryCaptureFailure {
    source: BackendFailure,
    state_preserved: bool,
    // Core unboxes the complete source before this independent host token drops.
    _host: HostPreparationAuthority,
}
impl OrdinaryCaptureFailure {
    pub(super) const fn state_preserved(&self) -> bool {
        self.state_preserved
    }
}
impl fmt::Display for OrdinaryCaptureFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(
            std::error::Error::source(&self.source).expect("capture source"),
            f,
        )
    }
}
impl std::error::Error for OrdinaryCaptureFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        std::error::Error::source(&self.source)
    }
}
impl Error {
    pub(crate) fn retain_ordinary_capture(self, host: Option<HostPreparationAuthority>) -> Self {
        match (self, host) {
            (error @ Self::OrdinaryCapture(_), _) | (error, None) => error,
            (cause, Some(host)) => {
                let state_preserved = cause.model_state_preserved();
                Self::OrdinaryCapture(OrdinaryCaptureFailure {
                    source: BackendFailure::from_error(cause),
                    state_preserved,
                    _host: host,
                })
            }
        }
    }
    pub(crate) fn is_ordinary_capture_failure(&self) -> bool {
        matches!(self, Self::OrdinaryCapture(_))
    }
}
/// Fixed core source/control representation; variable diagnostics remain ordinary.
#[cfg(test)]
pub(crate) fn control_peak_bytes() -> Option<usize> {
    BackendFailure::source_retention_peak_bytes::<Error>()?
        .checked_add(std::mem::size_of::<OrdinaryCaptureFailure>())?
        .checked_add(std::mem::size_of::<Result<(), Error>>())
}
#[cfg(test)]
mod tests;

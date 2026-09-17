//! Existing ordinary host custody for diagnostics returned outside a frame.
use super::{CaptureError, CaptureExecutionError, CaptureSession};
use eredu_core::HostPreparationAuthority;
pub(super) fn retain_result<T>(
    host: Option<HostPreparationAuthority>,
    result: Result<T, CaptureError>,
) -> Result<T, CaptureError> {
    result.map_err(|error| match host {
        Some(host) => error.retain_ordinary(host),
        None => error,
    })
}
pub(super) fn retain_admission<T, E: std::error::Error + 'static>(
    host: Option<HostPreparationAuthority>,
    result: Result<T, CaptureExecutionError<E>>,
) -> Result<T, CaptureExecutionError<E>> {
    result.map_err(|error| match (host, error) {
        (Some(host), CaptureExecutionError::Admission(error)) => {
            CaptureExecutionError::Admission(error.retain_ordinary(host))
        }
        (_, error) => error,
    })
}
impl CaptureSession {
    /// Borrows the already installed ordinary host token for native error delivery.
    /// This is not an allocation grant, finite quote, source or completion witness.
    /// It persists after p0 and cannot depend on a poisoned mutable custody lock.
    #[doc(hidden)]
    pub fn ordinary_error_custody(&self) -> Option<&HostPreparationAuthority> {
        self.ordinary_error_custody.as_ref()
    }
}
#[cfg(test)]
impl CaptureSession {
    pub(crate) fn poison_ordinary_host_for_test(&self) {
        self.owner.poison_for_test();
    }
}

//! Preserve concrete native causes without narrowing the portable policy contract.
use super::{moshi, Error};
use eredu_runtime::LayerwiseRuntimeError;

/// Owns the unchanged execution error under the actual frame source funding.
/// The portable policy error only requires Display; this concrete adapter knows
/// that its native policy errors also carry typed causes.
#[derive(Debug)]
pub(super) struct Source(pub(super) moshi::MoshiRealtimeExecutionError<Error>);

impl std::fmt::Display for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}
impl std::error::Error for Source {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.0 {
            moshi::MoshiRealtimeExecutionError::Execution(LayerwiseRuntimeError::Policy(cause)) => Some(cause),
            moshi::MoshiRealtimeExecutionError::DecisionBoundary(cause) => Some(cause),
            cause => Some(cause),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn realtime_policy_error_exposes_original_native_cause() {
        let failure = Source(moshi::MoshiRealtimeExecutionError::Execution(
            LayerwiseRuntimeError::Policy(Error::PrefillScopeUnavailable),
        ));
        assert!(matches!(failure.source().unwrap().downcast_ref::<Error>(),
            Some(Error::PrefillScopeUnavailable)));
        assert_eq!(failure.to_string(),
            "layerwise execution policy failed: original prefill scope role is unavailable");
    }
}

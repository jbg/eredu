//! Exact host metadata refusal, carried inline even when its account is exhausted.
use super::{BackendFailure, BackendFailureKind, SourceOwner};

/// Fixed refusal from the actual host-metadata account. Returning this value
/// never allocates, formats an error source, or grants native authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum HostMetadataFundingError {
    /// The requested constructor does not fit the account's remaining capacity.
    #[error("workspace metadata funding requires {required} bytes, with {available} available")]
    Capacity {
        /// Bytes requested by the next concrete producer.
        required: u64,
        /// Bytes available to that producer.
        available: u64,
    },
    /// A constructor layout or accounting total cannot be represented.
    #[error("workspace metadata funding layout overflow")]
    Overflow,
    /// The actual account cannot currently accept a metadata reservation.
    #[error("workspace metadata funding is unavailable")]
    Unavailable,
}

impl HostMetadataFundingError {
    /// Moves this fixed refusal into the public source chain without allocation.
    /// Capacity values and the concrete source type are preserved. This neither
    /// settles native work nor decides whether a caller may retry a snapshot.
    pub fn into_backend_failure(self) -> BackendFailure {
        BackendFailure {
            kind: match self {
                Self::Capacity { .. } | Self::Overflow => BackendFailureKind::ResourceExhausted,
                Self::Unavailable => BackendFailureKind::Other,
            },
            operation: "host metadata preparation",
            source: SourceOwner::metadata_funding(self),
        }
    }
}
impl From<HostMetadataFundingError> for BackendFailure {
    fn from(cause: HostMetadataFundingError) -> Self {
        cause.into_backend_failure()
    }
}

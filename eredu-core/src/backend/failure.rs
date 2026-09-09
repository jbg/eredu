use std::error::Error;

/// Backend-independent reason a model operation failed.
///
/// No error kind establishes successful settlement or safe session reuse.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum BackendFailureKind {
    /// Another submission still owns the session; settle it before retrying.
    Busy,
    /// The session is poisoned, mismatched, or otherwise invalid for this operation.
    InvalidSession,
    /// The backend identified insufficient memory or another exhausted resource.
    ResourceExhausted,
    /// The request requires an unavailable backend capability.
    Unsupported,
    /// The backend rejected invalid input or configuration.
    InvalidInput,
    /// A filesystem or host I/O operation failed.
    Io,
    /// The backend failed without a more precise portable classification.
    Other,
}

/// Common backend failure used by the application API on every backend.
///
/// Applications handle [`Self::kind`] without naming the backend's error type.
/// The original typed error and its cause chain remain available through
/// [`Error::source`], including optional downcasting for backend-specific diagnostics.
#[derive(Debug, thiserror::Error)]
#[error("{operation} failed ({kind:?}): {source}")]
pub struct BackendFailure {
    kind: BackendFailureKind,
    operation: &'static str,
    #[source]
    source: Box<dyn Error + Send + Sync + 'static>,
}

impl BackendFailure {
    /// Retains the original failure with a classification established by its owner.
    /// Use `Other` when only an unstructured diagnostic is available.
    pub fn new(kind: BackendFailureKind, source: impl Error + Send + Sync + 'static) -> Self {
        Self {
            kind,
            operation: "backend operation",
            source: Box::new(source),
        }
    }

    /// Returns the portable reason for failure.
    pub const fn kind(&self) -> BackendFailureKind {
        self.kind
    }

    /// Adds application operation context without replacing the original source.
    pub fn with_operation(mut self, operation: &'static str) -> Self {
        self.operation = operation;
        self
    }

    /// Returns the operation reported by the adapter or facade.
    pub const fn operation(&self) -> &'static str {
        self.operation
    }

    /// Preserves an error whose portable classification is not known.
    pub fn from_error(source: impl Error + Send + Sync + 'static) -> Self {
        // Preserve a classification already supplied by a lower adapter.
        let source: Box<dyn Error + Send + Sync> = Box::new(source);
        match source.downcast::<Self>() {
            Ok(error) => *error,
            Err(source) => {
                let kind = if let Some(error) =
                    source.downcast_ref::<crate::SessionAuthorityError>()
                {
                    match error {
                        crate::SessionAuthorityError::Busy => BackendFailureKind::Busy,
                        crate::SessionAuthorityError::TicketExhausted => {
                            BackendFailureKind::ResourceExhausted
                        }
                    }
                } else if source.is::<crate::SessionAdmissionError>() {
                    BackendFailureKind::InvalidSession
                } else if let Some(error) = source.downcast_ref::<std::io::Error>() {
                    match error.kind() {
                        std::io::ErrorKind::OutOfMemory => BackendFailureKind::ResourceExhausted,
                        std::io::ErrorKind::InvalidInput => BackendFailureKind::InvalidInput,
                        std::io::ErrorKind::Unsupported => BackendFailureKind::Unsupported,
                        _ => BackendFailureKind::Io,
                    }
                } else {
                    BackendFailureKind::Other
                };
                Self {
                    kind,
                    operation: "backend operation",
                    source,
                }
            }
        }
    }
}

impl From<crate::SessionAuthorityError> for BackendFailure {
    fn from(source: crate::SessionAuthorityError) -> Self {
        let kind = match source {
            crate::SessionAuthorityError::Busy => BackendFailureKind::Busy,
            crate::SessionAuthorityError::TicketExhausted => BackendFailureKind::ResourceExhausted,
        };
        Self::new(kind, source)
    }
}

impl From<crate::SessionAdmissionError> for BackendFailure {
    fn from(source: crate::SessionAdmissionError) -> Self {
        Self::new(BackendFailureKind::InvalidSession, source)
    }
}

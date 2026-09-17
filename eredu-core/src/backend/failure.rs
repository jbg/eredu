use std::{error::Error, fmt};

mod metadata_funding;
pub use metadata_funding::HostMetadataFundingError;
mod shared;
mod source;
pub use shared::SharedBackendFailure;
use source::SourceOwner;

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

/// Fixed rejection before an original generation-sequence bank is consumed.
///
/// These failures retain neither a bank nor funding custody. Their conversion
/// uses one of three static typed sources, so repeated rejected calls create no
/// error-source allocation. Caller-owned containers and arbitrary wrapping are
/// separate from this closed conversion.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
pub enum GenerationSequenceBankRejection {
    /// No sequence bank was admitted, or its one extraction already occurred.
    #[error("original generation sequence bank is unavailable or already taken")]
    Unavailable,
    /// The actual shared bank is currently borrowed by another operation.
    #[error("original generation sequence bank is busy")]
    Busy,
    /// The supplied preparation does not identify this original sequence bank.
    #[error("original generation sequence bank identity does not match")]
    IdentityMismatch,
}
impl GenerationSequenceBankRejection {
    /// Returns a neutral failure with a fixed static source and no allocation.
    ///
    /// The original typed rejection is available through `Error::source`.
    /// This neither spends nor clones a bank, and supplies no memory authority.
    pub fn into_backend_failure(self) -> BackendFailure {
        BackendFailure {
            kind: match self {
                Self::Busy => BackendFailureKind::Busy,
                Self::Unavailable | Self::IdentityMismatch => BackendFailureKind::InvalidSession,
            },
            operation: "generation sequence bank",
            source: SourceOwner::sequence_bank_rejection(self),
        }
    }
}

/// Fixed rejection outside a consumed original TokenIds input owner.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
pub enum TokenInputRejection {
    /// No originally admitted input remains.
    #[error("original token input is unavailable or spent")]
    Unavailable,
    /// The exact original input slot is borrowed or in flight.
    #[error("original token input is busy")]
    Busy,
    /// The input does not identify this original request.
    #[error("original token input identity mismatch")]
    IdentityMismatch,
    /// Backend has no originally constructed input contract.
    #[error("original token input is unsupported")]
    Unsupported,
    /// Text prefill requires a positive input extent.
    #[error("token input must not be empty")]
    Empty,
    /// Destination geometry exceeds its representation.
    #[error("token input extent overflow")]
    Overflow,
    /// Facade policy cannot resolve a supplied ID in its actual vocabulary.
    #[error("token input contains an unknown vocabulary ID")]
    InvalidToken,
}
impl TokenInputRejection {
    /// Fixed typed source without allocation or custody acquisition.
    pub fn into_backend_failure(self) -> BackendFailure {
        BackendFailure {
            kind: match self {
                Self::Busy => BackendFailureKind::Busy,
                Self::Unsupported => BackendFailureKind::Unsupported,
                Self::Empty | Self::Overflow | Self::InvalidToken => {
                    BackendFailureKind::InvalidInput
                }
                _ => BackendFailureKind::InvalidSession,
            },
            operation: "original token input",
            source: SourceOwner::token_input_rejection(self),
        }
    }
}

/// Fixed rejection before a complete original prepared request is accepted.
/// Missing components describe unfinished producer coverage, not architecture
/// inapplicability. Conversion retains a static typed source without allocation.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
pub enum PreparedRequestRejection {
    /// The backend has not opted into original prepared requests.
    #[error("original prepared request admission is unavailable")]
    Unsupported,
    /// Core has no genuine sequence request for this original preparation.
    #[error("original prepared input requires a generation sequence request")]
    MissingSequence,
    /// The prompt lacks its closed original source owner.
    #[error("original prepared source is unavailable")]
    SourceUnavailable,
    /// Actual source, selection, pool, domain or revision differs.
    #[error("original prepared source identity mismatch")]
    IdentityMismatch,
    /// Configuration, source mode or sequence descriptors differ.
    #[error("original prepared request mismatch")]
    RequestMismatch,
    /// Existing work prevents this admission attempt.
    #[error("original prepared request is busy")]
    Busy,
    /// Originally bounded inspection and diagnostic storage is not implemented.
    #[error("original prepared inspection storage is not yet bounded")]
    MissingInspectionStorage,
    /// Numerical or private workspace enforcement is not implemented.
    #[error("original prepared numerical workspace is not yet bounded")]
    MissingNumericalWorkspace,
    /// Reachable driver, readiness or completion controls are not complete.
    #[error("original prepared execution controls are not yet bounded")]
    MissingExecutionControls,
    /// The actual installed capture source has no complete original producer.
    #[error("original prepared capture storage is not yet bounded")]
    MissingCapture,
    /// The selected mechanism cannot close this controller declaration.
    #[error("original prepared controller storage is not yet bounded")]
    MissingController,
    /// Checked request geometry or a concrete layout overflows.
    #[error("original prepared request arithmetic overflow")]
    Overflow,
    /// The single original comparison rejects the complete request capacity.
    #[error("original prepared request exceeds available capacity")]
    CapacityExceeded,
}
impl PreparedRequestRejection {
    /// Returns a fixed typed source; this allocates and grants nothing.
    pub fn into_backend_failure(self) -> BackendFailure {
        BackendFailure {
            kind: match self {
                Self::Busy => BackendFailureKind::Busy,
                Self::Unsupported
                | Self::MissingInspectionStorage
                | Self::MissingNumericalWorkspace
                | Self::MissingExecutionControls
                | Self::MissingCapture
                | Self::MissingController => BackendFailureKind::Unsupported,
                Self::CapacityExceeded => BackendFailureKind::ResourceExhausted,
                Self::MissingSequence | Self::RequestMismatch | Self::Overflow => {
                    BackendFailureKind::InvalidInput
                }
                Self::SourceUnavailable | Self::IdentityMismatch => {
                    BackendFailureKind::InvalidSession
                }
            },
            operation: "original prepared request",
            source: SourceOwner::prepared_request_rejection(self),
        }
    }
}

/// Common backend failure used by the application API on every backend.
///
/// Applications handle [`Self::kind`] without naming the backend's error type.
/// The original typed error and its cause chain remain available through
/// [`Error::source`], including optional downcasting for backend-specific diagnostics.
#[derive(Debug)]
pub struct BackendFailure {
    kind: BackendFailureKind,
    operation: &'static str,
    source: SourceOwner,
}

impl BackendFailure {
    pub(super) fn text_context(cause: super::TextContextError) -> Self {
        Self {
            kind: BackendFailureKind::ResourceExhausted,
            operation: "text context identity",
            source: SourceOwner::text_context(cause),
        }
    }

    /// Retains the original failure with a classification established by its owner.
    /// Use `Other` when only an unstructured diagnostic is available.
    pub fn new(kind: BackendFailureKind, source: impl Error + Send + Sync + 'static) -> Self {
        Self {
            kind,
            operation: "backend operation",
            source: SourceOwner::new(source),
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
        from_concrete_error(source)
    }

    /// Requested storage and named constructor/retirement overlap for one
    /// concrete source of this error type. No source or allocation is created.
    ///
    /// This includes the actual source Box payload and core's erased owner,
    /// return, flattening and disposal controls. It excludes allocations nested
    /// inside `E`, caller-owned containers, allocator overhead and formatting.
    /// `None` reports checked size overflow. This fact grants no memory or
    /// completion authority; original admission must compose it before use.
    pub fn source_retention_peak_bytes<E: Error + Send + Sync + 'static>() -> Option<usize> {
        source::retention_peak_bytes::<E>()
    }
}

impl fmt::Display for BackendFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} failed ({:?}): {}",
            self.operation, self.kind, self.source
        )
    }
}
impl Error for BackendFailure {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        // Preserve the original leaf's type and its cause chain. The private
        // retirement wrapper is never inserted as an observable error source.
        Some(self.source.error())
    }
}

fn from_concrete_error<E: Error + Send + Sync + 'static>(source: E) -> BackendFailure {
    // A known neutral value already owns its complete source. Move it directly;
    // this Any check neither allocates nor invokes Error::source or callbacks.
    let mut source = Some(source);
    if let Some(neutral) =
        (&mut source as &mut dyn std::any::Any).downcast_mut::<Option<BackendFailure>>()
    {
        return neutral.take().expect("owned neutral failure");
    }
    if let Some(funding) =
        (&mut source as &mut dyn std::any::Any).downcast_mut::<Option<HostMetadataFundingError>>()
    {
        return funding
            .take()
            .expect("owned fixed funding refusal")
            .into_backend_failure();
    }
    let source = source.expect("unchanged concrete source");

    // Preserve the existing classification and flattening using one allocation.
    // A failed downcast is still exactly E; recover that same Box before closed
    // erasure. No user callback or raw source escape occurs during conversion.
    let source: Box<dyn Error + Send + Sync> = Box::new(source);
    match source.downcast::<BackendFailure>() {
        Ok(error) => source::unbox(error),
        Err(source) => {
            let kind = if let Some(error) = source.downcast_ref::<crate::SessionAuthorityError>() {
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
            let source = source.downcast::<E>().expect("original backend error type");
            BackendFailure {
                kind,
                operation: "backend operation",
                source: SourceOwner::from_box(source),
            }
        }
    }
}

#[cfg(test)]
pub(crate) fn source_retirement_count_for_test() -> usize {
    source::retirement_count()
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

pub(super) fn prepared_control_rejection(
    cause: super::PreparedControlInputError,
) -> BackendFailure {
    BackendFailure {
        kind: BackendFailureKind::Unsupported,
        operation: "prepared controlled input",
        source: SourceOwner::prepared_control_rejection(cause),
    }
}

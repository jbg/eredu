//! Preserve already constructed neural failures across the closed runtime wrappers.
use eredu_nn::{Error, workspace::{WorkspaceContext, WorkspaceMetadataError, HostMetadataFundingError}};
use eredu_runtime::{LayerwiseRuntimeError, PreparedLayeredObservationError};

pub(in crate::prepared_execution::workspace) trait QuoteError {
    fn into_quote_error(self, context: &WorkspaceContext) -> Error;
}
impl QuoteError for Error {
    fn into_quote_error(self, _: &WorkspaceContext) -> Error { self }
}
impl QuoteError for LayerwiseRuntimeError<Error, Error> {
    fn into_quote_error(self, context: &WorkspaceContext) -> Error {
        match self {
            Self::Architecture(cause) | Self::Policy(cause) => cause,
            Self::Submission(cause) => {
                if let Some(refusal) = std::error::Error::source(&cause)
                    .and_then(|cause| cause.downcast_ref::<HostMetadataFundingError>()) {
                    return WorkspaceMetadataError::Funding(*refusal).into();
                }
                context.metadata_source(Self::Submission(cause))
            }
            cause => context.metadata_source(cause),
        }
    }
}
impl<E> QuoteError for PreparedLayeredObservationError<E>
where E: QuoteError + std::error::Error + Send + Sync + 'static {
    fn into_quote_error(self, context: &WorkspaceContext) -> Error {
        match self {
            Self::Execution(cause) => cause.into_quote_error(context),
            cause => context.metadata_source(cause),
        }
    }
}

impl<P: std::error::Error + Send + Sync + 'static> QuoteError
    for eredu_runtime::ReplicatedTextSessionError<Error, P, std::convert::Infallible> {
    fn into_quote_error(self, context: &WorkspaceContext) -> Error {
        match self {
            Self::Architecture(error) => error,
            Self::Mechanism(never) => match never {},
            Self::BeforeStateMutation(error) => error.into_quote_error(context),
            cause => context.metadata_source(cause),
        }
    }
}

#[cfg(test)]
mod tests;

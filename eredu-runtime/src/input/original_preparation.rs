//! Original source construction under an explicit selected-session metadata owner.
use eredu_core::{BackendFailure, ModelRuntime, TextGenerationBackend, TokenInputRejection};
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use super::host::PreparedHostInputPlan;

/// Completed original input and the source-preparation account that produced it.
/// The input retires before its payer. This conveys no request/native permission.
#[derive(Debug)]
pub struct OriginalModelInput<P> {
    input: P,
    funding: WorkspaceMetadataFunding,
}
impl<P> OriginalModelInput<P> {
    /// Pays the exact publication/result shells before creating the returned owner.
    pub fn try_new(input: P, funding: WorkspaceMetadataFunding)
    -> Result<Self, OriginalModelInputPublicationError<P>> {
        let bytes = std::mem::size_of::<(Self, Result<Self, OriginalModelInputPublicationError<P>>)>();
        if let Err(cause) = funding.reserve_metadata(bytes) {
            return Err(OriginalModelInputPublicationError { input, cause, funding });
        }
        Ok(Self { input, funding })
    }
    /// Moves the same input and its account into an enclosing admitted driver.
    /// The account must remain live through the input's handoff or rejection.
    pub fn into_parts(self) -> (P, WorkspaceMetadataFunding) {
        (self.input, self.funding)
    }
}
/// A refused publication keeps the completed input and its actual payer.
#[derive(Debug)]
pub struct OriginalModelInputPublicationError<P> {
    input: P,
    cause: WorkspaceMetadataFundingError,
    funding: WorkspaceMetadataFunding,
}
impl<P> OriginalModelInputPublicationError<P> {
    /// Retire the input through its existing native owner, keeping funding until
    /// every synchronous destructor has run. Pending native aliases keep their
    /// own source custody; this operation establishes no terminal status.
    pub fn retire(self) -> (WorkspaceMetadataFundingError, WorkspaceMetadataFunding) {
        let Self { input, cause, funding } = self;
        drop(input);
        (cause, funding)
    }
}
/// Backend realization of the neutral original host-input recipe.
pub trait OriginalModelInputBackend: TextGenerationBackend {
    /// The native implementation uses the loaded execution's retained semantic
    /// source and exact input allocator. Caller policy supplies one total ceiling.
    /// No ordinary source may be promoted; default invokes no constructor.
    fn prepare_original_model_input(
        _runtime: &ModelRuntime<Self>,
        _plan: PreparedHostInputPlan<'_>,
        _capacity: u64,
    ) -> Result<OriginalModelInput<Self::Prompt>, BackendFailure> {
        Err(TokenInputRejection::Unsupported.into_backend_failure())
    }
}

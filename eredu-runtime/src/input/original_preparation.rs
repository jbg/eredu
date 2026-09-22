//! Original source construction under an explicit selected-session metadata owner.
use super::host::PreparedHostInputPlan;
use eredu_core::{BackendFailure, ModelRuntime, TextGenerationBackend, TokenInputRejection};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};

/// Completed original input and the source-preparation account that produced it.
/// The input retires before its payer. This conveys no request/native permission.
#[derive(Debug)]
pub struct OriginalModelInput<P> {
    input: P,
    custody: OriginalModelInputCustody,
}
/// Retains input preparation and, when present, the exact chat/source association.
/// Keep this owner through prompt handoff, failure, and the consuming session.
#[derive(Debug, Clone)]
pub struct OriginalModelInputCustody {
    pub(super) chat:
        Option<eredu_core::SharedStorageOwner<super::chat_binding::PreparedChatInputBinding>>,
    funding: HostMetadataFunding,
}
impl<P> OriginalModelInput<P> {
    pub(super) fn prompt(&self) -> &P {
        &self.input
    }
    /// Pays the exact publication/result shells before creating the returned owner.
    pub fn try_new(
        input: P,
        funding: HostMetadataFunding,
    ) -> Result<Self, OriginalModelInputPublicationError<P>> {
        let bytes =
            std::mem::size_of::<(Self, Result<Self, OriginalModelInputPublicationError<P>>)>();
        if let Err(cause) = funding.reserve_metadata(bytes) {
            return Err(OriginalModelInputPublicationError {
                input,
                cause,
                funding,
            });
        }
        Ok(Self {
            input,
            custody: OriginalModelInputCustody {
                chat: None,
                funding,
            },
        })
    }
    /// Moves the same input and its account into an enclosing admitted driver.
    /// The account must remain live through the input's handoff or rejection.
    pub fn into_parts(self) -> (P, OriginalModelInputCustody) {
        (self.input, self.custody)
    }
    /// Exact chat association, if this input was compiled from an original render.
    pub fn chat_binding(&self) -> Option<&super::chat_binding::PreparedChatInputBinding> {
        self.custody.chat.as_deref()
    }
    pub(super) fn with_chat_binding(
        mut self,
        binding: eredu_core::SharedStorageOwner<super::chat_binding::PreparedChatInputBinding>,
    ) -> Self {
        self.custody.chat = Some(binding);
        self
    }
}
impl OriginalModelInputCustody {
    /// Borrows the retained exact chat association without exporting its owners.
    pub fn chat_binding(&self) -> Option<&super::chat_binding::PreparedChatInputBinding> {
        self.chat.as_deref()
    }
}
/// A refused publication keeps the completed input and its actual payer.
#[derive(Debug)]
pub struct OriginalModelInputPublicationError<P> {
    input: P,
    cause: HostMetadataFundingError,
    funding: HostMetadataFunding,
}
impl<P> OriginalModelInputPublicationError<P> {
    /// Retire the input through its existing native owner, keeping funding until
    /// every synchronous destructor has run. Pending native aliases keep their
    /// own source custody; this operation establishes no terminal status.
    pub fn retire(self) -> (HostMetadataFundingError, HostMetadataFunding) {
        let Self {
            input,
            cause,
            funding,
        } = self;
        drop(input);
        (cause, funding)
    }
}
/// Backend realization of the neutral original host-input recipe.
pub trait OriginalModelInputBackend: TextGenerationBackend {
    /// Borrows the actual completed architecture semantics retained by this
    /// input. Returning a copied or independently compiled source is invalid.
    fn original_model_input_semantics(
        _input: &Self::Prompt,
    ) -> Option<&crate::working_memory::BoundCompositeSemanticStorage> {
        None
    }
    /// The native implementation uses the loaded execution's retained semantic
    /// source and exact input allocator. Caller policy supplies domain limits.
    /// No ordinary source may be promoted; default invokes no constructor.
    fn prepare_original_model_input(
        _runtime: &ModelRuntime<Self>,
        _plan: PreparedHostInputPlan<'_>,
        _limits: &eredu_core::MemoryLimitDeclarations,
    ) -> Result<OriginalModelInput<Self::Prompt>, BackendFailure> {
        Err(TokenInputRejection::Unsupported.into_backend_failure())
    }
}

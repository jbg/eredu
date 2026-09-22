//! Closed by-value chat operation errors and neutral concrete backend entrypoints.
use super::{
    OriginalChatFileError, OriginalChatRenderError, OriginalChatTemplate,
    OriginalChatTemplateError, OriginalRenderedChat,
};
use crate::working_memory::{OriginalTokenizer, OriginalTokenizerBackend};
use eredu_core::{ModelRuntime, TokenInputRejection};
use eredu_text::chat_storage::{ChatRenderContext, ChatSourceError, ChatTemplatePlan};
use std::mem::size_of;

/// Actual source/file operation error with no pre-admission error erasure.
/// Every admitted variant contains the real closed partial owner by value.
#[derive(Debug, thiserror::Error)]
pub enum OriginalChatSourceError {
    /// Fixed unsupported/foreign-source rejection before any operation.
    #[error(transparent)]
    Domain(#[from] TokenInputRejection),
    /// Checked source/config planning failure before fresh J admission.
    #[error(transparent)]
    Plan(#[from] ChatSourceError),
    /// Actual J constructor/admission error and retained compiler prefix.
    #[error(transparent)]
    Compile(#[from] OriginalChatTemplateError),
    /// Consuming config-file I/read/fresh-J error with both original lifetimes.
    #[error(transparent)]
    File(#[from] OriginalChatFileError),
    /// File preparation precedes the original byte boundary. Its real OS error
    /// is preserved by value; opening/error internals are not newly bounded.
    #[error(transparent)]
    FilePreparation(#[from] eredu_checkpoint::artifact::ArtifactFileReadError),
}
impl OriginalChatSourceError {
    fn fixed_controls() -> Option<usize> {
        size_of::<Self>()
            .checked_add(size_of::<TokenInputRejection>())?
            .checked_add(size_of::<Result<(), Self>>())
    }
    pub(super) fn source_controls() -> Option<usize> {
        Self::fixed_controls()?
            .checked_add(size_of::<Result<OriginalChatTemplate, Self>>())?
            .checked_add(size_of::<Option<OriginalChatTemplate>>())?
            .checked_add(size_of::<Result<Option<OriginalChatTemplate>, Self>>())
    }
}

/// Actual render failure, independent of source compiler storage.
#[derive(Debug, thiserror::Error)]
pub enum OriginalChatRenderOperationError {
    /// Fixed unsupported or foreign-source refusal before rendering.
    #[error(transparent)]
    Domain(#[from] TokenInputRejection),
    /// Actual render/admission failure retaining all partial render storage.
    #[error(transparent)]
    Render(#[from] OriginalChatRenderError),
}
impl OriginalChatRenderOperationError {
    pub(super) fn render_controls() -> Option<usize> {
        size_of::<Self>()
            .checked_add(size_of::<TokenInputRejection>())?
            .checked_add(size_of::<Result<(), Self>>())?
            .checked_add(size_of::<Result<OriginalRenderedChat, Self>>())?
            .checked_add(size_of::<Option<OriginalRenderedChat>>())?
            .checked_add(size_of::<Result<Option<OriginalRenderedChat>, Self>>())
    }
}
/// Original chat source/renderer operations. Template/request policy remains in
/// text/facade; the concrete backend supplies its existing neutral original pool.
/// Defaults return a fixed by-value rejection before reading or constructing.
pub trait OriginalChatBackend: OriginalTokenizerBackend {
    /// Compile a borrowed capture declaration against retained selected facts.
    /// Geometry comes from the actual paid prompt; the returned original source
    /// owns fresh declaration storage and does not grant execution authority.
    fn compile_original_capture_declaration(
        _runtime: &ModelRuntime<Self>,
        _plan: &eredu_core::capture::CapturePlan,
        _request: eredu_core::capture::CaptureRequestShape,
        _funding: &eredu_core::HostMetadataFunding,
    ) -> Result<
        crate::working_memory::OriginalCaptureSource,
        crate::working_memory::OriginalCaptureSourceError,
    > {
        Err(crate::working_memory::OriginalCaptureSourceError::rejected(
            crate::working_memory::WorkingMemoryError::UnknownBound,
        ))
    }

    /// Compile the same original intervention source used by restored children,
    /// using this logical session and exact capture geometry. No ordinary source
    /// may be relabeled; failure preserves the concrete preparation cause.
    fn compile_original_intervention_declaration(
        _runtime: &ModelRuntime<Self>,
        _plan: &eredu_core::intervention::InterventionPlan,
        _capture: &eredu_core::capture::SharedCapturePlan,
        _session_id: &str,
        _funding: &eredu_core::HostMetadataFunding,
    ) -> Result<crate::working_memory::OriginalInterventionSource, eredu_core::BackendFailure> {
        Err(TokenInputRejection::Unsupported.into_backend_failure())
    }

    /// Runs the existing neutral original compiler for exact forbidden inputs.
    /// Policy/trigger selection stays in the facade; no native tensor is built.
    fn compile_original_forbidden_source(
        _runtime: &ModelRuntime<Self>,
        _plan: eredu_core::speculative::PreparedForbiddenInputCopy<'_>,
    ) -> Result<
        crate::working_memory::OriginalForbiddenSource,
        crate::working_memory::OriginalForbiddenSourceError,
    > {
        Err(
            crate::working_memory::OriginalForbiddenSourceError::rejected(
                crate::working_memory::WorkingMemoryError::UnknownBound,
            ),
        )
    }

    /// Compiles exact forbidden input bytes directly from the paid immutable
    /// tokenizer. Trigger selection remains in facade semantic request policy.
    fn compile_original_forbidden_tokenizer_source(
        _runtime: &ModelRuntime<Self>,
        _tokenizer: &OriginalTokenizer,
        _trigger: &[u8],
    ) -> Result<
        crate::working_memory::OriginalForbiddenSource,
        crate::working_memory::OriginalForbiddenSourceError,
    > {
        Err(
            crate::working_memory::OriginalForbiddenSourceError::rejected(
                crate::working_memory::WorkingMemoryError::UnknownBound,
            ),
        )
    }

    /// Start source-qualified host preparation for facade-owned behavioral probes.
    /// The default performs no recognition, allocation, render or encoding.
    fn prepare_original_chat_profile(
        _runtime: &ModelRuntime<Self>,
        _template: &OriginalChatTemplate,
        _tokenizer: &OriginalTokenizer,
        _limits: &eredu_core::MemoryLimitDeclarations,
    ) -> Result<super::OriginalChatProfilePreparation, super::OriginalChatProfileError> {
        Err(TokenInputRejection::Unsupported.into())
    }

    /// Authenticate both real J/C owners against the selected runtime account.
    fn validate_original_chat_sources(
        _runtime: &ModelRuntime<Self>,
        _template: &OriginalChatTemplate,
        _tokenizer: &OriginalTokenizer,
    ) -> Result<(), TokenInputRejection> {
        Err(TokenInputRejection::Unsupported)
    }
    /// Authenticate the genuine H and its retained original sources in this pool.
    fn validate_original_chat_render(
        _runtime: &ModelRuntime<Self>,
        _render: &OriginalRenderedChat,
    ) -> Result<(), TokenInputRejection> {
        Err(TokenInputRejection::Unsupported)
    }
    /// Fresh originally funded J from the actual borrowed source plan.
    fn compile_original_chat_template(
        _runtime: &ModelRuntime<Self>,
        _plan: ChatTemplatePlan<'_>,
    ) -> Result<OriginalChatTemplate, OriginalChatSourceError> {
        Err(TokenInputRejection::Unsupported.into())
    }
    /// Consuming exact config-file I followed by fresh J in the same account.
    /// The facade supplies whether the request has tools; the portable text
    /// compiler owns named-template selection and full metadata validation.
    fn compile_original_chat_template_file(
        _runtime: &ModelRuntime<Self>,
        _read: eredu_checkpoint::artifact::PreparedArtifactFileRead,
        _model_id: &str,
        _has_tools: bool,
    ) -> Result<OriginalChatTemplate, OriginalChatSourceError> {
        Err(TokenInputRejection::Unsupported.into())
    }
    /// Exact borrowed defaults, caller values and message controls under the
    /// same original source and render ownership. No caller borrow escapes.
    fn render_original_chat(
        _runtime: &ModelRuntime<Self>,
        _template: &OriginalChatTemplate,
        _tokenizer: &OriginalTokenizer,
        _context: ChatRenderContext<'_>,
    ) -> Result<OriginalRenderedChat, OriginalChatRenderOperationError> {
        Err(TokenInputRejection::Unsupported.into())
    }
}

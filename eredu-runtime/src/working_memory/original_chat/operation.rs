//! Closed by-value chat operation errors and neutral concrete backend entrypoints.
use super::{
    OriginalChatFileError, OriginalChatRenderError, OriginalChatTemplate,
    OriginalChatTemplateError, OriginalRenderedChat,
};
use crate::working_memory::{OriginalTokenizer, OriginalTokenizerBackend};
use eredu_core::{ModelRuntime, TokenInputRejection};
use eredu_text::chat_storage::{
    ChatMessages, ChatRenderContext, ChatSourceError, ChatTemplatePlan,
};
use std::mem::size_of;

/// Actual source/render/file operation error with no pre-admission error erasure.
/// Every admitted variant contains the real closed partial owner by value.
#[derive(Debug, thiserror::Error)]
pub enum OriginalChatOperationError {
    /// Fixed unsupported/foreign-source rejection before any operation.
    #[error(transparent)]
    Domain(#[from] TokenInputRejection),
    /// Checked source/config planning failure before fresh J admission.
    #[error(transparent)]
    Plan(#[from] ChatSourceError),
    /// Actual J constructor/admission error and retained compiler prefix.
    #[error(transparent)]
    Compile(#[from] OriginalChatTemplateError),
    /// Actual H render/admission error and J/C/partial-render custody.
    #[error(transparent)]
    Render(#[from] OriginalChatRenderError),
    /// Consuming config-file I/read/fresh-J error with both original lifetimes.
    #[error(transparent)]
    File(#[from] OriginalChatFileError),
    /// File preparation precedes the original byte boundary. Its real OS error
    /// is preserved by value; opening/error internals are not newly bounded.
    #[error(transparent)]
    FilePreparation(#[from] eredu_checkpoint::artifact::ArtifactFileReadError),
}
impl OriginalChatOperationError {
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
    pub(super) fn render_controls() -> Option<usize> {
        Self::fixed_controls()?
            .checked_add(size_of::<Result<OriginalRenderedChat, Self>>())?
            .checked_add(size_of::<Option<OriginalRenderedChat>>())?
            .checked_add(size_of::<Result<Option<OriginalRenderedChat>, Self>>())
    }
}
/// Original chat source/renderer operations. Template/request policy remains in
/// text/facade; the concrete backend supplies its existing neutral original pool.
/// Defaults return a fixed by-value rejection before reading or constructing.
pub trait OriginalChatBackend: OriginalTokenizerBackend {
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
        _capacity: u64,
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
    ) -> Result<OriginalChatTemplate, OriginalChatOperationError> {
        Err(TokenInputRejection::Unsupported.into())
    }
    /// Consuming exact config-file I followed by fresh J in the same account.
    fn compile_original_chat_template_file(
        _runtime: &ModelRuntime<Self>,
        _read: eredu_checkpoint::artifact::PreparedArtifactFileRead,
        _model_id: &str,
    ) -> Result<OriginalChatTemplate, OriginalChatOperationError> {
        Err(TokenInputRejection::Unsupported.into())
    }
    /// Both exact borrowed-message renderings under one original H comparison.
    fn render_original_chat(
        _runtime: &ModelRuntime<Self>,
        _template: &OriginalChatTemplate,
        _tokenizer: &OriginalTokenizer,
        _messages: ChatMessages<'_>,
        _consumer: eredu_core::GenerationSequenceConsumerLayout,
    ) -> Result<OriginalRenderedChat, OriginalChatOperationError> {
        Err(TokenInputRejection::Unsupported.into())
    }
    /// Exact borrowed defaults/caller values, with the same original source and
    /// render ownership. Older backends may serve only an actually empty context.
    fn render_original_chat_with_context(
        runtime: &ModelRuntime<Self>,
        template: &OriginalChatTemplate,
        tokenizer: &OriginalTokenizer,
        context: ChatRenderContext<'_>,
        consumer: eredu_core::GenerationSequenceConsumerLayout,
    ) -> Result<OriginalRenderedChat, OriginalChatOperationError> {
        if !context.is_plain() {
            return Err(TokenInputRejection::Unsupported.into());
        }
        Self::render_original_chat(runtime, template, tokenizer, context.messages(), consumer)
    }
}

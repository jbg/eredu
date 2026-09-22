//! Original source compilation and canonical prepared-chat publication.
use super::original_token_input::chat::{
    compile_original_chat_file, prepare_original_chat, OriginalChatError,
};
use super::{LoadedModel, ManagedPlainTextSource};
use crate::runtime::chat::{ChatTemplateRequest, DependencyMemoryPolicy};
use eredu_core::{BackendFailure, GenerationCancellationToken, TokenInputRejection};
use eredu_runtime::working_memory::{
    OriginalChatBackend, OriginalChatSourceError, OriginalChatTemplate,
};

/// Actual input to fresh selected chat-template source preparation.
#[derive(Debug)]
pub enum ChatSourceInput {
    /// Read a consumed tokenizer-config JSON file under its original allowance.
    File(std::fs::File),
    /// Borrow the complete template selection retained by this loaded model.
    RetainedConfiguration,
}
impl From<std::fs::File> for ChatSourceInput {
    fn from(file: std::fs::File) -> Self {
        Self::File(file)
    }
}

/// Originally compiled tokenizer and chat template matched to loaded metadata.
/// Clones retain the same source accounts without exposing mutable compiler state.
#[derive(Clone, Debug)]
pub struct ManagedChatSource {
    tokenizer: ManagedPlainTextSource,
    template: OriginalChatTemplate,
}

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Input(#[from] TokenInputRejection),
    #[error("chat preparation requires a selected chat template")]
    MissingTemplate,
    #[error(transparent)]
    Chat(#[from] OriginalChatError),
    #[error(transparent)]
    Backend(#[from] BackendFailure),
    #[error(transparent)]
    Publication(#[from] crate::runtime::chat::PreparedChatPublicationError),
}
/// Backend-independent managed chat failure retaining original failed-prefix owners.
/// It carries no native error type or backend-error type parameter.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct ManagedChatError {
    #[source]
    cause: Cause,
}
impl ManagedChatError {
    fn new(cause: impl Into<Cause>) -> Self {
        Self {
            cause: cause.into(),
        }
    }
    /// Returns a fixed input/source identity refusal without error allocation.
    pub fn input_rejection(&self) -> Option<TokenInputRejection> {
        match &self.cause {
            Cause::Input(error) => Some(*error),
            Cause::Chat(error) => error.input_rejection().copied(),
            _ => None,
        }
    }
}

/// Template source compilation failure with its original partial compiler owner.
/// Generation failures do not carry this source-only compiler storage.
#[derive(Debug, thiserror::Error)]
pub enum ManagedChatSourceError {
    /// Loaded metadata or source identity failed validation.
    #[error(transparent)]
    Validation(#[from] ManagedChatError),
    /// The consumed file or selected template failed original preparation.
    #[error(transparent)]
    Source(#[from] OriginalChatSourceError),
}

impl<B: OriginalChatBackend> LoadedModel<B> {
    fn validate_managed_chat_metadata(
        &self,
        source: &ManagedChatSource,
        has_tools: bool,
    ) -> Result<(), ManagedChatError> {
        let template = self
            .chat_template
            .as_ref()
            .ok_or_else(|| ManagedChatError::new(Cause::MissingTemplate))?;
        if !source
            .tokenizer
            .original()
            .matches_configuration(&self.tokenizer)
            || !source
                .template
                .matches_selection(template, &self.model_id, has_tools)
        {
            return Err(ManagedChatError::new(TokenInputRejection::IdentityMismatch));
        }
        Ok(())
    }

    /// Compiles the actual selected template under the existing original J account.
    /// The tokenizer source is created with `compile_managed_plain_text_source`.
    /// `has_tools` selects the same named entry as the eventual request's
    /// non-empty declaration list. Exact source/name and tokenizer must match.
    /// File opening and loaded-model construction remain separate operations.
    pub fn compile_managed_chat_source(
        &self,
        tokenizer: &ManagedPlainTextSource,
        input: impl Into<ChatSourceInput>,
        has_tools: bool,
        cancellation: &GenerationCancellationToken,
    ) -> Result<Option<ManagedChatSource>, ManagedChatSourceError> {
        if cancellation.is_cancelled() {
            return Ok(None);
        }
        if self.chat_template.is_none() {
            return Err(ManagedChatError::new(Cause::MissingTemplate).into());
        }
        if !tokenizer.original().matches_configuration(&self.tokenizer) {
            return Err(ManagedChatError::new(TokenInputRejection::IdentityMismatch).into());
        }
        B::validate_original_tokenizer_source(&self.runtime, tokenizer.original())
            .map_err(ManagedChatError::new)?;
        let template = match input.into() {
            ChatSourceInput::File(file) => {
                let Some(template) = compile_original_chat_file(
                    &self.runtime,
                    file,
                    &self.model_id,
                    has_tools,
                    cancellation,
                )?
                else {
                    return Ok(None);
                };
                template
            }
            ChatSourceInput::RetainedConfiguration => {
                let plan = eredu_text::chat_storage::ChatTemplatePlan::prepare_model(
                    self.chat_template.as_ref().expect("template checked above"),
                    &self.model_id,
                    has_tools,
                )
                .map_err(OriginalChatSourceError::from)?;
                let template = B::compile_original_chat_template(&self.runtime, plan)?;
                if cancellation.is_cancelled() {
                    return Ok(None);
                }
                template
            }
        };
        let source = ManagedChatSource {
            tokenizer: tokenizer.clone(),
            template,
        };
        self.validate_managed_chat_metadata(&source, has_tools)?;
        B::validate_original_chat_sources(
            &self.runtime,
            &source.template,
            source.tokenizer.original(),
        )
        .map_err(ManagedChatError::new)?;
        Ok(Some(source))
    }

    /// Prepares one authenticated chat under the supplied managed limits.
    /// The actual request stays borrowed through profile, policy and rendering;
    /// the result retains their original allocations for every execution consumer.
    /// Cancellation returns `None` before the next expensive preparation stage.
    pub fn prepare_chat(
        &self,
        source: &ManagedChatSource,
        request: &ChatTemplateRequest,
        limits: &eredu_core::MemoryLimitDeclarations,
        cancellation: &GenerationCancellationToken,
    ) -> Result<Option<crate::runtime::chat::PreparedChat>, ManagedChatError> {
        self.prepare_chat_with_grammar_memory(
            source,
            request,
            limits,
            DependencyMemoryPolicy::default(),
            cancellation,
        )
    }

    /// Prepares a chat with selected planning headroom for stock grammar and
    /// tool-schema compilation, validation, sessions and snapshots. Controlled
    /// and uninterrupted execution retain this same policy with the prepared source.
    ///
    /// The allowance is an estimate, not a dependency or process memory ceiling.
    /// Tokenizer/trie, template rendering, native resources and first-party buffers
    /// retain their separate policies and admission. Cancellation returns `None`.
    pub fn prepare_chat_with_grammar_memory(
        &self,
        source: &ManagedChatSource,
        request: &ChatTemplateRequest,
        limits: &eredu_core::MemoryLimitDeclarations,
        grammar_memory: DependencyMemoryPolicy,
        cancellation: &GenerationCancellationToken,
    ) -> Result<Option<crate::runtime::chat::PreparedChat>, ManagedChatError> {
        if cancellation.is_cancelled() {
            return Ok(None);
        }
        self.validate_managed_chat_metadata(source, !request.tools.is_empty())?;
        let (_, named_entry) = self
            .chat_template
            .as_ref()
            .and_then(|template| template.selected_source(!request.tools.is_empty()))
            .ok_or_else(|| ManagedChatError::new(Cause::MissingTemplate))?;
        let Some((rendered, (policy, compilation))) = prepare_original_chat(
            &self.runtime,
            &source.template,
            source.tokenizer.original(),
            request,
            Some(self.tokenizer.template_kwargs()),
            &self.eos_token_ids,
            limits,
            grammar_memory,
            cancellation,
        )
        .map_err(ManagedChatError::new)?
        else {
            return Ok(None);
        };
        let prepared = rendered
            .publish(
                policy,
                compilation,
                named_entry,
                request.add_generation_prompt,
            )
            .map_err(ManagedChatError::new)?;
        Ok(Some(prepared))
    }
}

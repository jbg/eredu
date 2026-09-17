//! Loaded-model chat composition over the original renderer and shared text cursor.
use super::original_token_input::chat::{
    ManagedChatPolicyRejection, OriginalChatError, OriginalTextChatPreparation,
    compile_original_chat_file, prepare_original_text_chat_with_defaults_for,
    start_original_text_chat_for,
};
use super::{LoadedModel, ManagedPlainTextSession, ManagedPlainTextSource};
use crate::api::PreparedChatGenerationSettings;
use crate::runtime::chat::ChatTemplateRequest;
use eredu_core::{
    BackendFailure, GenerationCancellationToken, GenerationPlainTextEvent,
    GenerationPlainTextOutput, TokenInputRejection,
};
use eredu_runtime::working_memory::{
    OriginalChatBackend, OriginalChatOperationError, OriginalChatTemplate,
};

/// Originally compiled tokenizer and chat template matched to loaded metadata.
/// Clones retain the same source accounts without exposing mutable compiler state.
#[derive(Clone, Debug)]
pub struct ManagedChatSource {
    tokenizer: ManagedPlainTextSource,
    template: OriginalChatTemplate,
}

/// Borrowed text-chat request using the existing template and sampling policy.
/// Message and stop storage stays caller-owned during startup.
#[derive(Debug, Clone, Copy)]
pub struct ManagedChatRequest<'a> {
    /// Actual message/tool/reasoning/template policy; unintegrated policies reject.
    pub chat: &'a ChatTemplateRequest,
    /// Checkpoint overrides, sampler, seed and enforced caller memory ceiling.
    pub settings: PreparedChatGenerationSettings,
    /// Literal output stops in caller order.
    pub stop_sequences: &'a [&'a str],
    /// Omit special-token spellings from visible output.
    pub skip_special_tokens: bool,
}
impl<'a> ManagedChatRequest<'a> {
    /// Uses no literal stops and hides special-token output spellings.
    pub fn new(chat: &'a ChatTemplateRequest, settings: PreparedChatGenerationSettings) -> Self {
        Self {
            chat,
            settings,
            stop_sequences: &[],
            skip_special_tokens: true,
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Input(#[from] TokenInputRejection),
    #[error(transparent)]
    Policy(#[from] ManagedChatPolicyRejection),
    #[error(transparent)]
    Generation(#[from] eredu_core::generation::GenerationError),
    #[error(transparent)]
    Operation(#[from] OriginalChatOperationError),
    #[error(transparent)]
    Chat(#[from] OriginalChatError<BackendFailure>),
    #[error(transparent)]
    Backend(#[from] BackendFailure),
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
    fn chat<B: OriginalChatBackend>(error: OriginalChatError<B::Error>) -> Self {
        Self::new(error.into_neutral::<B>())
    }
    /// Returns a fixed input/source identity refusal without error allocation.
    pub fn input_rejection(&self) -> Option<TokenInputRejection> {
        match &self.cause {
            Cause::Input(error) => Some(*error),
            Cause::Chat(error) => error.input_rejection().copied(),
            Cause::Operation(OriginalChatOperationError::Domain(error)) => Some(*error),
            _ => None,
        }
    }
    /// Identifies an unfinished managed policy integration, not a model limitation.
    pub fn policy_rejection(&self) -> Option<ManagedChatPolicyRejection> {
        match &self.cause {
            Cause::Policy(error) => Some(*error),
            Cause::Chat(error) => error.policy_rejection(),
            _ => None,
        }
    }
}

// Public and private preparation/startup/error/terminal returns share the same
// actual H/S/E/I/R consumer specialization. Lifetimes affect borrows, not layout.
type PublicConsumer<'a, B> = (
    ManagedChatRequest<'static>,
    ManagedChatError,
    Result<Option<OriginalTextChatPreparation>, ManagedChatError>,
    Result<Option<ManagedPlainTextSession<'a, B>>, ManagedChatError>,
    Result<Option<GenerationPlainTextOutput>, ManagedChatError>,
);

impl<B: OriginalChatBackend> LoadedModel<B> {
    fn validate_managed_chat_metadata(
        &self,
        source: &ManagedChatSource,
    ) -> Result<(), ManagedChatError> {
        let template = self
            .chat_template
            .as_ref()
            .ok_or_else(|| ManagedChatError::new(ManagedChatPolicyRejection::MissingTemplate))?;
        if !source
            .tokenizer
            .original()
            .matches_configuration(&self.tokenizer)
            || !source
                .template
                .matches_configuration(template, &self.model_id)
        {
            return Err(ManagedChatError::new(TokenInputRejection::IdentityMismatch));
        }
        Ok(())
    }

    /// Compiles a consumed tokenizer-config file under the existing original J account.
    /// The tokenizer source is created with `compile_managed_plain_text_source`.
    /// Exact loaded template source/name and tokenizer configuration must match.
    /// File opening and loaded-model construction remain separate operations.
    pub fn compile_managed_chat_source(
        &self,
        tokenizer: &ManagedPlainTextSource,
        file: std::fs::File,
        cancellation: &GenerationCancellationToken,
    ) -> Result<Option<ManagedChatSource>, ManagedChatError> {
        if cancellation.is_cancelled() {
            return Ok(None);
        }
        if self.chat_template.is_none() {
            return Err(ManagedChatError::new(
                ManagedChatPolicyRejection::MissingTemplate,
            ));
        }
        if !tokenizer.original().matches_configuration(&self.tokenizer) {
            return Err(ManagedChatError::new(TokenInputRejection::IdentityMismatch));
        }
        B::validate_original_tokenizer_source(&self.runtime, tokenizer.original())
            .map_err(ManagedChatError::new)?;
        let Some(template) =
            compile_original_chat_file(&self.runtime, file, &self.model_id, cancellation)
                .map_err(ManagedChatError::new)?
        else {
            return Ok(None);
        };
        let source = ManagedChatSource {
            tokenizer: tokenizer.clone(),
            template,
        };
        self.validate_managed_chat_metadata(&source)?;
        B::validate_original_chat_sources(
            &self.runtime,
            &source.template,
            source.tokenizer.original(),
        )
        .map_err(ManagedChatError::new)?;
        Ok(Some(source))
    }

    /// Renders and starts the existing controlled managed text cursor.
    /// A caller memory ceiling is required and applies before rendering or S/E.
    /// Missing complete native fit and unfinished chat policies remain typed refusals.
    /// None means cancellation before a session was started.
    pub fn start_managed_chat<'a>(
        &'a mut self,
        source: &ManagedChatSource,
        request: ManagedChatRequest<'_>,
        cancellation: &GenerationCancellationToken,
    ) -> Result<Option<ManagedPlainTextSession<'a, B>>, ManagedChatError> {
        if cancellation.is_cancelled() {
            return Ok(None);
        }
        self.validate_managed_chat_metadata(source)?;
        let (config, _) = self
            .resolve_text_generation_settings(request.settings)
            .map_err(ManagedChatError::new)?;
        let Some(prepared) =
            prepare_original_text_chat_with_defaults_for::<B, PublicConsumer<'a, B>>(
                &self.runtime,
                &source.template,
                source.tokenizer.original(),
                request.chat,
                Some(self.tokenizer.template_kwargs()),
                config,
                cancellation,
            )
            .map_err(ManagedChatError::chat::<B>)?
        else {
            return Ok(None);
        };
        start_original_text_chat_for::<B, PublicConsumer<'a, B>>(
            &mut self.runtime,
            &source.template,
            source.tokenizer.original(),
            prepared,
            request.chat.add_generation_prompt,
            config,
            &self.eos_token_ids,
            request.stop_sequences,
            request.skip_special_tokens,
            cancellation,
        )
        .map(|session| session.map(ManagedPlainTextSession))
        .map_err(ManagedChatError::chat::<B>)
    }

    /// Runs the same session advancement as `start_managed_chat`, lending text events
    /// and retaining terminal output without an owned semantic-event conversion.
    pub fn generate_managed_chat(
        &mut self,
        source: &ManagedChatSource,
        request: ManagedChatRequest<'_>,
        cancellation: &GenerationCancellationToken,
        emit: &mut impl for<'e> FnMut(GenerationPlainTextEvent<'e>),
    ) -> Result<Option<GenerationPlainTextOutput>, ManagedChatError> {
        self.start_managed_chat(source, request, cancellation)?
            .map(|session| {
                session
                    .run(cancellation, emit)
                    .map_err(ManagedChatError::new)
            })
            .transpose()
    }
}

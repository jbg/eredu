//! Shared prompt authentication, source selection and semantic preparation.
use super::session::{ChatControllerError, PreparedSemanticInput};
use super::*;
use crate::api::PreparedChatGenerationSettings;
use crate::runtime::chat::PreparedChat;
use eredu_core::{ControlledTextGenerationError, TokenIdsInputPlan, TokenInputRejection};
use eredu_runtime::{input::OriginalModelInput, working_memory::OriginalChatBackend};

pub(crate) struct PreparedChatInvocation<'a, B: TextGenerationBackend> {
    pub(crate) prepared: chat::PreparedChatSemantics,
    pub(crate) config: TextGenerationConfig,
    pub(crate) input: PreparedSemanticInput<'a, B>,
}
impl<B: OriginalChatBackend> LoadedModel<B> {
    /// Check every lane before any paid semantic source is constructed. This
    /// borrows the exact input and shares validation with ordinary startup.
    pub(crate) fn validate_chat_invocation(
        &self,
        chat: &PreparedChat,
        input: &PreparedChatPrompt<'_, B::Prompt>,
        settings: PreparedChatGenerationSettings,
    ) -> Result<(TextGenerationConfig, std::num::NonZeroUsize), PreparedChatSessionError> {
        self.validate_prepared_chat(chat)?;
        if let PreparedChatPrompt::TokenIds(ids) = input {
            TokenIdsInputPlan::new(ids).map_err(PreparedChatSessionError::before)?;
            let domain = chat.tokenizer_source().generation_domain().ok_or_else(|| {
                PreparedChatSessionError::before(TokenInputRejection::IdentityMismatch)
            })?;
            if ids.iter().any(|&id| !domain.allows(id)) {
                return Err(PreparedChatSessionError::before(
                    TokenInputRejection::InvalidToken,
                ));
            }
        }
        let capacity = settings
            .inference
            .managed_memory_capacity_bytes
            .ok_or_else(|| {
                PreparedChatSessionError::before(TokenInputRejection::Unsupported)
            })?;
        if capacity != chat.capacity() {
            return Err(PreparedChatSessionError::before(
                TokenInputRejection::IdentityMismatch,
            ));
        }
        let (config, maximum) = self
            .resolve_text_generation_settings(settings)
            .map_err(PreparedChatSessionError::before)?;
        if let PreparedChatPrompt::Media(input) = input {
            let binding = input.chat_binding().ok_or_else(|| {
                PreparedChatSessionError::before(TokenInputRejection::IdentityMismatch)
            })?;
            if !binding.matches(chat.render(), binding.preparation(), chat.generation())
                || binding.preparation().capacity_bytes() != capacity
            {
                return Err(PreparedChatSessionError::before(
                    TokenInputRejection::IdentityMismatch,
                ));
            }
            B::validate_semantic_source(&self.runtime, binding.preparation())
                .map_err(PreparedChatSessionError::before)?;
        }
        Ok((config, maximum))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_chat_invocation<'a>(
        &self,
        chat: &'a PreparedChat,
        mut input: PreparedChatPrompt<'a, B::Prompt>,
        settings: PreparedChatGenerationSettings,
        event_window: std::num::NonZeroUsize,
        stop_sequences: &[String],
        skip_special_tokens: bool,
        output_mode: PreparedChatOutputMode,
    ) -> Result<PreparedChatInvocation<'a, B>, PreparedChatSessionError> {
        let prepared = (|| {
            let (config, maximum) = self.validate_chat_invocation(chat, &input, settings)?;
            let capacity = chat.capacity();
            let preparation = match &input {
                PreparedChatPrompt::Media(input) => input
                    .chat_binding()
                    .expect("validated original chat input binding")
                    .preparation()
                    .clone(),
                PreparedChatPrompt::Rendered | PreparedChatPrompt::TokenIds(_) => {
                    self.prepare_semantic_source(chat.tokenizer_source(), capacity)?
                }
            };
            let funding = preparation.metadata_funding().clone();
            let parts = [
                size_of::<PreparedChatRequest<'_, B::Prompt>>(),
                size_of::<PreparedChatInvocation<'_, B>>(),
                size_of::<Option<OriginalModelInput<B::Prompt>>>(),
                size_of::<PreparedSemanticInput<'_, B>>(),
                size_of::<ControlledTextGenerationError<B::Error, ChatControllerError>>(),
                size_of::<
                    Result<
                        (chat::PreparedChatSemantics, TextGenerationConfig),
                        PreparedChatSessionError,
                    >,
                >(),
            ];
            reserve(&funding, sum(&parts)).map_err(|cause| PreparedChatSessionError {
                cause,
                funding: Some(funding.clone()),
                input_funding: None,
            })?;
            let prepared = self.prepare_chat_semantics(
                preparation,
                chat,
                maximum.get(),
                event_window,
                stop_sequences,
                skip_special_tokens,
                output_mode,
            )?;
            Ok((prepared, config))
        })()
        .map_err(|mut failure: PreparedChatSessionError| {
            if let PreparedChatPrompt::Media(input) =
                std::mem::replace(&mut input, PreparedChatPrompt::Rendered)
            {
                let (prompt, custody) = input.into_parts();
                drop(prompt);
                failure.input_funding = Some(custody);
            }
            failure
        })?;
        let input = match input {
            PreparedChatPrompt::Media(input) => PreparedSemanticInput::Media {
                input,
                render: chat.render().clone(),
                generation_prompt: chat.generation(),
            },
            PreparedChatPrompt::Rendered => PreparedSemanticInput::Text(chat.rendered_prompt()),
            PreparedChatPrompt::TokenIds(ids) => PreparedSemanticInput::TokenIds(ids),
        };
        Ok(PreparedChatInvocation {
            prepared: prepared.0,
            config: prepared.1,
            input,
        })
    }
}

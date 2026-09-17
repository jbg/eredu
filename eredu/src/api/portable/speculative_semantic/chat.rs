//! Exact prepared-chat declarations into the existing original semantic host.
use super::*;
use crate::api::PreparedChatSpeculativeConstraint;
use crate::runtime::chat::PreparedChat;
use crate::runtime::{
    chat::constraints::{ConstraintController, OriginalControllerSourceError},
    generation::streaming::prepared_channels,
};
use eredu_runtime::working_memory::{OriginalChatBackend, OriginalSemanticChannelSource};

#[derive(Debug, thiserror::Error)]
pub(super) enum ChatCause {
    #[error("prepared chat has no executable semantic declaration")]
    Declaration,
    #[error("prepared chat controller has no original semantic source")]
    Controller,
    #[error("prepared chat stop reference population changed")]
    Population,
    #[error(transparent)]
    ControllerSource(#[from] OriginalControllerSourceError),
    #[error(transparent)]
    Channels(#[from] prepared_channels::Failure),
}

impl<B: OriginalChatBackend> LoadedModel<B> {
    /// This producer preserves the historical controller recipe and the actual
    /// selected generation decoder separately, matching ordinary prepared chat.
    /// It performs no prompt encoding, backend inference, or mode substitution.
    pub(crate) fn prepare_original_speculative_chat_host<'a, F: FnMut(SemanticEvent) + 'a>(
        &self,
        source: &ManagedPlainTextSource,
        prepared_chat: &PreparedChat,
        capacity: u64,
        maximum: usize,
        maximum_draft: usize,
        temperature: f32,
        caller_stops: &[String],
        callback: F,
    ) -> Result<
        (
            PreparedOriginalSpeculativeHost<'a>,
            PreparedChatSpeculativeConstraint,
        ),
        PreparedPlainSpeculativeSemanticError,
    > {
        if !source.original().matches_configuration(&self.tokenizer) {
            return Err(PreparedPlainSpeculativeSemanticError {
                cause: SpeculativeOutputError::Storage(
                    "semantic tokenizer does not match the loaded source",
                )
                .into(),
                funding: None,
            });
        }
        let preparation =
            B::prepare_original_speculative_semantic(&self.runtime, source.original(), capacity)
                .map_err(|cause| PreparedPlainSpeculativeSemanticError {
                    cause: cause.into(),
                    funding: None,
                })?;
        let funding = preparation.metadata_funding().clone();
        let result = (|| -> Result<_, Cause> {
            let parts = [
                size_of::<(
                    &Self,
                    &ManagedPlainTextSource,
                    &PreparedChat,
                    u64,
                    usize,
                    usize,
                    f32,
                    &[String],
                    F,
                )>(),
                size_of::<PreparedOriginalSpeculativeHost<'a>>(),
                size_of::<PreparedChatSpeculativeConstraint>(),
                size_of::<ConstraintController>(),
                size_of::<eredu_runtime::working_memory::OriginalSemanticControllerSource<'_>>(),
                size_of::<Option<eredu_runtime::working_memory::OriginalSemanticControllerSource<'_>>>(),
                eredu_core::speculative::PreparedGrammarSource::control_bytes().ok_or(
                    SpeculativeOutputError::HostFunding(eredu_core::HostMetadataFundingError::Overflow),
                )?,
                size_of::<(
                    PreparedOriginalSpeculativeHost<'a>,
                    PreparedChatSpeculativeConstraint,
                )>(),
                size_of::<
                    Result<
                        (
                            PreparedOriginalSpeculativeHost<'a>,
                            PreparedChatSpeculativeConstraint,
                        ),
                        PreparedPlainSpeculativeSemanticError,
                    >,
                >(),
                size_of::<Result<ConstraintController, OriginalControllerSourceError>>(),
                size_of::<Result<OriginalSemanticChannelSource, prepared_channels::Failure>>(),
                size_of::<StopCompilePlan<'_>>(),
                size_of::<Result<StopCompilePlan<'_>, eredu_text::stop_storage::StopSourceError>>(),
                size_of::<
                    Result<
                        eredu_runtime::working_memory::OriginalStopSource,
                        OriginalTextSourceError,
                    >,
                >(),
                size_of::<Result<SpeculativeSemanticOwner, SpeculativeOutputError>>(),
                size_of::<Result<SpeculativeConfiguration, OriginalSpeculativeHostError>>(),
                size_of::<Result<SpeculativeEventCallback<'a>, SpeculativeOutputError>>(),
                size_of::<ChatCause>(),
                size_of::<Cause>(),
                size_of::<PreparedPlainSpeculativeSemanticError>(),
            ];
            reserve(&funding, sum(&parts))?;
            let semantic_plan = prepared_chat
                .semantic_runtime_plan()
                .ok_or(ChatCause::Declaration)?;
            let generation_plan = prepared_chat
                .generation_runtime_plan()
                .ok_or(ChatCause::Declaration)?;
            let controller = ConstraintController::from_original_generation_plan::<B>(
                &self.runtime,
                generation_plan,
                self.token_validity.clone(),
                maximum,
                &funding,
            )
            .map_err(ChatCause::from)?;
            let controller_source = controller
                .original_semantic_inputs()
                .ok_or(ChatCause::Controller)?;
            let channels = prepared_channels::compile_source_for_controller(
                semantic_plan,
                source.original(),
                controller_source,
                &funding,
            )
            .map_err(ChatCause::from)?;
            let stop_count = semantic_plan
                .original_literal_stops()
                .count()
                .checked_add(caller_stops.len())
                .ok_or(SpeculativeOutputError::HostFunding(
                    eredu_core::HostMetadataFundingError::Overflow,
                ))?;
            let refs_host = host(
                &funding,
                SpeculativeBuffer::<&str>::retained_control_bytes(stop_count)
                    .and_then(|n| {
                        n.checked_add(size_of::<
                            Result<SpeculativeBuffer<&str>, SpeculativeBufferAllocationError>,
                        >())
                    })
                    .and_then(|n| {
                        n.checked_add(size_of::<Result<(), eredu_core::GenerationError>>())
                    })
                    .and_then(|n| n.checked_add(size_of::<std::slice::Iter<'_, String>>())),
            )?;
            let mut stops = SpeculativeBuffer::try_new_retained(stop_count, refs_host)?;
            stops
                .try_extend(
                    semantic_plan
                        .original_literal_stops()
                        .chain(caller_stops.iter().map(String::as_str)),
                )
                .map_err(|_| ChatCause::Population)?;
            let plan = StopCompilePlan::prepare_refs(&stops).map_err(|_| ChatCause::Declaration)?;
            let stops = B::compile_original_text_stop_source(&self.runtime, plan)?;
            let semantic =
                preparation.prepare_channels(&stops, &channels, maximum, maximum_draft, true)?;
            let configuration = preparation.prepare_configuration(
                maximum,
                maximum_draft,
                temperature,
                prepared_chat.eos_token_ids(),
            )?;
            let callback = preparation.prepare_callback(callback)?;
            Ok((
                PreparedOriginalSpeculativeHost {
                    semantic,
                    configuration,
                    callback,
                    preparation,
                },
                PreparedChatSpeculativeConstraint::new(controller),
            ))
        })();
        result.map_err(|cause| PreparedPlainSpeculativeSemanticError {
            cause,
            funding: Some(funding),
        })
    }
}

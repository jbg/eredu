//! Shared funded controller, stop and semantic state preparation.
use super::*;
use crate::runtime::chat::PreparedChat;
use crate::runtime::{
    chat::constraints::{ConstraintController, OriginalControllerSourceError},
    generation::streaming::prepared_channels,
};
use eredu_runtime::working_memory::{OriginalChatBackend, OriginalSemanticChannelSource};

#[derive(Debug, thiserror::Error)]
pub(super) enum ChatCause {
    #[error("prepared chat does not permit semantic output: {0}")]
    SemanticOutput(&'static str),
    #[error("prepared chat does not permit literal text output: {0}")]
    TextOutput(&'static str),
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
    #[error(transparent)]
    Plain(#[from] eredu_core::speculative::PlainControllerError),
    #[error(transparent)]
    Binding(#[from] eredu_runtime::working_memory::PreparedControllerBindingError),
    #[error(transparent)]
    Compilation(#[from] eredu_runtime::working_memory::WorkingMemoryError),
}

impl<B: OriginalChatBackend> LoadedModel<B> {
    /// Builds the exact controller and semantic state independently of a model
    /// execution strategy. The event window counts token transitions awaiting
    /// publication, independently of the execution scheduler.
    pub(crate) fn prepare_chat_semantics(
        &self,
        preparation: PreparedSemanticSource,
        prepared_chat: &PreparedChat,
        maximum: usize,
        event_window: std::num::NonZeroUsize,
        caller_stops: &[String],
        skip_special_tokens: bool,
        output_mode: PreparedChatOutputMode,
    ) -> Result<PreparedChatSemantics, PreparedChatSessionError> {
        if !preparation
            .tokenizer()
            .matches_configuration(&self.tokenizer)
        {
            return Err(PreparedChatSessionError {
                cause: eredu_core::TokenInputRejection::IdentityMismatch.into(),
                funding: Some(preparation.metadata_funding().clone()),
                input_funding: None,
            });
        }
        let funding = preparation.metadata_funding().clone();
        let result = (|| -> Result<_, Cause> {
            let parts = [
                size_of::<(
                    &Self,
                    PreparedSemanticSource,
                    &PreparedChat,
                    usize,
                    std::num::NonZeroUsize,
                    &[String],
                    bool,
                    PreparedChatOutputMode,
                )>(),
                size_of::<PreparedChatSemantics>(),
                size_of::<ConstraintController>(),
                size_of::<eredu_runtime::working_memory::OriginalSemanticControllerSource<'_>>(),
                size_of::<
                    Option<eredu_runtime::working_memory::OriginalSemanticControllerSource<'_>>,
                >(),
                eredu_core::speculative::PreparedGrammarSource::control_bytes().ok_or(
                    SpeculativeOutputError::HostFunding(
                        eredu_core::HostMetadataFundingError::Overflow,
                    ),
                )?,
                size_of::<Result<PreparedChatSemantics, PreparedChatSessionError>>(),
                size_of::<Result<ConstraintController, OriginalControllerSourceError>>(),
                size_of::<
                    Result<
                        ConstraintController,
                        eredu_runtime::working_memory::PreparedControllerBindingError,
                    >,
                >(),
                size_of::<Result<OriginalSemanticChannelSource, prepared_channels::Failure>>(),
                size_of::<StopCompilePlan<'_>>(),
                size_of::<Result<StopCompilePlan<'_>, eredu_text::stop_storage::StopSourceError>>(),
                size_of::<
                    Result<
                        eredu_runtime::working_memory::OriginalStopSource,
                        OriginalTextSourceError,
                    >,
                >(),
                size_of::<Result<SemanticStateOwner, SpeculativeOutputError>>(),
                size_of::<ChatCause>(),
                size_of::<Cause>(),
                size_of::<PreparedChatSessionError>(),
            ];
            reserve(&funding, sum(&parts))?;
            prepared_chat
                .compilation()
                .validate_preparation(prepared_chat.controller_sources(), &preparation)
                .map_err(ChatCause::from)?;
            let literal = output_mode == PreparedChatOutputMode::Text;
            if literal {
                if let crate::runtime::chat::CapabilitySupport::Unsupported { reason } =
                    prepared_chat.text_generation_support()
                {
                    return Err(ChatCause::TextOutput(reason).into());
                }
            } else if let crate::runtime::chat::SemanticSupport::Unsupported { reason } =
                prepared_chat.semantic_support()
            {
                return Err(ChatCause::SemanticOutput(reason).into());
            }
            let semantic_plan = (!literal)
                .then(|| prepared_chat.semantic_runtime_plan())
                .flatten();
            let generation_plan = (!literal)
                .then(|| prepared_chat.generation_runtime_plan())
                .flatten();
            let controller = match (semantic_plan, generation_plan) {
                (Some(_), Some(generation_plan)) => {
                    ConstraintController::from_original_generation_plan::<B>(
                        &self.runtime,
                        preparation.tokenizer(),
                        generation_plan,
                        prepared_chat.compilation(),
                        self.token_validity.clone(),
                        maximum,
                        &funding,
                    )
                    .map_err(ChatCause::from)?
                }
                (None, None) => {
                    let history =
                        eredu_core::speculative::PlainControllerHistory::copy_metadata_bytes(
                            maximum,
                        )
                        .ok_or(SpeculativeOutputError::HostFunding(
                            eredu_core::HostMetadataFundingError::Overflow,
                        ))?;
                    let controls = ConstraintController::text_metadata_bytes()
                        .and_then(|n| n.checked_add(history))
                        .and_then(|n| {
                            n.checked_add(size_of::<
                                Result<
                                    ConstraintController,
                                    eredu_core::speculative::PlainControllerError,
                                >,
                            >())
                        });
                    let authority = host(&funding, controls)?;
                    ConstraintController::text_prepared(
                        self.token_validity.clone(),
                        authority.clone(),
                    )
                    .copy_prepared_plain(maximum, authority)
                    .map_err(ChatCause::from)?
                }
                _ => return Err(ChatCause::Declaration.into()),
            };
            let controller = controller
                .bind_preparation(&preparation)
                .map_err(ChatCause::from)?;
            let channels = match semantic_plan {
                Some(plan) => Some(
                    prepared_channels::compile_source_for_controller(
                        plan,
                        preparation.tokenizer(),
                        controller
                            .original_semantic_inputs()
                            .ok_or(ChatCause::Controller)?,
                        &funding,
                    )
                    .map_err(ChatCause::from)?,
                ),
                None => None,
            };
            let literals = || {
                semantic_plan
                    .into_iter()
                    .flat_map(|plan| plan.original_literal_stops())
                    .chain(
                        prepared_chat
                            .profile_stop_sequences()
                            .iter()
                            .filter(|_| !literal && semantic_plan.is_none())
                            .map(String::as_str),
                    )
            };
            let stop_count = literals().count().checked_add(caller_stops.len()).ok_or(
                SpeculativeOutputError::HostFunding(eredu_core::HostMetadataFundingError::Overflow),
            )?;
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
                .try_extend(literals().chain(caller_stops.iter().map(String::as_str)))
                .map_err(|_| ChatCause::Population)?;
            let plan = StopCompilePlan::prepare_refs(&stops).map_err(|_| ChatCause::Declaration)?;
            let stops = B::compile_original_text_stop_source(&self.runtime, plan)?;
            let semantic = match channels {
                Some(channels) => preparation.prepare_channels(
                    &stops,
                    &channels,
                    maximum,
                    event_window,
                    skip_special_tokens,
                )?,
                None => preparation.prepare(&stops, maximum, event_window, skip_special_tokens)?,
            };
            Ok(PreparedChatSemantics {
                semantic,
                controller,
                preparation,
            })
        })();
        result.map_err(|cause| PreparedChatSessionError {
            cause,
            funding: Some(funding),
            input_funding: None,
        })
    }
}

/// Owns the original controller and semantic source through consuming assembly.
/// It grants no native permission and does not choose an inference scheduler.
pub(crate) struct PreparedChatSemantics {
    pub(crate) semantic: SemanticStateOwner,
    pub(crate) controller: ConstraintController,
    pub(crate) preparation: PreparedSemanticSource,
}
#[cfg(test)]
impl PreparedChatSemantics {
    pub(crate) fn controller_mut(&mut self) -> &mut ConstraintController {
        &mut self.controller
    }
}

impl<B: OriginalChatBackend> LoadedModel<B> {
    /// Speculation adds its own configuration and callback to shared preparation.
    pub(crate) fn prepare_original_speculative_chat_host<'a, F: FnMut(SemanticEvent) + 'a>(
        &self,
        prepared: PreparedChatSemantics,
        maximum: usize,
        maximum_draft: usize,
        temperature: f32,
        eos: &[u32],
        callback: F,
    ) -> Result<
        (
            PreparedOriginalSpeculativeHost<'a>,
            crate::api::PreparedChatSpeculativeConstraint,
        ),
        PreparedChatSessionError,
    > {
        let PreparedChatSemantics {
            semantic,
            controller,
            preparation,
        } = prepared;
        let funding = preparation.metadata_funding().clone();
        let result = (|| -> Result<_, Cause> {
            reserve(
                &funding,
                sum(&[
                    size_of::<PreparedOriginalSpeculativeHost<'a>>(),
                    size_of::<crate::api::PreparedChatSpeculativeConstraint>(),
                    size_of::<F>(),
                    size_of::<Result<SpeculativeConfiguration, OriginalSpeculativeHostError>>(),
                    size_of::<Result<SpeculativeEventCallback<'a>, SpeculativeOutputError>>(),
                    size_of::<
                        Result<
                            (
                                PreparedOriginalSpeculativeHost<'a>,
                                crate::api::PreparedChatSpeculativeConstraint,
                            ),
                            PreparedChatSessionError,
                        >,
                    >(),
                ]),
            )?;
            let configuration =
                preparation.prepare_configuration(maximum, maximum_draft, temperature, eos)?;
            let callback = preparation.prepare_callback(callback)?;
            Ok((
                PreparedOriginalSpeculativeHost {
                    semantic,
                    configuration,
                    callback,
                    preparation,
                },
                crate::api::PreparedChatSpeculativeConstraint::new(controller),
            ))
        })();
        result.map_err(|cause| PreparedChatSessionError {
            cause,
            funding: Some(funding),
            input_funding: None,
        })
    }
}

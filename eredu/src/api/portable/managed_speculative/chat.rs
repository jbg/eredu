//! Source-explicit prepared chat through the existing speculative drivers.
use super::*;
use crate::api::portable::prepared_semantic::PreparedSemanticInput;
use crate::{
    api::{PreparedChatGenerationSettings, PreparedChatPrompt},
    runtime::chat::PreparedChat,
};
type InputFunding = eredu_core::SpeculativeBuffer<eredu_runtime::input::OriginalModelInputCustody>;
use eredu_runtime::working_memory::OriginalChatBackend;

/// One caller-owned prepared chat and original target/draft request. The actual
/// rendered prompt is encoded by the original tokenizer worker; immutable
/// historical protocol/controller declarations are copied before execution.
pub struct PreparedChatSpeculativeRequest<'a, P, D, F> {
    /// Actual prepared prompt and its retained protocol/constraint recipe.
    pub chat: &'a PreparedChat,
    /// Rendered text, exact token IDs, or an authenticated media prompt.
    pub input: PreparedChatPrompt<'a, P>,
    /// Selected assistant, consumed by the shared speculative backend adapter.
    pub drafting: SpeculativeDraft<'a, D>,
    /// Sampling/token controls, including the required managed-memory ceiling.
    pub settings: PreparedChatGenerationSettings,
    /// Semantic channels or explicitly permitted literal text, shared with ordinary generation.
    pub output_mode: crate::api::PreparedChatOutputMode,
    /// Omit special-token spellings in the shared decoder.
    pub skip_special_tokens: bool,
    /// Existing proposal width and scheduler configuration.
    pub options: PreparedChatSpeculativeGenerationOptions,
    /// Literal caller stops combined with the actual profile's literal stops.
    pub caller_stop_sequences: &'a [String],
    /// Shared cooperative cancellation token.
    pub cancellation: GenerationCancellationToken,
    /// Receives only committed semantic events through the existing publisher.
    pub on_event: F,
}
impl<P, D, F> std::fmt::Debug for PreparedChatSpeculativeRequest<'_, P, D, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedChatSpeculativeRequest")
            .finish_non_exhaustive()
    }
}
/// Neutral source/preparation/execution failure retaining the actual host payer.
/// Native failures remain nested under the existing BackendFailure boundary.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct PreparedChatSpeculativeError {
    #[source]
    cause: Cause,
    funding: Option<HostMetadataFunding>,
    input_funding: Option<InputFunding>,
}
impl PreparedChatSpeculativeError {
    pub(crate) fn from_generation(cause: eredu_core::GenerationError) -> Self {
        Self {
            cause: cause.into(),
            funding: None,
            input_funding: None,
        }
    }
    pub(crate) fn from_capture(cause: eredu_core::capture::CaptureError) -> Self {
        Self {
            cause: cause.into(),
            funding: None,
            input_funding: None,
        }
    }
    /// A control action failed with this neutral policy or backend error.
    pub fn control_failure(&self) -> Option<&SpeculativeControlError> {
        match &self.cause {
            Cause::Control(cause) => Some(cause),
            _ => None,
        }
    }

    /// The retained template has no recognized semantic output protocol.
    pub fn semantic_output_rejection(&self) -> Option<crate::runtime::chat::SemanticSupport> {
        match &self.cause {
            Cause::Preparation(error) => error.semantic_output_rejection(),
            _ => None,
        }
    }
    /// Retained declaration policy rejected explicit literal text output.
    pub fn text_output_rejection(&self) -> Option<crate::runtime::chat::CapabilitySupport> {
        match &self.cause {
            Cause::Preparation(error) => error.text_output_rejection(),
            _ => None,
        }
    }
    /// Expected and returned lane counts when a backend violates single-lane delivery.
    pub fn output_cardinality(&self) -> Option<(usize, usize)> {
        match self.cause {
            Cause::Cardinality(actual) => Some((1, actual)),
            _ => None,
        }
    }
    /// Fixed source or missing original-policy rejection, when applicable.
    pub fn input_rejection(&self) -> Option<TokenInputRejection> {
        match &self.cause {
            Cause::Input(error) => Some(*error),
            Cause::Preparation(error) => error.input_rejection(),
            _ => None,
        }
    }
    /// Selected backend failure with the original retained source chain.
    pub fn backend_failure(&self) -> Option<&BackendFailure> {
        match &self.cause {
            Cause::Backend(error) | Cause::Control(SpeculativeControlError::Backend(error)) => Some(error),
            Cause::Preparation(error) => error.backend_failure(),
            _ => None,
        }
    }
}
impl<B: OriginalChatBackend + SpeculativeGenerationBackend> LoadedModel<B> {
    /// Generates semantic output from the actual prepared chat and original
    /// generation tokenizer. Active grammars, forbidden triggers and automatic
    /// activation use the same prepared controller as ordinary generation.
    pub fn generate_prepared_chat_speculative<'a, F: FnMut(SemanticEvent) + 'a>(
        &mut self,
        request: PreparedChatSpeculativeRequest<'a, B::Prompt, B::Drafter, F>,
    ) -> Result<SpeculativeGenerationOutput, PreparedChatSpeculativeError> {
        let driver = eredu_runtime::RunSpeculativeGeneration::new(request.options.scheduler);
        let mut funding = None;
        let mut input_funding = None;
        self.run_managed_prepared_chat_speculative(
            request,
            driver,
            &mut funding,
            &mut input_funding,
        )
        .map_err(|cause| PreparedChatSpeculativeError {
            cause,
            funding,
            input_funding,
        })
    }
    /// Lends the same source-bound request to the ordinary controlled driver.
    /// Commitment, cancellation, snapshots, restore and fork use the same semantic
    /// state and independently copied mutable destinations as uninterrupted runs.
    pub fn with_controlled_prepared_chat_speculative<'a, F, D>(
        &mut self,
        request: PreparedChatSpeculativeRequest<'a, B::Prompt, B::Drafter, F>,
        options: ControlledSpeculativeOptions,
        drive: D,
    ) -> Result<SpeculativeGenerationOutput, PreparedChatSpeculativeError>
    where
        F: FnMut(SemanticEvent) + 'a,
        D: FnOnce(&mut dyn ControlledSpeculativeSession) -> Result<(), SpeculativeControlError>,
    {
        if !request
            .chat
            .tokenizer_source()
            .matches_configuration(&self.tokenizer)
        {
            return Err(PreparedChatSpeculativeError {
                cause: TokenInputRejection::IdentityMismatch.into(),
                funding: None,
                input_funding: None,
            });
        }
        self.validate_controlled_speculative_options(request.settings, &options)
            .map_err(|cause| PreparedChatSpeculativeError {
                cause,
                funding: None,
                input_funding: None,
            })?;
        let vocabulary = request
            .chat
            .tokenizer_source()
            .ids()
            .max()
            .map_or(Some(0), |id| usize::try_from(id).ok()?.checked_add(1))
            .ok_or_else(|| PreparedChatSpeculativeError {
                cause: HostMetadataFundingError::Overflow.into(),
                funding: None,
                input_funding: None,
            })?;
        let mut failure = None;
        let driver = DriveControlledSpeculation::new(
            request.options.scheduler,
            options,
            drive,
            &mut failure,
        )
        .with_vocabulary(vocabulary);
        let mut funding = None;
        let mut input_funding = None;
        let result = self.run_managed_prepared_chat_speculative(
            request,
            driver,
            &mut funding,
            &mut input_funding,
        );
        if let Some(cause) = failure {
            return Err(PreparedChatSpeculativeError {
                cause: cause.into(),
                funding,
                input_funding,
            });
        }
        result.map_err(|cause| PreparedChatSpeculativeError {
            cause,
            funding,
            input_funding,
        })
    }
    /// Delivers each speculative step from the shared controlled session.
    pub fn generate_observed_prepared_chat_speculative<'a, F, D>(
        &mut self,
        request: PreparedChatSpeculativeRequest<'a, B::Prompt, B::Drafter, F>,
        options: ControlledSpeculativeOptions,
        on_step: D,
    ) -> Result<SpeculativeGenerationOutput, PreparedChatSpeculativeError>
    where
        F: FnMut(SemanticEvent) + 'a,
        D: FnMut(
            eredu_runtime::speculative::ControlledSpeculativeStep,
        ) -> std::ops::ControlFlow<()>,
    {
        self.with_controlled_prepared_chat_speculative(
            request,
            options,
            crate::api::controlled_speculative::continuous_steps(on_step),
        )
    }

    fn run_managed_prepared_chat_speculative<'a, F, V>(
        &mut self,
        request: PreparedChatSpeculativeRequest<'a, B::Prompt, B::Drafter, F>,
        driver: V,
        retained: &mut Option<HostMetadataFunding>,
        input_funding: &mut Option<InputFunding>,
    ) -> Result<SpeculativeGenerationOutput, Cause>
    where
        F: FnMut(SemanticEvent) + 'a,
        V: SpeculativeGenerationVisitor,
    {
        let request = self.prepare_managed_prepared_chat_speculative_request(
            request,
            retained,
            input_funding,
        )?;
        let funding = retained.as_ref().expect("prepared host account");
        reserve(
            funding,
            Some(
                size_of::<PreparedChatSpeculativeError>()
                    + size_of::<Result<SpeculativeGenerationOutput, PreparedChatSpeculativeError>>(
                    ),
            ),
        )?;
        self.execute_managed_speculative_request(request, driver, retained)
    }
    #[inline(never)]
    fn prepare_managed_prepared_chat_speculative_request<'a, F>(
        &self,
        request: PreparedChatSpeculativeRequest<'a, B::Prompt, B::Drafter, F>,
        retained: &mut Option<HostMetadataFunding>,
        input_funding: &mut Option<InputFunding>,
    ) -> Result<
        eredu_core::SpeculativeGenerationBatchRequest<
            'a,
            B,
            B::Drafter,
            PreparedChatSpeculativeConstraint,
        >,
        Cause,
    >
    where
        F: FnMut(SemanticEvent) + 'a,
    {
        let PreparedChatSpeculativeRequest {
            chat,
            input,
            drafting,
            settings,
            output_mode,
            skip_special_tokens,
            options,
            caller_stop_sequences,
            cancellation,
            on_event,
        } = request;
        let request = self.prepare_speculative_batch(
            drafting,
            std::iter::once(PreparedChatSpeculativeBatchLane {
                chat,
                input,
                settings,
                output_mode,
                skip_special_tokens,
                max_draft_tokens: options.max_draft_tokens,
                caller_stop_sequences,
                cancellation,
                on_event,
            }),
            Ok(()),
            options.scheduler,
            retained,
            |lane, retained| self.prepare_speculative_chat_lane(lane, retained, input_funding, 1),
        )?;
        Ok(request.expect("single prepared chat lane"))
    }

    fn prepare_speculative_chat_lane<'a, F: FnMut(SemanticEvent) + 'a>(
        &self,
        lane: PreparedChatSpeculativeBatchLane<'a, B::Prompt, F>,
        retained: &mut Option<HostMetadataFunding>,
        input_funding: &mut Option<InputFunding>,
        lane_count: usize,
    ) -> Result<preparation::PreparedLane<'a, B>, Cause> {
        let PreparedChatSpeculativeBatchLane {
            chat,
            input,
            settings,
            output_mode,
            skip_special_tokens,
            max_draft_tokens,
            caller_stop_sequences,
            cancellation,
            on_event,
        } = lane;
        let event_window = max_draft_tokens
            .get()
            .checked_add(1)
            .and_then(std::num::NonZeroUsize::new)
            .ok_or(HostMetadataFundingError::Overflow)?;
        let invocation = self.prepare_chat_invocation(
            chat,
            input,
            settings,
            event_window,
            caller_stop_sequences,
            skip_special_tokens,
            output_mode,
        )?;
        let generation = invocation.config;
        let maximum = generation
            .sampling()
            .max_new_tokens
            .expect("resolved finite maximum");
        let funding = invocation.prepared.preparation.metadata_funding().clone();
        if retained.is_none() {
            *retained = Some(funding.clone());
        }
        let input = match invocation.input {
            PreparedSemanticInput::Text(text) => preparation::PreparedPrompt::Text {
                text,
                add_special_tokens: false,
            },
            PreparedSemanticInput::TokenIds(ids) => preparation::PreparedPrompt::TokenIds(ids),
            PreparedSemanticInput::Media { input, .. } => {
                if input_funding.is_none() {
                    *input_funding = Some(preparation::buffer(lane_count, &funding)?);
                }
                let (prompt, custody) = input.into_parts();
                input_funding
                    .as_mut()
                    .expect("prepared input custody")
                    .try_push(custody)?;
                preparation::PreparedPrompt::Media(prompt)
            }
        };
        reserve(
            &funding,
            sum(&[
                size_of::<PreparedChatSpeculativeBatchLane<'a, B::Prompt, F>>(),
                size_of::<preparation::PreparedLane<'a, B>>(),
                size_of::<Result<preparation::PreparedLane<'a, B>, Cause>>(),
                size_of::<Cause>(),
                size_of::<PreparedChatSpeculativeError>(),
                size_of::<Option<HostMetadataFunding>>(),
                size_of::<Option<InputFunding>>(),
                size_of::<Option<SpeculativeControlError>>(),
            ])
            .and_then(|n| {
                n.checked_add(BackendFailure::source_retention_peak_bytes::<B::Error>()?)
            }),
        )?;
        let (host, constraint) = self.prepare_original_speculative_chat_host(
            invocation.prepared,
            maximum,
            max_draft_tokens.get(),
            generation.sampling().temperature,
            chat.eos_token_ids(),
            on_event,
        )?;
        Ok(preparation::PreparedLane {
            host,
            input,
            prefill_chunk_positions: settings.inference.prefill_chunk_positions,
            generation,
            constraint,
            cancellation,
            funding,
        })
    }

    /// Runs independent prepared chats through the same source-funded host and
    /// prompt stages as a single request, then the existing fair scheduler.
    pub fn generate_prepared_chat_speculative_batch<'a, F: FnMut(SemanticEvent) + 'a>(
        &mut self,
        request: PreparedChatSpeculativeBatchRequest<'a, B::Prompt, B::Drafter, F>,
    ) -> Result<eredu_core::SpeculativeGenerationBatchOutput, PreparedChatSpeculativeError> {
        let PreparedChatSpeculativeBatchRequest {
            drafting,
            lanes,
            scheduler,
        } = request;
        let mut funding = None;
        let mut input_funding = None;
        let count = lanes.len();
        let validation = lanes
            .iter()
            .try_for_each(|lane| {
                self.validate_chat_invocation(lane.chat, &lane.input, lane.settings)
                    .map(|_| ())
                    .map_err(Cause::from)
            });
        let result = self
            .prepare_speculative_batch(
                drafting,
                lanes.into_iter(),
                validation,
                scheduler,
                &mut funding,
                |lane, retained| {
                    self.prepare_speculative_chat_lane(lane, retained, &mut input_funding, count)
                },
            )
            .and_then(|request| self.execute_speculative_batch(request, scheduler, &funding));
        result.map_err(|cause| PreparedChatSpeculativeError {
            cause,
            funding,
            input_funding,
        })
    }
}

/// One prepared semantic chat with independent sampling and committed delivery.
pub struct PreparedChatSpeculativeBatchLane<'a, P, F> {
    /// The actual source-bound rendered chat and tool policy.
    pub chat: &'a PreparedChat,
    /// Rendered text, exact token IDs, or an authenticated media prompt.
    pub input: PreparedChatPrompt<'a, P>,
    /// Sampling, seed and required memory ceiling for this lane.
    pub settings: PreparedChatGenerationSettings,
    /// Semantic channels or explicitly permitted literal text, shared with ordinary generation.
    pub output_mode: crate::api::PreparedChatOutputMode,
    /// Omit special-token spellings in the shared decoder.
    pub skip_special_tokens: bool,
    /// Maximum assistant proposals verified together for this lane.
    pub max_draft_tokens: std::num::NonZeroUsize,
    /// Literal caller stops combined with this chat's profile stops.
    pub caller_stop_sequences: &'a [String],
    /// Cancellation belongs to this lane alone.
    pub cancellation: GenerationCancellationToken,
    /// Receives only this lane's committed semantic events.
    pub on_event: F,
}
/// Prepared semantic chats sharing one assistant and fair scheduler.
pub struct PreparedChatSpeculativeBatchRequest<'a, P, D, F> {
    /// Actual embedded or external assistant selection.
    pub drafting: SpeculativeDraft<'a, D>,
    /// Application-owned lane descriptions in stable output order.
    pub lanes: Vec<PreparedChatSpeculativeBatchLane<'a, P, F>>,
    /// Existing scheduler fairness and lookahead policy.
    pub scheduler: eredu_core::SpeculativeSchedulerOptions,
}

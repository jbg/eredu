//! Ordinary generation consumes the shared prepared semantic source and cursor.
mod snapshot;
mod branch;
pub use branch::PreparedChatBranch;
use super::*;
use crate::api::{ConstraintError, request::{BackendGenerationTokenSource, ScopedBackendGenerationTokenSource}};
use crate::runtime::generation::streaming::RetainedConsumerCursor;
use eredu_core::{
    BackendFailure, ControlledTextGeneration, ControlledTextGenerationError,
    GenerationSequenceRequest, GenerationTiming, GenerationTokenIds, ModelRuntime,
    TextGenerationInput, TokenIdsInputPlan, TokenInputRejection,
};
use eredu_runtime::{input::OriginalModelInput, working_memory::OriginalChatBackend};
pub use snapshot::{PreparedChatResumeSettings, PreparedChatSnapshot};
use std::time::{Duration, Instant};

/// Output interpretation over the same committed-token session.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum PreparedChatOutputMode {
    /// Parse the prepared protocol's text, reasoning and tool channels.
    #[default]
    Semantic,
    /// Emit decoded text with caller stops, without protocol parsing or stops.
    /// The prepared chat's text-generation capability must permit this policy.
    Text,
}

/// One ordinary semantic request for text or a completed authenticated media input.
/// The prepared chat supplies tokenizer, template and controller provenance.
pub enum PreparedChatPrompt<'a, P> {
    /// Encode the exact prepared chat render through its retained tokenizer.
    Rendered,
    /// Use an explicit canonical prefix with the prepared chat's output policy.
    /// Original admission copies and validates these IDs without text round trips.
    TokenIds(&'a [u32]),
    /// Completed media whose provenance authenticates this exact chat render.
    Media(OriginalModelInput<P>),
}

/// Ordinary input and output policy over the shared prepared-chat driver.
pub struct PreparedChatRequest<'a, P> {
    /// Originally prepared prompt and policy.
    pub chat: &'a crate::runtime::chat::PreparedChat,
    /// Ordinary sampling, prefill and managed-memory policy.
    pub settings: crate::api::PreparedChatGenerationSettings,
    /// Additional literal termination sequences.
    pub stop_sequences: &'a [String],
    /// Omit special-token spellings from visible decoded text.
    pub skip_special_tokens: bool,
    /// Select semantic interpretation or explicitly permitted literal text.
    pub output_mode: PreparedChatOutputMode,
    /// Rendered text, an explicit canonical prefix, or authenticated media.
    pub input: PreparedChatPrompt<'a, P>,
    /// Admitted capture and intervention sources for this exact execution.
    /// These use the backend's ordinary preparation and committed delivery.
    pub options: Option<eredu_core::TextPreparationOptions>,
    /// Borrowed capture declaration, admitted after the paid prompt geometry is
    /// known. Cannot be combined with an already admitted capture in `options`.
    pub capture: Option<&'a eredu_core::capture::CapturePlan>,
    /// Borrowed static edits, compiled against the same prompt/capture geometry
    /// and logical session. Cannot overlap `options.interventions`.
    pub intervention: Option<&'a eredu_core::intervention::InterventionPlan>,
}
impl<'a, P> PreparedChatRequest<'a, P> {
    /// Select the rendered text prompt. Media may be supplied in `input`.
    pub fn new(
        chat: &'a crate::runtime::chat::PreparedChat,
        settings: crate::api::PreparedChatGenerationSettings,
    ) -> Self {
        Self {
            chat,
            settings,
            stop_sequences: &[],
            skip_special_tokens: true,
            output_mode: PreparedChatOutputMode::Semantic,
            input: PreparedChatPrompt::Rendered,
            options: None,
            capture: None,
            intervention: None,
        }
    }
}

impl<B: OriginalChatBackend> LoadedModel<B> {
    pub(crate) fn validate_prepared_chat(
        &self,
        chat: &crate::runtime::chat::PreparedChat,
    ) -> Result<(), PreparedChatSessionError> {
        if !chat
            .tokenizer_source()
            .matches_configuration(&self.tokenizer)
            || !self.chat_template.as_ref().is_some_and(|template| {
                chat.render()
                    .template_source()
                    .matches_configuration(template, &self.model_id)
            })
        {
            return Err(PreparedChatSessionError::before(
                TokenInputRejection::IdentityMismatch,
            ));
        }
        B::validate_original_chat_render(&self.runtime, chat.render())
            .map_err(PreparedChatSessionError::before)
    }
    /// Start ordinary semantic generation through the shared committed-token
    /// cursor. Manual `advance` and uninterrupted `run` share that cursor; no
    /// speculative backend capability or drafter is required.
    pub fn start_prepared_chat<'a>(
        &'a mut self,
        request: PreparedChatRequest<'_, B::Prompt>,
        cancellation: &GenerationCancellationToken,
    ) -> Result<Option<PreparedChatSession<'a, B>>, PreparedChatSessionError> {
        if cancellation.is_cancelled() {
            return Ok(None);
        }
        let PreparedChatRequest {
            chat,
            settings,
            stop_sequences,
            skip_special_tokens,
            output_mode,
            input,
            options, capture, intervention,
        } = request;
        let invocation = self.prepare_chat_invocation(
            chat, input, settings, std::num::NonZeroUsize::MIN,
            stop_sequences, skip_special_tokens, output_mode,
        )?;
        PreparedChatSession::start(
            &mut self.runtime,
            invocation.prepared,
            invocation.input,
            invocation.config,
            chat.eos_token_ids(),
            Instrumentation { options, capture, intervention, session_id: &self.session_identity },
            cancellation,
        )
    }
}

impl<B: OriginalChatBackend + eredu_runtime::input::OriginalModelInputBackend> LoadedModel<B> {
    /// Compile host media/text parts against this exact rendered chat. The
    /// architecture's retained semantic projection authenticates every marker
    /// and decoder coordinate. The returned input is consumed by the same
    /// `start_prepared_chat` request as text and shares its semantic preparation.
    pub fn prepare_chat_input(
        &self,
        chat: &crate::runtime::chat::PreparedChat,
        parts: &[eredu_runtime::input::host::HostInputPart<'_>],
        cancellation: &GenerationCancellationToken,
    ) -> Result<Option<OriginalModelInput<B::Prompt>>, PreparedChatSessionError> {
        if cancellation.is_cancelled() {
            return Ok(None);
        }
        self.validate_prepared_chat(chat)?;
        let preparation = self.prepare_semantic_source(chat.tokenizer_source(), chat.capacity())?;
        let funding = preparation.metadata_funding().clone();
        let result = (|| -> Result<_, Cause> {
            let controls = [size_of::<(&Self, &crate::runtime::chat::PreparedChat,
                    &[eredu_runtime::input::host::HostInputPart<'_>], &GenerationCancellationToken)>(),
                size_of::<OriginalModelInput<B::Prompt>>(),
                size_of::<Result<Option<OriginalModelInput<B::Prompt>>, PreparedChatSessionError>>(),
                size_of::<Result<OriginalModelInput<B::Prompt>, eredu_runtime::input::PreparedChatInputError>>(),
                eredu_runtime::working_memory::OriginalControllerCompilation::validation_control_bytes()
                    .ok_or(eredu_core::TokenInputRejection::Overflow)?];
            reserve(&funding, sum(&controls))?;
            chat.compilation()
                .validate_preparation(chat.controller_sources(), &preparation)
                .map_err(chat::ChatCause::from)?;
            let input = eredu_runtime::input::prepare_original_chat_model_input::<B>(
                &self.runtime,
                &preparation,
                chat.render(),
                chat.generation(),
                parts,
            )?;
            if cancellation.is_cancelled() {
                return Ok(None);
            }
            Ok(Some(input))
        })();
        result.map_err(|cause| PreparedChatSessionError {
            cause,
            funding: Some(funding),
            input_funding: None,
        })
    }
}

fn startup_failure<B:TextGenerationBackend>(error:ControlledTextGenerationError<B::Error,ChatControllerError>)->Cause {
    match error {
        ControlledTextGenerationError::Backend(error)=>Cause::Backend(B::into_backend_failure(error)),
        ControlledTextGenerationError::Preparation(error)=>Cause::Backend(error),
        ControlledTextGenerationError::Controller(error)=>Cause::Choice(error),
    }
}

type ChatController = eredu_runtime::execution_control::TokenChoiceController<crate::runtime::chat::constraints::ConstraintController>;
pub(super) type ChatControllerError = eredu_runtime::execution_control::TokenChoiceError<ConstraintError>;
type SourceError = ControlledTextGenerationError<BackendFailure, ChatControllerError>;
type Cursor = RetainedConsumerCursor<SourceError, SpeculativeOutputError, true>;
pub(super) type CursorFailure = crate::runtime::generation::streaming::RetainedCursorFailure<SourceError, SpeculativeOutputError, true>;

/// The prompt mechanism changes; controller, decoder and commitment do not.
pub(crate) enum PreparedSemanticInput<'a, B: TextGenerationBackend> {
    Text(&'a str),
    TokenIds(&'a [u32]),
    Media {
        input: OriginalModelInput<B::Prompt>,
        render: eredu_runtime::working_memory::OriginalRenderedChat,
        generation_prompt: bool,
    },
}

pub(crate) struct Instrumentation<'a> {
    options: Option<eredu_core::TextPreparationOptions>,
    capture: Option<&'a eredu_core::capture::CapturePlan>,
    intervention: Option<&'a eredu_core::intervention::InterventionPlan>,
    session_id: &'a str,
}
impl Instrumentation<'_> {
    fn has_declarations(&self) -> bool { self.capture.is_some() || self.intervention.is_some() }
    fn compile<B: OriginalChatBackend>(self, runtime: &ModelRuntime<B>, positions: usize,
        maximum: usize, funding: &HostMetadataFunding)
        -> Result<Option<eredu_core::TextPreparationOptions>, Cause> {
        if !self.has_declarations() { return Ok(self.options); }
        reserve(funding, sum(&[size_of::<Self>(), size_of::<eredu_core::TextPreparationOptions>(),
            size_of::<eredu_core::capture::CapturePlan>(),
            size_of::<eredu_runtime::working_memory::OriginalCaptureSource>(),
            size_of::<eredu_runtime::working_memory::OriginalCaptureSourceError>(),
            size_of::<eredu_runtime::working_memory::OriginalInterventionSource>(),
            size_of::<eredu_core::BackendFailure>()]))?;
        let mut options = self.options.unwrap_or_default();
        if (self.capture.is_some() && options.capture.is_some())
            || (self.intervention.is_some() && options.interventions.is_some()) {
            return Err(TokenInputRejection::IdentityMismatch.into());
        }
        if self.capture.is_some() || options.capture.is_none() {
            let empty = eredu_core::capture::CapturePlan::none();
            let raw = self.capture.unwrap_or(&empty);
            let request = eredu_core::capture::CaptureRequestShape { batch: 1,
                prompt_tokens: u64::try_from(positions).map_err(|_| TokenInputRejection::Overflow)?,
                max_predictions: u64::try_from(maximum).map_err(|_| TokenInputRejection::Overflow)? };
            let source = B::compile_original_capture_declaration(runtime, raw, request, funding)?;
            options.capture = Some(source.plan().clone());
        }
        if let Some(raw) = self.intervention {
            let source = B::compile_original_intervention_declaration(runtime, raw,
                options.capture.as_ref().expect("compiled capture geometry"), self.session_id, funding)?;
            options.interventions = Some(source.plan().clone());
        }
        Ok(Some(options))
    }
}

/// Manual advancement of an ordinary source-bound semantic generation.
/// Each successful advancement publishes committed events exactly once. The
/// session retains its controller, decoder, input and original funding.
pub struct PreparedChatSession<'a, B: TextGenerationBackend> {
    source: BackendGenerationTokenSource<'a, B, ChatController>,
    semantic: SemanticStateOwner,
    effective_config: TextGenerationConfig,
    active: Duration,
    cursor: Cursor,
    lifecycle: eredu_runtime::execution_control::GenerationLifecycle,
    attribution: Option<eredu_core::SharedPromptAttribution>,
    input_funding: Option<eredu_runtime::input::OriginalModelInputCustody>,
    preparation: PreparedSemanticSource,
}

impl<'a, B: OriginalChatBackend> PreparedChatSession<'a, B> {
    pub(crate) fn consumer_layout() -> Option<eredu_core::GenerationSequenceConsumerLayout> {
        Cursor::layout()
    }
    /// Starts ordinary autoregressive generation; no speculative backend or
    /// scheduler is involved. Every native input is authenticated by core.
    pub(crate) fn start(
        runtime: &'a mut ModelRuntime<B>,
        prepared: chat::PreparedChatSemantics,
        input: PreparedSemanticInput<'_, B>,
        config: TextGenerationConfig,
        eos: &[u32],
        instrumentation: Instrumentation<'_>,
        cancellation: &GenerationCancellationToken,
    ) -> Result<Option<Self>, PreparedChatSessionError> {
        if cancellation.is_cancelled() {
            return Ok(None);
        }
        let started = Instant::now();
        let chat::PreparedChatSemantics {
            semantic,
            controller,
            preparation,
        } = prepared;
        let funding = preparation.metadata_funding().clone();
        let mut input_funding = None;
        let result = (|| -> Result<_, Cause> {
            B::validate_semantic_source(runtime, &preparation)?;
            let maximum = config
                .sampling()
                .max_new_tokens
                .ok_or(TokenInputRejection::Unsupported)?;
            let layout = Self::consumer_layout().ok_or(TokenInputRejection::Overflow)?;
            reserve(
                &funding,
                sum(&[
                    size_of::<Self>(),
                    size_of::<ScopedBackendGenerationTokenSource<'_, '_, '_, B, ChatController>>(),
                    size_of::<crate::api::request::TokenDeliveryFacts>(),
                    size_of::<eredu_core::TextTokenChoiceBoundary<'_, ChatController>>(),
                    size_of::<eredu_core::TextSamplingBoundary<'_, B>>(),
                    size_of::<eredu_core::execution_control::SamplingOverride>(),
                    size_of::<eredu_core::execution_control::SamplingStateFacts>(),
                    size_of::<(TextGenerationConfig, eredu_core::ResolvedGenerationConfig, u64, eredu_core::TextInferencePolicy)>(),
                    size_of::<Result<eredu_core::execution_control::SamplingStateFacts,
                        eredu_core::execution_control::SamplingOverrideError<B::Error>>>(),
                    size_of::<PreparedSemanticInput<'_, B>>(),
                size_of::<ControlledTextGenerationError<B::Error,ChatControllerError>>(),
                    size_of::<Result<Option<Self>, PreparedChatSessionError>>(),
                    size_of::<
                        Result<
                            ControlledTextGeneration<
                                '_,
                                B,
                                ChatController,
                            >,
                            ControlledTextGenerationError<B::Error, ChatControllerError>,
                        >,
                    >(),
                    size_of::<Option<eredu_runtime::working_memory::OriginalEncodedTokenIds>>(),
                    size_of::<
                        Result<
                            eredu_runtime::working_memory::OriginalEncodedTokenIds,
                            OriginalTextSourceError,
                        >,
                    >(),
                    size_of::<GenerationSequenceRequest<'_>>(),
                    size_of::<Instrumentation<'_>>(),
                    size_of::<Option<eredu_core::TextPreparationOptions>>(),
                    size_of::<Option<eredu_runtime::input::OriginalModelInputCustody>>(),
                    size_of::<Cause>(),
                    size_of::<PreparedChatSessionError>(),
                ])
                .and_then(|n| {
                    n.checked_add(BackendFailure::source_retention_peak_bytes::<SourceError>()?)
                }),
            )?;
            let domain = preparation.tokenizer().generation_domain()
                .and_then(eredu_core::TokenFilter::allowed_mask)
                .ok_or(TokenInputRejection::IdentityMismatch)?.len();
            let controller = ChatController::new(controller, eredu_runtime::TokenDomain::new(domain));
            let request = GenerationSequenceRequest::new(maximum, eos)
                .with_consumer(&layout)
                .with_semantic_state(&semantic);
            let mut attribution = None;
            let generator = match input {
                PreparedSemanticInput::TokenIds(ids) => {
                    let ids = TokenIdsInputPlan::new(ids)?;
                    attribution = Some(crate::api::control::records::PromptRecord::from_tokens(ids.tokens(), &funding)?.prepared().clone());
                    let options = instrumentation.compile::<B>(runtime, ids.tokens().len(), maximum, &funding)?;
                    ControlledTextGeneration::from_token_ids_with_sequence(runtime, ids,
                        config, controller, options, request).map_err(startup_failure::<B>)?
                }
                PreparedSemanticInput::Text(text) => {
                    let encoded =
                        B::encode_original_text_ids(runtime, preparation.tokenizer(), text, false)?;
                    if !encoded.matches_source(preparation.tokenizer()) {
                        return Err(TokenInputRejection::IdentityMismatch.into());
                    }
                    if cancellation.is_cancelled() {
                        return Ok(None);
                    }
                    attribution = Some(crate::api::control::records::PromptRecord::from_tokens(encoded.ids(), &funding)?.prepared().clone());
                    let ids = TokenIdsInputPlan::new(encoded.ids())?;
                    let options = instrumentation.compile::<B>(runtime, ids.tokens().len(), maximum, &funding)?;
                    let generator = ControlledTextGeneration::from_token_ids_with_sequence(
                        runtime, ids, config, controller, options, request,
                    )
                    .map_err(startup_failure::<B>)?;
                    generator
                }
                PreparedSemanticInput::Media {
                    input,
                    render,
                    generation_prompt,
                } => {
                    if !input.chat_binding().is_some_and(|binding| {
                        binding.matches(&render, &preparation, generation_prompt)
                    }) {
                        let (_, retained) = input.into_parts();
                        input_funding = Some(retained);
                        return Err(TokenInputRejection::IdentityMismatch.into());
                    }
                    if let Some(binding) = input.chat_binding().filter(|binding| binding.semantics().is_some()) {
                        attribution = Some(crate::api::control::records::PromptRecord::from_media(binding, &funding)?.prepared().clone());
                    }
                    let positions = if instrumentation.has_declarations() {
                        Some(input.chat_binding().and_then(|binding| binding.semantics())
                            .ok_or(TokenInputRejection::Unsupported)?.layout().positions())
                    } else { None };
                    let (prompt, retained) = input.into_parts();
                    input_funding = Some(retained);
                    B::validate_original_prepared_input_domain(
                        runtime,
                        &prompt,
                        preparation.tokenizer(),
                    )?;
                    let options = instrumentation.compile::<B>(runtime, positions.unwrap_or(0), maximum, &funding)?;
                    let generator = ControlledTextGeneration::from_input_with_sequence(
                        runtime,
                        TextGenerationInput::OriginalPrepared(prompt),
                        config,
                        controller,
                        options,
                        request,
                    )
                    .map_err(startup_failure::<B>)?;
                    generator
                }
            };
            let mut generator = generator;
            let sequence = generator
                .take_prepared_sequence()
                .expect("original request requires sequence storage");
            let cursor = Cursor::from_sequence(sequence)
                .unwrap_or_else(|_| unreachable!("core checked exact semantic consumer layout"));
            Ok(Some(Self {
                source: BackendGenerationTokenSource {
                    generator,
                    on_token: None,
                    delivery_failure: None,
                    generation_started: started,
                    time_to_first_token: None,
                },
                semantic,
                effective_config: config,
                active: started.elapsed(),
                cursor,
                lifecycle: eredu_runtime::execution_control::GenerationLifecycle::default(),
                attribution,
                input_funding: input_funding.take(),
                preparation,
            }))
        })();
        result.map_err(|cause| PreparedChatSessionError {
            cause,
            funding: Some(funding),
            input_funding,
        })
    }
}

impl<'a, B: TextGenerationBackend> PreparedChatSession<'a, B> {
    /// Lends the shared committed-token source a capture observer. Completed
    /// frames retain their original storage custody and arrive before the
    /// corresponding semantic events. A failed or cancelled step may deliver
    /// a frame without a committed token.
    ///
    /// This changes delivery only. Capture plans and their cumulative budgets
    /// are supplied in the request; the callback cannot install or reset them.
    pub fn with_capture_observer(
        mut self,
        observer: &'a mut dyn FnMut(
            Option<u32>,
            Option<eredu_core::capture::SharedCapturedStep>,
            f64,
        ),
    ) -> Self {
        self.source.on_token = Some(observer);
        self
    }
}

impl<B: TextGenerationBackend> PreparedChatSession<'_, B> {
    pub(crate) fn semantic_snapshot_bytes(&self) -> Option<u64> {
        let prepared = self.semantic.prepared_source()?.downcast_ref::<eredu_runtime::working_memory::PreparedSemanticState>()?;
        u64::try_from(prepared.copy_bytes()?).ok()
    }
    pub(crate) fn sampling_control_support(&self) -> eredu_core::execution_control::ControlSupport<&'static str> {
        B::text_sampling_control_support(self.source.generator.runtime())
    }
    pub(crate) fn metadata_funding(&self) -> &HostMetadataFunding {
        self.preparation.metadata_funding()
    }
    pub(crate) fn effective_config(&self) -> TextGenerationConfig { self.effective_config }
    pub(crate) fn last_committed_was_forced(&self) -> bool {
        self.source.generator.controller().last_committed_was_forced()
    }
    /// Records outside advancement use the same retained readiness source and
    /// exact cancellation vote; a local failure is never replaced by agreement.
    pub(crate) fn finish_record_delivery<T, E>(
        &self, local: Result<Option<T>, E>, map_backend: impl FnOnce(BackendFailure) -> E,
    ) -> Result<Option<T>, E> {
        self.source.generator.finish_text_preparation_cancellable(
            eredu_core::run_preparation::TextPreparationStage::Delivery, local, map_backend,
        )
    }
    /// Borrows the selected request geometry and its historical admission.
    /// This is the actual retained preparation, not current pool usage or a
    /// measured high-water mark. Reading it creates no execution authority.
    pub fn preparation_report(&self) -> Option<eredu_core::TextPreparationReport<'_>> {
        self.source.generator.preparation_report()
    }
    /// Borrow the actual immutable capture source installed on this machine.
    pub fn capture_source(&self) -> Option<&eredu_core::capture::SharedCapturePlan> {
        self.source.generator.capture_source()
    }
    /// Borrow the actual immutable edit source, including a resumed revision.
    pub fn intervention_source(&self) -> Option<&eredu_core::intervention::SharedInterventionPlan> {
        self.source.generator.intervention_source()
    }
    /// Exact original source parts and decoder coordinates when supplied by the
    /// backend's source contract. Media positions are never relabeled token IDs.
    pub fn prompt_attribution(&self) -> Option<&eredu_core::SharedPromptAttribution> {
        self.attribution.as_ref()
    }
    /// Completed-token lifecycle shared by manual and uninterrupted advancement.
    pub fn status(&self) -> eredu_core::execution_control::GenerationStatus {
        self.lifecycle.status()
    }

    /// Absolute prediction following the committed prefix, including after restore.
    pub fn next_prediction(&self) -> u64 {
        self.lifecycle.next_prediction()
    }

    /// Active execution time through first commitment; user pauses are excluded.
    pub fn timing(&self) -> GenerationTiming {
        GenerationTiming::new(self.source.time_to_first_token)
    }

    /// Pauses a completed session without consuming model input or randomness.
    /// A later explicit advancement resumes this same state.
    pub fn pause(&mut self) -> Result<(), PreparedChatSessionError> {
        self.control_boundary()?;
        self.lifecycle.pause().map_err(|cause| self.control_failure(cause.into()))
    }

    fn control_boundary(&mut self) -> Result<eredu_core::TextTokenChoiceBoundary<'_, ChatController>, PreparedChatSessionError> {
        if self.finish_reason().is_some() {
            return Err(self.control_failure(Cause::Boundary(eredu_core::TextContinuationError::Failed)));
        }
        let funding = Some(self.preparation.metadata_funding().clone());
        let input_funding = self.input_funding.clone();
        self.source.generator.token_choice_boundary().map_err(|error| {
            let error = crate::api::control::map_continuation_failure(error, B::into_backend_failure);
            PreparedChatSessionError { cause: Cause::Boundary(error), funding, input_funding }
        })
    }

    fn control_failure(&self, cause: Cause) -> PreparedChatSessionError {
        PreparedChatSessionError {
            cause,
            funding: Some(self.preparation.metadata_funding().clone()),
            input_funding: self.input_funding.clone(),
        }
    }

    /// Restricts the next ordinary sampling decision to a permitted canonical ID.
    /// Validation uses the retained source and consumes no prediction or RNG draw.
    /// The next advancement still applies sampling, penalties and commitment once.
    pub fn force_next_token(&mut self, token: u32) -> Result<(), PreparedChatSessionError> {
        let funding = self.preparation.metadata_funding().clone();
        let result = self.control_boundary()?.force_next(token, &funding);
        result.map_err(|cause| self.control_failure(cause.into()))
    }

    /// Removes a pending restriction without changing committed history or RNG.
    pub fn clear_forced_token(&mut self) -> Result<bool, PreparedChatSessionError> {
        Ok(self.control_boundary()?.clear_forced())
    }

    /// Canonical choice waiting for the next ordinary commitment, if any.
    pub fn pending_forced_token(&self) -> Option<u32> {
        self.source.generator.controller().pending_forced()
    }

    /// Borrows the exact committed token prefix, excluding tentative state.
    pub fn token_ids(&self) -> &[u32] {
        self.cursor.token_ids()
    }
    /// The terminal reason after committed output has finished.
    pub fn finish_reason(&self) -> Option<eredu_core::FinishReason> {
        self.cursor.finish_reason()
    }

    /// Advances the shared ordinary cursor once and lends it the event consumer.
    /// Failure consumes the session and retains its charged failure resources.
    /// Advancing terminal state preserves it without publishing another event.
    pub fn advance(
        self,
        cancellation: &GenerationCancellationToken,
        emit: &mut impl FnMut(SemanticEvent),
    ) -> Result<Self, PreparedChatSessionError> {
        self.advance_with_delivery(cancellation, None, None, emit)
    }

    /// Lends per-step record delivery to the same cursor and capture drain.
    /// Borrowed delivery state retires at this boundary, before the next call.
    pub(crate) fn advance_with_delivery(
        self,
        cancellation: &GenerationCancellationToken,
        mut observer: Option<&mut dyn FnMut(Option<u32>, Option<eredu_core::capture::SharedCapturedStep>, crate::api::request::TokenDeliveryFacts)>,
        delivery_failure: Option<&dyn Fn() -> Option<eredu_core::capture::CaptureError>>,
        emit: &mut impl FnMut(SemanticEvent),
    ) -> Result<Self, PreparedChatSessionError> {
        if self.finish_reason().is_some() {
            return Ok(self);
        }
        let Self {
            mut source,
            mut semantic,
            effective_config,
            active,
            cursor,
            mut lifecycle,
            attribution,
            input_funding,
            preparation,
        } = self;
        let committed_before = cursor.token_ids().len();
        lifecycle.begin_prediction().map_err(|cause| PreparedChatSessionError {
            cause: cause.into(),
            funding: Some(preparation.metadata_funding().clone()),
            input_funding: input_funding.clone(),
        })?;
        let started = Instant::now();
        source.generation_started = started
            .checked_sub(active)
            .expect("active process intervals");
        let has_observer = observer.is_some();
        let mut delivery = |token, capture, step_seconds, timing, controller: &ChatController| {
            if let Some(callback) = observer.as_deref_mut() {
                callback(token, capture, crate::api::request::TokenDeliveryFacts {
                    step_seconds, timing,
                    forced: token.is_some() && controller.last_committed_was_forced(),
                });
            }
        };
        let result = cursor.advance_semantic(
            &mut source.scoped(has_observer.then_some(&mut delivery), delivery_failure),
            &mut semantic, cancellation, emit,
        );
        match result {
            Ok(cursor) => {
                let transition = if cursor.token_ids().len() == committed_before {
                    match cursor.finish_reason() {
                        Some(reason) => lifecycle.finish_without_prediction(reason).map_err(Cause::from),
                        None => Err(Cause::MissingProgress),
                    }
                } else {
                    lifecycle.complete_prediction(cursor.finish_reason()).map_err(Cause::from)
                };
                transition.map_err(|cause| PreparedChatSessionError {
                    cause: cause.into(),
                    funding: Some(preparation.metadata_funding().clone()),
                    input_funding: input_funding.clone(),
                })?;
                Ok(Self {
                source,
                semantic,
                effective_config,
                active: active + started.elapsed(),
                cursor,
                lifecycle,
                attribution,
                input_funding,
                preparation,
            })},
            Err(failure) => {
                let funding = preparation.metadata_funding().clone();
                let (kind,operation) = match failure.cause() {
                    crate::runtime::generation::streaming::CommittedGenerationError::Source(
                        ControlledTextGenerationError::Preparation(error)|ControlledTextGenerationError::Backend(error),
                    ) => (error.kind(),error.operation()),
                    _ => (eredu_core::BackendFailureKind::Other,"prepared chat advancement"),
                };
                drop(source);
                let cause = failure.into_backend_failure(kind).with_operation(operation).into();
                Err(PreparedChatSessionError {
                    cause,
                    funding: Some(funding),
                    input_funding,
                })
            }
        }
    }

    /// Consumes terminal state into output whose token storage retains funding.
    /// Returns the unchanged session if generation has not finished.
    pub fn into_output(self) -> Result<eredu_core::GenerationOutput<(), GenerationTokenIds>, Self> {
        if self.finish_reason().is_none() {
            return Err(self);
        }
        let timing = GenerationTiming::new(self.source.time_to_first_token);
        let Self { source, cursor, .. } = self;
        drop(source);
        Ok(cursor
            .into_output(timing)
            .unwrap_or_else(|_| unreachable!("checked terminal")))
    }

    /// Runs the same advancement until termination, with no alternate scheduler.
    pub fn run(
        mut self,
        cancellation: &GenerationCancellationToken,
        emit: &mut impl FnMut(SemanticEvent),
    ) -> Result<eredu_core::GenerationOutput<(), GenerationTokenIds>, PreparedChatSessionError>
    {
        while self.finish_reason().is_none() {
            self = self.advance(cancellation, emit)?;
        }
        Ok(self
            .into_output()
            .unwrap_or_else(|_| unreachable!("terminal cursor")))
    }

    /// Advances the same cursor until a pause request or terminal outcome.
    /// The caller owns the cross-thread handle; it grants no extra memory or
    /// execution allowance. Cancellation takes precedence over a pending pause.
    pub fn run_until_paused(
        mut self,
        control: &eredu_core::execution_control::GenerationControlHandle,
        emit: &mut impl FnMut(SemanticEvent),
    ) -> Result<Self, PreparedChatSessionError> {
        while self.finish_reason().is_none() {
            if control.pause_requested() && !control.cancellation().is_cancelled() {
                self.pause()?;
                break;
            }
            self = self.advance(control.cancellation(), emit)?;
        }
        Ok(self)
    }

    /// Acknowledges a pending pause and continues the existing cursor.
    pub fn resume(
        self,
        control: &eredu_core::execution_control::GenerationControlHandle,
        emit: &mut impl FnMut(SemanticEvent),
    ) -> Result<Self, PreparedChatSessionError> {
        control.acknowledge_resume();
        self.run_until_paused(control, emit)
    }
}

impl<B: eredu_core::execution_control::TextSamplingControlBackend> PreparedChatSession<'_, B> {
    fn sampling_boundary(&mut self) -> Result<eredu_core::TextSamplingBoundary<'_, B>, PreparedChatSessionError> {
        let funding = Some(self.preparation.metadata_funding().clone());
        let input_funding = self.input_funding.clone();
        self.source.generator.sampling_boundary().map_err(|error| PreparedChatSessionError {
            cause: Cause::Boundary(crate::api::control::map_continuation_failure(error, B::into_backend_failure)),
            funding,
            input_funding,
        })
    }

    /// Reads the existing sampler's compatibility facts at a completed boundary,
    /// including terminal state. The same source-health and delivery checks apply.
    pub fn sampling_state(&mut self) -> Result<eredu_core::execution_control::SamplingStateFacts, PreparedChatSessionError> {
        Ok(self.sampling_boundary()?.facts())
    }

    /// Installs a prospective temperature or seed through the existing sampler.
    /// The native producer admits replacement storage and all future sampling
    /// work before installation. Refusal preserves history, RNG and live policy.
    pub fn override_sampling(
        &mut self,
        request: eredu_core::execution_control::SamplingOverride,
    ) -> Result<eredu_core::execution_control::SamplingStateFacts, PreparedChatSessionError> {
        if self.finish_reason().is_some() {
            return Err(self.control_failure(Cause::Boundary(eredu_core::TextContinuationError::Failed)));
        }
        let facts = self.sampling_boundary()?.apply(request).map_err(|error| {
            self.control_failure(Cause::Sampling(crate::api::control::map_sampling_failure(
                error, B::into_backend_failure,
            )))
        })?;
        let mut sampling = self.effective_config.sampling();
        sampling.temperature = facts.temperature;
        sampling.do_sample = facts.temperature > 0.0;
        self.effective_config = saved_configuration(
            self.effective_config, sampling,
            request.reseed.unwrap_or(self.effective_config.seed()),
            self.effective_config.inference_policy(),
        );
        Ok(facts)
    }
}

// Rebuilds an already validated configuration without resolving checkpoint
// defaults. Native state supplies the RNG continuation; the seed is attribution.
fn saved_configuration(
    saved: TextGenerationConfig,
    sampling: eredu_core::ResolvedGenerationConfig,
    seed: u64,
    inference: eredu_core::TextInferencePolicy,
) -> TextGenerationConfig {
    let config = TextGenerationConfig::new(sampling).with_seed(seed).with_inference_policy(inference);
    match saved.strategy() {
        eredu_core::TextSamplingStrategy::Standard => config,
        eredu_core::TextSamplingStrategy::MirostatV2 { tau, eta } => config
            .with_mirostat_v2(tau, eta)
            .expect("installed sampling facts preserve the validated strategy"),
    }
}

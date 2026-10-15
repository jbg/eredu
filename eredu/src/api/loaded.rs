//! Loaded-model facade and prepared generation entry points.

use std::{num::NonZeroUsize, path::Path};

use eredu_core::{
    SpeculativeCapability, SpeculativeGenerationBackend, SpeculativeGenerationBatchOutput,
    SpeculativeGenerationBatchRequest, SpeculativeGenerationLane, SpeculativeGenerationOutput,
    generation::{FinishReason, SemanticEvent, resolve_generation_config},
};
use eredu_text::{
    gguf::GgufTokenizer,
    tokenizer::{
        chat_template_kwargs as inspect_chat_template_kwargs, ChatTemplateIdentity,
        Tokenizer as ChatTokenizer,
    },
};

use super::{
    request::SourceRequest, GenerationOutput, GenerationRequest, LoadedModel,
    LoadedTextModelConfig, PlannedModel, PreparedChatError, PreparedChatGenerationSettings,
    PreparedChatInput, PreparedChatSpeculativeBatchLane, PreparedChatSpeculativeBatchRequest,
    PreparedChatSpeculativeConstraint, TextDecoderError, TextMetadataError, TextModelError,
    TextModelOptions,
};
use crate::runtime::chat::PreparedChat;
use crate::{
    api::{
        metadata::{
            eos_token_ids_from_sidecar_dir, gguf_eos_token_ids, merge_eos_token_id_sources,
            read_checkpoint_generation_config,
        },
        request::{
            prepare_chat_from_parts, BackendGenerationTokenSource, OutputMode,
            PreparedChatSemanticState, PreparedChatSetupError, PreparedChatTokenDecoder,
        },
        tokenizer::{
            gguf_sidecar_dir, load_gguf_tokenizer_from_metadata, load_tokenizer_for_kind,
            load_tokenizer_template_kwargs, resolve_chat_template,
        },
    },
    runtime::{
        chat::{constraints::ConstraintCompiler, ChatTemplateRequest},
        generation::streaming::{
            drive_committed_generation_cancellable, CommittedGenerationError,
            CommittedTokenPipeline, CommittedTokenPipelineError, RawTokenDecoder,
        },
    },
};

/// Failure while assembling a tokenizer-aware model around a selected backend.
#[derive(Debug, thiserror::Error)]
pub enum LoadedModelLoadError {
    /// Portable artifact inspection or preparation planning failed.
    #[error(transparent)]
    Artifact(#[from] eredu_core::artifact::ArtifactError),
    /// Backend materialization or session creation failed.
    #[error("selected backend failed to load the model: {0}")]
    Backend(#[source] eredu_core::BackendFailure),
    /// The inspected model/session route lacks a required capability.
    #[error(transparent)]
    SessionCapability(#[from] eredu_core::SessionCapabilityError),
    /// Portable tokenizer, chat-template, or generation sidecar loading failed.
    #[error(transparent)]
    Metadata(#[from] TextMetadataError),
    /// The execution plan contains a drafting mode unknown to this facade version.
    #[error("execution plan selects an unsupported speculative drafting mode")]
    UnsupportedDraftingPlan,
}

/// Failure while planning, realizing, or loading a model through a backend factory.
#[derive(Debug, thiserror::Error)]
pub enum PlannedModelLoadError {
    /// Portable planning or backend/device realization failed.
    #[error(transparent)]
    Planning(#[from] eredu_core::AutomaticPlanningError),
    /// Artifact, metadata, materialization, or session creation failed.
    #[error(transparent)]
    Loading(#[from] LoadedModelLoadError),
}

pub(super) fn map_prepared_chat_setup_error(error: PreparedChatSetupError) -> PreparedChatError {
    match error {
        PreparedChatSetupError::Constraint(error) => PreparedChatError::Constraint(error),
        PreparedChatSetupError::Semantic(error) => PreparedChatError::Semantic(error),
    }
}

pub(super) fn map_controlled_generation_error<E>(
    error: eredu_core::ControlledTextGenerationError<E, crate::api::ConstraintError>,
) -> PreparedChatError
where
    E: std::error::Error + Send + Sync + 'static,
{
    match error {
        eredu_core::ControlledTextGenerationError::Backend(error) => {
            PreparedChatError::Backend(eredu_core::BackendFailure::from_error(error))
        }
        eredu_core::ControlledTextGenerationError::Controller(error) => {
            PreparedChatError::Constraint(error)
        }
        eredu_core::ControlledTextGenerationError::Preparation(error) => {
            PreparedChatError::Backend(error)
        }
        eredu_core::ControlledTextGenerationError::Operation(error) => {
            PreparedChatError::Operation(error)
        }
    }
}

fn map_committed_generation_error<E>(
    error: CommittedGenerationError<
        eredu_core::ControlledTextGenerationError<E, crate::api::ConstraintError>,
        TextDecoderError,
    >,
) -> PreparedChatError
where
    E: std::error::Error + Send + Sync + 'static,
{
    match error {
        CommittedGenerationError::Source(error) => map_controlled_generation_error(error),
        CommittedGenerationError::Pipeline(CommittedTokenPipelineError::Decoder(error)) => {
            PreparedChatError::Tokenizer(error)
        }
        CommittedGenerationError::Pipeline(CommittedTokenPipelineError::Semantic(error)) => {
            PreparedChatError::Semantic(error)
        }
        CommittedGenerationError::Lifecycle(error) => PreparedChatError::Generation(error),
        CommittedGenerationError::Delivery(error) => PreparedChatError::Capture(error),
        CommittedGenerationError::MissingTerminalToken => PreparedChatError::MissingTerminalToken,
    }
}

impl<B: eredu_core::TextGenerationBackend> LoadedModel<B> {
    pub(super) fn validate_refinement_canvas(
        &self,
        canvas: Option<&eredu_core::refinement::RefinementCanvas>,
    ) -> Result<(), eredu_core::generation::GenerationError> {
        if let Some(canvas) = canvas {
            let length = B::refinement_canvas_length(&self.runtime)
                .ok_or(eredu_core::generation::GenerationError::RefinementUnavailable)?;
            let vocabulary = B::refinement_vocabulary_size(&self.runtime)
                .ok_or(eredu_core::generation::GenerationError::RefinementUnavailable)?;
            canvas.validate(length.get(), vocabulary.get() as usize)?;
            if canvas
                .seed_tokens
                .iter()
                .flatten()
                .any(|id| !self.token_validity.allows(*id))
            {
                return Err(eredu_core::refinement::RefinementPolicyError::Canvas.into());
            }
        }
        Ok(())
    }

    pub(super) fn resolve_text_generation_settings(
        &self,
        settings: PreparedChatGenerationSettings,
    ) -> Result<
        (eredu_core::TextGenerationConfig, NonZeroUsize),
        eredu_core::generation::GenerationError,
    > {
        let mode = settings.decoding_mode.unwrap_or_else(|| B::default_decoding_mode(&self.runtime));
        if matches!(mode, eredu_core::decoding::DecodingMode::Diffusion { .. })
            && ((!B::decoding_capabilities(&self.runtime).autoregressive
                && settings.overrides.has_sampling_overrides())
                || !matches!(settings.strategy, eredu_core::TextSamplingStrategy::Standard))
        {
            return Err(eredu_core::generation::GenerationError::RequiresCausalConditionalScores);
        }
        let resolved = resolve_generation_config(
            self.checkpoint_generation_config.as_ref(),
            settings.overrides,
        )?;
        let max_tokens = resolved
            .max_new_tokens
            .and_then(NonZeroUsize::new)
            .unwrap_or_else(|| NonZeroUsize::new(256).expect("256 is non-zero"));
        let config = eredu_core::TextGenerationConfig::new(resolved)
            .with_seed(settings.seed)
            .with_prefill_chunk_policy(settings.prefill)
            .with_decoding_mode(settings.decoding_mode)
            .with_draft_events(settings.draft_events);
        let config = match settings.strategy {
            eredu_core::TextSamplingStrategy::Standard => config,
            eredu_core::TextSamplingStrategy::MirostatV2 { tau, eta } => {
                config.with_mirostat_v2(tau, eta)?
            }
        };
        Ok((config, max_tokens))
    }

    pub(super) fn generate_request<F>(
        &mut self,
        request: GenerationRequest<'_, B, F>,
        started: std::time::Instant,
    ) -> Result<GenerationOutput, PreparedChatError>
    where
        F: FnMut(SemanticEvent),
    {
        let mode = request.output_mode;
        self.generate_prepared(request, None, started, mode, None)
    }

    pub(super) fn generate_prepared<'a, F>(
        &'a mut self,
        request: GenerationRequest<'_, B, F>,
        capture: Option<(
            eredu_core::capture::AdmittedCapturePlan,
            Option<eredu_core::intervention::AdmittedInterventionPlan>,
            &'a mut dyn FnMut(
                super::request::GenerationStepKind,
                Option<eredu_core::capture::CapturedStep>,
                f64,
            ),
            &'a dyn Fn() -> Option<eredu_core::capture::CaptureError>,
        )>,
        generation_started: std::time::Instant,
        mode: OutputMode,
        canvas: Option<std::sync::Arc<eredu_core::refinement::RefinementCanvas>>,
    ) -> Result<GenerationOutput, PreparedChatError>
    where
        F: FnMut(SemanticEvent),
    {
        let GenerationRequest {
            output_mode: _,
            input,
            settings,
            caller_stop_sequences,
            cancellation,
            control_handle,
            mut on_event,
        } = request;
        use eredu_core::run_preparation::TextPreparationStage as Stage;
        let prepared_chat = input.prepared_chat();
        let local = (|| {
            if cancellation.is_cancelled() {
                return Ok(None);
            }
            input
                .validate_session(&self.session_identity)
                .map_err(PreparedChatError::Backend)?;
            let (config, max_tokens) = self.resolve_text_generation_settings(settings)?;
            self.validate_refinement_canvas(canvas.as_deref())?;
            let config = config.with_refinement_canvas(canvas);
            let control = mode
                .prepare_control(
                    prepared_chat,
                    caller_stop_sequences,
                    self.token_validity.clone(),
                )
                .map_err(map_prepared_chat_setup_error)?;
            // Tokenization is part of host readiness. Native construction follows
            // only after every rank accepted its local policy and exact prefix.
            let input = match input.into_unbound() {
                PreparedChatInput::RenderedPrompt(chat) => PreparedChatInput::token_ids(
                    chat,
                    self.tokenizer
                        .encode(chat.rendered_prompt(), false)
                        .map_err(TextDecoderError::Tokenizer)?
                        .get_ids()
                        .to_vec(),
                ),
                input => input,
            };
            Ok::<_, PreparedChatError>(Some((config, max_tokens, control, input)))
        })();
        let prepared = self.runtime.finish_text_preparation_cancellable(
            Stage::Request,
            local,
            PreparedChatError::Backend,
        )?;
        let Some((config, max_tokens, control, input)) = prepared else {
            on_event(SemanticEvent::Finished {
                reason: FinishReason::Cancelled,
            });
            return Ok(GenerationOutput::new(
                Vec::new(),
                FinishReason::Cancelled,
                eredu_core::GenerationTiming::default(),
                super::GenerationStats::default(),
            ));
        };
        let decoder = PreparedChatTokenDecoder {
            decoder: self.text_decoder(true),
        };
        let raw_decoder =
            RawTokenDecoder::with_structural_tokens(decoder, control.structural_tokens);
        let mut pipeline = CommittedTokenPipeline::new(raw_decoder, control.parser);
        let prompt = match input {
            PreparedChatInput::TokenIds { token_ids, .. } => {
                B::prepare_text_prompt(self.runtime.backend(), token_ids).map_err(|error| {
                    PreparedChatError::Backend(eredu_core::BackendFailure::from_error(error))
                })
            }
            PreparedChatInput::PreparedBackendInput { prompt, .. } => Ok(prompt),
            PreparedChatInput::RenderedPrompt(_) => {
                unreachable!("host preparation tokenized the prompt")
            }
            PreparedChatInput::SessionBound { .. } => {
                unreachable!("host preparation validated the session")
            }
        };
        let prompt = self.runtime.finish_text_preparation(
            Stage::Prompt,
            prompt,
            PreparedChatError::Backend,
        )?;
        let mut generator = eredu_core::ControlledTextGeneration::from_prompt(
            &mut self.runtime,
            prompt,
            config,
            control.controller,
        )
        .map_err(map_controlled_generation_error)?;
        generator.set_cancellation_token(cancellation.clone());
        if let Some(handle) = control_handle {
            generator
                .bind_control_handle(handle)
                .map_err(eredu_core::TextOperationPolicyError::DecodingMode)?;
        }
        let (on_token, delivery_failure) =
            if let Some((plan, intervention, on_token, delivery_failure)) = capture {
                if let Some(intervention) = intervention {
                    generator.enable_interventions(plan, intervention)?;
                } else {
                    generator.enable_capture(plan)?;
                }
                (Some(on_token), Some(delivery_failure))
            } else {
                generator.finish_text_preparation(
                    Stage::Instrumentation,
                    Ok(()),
                    PreparedChatError::Backend,
                )?;
                (None, None)
            };
        let ready = generator.finish_text_preparation_cancellable(
            Stage::Delivery,
            Ok(if cancellation.is_cancelled() {
                None
            } else {
                Some(())
            }),
            PreparedChatError::Backend,
        )?;
        if ready.is_none() {
            cancellation.cancel();
        }
        let mut source = BackendGenerationTokenSource {
            generator,
            on_token,
            delivery_failure,
            generation_started,
            time_to_first_token: None,
        };
        let (token_ids, finish_reason) = drive_committed_generation_cancellable(
            &mut source,
            &mut pipeline,
            prepared_chat.eos_token_ids(),
            max_tokens,
            &cancellation,
            &mut on_event,
        )
        .map_err(map_committed_generation_error)?;
        Ok(GenerationOutput::new(
            token_ids,
            finish_reason,
            eredu_core::GenerationTiming::new(source.time_to_first_token),
            super::GenerationStats::default(),
        ))
    }

    /// Reports fail-closed speculative support for this backend model session.
    pub fn speculative_capability(&self) -> SpeculativeCapability
    where
        B: SpeculativeGenerationBackend,
    {
        B::speculative_capability(&self.runtime)
    }

    pub(super) fn generate_speculative_request<'a, F>(
        &mut self,
        request: SourceRequest<'a, B, B::Drafter, F>,
    ) -> Result<SpeculativeGenerationOutput, PreparedChatError>
    where
        B: SpeculativeGenerationBackend,
        F: FnMut(SemanticEvent),
    {
        let mode = request.request.output_mode;
        let driver =
            eredu_runtime::RunSpeculativeGeneration::new_at(source_request_scheduler(&request), request.started)
                .with_speculation_limits(request.limits.map(Into::into));
        self.generate_prepared_speculative_with(request, mode, driver)
    }

    pub(super) fn generate_prepared_speculative_with<'a, F, V>(
        &mut self,
        request: SourceRequest<'a, B, <B as SpeculativeGenerationBackend>::Drafter, F>,
        mode: OutputMode,
        driver: V,
    ) -> Result<SpeculativeGenerationOutput, PreparedChatError>
    where
        B: SpeculativeGenerationBackend,
        F: FnMut(SemanticEvent),
        V: eredu_core::SpeculativeGenerationVisitor,
    {
        let scheduler = source_request_scheduler(&request);
        let SourceRequest {
            drafting,
            limits,
            started: _,
            request:
                GenerationRequest {
                    output_mode: _,
                    input,
                    settings,
                    caller_stop_sequences,
                    cancellation,
                    control_handle,
                    on_event,
                },
        } = request;
        let lanes = self.prepare_speculative_lanes(
            vec![PreparedChatSpeculativeBatchLane {
                input,
                settings,
                caller_stop_sequences,
                cancellation,
                control_handle,
                on_event: Box::new(on_event),
            }],
            scheduler,
            limits,
            matches!(drafting, eredu_core::SpeculativeDraft::SharedBackbone)
                || matches!(drafting, eredu_core::SpeculativeDraft::Target)
                    && B::decoding_capabilities(&self.runtime).speculation,
            matches!(drafting, eredu_core::SpeculativeDraft::Target),
            mode,
        )?;
        let output = B::with_speculative_execution(
            &mut self.runtime,
            SpeculativeGenerationBatchRequest::new(drafting, lanes, self.tokenizer_fingerprint),
            driver,
        )
        .map_err(|error| {
            PreparedChatError::Backend(eredu_core::BackendFailure::from_error(error))
        })?;
        let request: Result<[SpeculativeGenerationOutput; 1], _> =
            output.into_requests().try_into();
        match request {
            Ok([request]) => Ok(request),
            Err(requests) => Err(PreparedChatError::OutputCardinality {
                expected: 1,
                actual: requests.len(),
            }),
        }
    }

    /// Runs independent lanes through one fair scheduler with a common output policy.
    pub fn generate_speculative_batch<'a>(
        &mut self,
        request: PreparedChatSpeculativeBatchRequest<
            'a,
            B,
            <B as SpeculativeGenerationBackend>::Drafter,
        >,
    ) -> Result<SpeculativeGenerationBatchOutput, PreparedChatError>
    where
        B: SpeculativeGenerationBackend,
    {
        let mode = request.output_mode;
        let scheduler = request
            .lanes
            .first()
            .map(|lane| source_scheduler(lane.settings))
            .unwrap_or_default();
        let driver = eredu_runtime::RunSpeculativeGeneration::new(scheduler);
        let PreparedChatSpeculativeBatchRequest {
            drafting, lanes, ..
        } = request;
        let shared = matches!(drafting, eredu_core::SpeculativeDraft::SharedBackbone)
            || matches!(drafting, eredu_core::SpeculativeDraft::Target)
                && B::decoding_capabilities(&self.runtime).speculation;
        let prepared_lanes = self.prepare_speculative_lanes(
            lanes,
            scheduler,
            None,
            shared,
            matches!(drafting, eredu_core::SpeculativeDraft::Target),
            mode,
        )?;
        B::with_speculative_execution(
            &mut self.runtime,
            SpeculativeGenerationBatchRequest::new(
                drafting,
                prepared_lanes,
                self.tokenizer_fingerprint,
            ),
            driver,
        )
        .map_err(|error| PreparedChatError::Backend(eredu_core::BackendFailure::from_error(error)))
    }

    fn prepare_speculative_lanes<'a>(
        &self,
        lanes: Vec<PreparedChatSpeculativeBatchLane<'a, B>>,
        scheduler: eredu_core::SpeculativeSchedulerOptions,
        limits: Option<super::SpeculationOptions>,
        shared_backbone: bool,
        target_only: bool,
        mode: OutputMode,
    ) -> Result<
        Vec<SpeculativeGenerationLane<'a, B, PreparedChatSpeculativeConstraint>>,
        PreparedChatError,
    > {
        use eredu_core::run_preparation::TextPreparationStage as Stage;
        // One vote for the entire batch: a failure in a later lane must not leave
        // peers waiting for a per-lane vote or executing an earlier lane.
        let host = (|| {
            if !target_only
                && B::generation_score_kind(&self.runtime)
                    != eredu_core::refinement::PredictionScoreKind::CausalConditional
            {
                return Err(PreparedChatError::Generation(
                    eredu_core::generation::GenerationError::RequiresCausalConditionalScores,
                ));
            }
            scheduler.validate()?;
            lanes
                .into_iter()
                .map(|lane| {
                    let mut settings = lane.settings;
                    let requested = settings.decoding_mode.unwrap_or(eredu_core::decoding::DecodingMode::Autoregressive);
                    if target_only && !shared_backbone && matches!(requested, eredu_core::decoding::DecodingMode::SelfSpeculation(_)) {
                        return Err(eredu_core::GenerationError::NoBackendDraftCapacity.into());
                    }
                    if let (Some(limits), eredu_core::decoding::DecodingMode::SelfSpeculation(options)) = (limits, requested) {
                        eredu_core::decoding::SpeculationLimits::from(limits).validate(options)?;
                    }
                    if limits.is_none() && source_scheduler(settings) != scheduler {
                        return Err(eredu_core::decoding::DecodingModeError::Unsupported("source-backed lane schedulers disagree").into());
                    }
                    if let eredu_core::decoding::DecodingMode::SelfSpeculation(options) = requested {
                        if !shared_backbone && (options.request().draft_adapter || options.request().refinement.is_some()) {
                            return Err(eredu_core::decoding::DecodingModeError::Unsupported("masked refinement and draft adapters require a shared-backbone source").into());
                        }
                    }
                    settings.decoding_mode = Some(requested);
                    let generation = self.resolve_text_generation_settings(settings)?;
                    let prepared = self.prepare_speculative_generation(
                        lane.input,
                        generation,
                        lane.caller_stop_sequences,
                        mode,
                    )?;
                    Ok((prepared, lane.cancellation, lane.on_event, lane.control_handle.cloned()))
                })
                .collect::<Result<Vec<_>, PreparedChatError>>()
        })();
        let host = self.runtime.finish_text_preparation(
            Stage::Request,
            host,
            PreparedChatError::Backend,
        )?;
        let prompts = host
            .into_iter()
            .map(
                |(
                    (input, generation, config, constraint, semantic),
                    cancellation,
                    on_event,
                    control_handle,
                )| {
                    let prompt = match input {
                        PreparedChatInput::TokenIds { token_ids, .. } => {
                            B::prepare_text_prompt(self.runtime.backend(), token_ids).map_err(
                                |error| {
                                    PreparedChatError::Backend(
                                        eredu_core::BackendFailure::from_error(error),
                                    )
                                },
                            )?
                        }
                        PreparedChatInput::PreparedBackendInput { prompt, .. } => prompt,
                        PreparedChatInput::RenderedPrompt(_) => {
                            unreachable!("host preparation tokenized the prompt")
                        }
                        PreparedChatInput::SessionBound { .. } => {
                            unreachable!("host preparation validated the session")
                        }
                    };
                    Ok(SpeculativeGenerationLane::new(
                        prompt,
                        generation,
                        config,
                        constraint,
                        semantic,
                        cancellation,
                        on_event,
                    )
                    .with_control_handle(control_handle))
                },
            )
            .collect::<Result<Vec<_>, PreparedChatError>>();
        self.runtime
            .finish_text_preparation(Stage::Prompt, prompts, PreparedChatError::Backend)
    }

    #[allow(clippy::type_complexity)]
    fn prepare_speculative_generation<'a>(
        &self,
        input: PreparedChatInput<'a, B>,
        generation: (eredu_core::TextGenerationConfig, NonZeroUsize),
        caller_stop_sequences: &[String],
        mode: OutputMode,
    ) -> Result<
        (
            PreparedChatInput<'a, B>,
            eredu_core::TextGenerationConfig,
            eredu_core::generation::GenerationRequestConfig,
            PreparedChatSpeculativeConstraint,
            Box<dyn eredu_core::SpeculativeSemanticState>,
        ),
        PreparedChatError,
    > {
        input
            .validate_session(&self.session_identity)
            .map_err(PreparedChatError::Backend)?;
        let input = input.into_unbound();
        let prepared_chat = input.prepared_chat();
        let control = mode
            .prepare_control(
                prepared_chat,
                caller_stop_sequences,
                self.token_validity.clone(),
            )
            .map_err(|error| match error {
                PreparedChatSetupError::Constraint(error) => PreparedChatError::Constraint(error),
                PreparedChatSetupError::Semantic(error) => PreparedChatError::Semantic(error),
            })?;
        let constraint = PreparedChatSpeculativeConstraint::new(control.controller);
        let semantic = PreparedChatSemanticState::new(
            PreparedChatTokenDecoder {
                decoder: self.text_decoder(true),
            },
            control.parser,
            control.structural_tokens,
        );
        let eos_token_ids = prepared_chat.eos_token_ids().to_vec();
        let (generation, max_tokens) = generation;
        let temperature = generation.sampling().temperature;
        let decoding_mode = generation.decoding_mode().unwrap_or_default();
        let input = match input {
            PreparedChatInput::RenderedPrompt(prepared_chat) => {
                let token_ids = self.encode(prepared_chat.rendered_prompt(), false)?;
                PreparedChatInput::token_ids(prepared_chat, token_ids)
            }
            input => input,
        };
        Ok((
            input,
            generation,
            eredu_core::generation::GenerationRequestConfig {
                max_tokens: max_tokens.get(),
                decoding_mode,
                temperature,
                eos_token_ids,
            },
            constraint,
            Box::new(semantic),
        ))
    }

    /// Returns the stable identity of the template selected for `tools`.
    pub fn selected_chat_template_identity(
        &self,
        tools: Option<&[serde_json::Value]>,
    ) -> Result<Option<ChatTemplateIdentity>, TextModelError> {
        self.chat_template
            .as_ref()
            .map(|templates| {
                templates
                    .select(tools)
                    .map(|selected| selected.identity().clone())
            })
            .transpose()
            .map_err(TextModelError::Template)
    }

    /// Returns likely user-provided kwargs referenced by the chat template.
    pub fn chat_template_kwargs(&self) -> Result<Vec<String>, TextModelError> {
        let Some(template) = &self.chat_template else {
            return Ok(Vec::new());
        };
        let selected = template.select(None)?;
        Ok(
            inspect_chat_template_kwargs(selected.template(), &self.model_id)?
                .into_iter()
                .filter(|name| !self.tokenizer.template_kwargs().contains_key(name))
                .collect(),
        )
    }

    /// Renders and validates one JSON-valued chat for generation.
    ///
    /// Tool declarations and history use application names. Preparation assigns
    /// protocol-valid wire aliases when needed; semantic tool-call events restore
    /// application names. Rendered prompts and literal token/text output retain
    /// the actual wire representation.
    ///
    /// Returns [`TextModelError::MissingChatTemplate`] when no template is
    /// attached. Supply one through [`TextModelOptions`] while loading
    /// or [`Self::set_chat_template`] before preparing a chat.
    pub fn prepare_chat(
        &mut self,
        request: ChatTemplateRequest,
    ) -> Result<PreparedChat, TextModelError> {
        let template = self
            .chat_template
            .clone()
            .ok_or(TextModelError::MissingChatTemplate)?;
        let constraint_compiler =
            ConstraintCompiler::from_tokenizer(&self.tokenizer, &self.eos_token_ids);
        prepare_chat_from_parts(
            &mut self.tokenizer,
            template,
            &self.model_id,
            &self.eos_token_ids,
            Some(&constraint_compiler),
            request,
        )
    }
}

impl<B> LoadedModel<B>
where
    B: eredu_core::TextGenerationBackend + eredu_core::ModelLoadingBackend,
    B::ConfigurationResolver: eredu_core::ModelConfigurationResolver<
        ArtifactPlan = eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
{
    /// Realizes a complete portable execution plan through the selected factory.
    ///
    /// The factory first translates and selects the backend preparation, then
    /// owns device and queue construction. Generic callers never construct
    /// backend streams, assistants, or backend-specific load options. The
    /// returned [`PlannedModel`] owns the target and complete drafting mode.
    /// Use [`Self::load_execution_plan_with_text_options`] to override the
    /// checkpoint chat template.
    pub fn load_execution_plan<F>(
        factory: &F,
        artifact: impl AsRef<Path>,
        plan: &eredu_core::ExecutionPlan,
    ) -> Result<PlannedModel<B, eredu_core::RealizedDrafting<F::Drafter>>, PlannedModelLoadError>
    where
        F: eredu_core::ExecutionPlanBackendFactory<
            Backend = B,
            DrafterPreparation = eredu_architectures::ExternalDraftPreparation,
        >,
    {
        Self::load_execution_plan_with_text_options(
            factory,
            artifact,
            plan,
            TextModelOptions::default(),
        )
    }

    /// Realizes a complete execution plan with facade-owned text overrides.
    ///
    /// The template override is selected before backend realization and is
    /// used by subsequent chat preparation and template inspection.
    pub fn load_execution_plan_with_text_options<F>(
        factory: &F,
        artifact: impl AsRef<Path>,
        plan: &eredu_core::ExecutionPlan,
        text_options: TextModelOptions,
    ) -> Result<PlannedModel<B, eredu_core::RealizedDrafting<F::Drafter>>, PlannedModelLoadError>
    where
        F: eredu_core::ExecutionPlanBackendFactory<
            Backend = B,
            DrafterPreparation = eredu_architectures::ExternalDraftPreparation,
        >,
    {
        let artifact = artifact.as_ref();
        let inspection = eredu_architectures::configuration::inspect_artifact(artifact)
            .map_err(LoadedModelLoadError::Artifact)?;
        super::PreparedTextModel::from_artifact(inspection, text_options)?
            .select_execution_plan(factory, plan)?
            .load()
    }

    /// Plans and loads one complete model session without exposing backend construction.
    ///
    /// The returned report is the exact portable plan used to realize the
    /// backend and remains suitable for persistence and execution telemetry.
    /// Use [`Self::plan_and_load_with_text_options`] to override the checkpoint
    /// chat template.
    #[allow(clippy::type_complexity)]
    pub fn plan_and_load<F>(
        factory: &F,
        planner: &eredu_core::AutomaticPlanner,
        request: &eredu_core::AutomaticPlanRequest,
    ) -> Result<
        (
            PlannedModel<B, eredu_core::RealizedDrafting<F::Drafter>>,
            eredu_core::ExecutionPlanReport,
        ),
        PlannedModelLoadError,
    >
    where
        F: eredu_core::ExecutionPlanBackendFactory<
                Backend = B,
                DrafterPreparation = eredu_architectures::ExternalDraftPreparation,
            > + eredu_core::AutomaticPlanningBackend<
                Inspection = eredu_core::ArtifactInspection<
                    eredu_architectures::processor_plan::ArtifactArchitecturePlan,
                >,
            >,
    {
        Self::plan_and_load_with_text_options(
            factory,
            planner,
            request,
            TextModelOptions::default(),
        )
    }

    /// Plans and loads a complete model session with facade-owned text overrides.
    ///
    /// Text options apply to the loaded target without changing the execution
    /// plan or its retained artifact inspection.
    #[allow(clippy::type_complexity)]
    pub fn plan_and_load_with_text_options<F>(
        factory: &F,
        planner: &eredu_core::AutomaticPlanner,
        request: &eredu_core::AutomaticPlanRequest,
        text_options: TextModelOptions,
    ) -> Result<
        (
            PlannedModel<B, eredu_core::RealizedDrafting<F::Drafter>>,
            eredu_core::ExecutionPlanReport,
        ),
        PlannedModelLoadError,
    >
    where
        F: eredu_core::ExecutionPlanBackendFactory<
                Backend = B,
                DrafterPreparation = eredu_architectures::ExternalDraftPreparation,
            > + eredu_core::AutomaticPlanningBackend<
                Inspection = eredu_core::ArtifactInspection<
                    eredu_architectures::processor_plan::ArtifactArchitecturePlan,
                >,
            >,
    {
        let retained = planner.plan_retained(factory, request)?;
        let (report, inspection) = retained.into_parts();
        let model = super::PreparedTextModel::from_artifact(inspection, text_options)?
            .select_execution_plan(factory, &report.plan)?
            .load()?;
        Ok((model, report))
    }

    /// Loads one artifact, its tokenizer, and its chat metadata on `backend`.
    ///
    /// The backend already owns device placement, execution queues, transfer
    /// queues, and optional communication. Artifact inspection occurs exactly
    /// once and is shared by portable metadata assembly and backend planning.
    /// Use [`Self::load_with_text_options`] to override the checkpoint chat template.
    pub fn load(
        backend: B,
        artifact: impl AsRef<Path>,
        options: B::LoadOptions,
    ) -> Result<Self, LoadedModelLoadError> {
        Self::load_with_text_options(backend, artifact, options, TextModelOptions::default())
    }

    /// Loads an artifact on an existing backend with facade-owned text overrides.
    pub fn load_with_text_options(
        backend: B,
        artifact: impl AsRef<Path>,
        options: B::LoadOptions,
        text_options: TextModelOptions,
    ) -> Result<Self, LoadedModelLoadError> {
        let artifact = artifact.as_ref();
        let inspection = eredu_architectures::configuration::inspect_artifact(artifact)?;
        super::PreparedTextModel::from_artifact(inspection, text_options)?.load_on(backend, options)
    }

    pub(super) fn from_inspected(
        backend: B,
        inspection: eredu_core::ArtifactInspection<
            eredu_architectures::processor_plan::ArtifactArchitecturePlan,
        >,
        options: B::LoadOptions,
        tokenizer: ChatTokenizer,
        config: LoadedTextModelConfig,
    ) -> Result<Self, LoadedModelLoadError> {
        let prepared = match eredu_core::prepare_inspected_model(&backend, inspection, options) {
            Ok(prepared) => prepared,
            Err(error) => return Err(map_model_load_error(error)),
        };
        let runtime =
            eredu_core::ModelRuntime::from_prepared(backend, prepared).map_err(|error| {
                LoadedModelLoadError::Backend(eredu_core::BackendFailure::from_error(error))
            })?;
        Ok(Self::from_runtime(runtime, tokenizer, config))
    }
}

pub(super) fn map_model_load_error<E: std::error::Error + Send + Sync + 'static>(
    error: eredu_core::ModelLoadError<E>,
) -> LoadedModelLoadError {
    match error {
        eredu_core::ModelLoadError::Artifact(error) => LoadedModelLoadError::Artifact(error),
        eredu_core::ModelLoadError::Backend(error) => {
            LoadedModelLoadError::Backend(eredu_core::BackendFailure::from_error(error))
        }
        eredu_core::ModelLoadError::SessionCapability(error) => {
            LoadedModelLoadError::SessionCapability(error)
        }
        error => {
            LoadedModelLoadError::Artifact(eredu_core::artifact::ArtifactError::InvalidArtifact(
                format!("unsupported model-loading failure: {error}"),
            ))
        }
    }
}

pub(super) fn bind_gguf_special_token_ids(
    inspection: &mut eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    tokenizer: &ChatTokenizer,
) -> Result<(), TextMetadataError> {
    use eredu_architectures::processor_plan::{
        GgufSpecialTokenIds, GgufSpecialTokenKind, QwenMediaTokenIds,
    };

    let Some(kind) = inspection
        .architecture_plan()
        .required_gguf_special_tokens()
    else {
        return Ok(());
    };
    let required = |token: &str| {
        tokenizer.token_to_id(token).ok_or_else(|| {
            TextMetadataError::TokenizerConfiguration(format!(
                "GGUF tokenizer is missing required media token {token:?}"
            ))
        })
    };
    let ids = match kind {
        GgufSpecialTokenKind::Qwen => GgufSpecialTokenIds::Qwen(QwenMediaTokenIds {
            image_token_id: required("<|image_pad|>")?,
            video_token_id: required("<|video_pad|>")?,
            vision_start_token_id: required("<|vision_start|>")?,
            vision_end_token_id: required("<|vision_end|>")?,
        }),
        GgufSpecialTokenKind::Inkling => GgufSpecialTokenIds::Inkling {
            image_bos_token_id: required("<|content_image|>")?,
            audio_bos_token_id: required("<|content_audio_input|>")?,
        },
    };
    inspection
        .architecture_plan_mut()
        .bind_gguf_special_token_ids(ids)
        .map_err(|error| TextMetadataError::TokenizerConfiguration(error.to_string()))
}

pub(super) fn source_request_scheduler<B: eredu_core::TextGenerationBackend, D, F>(
    request: &SourceRequest<'_, B, D, F>,
) -> eredu_core::SpeculativeSchedulerOptions {
    request
        .limits
        .map(|options| options.request().scheduler)
        .unwrap_or_else(|| source_scheduler(request.request.settings))
}

pub(super) fn source_scheduler(
    settings: PreparedChatGenerationSettings,
) -> eredu_core::SpeculativeSchedulerOptions {
    match settings.decoding_mode {
        Some(eredu_core::decoding::DecodingMode::SelfSpeculation(options)) => {
            options.request().scheduler
        }
        _ => eredu_core::SpeculativeSchedulerOptions::default(),
    }
}

pub(super) fn loaded_text_artifact(
    inspection: &eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    text_options: TextModelOptions,
) -> Result<(ChatTokenizer, LoadedTextModelConfig), TextMetadataError> {
    let path = inspection.path();
    let configuration = inspection.configuration();
    let kind = inspection.architecture_plan().model_kind();
    if kind.loading_protocol() == eredu_core::LoadingProtocol::Realtime {
        return Err(TextMetadataError::UnsupportedArchitecture(
            "Moshi-family models use a realtime speech-to-speech token contract; use the realtime backend contract instead of LoadedModel".into(),
        ));
    }

    let sidecar_dir = match inspection.format() {
        eredu_core::ArtifactFormat::SafeTensors => path,
        eredu_core::ArtifactFormat::Gguf => gguf_sidecar_dir(path),
        _ => {
            return Err(TextMetadataError::UnsupportedArchitecture(
                "unsupported artifact format for text metadata".into(),
            ));
        }
    };
    let checkpoint_generation_config = read_checkpoint_generation_config(sidecar_dir)?;
    let sidecar_eos_token_ids = eos_token_ids_from_sidecar_dir(sidecar_dir)?;

    let (mut tokenizer, chat_template, eos_token_ids, model_id) = match inspection.format() {
        eredu_core::ArtifactFormat::SafeTensors => {
            let tokenizer = load_tokenizer_for_kind(kind, path)?;
            (
                ChatTokenizer::from_tokenizer(tokenizer),
                resolve_chat_template(path, None, &text_options)?,
                sidecar_eos_token_ids,
                configuration.effective_model_type().to_owned(),
            )
        }
        eredu_core::ArtifactFormat::Gguf => {
            let metadata = inspection
                .gguf_checkpoint()
                .expect("GGUF inspection owns a portable checkpoint")
                .metadata()
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect::<std::collections::HashMap<_, _>>();
            let chat_template = resolve_chat_template(sidecar_dir, Some(&metadata), &text_options)?;
            let GgufTokenizer {
                tokenizer,
                template_kwargs,
            } = load_gguf_tokenizer_from_metadata(path, &metadata)?;
            let mut tokenizer = ChatTokenizer::from_tokenizer(tokenizer);
            tokenizer.set_template_kwargs(template_kwargs);
            (
                tokenizer,
                chat_template,
                merge_eos_token_id_sources([sidecar_eos_token_ids, gguf_eos_token_ids(&metadata)?]),
                path.display().to_string(),
            )
        }
        _ => {
            return Err(TextMetadataError::UnsupportedArchitecture(
                "unsupported artifact format for tokenizer loading".into(),
            ));
        }
    };
    if inspection.format() == eredu_core::ArtifactFormat::SafeTensors {
        tokenizer.set_template_kwargs(load_tokenizer_template_kwargs(path)?);
    }

    Ok((
        tokenizer,
        LoadedTextModelConfig {
            model_family: kind,
            effective_model_type: configuration.effective_model_type().to_owned(),
            model_id,
            chat_template,
            eos_token_ids,
            checkpoint_generation_config,
        },
    ))
}

//! Loaded-model facade and prepared generation entry points.

use std::{num::NonZeroUsize, path::Path};

use eredu_core::{
    generation::{resolve_generation_config, FinishReason, SemanticEvent},
    DraftingPlan, ExternalDraftArtifact, SpeculativeCapability, SpeculativeGenerationBackend,
    SpeculativeGenerationBatchOutput, SpeculativeGenerationBatchRequest, SpeculativeGenerationLane,
    SpeculativeGenerationOutput, TokenizerCompatibilityProof,
};
use eredu_text::{
    gguf::GgufTokenizer,
    tokenizer::{
        chat_template_kwargs as inspect_chat_template_kwargs, ChatTemplateIdentity,
        Tokenizer as ChatTokenizer,
    },
};

use super::{
    LoadedModel, LoadedTextModelConfig, PlannedModel, PreparedChatError,
    PreparedChatGenerationOutput, PreparedChatGenerationRequest, PreparedChatGenerationSettings,
    PreparedChatInput, PreparedChatSpeculativeBatchRequest, PreparedChatSpeculativeConstraint,
    PreparedChatSpeculativeError, PreparedChatSpeculativeGenerationRequest, TextDecoderError,
    TextMetadataError, TextModelError, TextModelOptions,
};
use crate::runtime::chat::PreparedChat;
use crate::{
    api::{
        metadata::{
            eos_token_ids_from_sidecar_dir, gguf_eos_token_ids, merge_eos_token_id_sources,
            read_checkpoint_generation_config,
        },
        request::{
            prepare_chat_from_parts, BackendGenerationTokenSource, PreparedChatSemanticState,
            PreparedChatSetupError, PreparedChatTokenDecoder, PreparedGenerationMode,
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
        CommittedGenerationError::MissingTerminalToken => PreparedChatError::MissingTerminalToken,
    }
}

impl<B: eredu_core::TextGenerationBackend> LoadedModel<B> {
    pub(super) fn resolve_text_generation_settings(
        &self,
        settings: PreparedChatGenerationSettings,
    ) -> Result<
        (eredu_core::TextGenerationConfig, NonZeroUsize),
        eredu_core::generation::GenerationError,
    > {
        let resolved = resolve_generation_config(
            self.checkpoint_generation_config.as_ref(),
            settings.overrides,
        )?;
        let max_tokens = resolved
            .max_new_tokens
            .and_then(NonZeroUsize::new)
            .unwrap_or_else(|| NonZeroUsize::new(256).expect("256 is non-zero"));
        let config = eredu_core::TextGenerationConfig::new(resolved).with_seed(settings.seed);
        let config = match settings.strategy {
            eredu_core::TextSamplingStrategy::Standard => config,
            eredu_core::TextSamplingStrategy::MirostatV2 { tau, eta } => {
                config.with_mirostat_v2(tau, eta)?
            }
        };
        Ok((config, max_tokens))
    }

    /// Generates one constrained semantic response through the selected backend.
    /// Use [`Self::reset`] to establish fresh request state when reusing a model,
    /// and [`Self::synchronize`] to confirm settlement after cancellation.
    pub fn generate_prepared_chat<F>(
        &mut self,
        request: PreparedChatGenerationRequest<'_, B, F>,
    ) -> Result<PreparedChatGenerationOutput, PreparedChatError>
    where
        F: FnMut(SemanticEvent),
    {
        self.generate_prepared_chat_captured(request, None, std::time::Instant::now())
    }

    /// Generates literal text from a prepared prompt, including unrecognized
    /// templates, with committed-token timing in the returned output.
    ///
    /// Uses the same request and output as [`Self::generate_prepared_chat`], but
    /// emits only `TextDelta` and `Finished`. Retains sampling, tokenizer validity,
    /// EOS, caller stops and cancellation; skips special tokens without parsing
    /// tools, reasoning or profile-specific stops. Admission follows
    /// [`PreparedChat::text_generation_support`]: native tool declarations and
    /// required calls are rejected; explicit thinking requires opting into
    /// unparsed reasoning. Application-emulated tools can use ordinary prompt text.
    /// Use [`Self::reset`] before reusing request state and [`Self::synchronize`]
    /// to confirm settlement after cancellation.
    pub fn generate_prepared_text<F>(
        &mut self,
        request: PreparedChatGenerationRequest<'_, B, F>,
    ) -> Result<PreparedChatGenerationOutput, PreparedChatError>
    where
        F: FnMut(SemanticEvent),
    {
        self.generate_prepared(
            request,
            None,
            std::time::Instant::now(),
            PreparedGenerationMode::Text,
        )
    }

    pub(super) fn generate_prepared_chat_captured<'a, F>(
        &'a mut self,
        request: PreparedChatGenerationRequest<'_, B, F>,
        capture: Option<(
            eredu_core::capture::AdmittedCapturePlan,
            Option<eredu_core::intervention::AdmittedInterventionPlan>,
            &'a mut dyn FnMut(Option<u32>, Option<eredu_core::capture::CapturedStep>, f64),
        )>,
        generation_started: std::time::Instant,
    ) -> Result<PreparedChatGenerationOutput, PreparedChatError>
    where
        F: FnMut(SemanticEvent),
    {
        self.generate_prepared(
            request,
            capture,
            generation_started,
            PreparedGenerationMode::Semantic,
        )
    }

    fn generate_prepared<'a, F>(
        &'a mut self,
        request: PreparedChatGenerationRequest<'_, B, F>,
        capture: Option<(
            eredu_core::capture::AdmittedCapturePlan,
            Option<eredu_core::intervention::AdmittedInterventionPlan>,
            &'a mut dyn FnMut(Option<u32>, Option<eredu_core::capture::CapturedStep>, f64),
        )>,
        generation_started: std::time::Instant,
        mode: PreparedGenerationMode,
    ) -> Result<PreparedChatGenerationOutput, PreparedChatError>
    where
        F: FnMut(SemanticEvent),
    {
        let PreparedChatGenerationRequest {
            input,
            settings,
            caller_stop_sequences,
            cancellation,
            mut on_event,
        } = request;
        if cancellation.is_cancelled() {
            on_event(SemanticEvent::Finished {
                reason: FinishReason::Cancelled,
            });
            return Ok(PreparedChatGenerationOutput::new(
                Vec::new(),
                FinishReason::Cancelled,
                eredu_core::GenerationTiming::default(),
                (),
            ));
        }

        let prepared_chat = input.prepared_chat();
        let (config, max_tokens) = self.resolve_text_generation_settings(settings)?;
        let control = mode
            .prepare_control(
                prepared_chat,
                caller_stop_sequences,
                self.token_validity.clone(),
            )
            .map_err(map_prepared_chat_setup_error)?;
        let decoder = PreparedChatTokenDecoder {
            decoder: self.text_decoder(true),
        };
        let raw_decoder =
            RawTokenDecoder::with_structural_tokens(decoder, control.structural_tokens);
        let mut pipeline = CommittedTokenPipeline::new(raw_decoder, control.parser);
        let mut generator = match input {
            PreparedChatInput::RenderedPrompt(prepared_chat) => {
                let prompt = self
                    .tokenizer
                    .encode(prepared_chat.rendered_prompt(), false)
                    .map_err(TextDecoderError::Tokenizer)?
                    .get_ids()
                    .to_vec();
                eredu_core::ControlledTextGeneration::new(
                    &mut self.runtime,
                    prompt,
                    config,
                    control.controller,
                )
            }
            PreparedChatInput::PreparedBackendInput { prompt, .. } => {
                eredu_core::ControlledTextGeneration::from_prompt(
                    &mut self.runtime,
                    prompt,
                    config,
                    control.controller,
                )
            }
        }
        .map_err(map_controlled_generation_error)?;
        let (on_token, capture_enabled) = if let Some((plan, intervention, on_token)) = capture {
            let enabled = !plan.is_empty() || intervention.as_ref().is_some_and(|p| !p.is_empty());
            if let Some(intervention) = intervention {
                generator.enable_interventions(plan, intervention)?;
            } else if enabled {
                generator.enable_capture(plan)?;
            }
            (Some(on_token), enabled)
        } else {
            (None, false)
        };
        let mut source = BackendGenerationTokenSource {
            generator,
            on_token,
            capture_enabled,
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
        Ok(PreparedChatGenerationOutput::new(
            token_ids,
            finish_reason,
            eredu_core::GenerationTiming::new(source.time_to_first_token),
            (),
        ))
    }

    /// Reports fail-closed speculative support for this backend model session.
    pub fn speculative_capability(&self) -> SpeculativeCapability
    where
        B: SpeculativeGenerationBackend,
    {
        B::speculative_capability(&self.runtime)
    }

    /// Generates one structured response using embedded or external drafting.
    pub fn generate_prepared_chat_speculative<'a, F>(
        &mut self,
        request: PreparedChatSpeculativeGenerationRequest<
            'a,
            B,
            <B as SpeculativeGenerationBackend>::Drafter,
            F,
        >,
    ) -> Result<SpeculativeGenerationOutput, PreparedChatSpeculativeError>
    where
        B: SpeculativeGenerationBackend,
        F: FnMut(SemanticEvent),
    {
        self.generate_prepared_speculative(request, PreparedGenerationMode::Semantic)
    }

    /// Generates literal text using embedded or external drafting, including
    /// unrecognized templates. Uses the text admission and output semantics of
    /// [`Self::generate_prepared_text`] with the existing speculative request type.
    /// TTFT measures the first target-committed token, including buffered or EOS
    /// tokens. The selected model must support the requested drafting mode.
    pub fn generate_prepared_text_speculative<'a, F>(
        &mut self,
        request: PreparedChatSpeculativeGenerationRequest<
            'a,
            B,
            <B as SpeculativeGenerationBackend>::Drafter,
            F,
        >,
    ) -> Result<SpeculativeGenerationOutput, PreparedChatSpeculativeError>
    where
        B: SpeculativeGenerationBackend,
        F: FnMut(SemanticEvent),
    {
        self.generate_prepared_speculative(request, PreparedGenerationMode::Text)
    }

    fn generate_prepared_speculative<'a, F>(
        &mut self,
        request: PreparedChatSpeculativeGenerationRequest<
            'a,
            B,
            <B as SpeculativeGenerationBackend>::Drafter,
            F,
        >,
        mode: PreparedGenerationMode,
    ) -> Result<SpeculativeGenerationOutput, PreparedChatSpeculativeError>
    where
        B: SpeculativeGenerationBackend,
        F: FnMut(SemanticEvent),
    {
        let driver = eredu_runtime::RunSpeculativeGeneration::new(request.options.scheduler);
        self.generate_prepared_speculative_with(request, mode, driver)
    }

    pub(super) fn generate_prepared_speculative_with<'a, F, V>(
        &mut self,
        request: PreparedChatSpeculativeGenerationRequest<
            'a,
            B,
            <B as SpeculativeGenerationBackend>::Drafter,
            F,
        >,
        mode: PreparedGenerationMode,
        driver: V,
    ) -> Result<SpeculativeGenerationOutput, PreparedChatSpeculativeError>
    where
        B: SpeculativeGenerationBackend,
        F: FnMut(SemanticEvent),
        V: eredu_core::SpeculativeGenerationVisitor,
    {
        let PreparedChatSpeculativeGenerationRequest {
            input,
            drafting,
            settings,
            options,
            caller_stop_sequences,
            cancellation,
            on_event,
        } = request;
        let generation = self.resolve_text_generation_settings(settings)?;
        let (prompt, generation, config, constraint, semantic) = self
            .prepare_speculative_generation(
                input,
                generation,
                options.max_draft_tokens,
                caller_stop_sequences,
                mode,
            )?;
        let output = B::with_speculative_execution(
            &mut self.runtime,
            SpeculativeGenerationBatchRequest::new(
                drafting,
                vec![SpeculativeGenerationLane::new(
                    prompt,
                    generation,
                    config,
                    constraint,
                    semantic,
                    cancellation,
                    Box::new(on_event),
                )],
                self.tokenizer_fingerprint,
            ),
            driver,
        )
        .map_err(|error| {
            PreparedChatSpeculativeError::Backend(eredu_core::BackendFailure::from_error(error))
        })?;
        let request: Result<[SpeculativeGenerationOutput; 1], _> =
            output.into_requests().try_into();
        match request {
            Ok([request]) => Ok(request),
            Err(requests) => Err(PreparedChatSpeculativeError::OutputCardinality {
                expected: 1,
                actual: requests.len(),
            }),
        }
    }

    /// Generates independent prepared chats through one fair speculative scheduler.
    pub fn generate_prepared_chat_speculative_batch<'a>(
        &mut self,
        request: PreparedChatSpeculativeBatchRequest<
            'a,
            B,
            <B as SpeculativeGenerationBackend>::Drafter,
        >,
    ) -> Result<SpeculativeGenerationBatchOutput, PreparedChatSpeculativeError>
    where
        B: SpeculativeGenerationBackend,
    {
        self.generate_prepared_speculative_batch(request, PreparedGenerationMode::Semantic)
    }

    /// Generates independent literal-text lanes through the fair speculative
    /// scheduler. Each lane follows [`Self::generate_prepared_text`] admission
    /// and stopping rules; TTFT includes preparation and queueing from this call.
    pub fn generate_prepared_text_speculative_batch<'a>(
        &mut self,
        request: PreparedChatSpeculativeBatchRequest<
            'a,
            B,
            <B as SpeculativeGenerationBackend>::Drafter,
        >,
    ) -> Result<SpeculativeGenerationBatchOutput, PreparedChatSpeculativeError>
    where
        B: SpeculativeGenerationBackend,
    {
        self.generate_prepared_speculative_batch(request, PreparedGenerationMode::Text)
    }

    fn generate_prepared_speculative_batch<'a>(
        &mut self,
        request: PreparedChatSpeculativeBatchRequest<
            'a,
            B,
            <B as SpeculativeGenerationBackend>::Drafter,
        >,
        mode: PreparedGenerationMode,
    ) -> Result<SpeculativeGenerationBatchOutput, PreparedChatSpeculativeError>
    where
        B: SpeculativeGenerationBackend,
    {
        let driver = eredu_runtime::RunSpeculativeGeneration::new(request.scheduler);
        let PreparedChatSpeculativeBatchRequest {
            drafting,
            lanes,
            scheduler: _,
        } = request;
        // Validate every lane before preparing any backend prompt or execution.
        let generations = lanes
            .iter()
            .map(|lane| self.resolve_text_generation_settings(lane.settings))
            .collect::<Result<Vec<_>, _>>()?;
        let mut prepared_lanes = Vec::with_capacity(lanes.len());
        for (lane, generation) in lanes.into_iter().zip(generations) {
            let (prompt, generation, config, constraint, semantic) = self
                .prepare_speculative_generation(
                    lane.input,
                    generation,
                    lane.max_draft_tokens,
                    lane.caller_stop_sequences,
                    mode,
                )?;
            prepared_lanes.push(SpeculativeGenerationLane::new(
                prompt,
                generation,
                config,
                constraint,
                semantic,
                lane.cancellation,
                lane.on_event,
            ));
        }
        B::with_speculative_execution(
            &mut self.runtime,
            SpeculativeGenerationBatchRequest::new(
                drafting,
                prepared_lanes,
                self.tokenizer_fingerprint,
            ),
            driver,
        )
        .map_err(|error| {
            PreparedChatSpeculativeError::Backend(eredu_core::BackendFailure::from_error(error))
        })
    }

    #[allow(clippy::type_complexity)]
    fn prepare_speculative_generation(
        &self,
        input: PreparedChatInput<'_, B>,
        generation: (eredu_core::TextGenerationConfig, NonZeroUsize),
        max_draft_tokens: NonZeroUsize,
        caller_stop_sequences: &[String],
        mode: PreparedGenerationMode,
    ) -> Result<
        (
            B::Prompt,
            eredu_core::TextGenerationConfig,
            eredu_core::generation::SpeculativeConfig,
            PreparedChatSpeculativeConstraint,
            Box<dyn eredu_core::SpeculativeSemanticState>,
        ),
        PreparedChatSpeculativeError,
    > {
        let prepared_chat = input.prepared_chat();
        let control = mode
            .prepare_control(
                prepared_chat,
                caller_stop_sequences,
                self.token_validity.clone(),
            )
            .map_err(|error| match error {
                PreparedChatSetupError::Constraint(error) => {
                    PreparedChatSpeculativeError::Constraint(error)
                }
                PreparedChatSetupError::Semantic(error) => {
                    PreparedChatSpeculativeError::Semantic(error)
                }
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
        let prompt = match input {
            PreparedChatInput::RenderedPrompt(prepared_chat) => {
                let token_ids = self.encode(prepared_chat.rendered_prompt(), false)?;
                B::prepare_text_prompt(self.runtime.backend(), token_ids).map_err(|error| {
                    PreparedChatSpeculativeError::Backend(eredu_core::BackendFailure::from_error(
                        error,
                    ))
                })?
            }
            PreparedChatInput::PreparedBackendInput { prompt, .. } => prompt,
        };
        Ok((
            prompt,
            generation,
            eredu_core::generation::SpeculativeConfig {
                max_tokens: max_tokens.get(),
                max_draft_tokens: max_draft_tokens.get(),
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
    ) -> Result<PlannedModel<B, F::Drafter>, PlannedModelLoadError>
    where
        F: eredu_core::ExecutionPlanBackendFactory<
            Backend = B,
            DrafterPreparation = eredu_architectures::ExternalAssistantPreparation,
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
    ) -> Result<PlannedModel<B, F::Drafter>, PlannedModelLoadError>
    where
        F: eredu_core::ExecutionPlanBackendFactory<
            Backend = B,
            DrafterPreparation = eredu_architectures::ExternalAssistantPreparation,
        >,
    {
        let artifact = artifact.as_ref();
        let inspection = eredu_architectures::configuration::inspect_artifact(artifact)
            .map_err(LoadedModelLoadError::Artifact)?;
        Self::load_inspected_execution_plan_with_text_options(
            factory,
            inspection,
            plan,
            text_options,
        )
    }

    /// Realizes a plan using the exact artifact inspection retained by the planner.
    /// Use [`Self::load_inspected_execution_plan_with_text_options`] to override
    /// the checkpoint chat template.
    pub fn load_inspected_execution_plan<F>(
        factory: &F,
        inspection: eredu_core::ArtifactInspection<
            eredu_architectures::processor_plan::ArtifactArchitecturePlan,
        >,
        plan: &eredu_core::ExecutionPlan,
    ) -> Result<PlannedModel<B, F::Drafter>, PlannedModelLoadError>
    where
        F: eredu_core::ExecutionPlanBackendFactory<
            Backend = B,
            DrafterPreparation = eredu_architectures::ExternalAssistantPreparation,
        >,
    {
        Self::load_inspected_execution_plan_with_text_options(
            factory,
            inspection,
            plan,
            TextModelOptions::default(),
        )
    }

    /// Realizes a retained artifact inspection with facade-owned text overrides.
    pub fn load_inspected_execution_plan_with_text_options<F>(
        factory: &F,
        mut inspection: eredu_core::ArtifactInspection<
            eredu_architectures::processor_plan::ArtifactArchitecturePlan,
        >,
        plan: &eredu_core::ExecutionPlan,
        text_options: TextModelOptions,
    ) -> Result<PlannedModel<B, F::Drafter>, PlannedModelLoadError>
    where
        F: eredu_core::ExecutionPlanBackendFactory<
            Backend = B,
            DrafterPreparation = eredu_architectures::ExternalAssistantPreparation,
        >,
    {
        let (tokenizer, config) = loaded_text_artifact(&inspection, text_options)
            .map_err(LoadedModelLoadError::Metadata)?;
        bind_gguf_special_token_ids(&mut inspection, &tokenizer)
            .map_err(LoadedModelLoadError::Metadata)?;
        let target_tokenizer_fingerprint =
            eredu_text::tokenizer::vocabulary_fingerprint(&tokenizer);
        let external_artifact = match plan.drafting() {
            DraftingPlan::External { model, .. } => {
                let preparation = eredu_architectures::prepare_external_assistant(model)
                    .map_err(LoadedModelLoadError::Artifact)?;
                let draft_tokenizer = super::tokenizer::load_tokenizer_for_kind(
                    preparation.tokenizer_model_kind(),
                    Path::new(model),
                )
                .map_err(LoadedModelLoadError::Metadata)?;
                let tokenizer_compatibility = TokenizerCompatibilityProof::prove(
                    target_tokenizer_fingerprint,
                    eredu_text::tokenizer::vocabulary_fingerprint(&draft_tokenizer),
                )
                .map_err(|error| eredu_core::AutomaticPlanningError::Invalid(error.to_string()))?;
                Some(ExternalDraftArtifact {
                    preparation,
                    tokenizer_compatibility,
                })
            }
            DraftingPlan::Disabled | DraftingPlan::Embedded { .. } => None,
            _ => return Err(LoadedModelLoadError::UnsupportedDraftingPlan.into()),
        };
        let selected = eredu_core::select_execution_plan_target(factory, plan, inspection)?;
        let external_artifact = eredu_core::select_execution_plan_drafting(
            factory,
            plan,
            &selected,
            external_artifact,
        )?;
        let realization = eredu_core::realize_execution_plan_target(factory, plan, selected)?;
        let runtime = match realization.into_runtime() {
            Ok(runtime) => runtime,
            Err(error) => return Err(map_model_load_error(error).into()),
        };
        let model = Self::from_runtime(runtime, tokenizer, config);
        let drafting = eredu_core::realize_execution_plan_drafting(
            factory,
            plan,
            model.runtime(),
            external_artifact,
        )?;
        Ok(PlannedModel::new(model, drafting, plan.drafting().clone()))
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
    ) -> Result<(PlannedModel<B, F::Drafter>, eredu_core::ExecutionPlanReport), PlannedModelLoadError>
    where
        F: eredu_core::ExecutionPlanBackendFactory<
                Backend = B,
                DrafterPreparation = eredu_architectures::ExternalAssistantPreparation,
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
    ) -> Result<(PlannedModel<B, F::Drafter>, eredu_core::ExecutionPlanReport), PlannedModelLoadError>
    where
        F: eredu_core::ExecutionPlanBackendFactory<
                Backend = B,
                DrafterPreparation = eredu_architectures::ExternalAssistantPreparation,
            > + eredu_core::AutomaticPlanningBackend<
                Inspection = eredu_core::ArtifactInspection<
                    eredu_architectures::processor_plan::ArtifactArchitecturePlan,
                >,
            >,
    {
        let retained = planner.plan_retained(factory, request)?;
        let (report, inspection) = retained.into_parts();
        let model = Self::load_inspected_execution_plan_with_text_options(
            factory,
            inspection,
            &report.plan,
            text_options,
        )?;
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
        let (tokenizer, config) = loaded_text_artifact(&inspection, text_options)?;
        Self::from_inspected(backend, inspection, options, tokenizer, config)
    }

    fn from_inspected(
        backend: B,
        mut inspection: eredu_core::ArtifactInspection<
            eredu_architectures::processor_plan::ArtifactArchitecturePlan,
        >,
        options: B::LoadOptions,
        tokenizer: ChatTokenizer,
        config: LoadedTextModelConfig,
    ) -> Result<Self, LoadedModelLoadError> {
        bind_gguf_special_token_ids(&mut inspection, &tokenizer)?;
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

fn map_model_load_error<E: std::error::Error + Send + Sync + 'static>(
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

fn bind_gguf_special_token_ids(
    inspection: &mut eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    tokenizer: &ChatTokenizer,
) -> Result<(), TextMetadataError> {
    use eredu_architectures::processor_plan::{GgufSpecialTokenIds, GgufSpecialTokenKind};

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
        GgufSpecialTokenKind::Qwen => GgufSpecialTokenIds::Qwen {
            image_token_id: required("<|image_pad|>")?,
            video_token_id: required("<|video_pad|>")?,
            vision_start_token_id: required("<|vision_start|>")?,
            vision_end_token_id: required("<|vision_end|>")?,
        },
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

fn loaded_text_artifact(
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

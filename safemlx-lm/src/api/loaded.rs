//! Loaded-model facade and prepared generation entry points.

use std::{num::NonZeroUsize, path::Path};

use safemlx_gguf::MetadataValue as GgufMetadataValue;
use safemlx_lm_core::{
    generation::{resolve_generation_config, FinishReason, SemanticEvent},
    ModelKind, MtpCapability, SpeculativeGenerationBackend, SpeculativeGenerationBatchOutput,
    SpeculativeGenerationBatchRequest, SpeculativeGenerationLane, SpeculativeGenerationOutput,
    SpeculativeGenerationRequest,
};
use safemlx_lm_utils::{
    gguf::GgufTokenizer,
    tokenizer::{
        chat_template_kwargs as inspect_chat_template_kwargs, ApplyChatTemplateArgs, Chat,
        ModelChatTemplate, Tokenizer as ChatTokenizer,
    },
};
use serde::Serialize;

use super::{
    LoadedModel, LoadedTextModelConfig, PreparedChat, PreparedChatError,
    PreparedChatGenerationOutput, PreparedChatGenerationRequest, PreparedChatGenerationSettings,
    PreparedChatInput, PreparedChatMtpBatchRequest, PreparedChatMtpError,
    PreparedChatMtpGenerationRequest, PreparedChatSpeculativeConstraint, TextDecoderError,
    TextMetadataError, TextModelError,
};
use crate::{
    api::{
        metadata::{
            eos_token_ids_from_sidecar_dir, gguf_eos_token_ids, merge_eos_token_id_sources,
            read_checkpoint_generation_config,
        },
        request::{
            prepare_chat_from_parts, prepared_chat_control_runtime, BackendGenerationTokenSource,
            PreparedChatSemanticState, PreparedChatSetupError, PreparedChatTokenDecoder,
        },
        tokenizer::{
            gguf_sidecar_dir, load_chat_template, load_gguf_tokenizer_from_metadata,
            load_tokenizer_for_kind, load_tokenizer_template_kwargs,
        },
    },
    runtime::{
        chat::{constraints::ConstraintCompiler, ChatTemplateIdentity, ChatTemplateRequest},
        generation::streaming::{
            drive_committed_generation_cancellable, CommittedGenerationError,
            CommittedTokenPipeline, CommittedTokenPipelineError, RawTokenDecoder,
        },
    },
};

/// Failure while assembling a tokenizer-aware model around a selected backend.
#[derive(Debug, thiserror::Error)]
pub enum LoadedModelLoadError<E: std::error::Error + Send + Sync + 'static> {
    /// Portable artifact inspection or preparation planning failed.
    #[error(transparent)]
    Artifact(#[from] safemlx_lm_core::artifact::ArtifactError),
    /// Backend materialization or session creation failed.
    #[error("selected backend failed to load the model: {0}")]
    Backend(#[source] E),
    /// Portable tokenizer, chat-template, or generation sidecar loading failed.
    #[error(transparent)]
    Metadata(#[from] TextMetadataError),
}

fn map_prepared_chat_setup_error<E>(error: PreparedChatSetupError) -> PreparedChatError<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    match error {
        PreparedChatSetupError::Constraint(error) => PreparedChatError::Constraint(error),
        PreparedChatSetupError::Semantic(error) => PreparedChatError::Semantic(error),
    }
}

fn map_controlled_generation_error<E>(
    error: safemlx_lm_core::ControlledTextGenerationError<
        E,
        crate::runtime::chat::constraints::ConstraintError,
    >,
) -> PreparedChatError<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    match error {
        safemlx_lm_core::ControlledTextGenerationError::Backend(error) => {
            PreparedChatError::Backend(error)
        }
        safemlx_lm_core::ControlledTextGenerationError::Controller(error) => {
            PreparedChatError::Constraint(error)
        }
    }
}

fn map_committed_generation_error<E>(
    error: CommittedGenerationError<
        safemlx_lm_core::ControlledTextGenerationError<
            E,
            crate::runtime::chat::constraints::ConstraintError,
        >,
        TextDecoderError,
    >,
) -> PreparedChatError<E>
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

impl<B: safemlx_lm_core::TextGenerationBackend> LoadedModel<B> {
    fn resolve_text_generation_settings(
        &self,
        settings: PreparedChatGenerationSettings,
    ) -> Result<
        (safemlx_lm_core::TextGenerationConfig, NonZeroUsize),
        crate::core::generation::GenerationError,
    > {
        let resolved = resolve_generation_config(
            self.checkpoint_generation_config.as_ref(),
            settings.overrides,
        )?;
        let max_tokens = resolved
            .max_new_tokens
            .and_then(NonZeroUsize::new)
            .unwrap_or_else(|| NonZeroUsize::new(256).expect("256 is non-zero"));
        Ok((
            safemlx_lm_core::TextGenerationConfig::new(resolved).with_seed(settings.seed),
            max_tokens,
        ))
    }

    /// Generates one constrained semantic response through the selected backend.
    pub fn generate_prepared_chat<F>(
        &mut self,
        request: PreparedChatGenerationRequest<'_, B, F>,
    ) -> Result<PreparedChatGenerationOutput, PreparedChatError<B::Error>>
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
            return Ok(PreparedChatGenerationOutput {
                token_ids: Vec::new(),
                finish_reason: FinishReason::Cancelled,
            });
        }

        let prepared_chat = input.prepared_chat();
        let (config, max_tokens) = self.resolve_text_generation_settings(settings)?;
        let control = prepared_chat_control_runtime(prepared_chat, caller_stop_sequences)
            .map_err(map_prepared_chat_setup_error)?;
        let decoder = PreparedChatTokenDecoder {
            decoder: self.text_decoder(true),
        };
        let raw_decoder =
            RawTokenDecoder::with_structural_tokens(decoder, control.structural_tokens);
        let mut pipeline = CommittedTokenPipeline::new(raw_decoder, control.parser);
        let generator = match input {
            PreparedChatInput::RenderedPrompt(prepared_chat) => {
                let prompt = self
                    .tokenizer
                    .encode(prepared_chat.rendered_prompt(), false)
                    .map_err(TextDecoderError::Tokenizer)?
                    .get_ids()
                    .to_vec();
                safemlx_lm_core::ControlledTextGeneration::new(
                    &mut self.runtime,
                    prompt,
                    config,
                    control.controller,
                )
            }
            PreparedChatInput::PreparedBackendInput { prompt, .. } => {
                safemlx_lm_core::ControlledTextGeneration::from_prompt(
                    &mut self.runtime,
                    prompt,
                    config,
                    control.controller,
                )
            }
        }
        .map_err(map_controlled_generation_error)?;
        let mut source = BackendGenerationTokenSource { generator };
        let (token_ids, finish_reason) = drive_committed_generation_cancellable(
            &mut source,
            &mut pipeline,
            prepared_chat.eos_token_ids(),
            max_tokens,
            &cancellation,
            &mut on_event,
        )
        .map_err(map_committed_generation_error)?;
        Ok(PreparedChatGenerationOutput {
            token_ids,
            finish_reason,
        })
    }

    /// Reports fail-closed speculative support for this backend model session.
    pub fn mtp_capability(&self) -> MtpCapability
    where
        B: SpeculativeGenerationBackend,
    {
        B::mtp_capability(&self.runtime)
    }

    /// Generates one structured response using embedded or external drafting.
    pub fn generate_prepared_chat_mtp<'a, F>(
        &mut self,
        request: PreparedChatMtpGenerationRequest<
            'a,
            B,
            <B as SpeculativeGenerationBackend>::Drafter,
            F,
        >,
    ) -> Result<SpeculativeGenerationOutput, PreparedChatMtpError<B::Error>>
    where
        B: SpeculativeGenerationBackend,
        F: FnMut(SemanticEvent),
    {
        let PreparedChatMtpGenerationRequest {
            input,
            drafting,
            settings,
            options,
            caller_stop_sequences,
            cancellation,
            on_event,
        } = request;
        let (prompt, generation, config, constraint, semantic) = self.prepare_speculative_chat(
            input,
            settings,
            options.max_draft_tokens,
            caller_stop_sequences,
        )?;
        B::execute_speculative(
            &mut self.runtime,
            SpeculativeGenerationRequest {
                prompt,
                drafting,
                generation,
                config,
                constraint,
                semantic,
                scheduler: options.scheduler,
                cancellation,
                tokenizer_fingerprint: self.tokenizer_fingerprint,
                on_event,
            },
        )
        .map_err(PreparedChatMtpError::Backend)
    }

    /// Generates independent prepared chats through one fair speculative scheduler.
    pub fn generate_prepared_chat_mtp_batch<'a>(
        &mut self,
        request: PreparedChatMtpBatchRequest<'a, B, <B as SpeculativeGenerationBackend>::Drafter>,
    ) -> Result<SpeculativeGenerationBatchOutput, PreparedChatMtpError<B::Error>>
    where
        B: SpeculativeGenerationBackend,
    {
        let PreparedChatMtpBatchRequest {
            drafting,
            lanes,
            scheduler,
        } = request;
        let mut prepared_lanes = Vec::with_capacity(lanes.len());
        for lane in lanes {
            let (prompt, generation, config, constraint, semantic) = self
                .prepare_speculative_chat(
                    lane.input,
                    lane.settings,
                    lane.max_draft_tokens,
                    lane.caller_stop_sequences,
                )?;
            prepared_lanes.push(SpeculativeGenerationLane {
                prompt,
                generation,
                config,
                constraint,
                semantic,
                cancellation: lane.cancellation,
                on_event: lane.on_event,
            });
        }
        B::execute_speculative_batch(
            &mut self.runtime,
            SpeculativeGenerationBatchRequest {
                drafting,
                lanes: prepared_lanes,
                tokenizer_fingerprint: self.tokenizer_fingerprint,
                scheduler,
            },
        )
        .map_err(PreparedChatMtpError::Backend)
    }

    fn prepare_speculative_chat(
        &self,
        input: PreparedChatInput<'_, B>,
        settings: PreparedChatGenerationSettings,
        max_draft_tokens: NonZeroUsize,
        caller_stop_sequences: &[String],
    ) -> Result<
        (
            B::Prompt,
            safemlx_lm_core::TextGenerationConfig,
            crate::core::generation::MtpConfig,
            PreparedChatSpeculativeConstraint,
            Box<dyn safemlx_lm_core::SpeculativeSemanticState>,
        ),
        PreparedChatMtpError<B::Error>,
    > {
        let prepared_chat = input.prepared_chat();
        let semantic_plan = match prepared_chat.semantic_support() {
            crate::runtime::chat::SemanticSupport::Supported => prepared_chat
                .semantic_runtime_plan()
                .expect("supported prepared chats carry a semantic runtime plan")
                .clone(),
            crate::runtime::chat::SemanticSupport::Unsupported { reason } => {
                return Err(PreparedChatMtpError::Semantic(format!(
                    "prepared chat does not have an executable semantic plan: {reason}"
                )));
            }
        };
        let constraint = PreparedChatSpeculativeConstraint::from_prepared_chat(prepared_chat)?;
        let semantic = PreparedChatSemanticState::new(
            PreparedChatTokenDecoder {
                decoder: self.text_decoder(true),
            },
            semantic_plan,
            caller_stop_sequences,
        )
        .map_err(|error| PreparedChatMtpError::Semantic(error.to_string()))?;
        let eos_token_ids = prepared_chat.eos_token_ids().to_vec();
        let (generation, max_tokens) = self.resolve_text_generation_settings(settings)?;
        let temperature = generation.sampling().temperature;
        let prompt = match input {
            PreparedChatInput::RenderedPrompt(prepared_chat) => {
                let token_ids = self.encode(prepared_chat.rendered_prompt(), false)?;
                B::prepare_text_prompt(self.runtime.backend(), token_ids)
                    .map_err(PreparedChatMtpError::Backend)?
            }
            PreparedChatInput::PreparedBackendInput { prompt, .. } => prompt,
        };
        Ok((
            prompt,
            generation,
            crate::core::generation::MtpConfig {
                max_tokens: max_tokens.get(),
                max_draft_tokens: max_draft_tokens.get(),
                temperature,
                eos_token_ids,
            },
            constraint,
            Box::new(semantic),
        ))
    }

    /// Returns the model id passed to chat-template rendering.
    pub fn model_id_for_template(&self) -> &str {
        &self.model_id
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

    /// Prepares one JSON-valued chat for generation.
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

    /// Applies the selected chat template to structured conversations.
    pub fn apply_chat_template<'a, I, R, T>(
        &'a mut self,
        conversations: I,
        tools: Option<&'a [serde_json::Value]>,
        add_generation_prompt: bool,
    ) -> Result<Option<String>, TextModelError>
    where
        I: IntoIterator<Item = Chat<'a, R, T>>,
        R: Serialize + 'a,
        T: Serialize + 'a,
    {
        self.apply_chat_template_with_kwargs(conversations, tools, add_generation_prompt, None)
    }

    /// Applies the selected chat template with extra template variables.
    pub fn apply_chat_template_with_kwargs<'a, I, R, T>(
        &'a mut self,
        conversations: I,
        tools: Option<&'a [serde_json::Value]>,
        add_generation_prompt: bool,
        template_kwargs: Option<&'a serde_json::Map<String, serde_json::Value>>,
    ) -> Result<Option<String>, TextModelError>
    where
        I: IntoIterator<Item = Chat<'a, R, T>>,
        R: Serialize + 'a,
        T: Serialize + 'a,
    {
        let Some(template) = self.chat_template.clone() else {
            return Ok(None);
        };
        let rendered = self.tokenizer.apply_chat_template(
            template,
            ApplyChatTemplateArgs {
                conversations,
                tools,
                documents: None,
                model_id: &self.model_id,
                chat_template_id: None,
                add_generation_prompt: Some(add_generation_prompt),
                continue_final_message: None,
                template_kwargs,
            },
        )?;
        Ok(rendered.into_iter().next())
    }

    /// Applies the selected chat template to JSON-valued conversations.
    pub fn apply_chat_template_json(
        &mut self,
        conversations: impl IntoIterator<Item = Vec<serde_json::Value>>,
        tools: Option<&[serde_json::Value]>,
        add_generation_prompt: bool,
    ) -> Result<Option<String>, TextModelError> {
        self.apply_chat_template_json_with_kwargs(conversations, tools, add_generation_prompt, None)
    }

    /// Applies the selected chat template with extra JSON template variables.
    pub fn apply_chat_template_json_with_kwargs(
        &mut self,
        conversations: impl IntoIterator<Item = Vec<serde_json::Value>>,
        tools: Option<&[serde_json::Value]>,
        add_generation_prompt: bool,
        template_kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Result<Option<String>, TextModelError> {
        let Some(template) = self.chat_template.clone() else {
            return Ok(None);
        };
        let rendered = self.tokenizer.apply_chat_template_json(
            template,
            conversations,
            tools,
            &self.model_id,
            add_generation_prompt,
            template_kwargs,
        )?;
        Ok(rendered.into_iter().next())
    }
}

impl<B> LoadedModel<B>
where
    B: safemlx_lm_core::TextGenerationBackend + safemlx_lm_core::ModelLoadingBackend,
{
    /// Loads one artifact, its tokenizer, and its chat metadata on `backend`.
    ///
    /// The backend already owns device placement, execution queues, transfer
    /// queues, and optional communication. Artifact inspection occurs exactly
    /// once and is shared by portable metadata assembly and backend planning.
    pub fn load(
        backend: B,
        artifact: impl AsRef<Path>,
        options: B::LoadOptions,
    ) -> Result<Self, LoadedModelLoadError<B::Error>> {
        let artifact = artifact.as_ref();
        let inspection = safemlx_lm_core::inspect_artifact(artifact)?;
        let (tokenizer, config) = loaded_text_artifact(&inspection)?;
        let prepared = match safemlx_lm_core::prepare_inspected_model(&backend, inspection, options)
        {
            Ok(prepared) => prepared,
            Err(safemlx_lm_core::ModelLoadError::Artifact(error)) => {
                return Err(LoadedModelLoadError::Artifact(error));
            }
            Err(safemlx_lm_core::ModelLoadError::Backend(error)) => {
                return Err(LoadedModelLoadError::Backend(error));
            }
        };
        let runtime = safemlx_lm_core::ModelRuntime::from_prepared(backend, prepared)
            .map_err(LoadedModelLoadError::Backend)?;
        Ok(Self::from_runtime(runtime, tokenizer, config))
    }
}

fn loaded_text_artifact(
    inspection: &safemlx_lm_core::ArtifactInspection,
) -> Result<(ChatTokenizer, LoadedTextModelConfig), TextMetadataError> {
    let path = inspection.path();
    let configuration = inspection.configuration();
    if configuration.kind == ModelKind::PersonaPlex {
        return Err(TextMetadataError::UnsupportedArchitecture(
            "PersonaPlex is a realtime speech-to-speech token model; use the realtime backend contract instead of LoadedModel".into(),
        ));
    }

    let sidecar_dir = match inspection.format() {
        safemlx_lm_core::ArtifactFormat::SafeTensors => path,
        safemlx_lm_core::ArtifactFormat::Gguf => gguf_sidecar_dir(path),
    };
    let checkpoint_generation_config = read_checkpoint_generation_config(sidecar_dir)?;
    let sidecar_eos_token_ids = eos_token_ids_from_sidecar_dir(sidecar_dir)?;

    let (mut tokenizer, chat_template, eos_token_ids, model_id) = match inspection.format() {
        safemlx_lm_core::ArtifactFormat::SafeTensors => {
            let tokenizer = load_tokenizer_for_kind(configuration.kind, path)?;
            (
                ChatTokenizer::from_tokenizer(tokenizer),
                load_chat_template(path)?,
                sidecar_eos_token_ids,
                configuration.effective_model_type.clone(),
            )
        }
        safemlx_lm_core::ArtifactFormat::Gguf => {
            let metadata = inspection
                .gguf_checkpoint()
                .expect("GGUF inspection owns a portable checkpoint")
                .metadata()
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect::<std::collections::HashMap<_, _>>();
            let embedded_chat_template = match metadata.get("tokenizer.chat_template") {
                Some(GgufMetadataValue::String(template)) => {
                    Some(ModelChatTemplate::Single(template.clone()))
                }
                Some(_) => {
                    return Err(TextMetadataError::GgufTokenizer(
                        "tokenizer.chat_template must be a string".into(),
                    ));
                }
                None => None,
            };
            let GgufTokenizer {
                tokenizer,
                template_kwargs,
            } = load_gguf_tokenizer_from_metadata(path, &metadata)?;
            let mut tokenizer = ChatTokenizer::from_tokenizer(tokenizer);
            tokenizer.set_template_kwargs(template_kwargs);
            (
                tokenizer,
                embedded_chat_template.or(load_chat_template(sidecar_dir)?),
                merge_eos_token_id_sources([sidecar_eos_token_ids, gguf_eos_token_ids(&metadata)?]),
                path.display().to_string(),
            )
        }
    };
    if inspection.format() == safemlx_lm_core::ArtifactFormat::SafeTensors {
        tokenizer.set_template_kwargs(load_tokenizer_template_kwargs(path)?);
    }

    Ok((
        tokenizer,
        LoadedTextModelConfig {
            model_type: configuration.effective_model_type.clone(),
            model_id,
            chat_template,
            eos_token_ids,
            checkpoint_generation_config,
        },
    ))
}

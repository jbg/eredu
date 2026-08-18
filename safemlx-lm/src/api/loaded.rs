//! Loaded-model facade and prepared generation entry points.

use std::{num::NonZeroUsize, path::Path};

use safemlx_gguf::MetadataValue as GgufMetadataValue;
use safemlx_lm_core::{
    generation::{resolve_generation_config, FinishReason, SemanticEvent},
    ModelKind, MtpCapability,
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
    PreparedChatInput, PreparedChatMtpBatchExecutionRequest, PreparedChatMtpBatchOutput,
    PreparedChatMtpBatchRequest, PreparedChatMtpError, PreparedChatMtpExecutionLane,
    PreparedChatMtpExecutionRequest, PreparedChatMtpGenerationOutput,
    PreparedChatMtpGenerationRequest, PreparedChatSpeculativeBackend,
    PreparedChatSpeculativeConstraint, TextDecoderError, TextMetadataError, TextModelError,
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
        B: PreparedChatSpeculativeBackend,
    {
        B::mtp_capability(self)
    }

    /// Generates one structured response using embedded or external drafting.
    pub fn generate_prepared_chat_mtp<'a, F>(
        &mut self,
        request: PreparedChatMtpGenerationRequest<'a, B, B::Drafter, F>,
    ) -> Result<PreparedChatMtpGenerationOutput, PreparedChatMtpError<B::Error>>
    where
        B: PreparedChatSpeculativeBackend,
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
        B::execute_prepared_chat_mtp(
            self,
            PreparedChatMtpExecutionRequest {
                prompt,
                drafting,
                generation,
                config,
                constraint,
                semantic,
                scheduler: options.scheduler,
                cancellation,
                on_event,
            },
        )
        .map_err(PreparedChatMtpError::Backend)
    }

    /// Generates independent prepared chats through one fair speculative scheduler.
    pub fn generate_prepared_chat_mtp_batch<'a>(
        &mut self,
        request: PreparedChatMtpBatchRequest<'a, B, B::Drafter>,
    ) -> Result<PreparedChatMtpBatchOutput, PreparedChatMtpError<B::Error>>
    where
        B: PreparedChatSpeculativeBackend,
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
            prepared_lanes.push(PreparedChatMtpExecutionLane {
                prompt,
                generation,
                config,
                constraint,
                semantic,
                cancellation: lane.cancellation,
                on_event: lane.on_event,
            });
        }
        B::execute_prepared_chat_mtp_batch(
            self,
            PreparedChatMtpBatchExecutionRequest {
                drafting,
                lanes: prepared_lanes,
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

#[cfg(feature = "mlx")]
use crate::api::request::PreparedChatDraft;
#[cfg(feature = "mlx")]
use crate::architectures::deepseek_v3::model as deepseek_v3;
#[cfg(feature = "mlx")]
use crate::architectures::gemma4::model as gemma4;
#[cfg(feature = "mlx")]
use crate::architectures::inkling::model as inkling;
#[cfg(feature = "mlx")]
use crate::architectures::nemotron_h::model as nemotron_h;
#[cfg(feature = "mlx")]
use crate::architectures::qwen::hybrid::qwen3_5;
#[cfg(feature = "mlx")]
use crate::backend::mlx::speculative::{
    scheduler::MlxMtpScheduler, MlxDrafter, MlxDrafterKind, MlxMtpCache, MtpExecutionStreams,
};
#[cfg(feature = "mlx")]
use crate::backend::mlx::{validate_gemma4_drafter, Model, ModelCache};
#[cfg(feature = "mlx")]
use crate::core::generation::{GenerationCancellationToken, MtpConfig, MtpSchedulerOptions};
#[cfg(feature = "mlx")]
use crate::core::SpeculativeSemanticState;
#[cfg(feature = "mlx")]
use crate::error::Error;
#[cfg(feature = "mlx")]
use crate::runtime::generation::sampler::ConstrainedSampler;
#[cfg(feature = "mlx")]
use safemlx::{error::Exception, Array};

#[cfg(feature = "mlx")]
struct PreparedChatMtpLaneRuntime<'a> {
    input: crate::backend::mlx::MlxModelInput,
    cache: &'a mut ModelCache,
    config: MtpConfig,
    prng_key: Option<Array>,
    sampler: ConstrainedSampler<crate::runtime::generation::sampler::GenerationSampler>,
    semantic: Box<dyn SpeculativeSemanticState>,
    cancellation: GenerationCancellationToken,
    on_event: Box<dyn FnMut(SemanticEvent) + 'a>,
}

#[cfg(feature = "mlx")]
fn run_prepared_chat_mtp_batch<'a, B>(
    backend: &'a mut B,
    lanes: Vec<PreparedChatMtpLaneRuntime<'a>>,
    cache_for_lane: fn(&mut ModelCache) -> Option<&mut B::Cache>,
    cache_kind: &str,
    streams: MtpExecutionStreams<'a>,
    options: MtpSchedulerOptions,
) -> Result<PreparedChatMtpBatchOutput, Exception>
where
    B: crate::backend::mlx::speculative::scheduler::MlxSpeculativeRuntime<'a>,
{
    let mut scheduler = MlxMtpScheduler::new(backend, streams, options)?;
    for (lane_index, lane) in lanes.into_iter().enumerate() {
        let PreparedChatMtpLaneRuntime {
            input,
            cache,
            config,
            prng_key,
            sampler,
            semantic,
            cancellation,
            on_event,
        } = lane;
        let cache = cache_for_lane(cache).ok_or_else(|| {
            Exception::custom(format!(
                "prepared-chat {cache_kind} MTP cache type mismatch at lane {lane_index}"
            ))
        })?;
        input.with_borrowed(|input| {
            scheduler.submit_with_semantics_cancellable(
                cache,
                input,
                config,
                prng_key,
                sampler,
                semantic,
                cancellation,
                on_event,
            )
        })?;
    }
    scheduler.run()?;
    let output = scheduler.finish()?;
    let requests = output
        .requests
        .into_iter()
        .map(|request| {
            let finish_reason = request.finish_reason.ok_or_else(|| {
                Exception::custom(format!(
                    "completed prepared-chat MTP request {} has no finish reason",
                    request.id.index()
                ))
            })?;
            Ok(PreparedChatMtpGenerationOutput {
                token_ids: request.token_ids,
                finish_reason,
                stats: request.stats,
            })
        })
        .collect::<Result<Vec<_>, Exception>>()?;
    Ok(PreparedChatMtpBatchOutput {
        requests,
        scheduler: output.scheduler,
    })
}

#[cfg(feature = "mlx")]
fn gemma4_mtp_cache(cache: &mut ModelCache) -> Option<&mut gemma4::Cache> {
    match cache {
        ModelCache::Gemma4(cache) => Some(cache),
        _ => None,
    }
}

#[cfg(feature = "mlx")]
fn model_mtp_cache(cache: &mut ModelCache) -> Option<&mut ModelCache> {
    Some(cache)
}

#[cfg(feature = "mlx")]
fn qwen_next_mtp_cache(cache: &mut ModelCache) -> Option<&mut qwen3_5::Cache> {
    match cache {
        ModelCache::Qwen3Next(cache) => Some(cache),
        _ => None,
    }
}

#[cfg(feature = "mlx")]
fn qwen35_mtp_cache(cache: &mut ModelCache) -> Option<&mut qwen3_5::Cache> {
    match cache {
        ModelCache::Qwen35(cache) => Some(cache),
        _ => None,
    }
}

#[cfg(feature = "mlx")]
fn deepseek_mtp_cache(cache: &mut ModelCache) -> Option<&mut deepseek_v3::Cache> {
    match cache {
        ModelCache::DeepSeekV3(cache) => Some(cache),
        _ => None,
    }
}

#[cfg(feature = "mlx")]
fn inkling_mtp_cache(cache: &mut ModelCache) -> Option<&mut inkling::Cache> {
    match cache {
        ModelCache::Inkling(cache) => Some(cache),
        _ => None,
    }
}

#[cfg(feature = "mlx")]
fn nemotron_mtp_cache(cache: &mut ModelCache) -> Option<&mut nemotron_h::Cache> {
    match cache {
        ModelCache::NemotronH(cache) => Some(cache),
        _ => None,
    }
}

#[cfg(feature = "mlx")]
impl LoadedModel<crate::backend::mlx::MlxBackend<'static>> {
    fn model(&self) -> &Model {
        self.runtime.session().complete_model()
    }

    fn model_and_cache(&mut self) -> (&mut Model, &mut ModelCache) {
        self.runtime.session_mut().complete_parts_mut()
    }

    fn prepare_mlx_speculative_sampling(
        generation: safemlx_lm_core::TextGenerationConfig,
        constraint: PreparedChatSpeculativeConstraint,
    ) -> Result<
        (
            Option<Array>,
            ConstrainedSampler<crate::runtime::generation::sampler::GenerationSampler>,
        ),
        Error,
    > {
        let resolved = generation.sampling();
        let prng_key = (resolved.temperature != 0.0)
            .then(|| safemlx::random::key(generation.seed()))
            .transpose()?;
        Ok((
            prng_key,
            ConstrainedSampler::with_controller(
                crate::runtime::generation::sampler::GenerationSampler::from_resolved(resolved),
                constraint.into_controller(),
            ),
        ))
    }

    /// Validates the observable target/assistant contract used by external MTP.
    ///
    /// Repository names and revisions are deliberately not compatibility keys.
    /// The validation covers the target architecture, shared tensor geometry,
    /// and the token-id vocabulary mapping when the drafter carries tokenizer
    /// metadata.
    fn validate_drafter_compatibility(&self, drafter: &MlxDrafter) -> Result<(), Error> {
        match (self.model(), drafter.kind()) {
            (Model::Gemma4(target), MlxDrafterKind::Gemma4Assistant) => {
                validate_gemma4_drafter(target.args(), drafter.gemma4())?
            }
            (Model::MuseGlimmer(target), MlxDrafterKind::MuseGlimmerDFlash) => {
                let assistant = drafter.muse_glimmer();
                let target_args = target.args();
                if assistant.config.hidden_size != target_args.hidden_size
                    || assistant
                        .config
                        .target_layer_ids
                        .iter()
                        .any(|layer| *layer >= target_args.num_hidden_layers as usize)
                    || assistant.config.mask_token_id >= target_args.vocab_size as u32
                    || assistant.config.block_size != 16
                {
                    return Err(Error::UnsupportedArchitecture(
                    "Muse-Glimmer DFlash hidden geometry, layer mapping, mask token, or block size does not match the target"
                        .into(),
                ));
                }
            }
            (model, kind) => {
                return Err(Error::UnsupportedArchitecture(format!(
                    "drafter {kind:?} is incompatible with target {} ({:?})",
                    model.model_type(),
                    model.mtp_capability()
                )))
            }
        }
        if drafter.tokenizer_fingerprint() != self.tokenizer_fingerprint {
            return Err(Error::UnsupportedArchitecture(
                "assistant token-id vocabulary mapping does not match the target".into(),
            ));
        }
        Ok(())
    }

    fn prepare_chat_mtp_batch_lanes<'a>(
        &self,
        lanes: Vec<PreparedChatMtpExecutionLane<'a, crate::backend::mlx::MlxBackend<'static>>>,
        cache: &'a mut MlxMtpCache,
    ) -> Result<Vec<PreparedChatMtpLaneRuntime<'a>>, Error> {
        if cache.len() != lanes.len() {
            return Err(Error::Speculative(format!(
                "MTP cache has {} lanes but the request has {} lanes",
                cache.len(),
                lanes.len()
            )));
        }
        let mut prepared_lanes = Vec::with_capacity(lanes.len());
        for (lane, cache) in lanes.into_iter().zip(cache.lanes.iter_mut()) {
            let PreparedChatMtpExecutionLane {
                prompt,
                generation,
                config,
                constraint,
                semantic,
                cancellation,
                on_event,
            } = lane;
            let (prng_key, sampler) =
                Self::prepare_mlx_speculative_sampling(generation, constraint)?;
            prepared_lanes.push(PreparedChatMtpLaneRuntime {
                input: prompt,
                cache,
                config,
                prng_key,
                sampler,
                semantic,
                cancellation,
                on_event,
            });
        }
        Ok(prepared_lanes)
    }

    /// Generates multiple independent prepared chats through one fair MTP scheduler.
    pub(crate) fn execute_prepared_chat_mtp_batch_mlx(
        &mut self,
        request: PreparedChatMtpBatchExecutionRequest<
            '_,
            crate::backend::mlx::MlxBackend<'static>,
            MlxDrafter,
        >,
    ) -> Result<PreparedChatMtpBatchOutput, Error> {
        let PreparedChatMtpBatchExecutionRequest {
            drafting,
            lanes,
            scheduler,
        } = request;
        match drafting {
            PreparedChatDraft::External(drafter) => {
                self.generate_prepared_chat_mtp_batch_with_external_draft(drafter, lanes, scheduler)
            }
            PreparedChatDraft::Embedded => {
                self.generate_prepared_chat_mtp_batch_with_embedded_draft(lanes, scheduler)
            }
        }
    }

    fn generate_prepared_chat_mtp_batch_with_external_draft(
        &mut self,
        drafter: &mut MlxDrafter,
        lanes: Vec<PreparedChatMtpExecutionLane<'_, crate::backend::mlx::MlxBackend<'static>>>,
        scheduler: MtpSchedulerOptions,
    ) -> Result<PreparedChatMtpBatchOutput, Error> {
        self.validate_drafter_compatibility(drafter)?;
        let target_stream = self.runtime.backend().stream().clone();
        let draft_stream = drafter.stream().clone();
        let streams = MtpExecutionStreams::new(&target_stream, &draft_stream)?;
        let mut cache = self.new_mtp_cache(lanes.len());
        let prepared_lanes = self.prepare_chat_mtp_batch_lanes(lanes, &mut cache)?;

        match self.model_and_cache().0 {
            Model::Gemma4(target) => {
                let assistant = drafter.gemma4_mut();
                validate_gemma4_drafter(target.args(), assistant)?;
                let mut backend =
                    crate::architectures::gemma4::mtp::Gemma4MtpExecutor::new(target, assistant);
                run_prepared_chat_mtp_batch(
                    &mut backend,
                    prepared_lanes,
                    gemma4_mtp_cache,
                    "Gemma 4",
                    streams,
                    scheduler,
                )
                .map_err(|error| Error::Speculative(error.to_string()))
            }
            Model::MuseGlimmer(target) => {
                let assistant = drafter.muse_glimmer_mut();
                let mut backend =
                    crate::architectures::muse_glimmer::mtp::MuseGlimmerMtpExecutor::new(
                        target, assistant,
                    );
                run_prepared_chat_mtp_batch(
                    &mut backend,
                    prepared_lanes,
                    model_mtp_cache,
                    "Muse-Glimmer DFlash",
                    streams,
                    scheduler,
                )
                .map_err(|error| Error::Speculative(error.to_string()))
            }
            model => Err(Error::Speculative(format!(
                "MTP runtime adapter is unavailable for model type {} ({:?})",
                model.model_type(),
                model.mtp_capability()
            ))),
        }
    }

    fn generate_prepared_chat_mtp_batch_with_embedded_draft(
        &mut self,
        lanes: Vec<PreparedChatMtpExecutionLane<'_, crate::backend::mlx::MlxBackend<'static>>>,
        scheduler: MtpSchedulerOptions,
    ) -> Result<PreparedChatMtpBatchOutput, Error> {
        let stream = self.runtime.backend().stream().clone();
        let mut cache = self.new_mtp_cache(lanes.len());
        let prepared_lanes = self.prepare_chat_mtp_batch_lanes(lanes, &mut cache)?;
        let streams = MtpExecutionStreams::single(&stream);
        match self.model_and_cache().0 {
            Model::DeepSeekV3(target) => {
                let mut backend =
                    crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(target);
                run_prepared_chat_mtp_batch(
                    &mut backend,
                    prepared_lanes,
                    deepseek_mtp_cache,
                    "DeepSeek embedded",
                    streams,
                    scheduler,
                )
                .map_err(|error| Error::Speculative(error.to_string()))
            }
            Model::Inkling(target) => {
                let mut backend =
                    crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(target);
                run_prepared_chat_mtp_batch(
                    &mut backend,
                    prepared_lanes,
                    inkling_mtp_cache,
                    "Inkling embedded",
                    streams,
                    scheduler,
                )
                .map_err(|error| Error::Speculative(error.to_string()))
            }
            Model::NemotronH(target) => {
                let mut backend =
                    crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(target);
                run_prepared_chat_mtp_batch(
                    &mut backend,
                    prepared_lanes,
                    nemotron_mtp_cache,
                    "Nemotron-H embedded",
                    streams,
                    scheduler,
                )
                .map_err(|error| Error::Speculative(error.to_string()))
            }
            Model::Qwen3Next(target) => {
                let mut backend =
                    crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(target);
                run_prepared_chat_mtp_batch(
                    &mut backend,
                    prepared_lanes,
                    qwen_next_mtp_cache,
                    "Qwen3-Next embedded",
                    streams,
                    scheduler,
                )
                .map_err(|error| Error::Speculative(error.to_string()))
            }
            Model::Qwen35(target) => {
                let mut backend =
                    crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(target);
                run_prepared_chat_mtp_batch(
                    &mut backend,
                    prepared_lanes,
                    qwen35_mtp_cache,
                    "Qwen3.5 embedded",
                    streams,
                    scheduler,
                )
                .map_err(|error| Error::Speculative(error.to_string()))
            }
            model => Err(Error::Speculative(format!(
            "scheduled prepared-chat embedded MTP batch is unavailable for model type {} ({:?})",
            model.model_type(),
            model.mtp_capability()
        ))),
        }
    }

    /// Generates one structured response using embedded or external drafting.
    pub(crate) fn execute_prepared_chat_mtp_mlx<'a, F>(
        &mut self,
        request: PreparedChatMtpExecutionRequest<
            'a,
            crate::backend::mlx::MlxBackend<'static>,
            MlxDrafter,
            F,
        >,
    ) -> Result<PreparedChatMtpGenerationOutput, Error>
    where
        F: FnMut(SemanticEvent),
    {
        let PreparedChatMtpExecutionRequest {
            prompt,
            drafting,
            generation,
            config,
            constraint,
            semantic,
            scheduler,
            cancellation,
            on_event,
        } = request;
        let (prng_key, sampler) = Self::prepare_mlx_speculative_sampling(generation, constraint)?;
        match drafting {
            PreparedChatDraft::External(drafter) => self
                .generate_prepared_chat_mtp_with_external_draft(
                    prompt,
                    drafter,
                    config,
                    prng_key,
                    sampler,
                    semantic,
                    scheduler,
                    cancellation,
                    on_event,
                ),
            PreparedChatDraft::Embedded => self.generate_prepared_chat_mtp_with_embedded_draft(
                prompt,
                config,
                prng_key,
                sampler,
                semantic,
                scheduler,
                cancellation,
                on_event,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_prepared_chat_mtp_with_external_draft<F>(
        &mut self,
        input: crate::backend::mlx::MlxModelInput,
        drafter: &mut MlxDrafter,
        config: MtpConfig,
        prng_key: Option<Array>,
        mut sampler: ConstrainedSampler<crate::runtime::generation::sampler::GenerationSampler>,
        semantic: Box<dyn SpeculativeSemanticState>,
        scheduler: MtpSchedulerOptions,
        cancellation: GenerationCancellationToken,
        on_event: F,
    ) -> Result<PreparedChatMtpGenerationOutput, Error>
    where
        F: FnMut(SemanticEvent),
    {
        self.validate_drafter_compatibility(drafter)?;
        let target_stream = self.runtime.backend().stream().clone();
        let draft_stream = drafter.stream().clone();
        let streams = MtpExecutionStreams::new(&target_stream, &draft_stream)?;
        input.with_borrowed(|model_input| {
            let (model, cache) = self.model_and_cache();
            let (token_ids, stats, finish_reason) = model
                .generate_mtp_input_with_semantics_and_options(
                    drafter,
                    cache,
                    model_input,
                    &config,
                    prng_key,
                    &mut sampler,
                    semantic,
                    cancellation,
                    streams,
                    scheduler,
                    on_event,
                )
                .map_err(|error| Error::Speculative(error.to_string()))?;
            Ok(PreparedChatMtpGenerationOutput {
                token_ids,
                finish_reason,
                stats,
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_prepared_chat_mtp_with_embedded_draft<F>(
        &mut self,
        input: crate::backend::mlx::MlxModelInput,
        config: MtpConfig,
        prng_key: Option<Array>,
        mut sampler: ConstrainedSampler<crate::runtime::generation::sampler::GenerationSampler>,
        semantic: Box<dyn SpeculativeSemanticState>,
        scheduler: MtpSchedulerOptions,
        cancellation: GenerationCancellationToken,
        on_event: F,
    ) -> Result<PreparedChatMtpGenerationOutput, Error>
    where
        F: FnMut(SemanticEvent),
    {
        let stream = self.runtime.backend().stream().clone();
        input.with_borrowed(|model_input| {
            let (model, cache) = self.model_and_cache();
            let (token_ids, stats, finish_reason) = model
                .generate_embedded_mtp_input_with_semantics_and_options(
                    cache,
                    model_input,
                    &config,
                    prng_key,
                    &mut sampler,
                    semantic,
                    cancellation,
                    &stream,
                    scheduler,
                    on_event,
                )
                .map_err(|error| Error::Speculative(error.to_string()))?;
            Ok(PreparedChatMtpGenerationOutput {
                token_ids,
                finish_reason,
                stats,
            })
        })
    }

    /// Reports whether and how this target can perform MTP generation.
    pub(crate) fn mlx_mtp_capability(&self) -> MtpCapability {
        self.model().mtp_capability()
    }

    // Independent target caches are an implementation detail of prepared-chat MTP.
    fn new_mtp_cache(&self, batch_size: usize) -> MlxMtpCache {
        MlxMtpCache::new(
            (0..batch_size)
                .map(|_| self.runtime.session().new_complete_cache())
                .collect(),
        )
    }
}

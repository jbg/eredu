//! Loaded-model facade and prepared generation entry points.

use super::*;

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
        crate::runtime::generation::sampler::ConstraintError,
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
            crate::runtime::generation::sampler::ConstraintError,
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
    ) -> Result<PreparedChatMtpGenerationOutput, B::SpeculativeError>
    where
        B: PreparedChatSpeculativeBackend,
        F: FnMut(SemanticEvent),
    {
        B::execute_prepared_chat_mtp(self, request)
    }

    /// Generates independent prepared chats through one fair speculative scheduler.
    pub fn generate_prepared_chat_mtp_batch<'a>(
        &mut self,
        request: PreparedChatMtpBatchRequest<'a, B, B::Drafter>,
    ) -> Result<PreparedChatMtpBatchOutput, B::SpeculativeError>
    where
        B: PreparedChatSpeculativeBackend,
    {
        B::execute_prepared_chat_mtp_batch(self, request)
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

struct PreparedChatMtpLaneRuntime<'a> {
    input: PreparedChatModelInput,
    cache: &'a mut ModelCache,
    config: MtpConfig,
    prng_key: Option<Array>,
    sampler: ConstrainedSampler<crate::runtime::generation::sampler::GenerationSampler>,
    semantic: Box<dyn SpeculativeSemanticState>,
    cancellation: GenerationCancellationToken,
    on_event: Box<dyn FnMut(SemanticEvent) + 'a>,
}

enum PreparedChatModelInput {
    RenderedPrompt(Array),
    Prepared(crate::backend::mlx::MlxModelInput),
}

impl PreparedChatModelInput {
    fn with_model_input<T>(&self, function: impl FnOnce(input::ModelInput<'_>) -> T) -> T {
        match self {
            Self::RenderedPrompt(prompt) => {
                let parts = [input::InputPart::text_token_ids(prompt)];
                function(input::ModelInput::new(&parts))
            }
            Self::Prepared(input) => input.with_borrowed(function),
        }
    }
}

#[cfg(all(test, feature = "image-processing"))]
mod prepared_chat_model_input_tests {
    use super::PreparedChatModelInput;
    use crate::runtime::media::{
        input::Modality, prepared_model_input, OwnedInputMetadata, PreparedInputPart,
    };
    use safemlx::Array;

    #[test]
    fn prepared_chat_model_input_preserves_multimodal_parts() {
        let prepared = prepared_model_input(vec![
            PreparedInputPart::text_token_ids(&[7]),
            PreparedInputPart::media_tensor(
                Modality::Image,
                Array::from_slice(&[1.0_f32; 4], &[1, 2, 2]),
                OwnedInputMetadata::default(),
            ),
            PreparedInputPart::text_token_ids(&[8]),
        ])
        .unwrap();
        let prepared =
            prepared.with_model_input(|input| crate::backend::mlx::MlxModelInput::from(input));

        PreparedChatModelInput::Prepared(prepared).with_model_input(|input| {
            assert_eq!(input.parts.len(), 3);
            assert_eq!(input.parts[0].modality, Modality::Text);
            assert_eq!(input.parts[1].modality, Modality::Image);
            assert_eq!(input.parts[2].modality, Modality::Text);
        });
    }
}

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
        input.with_model_input(|input| {
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

fn gemma4_mtp_cache(cache: &mut ModelCache) -> Option<&mut gemma4::Cache> {
    match cache {
        ModelCache::Gemma4(cache) => Some(cache),
        _ => None,
    }
}

fn model_mtp_cache(cache: &mut ModelCache) -> Option<&mut ModelCache> {
    Some(cache)
}

struct ExternalMtpBatch<'a, B, S>
where
    B: crate::backend::mlx::speculative::scheduler::MlxSpeculativeRuntime<'a>,
{
    lanes: &'a mut [ModelCache],
    cache_for_lane: fn(&mut ModelCache) -> Option<&mut B::Cache>,
    cache_kind: &'static str,
    prompt_tokens: &'a Array,
    config: &'a MtpConfig,
    prng_key: Option<Array>,
    sampler: &'a S,
    streams: MtpExecutionStreams<'a>,
}

fn run_external_mtp_batch<'a, B, S>(
    backend: &'a mut B,
    batch: ExternalMtpBatch<'a, B, S>,
) -> Result<MtpBatchOutput, Exception>
where
    B: crate::backend::mlx::speculative::scheduler::MlxSpeculativeRuntime<'a>,
    S: SpeculativeSampler + Clone + 'a,
{
    let ExternalMtpBatch {
        lanes,
        cache_for_lane,
        cache_kind,
        prompt_tokens,
        config,
        prng_key,
        sampler,
        streams,
    } = batch;
    let mut batch_prng = prng_key.map(RandomState::from_key);
    let mut scheduler = MlxMtpScheduler::new(backend, streams, MtpSchedulerOptions::default())?;
    for (lane, lane_cache) in lanes.iter_mut().enumerate() {
        let lane_cache = cache_for_lane(lane_cache).ok_or_else(|| {
            Exception::custom(format!(
                "scheduled {cache_kind} MTP cache type mismatch at lane {lane}"
            ))
        })?;
        let row = prompt_tokens.try_index_device((lane as i32, NewAxis, ..), streams.target())?;
        let lane_key = batch_prng
            .as_mut()
            .map(|state| state.next_key(streams.target()))
            .transpose()?;
        let parts = [input::InputPart::text_token_ids(&row)];
        scheduler.submit(
            lane_cache,
            input::ModelInput::new(&parts),
            config.clone(),
            lane_key,
            sampler.clone(),
            |_| Ok(()),
        )?;
    }
    scheduler.run()?;
    let output = scheduler.finish()?;
    let mut token_ids = Vec::with_capacity(output.requests.len());
    let mut stats = Vec::with_capacity(output.requests.len());
    for request in output.requests {
        token_ids.push(request.token_ids);
        stats.push(request.stats);
    }
    Ok(MtpBatchOutput {
        token_ids,
        stats,
        scheduler: output.scheduler,
    })
}

fn qwen_next_mtp_cache(cache: &mut ModelCache) -> Option<&mut qwen3_5::Cache> {
    match cache {
        ModelCache::Qwen3Next(cache) => Some(cache),
        _ => None,
    }
}

fn qwen35_mtp_cache(cache: &mut ModelCache) -> Option<&mut qwen3_5::Cache> {
    match cache {
        ModelCache::Qwen35(cache) => Some(cache),
        _ => None,
    }
}

fn deepseek_mtp_cache(cache: &mut ModelCache) -> Option<&mut deepseek_v3::Cache> {
    match cache {
        ModelCache::DeepSeekV3(cache) => Some(cache),
        _ => None,
    }
}

fn inkling_mtp_cache(cache: &mut ModelCache) -> Option<&mut inkling::Cache> {
    match cache {
        ModelCache::Inkling(cache) => Some(cache),
        _ => None,
    }
}

fn nemotron_mtp_cache(cache: &mut ModelCache) -> Option<&mut nemotron_h::Cache> {
    match cache {
        ModelCache::NemotronH(cache) => Some(cache),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn run_embedded_mtp_batch<'a, B, C: 'a, S>(
    backend: &'a mut B,
    lanes: &'a mut [ModelCache],
    cache_for_lane: fn(&mut ModelCache) -> Option<&mut C>,
    prompt_tokens: &Array,
    config: &MtpConfig,
    prng_key: Option<Array>,
    sampler: &S,
    stream: &'a Stream,
) -> Result<MtpBatchOutput, Exception>
where
    B: crate::backend::mlx::speculative::scheduler::MlxSpeculativeRuntime<'a>
        + safemlx_lm_core::SpeculativeExecutor<Cache = C>,
    S: SpeculativeSampler + Clone + 'a,
{
    let streams = MtpExecutionStreams::single(stream);
    let mut batch_prng = prng_key.map(RandomState::from_key);
    let mut scheduler = MlxMtpScheduler::new(backend, streams, MtpSchedulerOptions::default())?;
    for (lane, lane_cache) in lanes.iter_mut().enumerate() {
        let lane_cache = cache_for_lane(lane_cache).ok_or_else(|| {
            Exception::custom(format!(
                "scheduled embedded MTP cache type mismatch at lane {lane}"
            ))
        })?;
        let row = prompt_tokens.try_index_device((lane as i32, NewAxis, ..), stream)?;
        let lane_key = batch_prng
            .as_mut()
            .map(|state| state.next_key(stream))
            .transpose()?;
        let parts = [input::InputPart::text_token_ids(&row)];
        scheduler.submit(
            lane_cache,
            input::ModelInput::new(&parts),
            config.clone(),
            lane_key,
            sampler.clone(),
            |_| Ok(()),
        )?;
    }
    scheduler.run()?;
    let output = scheduler.finish()?;
    let mut token_ids = Vec::with_capacity(output.requests.len());
    let mut stats = Vec::with_capacity(output.requests.len());
    for request in output.requests {
        token_ids.push(request.token_ids);
        stats.push(request.stats);
    }
    Ok(MtpBatchOutput {
        token_ids,
        stats,
        scheduler: output.scheduler,
    })
}

impl LoadedModel<crate::backend::mlx::MlxBackend<'static>> {
    pub(crate) fn model(&self) -> &Model {
        self.runtime.session().complete_model()
    }

    fn model_and_cache(&mut self) -> (&mut Model, &mut ModelCache) {
        self.runtime.session_mut().complete_parts_mut()
    }

    fn resolve_prepared_chat_generation_settings(
        &self,
        settings: PreparedChatGenerationSettings,
    ) -> Result<ResolvedPreparedChatGenerationSettings, Error> {
        let (config, max_tokens) = self.resolve_text_generation_settings(settings)?;
        let resolved = config.sampling();
        let prng_key = (resolved.temperature != 0.0)
            .then(|| safemlx::random::key(config.seed()))
            .transpose()?;
        Ok(ResolvedPreparedChatGenerationSettings {
            temperature: resolved.temperature,
            max_tokens,
            prng_key,
            checkpoint_sampler:
                crate::runtime::generation::sampler::GenerationSampler::from_resolved(resolved),
        })
    }

    /// Validates the observable target/assistant contract used by external MTP.
    ///
    /// Repository names and revisions are deliberately not compatibility keys.
    /// The validation covers the target architecture, shared tensor geometry,
    /// and the token-id vocabulary mapping when the drafter carries tokenizer
    /// metadata.
    pub fn validate_drafter_compatibility(&self, drafter: &MlxDrafter) -> Result<(), Error> {
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

    fn prepare_chat_model_input<'a>(
        &self,
        input: PreparedChatInput<'a, crate::backend::mlx::MlxBackend<'static>>,
        stream: &Stream,
    ) -> Result<PreparedChatModelInput, Error> {
        match input {
            PreparedChatInput::RenderedPrompt(prepared_chat) => {
                Ok(PreparedChatModelInput::RenderedPrompt(
                    self.encode_to_array(prepared_chat.rendered_prompt(), false, stream)?,
                ))
            }
            PreparedChatInput::PreparedBackendInput { prompt, .. } => {
                Ok(PreparedChatModelInput::Prepared(prompt))
            }
        }
    }

    fn prepare_chat_mtp_batch_lanes<'a>(
        &self,
        lanes: Vec<PreparedChatMtpBatchLane<'a, crate::backend::mlx::MlxBackend<'static>>>,
        cache: &'a mut MlxMtpCache,
        stream: &Stream,
    ) -> Result<Vec<PreparedChatMtpLaneRuntime<'a>>, Error> {
        if cache.len() != lanes.len() {
            return Err(Error::Speculative(format!(
                "MTP cache has {} lanes but the request has {} lanes",
                cache.len(),
                lanes.len()
            )));
        }
        let mut prepared_lanes = Vec::with_capacity(lanes.len());
        for (lane_index, (lane, cache)) in lanes.into_iter().zip(cache.lanes.iter_mut()).enumerate()
        {
            let PreparedChatMtpBatchLane {
                input,
                settings,
                max_draft_tokens,
                caller_stop_sequences,
                cancellation,
                on_event,
            } = lane;
            let prepared_chat = input.prepared_chat();
            let semantic_plan = match prepared_chat.semantic_support() {
                SemanticSupport::Supported => prepared_chat
                    .semantic_runtime_plan()
                    .expect("supported prepared chats carry a semantic runtime plan")
                    .clone(),
                SemanticSupport::Unsupported { reason } => {
                    return Err(Error::PreparedChatSemantic(format!(
                        "prepared chat lane {lane_index} does not have an executable semantic plan: {reason}"
                    )));
                }
            };
            let generation_plan = prepared_chat
                .generation_runtime_plan()
                .expect("supported prepared chats carry a generation runtime plan");
            let settings = self.resolve_prepared_chat_generation_settings(settings)?;
            let sampler = ConstrainedSampler::from_generation_plan(
                settings.checkpoint_sampler,
                generation_plan,
            )
            .map_err(|error| Error::Speculative(error.to_string()))?;
            let decoder = PreparedChatTokenDecoder {
                decoder: self.text_decoder(true),
            };
            let semantic =
                PreparedChatSemanticState::new(decoder, semantic_plan, caller_stop_sequences)
                    .map_err(|error| Error::PreparedChatSemantic(error.to_string()))?;
            let input = self.prepare_chat_model_input(input, stream)?;
            prepared_lanes.push(PreparedChatMtpLaneRuntime {
                input,
                cache,
                config: MtpConfig {
                    max_tokens: settings.max_tokens.get(),
                    max_draft_tokens: max_draft_tokens.get(),
                    temperature: settings.temperature,
                    eos_token_ids: prepared_chat.eos_token_ids().to_vec(),
                },
                prng_key: settings.prng_key,
                sampler,
                semantic: Box::new(semantic),
                cancellation,
                on_event,
            });
        }
        Ok(prepared_lanes)
    }

    /// Generates multiple independent prepared chats through one fair MTP scheduler.
    pub(crate) fn execute_prepared_chat_mtp_batch_mlx(
        &mut self,
        request: PreparedChatMtpBatchRequest<
            '_,
            crate::backend::mlx::MlxBackend<'static>,
            MlxDrafter,
        >,
    ) -> Result<PreparedChatMtpBatchOutput, Error> {
        let PreparedChatMtpBatchRequest {
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
        lanes: Vec<PreparedChatMtpBatchLane<'_, crate::backend::mlx::MlxBackend<'static>>>,
        scheduler: MtpSchedulerOptions,
    ) -> Result<PreparedChatMtpBatchOutput, Error> {
        self.validate_drafter_compatibility(drafter)?;
        let target_stream = self.runtime.backend().stream().clone();
        let draft_stream = drafter.stream().clone();
        let streams = MtpExecutionStreams::new(&target_stream, &draft_stream)?;
        let mut cache = self.new_mtp_cache(lanes.len());
        let prepared_lanes =
            self.prepare_chat_mtp_batch_lanes(lanes, &mut cache, &target_stream)?;

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
        lanes: Vec<PreparedChatMtpBatchLane<'_, crate::backend::mlx::MlxBackend<'static>>>,
        scheduler: MtpSchedulerOptions,
    ) -> Result<PreparedChatMtpBatchOutput, Error> {
        let stream = self.runtime.backend().stream().clone();
        let mut cache = self.new_mtp_cache(lanes.len());
        let prepared_lanes = self.prepare_chat_mtp_batch_lanes(lanes, &mut cache, &stream)?;
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
                let mut backend = crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(target);
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
                let mut backend = crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(target);
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
        request: PreparedChatMtpGenerationRequest<
            'a,
            crate::backend::mlx::MlxBackend<'static>,
            MlxDrafter,
            F,
        >,
    ) -> Result<PreparedChatMtpGenerationOutput, Error>
    where
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
        match drafting {
            PreparedChatDraft::External(drafter) => self
                .generate_prepared_chat_mtp_with_external_draft(
                    input,
                    drafter,
                    settings,
                    options,
                    caller_stop_sequences,
                    cancellation,
                    on_event,
                ),
            PreparedChatDraft::Embedded => self.generate_prepared_chat_mtp_with_embedded_draft(
                input,
                settings,
                options,
                caller_stop_sequences,
                cancellation,
                on_event,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_prepared_chat_mtp_with_external_draft<F>(
        &mut self,
        input: PreparedChatInput<'_, crate::backend::mlx::MlxBackend<'static>>,
        drafter: &mut MlxDrafter,
        settings: PreparedChatGenerationSettings,
        options: PreparedChatMtpGenerationOptions,
        caller_stop_sequences: &[String],
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
        let prepared_chat = input.prepared_chat();
        let semantic_plan = match prepared_chat.semantic_support() {
            SemanticSupport::Supported => prepared_chat
                .semantic_runtime_plan()
                .expect("supported prepared chats carry a semantic runtime plan")
                .clone(),
            SemanticSupport::Unsupported { reason } => {
                return Err(Error::PreparedChatSemantic(format!(
                    "prepared chat does not have an executable semantic plan: {reason}"
                )));
            }
        };
        let settings = self.resolve_prepared_chat_generation_settings(settings)?;
        with_prepared_chat_runtime(prepared_chat, settings.checkpoint_sampler, |runtime| {
            let decoder = PreparedChatTokenDecoder {
                decoder: self.text_decoder(true),
            };
            let semantic =
                PreparedChatSemanticState::new(decoder, semantic_plan, caller_stop_sequences)
                    .map_err(|error| Error::PreparedChatSemantic(error.to_string()))?;
            let config = MtpConfig {
                max_tokens: settings.max_tokens.get(),
                max_draft_tokens: options.max_draft_tokens.get(),
                temperature: settings.temperature,
                eos_token_ids: prepared_chat.eos_token_ids().to_vec(),
            };
            let mut sampler = runtime.sampler;
            let model_input = self.prepare_chat_model_input(input, streams.target())?;
            model_input.with_model_input(|model_input| {
                let (model, cache) = self.model_and_cache();
                let (token_ids, stats, finish_reason) = model
                    .generate_mtp_input_with_semantics_and_options(
                        drafter,
                        cache,
                        model_input,
                        &config,
                        settings.prng_key,
                        &mut sampler,
                        Box::new(semantic),
                        cancellation,
                        streams,
                        options.scheduler,
                        on_event,
                    )
                    .map_err(|error| Error::Speculative(error.to_string()))?;
                Ok(PreparedChatMtpGenerationOutput {
                    token_ids,
                    finish_reason,
                    stats,
                })
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_prepared_chat_mtp_with_embedded_draft<F>(
        &mut self,
        input: PreparedChatInput<'_, crate::backend::mlx::MlxBackend<'static>>,
        settings: PreparedChatGenerationSettings,
        options: PreparedChatMtpGenerationOptions,
        caller_stop_sequences: &[String],
        cancellation: GenerationCancellationToken,
        on_event: F,
    ) -> Result<PreparedChatMtpGenerationOutput, Error>
    where
        F: FnMut(SemanticEvent),
    {
        let stream = self.runtime.backend().stream().clone();
        let prepared_chat = input.prepared_chat();
        let semantic_plan = match prepared_chat.semantic_support() {
            SemanticSupport::Supported => prepared_chat
                .semantic_runtime_plan()
                .expect("supported prepared chats carry a semantic runtime plan")
                .clone(),
            SemanticSupport::Unsupported { reason } => {
                return Err(Error::PreparedChatSemantic(format!(
                    "prepared chat does not have an executable semantic plan: {reason}"
                )));
            }
        };
        let settings = self.resolve_prepared_chat_generation_settings(settings)?;
        with_prepared_chat_runtime(prepared_chat, settings.checkpoint_sampler, |runtime| {
            let decoder = PreparedChatTokenDecoder {
                decoder: self.text_decoder(true),
            };
            let semantic =
                PreparedChatSemanticState::new(decoder, semantic_plan, caller_stop_sequences)
                    .map_err(|error| Error::PreparedChatSemantic(error.to_string()))?;
            let config = MtpConfig {
                max_tokens: settings.max_tokens.get(),
                max_draft_tokens: options.max_draft_tokens.get(),
                temperature: settings.temperature,
                eos_token_ids: prepared_chat.eos_token_ids().to_vec(),
            };
            let mut sampler = runtime.sampler;
            let model_input = self.prepare_chat_model_input(input, &stream)?;
            model_input.with_model_input(|model_input| {
                let (model, cache) = self.model_and_cache();
                let (token_ids, stats, finish_reason) = model
                    .generate_embedded_mtp_input_with_semantics_and_options(
                        cache,
                        model_input,
                        &config,
                        settings.prng_key,
                        &mut sampler,
                        Box::new(semantic),
                        cancellation,
                        &stream,
                        options.scheduler,
                        on_event,
                    )
                    .map_err(|error| Error::Speculative(error.to_string()))?;
                Ok(PreparedChatMtpGenerationOutput {
                    token_ids,
                    finish_reason,
                    stats,
                })
            })
        })
    }

    /// Reports whether and how this target can perform MTP generation.
    pub(crate) fn mlx_mtp_capability(&self) -> MtpCapability {
        self.model().mtp_capability()
    }

    /// Creates independent target caches for an MTP text batch.
    pub fn new_mtp_cache(&self, batch_size: usize) -> MlxMtpCache {
        MlxMtpCache::new(
            (0..batch_size)
                .map(|_| self.runtime.session().new_complete_cache())
                .collect(),
        )
    }

    /// Generates through the architecture-independent MTP path.
    pub fn generate_mtp_input(
        &mut self,
        drafter: &mut MlxDrafter,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        stream: &Stream,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        self.generate_mtp_input_with_sampler(
            drafter,
            input,
            config,
            prng_key,
            &mut DefaultSampler,
            stream,
        )
    }

    /// Generates through MTP with separate target and draft streams.
    pub fn generate_mtp_input_with_streams(
        &mut self,
        drafter: &mut MlxDrafter,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        self.generate_mtp_input_with_sampler_and_streams(
            drafter,
            input,
            config,
            prng_key,
            &mut DefaultSampler,
            streams,
        )
    }

    /// Generates through MTP with a caller-provided speculative sampling policy.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_mtp_input_with_sampler<S: SpeculativeSampler + Clone>(
        &mut self,
        drafter: &mut MlxDrafter,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        stream: &Stream,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        self.generate_mtp_input_with_sampler_and_streams(
            drafter,
            input,
            config,
            prng_key,
            sampler,
            MtpExecutionStreams::single(stream),
        )
    }

    /// Generates through MTP with a caller-provided sampler and separate
    /// target/draft streams.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_mtp_input_with_sampler_and_streams<S: SpeculativeSampler + Clone>(
        &mut self,
        drafter: &mut MlxDrafter,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        let mut config = config.clone();
        if config.eos_token_ids.is_empty() {
            config.eos_token_ids.clone_from(&self.eos_token_ids);
        }
        let (model, cache) = self.model_and_cache();
        model.generate_mtp_input_with_sampler_callback_and_streams(
            drafter,
            cache,
            input,
            &config,
            prng_key,
            sampler,
            streams,
            |_| Ok(()),
        )
    }

    /// Generates through MTP with separate streams and reports committed
    /// tokens as they become available.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_mtp_input_with_sampler_callback_and_streams<S, F>(
        &mut self,
        drafter: &mut MlxDrafter,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        streams: MtpExecutionStreams<'_>,
        on_token: F,
    ) -> Result<(Vec<u32>, MtpStats), Exception>
    where
        S: SpeculativeSampler + Clone,
        F: FnMut(u32) -> Result<(), Exception>,
    {
        self.generate_mtp_input_with_sampler_callback_and_streams_and_options(
            drafter,
            input,
            config,
            prng_key,
            sampler,
            streams,
            MtpSchedulerOptions::default(),
            on_token,
        )
    }

    /// Generates through MTP with explicit streams and scheduler controls.
    ///
    /// `MtpSchedulerOptions::with_lookahead(false)` provides an equivalent
    /// canonical A/B path with no optimistic branch work.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_mtp_input_with_sampler_callback_and_streams_and_options<S, F>(
        &mut self,
        drafter: &mut MlxDrafter,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        streams: MtpExecutionStreams<'_>,
        scheduler_options: MtpSchedulerOptions,
        on_token: F,
    ) -> Result<(Vec<u32>, MtpStats), Exception>
    where
        S: SpeculativeSampler + Clone,
        F: FnMut(u32) -> Result<(), Exception>,
    {
        self.validate_drafter_compatibility(drafter)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let mut config = config.clone();
        if config.eos_token_ids.is_empty() {
            config.eos_token_ids.clone_from(&self.eos_token_ids);
        }
        let (model, cache) = self.model_and_cache();
        model.generate_mtp_input_with_sampler_callback_and_streams_and_options(
            drafter,
            cache,
            input,
            &config,
            prng_key,
            sampler,
            streams,
            scheduler_options,
            on_token,
        )
    }

    /// Generates through MTP and reports each committed token as it becomes available.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_mtp_input_with_sampler_callback<S, F>(
        &mut self,
        drafter: &mut MlxDrafter,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        stream: &Stream,
        on_token: F,
    ) -> Result<(Vec<u32>, MtpStats), Exception>
    where
        S: SpeculativeSampler + Clone,
        F: FnMut(u32) -> Result<(), Exception>,
    {
        self.generate_mtp_input_with_sampler_callback_and_streams(
            drafter,
            input,
            config,
            prng_key,
            sampler,
            MtpExecutionStreams::single(stream),
            on_token,
        )
    }

    /// Generates through MTP weights embedded in the target checkpoint.
    pub fn generate_embedded_mtp_input(
        &mut self,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        stream: &Stream,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        self.generate_embedded_mtp_input_with_sampler(
            input,
            config,
            prng_key,
            &mut DefaultSampler,
            stream,
        )
    }

    /// Generates through embedded MTP weights with a caller-provided sampler.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_embedded_mtp_input_with_sampler<S: SpeculativeSampler + Clone>(
        &mut self,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        stream: &Stream,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        let mut config = config.clone();
        if config.eos_token_ids.is_empty() {
            config.eos_token_ids.clone_from(&self.eos_token_ids);
        }
        let (model, cache) = self.model_and_cache();
        model.generate_embedded_mtp_input_with_sampler(
            cache, input, &config, prng_key, sampler, stream,
        )
    }

    /// Generates through embedded MTP and reports each committed token as it becomes available.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_embedded_mtp_input_with_sampler_callback<S, F>(
        &mut self,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        stream: &Stream,
        on_token: F,
    ) -> Result<(Vec<u32>, MtpStats), Exception>
    where
        S: SpeculativeSampler + Clone,
        F: FnMut(u32) -> Result<(), Exception>,
    {
        let mut config = config.clone();
        if config.eos_token_ids.is_empty() {
            config.eos_token_ids.clone_from(&self.eos_token_ids);
        }
        let (model, cache) = self.model_and_cache();
        model.generate_embedded_mtp_input_with_sampler_callback(
            cache, input, &config, prng_key, sampler, stream, on_token,
        )
    }

    /// Generates an independently accepting and stopping batch of text prompts.
    ///
    /// Each lane owns a separate cache so rejection lengths and EOS positions
    /// may diverge without padding rejected state back into another sequence.
    pub fn generate_mtp_text_batch<S: SpeculativeSampler + Clone>(
        &mut self,
        drafter: &mut MlxDrafter,
        prompt_tokens: &Array,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &S,
        stream: &Stream,
    ) -> Result<MtpBatchOutput, Exception> {
        let batch_size = if prompt_tokens.ndim() == 2 {
            prompt_tokens.dim(0) as usize
        } else {
            0
        };
        let mut cache = self.new_mtp_cache(batch_size);
        self.generate_mtp_text_batch_with_cache(
            drafter,
            &mut cache,
            prompt_tokens,
            config,
            prng_key,
            sampler,
            stream,
        )
    }

    /// Generates a text batch using reusable independent per-lane caches.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_mtp_text_batch_with_cache<S: SpeculativeSampler + Clone>(
        &mut self,
        drafter: &mut MlxDrafter,
        cache: &mut MlxMtpCache,
        prompt_tokens: &Array,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &S,
        stream: &Stream,
    ) -> Result<MtpBatchOutput, Exception> {
        self.generate_mtp_text_batch_with_cache_and_streams(
            drafter,
            cache,
            prompt_tokens,
            config,
            prng_key,
            sampler,
            MtpExecutionStreams::single(stream),
        )
    }

    /// Generates a fair scheduled text batch with explicit target/draft streams.
    ///
    /// All lanes are submitted before decoding begins. With distinct target
    /// and draft streams on one device or across devices, a lane drafts
    /// optimistically or another ready lane drafts before the scheduler
    /// resolves an in-flight verification.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_mtp_text_batch_with_cache_and_streams<S: SpeculativeSampler + Clone>(
        &mut self,
        drafter: &mut MlxDrafter,
        cache: &mut MlxMtpCache,
        prompt_tokens: &Array,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &S,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<MtpBatchOutput, Exception> {
        self.validate_drafter_compatibility(drafter)
            .map_err(|error| Exception::custom(error.to_string()))?;
        if prompt_tokens.ndim() != 2 || prompt_tokens.dim(1) == 0 {
            return Err(Exception::custom(format!(
                "MTP text batch must be shaped [batch, nonzero sequence], got {:?}",
                prompt_tokens.shape()
            )));
        }
        if cache.len() != prompt_tokens.dim(0) as usize {
            return Err(Exception::custom(format!(
                "MTP cache has {} lanes but text input has batch size {}",
                cache.len(),
                prompt_tokens.dim(0)
            )));
        }
        if config.temperature != 0.0 && prng_key.is_none() {
            return Err(Exception::custom(
                "random operations require an explicit PRNG key",
            ));
        }
        let mut config = config.clone();
        if config.eos_token_ids.is_empty() {
            config.eos_token_ids.clone_from(&self.eos_token_ids);
        }
        match self.model_and_cache().0 {
            Model::Gemma4(target) => {
                let assistant = drafter.gemma4_mut();
                validate_gemma4_drafter(target.args(), assistant)?;
                let mut backend =
                    crate::architectures::gemma4::mtp::Gemma4MtpExecutor::new(target, assistant);
                run_external_mtp_batch(
                    &mut backend,
                    ExternalMtpBatch {
                        lanes: &mut cache.lanes,
                        cache_for_lane: gemma4_mtp_cache,
                        cache_kind: "Gemma 4",
                        prompt_tokens,
                        config: &config,
                        prng_key,
                        sampler,
                        streams,
                    },
                )
            }
            Model::MuseGlimmer(target) => {
                let assistant = drafter.muse_glimmer_mut();
                let mut backend =
                    crate::architectures::muse_glimmer::mtp::MuseGlimmerMtpExecutor::new(
                        target, assistant,
                    );
                run_external_mtp_batch(
                    &mut backend,
                    ExternalMtpBatch {
                        lanes: &mut cache.lanes,
                        cache_for_lane: model_mtp_cache,
                        cache_kind: "Muse-Glimmer DFlash",
                        prompt_tokens,
                        config: &config,
                        prng_key,
                        sampler,
                        streams,
                    },
                )
            }
            model => Err(Exception::custom(format!(
                "scheduled external MTP batch is unavailable for model type {} ({:?})",
                model.model_type(),
                model.mtp_capability()
            ))),
        }
    }

    /// Generates an independently accepting text batch with embedded MTP weights.
    pub fn generate_embedded_mtp_text_batch<S: SpeculativeSampler + Clone>(
        &mut self,
        prompt_tokens: &Array,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &S,
        stream: &Stream,
    ) -> Result<MtpBatchOutput, Exception> {
        let batch_size = if prompt_tokens.ndim() == 2 {
            prompt_tokens.dim(0) as usize
        } else {
            0
        };
        let mut cache = self.new_mtp_cache(batch_size);
        self.generate_embedded_mtp_text_batch_with_cache(
            &mut cache,
            prompt_tokens,
            config,
            prng_key,
            sampler,
            stream,
        )
    }

    /// Generates a text batch with embedded MTP weights and reusable lane caches.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_embedded_mtp_text_batch_with_cache<S: SpeculativeSampler + Clone>(
        &mut self,
        cache: &mut MlxMtpCache,
        prompt_tokens: &Array,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &S,
        stream: &Stream,
    ) -> Result<MtpBatchOutput, Exception> {
        if prompt_tokens.ndim() != 2 || prompt_tokens.dim(1) == 0 {
            return Err(Exception::custom(format!(
                "MTP text batch must be shaped [batch, nonzero sequence], got {:?}",
                prompt_tokens.shape()
            )));
        }
        if cache.len() != prompt_tokens.dim(0) as usize {
            return Err(Exception::custom(format!(
                "MTP cache has {} lanes but text input has batch size {}",
                cache.len(),
                prompt_tokens.dim(0)
            )));
        }
        if config.temperature != 0.0 && prng_key.is_none() {
            return Err(Exception::custom(
                "random operations require an explicit PRNG key",
            ));
        }
        let mut config = config.clone();
        if config.eos_token_ids.is_empty() {
            config.eos_token_ids.clone_from(&self.eos_token_ids);
        }
        match self.model_and_cache().0 {
            Model::DeepSeekV3(target) => {
                let mut backend =
                    crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(target);
                run_embedded_mtp_batch(
                    &mut backend,
                    &mut cache.lanes,
                    deepseek_mtp_cache,
                    prompt_tokens,
                    &config,
                    prng_key,
                    sampler,
                    stream,
                )
            }
            Model::Inkling(target) => {
                let mut backend =
                    crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(target);
                run_embedded_mtp_batch(
                    &mut backend,
                    &mut cache.lanes,
                    inkling_mtp_cache,
                    prompt_tokens,
                    &config,
                    prng_key,
                    sampler,
                    stream,
                )
            }
            Model::NemotronH(target) => {
                let mut backend =
                    crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(target);
                run_embedded_mtp_batch(
                    &mut backend,
                    &mut cache.lanes,
                    nemotron_mtp_cache,
                    prompt_tokens,
                    &config,
                    prng_key,
                    sampler,
                    stream,
                )
            }
            Model::Qwen3Next(target) => {
                let mut backend =
                    crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(target);
                run_embedded_mtp_batch(
                    &mut backend,
                    &mut cache.lanes,
                    qwen_next_mtp_cache,
                    prompt_tokens,
                    &config,
                    prng_key,
                    sampler,
                    stream,
                )
            }
            Model::Qwen35(target) => {
                let mut backend =
                    crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(target);
                run_embedded_mtp_batch(
                    &mut backend,
                    &mut cache.lanes,
                    qwen35_mtp_cache,
                    prompt_tokens,
                    &config,
                    prng_key,
                    sampler,
                    stream,
                )
            }
            model => Err(Exception::custom(format!(
                "scheduled embedded MTP batch is unavailable for model type {} ({:?})",
                model.model_type(),
                model.mtp_capability()
            ))),
        }
    }

    /// Returns residency telemetry when bounded layer execution was selected.
    pub fn residency_report(
        &self,
    ) -> Result<Option<crate::runtime::residency::manager::ResidencyReport>, Error> {
        self.runtime.session().residency_report()
    }

    /// Returns experimental dense-stream telemetry when enabled.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<crate::runtime::execution::layerwise::DenseDiskStreamReport>, Error> {
        self.runtime.session().dense_stream_report()
    }

    /// Returns sparse routed-expert cache telemetry when enabled.
    pub fn expert_cache_report(
        &self,
    ) -> Result<Option<crate::runtime::residency::expert_cache::ExpertCacheReport>, Error> {
        self.runtime.session().expert_cache_report()
    }

    /// Returns checkpoint-native quantization storage statistics when available.
    pub fn native_quantization_stats(
        &self,
    ) -> Option<&safemlx::native_quantization::NativeQuantizationStats> {
        self.model().native_quantization_stats()
    }

    /// Returns whether this model directory includes a supported media processor.
    #[cfg(feature = "media-processing")]
    pub fn has_processor(&self) -> bool {
        self.runtime.session().processor().is_some()
    }

    /// Returns the loaded architecture-dispatched media processor, if available.
    #[cfg(feature = "media-processing")]
    pub fn processor(&self) -> Option<&ModelProcessor> {
        self.runtime.session().processor()
    }

    /// Tokenizes and preprocesses ordered text and media segments.
    #[cfg(feature = "media-processing")]
    pub fn prepare_input(
        &self,
        input: &[ProcessorInput<'_>],
    ) -> Result<crate::backend::mlx::MlxModelInput, Error> {
        let processor = self.runtime.session().processor().ok_or_else(|| {
            Error::Processor(format!(
                "model type '{}' does not have a loaded media processor",
                self.model_type()
            ))
        })?;
        let prepared = processor.prepare_input(input, &mut |text| self.encode(text, false))?;
        Ok(crate::backend::mlx::MlxModelInput::from_prepared(&prepared))
    }

    /// Composes a prepared chat with decoded media at checked placeholders.
    ///
    /// Each binding removes one complete placeholder spelling from
    /// [`PreparedChat::rendered_prompt`] and inserts processor-owned media
    /// boundary tokens and tensors at that exact position. Occurrence counts
    /// and binding order are validated before preprocessing begins.
    #[cfg(feature = "media-processing")]
    pub fn prepare_chat_input(
        &self,
        prepared_chat: &PreparedChat,
        bindings: &[ChatMediaBinding<'_>],
    ) -> Result<crate::backend::mlx::MlxModelInput, Error> {
        let processor = self.runtime.session().processor().ok_or_else(|| {
            Error::Processor(format!(
                "model type '{}' does not have a loaded media processor",
                self.model_type()
            ))
        })?;
        let prepared = processor.prepare_chat_input(
            prepared_chat.rendered_prompt(),
            bindings,
            &mut |text| self.encode(text, false),
        )?;
        Ok(crate::backend::mlx::MlxModelInput::from_prepared(&prepared))
    }

    /// Encodes text and returns a `[1, len]` token-id array on `stream`.
    pub fn encode_to_array(
        &self,
        text: &str,
        add_special_tokens: bool,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let ids = self.encode(text, add_special_tokens)?;
        Ok(Array::from(ids.as_slice()).try_index_device(NewAxis, stream)?)
    }

    /// Clears the backend-owned cache for a new independent sequence.
    pub fn reset_session(&mut self) -> Result<(), Error> {
        self.runtime.session_mut().reset()
    }

    /// Replaces backend-owned cache state under an explicit residency policy.
    pub fn configure_cache(&mut self, policy: CacheResidencyPolicy) -> Result<(), Error> {
        self.runtime.session_mut().configure_cache(policy)
    }

    /// Returns residency telemetry for the backend-owned session cache.
    pub fn cache_residency_report(&self) -> Result<Option<CacheResidencyReport>, Error> {
        self.runtime.session().cache_residency_report()
    }

    /// Configures a paged cache attached to an aggregate process-wide pool.
    pub fn configure_cache_in_pool(
        &mut self,
        options: PagedCacheOptions,
        pool: CacheResidencyPool,
    ) -> Result<(), Error> {
        let options = options
            .with_pool(pool)
            .map_err(|error| Error::Exception(Exception::custom(error.to_string())))?;
        self.configure_cache(CacheResidencyPolicy::Paged(options))
    }

    /// Returns the canonical cache-relevant architecture identity for this loaded model.
    pub fn prompt_cache_architecture_fingerprint(&self) -> Result<String, Exception> {
        self.model().prompt_cache_architecture_fingerprint()
    }

    /// Returns the exact ordered prompt-cache state and attention layout.
    pub fn prompt_cache_layer_layout(&self) -> Result<LayerSchedule<LayerCachePolicy>, Exception> {
        self.model().prompt_cache_layer_layout()
    }

    /// Returns each owned layer's processed-token delta from the persisted prefix.
    pub fn prompt_cache_layer_prefix_offsets(&self) -> Result<Vec<i32>, Exception> {
        self.model().prompt_cache_layer_prefix_offsets()
    }

    /// Opens a compatible reusable prefix for this loaded model.
    ///
    /// The compatible prefix replaces the cache owned by this model session.
    pub fn load_prompt_cache(
        &mut self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        let (backend, session) = self.runtime.parts_mut();
        session.load_prompt_cache(backend, directory, expected, prefix_token_ids, options)
    }

    /// Atomically saves a completed immutable prefix with model-owned state validation.
    pub fn save_prompt_cache(
        &mut self,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        let (backend, session) = self.runtime.parts_mut();
        session.save_prompt_cache(backend, destination, descriptor, prefix_token_ids, options)
    }

    /// Submits prompt prefill against this model session's cache.
    pub fn submit_prefill(
        &mut self,
        input: input::ModelInput<'_>,
    ) -> Result<safemlx_lm_core::Submission<Array, crate::backend::mlx::MlxSessionCompletion>, Error>
    {
        let submission = self
            .runtime
            .prefill(crate::backend::mlx::MlxModelInput::from(input))?;
        let output = submission.output.into_logits().ok_or_else(|| {
            Error::Parallel("replicated loaded model produced no local logits".into())
        })?;
        Ok(safemlx_lm_core::Submission {
            output,
            completion: submission.completion,
        })
    }

    /// Submits cached decode using an existing model cache.
    pub fn submit_decode(
        &mut self,
        input: Array,
    ) -> Result<safemlx_lm_core::Submission<Array, crate::backend::mlx::MlxSessionCompletion>, Error>
    {
        let submission = self.runtime.decode(input)?;
        let output = submission.output.into_logits().ok_or_else(|| {
            Error::Parallel("replicated loaded model produced no local logits".into())
        })?;
        Ok(safemlx_lm_core::Submission {
            output,
            completion: submission.completion,
        })
    }

    /// Submits prompt prefill from an owned processor result.
    #[cfg(feature = "media-processing")]
    pub fn submit_prepared_prefill(
        &mut self,
        input: &PreparedModelInput,
    ) -> Result<safemlx_lm_core::Submission<Array, crate::backend::mlx::MlxSessionCompletion>, Error>
    {
        input.with_model_input(|input| self.submit_prefill(input))
    }

    /// Submits prompt prefill while reporting detailed activations.
    ///
    /// The submitted output has shape `[batch, vocab]` and matches
    /// [`LoadedModel::submit_prefill`] for the same model/cache.
    pub fn submit_prefill_with_observer(
        &mut self,
        input: input::ModelInput<'_>,
        observer: &mut impl ActivationObserver,
    ) -> Result<safemlx_lm_core::Submission<Array, crate::backend::mlx::MlxCompletion>, Error> {
        let stream = self.runtime.backend().stream().clone();
        let (model, cache) = self.model_and_cache();
        crate::backend::mlx::MlxModelSession::submit_prefill_with_observer(
            model,
            input.into(),
            cache,
            &stream,
            observer,
        )
    }

    /// Submits an owned processor result while observing activations.
    #[cfg(feature = "media-processing")]
    pub fn submit_prepared_prefill_with_observer(
        &mut self,
        input: &PreparedModelInput,
        observer: &mut impl ActivationObserver,
    ) -> Result<safemlx_lm_core::Submission<Array, crate::backend::mlx::MlxCompletion>, Error> {
        input.with_model_input(|input| self.submit_prefill_with_observer(input, observer))
    }

    /// Creates a token iterator using checkpoint generation defaults plus typed overrides.
    pub fn generate_input<'a>(
        &'a mut self,
        input: input::ModelInput<'_>,
        overrides: GenerationConfigOverrides,
        prng_key: Option<Array>,
    ) -> Result<MlxGeneration<'a, crate::runtime::generation::sampler::GenerationSampler>, Error>
    {
        let resolved = self.resolve_generation_config(overrides)?;
        let prng_key = match (prng_key, resolved.temperature) {
            (Some(key), _) => Some(key),
            (None, 0.0) => None,
            (None, _) => Some(safemlx::random::key(0)?),
        };
        Ok(MlxGeneration::with_sampler(
            &mut self.runtime,
            resolved.temperature,
            input,
            prng_key,
            crate::runtime::generation::sampler::GenerationSampler::from_resolved(resolved),
        ))
    }

    /// Creates a token iterator from typed input with a caller-provided sampler.
    pub fn generate_input_with_sampler<'a, S>(
        &'a mut self,
        temp: f32,
        input: input::ModelInput<'_>,
        prng_key: Option<Array>,
        sampler: S,
    ) -> MlxGeneration<'a, S>
    where
        S: Sampler,
    {
        MlxGeneration::with_sampler(&mut self.runtime, temp, input, prng_key, sampler)
    }
}

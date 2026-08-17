//! Loaded-model facade and prepared generation entry points.

use super::*;

/// A model directory or GGUF file loaded together with its tokenizer and chat template.
///
/// This is the most convenient entry point for text generation: it owns the
/// architecture-specific [`Model`], tokenizer, optional chat template, model id
/// used by the template renderer, and EOS token ids collected from checkpoint
/// and sidecar metadata.
pub struct LoadedModel {
    pub(super) model: Model,
    #[cfg(feature = "media-processing")]
    pub(super) processor: Option<ModelProcessor>,
    pub(super) tokenizer: ChatTokenizer,
    pub(super) tokenizer_fingerprint: [u8; 32],
    pub(super) chat_template: Option<ModelChatTemplate>,
    pub(super) model_id: String,
    pub(super) eos_token_ids: Vec<u32>,
    pub(super) checkpoint_generation_config: Option<CheckpointGenerationConfig>,
    pub(super) constraint_compiler: Result<ConstraintCompiler, String>,
}

struct PreparedChatMtpLaneRuntime<'a, S> {
    input: PreparedChatModelInput<'a>,
    cache: &'a mut ModelCache,
    config: MtpConfig,
    prng_key: Option<Array>,
    sampler: ConstrainedSampler<CheckpointConfiguredSampler<S>>,
    semantic: Box<dyn MtpSemanticState>,
    cancellation: GenerationCancellationToken,
    on_event: Box<dyn FnMut(SemanticEvent) + 'a>,
}

enum PreparedChatModelInput<'a> {
    RenderedPrompt(Array),
    Prepared(&'a PreparedModelInput),
}

impl PreparedChatModelInput<'_> {
    fn with_model_input<T>(&self, function: impl FnOnce(input::ModelInput<'_>) -> T) -> T {
        match self {
            Self::RenderedPrompt(prompt) => {
                let parts = [input::InputPart::text_token_ids(prompt)];
                function(input::ModelInput::new(&parts))
            }
            Self::Prepared(input) => input.with_model_input(function),
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

        PreparedChatModelInput::Prepared(&prepared).with_model_input(|input| {
            assert_eq!(input.parts.len(), 3);
            assert_eq!(input.parts[0].modality, Modality::Text);
            assert_eq!(input.parts[1].modality, Modality::Image);
            assert_eq!(input.parts[2].modality, Modality::Text);
        });
    }
}

fn run_prepared_chat_mtp_batch<'a, B, S>(
    backend: &'a mut B,
    lanes: Vec<PreparedChatMtpLaneRuntime<'a, S>>,
    cache_for_lane: fn(&mut ModelCache) -> Option<&mut B::Cache>,
    cache_kind: &str,
    streams: MtpExecutionStreams<'a>,
    options: MtpSchedulerOptions,
) -> Result<PreparedChatMtpBatchOutput, Exception>
where
    B: crate::runtime::generation::speculative::MtpBackend,
    S: SpeculativeSampler + Clone + 'a,
{
    let mut scheduler = MtpScheduler::new(backend, streams, options)?;
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
    B: crate::runtime::generation::speculative::MtpBackend,
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
    B: crate::runtime::generation::speculative::MtpBackend,
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
    let mut scheduler = MtpScheduler::new(backend, streams, MtpSchedulerOptions::default())?;
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
fn run_embedded_mtp_batch<'a, B, C, S>(
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
    B: crate::runtime::generation::speculative::MtpBackend<Cache = C>,
    S: SpeculativeSampler + Clone + 'a,
{
    let streams = MtpExecutionStreams::single(stream);
    let mut batch_prng = prng_key.map(RandomState::from_key);
    let mut scheduler = MtpScheduler::new(backend, streams, MtpSchedulerOptions::default())?;
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

impl LoadedModel {
    fn resolve_prepared_chat_generation_settings(
        &self,
        settings: PreparedChatGenerationSettings,
    ) -> Result<ResolvedPreparedChatGenerationSettings, Error> {
        let resolved = self.resolve_generation_config(settings.overrides)?;
        let max_tokens = resolved
            .max_new_tokens
            .and_then(NonZeroUsize::new)
            .unwrap_or_else(|| NonZeroUsize::new(256).expect("256 is non-zero"));
        let prng_key = match (settings.prng_key, resolved.temperature) {
            (Some(key), _) => Some(key),
            (None, 0.0) => None,
            (None, _) => Some(safemlx::random::key(0)?),
        };
        Ok(ResolvedPreparedChatGenerationSettings {
            temperature: resolved.temperature,
            max_tokens,
            prng_key,
            checkpoint_sampler: resolved.sampler(),
        })
    }

    /// Validates the observable target/assistant contract used by external MTP.
    ///
    /// Repository names and revisions are deliberately not compatibility keys.
    /// The validation covers the target architecture, shared tensor geometry,
    /// and the token-id vocabulary mapping when the drafter carries tokenizer
    /// metadata.
    pub fn validate_drafter_compatibility(&self, drafter: &LoadedDrafter) -> Result<(), Error> {
        match (&self.model, drafter.kind()) {
            (Model::Gemma4(target), DrafterKind::Gemma4Assistant) => {
                validate_gemma4_drafter(target.args(), drafter.gemma4())?
            }
            (Model::MuseGlimmer(target), DrafterKind::MuseGlimmerDFlash) => {
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
        if let Some(draft_fingerprint) = drafter.tokenizer_fingerprint() {
            if draft_fingerprint != self.tokenizer_fingerprint {
                return Err(Error::UnsupportedArchitecture(
                    "assistant token-id vocabulary mapping does not match the target".into(),
                ));
            }
        }
        Ok(())
    }

    fn prepare_chat_model_input<'a>(
        &self,
        input: PreparedChatInput<'a>,
        stream: &Stream,
    ) -> Result<PreparedChatModelInput<'a>, Error> {
        match input {
            PreparedChatInput::RenderedPrompt(prepared_chat) => {
                Ok(PreparedChatModelInput::RenderedPrompt(
                    self.encode_to_array(prepared_chat.rendered_prompt(), false, stream)?,
                ))
            }
            PreparedChatInput::PreparedModelInput { model_input, .. } => {
                Ok(PreparedChatModelInput::Prepared(model_input))
            }
        }
    }

    fn prepare_chat_mtp_batch_lanes<'a, S>(
        &self,
        lanes: Vec<PreparedChatMtpBatchLane<'a, S>>,
        stream: &Stream,
    ) -> Result<Vec<PreparedChatMtpLaneRuntime<'a, S>>, Error>
    where
        S: SpeculativeSampler + Clone,
    {
        let mut prepared_lanes = Vec::with_capacity(lanes.len());
        for (lane_index, lane) in lanes.into_iter().enumerate() {
            let PreparedChatMtpBatchLane {
                input,
                cache,
                sampling_policy,
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
                    return Err(Error::PreparedChatGeneration(format!(
                        "prepared chat lane {lane_index} does not have an executable semantic plan: {reason}"
                    )));
                }
            };
            let generation_plan = prepared_chat
                .generation_runtime_plan()
                .expect("supported prepared chats carry a generation runtime plan");
            let use_checkpoint_sampler =
                SpeculativeSampler::uses_checkpoint_defaults(&sampling_policy);
            let settings = self.resolve_prepared_chat_generation_settings(settings)?;
            let sampling_policy = CheckpointConfiguredSampler::for_sampler(
                sampling_policy,
                settings.checkpoint_sampler,
                use_checkpoint_sampler,
            );
            let sampler =
                ConstrainedSampler::from_generation_plan(sampling_policy, generation_plan)
                    .map_err(|error| Error::PreparedChatGeneration(error.to_string()))?;
            let decoder = PreparedChatTokenDecoder {
                decoder: self.text_decoder(true),
            };
            let semantic =
                PreparedChatSemanticState::new(decoder, semantic_plan, caller_stop_sequences)
                    .map_err(|error| Error::PreparedChatGeneration(error.to_string()))?;
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

    /// Creates an independent stateful decoder for streaming generated tokens.
    pub fn text_decoder(&self, skip_special_tokens: bool) -> TextDecoder {
        TextDecoder {
            tokenizer: (*self.tokenizer).clone(),
            skip_special_tokens,
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
        }
    }

    /// Generates multiple independent prepared chats through one fair MTP
    /// scheduler using an external assistant.
    ///
    /// Model parameters and execution streams are shared. Every submitted lane
    /// receives a fresh executable constraint/parser runtime, cache, callback,
    /// and target/draft PRNG roots. Events are published only after the
    /// corresponding target cache transaction commits. Each lane's cancellation
    /// token is observed independently.
    pub fn generate_prepared_chat_mtp_batch<S>(
        &mut self,
        request: PreparedChatMtpBatchRequest<'_, S>,
    ) -> Result<PreparedChatMtpBatchOutput, Error>
    where
        S: SpeculativeSampler + Clone,
    {
        let PreparedChatMtpBatchRequest {
            drafter,
            lanes,
            streams,
            scheduler,
        } = request;
        self.validate_drafter_compatibility(drafter)?;
        let prepared_lanes = self.prepare_chat_mtp_batch_lanes(lanes, streams.target())?;

        match &mut self.model {
            Model::Gemma4(target) => {
                let assistant = drafter.gemma4_mut();
                validate_gemma4_drafter(target.args(), assistant)?;
                let mut backend =
                    crate::architectures::gemma4::mtp::Gemma4MtpBackend::new(target, assistant);
                run_prepared_chat_mtp_batch(
                    &mut backend,
                    prepared_lanes,
                    gemma4_mtp_cache,
                    "Gemma 4",
                    streams,
                    scheduler,
                )
                .map_err(|error| Error::PreparedChatGeneration(error.to_string()))
            }
            Model::MuseGlimmer(target) => {
                let assistant = drafter.muse_glimmer_mut();
                let mut backend =
                    crate::architectures::muse_glimmer::mtp::MuseGlimmerMtpBackend::new(
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
                .map_err(|error| Error::PreparedChatGeneration(error.to_string()))
            }
            model => Err(Error::PreparedChatGeneration(format!(
                "MTP runtime adapter is unavailable for model type {} ({:?})",
                model.model_type(),
                model.mtp_capability()
            ))),
        }
    }

    /// Generates multiple independent prepared chats through one fair scheduler
    /// using checkpoint-embedded Qwen MTP heads.
    ///
    /// Model parameters and the execution stream are shared. Every submitted
    /// lane receives a fresh executable constraint/parser runtime, cache,
    /// callback, and PRNG root. Events are published only after the
    /// corresponding target cache transaction commits. Each lane's cancellation
    /// token is observed independently.
    pub fn generate_prepared_chat_embedded_mtp_batch<S>(
        &mut self,
        request: PreparedChatEmbeddedMtpBatchRequest<'_, S>,
    ) -> Result<PreparedChatMtpBatchOutput, Error>
    where
        S: SpeculativeSampler + Clone,
    {
        let PreparedChatEmbeddedMtpBatchRequest {
            lanes,
            stream,
            scheduler,
        } = request;
        let prepared_lanes = self.prepare_chat_mtp_batch_lanes(lanes, stream)?;
        let streams = MtpExecutionStreams::single(stream);
        match &mut self.model {
            Model::DeepSeekV3(target) => {
                let mut backend =
                    crate::runtime::generation::embedded_mtp::EmbeddedMtpBackend::new(target);
                run_prepared_chat_mtp_batch(
                    &mut backend,
                    prepared_lanes,
                    deepseek_mtp_cache,
                    "DeepSeek embedded",
                    streams,
                    scheduler,
                )
                .map_err(|error| Error::PreparedChatGeneration(error.to_string()))
            }
            Model::Inkling(target) => {
                let mut backend =
                    crate::runtime::generation::embedded_mtp::EmbeddedMtpBackend::new(target);
                run_prepared_chat_mtp_batch(
                    &mut backend,
                    prepared_lanes,
                    inkling_mtp_cache,
                    "Inkling embedded",
                    streams,
                    scheduler,
                )
                .map_err(|error| Error::PreparedChatGeneration(error.to_string()))
            }
            Model::NemotronH(target) => {
                let mut backend =
                    crate::runtime::generation::embedded_mtp::EmbeddedMtpBackend::new(target);
                run_prepared_chat_mtp_batch(
                    &mut backend,
                    prepared_lanes,
                    nemotron_mtp_cache,
                    "Nemotron-H embedded",
                    streams,
                    scheduler,
                )
                .map_err(|error| Error::PreparedChatGeneration(error.to_string()))
            }
            Model::Qwen3Next(target) => {
                let mut backend = crate::architectures::qwen::hybrid::mtp::QwenMtpBackend::new(target);
                run_prepared_chat_mtp_batch(
                    &mut backend,
                    prepared_lanes,
                    qwen_next_mtp_cache,
                    "Qwen3-Next embedded",
                    streams,
                    scheduler,
                )
                .map_err(|error| Error::PreparedChatGeneration(error.to_string()))
            }
            Model::Qwen35(target) => {
                let mut backend = crate::architectures::qwen::hybrid::mtp::QwenMtpBackend::new(target);
                run_prepared_chat_mtp_batch(
                    &mut backend,
                    prepared_lanes,
                    qwen35_mtp_cache,
                    "Qwen3.5 embedded",
                    streams,
                    scheduler,
                )
                .map_err(|error| Error::PreparedChatGeneration(error.to_string()))
            }
            model => Err(Error::PreparedChatGeneration(format!(
                "scheduled prepared-chat embedded MTP batch is unavailable for model type {} ({:?})",
                model.model_type(),
                model.mtp_capability()
            ))),
        }
    }

    /// Generates one ordinary structured response from a prepared chat.
    ///
    /// This method validates native-tool support and constructs the constrained
    /// sampler plus a fresh dialect parser before prompt prefill. It then uses
    /// the existing architecture-dispatched token iterator, committing each
    /// token through tokenizer-aware decoding, selective structural-token
    /// preservation, UTF-8 assembly, combined profile/caller stop matching, and
    /// immediate [`SemanticEvent`] delivery.
    /// The rendered chat template is encoded without adding a second layer of
    /// tokenizer special tokens.
    ///
    /// When terminal conditions coincide on one committed token, precedence is
    /// decoded stop sequence, grammar completion, EOS, then max tokens. Grammar
    /// completion is inspected before requesting the next token. Cancellation
    /// is checked before model work, after each synchronous event callback, and
    /// before requesting another token. It emits only `Finished(Cancelled)` as
    /// finalization and does not close incomplete protocol structures.
    pub fn generate_prepared_chat<S, F>(
        &mut self,
        request: PreparedChatGenerationRequest<'_, S, F>,
    ) -> Result<PreparedChatGenerationOutput, Error>
    where
        S: Sampler + Clone,
        F: FnMut(SemanticEvent),
    {
        let PreparedChatGenerationRequest {
            input,
            cache,
            sampling_policy,
            settings,
            caller_stop_sequences,
            stream,
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
        let use_checkpoint_sampler = Sampler::uses_checkpoint_defaults(&sampling_policy);
        let settings = self.resolve_prepared_chat_generation_settings(settings)?;

        with_prepared_chat_runtime(
            prepared_chat,
            sampling_policy,
            settings.checkpoint_sampler,
            use_checkpoint_sampler,
            caller_stop_sequences,
            |runtime| {
                // This closure is the execution boundary: unsupported plans
                // and runtime-construction failures return before it is called.
                let decoder = PreparedChatTokenDecoder {
                    decoder: self.text_decoder(true),
                };
                let raw_decoder =
                    RawTokenDecoder::with_structural_tokens(decoder, runtime.structural_tokens);
                let mut pipeline = CommittedTokenPipeline::new(raw_decoder, runtime.parser);
                let model_input = self.prepare_chat_model_input(input, stream)?;
                model_input.with_model_input(|model_input| {
                    let generator = self.generate_input_with_cache_sampler(
                        cache,
                        settings.temperature,
                        model_input,
                        settings.prng_key,
                        stream,
                        runtime.sampler,
                    );
                    let mut source = GenerationTokenSource { generator, stream };
                    let (token_ids, finish_reason) = drive_committed_generation_cancellable(
                        &mut source,
                        &mut pipeline,
                        prepared_chat.eos_token_ids(),
                        settings.max_tokens,
                        &cancellation,
                        &mut on_event,
                    )
                    .map_err(Error::PreparedChatGeneration)?;
                    Ok(PreparedChatGenerationOutput {
                        token_ids,
                        finish_reason,
                    })
                })
            },
        )
    }

    /// Generates one structured response with an external MTP assistant.
    ///
    /// Target verification, constrained-sampler state, decoded stop matching,
    /// protocol parsing, and event publication share one committed prefix.
    /// Cancellation waits for any in-flight verification to reach its safe
    /// cache boundary before publishing `Finished(Cancelled)`.
    pub fn generate_prepared_chat_mtp<S, F>(
        &mut self,
        request: PreparedChatMtpGenerationRequest<'_, S, F>,
    ) -> Result<PreparedChatMtpGenerationOutput, Error>
    where
        S: SpeculativeSampler + Clone,
        F: FnMut(SemanticEvent),
    {
        let PreparedChatMtpGenerationRequest {
            input,
            drafter,
            cache,
            sampling_policy,
            settings,
            options,
            caller_stop_sequences,
            streams,
            cancellation,
            on_event,
        } = request;
        self.validate_drafter_compatibility(drafter)?;
        let prepared_chat = input.prepared_chat();
        let semantic_plan = match prepared_chat.semantic_support() {
            SemanticSupport::Supported => prepared_chat
                .semantic_runtime_plan()
                .expect("supported prepared chats carry a semantic runtime plan")
                .clone(),
            SemanticSupport::Unsupported { reason } => {
                return Err(Error::PreparedChatGeneration(format!(
                    "prepared chat does not have an executable semantic plan: {reason}"
                )));
            }
        };
        let use_checkpoint_sampler = SpeculativeSampler::uses_checkpoint_defaults(&sampling_policy);
        let settings = self.resolve_prepared_chat_generation_settings(settings)?;
        with_prepared_chat_runtime(
            prepared_chat,
            sampling_policy,
            settings.checkpoint_sampler,
            use_checkpoint_sampler,
            caller_stop_sequences,
            |runtime| {
                let decoder = PreparedChatTokenDecoder {
                    decoder: self.text_decoder(true),
                };
                let semantic =
                    PreparedChatSemanticState::new(decoder, semantic_plan, caller_stop_sequences)
                        .map_err(|error| Error::PreparedChatGeneration(error.to_string()))?;
                let config = MtpConfig {
                    max_tokens: settings.max_tokens.get(),
                    max_draft_tokens: options.max_draft_tokens.get(),
                    temperature: settings.temperature,
                    eos_token_ids: prepared_chat.eos_token_ids().to_vec(),
                };
                let mut sampler = runtime.sampler;
                let model_input = self.prepare_chat_model_input(input, streams.target())?;
                model_input.with_model_input(|model_input| {
                    let (token_ids, stats, finish_reason) = self
                        .model
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
                        .map_err(|error| Error::PreparedChatGeneration(error.to_string()))?;
                    Ok(PreparedChatMtpGenerationOutput {
                        token_ids,
                        finish_reason,
                        stats,
                    })
                })
            },
        )
    }

    /// Generates one structured response with checkpoint-embedded MTP heads.
    /// Cancellation waits for any in-flight verification to reach its safe
    /// cache boundary before publishing `Finished(Cancelled)`.
    pub fn generate_prepared_chat_embedded_mtp<S, F>(
        &mut self,
        request: PreparedChatEmbeddedMtpGenerationRequest<'_, S, F>,
    ) -> Result<PreparedChatMtpGenerationOutput, Error>
    where
        S: SpeculativeSampler + Clone,
        F: FnMut(SemanticEvent),
    {
        let PreparedChatEmbeddedMtpGenerationRequest {
            input,
            cache,
            sampling_policy,
            settings,
            options,
            caller_stop_sequences,
            stream,
            cancellation,
            on_event,
        } = request;
        let prepared_chat = input.prepared_chat();
        let semantic_plan = match prepared_chat.semantic_support() {
            SemanticSupport::Supported => prepared_chat
                .semantic_runtime_plan()
                .expect("supported prepared chats carry a semantic runtime plan")
                .clone(),
            SemanticSupport::Unsupported { reason } => {
                return Err(Error::PreparedChatGeneration(format!(
                    "prepared chat does not have an executable semantic plan: {reason}"
                )));
            }
        };
        let use_checkpoint_sampler = SpeculativeSampler::uses_checkpoint_defaults(&sampling_policy);
        let settings = self.resolve_prepared_chat_generation_settings(settings)?;
        with_prepared_chat_runtime(
            prepared_chat,
            sampling_policy,
            settings.checkpoint_sampler,
            use_checkpoint_sampler,
            caller_stop_sequences,
            |runtime| {
                let decoder = PreparedChatTokenDecoder {
                    decoder: self.text_decoder(true),
                };
                let semantic =
                    PreparedChatSemanticState::new(decoder, semantic_plan, caller_stop_sequences)
                        .map_err(|error| Error::PreparedChatGeneration(error.to_string()))?;
                let config = MtpConfig {
                    max_tokens: settings.max_tokens.get(),
                    max_draft_tokens: options.max_draft_tokens.get(),
                    temperature: settings.temperature,
                    eos_token_ids: prepared_chat.eos_token_ids().to_vec(),
                };
                let mut sampler = runtime.sampler;
                let model_input = self.prepare_chat_model_input(input, stream)?;
                model_input.with_model_input(|model_input| {
                    let (token_ids, stats, finish_reason) = self
                        .model
                        .generate_embedded_mtp_input_with_semantics_and_options(
                            cache,
                            model_input,
                            &config,
                            settings.prng_key,
                            &mut sampler,
                            Box::new(semantic),
                            cancellation,
                            stream,
                            options.scheduler,
                            on_event,
                        )
                        .map_err(|error| Error::PreparedChatGeneration(error.to_string()))?;
                    Ok(PreparedChatMtpGenerationOutput {
                        token_ids,
                        finish_reason,
                        stats,
                    })
                })
            },
        )
    }

    /// Reports whether and how this target can perform MTP generation.
    pub fn mtp_capability(&self) -> MtpCapability {
        self.model.mtp_capability()
    }

    /// Creates independent target caches for an MTP text batch.
    pub fn new_mtp_cache(&self, batch_size: usize) -> MtpCache {
        MtpCache::new((0..batch_size).map(|_| self.new_cache()).collect())
    }

    /// Generates through the architecture-independent MTP path.
    pub fn generate_mtp_input(
        &mut self,
        drafter: &mut LoadedDrafter,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        stream: &Stream,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        self.generate_mtp_input_with_sampler(
            drafter,
            cache,
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
        drafter: &mut LoadedDrafter,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        streams: MtpExecutionStreams<'_>,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        self.generate_mtp_input_with_sampler_and_streams(
            drafter,
            cache,
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
        drafter: &mut LoadedDrafter,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        stream: &Stream,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        self.generate_mtp_input_with_sampler_and_streams(
            drafter,
            cache,
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
        drafter: &mut LoadedDrafter,
        cache: &mut ModelCache,
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
        self.model
            .generate_mtp_input_with_sampler_callback_and_streams(
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
        drafter: &mut LoadedDrafter,
        cache: &mut ModelCache,
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
            cache,
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
        drafter: &mut LoadedDrafter,
        cache: &mut ModelCache,
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
        self.model
            .generate_mtp_input_with_sampler_callback_and_streams_and_options(
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
        drafter: &mut LoadedDrafter,
        cache: &mut ModelCache,
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
            cache,
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
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        stream: &Stream,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        self.generate_embedded_mtp_input_with_sampler(
            cache,
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
        cache: &mut ModelCache,
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
        self.model.generate_embedded_mtp_input_with_sampler(
            cache, input, &config, prng_key, sampler, stream,
        )
    }

    /// Generates through embedded MTP and reports each committed token as it becomes available.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_embedded_mtp_input_with_sampler_callback<S, F>(
        &mut self,
        cache: &mut ModelCache,
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
        self.model
            .generate_embedded_mtp_input_with_sampler_callback(
                cache, input, &config, prng_key, sampler, stream, on_token,
            )
    }

    /// Generates an independently accepting and stopping batch of text prompts.
    ///
    /// Each lane owns a separate cache so rejection lengths and EOS positions
    /// may diverge without padding rejected state back into another sequence.
    pub fn generate_mtp_text_batch<S: SpeculativeSampler + Clone>(
        &mut self,
        drafter: &mut LoadedDrafter,
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
        drafter: &mut LoadedDrafter,
        cache: &mut MtpCache,
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
        drafter: &mut LoadedDrafter,
        cache: &mut MtpCache,
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
        match &mut self.model {
            Model::Gemma4(target) => {
                let assistant = drafter.gemma4_mut();
                validate_gemma4_drafter(target.args(), assistant)?;
                let mut backend =
                    crate::architectures::gemma4::mtp::Gemma4MtpBackend::new(target, assistant);
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
                    crate::architectures::muse_glimmer::mtp::MuseGlimmerMtpBackend::new(
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
        cache: &mut MtpCache,
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
        match &mut self.model {
            Model::DeepSeekV3(target) => {
                let mut backend =
                    crate::runtime::generation::embedded_mtp::EmbeddedMtpBackend::new(target);
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
                    crate::runtime::generation::embedded_mtp::EmbeddedMtpBackend::new(target);
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
                    crate::runtime::generation::embedded_mtp::EmbeddedMtpBackend::new(target);
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
                    crate::architectures::qwen::hybrid::mtp::QwenMtpBackend::new(target);
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
                    crate::architectures::qwen::hybrid::mtp::QwenMtpBackend::new(target);
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
        self.model.residency_report()
    }

    /// Returns experimental dense-stream telemetry when enabled.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<crate::runtime::execution::layerwise::DenseDiskStreamReport>, Error> {
        self.model.dense_stream_report()
    }

    /// Returns sparse routed-expert cache telemetry when enabled.
    pub fn expert_cache_report(
        &self,
    ) -> Result<Option<crate::runtime::residency::expert_cache::ExpertCacheReport>, Error> {
        self.model.expert_cache_report()
    }

    /// Loads a supported model directory or GGUF file with its tokenizer.
    ///
    /// GGUF tokenizers are reconstructed from embedded metadata. A sibling
    /// `tokenizer.json` is used only when the embedded tokenizer is absent or
    /// uses an unsupported tokenizer model.
    pub fn load(
        model_dir: impl AsRef<Path>,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<Self, Error> {
        Self::load_with_options(
            model_dir,
            ModelLoadOptions::default(),
            stream,
            weights_stream,
        )
    }

    /// Loads a supported model using architecture-independent weight options.
    pub fn load_with_options(
        model_dir: impl AsRef<Path>,
        options: ModelLoadOptions,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<Self, Error> {
        let model_dir = model_dir.as_ref();
        ensure_executable_load_options(options)?;
        if is_gguf_file(model_dir) {
            let sidecar_dir = gguf_sidecar_dir(model_dir);
            let checkpoint_generation_config = read_checkpoint_generation_config(sidecar_dir)?;
            let LoadedGgufModel {
                model,
                eos_token_ids,
                chat_template,
                tokenizer,
                #[cfg(feature = "media-processing")]
                processor,
            } = load_gguf_model_data(model_dir, true, options, stream, weights_stream)?;
            let GgufTokenizer {
                tokenizer,
                template_kwargs,
            } = tokenizer.expect("GGUF tokenizer requested by the combined loader");
            let mut tokenizer = ChatTokenizer::from_tokenizer(tokenizer);
            tokenizer.set_template_kwargs(template_kwargs);
            let tokenizer_fingerprint = crate::api::tokenizer_vocabulary_fingerprint(&tokenizer);
            let chat_template = chat_template.or(load_chat_template(sidecar_dir)?);
            let constraint_compiler =
                ConstraintCompiler::from_tokenizer(&tokenizer, &eos_token_ids);
            return Ok(Self {
                model,
                #[cfg(feature = "media-processing")]
                processor,
                tokenizer,
                tokenizer_fingerprint,
                chat_template,
                model_id: model_dir.display().to_string(),
                eos_token_ids,
                checkpoint_generation_config,
                constraint_compiler,
            });
        }
        let metadata = read_model_metadata(model_dir)?;
        let checkpoint_generation_config = read_checkpoint_generation_config(model_dir)?;
        let eos_token_ids = eos_token_ids_from_sidecar_dir(model_dir)?;
        let model_type = effective_model_type(&metadata);
        let kind = ModelKind::from_model_type(&model_type)?;
        let mut tokenizer = ChatTokenizer::from_tokenizer(load_tokenizer(model_dir)?);
        tokenizer.set_template_kwargs(load_tokenizer_template_kwargs(model_dir)?);
        let tokenizer_fingerprint = crate::api::tokenizer_vocabulary_fingerprint(&tokenizer);
        let constraint_compiler = ConstraintCompiler::from_tokenizer(&tokenizer, &eos_token_ids);
        let chat_template = load_chat_template(model_dir)?;
        #[cfg(feature = "media-processing")]
        let processor = load_processor(model_dir)?;
        let model = match kind {
            ModelKind::PersonaPlex => {
                return Err(Error::UnsupportedArchitecture(
                    "PersonaPlex is a realtime speech-to-speech token model; use architectures::moshi::personaplex instead of LoadedModel".into(),
                ));
            }
            _ => load_model_for_kind(kind, model_dir, options, stream, weights_stream)?,
        };

        Ok(Self {
            model,
            #[cfg(feature = "media-processing")]
            processor,
            tokenizer,
            tokenizer_fingerprint,
            chat_template,
            model_id: model_type,
            eos_token_ids,
            checkpoint_generation_config,
            constraint_compiler,
        })
    }

    /// Returns the effective runtime model type.
    pub fn model_type(&self) -> &str {
        self.model.model_type()
    }

    /// Returns sampling values declared by `generation_config.json`, if present.
    pub fn checkpoint_generation_config(&self) -> Option<&CheckpointGenerationConfig> {
        self.checkpoint_generation_config.as_ref()
    }

    /// Resolves request overrides over checkpoint recommendations and SafeMLX fallbacks.
    pub fn resolve_generation_config(
        &self,
        overrides: GenerationConfigOverrides,
    ) -> Result<ResolvedGenerationConfig, Error> {
        resolve_generation_config(self.checkpoint_generation_config.as_ref(), overrides)
    }

    /// Returns checkpoint-native quantization storage statistics when available.
    pub fn native_quantization_stats(
        &self,
    ) -> Option<&safemlx::native_quantization::NativeQuantizationStats> {
        self.model.native_quantization_stats()
    }

    /// Returns the model id passed to chat-template rendering.
    pub fn model_id_for_template(&self) -> &str {
        &self.model_id
    }

    /// Returns whether a chat template is available for this model.
    pub fn has_chat_template(&self) -> bool {
        self.chat_template.is_some()
    }

    /// Returns the stable identity of the template that would be selected for
    /// the supplied tools.
    pub fn selected_chat_template_identity(
        &self,
        tools: Option<&[serde_json::Value]>,
    ) -> Result<Option<ChatTemplateIdentity>, Error> {
        self.chat_template
            .as_ref()
            .map(|templates| {
                templates
                    .select(tools)
                    .map(|selected| selected.identity().clone())
            })
            .transpose()
            .map_err(Into::into)
    }

    /// Returns whether this model directory includes a supported media processor.
    #[cfg(feature = "media-processing")]
    pub fn has_processor(&self) -> bool {
        self.processor.is_some()
    }

    /// Returns the loaded architecture-dispatched media processor, if available.
    #[cfg(feature = "media-processing")]
    pub fn processor(&self) -> Option<&ModelProcessor> {
        self.processor.as_ref()
    }

    /// Tokenizes and preprocesses ordered text and media segments.
    #[cfg(feature = "media-processing")]
    pub fn prepare_input(&self, input: &[ProcessorInput<'_>]) -> Result<PreparedModelInput, Error> {
        let processor = self.processor.as_ref().ok_or_else(|| {
            Error::Processor(format!(
                "model type '{}' does not have a loaded media processor",
                self.model_type()
            ))
        })?;
        processor.prepare_input(input, &mut |text| self.encode(text, false))
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
    ) -> Result<PreparedModelInput, Error> {
        let processor = self.processor.as_ref().ok_or_else(|| {
            Error::Processor(format!(
                "model type '{}' does not have a loaded media processor",
                self.model_type()
            ))
        })?;
        processor.prepare_chat_input(prepared_chat.rendered_prompt(), bindings, &mut |text| {
            self.encode(text, false)
        })
    }

    /// Returns likely user-provided kwargs referenced by the loaded chat template.
    ///
    /// This is static template analysis and does not infer value types or
    /// defaults. Standard chat-template variables supplied by this crate are
    /// excluded.
    pub fn chat_template_kwargs(&self) -> Result<Vec<String>, Error> {
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
    ///
    /// The selected checkpoint template is rendered with and without its
    /// generation prompt so the appended contribution is available
    /// independently. Native tool support is reported only when atomic
    /// structural-token facts and bounded render probes establish one
    /// unambiguous wire protocol.
    pub fn prepare_chat(&mut self, request: ChatTemplateRequest) -> Result<PreparedChat, Error> {
        let template = self
            .chat_template
            .clone()
            .ok_or(Error::MissingChatTemplate)?;
        prepare_chat_from_parts(
            &mut self.tokenizer,
            template,
            &self.model_id,
            &self.eos_token_ids,
            Some(&self.constraint_compiler),
            request,
        )
    }

    /// Applies the loaded chat template to structured conversations.
    ///
    /// Returns `Ok(None)` when no chat template is available.
    pub fn apply_chat_template<'a, I, R, T>(
        &'a mut self,
        conversations: I,
        tools: Option<&'a [serde_json::Value]>,
        add_generation_prompt: bool,
    ) -> Result<Option<String>, Error>
    where
        I: IntoIterator<Item = Chat<'a, R, T>>,
        R: Serialize + 'a,
        T: Serialize + 'a,
    {
        self.apply_chat_template_with_kwargs(conversations, tools, add_generation_prompt, None)
    }

    /// Applies the loaded chat template to structured conversations with extra template variables.
    ///
    /// Returns `Ok(None)` when no chat template is available.
    pub fn apply_chat_template_with_kwargs<'a, I, R, T>(
        &'a mut self,
        conversations: I,
        tools: Option<&'a [serde_json::Value]>,
        add_generation_prompt: bool,
        template_kwargs: Option<&'a serde_json::Map<String, serde_json::Value>>,
    ) -> Result<Option<String>, Error>
    where
        I: IntoIterator<Item = Chat<'a, R, T>>,
        R: Serialize + 'a,
        T: Serialize + 'a,
    {
        let Some(template) = self.chat_template.clone() else {
            return Ok(None);
        };

        let rendered = self.tokenizer.apply_chat_template(
            template.clone(),
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

    /// Applies the loaded chat template to JSON-valued conversations.
    ///
    /// Returns `Ok(None)` when no chat template is available.
    pub fn apply_chat_template_json(
        &mut self,
        conversations: impl IntoIterator<Item = Vec<serde_json::Value>>,
        tools: Option<&[serde_json::Value]>,
        add_generation_prompt: bool,
    ) -> Result<Option<String>, Error> {
        self.apply_chat_template_json_with_kwargs(conversations, tools, add_generation_prompt, None)
    }

    /// Applies the loaded chat template to JSON-valued conversations with extra template variables.
    ///
    /// Returns `Ok(None)` when no chat template is available.
    pub fn apply_chat_template_json_with_kwargs(
        &mut self,
        conversations: impl IntoIterator<Item = Vec<serde_json::Value>>,
        tools: Option<&[serde_json::Value]>,
        add_generation_prompt: bool,
        template_kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Result<Option<String>, Error> {
        let Some(template) = self.chat_template.clone() else {
            return Ok(None);
        };

        let rendered = self.tokenizer.apply_chat_template_json(
            template.clone(),
            conversations,
            tools,
            &self.model_id,
            add_generation_prompt,
            template_kwargs,
        )?;
        Ok(rendered.into_iter().next())
    }

    /// Encodes text to tokenizer ids.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, Error> {
        Ok(self
            .tokenizer
            .encode(text, add_special_tokens)?
            .get_ids()
            .to_vec())
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

    /// Decodes tokenizer ids back to text.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, Error> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(Into::into)
    }

    /// Returns EOS token ids collected from the model's checkpoint metadata.
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    /// Returns true when `id` is one of the configured EOS token ids.
    pub fn is_eos_token(&self, id: u32) -> bool {
        self.eos_token_ids.contains(&id)
    }

    /// Creates an empty cache value appropriate for the loaded model.
    pub fn new_cache(&self) -> ModelCache {
        self.model.new_cache()
    }

    /// Creates cache state under an explicit cache-residency policy.
    pub fn new_cache_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<ModelCache, Exception> {
        self.model.new_cache_with_options(policy)
    }

    /// Creates a paged cache attached to an aggregate process-wide pool.
    pub fn new_cache_in_pool(
        &self,
        options: PagedCacheOptions,
        pool: CacheResidencyPool,
    ) -> Result<ModelCache, Exception> {
        let options = options
            .with_pool(pool)
            .map_err(|error| Exception::custom(error.to_string()))?;
        self.model
            .new_cache_with_options(CacheResidencyPolicy::Paged(options))
    }

    /// Returns the canonical cache-relevant architecture identity for this loaded model.
    pub fn prompt_cache_architecture_fingerprint(&self) -> Result<String, Exception> {
        self.model.prompt_cache_architecture_fingerprint()
    }

    /// Returns the exact ordered prompt-cache state and attention layout.
    pub fn prompt_cache_layer_layout(&self) -> Result<LayerSchedule<LayerCachePolicy>, Exception> {
        self.model.prompt_cache_layer_layout()
    }

    /// Returns each owned layer's processed-token delta from the persisted prefix.
    pub fn prompt_cache_layer_prefix_offsets(&self) -> Result<Vec<i32>, Exception> {
        self.model.prompt_cache_layer_prefix_offsets()
    }

    /// Opens a compatible reusable prefix for this loaded model.
    ///
    /// Block-paged families retain mmap-lazy attention shards. Resident hybrid
    /// and multimodal families materialize their validated fixed state on
    /// `stream` before returning.
    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(ModelCache, PromptCacheManifest), Exception> {
        self.model
            .load_prompt_cache(directory, expected, prefix_token_ids, options, stream)
    }

    /// Atomically saves a completed immutable prefix with model-owned state validation.
    pub fn save_prompt_cache(
        &self,
        cache: &mut ModelCache,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Exception> {
        self.model.save_prompt_cache(
            cache,
            destination,
            descriptor,
            prefix_token_ids,
            options,
            stream,
        )
    }

    /// Submits prompt prefill using a cache returned by [`LoadedModel::new_cache`].
    pub fn submit_prefill(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut ModelCache,
        stream: &Stream,
    ) -> Result<safemlx_lm_core::Submission<Array, crate::backend::mlx::MlxCompletion>, Error> {
        self.model.submit_prefill(input, cache, stream)
    }

    /// Submits cached decode using an existing model cache.
    pub fn submit_decode(
        &mut self,
        input: Array,
        cache: &mut ModelCache,
        stream: &Stream,
    ) -> Result<safemlx_lm_core::Submission<Array, crate::backend::mlx::MlxCompletion>, Error> {
        self.model.submit_decode(input, cache, stream)
    }

    /// Submits prompt prefill from an owned processor result.
    #[cfg(feature = "media-processing")]
    pub fn submit_prepared_prefill(
        &mut self,
        input: &PreparedModelInput,
        cache: &mut ModelCache,
        stream: &Stream,
    ) -> Result<safemlx_lm_core::Submission<Array, crate::backend::mlx::MlxCompletion>, Error> {
        input.with_model_input(|input| self.submit_prefill(input, cache, stream))
    }

    /// Submits prompt prefill while reporting detailed activations.
    ///
    /// The submitted output has shape `[batch, vocab]` and matches
    /// [`LoadedModel::submit_prefill`] for the same model/cache.
    pub fn submit_prefill_with_observer(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut ModelCache,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<safemlx_lm_core::Submission<Array, crate::backend::mlx::MlxCompletion>, Error> {
        self.model
            .submit_prefill_with_observer(input, cache, stream, observer)
    }

    /// Submits an owned processor result while observing activations.
    #[cfg(feature = "media-processing")]
    pub fn submit_prepared_prefill_with_observer(
        &mut self,
        input: &PreparedModelInput,
        cache: &mut ModelCache,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<safemlx_lm_core::Submission<Array, crate::backend::mlx::MlxCompletion>, Error> {
        input.with_model_input(|input| {
            self.submit_prefill_with_observer(input, cache, stream, observer)
        })
    }

    /// Creates a token iterator using checkpoint generation defaults plus typed overrides.
    pub fn generate_input_with_cache<'a>(
        &'a mut self,
        cache: &'a mut ModelCache,
        input: input::ModelInput<'_>,
        overrides: GenerationConfigOverrides,
        prng_key: Option<Array>,
        stream: &'a Stream,
    ) -> Result<MlxGeneration<'a, crate::runtime::generation::sampler::GenerationSampler>, Error>
    {
        let resolved = self.resolve_generation_config(overrides)?;
        let prng_key = match (prng_key, resolved.temperature) {
            (Some(key), _) => Some(key),
            (None, 0.0) => None,
            (None, _) => Some(safemlx::random::key(0)?),
        };
        Ok(self.model.generate_input_with_cache_sampler(
            cache,
            resolved.temperature,
            input,
            prng_key,
            stream,
            resolved.sampler(),
        ))
    }

    /// Creates a token iterator from typed input with a caller-provided sampler.
    pub fn generate_input_with_cache_sampler<'a, S>(
        &'a mut self,
        cache: &'a mut ModelCache,
        temp: f32,
        input: input::ModelInput<'_>,
        prng_key: Option<Array>,
        stream: &'a Stream,
        sampler: S,
    ) -> MlxGeneration<'a, S>
    where
        S: Sampler,
    {
        self.model
            .generate_input_with_cache_sampler(cache, temp, input, prng_key, stream, sampler)
    }

    /// Returns a mutable reference to the underlying architecture-specific model.
    pub fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }
}

pub(super) struct LoadedGgufModel {
    pub(super) model: Model,
    #[cfg(feature = "media-processing")]
    processor: Option<ModelProcessor>,
    eos_token_ids: Vec<u32>,
    chat_template: Option<ModelChatTemplate>,
    tokenizer: Option<GgufTokenizer>,
}

pub(super) fn load_gguf_model_data(
    gguf_file: &Path,
    load_tokenizer: bool,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedGgufModel, Error> {
    let checkpoint = GgufCheckpoint::open(gguf_file)?;
    let metadata = crate::runtime::checkpoint::load::gguf_metadata(&checkpoint);
    let sidecar_eos_token_ids = eos_token_ids_from_sidecar_dir(gguf_sidecar_dir(gguf_file))?;
    let gguf_eos_token_ids = gguf_eos_token_ids(&metadata)?;
    let architecture = match metadata.get("general.architecture") {
        Some(GgufMetadataValue::String(architecture)) => architecture.clone(),
        Some(_) => {
            return Err(Error::UnsupportedArchitecture(
                "GGUF metadata key \"general.architecture\" has the wrong type".into(),
            ));
        }
        None => {
            return Err(Error::UnsupportedArchitecture(
                "GGUF metadata is missing required key \"general.architecture\"".into(),
            ));
        }
    };
    let gguf_architecture = GgufArchitecture::resolve(&architecture)?;
    gguf_architecture.validate_catalog(&checkpoint, &metadata)?;
    gguf_architecture.validate_load_policy(options)?;
    super::structural::validate_gguf(gguf_architecture, &checkpoint, &metadata, options)
        .into_loader_result()?;
    let chat_template = match metadata.get("tokenizer.chat_template") {
        Some(GgufMetadataValue::String(template)) => {
            Some(ModelChatTemplate::Single(template.clone()))
        }
        Some(_) => {
            return Err(Error::UnsupportedArchitecture(
                "GGUF metadata key \"tokenizer.chat_template\" has the wrong type".into(),
            ));
        }
        None => None,
    };
    let tokenizer = load_tokenizer
        .then(|| load_gguf_tokenizer_from_metadata(gguf_file, &metadata))
        .transpose()?;
    validate_gguf_quantization_source(&checkpoint, &metadata, options.quantization)?;
    #[cfg(feature = "media-processing")]
    let mut processor = None;

    let (model, architecture_eos_token_ids) = if let Some(quantization) = options
        .quantization
        .filter(|_| options.weight_residency.is_fully_resident())
    {
        match gguf_architecture {
            GgufArchitecture::KimiLinear => {
                let loaded = kimi_linear::load_gguf_checkpoint(
                    &checkpoint,
                    metadata.clone(),
                    Some(quantization),
                    stream,
                    weights_stream,
                )?;
                let model = crate::architectures::kimi_linear::layerwise::execute_transformed_kimi_linear_model(
                    loaded.model, stream, weights_stream,
                )?;
                (Model::KimiLinear(model), loaded.eos_token_ids)
            }
            GgufArchitecture::DeepSeek2 => {
                let loaded = deepseek_v3::load_gguf_checkpoint(
                    &checkpoint,
                    metadata.clone(),
                    Some(quantization),
                    stream,
                    weights_stream,
                )?;
                let model = crate::architectures::deepseek_v3::layerwise::execute_transformed_deepseek_v3_model(
                    loaded.model, stream, weights_stream,
                )?;
                (Model::DeepSeekV3(model), loaded.eos_token_ids)
            }
            GgufArchitecture::DeepSeek4 => {
                let (loaded, eos_token_ids) = crate::architectures::deepseek_v4::layerwise::load_deepseek_v4_gguf_layerwise_model(
                    &checkpoint,
                    &metadata,
                    options.weight_residency,
                    Some(quantization),
                    stream,
                    weights_stream,
                )?;
                (Model::DeepSeekV4Layerwise(Box::new(loaded)), eos_token_ids)
            }
            GgufArchitecture::GptOss => {
                let loaded = gpt_oss::load_gguf_checkpoint(
                    &checkpoint,
                    metadata.clone(),
                    Some(quantization),
                    stream,
                    weights_stream,
                )?;
                let model =
                    crate::architectures::gpt_oss::layerwise::execute_transformed_gpt_oss_model(
                        loaded.model,
                        stream,
                        weights_stream,
                    )?;
                (Model::GptOss(model), loaded.eos_token_ids)
            }
            GgufArchitecture::Gemma4 => {
                let mmproj = gemma4::open_sibling_mmproj(gguf_file)?;
                #[cfg(any(feature = "image-processing", feature = "audio-processing"))]
                if let Some(mmproj) = &mmproj {
                    processor = Some(ModelProcessor::load_gemma4_gguf(
                        &metadata,
                        &mmproj.metadata,
                    )?);
                }
                let loaded = gemma4::load_gemma4_gguf_checkpoint(
                    &checkpoint,
                    metadata.clone(),
                    mmproj.as_ref(),
                    Some(quantization),
                    stream,
                    weights_stream,
                )?;
                let model =
                    crate::architectures::gemma4::layerwise::execute_transformed_gemma4_model_with_modalities(
                        loaded.model,
                        loaded.vision_config,
                        loaded.audio_config,
                        stream,
                        weights_stream,
                    )?;
                (Model::Gemma4(Box::new(model)), loaded.eos_token_ids)
            }
            GgufArchitecture::Llama | GgufArchitecture::Mistral => {
                let loaded = llama::load_llama_gguf_checkpoint(
                    &checkpoint,
                    metadata.clone(),
                    Some(quantization),
                    stream,
                    weights_stream,
                )?;
                let model =
                    crate::architectures::llama::layerwise::execute_transformed_llama_model(
                        loaded.model,
                        stream,
                        weights_stream,
                    )?;
                (Model::Llama(model), loaded.eos_token_ids)
            }
            GgufArchitecture::MuseGlimmer => {
                #[cfg(feature = "image-processing")]
                if let Some(mmproj) =
                    crate::architectures::muse_glimmer::open_sibling_mmproj(gguf_file)?
                {
                    processor = Some(ModelProcessor::load_muse_glimmer_gguf(&mmproj.metadata)?);
                }
                let (loaded, eos_token_ids) =
                    crate::architectures::muse_glimmer::layerwise::load_gguf_checkpoint(
                        &checkpoint,
                        &metadata,
                        &architecture,
                        options.weight_residency,
                        Some(quantization),
                        stream,
                        weights_stream,
                    )?;
                (Model::MuseGlimmer(loaded), eos_token_ids)
            }
            GgufArchitecture::Lfm2 | GgufArchitecture::Lfm2Moe => {
                let loaded = lfm2::load_gguf_checkpoint(
                    &checkpoint,
                    metadata.clone(),
                    Some(quantization),
                    stream,
                    weights_stream,
                )?;
                let model = crate::architectures::lfm2::layerwise::execute_transformed_lfm2_model(
                    loaded.model,
                    stream,
                    weights_stream,
                )?;
                (Model::Lfm2(model), loaded.eos_token_ids)
            }
            GgufArchitecture::Qwen2 | GgufArchitecture::Qwen3 | GgufArchitecture::Qwen3Moe => {
                let loaded = dense_qwen::load_gguf_checkpoint(
                    &checkpoint,
                    metadata.clone(),
                    Some(quantization),
                    stream,
                    weights_stream,
                )?;
                let model =
                    crate::architectures::qwen::dense::layerwise::execute_transformed_model(
                        loaded.model,
                        stream,
                        weights_stream,
                    )?;
                (Model::DenseQwen(model), loaded.eos_token_ids)
            }
            GgufArchitecture::Qwen3Vl | GgufArchitecture::Qwen3VlMoe => {
                let mmproj_file = qwen3_vl::find_qwen3_vl_mmproj(gguf_file)?;
                let vision_checkpoint = GgufCheckpoint::open(mmproj_file)?;
                let vision_metadata =
                    crate::runtime::checkpoint::load::gguf_metadata(&vision_checkpoint);
                let loaded = qwen3_vl::load_qwen3_vl_gguf_checkpoint(
                    &checkpoint,
                    metadata.clone(),
                    &vision_checkpoint,
                    vision_metadata,
                    Some(quantization),
                    stream,
                    weights_stream,
                )?;
                let model =
                    crate::architectures::qwen::vl::layerwise::execute_transformed_qwen3_vl_model(
                        loaded.model,
                        stream,
                        weights_stream,
                    )?;
                let model = if gguf_architecture == GgufArchitecture::Qwen3VlMoe {
                    Model::Qwen3VlMoe(model)
                } else {
                    Model::Qwen3Vl(model)
                };
                (model, loaded.eos_token_ids)
            }
            GgufArchitecture::Qwen35
            | GgufArchitecture::Qwen35Moe
            | GgufArchitecture::Qwen3Next => {
                let mmproj = if gguf_architecture == GgufArchitecture::Qwen3Next {
                    None
                } else {
                    qwen3_5::open_sibling_mmproj(gguf_file)?
                };
                #[cfg(feature = "image-processing")]
                if mmproj.is_some() {
                    processor = ModelProcessor::load_qwen(gguf_sidecar_dir(gguf_file))?;
                }
                let loaded = qwen3_5::load_qwen3_5_gguf_checkpoint(
                    &checkpoint,
                    metadata.clone(),
                    mmproj.as_ref(),
                    Some(quantization),
                    stream,
                    weights_stream,
                )?;
                let is_next = gguf_architecture == GgufArchitecture::Qwen3Next;
                let model = crate::architectures::qwen::hybrid::layerwise::execute_transformed_qwen_hybrid_model(
                    loaded.model, quantization, stream, weights_stream,
                )?;
                let model = if is_next {
                    Model::Qwen3Next(model)
                } else {
                    Model::Qwen35(model)
                };
                (model, loaded.eos_token_ids)
            }
            GgufArchitecture::Inkling => {
                let mmproj = inkling::open_sibling_mmproj(gguf_file)?;
                #[cfg(feature = "media-processing")]
                if mmproj.is_some() {
                    processor = Some(ModelProcessor::load_inkling_gguf(&metadata)?);
                }
                let (loaded, eos_token_ids) =
                    crate::architectures::inkling::layerwise::load_inkling_gguf_layerwise_model(
                        &checkpoint,
                        &metadata,
                        mmproj.as_ref(),
                        options.weight_residency,
                        Some(quantization),
                        stream,
                        weights_stream,
                    )?;
                (Model::Inkling(loaded), eos_token_ids)
            }
            GgufArchitecture::NemotronH | GgufArchitecture::NemotronHMoe => {
                let (loaded, eos_token_ids) = crate::architectures::nemotron_h::layerwise::load_nemotron_h_gguf_layerwise_model(
                    &checkpoint,
                    &metadata,
                    options.weight_residency,
                    Some(quantization),
                    stream,
                    weights_stream,
                )?;
                (Model::NemotronH(loaded), eos_token_ids)
            }
        }
    } else {
        match gguf_architecture {
            GgufArchitecture::KimiLinear => {
                let (loaded, eos_token_ids) =
                    crate::architectures::kimi_linear::layerwise::load_kimi_linear_gguf_layerwise_model(
                        &checkpoint,
                        &metadata,
                        options.weight_residency,
                        options.quantization,
                        stream,
                        weights_stream,
                    )?;
                (Model::KimiLinear(loaded), eos_token_ids)
            }
            GgufArchitecture::DeepSeek2 => {
                let (loaded, eos_token_ids) =
                    crate::architectures::deepseek_v3::layerwise::load_deepseek_v3_gguf_layerwise_model(
                        &checkpoint,
                        &metadata,
                        options.weight_residency,
                        options.quantization,
                        stream,
                        weights_stream,
                    )?;
                (Model::DeepSeekV3(loaded), eos_token_ids)
            }
            GgufArchitecture::DeepSeek4 => {
                let (loaded, eos_token_ids) = crate::architectures::deepseek_v4::layerwise::load_deepseek_v4_gguf_layerwise_model(
                    &checkpoint,
                    &metadata,
                    options.weight_residency,
                    options.quantization,
                    stream,
                    weights_stream,
                )?;
                (Model::DeepSeekV4Layerwise(Box::new(loaded)), eos_token_ids)
            }
            GgufArchitecture::GptOss => {
                let (loaded, eos_token_ids) =
                    crate::architectures::gpt_oss::layerwise::load_gpt_oss_gguf_layerwise_model(
                        &checkpoint,
                        &metadata,
                        options.weight_residency,
                        options.quantization,
                        stream,
                        weights_stream,
                    )?;
                (Model::GptOss(loaded), eos_token_ids)
            }
            GgufArchitecture::Inkling => {
                let mmproj = inkling::open_sibling_mmproj(gguf_file)?;
                #[cfg(feature = "media-processing")]
                if mmproj.is_some() {
                    processor = Some(ModelProcessor::load_inkling_gguf(&metadata)?);
                }
                let (loaded, eos_token_ids) =
                    crate::architectures::inkling::layerwise::load_inkling_gguf_layerwise_model(
                        &checkpoint,
                        &metadata,
                        mmproj.as_ref(),
                        options.weight_residency,
                        options.quantization,
                        stream,
                        weights_stream,
                    )?;
                (Model::Inkling(loaded), eos_token_ids)
            }
            GgufArchitecture::Gemma4 => {
                let mmproj = gemma4::open_sibling_mmproj(gguf_file)?;
                #[cfg(any(feature = "image-processing", feature = "audio-processing"))]
                if let Some(mmproj) = &mmproj {
                    processor = Some(ModelProcessor::load_gemma4_gguf(
                        &metadata,
                        &mmproj.metadata,
                    )?);
                }
                let (loaded, eos_token_ids) =
                    crate::architectures::gemma4::layerwise::load_gemma4_gguf_layerwise_model(
                        &checkpoint,
                        &metadata,
                        mmproj.as_ref(),
                        options.weight_residency,
                        options.quantization,
                        stream,
                        weights_stream,
                    )?;
                (Model::Gemma4(Box::new(loaded)), eos_token_ids)
            }
            GgufArchitecture::Llama | GgufArchitecture::Mistral => {
                let (loaded, eos_token_ids) =
                    crate::architectures::llama::layerwise::load_llama_gguf_model(
                        &checkpoint,
                        &metadata,
                        options.weight_residency,
                        options.quantization,
                        stream,
                        weights_stream,
                    )?;
                (Model::Llama(loaded), eos_token_ids)
            }
            GgufArchitecture::MuseGlimmer => {
                #[cfg(feature = "image-processing")]
                if let Some(mmproj) =
                    crate::architectures::muse_glimmer::open_sibling_mmproj(gguf_file)?
                {
                    processor = Some(ModelProcessor::load_muse_glimmer_gguf(&mmproj.metadata)?);
                }
                let (loaded, eos_token_ids) =
                    crate::architectures::muse_glimmer::layerwise::load_gguf_checkpoint(
                        &checkpoint,
                        &metadata,
                        &architecture,
                        options.weight_residency,
                        options.quantization,
                        stream,
                        weights_stream,
                    )?;
                (Model::MuseGlimmer(loaded), eos_token_ids)
            }
            GgufArchitecture::Lfm2 | GgufArchitecture::Lfm2Moe => {
                let (loaded, eos_token_ids) =
                    crate::architectures::lfm2::layerwise::load_lfm2_gguf_layerwise_model(
                        &checkpoint,
                        &metadata,
                        options.weight_residency,
                        options.quantization,
                        stream,
                        weights_stream,
                    )?;
                (Model::Lfm2(loaded), eos_token_ids)
            }
            GgufArchitecture::NemotronH | GgufArchitecture::NemotronHMoe => {
                let (loaded, eos_token_ids) =
                crate::architectures::nemotron_h::layerwise::load_nemotron_h_gguf_layerwise_model(
                    &checkpoint,
                    &metadata,
                    options.weight_residency,
                    options.quantization,
                    stream,
                    weights_stream,
                )?;
                (Model::NemotronH(loaded), eos_token_ids)
            }
            GgufArchitecture::Qwen2 | GgufArchitecture::Qwen3 | GgufArchitecture::Qwen3Moe => {
                let (loaded, eos_token_ids) =
                    crate::architectures::qwen::dense::layerwise::load_gguf_checkpoint(
                        &checkpoint,
                        &metadata,
                        &architecture,
                        options.weight_residency,
                        options.quantization,
                        stream,
                        weights_stream,
                    )?;
                (Model::DenseQwen(loaded), eos_token_ids)
            }
            GgufArchitecture::Qwen3Vl | GgufArchitecture::Qwen3VlMoe => {
                let mmproj_file = qwen3_vl::find_qwen3_vl_mmproj(gguf_file)?;
                let vision_checkpoint = GgufCheckpoint::open(mmproj_file)?;
                let vision_metadata =
                    crate::runtime::checkpoint::load::gguf_metadata(&vision_checkpoint);
                let (loaded, eos_token_ids) =
                    crate::architectures::qwen::vl::layerwise::load_qwen3_vl_gguf_layerwise_model(
                        &checkpoint,
                        &metadata,
                        &vision_checkpoint,
                        &vision_metadata,
                        options.weight_residency,
                        options.quantization,
                        stream,
                        weights_stream,
                    )?;
                let model = if gguf_architecture == GgufArchitecture::Qwen3VlMoe {
                    Model::Qwen3VlMoe(loaded)
                } else {
                    Model::Qwen3Vl(loaded)
                };
                (model, eos_token_ids)
            }
            GgufArchitecture::Qwen35
            | GgufArchitecture::Qwen35Moe
            | GgufArchitecture::Qwen3Next => {
                let mmproj = if gguf_architecture == GgufArchitecture::Qwen3Next {
                    None
                } else {
                    qwen3_5::open_sibling_mmproj(gguf_file)?
                };
                #[cfg(feature = "image-processing")]
                if mmproj.is_some() {
                    processor = ModelProcessor::load_qwen(gguf_sidecar_dir(gguf_file))?;
                }
                let (loaded, eos_token_ids, is_next) =
                    crate::architectures::qwen::hybrid::layerwise::load_qwen_hybrid_gguf_layerwise_model(
                        &checkpoint,
                        &metadata,
                        mmproj.as_ref(),
                        options.weight_residency,
                        options.quantization,
                        stream,
                        weights_stream,
                    )?;
                let model = if is_next {
                    Model::Qwen3Next(loaded)
                } else {
                    Model::Qwen35(loaded)
                };
                (model, eos_token_ids)
            }
        }
    };
    let eos_token_ids = merge_eos_token_id_sources([
        sidecar_eos_token_ids,
        architecture_eos_token_ids,
        gguf_eos_token_ids,
    ]);
    Ok(LoadedGgufModel {
        model,
        #[cfg(feature = "media-processing")]
        processor,
        eos_token_ids,
        chat_template,
        tokenizer,
    })
}

pub(crate) fn validate_gguf_quantization_source<
    S: crate::runtime::checkpoint::load::GgufTensorNames,
>(
    source: &S,
    metadata: &std::collections::HashMap<String, GgufMetadataValue>,
    quantization: Option<WeightQuantization>,
) -> Result<(), Error> {
    let Some(quantization) = quantization else {
        return Ok(());
    };
    quantization.validate()?;

    let has_packed_companions = source.has_affine_gguf_tensor();
    if has_packed_companions {
        return Err(Error::Quantization(
            "load-time quantization accepts only unquantized F32/F16/BF16 GGUF weights; packed GGUF tensors cannot be implicitly transcoded"
                .into(),
        ));
    }

    let file_type = metadata
        .get("general.file_type")
        .ok_or_else(|| {
            Error::Quantization(
                "GGUF general.file_type metadata is required to verify that load-time quantization is not transcoding packed weights"
                    .into(),
            )
        })?
        .as_i64()
        .ok_or_else(|| {
            Error::Quantization("GGUF general.file_type metadata must be an integer".into())
        })?;
    // llama.cpp's unquantized file types: ALL_F32, MOSTLY_F16, and MOSTLY_BF16.
    if !matches!(file_type, 0 | 1 | 32) {
        return Err(Error::Quantization(format!(
            "load-time quantization accepts only unquantized F32/F16/BF16 GGUF weights; general.file_type={file_type} is already quantized"
        )));
    }
    Ok(())
}

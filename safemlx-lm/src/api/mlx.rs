//! MLX model loading, architecture dispatch, and generation extensions.
//!
//! Use [`crate::api::LoadedModel`] when you want to load a model directory
//! together with its tokenizer and chat template. Use
//! [`crate::load_model`] and [`crate::api::load_tokenizer`] when you
//! want to manage those pieces separately.
//! Ordinary generation is available for every `TextGenerationBackend`;
//! prepared-chat speculative generation is available through the
//! [`PreparedChatSpeculativeBackend`] capability on the same `LoadedModel<B>`.

use safemlx::{error::Exception, Array};

use super::{
    portable::LoadedModel,
    request::{
        PreparedChatDraft, PreparedChatMtpBatchExecutionRequest, PreparedChatMtpBatchOutput,
        PreparedChatMtpExecutionLane, PreparedChatMtpExecutionRequest,
        PreparedChatMtpGenerationOutput, PreparedChatSpeculativeBackend,
        PreparedChatSpeculativeConstraint,
    },
};
use crate::{
    architectures::{
        deepseek_v3::model as deepseek_v3, gemma4::model as gemma4, inkling::model as inkling,
        nemotron_h::model as nemotron_h, qwen::hybrid::qwen3_5,
    },
    backend::mlx::{
        error::Error,
        runtime::generation::sampler::ConstrainedSampler,
        speculative::{
            scheduler::MlxMtpScheduler, MlxDrafter, MlxDrafterKind, MlxMtpCache,
            MtpExecutionStreams,
        },
        validate_gemma4_drafter, Model, ModelCache,
    },
    core::{
        generation::{GenerationCancellationToken, MtpConfig, MtpSchedulerOptions, SemanticEvent},
        MtpCapability, SpeculativeSemanticState,
    },
};

impl PreparedChatSpeculativeBackend for crate::backend::mlx::MlxBackend<'static> {
    type Drafter = MlxDrafter;

    fn mtp_capability(model: &LoadedModel<Self>) -> MtpCapability {
        model.mlx_mtp_capability()
    }

    fn execute_prepared_chat_mtp<'a, F>(
        model: &mut LoadedModel<Self>,
        request: PreparedChatMtpExecutionRequest<'a, Self, Self::Drafter, F>,
    ) -> Result<PreparedChatMtpGenerationOutput, Error>
    where
        F: FnMut(SemanticEvent),
    {
        model.execute_prepared_chat_mtp_mlx(request)
    }

    fn execute_prepared_chat_mtp_batch<'a>(
        model: &mut LoadedModel<Self>,
        request: PreparedChatMtpBatchExecutionRequest<'a, Self, Self::Drafter>,
    ) -> Result<PreparedChatMtpBatchOutput, Error> {
        model.execute_prepared_chat_mtp_batch_mlx(request)
    }
}

struct PreparedChatMtpLaneRuntime<'a> {
    input: crate::backend::mlx::MlxModelInput,
    cache: &'a mut ModelCache,
    config: MtpConfig,
    prng_key: Option<Array>,
    sampler:
        ConstrainedSampler<crate::backend::mlx::runtime::generation::sampler::GenerationSampler>,
    semantic: Box<dyn SpeculativeSemanticState>,
    cancellation: GenerationCancellationToken,
    on_event: Box<dyn FnMut(SemanticEvent) + 'a>,
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

fn gemma4_mtp_cache(cache: &mut ModelCache) -> Option<&mut gemma4::Cache> {
    match cache {
        ModelCache::Gemma4(cache) => Some(cache),
        _ => None,
    }
}

fn model_mtp_cache(cache: &mut ModelCache) -> Option<&mut ModelCache> {
    Some(cache)
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
            ConstrainedSampler<
                crate::backend::mlx::runtime::generation::sampler::GenerationSampler,
            >,
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
                crate::backend::mlx::runtime::generation::sampler::GenerationSampler::from_resolved(
                    resolved,
                ),
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
        mut sampler: ConstrainedSampler<
            crate::backend::mlx::runtime::generation::sampler::GenerationSampler,
        >,
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
        mut sampler: ConstrainedSampler<
            crate::backend::mlx::runtime::generation::sampler::GenerationSampler,
        >,
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

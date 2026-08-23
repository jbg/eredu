//! Whole-session MLX speculative generation capability.

use eredu_core::{
    generation::{GenerationCancellationToken, MtpConfig, MtpSchedulerOptions, SemanticEvent},
    ModelRuntime, MtpCapability, SpeculativeDraft, SpeculativeGenerationBackend,
    SpeculativeGenerationBatchOutput, SpeculativeGenerationBatchRequest, SpeculativeGenerationLane,
    SpeculativeGenerationOutput, SpeculativeGenerationRequest, SpeculativeSemanticState,
    SpeculativeTokenFilterController,
};
use eredu_runtime::{ConstrainedSampler, GenerationSampler};
use safemlx::{error::Exception, Array};

use super::{
    speculative::{
        scheduler::MlxMtpScheduler, MlxDrafter, MlxDrafterKind, MlxMtpCache, MtpExecutionStreams,
    },
    MlxBackend, MlxModelInput, Model, ModelCache,
};
use crate::backend::mlx::error::Error;

impl<'world> SpeculativeGenerationBackend for MlxBackend<'world> {
    type Drafter = MlxDrafter;

    fn mtp_capability(runtime: &ModelRuntime<Self>) -> MtpCapability {
        runtime.session().mtp_capability()
    }

    fn execute_speculative<C, F>(
        runtime: &mut ModelRuntime<Self>,
        request: SpeculativeGenerationRequest<'_, Self, Self::Drafter, C, F>,
    ) -> Result<SpeculativeGenerationOutput, Error>
    where
        C: SpeculativeTokenFilterController,
        F: FnMut(SemanticEvent),
    {
        let tokenizer_fingerprint = request.tokenizer_fingerprint;
        MlxSpeculativeSession::new(runtime, tokenizer_fingerprint).execute(request)
    }

    fn execute_speculative_batch<C>(
        runtime: &mut ModelRuntime<Self>,
        request: SpeculativeGenerationBatchRequest<'_, Self, Self::Drafter, C>,
    ) -> Result<SpeculativeGenerationBatchOutput, Error>
    where
        C: SpeculativeTokenFilterController,
    {
        let tokenizer_fingerprint = request.tokenizer_fingerprint;
        MlxSpeculativeSession::new(runtime, tokenizer_fingerprint).execute_batch(request)
    }
}

struct MlxSpeculativeSession<'runtime, 'world> {
    runtime: &'runtime mut ModelRuntime<MlxBackend<'world>>,
    tokenizer_fingerprint: [u8; 32],
}

struct MlxSpeculativeLaneRuntime<'a, C> {
    input: MlxModelInput,
    cache: &'a mut ModelCache,
    config: MtpConfig,
    prng_key: Option<Array>,
    sampler: MlxPreparedSampler<C>,
    semantic: Box<dyn SpeculativeSemanticState>,
    cancellation: GenerationCancellationToken,
    on_event: Box<dyn FnMut(SemanticEvent) + 'a>,
}

type MlxPreparedSampler<C> = ConstrainedSampler<GenerationSampler, C>;

pub fn validate_external_drafter(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    target_tokenizer_fingerprint: [u8; 32],
    drafter: &MlxDrafter,
) -> Result<(), Error> {
    let model = runtime.session().complete_model();
    match (model, drafter.kind()) {
        (Model::Gemma4(target), MlxDrafterKind::Gemma4Assistant) => {
            let assistant = drafter.gemma4();
            let target = &target.args().text;
            let _compatibility = assistant
                .config
                .prove_compatibility(target)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        }
        (Model::MuseGlimmer(target), MlxDrafterKind::MuseGlimmerDFlash) => {
            let assistant = drafter.muse_glimmer();
            let _compatibility = assistant
                .config
                .prove_compatibility(target.args())
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        }
        (model, kind) => {
            return Err(Error::UnsupportedArchitecture(format!(
                "drafter {kind:?} is incompatible with target {} ({:?})",
                model.effective_model_type(),
                model.mtp_capability()
            )))
        }
    }
    if drafter.tokenizer_fingerprint() != target_tokenizer_fingerprint {
        return Err(Error::UnsupportedArchitecture(
            "assistant token-id vocabulary mapping does not match the target".into(),
        ));
    }
    Ok(())
}

fn run_speculative_batch<'a, B, C>(
    backend: &'a mut B,
    lanes: Vec<MlxSpeculativeLaneRuntime<'a, C>>,
    cache_for_lane: fn(&mut ModelCache) -> Option<&mut B::Cache>,
    cache_kind: &str,
    streams: MtpExecutionStreams<'a>,
    options: MtpSchedulerOptions,
) -> Result<SpeculativeGenerationBatchOutput, Exception>
where
    B: crate::composition::mlx::speculative::scheduler::MlxSpeculativeRuntime<'a>,
    C: SpeculativeTokenFilterController + 'a,
{
    let mut scheduler = MlxMtpScheduler::new(backend, streams, options)?;
    for (lane_index, lane) in lanes.into_iter().enumerate() {
        let MlxSpeculativeLaneRuntime {
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
            Ok(SpeculativeGenerationOutput {
                token_ids: request.token_ids,
                finish_reason,
                stats: request.stats,
            })
        })
        .collect::<Result<Vec<_>, Exception>>()?;
    Ok(SpeculativeGenerationBatchOutput {
        requests,
        scheduler: output.scheduler,
    })
}

fn neutral_gemma_mtp_cache(
    cache: &mut ModelCache,
) -> Option<&mut crate::backend::mlx::runtime::cache::state::MlxHybridState> {
    match cache {
        ModelCache::Hybrid(cache) => Some(cache),
        _ => None,
    }
}

fn neutral_muse_mtp_cache(
    cache: &mut ModelCache,
) -> Option<&mut crate::backend::mlx::runtime::cache::state::MlxKeyValueState> {
    match cache {
        ModelCache::MuseGlimmer(cache) => Some(cache),
        _ => None,
    }
}

fn neutral_inkling_mtp_cache(
    cache: &mut ModelCache,
) -> Option<&mut crate::composition::inkling::InklingState> {
    match cache {
        ModelCache::Inkling(cache) => Some(cache),
        _ => None,
    }
}

fn qwen_next_mtp_cache(
    cache: &mut ModelCache,
) -> Option<&mut crate::backend::mlx::runtime::cache::state::MlxHybridState> {
    match cache {
        ModelCache::Qwen3Next(cache) => Some(cache),
        _ => None,
    }
}

fn qwen35_mtp_cache(
    cache: &mut ModelCache,
) -> Option<&mut crate::backend::mlx::runtime::cache::state::MlxHybridState> {
    match cache {
        ModelCache::Qwen35(cache) => Some(cache),
        _ => None,
    }
}

fn neutral_deepseek_mtp_cache(
    cache: &mut ModelCache,
) -> Option<&mut crate::composition::deepseek::DeepSeekState> {
    match cache {
        ModelCache::DeepSeek(cache) => Some(cache),
        _ => None,
    }
}

fn nemotron_mtp_cache(
    cache: &mut ModelCache,
) -> Option<&mut crate::backend::mlx::runtime::cache::state::MlxHybridState> {
    match cache {
        ModelCache::Hybrid(cache) => Some(cache),
        _ => None,
    }
}

impl<'runtime, 'world> MlxSpeculativeSession<'runtime, 'world> {
    fn new(
        runtime: &'runtime mut ModelRuntime<MlxBackend<'world>>,
        tokenizer_fingerprint: [u8; 32],
    ) -> Self {
        Self {
            runtime,
            tokenizer_fingerprint,
        }
    }

    fn model_and_cache(&mut self) -> (&mut Model, &mut ModelCache) {
        self.runtime.session_mut().complete_parts_mut()
    }

    fn prepare_mlx_speculative_sampling<C>(
        generation: eredu_core::TextGenerationConfig,
        constraint: C,
    ) -> Result<(Option<Array>, MlxPreparedSampler<C>), Error>
    where
        C: SpeculativeTokenFilterController,
    {
        let resolved = generation.sampling();
        let prng_key = (resolved.temperature != 0.0)
            .then(|| safemlx::random::key(generation.seed()))
            .transpose()?;
        Ok((
            prng_key,
            ConstrainedSampler::new(GenerationSampler::from_resolved(resolved), constraint),
        ))
    }

    /// Validates the observable target/assistant contract used by external MTP.
    ///
    /// Repository names and revisions are deliberately not compatibility keys.
    /// The validation covers the target architecture, shared tensor geometry,
    /// and the token-id vocabulary mapping when the drafter carries tokenizer
    /// metadata.
    fn validate_drafter_compatibility(&self, drafter: &MlxDrafter) -> Result<(), Error> {
        validate_external_drafter(self.runtime, self.tokenizer_fingerprint, drafter)
    }

    fn prepare_speculative_batch_lanes<'a, C>(
        &self,
        lanes: Vec<SpeculativeGenerationLane<'a, MlxBackend<'world>, C>>,
        cache: &'a mut MlxMtpCache,
    ) -> Result<Vec<MlxSpeculativeLaneRuntime<'a, C>>, Error>
    where
        C: SpeculativeTokenFilterController,
    {
        if cache.len() != lanes.len() {
            return Err(Error::Speculative(format!(
                "MTP cache has {} lanes but the request has {} lanes",
                cache.len(),
                lanes.len()
            )));
        }
        let mut prepared_lanes = Vec::with_capacity(lanes.len());
        for (lane, cache) in lanes.into_iter().zip(cache.lanes.iter_mut()) {
            let SpeculativeGenerationLane {
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
            prepared_lanes.push(MlxSpeculativeLaneRuntime {
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

    fn execute_batch<C>(
        &mut self,
        request: SpeculativeGenerationBatchRequest<'_, MlxBackend<'world>, MlxDrafter, C>,
    ) -> Result<SpeculativeGenerationBatchOutput, Error>
    where
        C: SpeculativeTokenFilterController,
    {
        let SpeculativeGenerationBatchRequest {
            drafting,
            lanes,
            scheduler,
            tokenizer_fingerprint: _,
        } = request;
        match drafting {
            SpeculativeDraft::External(drafter) => {
                self.generate_speculative_batch_with_external_draft(drafter, lanes, scheduler)
            }
            SpeculativeDraft::Embedded => {
                self.generate_speculative_batch_with_embedded_draft(lanes, scheduler)
            }
        }
    }

    fn generate_speculative_batch_with_external_draft<C>(
        &mut self,
        drafter: &mut MlxDrafter,
        lanes: Vec<SpeculativeGenerationLane<'_, MlxBackend<'world>, C>>,
        scheduler: MtpSchedulerOptions,
    ) -> Result<SpeculativeGenerationBatchOutput, Error>
    where
        C: SpeculativeTokenFilterController,
    {
        self.validate_drafter_compatibility(drafter)?;
        let target_stream = self.runtime.backend().stream().clone();
        let draft_stream = drafter.stream().clone();
        let streams = MtpExecutionStreams::new(&target_stream, &draft_stream)?;
        let mut cache = self.new_mtp_cache(lanes.len());
        let prepared_lanes = self.prepare_speculative_batch_lanes(lanes, &mut cache)?;

        match (self.model_and_cache().0, drafter.kind()) {
            (Model::Gemma4(target), MlxDrafterKind::Gemma4Assistant) => {
                let mut backend =
                    crate::composition::mlx::speculative::external::Gemma4ExternalExecutor::new(
                        target,
                        drafter.gemma4_mut(),
                    );
                run_speculative_batch(
                    &mut backend,
                    prepared_lanes,
                    neutral_gemma_mtp_cache,
                    "Gemma 4 external assistant",
                    streams,
                    scheduler,
                )
                .map_err(|error| Error::Speculative(error.to_string()))
            }
            (Model::MuseGlimmer(target), MlxDrafterKind::MuseGlimmerDFlash) => {
                let mut backend = crate::composition::mlx::speculative::external::MuseGlimmerExternalExecutor::new(
                    target,
                    drafter.muse_glimmer_mut(),
                );
                run_speculative_batch(
                    &mut backend,
                    prepared_lanes,
                    neutral_muse_mtp_cache,
                    "Muse-Glimmer DFlash",
                    streams,
                    scheduler,
                )
                .map_err(|error| Error::Speculative(error.to_string()))
            }
            (model, kind) => Err(Error::Speculative(format!(
                "MTP runtime adapter is unavailable for model type {} ({:?})",
                model.effective_model_type(),
                kind
            ))),
        }
    }

    fn generate_speculative_batch_with_embedded_draft<C>(
        &mut self,
        lanes: Vec<SpeculativeGenerationLane<'_, MlxBackend<'world>, C>>,
        scheduler: MtpSchedulerOptions,
    ) -> Result<SpeculativeGenerationBatchOutput, Error>
    where
        C: SpeculativeTokenFilterController,
    {
        let stream = self.runtime.backend().stream().clone();
        let mut cache = self.new_mtp_cache(lanes.len());
        let prepared_lanes = self.prepare_speculative_batch_lanes(lanes, &mut cache)?;
        let streams = MtpExecutionStreams::single(&stream);
        match self.model_and_cache().0 {
            Model::DeepSeek(target) => {
                let mut backend =
                    crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        target.as_mut(),
                    );
                run_speculative_batch(
                    &mut backend,
                    prepared_lanes,
                    neutral_deepseek_mtp_cache,
                    "DeepSeek embedded",
                    streams,
                    scheduler,
                )
                .map_err(|error| Error::Speculative(error.to_string()))
            }
            Model::NemotronH(target) => {
                let mut backend =
                    crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        target,
                    );
                run_speculative_batch(
                    &mut backend,
                    prepared_lanes,
                    nemotron_mtp_cache,
                    "Nemotron-H embedded",
                    streams,
                    scheduler,
                )
                .map_err(|error| Error::Speculative(error.to_string()))
            }
            Model::Inkling(target) => {
                let mut backend =
                    crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        target,
                    );
                run_speculative_batch(
                    &mut backend,
                    prepared_lanes,
                    neutral_inkling_mtp_cache,
                    "Inkling embedded",
                    streams,
                    scheduler,
                )
                .map_err(|error| Error::Speculative(error.to_string()))
            }
            Model::Qwen3Next(target) => {
                let mut backend =
                    crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        target,
                    );
                run_speculative_batch(
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
                    crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        target,
                    );
                run_speculative_batch(
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
            model.effective_model_type(),
            model.mtp_capability()
        ))),
        }
    }

    fn execute<C, F>(
        &mut self,
        request: SpeculativeGenerationRequest<'_, MlxBackend<'world>, MlxDrafter, C, F>,
    ) -> Result<SpeculativeGenerationOutput, Error>
    where
        C: SpeculativeTokenFilterController,
        F: FnMut(SemanticEvent),
    {
        let SpeculativeGenerationRequest {
            prompt,
            drafting,
            generation,
            config,
            constraint,
            semantic,
            scheduler,
            cancellation,
            tokenizer_fingerprint: _,
            on_event,
        } = request;
        let (prng_key, sampler) = Self::prepare_mlx_speculative_sampling(generation, constraint)?;
        match drafting {
            SpeculativeDraft::External(drafter) => self.generate_speculative_with_external_draft(
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
            SpeculativeDraft::Embedded => self.generate_speculative_with_embedded_draft(
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
    fn generate_speculative_with_external_draft<C, F>(
        &mut self,
        input: MlxModelInput,
        drafter: &mut MlxDrafter,
        config: MtpConfig,
        prng_key: Option<Array>,
        mut sampler: MlxPreparedSampler<C>,
        semantic: Box<dyn SpeculativeSemanticState>,
        scheduler: MtpSchedulerOptions,
        cancellation: GenerationCancellationToken,
        on_event: F,
    ) -> Result<SpeculativeGenerationOutput, Error>
    where
        C: SpeculativeTokenFilterController,
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
            Ok(SpeculativeGenerationOutput {
                token_ids,
                finish_reason,
                stats,
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_speculative_with_embedded_draft<C, F>(
        &mut self,
        input: MlxModelInput,
        config: MtpConfig,
        prng_key: Option<Array>,
        mut sampler: MlxPreparedSampler<C>,
        semantic: Box<dyn SpeculativeSemanticState>,
        scheduler: MtpSchedulerOptions,
        cancellation: GenerationCancellationToken,
        on_event: F,
    ) -> Result<SpeculativeGenerationOutput, Error>
    where
        C: SpeculativeTokenFilterController,
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
            Ok(SpeculativeGenerationOutput {
                token_ids,
                finish_reason,
                stats,
            })
        })
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

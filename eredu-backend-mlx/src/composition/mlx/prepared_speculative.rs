//! Whole-session MLX speculative generation capability.

use eredu_core::{
    generation::{GenerationCancellationToken, SemanticEvent, SpeculativeConfig},
    GenerationSequence, ModelRuntime, PreparedSpeculativeLane, SpeculativeCallbackPublisher,
    SpeculativeCapability, SpeculativeDraft, SpeculativeGenerationBackend,
    SpeculativeGenerationBatchOutput, SpeculativeGenerationBatchRequest, SpeculativeGenerationLane,
    SpeculativeGenerationVisitor, SpeculativeOutputRuntime, SpeculativeSampling,
    SpeculativeSemanticConstraint, SpeculativeSemanticState, SpeculativeTokenFilterController,
};
use eredu_runtime::{ConstrainedSampler, GenerationSampler, SpeculativeSampler};
use safemlx::{error::Exception, transforms::async_eval_with_event, Array, Stream};

use super::{
    session::MlxSpeculativeSessionParts,
    speculative::{
        scheduler::{component_timing_enabled, MlxSpeculativeRuntime},
        MlxDrafter, MlxDrafterKind, MlxSpeculativeSampling, SpeculativeExecutionStreams,
    },
    Executable, MlxBackend, MlxModelInput,
};
use crate::backend::error::Error;
use crate::backend::nn::tensor::TokenValidationScope;
use crate::backend::runtime::generation::MlxSamplingBackend;
use crate::backend::runtime::media::input::ModelInput;
use crate::MlxTensor;

struct NeutralEmbeddedPredictionTarget<'a> {
    target: &'a mut dyn super::replicated_text::ErasedReplicatedTextExecutable,
    depth: usize,
}

fn with_neutral_target_validation<T>(
    operation: impl FnOnce() -> Result<T, Error>,
) -> Result<T, Exception> {
    let scope = TokenValidationScope::begin()?;
    let output = operation().map_err(|error| Exception::custom(error.to_string()))?;
    let validations = scope.finish();
    if !validations.is_empty() {
        async_eval_with_event(validations.arrays())?.synchronize()?;
        validations.validate_completed()?;
    }
    Ok(output)
}

fn with_neutral_target_transaction<T>(
    cache: &mut super::replicated_text::MlxPredictionTargetCache,
    stream: &Stream,
    operation: impl FnOnce(&mut super::replicated_text::MlxPredictionTargetCache) -> Result<T, Error>,
) -> Result<T, Exception> {
    let checkpoint = cache.clone();
    match with_neutral_target_validation(|| operation(cache)) {
        Ok(output) => Ok(output),
        Err(error) => {
            if let Err(restore) = cache.restore_checkpoint(&checkpoint, stream) {
                return Err(Exception::custom(format!(
                    "prediction target operation failed: {error}; rollback failed: {restore}"
                )));
            }
            Err(error)
        }
    }
}

impl super::speculative::embedded::EmbeddedMtpTarget for NeutralEmbeddedPredictionTarget<'_> {
    type Cache = super::replicated_text::MlxPredictionTargetCache;
    type DraftCache = super::replicated_text::MlxPredictionDraftCache;

    fn prefill_target(
        &mut self,
        input: ModelInput<'_>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<super::speculative::embedded::EmbeddedMtpOutput, Exception> {
        with_neutral_target_transaction(cache, stream, |cache| {
            self.target.prefill_prediction_target(input, cache)
        })
    }

    fn verify_target(
        &mut self,
        tokens: &MlxTensor,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<super::speculative::embedded::EmbeddedMtpOutput, Exception> {
        with_neutral_target_transaction(cache, stream, |cache| {
            self.target.verify_prediction_target(tokens, cache)
        })
    }

    fn prefill_draft_cache(
        &mut self,
        output: &super::speculative::embedded::EmbeddedMtpOutput,
        tokens: &MlxTensor,
        cache: &mut Self::Cache,
        _stream: &Stream,
    ) -> Result<(), Exception> {
        let checkpoint = cache.draft_cache();
        match with_neutral_target_validation(|| {
            self.target
                .prefill_prediction_extension(output, tokens, cache)
        }) {
            Ok(()) => Ok(()),
            Err(error) => {
                cache.commit_draft_cache(&checkpoint);
                Err(error)
            }
        }
    }

    fn draft_cache(&self, cache: &Self::Cache) -> Self::DraftCache {
        cache.draft_cache()
    }

    fn commit_draft_cache(&self, cache: &mut Self::Cache, draft: &Self::DraftCache) {
        cache.commit_draft_cache(draft);
    }

    fn restore_target_checkpoint(
        cache: &mut Self::Cache,
        checkpoint: &Self::Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        cache.restore_checkpoint(checkpoint, stream)
    }

    fn draft_logits(
        &mut self,
        hidden: &MlxTensor,
        last_token: u32,
        draft_index: usize,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(MlxTensor, MlxTensor), Exception> {
        cache.with_extension(|cache| {
            with_neutral_target_transaction(cache, stream, |cache| {
                self.target
                    .prediction_extension_logits(hidden, last_token, draft_index, cache)
            })
        })
    }

    fn advance_draft_cache(
        &mut self,
        hidden: &MlxTensor,
        tokens: &MlxTensor,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        cache.with_extension(|cache| {
            with_neutral_target_transaction(cache, stream, |cache| {
                self.target
                    .advance_prediction_extension(hidden, tokens, cache)
            })
        })
    }

    fn max_draft_tokens(&self) -> usize {
        self.depth
    }
}

impl<'world> SpeculativeGenerationBackend for MlxBackend<'world> {
    type Drafter = MlxDrafter;

    fn speculative_capability(runtime: &ModelRuntime<Self>) -> SpeculativeCapability {
        runtime.session().speculative_capability()
    }

    fn with_speculative_execution<C, V>(
        runtime: &mut ModelRuntime<Self>,
        request: SpeculativeGenerationBatchRequest<'_, Self, Self::Drafter, C>,
        visitor: V,
    ) -> Result<SpeculativeGenerationBatchOutput, Error>
    where
        C: SpeculativeTokenFilterController,
        V: SpeculativeGenerationVisitor,
    {
        let tokenizer_fingerprint = request.tokenizer_fingerprint();
        MlxSpeculativeSession::new(runtime, tokenizer_fingerprint).with_execution(request, visitor)
    }
}

struct MlxSpeculativeSession<'runtime, 'world> {
    runtime: &'runtime mut ModelRuntime<MlxBackend<'world>>,
    tokenizer_fingerprint: [u8; 32],
}

struct MlxSpeculativeLaneRuntime<'a, C> {
    input: MlxModelInput,
    config: SpeculativeConfig,
    prng_key: Option<Array>,
    sampler: MlxPreparedSampler<C>,
    semantic: Box<dyn SpeculativeSemanticState>,
    cancellation: GenerationCancellationToken,
    on_event: Box<dyn FnMut(SemanticEvent) + 'a>,
}

type MlxPreparedSampler<C> = ConstrainedSampler<GenerationSampler, C>;

fn run_speculative_batch<'a, B, C, S>(
    backend: &'a mut B,
    lanes: Vec<MlxSpeculativeLaneRuntime<'a, C>>,
    caches: &'a mut [B::Cache],
    wrap_sampler: impl Fn(MlxPreparedSampler<C>) -> Result<S, Exception>,
    streams: SpeculativeExecutionStreams<'a>,
    visitor: impl SpeculativeGenerationVisitor,
) -> Result<SpeculativeGenerationBatchOutput, Exception>
where
    B: MlxSpeculativeRuntime<'a>,
    C: SpeculativeTokenFilterController + 'a,
    S: SpeculativeSampler<MlxSamplingBackend> + Clone + 'a,
{
    if caches.len() != lanes.len() {
        return Err(Exception::custom(format!(
            "speculative cache has {} lanes but the request has {} lanes",
            caches.len(),
            lanes.len()
        )));
    }
    let topology = streams.topology();
    let component_timings_collected = component_timing_enabled() && backend.supports_telemetry();
    let mut prepared = Vec::with_capacity(lanes.len());
    for (lane, cache) in lanes.into_iter().zip(caches.iter_mut()) {
        let MlxSpeculativeLaneRuntime {
            input,
            config,
            prng_key,
            sampler,
            semantic,
            cancellation,
            on_event,
        } = lane;
        let sampling = MlxSpeculativeSampling::new(wrap_sampler(sampler)?);
        let randomness = <MlxSpeculativeSampling<S> as SpeculativeSampling>::initialize_randomness(
            prng_key,
            config.temperature,
            streams,
        )?;
        let sequence =
            GenerationSequence::new(config.max_tokens, config.eos_token_ids.iter().copied());
        prepared.push(PreparedSpeculativeLane::new(
            cache,
            input,
            config,
            SpeculativeOutputRuntime::new(
                sampling,
                sequence,
                SpeculativeSemanticConstraint::semantic(semantic),
                SpeculativeCallbackPublisher::semantic(on_event),
                cancellation,
            ),
            randomness,
        ));
    }
    visitor
        .run(
            backend,
            prepared,
            topology,
            streams.is_split(),
            component_timings_collected,
            streams,
        )
        .map_err(|error| Exception::custom(error.to_string()))
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

    /// Validates the observable target/assistant contract used by external drafting.
    ///
    /// Repository names and revisions are deliberately not compatibility keys.
    /// The validation covers the target architecture, shared tensor geometry,
    /// and the portable token-id vocabulary compatibility contract.
    fn validate_drafter_compatibility(&self, drafter: &MlxDrafter) -> Result<(), Error> {
        self.runtime.session().validate_external_drafter(drafter)?;
        drafter
            .tokenizer_compatibility()
            .validate_target(self.tokenizer_fingerprint)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(())
    }

    fn prepare_speculative_batch_lanes<'a, C>(
        &self,
        lanes: Vec<SpeculativeGenerationLane<'a, MlxBackend<'world>, C>>,
    ) -> Result<Vec<MlxSpeculativeLaneRuntime<'a, C>>, Error>
    where
        C: SpeculativeTokenFilterController,
    {
        let mut prepared_lanes = Vec::with_capacity(lanes.len());
        for mut lane in lanes {
            let prompt = lane.take_prompt();
            let generation = lane.take_generation();
            let config = lane.take_config();
            let constraint = lane.take_constraint();
            let semantic = lane.take_semantic();
            let cancellation = lane.take_cancellation();
            let on_event = lane.take_on_event();
            let (prng_key, sampler) =
                Self::prepare_mlx_speculative_sampling(generation, constraint)?;
            prepared_lanes.push(MlxSpeculativeLaneRuntime {
                input: prompt,
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

    fn with_execution<C, V>(
        &mut self,
        mut request: SpeculativeGenerationBatchRequest<'_, MlxBackend<'world>, MlxDrafter, C>,
        visitor: V,
    ) -> Result<SpeculativeGenerationBatchOutput, Error>
    where
        C: SpeculativeTokenFilterController,
        V: SpeculativeGenerationVisitor,
    {
        let drafting = request.take_drafting();
        let lanes = request.take_lanes();
        match drafting {
            SpeculativeDraft::External(drafter) => {
                self.generate_speculative_batch_with_external_draft(drafter, lanes, visitor)
            }
            SpeculativeDraft::Embedded => {
                self.generate_speculative_batch_with_embedded_draft(lanes, visitor)
            }
            _ => Err(Error::ArchitectureModel(
                "unsupported speculative draft source".to_string(),
            )),
        }
    }

    fn generate_speculative_batch_with_external_draft<C, V>(
        &mut self,
        drafter: &mut MlxDrafter,
        lanes: Vec<SpeculativeGenerationLane<'_, MlxBackend<'world>, C>>,
        visitor: V,
    ) -> Result<SpeculativeGenerationBatchOutput, Error>
    where
        C: SpeculativeTokenFilterController,
        V: SpeculativeGenerationVisitor,
    {
        self.validate_drafter_compatibility(drafter)?;
        let target_stream = self.runtime.backend().stream().clone();
        let draft_stream = drafter.stream().clone();
        let streams = SpeculativeExecutionStreams::new(&target_stream, &draft_stream)?;
        let prepared_lanes = self.prepare_speculative_batch_lanes(lanes)?;
        let lane_count = prepared_lanes.len();
        let model = match self.runtime.session_mut().speculative_parts_mut()? {
            MlxSpeculativeSessionParts::Complete { model, .. } => model,
        };
        match (model, drafter.kind()) {
            (Executable::ReplicatedText(_, target), MlxDrafterKind::Gemma4Assistant) => {
                let capture = target
                    .external_prediction_capture_request(drafter)?
                    .ok_or_else(|| {
                        Error::Speculative(
                            "neutral target does not admit the Gemma external assistant".into(),
                        )
                    })?;
                let mut caches = (0..lane_count)
                    .map(|_| target.prepare_external_prediction_target_cache())
                    .collect::<Result<Vec<_>, _>>()?;
                let mut backend =
                    crate::composition::mlx::speculative::external::Gemma4ExternalExecutor::new(
                        target.as_mut(),
                        drafter.gemma4_mut(),
                        capture,
                    );
                run_speculative_batch(
                    &mut backend,
                    prepared_lanes,
                    &mut caches,
                    Ok,
                    streams,
                    visitor,
                )
                .map_err(|error| Error::Speculative(error.to_string()))
            }
            (Executable::ReplicatedText(_, target), MlxDrafterKind::MuseGlimmerDFlash) => {
                let capture = target
                    .external_prediction_capture_request(drafter)?
                    .ok_or_else(|| {
                        Error::Speculative(
                            "neutral target does not admit the Muse-Glimmer DFlash assistant"
                                .into(),
                        )
                    })?;
                let mut caches = (0..lane_count)
                    .map(|_| target.prepare_external_prediction_target_cache())
                    .collect::<Result<Vec<_>, _>>()?;
                let mut backend = crate::composition::mlx::speculative::external::MuseGlimmerExternalExecutor::new(
                    target.as_mut(),
                    drafter.muse_glimmer_mut(),
                    capture,
                );
                run_speculative_batch(
                    &mut backend,
                    prepared_lanes,
                    &mut caches,
                    Ok,
                    streams,
                    visitor,
                )
                .map_err(|error| Error::Speculative(error.to_string()))
            }
            (
                model @ (Executable::DeepSeek(_, _, _)
                | Executable::Gemma4(_, _, _)
                | Executable::GptOss(_, _, _)
                | Executable::Inkling(_, _, _)
                | Executable::KimiLinear(_, _, _)
                | Executable::Lfm2(_, _, _)
                | Executable::MuseGlimmer(_, _, _)
                | Executable::NemotronH(_, _, _)
                | Executable::Qwen(_, _, _)
                | Executable::Qwen3Next(_, _, _)
                | Executable::Qwen35(_, _, _)),
                kind,
            ) => Err(Error::Speculative(format!(
                "speculative runtime adapter is unavailable for model type {} ({:?})",
                model.effective_model_type(),
                kind
            ))),
        }
    }

    fn generate_speculative_batch_with_embedded_draft<C, V>(
        &mut self,
        lanes: Vec<SpeculativeGenerationLane<'_, MlxBackend<'world>, C>>,
        visitor: V,
    ) -> Result<SpeculativeGenerationBatchOutput, Error>
    where
        C: SpeculativeTokenFilterController,
        V: SpeculativeGenerationVisitor,
    {
        let stream = self.runtime.backend().stream().clone();
        let prepared_lanes = self.prepare_speculative_batch_lanes(lanes)?;
        let lane_count = prepared_lanes.len();
        let streams = SpeculativeExecutionStreams::single(&stream);
        let MlxSpeculativeSessionParts::Complete { model, .. } =
            self.runtime.session_mut().speculative_parts_mut()?;
        let Executable::ReplicatedText(_, target) = model else {
            return Err(Error::Speculative(format!(
                "embedded prediction requires the selected neutral target, got {}",
                model.effective_model_type()
            )));
        };
        let depth = target.prediction_extension_depth().ok_or_else(|| {
            Error::Speculative(
                "neutral target has no installed prediction-extension contract".into(),
            )
        })?;
        let mut caches = (0..lane_count)
            .map(|_| target.prepare_prediction_target_cache())
            .collect::<Result<Vec<_>, _>>()?;
        let mut target = NeutralEmbeddedPredictionTarget {
            target: target.as_mut(),
            depth,
        };
        let mut backend =
            crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(&mut target);
        run_speculative_batch(
            &mut backend,
            prepared_lanes,
            &mut caches,
            Ok,
            streams,
            visitor,
        )
        .map_err(|error| Error::Speculative(error.to_string()))
    }
}

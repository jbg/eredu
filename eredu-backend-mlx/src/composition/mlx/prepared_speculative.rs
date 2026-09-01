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
use safemlx::{error::Exception, Array};

use super::{
    distributed::pipeline::PipelineEmbeddedMtpTarget,
    session::MlxSpeculativeSessionParts,
    speculative::{
        embedded::DistributedEmbeddedMtpSampler,
        scheduler::{component_timing_enabled, MlxSpeculativeRuntime},
        MlxDrafter, MlxDrafterKind, MlxSpeculativeSampling, SpeculativeExecutionStreams,
    },
    Executable, MlxBackend, MlxModelInput,
};
use crate::backend::error::Error;
use crate::backend::runtime::generation::MlxSamplingBackend;

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
        let tokenizer_fingerprint = request.tokenizer_fingerprint;
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
        prepared.push(PreparedSpeculativeLane {
            cache,
            input,
            runtime: SpeculativeOutputRuntime::new(
                sampling,
                GenerationSequence::new(config.max_tokens, config.eos_token_ids.iter().copied()),
                SpeculativeSemanticConstraint::semantic(semantic),
                SpeculativeCallbackPublisher::semantic(on_event),
                cancellation,
            ),
            config,
            randomness,
        });
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
        for lane in lanes {
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
        request: SpeculativeGenerationBatchRequest<'_, MlxBackend<'world>, MlxDrafter, C>,
        visitor: V,
    ) -> Result<SpeculativeGenerationBatchOutput, Error>
    where
        C: SpeculativeTokenFilterController,
        V: SpeculativeGenerationVisitor,
    {
        let SpeculativeGenerationBatchRequest {
            drafting,
            lanes,
            tokenizer_fingerprint: _,
        } = request;
        match drafting {
            SpeculativeDraft::External(drafter) => {
                self.generate_speculative_batch_with_external_draft(drafter, lanes, visitor)
            }
            SpeculativeDraft::Embedded => {
                self.generate_speculative_batch_with_embedded_draft(lanes, visitor)
            }
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
            MlxSpeculativeSessionParts::Pipeline { .. } => {
                return Err(Error::Speculative(
                    "external drafting is unavailable for pipeline sessions".into(),
                ))
            }
        };
        match (model, drafter.kind()) {
            (Executable::Gemma4(_, target, _), MlxDrafterKind::Gemma4Assistant) => {
                let mut caches = (0..lane_count)
                    .map(|_| target.new_cache())
                    .collect::<Vec<_>>();
                let mut backend =
                    crate::composition::mlx::speculative::external::Gemma4ExternalExecutor::new(
                        target,
                        drafter.gemma4_mut(),
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
            (Executable::MuseGlimmer(_, target, _), MlxDrafterKind::MuseGlimmerDFlash) => {
                let mut caches = (0..lane_count)
                    .map(|_| target.new_cache())
                    .collect::<Vec<_>>();
                let mut backend = crate::composition::mlx::speculative::external::MuseGlimmerExternalExecutor::new(
                    target,
                    drafter.muse_glimmer_mut(),
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
            (model @ Executable::Gemma4(_, _, _), kind @ MlxDrafterKind::MuseGlimmerDFlash)
            | (model @ Executable::MuseGlimmer(_, _, _), kind @ MlxDrafterKind::Gemma4Assistant)
            | (
                model @ (Executable::DeepSeek(_, _, _)
                | Executable::GptOss(_, _, _)
                | Executable::Inkling(_, _, _)
                | Executable::KimiLinear(_, _, _)
                | Executable::Lfm2(_, _, _)
                | Executable::PartitionedLlama(_, _, _)
                | Executable::ReplicatedText(_, _)
                | Executable::NemotronH(_, _, _)
                | Executable::Qwen(_, _, _)
                | Executable::Qwen3Next(_, _, _)
                | Executable::Qwen3Vl(_, _, _)
                | Executable::Qwen3VlMoe(_, _, _)
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
        match self.runtime.session_mut().speculative_parts_mut()? {
            MlxSpeculativeSessionParts::Complete { model, execution } => {
                if let Some(execution) = execution {
                    let topology = model
                        .parallel_info()
                        .map(|info| info.topology())
                        .ok_or_else(|| {
                            Error::Speculative(
                                "distributed embedded MTP requires selected topology".into(),
                            )
                        })?;
                    if topology.pipeline_parallel_size != 1 || topology.expert_parallel_size != 1 {
                        return Err(Error::Speculative(
                            "complete-model embedded MTP requires pure tensor parallelism".into(),
                        ));
                    }
                    super::distributed::topology::validate_session(topology, execution)?;
                    let tensor = execution
                        .partitioned_context(crate::backend::distributed::SHARD_GROUP_ID)?;
                    let tensor_group = tensor.group().ok_or_else(|| {
                        Error::Speculative(
                            "distributed embedded MTP requires an active tensor subgroup".into(),
                        )
                    })?;
                    let sampling_rank = topology
                        .global_rank_for(eredu_core::ParallelCoordinates {
                            tensor: 0,
                            pipeline: 0,
                            expert: 0,
                            data: topology.data_parallel_rank,
                        })
                        .map_err(|error| Error::Parallel(error.to_string()))?;
                    let world = execution.world();
                    let synchronized = |sampler| {
                        DistributedEmbeddedMtpSampler::new(sampler, sampling_rank, world)
                            .map_err(|error| Exception::custom(error.to_string()))
                    };
                    return match model {
                        Executable::NemotronH(_, target, _) => {
                            let mut caches = (0..lane_count)
                                .map(|_| target.new_cache())
                                .collect::<Vec<_>>();
                            let mut target =
                                crate::composition::nemotron_h::NemotronHTensorMtpTarget::new(
                                    target,
                                    tensor_group,
                                );
                            let mut backend = crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(&mut target);
                            run_speculative_batch(
                                &mut backend,
                                prepared_lanes,
                                &mut caches,
                                synchronized,
                                streams,
                                visitor,
                            )
                            .map_err(|error| Error::Speculative(error.to_string()))
                        }
                        Executable::Inkling(_, target, _) => {
                            let mut caches = (0..lane_count)
                                .map(|_| target.new_cache())
                                .collect::<Vec<_>>();
                            let mut target =
                                crate::composition::inkling::InklingTensorMtpTarget::new(
                                    target,
                                    tensor_group,
                                );
                            let mut backend = crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(&mut target);
                            run_speculative_batch(
                                &mut backend,
                                prepared_lanes,
                                &mut caches,
                                synchronized,
                                streams,
                                visitor,
                            )
                            .map_err(|error| Error::Speculative(error.to_string()))
                        }
                        Executable::Qwen3Next(_, target, _) => {
                            let mut caches = (0..lane_count)
                                .map(|_| target.new_cache())
                                .collect::<Vec<_>>();
                            let mut backend = crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(target);
                            run_speculative_batch(
                                &mut backend,
                                prepared_lanes,
                                &mut caches,
                                synchronized,
                                streams,
                                visitor,
                            )
                            .map_err(|error| Error::Speculative(error.to_string()))
                        }
                        Executable::Qwen35(_, target, _) => {
                            let mut caches = (0..lane_count)
                                .map(|_| target.new_cache())
                                .collect::<Vec<_>>();
                            let mut backend = crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(target);
                            run_speculative_batch(
                                &mut backend,
                                prepared_lanes,
                                &mut caches,
                                synchronized,
                                streams,
                                visitor,
                            )
                            .map_err(|error| Error::Speculative(error.to_string()))
                        }
                        model @ (Executable::DeepSeek(_, _, _)
                        | Executable::Gemma4(_, _, _)
                        | Executable::GptOss(_, _, _)
                        | Executable::KimiLinear(_, _, _)
                        | Executable::Lfm2(_, _, _)
                        | Executable::PartitionedLlama(_, _, _)
                        | Executable::ReplicatedText(_, _)
                        | Executable::MuseGlimmer(_, _, _)
                        | Executable::Qwen(_, _, _)
                        | Executable::Qwen3Vl(_, _, _)
                        | Executable::Qwen3VlMoe(_, _, _)) => Err(Error::Speculative(format!(
                            "distributed prepared-chat embedded MTP is unavailable for model type {} ({:?})",
                            model.effective_model_type(),
                            model.speculative_capability()
                        ))),
                    };
                }
                match model {
            Executable::DeepSeek(_, target, _) => {
                let mut caches = (0..lane_count)
                    .map(|_| target.new_state())
                    .collect::<Result<Vec<_>, _>>()?;
                let mut backend =
                    crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        target.as_mut(),
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
            Executable::NemotronH(_, target, _) => {
                let mut caches = (0..lane_count)
                    .map(|_| target.new_cache())
                    .collect::<Vec<_>>();
                let mut backend =
                    crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        target,
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
            Executable::Inkling(_, target, _) => {
                let mut caches = (0..lane_count)
                    .map(|_| target.new_cache())
                    .collect::<Vec<_>>();
                let mut backend =
                    crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        target,
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
            Executable::Qwen3Next(_, target, _) => {
                let mut caches = (0..lane_count)
                    .map(|_| target.new_cache())
                    .collect::<Vec<_>>();
                let mut backend =
                    crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        target,
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
            Executable::Qwen35(_, target, _) => {
                let mut caches = (0..lane_count)
                    .map(|_| target.new_cache())
                    .collect::<Vec<_>>();
                let mut backend =
                    crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        target,
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
            model @ (Executable::Gemma4(_, _, _)
            | Executable::GptOss(_, _, _)
            | Executable::KimiLinear(_, _, _)
            | Executable::Lfm2(_, _, _)
            | Executable::PartitionedLlama(_, _, _)
            | Executable::ReplicatedText(_, _)
            | Executable::MuseGlimmer(_, _, _)
            | Executable::Qwen(_, _, _)
            | Executable::Qwen3Vl(_, _, _)
            | Executable::Qwen3VlMoe(_, _, _)) => Err(Error::Speculative(format!(
                "scheduled prepared-chat embedded MTP batch is unavailable for model type {} ({:?})",
                model.effective_model_type(),
                model.speculative_capability()
            ))),
                }
            }
            MlxSpeculativeSessionParts::Pipeline { model, execution } => {
                let mut caches = (0..lane_count)
                    .map(|_| model.new_cache())
                    .collect::<Result<Vec<_>, _>>()?;
                let topology = model.stage_info().topology;
                super::distributed::topology::validate_session(topology, execution)?;
                let sampling_rank = topology
                    .global_rank_for(eredu_core::ParallelCoordinates {
                        tensor: 0,
                        pipeline: topology.pipeline_parallel_size - 1,
                        expert: 0,
                        data: topology.data_parallel_rank,
                    })
                    .map_err(|error| Error::Parallel(error.to_string()))?;
                let world = execution.world();
                let mut target = PipelineEmbeddedMtpTarget::new(model, execution);
                let mut backend =
                    crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        &mut target,
                    );
                run_speculative_batch(
                    &mut backend,
                    prepared_lanes,
                    &mut caches,
                    |sampler| {
                        DistributedEmbeddedMtpSampler::new(sampler, sampling_rank, world)
                            .map_err(|error| Exception::custom(error.to_string()))
                    },
                    streams,
                    visitor,
                )
                .map_err(|error| Error::Speculative(error.to_string()))
            }
        }
    }
}

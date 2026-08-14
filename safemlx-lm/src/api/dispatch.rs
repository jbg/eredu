//! Architecture-erased model, cache, and generation dispatch.

use super::*;

/// Loaded model value for any architecture supported by this crate.
pub enum Model {
    /// DeepSeek-V3/R1 model.
    DeepSeekV3(crate::architectures::deepseek_v3::layerwise::DeepSeekV3LayerwiseModel),
    /// DeepSeek-V4 target model.
    DeepSeekV4(Box<crate::architectures::deepseek_v4::model::Model>),
    /// DeepSeek-V4 model using generalized bounded residency.
    DeepSeekV4Layerwise(
        Box<crate::architectures::deepseek_v4::layerwise::DeepSeekV4LayerwiseModel>,
    ),
    /// Gemma 4 text and multimodal model.
    Gemma4(Box<crate::architectures::gemma4::layerwise::Gemma4LayerwiseModel>),
    /// OpenAI GPT-OSS model.
    GptOss(crate::architectures::gpt_oss::layerwise::GptOssLayerwiseModel),
    /// Moonshot Kimi Linear hybrid KDA/MLA sparse decoder.
    KimiLinear(crate::architectures::kimi_linear::layerwise::KimiLinearLayerwiseModel),
    /// Thinking Machines Lab Inkling multimodal model.
    Inkling(crate::architectures::inkling::layerwise::InklingLayerwiseModel),
    /// Llama-compatible dense model.
    Llama(crate::architectures::llama::layerwise::LlamaModel),
    /// Meta Muse-Glimmer dense multimodal model.
    MuseGlimmer(crate::architectures::muse_glimmer::layerwise::LayerwiseDecoder),
    /// Liquid AI LFM2/LFM2.5 model.
    Lfm2(crate::architectures::lfm2::layerwise::Lfm2LayerwiseModel),
    /// Nemotron-H hybrid model.
    NemotronH(crate::architectures::nemotron_h::layerwise::NemotronHLayerwiseModel),
    /// Dense Qwen2/Qwen2.5/Qwen3 model.
    DenseQwen(crate::architectures::qwen::dense::layerwise::LayerwiseDecoder),
    /// Qwen3-Next model.
    Qwen3Next(crate::architectures::qwen::hybrid::layerwise::QwenHybridLayerwiseModel),
    /// Qwen3-VL multimodal model.
    Qwen3Vl(crate::architectures::qwen::vl::layerwise::Qwen3VlLayerwiseModel),
    /// Qwen3-VL-MoE multimodal model.
    Qwen3VlMoe(crate::architectures::qwen::vl::layerwise::Qwen3VlLayerwiseModel),
    /// Qwen3.5 dense or MoE model, optionally multimodal.
    Qwen35(crate::architectures::qwen::hybrid::layerwise::QwenHybridLayerwiseModel),
}

impl Model {
    /// Returns architecture-neutral rank-local placement information when this
    /// model was loaded through generalized parallel execution groups.
    pub fn parallel_info(&self) -> Option<&crate::ParallelModelInfo> {
        match self {
            Self::Llama(model) => model.parallel_info(),
            Self::MuseGlimmer(model) => model.parallel_info(),
            Self::DeepSeekV3(model) => model.parallel_info(),
            Self::DeepSeekV4(_) | Self::DeepSeekV4Layerwise(_) => None,
            Self::GptOss(model) => model.parallel_info(),
            Self::DenseQwen(model) => model.parallel_info(),
            Self::KimiLinear(model) => model.parallel_info(),
            Self::Lfm2(model) => model.parallel_info(),
            Self::NemotronH(model) => model.parallel_info(),
            Self::Qwen3Next(model) | Self::Qwen35(model) => model.parallel_info(),
            Self::Qwen3Vl(model) | Self::Qwen3VlMoe(model) => model.parallel_info(),
            Self::Gemma4(model) => model.parallel_info(),
            Self::Inkling(model) => model.parallel_info(),
        }
    }

    /// Reports how this model architecture exposes MTP weights.
    pub fn mtp_capability(&self) -> MtpCapability {
        match self {
            Self::Gemma4(_) | Self::MuseGlimmer(_) => MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Separate,
            },
            Self::DeepSeekV3(model) if model.mtp_len() > 0 => MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Embedded,
            },
            Self::DeepSeekV4(model) if model.mtp_len() > 0 => MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Embedded,
            },
            Self::DeepSeekV4Layerwise(model) if model.mtp_len() > 0 => MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Embedded,
            },
            Self::Inkling(model) if model.mtp_len() > 0 => MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Embedded,
            },
            Self::Qwen3Next(model) if model.mtp_len() > 0 => MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Embedded,
            },
            Self::Qwen35(model) if model.mtp_len() > 0 => MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Embedded,
            },
            Self::NemotronH(model) if model.mtp_len() > 0 => MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Embedded,
            },
            _ => MtpCapability::Unavailable,
        }
    }

    /// Generates with MTP using the default lossless sampling policy.
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

    /// Generates with MTP using a caller-provided lossless sampling policy.
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
        self.generate_mtp_input_with_sampler_callback_and_streams(
            drafter,
            cache,
            input,
            config,
            prng_key,
            sampler,
            MtpExecutionStreams::single(stream),
            |_| Ok(()),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn generate_mtp_input_with_sampler_callback_and_streams<S, F>(
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

    #[allow(clippy::too_many_arguments)]
    pub(super) fn generate_mtp_input_with_sampler_callback_and_streams_and_options<S, F>(
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
        match self {
            Self::Gemma4(target) => {
                let ModelCache::Gemma4(cache) = cache else {
                    return Err(Exception::custom("Gemma 4 MTP cache type mismatch"));
                };
                let assistant = drafter.gemma4_mut();
                validate_gemma4_drafter(target.args(), assistant)?;
                crate::architectures::gemma4::mtp::generate_with_streams_and_callback_and_options(
                    target,
                    assistant,
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    streams,
                    scheduler_options,
                    on_token,
                )
            }
            Self::MuseGlimmer(target) => {
                let assistant = drafter.muse_glimmer_mut();
                let mut backend =
                    crate::architectures::muse_glimmer::mtp::MuseGlimmerMtpBackend::new(
                        target, assistant,
                    );
                crate::runtime::generation::speculative::generate_with_streams_and_callback_and_options(
                    &mut backend,
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    streams,
                    scheduler_options,
                    on_token,
                )
            }
            model => Err(Exception::custom(format!(
                "MTP runtime adapter is unavailable for model type {} ({:?})",
                model.model_type(),
                model.mtp_capability()
            ))),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn generate_mtp_input_with_semantics_and_options<S, F>(
        &mut self,
        drafter: &mut LoadedDrafter,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        semantic: Box<dyn MtpSemanticState>,
        cancellation: GenerationCancellationToken,
        streams: MtpExecutionStreams<'_>,
        scheduler_options: MtpSchedulerOptions,
        on_event: F,
    ) -> Result<(Vec<u32>, MtpStats, FinishReason), Exception>
    where
        S: SpeculativeSampler + Clone,
        F: FnMut(SemanticEvent),
    {
        match self {
            Self::Gemma4(target) => {
                let ModelCache::Gemma4(cache) = cache else {
                    return Err(Exception::custom("Gemma 4 MTP cache type mismatch"));
                };
                let assistant = drafter.gemma4_mut();
                validate_gemma4_drafter(target.args(), assistant)?;
                crate::architectures::gemma4::mtp::generate_with_semantics_and_options(
                    target,
                    assistant,
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    semantic,
                    cancellation,
                    streams,
                    scheduler_options,
                    on_event,
                )
            }
            Self::MuseGlimmer(target) => {
                let assistant = drafter.muse_glimmer_mut();
                let mut backend =
                    crate::architectures::muse_glimmer::mtp::MuseGlimmerMtpBackend::new(
                        target, assistant,
                    );
                crate::runtime::generation::speculative::generate_with_semantics_and_options(
                    &mut backend,
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    semantic,
                    cancellation,
                    streams,
                    scheduler_options,
                    on_event,
                )
            }
            model => Err(Exception::custom(format!(
                "MTP runtime adapter is unavailable for model type {} ({:?})",
                model.model_type(),
                model.mtp_capability()
            ))),
        }
    }

    /// Generates with MTP weights embedded in the target checkpoint.
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

    /// Generates through embedded predictor layers on a non-pipeline
    /// Cartesian model. TP collectives and rank-synchronized sampling are
    /// derived from `execution`; EP and PP models use their architecture-erased
    /// distributed model containers.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_embedded_mtp_cartesian<S: SpeculativeSampler + Clone>(
        &mut self,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        execution: &crate::CartesianExecution<'_>,
        stream: &Stream,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        let topology = execution.topology();
        if topology.pipeline_parallel_size != 1 || topology.expert_parallel_size != 1 {
            return Err(Exception::custom(
                "architecture-erased Cartesian MTP requires a non-pipeline, non-EP model; use PipelineModel or ExpertParallelModel for active PP/EP axes",
            ));
        }
        if self.parallel_info().map(|info| info.topology()) != Some(topology) {
            return Err(Exception::custom(
                "embedded MTP model topology does not match Cartesian execution",
            ));
        }
        let tensor = execution
            .tensor_context(stream)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let tensor_group = tensor.group().ok_or_else(|| {
            Exception::custom("Cartesian embedded MTP requires an active TP subgroup")
        })?;
        let sampling_rank = topology
            .global_rank_for(crate::ParallelCoordinates {
                tensor: 0,
                pipeline: 0,
                expert: 0,
            })
            .map_err(|error| Exception::custom(error.to_string()))?;
        let mut synchronized =
            crate::runtime::generation::embedded_mtp::DistributedEmbeddedMtpSampler::new(
                sampler.clone(),
                sampling_rank,
                execution.world(),
            )
            .map_err(|error| Exception::custom(error.to_string()))?;
        let result = match (self, cache) {
            (Self::DeepSeekV3(model), ModelCache::DeepSeekV3(cache)) => {
                let mut target =
                    crate::architectures::deepseek_v3::layerwise::DeepSeekTensorMtpTarget::new(
                        model,
                        tensor_group,
                    );
                crate::runtime::generation::embedded_mtp::generate_with_callback(
                    &mut target,
                    cache,
                    input,
                    config,
                    prng_key,
                    &mut synchronized,
                    stream,
                    |_| Ok(()),
                )
            }
            (Self::Inkling(model), ModelCache::Inkling(cache)) => {
                let mut target =
                    crate::architectures::inkling::layerwise::InklingTensorMtpTarget::new(
                        model,
                        tensor_group,
                    );
                crate::runtime::generation::embedded_mtp::generate_with_callback(
                    &mut target,
                    cache,
                    input,
                    config,
                    prng_key,
                    &mut synchronized,
                    stream,
                    |_| Ok(()),
                )
            }
            (Self::NemotronH(model), ModelCache::NemotronH(cache)) => {
                let mut target =
                    crate::architectures::nemotron_h::layerwise::NemotronHTensorMtpTarget::new(
                        model,
                        tensor_group,
                    );
                crate::runtime::generation::embedded_mtp::generate_with_callback(
                    &mut target,
                    cache,
                    input,
                    config,
                    prng_key,
                    &mut synchronized,
                    stream,
                    |_| Ok(()),
                )
            }
            (Self::Qwen3Next(model), ModelCache::Qwen3Next(cache))
            | (Self::Qwen35(model), ModelCache::Qwen35(cache)) => {
                let mut target =
                    crate::architectures::qwen::hybrid::layerwise::QwenHybridTensorMtpTarget::new(
                        model,
                        tensor_group,
                    );
                crate::architectures::qwen::hybrid::mtp::generate_with_callback(
                    &mut target,
                    cache,
                    input,
                    config,
                    prng_key,
                    &mut synchronized,
                    stream,
                    |_| Ok(()),
                )
            }
            (model, _) => Err(Exception::custom(format!(
                "Cartesian embedded MTP runtime adapter is unavailable for model type {} ({:?})",
                model.model_type(),
                model.mtp_capability()
            ))),
        };
        *sampler = synchronized.into_inner();
        result
    }

    /// Generates with embedded MTP weights and a caller-provided sampler.
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
        self.generate_embedded_mtp_input_with_sampler_callback(
            cache,
            input,
            config,
            prng_key,
            sampler,
            stream,
            |_| Ok(()),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn generate_embedded_mtp_input_with_sampler_callback<S, F>(
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
        match (self, cache) {
            (Self::DeepSeekV3(target), ModelCache::DeepSeekV3(cache)) => {
                crate::runtime::generation::embedded_mtp::generate_with_callback(
                    target, cache, input, config, prng_key, sampler, stream, on_token,
                )
            }
            (Self::DeepSeekV4(target), ModelCache::DeepSeekV4(cache)) => {
                crate::runtime::generation::embedded_mtp::generate_with_callback(
                    target.as_mut(),
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    stream,
                    on_token,
                )
            }
            (Self::DeepSeekV4Layerwise(target), ModelCache::DeepSeekV4(cache)) => {
                crate::runtime::generation::embedded_mtp::generate_with_callback(
                    target.as_mut(),
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    stream,
                    on_token,
                )
            }
            (Self::Inkling(target), ModelCache::Inkling(cache)) => {
                crate::runtime::generation::embedded_mtp::generate_with_callback(
                    target, cache, input, config, prng_key, sampler, stream, on_token,
                )
            }
            (Self::NemotronH(target), ModelCache::NemotronH(cache)) => {
                crate::runtime::generation::embedded_mtp::generate_with_callback(
                    target, cache, input, config, prng_key, sampler, stream, on_token,
                )
            }
            (Self::Qwen3Next(target), ModelCache::Qwen3Next(cache))
            | (Self::Qwen35(target), ModelCache::Qwen35(cache)) => {
                crate::architectures::qwen::hybrid::mtp::generate_with_callback(
                    target, cache, input, config, prng_key, sampler, stream, on_token,
                )
            }
            (model, _) => Err(Exception::custom(format!(
                "embedded MTP runtime adapter is unavailable for model type {} ({:?})",
                model.model_type(),
                model.mtp_capability()
            ))),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn generate_embedded_mtp_input_with_semantics_and_options<S, F>(
        &mut self,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        semantic: Box<dyn MtpSemanticState>,
        cancellation: GenerationCancellationToken,
        stream: &Stream,
        scheduler_options: MtpSchedulerOptions,
        on_event: F,
    ) -> Result<(Vec<u32>, MtpStats, FinishReason), Exception>
    where
        S: SpeculativeSampler + Clone,
        F: FnMut(SemanticEvent),
    {
        match (self, cache) {
            (Self::DeepSeekV3(target), ModelCache::DeepSeekV3(cache)) => {
                crate::runtime::generation::embedded_mtp::generate_with_semantics_and_options(
                    target,
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    semantic,
                    cancellation,
                    stream,
                    scheduler_options,
                    on_event,
                )
            }
            (Self::DeepSeekV4(target), ModelCache::DeepSeekV4(cache)) => {
                crate::runtime::generation::embedded_mtp::generate_with_semantics_and_options(
                    target.as_mut(),
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    semantic,
                    cancellation,
                    stream,
                    scheduler_options,
                    on_event,
                )
            }
            (Self::DeepSeekV4Layerwise(target), ModelCache::DeepSeekV4(cache)) => {
                crate::runtime::generation::embedded_mtp::generate_with_semantics_and_options(
                    target.as_mut(),
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    semantic,
                    cancellation,
                    stream,
                    scheduler_options,
                    on_event,
                )
            }
            (Self::Inkling(target), ModelCache::Inkling(cache)) => {
                crate::runtime::generation::embedded_mtp::generate_with_semantics_and_options(
                    target,
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    semantic,
                    cancellation,
                    stream,
                    scheduler_options,
                    on_event,
                )
            }
            (Self::NemotronH(target), ModelCache::NemotronH(cache)) => {
                crate::runtime::generation::embedded_mtp::generate_with_semantics_and_options(
                    target,
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    semantic,
                    cancellation,
                    stream,
                    scheduler_options,
                    on_event,
                )
            }
            (Self::Qwen3Next(target), ModelCache::Qwen3Next(cache))
            | (Self::Qwen35(target), ModelCache::Qwen35(cache)) => {
                crate::architectures::qwen::hybrid::mtp::generate_with_semantics_and_options(
                    target,
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    semantic,
                    cancellation,
                    stream,
                    scheduler_options,
                    on_event,
                )
            }
            (model, _) => Err(Exception::custom(format!(
                "embedded MTP runtime adapter is unavailable for model type {} ({:?})",
                model.model_type(),
                model.mtp_capability()
            ))),
        }
    }

    /// Returns residency telemetry when this model uses bounded layer execution.
    pub fn residency_report(
        &self,
    ) -> Result<Option<crate::runtime::residency::manager::ResidencyReport>, Error> {
        match self {
            Self::DeepSeekV3(model) => Ok(Some(model.residency_report()?)),
            Self::DeepSeekV4(_) => Ok(None),
            Self::DeepSeekV4Layerwise(model) => Ok(Some(model.residency_report()?)),
            Self::Gemma4(model) => Ok(Some(model.residency_report()?)),
            Self::Inkling(model) => Ok(Some(model.residency_report()?)),
            Self::KimiLinear(model) => Ok(Some(model.residency_report()?)),
            Self::Llama(model) => model.residency_report(),
            Self::GptOss(model) => Ok(Some(model.residency_report()?)),
            Self::Lfm2(model) => Ok(Some(model.residency_report()?)),
            Self::NemotronH(model) => Ok(Some(model.residency_report()?)),
            Self::Qwen3Next(model) | Self::Qwen35(model) => Ok(Some(model.residency_report()?)),
            Self::DenseQwen(model) => Ok(Some(model.residency_report()?)),
            Self::MuseGlimmer(model) => Ok(Some(model.residency_report()?)),
            Self::Qwen3Vl(model) | Self::Qwen3VlMoe(model) => Ok(Some(model.residency_report()?)),
        }
    }

    /// Returns experimental dense-stream telemetry when enabled.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<crate::runtime::execution::layerwise::DenseDiskStreamReport>, Error> {
        match self {
            Self::DeepSeekV3(model) => model.dense_stream_report(),
            Self::DeepSeekV4(_) => Ok(None),
            Self::DeepSeekV4Layerwise(model) => model.dense_stream_report(),
            Self::Gemma4(model) => model.dense_stream_report(),
            Self::Inkling(model) => model.dense_stream_report(),
            Self::KimiLinear(model) => model.dense_stream_report(),
            Self::Llama(model) => model.dense_stream_report(),
            Self::GptOss(model) => model.dense_stream_report(),
            Self::Lfm2(model) => model.dense_stream_report(),
            Self::NemotronH(model) => model.dense_stream_report(),
            Self::Qwen3Next(model) | Self::Qwen35(model) => model.dense_stream_report(),
            Self::DenseQwen(model) => model.dense_stream_report(),
            Self::MuseGlimmer(model) => model.dense_stream_report(),
            Self::Qwen3Vl(model) | Self::Qwen3VlMoe(model) => model.dense_stream_report(),
        }
    }

    /// Returns sparse routed-expert cache telemetry when enabled.
    pub fn expert_cache_report(
        &self,
    ) -> Result<Option<crate::runtime::residency::expert_cache::ExpertCacheReport>, Error> {
        match self {
            Self::DeepSeekV3(model) => model.expert_cache_report(),
            Self::DeepSeekV4Layerwise(model) => model.expert_cache_report(),
            Self::Gemma4(model) => model.expert_cache_report(),
            Self::KimiLinear(model) => model.expert_cache_report(),
            Self::GptOss(model) => model.expert_cache_report(),
            Self::Inkling(model) => model.expert_cache_report(),
            Self::Lfm2(model) => model.expert_cache_report(),
            Self::NemotronH(model) => model.expert_cache_report(),
            Self::DenseQwen(model) => model.expert_cache_report(),
            Self::Qwen3Next(model) | Self::Qwen35(model) => model.expert_cache_report(),
            Self::Qwen3VlMoe(model) => model.expert_cache_report(),
            _ => Ok(None),
        }
    }

    /// Returns the effective model type used for dispatch.
    pub fn model_type(&self) -> &str {
        match self {
            Self::DeepSeekV3(model) => &model.args().model_type,
            Self::DeepSeekV4(model) => &model.args.model_type,
            Self::DeepSeekV4Layerwise(model) => &model.args().model_type,
            Self::Gemma4(model) => &model.args().model_type,
            Self::GptOss(model) => &model.args().model_type,
            Self::Inkling(model) => &model.args().model_type,
            Self::KimiLinear(model) => &model.args().model_type,
            Self::Llama(model) => &model.args().model_type,
            Self::Lfm2(model) => &model.args().model_type,
            Self::NemotronH(model) => &model.args().model_type,
            Self::DenseQwen(model) => &model.args().model_type,
            Self::MuseGlimmer(model) => &model.args().model_type,
            Self::Qwen3Next(model) => &model.args().model_type,
            Self::Qwen3Vl(model) => model.model_type(),
            Self::Qwen3VlMoe(model) => model.model_type(),
            Self::Qwen35(model) => &model.args().model_type,
        }
    }

    /// Returns checkpoint-native quantization storage statistics when available.
    pub fn native_quantization_stats(
        &self,
    ) -> Option<&safemlx::native_quantization::NativeQuantizationStats> {
        None
    }

    /// Returns the canonical cache-relevant architecture identity derived from the loaded model.
    pub fn prompt_cache_architecture_fingerprint(&self) -> Result<String, Exception> {
        match self {
            Self::Gemma4(model) => Ok(gemma4::prompt_cache_architecture_fingerprint(model.args())),
            Self::Llama(model) => Ok(llama::prompt_cache_architecture_fingerprint(model.args())),
            Self::DeepSeekV3(model) => Ok(deepseek_v3::prompt_cache_architecture_fingerprint(
                model.args(),
            )),
            Self::DeepSeekV4(model) => Ok(deepseek_v4::prompt_cache_architecture_fingerprint(
                &model.args,
            )),
            Self::DeepSeekV4Layerwise(model) => Ok(
                deepseek_v4::prompt_cache_architecture_fingerprint(model.args()),
            ),
            Self::GptOss(model) => Ok(gpt_oss::prompt_cache_architecture_fingerprint(model.args())),
            Self::Inkling(model) => {
                Ok(inkling::prompt_cache_architecture_fingerprint(model.args()))
            }
            Self::KimiLinear(model) => Ok(kimi_linear::prompt_cache_architecture_fingerprint(
                model.args(),
            )),
            Self::Lfm2(model) => Ok(lfm2::prompt_cache_architecture_fingerprint(model.args())),
            Self::DenseQwen(model) => Ok(dense_qwen::prompt_cache_architecture_fingerprint(
                model.args(),
            )),
            Self::MuseGlimmer(model) => Ok(muse_glimmer::prompt_cache_architecture_fingerprint(
                model.args(),
            )),
            Self::NemotronH(model) => Ok(nemotron_h::prompt_cache_architecture_fingerprint(
                model.args(),
            )),
            Self::Qwen3Next(model) | Self::Qwen35(model) => {
                Ok(qwen3_5::prompt_cache_architecture_fingerprint(model.args()))
            }
            Self::Qwen3Vl(model) | Self::Qwen3VlMoe(model) => Ok(
                qwen3_vl::prompt_cache_architecture_fingerprint(model.args()),
            ),
        }
    }

    /// Returns the exact ordered prompt-cache state and attention layout.
    pub fn prompt_cache_layer_layout(&self) -> Result<LayerSchedule<LayerCachePolicy>, Exception> {
        match self {
            Self::Llama(model) => model.prompt_cache_layer_layout(),
            Self::DeepSeekV3(model) => model.prompt_cache_layer_layout(),
            Self::DeepSeekV4(model) => model.prompt_cache_layer_layout(),
            Self::DeepSeekV4Layerwise(model) => model.prompt_cache_layer_layout(),
            Self::GptOss(model) => model.prompt_cache_layer_layout(),
            Self::DenseQwen(model) => model.prompt_cache_layer_layout(),
            Self::MuseGlimmer(model) => model.prompt_cache_layer_layout(),
            Self::KimiLinear(model) => model.prompt_cache_layer_layout(),
            Self::Lfm2(model) => model.prompt_cache_layer_layout(),
            Self::NemotronH(model) => model.prompt_cache_layer_layout(),
            Self::Qwen3Next(model) | Self::Qwen35(model) => model.prompt_cache_layer_layout(),
            Self::Qwen3Vl(model) | Self::Qwen3VlMoe(model) => model.prompt_cache_layer_layout(),
            Self::Gemma4(model) => model.prompt_cache_layer_layout(),
            Self::Inkling(model) => model.prompt_cache_layer_layout(),
        }
        .map_err(|error| Exception::custom(error.to_string()))
    }

    /// Runs an instrumented pass through the canonical generalized executor.
    pub fn forward_with_observer(
        &mut self,
        input_tokens: &Array,
        mask: Option<&Array>,
        cache: &mut ModelCache,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        let result = match (self, cache) {
            (Self::DeepSeekV3(model), ModelCache::DeepSeekV3(cache)) => {
                if mask.is_some() {
                    return Err(Exception::custom(
                        "an explicit DeepSeek observer mask is unsupported; the adapter constructs the causal mask from cache state",
                    ));
                }
                model.forward_with_observer(input_tokens, cache, stream, observer)
            }
            (Self::KimiLinear(model), ModelCache::KimiLinear(cache)) => {
                if mask.is_some() {
                    return Err(Exception::custom(
                        "an explicit Kimi Linear observer mask is unsupported; the adapter constructs the causal mask from cache state",
                    ));
                }
                model.forward_with_observer(input_tokens, cache, stream, observer)
            }
            (Self::Llama(model), ModelCache::Llama(cache)) => {
                model.forward_with_observer(input_tokens, mask, cache, stream, observer)
            }
            (Self::DenseQwen(model), ModelCache::KeyValue(cache)) => {
                model.forward_with_observer(input_tokens, mask, cache, stream, observer)
            }
            (Self::DenseQwen(model), ModelCache::PagedKeyValue(cache)) => {
                model.forward_paged_with_observer(input_tokens, mask, cache, stream, observer)
            }
            (Self::MuseGlimmer(model), ModelCache::KeyValue(cache)) => {
                model.forward_with_observer(input_tokens, mask, cache, stream, observer)
            }
            (Self::MuseGlimmer(model), ModelCache::PagedKeyValue(cache)) => {
                model.forward_paged_with_observer(input_tokens, mask, cache, stream, observer)
            }
            (Self::Qwen3Next(model), ModelCache::Qwen3Next(cache))
            | (Self::Qwen35(model), ModelCache::Qwen35(cache)) => {
                if mask.is_some() {
                    return Err(Exception::custom(
                        "an explicit Qwen hybrid observer mask is unsupported; the adapter constructs the causal mask from cache state",
                    ));
                }
                model.forward_with_observer(input_tokens, cache, stream, observer)
            }
            (Self::Gemma4(model), ModelCache::Gemma4(cache)) => {
                if mask.is_some() {
                    return Err(Exception::custom(
                        "an explicit Gemma observer mask is unsupported; the adapter constructs its per-layer masks from cache state",
                    ));
                }
                model.forward_with_observer(input_tokens, cache, stream, observer)
            }
            (model, _) => {
                return Err(Exception::custom(format!(
                    "activation observation is unavailable for model type {} or the supplied cache does not match",
                    model.model_type()
                )))
            }
        };
        result.map_err(|error| Exception::custom(error.to_string()))
    }

    /// Computes initial prompt logits while reporting detailed activations.
    pub fn prefill_input_with_observer(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut ModelCache,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        if let (Self::Gemma4(model), ModelCache::Gemma4(cache)) = (&mut *self, &mut *cache) {
            return model
                .prefill_input_with_observer(input, cache, stream, observer)
                .map_err(|error| Exception::custom(error.to_string()))?
                .try_index_device((.., -1, ..), stream);
        }
        match (&mut *self, &mut *cache) {
            (Self::Qwen3Next(model), ModelCache::Qwen3Next(cache))
            | (Self::Qwen35(model), ModelCache::Qwen35(cache)) => {
                return model
                    .prefill_input_with_observer(input, cache, stream, observer)
                    .map_err(|error| Exception::custom(error.to_string()))?
                    .try_index_device((.., -1, ..), stream);
            }
            _ => {}
        }
        let tokens = input::text_token_ids(input, stream)?;
        self.forward_with_observer(&tokens, None, cache, stream, observer)?
            .try_index_device((.., -1, ..), stream)
    }

    /// Creates an empty cache value appropriate for this model.
    pub fn new_cache(&self) -> ModelCache {
        match self {
            Self::DeepSeekV3(model) => ModelCache::DeepSeekV3(model.new_cache()),
            Self::DeepSeekV4(model) => ModelCache::DeepSeekV4(
                model
                    .new_cache()
                    .expect("validated DeepSeek-V4 cache geometry"),
            ),
            Self::DeepSeekV4Layerwise(model) => ModelCache::DeepSeekV4(
                model
                    .new_cache()
                    .expect("validated layerwise DeepSeek-V4 cache geometry"),
            ),
            Self::Gemma4(model) => ModelCache::Gemma4(model.new_cache()),
            Self::GptOss(model) => ModelCache::GptOss(model.new_cache()),
            Self::Inkling(model) => ModelCache::Inkling(model.new_cache()),
            Self::KimiLinear(model) => ModelCache::KimiLinear(model.new_cache()),
            Self::Llama(model) => ModelCache::Llama(model.new_cache()),
            Self::Lfm2(model) => ModelCache::Lfm2(model.new_cache()),
            Self::DenseQwen(model) => ModelCache::KeyValue(model.new_cache()),
            Self::MuseGlimmer(model) => ModelCache::KeyValue(model.new_cache()),
            Self::Qwen3Next(model) => ModelCache::Qwen3Next(model.new_cache()),
            Self::Qwen3Vl(model) => ModelCache::Qwen3Vl(model.new_cache()),
            Self::Qwen3VlMoe(model) => ModelCache::Qwen3VlMoe(model.new_cache()),
            Self::NemotronH(model) => ModelCache::NemotronH(model.new_cache()),
            Self::Qwen35(model) => ModelCache::Qwen35(model.new_cache()),
        }
    }

    /// Creates ordinary cache state or an explicitly bounded paged cache.
    pub fn new_cache_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<ModelCache, Exception> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => match self {
                Self::Llama(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::Llama)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::DeepSeekV3(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::DeepSeekV3)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::DeepSeekV4(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::DeepSeekV4)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::DeepSeekV4Layerwise(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::DeepSeekV4)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::KimiLinear(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::KimiLinear)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::GptOss(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::GptOss)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::DenseQwen(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .and_then(|cache| match cache {
                        crate::architectures::qwen::dense::layerwise::DenseQwenLayerwiseCache::Paged(caches) => Ok(ModelCache::PagedKeyValue(caches)),
                        _ => Err(Error::UnsupportedArchitecture(
                            "dense-Qwen paged cache construction returned device state".into(),
                        )),
                    })
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::MuseGlimmer(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .and_then(|cache| match cache {
                        crate::architectures::muse_glimmer::layerwise::MuseGlimmerLayerwiseCache::Paged(caches) => Ok(ModelCache::PagedKeyValue(caches)),
                        _ => Err(Error::UnsupportedArchitecture(
                            "Muse-Glimmer paged cache construction returned device state".into(),
                        )),
                    })
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::Inkling(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::Inkling)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::NemotronH(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::NemotronH)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::Lfm2(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::Lfm2)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::Qwen3Next(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::Qwen3Next)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::Qwen35(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::Qwen35)
                    .map_err(|error| Exception::custom(error.to_string())),
                _ => Err(Exception::custom(format!(
                    "paged cache residency is unsupported for model type {}",
                    self.model_type()
                ))),
            },
        }
    }

    /// Lazily catalogs a compatible persisted text prefix for a fresh cache.
    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(ModelCache, PromptCacheManifest), Exception> {
        macro_rules! load {
            ($model:expr, $variant:path) => {
                $model
                    .load_prompt_cache(directory, expected, prefix_token_ids, options, stream)
                    .map(|(cache, manifest)| ($variant(cache), manifest))
                    .map_err(|error| Exception::custom(error.to_string()))
            };
        }
        match self {
            Self::Llama(model) => load!(model, ModelCache::Llama),
            Self::DeepSeekV3(model) => load!(model, ModelCache::DeepSeekV3),
            Self::DeepSeekV4(model) => load!(model, ModelCache::DeepSeekV4),
            Self::DeepSeekV4Layerwise(model) => load!(model, ModelCache::DeepSeekV4),
            Self::GptOss(model) => load!(model, ModelCache::GptOss),
            Self::DenseQwen(model) => load!(model, ModelCache::KeyValue),
            Self::MuseGlimmer(model) => load!(model, ModelCache::KeyValue),
            Self::KimiLinear(model) => load!(model, ModelCache::KimiLinear),
            Self::Qwen3Next(model) => load!(model, ModelCache::Qwen3Next),
            Self::Qwen35(model) => load!(model, ModelCache::Qwen35),
            Self::Qwen3Vl(model) => load!(model, ModelCache::Qwen3Vl),
            Self::Qwen3VlMoe(model) => load!(model, ModelCache::Qwen3VlMoe),
            Self::Gemma4(model) => load!(model, ModelCache::Gemma4),
            Self::Inkling(model) => load!(model, ModelCache::Inkling),
            Self::Lfm2(model) => load!(model, ModelCache::Lfm2),
            Self::NemotronH(model) => load!(model, ModelCache::NemotronH),
        }
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
        match (self, &mut *cache) {
            (Self::Llama(model), ModelCache::Llama(cache)) => {
                return model
                    .save_prompt_cache(
                        cache,
                        &destination,
                        descriptor,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()));
            }
            (Self::DeepSeekV3(model), ModelCache::DeepSeekV3(cache)) => {
                return model
                    .save_prompt_cache(
                        cache,
                        &destination,
                        descriptor,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()));
            }
            (Self::DeepSeekV4(model), ModelCache::DeepSeekV4(cache)) => {
                return model
                    .save_prompt_cache(
                        cache,
                        &destination,
                        descriptor,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()));
            }
            (Self::DeepSeekV4Layerwise(model), ModelCache::DeepSeekV4(cache)) => {
                return model
                    .save_prompt_cache(
                        cache,
                        &destination,
                        descriptor,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()));
            }
            (Self::GptOss(model), ModelCache::GptOss(cache)) => {
                return model
                    .save_prompt_cache(
                        cache,
                        &destination,
                        descriptor,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()));
            }
            (Self::DenseQwen(model), ModelCache::KeyValue(cache)) => {
                return model
                    .save_prompt_cache(
                        cache,
                        &destination,
                        descriptor,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()));
            }
            (Self::MuseGlimmer(model), ModelCache::KeyValue(cache)) => {
                return model
                    .save_prompt_cache(
                        cache,
                        &destination,
                        descriptor,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()));
            }
            (Self::KimiLinear(model), ModelCache::KimiLinear(cache)) => {
                return model
                    .save_prompt_cache(
                        cache,
                        &destination,
                        descriptor,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()));
            }
            (Self::Lfm2(model), ModelCache::Lfm2(cache)) => {
                return model
                    .save_prompt_cache(
                        cache,
                        &destination,
                        descriptor,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()));
            }
            (Self::NemotronH(model), ModelCache::NemotronH(cache)) => {
                return model
                    .save_prompt_cache(
                        cache,
                        &destination,
                        descriptor,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()));
            }
            (Self::Qwen3Next(model), ModelCache::Qwen3Next(cache))
            | (Self::Qwen35(model), ModelCache::Qwen35(cache)) => {
                return model
                    .save_prompt_cache(
                        cache,
                        &destination,
                        descriptor,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()));
            }
            (Self::Gemma4(model), ModelCache::Gemma4(cache)) => {
                return model
                    .save_prompt_cache(
                        cache,
                        &destination,
                        descriptor,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()));
            }
            (Self::Qwen3Vl(model), ModelCache::Qwen3Vl(cache))
            | (Self::Qwen3VlMoe(model), ModelCache::Qwen3VlMoe(cache)) => {
                return model
                    .save_prompt_cache(
                        cache,
                        &destination,
                        descriptor,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()));
            }
            (Self::Inkling(model), ModelCache::Inkling(cache)) => {
                return model
                    .save_prompt_cache(
                        cache,
                        &destination,
                        descriptor,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()));
            }
            _ => {}
        }
        let layer_layout = self.prompt_cache_layer_layout()?;
        let model_family = match self {
            Self::Llama(_) => "llama",
            Self::DeepSeekV3(_) => "deepseek_v3",
            Self::DeepSeekV4(_) | Self::DeepSeekV4Layerwise(_) => "deepseek_v4",
            Self::GptOss(_) => "gpt_oss",
            Self::DenseQwen(_) => "dense_qwen",
            Self::MuseGlimmer(_) => "muse_glimmer",
            Self::KimiLinear(_) => "kimi_linear",
            Self::Lfm2(_) => "lfm2",
            Self::NemotronH(_) => "nemotron_h",
            Self::Qwen3Next(_) | Self::Qwen35(_) => "qwen_hybrid",
            Self::Qwen3Vl(_) | Self::Qwen3VlMoe(_) => "qwen3_vl",
            Self::Gemma4(_) => "gemma4",
            Self::Inkling(_) => "inkling",
        };
        let layer_count = layer_layout.len();
        let identity = PromptCacheModelIdentity {
            model_family: model_family.into(),
            effective_model_type: self.model_type().into(),
            architecture_fingerprint: self.prompt_cache_architecture_fingerprint()?,
            layer_count,
            global_layer_start: 0,
            global_layer_end: layer_count,
            sink_tokens: 0,
            topology: Default::default(),
            layer_layout,
        };
        validate_prompt_cache_model_identity(&descriptor, &identity)
            .map_err(|error| Exception::custom(error.to_string()))?;
        match (self, cache) {
            (Self::DenseQwen(_), ModelCache::PagedKeyValue(caches)) => {
                for cache in caches.iter_mut().flatten() {
                    cache.finalize()?;
                }
                caches
                    .iter()
                    .flatten()
                    .next()
                    .ok_or_else(|| Exception::custom("cannot persist an empty paged cache"))?
                    .manager()
                    .save_prompt_cache(destination, descriptor, prefix_token_ids, &[], options)
                    .map_err(|error| Exception::custom(error.to_string()))
            }
            (Self::MuseGlimmer(_), ModelCache::PagedKeyValue(caches)) => {
                for cache in caches.iter_mut().flatten() {
                    cache.finalize()?;
                }
                caches
                    .iter()
                    .flatten()
                    .next()
                    .ok_or_else(|| Exception::custom("cannot persist an empty paged cache"))?
                    .manager()
                    .save_prompt_cache(destination, descriptor, prefix_token_ids, &[], options)
                    .map_err(|error| Exception::custom(error.to_string()))
            }
            _ => Err(Exception::custom(
                "model and cache representations do not match for prompt-cache publication",
            )),
        }
    }

    /// Computes logits for an initial typed input using a cache returned by [`Model::new_cache`].
    pub fn prefill_input_with_cache(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut ModelCache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        match (self, cache) {
            (Self::Gemma4(model), ModelCache::Gemma4(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::GptOss(model), ModelCache::GptOss(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Inkling(model), ModelCache::Inkling(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Llama(model), ModelCache::Llama(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Lfm2(model), ModelCache::Lfm2(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::NemotronH(model), ModelCache::NemotronH(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::DenseQwen(model), ModelCache::KeyValue(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::DenseQwen(model), ModelCache::PagedKeyValue(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::MuseGlimmer(model), ModelCache::KeyValue(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::MuseGlimmer(model), ModelCache::PagedKeyValue(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen3Vl(model), ModelCache::Qwen3Vl(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen3VlMoe(model), ModelCache::Qwen3VlMoe(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen3Next(model), ModelCache::Qwen3Next(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen35(model), ModelCache::Qwen35(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::DeepSeekV3(model), ModelCache::DeepSeekV3(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::DeepSeekV4(model), ModelCache::DeepSeekV4(cache)) => {
                let tokens = input::text_token_ids(input, stream)?;
                model
                    .forward(&tokens, Some(cache), stream)?
                    .try_index_device((.., -1, ..), stream)
            }
            (Self::DeepSeekV4Layerwise(model), ModelCache::DeepSeekV4(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::KimiLinear(model), ModelCache::KimiLinear(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            _ => Err(Exception::custom(
                "model cache type does not match model kind",
            )),
        }
    }

    /// Creates a token iterator from typed input using a cache returned by [`Model::new_cache`].
    pub fn generate_input_with_cache<'a>(
        &'a mut self,
        cache: &'a mut ModelCache,
        temp: f32,
        input: input::ModelInput<'a>,
        prng_key: Option<Array>,
        stream: &'a Stream,
    ) -> ModelGenerate<'a> {
        self.generate_input_with_cache_sampler(cache, temp, input, prng_key, stream, DefaultSampler)
    }

    /// Creates a token iterator from typed input with a caller-provided sampler.
    pub fn generate_input_with_cache_sampler<'a, S>(
        &'a mut self,
        cache: &'a mut ModelCache,
        temp: f32,
        input: input::ModelInput<'a>,
        prng_key: Option<Array>,
        stream: &'a Stream,
        sampler: S,
    ) -> ModelGenerate<'a, S>
    where
        S: Sampler,
    {
        match (self, cache) {
            (Self::Gemma4(model), ModelCache::Gemma4(cache)) => ModelGenerate::Gemma4(
                crate::architectures::gemma4::layerwise::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ),
            ),
            (Self::Lfm2(model), ModelCache::Lfm2(cache)) => ModelGenerate::Lfm2(
                crate::architectures::lfm2::layerwise::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ),
            ),
            (Self::GptOss(model), ModelCache::GptOss(cache)) => ModelGenerate::GptOss(
                crate::architectures::gpt_oss::layerwise::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ),
            ),
            (Self::Inkling(model), ModelCache::Inkling(cache)) => ModelGenerate::Inkling(
                crate::architectures::inkling::layerwise::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ),
            ),
            (Self::Llama(model), ModelCache::Llama(cache)) => {
                ModelGenerate::Llama(common::generation::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::DenseQwen(model), ModelCache::KeyValue(cache)) => {
                ModelGenerate::DenseQwen(common::generation::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::DenseQwen(model), ModelCache::PagedKeyValue(cache)) => {
                ModelGenerate::DenseQwenPaged(common::generation::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::MuseGlimmer(model), ModelCache::KeyValue(cache)) => {
                ModelGenerate::MuseGlimmer(common::generation::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::MuseGlimmer(model), ModelCache::PagedKeyValue(cache)) => {
                ModelGenerate::MuseGlimmerPaged(common::generation::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Qwen3Vl(model), ModelCache::Qwen3Vl(cache)) => ModelGenerate::Qwen3Vl(
                crate::architectures::qwen::vl::layerwise::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ),
            ),
            (Self::Qwen3VlMoe(model), ModelCache::Qwen3VlMoe(cache)) => ModelGenerate::Qwen3VlMoe(
                crate::architectures::qwen::vl::layerwise::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ),
            ),
            (Self::NemotronH(model), ModelCache::NemotronH(cache)) => ModelGenerate::NemotronH(
                crate::architectures::nemotron_h::layerwise::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ),
            ),
            (Self::Qwen35(model), ModelCache::Qwen35(cache)) => ModelGenerate::Qwen35(
                crate::architectures::qwen::hybrid::layerwise::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ),
            ),
            (Self::Qwen3Next(model), ModelCache::Qwen3Next(cache)) => ModelGenerate::Qwen3Next(
                crate::architectures::qwen::hybrid::layerwise::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ),
            ),
            (Self::DeepSeekV3(model), ModelCache::DeepSeekV3(cache)) => ModelGenerate::DeepSeekV3(
                crate::architectures::deepseek_v3::layerwise::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ),
            ),
            (Self::DeepSeekV4(model), ModelCache::DeepSeekV4(cache)) => ModelGenerate::DeepSeekV4(
                crate::architectures::deepseek_v4::model::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ),
            ),
            (Self::DeepSeekV4Layerwise(model), ModelCache::DeepSeekV4(cache)) => {
                ModelGenerate::DeepSeekV4Layerwise(
                    crate::architectures::deepseek_v4::layerwise::Generate::with_sampler(
                        model, cache, temp, input, prng_key, stream, sampler,
                    ),
                )
            }
            (Self::KimiLinear(model), ModelCache::KimiLinear(cache)) => ModelGenerate::KimiLinear(
                crate::architectures::kimi_linear::layerwise::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ),
            ),
            _ => panic!("model cache type does not match model kind"),
        }
    }
}

/// Cache value matching a [`Model`] variant.
#[derive(Clone)]
pub enum ModelCache {
    /// Compressed latent MLA cache for DeepSeek-V3/R1.
    DeepSeekV3(deepseek_v3::Cache),
    /// Local and compressed attention caches for DeepSeek-V4.
    DeepSeekV4(crate::architectures::deepseek_v4::model::Cache),
    /// Gemma 4 generation cache.
    Gemma4(gemma4::Cache),
    /// GPT-OSS cache following its canonical per-layer attention schedule.
    GptOss(gpt_oss::Cache),
    /// Alternating global/local attention and short-convolution Inkling cache.
    Inkling(inkling::Cache),
    /// Per-layer key/value caches whose bounds follow the model schedule.
    KeyValue(Vec<Option<ConcatKeyValueCache>>),
    /// Unified Llama cache used by bounded layer execution.
    Llama(crate::architectures::llama::layerwise::LlamaCache),
    /// Qwen3-VL key/value cache and multimodal position state.
    Qwen3Vl(qwen3_vl::Cache),
    /// Qwen3-VL-MoE key/value cache and multimodal position state.
    Qwen3VlMoe(qwen3_vl_moe::Cache),
    /// Homogeneous block-addressable key/value cache under one global budget.
    PagedKeyValue(Vec<Option<PagedKeyValueCache>>),
    /// Heterogeneous Nemotron-H cache.
    NemotronH(nemotron_h::Cache),
    /// Heterogeneous LFM2 attention/convolution cache.
    Lfm2(lfm2::Cache),
    /// Heterogeneous Kimi Linear KDA/MLA cache.
    KimiLinear(kimi_linear::Cache),
    /// Heterogeneous Qwen3.5 MoE cache.
    Qwen35(qwen3_5::Cache),
    /// Heterogeneous Qwen3-Next cache.
    Qwen3Next(qwen3_next::Cache),
}

pub(super) fn validate_gemma4_drafter(
    target: &gemma4::ModelArgs,
    assistant: &gemma4_assistant::Gemma4AssistantDraftModel,
) -> Result<(), Exception> {
    if assistant.config.model_type != "gemma4_assistant" {
        return Err(Exception::custom(format!(
            "expected a gemma4_assistant checkpoint, got {:?}",
            assistant.config.model_type
        )));
    }
    if assistant.config.backbone_hidden_size != target.hidden_size {
        return Err(Exception::custom(format!(
            "Gemma 4 assistant backbone hidden size {} does not match target hidden size {}",
            assistant.config.backbone_hidden_size, target.hidden_size
        )));
    }
    if assistant.config.text_config.vocab_size != target.vocab_size {
        return Err(Exception::custom(format!(
            "Gemma 4 assistant vocabulary size {} does not match target vocabulary size {}",
            assistant.config.text_config.vocab_size, target.vocab_size
        )));
    }
    let draft = &assistant.config.text_config;
    for (layer, draft_policy) in draft.layer_schedule.iter().enumerate() {
        let attention = draft_policy.attention;
        let Some(target_policy) = target
            .layer_schedule
            .iter()
            .find(|policy| policy.attention == attention && policy.key_value.publishes_state())
        else {
            return Err(Exception::custom(format!(
                "Gemma 4 assistant layer {layer} requires {attention:?} shared KV state, but the target has no matching publishing layer"
            )));
        };
        let target_kv_heads = target_policy.num_key_value_heads.get();
        let draft_kv_heads = draft_policy.num_key_value_heads.get();
        if draft_kv_heads != target_kv_heads {
            return Err(Exception::custom(format!(
                "Gemma 4 assistant layer {layer} {attention:?} KV-head count {draft_kv_heads} does not match target count {target_kv_heads}"
            )));
        }

        let target_head_dim = target_policy.head_dim.get();
        let draft_head_dim = draft_policy.head_dim.get();
        if draft_head_dim != target_head_dim {
            return Err(Exception::custom(format!(
                "Gemma 4 assistant layer {layer} {attention:?} head dimension {draft_head_dim} does not match target dimension {target_head_dim}"
            )));
        }

        let target_rope_theta = target.rope_theta_for_layer(attention);
        let draft_rope_theta = draft.rope_theta_for_layer(attention);
        if draft_rope_theta.to_bits() != target_rope_theta.to_bits() {
            return Err(Exception::custom(format!(
                "Gemma 4 assistant layer {layer} {attention:?} RoPE base {draft_rope_theta} does not match target base {target_rope_theta}"
            )));
        }
    }
    if assistant.block_size() <= 1 {
        return Err(Exception::custom(
            "Gemma 4 assistant block_size must permit at least one draft token",
        ));
    }
    Ok(())
}

impl ModelCache {
    /// Returns aggregate cache-residency telemetry when paging is active.
    pub fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        match self {
            Self::PagedKeyValue(caches) => caches
                .iter()
                .flatten()
                .next()
                .map(PagedKeyValueCache::report)
                .transpose(),
            Self::Llama(cache) => cache
                .residency_report()
                .map_err(|error| Exception::custom(error.to_string())),
            Self::DeepSeekV3(cache) => cache.residency_report(),
            Self::DeepSeekV4(cache) => cache.residency_report(),
            Self::GptOss(cache) => cache.residency_report(),
            Self::Inkling(cache) => cache.residency_report(),
            Self::KimiLinear(cache) => cache.residency_report(),
            Self::NemotronH(cache) => cache.residency_report(),
            _ => Ok(None),
        }
    }
}

/// Token iterator for any supported model variant.
pub enum ModelGenerate<'a, S = DefaultSampler>
where
    S: Sampler,
{
    /// DeepSeek-V3/R1 generation iterator.
    DeepSeekV3(crate::architectures::deepseek_v3::layerwise::Generate<'a, S>),
    /// DeepSeek-V4 generation iterator.
    DeepSeekV4(crate::architectures::deepseek_v4::model::Generate<'a, S>),
    /// Bounded-residency DeepSeek-V4 generation iterator.
    DeepSeekV4Layerwise(crate::architectures::deepseek_v4::layerwise::Generate<'a, S>),
    /// Gemma 4 generation iterator.
    Gemma4(crate::architectures::gemma4::layerwise::Generate<'a, S>),
    /// GPT-OSS generation iterator.
    GptOss(crate::architectures::gpt_oss::layerwise::Generate<'a, S>),
    /// Inkling generation iterator.
    Inkling(crate::architectures::inkling::layerwise::Generate<'a, S>),
    /// Kimi Linear generation iterator.
    KimiLinear(crate::architectures::kimi_linear::layerwise::Generate<'a, S>),
    /// Llama generation iterator.
    Llama(
        common::generation::Generate<
            'a,
            crate::architectures::llama::layerwise::LlamaModel,
            crate::architectures::llama::layerwise::LlamaCache,
            S,
        >,
    ),
    /// Dense-Qwen generation iterator.
    DenseQwen(
        common::generation::Generate<
            'a,
            crate::architectures::qwen::dense::layerwise::LayerwiseDecoder,
            Vec<Option<ConcatKeyValueCache>>,
            S,
        >,
    ),
    /// Dense-Qwen generation using a paged KV cache.
    DenseQwenPaged(
        common::generation::Generate<
            'a,
            crate::architectures::qwen::dense::layerwise::LayerwiseDecoder,
            Vec<Option<PagedKeyValueCache>>,
            S,
        >,
    ),
    /// Muse-Glimmer generation iterator.
    MuseGlimmer(
        common::generation::Generate<
            'a,
            crate::architectures::muse_glimmer::layerwise::LayerwiseDecoder,
            Vec<Option<ConcatKeyValueCache>>,
            S,
        >,
    ),
    /// Muse-Glimmer generation using a paged KV cache.
    MuseGlimmerPaged(
        common::generation::Generate<
            'a,
            crate::architectures::muse_glimmer::layerwise::LayerwiseDecoder,
            Vec<Option<PagedKeyValueCache>>,
            S,
        >,
    ),
    /// Qwen3-VL generation iterator.
    Qwen3Vl(crate::architectures::qwen::vl::layerwise::Generate<'a, S>),
    /// Qwen3-VL-MoE generation iterator.
    Qwen3VlMoe(crate::architectures::qwen::vl::layerwise::Generate<'a, S>),
    /// Nemotron-H generation iterator.
    NemotronH(crate::architectures::nemotron_h::layerwise::Generate<'a, S>),
    /// LFM2 generation iterator.
    Lfm2(crate::architectures::lfm2::layerwise::Generate<'a, S>),
    /// Qwen3.5 MoE generation iterator.
    Qwen35(crate::architectures::qwen::hybrid::layerwise::Generate<'a, S>),
    /// Qwen3-Next generation iterator.
    Qwen3Next(crate::architectures::qwen::hybrid::layerwise::Generate<'a, S>),
}

impl<S> ModelGenerate<'_, S>
where
    S: Sampler,
{
    /// Returns the architecture iterator's sampler at its committed prefix.
    pub fn sampler_mut(&mut self) -> &mut S {
        match self {
            Self::DeepSeekV3(generate) => generate.sampler_mut(),
            Self::DeepSeekV4(generate) => generate.sampler_mut(),
            Self::DeepSeekV4Layerwise(generate) => generate.sampler_mut(),
            Self::Gemma4(generate) => generate.sampler_mut(),
            Self::GptOss(generate) => generate.sampler_mut(),
            Self::Inkling(generate) => generate.sampler_mut(),
            Self::KimiLinear(generate) => generate.sampler_mut(),
            Self::Llama(generate) => generate.sampler_mut(),
            Self::Lfm2(generate) => generate.sampler_mut(),
            Self::NemotronH(generate) => generate.sampler_mut(),
            Self::DenseQwenPaged(generate) => generate.sampler_mut(),
            Self::DenseQwen(generate) => generate.sampler_mut(),
            Self::MuseGlimmerPaged(generate) => generate.sampler_mut(),
            Self::MuseGlimmer(generate) => generate.sampler_mut(),
            Self::Qwen3Vl(generate) => generate.sampler_mut(),
            Self::Qwen3VlMoe(generate) => generate.sampler_mut(),
            Self::Qwen35(generate) => generate.sampler_mut(),
            Self::Qwen3Next(generate) => generate.sampler_mut(),
        }
    }
}

impl<S> Iterator for ModelGenerate<'_, S>
where
    S: Sampler,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::DeepSeekV3(generate) => generate.next(),
            Self::DeepSeekV4(generate) => generate.next(),
            Self::DeepSeekV4Layerwise(generate) => generate.next(),
            Self::Gemma4(generate) => generate.next(),
            Self::GptOss(generate) => generate.next(),
            Self::Inkling(generate) => generate.next(),
            Self::KimiLinear(generate) => generate.next(),
            Self::Llama(generate) => generate.next(),
            Self::Lfm2(generate) => generate.next(),
            Self::NemotronH(generate) => generate.next(),
            Self::DenseQwenPaged(generate) => generate.next(),
            Self::DenseQwen(generate) => generate.next(),
            Self::MuseGlimmerPaged(generate) => generate.next(),
            Self::MuseGlimmer(generate) => generate.next(),
            Self::Qwen3Vl(generate) => generate.next(),
            Self::Qwen3VlMoe(generate) => generate.next(),
            Self::Qwen35(generate) => generate.next(),
            Self::Qwen3Next(generate) => generate.next(),
        }
    }
}

#[cfg(test)]
mod gemma4_drafter_compatibility_tests {
    use safemlx::{Device, DeviceType, Stream};

    use super::validate_gemma4_drafter;
    use crate::architectures::gemma4::{
        assistant::{Gemma4AssistantConfig, Gemma4AssistantDraftModel},
        model::{model_args_from_config_value, ModelArgs},
    };
    use crate::runtime::attention::{AttentionPolicy, LayerSchedule};

    fn target_args() -> ModelArgs {
        model_args_from_config_value(&serde_json::json!({
            "model_type": "gemma4",
            "hidden_size": 8,
            "num_hidden_layers": 4,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "rms_norm_eps": 0.00001,
            "vocab_size": 32,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "head_dim": 4,
            "global_head_dim": 4,
            "tie_word_embeddings": true,
            "num_kv_shared_layers": 2,
            "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
            "sliding_window": 64
        }))
        .unwrap()
    }

    fn assistant(target: &ModelArgs) -> Gemma4AssistantDraftModel {
        let config = Gemma4AssistantConfig {
            model_type: "gemma4_assistant".into(),
            backbone_hidden_size: target.hidden_size,
            use_ordered_embeddings: false,
            num_centroids: 2048,
            centroid_intermediate_top_k: 32,
            tie_word_embeddings: true,
            block_size: 4,
            text_config: target.clone(),
            quantization: None,
        };
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        Gemma4AssistantDraftModel::new(config, &stream).unwrap()
    }

    #[test]
    fn validates_shared_kv_geometry_instead_of_provenance() {
        let target = target_args();
        let mut assistant = assistant(&target);
        validate_gemma4_drafter(&target, &assistant).unwrap();

        let mut policies = assistant
            .config
            .text_config
            .layer_schedule
            .iter()
            .copied()
            .collect::<Vec<_>>();
        for policy in &mut policies {
            if policy.attention == AttentionPolicy::Full {
                policy.num_key_value_heads = std::num::NonZeroU32::new(2).unwrap();
            }
        }
        assistant.config.text_config.layer_schedule = LayerSchedule::new(4, policies).unwrap();
        let error = validate_gemma4_drafter(&target, &assistant)
            .unwrap_err()
            .to_string();
        assert!(error.contains("Full KV-head count"));

        let mut policies = assistant
            .config
            .text_config
            .layer_schedule
            .iter()
            .copied()
            .collect::<Vec<_>>();
        policies[0].attention = AttentionPolicy::sliding(32).unwrap();
        assistant.config.text_config.layer_schedule = LayerSchedule::new(4, policies).unwrap();
        let error = validate_gemma4_drafter(&target, &assistant)
            .unwrap_err()
            .to_string();
        assert!(error.contains("no matching publishing layer"));
    }
}

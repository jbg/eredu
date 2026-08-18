//! Architecture-erased model, cache, and generation dispatch.

use std::path::Path;

use safemlx::{error::Exception, Array, Stream};
use safemlx_lm_core::cache::{
    validate_prompt_cache_model_identity, PromptCacheDescriptor, PromptCacheManifest,
    PromptCacheModelIdentity, PromptCacheOptions,
};
use safemlx_lm_core::generation::{
    FinishReason, GenerationCancellationToken, MtpConfig, MtpSchedulerOptions, SemanticEvent,
};
use safemlx_lm_core::{MtpCapability, MtpCheckpointKind, MtpStats, SpeculativeSemanticState};

use crate::architectures::{
    deepseek_v3::model as deepseek_v3,
    deepseek_v4::model as deepseek_v4,
    gemma4::assistant as gemma4_assistant,
    gemma4::model as gemma4,
    gpt_oss::model as gpt_oss,
    inkling::model as inkling,
    kimi_linear::model as kimi_linear,
    lfm2::model as lfm2,
    llama::model as llama,
    muse_glimmer,
    nemotron_h::model as nemotron_h,
    qwen::{
        dense as dense_qwen,
        hybrid::{qwen3_5, qwen3_next},
        vl::{model as qwen3_vl, moe as qwen3_vl_moe},
    },
};
use crate::backend::mlx::speculative::{MlxDrafter, MtpExecutionStreams};
use crate::error::Error;
use crate::runtime::cache::residency::{
    CacheResidencyPolicy, CacheResidencyReport, PagedCacheOptions,
};
use crate::runtime::cache::{ConcatKeyValueCache, PagedKeyValueCache};
use crate::runtime::execution::inspection::ActivationObserver;
use crate::runtime::generation::sampler::SpeculativeSampler;
use crate::runtime::media::input;
use crate::{LayerCachePolicy, LayerSchedule};

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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_mtp_input_with_semantics_and_options<S, F>(
        &mut self,
        drafter: &mut MlxDrafter,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        semantic: Box<dyn SpeculativeSemanticState>,
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
                let mut executor =
                    crate::architectures::gemma4::mtp::Gemma4MtpExecutor::new(target, assistant);
                crate::backend::mlx::speculative::scheduler::generate_semantic(
                    &mut executor,
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
                    crate::architectures::muse_glimmer::mtp::MuseGlimmerMtpExecutor::new(
                        target, assistant,
                    );
                crate::backend::mlx::speculative::scheduler::generate_semantic(
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

    /// Generates through embedded predictor layers on a tensor-parallel model.
    /// TP collectives and rank-synchronized sampling are
    /// derived from `execution`; EP and PP models use their architecture-erased
    /// distributed model containers.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_embedded_mtp_distributed<S: SpeculativeSampler + Clone>(
        &mut self,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        execution: &crate::MlxDistributedSession<'_>,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        let topology = execution.topology();
        if topology.pipeline_parallel_size != 1 || topology.expert_parallel_size != 1 {
            return Err(Exception::custom(
                "architecture-erased distributed MTP requires a non-pipeline, non-EP model; use PipelineModel or ExpertParallelModel for active PP/EP axes",
            ));
        }
        if self.parallel_info().map(|info| info.topology()) != Some(topology) {
            return Err(Exception::custom(
                "embedded MTP model topology does not match distributed session",
            ));
        }
        let stream = execution.stream();
        let tensor = execution
            .tensor_context()
            .map_err(|error| Exception::custom(error.to_string()))?;
        let tensor_group = tensor.group().ok_or_else(|| {
            Exception::custom("distributed embedded MTP requires an active TP subgroup")
        })?;
        let sampling_rank = topology
            .global_rank_for(crate::ParallelCoordinates {
                tensor: 0,
                pipeline: 0,
                expert: 0,
                data: topology.data_parallel_rank,
            })
            .map_err(|error| Exception::custom(error.to_string()))?;
        let mut synchronized =
            crate::backend::mlx::speculative::embedded::DistributedEmbeddedMtpSampler::new(
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
                let mut executor =
                    crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        &mut target,
                    );
                crate::backend::mlx::speculative::scheduler::generate_tokens(
                    &mut executor,
                    cache,
                    input,
                    config,
                    prng_key,
                    &mut synchronized,
                    MtpExecutionStreams::single(stream),
                    MtpSchedulerOptions::default(),
                    |_| Ok(()),
                )
            }
            (Self::Inkling(model), ModelCache::Inkling(cache)) => {
                let mut target =
                    crate::architectures::inkling::layerwise::InklingTensorMtpTarget::new(
                        model,
                        tensor_group,
                    );
                let mut executor =
                    crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        &mut target,
                    );
                crate::backend::mlx::speculative::scheduler::generate_tokens(
                    &mut executor,
                    cache,
                    input,
                    config,
                    prng_key,
                    &mut synchronized,
                    MtpExecutionStreams::single(stream),
                    MtpSchedulerOptions::default(),
                    |_| Ok(()),
                )
            }
            (Self::NemotronH(model), ModelCache::NemotronH(cache)) => {
                let mut target =
                    crate::architectures::nemotron_h::layerwise::NemotronHTensorMtpTarget::new(
                        model,
                        tensor_group,
                    );
                let mut executor =
                    crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        &mut target,
                    );
                crate::backend::mlx::speculative::scheduler::generate_tokens(
                    &mut executor,
                    cache,
                    input,
                    config,
                    prng_key,
                    &mut synchronized,
                    MtpExecutionStreams::single(stream),
                    MtpSchedulerOptions::default(),
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
                let mut executor =
                    crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        &mut target,
                    );
                crate::backend::mlx::speculative::scheduler::generate_tokens(
                    &mut executor,
                    cache,
                    input,
                    config,
                    prng_key,
                    &mut synchronized,
                    MtpExecutionStreams::single(stream),
                    MtpSchedulerOptions::default(),
                    |_| Ok(()),
                )
            }
            (model, _) => Err(Exception::custom(format!(
                "distributed embedded MTP runtime adapter is unavailable for model type {} ({:?})",
                model.model_type(),
                model.mtp_capability()
            ))),
        };
        *sampler = synchronized.into_inner();
        result
    }

    /// Generates with embedded MTP weights and a caller-provided sampler.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_embedded_mtp_input_with_sampler<S: SpeculativeSampler + Clone>(
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
    pub(crate) fn generate_embedded_mtp_input_with_sampler_callback<S, F>(
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
                let mut executor =
                    crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(target);
                crate::backend::mlx::speculative::scheduler::generate_tokens(
                    &mut executor,
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    MtpExecutionStreams::single(stream),
                    MtpSchedulerOptions::default(),
                    on_token,
                )
            }
            (Self::DeepSeekV4(target), ModelCache::DeepSeekV4(cache)) => {
                let mut executor =
                    crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        target.as_mut(),
                    );
                crate::backend::mlx::speculative::scheduler::generate_tokens(
                    &mut executor,
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    MtpExecutionStreams::single(stream),
                    MtpSchedulerOptions::default(),
                    on_token,
                )
            }
            (Self::DeepSeekV4Layerwise(target), ModelCache::DeepSeekV4(cache)) => {
                let mut executor =
                    crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        target.as_mut(),
                    );
                crate::backend::mlx::speculative::scheduler::generate_tokens(
                    &mut executor,
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    MtpExecutionStreams::single(stream),
                    MtpSchedulerOptions::default(),
                    on_token,
                )
            }
            (Self::Inkling(target), ModelCache::Inkling(cache)) => {
                let mut executor =
                    crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(target);
                crate::backend::mlx::speculative::scheduler::generate_tokens(
                    &mut executor,
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    MtpExecutionStreams::single(stream),
                    MtpSchedulerOptions::default(),
                    on_token,
                )
            }
            (Self::NemotronH(target), ModelCache::NemotronH(cache)) => {
                let mut executor =
                    crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(target);
                crate::backend::mlx::speculative::scheduler::generate_tokens(
                    &mut executor,
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    MtpExecutionStreams::single(stream),
                    MtpSchedulerOptions::default(),
                    on_token,
                )
            }
            (Self::Qwen3Next(target), ModelCache::Qwen3Next(cache))
            | (Self::Qwen35(target), ModelCache::Qwen35(cache)) => {
                let mut executor =
                    crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(target);
                crate::backend::mlx::speculative::scheduler::generate_tokens(
                    &mut executor,
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    MtpExecutionStreams::single(stream),
                    MtpSchedulerOptions::default(),
                    on_token,
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
    pub(crate) fn generate_embedded_mtp_input_with_semantics_and_options<S, F>(
        &mut self,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        semantic: Box<dyn SpeculativeSemanticState>,
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
                let mut executor =
                    crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(target);
                crate::backend::mlx::speculative::scheduler::generate_semantic(
                    &mut executor,
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    semantic,
                    cancellation,
                    MtpExecutionStreams::single(stream),
                    scheduler_options,
                    on_event,
                )
            }
            (Self::DeepSeekV4(target), ModelCache::DeepSeekV4(cache)) => {
                let mut executor =
                    crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        target.as_mut(),
                    );
                crate::backend::mlx::speculative::scheduler::generate_semantic(
                    &mut executor,
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    semantic,
                    cancellation,
                    MtpExecutionStreams::single(stream),
                    scheduler_options,
                    on_event,
                )
            }
            (Self::DeepSeekV4Layerwise(target), ModelCache::DeepSeekV4(cache)) => {
                let mut executor =
                    crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        target.as_mut(),
                    );
                crate::backend::mlx::speculative::scheduler::generate_semantic(
                    &mut executor,
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    semantic,
                    cancellation,
                    MtpExecutionStreams::single(stream),
                    scheduler_options,
                    on_event,
                )
            }
            (Self::Inkling(target), ModelCache::Inkling(cache)) => {
                let mut executor =
                    crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(target);
                crate::backend::mlx::speculative::scheduler::generate_semantic(
                    &mut executor,
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    semantic,
                    cancellation,
                    MtpExecutionStreams::single(stream),
                    scheduler_options,
                    on_event,
                )
            }
            (Self::NemotronH(target), ModelCache::NemotronH(cache)) => {
                let mut executor =
                    crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(target);
                crate::backend::mlx::speculative::scheduler::generate_semantic(
                    &mut executor,
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    semantic,
                    cancellation,
                    MtpExecutionStreams::single(stream),
                    scheduler_options,
                    on_event,
                )
            }
            (Self::Qwen3Next(target), ModelCache::Qwen3Next(cache))
            | (Self::Qwen35(target), ModelCache::Qwen35(cache)) => {
                let mut executor =
                    crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(target);
                crate::backend::mlx::speculative::scheduler::generate_semantic(
                    &mut executor,
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    semantic,
                    cancellation,
                    MtpExecutionStreams::single(stream),
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

    /// Returns each owned layer's processed-token delta relative to the
    /// persisted prefix. Ordinary decoder layers use zero; speculative layers
    /// may trail the target frontier.
    pub fn prompt_cache_layer_prefix_offsets(&self) -> Result<Vec<i32>, Exception> {
        match self {
            Self::DeepSeekV4(model) => deepseek_v4::prompt_cache_model_identity(
                &model.args,
                crate::PromptCacheTopology::default(),
            )
            .map(|identity| identity.layer_prefix_offsets)
            .map_err(|error| Exception::custom(error.to_string())),
            Self::DeepSeekV4Layerwise(model) => model
                .prompt_cache_model_identity()
                .map(|identity| identity.layer_prefix_offsets)
                .map_err(|error| Exception::custom(error.to_string())),
            _ => Ok(vec![0; self.prompt_cache_layer_layout()?.len()]),
        }
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
        crate::backend::mlx::MlxModelSession::forward_with_observer(
            self,
            input_tokens,
            mask,
            cache,
            stream,
            observer,
        )
        .map_err(|error| Exception::custom(error.to_string()))
    }

    /// Submits prompt prefill while reporting detailed activations.
    pub fn submit_prefill_with_observer(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut ModelCache,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<safemlx_lm_core::Submission<Array, crate::backend::mlx::MlxCompletion>, Error> {
        crate::backend::mlx::MlxModelSession::submit_complete_prefill_with_observer(
            self,
            input.into(),
            cache,
            stream,
            observer,
        )
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
            layer_prefix_offsets: vec![0; layer_count],
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

    /// Submits prompt prefill through the selected MLX model session.
    pub fn submit_prefill(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut ModelCache,
        stream: &Stream,
    ) -> Result<safemlx_lm_core::Submission<Array, crate::backend::mlx::MlxCompletion>, Error> {
        crate::backend::mlx::submit_prefill_with_cache(self, cache, input.into(), stream)
    }

    /// Submits cached decode through the selected MLX model session.
    pub fn submit_decode(
        &mut self,
        input: Array,
        cache: &mut ModelCache,
        stream: &Stream,
    ) -> Result<safemlx_lm_core::Submission<Array, crate::backend::mlx::MlxCompletion>, Error> {
        crate::backend::mlx::submit_decode_with_cache(self, cache, input, stream)
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

pub(crate) fn validate_gemma4_drafter(
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

//! Architecture-erased model, cache, and generation dispatch.

use std::path::Path;

use eredu_core::cache::{
    validate_prompt_cache_model_identity, PromptCacheDescriptor, PromptCacheManifest,
    PromptCacheModelIdentity, PromptCacheOptions,
};
use eredu_core::generation::{
    FinishReason, GenerationCancellationToken, MtpConfig, MtpSchedulerOptions, SemanticEvent,
};
use eredu_core::{MtpCapability, MtpCheckpointKind, MtpStats, SpeculativeSemanticState};
use safemlx::{error::Exception, Array, Stream};

use crate::backend::error::Error;
use crate::backend::runtime::generation::sampler::SpeculativeSampler;
use crate::backend::runtime::media::input;
use crate::composition::gpt_oss;
use crate::composition::mlx::speculative::{MlxDrafter, MtpExecutionStreams};
use eredu_architectures::kimi_linear;
use eredu_core::cache::LayerCachePolicy;
use eredu_core::LayerSchedule;
use eredu_runtime::ActivationObserver as RuntimeActivationObserver;
use eredu_runtime::CacheResidencyReport;
use eredu_runtime::{CacheResidencyPolicy, PagedCacheOptions};

/// Loaded model value for any architecture supported by this crate.
pub enum Model {
    /// Neutral DeepSeek-V3/V4 architecture with policy-selected residency.
    DeepSeek(Box<crate::composition::deepseek::DeepSeekModel>),
    /// Gemma 4 text and multimodal model.
    Gemma4(crate::composition::gemma4::Gemma4Model),
    /// OpenAI GPT-OSS model.
    GptOss(crate::composition::gpt_oss::GptOssModel),
    /// Moonshot Kimi Linear hybrid KDA/MLA sparse decoder.
    KimiLinear(crate::composition::kimi_linear::KimiLinearModel),
    /// Thinking Machines Lab Inkling multimodal model.
    Inkling(crate::composition::inkling::InklingModel),
    /// Llama-compatible dense model.
    Llama(crate::composition::llama::LlamaModel),
    /// Meta Muse-Glimmer dense multimodal model.
    MuseGlimmer(crate::composition::muse_glimmer::MuseGlimmerModel),
    /// Liquid AI LFM2/LFM2.5 model.
    Lfm2(crate::composition::lfm2::Lfm2Model),
    /// Nemotron-H hybrid model.
    NemotronH(crate::composition::nemotron_h::NemotronHModel),
    /// Neutral Qwen2/Qwen2.5/Qwen3/Qwen3-MoE model.
    Qwen(crate::composition::qwen::QwenModel),
    /// Qwen3-Next model.
    Qwen3Next(crate::composition::qwen::hybrid::QwenHybridModel),
    /// Qwen3-VL multimodal model.
    Qwen3Vl(crate::composition::qwen::vl::QwenVlModel),
    /// Qwen3-VL-MoE multimodal model.
    Qwen3VlMoe(crate::composition::qwen::vl::QwenVlModel),
    /// Qwen3.5 dense or MoE model, optionally multimodal.
    Qwen35(crate::composition::qwen::hybrid::QwenHybridModel),
}

impl Model {
    /// Returns architecture-neutral rank-local placement information when this
    /// model was loaded through generalized parallel execution groups.
    pub fn parallel_info(
        &self,
    ) -> Option<&eredu_runtime::ParallelModelInfo<crate::backend::MlxParallelContext>> {
        match self {
            Self::DeepSeek(_) => None,
            Self::Llama(model) => model.parallel_info(),
            Self::MuseGlimmer(model) => model.parallel_info(),
            Self::GptOss(model) => model.parallel_info(),
            Self::Qwen(model) => model.parallel_info(),
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
            Self::DeepSeek(model) if model.mtp_len() > 0 => MtpCapability::Ready {
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
            Self::Inkling(model) if model.mtp_len() > 0 => MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Embedded,
            },
            _ => MtpCapability::Unavailable,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_mtp_input_with_semantics_and_options<S, F>(
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
        match (self, cache, drafter.kind()) {
            (
                Self::Gemma4(target),
                ModelCache::Hybrid(cache),
                crate::composition::mlx::speculative::MlxDrafterKind::Gemma4Assistant,
            ) => {
                let mut executor =
                    crate::composition::mlx::speculative::external::Gemma4ExternalExecutor::new(
                        target,
                        drafter.gemma4_mut(),
                    );
                crate::composition::mlx::speculative::scheduler::generate_semantic(
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
            (
                Self::MuseGlimmer(target),
                ModelCache::MuseGlimmer(cache),
                crate::composition::mlx::speculative::MlxDrafterKind::MuseGlimmerDFlash,
            ) => {
                let mut executor = crate::composition::mlx::speculative::external::MuseGlimmerExternalExecutor::new(
                    target,
                    drafter.muse_glimmer_mut(),
                );
                crate::composition::mlx::speculative::scheduler::generate_semantic(
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
            (model, _, kind) => Err(Exception::custom(format!(
                "drafter {kind:?} has no runtime adapter for model type {} ({:?})",
                model.effective_model_type(),
                model.mtp_capability()
            ))),
        }
    }

    /// Generates through embedded predictor layers on a tensor-parallel model.
    /// TP collectives and rank-synchronized sampling are
    /// derived from `execution`; EP and PP models use their architecture-erased
    /// distributed model containers.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_embedded_mtp_distributed<S: SpeculativeSampler + Clone>(
        &mut self,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        execution: &crate::backend::MlxDistributedSession<'_>,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        let topology = execution.topology();
        if topology.pipeline_parallel_size != 1 || topology.expert_parallel_size != 1 {
            return Err(Exception::custom(
                "architecture-erased distributed MTP requires a non-distributed model; use PipelineModel for active PP/EP axes",
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
            .global_rank_for(eredu_core::ParallelCoordinates {
                tensor: 0,
                pipeline: 0,
                expert: 0,
                data: topology.data_parallel_rank,
            })
            .map_err(|error| Exception::custom(error.to_string()))?;
        let mut synchronized =
            crate::composition::mlx::speculative::embedded::DistributedEmbeddedMtpSampler::new(
                sampler.clone(),
                sampling_rank,
                execution.world(),
            )
            .map_err(|error| Exception::custom(error.to_string()))?;
        let result = match (self, cache) {
            (Self::NemotronH(model), ModelCache::Hybrid(cache)) => {
                let mut target = crate::composition::nemotron_h::NemotronHTensorMtpTarget::new(
                    model,
                    tensor_group,
                );
                let mut executor =
                    crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        &mut target,
                    );
                crate::composition::mlx::speculative::scheduler::generate_tokens(
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
                    crate::composition::inkling::InklingTensorMtpTarget::new(model, tensor_group);
                let mut executor =
                    crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        &mut target,
                    );
                crate::composition::mlx::speculative::scheduler::generate_tokens(
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
                let mut executor =
                    crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(model);
                crate::composition::mlx::speculative::scheduler::generate_tokens(
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
                model.effective_model_type(),
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
        match (self, cache) {
            (Self::DeepSeek(target), ModelCache::DeepSeek(cache)) => {
                let mut executor =
                    crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        target.as_mut(),
                    );
                crate::composition::mlx::speculative::scheduler::generate_tokens(
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
            (Self::NemotronH(target), ModelCache::Hybrid(cache)) => {
                let mut executor =
                    crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        target,
                    );
                crate::composition::mlx::speculative::scheduler::generate_tokens(
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
                    crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        target,
                    );
                crate::composition::mlx::speculative::scheduler::generate_tokens(
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
                    crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        target,
                    );
                crate::composition::mlx::speculative::scheduler::generate_tokens(
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
                model.effective_model_type(),
                model.mtp_capability()
            ))),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_embedded_mtp_input_with_semantics_and_options<S, F>(
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
            (Self::DeepSeek(target), ModelCache::DeepSeek(cache)) => {
                let mut executor =
                    crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        target.as_mut(),
                    );
                crate::composition::mlx::speculative::scheduler::generate_semantic(
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
            (Self::NemotronH(target), ModelCache::Hybrid(cache)) => {
                let mut executor =
                    crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        target,
                    );
                crate::composition::mlx::speculative::scheduler::generate_semantic(
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
                    crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        target,
                    );
                crate::composition::mlx::speculative::scheduler::generate_semantic(
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
                    crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(
                        target,
                    );
                crate::composition::mlx::speculative::scheduler::generate_semantic(
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
                model.effective_model_type(),
                model.mtp_capability()
            ))),
        }
    }

    /// Returns residency telemetry when this model uses bounded layer execution.
    pub fn residency_report(&self) -> Result<Option<eredu_runtime::ResidencyReport>, Error> {
        match self {
            Self::DeepSeek(model) => Ok(Some(model.residency_report()?)),
            Self::Gemma4(model) => model.residency_report(),
            Self::Inkling(model) => model.residency_report(),
            Self::KimiLinear(model) => Ok(Some(model.residency_report()?)),
            Self::Llama(model) => model.residency_report(),
            Self::GptOss(model) => Ok(Some(model.residency_report()?)),
            Self::Lfm2(model) => Ok(Some(model.residency_report()?)),
            Self::NemotronH(model) => Ok(Some(model.residency_report()?)),
            Self::Qwen3Next(model) | Self::Qwen35(model) => Ok(Some(model.residency_report()?)),
            Self::Qwen(model) => model.residency_report(),
            Self::MuseGlimmer(model) => model.residency_report(),
            Self::Qwen3Vl(model) | Self::Qwen3VlMoe(model) => Ok(Some(model.residency_report()?)),
        }
    }

    /// Returns experimental dense-stream telemetry when enabled.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match self {
            Self::DeepSeek(model) => model.dense_stream_report(),
            Self::Gemma4(model) => model.dense_stream_report(),
            Self::Inkling(model) => model.dense_stream_report(),
            Self::KimiLinear(model) => model.dense_stream_report(),
            Self::Llama(model) => model.dense_stream_report(),
            Self::GptOss(model) => model.dense_stream_report(),
            Self::Lfm2(model) => model.dense_stream_report(),
            Self::NemotronH(model) => model.dense_stream_report(),
            Self::Qwen3Next(model) | Self::Qwen35(model) => model.dense_stream_report(),
            Self::Qwen(model) => model.dense_stream_report(),
            Self::MuseGlimmer(model) => model.dense_stream_report(),
            Self::Qwen3Vl(model) | Self::Qwen3VlMoe(model) => model.dense_stream_report(),
        }
    }

    /// Returns sparse routed-expert cache telemetry when enabled.
    pub fn expert_cache_report(
        &self,
    ) -> Result<Option<crate::backend::runtime::residency::expert_cache::ExpertCacheReport>, Error>
    {
        match self {
            Self::DeepSeek(model) => model.expert_cache_report(),
            Self::Gemma4(model) => model.expert_cache_report(),
            Self::KimiLinear(model) => model.expert_cache_report(),
            Self::GptOss(model) => model.expert_cache_report(),
            Self::Inkling(model) => model.expert_cache_report(),
            Self::Lfm2(model) => model.expert_cache_report(),
            Self::NemotronH(model) => model.expert_cache_report(),
            Self::Qwen(model) => model.expert_cache_report(),
            Self::Qwen3Next(model) | Self::Qwen35(model) => model.expert_cache_report(),
            Self::Qwen3VlMoe(model) => model.expert_cache_report(),
            Self::MuseGlimmer(model) => model.expert_cache_report(),
            _ => Ok(None),
        }
    }

    /// Returns the canonical architecture family for this loaded model.
    pub fn model_family(&self) -> eredu_architectures::ModelKind {
        eredu_architectures::ModelKind::resolve_model_type(self.effective_model_type())
            .expect("loaded MLX model has a validated effective model type")
    }

    /// Returns the effective model type preserved from the parsed configuration.
    pub fn effective_model_type(&self) -> &str {
        match self {
            Self::DeepSeek(model) => model.model_type(),
            Self::Gemma4(model) => &model.args().model_type,
            Self::GptOss(model) => &model.args().model_type,
            Self::Inkling(model) => &model.args().model_type,
            Self::KimiLinear(model) => &model.args().model_type,
            Self::Llama(model) => &model.args().model_type,
            Self::Lfm2(model) => &model.args().model_type,
            Self::NemotronH(model) => &model.args().model_type,
            Self::Qwen(model) => &model.args().model_type,
            Self::MuseGlimmer(model) => &model.args().model_type,
            Self::Qwen3Next(model) => &model.args().model_type,
            Self::Qwen3Vl(model) => model.model_type(),
            Self::Qwen3VlMoe(model) => model.model_type(),
            Self::Qwen35(model) => &model.args().model_type,
        }
    }

    /// Returns the canonical cache-relevant architecture identity derived from the loaded model.
    pub fn prompt_cache_architecture_fingerprint(&self) -> Result<String, Exception> {
        match self {
            Self::DeepSeek(model) => model
                .prompt_cache_identity()
                .map(|identity| identity.architecture_fingerprint)
                .map_err(|error| Exception::custom(error.to_string())),
            Self::Gemma4(model) => model
                .prompt_cache_model_identity()
                .map(|identity| identity.architecture_fingerprint)
                .map_err(|error| Exception::custom(error.to_string())),
            Self::Llama(model) => {
                Ok(eredu_architectures::llama::prompt_cache_architecture_fingerprint(model.args()))
            }
            Self::GptOss(model) => Ok(model.prompt_cache_architecture_fingerprint()),
            Self::Inkling(model) => Ok(model.args().architecture_fingerprint()),
            Self::KimiLinear(model) => Ok(kimi_linear::prompt_cache_architecture_fingerprint(
                model.args(),
            )),
            Self::Lfm2(model) => model
                .prompt_cache_architecture_fingerprint()
                .map_err(|error| Exception::custom(error.to_string())),
            Self::Qwen(model) => Ok(model.prompt_cache_architecture_fingerprint()),
            Self::MuseGlimmer(model) => model
                .prompt_cache_model_identity()
                .map(|identity| identity.architecture_fingerprint)
                .map_err(|error| Exception::custom(error.to_string())),
            Self::NemotronH(model) => model
                .prompt_cache_architecture_fingerprint()
                .map_err(|error| Exception::custom(error.to_string())),
            Self::Qwen3Next(model) | Self::Qwen35(model) => {
                Ok(model.prompt_cache_architecture_fingerprint())
            }
            Self::Qwen3Vl(model) | Self::Qwen3VlMoe(model) => {
                Ok(model.prompt_cache_architecture_fingerprint())
            }
        }
    }

    /// Returns the exact ordered prompt-cache state and attention layout.
    pub fn prompt_cache_layer_layout(&self) -> Result<LayerSchedule<LayerCachePolicy>, Exception> {
        match self {
            Self::DeepSeek(model) => model
                .prompt_cache_identity()
                .map(|identity| identity.layer_layout),
            Self::Llama(model) => model.prompt_cache_layer_layout(),
            Self::GptOss(model) => model.prompt_cache_layer_layout(),
            Self::Qwen(model) => model.prompt_cache_layer_layout(),
            Self::MuseGlimmer(model) => model
                .prompt_cache_model_identity()
                .map(|identity| identity.layer_layout),
            Self::KimiLinear(model) => model.prompt_cache_layer_layout(),
            Self::Lfm2(model) => model.prompt_cache_layer_layout(),
            Self::NemotronH(model) => model.prompt_cache_layer_layout(),
            Self::Qwen3Next(model) | Self::Qwen35(model) => model.prompt_cache_layer_layout(),
            Self::Qwen3Vl(model) | Self::Qwen3VlMoe(model) => model.prompt_cache_layer_layout(),
            Self::Gemma4(model) => model
                .prompt_cache_model_identity()
                .map(|identity| identity.layer_layout),
            Self::Inkling(model) => model.prompt_cache_layer_layout(),
        }
        .map_err(|error| Exception::custom(error.to_string()))
    }

    /// Returns each owned layer's processed-token delta relative to the
    /// persisted prefix. Ordinary decoder layers use zero; speculative layers
    /// may trail the target frontier.
    pub fn prompt_cache_layer_prefix_offsets(&self) -> Result<Vec<i32>, Exception> {
        match self {
            Self::DeepSeek(model) => model
                .prompt_cache_identity()
                .map(|identity| identity.layer_prefix_offsets)
                .map_err(|error| Exception::custom(error.to_string())),
            Self::Gemma4(model) => model
                .prompt_cache_model_identity()
                .map(|identity| identity.layer_prefix_offsets)
                .map_err(|error| Exception::custom(error.to_string())),
            Self::MuseGlimmer(model) => model
                .prompt_cache_model_identity()
                .map(|identity| identity.layer_prefix_offsets)
                .map_err(|error| Exception::custom(error.to_string())),
            Self::Inkling(model) => model
                .prompt_cache_layer_prefix_offsets()
                .map_err(|error| Exception::custom(error.to_string())),
            Self::NemotronH(model) => model
                .prompt_cache_model_identity()
                .map(|identity| identity.layer_prefix_offsets)
                .map_err(|error| Exception::custom(error.to_string())),
            Self::Qwen3Next(model) | Self::Qwen35(model) => model
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
        observer: &mut impl RuntimeActivationObserver<crate::MlxTensor, Exception>,
    ) -> Result<Array, Exception> {
        crate::composition::mlx::MlxModelSession::forward_with_observer(
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
        observer: &mut impl RuntimeActivationObserver<crate::MlxTensor, Exception>,
    ) -> Result<eredu_core::Submission<Array, crate::backend::MlxCompletion>, Error> {
        crate::composition::mlx::MlxModelSession::submit_complete_prefill_with_observer(
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
            Self::DeepSeek(model) => ModelCache::DeepSeek(
                model
                    .new_state()
                    .expect("validated DeepSeek state geometry"),
            ),
            Self::Gemma4(model) => ModelCache::Hybrid(model.new_cache()),
            Self::GptOss(model) => ModelCache::GptOss(model.new_cache()),
            Self::Inkling(model) => ModelCache::Inkling(model.new_cache()),
            Self::KimiLinear(model) => ModelCache::Hybrid(model.new_cache()),
            Self::Llama(model) => ModelCache::Llama(model.new_cache()),
            Self::Lfm2(model) => ModelCache::Hybrid(model.new_cache()),
            Self::Qwen(model) => ModelCache::Qwen(model.new_cache()),
            Self::MuseGlimmer(model) => ModelCache::MuseGlimmer(model.new_cache()),
            Self::Qwen3Next(model) => ModelCache::Qwen3Next(model.new_cache()),
            Self::Qwen3Vl(model) => ModelCache::Qwen3Vl(model.new_cache()),
            Self::Qwen3VlMoe(model) => ModelCache::Qwen3VlMoe(model.new_cache()),
            Self::NemotronH(model) => ModelCache::Hybrid(model.new_cache()),
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
                Self::DeepSeek(model) => model
                    .new_state_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::DeepSeek)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::Llama(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::Llama)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::KimiLinear(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::Hybrid)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::GptOss(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::GptOss)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::Qwen(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::Qwen)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::MuseGlimmer(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::MuseGlimmer)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::Inkling(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::Inkling)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::Gemma4(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::Hybrid)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::NemotronH(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::Hybrid)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::Lfm2(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::Hybrid)
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
                    self.effective_model_type()
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
            Self::DeepSeek(model) => model
                .load_prompt_cache(directory, expected, prefix_token_ids, options, stream)
                .map(|(state, manifest)| (ModelCache::DeepSeek(state), manifest))
                .map_err(|error| Exception::custom(error.to_string())),
            Self::Llama(model) => load!(model, ModelCache::Llama),
            Self::GptOss(model) => load!(model, ModelCache::GptOss),
            Self::Qwen(model) => load!(model, ModelCache::Qwen),
            Self::MuseGlimmer(model) => load!(model, ModelCache::MuseGlimmer),
            Self::KimiLinear(model) => load!(model, ModelCache::Hybrid),
            Self::Qwen3Next(model) => load!(model, ModelCache::Qwen3Next),
            Self::Qwen35(model) => load!(model, ModelCache::Qwen35),
            Self::Qwen3Vl(model) => load!(model, ModelCache::Qwen3Vl),
            Self::Qwen3VlMoe(model) => load!(model, ModelCache::Qwen3VlMoe),
            Self::Gemma4(model) => load!(model, ModelCache::Hybrid),
            Self::Inkling(model) => load!(model, ModelCache::Inkling),
            Self::Lfm2(model) => load!(model, ModelCache::Hybrid),
            Self::NemotronH(model) => load!(model, ModelCache::Hybrid),
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
            (Self::DeepSeek(model), ModelCache::DeepSeek(state)) => {
                return model
                    .save_prompt_cache(state, &destination, descriptor, prefix_token_ids, options)
                    .map_err(|error| Exception::custom(error.to_string()));
            }
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
            (Self::Qwen(model), ModelCache::Qwen(cache)) => {
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
            (Self::MuseGlimmer(model), ModelCache::MuseGlimmer(cache)) => {
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
            (Self::KimiLinear(model), ModelCache::Hybrid(cache)) => {
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
            (Self::Lfm2(model), ModelCache::Hybrid(cache)) => {
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
            (Self::NemotronH(model), ModelCache::Hybrid(cache)) => {
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
            (Self::Gemma4(model), ModelCache::Hybrid(cache)) => {
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
            Self::DeepSeek(model) => model.model_type(),
            Self::Llama(_) => "llama",
            Self::GptOss(_) => "gpt_oss",
            Self::Qwen(_) => "qwen",
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
            effective_model_type: self.effective_model_type().into(),
            architecture_fingerprint: self.prompt_cache_architecture_fingerprint()?,
            layer_count,
            global_layer_start: 0,
            global_layer_end: layer_count,
            sink_tokens: 0,
            layer_prefix_offsets: self.prompt_cache_layer_prefix_offsets()?,
            topology: Default::default(),
            layer_layout,
        };
        validate_prompt_cache_model_identity(&descriptor, &identity)
            .map_err(|error| Exception::custom(error.to_string()))?;
        Err(Exception::custom(
            "model and cache representations do not match for prompt-cache publication",
        ))
    }

    /// Submits prompt prefill through the selected MLX model session.
    pub fn submit_prefill(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut ModelCache,
        stream: &Stream,
    ) -> Result<eredu_core::Submission<Array, crate::backend::MlxCompletion>, Error> {
        crate::composition::mlx::submit_prefill_with_cache(self, cache, input.into(), stream)
    }

    /// Submits cached decode through the selected MLX model session.
    pub fn submit_decode(
        &mut self,
        input: Array,
        cache: &mut ModelCache,
        stream: &Stream,
    ) -> Result<eredu_core::Submission<Array, crate::backend::MlxCompletion>, Error> {
        crate::composition::mlx::submit_decode_with_cache(self, cache, input, stream)
    }
}

/// Cache value matching a [`Model`] variant.
#[derive(Clone)]
pub enum ModelCache {
    /// Architecture-declared DeepSeek state.
    DeepSeek(crate::composition::deepseek::DeepSeekState),
    /// GPT-OSS cache following its canonical per-layer attention schedule.
    GptOss(gpt_oss::Cache),
    /// Neutral Muse-Glimmer key/value state.
    MuseGlimmer(crate::backend::runtime::cache::state::MlxKeyValueState),
    /// Runtime-policy-selected MLX key/value state.
    Llama(crate::backend::runtime::cache::state::MlxKeyValueState),
    /// Runtime-policy-selected Qwen key/value state.
    Qwen(crate::backend::runtime::cache::state::MlxKeyValueState),
    /// Qwen3-VL key/value cache and multimodal position state.
    Qwen3Vl(crate::backend::runtime::cache::state::MlxHybridState),
    /// Qwen3-VL-MoE key/value cache and multimodal position state.
    Qwen3VlMoe(crate::backend::runtime::cache::state::MlxHybridState),
    /// Architecture-declared heterogeneous append-only and fixed state.
    Hybrid(crate::backend::runtime::cache::state::MlxHybridState),
    /// Neutral Inkling target and checkpoint-embedded predictor state.
    Inkling(crate::composition::inkling::InklingState),
    /// Heterogeneous Qwen3.5 MoE cache.
    Qwen35(crate::backend::runtime::cache::state::MlxHybridState),
    /// Heterogeneous Qwen3-Next cache.
    Qwen3Next(crate::backend::runtime::cache::state::MlxHybridState),
}

impl ModelCache {
    /// Returns aggregate cache-residency telemetry when paging is active.
    pub fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        match self {
            Self::DeepSeek(cache) => cache.residency_report(),
            Self::GptOss(cache) => cache.residency_report(),
            Self::Inkling(cache) => cache.target().residency_report(),
            Self::MuseGlimmer(cache) | Self::Llama(cache) | Self::Qwen(cache) => {
                cache.residency_report()
            }
            Self::Qwen3Vl(cache)
            | Self::Qwen3VlMoe(cache)
            | Self::Hybrid(cache)
            | Self::Qwen35(cache)
            | Self::Qwen3Next(cache) => cache.residency_report(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::runtime::cache::{
        residency::CacheResidencyManager,
        state::{MlxHybridState, MlxKeyValueState},
    };
    use eredu_core::{cache::LayerCachePolicy, AttentionPolicy};
    use eredu_runtime::StateLayout;

    fn paged_state_layout() -> StateLayout {
        StateLayout::new(
            LayerSchedule::new(
                1,
                vec![LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 8).unwrap()],
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn cache_residency_manager() -> CacheResidencyManager {
        CacheResidencyManager::new(
            PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
                .unwrap()
                .with_full_attention(true),
        )
        .unwrap()
    }

    #[test]
    fn qwen_cache_variants_forward_paged_residency_reports() {
        let layout = paged_state_layout();
        let manager = cache_residency_manager();
        let caches = [
            (
                "Qwen",
                ModelCache::Qwen(
                    MlxKeyValueState::paged(layout.clone(), manager.clone(), None).unwrap(),
                ),
            ),
            (
                "Qwen3-Next",
                ModelCache::Qwen3Next(
                    MlxHybridState::paged(layout.clone(), manager.clone(), None).unwrap(),
                ),
            ),
            (
                "Qwen3.5",
                ModelCache::Qwen35(MlxHybridState::paged(layout, manager, None).unwrap()),
            ),
        ];

        for (family, cache) in caches {
            assert!(
                cache.residency_report().unwrap().is_some(),
                "{family} paged-cache telemetry should be available"
            );
        }
    }
}

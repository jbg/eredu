//! Reusable expert-parallel assignment, routing, and exchange infrastructure.
//!
//! Pure expert parallelism keeps ordinary model state replicated and partitions
//! only routed expert banks. [`crate::runtime::distributed::expert::dispatch_replicated`]
//! exploits the replicated
//! token layout: ranks compact only routes owned by their experts and all-sum
//! the resulting token buffer. [`crate::runtime::distributed::expert::all_to_all_v`]
//! is the general sharded-token
//! transport.  It is intentionally an all-gather fallback and therefore uses
//! `O(group_size)` temporary replication until MLX exposes native all-to-all.

use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use safemlx::{
    distributed::{self, Group},
    error::Exception,
    module::{ModuleParameters, Param},
    ops::{indexing::TryIndexOp, zeros_dtype, GgufCheckpoint, GgufMetadataValue},
    transforms::eval,
    Array, Stream,
};

use crate::{
    api::{
        deepseek_v3, gemma4, gpt_oss, inkling, input as runtime_input, kimi_linear, lfm2,
        nemotron_h, qwen3_5, qwen3_next, qwen3_vl, ModelKind, ModelLoadOptions,
    },
    architectures::distributed::pipeline::{assign_module, load_deepseek_experts},
    error::Error,
    runtime::cache::residency::{
        open_prompt_cache, validate_prompt_cache_model_identity, CacheRankIdentity,
        CacheResidencyManager, CacheResidencyPolicy, CacheResidencyReport, PagedCacheOptions,
        PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
        PromptCacheTopology,
    },
    runtime::cache::{
        ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache, SlidingKeyValueCache,
    },
    runtime::checkpoint::load::{transform_split_swiglu_experts, StrictLoadConfig},
    runtime::checkpoint::quantization::{should_quantize_on_load, WeightQuantization},
    runtime::checkpoint::store::{GgufWeightStore, SafetensorsWeightStore, WeightStore},
    runtime::distributed::parallel::{
        sample_and_synchronize, ParallelBuildContext, ShardingPolicy, SynchronizedToken,
    },
    runtime::distributed::topology::{
        load_partition_from_store_on_streams, ParallelTopology, PlacementPlan, TensorPlacement,
    },
    runtime::execution::inspection::ActivationObserver,
    runtime::generation::sampler::{DefaultSampler, Sampler, SpeculativeSampler},
    runtime::generation::speculative::{MtpCapability, MtpCheckpointKind, MtpConfig, MtpStats},
    runtime::residency::expert_cache::{
        AcquiredExperts, ExpertCache, ExpertCacheError, ExpertCacheLoadOptions, ExpertCacheReport,
        ExpertCatalogEntry, ExpertPass, ExpertRouteBatch,
    },
};

use crate::runtime::execution::layerwise::LayerWeightResidency;
#[cfg(test)]
use crate::runtime::execution::layerwise::WeightResidency;

use crate::{
    architectures::{deepseek_v3::model::RoutedExperts, qwen::dense as dense_qwen},
    nn::moe::{quantize_expert_bank, PackedSwiGluExperts},
};

pub use crate::runtime::distributed::expert::*;

impl LocalExpertBank for RoutedExperts {
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        local_expert_ids: &Array,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let ids = local_expert_ids.reshape(&[-1, 1], stream)?;
        let weights = safemlx::ops::ones_dtype(&[hidden.dim(0), 1], hidden.dtype(), stream)?;
        Ok(self.forward_local(hidden, &ids, &weights, stream)?)
    }
}

impl LocalExpertBank for gpt_oss::Experts {
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        local_expert_ids: &Array,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let ids = local_expert_ids.reshape(&[-1, 1], stream)?;
        let weights = unit_route_weights(hidden.dim(0), hidden.dtype(), stream)?;
        Ok(self.forward(hidden, &ids, &weights, stream)?)
    }
}

impl LocalExpertBank for nemotron_h::Experts {
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        local_expert_ids: &Array,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let ids = local_expert_ids.reshape(&[-1, 1], stream)?;
        let weights = unit_route_weights(hidden.dim(0), hidden.dtype(), stream)?;
        Ok(self.forward(hidden, &ids, &weights, stream)?)
    }
}

impl LocalExpertBank for qwen3_5::Experts {
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        local_expert_ids: &Array,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let ids = local_expert_ids.reshape(&[-1, 1], stream)?;
        let weights = unit_route_weights(hidden.dim(0), hidden.dtype(), stream)?;
        Ok(self.forward_chunked(hidden, &ids, &weights, stream)?)
    }
}

impl LocalExpertBank for gemma4::GemmaExperts {
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        local_expert_ids: &Array,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let ids = local_expert_ids.reshape(&[-1, 1], stream)?;
        let weights = unit_route_weights(hidden.dim(0), hidden.dtype(), stream)?;
        Ok(self.forward(hidden, &ids, &weights, stream)?)
    }
}

/// Physical residency of the routed experts owned by this EP rank.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum RoutedExpertResidency {
    /// Every owned expert is materialized and pinned on the execution device.
    FullyResident,
    /// Owned experts are materialized on demand under explicit cache budgets.
    SparseCache,
}

/// Immutable description of a rank-local expert-parallel model.
#[derive(Debug, Clone)]
pub struct ExpertParallelInfo {
    /// Complete Cartesian topology and local TP/PP/EP coordinates.
    pub topology: ParallelTopology,
    /// Global rank.
    pub global_rank: usize,
    /// Rank in the EP group.
    pub expert_parallel_rank: usize,
    /// EP group size.
    pub expert_parallel_size: usize,
    /// Loaded architecture.
    pub model_kind: ModelKind,
    /// Assignment metadata.
    pub assignment: ExpertAssignment,
    /// Bytes in all locally materialized parameters.
    pub local_parameter_bytes: usize,
    /// Bytes in local routed-expert tensors.
    pub routed_expert_bytes: usize,
    /// Bytes in all cold, warm, or hot routed experts owned by this rank.
    pub owned_expert_bytes: usize,
    /// Physical residency policy for rank-owned routed experts.
    pub routed_expert_residency: RoutedExpertResidency,
    /// Bytes in replicated tensors.
    pub replicated_parameter_bytes: usize,
    /// Checkpoint shards opened by this rank.
    pub opened_checkpoint_shards: Vec<PathBuf>,
    /// Active route transport.
    pub exchange_strategy: ExpertExchangeStrategy,
}

/// Architecture-checked replicated attention cache used by an EP model.
#[derive(Debug, Clone)]
pub enum ExpertParallelCache {
    /// DeepSeek compressed-latent attention cache.
    DeepSeek(deepseek_v3::Cache),
    /// Kimi Linear heterogeneous KDA/MLA cache.
    KimiLinear(kimi_linear::Cache),
    /// Dense-Qwen standard key/value cache.
    DenseQwen(Vec<Option<ConcatKeyValueCache>>),
    /// Dense-Qwen bounded sliding-window key/value cache.
    DenseQwenSliding(Vec<Option<SlidingKeyValueCache>>),
    /// Dense-Qwen globally budgeted paged key/value cache.
    DenseQwenPaged(Vec<Option<PagedKeyValueCache>>),
    /// GPT-OSS cache following its canonical per-layer attention schedule.
    GptOss(gpt_oss::Cache),
    /// Inkling attention and convolution cache.
    Inkling(inkling::Cache),
    /// LFM2 heterogeneous attention/convolution cache.
    Lfm2(lfm2::Cache),
    /// Nemotron-H heterogeneous recurrent/attention cache.
    NemotronH(nemotron_h::Cache),
    /// Qwen3-Next/Qwen3.5 heterogeneous attention cache.
    QwenHybrid(qwen3_5::Cache),
    /// Qwen3-VL-MoE multimodal-RoPE text cache.
    Qwen3Vl(qwen3_vl::Cache),
    /// Gemma 4 alternating/shared key/value state.
    Gemma4(gemma4::Cache),
}

impl ExpertParallelCache {
    /// Clears all cached attention state.
    pub fn reset(&mut self) -> Result<(), Error> {
        match self {
            Self::DeepSeek(cache) => {
                for cache in &mut cache.layers {
                    cache.clear()?;
                }
            }
            Self::KimiLinear(cache) => cache.reset()?,
            Self::DenseQwen(cache) => cache
                .iter_mut()
                .flatten()
                .for_each(ConcatKeyValueCache::clear),
            Self::DenseQwenSliding(cache) => cache
                .iter_mut()
                .flatten()
                .for_each(SlidingKeyValueCache::clear),
            Self::DenseQwenPaged(caches) => {
                if let Some(first) = caches.iter().flatten().next() {
                    first
                        .manager()
                        .clear()
                        .map_err(|error| Error::Parallel(error.to_string()))?;
                }
                for cache in caches.iter_mut().flatten() {
                    cache.reset_local_after_manager_clear();
                }
            }
            Self::GptOss(cache) => cache.reset()?,
            Self::Inkling(cache) => cache.reset()?,
            Self::Lfm2(cache) => cache.reset(),
            Self::NemotronH(cache) => cache.reset()?,
            Self::QwenHybrid(cache) => cache.reset(),
            Self::Qwen3Vl(cache) => *cache = qwen3_vl::Cache::default(),
            Self::Gemma4(cache) => cache.clear()?,
        }
        Ok(())
    }

    /// Returns the common replicated cache offset.
    pub fn offset(&self) -> i32 {
        match self {
            Self::DeepSeek(cache) => cache.offset(),
            Self::KimiLinear(cache) => cache.offset(),
            Self::DenseQwen(cache) => cache
                .first()
                .and_then(Option::as_ref)
                .map_or(0, KeyValueCache::offset),
            Self::DenseQwenSliding(cache) => cache
                .first()
                .and_then(Option::as_ref)
                .map_or(0, KeyValueCache::offset),
            Self::DenseQwenPaged(cache) => cache
                .first()
                .and_then(Option::as_ref)
                .map_or(0, KeyValueCache::offset),
            Self::GptOss(cache) => cache.offset(),
            Self::Inkling(cache) => cache.offset(),
            Self::Lfm2(cache) => cache.offset(),
            Self::NemotronH(cache) => cache.offset(),
            Self::QwenHybrid(cache) => cache.offset(),
            Self::Qwen3Vl(cache) => cache
                .kv
                .first()
                .and_then(Option::as_ref)
                .map_or(0, KeyValueCache::offset),
            Self::Gemma4(cache) => cache.offset(),
        }
    }
}

#[allow(dead_code)]
enum ExpertArchitecture {
    DeepSeek(Box<deepseek_v3::Model>),
    DeepSeekLayerwise(Box<crate::architectures::deepseek_v3::layerwise::DeepSeekV3LayerwiseModel>),
    KimiLinear(Box<kimi_linear::Model>),
    KimiLinearLayerwise(
        Box<crate::architectures::kimi_linear::layerwise::KimiLinearLayerwiseModel>,
    ),
    DenseQwen(Box<dense_qwen::Model>),
    DenseQwenLayerwise(Box<crate::architectures::qwen::dense::layerwise::LayerwiseDecoder>),
    GptOss(Box<gpt_oss::Model>),
    GptOssLayerwise(Box<crate::architectures::gpt_oss::layerwise::GptOssLayerwiseModel>),
    Inkling(Box<inkling::Model>),
    InklingLayerwise(Box<crate::architectures::inkling::layerwise::InklingLayerwiseModel>),
    Lfm2(Box<lfm2::Model>),
    Lfm2Layerwise(Box<crate::architectures::lfm2::layerwise::Lfm2LayerwiseModel>),
    NemotronH(Box<nemotron_h::Model>),
    NemotronHLayerwise(Box<crate::architectures::nemotron_h::layerwise::NemotronHLayerwiseModel>),
    QwenHybrid(Box<qwen3_5::Model>),
    QwenHybridLayerwise(
        Box<crate::architectures::qwen::hybrid::layerwise::QwenHybridLayerwiseModel>,
    ),
    Qwen3Vl(Box<qwen3_vl::Model>),
    Qwen3VlLayerwise(Box<crate::architectures::qwen::vl::layerwise::Qwen3VlLayerwiseModel>),
    Gemma4Layerwise(Box<crate::architectures::gemma4::layerwise::Gemma4LayerwiseModel>),
}

impl ExpertArchitecture {
    fn bind_parallel_topology(&mut self, topology: ParallelTopology) {
        match self {
            Self::DeepSeekLayerwise(model) => model.bind_parallel_topology(topology),
            Self::KimiLinearLayerwise(model) => model.bind_parallel_topology(topology),
            Self::DenseQwenLayerwise(model) => model.bind_parallel_topology(topology),
            Self::GptOssLayerwise(model) => model.bind_parallel_topology(topology),
            Self::InklingLayerwise(model) => model.bind_parallel_topology(topology),
            Self::Lfm2Layerwise(model) => model.bind_parallel_topology(topology),
            Self::NemotronHLayerwise(model) => model.bind_parallel_topology(topology),
            Self::QwenHybridLayerwise(model) => model.bind_parallel_topology(topology),
            Self::Qwen3VlLayerwise(model) => model.bind_parallel_topology(topology),
            Self::Gemma4Layerwise(model) => model.bind_parallel_topology(topology),
            _ => {}
        }
    }
}

/// Executable rank-local EP or TP+EP model.
pub struct ExpertParallelModel {
    topology: ParallelTopology,
    info: ExpertParallelInfo,
    architecture: ExpertArchitecture,
    expert_cache: Option<ExpertCache>,
    latest_statistics: RoutingStatistics,
    cumulative_statistics: RoutingStatistics,
}

struct ExpertParallelQwenMtpTarget<'a> {
    model: &'a mut ExpertParallelModel,
    group: &'a Group,
}

#[derive(Clone)]
struct ExpertParallelSpeculativeSampler<'a, S> {
    sampler: S,
    sampling_rank: usize,
    group: &'a Group,
}

impl<S: SpeculativeSampler> SpeculativeSampler for ExpertParallelSpeculativeSampler<'_, S> {
    fn supports_exact_optimistic_promotion(&self) -> bool {
        false
    }

    fn grammar_is_complete(&mut self) -> Result<bool, Exception> {
        self.sampler.grammar_is_complete()
    }

    fn prefix_is_complete(&self, history: &[u32]) -> Result<bool, Exception> {
        self.sampler.prefix_is_complete(history)
    }

    fn process_logits(
        &mut self,
        logits: &Array,
        temperature: f32,
        history: &[u32],
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.sampler
            .process_logits(logits, temperature, history, stream)
    }

    fn sample_processed(
        &self,
        logits: &Array,
        temperature: f32,
        prng_state: Option<&mut safemlx::random::RandomState>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        // Sample on every rank so identical PRNG states remain aligned, then
        // retain only the designated rank's choice before synchronizing it.
        let sampled = self
            .sampler
            .sample_processed(logits, temperature, prng_state, stream)?;
        let selected = if self.group.rank() == self.sampling_rank {
            sampled
        } else {
            zeros_dtype(sampled.shape(), sampled.dtype(), stream)?
        };
        distributed::all_sum(&selected, self.group, stream)
    }

    fn commit_token(
        &mut self,
        processed_logits: &Array,
        token: u32,
        stream: &Stream,
    ) -> Result<(), Exception> {
        self.sampler.commit_token(processed_logits, token, stream)
    }
}

impl std::fmt::Debug for ExpertParallelModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExpertParallelModel")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

impl crate::architectures::qwen::hybrid::mtp::QwenMtpTarget for ExpertParallelQwenMtpTarget<'_> {
    fn prefill_mtp_target(
        &mut self,
        input: runtime_input::ModelInput<'_>,
        cache: &mut qwen3_5::Cache,
        stream: &Stream,
    ) -> Result<qwen3_5::QwenMtpStepOutput, Exception> {
        let tokens = runtime_input::text_token_ids(input, stream)?;
        cache.reset();
        self.model
            .forward_qwen_mtp_target(&tokens, cache, self.group, stream)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn verify_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut qwen3_5::Cache,
        stream: &Stream,
    ) -> Result<qwen3_5::QwenMtpStepOutput, Exception> {
        self.model
            .forward_qwen_mtp_target(tokens, cache, self.group, stream)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn forward_mtp_drafter(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut [qwen3_5::LayerCache],
        stream: &Stream,
    ) -> Result<Array, Exception> {
        match &mut self.model.architecture {
            ExpertArchitecture::QwenHybrid(model) => {
                model.forward_mtp_head(hidden, tokens, cache, stream)
            }
            ExpertArchitecture::QwenHybridLayerwise(model) => {
                model.forward_mtp_head(hidden, tokens, cache, stream)
            }
            _ => Err(Exception::custom(
                "embedded Qwen MTP requires a Qwen3-Next or Qwen3.5 EP model",
            )),
        }
    }

    fn mtp_layer_count(&self) -> usize {
        match &self.model.architecture {
            ExpertArchitecture::QwenHybrid(model) => model.mtp_len(),
            ExpertArchitecture::QwenHybridLayerwise(model) => model.mtp_len(),
            _ => 0,
        }
    }
}

impl ExpertParallelModel {
    /// Returns placement, assignment, and memory diagnostics.
    pub fn info(&self) -> &ExpertParallelInfo {
        &self.info
    }

    /// Reports whether this EP target can perform embedded MTP generation.
    pub fn mtp_capability(&self) -> MtpCapability {
        match &self.architecture {
            ExpertArchitecture::QwenHybrid(model) if model.mtp_len() > 0 => MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Embedded,
            },
            ExpertArchitecture::QwenHybridLayerwise(model) if model.mtp_len() > 0 => {
                MtpCapability::Ready {
                    checkpoint: MtpCheckpointKind::Embedded,
                }
            }
            _ => MtpCapability::Unavailable,
        }
    }

    /// Returns rank-owned expert residency for sparse-cached and externally
    /// pinned fully resident execution.
    pub fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        self.expert_cache
            .as_ref()
            .map(ExpertCache::report)
            .transpose()
            .map_err(Error::from)
    }

    /// Returns dense nonexpert streaming telemetry when that EP residency
    /// policy is active.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<crate::runtime::execution::layerwise::DenseDiskStreamReport>, Error> {
        match &self.architecture {
            ExpertArchitecture::DeepSeekLayerwise(model) => model.dense_stream_report(),
            ExpertArchitecture::KimiLinearLayerwise(model) => model.dense_stream_report(),
            ExpertArchitecture::DenseQwenLayerwise(model) => model.dense_stream_report(),
            ExpertArchitecture::GptOssLayerwise(model) => model.dense_stream_report(),
            ExpertArchitecture::InklingLayerwise(model) => model.dense_stream_report(),
            ExpertArchitecture::Lfm2Layerwise(model) => model.dense_stream_report(),
            ExpertArchitecture::NemotronHLayerwise(model) => model.dense_stream_report(),
            ExpertArchitecture::QwenHybridLayerwise(model) => model.dense_stream_report(),
            ExpertArchitecture::Qwen3VlLayerwise(model) => model.dense_stream_report(),
            ExpertArchitecture::Gemma4Layerwise(model) => model.dense_stream_report(),
            _ => Ok(None),
        }
    }

    /// Allocates an empty architecture-appropriate replicated cache.
    pub fn new_cache(&self) -> ExpertParallelCache {
        match &self.architecture {
            ExpertArchitecture::DeepSeek(model) => ExpertParallelCache::DeepSeek(model.new_cache()),
            ExpertArchitecture::DeepSeekLayerwise(model) => {
                ExpertParallelCache::DeepSeek(model.new_cache())
            }
            ExpertArchitecture::KimiLinear(model) => {
                ExpertParallelCache::KimiLinear(model.new_cache())
            }
            ExpertArchitecture::KimiLinearLayerwise(model) => {
                ExpertParallelCache::KimiLinear(model.new_cache())
            }
            ExpertArchitecture::DenseQwen(_) => ExpertParallelCache::DenseQwen(Vec::new()),
            ExpertArchitecture::DenseQwenLayerwise(model) => {
                ExpertParallelCache::DenseQwen(model.new_cache())
            }
            ExpertArchitecture::GptOss(model) => ExpertParallelCache::GptOss(model.new_cache()),
            ExpertArchitecture::GptOssLayerwise(model) => {
                ExpertParallelCache::GptOss(model.new_cache())
            }
            ExpertArchitecture::Inkling(model) => ExpertParallelCache::Inkling(model.new_cache()),
            ExpertArchitecture::InklingLayerwise(model) => {
                ExpertParallelCache::Inkling(model.new_cache())
            }
            ExpertArchitecture::Lfm2(model) => ExpertParallelCache::Lfm2(model.new_cache()),
            ExpertArchitecture::Lfm2Layerwise(model) => {
                ExpertParallelCache::Lfm2(model.new_cache())
            }
            ExpertArchitecture::NemotronH(model) => {
                ExpertParallelCache::NemotronH(model.new_cache())
            }
            ExpertArchitecture::NemotronHLayerwise(model) => {
                ExpertParallelCache::NemotronH(model.new_cache())
            }
            ExpertArchitecture::QwenHybrid(model) => {
                ExpertParallelCache::QwenHybrid(model.new_cache())
            }
            ExpertArchitecture::QwenHybridLayerwise(model) => {
                ExpertParallelCache::QwenHybrid(model.new_cache())
            }
            ExpertArchitecture::Qwen3Vl(model) => ExpertParallelCache::Qwen3Vl(model.new_cache()),
            ExpertArchitecture::Qwen3VlLayerwise(model) => {
                ExpertParallelCache::Qwen3Vl(model.new_cache())
            }
            ExpertArchitecture::Gemma4Layerwise(model) => {
                ExpertParallelCache::Gemma4(model.new_cache())
            }
        }
    }

    /// Allocates replicated attention state under an explicit cache policy.
    ///
    /// DeepSeek compressed attention, GPT-OSS scheduled attention, Qwen3 KV,
    /// Inkling relative-position attention, and Nemotron-H attention are
    /// supported. Bounded Inkling convolution and Nemotron-H Mamba state remain
    /// resident alongside the paged attention blocks.
    pub fn new_cache_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<ExpertParallelCache, Error> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => match &self.architecture {
                ExpertArchitecture::DeepSeek(model) => {
                    let manager = CacheResidencyManager::new(options)
                        .map_err(|error| Error::Parallel(error.to_string()))?;
                    let rank = CacheRankIdentity {
                        pipeline_rank: None,
                        tensor_parallel_rank: (self.topology.tensor_parallel_size > 1)
                            .then_some(self.topology.tensor_parallel_rank),
                        expert_parallel_rank: Some(self.topology.expert_parallel_rank),
                    };
                    model
                        .new_cache_with_manager(manager, Some(rank))
                        .map(ExpertParallelCache::DeepSeek)
                        .map_err(Into::into)
                }
                ExpertArchitecture::DeepSeekLayerwise(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ExpertParallelCache::DeepSeek),
                ExpertArchitecture::KimiLinear(model) => {
                    let rank = Some(CacheRankIdentity {
                        pipeline_rank: None,
                        tensor_parallel_rank: (self.topology.tensor_parallel_size > 1)
                            .then_some(self.topology.tensor_parallel_rank),
                        expert_parallel_rank: Some(self.topology.expert_parallel_rank),
                    });
                    kimi_linear::Cache::new_with_options_and_rank(
                        &model.args,
                        CacheResidencyPolicy::Paged(options),
                        rank,
                    )
                    .map(ExpertParallelCache::KimiLinear)
                    .map_err(Into::into)
                }
                ExpertArchitecture::KimiLinearLayerwise(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ExpertParallelCache::KimiLinear),
                ExpertArchitecture::GptOss(model) => {
                    let manager = CacheResidencyManager::new(options)
                        .map_err(|error| Error::Parallel(error.to_string()))?;
                    let rank = CacheRankIdentity {
                        pipeline_rank: None,
                        tensor_parallel_rank: (self.topology.tensor_parallel_size > 1)
                            .then_some(self.topology.tensor_parallel_rank),
                        expert_parallel_rank: Some(self.topology.expert_parallel_rank),
                    };
                    model
                        .new_cache_with_manager(manager, Some(rank))
                        .map(ExpertParallelCache::GptOss)
                        .map_err(Into::into)
                }
                ExpertArchitecture::GptOssLayerwise(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ExpertParallelCache::GptOss),
                ExpertArchitecture::DenseQwen(model) => {
                    let manager = CacheResidencyManager::new(options)
                        .map_err(|error| Error::Parallel(error.to_string()))?;
                    let rank = Some(CacheRankIdentity {
                        pipeline_rank: None,
                        tensor_parallel_rank: (self.topology.tensor_parallel_size > 1)
                            .then_some(self.topology.tensor_parallel_rank),
                        expert_parallel_rank: Some(self.topology.expert_parallel_rank),
                    });
                    let caches = dense_qwen::new_paged_cache_with_manager(
                        &model.args,
                        manager,
                        rank,
                    )?;
                    Ok(ExpertParallelCache::DenseQwenPaged(caches))
                }
                ExpertArchitecture::DenseQwenLayerwise(model) => {
                    let manager = CacheResidencyManager::new(options)
                        .map_err(|error| Error::Parallel(error.to_string()))?;
                    let rank = Some(CacheRankIdentity {
                        pipeline_rank: None,
                        tensor_parallel_rank: (self.topology.tensor_parallel_size > 1)
                            .then_some(self.topology.tensor_parallel_rank),
                        expert_parallel_rank: Some(self.topology.expert_parallel_rank),
                    });
                    let caches = dense_qwen::new_paged_cache_with_manager(
                        model.args(),
                        manager,
                        rank,
                    )?;
                    Ok(ExpertParallelCache::DenseQwenPaged(caches))
                }
                ExpertArchitecture::Inkling(model) => {
                    let manager = CacheResidencyManager::new(options)
                        .map_err(|error| Error::Parallel(error.to_string()))?;
                    let rank = Some(CacheRankIdentity {
                        pipeline_rank: None,
                        tensor_parallel_rank: (self.topology.tensor_parallel_size > 1)
                            .then_some(self.topology.tensor_parallel_rank),
                        expert_parallel_rank: Some(self.topology.expert_parallel_rank),
                    });
                    model
                        .new_paged_cache_with_manager(manager, rank)
                        .map(ExpertParallelCache::Inkling)
                        .map_err(Into::into)
                }
                ExpertArchitecture::InklingLayerwise(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ExpertParallelCache::Inkling),
                ExpertArchitecture::NemotronH(model) => {
                    let rank = Some(CacheRankIdentity {
                        pipeline_rank: None,
                        tensor_parallel_rank: (self.topology.tensor_parallel_size > 1)
                            .then_some(self.topology.tensor_parallel_rank),
                        expert_parallel_rank: Some(self.topology.expert_parallel_rank),
                    });
                    nemotron_h::Cache::new_with_options_and_rank(
                        &model.args,
                        CacheResidencyPolicy::Paged(options),
                        rank,
                    )
                    .map(ExpertParallelCache::NemotronH)
                    .map_err(Into::into)
                }
                ExpertArchitecture::NemotronHLayerwise(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ExpertParallelCache::NemotronH),
                _ => Err(Error::Parallel(
                    "paged cache residency is unsupported for this expert-parallel cache representation"
                        .into(),
                )),
            },
        }
    }

    /// Returns aggregate cache-residency telemetry for replicated paged attention state.
    pub fn cache_residency_report(
        &self,
        cache: &ExpertParallelCache,
    ) -> Result<Option<CacheResidencyReport>, Error> {
        match cache {
            ExpertParallelCache::DeepSeek(cache) => cache.residency_report().map_err(Into::into),
            ExpertParallelCache::GptOss(cache) => cache.residency_report().map_err(Into::into),
            ExpertParallelCache::KimiLinear(cache) => cache.residency_report().map_err(Into::into),
            ExpertParallelCache::DenseQwenPaged(caches) => caches
                .iter()
                .flatten()
                .next()
                .map(PagedKeyValueCache::report)
                .transpose()
                .map_err(Into::into),
            ExpertParallelCache::Inkling(cache) => cache.residency_report().map_err(Into::into),
            ExpertParallelCache::NemotronH(cache) => cache.residency_report().map_err(Into::into),
            _ => Ok(None),
        }
    }

    /// Persists this rank's replicated prompt state below a shared root.
    pub fn save_prompt_cache(
        &self,
        cache: &mut ExpertParallelCache,
        root: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        let identity = self.prompt_cache_model_identity()?;
        validate_prompt_cache_model_identity(&descriptor, &identity)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let directory = self.prompt_cache_rank_directory(root.as_ref());
        let rank = Some(CacheRankIdentity {
            pipeline_rank: None,
            tensor_parallel_rank: (self.topology.tensor_parallel_size > 1)
                .then_some(self.topology.tensor_parallel_rank),
            expert_parallel_rank: Some(self.topology.expert_parallel_rank),
        });
        match (&self.architecture, cache) {
            (_, ExpertParallelCache::DeepSeek(cache)) => cache
                .save_prompt_cache(directory, descriptor, prefix_token_ids, options)
                .map_err(Into::into),
            (_, ExpertParallelCache::GptOss(cache)) => cache
                .save_prompt_cache(directory, descriptor, prefix_token_ids, options)
                .map_err(Into::into),
            (
                ExpertArchitecture::KimiLinear(_) | ExpertArchitecture::KimiLinearLayerwise(_),
                ExpertParallelCache::KimiLinear(cache),
            ) => kimi_linear::Model::save_prompt_cache_with_rank(
                cache,
                directory,
                descriptor,
                prefix_token_ids,
                options,
                rank,
            )
            .map_err(Into::into),
            (
                ExpertArchitecture::Lfm2(_) | ExpertArchitecture::Lfm2Layerwise(_),
                ExpertParallelCache::Lfm2(cache),
            ) => lfm2::Model::save_prompt_cache_with_rank(
                cache,
                directory,
                descriptor,
                prefix_token_ids,
                options,
                rank,
                stream,
            )
            .map_err(Into::into),
            (
                ExpertArchitecture::Inkling(_) | ExpertArchitecture::InklingLayerwise(_),
                ExpertParallelCache::Inkling(cache),
            ) => inkling::Model::save_prompt_cache(
                cache,
                directory,
                descriptor,
                prefix_token_ids,
                options,
                stream,
            )
            .map_err(Into::into),
            (
                ExpertArchitecture::NemotronH(_) | ExpertArchitecture::NemotronHLayerwise(_),
                ExpertParallelCache::NemotronH(cache),
            ) => nemotron_h::Model::save_prompt_cache_with_rank(
                cache,
                directory,
                descriptor,
                prefix_token_ids,
                options,
                rank,
                stream,
            )
            .map_err(Into::into),
            (ExpertArchitecture::Qwen3VlLayerwise(model), ExpertParallelCache::Qwen3Vl(cache)) => {
                model.save_prompt_cache_with_validated_identity(
                    cache,
                    &directory,
                    descriptor,
                    prefix_token_ids,
                    options,
                    stream,
                )
            }
            (ExpertArchitecture::Gemma4Layerwise(model), ExpertParallelCache::Gemma4(cache)) => {
                model.save_prompt_cache(
                    cache,
                    directory,
                    descriptor,
                    prefix_token_ids,
                    options,
                    stream,
                )
            }
            _ => Err(Error::Parallel(
                "expert-parallel model and prompt-cache representations do not match".into(),
            )),
        }
    }

    /// Opens this rank's compatible replicated prefix.
    pub fn load_prompt_cache(
        &self,
        root: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(ExpertParallelCache, PromptCacheManifest), Error> {
        let identity = self.prompt_cache_model_identity()?;
        validate_prompt_cache_model_identity(expected, &identity)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let directory = self.prompt_cache_rank_directory(root.as_ref());
        match &self.architecture {
            ExpertArchitecture::KimiLinear(model) => {
                return kimi_linear::Model::load_paged_prompt_cache_with_identity(
                    &model.args,
                    directory,
                    expected,
                    prefix_token_ids,
                    &identity,
                    options,
                    stream,
                )
                .map(|(cache, manifest)| (ExpertParallelCache::KimiLinear(cache), manifest))
                .map_err(Into::into);
            }
            ExpertArchitecture::KimiLinearLayerwise(model) => {
                return kimi_linear::Model::load_paged_prompt_cache_with_identity(
                    model.args(),
                    directory,
                    expected,
                    prefix_token_ids,
                    &identity,
                    options,
                    stream,
                )
                .map(|(cache, manifest)| (ExpertParallelCache::KimiLinear(cache), manifest))
                .map_err(Into::into);
            }
            ExpertArchitecture::Lfm2(model) => {
                return lfm2::Model::load_prompt_cache_with_identity(
                    &model.args,
                    directory,
                    expected,
                    prefix_token_ids,
                    identity,
                    stream,
                )
                .map(|(cache, manifest)| (ExpertParallelCache::Lfm2(cache), manifest))
                .map_err(Into::into);
            }
            ExpertArchitecture::Lfm2Layerwise(model) => {
                return lfm2::Model::load_prompt_cache_with_identity(
                    model.args(),
                    directory,
                    expected,
                    prefix_token_ids,
                    identity,
                    stream,
                )
                .map(|(cache, manifest)| (ExpertParallelCache::Lfm2(cache), manifest))
                .map_err(Into::into);
            }
            ExpertArchitecture::Inkling(model) => {
                return inkling::Model::load_prompt_cache_with_identity(
                    &model.args,
                    directory,
                    expected,
                    prefix_token_ids,
                    &identity,
                    stream,
                )
                .map(|(cache, manifest)| (ExpertParallelCache::Inkling(cache), manifest))
                .map_err(Into::into);
            }
            ExpertArchitecture::InklingLayerwise(model) => {
                return inkling::Model::load_prompt_cache_with_identity(
                    model.args(),
                    directory,
                    expected,
                    prefix_token_ids,
                    &identity,
                    stream,
                )
                .map(|(cache, manifest)| (ExpertParallelCache::Inkling(cache), manifest))
                .map_err(Into::into);
            }
            ExpertArchitecture::NemotronH(model) => {
                return nemotron_h::Model::load_paged_prompt_cache_with_identity(
                    &model.args,
                    directory,
                    expected,
                    prefix_token_ids,
                    &identity,
                    options,
                    stream,
                )
                .map(|(cache, manifest)| (ExpertParallelCache::NemotronH(cache), manifest))
                .map_err(Into::into);
            }
            ExpertArchitecture::NemotronHLayerwise(model) => {
                return nemotron_h::Model::load_paged_prompt_cache_with_identity(
                    model.args(),
                    directory,
                    expected,
                    prefix_token_ids,
                    &identity,
                    options,
                    stream,
                )
                .map(|(cache, manifest)| (ExpertParallelCache::NemotronH(cache), manifest))
                .map_err(Into::into);
            }
            ExpertArchitecture::Qwen3VlLayerwise(model) => {
                return model
                    .load_prompt_cache_with_validated_identity(
                        &directory,
                        expected,
                        &identity,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map(|(cache, manifest)| (ExpertParallelCache::Qwen3Vl(cache), manifest));
            }
            ExpertArchitecture::Gemma4Layerwise(model) => {
                return model
                    .load_prompt_cache(&directory, expected, prefix_token_ids, options, stream)
                    .map(|(cache, manifest)| (ExpertParallelCache::Gemma4(cache), manifest));
            }
            _ => {}
        }
        let (manager, manifest) =
            open_prompt_cache(directory, expected, &identity, prefix_token_ids, options)
                .map_err(|error| Error::Parallel(error.to_string()))?;
        let rank = Some(CacheRankIdentity {
            pipeline_rank: None,
            tensor_parallel_rank: (self.topology.tensor_parallel_size > 1)
                .then_some(self.topology.tensor_parallel_rank),
            expert_parallel_rank: Some(self.topology.expert_parallel_rank),
        });
        let cache =
            match &self.architecture {
                ExpertArchitecture::DeepSeek(model) => model
                    .new_cache_with_manager(manager, rank)
                    .map(ExpertParallelCache::DeepSeek)?,
                ExpertArchitecture::GptOss(model) => model
                    .new_cache_with_manager(manager, rank)
                    .map(ExpertParallelCache::GptOss)?,
                _ => return Err(Error::Parallel(
                    "expert-parallel prompt loading is unsupported for this cache representation"
                        .into(),
                )),
            };
        Ok((cache, manifest))
    }

    fn prompt_cache_rank_directory(&self, root: &Path) -> PathBuf {
        root.join(format!("rank-{:05}", self.topology.global_rank))
    }

    /// Returns the canonical cache-relevant architecture identity for this rank.
    pub fn prompt_cache_architecture_fingerprint(&self) -> Result<String, Error> {
        match &self.architecture {
            ExpertArchitecture::KimiLinear(model) => {
                return Ok(
                    crate::architectures::kimi_linear::model::prompt_cache_architecture_fingerprint(
                        &model.args,
                    ),
                );
            }
            ExpertArchitecture::KimiLinearLayerwise(model) => {
                return Ok(
                    crate::architectures::kimi_linear::model::prompt_cache_architecture_fingerprint(
                        model.args(),
                    ),
                );
            }
            _ => {}
        }
        Ok(self.prompt_cache_model_identity()?.architecture_fingerprint)
    }

    /// Returns this rank's exact ordered prompt-cache layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        Ok(self.prompt_cache_model_identity()?.layer_layout)
    }

    fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        let layerwise_identity = match &self.architecture {
            ExpertArchitecture::KimiLinearLayerwise(model) => {
                Some(model.prompt_cache_model_identity()?)
            }
            ExpertArchitecture::DenseQwenLayerwise(model) => {
                Some(model.prompt_cache_model_identity()?)
            }
            ExpertArchitecture::GptOssLayerwise(model) => {
                Some(model.prompt_cache_model_identity()?)
            }
            ExpertArchitecture::InklingLayerwise(model) => {
                Some(model.prompt_cache_model_identity()?)
            }
            ExpertArchitecture::Lfm2Layerwise(model) => Some(model.prompt_cache_model_identity()?),
            ExpertArchitecture::Qwen3VlLayerwise(model) => {
                Some(model.prompt_cache_model_identity()?)
            }
            ExpertArchitecture::Gemma4Layerwise(model) => {
                Some(model.prompt_cache_model_identity()?)
            }
            _ => None,
        };
        if let Some(mut identity) = layerwise_identity {
            identity.topology = PromptCacheTopology::for_parallel_topology(self.topology);
            return Ok(identity);
        }
        let (model_family, effective_model_type, architecture_fingerprint, layer_count) =
            match &self.architecture {
                ExpertArchitecture::DeepSeek(model) => (
                    "deepseek_v3".to_string(),
                    model.args.model_type.clone(),
                    crate::architectures::deepseek_v3::model::prompt_cache_architecture_fingerprint(
                        &model.args,
                    ),
                    model.args.layer_schedule.len(),
                ),
                ExpertArchitecture::KimiLinear(model) => (
                    "kimi_linear".to_string(),
                    model.args.model_type.clone(),
                    kimi_linear::prompt_cache_architecture_fingerprint(&model.args),
                    model.args.layer_schedule.len(),
                ),
                ExpertArchitecture::GptOss(model) => (
                    "gpt_oss".to_string(),
                    model.args.model_type.clone(),
                    crate::architectures::gpt_oss::model::prompt_cache_architecture_fingerprint(
                        &model.args,
                    ),
                    usize::try_from(model.args.num_hidden_layers)
                        .map_err(|_| Error::Parallel("invalid GPT-OSS layer count".into()))?,
                ),
                ExpertArchitecture::Inkling(model) => (
                    "inkling".to_string(),
                    model.args.model_type.clone(),
                    inkling::prompt_cache_architecture_fingerprint(&model.args),
                    usize::try_from(model.args.text_config.num_hidden_layers)
                        .map_err(|_| Error::Parallel("invalid Inkling layer count".into()))?,
                ),
                ExpertArchitecture::Lfm2(model) => (
                    "lfm2".to_string(),
                    model.args.model_type.clone(),
                    lfm2::prompt_cache_architecture_fingerprint(&model.args),
                    usize::try_from(model.args.num_hidden_layers)
                        .map_err(|_| Error::Parallel("invalid LFM2 layer count".into()))?,
                ),
                ExpertArchitecture::Lfm2Layerwise(model) => (
                    "lfm2".to_string(),
                    model.args().model_type.clone(),
                    lfm2::prompt_cache_architecture_fingerprint(model.args()),
                    usize::try_from(model.args().num_hidden_layers)
                        .map_err(|_| Error::Parallel("invalid LFM2 layer count".into()))?,
                ),
                ExpertArchitecture::NemotronH(model) => (
                    "nemotron_h".to_string(),
                    model.args.model_type.clone(),
                    nemotron_h::prompt_cache_architecture_fingerprint(&model.args),
                    usize::try_from(model.args.num_hidden_layers)
                        .map_err(|_| Error::Parallel("invalid Nemotron-H layer count".into()))?,
                ),
                ExpertArchitecture::NemotronHLayerwise(model) => (
                    "nemotron_h".to_string(),
                    model.args().model_type.clone(),
                    nemotron_h::prompt_cache_architecture_fingerprint(model.args()),
                    usize::try_from(model.args().num_hidden_layers)
                        .map_err(|_| Error::Parallel("invalid Nemotron-H layer count".into()))?,
                ),
                _ => return Err(Error::Parallel(
                    "prompt-cache persistence is unsupported for this expert-parallel architecture"
                        .into(),
                )),
            };
        let layer_layout = match &self.architecture {
            ExpertArchitecture::DeepSeek(model) => PromptCacheModelIdentity::compressed_layouts(
                layer_count,
                model.args.kv_lora_rank,
                model.args.qk_rope_head_dim,
            )
            .map_err(|error| Error::Parallel(error.to_string()))?,
            ExpertArchitecture::KimiLinear(model) => {
                kimi_linear::prompt_cache_layer_layout(&model.args)?
            }
            ExpertArchitecture::GptOss(model) => PromptCacheModelIdentity::key_value_layouts(
                model.args.attention_schedule.iter().map(|policy| {
                    policy.window().map(|window| {
                        i32::try_from(window.get())
                            .expect("validated GPT-OSS sliding window fits i32")
                    })
                }),
                model.args.num_key_value_heads,
                model.args.head_dim,
            )
            .map_err(|error| Error::Parallel(error.to_string()))?,
            ExpertArchitecture::Inkling(model) => inkling::prompt_cache_layer_layout(&model.args)?,
            ExpertArchitecture::Lfm2(model) => lfm2::prompt_cache_layer_layout(&model.args)?,
            ExpertArchitecture::Lfm2Layerwise(model) => {
                lfm2::prompt_cache_layer_layout(model.args())?
            }
            ExpertArchitecture::NemotronH(model) => {
                nemotron_h::prompt_cache_layer_layout(&model.args)?
            }
            ExpertArchitecture::NemotronHLayerwise(model) => {
                nemotron_h::prompt_cache_layer_layout(model.args())?
            }
            _ => unreachable!("identity rejects unsupported expert architectures"),
        };
        Ok(PromptCacheModelIdentity {
            model_family,
            effective_model_type,
            architecture_fingerprint,
            layer_count,
            global_layer_start: 0,
            global_layer_end: layer_count,
            sink_tokens: 0,
            topology: PromptCacheTopology {
                pipeline: None,
                tensor_parallel: (self.topology.tensor_parallel_size > 1).then_some((
                    self.topology.tensor_parallel_size,
                    self.topology.tensor_parallel_rank,
                )),
                expert_parallel: Some((
                    self.topology.expert_parallel_size,
                    self.topology.expert_parallel_rank,
                )),
                expert_parallel_cache_replicated: true,
            },
            layer_layout,
        })
    }

    /// Allocates a bounded Qwen3 sliding-window cache.
    pub fn new_qwen3_sliding_cache(
        &self,
        max_size: i32,
        options: PagedCacheOptions,
    ) -> Result<ExpertParallelCache, Error> {
        if max_size <= 0 {
            return Err(Error::Parallel(
                "Qwen3 sliding cache size must be positive".into(),
            ));
        }
        match &self.architecture {
            ExpertArchitecture::DenseQwen(model) => {
                let manager = CacheResidencyManager::new(options)
                    .map_err(|error| Error::Parallel(error.to_string()))?;
                let rank = Some(CacheRankIdentity {
                    pipeline_rank: None,
                    tensor_parallel_rank: (self.topology.tensor_parallel_size > 1)
                        .then_some(self.topology.tensor_parallel_rank),
                    expert_parallel_rank: Some(self.topology.expert_parallel_rank),
                });
                Ok(ExpertParallelCache::DenseQwenPaged(
                    (0..model.model.layers.len())
                        .map(|layer| {
                            PagedKeyValueCache::new_with_layout(
                                manager.clone(),
                                layer,
                                Some(max_size),
                                0,
                                rank,
                            )
                            .map(Some)
                            .map_err(Error::from)
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                ))
            }
            ExpertArchitecture::DenseQwenLayerwise(model) => {
                let manager = CacheResidencyManager::new(options)
                    .map_err(|error| Error::Parallel(error.to_string()))?;
                let rank = Some(CacheRankIdentity {
                    pipeline_rank: None,
                    tensor_parallel_rank: (self.topology.tensor_parallel_size > 1)
                        .then_some(self.topology.tensor_parallel_rank),
                    expert_parallel_rank: Some(self.topology.expert_parallel_rank),
                });
                let layer_count = usize::try_from(model.args().num_hidden_layers)
                    .map_err(|_| Error::Parallel("invalid dense-Qwen layer count".into()))?;
                Ok(ExpertParallelCache::DenseQwenPaged(
                    (0..layer_count)
                        .map(|layer| {
                            PagedKeyValueCache::new_with_layout(
                                manager.clone(),
                                layer,
                                Some(max_size),
                                0,
                                rank,
                            )
                            .map(Some)
                            .map_err(Error::from)
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                ))
            }
            _ => Err(Error::Parallel(
                "sliding key/value caches are only available for Qwen3 expert parallelism".into(),
            )),
        }
    }

    /// Counters from the most recent complete model forward.
    pub fn latest_routing_statistics(&self) -> &RoutingStatistics {
        &self.latest_statistics
    }

    /// Counters accumulated across all forwards.
    pub fn cumulative_routing_statistics(&self) -> &RoutingStatistics {
        &self.cumulative_statistics
    }

    /// Runs prefill or decode with identical input tokens on every EP rank.
    pub fn forward(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut ExpertParallelCache,
        group: &Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward_impl(tokens, mask, cache, None, group, None, stream)
    }

    /// Runs combined TP+EP inference with communicators derived from one topology.
    pub fn forward_cartesian(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut ExpertParallelCache,
        execution: &crate::CartesianExecution<'_>,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if execution.topology() != self.topology {
            return Err(Error::Parallel(format!(
                "model topology {:?} does not match Cartesian execution topology {:?}",
                self.topology,
                execution.topology()
            )));
        }
        let tensor_context = execution.tensor_context(stream)?;
        let tensor_group = tensor_context.group();
        let expert_group = execution.expert_group().ok_or_else(|| {
            Error::Parallel("combined tensor/expert execution requires an EP group".into())
        })?;
        self.forward_impl(
            tokens,
            mask,
            cache,
            tensor_group,
            expert_group,
            None,
            stream,
        )
    }

    /// Runs expert-parallel inference while exposing global router decisions,
    /// rank-local expert contributions, reduced routed outputs, and shared experts.
    pub fn forward_with_observer(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut ExpertParallelCache,
        group: &Group,
        observer: &mut impl ActivationObserver,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward_impl(tokens, mask, cache, None, group, Some(observer), stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_impl(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut ExpertParallelCache,
        tensor_group: Option<&Group>,
        group: &Group,
        observer: Option<&mut dyn ActivationObserver>,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let total_started = Instant::now();
        self.validate_expert_group(group)?;
        self.topology.validate_execution_stream(stream)?;
        if tokens.ndim() != 2 {
            return Err(Error::Parallel(format!(
                "expert-parallel token input must be [batch, sequence], got {:?}",
                tokens.shape()
            )));
        }
        let mut statistics = RoutingStatistics::default();
        let logits = if let Some(expert_cache) = self.expert_cache.as_ref() {
            if observer.is_some() {
                return Err(Error::UnsupportedArchitecture(
                    "detailed activation observation is unavailable for external-expert parallelism"
                        .into(),
                ));
            }
            let pass = if tokens.dim(1) > 1 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            };
            let assignment = &self.info.assignment;
            match (&mut self.architecture, cache) {
                (ExpertArchitecture::DeepSeek(model), ExpertParallelCache::DeepSeek(cache)) => {
                    let args = model.args.clone();
                    model.forward_cached_expert_parallel(
                        tokens,
                        mask,
                        cache,
                        |layer, hidden, ids, weights, stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    execute_cached_deepseek(
                                        &args,
                                        layer,
                                        routes,
                                        pass,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        },
                        stream,
                    )?
                }
                (
                    ExpertArchitecture::DeepSeekLayerwise(model),
                    ExpertParallelCache::DeepSeek(cache),
                ) => {
                    let args = model.args().clone();
                    let mut execute =
                        |layer, hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    execute_cached_deepseek(
                                        &args,
                                        layer,
                                        routes,
                                        pass,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        };
                    match tensor_group {
                        Some(tensor_group) => model.forward_tensor_expert_parallel(
                            tokens,
                            mask,
                            cache,
                            tensor_group,
                            &mut execute,
                            stream,
                        )?,
                        None => model.forward_with_expert_executor(
                            tokens,
                            mask,
                            cache,
                            &mut execute,
                            stream,
                        )?,
                    }
                }
                (ExpertArchitecture::KimiLinear(model), ExpertParallelCache::KimiLinear(cache)) => {
                    let args = model.args.clone();
                    model.forward_cached_expert_parallel(
                        tokens,
                        mask,
                        cache,
                        |layer, hidden, ids, weights, stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    execute_cached_kimi_linear(
                                        &args,
                                        layer,
                                        routes,
                                        pass,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        },
                        stream,
                    )?
                }
                (
                    ExpertArchitecture::KimiLinearLayerwise(model),
                    ExpertParallelCache::KimiLinear(cache),
                ) => {
                    let args = model.args().clone();
                    let mut execute =
                        |layer, hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    execute_cached_kimi_linear(
                                        &args,
                                        layer,
                                        routes,
                                        pass,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        };
                    match tensor_group {
                        Some(tensor_group) => model.forward_tensor_expert_parallel(
                            tokens,
                            mask,
                            cache,
                            tensor_group,
                            &mut execute,
                            stream,
                        )?,
                        None => model.forward_with_expert_executor(
                            tokens,
                            mask,
                            cache,
                            &mut execute,
                            stream,
                        )?,
                    }
                }
                (ExpertArchitecture::DenseQwen(model), ExpertParallelCache::DenseQwen(cache)) => {
                    let args = model.args.clone();
                    model.forward_cached_expert_parallel(
                        dense_qwen::ModelInput {
                            inputs: tokens,
                            mask,
                            cache,
                        },
                        |layer, hidden, ids, weights, stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    execute_cached_qwen3(
                                        &args,
                                        layer,
                                        routes,
                                        pass,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        },
                        stream,
                    )?
                }
                (
                    ExpertArchitecture::DenseQwenLayerwise(model),
                    ExpertParallelCache::DenseQwen(cache),
                ) => {
                    let args = model.args().clone();
                    let mut execute =
                        |layer, hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    execute_cached_qwen3(
                                        &args,
                                        layer,
                                        routes,
                                        pass,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        };
                    match tensor_group {
                        Some(tensor_group) => model.forward_tensor_expert_parallel(
                            tokens,
                            mask,
                            cache,
                            tensor_group,
                            &mut execute,
                            stream,
                        )?,
                        None => model.forward_with_expert_executor(
                            tokens,
                            mask,
                            cache,
                            &mut execute,
                            stream,
                        )?,
                    }
                }
                (
                    ExpertArchitecture::DenseQwenLayerwise(model),
                    ExpertParallelCache::DenseQwenSliding(cache),
                ) => {
                    let args = model.args().clone();
                    let mut execute =
                        |layer, hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    execute_cached_qwen3(
                                        &args,
                                        layer,
                                        routes,
                                        pass,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        };
                    match tensor_group {
                        Some(tensor_group) => model.forward_tensor_expert_parallel_sliding(
                            tokens,
                            mask,
                            cache,
                            tensor_group,
                            &mut execute,
                            stream,
                        )?,
                        None => model.forward_with_sliding_expert_executor(
                            tokens,
                            mask,
                            cache,
                            &mut execute,
                            stream,
                        )?,
                    }
                }
                (
                    ExpertArchitecture::DenseQwenLayerwise(model),
                    ExpertParallelCache::DenseQwenPaged(cache),
                ) => {
                    let args = model.args().clone();
                    let mut execute =
                        |layer, hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    execute_cached_qwen3(
                                        &args,
                                        layer,
                                        routes,
                                        pass,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        };
                    match tensor_group {
                        Some(tensor_group) => model.forward_tensor_expert_parallel_paged(
                            tokens,
                            mask,
                            cache,
                            tensor_group,
                            &mut execute,
                            stream,
                        )?,
                        None => model.forward_with_paged_expert_executor(
                            tokens,
                            mask,
                            cache,
                            &mut execute,
                            stream,
                        )?,
                    }
                }
                (
                    ExpertArchitecture::DenseQwen(model),
                    ExpertParallelCache::DenseQwenSliding(cache),
                ) => {
                    let args = model.args.clone();
                    model.forward_cached_expert_parallel(
                        dense_qwen::ModelInput {
                            inputs: tokens,
                            mask,
                            cache,
                        },
                        |layer, hidden, ids, weights, stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    execute_cached_qwen3(
                                        &args,
                                        layer,
                                        routes,
                                        pass,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        },
                        stream,
                    )?
                }
                (
                    ExpertArchitecture::DenseQwen(model),
                    ExpertParallelCache::DenseQwenPaged(cache),
                ) => {
                    let args = model.args.clone();
                    model.forward_cached_expert_parallel(
                        dense_qwen::ModelInput {
                            inputs: tokens,
                            mask,
                            cache,
                        },
                        |layer, hidden, ids, weights, stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    execute_cached_qwen3(
                                        &args,
                                        layer,
                                        routes,
                                        pass,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        },
                        stream,
                    )?
                }
                (ExpertArchitecture::GptOss(model), ExpertParallelCache::GptOss(cache)) => {
                    let args = model.args.clone();
                    model.forward_cached_expert_parallel(
                        tokens,
                        cache,
                        |layer, hidden, ids, weights, stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    execute_cached_gpt_oss(
                                        &args,
                                        layer,
                                        routes,
                                        pass,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        },
                        stream,
                    )?
                }
                (
                    ExpertArchitecture::GptOssLayerwise(model),
                    ExpertParallelCache::GptOss(cache),
                ) => {
                    let args = model.args().clone();
                    let mut execute =
                        |layer, hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    execute_cached_gpt_oss(
                                        &args,
                                        layer,
                                        routes,
                                        pass,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        };
                    match tensor_group {
                        Some(tensor_group) => model.forward_tensor_expert_parallel(
                            tokens,
                            cache,
                            tensor_group,
                            &mut execute,
                            stream,
                        )?,
                        None => model.forward_with_expert_executor(
                            tokens,
                            cache,
                            &mut execute,
                            stream,
                        )?,
                    }
                }
                (ExpertArchitecture::Inkling(model), ExpertParallelCache::Inkling(cache)) => {
                    let args = model.args.clone();
                    model.forward_cached_expert_parallel(
                        tokens,
                        cache,
                        |layer, hidden, ids, weights, stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    execute_cached_inkling(
                                        &args,
                                        layer,
                                        routes,
                                        pass,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        },
                        stream,
                    )?
                }
                (
                    ExpertArchitecture::InklingLayerwise(model),
                    ExpertParallelCache::Inkling(cache),
                ) => {
                    let args = model.args().clone();
                    let mut execute =
                        |layer, hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    execute_cached_inkling(
                                        &args,
                                        layer,
                                        routes,
                                        pass,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        };
                    match tensor_group {
                        Some(tensor_group) => model.forward_tensor_expert_parallel(
                            tokens,
                            cache,
                            tensor_group,
                            &mut execute,
                            stream,
                        )?,
                        None => model.forward_with_expert_executor(
                            tokens,
                            cache,
                            &mut execute,
                            stream,
                        )?,
                    }
                }
                (ExpertArchitecture::Lfm2(model), ExpertParallelCache::Lfm2(cache)) => {
                    let args = model.args.clone();
                    model.forward_cached_expert_parallel(
                        tokens,
                        cache,
                        |layer, hidden, ids, weights, stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    execute_cached_lfm2(
                                        &args,
                                        layer,
                                        routes,
                                        pass,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        },
                        stream,
                    )?
                }
                (ExpertArchitecture::Lfm2Layerwise(model), ExpertParallelCache::Lfm2(cache)) => {
                    let args = model.args().clone();
                    let mut execute =
                        |layer, hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    execute_cached_lfm2(
                                        &args,
                                        layer,
                                        routes,
                                        pass,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        };
                    match tensor_group {
                        Some(tensor_group) => model.forward_tensor_expert_parallel(
                            tokens,
                            cache,
                            tensor_group,
                            &mut execute,
                            stream,
                        )?,
                        None => model.forward_with_expert_executor(
                            tokens,
                            cache,
                            &mut execute,
                            stream,
                        )?,
                    }
                }
                (ExpertArchitecture::NemotronH(model), ExpertParallelCache::NemotronH(cache)) => {
                    let args = model.args.clone();
                    model.forward_cached_expert_parallel(
                        tokens,
                        cache,
                        |layer, hidden, ids, weights, stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    execute_cached_nemotron_h(
                                        &args,
                                        layer,
                                        routes,
                                        pass,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        },
                        stream,
                    )?
                }
                (
                    ExpertArchitecture::NemotronHLayerwise(model),
                    ExpertParallelCache::NemotronH(cache),
                ) => {
                    let args = model.args().clone();
                    let mut execute =
                        |layer, hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    execute_cached_nemotron_h(
                                        &args,
                                        layer,
                                        routes,
                                        pass,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        };
                    match tensor_group {
                        Some(tensor_group) => model.forward_tensor_expert_parallel(
                            tokens,
                            cache,
                            tensor_group,
                            &mut execute,
                            stream,
                        )?,
                        None => model.forward_with_expert_executor(
                            tokens,
                            cache,
                            &mut execute,
                            stream,
                        )?,
                    }
                }
                (ExpertArchitecture::QwenHybrid(model), ExpertParallelCache::QwenHybrid(cache)) => {
                    let args = model.args.clone();
                    model.forward_cached_expert_parallel(
                        tokens,
                        cache,
                        |layer, hidden, ids, weights, stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    execute_cached_qwen_hybrid(
                                        &args,
                                        layer,
                                        routes,
                                        pass,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        },
                        stream,
                    )?
                }
                (
                    ExpertArchitecture::QwenHybridLayerwise(model),
                    ExpertParallelCache::QwenHybrid(cache),
                ) => {
                    let args = model.args().clone();
                    let mut execute = |layer: usize,
                                       hidden: &Array,
                                       ids: &Array,
                                       weights: &Array,
                                       stream: &Stream| {
                        let returned = dispatch_replicated_with(
                            hidden,
                            ids,
                            weights,
                            assignment,
                            group,
                            stream,
                            |routes, stream| {
                                execute_cached_qwen_hybrid(
                                    &args,
                                    layer,
                                    routes,
                                    pass,
                                    expert_cache,
                                    stream,
                                )
                            },
                        )
                        .map_err(|error| Exception::custom(error.to_string()))?;
                        statistics.accumulate(&returned.statistics);
                        Ok(returned.reduced_output)
                    };
                    match tensor_group {
                        Some(tensor_group) => model.forward_tensor_expert_parallel(
                            tokens,
                            cache,
                            tensor_group,
                            &mut execute,
                            stream,
                        )?,
                        None => model.forward_with_expert_executor(
                            tokens,
                            cache,
                            &mut execute,
                            stream,
                        )?,
                    }
                }
                (ExpertArchitecture::Qwen3Vl(model), ExpertParallelCache::Qwen3Vl(cache)) => {
                    let args = model.args.text_config.clone();
                    model.forward_cached_expert_parallel(
                        tokens,
                        cache,
                        |layer, hidden, ids, weights, stream| {
                            let returned = dispatch_replicated_with(
                                hidden,
                                ids,
                                weights,
                                assignment,
                                group,
                                stream,
                                |routes, stream| {
                                    execute_cached_qwen3_at(
                                        &args,
                                        layer,
                                        "model.language_model.layers",
                                        routes,
                                        pass,
                                        expert_cache,
                                        stream,
                                    )
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))?;
                            statistics.accumulate(&returned.statistics);
                            Ok(returned.reduced_output)
                        },
                        stream,
                    )?
                }
                (
                    ExpertArchitecture::Qwen3VlLayerwise(model),
                    ExpertParallelCache::Qwen3Vl(cache),
                ) => {
                    let args = model.args().text_config.clone();
                    let mut execute = |layer: usize,
                                       hidden: &Array,
                                       ids: &Array,
                                       weights: &Array,
                                       stream: &Stream| {
                        let returned = dispatch_replicated_with(
                            hidden,
                            ids,
                            weights,
                            assignment,
                            group,
                            stream,
                            |routes, stream| {
                                execute_cached_qwen3_at(
                                    &args,
                                    layer,
                                    "model.language_model.layers",
                                    routes,
                                    pass,
                                    expert_cache,
                                    stream,
                                )
                            },
                        )
                        .map_err(|error| Exception::custom(error.to_string()))?;
                        statistics.accumulate(&returned.statistics);
                        Ok(returned.reduced_output)
                    };
                    match tensor_group {
                        Some(tensor_group) => model.decode_tensor_expert_parallel(
                            tokens,
                            cache,
                            tensor_group,
                            &mut execute,
                            stream,
                        )?,
                        None => model.decode_with_expert_executor(
                            tokens,
                            cache,
                            &mut execute,
                            stream,
                        )?,
                    }
                }
                (
                    ExpertArchitecture::Gemma4Layerwise(model),
                    ExpertParallelCache::Gemma4(cache),
                ) => {
                    let args = model.args().clone();
                    let mut execute = |layer: usize,
                                       hidden: &Array,
                                       ids: &Array,
                                       weights: &Array,
                                       stream: &Stream| {
                        let returned = dispatch_replicated_with(
                            hidden,
                            ids,
                            weights,
                            assignment,
                            group,
                            stream,
                            |routes, stream| {
                                execute_cached_gemma4(
                                    &args,
                                    layer,
                                    routes,
                                    pass,
                                    expert_cache,
                                    stream,
                                )
                            },
                        )
                        .map_err(|error| Exception::custom(error.to_string()))?;
                        statistics.accumulate(&returned.statistics);
                        Ok(returned.reduced_output)
                    };
                    match tensor_group {
                        Some(tensor_group) => model.decode_tensor_expert_parallel(
                            tokens,
                            cache,
                            tensor_group,
                            &mut execute,
                            stream,
                        )?,
                        None => model.decode_with_expert_executor(
                            tokens,
                            cache,
                            &mut execute,
                            stream,
                        )?,
                    }
                }
                _ => {
                    return Err(Error::Parallel(
                        "expert-parallel cache architecture mismatch".into(),
                    ))
                }
            }
        } else {
            match (&mut self.architecture, cache) {
                (ExpertArchitecture::DeepSeek(model), ExpertParallelCache::DeepSeek(cache)) => {
                    model.forward_expert_parallel(
                        tokens,
                        mask,
                        cache,
                        &self.info.assignment,
                        group,
                        &mut statistics,
                        observer,
                        stream,
                    )?
                }
                (ExpertArchitecture::KimiLinear(model), ExpertParallelCache::KimiLinear(cache)) => {
                    model.forward_expert_parallel(
                        tokens,
                        mask,
                        cache,
                        &self.info.assignment,
                        group,
                        &mut statistics,
                        observer,
                        stream,
                    )?
                }
                (ExpertArchitecture::DenseQwen(model), ExpertParallelCache::DenseQwen(cache)) => {
                    model.forward_expert_parallel(
                        dense_qwen::ModelInput {
                            inputs: tokens,
                            mask,
                            cache,
                        },
                        &self.info.assignment,
                        group,
                        &mut statistics,
                        observer,
                        stream,
                    )?
                }
                (
                    ExpertArchitecture::DenseQwen(model),
                    ExpertParallelCache::DenseQwenSliding(cache),
                ) => model.forward_expert_parallel(
                    dense_qwen::ModelInput {
                        inputs: tokens,
                        mask,
                        cache,
                    },
                    &self.info.assignment,
                    group,
                    &mut statistics,
                    observer,
                    stream,
                )?,
                (
                    ExpertArchitecture::DenseQwen(model),
                    ExpertParallelCache::DenseQwenPaged(cache),
                ) => model.forward_expert_parallel(
                    dense_qwen::ModelInput {
                        inputs: tokens,
                        mask,
                        cache,
                    },
                    &self.info.assignment,
                    group,
                    &mut statistics,
                    observer,
                    stream,
                )?,
                _ => {
                    return Err(Error::Parallel(
                        "expert-parallel cache architecture mismatch".into(),
                    ))
                }
            }
        };
        materialize_timing_phase([&logits])?;
        statistics.model_time = total_started.elapsed();
        self.latest_statistics = statistics;
        self.cumulative_statistics
            .accumulate(&self.latest_statistics);
        Ok(logits)
    }

    fn forward_qwen_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut qwen3_5::Cache,
        group: &Group,
        stream: &Stream,
    ) -> Result<qwen3_5::QwenMtpStepOutput, Error> {
        let total_started = Instant::now();
        self.validate_group(group)?;
        self.topology.validate_execution_stream(stream)?;
        if tokens.ndim() != 2 {
            return Err(Error::Parallel(format!(
                "expert-parallel token input must be [batch, sequence], got {:?}",
                tokens.shape()
            )));
        }
        let expert_cache = self.expert_cache.as_ref().ok_or_else(|| {
            Error::UnsupportedArchitecture(
                "Qwen embedded MTP currently requires sparse expert-cached EP loading".into(),
            )
        })?;
        let pass = if tokens.dim(1) > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let assignment = &self.info.assignment;
        let mut statistics = RoutingStatistics::default();
        let output = match &mut self.architecture {
            ExpertArchitecture::QwenHybrid(model) => {
                let args = model.args.clone();
                model.forward_cached_expert_parallel_mtp(
                    tokens,
                    cache,
                    |layer, hidden, ids, weights, stream| {
                        let returned = dispatch_replicated_with(
                            hidden,
                            ids,
                            weights,
                            assignment,
                            group,
                            stream,
                            |routes, stream| {
                                execute_cached_qwen_hybrid(
                                    &args,
                                    layer,
                                    routes,
                                    pass,
                                    expert_cache,
                                    stream,
                                )
                            },
                        )
                        .map_err(|error| Exception::custom(error.to_string()))?;
                        statistics.accumulate(&returned.statistics);
                        Ok(returned.reduced_output)
                    },
                    stream,
                )?
            }
            ExpertArchitecture::QwenHybridLayerwise(model) => {
                let args = model.args().clone();
                model.forward_mtp_with_expert_executor(
                    tokens,
                    cache,
                    |layer, hidden, ids, weights, stream| {
                        let returned = dispatch_replicated_with(
                            hidden,
                            ids,
                            weights,
                            assignment,
                            group,
                            stream,
                            |routes, stream| {
                                execute_cached_qwen_hybrid(
                                    &args,
                                    layer,
                                    routes,
                                    pass,
                                    expert_cache,
                                    stream,
                                )
                            },
                        )
                        .map_err(|error| Exception::custom(error.to_string()))?;
                        statistics.accumulate(&returned.statistics);
                        Ok(returned.reduced_output)
                    },
                    stream,
                )?
            }
            _ => {
                return Err(Error::UnsupportedArchitecture(
                    "embedded Qwen MTP requires a Qwen3-Next or Qwen3.5 EP model".into(),
                ))
            }
        };
        materialize_timing_phase([&output.logits])?;
        statistics.model_time = total_started.elapsed();
        self.latest_statistics = statistics;
        self.cumulative_statistics
            .accumulate(&self.latest_statistics);
        Ok(output)
    }

    /// Generates with replicated Qwen MTP weights and EP target verification.
    ///
    /// Every rank must call this method with identical inputs and PRNG keys.
    /// Sampling decisions are selected on `sampling_rank` and synchronized
    /// across the EP group.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_embedded_mtp_input(
        &mut self,
        cache: &mut ExpertParallelCache,
        input: runtime_input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampling_rank: usize,
        group: &Group,
        stream: &Stream,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        self.generate_embedded_mtp_input_with_sampler(
            cache,
            input,
            config,
            prng_key,
            &mut DefaultSampler,
            sampling_rank,
            group,
            stream,
        )
    }

    /// Generates through embedded Qwen MTP with a caller-provided sampler.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_embedded_mtp_input_with_sampler<S: SpeculativeSampler + Clone>(
        &mut self,
        cache: &mut ExpertParallelCache,
        input: runtime_input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        sampling_rank: usize,
        group: &Group,
        stream: &Stream,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        self.generate_embedded_mtp_input_with_sampler_callback(
            cache,
            input,
            config,
            prng_key,
            sampler,
            sampling_rank,
            group,
            stream,
            |_| Ok(()),
        )
    }

    /// Generates through embedded Qwen MTP and reports committed tokens on the
    /// designated sampling rank.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_embedded_mtp_input_with_sampler_callback<S, F>(
        &mut self,
        cache: &mut ExpertParallelCache,
        input: runtime_input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        sampling_rank: usize,
        group: &Group,
        stream: &Stream,
        mut on_token: F,
    ) -> Result<(Vec<u32>, MtpStats), Exception>
    where
        S: SpeculativeSampler + Clone,
        F: FnMut(u32) -> Result<(), Exception>,
    {
        self.validate_group(group)
            .map_err(|error| Exception::custom(error.to_string()))?;
        if sampling_rank >= group.size() {
            return Err(Exception::custom(format!(
                "sampling rank {sampling_rank} is outside EP size {}",
                group.size()
            )));
        }
        if !matches!(
            self.mtp_capability(),
            MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Embedded
            }
        ) {
            return Err(Exception::custom(format!(
                "embedded MTP runtime adapter is unavailable for EP model type {} ({:?})",
                self.info.model_kind.model_type_name(),
                self.mtp_capability()
            )));
        }
        let ExpertParallelCache::QwenHybrid(cache) = cache else {
            return Err(Exception::custom(
                "embedded Qwen MTP requires a Qwen hybrid EP cache",
            ));
        };
        let mut synchronized_sampler = ExpertParallelSpeculativeSampler {
            sampler: sampler.clone(),
            sampling_rank,
            group,
        };
        let emit_callbacks = group.rank() == sampling_rank;
        let mut target = ExpertParallelQwenMtpTarget { model: self, group };
        let result = crate::architectures::qwen::hybrid::mtp::generate_with_callback(
            &mut target,
            cache,
            input,
            config,
            prng_key,
            &mut synchronized_sampler,
            stream,
            |token| {
                if emit_callbacks {
                    on_token(token)
                } else {
                    Ok(())
                }
            },
        );
        *sampler = synchronized_sampler.sampler;
        result
    }

    /// Samples on one rank and synchronizes only token ids and stop state.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_and_synchronize<S: Sampler>(
        &self,
        logits: &Array,
        sampler: &mut S,
        temperature: f32,
        prng_state: Option<&mut safemlx::random::RandomState>,
        finished: bool,
        sampling_rank: usize,
        group: &Group,
        stream: &Stream,
    ) -> Result<SynchronizedToken, Error> {
        self.validate_group(group)?;
        sample_and_synchronize(
            Some(logits),
            logits.dim(0),
            sampler,
            temperature,
            prng_state,
            finished,
            sampling_rank,
            group,
            stream,
        )
    }

    fn validate_group(&self, group: &Group) -> Result<(), Error> {
        if group.rank() != self.topology.global_rank || group.size() != self.topology.world_size {
            return Err(Error::Parallel(format!(
                "expert-parallel topology expects group rank {}/{} but received {}/{}",
                self.topology.global_rank,
                self.topology.world_size,
                group.rank(),
                group.size()
            )));
        }
        Ok(())
    }

    fn validate_expert_group(&self, group: &Group) -> Result<(), Error> {
        if group.rank() != self.topology.expert_parallel_rank
            || group.size() != self.topology.expert_parallel_size
        {
            return Err(Error::Parallel(format!(
                "expert topology expects EP subgroup rank {}/{} but received {}/{}",
                self.topology.expert_parallel_rank,
                self.topology.expert_parallel_size,
                group.rank(),
                group.size()
            )));
        }
        Ok(())
    }
}

fn validate_expert_topology(topology: ParallelTopology) -> Result<(), Error> {
    if topology.expert_parallel_size <= 1 {
        return Err(Error::Parallel(
            "expert-parallel loading requires expert_parallel_size > 1".into(),
        ));
    }
    if topology.pipeline_parallel_size != 1 {
        return Err(Error::Parallel(format!(
            "expert model execution accepts EP or TP+EP and therefore requires PP=1, got TP={} PP={} EP={}",
            topology.tensor_parallel_size, topology.pipeline_parallel_size, topology.expert_parallel_size
        )));
    }
    topology.preflight(None, Some(topology.expert_parallel_size))?;
    Ok(())
}

/// Loads an executable expert-parallel MoE model.
///
/// SafeTensors uses the architecture's resident or sparse-cache adapter.
/// Registered GGUF architectures support resident partitioning and
/// rank-filtered sparse expert caching.
pub fn load_expert_parallel_model(
    model_dir: impl AsRef<Path>,
    topology: ParallelTopology,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    load_expert_parallel_model_with_options(
        model_dir,
        ModelLoadOptions::with_parallel(topology),
        stream,
        weights_stream,
    )
}

/// Loads an executable EP or TP+EP model with a caller-supplied assignment.
///
/// The assignment must describe the checkpoint's complete routed-expert set
/// and match this process's EP rank and group size.
pub fn load_expert_parallel_model_with_assignment(
    model_dir: impl AsRef<Path>,
    topology: ParallelTopology,
    assignment: ExpertAssignment,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    load_expert_parallel_model_with_options_and_assignment(
        model_dir,
        ModelLoadOptions::with_parallel(topology),
        assignment,
        stream,
        weights_stream,
    )
}

/// Loads an executable EP or TP+EP model with explicit load options.
pub fn load_expert_parallel_model_with_options(
    model_dir: impl AsRef<Path>,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    load_expert_parallel_model_impl(model_dir, options, None, stream, weights_stream)
}

/// Loads an executable EP or TP+EP model with explicit model options and expert
/// assignment.
pub fn load_expert_parallel_model_with_options_and_assignment(
    model_dir: impl AsRef<Path>,
    options: ModelLoadOptions,
    assignment: ExpertAssignment,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    load_expert_parallel_model_impl(model_dir, options, Some(assignment), stream, weights_stream)
}

fn load_expert_parallel_model_impl(
    model_dir: impl AsRef<Path>,
    options: ModelLoadOptions,
    assignment: Option<ExpertAssignment>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    let model_dir = model_dir.as_ref();
    let topology = options.parallel.ok_or_else(|| {
        Error::Parallel("expert-parallel loading requires ModelLoadOptions::parallel".into())
    })?;
    validate_expert_topology(topology)?;
    topology.validate_execution_stream(stream)?;
    if model_dir
        .extension()
        .is_some_and(|extension| extension == "gguf")
    {
        let checkpoint = GgufCheckpoint::open(model_dir)?;
        let metadata = crate::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        let architecture = match metadata.get("general.architecture") {
            Some(GgufMetadataValue::String(architecture)) => architecture.clone(),
            Some(_) => {
                return Err(Error::UnsupportedArchitecture(
                    "GGUF general.architecture metadata has the wrong type".into(),
                ))
            }
            None => {
                return Err(Error::UnsupportedArchitecture(
                    "GGUF is missing general.architecture metadata".into(),
                ))
            }
        };
        return load_gguf_ep(
            &architecture,
            model_dir,
            &checkpoint,
            metadata,
            topology,
            options,
            assignment,
            stream,
            weights_stream,
        );
    }
    let config: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(model_dir.join("config.json"))?)?;
    let model_type = config.get("model_type").and_then(serde_json::Value::as_str);
    if topology.tensor_parallel_size > 1
        && !matches!(
            model_type,
            Some(
                "deepseek_v3"
                    | "kimi_linear"
                    | "inkling_mm_model"
                    | "qwen3"
                    | "qwen3_moe"
                    | "gpt_oss"
                    | "gemma4"
                    | "gemma4_text"
                    | "gemma4_unified"
                    | "gemma4_unified_text"
                    | "lfm2_moe"
                    | "nemotron_h"
                    | "qwen3_next"
                    | "qwen3_vl_moe"
                    | "qwen3_vl_moe_text"
                    | "qwen3_5"
                    | "qwen3_5_text"
                    | "qwen3_5_moe"
                    | "qwen3_5_moe_text"
            )
        )
    {
        return Err(Error::Parallel(format!(
            "TP+EP preflight requires an architecture with a shared TP semantic plan and EP expert executor; model_type {:?} is not registered",
            model_type.unwrap_or("<missing>")
        )));
    }
    match model_type {
        Some("deepseek_v3") => load_deepseek_ep(
            model_dir,
            topology,
            options,
            assignment,
            stream,
            weights_stream,
        ),
        Some("kimi_linear") => load_kimi_linear_ep(
            model_dir,
            topology,
            options,
            assignment,
            stream,
            weights_stream,
        ),
        Some("qwen3" | "qwen3_moe") => {
            load_qwen3_ep(
                model_dir,
                topology,
                options,
                assignment,
                stream,
                weights_stream,
            )
        }
        Some("gpt_oss") => load_additional_ep(
            model_dir, topology, options, assignment, ModelKind::GptOss, stream, weights_stream,
        ),
        Some("gemma4" | "gemma4_text" | "gemma4_unified" | "gemma4_unified_text") => {
            load_additional_ep(
                model_dir,
                topology,
                options,
                assignment,
                ModelKind::Gemma4,
                stream,
                weights_stream,
            )
        }
        Some("inkling_mm_model") => load_additional_ep(
            model_dir, topology, options, assignment, ModelKind::Inkling, stream, weights_stream,
        ),
        Some("lfm2" | "lfm2_moe") => load_additional_ep(
            model_dir, topology, options, assignment, ModelKind::Lfm2, stream, weights_stream,
        ),
        Some("nemotron_h") => load_additional_ep(
            model_dir, topology, options, assignment, ModelKind::NemotronH, stream, weights_stream,
        ),
        Some("qwen3_next") => load_additional_ep(
            model_dir, topology, options, assignment, ModelKind::Qwen3Next, stream, weights_stream,
        ),
        Some("qwen3_vl_moe" | "qwen3_vl_moe_text") => load_additional_ep(
            model_dir, topology, options, assignment, ModelKind::Qwen3VlMoe, stream, weights_stream,
        ),
        Some("qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text") => load_additional_ep(
            model_dir, topology, options, assignment, ModelKind::Qwen35, stream, weights_stream,
        ),
        Some(model_type) => Err(Error::UnsupportedArchitecture(format!(
            "expert-parallel execution requires a supported safetensors MoE architecture, not {model_type}"
        ))),
        None => Err(Error::UnsupportedArchitecture("expert-parallel model config is missing model_type".into())),
    }
}

fn resolve_model_assignment(
    assignment: Option<ExpertAssignment>,
    global_experts: usize,
    topology: ParallelTopology,
) -> Result<ExpertAssignment, Error> {
    let assignment = assignment.map_or_else(
        || {
            ExpertAssignment::balanced(
                global_experts,
                topology.expert_parallel_size,
                topology.expert_parallel_rank,
            )
        },
        Ok,
    )?;
    if assignment.global_expert_count() != global_experts
        || assignment.group_size() != topology.expert_parallel_size
        || assignment.rank() != topology.expert_parallel_rank
    {
        return Err(Error::Parallel(format!(
            "expert assignment describes {} experts at rank {}/{}, but the model and topology require {global_experts} experts at rank {}/{}",
            assignment.global_expert_count(),
            assignment.rank(),
            assignment.group_size(),
            topology.expert_parallel_rank,
            topology.expert_parallel_size,
        )));
    }
    if assignment.local_expert_count() == 0 {
        return Err(Error::Parallel(format!(
            "expert-parallel model loading does not support an empty local expert bank on rank {}",
            assignment.rank()
        )));
    }
    Ok(assignment)
}

fn slice_axis_zero(
    value: &Array,
    assignment: &ExpertAssignment,
    stream: &Stream,
) -> Result<Array, Error> {
    let ids = assignment.local_global_expert_ids();
    let contiguous = ids.windows(2).all(|pair| pair[1] == pair[0] + 1);
    if contiguous {
        Ok(value.try_index_device(ids[0] as i32..(ids[ids.len() - 1] + 1) as i32, stream)?)
    } else {
        let ids = Array::from_slice(
            &ids.iter().map(|id| *id as i32).collect::<Vec<_>>(),
            &[ids.len() as i32],
        );
        Ok(value.take_axis(&ids, 0, stream)?)
    }
}

fn slice_optional(
    param: &mut Param<Option<Array>>,
    assignment: &ExpertAssignment,
    stream: &Stream,
) -> Result<usize, Error> {
    if let Some(value) = param.as_ref() {
        let local = slice_axis_zero(value, assignment, stream)?;
        let bytes = local.nbytes();
        *param = Param::new(Some(local));
        Ok(bytes)
    } else {
        Ok(0)
    }
}

fn slice_required(
    param: &mut Param<Array>,
    assignment: &ExpertAssignment,
    stream: &Stream,
) -> Result<usize, Error> {
    let local = slice_axis_zero(param.as_ref(), assignment, stream)?;
    let bytes = local.nbytes();
    *param = Param::new(local);
    Ok(bytes)
}

fn parameter_bytes(module: &impl ModuleParameters) -> usize {
    module
        .parameters()
        .flatten()
        .into_values()
        .map(|value| value.nbytes())
        .sum()
}

pub(crate) fn execute_cached_deepseek(
    args: &deepseek_v3::ModelArgs,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    Ok(cache.execute_routes_bounded(
        ExpertRouteBatch::new(
            layer,
            &routes.hidden,
            &routes.global_expert_ids,
            &routes.weights,
            pass,
        ),
        stream,
        |hidden, acquired, _weights, stream| {
            let started = Instant::now();
            let mut bank = RoutedExperts::new_compact(
                args,
                layer as i32,
                acquired.identities().len() as i32,
                stream,
            )?;
            macro_rules! required {
                ($field:ident, $name:literal) => {
                    bank.$field = Param::new(Some(acquired.compact_binding($name, stream)?));
                };
            }
            macro_rules! optional {
                ($field:ident, $name:literal) => {
                    bank.$field = Param::new(acquired.optional_compact_binding($name, stream)?);
                };
            }
            required!(gate_proj, "gate_proj");
            optional!(gate_proj_scale_inv, "gate_proj_scale_inv");
            optional!(gate_proj_scales, "gate_proj_scales");
            optional!(gate_proj_biases, "gate_proj_biases");
            required!(up_proj, "up_proj");
            optional!(up_proj_scale_inv, "up_proj_scale_inv");
            optional!(up_proj_scales, "up_proj_scales");
            optional!(up_proj_biases, "up_proj_biases");
            required!(down_proj, "down_proj");
            optional!(down_proj_scale_inv, "down_proj_scale_inv");
            optional!(down_proj_scales, "down_proj_scales");
            optional!(down_proj_biases, "down_proj_biases");
            cache.record_compact_bank(
                acquired.pass(),
                acquired.scratch_bytes(),
                started.elapsed(),
            )?;
            let weights = safemlx::ops::ones_dtype(&[hidden.dim(0), 1], hidden.dtype(), stream)?;
            Ok(bank.forward_local(hidden, acquired.compact_routes(), &weights, stream)?)
        },
    )?)
}

fn execute_cached_qwen3(
    args: &dense_qwen::DecoderConfig,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    execute_cached_qwen3_at(args, layer, "model.layers", routes, pass, cache, stream)
}

pub(crate) fn execute_cached_gemma4(
    args: &gemma4::ModelArgs,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    Ok(cache.execute_routes_bounded(
        ExpertRouteBatch::new(
            layer,
            &routes.hidden,
            &routes.global_expert_ids,
            &routes.weights,
            pass,
        ),
        stream,
        |hidden, acquired, weights, stream| {
            Ok(
                crate::architectures::gemma4::layerwise::execute_acquired_gemma_experts(
                    args, layer, hidden, acquired, weights, cache, stream,
                )?,
            )
        },
    )?)
}

pub(crate) fn execute_cached_kimi_linear(
    args: &kimi_linear::ModelArgs,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    Ok(cache.execute_routes_bounded(
        ExpertRouteBatch::new(
            layer,
            &routes.hidden,
            &routes.global_expert_ids,
            &routes.weights,
            pass,
        ),
        stream,
        |hidden, acquired, _weights, stream| {
            let started = Instant::now();
            let prefix = format!("model.layers.{layer}.mlp.experts");
            let mut bank = PackedSwiGluExperts::new(
                acquired.identities().len() as i32,
                args.hidden_size,
                args.moe_intermediate_size,
                args.weight_quantization_for(&format!("{prefix}.gate_up_proj")),
                args.weight_quantization_for(&format!("{prefix}.down_proj")),
                stream,
            )?;
            bank.gate_up_proj = Param::new(acquired.compact_binding("gate_up_proj", stream)?);
            bank.gate_up_proj_scales =
                Param::new(acquired.optional_compact_binding("gate_up_proj_scales", stream)?);
            bank.gate_up_proj_biases =
                Param::new(acquired.optional_compact_binding("gate_up_proj_biases", stream)?);
            bank.down_proj = Param::new(acquired.compact_binding("down_proj", stream)?);
            bank.down_proj_scales =
                Param::new(acquired.optional_compact_binding("down_proj_scales", stream)?);
            bank.down_proj_biases =
                Param::new(acquired.optional_compact_binding("down_proj_biases", stream)?);
            cache.record_compact_bank(
                acquired.pass(),
                acquired.scratch_bytes(),
                started.elapsed(),
            )?;
            let weights = safemlx::ops::ones_dtype(&[hidden.dim(0), 1], hidden.dtype(), stream)?;
            Ok(bank.forward(hidden, acquired.compact_routes(), &weights, stream)?)
        },
    )?)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_cached_qwen3_at(
    args: &dense_qwen::DecoderConfig,
    layer: usize,
    layer_root: &str,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    Ok(cache.execute_routes_bounded(
        ExpertRouteBatch::new(
            layer,
            &routes.hidden,
            &routes.global_expert_ids,
            &routes.weights,
            pass,
        ),
        stream,
        |hidden, acquired, _weights, stream| {
            let started = Instant::now();
            let prefix = format!("{layer_root}.{layer}.mlp.experts");
            let mut bank = PackedSwiGluExperts::new(
                acquired.identities().len() as i32,
                args.hidden_size,
                args.moe_intermediate_size,
                args.weight_quantization_for(&format!("{prefix}.gate_up_proj")),
                args.weight_quantization_for(&format!("{prefix}.down_proj")),
                stream,
            )?;
            bank.gate_up_proj = Param::new(acquired.compact_binding("gate_up_proj", stream)?);
            bank.gate_up_proj_scales =
                Param::new(acquired.optional_compact_binding("gate_up_proj_scales", stream)?);
            bank.gate_up_proj_biases =
                Param::new(acquired.optional_compact_binding("gate_up_proj_biases", stream)?);
            bank.down_proj = Param::new(acquired.compact_binding("down_proj", stream)?);
            bank.down_proj_scales =
                Param::new(acquired.optional_compact_binding("down_proj_scales", stream)?);
            bank.down_proj_biases =
                Param::new(acquired.optional_compact_binding("down_proj_biases", stream)?);
            cache.record_compact_bank(
                acquired.pass(),
                acquired.scratch_bytes(),
                started.elapsed(),
            )?;
            let weights = safemlx::ops::ones_dtype(&[hidden.dim(0), 1], hidden.dtype(), stream)?;
            Ok(bank.forward(hidden, acquired.compact_routes(), &weights, stream)?)
        },
    )?)
}

fn execute_cached_gpt_oss(
    args: &gpt_oss::ModelArgs,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    execute_cached_gpt_oss_at(args, layer, routes, pass, cache, 1, stream)
}

pub(crate) fn execute_cached_gpt_oss_at(
    args: &gpt_oss::ModelArgs,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    tensor_parallel_size: usize,
    stream: &Stream,
) -> Result<Array, Error> {
    if tensor_parallel_size == 0 {
        return Err(Error::Parallel(
            "GPT-OSS cached expert execution requires a positive TP size".into(),
        ));
    }
    Ok(cache.execute_routes_bounded(
        ExpertRouteBatch::new(
            layer,
            &routes.hidden,
            &routes.global_expert_ids,
            &routes.weights,
            pass,
        ),
        stream,
        |hidden, acquired, _weights, stream| {
            let started = Instant::now();
            let gate_up_proj_blocks = acquired.compact_binding("gate_up_proj_blocks", stream)?;
            let gate_up_proj_scales = acquired.compact_binding("gate_up_proj_scales", stream)?;
            let gate_up_proj_bias = acquired.compact_binding("gate_up_proj_bias", stream)?;
            let down_proj_blocks = acquired.compact_binding("down_proj_blocks", stream)?;
            let down_proj_scales = acquired.compact_binding("down_proj_scales", stream)?;
            let down_proj_bias = acquired.compact_binding("down_proj_bias", stream)?;
            let mut compact_args = args.clone();
            compact_args.num_local_experts = acquired.identities().len() as i32;
            compact_args.intermediate_size = gate_up_proj_bias.dim(-1) / 2;
            let mut bank = gpt_oss::Experts::new(&compact_args, stream)?;
            bank.gate_up_proj_blocks = Param::new(gate_up_proj_blocks);
            bank.gate_up_proj_scales = Param::new(gate_up_proj_scales);
            bank.gate_up_proj_bias = Param::new(gate_up_proj_bias);
            bank.down_proj_blocks = Param::new(down_proj_blocks);
            bank.down_proj_scales = Param::new(down_proj_scales);
            bank.down_proj_bias = Param::new(down_proj_bias);
            cache.record_compact_bank(
                acquired.pass(),
                acquired.scratch_bytes(),
                started.elapsed(),
            )?;
            let weights = safemlx::ops::ones_dtype(&[hidden.dim(0), 1], hidden.dtype(), stream)?;
            let output = bank.forward(hidden, acquired.compact_routes(), &weights, stream)?;
            if tensor_parallel_size == 1 {
                return Ok(output);
            }
            let bias =
                bank.down_proj_bias
                    .as_ref()
                    .take_axis(acquired.compact_routes(), 0, stream)?;
            Ok(output.subtract(&bias, stream)?.add(
                bias.divide(Array::from_f32(tensor_parallel_size as f32), stream)?,
                stream,
            )?)
        },
    )?)
}

pub(crate) fn execute_cached_inkling(
    args: &inkling::ModelArgs,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    Ok(cache.execute_routes_bounded(
        ExpertRouteBatch::new(
            layer,
            &routes.hidden,
            &routes.global_expert_ids,
            &routes.weights,
            pass,
        ),
        stream,
        |hidden, acquired, _weights, stream| {
            let started = Instant::now();
            let text = &args.text_config;
            let prefix = format!("model.layers.{layer}.moe.experts");
            let mut bank = PackedSwiGluExperts::new_with_dtype(
                acquired.identities().len() as i32,
                text.hidden_size,
                text.moe_intermediate_size(),
                text.weight_quantization_for(&format!("{prefix}.gate_up_proj")),
                text.weight_quantization_for(&format!("{prefix}.down_proj")),
                text.weight_dtype(),
                stream,
            )?;
            bank.gate_up_proj = Param::new(acquired.compact_binding("gate_up_proj", stream)?);
            bank.gate_up_proj_scales =
                Param::new(acquired.optional_compact_binding("gate_up_proj_scales", stream)?);
            bank.gate_up_proj_biases =
                Param::new(acquired.optional_compact_binding("gate_up_proj_biases", stream)?);
            bank.down_proj = Param::new(acquired.compact_binding("down_proj", stream)?);
            bank.down_proj_scales =
                Param::new(acquired.optional_compact_binding("down_proj_scales", stream)?);
            bank.down_proj_biases =
                Param::new(acquired.optional_compact_binding("down_proj_biases", stream)?);
            cache.record_compact_bank(
                acquired.pass(),
                acquired.scratch_bytes(),
                started.elapsed(),
            )?;
            let weights = safemlx::ops::ones_dtype(&[hidden.dim(0), 1], hidden.dtype(), stream)?;
            Ok(bank.forward(hidden, acquired.compact_routes(), &weights, stream)?)
        },
    )?)
}

pub(crate) fn execute_cached_lfm2(
    args: &lfm2::ModelArgs,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    Ok(cache.execute_routes_bounded(
        ExpertRouteBatch::new(
            layer,
            &routes.hidden,
            &routes.global_expert_ids,
            &routes.weights,
            pass,
        ),
        stream,
        |hidden, acquired, _weights, stream| {
            let started = Instant::now();
            let prefix = format!("model.layers.{layer}.feed_forward.experts");
            let mut bank = PackedSwiGluExperts::new(
                acquired.identities().len() as i32,
                args.hidden_size,
                args.moe_intermediate_size,
                args.weight_quantization_for(&format!("{prefix}.gate_up_proj")),
                args.weight_quantization_for(&format!("{prefix}.down_proj")),
                stream,
            )?;
            populate_swiglu_bank(&mut bank, acquired, stream)?;
            cache.record_compact_bank(
                acquired.pass(),
                acquired.scratch_bytes(),
                started.elapsed(),
            )?;
            let weights = safemlx::ops::ones_dtype(&[hidden.dim(0), 1], hidden.dtype(), stream)?;
            Ok(bank.forward(hidden, acquired.compact_routes(), &weights, stream)?)
        },
    )?)
}

pub(crate) fn execute_cached_nemotron_h(
    args: &nemotron_h::ModelArgs,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    Ok(cache.execute_routes_bounded(
        ExpertRouteBatch::new(
            layer,
            &routes.hidden,
            &routes.global_expert_ids,
            &routes.weights,
            pass,
        ),
        stream,
        |hidden, acquired, _weights, stream| {
            let started = Instant::now();
            let prefix = format!("model.layers.{layer}.moe.experts");
            let mut bank = nemotron_h::Experts::new(
                acquired.identities().len() as i32,
                args.hidden_size,
                args.moe_intermediate_size,
                [
                    args.weight_quantization_for(&format!("{prefix}.up_proj")),
                    args.weight_quantization_for(&format!("{prefix}.down_proj")),
                ],
                stream,
            )?;
            bank.up_proj = Param::new(acquired.compact_binding("up_proj", stream)?);
            bank.up_proj_scales =
                Param::new(acquired.optional_compact_binding("up_proj_scales", stream)?);
            bank.up_proj_biases =
                Param::new(acquired.optional_compact_binding("up_proj_biases", stream)?);
            bank.down_proj = Param::new(acquired.compact_binding("down_proj", stream)?);
            bank.down_proj_scales =
                Param::new(acquired.optional_compact_binding("down_proj_scales", stream)?);
            bank.down_proj_biases =
                Param::new(acquired.optional_compact_binding("down_proj_biases", stream)?);
            cache.record_compact_bank(
                acquired.pass(),
                acquired.scratch_bytes(),
                started.elapsed(),
            )?;
            let weights = safemlx::ops::ones_dtype(&[hidden.dim(0), 1], hidden.dtype(), stream)?;
            Ok(bank.forward(hidden, acquired.compact_routes(), &weights, stream)?)
        },
    )?)
}

pub(crate) fn execute_cached_qwen_hybrid(
    args: &qwen3_5::ModelArgs,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    Ok(cache.execute_routes_bounded(
        ExpertRouteBatch::new(
            layer,
            &routes.hidden,
            &routes.global_expert_ids,
            &routes.weights,
            pass,
        ),
        stream,
        |hidden, acquired, _weights, stream| {
            let started = Instant::now();
            let mut compact_args = args.clone();
            compact_args.num_experts = acquired.identities().len() as i32;
            let mut bank = qwen3_5::Experts::new(&compact_args, layer, stream)?;
            bank.gate_up_proj = Param::new(acquired.compact_binding("gate_up_proj", stream)?);
            bank.gate_up_proj_scale_inv =
                Param::new(acquired.optional_compact_binding("gate_up_proj_scale_inv", stream)?);
            bank.gate_up_proj_scales =
                Param::new(acquired.optional_compact_binding("gate_up_proj_scales", stream)?);
            bank.gate_up_proj_biases =
                Param::new(acquired.optional_compact_binding("gate_up_proj_biases", stream)?);
            bank.down_proj = Param::new(acquired.compact_binding("down_proj", stream)?);
            bank.down_proj_scale_inv =
                Param::new(acquired.optional_compact_binding("down_proj_scale_inv", stream)?);
            bank.down_proj_scales =
                Param::new(acquired.optional_compact_binding("down_proj_scales", stream)?);
            bank.down_proj_biases =
                Param::new(acquired.optional_compact_binding("down_proj_biases", stream)?);
            cache.record_compact_bank(
                acquired.pass(),
                acquired.scratch_bytes(),
                started.elapsed(),
            )?;
            let routes = acquired.compact_routes().reshape(&[-1, 1], stream)?;
            let weights = safemlx::ops::ones_dtype(&[hidden.dim(0), 1], hidden.dtype(), stream)?;
            Ok(bank.forward_chunked(hidden, &routes, &weights, stream)?)
        },
    )?)
}

fn populate_swiglu_bank(
    bank: &mut PackedSwiGluExperts,
    acquired: &AcquiredExperts,
    stream: &Stream,
) -> Result<(), ExpertCacheError> {
    bank.gate_up_proj = Param::new(acquired.compact_binding("gate_up_proj", stream)?);
    bank.gate_up_proj_scales =
        Param::new(acquired.optional_compact_binding("gate_up_proj_scales", stream)?);
    bank.gate_up_proj_biases =
        Param::new(acquired.optional_compact_binding("gate_up_proj_biases", stream)?);
    bank.down_proj = Param::new(acquired.compact_binding("down_proj", stream)?);
    bank.down_proj_scales =
        Param::new(acquired.optional_compact_binding("down_proj_scales", stream)?);
    bank.down_proj_biases =
        Param::new(acquired.optional_compact_binding("down_proj_biases", stream)?);
    Ok(())
}

fn expert_bank_needs_slicing(
    bank_experts: i32,
    assignment: &ExpertAssignment,
) -> Result<bool, Error> {
    let bank_experts = usize::try_from(bank_experts).map_err(|_| {
        Error::Parallel(format!(
            "expert bank has invalid negative expert count {bank_experts}"
        ))
    })?;
    if bank_experts == assignment.global_expert_count() {
        Ok(true)
    } else if bank_experts == assignment.local_expert_count() {
        Ok(false)
    } else {
        Err(Error::Parallel(format!(
            "expert bank contains {bank_experts} experts, expected either {} global experts or {} experts local to EP rank {}",
            assignment.global_expert_count(),
            assignment.local_expert_count(),
            assignment.rank(),
        )))
    }
}

fn finalize_deepseek_expert_bank(
    bank: &mut RoutedExperts,
    assignment: &ExpertAssignment,
    stream: &Stream,
) -> Result<usize, Error> {
    if !expert_bank_needs_slicing(bank.num_experts, assignment)? {
        return Ok(parameter_bytes(bank));
    }
    let mut bytes = 0;
    bytes += slice_optional(&mut bank.gate_proj, assignment, stream)?;
    bytes += slice_optional(&mut bank.gate_proj_scale_inv, assignment, stream)?;
    bytes += slice_optional(&mut bank.gate_proj_scales, assignment, stream)?;
    bytes += slice_optional(&mut bank.gate_proj_biases, assignment, stream)?;
    bytes += slice_optional(&mut bank.up_proj, assignment, stream)?;
    bytes += slice_optional(&mut bank.up_proj_scale_inv, assignment, stream)?;
    bytes += slice_optional(&mut bank.up_proj_scales, assignment, stream)?;
    bytes += slice_optional(&mut bank.up_proj_biases, assignment, stream)?;
    bytes += slice_optional(&mut bank.down_proj, assignment, stream)?;
    bytes += slice_optional(&mut bank.down_proj_scale_inv, assignment, stream)?;
    bytes += slice_optional(&mut bank.down_proj_scales, assignment, stream)?;
    bytes += slice_optional(&mut bank.down_proj_biases, assignment, stream)?;
    bank.num_experts = assignment.local_expert_count() as i32;
    Ok(bytes)
}

pub(crate) fn finalize_qwen3_expert_bank(
    bank: &mut PackedSwiGluExperts,
    assignment: &ExpertAssignment,
    stream: &Stream,
) -> Result<usize, Error> {
    if !expert_bank_needs_slicing(bank.num_experts, assignment)? {
        return Ok(parameter_bytes(bank));
    }
    let mut bytes = 0;
    bytes += slice_required(&mut bank.gate_up_proj, assignment, stream)?;
    bytes += slice_optional(&mut bank.gate_up_proj_scales, assignment, stream)?;
    bytes += slice_optional(&mut bank.gate_up_proj_biases, assignment, stream)?;
    bytes += slice_required(&mut bank.down_proj, assignment, stream)?;
    bytes += slice_optional(&mut bank.down_proj_scales, assignment, stream)?;
    bytes += slice_optional(&mut bank.down_proj_biases, assignment, stream)?;
    bank.num_experts = assignment.local_expert_count() as i32;
    Ok(bytes)
}

fn split_expert_id(name: &str) -> Option<usize> {
    let (_, rest) = name.split_once(".mlp.experts.")?;
    rest.split('.').next()?.parse().ok()
}

fn localize_split_expert_name(name: &str, assignment: &ExpertAssignment) -> Option<String> {
    let global = split_expert_id(name)?;
    if assignment.owner(global)? != assignment.rank() {
        return None;
    }
    let local = assignment.owner_local_id(global)?;
    let marker = format!(".mlp.experts.{global}.");
    Some(name.replacen(&marker, &format!(".mlp.experts.{local}."), 1))
}

fn expert_placement_plan(
    store: &(impl WeightStore + ?Sized),
    topology: ParallelTopology,
    assignment: &ExpertAssignment,
) -> Result<(PlacementPlan, bool), Error> {
    let mut plan = PlacementPlan::replicated(topology);
    let mut has_split = false;
    for key in store.keys() {
        if let Some(global) = split_expert_id(&key) {
            has_split = true;
            let placement = if assignment.owner(global) == Some(assignment.rank()) {
                TensorPlacement::Local
            } else {
                TensorPlacement::Omit
            };
            plan.insert(key, placement);
        } else if key.contains(".mlp.experts.")
            && matches!(
                key.rsplit('.').next(),
                Some(
                    "gate_up_proj"
                        | "gate_proj"
                        | "up_proj"
                        | "down_proj"
                        | "gate_proj_scale_inv"
                        | "up_proj_scale_inv"
                        | "down_proj_scale_inv"
                        | "gate_up_proj_scales"
                        | "gate_up_proj_biases"
                        | "gate_proj_scales"
                        | "gate_proj_biases"
                        | "up_proj_scales"
                        | "up_proj_biases"
                        | "down_proj_scales"
                        | "down_proj_biases"
                )
            )
        {
            let ids = assignment.local_global_expert_ids();
            let placement = if ids.windows(2).all(|pair| pair[1] == pair[0] + 1) {
                TensorPlacement::Range {
                    axis: 0,
                    start: ids[0],
                    end: ids[ids.len() - 1] + 1,
                }
            } else {
                TensorPlacement::Indices {
                    axis: 0,
                    indices: ids.to_vec(),
                }
            };
            plan.insert(key, placement);
        }
    }
    Ok((plan, has_split))
}

fn quantize_qwen3_local_experts(
    tensors: &mut std::collections::HashMap<String, Array>,
    num_hidden_layers: i32,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<(), Error> {
    for layer in 0..num_hidden_layers {
        for projection in ["gate_up_proj", "down_proj"] {
            let key = format!("model.layers.{layer}.mlp.experts.{projection}");
            let value = tensors
                .remove(&key)
                .ok_or_else(|| Error::StrictLoadValidation {
                    missing: vec![key.clone()],
                    unused: Vec::new(),
                })?;
            let quantized = quantize_expert_bank(&value, quantization, stream)?;
            eval(
                [&quantized.weight, &quantized.scales]
                    .into_iter()
                    .chain(quantized.biases.as_ref()),
            )?;
            tensors.insert(key.clone(), quantized.weight);
            tensors.insert(format!("{key}_scales"), quantized.scales);
            if let Some(biases) = quantized.biases {
                tensors.insert(format!("{key}_biases"), biases);
            }
        }
    }
    Ok(())
}

fn load_kimi_linear_ep(
    model_dir: &Path,
    topology: ParallelTopology,
    options: ModelLoadOptions,
    assignment: Option<ExpertAssignment>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    if let Some(expert_options) = options.weight_residency.expert_cache() {
        return load_kimi_linear_external_ep(
            model_dir,
            topology,
            options,
            options.weight_residency.layers(),
            ExternalExpertResidency::SparseCache(expert_options),
            options.weight_residency.max_mapped_shards(),
            assignment,
            stream,
            weights_stream,
        );
    }
    if !options.weight_residency.is_fully_resident() {
        return Err(Error::Parallel(
            "Kimi Linear expert-parallel loading requires fully resident or sparse-expert-cache residency"
                .into(),
        ));
    }
    if topology.tensor_parallel_size > 1 {
        return load_kimi_linear_external_ep(
            model_dir,
            topology,
            options,
            LayerWeightResidency::FullyResident,
            ExternalExpertResidency::FullyResident,
            crate::runtime::checkpoint::store::DEFAULT_MAX_MAPPED_SHARDS,
            assignment,
            stream,
            weights_stream,
        );
    }
    let args = kimi_linear::get_model_args(model_dir)?;
    if args.num_experts <= 0 {
        return Err(Error::Parallel(
            "Kimi Linear config has no routed experts".into(),
        ));
    }
    let assignment = resolve_model_assignment(assignment, args.num_experts as usize, topology)?;
    let mut model = if let Some(quantization) = options.quantization {
        kimi_linear::load_model_quantized(model_dir, quantization, stream, weights_stream)?
    } else {
        kimi_linear::load_model(model_dir, stream, weights_stream)?
    };
    let routed_expert_bytes = model.partition_routed_experts(&assignment, stream)?;
    let local_parameter_bytes = parameter_bytes(&model);
    let mut opened_checkpoint_shards = std::fs::read_dir(model_dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "safetensors")
        })
        .collect::<Vec<_>>();
    opened_checkpoint_shards.sort();
    Ok(ExpertParallelModel {
        topology,
        info: ExpertParallelInfo {
            topology,
            global_rank: topology.global_rank,
            expert_parallel_rank: topology.expert_parallel_rank,
            expert_parallel_size: topology.expert_parallel_size,
            model_kind: ModelKind::KimiLinear,
            assignment,
            local_parameter_bytes,
            routed_expert_bytes,
            owned_expert_bytes: routed_expert_bytes,
            routed_expert_residency: RoutedExpertResidency::FullyResident,
            replicated_parameter_bytes: local_parameter_bytes - routed_expert_bytes,
            opened_checkpoint_shards,
            exchange_strategy: ExpertExchangeStrategy::ReplicatedInputAllSum,
        },
        architecture: ExpertArchitecture::KimiLinear(Box::new(model)),
        expert_cache: None,
        latest_statistics: Default::default(),
        cumulative_statistics: Default::default(),
    })
}

#[allow(clippy::too_many_arguments)]
fn load_gguf_ep(
    architecture: &str,
    gguf_file: &Path,
    checkpoint: &GgufCheckpoint,
    metadata: std::collections::HashMap<String, GgufMetadataValue>,
    topology: ParallelTopology,
    options: ModelLoadOptions,
    assignment: Option<ExpertAssignment>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    if topology.tensor_parallel_size > 1
        && (!matches!(
            architecture,
            "kimi-linear"
                | "deepseek2"
                | "inkling"
                | "qwen3moe"
                | "qwen3vlmoe"
                | "gpt-oss"
                | "gemma4"
                | "lfm2moe"
                | "nemotron_h_moe"
                | "qwen35moe"
                | "qwen3next"
        ) || !(options.weight_residency.is_fully_resident()
            || options.weight_residency.expert_cache().is_some()))
    {
        return Err(Error::Parallel(format!(
            "GGUF TP+EP preflight requires kimi-linear, deepseek2, inkling, qwen3moe, qwen3vlmoe, gpt-oss, gemma4, lfm2moe, nemotron_h_moe, qwen35moe, or qwen3next with fully resident weights or non-expert residency plus an independent expert cache, got architecture {architecture} and residency {:?}",
            options.weight_residency
        )));
    }
    crate::api::validate_gguf_quantization_source(checkpoint, &metadata, options.quantization)?;
    let inkling_mmproj = if architecture == "inkling" {
        inkling::open_sibling_mmproj(gguf_file)?
    } else {
        None
    };
    let qwen3_vl_mmproj = if architecture == "qwen3vlmoe" {
        Some(qwen3_vl::find_qwen3_vl_mmproj(gguf_file)?)
    } else {
        None
    };
    let qwen35_mmproj = if architecture == "qwen35moe" {
        qwen3_5::open_sibling_mmproj(gguf_file)?
    } else {
        None
    };
    if let Some(expert_options) = options.weight_residency.expert_cache() {
        reject_external_ep_quantization(options.quantization)?;
        return load_external_gguf_ep(
            architecture,
            checkpoint,
            &metadata,
            inkling_mmproj.as_ref(),
            qwen3_vl_mmproj.as_deref(),
            qwen35_mmproj.as_ref(),
            topology,
            assignment,
            options.weight_residency.layers(),
            ExternalExpertResidency::SparseCache(expert_options),
            options.weight_residency.max_mapped_shards(),
            stream,
            weights_stream,
        );
    }
    if !options.weight_residency.is_fully_resident() {
        return Err(Error::Parallel(
            "GGUF expert-parallel loading requires fully resident, sparse-expert-cache, or sparse-expert-cache-with-dense-layers residency"
                .into(),
        ));
    }
    if topology.tensor_parallel_size > 1 || matches!(architecture, "qwen3vlmoe" | "gemma4") {
        reject_external_ep_quantization(options.quantization)?;
        return load_external_gguf_ep(
            architecture,
            checkpoint,
            &metadata,
            inkling_mmproj.as_ref(),
            qwen3_vl_mmproj.as_deref(),
            qwen35_mmproj.as_ref(),
            topology,
            assignment,
            LayerWeightResidency::FullyResident,
            ExternalExpertResidency::FullyResident,
            crate::runtime::checkpoint::store::DEFAULT_MAX_MAPPED_SHARDS,
            stream,
            weights_stream,
        );
    }
    let opened_checkpoint_shards = checkpoint
        .catalog()
        .shards()
        .iter()
        .map(|shard| shard.path().to_path_buf())
        .collect::<Vec<_>>();
    match architecture {
        "kimi-linear" => {
            let loaded = kimi_linear::load_gguf_checkpoint(
                checkpoint,
                metadata,
                options.quantization,
                stream,
                weights_stream,
            )?;
            let mut model = loaded.model;
            if model.args.num_experts <= 0 {
                return Err(Error::Parallel(
                    "Kimi Linear GGUF config has no routed experts".into(),
                ));
            }
            let assignment =
                resolve_model_assignment(assignment, model.args.num_experts as usize, topology)?;
            let routed_expert_bytes = model.partition_routed_experts(&assignment, stream)?;
            let local_parameter_bytes = parameter_bytes(&model);
            Ok(finish_resident_gguf_ep(
                topology,
                ModelKind::KimiLinear,
                assignment,
                ExpertArchitecture::KimiLinear(Box::new(model)),
                local_parameter_bytes,
                routed_expert_bytes,
                opened_checkpoint_shards,
            ))
        }
        "deepseek2" => {
            let loaded = deepseek_v3::load_gguf_checkpoint(
                checkpoint,
                metadata,
                options.quantization,
                stream,
                weights_stream,
            )?;
            let mut model = loaded.model;
            if model.args.n_routed_experts <= 0 {
                return Err(Error::Parallel(
                    "DeepSeek GGUF config has no routed experts".into(),
                ));
            }
            let assignment = resolve_model_assignment(
                assignment,
                model.args.n_routed_experts as usize,
                topology,
            )?;
            let mut routed_expert_bytes = 0;
            for layer in &mut model.model.layers {
                if let Some(moe) = layer.mlp.moe_mut() {
                    routed_expert_bytes +=
                        finalize_deepseek_expert_bank(&mut moe.experts, &assignment, stream)?;
                }
            }
            let local_parameter_bytes = parameter_bytes(&model);
            Ok(finish_resident_gguf_ep(
                topology,
                ModelKind::DeepSeekV3,
                assignment,
                ExpertArchitecture::DeepSeek(Box::new(model)),
                local_parameter_bytes,
                routed_expert_bytes,
                opened_checkpoint_shards,
            ))
        }
        "qwen3moe" => {
            let loaded = dense_qwen::load_gguf_checkpoint(
                checkpoint,
                metadata,
                options.quantization,
                stream,
                weights_stream,
            )?;
            let mut model = loaded.model;
            let assignment =
                resolve_model_assignment(assignment, model.args.num_experts as usize, topology)?;
            let mut routed_expert_bytes = 0;
            for layer in &mut model.model.layers {
                if let dense_qwen::FeedForward::Moe(moe) = &mut layer.mlp {
                    routed_expert_bytes +=
                        finalize_qwen3_expert_bank(&mut moe.experts, &assignment, stream)?;
                }
            }
            let local_parameter_bytes = parameter_bytes(&model);
            Ok(finish_resident_gguf_ep(
                topology,
                ModelKind::Qwen3,
                assignment,
                ExpertArchitecture::DenseQwen(Box::new(model)),
                local_parameter_bytes,
                routed_expert_bytes,
                opened_checkpoint_shards,
            ))
        }
        other => Err(Error::Parallel(format!(
            "expert-parallel GGUF architecture {other} is unsupported; registered GGUF EP architectures are kimi-linear, deepseek2, inkling, qwen3moe, qwen3vlmoe, gpt-oss, gemma4, lfm2moe, nemotron_h_moe, qwen35moe, and qwen3next"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn load_external_gguf_ep(
    architecture: &str,
    checkpoint: &GgufCheckpoint,
    metadata: &std::collections::HashMap<String, GgufMetadataValue>,
    inkling_mmproj: Option<&inkling::InklingMmprojGguf>,
    qwen3_vl_mmproj: Option<&Path>,
    qwen35_mmproj: Option<&qwen3_5::Qwen35MmprojGguf>,
    topology: ParallelTopology,
    assignment: Option<ExpertAssignment>,
    non_expert: LayerWeightResidency,
    expert_residency: ExternalExpertResidency,
    max_mapped_shards: usize,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    match architecture {
        "kimi-linear" => {
            let prepared = kimi_linear::prepare_gguf_checkpoint(
                checkpoint,
                metadata,
                None,
                weights_stream,
            )?;
            let args = prepared.args;
            let assignment =
                resolve_model_assignment(assignment, args.num_experts as usize, topology)?;
            let store: std::sync::Arc<dyn WeightStore + Send + Sync> =
                std::sync::Arc::new(GgufWeightStore::new_with_max_mapped_shards(
                    checkpoint.clone(),
                    kimi_linear::translate_gguf_weight_name,
                    max_mapped_shards,
                )?);
            let model = if topology.tensor_parallel_size > 1 {
                crate::architectures::kimi_linear::layerwise::
                    load_kimi_linear_sparse_tp_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        ParallelBuildContext::new(topology, ShardingPolicy::Require),
                        stream,
                        weights_stream,
                    )?
            } else {
                crate::architectures::kimi_linear::layerwise::
                    load_kimi_linear_sparse_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        stream,
                        weights_stream,
                    )?
            };
            let entries =
                crate::architectures::kimi_linear::layerwise::kimi_expert_catalog(
                    &args,
                    store.as_ref(),
                )?;
            let replicated_parameter_bytes =
                planned_replicated_bytes(&model.residency_report()?)?;
            finish_external_ep(
                topology,
                ModelKind::KimiLinear,
                assignment,
                ExpertArchitecture::KimiLinearLayerwise(Box::new(model)),
                store,
                entries,
                expert_residency,
                replicated_parameter_bytes,
                stream,
                weights_stream,
            )
        }
        "deepseek2" => {
            let prepared = deepseek_v3::prepare_gguf_checkpoint(
                checkpoint,
                metadata,
                None,
                weights_stream,
            )?;
            let args = prepared.args;
            let assignment = resolve_model_assignment(
                assignment,
                args.n_routed_experts as usize,
                topology,
            )?;
            let store: std::sync::Arc<dyn WeightStore + Send + Sync> =
                std::sync::Arc::new(GgufWeightStore::new_with_max_mapped_shards(
                    checkpoint.clone(),
                    deepseek_v3::translate_gguf_weight_name,
                    max_mapped_shards,
                )?);
            let model = if topology.tensor_parallel_size > 1 {
                crate::architectures::deepseek_v3::layerwise::
                    load_deepseek_v3_sparse_tp_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        ParallelBuildContext::new(topology, ShardingPolicy::Require),
                        stream,
                        weights_stream,
                    )?
            } else {
                crate::architectures::deepseek_v3::layerwise::
                    load_deepseek_v3_sparse_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        stream,
                        weights_stream,
                    )?
            };
            let entries =
                crate::architectures::deepseek_v3::layerwise::deepseek_expert_catalog(
                    &args,
                    store.as_ref(),
                )?;
            let replicated_parameter_bytes =
                planned_replicated_bytes(&model.residency_report()?)?;
            finish_external_ep(
                topology,
                ModelKind::DeepSeekV3,
                assignment,
                ExpertArchitecture::DeepSeekLayerwise(Box::new(model)),
                store,
                entries,
                expert_residency,
                replicated_parameter_bytes,
                stream,
                weights_stream,
            )
        }
        "qwen3moe" => {
            let (args, _) = dense_qwen::prepare_gguf_checkpoint(
                checkpoint,
                metadata,
                architecture,
                true,
            )?;
            let assignment =
                resolve_model_assignment(assignment, args.num_experts as usize, topology)?;
            let store: std::sync::Arc<dyn WeightStore + Send + Sync> =
                std::sync::Arc::new(GgufWeightStore::new_with_max_mapped_shards(
                    checkpoint.clone(),
                    |name| dense_qwen::translate_gguf_weight_name(name, true),
                    max_mapped_shards,
                )?);
            let model = if topology.tensor_parallel_size > 1 {
                crate::architectures::qwen::dense::layerwise::
                    load_qwen3_sparse_tp_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        ParallelBuildContext::new(topology, ShardingPolicy::Require),
                        stream,
                        weights_stream,
                    )?
            } else {
                crate::architectures::qwen::dense::layerwise::
                    load_qwen3_sparse_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        stream,
                        weights_stream,
                    )?
            };
            let entries =
                crate::architectures::qwen::dense::layerwise::qwen3_expert_catalog(
                    &args,
                    store.as_ref(),
                )?;
            let replicated_parameter_bytes =
                planned_replicated_bytes(&model.residency_report()?)?;
            finish_external_ep(
                topology,
                ModelKind::Qwen3,
                assignment,
                ExpertArchitecture::DenseQwenLayerwise(Box::new(model)),
                store,
                entries,
                expert_residency,
                replicated_parameter_bytes,
                stream,
                weights_stream,
            )
        }
        "qwen3vlmoe" => {
            let vision_path = qwen3_vl_mmproj.ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "Qwen3-VL-MoE GGUF expert parallelism requires a sibling qwen3vl mmproj"
                        .into(),
                )
            })?;
            let vision_checkpoint = GgufCheckpoint::open(vision_path)?;
            let vision_metadata =
                crate::runtime::checkpoint::load::gguf_metadata(&vision_checkpoint);
            let prepared = qwen3_vl::prepare_qwen3_vl_gguf_checkpoint(
                checkpoint,
                metadata,
                &vision_checkpoint,
                &vision_metadata,
            )?;
            let args = prepared.args;
            let assignment = resolve_model_assignment(
                assignment,
                args.text_config.num_experts as usize,
                topology,
            )?;
            let store = crate::architectures::qwen::vl::layerwise::qwen3_vl_gguf_store(
                checkpoint,
                &vision_checkpoint,
                &args,
                max_mapped_shards,
            )?;
            let model = if topology.tensor_parallel_size > 1 {
                crate::architectures::qwen::vl::layerwise::
                    load_qwen3_vl_sparse_tp_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        ParallelBuildContext::new(topology, ShardingPolicy::Require),
                        stream,
                        weights_stream,
                    )?
            } else {
                crate::architectures::qwen::vl::layerwise::
                    load_qwen3_vl_sparse_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        stream,
                        weights_stream,
                    )?
            };
            let entries =
                crate::architectures::qwen::dense::layerwise::qwen3_expert_catalog_at(
                    &args.text_config,
                    store.as_ref(),
                    "model.language_model.layers",
                )?;
            let replicated_parameter_bytes =
                planned_replicated_bytes(&model.residency_report()?)?;
            finish_external_ep(
                topology,
                ModelKind::Qwen3VlMoe,
                assignment,
                ExpertArchitecture::Qwen3VlLayerwise(Box::new(model)),
                store,
                entries,
                expert_residency,
                replicated_parameter_bytes,
                stream,
                weights_stream,
            )
        }
        "gemma4" => {
            let prepared = gemma4::prepare_gemma4_gguf_checkpoint(
                checkpoint,
                metadata,
                None,
                None,
            )?;
            let args = prepared.args;
            let global_experts = args.num_experts.ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "Gemma 4 GGUF expert parallelism requires routed MoE layers".into(),
                )
            })?;
            let assignment = resolve_model_assignment(
                assignment,
                usize::try_from(global_experts)
                    .map_err(|_| Error::Parallel("Gemma 4 expert count is negative".into()))?,
                topology,
            )?;
            let store = crate::architectures::gemma4::layerwise::gemma4_gguf_store(
                checkpoint,
                None,
                max_mapped_shards,
            )?;
            let model = if topology.tensor_parallel_size > 1 {
                crate::architectures::gemma4::layerwise::
                    load_gemma4_sparse_tp_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        ParallelBuildContext::new(topology, ShardingPolicy::Require),
                        stream,
                        weights_stream,
                    )?
            } else {
                crate::architectures::gemma4::layerwise::load_gemma4_sparse_ep_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    stream,
                    weights_stream,
                )?
            };
            let entries = crate::architectures::gemma4::layerwise::gemma4_expert_catalog(
                &args,
                store.as_ref(),
            )?;
            let replicated_parameter_bytes =
                planned_replicated_bytes(&model.residency_report()?)?;
            finish_external_ep(
                topology,
                ModelKind::Gemma4,
                assignment,
                ExpertArchitecture::Gemma4Layerwise(Box::new(model)),
                store,
                entries,
                expert_residency,
                replicated_parameter_bytes,
                stream,
                weights_stream,
            )
        }
        "gpt-oss" => {
            let prepared =
                gpt_oss::prepare_gguf_checkpoint(checkpoint, metadata, weights_stream)?;
            let args = prepared.args;
            let assignment = resolve_model_assignment(
                assignment,
                args.num_local_experts as usize,
                topology,
            )?;
            let store: std::sync::Arc<dyn WeightStore + Send + Sync> =
                std::sync::Arc::new(GgufWeightStore::new_with_max_mapped_shards(
                    checkpoint.clone(),
                    gpt_oss::translate_gguf_weight_name,
                    max_mapped_shards,
                )?);
            let model = if topology.tensor_parallel_size > 1 {
                crate::architectures::gpt_oss::layerwise::
                    load_gpt_oss_sparse_tp_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        ParallelBuildContext::new(topology, ShardingPolicy::Require),
                        stream,
                        weights_stream,
                    )?
            } else {
                crate::architectures::gpt_oss::layerwise::
                    load_gpt_oss_sparse_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        stream,
                        weights_stream,
                    )?
            };
            let entries = crate::architectures::gpt_oss::layerwise::gpt_oss_expert_catalog(
                &args,
                store.as_ref(),
            )?;
            let replicated_parameter_bytes =
                planned_replicated_bytes(&model.residency_report()?)?;
            finish_external_ep(
                topology,
                ModelKind::GptOss,
                assignment,
                ExpertArchitecture::GptOssLayerwise(Box::new(model)),
                store,
                entries,
                expert_residency,
                replicated_parameter_bytes,
                stream,
                weights_stream,
            )
        }
        "inkling" => {
            let prepared = inkling::prepare_gguf_checkpoint_with_mmproj(
                checkpoint,
                metadata,
                inkling_mmproj,
            )?;
            let args = prepared.args;
            let assignment = resolve_model_assignment(
                assignment,
                args.text_config.n_routed_experts as usize,
                topology,
            )?;
            let store = crate::architectures::inkling::layerwise::inkling_gguf_store(
                checkpoint,
                inkling_mmproj,
                max_mapped_shards,
            )?;
            let model = if topology.tensor_parallel_size > 1 {
                crate::architectures::inkling::layerwise::
                    load_inkling_sparse_tp_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        ParallelBuildContext::new(topology, ShardingPolicy::Require),
                        stream,
                        weights_stream,
                    )?
            } else {
                crate::architectures::inkling::layerwise::
                    load_inkling_sparse_ep_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    stream,
                    weights_stream,
                )?
            };
            let entries = crate::architectures::inkling::layerwise::inkling_expert_catalog(
                &args,
                store.as_ref(),
            )?;
            let replicated_parameter_bytes =
                planned_replicated_bytes(&model.residency_report()?)?;
            finish_external_ep(
                topology,
                ModelKind::Inkling,
                assignment,
                ExpertArchitecture::InklingLayerwise(Box::new(model)),
                store,
                entries,
                expert_residency,
                replicated_parameter_bytes,
                stream,
                weights_stream,
            )
        }
        "lfm2moe" => {
            let prepared = lfm2::prepare_gguf_checkpoint(checkpoint, metadata, weights_stream)?;
            let args = prepared.args;
            let assignment =
                resolve_model_assignment(assignment, args.num_experts as usize, topology)?;
            let store: std::sync::Arc<dyn WeightStore + Send + Sync> =
                std::sync::Arc::new(GgufWeightStore::new_with_max_mapped_shards(
                    checkpoint.clone(),
                    |name| lfm2::translate_gguf_weight_name(name, true),
                    max_mapped_shards,
                )?);
            let model = if topology.tensor_parallel_size > 1 {
                crate::architectures::lfm2::layerwise::load_lfm2_sparse_tp_ep_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    ParallelBuildContext::new(topology, ShardingPolicy::Require),
                    stream,
                    weights_stream,
                )?
            } else {
                crate::architectures::lfm2::layerwise::load_lfm2_sparse_ep_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    stream,
                    weights_stream,
                )?
            };
            let entries =
                crate::architectures::lfm2::layerwise::lfm2_expert_catalog(
                    &args,
                    store.as_ref(),
                )?;
            let replicated_parameter_bytes =
                planned_replicated_bytes(&model.residency_report()?)?;
            finish_external_ep(
                topology,
                ModelKind::Lfm2,
                assignment,
                ExpertArchitecture::Lfm2Layerwise(Box::new(model)),
                store,
                entries,
                expert_residency,
                replicated_parameter_bytes,
                stream,
                weights_stream,
            )
        }
        "nemotron_h_moe" => {
            let prepared =
                nemotron_h::prepare_nemotron_h_gguf_checkpoint(checkpoint, metadata, weights_stream)?;
            let args = prepared.args;
            let assignment =
                resolve_model_assignment(assignment, args.n_routed_experts as usize, topology)?;
            let store: std::sync::Arc<dyn WeightStore + Send + Sync> =
                std::sync::Arc::new(GgufWeightStore::new_with_max_mapped_shards(
                    checkpoint.clone(),
                    nemotron_h::translate_gguf_weight_name,
                    max_mapped_shards,
                )?);
            let model = if topology.tensor_parallel_size > 1 {
                crate::architectures::nemotron_h::layerwise::
                    load_nemotron_h_sparse_tp_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        ParallelBuildContext::new(topology, ShardingPolicy::Require),
                        stream,
                        weights_stream,
                    )?
            } else {
                crate::architectures::nemotron_h::layerwise::
                    load_nemotron_h_sparse_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        stream,
                        weights_stream,
                    )?
            };
            let entries =
                crate::architectures::nemotron_h::layerwise::nemotron_h_expert_catalog(
                    &args,
                    store.as_ref(),
                )?;
            let replicated_parameter_bytes =
                planned_replicated_bytes(&model.residency_report()?)?;
            finish_external_ep(
                topology,
                ModelKind::NemotronH,
                assignment,
                ExpertArchitecture::NemotronHLayerwise(Box::new(model)),
                store,
                entries,
                expert_residency,
                replicated_parameter_bytes,
                stream,
                weights_stream,
            )
        }
        "qwen35moe" | "qwen3next" => {
            let prepared = qwen3_5::prepare_qwen35_gguf_checkpoint(
                checkpoint,
                metadata,
                qwen35_mmproj,
                weights_stream,
            )?;
            let args = prepared.args;
            let is_next = prepared.architecture == "qwen3next";
            let kind = if is_next {
                ModelKind::Qwen3Next
            } else {
                ModelKind::Qwen35
            };
            let assignment =
                resolve_model_assignment(assignment, args.num_experts as usize, topology)?;
            let store = crate::architectures::qwen::hybrid::layerwise::qwen_hybrid_gguf_store(
                checkpoint,
                qwen35_mmproj,
                prepared.modalities.vision_config.as_ref(),
                max_mapped_shards,
            )?;
            let model = if topology.tensor_parallel_size > 1 {
                crate::architectures::qwen::hybrid::layerwise::
                    load_qwen_hybrid_sparse_tp_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        is_next,
                        prepared.modalities.clone(),
                        non_expert,
                        ParallelBuildContext::new(topology, ShardingPolicy::Require),
                        stream,
                        weights_stream,
                    )?
            } else {
                crate::architectures::qwen::hybrid::layerwise::
                    load_qwen_hybrid_sparse_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        is_next,
                        prepared.modalities.clone(),
                        non_expert,
                        stream,
                        weights_stream,
                    )?
            };
            let entries =
                crate::architectures::qwen::hybrid::layerwise::qwen_hybrid_expert_catalog(
                    &args,
                    store.as_ref(),
                )?;
            let replicated_parameter_bytes =
                planned_replicated_bytes(&model.residency_report()?)?;
            finish_external_ep(
                topology,
                kind,
                assignment,
                ExpertArchitecture::QwenHybridLayerwise(Box::new(model)),
                store,
                entries,
                expert_residency,
                replicated_parameter_bytes,
                stream,
                weights_stream,
            )
        }
        other => Err(Error::Parallel(format!(
            "external-expert GGUF architecture {other} is unsupported; registered architectures are kimi-linear, deepseek2, gemma4, gpt-oss, inkling, qwen3moe, qwen3vlmoe, lfm2moe, nemotron_h_moe, qwen35moe, and qwen3next"
        ))),
    }
}

fn finish_resident_gguf_ep(
    topology: ParallelTopology,
    model_kind: ModelKind,
    assignment: ExpertAssignment,
    architecture: ExpertArchitecture,
    local_parameter_bytes: usize,
    routed_expert_bytes: usize,
    opened_checkpoint_shards: Vec<PathBuf>,
) -> ExpertParallelModel {
    ExpertParallelModel {
        topology,
        info: ExpertParallelInfo {
            topology,
            global_rank: topology.global_rank,
            expert_parallel_rank: topology.expert_parallel_rank,
            expert_parallel_size: topology.expert_parallel_size,
            model_kind,
            assignment,
            local_parameter_bytes,
            routed_expert_bytes,
            owned_expert_bytes: routed_expert_bytes,
            routed_expert_residency: RoutedExpertResidency::FullyResident,
            replicated_parameter_bytes: local_parameter_bytes - routed_expert_bytes,
            opened_checkpoint_shards,
            exchange_strategy: ExpertExchangeStrategy::ReplicatedInputAllSum,
        },
        architecture,
        expert_cache: None,
        latest_statistics: Default::default(),
        cumulative_statistics: Default::default(),
    }
}

fn planned_replicated_bytes(
    report: &crate::runtime::residency::manager::ResidencyReport,
) -> Result<usize, Error> {
    let bytes = report.units().iter().try_fold(0u64, |total, unit| {
        total
            .checked_add(unit.expected_bytes())
            .ok_or_else(|| Error::Parallel("streamed replicated byte total overflowed".into()))
    })?;
    usize::try_from(bytes)
        .map_err(|_| Error::Parallel("streamed replicated bytes exceed usize".into()))
}

#[allow(clippy::too_many_arguments)]
fn finish_external_ep(
    topology: ParallelTopology,
    kind: ModelKind,
    assignment: ExpertAssignment,
    mut architecture: ExpertArchitecture,
    store: std::sync::Arc<dyn WeightStore + Send + Sync>,
    entries: Vec<ExpertCatalogEntry>,
    residency: ExternalExpertResidency,
    replicated_parameter_bytes: usize,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    architecture.bind_parallel_topology(topology);
    let routed_expert_residency = residency.kind();
    let (expert_cache, owned_expert_bytes) = match residency {
        ExternalExpertResidency::FullyResident => rank_owned_resident_experts(
            store.clone(),
            entries,
            &assignment,
            stream,
            weights_stream,
        )?,
        ExternalExpertResidency::SparseCache(options) => rank_owned_expert_cache(
            store.clone(),
            entries,
            &assignment,
            options,
            stream,
            weights_stream,
        )?,
    };
    let opened_checkpoint_shards = store.diagnostics()?.touched_shard_paths;
    Ok(finish_additional_cached_ep(
        topology,
        kind,
        assignment,
        architecture,
        expert_cache,
        owned_expert_bytes,
        replicated_parameter_bytes,
        routed_expert_residency,
        opened_checkpoint_shards,
    ))
}

#[derive(Debug, Clone, Copy)]
enum ExternalExpertResidency {
    FullyResident,
    SparseCache(ExpertCacheLoadOptions),
}

impl ExternalExpertResidency {
    const fn kind(self) -> RoutedExpertResidency {
        match self {
            Self::FullyResident => RoutedExpertResidency::FullyResident,
            Self::SparseCache(_) => RoutedExpertResidency::SparseCache,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn load_kimi_linear_external_ep(
    model_dir: &Path,
    topology: ParallelTopology,
    options: ModelLoadOptions,
    non_expert: LayerWeightResidency,
    expert_residency: ExternalExpertResidency,
    max_mapped_shards: usize,
    assignment: Option<ExpertAssignment>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    reject_external_ep_quantization(options.quantization)?;
    let args = kimi_linear::get_model_args(model_dir)?;
    args.validate()?;
    let assignment = resolve_model_assignment(assignment, args.num_experts as usize, topology)?;
    let store: std::sync::Arc<dyn WeightStore + Send + Sync> = std::sync::Arc::new(
        SafetensorsWeightStore::open_with_max_mapped_shards(model_dir, max_mapped_shards)?,
    );
    let model = if topology.tensor_parallel_size > 1 {
        crate::architectures::kimi_linear::layerwise::load_kimi_linear_sparse_tp_ep_base_with_store(
            store.clone(),
            args.clone(),
            non_expert,
            ParallelBuildContext::new(topology, ShardingPolicy::Require),
            stream,
            weights_stream,
        )?
    } else {
        crate::architectures::kimi_linear::layerwise::load_kimi_linear_sparse_ep_base_with_store(
            store.clone(),
            args.clone(),
            non_expert,
            stream,
            weights_stream,
        )?
    };
    let entries =
        crate::architectures::kimi_linear::layerwise::kimi_expert_catalog(&args, store.as_ref())?;
    let replicated_parameter_bytes = planned_replicated_bytes(&model.residency_report()?)?;
    finish_external_ep(
        topology,
        ModelKind::KimiLinear,
        assignment,
        ExpertArchitecture::KimiLinearLayerwise(Box::new(model)),
        store,
        entries,
        expert_residency,
        replicated_parameter_bytes,
        stream,
        weights_stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn load_deepseek_external_ep(
    model_dir: &Path,
    topology: ParallelTopology,
    options: ModelLoadOptions,
    non_expert: LayerWeightResidency,
    expert_residency: ExternalExpertResidency,
    max_mapped_shards: usize,
    assignment: Option<ExpertAssignment>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    reject_external_ep_quantization(options.quantization)?;
    let args = deepseek_v3::get_model_args(model_dir)?;
    args.validate()?;
    let assignment =
        resolve_model_assignment(assignment, args.n_routed_experts as usize, topology)?;
    let store: std::sync::Arc<dyn WeightStore + Send + Sync> = std::sync::Arc::new(
        SafetensorsWeightStore::open_with_max_mapped_shards(model_dir, max_mapped_shards)?,
    );
    let model = if topology.tensor_parallel_size > 1 {
        crate::architectures::deepseek_v3::layerwise::load_deepseek_v3_sparse_tp_ep_base_with_store(
            store.clone(),
            args.clone(),
            non_expert,
            ParallelBuildContext::new(topology, ShardingPolicy::Require),
            stream,
            weights_stream,
        )?
    } else {
        crate::architectures::deepseek_v3::layerwise::load_deepseek_v3_sparse_ep_base_with_store(
            store.clone(),
            args.clone(),
            non_expert,
            stream,
            weights_stream,
        )?
    };
    let entries = crate::architectures::deepseek_v3::layerwise::deepseek_expert_catalog(
        &args,
        store.as_ref(),
    )?;
    let replicated_parameter_bytes = planned_replicated_bytes(&model.residency_report()?)?;
    finish_external_ep(
        topology,
        ModelKind::DeepSeekV3,
        assignment,
        ExpertArchitecture::DeepSeekLayerwise(Box::new(model)),
        store,
        entries,
        expert_residency,
        replicated_parameter_bytes,
        stream,
        weights_stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn load_qwen3_external_ep(
    model_dir: &Path,
    topology: ParallelTopology,
    options: ModelLoadOptions,
    non_expert: LayerWeightResidency,
    expert_residency: ExternalExpertResidency,
    max_mapped_shards: usize,
    assignment: Option<ExpertAssignment>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    reject_external_ep_quantization(options.quantization)?;
    let args = dense_qwen::load_config(model_dir)?;
    if !args.is_moe() {
        return Err(Error::Parallel(
            "Qwen3 config is dense and has no routed experts".into(),
        ));
    }
    let assignment = resolve_model_assignment(assignment, args.num_experts as usize, topology)?;
    let store: std::sync::Arc<dyn WeightStore + Send + Sync> = std::sync::Arc::new(
        SafetensorsWeightStore::open_with_max_mapped_shards(model_dir, max_mapped_shards)?,
    );
    let model = if topology.tensor_parallel_size > 1 {
        crate::architectures::qwen::dense::layerwise::load_qwen3_sparse_tp_ep_base_with_store(
            store.clone(),
            args.clone(),
            non_expert,
            ParallelBuildContext::new(topology, ShardingPolicy::Require),
            stream,
            weights_stream,
        )?
    } else {
        crate::architectures::qwen::dense::layerwise::load_qwen3_sparse_ep_base_with_store(
            store.clone(),
            args.clone(),
            non_expert,
            stream,
            weights_stream,
        )?
    };
    let entries =
        crate::architectures::qwen::dense::layerwise::qwen3_expert_catalog(&args, store.as_ref())?;
    let replicated_parameter_bytes = planned_replicated_bytes(&model.residency_report()?)?;
    finish_external_ep(
        topology,
        ModelKind::Qwen3,
        assignment,
        ExpertArchitecture::DenseQwenLayerwise(Box::new(model)),
        store,
        entries,
        expert_residency,
        replicated_parameter_bytes,
        stream,
        weights_stream,
    )
}

fn reject_external_ep_quantization(quantization: Option<WeightQuantization>) -> Result<(), Error> {
    if quantization.is_some() {
        return Err(Error::Quantization(
            "load-time quantization is unsupported with external-expert parallelism; use checkpoint-native weights"
                .into(),
        ));
    }
    Ok(())
}

fn load_deepseek_ep(
    model_dir: &Path,
    topology: ParallelTopology,
    options: ModelLoadOptions,
    assignment: Option<ExpertAssignment>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    if let Some(expert_options) = options.weight_residency.expert_cache() {
        return load_deepseek_external_ep(
            model_dir,
            topology,
            options,
            options.weight_residency.layers(),
            ExternalExpertResidency::SparseCache(expert_options),
            options.weight_residency.max_mapped_shards(),
            assignment,
            stream,
            weights_stream,
        );
    }
    if !options.weight_residency.is_fully_resident() {
        return Err(Error::Parallel(
            "expert-parallel loading accepts fully resident weights or non-expert residency plus an independent expert cache"
                .into(),
        ));
    }
    if topology.tensor_parallel_size > 1 {
        return load_deepseek_external_ep(
            model_dir,
            topology,
            options,
            LayerWeightResidency::FullyResident,
            ExternalExpertResidency::FullyResident,
            crate::runtime::checkpoint::store::DEFAULT_MAX_MAPPED_SHARDS,
            assignment,
            stream,
            weights_stream,
        );
    }
    let source_args = deepseek_v3::get_model_args(model_dir)?;
    if source_args.n_routed_experts <= 0 {
        return Err(Error::Parallel(
            "DeepSeek config has no routed experts".into(),
        ));
    }
    if options.quantization.is_some() && source_args.native_fp8_config().is_some() {
        return Err(Error::Quantization(
            "native DeepSeek block-FP8 expert-parallel weights cannot be implicitly dequantized and requantized".into(),
        ));
    }
    let quantize_on_load = options
        .quantization
        .map(|requested| {
            should_quantize_on_load(
                "DeepSeek-V3 expert-parallel",
                source_args.affine_quantization()?,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let mut target_args = source_args.clone();
    if let Some(quantization) = quantize_on_load {
        target_args.quantization_config = None;
        target_args.quantization = Some(quantization);
    }
    let assignment =
        resolve_model_assignment(assignment, source_args.n_routed_experts as usize, topology)?;
    let store = SafetensorsWeightStore::open(model_dir)?;
    let (plan, _) = expert_placement_plan(&store, topology, &assignment)?;
    let mut strict = StrictLoadConfig::default();
    for index in 0..source_args.num_nextn_predict_layers {
        strict = strict.allow_unused_prefix(format!(
            "model.layers.{}.",
            source_args.num_hidden_layers + index
        ));
    }
    let partition =
        load_partition_from_store_on_streams(&store, &plan, weights_stream, stream, &strict)?;
    let opened_checkpoint_shards = partition.opened_shards().to_vec();
    let mut tensors = partition.into_tensors();
    let mut model = deepseek_v3::Model::new(target_args, stream)?;
    assign_module(&mut model, "", &mut tensors, quantize_on_load, stream)?;
    for layer_index in 0..source_args.layer_schedule.len() {
        let Some(moe) = model.model.layers[layer_index].mlp.moe_mut() else {
            continue;
        };
        let mut localized = Vec::new();
        for name in tensors.keys() {
            if name.starts_with(&format!("model.layers.{layer_index}.mlp.experts.")) {
                if let Some(local) = localize_split_expert_name(name, &assignment) {
                    localized.push((name.clone(), local));
                }
            }
        }
        let localized = localized
            .into_iter()
            .map(|(global, local)| {
                let value = tensors.remove(&global).expect("listed local expert tensor");
                (local, value)
            })
            .collect::<Vec<_>>();
        for (local, value) in localized {
            tensors.insert(local, value);
        }
        load_deepseek_experts(
            moe,
            layer_index,
            (
                assignment.local_expert_count() as i32,
                source_args.hidden_size,
                source_args.moe_intermediate_size,
            ),
            &mut tensors,
            quantize_on_load,
            stream,
        )?;
        moe.experts.num_experts = assignment.local_expert_count() as i32;
    }
    if !tensors.is_empty() {
        let mut unused = tensors.into_keys().collect::<Vec<_>>();
        unused.sort();
        return Err(Error::StrictLoadValidation {
            missing: Vec::new(),
            unused,
        });
    }
    let mut routed_expert_bytes = 0;
    for layer in &mut model.model.layers {
        if let Some(moe) = layer.mlp.moe_mut() {
            routed_expert_bytes +=
                finalize_deepseek_expert_bank(&mut moe.experts, &assignment, stream)?;
        }
    }
    let local_parameter_bytes = parameter_bytes(&model);
    Ok(ExpertParallelModel {
        topology,
        info: ExpertParallelInfo {
            topology,
            global_rank: topology.global_rank,
            expert_parallel_rank: topology.expert_parallel_rank,
            expert_parallel_size: topology.expert_parallel_size,
            model_kind: ModelKind::DeepSeekV3,
            assignment,
            local_parameter_bytes,
            routed_expert_bytes,
            owned_expert_bytes: routed_expert_bytes,
            routed_expert_residency: RoutedExpertResidency::FullyResident,
            replicated_parameter_bytes: local_parameter_bytes - routed_expert_bytes,
            opened_checkpoint_shards,
            exchange_strategy: ExpertExchangeStrategy::ReplicatedInputAllSum,
        },
        architecture: ExpertArchitecture::DeepSeek(Box::new(model)),
        expert_cache: None,
        latest_statistics: Default::default(),
        cumulative_statistics: Default::default(),
    })
}

fn load_qwen3_ep(
    model_dir: &Path,
    topology: ParallelTopology,
    options: ModelLoadOptions,
    assignment: Option<ExpertAssignment>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    if let Some(expert_options) = options.weight_residency.expert_cache() {
        return load_qwen3_external_ep(
            model_dir,
            topology,
            options,
            options.weight_residency.layers(),
            ExternalExpertResidency::SparseCache(expert_options),
            options.weight_residency.max_mapped_shards(),
            assignment,
            stream,
            weights_stream,
        );
    }
    if !options.weight_residency.is_fully_resident() {
        return Err(Error::Parallel(
            "expert-parallel loading accepts fully resident weights or non-expert residency plus an independent expert cache"
                .into(),
        ));
    }
    if topology.tensor_parallel_size > 1 {
        return load_qwen3_external_ep(
            model_dir,
            topology,
            options,
            LayerWeightResidency::FullyResident,
            ExternalExpertResidency::FullyResident,
            crate::runtime::checkpoint::store::DEFAULT_MAX_MAPPED_SHARDS,
            assignment,
            stream,
            weights_stream,
        );
    }
    let source_args = dense_qwen::load_config(model_dir)?;
    if source_args.num_experts <= 0 {
        return Err(Error::Parallel(
            "Qwen3 config is dense and has no routed experts".into(),
        ));
    }
    let source_quantization = source_args.quantization.or(source_args.quantization_config);
    let quantize_on_load = options
        .quantization
        .map(|requested| {
            should_quantize_on_load("Qwen3 expert-parallel", source_quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let mut target_args = source_args.clone();
    if let Some(quantization) = quantize_on_load {
        target_args.quantization = Some(quantization);
        target_args.quantization_config = None;
    }
    let assignment =
        resolve_model_assignment(assignment, source_args.num_experts as usize, topology)?;
    let store = SafetensorsWeightStore::open(model_dir)?;
    let (plan, has_split) = expert_placement_plan(&store, topology, &assignment)?;
    let partition = load_partition_from_store_on_streams(
        &store,
        &plan,
        weights_stream,
        stream,
        &StrictLoadConfig::default(),
    )?;
    let opened_checkpoint_shards = partition.opened_shards().to_vec();
    let mut tensors = partition.into_tensors();
    if has_split {
        let mut localized = std::collections::HashMap::new();
        for (name, value) in tensors {
            if split_expert_id(&name).is_some() {
                if let Some(local) = localize_split_expert_name(&name, &assignment) {
                    localized.insert(local, value);
                }
            } else {
                localized.insert(name, value);
            }
        }
        tensors = transform_split_swiglu_experts(
            localized,
            assignment.local_expert_count() as i32,
            stream,
        )?;
    }
    if let Some(quantization) = quantize_on_load {
        quantize_qwen3_local_experts(
            &mut tensors,
            source_args.num_hidden_layers,
            quantization,
            stream,
        )?;
    }
    let mut model = dense_qwen::Model::new(target_args.clone(), stream)?;
    for (layer_index, layer) in model.model.layers.iter_mut().enumerate() {
        if let dense_qwen::FeedForward::Moe(moe) = &mut layer.mlp {
            let prefix = format!("model.layers.{layer_index}.mlp.experts");
            moe.experts = PackedSwiGluExperts::new(
                assignment.local_expert_count() as i32,
                source_args.hidden_size,
                source_args.moe_intermediate_size,
                target_args.weight_quantization_for(&format!("{prefix}.gate_up_proj")),
                target_args.weight_quantization_for(&format!("{prefix}.down_proj")),
                stream,
            )?;
        }
    }
    assign_module(&mut model, "", &mut tensors, quantize_on_load, stream)?;
    if !tensors.is_empty() {
        let mut unused = tensors.into_keys().collect::<Vec<_>>();
        unused.sort();
        return Err(Error::StrictLoadValidation {
            missing: Vec::new(),
            unused,
        });
    }
    let mut routed_expert_bytes = 0;
    for layer in &mut model.model.layers {
        if let dense_qwen::FeedForward::Moe(moe) = &mut layer.mlp {
            routed_expert_bytes +=
                finalize_qwen3_expert_bank(&mut moe.experts, &assignment, stream)?;
        }
    }
    let local_parameter_bytes = parameter_bytes(&model);
    Ok(ExpertParallelModel {
        topology,
        info: ExpertParallelInfo {
            topology,
            global_rank: topology.global_rank,
            expert_parallel_rank: topology.expert_parallel_rank,
            expert_parallel_size: topology.expert_parallel_size,
            model_kind: ModelKind::Qwen3,
            assignment,
            local_parameter_bytes,
            routed_expert_bytes,
            owned_expert_bytes: routed_expert_bytes,
            routed_expert_residency: RoutedExpertResidency::FullyResident,
            replicated_parameter_bytes: local_parameter_bytes - routed_expert_bytes,
            opened_checkpoint_shards,
            exchange_strategy: ExpertExchangeStrategy::ReplicatedInputAllSum,
        },
        architecture: ExpertArchitecture::DenseQwen(Box::new(model)),
        expert_cache: None,
        latest_statistics: Default::default(),
        cumulative_statistics: Default::default(),
    })
}

#[cfg(test)]
fn is_routed_expert_key(kind: ModelKind, key: &str) -> bool {
    match kind {
        ModelKind::KimiLinear => {
            key.contains(".mlp.experts.") || key.contains(".block_sparse_moe.experts.")
        }
        ModelKind::Lfm2 => key.contains(".feed_forward.experts."),
        ModelKind::NemotronH => key.contains(".experts.") && !key.contains(".shared_experts."),
        ModelKind::Inkling => key.contains(".mlp.experts.") || key.contains(".moe.experts."),
        ModelKind::Qwen3Next | ModelKind::Qwen35 => is_qwen_hybrid_decoder_expert_key(key),
        _ => key.contains(".mlp.experts."),
    }
}

#[cfg(test)]
fn is_qwen_hybrid_decoder_expert_key(key: &str) -> bool {
    key.starts_with("model.layers.") && key.contains(".mlp.experts.")
}

#[cfg(test)]
fn is_auxiliary_checkpoint_key(kind: ModelKind, key: &str) -> bool {
    match kind {
        ModelKind::Inkling => key.starts_with("model.mtp."),
        ModelKind::Qwen3Next => key.starts_with("model.mtp."),
        _ => false,
    }
}

fn rank_owned_expert_cache(
    store: std::sync::Arc<dyn WeightStore + Send + Sync>,
    entries: Vec<ExpertCatalogEntry>,
    assignment: &ExpertAssignment,
    options: ExpertCacheLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(ExpertCache, usize), Error> {
    let entries = entries
        .into_iter()
        .filter(|entry| assignment.owner(entry.identity().global_expert) == Some(assignment.rank()))
        .collect::<Vec<_>>();
    let owned_expert_bytes =
        usize::try_from(entries.iter().map(ExpertCatalogEntry::bytes).sum::<u64>())
            .map_err(|_| Error::Parallel("owned expert bytes exceed usize".into()))?;
    let cache = ExpertCache::new_shared(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?;
    Ok((cache, owned_expert_bytes))
}

fn rank_owned_resident_experts(
    store: std::sync::Arc<dyn WeightStore + Send + Sync>,
    entries: Vec<ExpertCatalogEntry>,
    assignment: &ExpertAssignment,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(ExpertCache, usize), Error> {
    let entries = entries
        .into_iter()
        .filter(|entry| assignment.owner(entry.identity().global_expert) == Some(assignment.rank()))
        .collect::<Vec<_>>();
    let owned_expert_bytes =
        usize::try_from(entries.iter().map(ExpertCatalogEntry::bytes).sum::<u64>())
            .map_err(|_| Error::Parallel("owned expert bytes exceed usize".into()))?;
    let resident =
        ExpertCache::new_resident_shared(store, entries, weights_stream.clone(), stream.clone())?;
    Ok((resident, owned_expert_bytes))
}

#[allow(clippy::too_many_arguments)]
fn finish_additional_cached_ep(
    topology: ParallelTopology,
    kind: ModelKind,
    assignment: ExpertAssignment,
    architecture: ExpertArchitecture,
    expert_cache: ExpertCache,
    owned_expert_bytes: usize,
    replicated_parameter_bytes: usize,
    routed_expert_residency: RoutedExpertResidency,
    opened_checkpoint_shards: Vec<PathBuf>,
) -> ExpertParallelModel {
    let routed_expert_bytes = match routed_expert_residency {
        RoutedExpertResidency::FullyResident => owned_expert_bytes,
        RoutedExpertResidency::SparseCache => 0,
    };
    ExpertParallelModel {
        topology,
        info: ExpertParallelInfo {
            topology,
            global_rank: topology.global_rank,
            expert_parallel_rank: topology.expert_parallel_rank,
            expert_parallel_size: topology.expert_parallel_size,
            model_kind: kind,
            assignment,
            local_parameter_bytes: replicated_parameter_bytes.saturating_add(routed_expert_bytes),
            routed_expert_bytes,
            owned_expert_bytes,
            routed_expert_residency,
            replicated_parameter_bytes,
            opened_checkpoint_shards,
            exchange_strategy: ExpertExchangeStrategy::ReplicatedInputAllSum,
        },
        architecture,
        expert_cache: Some(expert_cache),
        latest_statistics: Default::default(),
        cumulative_statistics: Default::default(),
    }
}

fn open_external_safetensors_store(
    model_dir: &Path,
    max_mapped_shards: usize,
) -> Result<std::sync::Arc<dyn WeightStore + Send + Sync>, Error> {
    Ok(std::sync::Arc::new(
        SafetensorsWeightStore::open_with_max_mapped_shards(model_dir, max_mapped_shards)?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn load_additional_external_ep(
    model_dir: &Path,
    topology: ParallelTopology,
    options: ModelLoadOptions,
    non_expert: LayerWeightResidency,
    expert_residency: ExternalExpertResidency,
    max_mapped_shards: usize,
    assignment: Option<ExpertAssignment>,
    kind: ModelKind,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    reject_external_ep_quantization(options.quantization)?;
    let store = open_external_safetensors_store(model_dir, max_mapped_shards)?;
    let (assignment, architecture, entries, replicated_parameter_bytes) = match kind {
        ModelKind::Gemma4 => {
            let (args, _, _, _, _, _) = gemma4::get_gemma4_model_config(model_dir)?;
            let global_experts = args.num_experts.ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "expert parallelism requires Gemma 4 routed MoE text layers".into(),
                )
            })?;
            if !args
                .layer_schedule
                .iter()
                .any(|policy| policy.feed_forward == gemma4::FeedForwardPolicy::DenseWithSparseMoe)
            {
                return Err(Error::UnsupportedArchitecture(
                    "expert parallelism requires Gemma 4 routed MoE text layers".into(),
                ));
            }
            let assignment = resolve_model_assignment(
                assignment,
                usize::try_from(global_experts)
                    .map_err(|_| Error::Parallel("Gemma 4 expert count is negative".into()))?,
                topology,
            )?;
            let model = if topology.tensor_parallel_size > 1 {
                crate::architectures::gemma4::layerwise::load_gemma4_sparse_tp_ep_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    ParallelBuildContext::new(topology, ShardingPolicy::Require),
                    stream,
                    weights_stream,
                )?
            } else {
                crate::architectures::gemma4::layerwise::load_gemma4_sparse_ep_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    stream,
                    weights_stream,
                )?
            };
            let entries = crate::architectures::gemma4::layerwise::gemma4_expert_catalog(
                &args,
                store.as_ref(),
            )?;
            let replicated = planned_replicated_bytes(&model.residency_report()?)?;
            (
                assignment,
                ExpertArchitecture::Gemma4Layerwise(Box::new(model)),
                entries,
                replicated,
            )
        }
        ModelKind::GptOss => {
            let args = gpt_oss::get_model_args(model_dir)?;
            let assignment =
                resolve_model_assignment(assignment, args.num_local_experts as usize, topology)?;
            let model = if topology.tensor_parallel_size > 1 {
                crate::architectures::gpt_oss::layerwise::load_gpt_oss_sparse_tp_ep_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    ParallelBuildContext::new(topology, ShardingPolicy::Require),
                    stream,
                    weights_stream,
                )?
            } else {
                crate::architectures::gpt_oss::layerwise::load_gpt_oss_sparse_ep_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    stream,
                    weights_stream,
                )?
            };
            let entries = crate::architectures::gpt_oss::layerwise::gpt_oss_expert_catalog(
                &args,
                store.as_ref(),
            )?;
            let replicated = planned_replicated_bytes(&model.residency_report()?)?;
            (
                assignment,
                ExpertArchitecture::GptOssLayerwise(Box::new(model)),
                entries,
                replicated,
            )
        }
        ModelKind::Inkling => {
            let args = inkling::get_model_args(model_dir)?;
            let global_experts = usize::try_from(args.text_config.n_routed_experts)
                .map_err(|_| Error::Parallel("Inkling routed expert count is negative".into()))?;
            if global_experts == 0
                || !args.text_config.layer_schedule.iter().any(|policy| {
                    policy.feed_forward
                        == crate::architectures::inkling::model::FeedForwardPolicy::SparseMoe
                })
            {
                return Err(Error::UnsupportedArchitecture(
                    "expert parallelism requires an Inkling checkpoint with routed MoE layers"
                        .into(),
                ));
            }
            let assignment = resolve_model_assignment(assignment, global_experts, topology)?;
            let model = if topology.tensor_parallel_size > 1 {
                crate::architectures::inkling::layerwise::load_inkling_sparse_tp_ep_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    ParallelBuildContext::new(topology, ShardingPolicy::Require),
                    stream,
                    weights_stream,
                )?
            } else {
                crate::architectures::inkling::layerwise::load_inkling_sparse_ep_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    stream,
                    weights_stream,
                )?
            };
            let entries = crate::architectures::inkling::layerwise::inkling_expert_catalog(
                &args,
                store.as_ref(),
            )?;
            let replicated = planned_replicated_bytes(&model.residency_report()?)?;
            (
                assignment,
                ExpertArchitecture::InklingLayerwise(Box::new(model)),
                entries,
                replicated,
            )
        }
        ModelKind::Lfm2 => {
            let args = lfm2::get_model_args(model_dir)?;
            if !args.has_sparse_moe_layers() {
                return Err(Error::UnsupportedArchitecture(
                    "expert parallelism requires an LFM2 MoE checkpoint".into(),
                ));
            }
            let assignment =
                resolve_model_assignment(assignment, args.num_experts as usize, topology)?;
            let model = if topology.tensor_parallel_size > 1 {
                crate::architectures::lfm2::layerwise::load_lfm2_sparse_tp_ep_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    ParallelBuildContext::new(topology, ShardingPolicy::Require),
                    stream,
                    weights_stream,
                )?
            } else {
                crate::architectures::lfm2::layerwise::load_lfm2_sparse_ep_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    stream,
                    weights_stream,
                )?
            };
            let entries =
                crate::architectures::lfm2::layerwise::lfm2_expert_catalog(&args, store.as_ref())?;
            let replicated = planned_replicated_bytes(&model.residency_report()?)?;
            (
                assignment,
                ExpertArchitecture::Lfm2Layerwise(Box::new(model)),
                entries,
                replicated,
            )
        }
        ModelKind::NemotronH => {
            let args = nemotron_h::get_nemotron_h_model_args(model_dir)?;
            if !args
                .layer_schedule
                .iter()
                .any(|policy| *policy == nemotron_h::LayerPolicy::SparseMoe)
            {
                return Err(Error::UnsupportedArchitecture(
                    "expert parallelism requires a Nemotron-H MoE checkpoint".into(),
                ));
            }
            let assignment =
                resolve_model_assignment(assignment, args.n_routed_experts as usize, topology)?;
            let model = if topology.tensor_parallel_size > 1 {
                crate::architectures::nemotron_h::layerwise::
                    load_nemotron_h_sparse_tp_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        ParallelBuildContext::new(topology, ShardingPolicy::Require),
                        stream,
                        weights_stream,
                    )?
            } else {
                crate::architectures::nemotron_h::layerwise::
                    load_nemotron_h_sparse_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        stream,
                        weights_stream,
                    )?
            };
            let entries = crate::architectures::nemotron_h::layerwise::nemotron_h_expert_catalog(
                &args,
                store.as_ref(),
            )?;
            let replicated = planned_replicated_bytes(&model.residency_report()?)?;
            (
                assignment,
                ExpertArchitecture::NemotronHLayerwise(Box::new(model)),
                entries,
                replicated,
            )
        }
        ModelKind::Qwen3Next | ModelKind::Qwen35 => {
            let (args, modalities) = if kind == ModelKind::Qwen3Next {
                (
                    qwen3_next::get_qwen3_next_model_args(model_dir)?,
                    qwen3_5::Qwen35Modalities::default(),
                )
            } else {
                let (args, image_token_id, video_token_id, vision_config) =
                    qwen3_5::get_qwen3_5_model_args(model_dir)?;
                (
                    args,
                    qwen3_5::Qwen35Modalities {
                        image_token_id,
                        video_token_id,
                        vision_config,
                    },
                )
            };
            if !args.is_moe() {
                return Err(Error::UnsupportedArchitecture(format!(
                    "expert parallelism requires a {} MoE checkpoint",
                    kind.model_type_name()
                )));
            }
            if let Some(config) = &args.quantization_config {
                config.validate_supported()?;
            }
            let assignment =
                resolve_model_assignment(assignment, args.num_experts as usize, topology)?;
            let model = if topology.tensor_parallel_size > 1 {
                crate::architectures::qwen::hybrid::layerwise::
                    load_qwen_hybrid_sparse_tp_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        kind == ModelKind::Qwen3Next,
                        modalities.clone(),
                        non_expert,
                        ParallelBuildContext::new(topology, ShardingPolicy::Require),
                        stream,
                        weights_stream,
                    )?
            } else {
                crate::architectures::qwen::hybrid::layerwise::
                    load_qwen_hybrid_sparse_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        kind == ModelKind::Qwen3Next,
                        modalities.clone(),
                        non_expert,
                        stream,
                        weights_stream,
                    )?
            };
            let entries =
                crate::architectures::qwen::hybrid::layerwise::qwen_hybrid_expert_catalog(
                    &args,
                    store.as_ref(),
                )?;
            let replicated = planned_replicated_bytes(&model.residency_report()?)?;
            (
                assignment,
                ExpertArchitecture::QwenHybridLayerwise(Box::new(model)),
                entries,
                replicated,
            )
        }
        ModelKind::Qwen3VlMoe => {
            let args = qwen3_vl::get_qwen3_vl_model_args(model_dir)?;
            if !args.text_config.is_moe() {
                return Err(Error::UnsupportedArchitecture(
                    "expert parallelism requires a Qwen3-VL-MoE checkpoint".into(),
                ));
            }
            let assignment = resolve_model_assignment(
                assignment,
                args.text_config.num_experts as usize,
                topology,
            )?;
            let model = if topology.tensor_parallel_size > 1 {
                crate::architectures::qwen::vl::layerwise::
                    load_qwen3_vl_sparse_tp_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        ParallelBuildContext::new(topology, ShardingPolicy::Require),
                        stream,
                        weights_stream,
                    )?
            } else {
                crate::architectures::qwen::vl::layerwise::load_qwen3_vl_sparse_ep_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    stream,
                    weights_stream,
                )?
            };
            let entries = crate::architectures::qwen::dense::layerwise::qwen3_expert_catalog_at(
                &args.text_config,
                store.as_ref(),
                "model.language_model.layers",
            )?;
            let replicated = planned_replicated_bytes(&model.residency_report()?)?;
            (
                assignment,
                ExpertArchitecture::Qwen3VlLayerwise(Box::new(model)),
                entries,
                replicated,
            )
        }
        _ => {
            return Err(Error::UnsupportedArchitecture(format!(
                "{} is not an external-expert architecture",
                kind.model_type_name()
            )))
        }
    };
    finish_external_ep(
        topology,
        kind,
        assignment,
        architecture,
        store,
        entries,
        expert_residency,
        replicated_parameter_bytes,
        stream,
        weights_stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn load_additional_ep(
    model_dir: &Path,
    topology: ParallelTopology,
    options: ModelLoadOptions,
    assignment: Option<ExpertAssignment>,
    kind: ModelKind,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    let (non_expert, expert_residency, max_mapped_shards) = if let Some(expert_options) =
        options.weight_residency.expert_cache()
    {
        (
            options.weight_residency.layers(),
            ExternalExpertResidency::SparseCache(expert_options),
            options.weight_residency.max_mapped_shards(),
        )
    } else if options.weight_residency.is_fully_resident() {
        (
            LayerWeightResidency::FullyResident,
            ExternalExpertResidency::FullyResident,
            crate::runtime::checkpoint::store::DEFAULT_MAX_MAPPED_SHARDS,
        )
    } else {
        return Err(Error::Parallel(format!(
                "{} expert-parallel execution requires fully resident weights or independently cached experts, got {:?}",
                kind.model_type_name(), options.weight_residency
            )));
    };
    load_additional_external_ep(
        model_dir,
        topology,
        options,
        non_expert,
        expert_residency,
        max_mapped_shards,
        assignment,
        kind,
        stream,
        weights_stream,
    )
}

#[cfg(test)]
pub(crate) fn assert_rank_owned_sparse_ep_load(
    model_dir: &Path,
    expert_options: ExpertCacheLoadOptions,
    expected_kind: ModelKind,
    expected_owned_experts: usize,
    stream: &Stream,
    weights_stream: &Stream,
) {
    use crate::runtime::distributed::topology::DeviceAssignment;
    use crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions;
    use safemlx::DeviceType;

    let topology =
        ParallelTopology::from_rank(2, 1, 1, 1, 2, DeviceAssignment::new(DeviceType::Gpu, 0))
            .unwrap();
    let model = load_expert_parallel_model_with_options(
        model_dir,
        ModelLoadOptions {
            quantization: None,
            parallel: Some(topology),
            weight_residency: WeightResidency::with_expert_cache(
                crate::NonExpertWeightResidency::LayerwiseHost(Default::default()),
                expert_options,
            ),
        },
        stream,
        weights_stream,
    )
    .unwrap();
    assert_eq!(model.info().model_kind, expected_kind);
    assert_eq!(model.info().expert_parallel_rank, 1);
    assert_eq!(model.info().expert_parallel_size, 2);
    assert_eq!(model.info().routed_expert_bytes, 0);
    assert!(model.info().owned_expert_bytes > 0);
    assert_eq!(
        model.expert_cache_report().unwrap().unwrap().owned_experts,
        expected_owned_experts
    );

    let resident_non_experts = load_expert_parallel_model_with_options(
        model_dir,
        ModelLoadOptions {
            quantization: None,
            parallel: Some(topology),
            weight_residency: WeightResidency::with_expert_cache(
                crate::NonExpertWeightResidency::FullyResident,
                expert_options,
            ),
        },
        stream,
        weights_stream,
    )
    .unwrap();
    assert_eq!(resident_non_experts.info().model_kind, expected_kind);
    assert_eq!(resident_non_experts.info().routed_expert_bytes, 0);
    assert_eq!(
        resident_non_experts
            .expert_cache_report()
            .unwrap()
            .unwrap()
            .owned_experts,
        expected_owned_experts
    );

    let dense = DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1, 1).unwrap();
    let streamed = load_expert_parallel_model_with_options(
        model_dir,
        ModelLoadOptions {
            quantization: None,
            parallel: Some(topology),
            weight_residency: WeightResidency::with_expert_cache(
                crate::NonExpertWeightResidency::DenseDiskStream(dense),
                expert_options,
            ),
        },
        stream,
        weights_stream,
    )
    .unwrap();
    assert_eq!(streamed.info().model_kind, expected_kind);
    assert_eq!(streamed.info().routed_expert_bytes, 0);
    assert!(streamed.info().owned_expert_bytes > 0);
    assert_eq!(
        streamed
            .expert_cache_report()
            .unwrap()
            .unwrap()
            .owned_experts,
        expected_owned_experts
    );
    let dense = streamed.dense_stream_report().unwrap().unwrap();
    assert_eq!(dense.device_layers().current_layer_count(), 0);
    assert!(dense.planned_layer_count() > 0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::distributed::topology::DeviceAssignment;
    use safemlx::{
        distributed::Backend,
        module::ModuleParameters,
        ops::{indexing::TryIndexOp, zeros_dtype, GgufMetadataArray},
        Device, DeviceType, Dtype, ExecutionContext,
    };

    fn stream() -> Stream {
        Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
    }

    fn host_expert_residency(experts: ExpertCacheLoadOptions) -> WeightResidency {
        WeightResidency::with_expert_cache(
            crate::NonExpertWeightResidency::LayerwiseHost(Default::default()),
            experts,
        )
    }

    fn dense_expert_residency(
        experts: ExpertCacheLoadOptions,
        dense: crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions,
    ) -> WeightResidency {
        WeightResidency::with_expert_cache(
            crate::NonExpertWeightResidency::DenseDiskStream(dense),
            experts,
        )
    }

    #[test]
    fn timing_profiling_guard_restores_previous_state() {
        assert!(!timing_profiling_enabled());
        {
            let _outer = profile_expert_parallel_timings();
            assert!(timing_profiling_enabled());
            {
                let _inner = profile_expert_parallel_timings();
                assert!(timing_profiling_enabled());
            }
            assert!(timing_profiling_enabled());
        }
        assert!(!timing_profiling_enabled());
    }

    #[test]
    fn qwen_mtp_weights_are_replicated_while_decoder_experts_are_partitioned() {
        assert!(!is_auxiliary_checkpoint_key(
            ModelKind::Qwen3Next,
            "mtp.fc.weight"
        ));
        assert!(is_auxiliary_checkpoint_key(
            ModelKind::Qwen3Next,
            "model.mtp.fc.weight"
        ));
        assert!(!is_auxiliary_checkpoint_key(
            ModelKind::Qwen35,
            "mtp.fc.weight"
        ));
        assert!(is_auxiliary_checkpoint_key(
            ModelKind::Inkling,
            "model.mtp.fc.weight"
        ));
        assert!(!is_auxiliary_checkpoint_key(
            ModelKind::Inkling,
            "mtp.fc.weight"
        ));
        for kind in [ModelKind::Qwen3Next, ModelKind::Qwen35] {
            assert!(is_routed_expert_key(
                kind,
                "model.layers.0.mlp.experts.1.down_proj.weight"
            ));
            assert!(!is_routed_expert_key(
                kind,
                "mtp.layers.0.mlp.experts.1.down_proj.weight"
            ));
        }
    }

    fn save_zero_checkpoint(model: &impl ModuleParameters, directory: &Path, stream: &Stream) {
        let parameters = model.parameters().flatten();
        let arrays = parameters
            .iter()
            .map(|(name, parameter)| {
                (
                    name.to_string(),
                    zeros_dtype(parameter.shape(), parameter.dtype(), stream).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        Array::save_safetensors(
            arrays.iter().map(|(name, array)| (name.as_str(), array)),
            None,
            directory.join("model.safetensors"),
        )
        .unwrap();
    }

    fn rank_one_topology() -> ParallelTopology {
        ParallelTopology::from_rank(2, 1, 1, 1, 2, DeviceAssignment::new(DeviceType::Gpu, 0))
            .unwrap()
    }

    fn tensor_expert_topology(rank: usize) -> ParallelTopology {
        ParallelTopology::from_rank(4, rank, 2, 1, 2, DeviceAssignment::new(DeviceType::Gpu, 0))
            .unwrap()
    }

    fn kimi_test_args() -> kimi_linear::ModelArgs {
        kimi_linear::model_args_from_config_value(&serde_json::json!({
            "model_type": "kimi_linear",
            "vocab_size": 32,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "intermediate_size": 64,
            "head_dim": 8,
            "model_max_length": 128,
            "rms_norm_eps": 0.00001,
            "linear_attn_config": {
                "kda_layers": [1],
                "full_attn_layers": [2],
                "num_heads": 4,
                "head_dim": 8,
                "short_conv_kernel_size": 2
            },
            "num_experts": 4,
            "moe_intermediate_size": 64,
            "kv_lora_rank": 8,
            "q_lora_rank": null,
            "qk_nope_head_dim": 4,
            "qk_rope_head_dim": 4,
            "v_head_dim": 4,
            "mla_use_nope": true,
            "num_experts_per_token": 2,
            "num_shared_experts": 1,
            "moe_router_activation_func": "sigmoid",
            "moe_renormalize": true,
            "routed_scaling_factor": 1.0,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "use_grouped_topk": true,
            "num_expert_group": 1,
            "topk_group": 1,
            "tie_word_embeddings": false,
            "num_nextn_predict_layers": 0
        }))
        .unwrap()
    }

    fn kimi_runtime_to_gguf_name(name: &str) -> String {
        for (runtime, gguf) in [
            ("model.embed_tokens.weight", "token_embd.weight"),
            ("model.norm.weight", "output_norm.weight"),
            ("lm_head.weight", "output.weight"),
        ] {
            if name == runtime {
                return gguf.into();
            }
        }
        let rest = name
            .strip_prefix("model.layers.")
            .unwrap_or_else(|| panic!("unexpected Kimi parameter {name}"));
        let (layer, parameter) = rest
            .split_once('.')
            .unwrap_or_else(|| panic!("unexpected Kimi parameter {name}"));
        let parameter = match parameter {
            "self_attn.q_proj.weight" => "attn_q.weight",
            "self_attn.k_proj.weight" => "attn_k.weight",
            "self_attn.v_proj.weight" => "attn_v.weight",
            "self_attn.q_conv1d.weight" => "ssm_conv1d_q.weight",
            "self_attn.k_conv1d.weight" => "ssm_conv1d_k.weight",
            "self_attn.v_conv1d.weight" => "ssm_conv1d_v.weight",
            "self_attn.A_log" => "ssm_a.weight",
            "self_attn.dt_bias" => "ssm_dt.bias",
            "self_attn.f_a_proj.weight" => "ssm_f_a.weight",
            "self_attn.f_b_proj.weight" => "ssm_f_b.weight",
            "self_attn.b_proj.weight" => "ssm_beta.weight",
            "self_attn.g_a_proj.weight" => "ssm_g_a.weight",
            "self_attn.g_b_proj.weight" => "ssm_g_b.weight",
            "self_attn.o_norm.weight" => "ssm_norm.weight",
            "self_attn.kv_a_proj_with_mqa.weight" => "attn_kv_a_mqa.weight",
            "self_attn.kv_a_layernorm.weight" => "attn_kv_a_norm.weight",
            "self_attn.kv_b_proj.weight" => "attn_kv_b.weight",
            "self_attn.o_proj.weight" => "attn_output.weight",
            "input_layernorm.weight" => "attn_norm.weight",
            "post_attention_layernorm.weight" => "ffn_norm.weight",
            "mlp.gate_proj.weight" => "ffn_gate.weight",
            "mlp.up_proj.weight" => "ffn_up.weight",
            "mlp.down_proj.weight" => "ffn_down.weight",
            "mlp.shared_experts.gate_proj.weight" => "ffn_gate_shexp.weight",
            "mlp.shared_experts.up_proj.weight" => "ffn_up_shexp.weight",
            "mlp.shared_experts.down_proj.weight" => "ffn_down_shexp.weight",
            "mlp.gate.weight" => "ffn_gate_inp.weight",
            "mlp.gate.e_score_correction_bias" => "exp_probs_b.bias",
            _ => panic!("unexpected Kimi parameter {name}"),
        };
        format!("blk.{layer}.{parameter}")
    }

    fn synthetic_kimi_gguf(stream: &Stream) -> crate::test_utils::SyntheticGguf {
        let args = kimi_test_args();
        let source = kimi_linear::Model::new(args.clone(), stream).unwrap();
        let mut arrays = std::collections::HashMap::new();
        for (name, parameter) in source.parameters().flatten() {
            let value = if name.as_ref().ends_with(".A_log") {
                Array::full::<f32>(parameter.shape(), Array::from_f32(-1.0), stream).unwrap()
            } else {
                Array::zeros::<f32>(parameter.shape(), stream).unwrap()
            };
            if name.as_ref() == "model.layers.1.mlp.experts.gate_up_proj" {
                arrays.insert(
                    "blk.1.ffn_gate_exps.weight".into(),
                    value
                        .try_index_device((.., ..args.moe_intermediate_size, ..), stream)
                        .unwrap(),
                );
                arrays.insert(
                    "blk.1.ffn_up_exps.weight".into(),
                    value
                        .try_index_device((.., args.moe_intermediate_size.., ..), stream)
                        .unwrap(),
                );
            } else if name.as_ref() == "model.layers.1.mlp.experts.down_proj" {
                arrays.insert("blk.1.ffn_down_exps.weight".into(), value);
            } else {
                arrays.insert(kimi_runtime_to_gguf_name(name.as_ref()), value);
            }
        }
        let metadata = std::collections::HashMap::from([
            (
                "general.architecture".into(),
                GgufMetadataValue::String("kimi-linear".into()),
            ),
            (
                "kimi-linear.embedding_length".into(),
                GgufMetadataValue::Uint32(args.hidden_size as u32),
            ),
            (
                "kimi-linear.block_count".into(),
                GgufMetadataValue::Uint32(args.num_hidden_layers as u32),
            ),
            (
                "kimi-linear.feed_forward_length".into(),
                GgufMetadataValue::Uint32(args.intermediate_size as u32),
            ),
            (
                "kimi-linear.attention.head_count".into(),
                GgufMetadataValue::Uint32(args.num_attention_heads as u32),
            ),
            (
                "kimi-linear.attention.head_count_kv".into(),
                GgufMetadataValue::Array(GgufMetadataArray::Uint32(vec![0, 1])),
            ),
            (
                "kimi-linear.rope.dimension_count".into(),
                GgufMetadataValue::Uint32(args.qk_rope_head_dim as u32),
            ),
            (
                "kimi-linear.attention.key_length_mla".into(),
                GgufMetadataValue::Uint32((args.qk_nope_head_dim + args.qk_rope_head_dim) as u32),
            ),
            (
                "kimi-linear.attention.value_length_mla".into(),
                GgufMetadataValue::Uint32(args.v_head_dim as u32),
            ),
            (
                "kimi-linear.attention.kv_lora_rank".into(),
                GgufMetadataValue::Uint32(args.kv_lora_rank as u32),
            ),
            (
                "kimi-linear.attention.layer_norm_rms_epsilon".into(),
                GgufMetadataValue::Float32(args.rms_norm_eps),
            ),
            (
                "kimi-linear.context_length".into(),
                GgufMetadataValue::Uint32(args.model_max_length as u32),
            ),
            (
                "kimi-linear.kda.head_dim".into(),
                GgufMetadataValue::Uint32(args.kda_config.head_dim as u32),
            ),
            (
                "kimi-linear.ssm.conv_kernel".into(),
                GgufMetadataValue::Uint32(args.kda_config.short_conv_kernel_size as u32),
            ),
            (
                "kimi-linear.expert_count".into(),
                GgufMetadataValue::Uint32(args.num_experts as u32),
            ),
            (
                "kimi-linear.expert_feed_forward_length".into(),
                GgufMetadataValue::Uint32(args.moe_intermediate_size as u32),
            ),
            (
                "kimi-linear.expert_used_count".into(),
                GgufMetadataValue::Uint32(args.num_experts_per_token as u32),
            ),
            (
                "kimi-linear.vocab_size".into(),
                GgufMetadataValue::Uint32(args.vocab_size as u32),
            ),
        ]);
        crate::test_utils::SyntheticGguf::with_packed_tensors(&arrays, &metadata, |name, _| {
            (name.contains("ffn_") && name.contains("_exps.weight"))
                .then_some(safemlx_gguf::GgmlType::MxFp4)
        })
    }

    struct IdentityBank;

    impl LocalExpertBank for IdentityBank {
        fn execute_local_routes(
            &mut self,
            hidden: &Array,
            _local_expert_ids: &Array,
            _stream: &Stream,
        ) -> Result<Array, Error> {
            Ok(hidden.clone())
        }
    }

    #[test]
    fn assignment_policies_and_round_trips() {
        let balanced = ExpertAssignment::balanced(7, 3, 1).unwrap();
        assert_eq!(balanced.owners(), &[0, 0, 0, 1, 1, 2, 2]);
        assert_eq!(balanced.local_global_expert_ids(), &[3, 4]);
        assert_eq!(balanced.owner_local_id(4), Some(1));
        assert_eq!(balanced.global_id(1), Some(4));

        let rr = ExpertAssignment::round_robin(7, 3, 1).unwrap();
        assert_eq!(rr.local_global_expert_ids(), &[1, 4]);
        assert_eq!(rr.owner_local_id(4), Some(1));

        let explicit = ExpertAssignment::explicit(vec![1, 0, 1, 0], 2, 0).unwrap();
        assert_eq!(explicit.local_global_expert_ids(), &[1, 3]);
        assert_eq!(explicit.global_id(1), Some(3));
    }

    #[test]
    fn assignment_rejects_invalid_or_empty_ownership() {
        assert!(ExpertAssignment::balanced(0, 2, 0).is_err());
        assert!(ExpertAssignment::balanced(1, 2, 0).is_err());
        assert!(ExpertAssignment::explicit(vec![0, 2], 2, 0).is_err());
        assert!(ExpertAssignment::explicit(vec![0, 0], 2, 0).is_err());
        assert!(ExpertAssignment::explicit_with_empty(vec![0, 0], 2, 1, true).is_ok());
        assert!(resolve_model_assignment(
            Some(ExpertAssignment::balanced(6, 2, 1).unwrap()),
            4,
            rank_one_topology(),
        )
        .is_err());
    }

    #[test]
    fn singleton_dispatch_preserves_empty_route_participation_without_a_collective() {
        let stream = stream();
        let assignment = ExpertAssignment::balanced(2, 1, 0).unwrap();
        let hidden = zeros_dtype(&[0, 4], Dtype::Float32, &stream).unwrap();
        let ids = zeros_dtype(&[0, 1], Dtype::Int32, &stream).unwrap();
        let weights = zeros_dtype(&[0, 1], Dtype::Float32, &stream).unwrap();
        let returned = dispatch_local_with(
            &hidden,
            &ids,
            &weights,
            &assignment,
            &stream,
            |_routes, _stream| panic!("empty routes must not acquire or execute an expert"),
        )
        .unwrap();
        eval([&returned.reduced_output]).unwrap();
        assert_eq!(returned.reduced_output.shape(), &[0, 4]);
        assert_eq!(returned.statistics.total_routes, 0);
        assert_eq!(returned.statistics.local_routes, 0);
        assert_eq!(returned.statistics.sent_routes, 0);
        assert_eq!(returned.statistics.received_routes, 0);
        assert_eq!(returned.statistics.exchanged_bytes, 0);
    }

    #[test]
    fn already_local_expert_bank_is_not_sliced_again_on_nonzero_rank() {
        let stream = stream();
        let assignment = ExpertAssignment::balanced(4, 2, 1).unwrap();
        assert!(!expert_bank_needs_slicing(2, &assignment).unwrap());
        assert!(expert_bank_needs_slicing(4, &assignment).unwrap());
        assert!(expert_bank_needs_slicing(3, &assignment).is_err());

        let mut bank = PackedSwiGluExperts::new(2, 4, 3, None, None, &stream).unwrap();
        let gate_up_shape = bank.gate_up_proj.shape().to_vec();
        let down_shape = bank.down_proj.shape().to_vec();
        let expected_bytes = parameter_bytes(&bank);
        let bytes = finalize_qwen3_expert_bank(&mut bank, &assignment, &stream).unwrap();

        assert_eq!(bytes, expected_bytes);
        assert_eq!(bank.num_experts, 2);
        assert_eq!(bank.gate_up_proj.shape(), gate_up_shape);
        assert_eq!(bank.down_proj.shape(), down_shape);
    }

    #[test]
    fn kimi_mxfp4_gguf_loads_through_expert_parallel_api() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let fixture = synthetic_kimi_gguf(context.stream());
        let loaded = load_expert_parallel_model_with_options(
            fixture.path(),
            ModelLoadOptions::with_parallel(rank_one_topology()),
            context.stream(),
            weights_context.stream(),
        )
        .unwrap();

        assert_eq!(loaded.info.model_kind, ModelKind::KimiLinear);
        assert_eq!(loaded.info.assignment.local_global_expert_ids(), &[2, 3]);
        assert_eq!(loaded.info.opened_checkpoint_shards, [fixture.path()]);
        assert!(loaded.info.routed_expert_bytes > 0);
        assert!(loaded.info.replicated_parameter_bytes > 0);
        let ExpertArchitecture::KimiLinear(model) = &loaded.architecture else {
            panic!("expected Kimi Linear");
        };
        let parameters = model.parameters().flatten();
        let gate_up = parameters
            .get("model.layers.1.mlp.experts.gate_up_proj")
            .unwrap();
        let down = parameters
            .get("model.layers.1.mlp.experts.down_proj")
            .unwrap();
        assert_eq!(gate_up.shape(), &[2, 128, 4]);
        assert_eq!(down.shape(), &[2, 32, 8]);
        assert_eq!(gate_up.dtype(), Dtype::Uint32);
        assert_eq!(
            parameters
                .get("model.layers.1.mlp.experts.gate_up_proj_scales")
                .unwrap()
                .shape(),
            &[2, 128, 1]
        );
        assert!(parameters.contains_key("model.layers.1.mlp.shared_experts.gate_proj.weight"));
    }

    #[test]
    fn kimi_mxfp4_gguf_sparse_ep_keeps_owned_experts_cold() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let fixture = synthetic_kimi_gguf(context.stream());
        let loaded = load_expert_parallel_model_with_options(
            fixture.path(),
            ModelLoadOptions {
                quantization: None,
                parallel: Some(rank_one_topology()),
                weight_residency: host_expert_residency(ExpertCacheLoadOptions::default()),
            },
            context.stream(),
            weights_context.stream(),
        )
        .unwrap();

        assert_eq!(loaded.info.model_kind, ModelKind::KimiLinear);
        assert_eq!(loaded.info.assignment.local_global_expert_ids(), &[2, 3]);
        assert_eq!(loaded.info.routed_expert_bytes, 0);
        assert!(loaded.info.owned_expert_bytes > 0);
        assert!(loaded.info.replicated_parameter_bytes > 0);
        let report = loaded.expert_cache_report().unwrap().unwrap();
        assert_eq!(report.owned_experts, 2);
        assert_eq!(report.host_resident_experts, 0);
        assert_eq!(report.device_resident_experts, 0);
        assert_eq!(loaded.info.opened_checkpoint_shards, [fixture.path()]);

        let reads_before = report.residency.weight_store().physical_reads;
        let cache = loaded.expert_cache.as_ref().unwrap();
        let acquired = cache
            .acquire_route_slice(1, &[2], &[1, 1], ExpertPass::Decode, context.stream())
            .unwrap();
        let hot = cache.report().unwrap();
        assert_eq!(hot.device_resident_experts, 1);
        assert!(hot.residency.weight_store().physical_reads > reads_before);
        drop(acquired);
    }

    #[test]
    fn kimi_mxfp4_gguf_streamed_ep_keeps_dense_layers_and_experts_cold() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let fixture = synthetic_kimi_gguf(context.stream());
        let dense = crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions::new(
            u64::MAX,
            u64::MAX,
            1,
            1,
            1,
        )
        .unwrap();
        let residency = dense_expert_residency(ExpertCacheLoadOptions::default(), dense);
        let loaded = load_expert_parallel_model_with_options(
            fixture.path(),
            ModelLoadOptions {
                quantization: None,
                parallel: Some(rank_one_topology()),
                weight_residency: residency,
            },
            context.stream(),
            weights_context.stream(),
        )
        .unwrap();

        assert!(matches!(
            loaded.architecture,
            ExpertArchitecture::KimiLinearLayerwise(_)
        ));
        assert_eq!(loaded.info.routed_expert_bytes, 0);
        assert!(loaded.info.owned_expert_bytes > 0);
        let experts = loaded.expert_cache_report().unwrap().unwrap();
        assert_eq!(experts.owned_experts, 2);
        assert_eq!(experts.device_resident_experts, 0);
        let dense = loaded.dense_stream_report().unwrap().unwrap();
        assert_eq!(dense.planned_layer_count(), 2);
        assert_eq!(dense.device_layers().current_layer_count(), 0);
    }

    #[test]
    fn kimi_mxfp4_gguf_tp_ep_preflight_uses_sharded_hybrid_state() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let fixture = synthetic_kimi_gguf(context.stream());

        for rank in 0..4 {
            let topology = tensor_expert_topology(rank);
            let dense = crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions::new(
                u64::MAX,
                u64::MAX,
                1,
                1,
                1,
            )
            .unwrap();
            let residency = dense_expert_residency(ExpertCacheLoadOptions::default(), dense);
            let loaded = load_expert_parallel_model_with_options(
                fixture.path(),
                ModelLoadOptions::with_parallel(topology).with_weight_residency(residency),
                context.stream(),
                weights_context.stream(),
            )
            .unwrap();

            assert_eq!(loaded.info.topology, topology);
            assert_eq!(
                loaded.info.assignment.local_global_expert_ids(),
                if topology.expert_parallel_rank == 0 {
                    &[0, 1]
                } else {
                    &[2, 3]
                }
            );
            assert_eq!(loaded.prompt_cache_layer_layout().unwrap().len(), 2);
            let dense = loaded.dense_stream_report().unwrap().unwrap();
            assert_eq!(dense.planned_layer_count(), 2);
            assert!(loaded.info.owned_expert_bytes > 0);
        }
    }

    #[test]
    fn kimi_mxfp4_gguf_tp_ep_fully_resident_pins_only_owned_experts() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let fixture = synthetic_kimi_gguf(context.stream());

        for rank in 0..4 {
            let topology = tensor_expert_topology(rank);
            let loaded = load_expert_parallel_model_with_options(
                fixture.path(),
                ModelLoadOptions::with_parallel(topology),
                context.stream(),
                weights_context.stream(),
            )
            .unwrap();

            assert!(matches!(
                loaded.architecture,
                ExpertArchitecture::KimiLinearLayerwise(_)
            ));
            assert_eq!(
                loaded.info.routed_expert_residency,
                RoutedExpertResidency::FullyResident
            );
            assert_eq!(loaded.info.assignment.local_expert_count(), 2);
            assert!(loaded.info.routed_expert_bytes > 0);
            assert_eq!(
                loaded.info.local_parameter_bytes,
                loaded.info.replicated_parameter_bytes + loaded.info.routed_expert_bytes
            );
            let experts = loaded.expert_cache_report().unwrap().unwrap();
            assert_eq!(experts.owned_experts, 2);
            assert_eq!(experts.device_resident_experts, 2);
            assert_eq!(experts.device_resident_bytes, experts.owned_bytes);
            assert_eq!(experts.prefill.device.misses, 0);
            assert_eq!(experts.decode.device.misses, 0);
            assert!(loaded.dense_stream_report().unwrap().is_none());
        }
    }

    #[test]
    fn qwen3_round_robin_loader_materializes_only_rank_one_experts() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let weights_stream = weights_context.stream();
        let fixture = tempfile::tempdir().unwrap();
        std::fs::write(
            fixture.path().join("config.json"),
            r#"{
              "model_type":"qwen3_moe","hidden_size":32,"num_hidden_layers":1,
              "intermediate_size":64,"num_attention_heads":1,"num_key_value_heads":1,
              "head_dim":32,"rms_norm_eps":0.000001,"vocab_size":32,
              "max_position_embeddings":128,"rope_theta":1000000.0,
              "tie_word_embeddings":false,"rope_scaling":null,
              "moe_intermediate_size":32,"num_experts":4,
              "num_experts_per_tok":2,"norm_topk_prob":true
            }"#,
        )
        .unwrap();
        let args = dense_qwen::load_config(fixture.path()).unwrap();
        let source = dense_qwen::Model::new(args, stream).unwrap();
        save_zero_checkpoint(&source, fixture.path(), stream);

        let options = ModelLoadOptions::with_quantization(WeightQuantization::MxFp4)
            .with_parallel_topology(rank_one_topology());
        let assignment = ExpertAssignment::round_robin(4, 2, 1).unwrap();
        let loaded = load_expert_parallel_model_with_options_and_assignment(
            fixture.path(),
            options,
            assignment,
            stream,
            weights_stream,
        )
        .unwrap();

        assert_eq!(loaded.info.assignment.local_global_expert_ids(), &[1, 3]);
        assert_eq!(
            loaded.info.assignment.policy(),
            &ExpertAssignmentPolicy::RoundRobin
        );
        let ExpertArchitecture::DenseQwen(model) = &loaded.architecture else {
            panic!("expected Qwen3");
        };
        let dense_qwen::FeedForward::Moe(moe) = &model.model.layers[0].mlp else {
            panic!("expected sparse MoE layer");
        };
        assert_eq!(moe.experts.num_experts, 2);
        assert_eq!(moe.experts.gate_up_proj.shape(), &[2, 64, 4]);
        assert_eq!(moe.experts.down_proj.shape(), &[2, 32, 4]);
        assert_eq!(moe.experts.gate_up_proj.dtype(), Dtype::Uint32);
        assert_eq!(
            moe.experts
                .gate_up_proj_scales
                .as_ref()
                .as_ref()
                .unwrap()
                .shape(),
            &[2, 64, 1]
        );

        let expert_options = ExpertCacheLoadOptions::new(
            crate::runtime::residency::policy::OffloadConfig::new(Some(1 << 20), Some(0), 1)
                .unwrap(),
            1 << 20,
            1 << 20,
        )
        .unwrap();
        let options = ModelLoadOptions::with_parallel(rank_one_topology())
            .with_weight_residency(host_expert_residency(expert_options));
        let assignment = ExpertAssignment::round_robin(4, 2, 1).unwrap();
        let cached = load_expert_parallel_model_with_options_and_assignment(
            fixture.path(),
            options,
            assignment,
            stream,
            weights_stream,
        )
        .unwrap();
        let report = cached.expert_cache_report().unwrap().unwrap();
        assert_eq!(report.owned_experts, 2);
        assert_eq!(report.host_resident_experts, 0);
        assert_eq!(report.device_resident_experts, 0);
        assert_eq!(cached.info.routed_expert_bytes, 0);
        assert!(cached.info.owned_expert_bytes > 0);
        assert_eq!(
            report
                .residency
                .units()
                .iter()
                .map(|unit| unit.id().as_str())
                .collect::<Vec<_>>(),
            vec![
                "expert.layer.00000.global.00001",
                "expert.layer.00000.global.00003"
            ]
        );
    }

    #[test]
    fn qwen3_tensor_expert_preflight_shards_nonexperts_and_preserves_cache_identity() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let fixture = tempfile::tempdir().unwrap();
        std::fs::write(
            fixture.path().join("config.json"),
            r#"{
              "model_type":"qwen3_moe","hidden_size":32,"num_hidden_layers":2,
              "intermediate_size":64,"num_attention_heads":2,"num_key_value_heads":2,
              "head_dim":16,"rms_norm_eps":0.000001,"vocab_size":32,
              "max_position_embeddings":128,"rope_theta":1000000.0,
              "tie_word_embeddings":true,"rope_scaling":null,
              "moe_intermediate_size":32,"num_experts":4,
              "num_experts_per_tok":2,"norm_topk_prob":true
            }"#,
        )
        .unwrap();
        let args = dense_qwen::load_config(fixture.path()).unwrap();
        let source = dense_qwen::Model::new(args, context.stream()).unwrap();
        save_zero_checkpoint(&source, fixture.path(), context.stream());
        let dense = crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions::new(
            u64::MAX,
            u64::MAX,
            1,
            1,
            1,
        )
        .unwrap();
        let residency = dense_expert_residency(ExpertCacheLoadOptions::default(), dense);
        let topology = tensor_expert_topology(3);
        let loaded = load_expert_parallel_model_with_options(
            fixture.path(),
            ModelLoadOptions {
                quantization: None,
                parallel: Some(topology),
                weight_residency: residency,
            },
            context.stream(),
            weights_context.stream(),
        )
        .unwrap();

        assert_eq!(loaded.info.topology, topology);
        assert_eq!(loaded.info.assignment.local_global_expert_ids(), &[2, 3]);
        let ExpertArchitecture::DenseQwenLayerwise(model) = &loaded.architecture else {
            panic!("expected shared Qwen3 layerwise adapter")
        };
        assert_eq!(model.parallel_info().unwrap().topology(), topology);
        let identity = loaded.prompt_cache_model_identity().unwrap();
        assert_eq!(identity.topology.tensor_parallel, Some((2, 1)));
        assert_eq!(identity.topology.expert_parallel, Some((2, 1)));
        let reads = model.checkpoint_store().diagnostics().unwrap();
        assert!(reads.physical_reads < source.parameters().flatten().len() as u64);
    }

    #[test]
    fn gpt_oss_tensor_expert_preflight_shards_nonexperts_and_preserves_cache_identity() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let fixture = tempfile::tempdir().unwrap();
        let config = serde_json::json!({
            "model_type": "gpt_oss",
            "hidden_size": 64,
            "intermediate_size": 96,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": 64,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "rms_norm_eps": 0.00001,
            "sliding_window": 3,
            "max_position_embeddings": 128,
            "rope_theta": 150000.0,
            "layer_types": ["sliding_attention", "full_attention"],
            "quantization_config": {"quant_method": "mxfp4"},
            "swiglu_limit": 7.0
        });
        std::fs::write(
            fixture.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let args = gpt_oss::model_args_from_config_value(&config).unwrap();
        let source = gpt_oss::Model::new(args, context.stream()).unwrap();
        save_zero_checkpoint(&source, fixture.path(), context.stream());
        let dense = crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions::new(
            u64::MAX,
            u64::MAX,
            1,
            1,
            1,
        )
        .unwrap();
        let residency = dense_expert_residency(ExpertCacheLoadOptions::default(), dense);
        let topology = tensor_expert_topology(3);
        let loaded = load_expert_parallel_model_with_options(
            fixture.path(),
            ModelLoadOptions {
                quantization: None,
                parallel: Some(topology),
                weight_residency: residency,
            },
            context.stream(),
            weights_context.stream(),
        )
        .unwrap();

        assert_eq!(loaded.info.topology, topology);
        assert_eq!(loaded.info.assignment.local_global_expert_ids(), &[2, 3]);
        let ExpertArchitecture::GptOssLayerwise(model) = &loaded.architecture else {
            panic!("expected shared GPT-OSS layerwise adapter")
        };
        assert_eq!(model.parallel_info().unwrap().topology(), topology);
        let identity = loaded.prompt_cache_model_identity().unwrap();
        assert_eq!(identity.topology.tensor_parallel, Some((2, 1)));
        assert_eq!(identity.topology.expert_parallel, Some((2, 1)));
        assert!(matches!(
            identity.layer_layout.get(0),
            Some(crate::LayerCachePolicy::KeyValue {
                num_key_value_heads,
                head_dim,
                ..
            }) if num_key_value_heads.get() == 1 && head_dim.get() == 32
        ));
        let reads = model.checkpoint_store().diagnostics().unwrap();
        assert!(reads.physical_reads < source.parameters().flatten().len() as u64);
    }

    #[test]
    fn lfm2_tensor_expert_preflight_composes_hybrid_state_and_sparse_ownership() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let fixture = tempfile::tempdir().unwrap();
        let config = serde_json::json!({
            "model_type": "lfm2_moe",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "moe_intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "norm_eps": 0.00001,
            "layer_types": ["conv", "full_attention"],
            "conv_L_cache": 3,
            "conv_bias": false,
            "block_auto_adjust_ff_dim": false,
            "tie_word_embeddings": false,
            "num_dense_layers": 1,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "norm_topk_prob": true,
            "use_expert_bias": true
        });
        std::fs::write(
            fixture.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let args = lfm2::model_args_from_config_value(&config).unwrap();
        let source = lfm2::Model::new(args, context.stream()).unwrap();
        save_zero_checkpoint(&source, fixture.path(), context.stream());
        let dense = crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions::new(
            u64::MAX,
            u64::MAX,
            1,
            1,
            1,
        )
        .unwrap();
        let residency = dense_expert_residency(ExpertCacheLoadOptions::default(), dense);
        let topology = tensor_expert_topology(3);
        let loaded = load_expert_parallel_model_with_options(
            fixture.path(),
            ModelLoadOptions {
                quantization: None,
                parallel: Some(topology),
                weight_residency: residency,
            },
            context.stream(),
            weights_context.stream(),
        )
        .unwrap();

        assert_eq!(loaded.info.topology, topology);
        assert_eq!(loaded.info.assignment.local_global_expert_ids(), &[2, 3]);
        let ExpertArchitecture::Lfm2Layerwise(model) = &loaded.architecture else {
            panic!("expected shared LFM2 layerwise adapter")
        };
        assert_eq!(model.parallel_info().unwrap().topology(), topology);
        let identity = loaded.prompt_cache_model_identity().unwrap();
        assert_eq!(identity.topology.tensor_parallel, Some((2, 1)));
        assert_eq!(identity.topology.expert_parallel, Some((2, 1)));
        assert_eq!(identity.layer_layout.len(), 2);
        assert!(matches!(
            identity.layer_layout.get(1),
            Some(crate::LayerCachePolicy::KeyValue {
                num_key_value_heads,
                head_dim,
                ..
            }) if num_key_value_heads.get() == 1 && head_dim.get() == 4
        ));
        let reads = model.checkpoint_store().diagnostics().unwrap();
        assert!(reads.physical_reads < source.parameters().flatten().len() as u64);
    }

    #[test]
    fn deepseek_explicit_loader_materializes_only_rank_one_experts() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let weights_stream = weights_context.stream();
        let fixture = tempfile::tempdir().unwrap();
        std::fs::write(
            fixture.path().join("config.json"),
            r#"{
              "model_type":"deepseek_v3","hidden_size":32,"intermediate_size":64,
              "moe_intermediate_size":32,"num_hidden_layers":1,"num_attention_heads":1,
              "vocab_size":32,"rms_norm_eps":0.000001,"max_position_embeddings":128,
              "rope_theta":10000,"q_lora_rank":null,"kv_lora_rank":32,
              "qk_nope_head_dim":32,"qk_rope_head_dim":8,"v_head_dim":32,
              "first_k_dense_replace":0,"moe_layer_freq":1,"n_routed_experts":4,
              "n_shared_experts":1,"num_experts_per_tok":2,"n_group":2,
              "topk_group":1,"topk_method":"noaux_tc","scoring_func":"sigmoid",
              "norm_topk_prob":true,"routed_scaling_factor":1.0,
              "num_nextn_predict_layers":0,"tie_word_embeddings":false
            }"#,
        )
        .unwrap();
        let args = deepseek_v3::get_model_args(fixture.path()).unwrap();
        let source = deepseek_v3::Model::new(args.clone(), stream).unwrap();
        let parameters = source.parameters().flatten();
        let mut arrays = parameters
            .iter()
            .map(|(name, parameter)| {
                (
                    name.to_string(),
                    zeros_dtype(parameter.shape(), parameter.dtype(), stream).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        for expert in 0..args.n_routed_experts {
            for (projection, shape) in [
                ("gate_proj", [args.moe_intermediate_size, args.hidden_size]),
                ("up_proj", [args.moe_intermediate_size, args.hidden_size]),
                ("down_proj", [args.hidden_size, args.moe_intermediate_size]),
            ] {
                arrays.push((
                    format!("model.layers.0.mlp.experts.{expert}.{projection}.weight"),
                    Array::zeros::<f32>(&shape, stream).unwrap(),
                ));
            }
        }
        Array::save_safetensors(
            arrays.iter().map(|(name, array)| (name.as_str(), array)),
            None,
            fixture.path().join("model.safetensors"),
        )
        .unwrap();

        let options = ModelLoadOptions::with_quantization(WeightQuantization::MxFp4)
            .with_parallel_topology(rank_one_topology());
        let assignment = ExpertAssignment::explicit(vec![1, 0, 0, 1], 2, 1).unwrap();
        let loaded = load_expert_parallel_model_with_options_and_assignment(
            fixture.path(),
            options,
            assignment,
            stream,
            weights_stream,
        )
        .unwrap();

        assert_eq!(loaded.info.assignment.local_global_expert_ids(), &[0, 3]);
        assert_eq!(
            loaded.info.assignment.policy(),
            &ExpertAssignmentPolicy::Explicit(vec![1, 0, 0, 1])
        );
        let ExpertArchitecture::DeepSeek(model) = &loaded.architecture else {
            panic!("expected DeepSeek");
        };
        let deepseek_v3::FeedForward::Moe(moe) = &model.model.layers[0].mlp else {
            panic!("expected sparse MoE layer");
        };
        assert_eq!(moe.experts.num_experts, 2);
        assert_eq!(
            moe.experts.gate_proj.as_ref().as_ref().unwrap().shape(),
            &[2, 32, 4]
        );
        assert_eq!(
            moe.experts
                .gate_proj_scales
                .as_ref()
                .as_ref()
                .unwrap()
                .shape(),
            &[2, 32, 1]
        );
    }

    #[test]
    fn compact_routes_preserves_tokens_slots_and_global_ids() {
        let stream = stream();
        let assignment = ExpertAssignment::balanced(4, 2, 1).unwrap();
        let hidden = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let ids = Array::from_slice(&[0i32, 2, 1, 3], &[2, 2]);
        let weights = Array::from_slice(&[0.1f32, 0.9, 0.25, 0.75], &[2, 2]);
        let (routes, stats) =
            compact_local_routes(&hidden, &ids, &weights, &assignment, &stream).unwrap();
        eval([
            &routes.global_expert_ids,
            &routes.local_expert_ids,
            &routes.token_indices,
            &routes.slot_indices,
            &routes.weights,
        ])
        .unwrap();
        assert_eq!(stats.total_routes, 4);
        assert_eq!(stats.local_routes, 2);
        assert_eq!(stats.synchronization_count, 1);
        assert_eq!(
            routes
                .global_expert_ids
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[2, 3]
        );
        assert_eq!(
            routes
                .local_expert_ids
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[0, 1]
        );
        assert_eq!(
            routes.token_indices.evaluated().unwrap().as_slice::<i32>(),
            &[0, 1]
        );
        assert_eq!(
            routes.slot_indices.evaluated().unwrap().as_slice::<i32>(),
            &[1, 1]
        );
    }

    #[test]
    fn replicated_dispatch_recombines_weights_exactly() {
        let stream = stream();
        let group = Group::init(false, Backend::Any).unwrap();
        assert_eq!(group.size(), 1);
        let assignment = ExpertAssignment::balanced(3, 1, 0).unwrap();
        let hidden = Array::from_slice(&[2.0f32, 4.0, 10.0, 20.0], &[2, 2]);
        let ids = Array::from_slice(&[0i32, 2, 1, 1], &[2, 2]);
        let weights = Array::from_slice(&[0.25f32, 0.75, 0.4, 0.6], &[2, 2]);
        let returned = dispatch_replicated(
            &hidden,
            &ids,
            &weights,
            &assignment,
            &mut IdentityBank,
            &group,
            &stream,
        )
        .unwrap();
        eval([&returned.reduced_output]).unwrap();
        assert_eq!(
            returned
                .reduced_output
                .evaluated()
                .unwrap()
                .as_slice::<f32>(),
            &[2.0, 4.0, 10.0, 20.0]
        );
    }

    #[test]
    fn all_to_all_v_singleton_preserves_payload_and_zero_counts() {
        let stream = stream();
        let group = Group::init(false, Backend::Any).unwrap();
        let payload = Array::from_slice(&[1i32, 2, 3, 4], &[2, 2]);
        let received = all_to_all_v(&[payload], &group, &stream).unwrap();
        eval([&received.received]).unwrap();
        assert_eq!(received.source_counts, vec![2]);
        assert_eq!(
            received.received.evaluated().unwrap().as_slice::<i32>(),
            &[1, 2, 3, 4]
        );

        let empty = Array::from_slice::<i32>(&[], &[0, 2]);
        let received = all_to_all_v(&[empty], &group, &stream).unwrap();
        assert_eq!(received.source_counts, vec![0]);
        assert_eq!(received.received.shape(), &[0, 2]);
    }

    #[test]
    fn sharded_and_replicated_dispatch_match_on_singleton() {
        let stream = stream();
        let group = Group::init(false, Backend::Any).unwrap();
        let assignment = ExpertAssignment::balanced(2, 1, 0).unwrap();
        let hidden = Array::from_slice(&[2.0f32, 4.0, 10.0, 20.0], &[2, 2]);
        let ids = Array::from_slice(&[0i32, 1, 1, 0], &[2, 2]);
        let weights = Array::from_slice(&[0.25f32, 0.75, 0.4, 0.6], &[2, 2]);
        let replicated = dispatch_replicated(
            &hidden,
            &ids,
            &weights,
            &assignment,
            &mut IdentityBank,
            &group,
            &stream,
        )
        .unwrap();
        let route_tokens = Array::from_slice(&[0i32, 0, 1, 1], &[4]);
        let routed_hidden = hidden.take_axis(&route_tokens, 0, &stream).unwrap();
        let sharded = dispatch_sharded(
            ShardedRouteBlocks {
                hidden: vec![routed_hidden],
                global_expert_ids: vec![ids.reshape(&[4], &stream).unwrap()],
                original_route_indices: vec![
                    Array::arange::<i32, i32>(Some(0), 4, None, &stream).unwrap()
                ],
                weights: vec![weights.reshape(&[4], &stream).unwrap()],
                top_k: 2,
                source_tokens: 2,
            },
            &assignment,
            &mut IdentityBank,
            &group,
            &stream,
        )
        .unwrap();
        eval([&replicated.reduced_output, &sharded.output]).unwrap();
        assert_eq!(
            replicated
                .reduced_output
                .evaluated()
                .unwrap()
                .as_slice::<f32>(),
            sharded.output.evaluated().unwrap().as_slice::<f32>()
        );
    }
}

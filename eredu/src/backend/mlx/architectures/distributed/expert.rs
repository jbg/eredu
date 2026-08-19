//! Reusable expert-parallel assignment, routing, and exchange infrastructure.

//!
//! Pure expert parallelism keeps ordinary model state replicated and partitions
//! only routed expert banks. [`crate::backend::mlx::runtime::distributed::expert::dispatch_replicated`]
//! exploits the replicated
//! token layout: ranks compact only routes owned by their experts and all-sum
//! the resulting token buffer. Sharded-token dispatch uses compact native or
//! topology-routed variable-count all-to-all payload exchange.

use eredu_checkpoint::WeightQuantization;

use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use safemlx::{
    distributed::Group,
    error::Exception,
    module::{ModuleParameters, Param},
    ops::{indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue},
    transforms::eval,
    Array, Stream,
};

use crate::core::cache::{
    validate_prompt_cache_model_identity, PromptCacheDescriptor, PromptCacheManifest,
    PromptCacheModelIdentity, PromptCacheOptions, PromptCacheTopology,
};

use crate::{
    backend::mlx::architectures::distributed::pipeline::{assign_module, load_deepseek_experts},
    backend::mlx::architectures::{
        deepseek_v3::model as deepseek_v3,
        deepseek_v4::model as deepseek_v4,
        gemma4::model as gemma4,
        gpt_oss::model as gpt_oss,
        inkling::model as inkling,
        kimi_linear::model as kimi_linear,
        lfm2::model as lfm2,
        nemotron_h::model as nemotron_h,
        qwen::{
            hybrid::{qwen3_5, qwen3_next},
            vl::model as qwen3_vl,
        },
    },
    backend::mlx::error::Error,
    backend::mlx::runtime::cache::residency::{
        open_prompt_cache, CacheResidencyManager, CacheResidencyPolicy, CacheResidencyReport,
        PagedCacheOptions,
    },
    backend::mlx::runtime::cache::{ConcatKeyValueCache, PagedKeyValueCache, SlidingKeyValueCache},
    backend::mlx::runtime::checkpoint::load::StrictLoadConfig,
    backend::mlx::runtime::checkpoint::quantization::should_quantize_on_load,
    backend::mlx::runtime::checkpoint::store::{GgufWeightStore, SafetensorsWeightStore},
    backend::mlx::runtime::distributed::parallel::{ParallelBuildContext, ShardingPolicy},
    backend::mlx::runtime::distributed::topology::{
        load_partition_from_store_on_streams, PlacementPlan, TensorPlacement,
    },
    backend::mlx::runtime::execution::inspection::ActivationObserver,
    backend::mlx::runtime::generation::sampler::SpeculativeSampler,
    backend::mlx::runtime::media::input as runtime_input,
    backend::mlx::runtime::residency::expert_cache::{
        AcquiredExperts, ExpertCache, ExpertCacheError, ExpertCacheLoadOptions, ExpertCacheReport,
        ExpertCatalogEntry, ExpertPass, ExpertRouteBatch,
    },
    backend::mlx::speculative::embedded::{
        DistributedEmbeddedMtpSampler, EmbeddedMtpOutput, EmbeddedMtpTarget,
    },
    backend::mlx::{MlxParallelContext, ModelLoadOptions},
    core::cache::CacheRankIdentity,
    core::generation::MtpConfig,
    core::ModelKind,
    core::{MtpCapability, MtpCheckpointKind, MtpStats},
};

use crate::backend::mlx::runtime::execution::layerwise::LayerWeightResidency;
#[cfg(test)]
use crate::backend::mlx::runtime::execution::layerwise::WeightResidency;

use crate::{
    backend::mlx::architectures::{deepseek_v3::model::RoutedExperts, qwen::dense as dense_qwen},
    backend::mlx::nn::moe::{quantize_expert_bank, PackedSwiGluExperts},
};

pub use crate::backend::mlx::runtime::distributed::expert::*;

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
    pub topology: MlxParallelContext,
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
    /// DeepSeek V4 compressed/sparse attention cache.
    DeepSeekV4(deepseek_v4::Cache),
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
            Self::DeepSeekV4(cache) => cache.reset()?,
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
            Self::Lfm2(cache) => cache.reset()?,
            Self::NemotronH(cache) => cache.reset()?,
            Self::QwenHybrid(cache) => cache.reset()?,
            Self::Qwen3Vl(cache) => *cache = qwen3_vl::Cache::default(),
            Self::Gemma4(cache) => cache.clear()?,
        }
        Ok(())
    }
}

#[cfg(test)]
impl ExpertParallelCache {
    pub fn offset(&self) -> i32 {
        match self {
            Self::DeepSeek(cache) => cache.offset(),
            Self::DeepSeekV4(cache) => cache.offset(),
            Self::KimiLinear(cache) => cache.offset(),
            Self::DenseQwen(cache) => cache.first().and_then(Option::as_ref).map_or(
                0,
                crate::backend::mlx::runtime::cache::KeyValueCache::offset,
            ),
            Self::DenseQwenSliding(cache) => cache.first().and_then(Option::as_ref).map_or(
                0,
                crate::backend::mlx::runtime::cache::KeyValueCache::offset,
            ),
            Self::DenseQwenPaged(cache) => cache.first().and_then(Option::as_ref).map_or(
                0,
                crate::backend::mlx::runtime::cache::KeyValueCache::offset,
            ),
            Self::GptOss(cache) => cache.offset(),
            Self::Inkling(cache) => cache.offset(),
            Self::Lfm2(cache) => cache.offset(),
            Self::NemotronH(cache) => cache.offset(),
            Self::QwenHybrid(cache) => cache.offset(),
            Self::Qwen3Vl(cache) => cache.kv.first().and_then(Option::as_ref).map_or(
                0,
                crate::backend::mlx::runtime::cache::KeyValueCache::offset,
            ),
            Self::Gemma4(cache) => i32::try_from(cache.mtp_len()).unwrap_or(i32::MAX),
        }
    }
}

enum ExpertArchitecture {
    DeepSeekLayerwise(
        Box<crate::backend::mlx::architectures::deepseek_v3::layerwise::DeepSeekV3LayerwiseModel>,
    ),
    DeepSeekV4Layerwise(
        Box<crate::backend::mlx::architectures::deepseek_v4::layerwise::DeepSeekV4LayerwiseModel>,
    ),
    KimiLinearLayerwise(
        Box<crate::backend::mlx::architectures::kimi_linear::layerwise::KimiLinearLayerwiseModel>,
    ),
    DenseQwenLayerwise(
        Box<crate::backend::mlx::architectures::qwen::dense::layerwise::LayerwiseDecoder>,
    ),
    GptOssLayerwise(
        Box<crate::backend::mlx::architectures::gpt_oss::layerwise::GptOssLayerwiseModel>,
    ),
    InklingLayerwise(
        Box<crate::backend::mlx::architectures::inkling::layerwise::InklingLayerwiseModel>,
    ),
    Lfm2Layerwise(Box<crate::backend::mlx::architectures::lfm2::layerwise::Lfm2LayerwiseModel>),
    NemotronHLayerwise(
        Box<crate::backend::mlx::architectures::nemotron_h::layerwise::NemotronHLayerwiseModel>,
    ),
    QwenHybridLayerwise(
        Box<crate::backend::mlx::architectures::qwen::hybrid::layerwise::QwenHybridLayerwiseModel>,
    ),
    Qwen3VlLayerwise(
        Box<crate::backend::mlx::architectures::qwen::vl::layerwise::Qwen3VlLayerwiseModel>,
    ),
    Gemma4Layerwise(
        Box<crate::backend::mlx::architectures::gemma4::layerwise::Gemma4LayerwiseModel>,
    ),
}

impl ExpertArchitecture {
    fn bind_parallel_topology(&mut self, topology: MlxParallelContext) {
        match self {
            Self::DeepSeekLayerwise(model) => model.bind_parallel_topology(topology),
            Self::DeepSeekV4Layerwise(model) => model.bind_parallel_topology(topology),
            Self::KimiLinearLayerwise(model) => model.bind_parallel_topology(topology),
            Self::DenseQwenLayerwise(model) => model.bind_parallel_topology(topology),
            Self::GptOssLayerwise(model) => model.bind_parallel_topology(topology),
            Self::InklingLayerwise(model) => model.bind_parallel_topology(topology),
            Self::Lfm2Layerwise(model) => model.bind_parallel_topology(topology),
            Self::NemotronHLayerwise(model) => model.bind_parallel_topology(topology),
            Self::QwenHybridLayerwise(model) => model.bind_parallel_topology(topology),
            Self::Qwen3VlLayerwise(model) => model.bind_parallel_topology(topology),
            Self::Gemma4Layerwise(model) => model.bind_parallel_topology(topology),
        }
    }
}

/// Executable rank-local EP or TP+EP model.
pub struct ExpertParallelModel {
    topology: MlxParallelContext,
    info: ExpertParallelInfo,
    architecture: ExpertArchitecture,
    expert_cache: ExpertCache,
    latest_statistics: RoutingStatistics,
    cumulative_statistics: RoutingStatistics,
}

struct ExpertParallelQwenMtpTarget<'a> {
    model: &'a mut ExpertParallelModel,
    tensor_group: Option<&'a Group>,
    group: &'a Group,
}

#[derive(Clone)]
enum ExpertParallelMtpDraftCache {
    DeepSeek(Vec<crate::backend::mlx::runtime::cache::CompressedLatentCache>),
    DeepSeekV4(deepseek_v4::DraftCache),
    Inkling(Vec<inkling::LayerCache>),
    NemotronH(Vec<nemotron_h::LayerCache>),
}

struct ExpertParallelEmbeddedMtpTarget<'a> {
    model: &'a mut ExpertParallelModel,
    tensor_group: Option<&'a Group>,
    expert_group: &'a Group,
}

impl std::fmt::Debug for ExpertParallelModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExpertParallelModel")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

impl crate::backend::mlx::architectures::qwen::hybrid::mtp::QwenMtpTarget
    for ExpertParallelQwenMtpTarget<'_>
{
    fn prefill_mtp_target(
        &mut self,
        input: runtime_input::ModelInput<'_>,
        cache: &mut qwen3_5::Cache,
        stream: &Stream,
    ) -> Result<qwen3_5::QwenMtpStepOutput, Exception> {
        let tokens = runtime_input::text_token_ids(input, stream)?;
        cache.reset()?;
        self.model
            .forward_qwen_mtp_target(&tokens, cache, self.tensor_group, self.group, stream)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn verify_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut qwen3_5::Cache,
        stream: &Stream,
    ) -> Result<qwen3_5::QwenMtpStepOutput, Exception> {
        self.model
            .forward_qwen_mtp_target(tokens, cache, self.tensor_group, self.group, stream)
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
            ExpertArchitecture::QwenHybridLayerwise(model) => {
                let expert_cache = &self.model.expert_cache;
                let assignment = &self.model.info.assignment;
                let args = model.args().clone();
                let mut statistics = RoutingStatistics::default();
                let output = model.forward_mtp_head_with_expert_executor(
                    hidden,
                    tokens,
                    cache,
                    self.tensor_group,
                    &mut |layer, hidden, ids, weights, stream| {
                        let returned = dispatch_replicated_with(
                            hidden,
                            ids,
                            weights,
                            assignment,
                            self.group,
                            stream,
                            |routes, stream| {
                                execute_cached_qwen_hybrid(
                                    &args,
                                    layer,
                                    routes,
                                    ExpertPass::Decode,
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
                );
                self.model.latest_statistics = statistics;
                self.model
                    .cumulative_statistics
                    .accumulate(&self.model.latest_statistics);
                output
            }
            _ => Err(Exception::custom(
                "embedded Qwen MTP requires a Qwen3-Next or Qwen3.5 EP model",
            )),
        }
    }

    fn mtp_layer_count(&self) -> usize {
        match &self.model.architecture {
            ExpertArchitecture::QwenHybridLayerwise(model) => model.mtp_len(),
            _ => 0,
        }
    }
}

impl EmbeddedMtpTarget for ExpertParallelEmbeddedMtpTarget<'_> {
    type Cache = ExpertParallelCache;
    type DraftCache = ExpertParallelMtpDraftCache;

    fn prefill_target(
        &mut self,
        input: runtime_input::ModelInput<'_>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Exception> {
        let tokens = runtime_input::text_token_ids(input, stream)?;
        cache
            .reset()
            .map_err(|error| Exception::custom(error.to_string()))?;
        self.model.forward_embedded_mtp_target(
            &tokens,
            cache,
            self.tensor_group,
            self.expert_group,
            stream,
        )
    }

    fn verify_target(
        &mut self,
        tokens: &Array,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Exception> {
        self.model.forward_embedded_mtp_target(
            tokens,
            cache,
            self.tensor_group,
            self.expert_group,
            stream,
        )
    }

    fn prefill_draft_cache(
        &mut self,
        output: &EmbeddedMtpOutput,
        tokens: &Array,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        if let (
            ExpertArchitecture::DeepSeekV4Layerwise(model),
            ExpertParallelCache::DeepSeekV4(cache),
        ) = (&mut self.model.architecture, &mut *cache)
        {
            return <crate::backend::mlx::architectures::deepseek_v4::layerwise::DeepSeekV4LayerwiseModel as EmbeddedMtpTarget>::prefill_draft_cache(
                model, output, tokens, cache, stream,
            );
        }
        let sequence = tokens.dim(1);
        if sequence <= 1 {
            return Ok(());
        }
        let hidden = output
            .hidden
            .try_index_device((.., ..sequence - 1, ..), stream)?;
        let next = tokens.try_index_device((.., 1..), stream)?;
        let mut draft = Self::draft_cache(cache);
        for depth in 0..self.max_draft_tokens() {
            let _ = self.model.forward_embedded_mtp_draft(
                &hidden,
                &next,
                depth,
                &mut draft,
                self.tensor_group,
                self.expert_group,
                stream,
            )?;
        }
        Self::commit_draft_cache(cache, &draft);
        Ok(())
    }

    fn draft_cache(cache: &Self::Cache) -> Self::DraftCache {
        match cache {
            ExpertParallelCache::DeepSeek(cache) => {
                ExpertParallelMtpDraftCache::DeepSeek(cache.mtp_layers.clone())
            }
            ExpertParallelCache::DeepSeekV4(cache) => {
                ExpertParallelMtpDraftCache::DeepSeekV4(cache.mtp_layers.clone())
            }
            ExpertParallelCache::Inkling(cache) => {
                ExpertParallelMtpDraftCache::Inkling(cache.mtp_layers.clone())
            }
            ExpertParallelCache::NemotronH(cache) => {
                ExpertParallelMtpDraftCache::NemotronH(cache.mtp_layers.clone())
            }
            _ => unreachable!("capability preflight rejects non-shared embedded MTP caches"),
        }
    }

    fn commit_draft_cache(cache: &mut Self::Cache, draft: &Self::DraftCache) {
        match (cache, draft) {
            (
                ExpertParallelCache::DeepSeek(cache),
                ExpertParallelMtpDraftCache::DeepSeek(draft),
            ) => cache.mtp_layers.clone_from(draft),
            (
                ExpertParallelCache::DeepSeekV4(cache),
                ExpertParallelMtpDraftCache::DeepSeekV4(draft),
            ) => cache.mtp_layers.clone_from(draft),
            (ExpertParallelCache::Inkling(cache), ExpertParallelMtpDraftCache::Inkling(draft)) => {
                cache.mtp_layers.clone_from(draft)
            }
            (
                ExpertParallelCache::NemotronH(cache),
                ExpertParallelMtpDraftCache::NemotronH(draft),
            ) => cache.mtp_layers.clone_from(draft),
            _ => unreachable!("embedded MTP draft-cache type changed after preflight"),
        }
    }

    fn draft_logits(
        &mut self,
        hidden: &Array,
        last_token: u32,
        draft_index: usize,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        let token = Array::from_slice(&[last_token], &[1, 1]);
        let output = self.model.forward_embedded_mtp_draft(
            hidden,
            &token,
            draft_index,
            cache,
            self.tensor_group,
            self.expert_group,
            stream,
        )?;
        Ok((output.logits, output.hidden))
    }

    fn advance_draft_cache(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        if let (
            ExpertArchitecture::DeepSeekV4Layerwise(model),
            ExpertParallelMtpDraftCache::DeepSeekV4(cache),
        ) = (&mut self.model.architecture, &mut *cache)
        {
            return <crate::backend::mlx::architectures::deepseek_v4::layerwise::DeepSeekV4LayerwiseModel as EmbeddedMtpTarget>::advance_draft_cache(
                model, hidden, tokens, cache, stream,
            );
        }
        for depth in 0..self.max_draft_tokens() {
            let _ = self.model.forward_embedded_mtp_draft(
                hidden,
                tokens,
                depth,
                cache,
                self.tensor_group,
                self.expert_group,
                stream,
            )?;
        }
        Ok(())
    }

    fn fused_draft_logits(
        &mut self,
        hidden: &Array,
        last_token: u32,
        proposal_capacity: usize,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<Option<Array>, Exception> {
        match (&mut self.model.architecture, cache) {
            (
                ExpertArchitecture::DeepSeekV4Layerwise(model),
                ExpertParallelMtpDraftCache::DeepSeekV4(cache),
            ) => <crate::backend::mlx::architectures::deepseek_v4::layerwise::DeepSeekV4LayerwiseModel as EmbeddedMtpTarget>::fused_draft_logits(
                model, hidden, last_token, proposal_capacity, cache, stream,
            ),
            _ => Ok(None),
        }
    }

    fn adjust_fused_draft_logits(
        &mut self,
        logits: Array,
        last_token: u32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        match &mut self.model.architecture {
            ExpertArchitecture::DeepSeekV4Layerwise(model) => {
                <crate::backend::mlx::architectures::deepseek_v4::layerwise::DeepSeekV4LayerwiseModel as EmbeddedMtpTarget>::adjust_fused_draft_logits(
                    model, logits, last_token, stream,
                )
            }
            _ => Ok(logits),
        }
    }

    fn max_draft_tokens(&self) -> usize {
        match &self.model.architecture {
            ExpertArchitecture::DeepSeekLayerwise(model) => model.mtp_len(),
            ExpertArchitecture::DeepSeekV4Layerwise(model) => model.mtp_len(),
            ExpertArchitecture::InklingLayerwise(model) => model.mtp_len(),
            ExpertArchitecture::NemotronHLayerwise(model) => model.mtp_len(),
            _ => 0,
        }
    }
}

impl ExpertParallelModel {
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

    /// Returns placement, assignment, and memory diagnostics.
    pub fn info(&self) -> &ExpertParallelInfo {
        &self.info
    }

    /// Reports whether this EP target can perform embedded MTP generation.
    pub fn mtp_capability(&self) -> MtpCapability {
        match &self.architecture {
            ExpertArchitecture::DeepSeekLayerwise(model) if model.mtp_len() > 0 => {
                MtpCapability::Ready {
                    checkpoint: MtpCheckpointKind::Embedded,
                }
            }
            ExpertArchitecture::DeepSeekV4Layerwise(model) if model.mtp_len() > 0 => {
                MtpCapability::Ready {
                    checkpoint: MtpCheckpointKind::Embedded,
                }
            }
            ExpertArchitecture::InklingLayerwise(model) if model.mtp_len() > 0 => {
                MtpCapability::Ready {
                    checkpoint: MtpCheckpointKind::Embedded,
                }
            }
            ExpertArchitecture::NemotronHLayerwise(model) if model.mtp_len() > 0 => {
                MtpCapability::Ready {
                    checkpoint: MtpCheckpointKind::Embedded,
                }
            }
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
        self.expert_cache.report().map(Some).map_err(Error::from)
    }

    /// Returns dense nonexpert streaming telemetry when that EP residency
    /// policy is active.
    pub fn dense_stream_report(
        &self,
    ) -> Result<
        Option<crate::backend::mlx::runtime::execution::layerwise::DenseDiskStreamReport>,
        Error,
    > {
        match &self.architecture {
            ExpertArchitecture::DeepSeekLayerwise(model) => model.dense_stream_report(),
            ExpertArchitecture::DeepSeekV4Layerwise(model) => model.dense_stream_report(),
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
            ExpertArchitecture::DeepSeekLayerwise(model) => {
                ExpertParallelCache::DeepSeek(model.new_cache())
            }
            ExpertArchitecture::DeepSeekV4Layerwise(model) => ExpertParallelCache::DeepSeekV4(
                model
                    .new_cache()
                    .expect("validated DeepSeek V4 distributed cache geometry"),
            ),
            ExpertArchitecture::KimiLinearLayerwise(model) => {
                ExpertParallelCache::KimiLinear(model.new_cache())
            }
            ExpertArchitecture::DenseQwenLayerwise(model) => {
                ExpertParallelCache::DenseQwen(model.new_cache())
            }
            ExpertArchitecture::GptOssLayerwise(model) => {
                ExpertParallelCache::GptOss(model.new_cache())
            }
            ExpertArchitecture::InklingLayerwise(model) => {
                ExpertParallelCache::Inkling(model.new_cache())
            }
            ExpertArchitecture::Lfm2Layerwise(model) => {
                ExpertParallelCache::Lfm2(model.new_cache())
            }
            ExpertArchitecture::NemotronHLayerwise(model) => {
                ExpertParallelCache::NemotronH(model.new_cache())
            }
            ExpertArchitecture::QwenHybridLayerwise(model) => {
                ExpertParallelCache::QwenHybrid(model.new_cache())
            }
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
                ExpertArchitecture::DeepSeekLayerwise(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ExpertParallelCache::DeepSeek),
                ExpertArchitecture::DeepSeekV4Layerwise(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ExpertParallelCache::DeepSeekV4),
                ExpertArchitecture::KimiLinearLayerwise(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ExpertParallelCache::KimiLinear),
                ExpertArchitecture::GptOssLayerwise(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ExpertParallelCache::GptOss),
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
                ExpertArchitecture::InklingLayerwise(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ExpertParallelCache::Inkling),
                ExpertArchitecture::NemotronHLayerwise(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ExpertParallelCache::NemotronH),
                ExpertArchitecture::Lfm2Layerwise(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ExpertParallelCache::Lfm2),
                ExpertArchitecture::QwenHybridLayerwise(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ExpertParallelCache::QwenHybrid),
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
            ExpertParallelCache::DeepSeekV4(cache) => cache.residency_report().map_err(Into::into),
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
            ExpertParallelCache::Lfm2(cache) => cache.residency_report().map_err(Into::into),
            ExpertParallelCache::QwenHybrid(cache) => cache.residency_report().map_err(Into::into),
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
            (ExpertArchitecture::DeepSeekLayerwise(_), ExpertParallelCache::DeepSeek(cache)) => {
                cache
                    .save_prompt_cache(directory, descriptor, prefix_token_ids, options)
                    .map_err(Into::into)
            }
            (ExpertArchitecture::GptOssLayerwise(_), ExpertParallelCache::GptOss(cache)) => cache
                .save_prompt_cache(directory, descriptor, prefix_token_ids, options)
                .map_err(Into::into),
            (
                ExpertArchitecture::DeepSeekV4Layerwise(model),
                ExpertParallelCache::DeepSeekV4(cache),
            ) => model.save_prompt_cache(
                cache,
                directory,
                descriptor,
                prefix_token_ids,
                options,
                stream,
            ),
            (
                ExpertArchitecture::KimiLinearLayerwise(_),
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
            (ExpertArchitecture::Lfm2Layerwise(_), ExpertParallelCache::Lfm2(cache)) => {
                lfm2::Model::save_prompt_cache_with_rank(
                    cache,
                    directory,
                    descriptor,
                    prefix_token_ids,
                    options,
                    rank,
                    stream,
                )
                .map_err(Into::into)
            }
            (ExpertArchitecture::InklingLayerwise(_), ExpertParallelCache::Inkling(cache)) => {
                inkling::Model::save_prompt_cache(
                    cache,
                    directory,
                    descriptor,
                    prefix_token_ids,
                    options,
                    stream,
                )
                .map_err(Into::into)
            }
            (ExpertArchitecture::NemotronHLayerwise(_), ExpertParallelCache::NemotronH(cache)) => {
                nemotron_h::Model::save_prompt_cache_with_rank(
                    cache,
                    directory,
                    descriptor,
                    prefix_token_ids,
                    options,
                    rank,
                    stream,
                )
                .map_err(Into::into)
            }
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
            ExpertArchitecture::DeepSeekLayerwise(model) => model
                .load_prompt_cache(&directory, expected, prefix_token_ids, options, stream)
                .map(|(cache, manifest)| (ExpertParallelCache::DeepSeek(cache), manifest)),
            ExpertArchitecture::DeepSeekV4Layerwise(model) => model
                .load_prompt_cache(directory, expected, prefix_token_ids, options, stream)
                .map(|(cache, manifest)| (ExpertParallelCache::DeepSeekV4(cache), manifest)),
            ExpertArchitecture::KimiLinearLayerwise(model) => {
                kimi_linear::Model::load_paged_prompt_cache_with_identity(
                    model.args(),
                    &directory,
                    expected,
                    prefix_token_ids,
                    &identity,
                    options,
                    stream,
                )
                .map(|(cache, manifest)| (ExpertParallelCache::KimiLinear(cache), manifest))
                .map_err(Into::into)
            }
            ExpertArchitecture::Lfm2Layerwise(model) => {
                lfm2::Model::load_prompt_cache_with_identity(
                    model.args(),
                    &directory,
                    expected,
                    prefix_token_ids,
                    identity,
                    stream,
                )
                .map(|(cache, manifest)| (ExpertParallelCache::Lfm2(cache), manifest))
                .map_err(Into::into)
            }
            ExpertArchitecture::InklingLayerwise(model) => {
                inkling::Model::load_prompt_cache_with_identity(
                    model.args(),
                    &directory,
                    expected,
                    prefix_token_ids,
                    &identity,
                    stream,
                )
                .map(|(cache, manifest)| (ExpertParallelCache::Inkling(cache), manifest))
                .map_err(Into::into)
            }
            ExpertArchitecture::NemotronHLayerwise(model) => {
                nemotron_h::Model::load_paged_prompt_cache_with_identity(
                    model.args(),
                    &directory,
                    expected,
                    prefix_token_ids,
                    &identity,
                    options,
                    stream,
                )
                .map(|(cache, manifest)| (ExpertParallelCache::NemotronH(cache), manifest))
                .map_err(Into::into)
            }
            ExpertArchitecture::DenseQwenLayerwise(model) => model
                .load_prompt_cache(&directory, expected, prefix_token_ids, options, stream)
                .map(|(cache, manifest)| (ExpertParallelCache::DenseQwen(cache), manifest)),
            ExpertArchitecture::GptOssLayerwise(model) => model
                .load_prompt_cache(&directory, expected, prefix_token_ids, options, stream)
                .map(|(cache, manifest)| (ExpertParallelCache::GptOss(cache), manifest)),
            ExpertArchitecture::QwenHybridLayerwise(model) => model
                .load_prompt_cache(&directory, expected, prefix_token_ids, options, stream)
                .map(|(cache, manifest)| (ExpertParallelCache::QwenHybrid(cache), manifest)),
            ExpertArchitecture::Qwen3VlLayerwise(model) => model
                .load_prompt_cache_with_validated_identity(
                    &directory,
                    expected,
                    &identity,
                    prefix_token_ids,
                    options,
                    stream,
                )
                .map(|(cache, manifest)| (ExpertParallelCache::Qwen3Vl(cache), manifest)),
            ExpertArchitecture::Gemma4Layerwise(model) => model
                .load_prompt_cache(&directory, expected, prefix_token_ids, options, stream)
                .map(|(cache, manifest)| (ExpertParallelCache::Gemma4(cache), manifest)),
        }
    }

    fn prompt_cache_rank_directory(&self, root: &Path) -> PathBuf {
        root.join(format!("rank-{:05}", self.topology.global_rank))
    }

    /// Returns the canonical cache-relevant architecture identity for this rank.
    pub fn prompt_cache_architecture_fingerprint(&self) -> Result<String, Error> {
        Ok(self.prompt_cache_model_identity()?.architecture_fingerprint)
    }

    /// Returns this rank's exact ordered prompt-cache layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        Ok(self.prompt_cache_model_identity()?.layer_layout)
    }

    /// Returns each owned layer's processed-token delta from the persisted prefix.
    pub fn prompt_cache_layer_prefix_offsets(&self) -> Result<Vec<i32>, Error> {
        Ok(self.prompt_cache_model_identity()?.layer_prefix_offsets)
    }

    pub(crate) fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        let mut identity = match &self.architecture {
            ExpertArchitecture::DeepSeekLayerwise(model) => model.prompt_cache_model_identity()?,
            ExpertArchitecture::DeepSeekV4Layerwise(model) => {
                model.prompt_cache_model_identity()?
            }
            ExpertArchitecture::KimiLinearLayerwise(model) => {
                model.prompt_cache_model_identity()?
            }
            ExpertArchitecture::DenseQwenLayerwise(model) => model.prompt_cache_model_identity()?,
            ExpertArchitecture::GptOssLayerwise(model) => model.prompt_cache_model_identity()?,
            ExpertArchitecture::InklingLayerwise(model) => model.prompt_cache_model_identity()?,
            ExpertArchitecture::Lfm2Layerwise(model) => model.prompt_cache_model_identity()?,
            ExpertArchitecture::NemotronHLayerwise(model) => model.prompt_cache_model_identity()?,
            ExpertArchitecture::QwenHybridLayerwise(model) => {
                model.prompt_cache_model_identity()?
            }
            ExpertArchitecture::Qwen3VlLayerwise(model) => model.prompt_cache_model_identity()?,
            ExpertArchitecture::Gemma4Layerwise(model) => model.prompt_cache_model_identity()?,
        };
        identity.topology = crate::backend::mlx::cache::prompt_cache_topology(self.topology);
        Ok(identity)
    }

    /// Runs prefill or decode through the selected distributed backend session.
    pub fn forward(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut ExpertParallelCache,
        execution: &crate::backend::mlx::MlxDistributedSession<'_>,
    ) -> Result<Array, Error> {
        if execution.topology() != self.topology {
            return Err(Error::Parallel(format!(
                "model topology {:?} does not match distributed session topology {:?}",
                self.topology,
                execution.topology()
            )));
        }
        let tensor_context = execution.tensor_context()?;
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
            execution.stream(),
        )
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
        let logits = {
            let expert_cache = &self.expert_cache;
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
                (
                    ExpertArchitecture::DeepSeekV4Layerwise(model),
                    ExpertParallelCache::DeepSeekV4(cache),
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
                                    execute_cached_deepseek_v4(
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
        };
        materialize_timing_phase([&logits])?;
        statistics.model_time = total_started.elapsed();
        self.latest_statistics = statistics;
        self.cumulative_statistics
            .accumulate(&self.latest_statistics);
        Ok(logits)
    }

    fn forward_embedded_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut ExpertParallelCache,
        tensor_group: Option<&Group>,
        expert_group: &Group,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Exception> {
        self.validate_expert_group(expert_group)
            .map_err(|error| Exception::custom(error.to_string()))?;
        self.topology
            .validate_execution_stream(stream)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let expert_cache = &self.expert_cache;
        let pass = if tokens.dim(1) > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let assignment = &self.info.assignment;
        let mut statistics = RoutingStatistics::default();
        let output = match (&mut self.architecture, cache) {
            (
                ExpertArchitecture::DeepSeekLayerwise(model),
                ExpertParallelCache::DeepSeek(cache),
            ) => {
                let args = model.args().clone();
                model.forward_mtp_target_with_expert_executor(
                    tokens,
                    cache,
                    tensor_group,
                    |layer, hidden, ids, weights, stream| {
                        let returned = dispatch_replicated_with(
                            hidden,
                            ids,
                            weights,
                            assignment,
                            expert_group,
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
                ExpertArchitecture::DeepSeekV4Layerwise(model),
                ExpertParallelCache::DeepSeekV4(cache),
            ) => {
                let args = model.args().clone();
                model.forward_mtp_target_with_expert_executor(
                    tokens,
                    cache,
                    tensor_group,
                    |layer, hidden, ids, weights, stream| {
                        let returned = dispatch_replicated_with(
                            hidden,
                            ids,
                            weights,
                            assignment,
                            expert_group,
                            stream,
                            |routes, stream| {
                                execute_cached_deepseek_v4(
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
            (ExpertArchitecture::InklingLayerwise(model), ExpertParallelCache::Inkling(cache)) => {
                let args = model.args().clone();
                model.forward_mtp_target_with_expert_executor(
                    tokens,
                    cache,
                    tensor_group,
                    |layer, hidden, ids, weights, stream| {
                        let returned = dispatch_replicated_with(
                            hidden,
                            ids,
                            weights,
                            assignment,
                            expert_group,
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
                ExpertArchitecture::NemotronHLayerwise(model),
                ExpertParallelCache::NemotronH(cache),
            ) => {
                let args = model.args().clone();
                model.forward_mtp_target_with_expert_executor(
                    tokens,
                    cache,
                    tensor_group,
                    |layer, hidden, ids, weights, stream| {
                        let returned = dispatch_replicated_with(
                            hidden,
                            ids,
                            weights,
                            assignment,
                            expert_group,
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
            _ => {
                return Err(Exception::custom(
                    "embedded MTP architecture/cache mismatch for EP execution",
                ))
            }
        };
        materialize_timing_phase([&output.logits])
            .map_err(|error| Exception::custom(error.to_string()))?;
        self.latest_statistics = statistics;
        self.cumulative_statistics
            .accumulate(&self.latest_statistics);
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_embedded_mtp_draft(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut ExpertParallelMtpDraftCache,
        tensor_group: Option<&Group>,
        expert_group: &Group,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Exception> {
        let expert_cache = &self.expert_cache;
        let assignment = &self.info.assignment;
        let pass = ExpertPass::Decode;
        let mut statistics = RoutingStatistics::default();
        let output = match (&mut self.architecture, cache) {
            (
                ExpertArchitecture::DeepSeekLayerwise(model),
                ExpertParallelMtpDraftCache::DeepSeek(cache),
            ) => {
                let args = model.args().clone();
                model.forward_mtp_draft_with_expert_executor(
                    hidden,
                    tokens,
                    depth,
                    cache,
                    tensor_group,
                    |layer, hidden, ids, weights, stream| {
                        let returned = dispatch_replicated_with(
                            hidden,
                            ids,
                            weights,
                            assignment,
                            expert_group,
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
                ExpertArchitecture::DeepSeekV4Layerwise(model),
                ExpertParallelMtpDraftCache::DeepSeekV4(cache),
            ) => {
                let last_token = tokens.clone().item::<u32>(stream);
                let (logits, hidden) = <crate::backend::mlx::architectures::deepseek_v4::layerwise::DeepSeekV4LayerwiseModel as EmbeddedMtpTarget>::draft_logits(
                    model, hidden, last_token, depth, cache, stream,
                )?;
                EmbeddedMtpOutput {
                    logits,
                    hidden,
                    tokens: tokens.clone(),
                }
            }
            (
                ExpertArchitecture::InklingLayerwise(model),
                ExpertParallelMtpDraftCache::Inkling(cache),
            ) => model.forward_mtp_draft_cartesian(
                hidden,
                tokens,
                depth,
                cache,
                tensor_group,
                stream,
            )?,
            (
                ExpertArchitecture::NemotronHLayerwise(model),
                ExpertParallelMtpDraftCache::NemotronH(cache),
            ) => {
                let args = model.args().clone();
                model.forward_mtp_draft_with_expert_executor(
                    hidden,
                    tokens,
                    depth,
                    cache,
                    tensor_group,
                    |layer, hidden, ids, weights, stream| {
                        let returned = dispatch_replicated_with(
                            hidden,
                            ids,
                            weights,
                            assignment,
                            expert_group,
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
            _ => {
                return Err(Exception::custom(
                    "embedded MTP draft-cache architecture mismatch for EP execution",
                ))
            }
        };
        self.latest_statistics = statistics;
        self.cumulative_statistics
            .accumulate(&self.latest_statistics);
        Ok(output)
    }

    fn forward_qwen_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut qwen3_5::Cache,
        tensor_group: Option<&Group>,
        group: &Group,
        stream: &Stream,
    ) -> Result<qwen3_5::QwenMtpStepOutput, Error> {
        let total_started = Instant::now();
        self.validate_expert_group(group)?;
        self.topology.validate_execution_stream(stream)?;
        if tokens.ndim() != 2 {
            return Err(Error::Parallel(format!(
                "expert-parallel token input must be [batch, sequence], got {:?}",
                tokens.shape()
            )));
        }
        let expert_cache = &self.expert_cache;
        let pass = if tokens.dim(1) > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let assignment = &self.info.assignment;
        let mut statistics = RoutingStatistics::default();
        let output = match &mut self.architecture {
            ExpertArchitecture::QwenHybridLayerwise(model) => {
                let args = model.args().clone();
                model.forward_mtp_with_expert_executor(
                    tokens,
                    cache,
                    tensor_group,
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

    /// Generates with Cartesian TP+EP predictor ownership and synchronized
    /// sampling. Predictor TP collectives and expert exchanges are scoped to
    /// the topology-derived subgroups supplied by `execution`.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_embedded_mtp_distributed<S: SpeculativeSampler + Clone>(
        &mut self,
        cache: &mut ExpertParallelCache,
        input: runtime_input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        execution: &crate::backend::mlx::MlxDistributedSession<'_>,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        let topology = execution.topology();
        if topology != self.topology {
            return Err(Exception::custom(
                "embedded MTP model topology does not match distributed session",
            ));
        }
        let stream = execution.stream();
        if topology.pipeline_parallel_size != 1 {
            return Err(Exception::custom(
                "ExpertParallelModel embedded MTP cannot own a pipeline axis; use PipelineModel",
            ));
        }
        let expert_group = execution.expert_group().ok_or_else(|| {
            Exception::custom("distributed embedded MTP requires an active EP subgroup")
        })?;
        let tensor = execution
            .tensor_context()
            .map_err(|error| Exception::custom(error.to_string()))?;
        let tensor_group = tensor.group();
        let sampling_rank = topology
            .global_rank_for(crate::ParallelCoordinates {
                tensor: 0,
                pipeline: 0,
                expert: 0,
                data: topology.data_parallel_rank,
            })
            .map_err(|error| Exception::custom(error.to_string()))?;
        let mut synchronized =
            DistributedEmbeddedMtpSampler::new(sampler.clone(), sampling_rank, execution.world())
                .map_err(|error| Exception::custom(error.to_string()))?;
        if !matches!(
            self.mtp_capability(),
            MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Embedded
            }
        ) {
            return Err(Exception::custom(format!(
                "embedded MTP runtime adapter is unavailable for distributed EP model type {} ({:?})",
                self.info.model_kind.model_type_name(),
                self.mtp_capability()
            )));
        }
        if matches!(cache, ExpertParallelCache::QwenHybrid(_)) {
            let ExpertParallelCache::QwenHybrid(cache) = cache else {
                unreachable!()
            };
            let mut target = ExpertParallelQwenMtpTarget {
                model: self,
                tensor_group,
                group: expert_group,
            };
            let mut executor =
                crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(&mut target);
            let result = crate::backend::mlx::speculative::scheduler::generate_tokens(
                &mut executor,
                cache,
                input,
                config,
                prng_key,
                &mut synchronized,
                crate::backend::mlx::speculative::MtpExecutionStreams::single(stream),
                crate::core::generation::MtpSchedulerOptions::default(),
                |_| Ok(()),
            );
            *sampler = synchronized.into_inner();
            return result;
        }
        let mut target = ExpertParallelEmbeddedMtpTarget {
            model: self,
            tensor_group,
            expert_group,
        };
        let mut executor =
            crate::backend::mlx::speculative::embedded::EmbeddedMtpExecutor::new(&mut target);
        let result = crate::backend::mlx::speculative::scheduler::generate_tokens(
            &mut executor,
            cache,
            input,
            config,
            prng_key,
            &mut synchronized,
            crate::backend::mlx::speculative::MtpExecutionStreams::single(stream),
            crate::core::generation::MtpSchedulerOptions::default(),
            |_| Ok(()),
        );
        *sampler = synchronized.into_inner();
        result
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

fn validate_expert_topology(topology: MlxParallelContext) -> Result<(), Error> {
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

/// Materializes an executable EP or TP+EP model for the MLX backend.
pub(crate) fn load_expert_parallel_model_with_options(
    model_dir: impl AsRef<Path>,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    load_expert_parallel_model_impl(model_dir, options, None, stream, weights_stream)
}

#[cfg(test)]
pub(crate) fn load_expert_parallel_model_with_options_and_assignment(
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
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
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
                    | "deepseek_v4"
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
        Some("deepseek_v4") => load_deepseek_v4_ep(
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
    topology: MlxParallelContext,
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
impl ExpertParallelModel {
    pub fn latest_routing_statistics(&self) -> &RoutingStatistics {
        &self.latest_statistics
    }

    pub fn forward_with_observer(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut ExpertParallelCache,
        execution: &crate::backend::mlx::MlxDistributedSession<'_>,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Error> {
        let tensor = execution.tensor_context()?;
        let expert = execution.expert_group().ok_or_else(|| {
            Error::Parallel("expert observation requires an active EP group".into())
        })?;
        self.forward_impl(
            tokens,
            mask,
            cache,
            tensor.group(),
            expert,
            Some(observer),
            execution.stream(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample_and_synchronize<S: crate::backend::mlx::runtime::generation::sampler::Sampler>(
        &self,
        logits: &Array,
        sampler: &mut S,
        temperature: f32,
        prng_state: Option<&mut safemlx::random::RandomState>,
        finished: bool,
        sampling_rank: usize,
        execution: &crate::backend::mlx::MlxDistributedSession<'_>,
    ) -> Result<crate::backend::mlx::runtime::distributed::parallel::SynchronizedToken, Error> {
        execution.sample_and_synchronize_on_rank(
            Some(logits),
            logits.dim(0),
            sampler,
            temperature,
            prng_state,
            finished,
            sampling_rank,
        )
    }
}

#[cfg(test)]
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

pub(crate) fn execute_cached_deepseek_v4(
    args: &deepseek_v4::ModelArgs,
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
            let quantization = match args.expert_dtype.as_deref() {
                Some("fp4") => Some(WeightQuantization::MxFp4),
                Some("fp8") => None,
                None => args.quantization,
                Some(_) => unreachable!("validated expert dtype"),
            };
            let bank = PackedSwiGluExperts::new(
                acquired.identities().len() as i32,
                args.hidden_size,
                args.moe_intermediate_size,
                quantization,
                quantization,
                stream,
            )?;
            let mut bank = if args.expert_dtype.as_deref() == Some("fp8") {
                bank.with_native_fp8_e8m0(stream)?
            } else {
                bank
            }
            .with_swiglu_limit(args.swiglu_limit)?;
            bank.gate_up_proj = Param::new(acquired.compact_binding("gate_up_proj", stream)?);
            bank.gate_up_proj_scales =
                Param::new(acquired.optional_compact_binding("gate_up_proj_scales", stream)?);
            bank.down_proj = Param::new(acquired.compact_binding("down_proj", stream)?);
            bank.down_proj_scales =
                Param::new(acquired.optional_compact_binding("down_proj_scales", stream)?);
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
                crate::backend::mlx::architectures::gemma4::layerwise::execute_acquired_gemma_experts(
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

#[cfg(test)]
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

#[cfg(test)]
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

fn load_kimi_linear_ep(
    model_dir: &Path,
    topology: MlxParallelContext,
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
            ExternalExpertResidency::SparseCache(expert_options, options.quantization),
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
    load_kimi_linear_external_ep(
        model_dir,
        topology,
        options,
        LayerWeightResidency::FullyResident,
        ExternalExpertResidency::FullyResident(options.quantization),
        crate::core::DEFAULT_MAX_MAPPED_SHARDS,
        assignment,
        stream,
        weights_stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn load_gguf_ep(
    architecture: &str,
    gguf_file: &Path,
    checkpoint: &GgufCheckpoint,
    metadata: std::collections::HashMap<String, GgufMetadataValue>,
    topology: MlxParallelContext,
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
                | "deepseek4"
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
    crate::backend::mlx::validate_gguf_quantization_source(
        checkpoint,
        &metadata,
        options.quantization,
    )?;
    if architecture == "deepseek4" {
        let mut structural_options = options;
        structural_options.parallel = None;
        structural_options.weight_residency =
            crate::backend::mlx::runtime::execution::layerwise::WeightResidency::fully_resident();
        crate::backend::mlx::structural::validate_gguf(
            crate::core::GgufArchitecture::DeepSeek4,
            checkpoint,
            &metadata,
            structural_options,
        )
        .into_loader_result()?;
    }
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
        reject_external_gguf_ep_quantization(options.quantization)?;
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
            ExternalExpertResidency::SparseCache(expert_options, options.quantization),
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
    reject_external_gguf_ep_quantization(options.quantization)?;
    load_external_gguf_ep(
        architecture,
        checkpoint,
        &metadata,
        inkling_mmproj.as_ref(),
        qwen3_vl_mmproj.as_deref(),
        qwen35_mmproj.as_ref(),
        topology,
        assignment,
        LayerWeightResidency::FullyResident,
        ExternalExpertResidency::FullyResident(options.quantization),
        crate::core::DEFAULT_MAX_MAPPED_SHARDS,
        stream,
        weights_stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn load_external_gguf_ep(
    architecture: &str,
    checkpoint: &GgufCheckpoint,
    metadata: &std::collections::HashMap<String, GgufMetadataValue>,
    inkling_mmproj: Option<&inkling::InklingMmprojGguf>,
    qwen3_vl_mmproj: Option<&Path>,
    qwen35_mmproj: Option<&qwen3_5::Qwen35MmprojGguf>,
    topology: MlxParallelContext,
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
            let gguf_plan = crate::backend::mlx::architectures::kimi_linear::checkpoint::gguf_plan(&args)
                .map_err(Error::UnsupportedArchitecture)?;
            let store: std::sync::Arc<dyn eredu_checkpoint::store::CheckpointSource> =
                std::sync::Arc::new(GgufWeightStore::new_with_max_mapped_shards(
                    checkpoint.clone(),
                    &gguf_plan,
                    kimi_linear::translate_gguf_weight_name,
                    max_mapped_shards,
                )?);
            let model = if topology.tensor_parallel_size > 1 {
                crate::backend::mlx::architectures::kimi_linear::layerwise::
                    load_kimi_linear_sparse_tp_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        ParallelBuildContext::new(topology, ShardingPolicy::Require),
                        stream,
                        weights_stream,
                    )?
            } else {
                crate::backend::mlx::architectures::kimi_linear::layerwise::
                    load_kimi_linear_sparse_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        stream,
                        weights_stream,
                    )?
            };
            let store = model.checkpoint_store_arc();
            let entries =
                crate::backend::mlx::architectures::kimi_linear::layerwise::kimi_expert_catalog(
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
        "deepseek4" => {
            let prepared = deepseek_v4::prepare_gguf_checkpoint(checkpoint, metadata)?;
            let args = prepared.args;
            let assignment = resolve_model_assignment(
                assignment,
                args.n_routed_experts as usize,
                topology,
            )?;
            let gguf_plan = crate::backend::mlx::architectures::deepseek_v4::checkpoint::gguf_plan(&args)
                .map_err(Error::UnsupportedArchitecture)?;
            let store: std::sync::Arc<dyn eredu_checkpoint::store::CheckpointSource> =
                std::sync::Arc::new(GgufWeightStore::new_with_max_mapped_shards(
                    checkpoint.clone(),
                    &gguf_plan,
                    deepseek_v4::translate_gguf_weight_name,
                    max_mapped_shards,
                )?);
            let model = if topology.tensor_parallel_size > 1 {
                crate::backend::mlx::architectures::deepseek_v4::layerwise::
                    load_deepseek_v4_sparse_tp_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        expert_residency.quantization(),
                        ParallelBuildContext::new(topology, ShardingPolicy::Require),
                        stream,
                        weights_stream,
                    )?
            } else {
                crate::backend::mlx::architectures::deepseek_v4::layerwise::
                    load_deepseek_v4_sparse_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        expert_residency.quantization(),
                        stream,
                        weights_stream,
                    )?
            };
            let store = model.checkpoint_store_arc();
            let entries = crate::backend::mlx::architectures::deepseek_v4::layerwise::expert_catalog(
                &args,
                store.as_ref(),
            )?;
            let replicated_parameter_bytes =
                planned_replicated_bytes(&model.residency_report()?)?;
            finish_external_ep(
                topology,
                ModelKind::DeepSeekV4,
                assignment,
                ExpertArchitecture::DeepSeekV4Layerwise(Box::new(model)),
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
            let gguf_plan = crate::backend::mlx::architectures::deepseek_v3::checkpoint::gguf_plan(&args)
                .map_err(Error::UnsupportedArchitecture)?;
            let store: std::sync::Arc<dyn eredu_checkpoint::store::CheckpointSource> =
                std::sync::Arc::new(GgufWeightStore::new_with_max_mapped_shards(
                    checkpoint.clone(),
                    &gguf_plan,
                    deepseek_v3::translate_gguf_weight_name,
                    max_mapped_shards,
                )?);
            let model = if topology.tensor_parallel_size > 1 {
                crate::backend::mlx::architectures::deepseek_v3::layerwise::
                    load_deepseek_v3_sparse_tp_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        ParallelBuildContext::new(topology, ShardingPolicy::Require),
                        stream,
                        weights_stream,
                    )?
            } else {
                crate::backend::mlx::architectures::deepseek_v3::layerwise::
                    load_deepseek_v3_sparse_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        stream,
                        weights_stream,
                    )?
            };
            let store = model.checkpoint_store_arc();
            let entries =
                crate::backend::mlx::architectures::deepseek_v3::layerwise::deepseek_expert_catalog(
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
            let gguf_plan = crate::backend::mlx::architectures::qwen::dense::checkpoint::gguf_plan(
                &args,
                crate::backend::mlx::architectures::qwen::dense::checkpoint::GgufVariant::Qwen3Moe,
            )
            .map_err(Error::UnsupportedArchitecture)?;
            let store: std::sync::Arc<dyn eredu_checkpoint::store::CheckpointSource> =
                std::sync::Arc::new(GgufWeightStore::new_with_max_mapped_shards(
                    checkpoint.clone(),
                    &gguf_plan,
                    |name| dense_qwen::translate_gguf_weight_name(name, true),
                    max_mapped_shards,
                )?);
            let model = if topology.tensor_parallel_size > 1 {
                crate::backend::mlx::architectures::qwen::dense::layerwise::
                    load_qwen3_sparse_tp_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        ParallelBuildContext::new(topology, ShardingPolicy::Require),
                        stream,
                        weights_stream,
                    )?
            } else {
                crate::backend::mlx::architectures::qwen::dense::layerwise::
                    load_qwen3_sparse_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        stream,
                        weights_stream,
                    )?
            };
            let store = model.checkpoint_store_arc();
            let entries =
                crate::backend::mlx::architectures::qwen::dense::layerwise::qwen3_expert_catalog(
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
                crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&vision_checkpoint);
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
            let store = crate::backend::mlx::architectures::qwen::vl::layerwise::qwen3_vl_gguf_store(
                checkpoint,
                &vision_checkpoint,
                &args,
                max_mapped_shards,
            )?;
            let model = if topology.tensor_parallel_size > 1 {
                crate::backend::mlx::architectures::qwen::vl::layerwise::
                    load_qwen3_vl_sparse_tp_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        ParallelBuildContext::new(topology, ShardingPolicy::Require),
                        stream,
                        weights_stream,
                    )?
            } else {
                crate::backend::mlx::architectures::qwen::vl::layerwise::
                    load_qwen3_vl_sparse_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        stream,
                        weights_stream,
                    )?
            };
            let store = model.checkpoint_store_arc();
            let entries =
                crate::backend::mlx::architectures::qwen::dense::layerwise::qwen3_expert_catalog_at(
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
            let store = crate::backend::mlx::architectures::gemma4::layerwise::gemma4_gguf_store(
                checkpoint,
                None,
                &args,
                prepared.vision_config.as_ref(),
                prepared.audio_config.as_ref(),
                max_mapped_shards,
            )?;
            let model = if topology.tensor_parallel_size > 1 {
                crate::backend::mlx::architectures::gemma4::layerwise::
                    load_gemma4_sparse_tp_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        ParallelBuildContext::new(topology, ShardingPolicy::Require),
                        stream,
                        weights_stream,
                    )?
            } else {
                crate::backend::mlx::architectures::gemma4::layerwise::load_gemma4_sparse_ep_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    stream,
                    weights_stream,
                )?
            };
            let store = model.checkpoint_store_arc();
            let entries = crate::backend::mlx::architectures::gemma4::layerwise::gemma4_expert_catalog(
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
            let gguf_plan = crate::backend::mlx::architectures::gpt_oss::checkpoint::gguf_plan(&args)
                .map_err(Error::UnsupportedArchitecture)?;
            let store: std::sync::Arc<dyn eredu_checkpoint::store::CheckpointSource> =
                std::sync::Arc::new(GgufWeightStore::new_with_max_mapped_shards(
                    checkpoint.clone(),
                    &gguf_plan,
                    gpt_oss::translate_gguf_weight_name,
                    max_mapped_shards,
                )?);
            let model = if topology.tensor_parallel_size > 1 {
                crate::backend::mlx::architectures::gpt_oss::layerwise::
                    load_gpt_oss_sparse_tp_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        ParallelBuildContext::new(topology, ShardingPolicy::Require),
                        stream,
                        weights_stream,
                    )?
            } else {
                crate::backend::mlx::architectures::gpt_oss::layerwise::
                    load_gpt_oss_sparse_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        stream,
                        weights_stream,
                    )?
            };
            let store = model.checkpoint_store_arc();
            let entries = crate::backend::mlx::architectures::gpt_oss::layerwise::gpt_oss_expert_catalog(
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
            let store = crate::backend::mlx::architectures::inkling::layerwise::inkling_gguf_store(
                checkpoint,
                inkling_mmproj,
                &args,
                max_mapped_shards,
            )?;
            let model = if topology.tensor_parallel_size > 1 {
                crate::backend::mlx::architectures::inkling::layerwise::
                    load_inkling_sparse_tp_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        ParallelBuildContext::new(topology, ShardingPolicy::Require),
                        stream,
                        weights_stream,
                    )?
            } else {
                crate::backend::mlx::architectures::inkling::layerwise::
                    load_inkling_sparse_ep_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    stream,
                    weights_stream,
                )?
            };
            let store = model.checkpoint_store_arc();
            let entries = crate::backend::mlx::architectures::inkling::layerwise::inkling_expert_catalog(
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
            let gguf_plan = crate::backend::mlx::architectures::lfm2::checkpoint::gguf_plan(&args)
                .map_err(Error::UnsupportedArchitecture)?;
            let store: std::sync::Arc<dyn eredu_checkpoint::store::CheckpointSource> =
                std::sync::Arc::new(GgufWeightStore::new_with_max_mapped_shards(
                    checkpoint.clone(),
                    &gguf_plan,
                    |name| lfm2::translate_gguf_weight_name(name, true),
                    max_mapped_shards,
                )?);
            let model = if topology.tensor_parallel_size > 1 {
                crate::backend::mlx::architectures::lfm2::layerwise::load_lfm2_sparse_tp_ep_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    ParallelBuildContext::new(topology, ShardingPolicy::Require),
                    stream,
                    weights_stream,
                )?
            } else {
                crate::backend::mlx::architectures::lfm2::layerwise::load_lfm2_sparse_ep_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    stream,
                    weights_stream,
                )?
            };
            let store = model.checkpoint_store_arc();
            let entries =
                crate::backend::mlx::architectures::lfm2::layerwise::lfm2_expert_catalog(
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
            let gguf_plan = crate::backend::mlx::architectures::nemotron_h::checkpoint::gguf_plan(&args)
                .map_err(Error::UnsupportedArchitecture)?;
            let store: std::sync::Arc<dyn eredu_checkpoint::store::CheckpointSource> =
                std::sync::Arc::new(GgufWeightStore::new_with_max_mapped_shards(
                    checkpoint.clone(),
                    &gguf_plan,
                    nemotron_h::translate_gguf_weight_name,
                    max_mapped_shards,
                )?);
            let model = if topology.tensor_parallel_size > 1 {
                crate::backend::mlx::architectures::nemotron_h::layerwise::
                    load_nemotron_h_sparse_tp_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        ParallelBuildContext::new(topology, ShardingPolicy::Require),
                        stream,
                        weights_stream,
                    )?
            } else {
                crate::backend::mlx::architectures::nemotron_h::layerwise::
                    load_nemotron_h_sparse_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        stream,
                        weights_stream,
                    )?
            };
            let store = model.checkpoint_store_arc();
            let entries =
                crate::backend::mlx::architectures::nemotron_h::layerwise::nemotron_h_expert_catalog(
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
            let variant = if is_next {
                crate::backend::mlx::architectures::qwen::hybrid::checkpoint::GgufVariant::Qwen3Next
            } else {
                crate::backend::mlx::architectures::qwen::hybrid::checkpoint::GgufVariant::Qwen35Moe
            };
            let store = crate::backend::mlx::architectures::qwen::hybrid::layerwise::qwen_hybrid_gguf_store(
                checkpoint,
                qwen35_mmproj,
                &args,
                variant,
                prepared.modalities.vision_config.as_ref(),
                max_mapped_shards,
            )?;
            let model = if topology.tensor_parallel_size > 1 {
                crate::backend::mlx::architectures::qwen::hybrid::layerwise::
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
                crate::backend::mlx::architectures::qwen::hybrid::layerwise::
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
            let store = model.checkpoint_store_arc();
            let entries =
                crate::backend::mlx::architectures::qwen::hybrid::layerwise::qwen_hybrid_expert_catalog(
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

fn planned_replicated_bytes(report: &eredu_runtime::ResidencyReport) -> Result<usize, Error> {
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
    topology: MlxParallelContext,
    kind: ModelKind,
    assignment: ExpertAssignment,
    mut architecture: ExpertArchitecture,
    store: std::sync::Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    entries: Vec<ExpertCatalogEntry>,
    residency: ExternalExpertResidency,
    replicated_parameter_bytes: usize,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    architecture.bind_parallel_topology(topology);
    let routed_expert_residency = residency.kind();
    let (expert_cache, owned_expert_bytes) = match residency {
        ExternalExpertResidency::FullyResident(quantization) => rank_owned_resident_experts(
            store.clone(),
            entries,
            &assignment,
            quantization,
            stream,
            weights_stream,
        )?,
        ExternalExpertResidency::SparseCache(options, quantization) => rank_owned_expert_cache(
            store.clone(),
            entries,
            &assignment,
            options,
            quantization,
            stream,
            weights_stream,
        )?,
    };
    let opened_checkpoint_shards = store.source_diagnostics()?.touched_shard_paths;
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
    FullyResident(Option<WeightQuantization>),
    SparseCache(ExpertCacheLoadOptions, Option<WeightQuantization>),
}

impl ExternalExpertResidency {
    const fn kind(self) -> RoutedExpertResidency {
        match self {
            Self::FullyResident(_) => RoutedExpertResidency::FullyResident,
            Self::SparseCache(_, _) => RoutedExpertResidency::SparseCache,
        }
    }

    const fn quantization(self) -> Option<WeightQuantization> {
        match self {
            Self::FullyResident(quantization) | Self::SparseCache(_, quantization) => quantization,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn load_kimi_linear_external_ep(
    model_dir: &Path,
    topology: MlxParallelContext,
    _options: ModelLoadOptions,
    non_expert: LayerWeightResidency,
    expert_residency: ExternalExpertResidency,
    max_mapped_shards: usize,
    assignment: Option<ExpertAssignment>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    let args = kimi_linear::get_model_args(model_dir)?;
    args.validate()?;
    let assignment = resolve_model_assignment(assignment, args.num_experts as usize, topology)?;
    let store: std::sync::Arc<dyn eredu_checkpoint::store::CheckpointSource> = std::sync::Arc::new(
        SafetensorsWeightStore::open_with_max_mapped_shards(model_dir, max_mapped_shards)?,
    );
    let model = if topology.tensor_parallel_size > 1 {
        crate::backend::mlx::architectures::kimi_linear::layerwise::load_kimi_linear_sparse_tp_ep_base_with_store(
            store.clone(),
            args.clone(),
            non_expert,
            ParallelBuildContext::new(topology, ShardingPolicy::Require),
            stream,
            weights_stream,
        )?
    } else {
        crate::backend::mlx::architectures::kimi_linear::layerwise::load_kimi_linear_sparse_ep_base_with_store(
            store.clone(),
            args.clone(),
            non_expert,
            stream,
            weights_stream,
        )?
    };
    let store = model.checkpoint_store_arc();
    let entries = crate::backend::mlx::architectures::kimi_linear::layerwise::kimi_expert_catalog(
        &args,
        store.as_ref(),
    )?;
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
    topology: MlxParallelContext,
    _options: ModelLoadOptions,
    non_expert: LayerWeightResidency,
    expert_residency: ExternalExpertResidency,
    max_mapped_shards: usize,
    assignment: Option<ExpertAssignment>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    let args = deepseek_v3::get_model_args(model_dir)?;
    args.validate()?;
    let assignment =
        resolve_model_assignment(assignment, args.n_routed_experts as usize, topology)?;
    let store: std::sync::Arc<dyn eredu_checkpoint::store::CheckpointSource> = std::sync::Arc::new(
        SafetensorsWeightStore::open_with_max_mapped_shards(model_dir, max_mapped_shards)?,
    );
    let model = if topology.tensor_parallel_size > 1 {
        crate::backend::mlx::architectures::deepseek_v3::layerwise::load_deepseek_v3_sparse_tp_ep_base_with_store(
            store.clone(),
            args.clone(),
            non_expert,
            ParallelBuildContext::new(topology, ShardingPolicy::Require),
            stream,
            weights_stream,
        )?
    } else {
        crate::backend::mlx::architectures::deepseek_v3::layerwise::load_deepseek_v3_sparse_ep_base_with_store(
            store.clone(),
            args.clone(),
            non_expert,
            stream,
            weights_stream,
        )?
    };
    let store = model.checkpoint_store_arc();
    let entries =
        crate::backend::mlx::architectures::deepseek_v3::layerwise::deepseek_expert_catalog(
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
fn load_deepseek_v4_external_ep(
    model_dir: &Path,
    topology: MlxParallelContext,
    options: ModelLoadOptions,
    non_expert: LayerWeightResidency,
    expert_residency: ExternalExpertResidency,
    max_mapped_shards: usize,
    assignment: Option<ExpertAssignment>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    let args = deepseek_v4::get_model_args(model_dir)?;
    let assignment =
        resolve_model_assignment(assignment, args.n_routed_experts as usize, topology)?;
    let store: std::sync::Arc<dyn eredu_checkpoint::store::CheckpointSource> = std::sync::Arc::new(
        SafetensorsWeightStore::open_with_max_mapped_shards(model_dir, max_mapped_shards)?,
    );
    let model = if topology.tensor_parallel_size > 1 {
        crate::backend::mlx::architectures::deepseek_v4::layerwise::load_deepseek_v4_sparse_tp_ep_base_with_store(
            store.clone(),
            args.clone(),
            non_expert,
            options.quantization,
            ParallelBuildContext::new(topology, ShardingPolicy::Require),
            stream,
            weights_stream,
        )?
    } else {
        crate::backend::mlx::architectures::deepseek_v4::layerwise::load_deepseek_v4_sparse_ep_base_with_store(
            store.clone(),
            args.clone(),
            non_expert,
            options.quantization,
            stream,
            weights_stream,
        )?
    };
    let store = model.checkpoint_store_arc();
    let entries = crate::backend::mlx::architectures::deepseek_v4::layerwise::expert_catalog(
        &args,
        store.as_ref(),
    )?;
    let replicated_parameter_bytes = planned_replicated_bytes(&model.residency_report()?)?;
    finish_external_ep(
        topology,
        ModelKind::DeepSeekV4,
        assignment,
        ExpertArchitecture::DeepSeekV4Layerwise(Box::new(model)),
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
    topology: MlxParallelContext,
    _options: ModelLoadOptions,
    non_expert: LayerWeightResidency,
    expert_residency: ExternalExpertResidency,
    max_mapped_shards: usize,
    assignment: Option<ExpertAssignment>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    let args = dense_qwen::load_config(model_dir)?;
    if !args.is_moe() {
        return Err(Error::Parallel(
            "Qwen3 config is dense and has no routed experts".into(),
        ));
    }
    let assignment = resolve_model_assignment(assignment, args.num_experts as usize, topology)?;
    let store: std::sync::Arc<dyn eredu_checkpoint::store::CheckpointSource> = std::sync::Arc::new(
        SafetensorsWeightStore::open_with_max_mapped_shards(model_dir, max_mapped_shards)?,
    );
    let model = if topology.tensor_parallel_size > 1 {
        crate::backend::mlx::architectures::qwen::dense::layerwise::load_qwen3_sparse_tp_ep_base_with_store(
            store.clone(),
            args.clone(),
            non_expert,
            ParallelBuildContext::new(topology, ShardingPolicy::Require),
            stream,
            weights_stream,
        )?
    } else {
        crate::backend::mlx::architectures::qwen::dense::layerwise::load_qwen3_sparse_ep_base_with_store(
            store.clone(),
            args.clone(),
            non_expert,
            stream,
            weights_stream,
        )?
    };
    let store = model.checkpoint_store_arc();
    let entries = crate::backend::mlx::architectures::qwen::dense::layerwise::qwen3_expert_catalog(
        &args,
        store.as_ref(),
    )?;
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

fn reject_external_gguf_ep_quantization(
    quantization: Option<WeightQuantization>,
) -> Result<(), Error> {
    if quantization.is_some() {
        return Err(Error::Quantization(
            "GGUF expert-parallel load-time conversion is not yet connected to the bounded dense-span reader; use matching checkpoint-native packed weights"
                .into(),
        ));
    }
    Ok(())
}

fn load_deepseek_ep(
    model_dir: &Path,
    topology: MlxParallelContext,
    options: ModelLoadOptions,
    assignment: Option<ExpertAssignment>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    let (non_expert, expert_residency, max_mapped_shards) = match options
        .weight_residency
        .expert_cache()
    {
        Some(expert_options) => (
            options.weight_residency.layers(),
            ExternalExpertResidency::SparseCache(expert_options, options.quantization),
            options.weight_residency.max_mapped_shards(),
        ),
        None if options.weight_residency.is_fully_resident() => (
            LayerWeightResidency::FullyResident,
            ExternalExpertResidency::FullyResident(options.quantization),
            crate::core::DEFAULT_MAX_MAPPED_SHARDS,
        ),
        None => {
            return Err(Error::Parallel(
                "expert-parallel loading accepts fully resident weights or non-expert residency plus an independent expert cache"
                    .into(),
            ));
        }
    };
    load_deepseek_external_ep(
        model_dir,
        topology,
        options,
        non_expert,
        expert_residency,
        max_mapped_shards,
        assignment,
        stream,
        weights_stream,
    )
}

fn load_deepseek_v4_ep(
    model_dir: &Path,
    topology: MlxParallelContext,
    options: ModelLoadOptions,
    assignment: Option<ExpertAssignment>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    if let Some(expert_options) = options.weight_residency.expert_cache() {
        return load_deepseek_v4_external_ep(
            model_dir,
            topology,
            options,
            options.weight_residency.layers(),
            ExternalExpertResidency::SparseCache(expert_options, options.quantization),
            options.weight_residency.max_mapped_shards(),
            assignment,
            stream,
            weights_stream,
        );
    }
    if !options.weight_residency.is_fully_resident() {
        return Err(Error::Parallel(
            "DeepSeek V4 EP accepts fully resident weights or non-expert residency plus an independent expert cache"
                .into(),
        ));
    }
    load_deepseek_v4_external_ep(
        model_dir,
        topology,
        options,
        LayerWeightResidency::FullyResident,
        ExternalExpertResidency::FullyResident(options.quantization),
        crate::core::DEFAULT_MAX_MAPPED_SHARDS,
        assignment,
        stream,
        weights_stream,
    )
}
fn load_qwen3_ep(
    model_dir: &Path,
    topology: MlxParallelContext,
    options: ModelLoadOptions,
    assignment: Option<ExpertAssignment>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    let (non_expert, expert_residency, max_mapped_shards) = match options
        .weight_residency
        .expert_cache()
    {
        Some(expert_options) => (
            options.weight_residency.layers(),
            ExternalExpertResidency::SparseCache(expert_options, options.quantization),
            options.weight_residency.max_mapped_shards(),
        ),
        None if options.weight_residency.is_fully_resident() => (
            LayerWeightResidency::FullyResident,
            ExternalExpertResidency::FullyResident(options.quantization),
            crate::core::DEFAULT_MAX_MAPPED_SHARDS,
        ),
        None => {
            return Err(Error::Parallel(
                "expert-parallel loading accepts fully resident weights or non-expert residency plus an independent expert cache"
                    .into(),
            ));
        }
    };
    load_qwen3_external_ep(
        model_dir,
        topology,
        options,
        non_expert,
        expert_residency,
        max_mapped_shards,
        assignment,
        stream,
        weights_stream,
    )
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
    store: std::sync::Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    entries: Vec<ExpertCatalogEntry>,
    assignment: &ExpertAssignment,
    options: ExpertCacheLoadOptions,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(ExpertCache, usize), Error> {
    let entries = entries
        .into_iter()
        .filter(|entry| assignment.owner(entry.identity().global_expert) == Some(assignment.rank()))
        .collect::<Vec<_>>();
    let cache = match quantization {
        Some(quantization) => ExpertCache::new_quantized_shared(
            store,
            entries,
            options,
            quantization,
            weights_stream.clone(),
            stream.clone(),
        )?,
        None => ExpertCache::new_shared(
            store,
            entries,
            options,
            weights_stream.clone(),
            stream.clone(),
        )?,
    };
    let owned_expert_bytes = usize::try_from(cache.report()?.owned_bytes)
        .map_err(|_| Error::Parallel("owned expert bytes exceed usize".into()))?;
    Ok((cache, owned_expert_bytes))
}

fn rank_owned_resident_experts(
    store: std::sync::Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    entries: Vec<ExpertCatalogEntry>,
    assignment: &ExpertAssignment,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(ExpertCache, usize), Error> {
    let entries = entries
        .into_iter()
        .filter(|entry| assignment.owner(entry.identity().global_expert) == Some(assignment.rank()))
        .collect::<Vec<_>>();
    let resident = match quantization {
        Some(quantization) => ExpertCache::new_quantized_resident_shared(
            store,
            entries,
            quantization,
            weights_stream.clone(),
            stream.clone(),
        )?,
        None => ExpertCache::new_resident_shared(
            store,
            entries,
            weights_stream.clone(),
            stream.clone(),
        )?,
    };
    let owned_expert_bytes = usize::try_from(resident.report()?.owned_bytes)
        .map_err(|_| Error::Parallel("owned expert bytes exceed usize".into()))?;
    Ok((resident, owned_expert_bytes))
}

#[allow(clippy::too_many_arguments)]
fn finish_additional_cached_ep(
    topology: MlxParallelContext,
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
        expert_cache,
        latest_statistics: Default::default(),
        cumulative_statistics: Default::default(),
    }
}

fn open_external_safetensors_store(
    model_dir: &Path,
    max_mapped_shards: usize,
) -> Result<std::sync::Arc<dyn eredu_checkpoint::store::CheckpointSource>, Error> {
    Ok(std::sync::Arc::new(
        SafetensorsWeightStore::open_with_max_mapped_shards(model_dir, max_mapped_shards)?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn load_additional_external_ep(
    model_dir: &Path,
    topology: MlxParallelContext,
    _options: ModelLoadOptions,
    non_expert: LayerWeightResidency,
    expert_residency: ExternalExpertResidency,
    max_mapped_shards: usize,
    assignment: Option<ExpertAssignment>,
    kind: ModelKind,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    let store = open_external_safetensors_store(model_dir, max_mapped_shards)?;
    let (assignment, architecture, store, entries, replicated_parameter_bytes) = match kind {
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
                crate::backend::mlx::architectures::gemma4::layerwise::load_gemma4_sparse_tp_ep_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    ParallelBuildContext::new(topology, ShardingPolicy::Require),
                    stream,
                    weights_stream,
                )?
            } else {
                crate::backend::mlx::architectures::gemma4::layerwise::load_gemma4_sparse_ep_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    stream,
                    weights_stream,
                )?
            };
            let store = model.checkpoint_store_arc();
            let entries =
                crate::backend::mlx::architectures::gemma4::layerwise::gemma4_expert_catalog(
                    &args,
                    store.as_ref(),
                )?;
            let replicated = planned_replicated_bytes(&model.residency_report()?)?;
            (
                assignment,
                ExpertArchitecture::Gemma4Layerwise(Box::new(model)),
                store,
                entries,
                replicated,
            )
        }
        ModelKind::GptOss => {
            let args = gpt_oss::get_model_args(model_dir)?;
            let assignment =
                resolve_model_assignment(assignment, args.num_local_experts as usize, topology)?;
            let model = if topology.tensor_parallel_size > 1 {
                crate::backend::mlx::architectures::gpt_oss::layerwise::load_gpt_oss_sparse_tp_ep_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    ParallelBuildContext::new(topology, ShardingPolicy::Require),
                    stream,
                    weights_stream,
                )?
            } else {
                crate::backend::mlx::architectures::gpt_oss::layerwise::load_gpt_oss_sparse_ep_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    stream,
                    weights_stream,
                )?
            };
            let store = model.checkpoint_store_arc();
            let entries =
                crate::backend::mlx::architectures::gpt_oss::layerwise::gpt_oss_expert_catalog(
                    &args,
                    store.as_ref(),
                )?;
            let replicated = planned_replicated_bytes(&model.residency_report()?)?;
            (
                assignment,
                ExpertArchitecture::GptOssLayerwise(Box::new(model)),
                store,
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
                        == crate::backend::mlx::architectures::inkling::model::FeedForwardPolicy::SparseMoe
                })
            {
                return Err(Error::UnsupportedArchitecture(
                    "expert parallelism requires an Inkling checkpoint with routed MoE layers"
                        .into(),
                ));
            }
            let assignment = resolve_model_assignment(assignment, global_experts, topology)?;
            let model = if topology.tensor_parallel_size > 1 {
                crate::backend::mlx::architectures::inkling::layerwise::load_inkling_sparse_tp_ep_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    ParallelBuildContext::new(topology, ShardingPolicy::Require),
                    stream,
                    weights_stream,
                )?
            } else {
                crate::backend::mlx::architectures::inkling::layerwise::load_inkling_sparse_ep_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    stream,
                    weights_stream,
                )?
            };
            let store = model.checkpoint_store_arc();
            let entries =
                crate::backend::mlx::architectures::inkling::layerwise::inkling_expert_catalog(
                    &args,
                    store.as_ref(),
                )?;
            let replicated = planned_replicated_bytes(&model.residency_report()?)?;
            (
                assignment,
                ExpertArchitecture::InklingLayerwise(Box::new(model)),
                store,
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
                crate::backend::mlx::architectures::lfm2::layerwise::load_lfm2_sparse_tp_ep_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    ParallelBuildContext::new(topology, ShardingPolicy::Require),
                    stream,
                    weights_stream,
                )?
            } else {
                crate::backend::mlx::architectures::lfm2::layerwise::load_lfm2_sparse_ep_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    stream,
                    weights_stream,
                )?
            };
            let store = model.checkpoint_store_arc();
            let entries = crate::backend::mlx::architectures::lfm2::layerwise::lfm2_expert_catalog(
                &args,
                store.as_ref(),
            )?;
            let replicated = planned_replicated_bytes(&model.residency_report()?)?;
            (
                assignment,
                ExpertArchitecture::Lfm2Layerwise(Box::new(model)),
                store,
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
                crate::backend::mlx::architectures::nemotron_h::layerwise::
                    load_nemotron_h_sparse_tp_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        ParallelBuildContext::new(topology, ShardingPolicy::Require),
                        stream,
                        weights_stream,
                    )?
            } else {
                crate::backend::mlx::architectures::nemotron_h::layerwise::
                    load_nemotron_h_sparse_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        stream,
                        weights_stream,
                    )?
            };
            let store = model.checkpoint_store_arc();
            let entries = crate::backend::mlx::architectures::nemotron_h::layerwise::nemotron_h_expert_catalog(
                &args,
                store.as_ref(),
            )?;
            let replicated = planned_replicated_bytes(&model.residency_report()?)?;
            (
                assignment,
                ExpertArchitecture::NemotronHLayerwise(Box::new(model)),
                store,
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
                crate::backend::mlx::architectures::qwen::hybrid::layerwise::
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
                crate::backend::mlx::architectures::qwen::hybrid::layerwise::
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
            let store = model.checkpoint_store_arc();
            let entries =
                crate::backend::mlx::architectures::qwen::hybrid::layerwise::qwen_hybrid_expert_catalog(
                    &args,
                    store.as_ref(),
                )?;
            let replicated = planned_replicated_bytes(&model.residency_report()?)?;
            (
                assignment,
                ExpertArchitecture::QwenHybridLayerwise(Box::new(model)),
                store,
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
                crate::backend::mlx::architectures::qwen::vl::layerwise::
                    load_qwen3_vl_sparse_tp_ep_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        ParallelBuildContext::new(topology, ShardingPolicy::Require),
                        stream,
                        weights_stream,
                    )?
            } else {
                crate::backend::mlx::architectures::qwen::vl::layerwise::load_qwen3_vl_sparse_ep_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    stream,
                    weights_stream,
                )?
            };
            let store = model.checkpoint_store_arc();
            let entries = crate::backend::mlx::architectures::qwen::dense::layerwise::qwen3_expert_catalog_at(
                &args.text_config,
                store.as_ref(),
                "model.language_model.layers",
            )?;
            let replicated = planned_replicated_bytes(&model.residency_report()?)?;
            (
                assignment,
                ExpertArchitecture::Qwen3VlLayerwise(Box::new(model)),
                store,
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
    topology: MlxParallelContext,
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
            ExternalExpertResidency::SparseCache(expert_options, options.quantization),
            options.weight_residency.max_mapped_shards(),
        )
    } else if options.weight_residency.is_fully_resident() {
        (
            LayerWeightResidency::FullyResident,
            ExternalExpertResidency::FullyResident(options.quantization),
            crate::core::DEFAULT_MAX_MAPPED_SHARDS,
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
    use crate::backend::mlx::runtime::residency::dense_stream::DenseDiskStreamLoadOptions;
    use crate::backend::mlx::DeviceAssignment;
    use safemlx::DeviceType;

    let topology =
        MlxParallelContext::for_rank(1, 1, 1, 2, DeviceAssignment::new(DeviceType::Gpu, 0))
            .unwrap();
    let model = load_expert_parallel_model_with_options(
        model_dir,
        ModelLoadOptions {
            quantization: None,
            parallel: Some(topology),
            weight_residency: WeightResidency::with_expert_cache(
                crate::backend::mlx::runtime::execution::layerwise::NonExpertWeightResidency::LayerwiseHost(Default::default()),
                expert_options,
            ),
        },
        stream,
        weights_stream,
    )
    .unwrap();
    assert_eq!(model.info().model_kind, expected_kind);
    assert_eq!(model.info().topology.expert_parallel_rank, 1);
    assert_eq!(model.info().topology.expert_parallel_size, 2);
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
                crate::backend::mlx::runtime::execution::layerwise::NonExpertWeightResidency::FullyResident,
                expert_options,
            ),
        },
        stream,
        weights_stream,
    )
    .unwrap();
    assert_eq!(resident_non_experts.info().model_kind, expected_kind);
    assert_eq!(
        resident_non_experts
            .expert_cache_report()
            .unwrap()
            .unwrap()
            .owned_experts,
        expected_owned_experts
    );

    let dense = DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1).unwrap();
    let streamed = load_expert_parallel_model_with_options(
        model_dir,
        ModelLoadOptions {
            quantization: None,
            parallel: Some(topology),
            weight_residency: WeightResidency::with_expert_cache(
                crate::backend::mlx::runtime::execution::layerwise::NonExpertWeightResidency::DenseDiskStream(dense),
                expert_options,
            ),
        },
        stream,
        weights_stream,
    )
    .unwrap();
    assert_eq!(streamed.info().model_kind, expected_kind);
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
pub(crate) fn assert_rank_owned_quantized_sparse_ep_load(
    model_dir: &Path,
    expert_options: ExpertCacheLoadOptions,
    quantization: WeightQuantization,
    expected_kind: ModelKind,
    expected_owned_experts: usize,
    stream: &Stream,
    weights_stream: &Stream,
) {
    use crate::backend::mlx::DeviceAssignment;
    use safemlx::DeviceType;

    let topology =
        MlxParallelContext::for_rank(1, 1, 1, 2, DeviceAssignment::new(DeviceType::Gpu, 0))
            .unwrap();
    let model = load_expert_parallel_model_with_options(
        model_dir,
        ModelLoadOptions::with_quantization(quantization)
            .with_parallel_topology(topology)
            .with_weight_residency(WeightResidency::with_expert_cache(
                crate::backend::mlx::runtime::execution::layerwise::NonExpertWeightResidency::LayerwiseHost(Default::default()),
                expert_options,
            )),
        stream,
        weights_stream,
    )
    .unwrap();
    assert_eq!(model.info().model_kind, expected_kind);
    assert_eq!(model.info().topology.expert_parallel_rank, 1);
    assert_eq!(model.info().topology.expert_parallel_size, 2);
    let report = model.expert_cache_report().unwrap().unwrap();
    assert_eq!(report.owned_experts, expected_owned_experts);
    assert_eq!(report.weight_quantization, Some(quantization));
    let materialization = report.materialization.unwrap();
    assert!(materialization.transformed_weights > 0);
    assert!(materialization.source_tiles > 0);
    assert!(materialization.source_bytes_read > materialization.output_bytes);
    assert!(
        materialization.peak_planned_working_set_bytes <= expert_options.compact_bank_scratch_bytes
    );
}

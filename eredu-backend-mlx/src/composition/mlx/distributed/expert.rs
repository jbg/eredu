//! Reusable expert-parallel assignment, routing, and exchange infrastructure.

//!
//! Pure expert parallelism keeps ordinary model state replicated and partitions
//! only routed expert banks. [`crate::backend::mlx::runtime::distributed::expert::dispatch_replicated`]
//! exploits the replicated
//! token layout: ranks compact only routes owned by their experts and all-sum
//! the resulting token buffer. Sharded-token dispatch uses compact native or
//! topology-routed variable-count all-to-all payload exchange.

use eredu_architectures::{gpt_oss as gpt_oss_arch, kimi_linear as kimi_linear_arch};
use eredu_checkpoint::WeightQuantization;
use eredu_runtime::{ActivationObserver as RuntimeActivationObserver, ShardingPolicy};

use std::{
    path::{Path, PathBuf},
    time::Instant,
};

#[cfg(test)]
use safemlx::module::{ModuleParameters, Param};
use safemlx::{
    distributed::Group,
    error::Exception,
    ops::{indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue},
    Array, Stream,
};

use eredu_core::cache::{
    validate_prompt_cache_model_identity, PromptCacheDescriptor, PromptCacheManifest,
    PromptCacheModelIdentity, PromptCacheOptions,
};

use crate::{
    backend::mlx::error::Error,
    backend::mlx::runtime::cache::state::{MlxHybridState, MlxKeyValueState},
    backend::mlx::runtime::checkpoint::store::{
        open_gguf_checkpoint_source, SafetensorsWeightStore,
    },
    backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    backend::mlx::runtime::generation::sampler::SpeculativeSampler,
    backend::mlx::runtime::media::input as runtime_input,
    backend::mlx::runtime::residency::expert_cache::{
        ExpertCache, ExpertCacheReport, ExpertCatalogEntry,
    },
    backend::mlx::{MlxParallelContext, ModelLoadOptions},
    composition::mlx::speculative::embedded::{
        DistributedEmbeddedMtpSampler, EmbeddedMtpOutput, EmbeddedMtpTarget,
    },
    composition::{gpt_oss, kimi_linear, lfm2, nemotron_h},
};
use eredu_core::generation::MtpConfig;
use eredu_core::{ModelKind, MtpCapability, MtpCheckpointKind, MtpStats};
#[cfg(test)]
use eredu_runtime::NonExpertWeightResidency;
use eredu_runtime::{
    CacheResidencyPolicy, CacheResidencyReport, ExpertCacheLoadOptions, ExpertPass,
    LayerWeightResidency, PagedCacheOptions, RoutedExpertTensorParallelOutput, WeightResidency,
};

#[cfg(test)]
use crate::backend::mlx::nn::moe::PackedGatedProductExperts;

pub use crate::backend::mlx::runtime::distributed::expert::*;

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
#[allow(dead_code)]
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
    /// Neutral DeepSeek V3/V4 architecture state.
    DeepSeek(crate::composition::deepseek::DeepSeekState),
    /// Architecture-declared heterogeneous attention and fixed state.
    Hybrid(MlxHybridState),
    /// Neutral key/value state, device-resident or paged.
    KeyValue(MlxKeyValueState),
}

impl ExpertParallelCache {
    /// Clears all cached attention state.
    pub fn reset(&mut self) -> Result<(), Error> {
        match self {
            Self::DeepSeek(cache) => cache.clear()?,
            Self::Hybrid(cache) => cache.clear()?,
            Self::KeyValue(cache) => cache.clear()?,
        }
        Ok(())
    }
}

#[cfg(test)]
impl ExpertParallelCache {
    pub fn offset(&self) -> i32 {
        match self {
            Self::DeepSeek(cache) => cache.offset(),
            Self::Hybrid(cache) => cache.offset(),
            Self::KeyValue(cache) => cache.offset(),
        }
    }
}

trait NeutralHybridExpertArchitecture {
    fn dense_stream_report(&self) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error>;
    fn new_cache(&self) -> MlxHybridState;
    fn new_cache_with_options(&self, policy: CacheResidencyPolicy)
        -> Result<MlxHybridState, Error>;
    fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error>;
    fn save_prompt_cache(
        &self,
        cache: &mut MlxHybridState,
        directory: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        identity: &PromptCacheModelIdentity,
    ) -> Result<PromptCacheManifest, Error>;
    fn load_prompt_cache(
        &self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        identity: &PromptCacheModelIdentity,
        stream: &Stream,
    ) -> Result<(MlxHybridState, PromptCacheManifest), Error>;
    #[allow(clippy::too_many_arguments)]
    fn forward_expert_parallel(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut MlxHybridState,
        tensor_group: Option<&Group>,
        expert_group: &Group,
        assignment: &ExpertAssignment,
        expert_cache: &ExpertCache,
        pass: ExpertPass,
        statistics: &mut RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Error>;
    fn mtp_len(&self) -> usize {
        0
    }
    fn mtp_state_start(&self) -> Option<usize> {
        None
    }
    #[allow(clippy::too_many_arguments)]
    fn forward_mtp_target(
        &mut self,
        _tokens: &Array,
        _cache: &mut MlxHybridState,
        _tensor_group: Option<&Group>,
        _expert_group: &Group,
        _assignment: &ExpertAssignment,
        _expert_cache: &ExpertCache,
        _statistics: &mut RoutingStatistics,
        _stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Exception> {
        Err(Exception::custom(
            "neutral hybrid architecture has no embedded prediction groups",
        ))
    }
    #[allow(clippy::too_many_arguments)]
    fn forward_mtp_draft(
        &mut self,
        _hidden: &Array,
        _tokens: &Array,
        _depth: usize,
        _cache: &mut MlxHybridState,
        _tensor_group: Option<&Group>,
        _expert_group: &Group,
        _assignment: &ExpertAssignment,
        _expert_cache: &ExpertCache,
        _statistics: &mut RoutingStatistics,
        _stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Exception> {
        Err(Exception::custom(
            "neutral hybrid architecture has no embedded prediction groups",
        ))
    }
}

macro_rules! impl_neutral_hybrid_expert_architecture {
    ($model:ty, $execute:path $(, $extra:item)*) => {
        impl NeutralHybridExpertArchitecture for $model {
            fn dense_stream_report(
                &self,
            ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
                self.dense_stream_report()
            }

            fn new_cache(&self) -> MlxHybridState {
                self.new_cache()
            }

            fn new_cache_with_options(
                &self,
                policy: CacheResidencyPolicy,
            ) -> Result<MlxHybridState, Error> {
                self.new_cache_with_options(policy)
            }

            fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
                self.prompt_cache_model_identity()
            }

            fn save_prompt_cache(
                &self,
                cache: &mut MlxHybridState,
                directory: &Path,
                descriptor: PromptCacheDescriptor,
                prefix_token_ids: &[u32],
                options: &PromptCacheOptions,
                identity: &PromptCacheModelIdentity,
            ) -> Result<PromptCacheManifest, Error> {
                self.save_prompt_cache_with_identity(
                    cache,
                    directory,
                    descriptor,
                    prefix_token_ids,
                    options,
                    identity,
                )
            }

            fn load_prompt_cache(
                &self,
                directory: &Path,
                expected: &PromptCacheDescriptor,
                prefix_token_ids: &[u32],
                options: PagedCacheOptions,
                identity: &PromptCacheModelIdentity,
                stream: &Stream,
            ) -> Result<(MlxHybridState, PromptCacheManifest), Error> {
                self.load_prompt_cache_with_identity(
                    directory,
                    expected,
                    prefix_token_ids,
                    options,
                    identity,
                    stream,
                )
            }

            fn forward_expert_parallel(
                &mut self,
                tokens: &Array,
                mask: Option<&Array>,
                cache: &mut MlxHybridState,
                tensor_group: Option<&Group>,
                expert_group: &Group,
                assignment: &ExpertAssignment,
                expert_cache: &ExpertCache,
                pass: ExpertPass,
                statistics: &mut RoutingStatistics,
                stream: &Stream,
            ) -> Result<Array, Error> {
                let args = self.args().clone();
                let mut execute =
                    |layer, hidden: &Array, ids: &Array, weights: &Array, stream: &Stream| {
                        let returned = dispatch_replicated_with(
                            hidden,
                            ids,
                            weights,
                            assignment,
                            expert_group,
                            stream,
                            |routes, stream| {
                                $execute(&args, layer, routes, pass, expert_cache, stream)
                            },
                        )
                        .map_err(|error| Exception::custom(error.to_string()))?;
                        statistics.accumulate(&returned.statistics);
                        Ok(returned.reduced_output)
                    };
                match tensor_group {
                    Some(tensor_group) => self.forward_tensor_expert_parallel(
                        tokens,
                        mask,
                        cache,
                        tensor_group,
                        &mut execute,
                        stream,
                    ),
                    None => self.forward_with_expert_executor(
                        tokens,
                        mask,
                        cache,
                        &mut execute,
                        stream,
                    ),
                }
            }

            $($extra)*
        }
    };
}

impl_neutral_hybrid_expert_architecture!(kimi_linear::KimiLinearModel, execute_cached_kimi_linear);
impl_neutral_hybrid_expert_architecture!(lfm2::Lfm2Model, execute_cached_lfm2);
impl_neutral_hybrid_expert_architecture!(
    nemotron_h::NemotronHModel,
    execute_cached_nemotron_h,
    fn mtp_len(&self) -> usize {
        self.mtp_len()
    },
    fn mtp_state_start(&self) -> Option<usize> {
        Some(self.args().num_hidden_layers as usize)
    },
    fn forward_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut MlxHybridState,
        tensor_group: Option<&Group>,
        expert_group: &Group,
        assignment: &ExpertAssignment,
        expert_cache: &ExpertCache,
        statistics: &mut RoutingStatistics,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Exception> {
        let args = self.args().clone();
        self.forward_mtp_target_with_expert_executor(
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
        )
    },
    fn forward_mtp_draft(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut MlxHybridState,
        tensor_group: Option<&Group>,
        expert_group: &Group,
        assignment: &ExpertAssignment,
        expert_cache: &ExpertCache,
        statistics: &mut RoutingStatistics,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Exception> {
        let args = self.args().clone();
        self.forward_mtp_draft_with_expert_executor(
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
        )
    }
);

enum ExpertArchitecture {
    DeepSeek(Box<crate::composition::deepseek::DeepSeekModel>),
    NeutralHybrid(Box<dyn NeutralHybridExpertArchitecture>),
    Qwen(Box<crate::composition::qwen::QwenModel>),
    GptOss(Box<gpt_oss::GptOssModel>),
}

impl ExpertArchitecture {
    fn bind_parallel_topology(&mut self, topology: MlxParallelContext) {
        match self {
            Self::DeepSeek(_) => {}
            Self::NeutralHybrid(_) => {}
            Self::Qwen(_) => {}
            Self::GptOss(model) => model.bind_parallel_topology(topology),
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

#[derive(Clone)]
enum ExpertParallelMtpDraftCache {
    DeepSeek(crate::composition::deepseek::DeepSeekState),
    Hybrid(MlxHybridState),
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
        tokens: &crate::MlxTensor,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Exception> {
        self.model.forward_embedded_mtp_target(
            tokens.as_array(),
            cache,
            self.tensor_group,
            self.expert_group,
            stream,
        )
    }

    fn prefill_draft_cache(
        &mut self,
        output: &EmbeddedMtpOutput,
        tokens: &crate::MlxTensor,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let sequence = tokens.as_array().dim(1);
        if sequence <= 1 {
            return Ok(());
        }
        let hidden = output
            .hidden
            .as_array()
            .try_index_device((.., ..sequence - 1, ..), stream)?;
        let next = tokens.as_array().try_index_device((.., 1..), stream)?;
        let mut draft = self.draft_cache(cache);
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
        self.commit_draft_cache(cache, &draft);
        Ok(())
    }

    fn draft_cache(&self, cache: &Self::Cache) -> Self::DraftCache {
        match cache {
            ExpertParallelCache::DeepSeek(cache) => {
                ExpertParallelMtpDraftCache::DeepSeek(cache.clone())
            }
            ExpertParallelCache::Hybrid(cache) => ExpertParallelMtpDraftCache::Hybrid(
                cache
                    .deep_clone_state()
                    .expect("evaluated Nemotron-H EP draft state must be forkable"),
            ),
            _ => unreachable!("capability preflight rejects non-shared embedded MTP caches"),
        }
    }

    fn commit_draft_cache(&self, cache: &mut Self::Cache, draft: &Self::DraftCache) {
        match (cache, draft) {
            (
                ExpertParallelCache::DeepSeek(cache),
                ExpertParallelMtpDraftCache::DeepSeek(draft),
            ) => *cache = draft.clone(),
            (ExpertParallelCache::Hybrid(cache), ExpertParallelMtpDraftCache::Hybrid(draft)) => {
                let ExpertArchitecture::NeutralHybrid(model) = &self.model.architecture else {
                    unreachable!("hybrid draft cache requires a neutral hybrid model")
                };
                let start = model
                    .mtp_state_start()
                    .expect("embedded hybrid draft has appended prediction state");
                cache
                    .commit_layer_range_from(draft, start)
                    .expect("validated neutral hybrid EP draft state layout")
            }
            _ => unreachable!("embedded MTP draft-cache type changed after preflight"),
        }
    }

    fn restore_target_checkpoint(
        cache: &mut Self::Cache,
        checkpoint: &Self::Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        match (cache, checkpoint) {
            (ExpertParallelCache::Hybrid(cache), ExpertParallelCache::Hybrid(checkpoint)) => {
                cache.restore_checkpoint(checkpoint, stream)
            }
            (cache, checkpoint) => {
                cache.clone_from(checkpoint);
                Ok(())
            }
        }
    }

    fn draft_logits(
        &mut self,
        hidden: &crate::MlxTensor,
        last_token: u32,
        draft_index: usize,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(crate::MlxTensor, crate::MlxTensor), Exception> {
        let token = Array::from_slice(&[last_token], &[1, 1]);
        let output = self.model.forward_embedded_mtp_draft(
            hidden.as_array(),
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
        hidden: &crate::MlxTensor,
        tokens: &crate::MlxTensor,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        for depth in 0..self.max_draft_tokens() {
            let _ = self.model.forward_embedded_mtp_draft(
                hidden.as_array(),
                tokens.as_array(),
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
        hidden: &crate::MlxTensor,
        last_token: u32,
        proposal_capacity: usize,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<Option<crate::MlxTensor>, Exception> {
        let _ = (hidden, last_token, proposal_capacity, cache, stream);
        Ok(None)
    }

    fn adjust_fused_draft_logits(
        &mut self,
        logits: crate::MlxTensor,
        last_token: u32,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Exception> {
        let _ = (last_token, stream);
        Ok(logits)
    }

    fn max_draft_tokens(&self) -> usize {
        match &self.model.architecture {
            ExpertArchitecture::DeepSeek(model) => model.mtp_len(),
            ExpertArchitecture::NeutralHybrid(model) => model.mtp_len(),
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
            ExpertArchitecture::DeepSeek(model) if model.mtp_len() > 0 => MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Embedded,
            },
            ExpertArchitecture::NeutralHybrid(model) if model.mtp_len() > 0 => {
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
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match &self.architecture {
            ExpertArchitecture::DeepSeek(model) => model.dense_stream_report(),
            ExpertArchitecture::NeutralHybrid(model) => model.dense_stream_report(),
            ExpertArchitecture::Qwen(model) => model.dense_stream_report(),
            ExpertArchitecture::GptOss(model) => model.dense_stream_report(),
        }
    }

    /// Allocates an empty architecture-appropriate replicated cache.
    pub fn new_cache(&self) -> ExpertParallelCache {
        match &self.architecture {
            ExpertArchitecture::DeepSeek(model) => ExpertParallelCache::DeepSeek(
                model
                    .new_state()
                    .expect("validated DeepSeek distributed state geometry"),
            ),
            ExpertArchitecture::NeutralHybrid(model) => {
                ExpertParallelCache::Hybrid(model.new_cache())
            }
            ExpertArchitecture::Qwen(model) => ExpertParallelCache::KeyValue(model.new_cache()),
            ExpertArchitecture::GptOss(model) => ExpertParallelCache::KeyValue(model.new_cache()),
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
                ExpertArchitecture::DeepSeek(model) => model
                    .new_state_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ExpertParallelCache::DeepSeek),
                ExpertArchitecture::NeutralHybrid(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ExpertParallelCache::Hybrid),
                ExpertArchitecture::GptOss(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ExpertParallelCache::KeyValue),
                ExpertArchitecture::Qwen(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ExpertParallelCache::KeyValue),
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
            ExpertParallelCache::Hybrid(cache) => cache.residency_report().map_err(Into::into),
            ExpertParallelCache::KeyValue(cache) => cache.residency_report().map_err(Into::into),
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
        match (&self.architecture, cache) {
            (ExpertArchitecture::DeepSeek(model), ExpertParallelCache::DeepSeek(cache)) => {
                model.save_prompt_cache(cache, directory, descriptor, prefix_token_ids, options)
            }
            (ExpertArchitecture::GptOss(model), ExpertParallelCache::KeyValue(cache)) => model
                .save_prompt_cache(
                    cache,
                    directory,
                    descriptor,
                    prefix_token_ids,
                    options,
                    stream,
                ),
            (ExpertArchitecture::NeutralHybrid(model), ExpertParallelCache::Hybrid(cache)) => model
                .save_prompt_cache(
                    cache,
                    &directory,
                    descriptor,
                    prefix_token_ids,
                    options,
                    &identity,
                ),
            (ExpertArchitecture::Qwen(model), ExpertParallelCache::KeyValue(cache)) => model
                .save_prompt_cache(
                    cache,
                    directory,
                    descriptor,
                    prefix_token_ids,
                    options,
                    stream,
                ),
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
            ExpertArchitecture::DeepSeek(model) => model
                .load_prompt_cache(&directory, expected, prefix_token_ids, options, stream)
                .map(|(cache, manifest)| (ExpertParallelCache::DeepSeek(cache), manifest)),
            ExpertArchitecture::NeutralHybrid(model) => model
                .load_prompt_cache(
                    &directory,
                    expected,
                    prefix_token_ids,
                    options,
                    &identity,
                    stream,
                )
                .map(|(cache, manifest)| (ExpertParallelCache::Hybrid(cache), manifest)),
            ExpertArchitecture::Qwen(model) => model
                .load_prompt_cache(&directory, expected, prefix_token_ids, options, stream)
                .map(|(cache, manifest)| (ExpertParallelCache::KeyValue(cache), manifest)),
            ExpertArchitecture::GptOss(model) => model
                .load_prompt_cache(&directory, expected, prefix_token_ids, options, stream)
                .map(|(cache, manifest)| (ExpertParallelCache::KeyValue(cache), manifest)),
        }
    }

    fn prompt_cache_rank_directory(&self, root: &Path) -> PathBuf {
        root.join(format!("rank-{:05}", self.topology.global_rank))
    }

    /// Returns the canonical cache-relevant architecture identity for this rank.
    #[allow(dead_code)]
    pub fn prompt_cache_architecture_fingerprint(&self) -> Result<String, Error> {
        Ok(self.prompt_cache_model_identity()?.architecture_fingerprint)
    }

    /// Returns this rank's exact ordered prompt-cache layout.
    #[allow(dead_code)]
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<eredu_core::LayerSchedule<eredu_core::cache::LayerCachePolicy>, Error> {
        Ok(self.prompt_cache_model_identity()?.layer_layout)
    }

    /// Returns each owned layer's processed-token delta from the persisted prefix.
    #[allow(dead_code)]
    pub fn prompt_cache_layer_prefix_offsets(&self) -> Result<Vec<i32>, Error> {
        Ok(self.prompt_cache_model_identity()?.layer_prefix_offsets)
    }

    pub fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        let mut identity = match &self.architecture {
            ExpertArchitecture::DeepSeek(model) => model.prompt_cache_identity()?,
            ExpertArchitecture::NeutralHybrid(model) => model.prompt_cache_model_identity()?,
            ExpertArchitecture::Qwen(model) => model.prompt_cache_model_identity()?,
            ExpertArchitecture::GptOss(model) => model.prompt_cache_model_identity()?,
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
        observer: Option<&mut dyn RuntimeActivationObserver<Array, Exception>>,
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
                (ExpertArchitecture::DeepSeek(model), ExpertParallelCache::DeepSeek(cache)) => {
                    if mask.is_some() {
                        return Err(Error::Parallel(
                            "explicit masks are unsupported by neutral DeepSeek EP execution"
                                .into(),
                        ));
                    }
                    let mut execute = |execution, stream: &Stream| {
                        execute_cached_neutral_deepseek(
                            execution,
                            pass,
                            expert_cache,
                            assignment,
                            group,
                            &mut statistics,
                            stream,
                        )
                        .map_err(|error| Exception::custom(error.to_string()))
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
                (ExpertArchitecture::NeutralHybrid(model), ExpertParallelCache::Hybrid(cache)) => {
                    model.forward_expert_parallel(
                        tokens,
                        mask,
                        cache,
                        tensor_group,
                        group,
                        assignment,
                        expert_cache,
                        pass,
                        &mut statistics,
                        stream,
                    )?
                }
                (ExpertArchitecture::Qwen(model), ExpertParallelCache::KeyValue(cache)) => {
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
                                    execute_cached_neutral_qwen3(
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
                (ExpertArchitecture::GptOss(model), ExpertParallelCache::KeyValue(cache)) => {
                    let args = model.args().clone();
                    let mut provider = gpt_oss::expert::distributed_provider(
                        &args,
                        assignment,
                        Some(group),
                        expert_cache,
                        &mut statistics,
                    );
                    match tensor_group {
                        Some(tensor_group) => model.forward_tensor_expert_provider(
                            tokens,
                            mask,
                            cache,
                            tensor_group,
                            &mut provider,
                            stream,
                        )?,
                        None => model.forward_with_expert_provider(
                            tokens,
                            mask,
                            cache,
                            &mut provider,
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
            (ExpertArchitecture::DeepSeek(model), ExpertParallelCache::DeepSeek(cache)) => {
                let mut execute = |execution, stream: &Stream| {
                    execute_cached_neutral_deepseek(
                        execution,
                        pass,
                        expert_cache,
                        assignment,
                        expert_group,
                        &mut statistics,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()))
                };
                let input = eredu_architectures::deepseek::mtp::EmbeddedInput::target(tokens, None);
                let (logits, hidden) = match tensor_group {
                    Some(tensor_group) => model.forward_embedded_tensor_expert_parallel(
                        input,
                        cache,
                        tensor_group,
                        &mut execute,
                        stream,
                    )?,
                    None => model.forward_embedded_with_expert_executor(
                        input,
                        cache,
                        &mut execute,
                        stream,
                    )?,
                };
                EmbeddedMtpOutput {
                    logits: crate::MlxTensor::from_array(logits),
                    hidden: crate::MlxTensor::from_array(hidden),
                    tokens: crate::MlxTensor::from_array(tokens.clone()),
                }
            }
            (ExpertArchitecture::NeutralHybrid(model), ExpertParallelCache::Hybrid(cache)) => model
                .forward_mtp_target(
                    tokens,
                    cache,
                    tensor_group,
                    expert_group,
                    assignment,
                    expert_cache,
                    &mut statistics,
                    stream,
                )?,
            _ => {
                return Err(Exception::custom(
                    "embedded MTP architecture/cache mismatch for EP execution",
                ))
            }
        };
        materialize_timing_phase([output.logits.as_array()])
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
            (ExpertArchitecture::DeepSeek(model), ExpertParallelMtpDraftCache::DeepSeek(cache)) => {
                let mut execute = |execution, stream: &Stream| {
                    execute_cached_neutral_deepseek(
                        execution,
                        pass,
                        expert_cache,
                        assignment,
                        expert_group,
                        &mut statistics,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()))
                };
                let input =
                    eredu_architectures::deepseek::mtp::EmbeddedInput::draft(tokens, hidden, depth);
                let (logits, next_hidden) = match tensor_group {
                    Some(tensor_group) => model.forward_embedded_tensor_expert_parallel(
                        input,
                        cache,
                        tensor_group,
                        &mut execute,
                        stream,
                    )?,
                    None => model.forward_embedded_with_expert_executor(
                        input,
                        cache,
                        &mut execute,
                        stream,
                    )?,
                };
                EmbeddedMtpOutput {
                    logits: crate::MlxTensor::from_array(logits),
                    hidden: crate::MlxTensor::from_array(next_hidden),
                    tokens: crate::MlxTensor::from_array(tokens.clone()),
                }
            }
            (
                ExpertArchitecture::NeutralHybrid(model),
                ExpertParallelMtpDraftCache::Hybrid(cache),
            ) => model.forward_mtp_draft(
                hidden,
                tokens,
                depth,
                cache,
                tensor_group,
                expert_group,
                assignment,
                expert_cache,
                &mut statistics,
                stream,
            )?,
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
            .global_rank_for(eredu_core::ParallelCoordinates {
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
        let mut target = ExpertParallelEmbeddedMtpTarget {
            model: self,
            tensor_group,
            expert_group,
        };
        let mut executor =
            crate::composition::mlx::speculative::embedded::EmbeddedMtpExecutor::new(&mut target);
        let result = crate::composition::mlx::speculative::scheduler::generate_tokens(
            &mut executor,
            cache,
            input,
            config,
            prng_key,
            &mut synchronized,
            crate::composition::mlx::speculative::MtpExecutionStreams::single(stream),
            eredu_core::generation::MtpSchedulerOptions::default(),
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
pub fn load_expert_parallel_model_with_options(
    model_dir: impl AsRef<Path>,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    load_expert_parallel_model_impl(model_dir, options, None, stream, weights_stream)
}

#[cfg(test)]
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
    let kind = model_type
        .and_then(|model_type| ModelKind::from_model_type(model_type).ok())
        .or_else(|| {
            crate::composition::mlx::resolve_model_config(&config)
                .ok()
                .map(|resolved| resolved.kind)
        });
    if kind.is_some_and(ModelKind::requires_realtime_loader) {
        return Err(Error::UnsupportedArchitecture(
            "Moshi-family models use a realtime multi-stream temporal/depth contract, not expert-parallel decoder execution"
                .into(),
        ));
    }
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
        observer: &mut impl RuntimeActivationObserver<Array, Exception>,
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

struct CachedDeepSeekLocalBank<'a> {
    spec: &'a eredu_nn::GatedProductExpertBankSpec,
    layer: usize,
    pass: ExpertPass,
    cache: &'a ExpertCache,
    local_global_expert_ids: &'a [usize],
}

impl CachedDeepSeekLocalBank<'_> {
    fn global_ids(&self, local_ids: &Array, stream: &Stream) -> Result<Array, Error> {
        let ids = self
            .local_global_expert_ids
            .iter()
            .map(|id| i32::try_from(*id))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| Error::Parallel("DeepSeek expert id exceeds i32".into()))?;
        let lookup = Array::from_slice(&ids, &[ids.len() as i32]);
        Ok(lookup.take_axis(local_ids, 0, stream)?)
    }
}

impl LocalExpertBank for CachedDeepSeekLocalBank<'_> {
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        local_expert_ids: &Array,
        stream: &Stream,
    ) -> Result<Array, Error> {
        crate::backend::mlx::runtime::residency::expert_provider::execute_cached_gated_product_dispatched(
            self.cache,
            self.spec,
            self.layer,
            hidden,
            &self.global_ids(local_expert_ids, stream)?,
            self.pass,
            stream,
        )
    }

    fn execute_local_routes_tensor_parallel(
        &mut self,
        hidden: &Array,
        local_expert_ids: &Array,
        partitions: usize,
        stream: &Stream,
    ) -> Result<eredu_nn::TensorParallelExpertOutput<Array>, Error> {
        let expert_ids = self
            .global_ids(local_expert_ids, stream)?
            .reshape(&[-1, 1], stream)?;
        let route_weights = safemlx::ops::ones_dtype(&[hidden.dim(0), 1], hidden.dtype(), stream)?;
        crate::backend::mlx::runtime::residency::expert_provider::execute_cached_gated_product_tensor_parallel(
            self.cache,
            self.spec,
            self.layer,
            hidden,
            &expert_ids,
            &route_weights,
            self.pass,
            partitions,
            stream,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_cached_neutral_deepseek(
    execution: crate::backend::mlx::runtime::residency::expert_provider::GatedProductExpertExecution,
    pass: ExpertPass,
    cache: &ExpertCache,
    assignment: &ExpertAssignment,
    group: &Group,
    statistics: &mut RoutingStatistics,
    stream: &Stream,
) -> Result<RoutedExpertTensorParallelOutput<Array>, Error> {
    use crate::backend::mlx::runtime::residency::expert_provider::GatedProductExpertExecutionMode;

    let mut bank = CachedDeepSeekLocalBank {
        spec: &execution.spec,
        layer: execution.layer,
        pass,
        cache,
        local_global_expert_ids: assignment.local_global_expert_ids(),
    };
    match execution.mode {
        GatedProductExpertExecutionMode::Complete => {
            let returned = dispatch_replicated_with(
                &execution.hidden,
                &execution.expert_ids,
                &execution.route_weights,
                assignment,
                group,
                stream,
                |routes, stream| {
                    bank.execute_local_routes(&routes.hidden, &routes.local_expert_ids, stream)
                },
            )?;
            statistics.accumulate(&returned.statistics);
            Ok(RoutedExpertTensorParallelOutput::Complete(
                returned.reduced_output,
            ))
        }
        GatedProductExpertExecutionMode::TensorParallel { partitions } => {
            let returned = dispatch_replicated_tensor_parallel(
                &execution.hidden,
                &execution.expert_ids,
                &execution.route_weights,
                assignment,
                &mut bank,
                group,
                partitions,
                stream,
            )?;
            statistics.accumulate(&returned.statistics);
            Ok(RoutedExpertTensorParallelOutput::Partial(returned.output))
        }
    }
}

pub fn execute_cached_neutral_gemma4(
    args: &eredu_architectures::gemma4::ModelArgs,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    let spec = eredu_architectures::gemma4::text::expert_bank_spec(args, layer)?;
    crate::backend::mlx::runtime::residency::expert_provider::execute_cached_gated_product_dispatched(
        cache,
        &spec,
        layer,
        &routes.hidden,
        &routes.global_expert_ids,
        pass,
        stream,
    )
}

pub fn execute_cached_kimi_linear(
    args: &kimi_linear_arch::ModelArgs,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    let spec = kimi_linear_arch::moe::expert_bank_spec(args, layer)?;
    crate::backend::mlx::runtime::residency::expert_provider::execute_cached_gated_product_dispatched(
        cache,
        &spec,
        layer,
        &routes.hidden,
        &routes.global_expert_ids,
        pass,
        stream,
    )
}

pub fn execute_cached_neutral_qwen3(
    args: &eredu_architectures::qwen::ModelArgs,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    crate::composition::qwen::expert::execute_cached_dispatched(
        cache,
        args,
        layer,
        &routes.hidden,
        &routes.global_expert_ids,
        pass,
        stream,
    )
}

pub fn execute_cached_neutral_inkling(
    args: &eredu_architectures::inkling::ModelArgs,
    cache_layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    let spec = eredu_architectures::inkling::text::expert_bank_spec(args, cache_layer)?;
    crate::backend::mlx::runtime::residency::expert_provider::execute_cached_gated_product_dispatched(
        cache,
        &spec,
        cache_layer,
        &routes.hidden,
        &routes.global_expert_ids,
        pass,
        stream,
    )
}

pub fn execute_cached_lfm2(
    args: &eredu_architectures::lfm2::ModelArgs,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    let spec = eredu_architectures::lfm2::moe::expert_bank_spec(args, layer)?;
    crate::backend::mlx::runtime::residency::expert_provider::execute_cached_gated_product_dispatched(
        cache,
        &spec,
        layer,
        &routes.hidden,
        &routes.global_expert_ids,
        pass,
        stream,
    )
}

pub fn execute_cached_muse_glimmer(
    args: &eredu_architectures::muse_glimmer::DecoderConfig,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    let spec = eredu_architectures::muse_glimmer::text::expert_bank_spec(args, layer)?;
    crate::backend::mlx::runtime::residency::expert_provider::execute_cached_gated_product_dispatched(
        cache,
        &spec,
        layer,
        &routes.hidden,
        &routes.global_expert_ids,
        pass,
        stream,
    )
}

pub fn execute_cached_nemotron_h(
    args: &eredu_architectures::nemotron_h::ModelArgs,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    let prefix = if layer < args.num_hidden_layers as usize {
        format!("model.layers.{layer}.moe.experts")
    } else {
        format!(
            "model.mtp.layers.{}.mixer.experts",
            layer - args.num_hidden_layers as usize
        )
    };
    crate::backend::mlx::runtime::residency::expert_provider::execute_cached_relu2_dispatched(
        cache,
        crate::backend::mlx::runtime::residency::expert_provider::CachedRelu2BankSpec {
            hidden_dimensions: args.hidden_size,
            intermediate_dimensions: args.moe_intermediate_size,
            up_quantization: args.weight_quantization_for(&format!("{prefix}.up_proj")),
            down_quantization: args.weight_quantization_for(&format!("{prefix}.down_proj")),
        },
        layer,
        &routes.hidden,
        &routes.global_expert_ids,
        pass,
        stream,
    )
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
pub fn finalize_qwen3_expert_bank(
    bank: &mut PackedGatedProductExperts,
    assignment: &ExpertAssignment,
    stream: &Stream,
) -> Result<usize, Error> {
    if !expert_bank_needs_slicing(bank.num_experts, assignment)? {
        return Ok(parameter_bytes(bank));
    }
    let mut bytes = 0;
    bytes += slice_required(&mut bank.gate_up_proj, assignment, stream)?;
    bytes += slice_optional(&mut bank.gate_up_proj_bias, assignment, stream)?;
    bytes += slice_optional(&mut bank.gate_up_proj_scales, assignment, stream)?;
    bytes += slice_optional(&mut bank.gate_up_proj_biases, assignment, stream)?;
    bytes += slice_required(&mut bank.down_proj, assignment, stream)?;
    bytes += slice_optional(&mut bank.down_proj_bias, assignment, stream)?;
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
        eredu_core::DEFAULT_MAX_MAPPED_SHARDS,
        assignment,
        stream,
        weights_stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn load_gguf_ep(
    architecture: &str,
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
                | "gpt-oss"
                | "lfm2moe"
                | "nemotron_h_moe"
        ) || !(options.weight_residency.is_fully_resident()
            || options.weight_residency.expert_cache().is_some()))
    {
        return Err(Error::Parallel(format!(
            "GGUF TP+EP preflight requires kimi-linear, deepseek2, inkling, qwen3moe, gpt-oss, lfm2moe, or nemotron_h_moe with fully resident weights or non-expert residency plus an independent expert cache, got architecture {architecture} and residency {:?}",
            options.weight_residency
        )));
    }
    crate::composition::mlx::validate_gguf_quantization_source(
        checkpoint,
        &metadata,
        options.quantization,
    )?;
    if architecture == "deepseek4" {
        let mut structural_options = options;
        structural_options.parallel = None;
        structural_options.weight_residency = WeightResidency::fully_resident();
        crate::composition::mlx::structural::validate_gguf(
            eredu_core::GgufArchitecture::DeepSeek4,
            checkpoint,
            &metadata,
            structural_options,
        )
        .into_loader_result()?;
    }
    if let Some(expert_options) = options.weight_residency.expert_cache() {
        reject_external_gguf_ep_quantization(options.quantization)?;
        return load_external_gguf_ep(
            architecture,
            checkpoint,
            &metadata,
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
        topology,
        assignment,
        LayerWeightResidency::FullyResident,
        ExternalExpertResidency::FullyResident(options.quantization),
        eredu_core::DEFAULT_MAX_MAPPED_SHARDS,
        stream,
        weights_stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn load_external_gguf_ep(
    architecture: &str,
    checkpoint: &GgufCheckpoint,
    metadata: &std::collections::HashMap<String, GgufMetadataValue>,
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
            let prepared = kimi_linear::prepare_gguf(checkpoint, metadata)?;
            let args = prepared.args;
            let assignment =
                resolve_model_assignment(assignment, args.num_experts as usize, topology)?;
            let gguf_plan = eredu_architectures::kimi_linear::gguf_plan(&args)
                .map_err(Error::UnsupportedArchitecture)?;
            let store: std::sync::Arc<dyn eredu_checkpoint::store::CheckpointSource> =
                std::sync::Arc::new(open_gguf_checkpoint_source(
                    checkpoint.clone(),
                    &gguf_plan,
                    eredu_architectures::kimi_linear::translate_gguf_weight_name,
                    max_mapped_shards,
                )?);
            let model = if topology.tensor_parallel_size > 1 {
                kimi_linear::load_external_expert_parallel_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        ParallelBuildContext::new(topology, ShardingPolicy::Require),
                        stream,
                        weights_stream,
                    )?
            } else {
                kimi_linear::load_external_expert_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        stream,
                        weights_stream,
                    )?
            };
            let store = model.checkpoint_store_arc();
            let entries = kimi_linear::expert_catalog(&args, store.as_ref())?;
            let replicated_parameter_bytes =
                planned_replicated_bytes(&model.residency_report()?)?;
            finish_external_ep(
                topology,
                ModelKind::KimiLinear,
                assignment,
                ExpertArchitecture::NeutralHybrid(Box::new(model)),
                store,
                entries,
                expert_residency,
                replicated_parameter_bytes,
                stream,
                weights_stream,
            )
        }
        "deepseek4" => {
            let (model, _) = if topology.tensor_parallel_size > 1 {
                crate::composition::deepseek::load_gguf_external_experts_parallel(
                    checkpoint,
                    metadata,
                    true,
                    non_expert,
                    ParallelBuildContext::new(topology, ShardingPolicy::Require),
                    stream,
                    weights_stream,
                )?
            } else {
                crate::composition::deepseek::load_gguf_external_experts(
                    checkpoint,
                    metadata,
                    true,
                    non_expert,
                    stream,
                    weights_stream,
                )?
            };
            let args = model
                .v4_args()
                .cloned()
                .ok_or_else(|| Error::Parallel("expected DeepSeek V4 GGUF".into()))?;
            let assignment = resolve_model_assignment(
                assignment,
                args.n_routed_experts as usize,
                topology,
            )?;
            let store = model.checkpoint_store_arc();
            let entries = deepseek_v4_expert_catalog(&args, store.as_ref(), topology)?;
            let replicated_parameter_bytes =
                planned_replicated_bytes(&model.residency_report()?)?;
            finish_external_ep(
                topology,
                ModelKind::DeepSeekV4,
                assignment,
                ExpertArchitecture::DeepSeek(Box::new(model)),
                store,
                entries,
                expert_residency,
                replicated_parameter_bytes,
                stream,
                weights_stream,
            )
        }
        "deepseek2" => {
            let (model, _) = if topology.tensor_parallel_size > 1 {
                crate::composition::deepseek::load_gguf_external_experts_parallel(
                    checkpoint,
                    metadata,
                    false,
                    non_expert,
                    ParallelBuildContext::new(topology, ShardingPolicy::Require),
                    stream,
                    weights_stream,
                )?
            } else {
                crate::composition::deepseek::load_gguf_external_experts(
                    checkpoint,
                    metadata,
                    false,
                    non_expert,
                    stream,
                    weights_stream,
                )?
            };
            let args = model
                .v3_args()
                .cloned()
                .ok_or_else(|| Error::Parallel("expected DeepSeek V3 GGUF".into()))?;
            let assignment = resolve_model_assignment(
                assignment,
                args.n_routed_experts as usize,
                topology,
            )?;
            let store = model.checkpoint_store_arc();
            let entries = deepseek_v3_expert_catalog(&args, store.as_ref(), topology)?;
            let replicated_parameter_bytes =
                planned_replicated_bytes(&model.residency_report()?)?;
            finish_external_ep(
                topology,
                ModelKind::DeepSeekV3,
                assignment,
                ExpertArchitecture::DeepSeek(Box::new(model)),
                store,
                entries,
                expert_residency,
                replicated_parameter_bytes,
                stream,
                weights_stream,
            )
        }
        "qwen3moe" => {
            let prepared = crate::composition::qwen::prepare_qwen_gguf_checkpoint(
                checkpoint,
                metadata,
            )?;
            let args = prepared.args;
            let assignment =
                resolve_model_assignment(assignment, args.num_experts as usize, topology)?;
            let gguf_plan = eredu_architectures::qwen::gguf_plan(&args)
                .map_err(Error::UnsupportedArchitecture)?;
            let store: std::sync::Arc<dyn eredu_checkpoint::store::CheckpointSource> =
                std::sync::Arc::new(open_gguf_checkpoint_source(
                    checkpoint.clone(),
                    &gguf_plan,
                    |name| eredu_architectures::qwen::translate_gguf_weight_name(name, true),
                    max_mapped_shards,
                )?);
            let build = (topology.tensor_parallel_size > 1).then(|| {
                ParallelBuildContext::new(topology, ShardingPolicy::Require)
            });
            let model = crate::composition::qwen::load_qwen_external_experts_with_store(
                store.clone(),
                args.clone(),
                non_expert,
                build,
                stream,
                weights_stream,
            )?;
            let store = model.checkpoint_store_arc();
            let entries = crate::composition::qwen::expert::expert_catalog(&args, store.as_ref())?;
            let report = model.residency_report()?.ok_or_else(|| {
                Error::Parallel("Qwen external expert model has no residency report".into())
            })?;
            let replicated_parameter_bytes = planned_replicated_bytes(&report)?;
            finish_external_ep(
                topology,
                ModelKind::Qwen3,
                assignment,
                ExpertArchitecture::Qwen(Box::new(model)),
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
                gpt_oss::prepare_gpt_oss_gguf_checkpoint(checkpoint, metadata)?;
            let args = prepared.args;
            let assignment = resolve_model_assignment(
                assignment,
                args.num_local_experts as usize,
                topology,
            )?;
            let gguf_plan =
                gpt_oss_arch::gguf_plan(&args).map_err(Error::UnsupportedArchitecture)?;
            let store: std::sync::Arc<dyn eredu_checkpoint::store::CheckpointSource> =
                std::sync::Arc::new(open_gguf_checkpoint_source(
                    checkpoint.clone(),
                    &gguf_plan,
                    gpt_oss_arch::translate_gguf_weight_name,
                    max_mapped_shards,
                )?);
            let build = if topology.tensor_parallel_size > 1 {
                Some(ParallelBuildContext::new(
                    topology,
                    ShardingPolicy::Require,
                ))
            } else {
                None
            };
            let model = gpt_oss::load_external_experts_with_store(
                store.clone(),
                args.clone(),
                non_expert,
                build,
                stream,
                weights_stream,
            )?;
            let store = model.checkpoint_store_arc();
            let entries = model.external_expert_catalog()?;
            let replicated_parameter_bytes =
                planned_replicated_bytes(&model.residency_report()?)?;
            finish_external_ep(
                topology,
                ModelKind::GptOss,
                assignment,
                ExpertArchitecture::GptOss(Box::new(model)),
                store,
                entries,
                expert_residency,
                replicated_parameter_bytes,
                stream,
                weights_stream,
            )
        }
        "inkling" => Err(Error::Parallel(
            "Inkling expert parallelism is served by the neutral pipeline runtime".into(),
        )),
        "lfm2moe" => {
            let prepared = lfm2::prepare_gguf(checkpoint, metadata)?;
            let args = prepared.args;
            let assignment =
                resolve_model_assignment(assignment, args.num_experts as usize, topology)?;
            let gguf_plan = eredu_architectures::lfm2::gguf_plan(&args)
                .map_err(Error::UnsupportedArchitecture)?;
            let store: std::sync::Arc<dyn eredu_checkpoint::store::CheckpointSource> =
                std::sync::Arc::new(open_gguf_checkpoint_source(
                    checkpoint.clone(),
                    &gguf_plan,
                    |name| eredu_architectures::lfm2::translate_gguf_weight_name(name, true),
                    max_mapped_shards,
                )?);
            let model = if topology.tensor_parallel_size > 1 {
                lfm2::load_external_expert_parallel_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    ParallelBuildContext::new(topology, ShardingPolicy::Require),
                    stream,
                    weights_stream,
                )?
            } else {
                lfm2::load_external_expert_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    stream,
                    weights_stream,
                )?
            };
            let store = model.checkpoint_store_arc();
            let entries = lfm2::expert_catalog(&args, store.as_ref())?;
            let replicated_parameter_bytes =
                planned_replicated_bytes(&model.residency_report()?)?;
            finish_external_ep(
                topology,
                ModelKind::Lfm2,
                assignment,
                ExpertArchitecture::NeutralHybrid(Box::new(model)),
                store,
                entries,
                expert_residency,
                replicated_parameter_bytes,
                stream,
                weights_stream,
            )
        }
        "nemotron_h_moe" => {
            let prepared = nemotron_h::prepare_gguf(checkpoint, metadata)?;
            let args = prepared.args;
            let assignment =
                resolve_model_assignment(assignment, args.n_routed_experts as usize, topology)?;
            let gguf_plan = eredu_architectures::nemotron_h::gguf_plan(&args)
                .map_err(Error::UnsupportedArchitecture)?;
            let store: std::sync::Arc<dyn eredu_checkpoint::store::CheckpointSource> =
                std::sync::Arc::new(open_gguf_checkpoint_source(
                    checkpoint.clone(),
                    &gguf_plan,
                    eredu_architectures::nemotron_h::translate_gguf_weight_name,
                    max_mapped_shards,
                )?);
            let model = if topology.tensor_parallel_size > 1 {
                nemotron_h::load_external_expert_parallel_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        ParallelBuildContext::new(topology, ShardingPolicy::Require),
                        stream,
                        weights_stream,
                    )?
            } else {
                nemotron_h::load_external_expert_base_with_store(
                        store.clone(),
                        args.clone(),
                        non_expert,
                        stream,
                        weights_stream,
                    )?
            };
            let store = model.checkpoint_store_arc();
            let entries = nemotron_h::expert_catalog(&args, store.as_ref())?;
            let replicated_parameter_bytes =
                planned_replicated_bytes(&model.residency_report()?)?;
            finish_external_ep(
                topology,
                ModelKind::NemotronH,
                assignment,
                ExpertArchitecture::NeutralHybrid(Box::new(model)),
                store,
                entries,
                expert_residency,
                replicated_parameter_bytes,
                stream,
                weights_stream,
            )
        }
        other => Err(Error::Parallel(format!(
            "external-expert GGUF architecture {other} is unsupported; registered architectures are kimi-linear, deepseek2, gpt-oss, inkling, qwen3moe, qwen3vlmoe, lfm2moe, nemotron_h_moe, qwen35moe, and qwen3next"
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
    let args = eredu_architectures::kimi_linear::model_args_from_config_reader(
        std::fs::File::open(model_dir.join("config.json"))?,
    )
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let assignment = resolve_model_assignment(assignment, args.num_experts as usize, topology)?;
    let store: std::sync::Arc<dyn eredu_checkpoint::store::CheckpointSource> = std::sync::Arc::new(
        SafetensorsWeightStore::open_with_max_mapped_shards(model_dir, max_mapped_shards)?,
    );
    let model = if topology.tensor_parallel_size > 1 {
        kimi_linear::load_external_expert_parallel_base_with_store(
            store.clone(),
            args.clone(),
            non_expert,
            ParallelBuildContext::new(topology, ShardingPolicy::Require),
            stream,
            weights_stream,
        )?
    } else {
        kimi_linear::load_external_expert_base_with_store(
            store.clone(),
            args.clone(),
            non_expert,
            stream,
            weights_stream,
        )?
    };
    let store = model.checkpoint_store_arc();
    let entries = kimi_linear::expert_catalog(&args, store.as_ref())?;
    let replicated_parameter_bytes = planned_replicated_bytes(&model.residency_report()?)?;
    finish_external_ep(
        topology,
        ModelKind::KimiLinear,
        assignment,
        ExpertArchitecture::NeutralHybrid(Box::new(model)),
        store,
        entries,
        expert_residency,
        replicated_parameter_bytes,
        stream,
        weights_stream,
    )
}

fn deepseek_intermediate_range(
    global: i32,
    topology: MlxParallelContext,
) -> Result<std::ops::Range<usize>, Error> {
    let global = usize::try_from(global)
        .map_err(|_| Error::Parallel("invalid DeepSeek expert intermediate width".into()))?;
    if global % topology.tensor_parallel_size != 0 {
        return Err(Error::Parallel(format!(
            "DeepSeek expert intermediate width {global} is not divisible by tensor parallel size {}",
            topology.tensor_parallel_size
        )));
    }
    let width = global / topology.tensor_parallel_size;
    let start = topology
        .tensor_parallel_rank
        .checked_mul(width)
        .ok_or_else(|| Error::Parallel("DeepSeek expert intermediate range overflowed".into()))?;
    Ok(start..start + width)
}

fn deepseek_v3_expert_catalog(
    args: &eredu_architectures::deepseek::V3Args,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    topology: MlxParallelContext,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    if topology.tensor_parallel_size > 1 {
        crate::composition::deepseek_expert::v3_parallel_catalog(
            args,
            deepseek_intermediate_range(args.moe_intermediate_size, topology)?,
            store,
        )
    } else {
        crate::composition::deepseek_expert::v3_catalog(args, store)
    }
}

fn deepseek_v4_expert_catalog(
    args: &eredu_architectures::deepseek::V4Args,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    topology: MlxParallelContext,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    if topology.tensor_parallel_size > 1 {
        crate::composition::deepseek_expert::v4_parallel_catalog(
            args,
            deepseek_intermediate_range(args.moe_intermediate_size, topology)?,
            store,
        )
    } else {
        crate::composition::deepseek_expert::v4_catalog(args, store)
    }
}

#[allow(clippy::too_many_arguments)]
fn load_deepseek_external_ep(
    model_dir: &Path,
    topology: MlxParallelContext,
    options: ModelLoadOptions,
    non_expert: LayerWeightResidency,
    expert_residency: ExternalExpertResidency,
    _max_mapped_shards: usize,
    assignment: Option<ExpertAssignment>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    let model = if topology.tensor_parallel_size > 1 {
        crate::composition::deepseek::load_safetensors_external_experts_parallel(
            model_dir,
            non_expert,
            options.quantization,
            ParallelBuildContext::new(topology, ShardingPolicy::Require),
            stream,
            weights_stream,
        )?
    } else {
        crate::composition::deepseek::load_safetensors_external_experts(
            model_dir,
            non_expert,
            options.quantization,
            stream,
            weights_stream,
        )?
    };
    let args = model
        .v3_args()
        .cloned()
        .ok_or_else(|| Error::Parallel("expected a DeepSeek V3 checkpoint".into()))?;
    let assignment =
        resolve_model_assignment(assignment, args.n_routed_experts as usize, topology)?;
    let store = model.checkpoint_store_arc();
    let entries = deepseek_v3_expert_catalog(&args, store.as_ref(), topology)?;
    let replicated_parameter_bytes = planned_replicated_bytes(&model.residency_report()?)?;
    finish_external_ep(
        topology,
        ModelKind::DeepSeekV3,
        assignment,
        ExpertArchitecture::DeepSeek(Box::new(model)),
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
    _max_mapped_shards: usize,
    assignment: Option<ExpertAssignment>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ExpertParallelModel, Error> {
    let model = if topology.tensor_parallel_size > 1 {
        crate::composition::deepseek::load_safetensors_external_experts_parallel(
            model_dir,
            non_expert,
            options.quantization,
            ParallelBuildContext::new(topology, ShardingPolicy::Require),
            stream,
            weights_stream,
        )?
    } else {
        crate::composition::deepseek::load_safetensors_external_experts(
            model_dir,
            non_expert,
            options.quantization,
            stream,
            weights_stream,
        )?
    };
    let args = model
        .v4_args()
        .cloned()
        .ok_or_else(|| Error::Parallel("expected a DeepSeek V4 checkpoint".into()))?;
    let assignment =
        resolve_model_assignment(assignment, args.n_routed_experts as usize, topology)?;
    let store = model.checkpoint_store_arc();
    let entries = deepseek_v4_expert_catalog(&args, store.as_ref(), topology)?;
    let replicated_parameter_bytes = planned_replicated_bytes(&model.residency_report()?)?;
    finish_external_ep(
        topology,
        ModelKind::DeepSeekV4,
        assignment,
        ExpertArchitecture::DeepSeek(Box::new(model)),
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
    let args = crate::composition::qwen::load_model_args(model_dir)?;
    if !args.is_moe() {
        return Err(Error::Parallel(
            "Qwen3 config is dense and has no routed experts".into(),
        ));
    }
    let assignment = resolve_model_assignment(assignment, args.num_experts as usize, topology)?;
    let store: std::sync::Arc<dyn eredu_checkpoint::store::CheckpointSource> = std::sync::Arc::new(
        SafetensorsWeightStore::open_with_max_mapped_shards(model_dir, max_mapped_shards)?,
    );
    let build = (topology.tensor_parallel_size > 1)
        .then(|| ParallelBuildContext::new(topology, ShardingPolicy::Require));
    let model = crate::composition::qwen::load_qwen_external_experts_with_store(
        store.clone(),
        args.clone(),
        non_expert,
        build,
        stream,
        weights_stream,
    )?;
    let store = model.checkpoint_store_arc();
    let entries = crate::composition::qwen::expert::expert_catalog(&args, store.as_ref())?;
    let report = model.residency_report()?.ok_or_else(|| {
        Error::Parallel("Qwen external expert model has no residency report".into())
    })?;
    let replicated_parameter_bytes = planned_replicated_bytes(&report)?;
    finish_external_ep(
        topology,
        ModelKind::Qwen3,
        assignment,
        ExpertArchitecture::Qwen(Box::new(model)),
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
            eredu_core::DEFAULT_MAX_MAPPED_SHARDS,
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
        eredu_core::DEFAULT_MAX_MAPPED_SHARDS,
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
            eredu_core::DEFAULT_MAX_MAPPED_SHARDS,
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
#[allow(dead_code)]
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
#[allow(dead_code)]
fn is_qwen_hybrid_decoder_expert_key(key: &str) -> bool {
    key.starts_with("model.layers.") && key.contains(".mlp.experts.")
}

#[cfg(test)]
#[allow(dead_code)]
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
        ModelKind::GptOss => {
            let args = gpt_oss::load_model_args(model_dir)?;
            let assignment =
                resolve_model_assignment(assignment, args.num_local_experts as usize, topology)?;
            let build = if topology.tensor_parallel_size > 1 {
                Some(ParallelBuildContext::new(topology, ShardingPolicy::Require))
            } else {
                None
            };
            let model = gpt_oss::load_external_experts_with_store(
                store.clone(),
                args.clone(),
                non_expert,
                build,
                stream,
                weights_stream,
            )?;
            let store = model.checkpoint_store_arc();
            let entries = model.external_expert_catalog()?;
            let replicated = planned_replicated_bytes(&model.residency_report()?)?;
            (
                assignment,
                ExpertArchitecture::GptOss(Box::new(model)),
                store,
                entries,
                replicated,
            )
        }
        ModelKind::Inkling => {
            return Err(Error::Parallel(
                "Inkling expert parallelism is served by the neutral pipeline runtime".into(),
            ))
        }
        ModelKind::Lfm2 => {
            let args = lfm2::load_model_args(model_dir)?;
            if !args.has_sparse_moe_layers() {
                return Err(Error::UnsupportedArchitecture(
                    "expert parallelism requires an LFM2 MoE checkpoint".into(),
                ));
            }
            let assignment =
                resolve_model_assignment(assignment, args.num_experts as usize, topology)?;
            let model = if topology.tensor_parallel_size > 1 {
                lfm2::load_external_expert_parallel_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    ParallelBuildContext::new(topology, ShardingPolicy::Require),
                    stream,
                    weights_stream,
                )?
            } else {
                lfm2::load_external_expert_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    stream,
                    weights_stream,
                )?
            };
            let store = model.checkpoint_store_arc();
            let entries = lfm2::expert_catalog(&args, store.as_ref())?;
            let replicated = planned_replicated_bytes(&model.residency_report()?)?;
            (
                assignment,
                ExpertArchitecture::NeutralHybrid(Box::new(model)),
                store,
                entries,
                replicated,
            )
        }
        ModelKind::NemotronH => {
            let args = nemotron_h::load_model_args(model_dir)?;
            if !args.has_sparse_moe_layers() {
                return Err(Error::UnsupportedArchitecture(
                    "expert parallelism requires a Nemotron-H MoE checkpoint".into(),
                ));
            }
            let assignment =
                resolve_model_assignment(assignment, args.n_routed_experts as usize, topology)?;
            let model = if topology.tensor_parallel_size > 1 {
                nemotron_h::load_external_expert_parallel_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    ParallelBuildContext::new(topology, ShardingPolicy::Require),
                    stream,
                    weights_stream,
                )?
            } else {
                nemotron_h::load_external_expert_base_with_store(
                    store.clone(),
                    args.clone(),
                    non_expert,
                    stream,
                    weights_stream,
                )?
            };
            let store = model.checkpoint_store_arc();
            let entries = nemotron_h::expert_catalog(&args, store.as_ref())?;
            let replicated = planned_replicated_bytes(&model.residency_report()?)?;
            (
                assignment,
                ExpertArchitecture::NeutralHybrid(Box::new(model)),
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
            eredu_core::DEFAULT_MAX_MAPPED_SHARDS,
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
pub fn assert_rank_owned_sparse_ep_load(
    model_dir: &Path,
    expert_options: ExpertCacheLoadOptions,
    expected_kind: ModelKind,
    expected_owned_experts: usize,
    stream: &Stream,
    weights_stream: &Stream,
) {
    use crate::backend::mlx::DeviceAssignment;
    use eredu_runtime::DenseDiskStreamLoadOptions;
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
                NonExpertWeightResidency::LayerwiseHost(Default::default()),
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
                NonExpertWeightResidency::FullyResident,
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
                NonExpertWeightResidency::DenseDiskStream(dense),
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
pub fn assert_rank_owned_quantized_sparse_ep_load(
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
                NonExpertWeightResidency::LayerwiseHost(Default::default()),
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

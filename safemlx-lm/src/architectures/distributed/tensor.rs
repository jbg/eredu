//! Executable pure tensor-parallel inference for dense, MLA, MoE, and hybrid
//! text transformer families.
//!
//! Every rank executes every decoder layer. Q/K/V and gate/up projections are
//! column sharded, their intermediate activations stay local, and output/down
//! projections are row sharded followed by exactly one all-sum. Embeddings and
//! output logits use balanced contiguous vocabulary ranges. Recurrent heads,
//! convolution channels, and their cache state are sharded whenever they admit
//! the same projection-local decomposition; only operators without that
//! decomposition remain replicated.

use std::{
    collections::HashMap,
    num::NonZeroU32,
    ops::Range,
    path::{Path, PathBuf},
};

use safemlx::{
    distributed::{self, Group},
    module::{Module, ModuleParameters},
    nn,
    ops::{indexing::TryIndexOp, ones, tanh, zeros, zeros_like},
    quantization::MaybeQuantized,
    transforms::eval,
    Array, Dtype, Stream,
};

use crate::{
    api::{
        common::linear, deepseek_v3, gemma4, gpt_oss, kimi_linear, lfm2, llama, nemotron_h,
        qwen3_5_moe as qwen_hybrid, ModelKind, ModelLoadOptions,
    },
    architectures::distributed::pipeline::{
        assign_module, load_deepseek_experts, SynchronizedToken,
    },
    architectures::qwen::dense as qwen,
    error::Error,
    nn::tensor::create_causal_mask,
    runtime::attention::LayerSchedule,
    runtime::cache::residency::{
        open_prompt_cache, validate_prompt_cache_model_identity, CacheRankIdentity,
        CacheResidencyError, CacheResidencyManager, CacheResidencyPolicy, CacheResidencyReport,
        LayerCachePolicy, PagedCacheOptions, PromptCacheDescriptor, PromptCacheManifest,
        PromptCacheModelIdentity, PromptCacheOptions, PromptCacheTopology,
    },
    runtime::cache::{
        CompressedLatentCache, ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache,
    },
    runtime::checkpoint::binding::canonical_checkpoint_name,
    runtime::checkpoint::load::{
        transform_split_relu2_experts, transform_split_swiglu_experts, StrictLoadConfig,
    },
    runtime::checkpoint::quantization::{should_quantize_on_load, WeightQuantization},
    runtime::checkpoint::store::{SafetensorsWeightStore, WeightStore},
    runtime::distributed::parallel::{
        array_parameter_member as array_member, balanced_parallel_widths as vocabulary_widths,
        exact_parallel_division as exact_division, module_parameter_group as typed_module_group,
        register_projection_module, register_projection_module as register_projection,
        register_replicated_module, register_vocabulary_module,
        require_parallel_alignment as require_alignment,
        validate_pure_tensor_topology as validate_pure_tensor, MemberSharding, ParallelPlanBuilder,
        ParameterGroupSpec, ParameterMemberSpec, ParameterRole,
        ProjectionSharding as ProjectionPlacement,
    },
    runtime::distributed::topology::{
        balanced_contiguous_range, load_safetensors_partition_on_streams, ParallelTopology,
        PlacementPlan, RankPartition,
    },
    runtime::generation::sampler::Sampler,
};

/// Immutable description of one rank's tensor-parallel model state.
#[derive(Debug, Clone)]
pub struct TensorParallelInfo {
    /// Rank in the communication group.
    pub global_rank: usize,
    /// Tensor-parallel coordinate.
    pub tensor_parallel_rank: usize,
    /// Tensor-parallel process count.
    pub tensor_parallel_size: usize,
    /// Loaded architecture.
    pub model_kind: ModelKind,
    /// Global query-head count.
    pub global_attention_heads: i32,
    /// Local query-head count.
    pub local_attention_heads: i32,
    /// Global key/value-head count. DeepSeek MLA uses its query-head count.
    pub global_kv_heads: i32,
    /// Local key/value-head count.
    pub local_kv_heads: i32,
    /// Balanced local vocabulary range.
    pub local_vocabulary_range: Range<usize>,
    /// Names of checkpoint slices materialized on this rank.
    pub owned_tensors: Vec<String>,
    /// Bytes in locally selected checkpoint tensors before runtime packing.
    pub local_parameter_bytes: usize,
    /// Checkpoint payload shards actually opened by this rank.
    pub opened_checkpoint_shards: Vec<PathBuf>,
}

/// Llama cache storage with only this rank's local K/V heads.
#[derive(Debug, Clone)]
pub enum TensorParallelLlamaLayerCache {
    /// Unbounded concatenating cache.
    Standard(ConcatKeyValueCache),
    /// Block-addressable local-head state.
    Paged(PagedKeyValueCache),
}

/// Architecture-checked rank-local tensor-parallel cache.
#[derive(Debug, Clone)]
pub enum TensorParallelCache {
    /// Llama-compatible local-head caches.
    Llama(Vec<TensorParallelLlamaLayerCache>),
    /// Dense Qwen2/Qwen3 local-head caches.
    Qwen(Vec<TensorParallelLlamaLayerCache>),
    /// Gemma text local-head caches; shared-KV layers have no local entry.
    Gemma(Vec<Option<TensorParallelLlamaLayerCache>>),
    /// DeepSeek compressed-latent caches.
    DeepSeek(Vec<CompressedLatentCache>),
    /// GPT-OSS scheduled local-head caches.
    GptOss(gpt_oss::Cache),
    /// Kimi Linear rank-local KDA/MLA caches.
    Kimi(kimi_linear::Cache),
    /// LFM2 hybrid replicated-convolution/local-head-attention caches.
    Lfm2(lfm2::Cache),
    /// Qwen3-Next/Qwen3.5 hybrid full/recurrent attention caches.
    QwenHybrid(qwen_hybrid::Cache),
    /// Nemotron-H rank-local attention/Mamba caches.
    NemotronH(nemotron_h::Cache),
}

impl TensorParallelCache {
    /// Clears all retained sequence state.
    pub fn reset(&mut self) -> Result<(), Error> {
        match self {
            Self::Llama(caches) | Self::Qwen(caches) => {
                for cache in caches {
                    match cache {
                        TensorParallelLlamaLayerCache::Standard(cache) => cache.clear(),
                        TensorParallelLlamaLayerCache::Paged(cache) => cache.clear()?,
                    }
                }
            }
            Self::Gemma(caches) => {
                for cache in caches.iter_mut().flatten() {
                    match cache {
                        TensorParallelLlamaLayerCache::Standard(cache) => cache.clear(),
                        TensorParallelLlamaLayerCache::Paged(cache) => cache.clear()?,
                    }
                }
            }
            Self::DeepSeek(caches) => {
                for cache in caches {
                    cache.clear()?;
                }
            }
            Self::GptOss(cache) => cache.reset()?,
            Self::Kimi(cache) => cache.reset()?,
            Self::Lfm2(cache) => cache.reset(),
            Self::QwenHybrid(cache) => cache.reset(),
            Self::NemotronH(cache) => cache.reset(),
        }
        Ok(())
    }

    /// Returns the common cache sequence offset.
    pub fn offset(&self) -> i32 {
        match self {
            Self::Llama(caches) | Self::Qwen(caches) => {
                caches.first().map_or(0, |cache| match cache {
                    TensorParallelLlamaLayerCache::Standard(cache) => cache.offset(),
                    TensorParallelLlamaLayerCache::Paged(cache) => cache.offset(),
                })
            }
            Self::Gemma(caches) => caches
                .iter()
                .flatten()
                .next()
                .map_or(0, |cache| match cache {
                    TensorParallelLlamaLayerCache::Standard(cache) => cache.offset(),
                    TensorParallelLlamaLayerCache::Paged(cache) => cache.offset(),
                }),
            Self::DeepSeek(caches) => caches.first().map_or(0, CompressedLatentCache::offset),
            Self::GptOss(cache) => cache.offset(),
            Self::Kimi(cache) => cache.offset(),
            Self::Lfm2(cache) => cache.offset(),
            Self::QwenHybrid(cache) => cache.offset(),
            Self::NemotronH(cache) => cache.offset(),
        }
    }
}

struct LlamaTensorModel {
    global_args: llama::ModelArgs,
    embedding: MaybeQuantized<nn::Embedding>,
    layers: Vec<llama::TransformerBlock>,
    norm: nn::RmsNorm,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
}

struct QwenTensorModel {
    global_args: qwen::DecoderConfig,
    embedding: MaybeQuantized<nn::Embedding>,
    layers: Vec<qwen::TransformerBlock>,
    norm: nn::RmsNorm,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
}

struct GemmaTensorModel {
    global_args: gemma4::ModelArgs,
    text: gemma4::Gemma4TextModel,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    per_layer_vocabulary_range: Option<Range<usize>>,
}

struct DeepSeekTensorModel {
    global_args: deepseek_v3::ModelArgs,
    embedding: MaybeQuantized<nn::Embedding>,
    layers: Vec<deepseek_v3::DecoderLayer>,
    norm: nn::RmsNorm,
    lm_head: MaybeQuantized<nn::Linear>,
}

struct GptOssTensorModel {
    global_args: gpt_oss::ModelArgs,
    model: gpt_oss::Model,
}

struct KimiTensorModel {
    global_args: kimi_linear::ModelArgs,
    model: kimi_linear::Model,
}

struct Lfm2TensorModel {
    global_args: lfm2::ModelArgs,
    model: lfm2::Model,
}

struct QwenHybridTensorModel {
    global_args: qwen_hybrid::ModelArgs,
    model: qwen_hybrid::Model,
}

struct NemotronHTensorModel {
    global_args: nemotron_h::ModelArgs,
    model: nemotron_h::Model,
}

enum TensorArchitecture {
    Llama(LlamaTensorModel),
    Qwen(QwenTensorModel),
    Gemma(Box<GemmaTensorModel>),
    DeepSeek(DeepSeekTensorModel),
    GptOss(Box<GptOssTensorModel>),
    Kimi(Box<KimiTensorModel>),
    Lfm2(Box<Lfm2TensorModel>),
    QwenHybrid(Box<QwenHybridTensorModel>),
    NemotronH(Box<NemotronHTensorModel>),
}

/// Executable rank-local pure tensor-parallel model.
pub struct TensorParallelModel {
    topology: ParallelTopology,
    info: TensorParallelInfo,
    architecture: TensorArchitecture,
}

impl std::fmt::Debug for TensorParallelModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TensorParallelModel")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

impl TensorParallelModel {
    /// Returns immutable rank-local placement and memory information.
    pub fn info(&self) -> &TensorParallelInfo {
        &self.info
    }

    /// Allocates one rank-local cache entry per decoder layer.
    pub fn new_cache(&self) -> TensorParallelCache {
        match &self.architecture {
            TensorArchitecture::Llama(model) => TensorParallelCache::Llama(
                (0..model.layers.len())
                    .enumerate()
                    .map(|(layer, _)| {
                        let policy = model
                            .global_args
                            .attention_schedule
                            .get(layer)
                            .expect("validated Llama tensor-parallel layer range");
                        TensorParallelLlamaLayerCache::Standard(match policy.window() {
                            Some(window) => ConcatKeyValueCache::new_for_sliding_attention(
                                i32::try_from(window.get())
                                    .expect("validated Llama attention window fits i32"),
                            ),
                            None => ConcatKeyValueCache::new(),
                        })
                    })
                    .collect(),
            ),
            TensorArchitecture::Qwen(model) => TensorParallelCache::Qwen(
                model
                    .global_args
                    .attention_schedule
                    .iter()
                    .map(|policy| {
                        TensorParallelLlamaLayerCache::Standard(match policy.window() {
                            Some(window) => ConcatKeyValueCache::new_for_sliding_attention(
                                i32::try_from(window.get())
                                    .expect("validated Qwen attention window fits i32"),
                            ),
                            None => ConcatKeyValueCache::new(),
                        })
                    })
                    .collect(),
            ),
            TensorArchitecture::Gemma(model) => TensorParallelCache::Gemma(
                model
                    .global_args
                    .layer_schedule
                    .iter()
                    .map(|policy| {
                        policy.key_value.owns_state().then(|| {
                            TensorParallelLlamaLayerCache::Standard(
                                match policy.attention.window() {
                                    Some(window) => ConcatKeyValueCache::new_for_sliding_attention(
                                        i32::try_from(window.get())
                                            .expect("validated Gemma attention window fits i32"),
                                    ),
                                    None => ConcatKeyValueCache::new(),
                                },
                            )
                        })
                    })
                    .collect(),
            ),
            TensorArchitecture::DeepSeek(model) => TensorParallelCache::DeepSeek(
                (0..model.layers.len())
                    .map(|_| CompressedLatentCache::new())
                    .collect(),
            ),
            TensorArchitecture::GptOss(model) => {
                TensorParallelCache::GptOss(model.model.new_cache())
            }
            TensorArchitecture::Kimi(model) => TensorParallelCache::Kimi(model.model.new_cache()),
            TensorArchitecture::Lfm2(model) => TensorParallelCache::Lfm2(model.model.new_cache()),
            TensorArchitecture::QwenHybrid(model) => {
                TensorParallelCache::QwenHybrid(model.model.new_cache())
            }
            TensorArchitecture::NemotronH(model) => {
                TensorParallelCache::NemotronH(model.model.new_cache())
            }
        }
    }

    /// Allocates rank-local cache state under an explicit cache policy.
    pub fn new_cache_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<TensorParallelCache, Error> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => {
                let manager = CacheResidencyManager::new(options)
                    .map_err(|error| Error::Parallel(error.to_string()))?;
                let rank = Some(CacheRankIdentity {
                    pipeline_rank: (self.topology.pipeline_parallel_size > 1)
                        .then_some(self.topology.pipeline_parallel_rank),
                    tensor_parallel_rank: Some(self.topology.tensor_parallel_rank),
                    expert_parallel_rank: (self.topology.expert_parallel_size > 1)
                        .then_some(self.topology.expert_parallel_rank),
                });
                match &self.architecture {
                    TensorArchitecture::Llama(model) => Ok(TensorParallelCache::Llama(
                        (0..model.layers.len())
                            .map(|layer| {
                                PagedKeyValueCache::new_with_layout(
                                    manager.clone(),
                                    layer,
                                    model
                                        .global_args
                                        .attention_schedule
                                        .get(layer)
                                        .expect("validated Llama tensor-parallel layer range")
                                        .window()
                                        .map(|window| {
                                            i32::try_from(window.get())
                                                .expect("validated Llama attention window fits i32")
                                        }),
                                    0,
                                    rank,
                                )
                                .map(TensorParallelLlamaLayerCache::Paged)
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    )),
                    TensorArchitecture::Qwen(model) => Ok(TensorParallelCache::Qwen(
                        model
                            .global_args
                            .attention_schedule
                            .iter()
                            .enumerate()
                            .map(|(layer, policy)| {
                                PagedKeyValueCache::new_with_layout(
                                    manager.clone(),
                                    layer,
                                    policy.window().map(|window| {
                                        i32::try_from(window.get())
                                            .expect("validated Qwen attention window fits i32")
                                    }),
                                    0,
                                    rank,
                                )
                                .map(TensorParallelLlamaLayerCache::Paged)
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    )),
                    TensorArchitecture::Gemma(model) => Ok(TensorParallelCache::Gemma(
                        model
                            .global_args
                            .layer_schedule
                            .iter()
                            .enumerate()
                            .map(|(layer, policy)| {
                                policy
                                    .key_value
                                    .owns_state()
                                    .then(|| {
                                        PagedKeyValueCache::new_with_layout(
                                            manager.clone(),
                                            layer,
                                            policy.attention.window().map(|window| {
                                                i32::try_from(window.get()).expect(
                                                    "validated Gemma attention window fits i32",
                                                )
                                            }),
                                            0,
                                            rank,
                                        )
                                        .map(TensorParallelLlamaLayerCache::Paged)
                                    })
                                    .transpose()
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    )),
                    TensorArchitecture::DeepSeek(model) => Ok(TensorParallelCache::DeepSeek(
                        (0..model.layers.len())
                            .map(|layer| {
                                CompressedLatentCache::new_paged(manager.clone(), layer, rank)
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    )),
                    TensorArchitecture::GptOss(model) => Ok(TensorParallelCache::GptOss(
                        model.model.new_cache_with_manager(manager, rank)?,
                    )),
                    TensorArchitecture::Kimi(_) => Err(Error::Parallel(
                        "Kimi tensor-parallel caches contain recurrent KDA state and do not support paged residency".into(),
                    )),
                    TensorArchitecture::Lfm2(_) => Err(Error::Parallel(
                        "LFM2 tensor-parallel caches contain convolution state and do not support paged residency".into(),
                    )),
                    TensorArchitecture::QwenHybrid(_) => Err(Error::Parallel(
                        "Qwen hybrid tensor-parallel caches contain recurrent state and do not support paged residency".into(),
                    )),
                    TensorArchitecture::NemotronH(_) => Err(Error::Parallel(
                        "Nemotron-H tensor-parallel caches contain Mamba recurrent state and do not support paged residency".into(),
                    )),
                }
            }
        }
    }

    /// Returns aggregate cache-residency telemetry for a paged rank-local cache.
    pub fn cache_residency_report(
        &self,
        cache: &TensorParallelCache,
    ) -> Result<Option<CacheResidencyReport>, Error> {
        if let TensorParallelCache::GptOss(cache) = cache {
            return cache
                .residency_report()
                .map_err(|error| Error::Parallel(error.to_string()));
        }
        let manager = match cache {
            TensorParallelCache::Llama(caches) | TensorParallelCache::Qwen(caches) => {
                caches.iter().find_map(|cache| match cache {
                    TensorParallelLlamaLayerCache::Paged(cache) => Some(cache.manager()),
                    TensorParallelLlamaLayerCache::Standard(_) => None,
                })
            }
            TensorParallelCache::Gemma(caches) => {
                caches.iter().flatten().find_map(|cache| match cache {
                    TensorParallelLlamaLayerCache::Paged(cache) => Some(cache.manager()),
                    TensorParallelLlamaLayerCache::Standard(_) => None,
                })
            }
            TensorParallelCache::DeepSeek(caches) => caches
                .iter()
                .find_map(CompressedLatentCache::residency_manager),
            TensorParallelCache::GptOss(_) => unreachable!(),
            TensorParallelCache::Kimi(_) => None,
            TensorParallelCache::Lfm2(_) => None,
            TensorParallelCache::QwenHybrid(_) => None,
            TensorParallelCache::NemotronH(_) => None,
        };
        manager
            .map(|manager| {
                manager
                    .report()
                    .map_err(|error| Error::Parallel(error.to_string()))
            })
            .transpose()
    }

    /// Persists this rank's completed paged prefix below a shared cache root.
    pub fn save_prompt_cache(
        &self,
        cache: &mut TensorParallelCache,
        root: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        let identity = self.prompt_cache_model_identity()?;
        validate_prompt_cache_model_identity(&descriptor, &identity)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        if let TensorParallelCache::GptOss(cache) = cache {
            return cache
                .save_prompt_cache(
                    self.prompt_cache_rank_directory(root.as_ref()),
                    descriptor,
                    prefix_token_ids,
                    options,
                )
                .map_err(|error| Error::Parallel(error.to_string()));
        }
        if let TensorParallelCache::Kimi(cache) = cache {
            return kimi_linear::Model::save_prompt_cache(
                cache,
                self.prompt_cache_rank_directory(root.as_ref()),
                descriptor,
                prefix_token_ids,
                options,
            )
            .map_err(|error| Error::Parallel(error.to_string()));
        }
        if matches!(cache, TensorParallelCache::Lfm2(_)) {
            return Err(Error::Parallel(
                "LFM2 tensor-parallel prompt persistence requires an execution stream; use the resident model cache API".into(),
            ));
        }
        if matches!(cache, TensorParallelCache::QwenHybrid(_)) {
            return Err(Error::Parallel(
                "Qwen hybrid tensor-parallel prompt persistence requires an execution stream; use the resident model cache API".into(),
            ));
        }
        if matches!(cache, TensorParallelCache::NemotronH(_)) {
            return Err(Error::Parallel(
                "Nemotron-H tensor-parallel prompt persistence requires an execution stream; use the resident model cache API".into(),
            ));
        }
        let manager = match cache {
            TensorParallelCache::Llama(caches) | TensorParallelCache::Qwen(caches) => {
                let mut manager = None;
                for cache in caches {
                    let TensorParallelLlamaLayerCache::Paged(cache) = cache else {
                        return Err(Error::Parallel(
                            "tensor-parallel prompt persistence requires a paged cache".into(),
                        ));
                    };
                    cache.finalize()?;
                    manager.get_or_insert_with(|| cache.manager().clone());
                }
                manager
            }
            TensorParallelCache::Gemma(caches) => {
                let mut manager = None;
                for cache in caches.iter_mut().flatten() {
                    let TensorParallelLlamaLayerCache::Paged(cache) = cache else {
                        return Err(Error::Parallel(
                            "tensor-parallel prompt persistence requires a paged cache".into(),
                        ));
                    };
                    cache.finalize()?;
                    manager.get_or_insert_with(|| cache.manager().clone());
                }
                manager
            }
            TensorParallelCache::DeepSeek(caches) => {
                let mut manager = None;
                for cache in caches {
                    cache.finalize()?;
                    if let Some(paged) = cache.residency_manager() {
                        manager.get_or_insert_with(|| paged.clone());
                    } else {
                        return Err(Error::Parallel(
                            "tensor-parallel prompt persistence requires a paged cache".into(),
                        ));
                    }
                }
                manager
            }
            TensorParallelCache::GptOss(_) => unreachable!(),
            TensorParallelCache::Kimi(_) => unreachable!(),
            TensorParallelCache::Lfm2(_) => unreachable!(),
            TensorParallelCache::QwenHybrid(_) => unreachable!(),
            TensorParallelCache::NemotronH(_) => unreachable!(),
        }
        .ok_or_else(|| Error::Parallel("cannot persist an empty rank-local cache".into()))?;
        manager
            .save_prompt_cache(
                self.prompt_cache_rank_directory(root.as_ref()),
                descriptor,
                prefix_token_ids,
                &[],
                options,
            )
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    /// Opens this rank's compatible persisted prefix without eager array loading.
    pub fn load_prompt_cache(
        &self,
        root: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
    ) -> Result<(TensorParallelCache, PromptCacheManifest), Error> {
        let identity = self.prompt_cache_model_identity()?;
        validate_prompt_cache_model_identity(expected, &identity)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let (manager, manifest) = open_prompt_cache(
            self.prompt_cache_rank_directory(root.as_ref()),
            expected,
            &identity,
            prefix_token_ids,
            options,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let rank = Some(CacheRankIdentity {
            pipeline_rank: None,
            tensor_parallel_rank: Some(self.topology.tensor_parallel_rank),
            expert_parallel_rank: None,
        });
        let cache = match &self.architecture {
            TensorArchitecture::Llama(model) => TensorParallelCache::Llama(
                (0..model.layers.len())
                    .map(|layer| {
                        PagedKeyValueCache::new_with_layout(
                            manager.clone(),
                            layer,
                            model
                                .global_args
                                .attention_schedule
                                .get(layer)
                                .expect("validated Llama tensor-parallel layer range")
                                .window()
                                .map(|window| {
                                    i32::try_from(window.get())
                                        .expect("validated Llama attention window fits i32")
                                }),
                            0,
                            rank,
                        )
                        .map(TensorParallelLlamaLayerCache::Paged)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            TensorArchitecture::Gemma(model) => TensorParallelCache::Gemma(
                model
                    .global_args
                    .layer_schedule
                    .iter()
                    .enumerate()
                    .map(|(layer, policy)| {
                        policy
                            .key_value
                            .owns_state()
                            .then(|| {
                                PagedKeyValueCache::new_with_layout(
                                    manager.clone(),
                                    layer,
                                    policy.attention.window().map(|window| {
                                        i32::try_from(window.get())
                                            .expect("validated Gemma attention window fits i32")
                                    }),
                                    0,
                                    rank,
                                )
                                .map(TensorParallelLlamaLayerCache::Paged)
                            })
                            .transpose()
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            TensorArchitecture::Qwen(model) => TensorParallelCache::Qwen(
                model
                    .global_args
                    .attention_schedule
                    .iter()
                    .enumerate()
                    .map(|(layer, policy)| {
                        PagedKeyValueCache::new_with_layout(
                            manager.clone(),
                            layer,
                            policy.window().map(|window| {
                                i32::try_from(window.get())
                                    .expect("validated Qwen attention window fits i32")
                            }),
                            0,
                            rank,
                        )
                        .map(TensorParallelLlamaLayerCache::Paged)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            TensorArchitecture::DeepSeek(model) => TensorParallelCache::DeepSeek(
                (0..model.layers.len())
                    .map(|layer| CompressedLatentCache::new_paged(manager.clone(), layer, rank))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            TensorArchitecture::GptOss(model) => {
                TensorParallelCache::GptOss(model.model.new_cache_with_manager(manager, rank)?)
            }
            TensorArchitecture::Kimi(_) => {
                return Err(Error::Parallel(
                    "Kimi tensor-parallel prompt-cache loading uses resident snapshots and is unavailable through paged cache options".into(),
                ));
            }
            TensorArchitecture::Lfm2(_) => {
                return Err(Error::Parallel(
                    "LFM2 tensor-parallel prompt-cache loading uses resident snapshots and is unavailable through paged cache options".into(),
                ));
            }
            TensorArchitecture::QwenHybrid(_) => {
                return Err(Error::Parallel(
                    "Qwen hybrid tensor-parallel prompt-cache loading uses resident snapshots and is unavailable through paged cache options".into(),
                ));
            }
            TensorArchitecture::NemotronH(_) => {
                return Err(Error::Parallel(
                    "Nemotron-H tensor-parallel prompt-cache loading uses resident snapshots and is unavailable through paged cache options".into(),
                ));
            }
        };
        Ok((cache, manifest))
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

    fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        let (model_family, effective_model_type, architecture_fingerprint, layer_count) =
            match &self.architecture {
                TensorArchitecture::Llama(model) => (
                    "llama".to_string(),
                    model.global_args.model_type.clone(),
                    crate::architectures::llama::model::prompt_cache_architecture_fingerprint(
                        &model.global_args,
                    ),
                    usize::try_from(model.global_args.num_hidden_layers)
                        .map_err(|_| Error::Parallel("invalid Llama layer count".into()))?,
                ),
                TensorArchitecture::Qwen(model) => (
                    "dense_qwen".to_string(),
                    model.global_args.model_type.clone(),
                    qwen::prompt_cache_architecture_fingerprint(&model.global_args),
                    usize::try_from(model.global_args.num_hidden_layers)
                        .map_err(|_| Error::Parallel("invalid Qwen layer count".into()))?,
                ),
                TensorArchitecture::Gemma(model) => (
                    "gemma4".to_string(),
                    model.global_args.model_type.clone(),
                    gemma4::prompt_cache_architecture_fingerprint(&model.global_args),
                    usize::try_from(model.global_args.num_hidden_layers)
                        .map_err(|_| Error::Parallel("invalid Gemma layer count".into()))?,
                ),
                TensorArchitecture::DeepSeek(model) => (
                    "deepseek_v3".to_string(),
                    model.global_args.model_type.clone(),
                    crate::architectures::deepseek_v3::model::prompt_cache_architecture_fingerprint(
                        &model.global_args,
                    ),
                    usize::try_from(model.global_args.num_hidden_layers)
                        .map_err(|_| Error::Parallel("invalid DeepSeek layer count".into()))?,
                ),
                TensorArchitecture::GptOss(model) => (
                    "gpt_oss".to_string(),
                    model.global_args.model_type.clone(),
                    gpt_oss::prompt_cache_architecture_fingerprint(&model.global_args),
                    usize::try_from(model.global_args.num_hidden_layers)
                        .map_err(|_| Error::Parallel("invalid GPT-OSS layer count".into()))?,
                ),
                TensorArchitecture::Kimi(model) => (
                    "kimi_linear".to_string(),
                    model.global_args.model_type.clone(),
                    kimi_linear::prompt_cache_architecture_fingerprint(&model.global_args),
                    usize::try_from(model.global_args.num_hidden_layers)
                        .map_err(|_| Error::Parallel("invalid Kimi layer count".into()))?,
                ),
                TensorArchitecture::Lfm2(model) => (
                    "lfm2".to_string(),
                    model.global_args.model_type.clone(),
                    lfm2::prompt_cache_architecture_fingerprint(&model.global_args),
                    usize::try_from(model.global_args.num_hidden_layers)
                        .map_err(|_| Error::Parallel("invalid LFM2 layer count".into()))?,
                ),
                TensorArchitecture::QwenHybrid(model) => (
                    "qwen_hybrid".to_string(),
                    model.global_args.model_type.clone(),
                    qwen_hybrid::prompt_cache_architecture_fingerprint(&model.global_args),
                    usize::try_from(model.global_args.num_hidden_layers)
                        .map_err(|_| Error::Parallel("invalid Qwen hybrid layer count".into()))?,
                ),
                TensorArchitecture::NemotronH(model) => (
                    "nemotron_h".to_string(),
                    model.global_args.model_type.clone(),
                    nemotron_h::prompt_cache_architecture_fingerprint(&model.global_args),
                    usize::try_from(model.global_args.num_hidden_layers)
                        .map_err(|_| Error::Parallel("invalid Nemotron-H layer count".into()))?,
                ),
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
                tensor_parallel: Some((
                    self.topology.tensor_parallel_size,
                    self.topology.tensor_parallel_rank,
                )),
                expert_parallel: None,
                expert_parallel_cache_replicated: true,
            },
            layer_layout: match &self.architecture {
                TensorArchitecture::Llama(model) => {
                    let local_kv_heads = exact_division(
                        "Llama prompt-cache KV heads",
                        model.global_args.num_key_value_heads,
                        self.topology.tensor_parallel_size,
                    )?;
                    PromptCacheModelIdentity::key_value_layouts(
                        model.global_args.attention_schedule.iter().map(|policy| {
                            policy.window().map(|window| {
                                i32::try_from(window.get())
                                    .expect("validated Llama attention window fits i32")
                            })
                        }),
                        local_kv_heads,
                        model.global_args.head_dim,
                    )
                }
                TensorArchitecture::Qwen(model) => {
                    let local_kv_heads = exact_division(
                        "Qwen prompt-cache KV heads",
                        model.global_args.num_key_value_heads,
                        self.topology.tensor_parallel_size,
                    )?;
                    PromptCacheModelIdentity::key_value_layouts(
                        model.global_args.attention_schedule.iter().map(|policy| {
                            policy.window().map(|window| {
                                i32::try_from(window.get())
                                    .expect("validated Qwen attention window fits i32")
                            })
                        }),
                        local_kv_heads,
                        model.global_args.head_dim,
                    )
                }
                TensorArchitecture::Gemma(model) => (|| {
                    let policies = model
                        .global_args
                        .layer_schedule
                        .iter()
                        .map(|policy| {
                            if !policy.key_value.owns_state() {
                                return Ok(LayerCachePolicy::NoState);
                            }
                            let local_kv_heads = exact_division(
                                "Gemma prompt-cache KV heads",
                                i32::try_from(policy.num_key_value_heads.get()).map_err(|_| {
                                    CacheResidencyError::InvalidOptions(
                                        "Gemma KV heads exceed i32".into(),
                                    )
                                })?,
                                self.topology.tensor_parallel_size,
                            )
                            .map_err(|error| {
                                CacheResidencyError::InvalidOptions(error.to_string())
                            })?;
                            LayerCachePolicy::key_value(
                                policy.attention,
                                local_kv_heads,
                                i32::try_from(policy.head_dim.get()).map_err(|_| {
                                    CacheResidencyError::InvalidOptions(
                                        "Gemma head dimension exceeds i32".into(),
                                    )
                                })?,
                            )
                        })
                        .collect::<Result<Vec<_>, CacheResidencyError>>()?;
                    LayerSchedule::new(policies.len(), policies)
                        .map_err(|error| CacheResidencyError::InvalidOptions(error.to_string()))
                })(),
                TensorArchitecture::DeepSeek(model) => {
                    PromptCacheModelIdentity::compressed_layouts(
                        layer_count,
                        model.global_args.kv_lora_rank,
                        model.global_args.qk_rope_head_dim,
                    )
                }
                TensorArchitecture::GptOss(model) => {
                    let local_kv_heads = exact_division(
                        "GPT-OSS prompt-cache KV heads",
                        model.global_args.num_key_value_heads,
                        self.topology.tensor_parallel_size,
                    )?;
                    PromptCacheModelIdentity::key_value_layouts(
                        model.global_args.attention_schedule.iter().map(|policy| {
                            policy.window().map(|window| {
                                i32::try_from(window.get())
                                    .expect("validated GPT-OSS attention window fits i32")
                            })
                        }),
                        local_kv_heads,
                        model.global_args.head_dim,
                    )
                }
                TensorArchitecture::Kimi(model) => {
                    let mut local = model.global_args.clone();
                    local.num_attention_heads = exact_division(
                        "Kimi prompt-cache MLA heads",
                        local.num_attention_heads,
                        self.topology.tensor_parallel_size,
                    )?;
                    local.num_key_value_heads = exact_division(
                        "Kimi prompt-cache MLA KV heads",
                        local.num_key_value_heads,
                        self.topology.tensor_parallel_size,
                    )?;
                    local.kda_config.num_heads = exact_division(
                        "Kimi prompt-cache KDA heads",
                        local.kda_config.num_heads,
                        self.topology.tensor_parallel_size,
                    )?;
                    kimi_linear::prompt_cache_layer_layout(&local)
                        .map_err(|error| CacheResidencyError::InvalidOptions(error.to_string()))
                }
                TensorArchitecture::Lfm2(model) => {
                    let mut local = model.global_args.clone();
                    local.num_attention_heads = exact_division(
                        "LFM2 prompt-cache attention heads",
                        local.num_attention_heads,
                        self.topology.tensor_parallel_size,
                    )?;
                    local.num_key_value_heads = exact_division(
                        "LFM2 prompt-cache KV heads",
                        local.num_key_value_heads,
                        self.topology.tensor_parallel_size,
                    )?;
                    lfm2::prompt_cache_layer_layout(&local)
                        .map_err(|error| CacheResidencyError::InvalidOptions(error.to_string()))
                }
                TensorArchitecture::QwenHybrid(model) => {
                    let mut local = model.global_args.clone();
                    local.num_attention_heads = exact_division(
                        "Qwen hybrid prompt-cache attention heads",
                        local.num_attention_heads,
                        self.topology.tensor_parallel_size,
                    )?;
                    local.num_key_value_heads = exact_division(
                        "Qwen hybrid prompt-cache KV heads",
                        local.num_key_value_heads,
                        self.topology.tensor_parallel_size,
                    )?;
                    local.linear_num_key_heads = exact_division(
                        "Qwen hybrid prompt-cache recurrent key heads",
                        local.linear_num_key_heads,
                        self.topology.tensor_parallel_size,
                    )?;
                    local.linear_num_value_heads = exact_division(
                        "Qwen hybrid prompt-cache recurrent value heads",
                        local.linear_num_value_heads,
                        self.topology.tensor_parallel_size,
                    )?;
                    qwen_hybrid::prompt_cache_layer_layout(&local)
                        .map_err(|error| CacheResidencyError::InvalidOptions(error.to_string()))
                }
                TensorArchitecture::NemotronH(model) => {
                    let mut local = model.global_args.clone();
                    local.num_attention_heads = exact_division(
                        "Nemotron-H prompt-cache attention heads",
                        local.num_attention_heads,
                        self.topology.tensor_parallel_size,
                    )?;
                    local.num_key_value_heads = exact_division(
                        "Nemotron-H prompt-cache KV heads",
                        local.num_key_value_heads,
                        self.topology.tensor_parallel_size,
                    )?;
                    local.mamba_num_heads = exact_division(
                        "Nemotron-H prompt-cache Mamba heads",
                        local.mamba_num_heads,
                        self.topology.tensor_parallel_size,
                    )?;
                    local.n_groups = exact_division(
                        "Nemotron-H prompt-cache Mamba groups",
                        local.n_groups,
                        self.topology.tensor_parallel_size,
                    )?;
                    nemotron_h::prompt_cache_layer_layout(&local)
                        .map_err(|error| CacheResidencyError::InvalidOptions(error.to_string()))
                }
            }
            .map_err(|error| Error::Parallel(error.to_string()))?,
        })
    }

    /// Runs prefill or decode and returns this rank's vocabulary-logit shard.
    pub fn forward_local_logits(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut TensorParallelCache,
        group: &Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.validate_group(group)?;
        self.topology.validate_execution_stream(stream)?;
        if tokens.ndim() != 2 {
            return Err(Error::Parallel(format!(
                "tensor-parallel token input must have rank 2 [batch, sequence], got {:?}",
                tokens.shape()
            )));
        }
        match (&mut self.architecture, cache) {
            (TensorArchitecture::Llama(model), TensorParallelCache::Llama(caches)) => {
                forward_llama(model, tokens, mask, caches, &self.info, group, stream)
            }
            (TensorArchitecture::Qwen(model), TensorParallelCache::Qwen(caches)) => {
                forward_qwen(model, tokens, mask, caches, &self.info, group, stream)
            }
            (TensorArchitecture::Gemma(model), TensorParallelCache::Gemma(caches)) => {
                forward_gemma(model, tokens, mask, caches, &self.info, group, stream)
            }
            (TensorArchitecture::DeepSeek(model), TensorParallelCache::DeepSeek(caches)) => {
                forward_deepseek(model, tokens, mask, caches, &self.info, group, stream)
            }
            (TensorArchitecture::GptOss(model), TensorParallelCache::GptOss(cache)) => {
                forward_gpt_oss(model, tokens, mask, cache, &self.info, group, stream)
            }
            (TensorArchitecture::Kimi(model), TensorParallelCache::Kimi(cache)) => {
                forward_kimi(model, tokens, mask, cache, &self.info, group, stream)
            }
            (TensorArchitecture::Lfm2(model), TensorParallelCache::Lfm2(cache)) => {
                forward_lfm2(model, tokens, mask, cache, &self.info, group, stream)
            }
            (TensorArchitecture::QwenHybrid(model), TensorParallelCache::QwenHybrid(cache)) => {
                forward_qwen_hybrid(model, tokens, mask, cache, &self.info, group, stream)
            }
            (TensorArchitecture::NemotronH(model), TensorParallelCache::NemotronH(cache)) => {
                forward_nemotron_h(model, tokens, mask, cache, &self.info, group, stream)
            }
            (TensorArchitecture::Llama(_), _) => Err(Error::Parallel(
                "tensor-parallel cache does not match the Llama model".into(),
            )),
            (TensorArchitecture::Qwen(_), _) => Err(Error::Parallel(
                "tensor-parallel cache does not match the Qwen model".into(),
            )),
            (TensorArchitecture::Gemma(_), _) => Err(Error::Parallel(
                "tensor-parallel cache does not match the Gemma model".into(),
            )),
            (TensorArchitecture::DeepSeek(_), _) => Err(Error::Parallel(
                "tensor-parallel cache does not match the DeepSeek model".into(),
            )),
            (TensorArchitecture::GptOss(_), _) => Err(Error::Parallel(
                "tensor-parallel cache does not match the GPT-OSS model".into(),
            )),
            (TensorArchitecture::Kimi(_), _) => Err(Error::Parallel(
                "tensor-parallel cache does not match the Kimi model".into(),
            )),
            (TensorArchitecture::Lfm2(_), _) => Err(Error::Parallel(
                "tensor-parallel cache does not match the LFM2 model".into(),
            )),
            (TensorArchitecture::QwenHybrid(_), _) => Err(Error::Parallel(
                "tensor-parallel cache does not match the Qwen hybrid model".into(),
            )),
            (TensorArchitecture::NemotronH(_), _) => Err(Error::Parallel(
                "tensor-parallel cache does not match the Nemotron-H model".into(),
            )),
        }
    }

    /// Runs prefill or decode and gathers complete vocabulary logits on every rank.
    pub fn forward(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut TensorParallelCache,
        group: &Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let local = self.forward_local_logits(tokens, mask, cache, group, stream)?;
        let widths = vocabulary_widths(
            self.global_vocabulary_size(),
            self.topology.tensor_parallel_size,
        )?;
        Ok(distributed::all_gather_uneven_axis(
            &local, -1, &widths, group, stream,
        )?)
    }

    /// Alias for a prompt forward pass returning complete logits.
    pub fn prefill(
        &mut self,
        tokens: &Array,
        cache: &mut TensorParallelCache,
        group: &Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward(tokens, None, cache, group, stream)
    }

    /// Alias for one or more autoregressive decode tokens returning complete logits.
    pub fn decode(
        &mut self,
        tokens: &Array,
        cache: &mut TensorParallelCache,
        group: &Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward(tokens, None, cache, group, stream)
    }

    /// Samples only on `sampling_rank` and synchronizes the token and stop flag.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_and_synchronize<S: Sampler>(
        &self,
        complete_logits: &Array,
        sampler: &mut S,
        temperature: f32,
        prng_state: Option<&mut safemlx::random::RandomState>,
        finished: bool,
        sampling_rank: usize,
        group: &Group,
        stream: &Stream,
    ) -> Result<SynchronizedToken, Error> {
        self.validate_group(group)?;
        if sampling_rank >= group.size() {
            return Err(Error::Parallel(format!(
                "sampling rank {sampling_rank} is outside TP size {}",
                group.size()
            )));
        }
        let batch = complete_logits.dim(0);
        let local_token = if group.rank() == sampling_rank {
            let logits = if complete_logits.ndim() == 3 {
                complete_logits.try_index_device((.., -1, ..), stream)?
            } else {
                complete_logits.clone()
            };
            sampler
                .sample(&logits, temperature, prng_state, stream)?
                .reshape(&[batch, 1], stream)?
        } else {
            zeros::<u32>(&[batch, 1], stream)?
        };
        let token = distributed::all_sum(&local_token, group, stream)?;
        let local_finished = if group.rank() == sampling_rank && finished {
            ones::<i32>(&[], stream)?
        } else {
            zeros::<i32>(&[], stream)?
        };
        let finished = distributed::all_sum(&local_finished, group, stream)?;
        eval([&token, &finished])?;
        stream.synchronize()?;
        Ok(SynchronizedToken {
            token,
            finished: finished.try_item::<i32>(stream)? != 0,
        })
    }

    fn global_vocabulary_size(&self) -> usize {
        match &self.architecture {
            TensorArchitecture::Llama(model) => model.global_args.vocab_size as usize,
            TensorArchitecture::Qwen(model) => model.global_args.vocab_size as usize,
            TensorArchitecture::Gemma(model) => model.global_args.vocab_size as usize,
            TensorArchitecture::DeepSeek(model) => model.global_args.vocab_size as usize,
            TensorArchitecture::GptOss(model) => model.global_args.vocab_size as usize,
            TensorArchitecture::Kimi(model) => model.global_args.vocab_size as usize,
            TensorArchitecture::Lfm2(model) => model.global_args.vocab_size as usize,
            TensorArchitecture::QwenHybrid(model) => model.global_args.vocab_size as usize,
            TensorArchitecture::NemotronH(model) => model.global_args.vocab_size as usize,
        }
    }

    fn validate_group(&self, group: &Group) -> Result<(), Error> {
        if group.rank() != self.topology.global_rank || group.size() != self.topology.world_size {
            return Err(Error::Parallel(format!(
                "tensor-parallel topology expects group rank {}/{} but received {}/{}",
                self.topology.global_rank,
                self.topology.world_size,
                group.rank(),
                group.size()
            )));
        }
        Ok(())
    }
}

fn insert_llama_layer_plan(
    planner: &mut ParallelPlanBuilder,
    layer: &llama::TransformerBlock,
    index: usize,
) -> Result<(), Error> {
    let prefix = format!("model.layers.{index}");
    insert_llama_layer_plan_with_prefix(planner, layer, &prefix)
}

pub(crate) fn insert_llama_layer_plan_with_prefix(
    planner: &mut ParallelPlanBuilder,
    layer: &llama::TransformerBlock,
    prefix: &str,
) -> Result<(), Error> {
    let attention = &layer.self_attn;
    register_projection(
        planner,
        &attention.q_proj,
        &format!("{prefix}.self_attn.q_proj"),
        ProjectionPlacement::Column,
    )?;
    register_projection(
        planner,
        &attention.k_proj,
        &format!("{prefix}.self_attn.k_proj"),
        ProjectionPlacement::Column,
    )?;
    register_projection(
        planner,
        &attention.v_proj,
        &format!("{prefix}.self_attn.v_proj"),
        ProjectionPlacement::Column,
    )?;
    register_projection(
        planner,
        &attention.o_proj,
        &format!("{prefix}.self_attn.o_proj"),
        ProjectionPlacement::Row,
    )?;
    register_replicated_module(
        planner,
        &attention.rope,
        &format!("{prefix}.self_attn.rope"),
    )?;
    register_projection(
        planner,
        &layer.mlp.gate_proj,
        &format!("{prefix}.mlp.gate_proj"),
        ProjectionPlacement::Column,
    )?;
    register_projection(
        planner,
        &layer.mlp.up_proj,
        &format!("{prefix}.mlp.up_proj"),
        ProjectionPlacement::Column,
    )?;
    register_projection(
        planner,
        &layer.mlp.down_proj,
        &format!("{prefix}.mlp.down_proj"),
        ProjectionPlacement::Row,
    )?;
    register_replicated_module(
        planner,
        &layer.input_layernorm,
        &format!("{prefix}.input_layernorm"),
    )?;
    register_replicated_module(
        planner,
        &layer.post_attention_layernorm,
        &format!("{prefix}.post_attention_layernorm"),
    )?;
    Ok(())
}

fn insert_qwen_layer_plan(
    planner: &mut ParallelPlanBuilder,
    layer: &qwen::TransformerBlock,
    index: usize,
) -> Result<(), Error> {
    let prefix = format!("model.layers.{index}");
    insert_qwen_layer_plan_with_prefix(planner, layer, &prefix)
}

/// Registers a Qwen block under an arbitrary checkpoint prefix so composite
/// models can reuse the canonical dense/MoE placement declaration.
pub(crate) fn insert_qwen_layer_plan_with_prefix(
    planner: &mut ParallelPlanBuilder,
    layer: &qwen::TransformerBlock,
    prefix: &str,
) -> Result<(), Error> {
    let attention = &layer.self_attn;
    for (name, projection) in [
        ("q_proj", &attention.q_proj),
        ("k_proj", &attention.k_proj),
        ("v_proj", &attention.v_proj),
    ] {
        register_projection(
            planner,
            projection,
            &format!("{prefix}.self_attn.{name}"),
            ProjectionPlacement::Column,
        )?;
    }
    register_projection(
        planner,
        &attention.o_proj,
        &format!("{prefix}.self_attn.o_proj"),
        ProjectionPlacement::Row,
    )?;
    if let Some(norm) = &attention.q_norm {
        register_replicated_module(planner, norm, &format!("{prefix}.self_attn.q_norm"))?;
    }
    if let Some(norm) = &attention.k_norm {
        register_replicated_module(planner, norm, &format!("{prefix}.self_attn.k_norm"))?;
    }
    register_replicated_module(
        planner,
        &attention.rope,
        &format!("{prefix}.self_attn.rope"),
    )?;
    match &layer.mlp {
        qwen::FeedForward::Dense(mlp) => {
            for (name, projection, placement) in [
                ("gate_proj", &mlp.gate_proj, ProjectionPlacement::Column),
                ("up_proj", &mlp.up_proj, ProjectionPlacement::Column),
                ("down_proj", &mlp.down_proj, ProjectionPlacement::Row),
            ] {
                register_projection(
                    planner,
                    projection,
                    &format!("{prefix}.mlp.{name}"),
                    placement,
                )?;
            }
        }
        qwen::FeedForward::Moe(moe) => {
            register_replicated_module(planner, &moe.gate, &format!("{prefix}.mlp.gate"))?;
            let experts = &moe.experts;
            let intermediate = usize::try_from(experts.intermediate_dim)
                .map_err(|_| Error::Parallel("Qwen expert width exceeds usize".into()))?;
            let segments = vec![0..intermediate, intermediate..2 * intermediate];
            let mut gate_up = vec![array_member(
                format!("{prefix}.mlp.experts.gate_up_proj"),
                experts.gate_up_proj.as_ref(),
                MemberSharding::Segmented {
                    axis: 1,
                    segments: segments.clone(),
                },
            )?];
            for (name, value) in [
                (
                    "gate_up_proj_scales",
                    experts.gate_up_proj_scales.as_ref().as_ref(),
                ),
                (
                    "gate_up_proj_biases",
                    experts.gate_up_proj_biases.as_ref().as_ref(),
                ),
            ] {
                if let Some(value) = value {
                    gate_up.push(array_member(
                        format!("{prefix}.mlp.experts.{name}"),
                        value,
                        MemberSharding::Segmented {
                            axis: 1,
                            segments: segments.clone(),
                        },
                    )?);
                }
            }
            planner.register(ParameterGroupSpec::new(
                format!("{prefix}.mlp.experts.gate_up"),
                ParameterRole::ExpertIntermediate,
                gate_up,
            )?)?;
            let mut down = vec![array_member(
                format!("{prefix}.mlp.experts.down_proj"),
                experts.down_proj.as_ref(),
                MemberSharding::Equal { axis: 2 },
            )?];
            for (name, value) in [
                (
                    "down_proj_scales",
                    experts.down_proj_scales.as_ref().as_ref(),
                ),
                (
                    "down_proj_biases",
                    experts.down_proj_biases.as_ref().as_ref(),
                ),
            ] {
                if let Some(value) = value {
                    down.push(array_member(
                        format!("{prefix}.mlp.experts.{name}"),
                        value,
                        MemberSharding::Equal { axis: 2 },
                    )?);
                }
            }
            planner.register(ParameterGroupSpec::new(
                format!("{prefix}.mlp.experts.down"),
                ParameterRole::ExpertIntermediate,
                down,
            )?)?;
        }
    }
    register_replicated_module(
        planner,
        &layer.input_layernorm,
        &format!("{prefix}.input_layernorm"),
    )?;
    register_replicated_module(
        planner,
        &layer.post_attention_layernorm,
        &format!("{prefix}.post_attention_layernorm"),
    )?;
    Ok(())
}

pub(crate) fn insert_gpt_oss_layer_plan(
    planner: &mut ParallelPlanBuilder,
    layer: &gpt_oss::TransformerBlock,
    index: usize,
) -> Result<(), Error> {
    let prefix = format!("model.layers.{index}");
    let attention = &layer.self_attn;
    for (name, projection) in [
        ("q_proj", &attention.q_proj),
        ("k_proj", &attention.k_proj),
        ("v_proj", &attention.v_proj),
    ] {
        register_projection(
            planner,
            projection,
            &format!("{prefix}.self_attn.{name}"),
            ProjectionPlacement::Column,
        )?;
    }
    register_projection(
        planner,
        &attention.o_proj,
        &format!("{prefix}.self_attn.o_proj"),
        ProjectionPlacement::Row,
    )?;
    planner.register(ParameterGroupSpec::new(
        format!("{prefix}.self_attn.sinks"),
        ParameterRole::AttentionHeads,
        [array_member(
            format!("{prefix}.self_attn.sinks"),
            attention.sinks.as_ref(),
            MemberSharding::Equal { axis: 0 },
        )?],
    )?)?;
    register_replicated_module(
        planner,
        &layer.input_layernorm,
        &format!("{prefix}.input_layernorm"),
    )?;
    register_replicated_module(
        planner,
        &layer.post_attention_layernorm,
        &format!("{prefix}.post_attention_layernorm"),
    )?;
    register_projection(
        planner,
        &layer.mlp.router,
        &format!("{prefix}.mlp.router"),
        ProjectionPlacement::Replicated,
    )?;
    let experts = &layer.mlp.experts;
    planner.register(ParameterGroupSpec::new(
        format!("{prefix}.mlp.experts.gate_up"),
        ParameterRole::ExpertIntermediate,
        [
            array_member(
                format!("{prefix}.mlp.experts.gate_up_proj_blocks"),
                experts.gate_up_proj_blocks.as_ref(),
                MemberSharding::Equal { axis: 1 },
            )?,
            array_member(
                format!("{prefix}.mlp.experts.gate_up_proj_scales"),
                experts.gate_up_proj_scales.as_ref(),
                MemberSharding::Equal { axis: 1 },
            )?,
            array_member(
                format!("{prefix}.mlp.experts.gate_up_proj_bias"),
                experts.gate_up_proj_bias.as_ref(),
                MemberSharding::Equal { axis: 1 },
            )?,
        ],
    )?)?;
    planner.register(ParameterGroupSpec::new(
        format!("{prefix}.mlp.experts.down"),
        ParameterRole::ExpertIntermediate,
        [
            array_member(
                format!("{prefix}.mlp.experts.down_proj_blocks"),
                experts.down_proj_blocks.as_ref(),
                MemberSharding::Equal { axis: 2 },
            )?,
            array_member(
                format!("{prefix}.mlp.experts.down_proj_scales"),
                experts.down_proj_scales.as_ref(),
                MemberSharding::Equal { axis: 2 },
            )?,
            array_member(
                format!("{prefix}.mlp.experts.down_proj_bias"),
                experts.down_proj_bias.as_ref(),
                MemberSharding::Replicated,
            )?,
        ],
    )?)?;
    Ok(())
}

fn register_gemma_expert_projection(
    planner: &mut ParallelPlanBuilder,
    projection: &gemma4::ExpertProjection,
    prefix: &str,
    axis: usize,
) -> Result<(), Error> {
    let mut members = vec![array_member(
        format!("{prefix}.weight"),
        projection.weight.as_ref(),
        MemberSharding::Equal { axis },
    )?];
    if let Some(scales) = projection.scales.as_ref().as_ref() {
        members.push(array_member(
            format!("{prefix}.scales"),
            scales,
            MemberSharding::Equal { axis },
        )?);
    }
    if let Some(biases) = projection.biases.as_ref().as_ref() {
        members.push(array_member(
            format!("{prefix}.biases"),
            biases,
            MemberSharding::Equal { axis },
        )?);
    }
    planner.register(ParameterGroupSpec::new(
        prefix,
        ParameterRole::ExpertIntermediate,
        members,
    )?)
}

pub(crate) fn insert_gemma_layer_plan_with_prefix(
    planner: &mut ParallelPlanBuilder,
    layer: &gemma4::TransformerBlock,
    prefix: &str,
) -> Result<(), Error> {
    let attention = &layer.self_attn;
    register_projection(
        planner,
        &attention.q_proj,
        &format!("{prefix}.self_attn.q_proj"),
        ProjectionPlacement::Column,
    )?;
    if let Some(projection) = &attention.k_proj {
        register_projection(
            planner,
            projection,
            &format!("{prefix}.self_attn.k_proj"),
            ProjectionPlacement::Column,
        )?;
    }
    if let Some(projection) = &attention.v_proj {
        register_projection(
            planner,
            projection,
            &format!("{prefix}.self_attn.v_proj"),
            ProjectionPlacement::Column,
        )?;
    }
    register_projection(
        planner,
        &attention.o_proj,
        &format!("{prefix}.self_attn.o_proj"),
        ProjectionPlacement::Row,
    )?;
    register_replicated_module(
        planner,
        &attention.q_norm,
        &format!("{prefix}.self_attn.q_norm"),
    )?;
    if let Some(norm) = &attention.k_norm {
        register_replicated_module(planner, norm, &format!("{prefix}.self_attn.k_norm"))?;
    }
    register_replicated_module(
        planner,
        &attention.rope,
        &format!("{prefix}.self_attn.rope"),
    )?;
    for (name, projection, placement) in [
        (
            "gate_proj",
            &layer.mlp.gate_proj,
            ProjectionPlacement::Column,
        ),
        ("up_proj", &layer.mlp.up_proj, ProjectionPlacement::Column),
        ("down_proj", &layer.mlp.down_proj, ProjectionPlacement::Row),
    ] {
        register_projection(
            planner,
            projection,
            &format!("{prefix}.mlp.{name}"),
            placement,
        )?;
    }
    if let Some(router) = &layer.router {
        register_replicated_module(planner, router, &format!("{prefix}.router"))?;
    }
    if let Some(experts) = &layer.experts {
        let expert_prefix = format!("{prefix}.experts.switch_glu");
        register_gemma_expert_projection(
            planner,
            &experts.switch_glu.gate_proj,
            &format!("{expert_prefix}.gate_proj"),
            1,
        )?;
        register_gemma_expert_projection(
            planner,
            &experts.switch_glu.up_proj,
            &format!("{expert_prefix}.up_proj"),
            1,
        )?;
        register_gemma_expert_projection(
            planner,
            &experts.switch_glu.down_proj,
            &format!("{expert_prefix}.down_proj"),
            2,
        )?;
    }
    if let Some(projection) = &layer.per_layer_input_gate {
        register_projection(
            planner,
            projection,
            &format!("{prefix}.per_layer_input_gate"),
            ProjectionPlacement::Replicated,
        )?;
    }
    if let Some(projection) = &layer.per_layer_projection {
        register_projection(
            planner,
            projection,
            &format!("{prefix}.per_layer_projection"),
            ProjectionPlacement::Replicated,
        )?;
    }
    for (name, norm) in [
        ("input_layernorm", Some(&layer.input_layernorm)),
        (
            "post_attention_layernorm",
            Some(&layer.post_attention_layernorm),
        ),
        (
            "pre_feedforward_layernorm",
            Some(&layer.pre_feedforward_layernorm),
        ),
        (
            "post_feedforward_layernorm",
            Some(&layer.post_feedforward_layernorm),
        ),
        (
            "post_per_layer_input_norm",
            layer.post_per_layer_input_norm.as_ref(),
        ),
        (
            "post_feedforward_layernorm_1",
            layer.post_feedforward_layernorm_1.as_ref(),
        ),
        (
            "pre_feedforward_layernorm_2",
            layer.pre_feedforward_layernorm_2.as_ref(),
        ),
        (
            "post_feedforward_layernorm_2",
            layer.post_feedforward_layernorm_2.as_ref(),
        ),
    ] {
        if let Some(norm) = norm {
            register_replicated_module(planner, norm, &format!("{prefix}.{name}"))?;
        }
    }
    planner.register(ParameterGroupSpec::new(
        format!("{prefix}.layer_scalar"),
        ParameterRole::Replicated,
        [ParameterMemberSpec::new(
            format!("{prefix}.layer_scalar"),
            [1],
            MemberSharding::Replicated,
        )],
    )?)?;
    Ok(())
}

fn insert_gemma_layer_plan(
    planner: &mut ParallelPlanBuilder,
    layer: &gemma4::TransformerBlock,
    index: usize,
) -> Result<(), Error> {
    insert_gemma_layer_plan_with_prefix(
        planner,
        layer,
        &format!("model.language_model.layers.{index}"),
    )
}

pub(crate) fn insert_nemotron_layer_plan(
    planner: &mut ParallelPlanBuilder,
    layer: &nemotron_h::TransformerBlock,
    index: usize,
    split_experts: bool,
) -> Result<(), Error> {
    let prefix = format!("model.layers.{index}");
    register_replicated_module(planner, &layer.norm, &format!("{prefix}.norm"))?;
    match layer.policy {
        nemotron_h::LayerPolicy::Mamba => {
            let mamba = layer.mamba.as_ref().ok_or_else(|| {
                Error::Parallel(format!(
                    "Nemotron-H layer {index} is missing its Mamba mixer"
                ))
            })?;
            let intermediate = usize::try_from(mamba.intermediate_size)
                .map_err(|_| Error::Parallel("Nemotron Mamba width exceeds usize".into()))?;
            let grouped = usize::try_from(mamba.n_groups * mamba.ssm_state_size)
                .map_err(|_| Error::Parallel("Nemotron Mamba group width exceeds usize".into()))?;
            let heads = usize::try_from(mamba.num_heads)
                .map_err(|_| Error::Parallel("Nemotron Mamba heads exceed usize".into()))?;
            let in_segments = vec![
                0..intermediate,
                intermediate..2 * intermediate,
                2 * intermediate..2 * intermediate + grouped,
                2 * intermediate + grouped..2 * intermediate + 2 * grouped,
                2 * intermediate + 2 * grouped..2 * intermediate + 2 * grouped + heads,
            ];
            planner.register(typed_module_group(
                &format!("{prefix}.mamba.in_proj"),
                ParameterRole::Segmented,
                &mamba.in_proj,
                &format!("{prefix}.mamba.in_proj"),
                |_, _| {
                    Ok(MemberSharding::Segmented {
                        axis: 0,
                        segments: in_segments.clone(),
                    })
                },
            )?)?;
            let conv_segments = vec![
                0..intermediate,
                intermediate..intermediate + grouped,
                intermediate + grouped..intermediate + 2 * grouped,
            ];
            let mut convolution = vec![array_member(
                format!("{prefix}.mamba.conv1d.weight"),
                mamba.conv1d.weight.as_ref(),
                MemberSharding::Segmented {
                    axis: 0,
                    segments: conv_segments.clone(),
                },
            )?];
            if let Some(bias) = mamba.conv1d.bias.as_ref().as_ref() {
                convolution.push(array_member(
                    format!("{prefix}.mamba.conv1d.bias"),
                    bias,
                    MemberSharding::Segmented {
                        axis: 0,
                        segments: conv_segments,
                    },
                )?);
            }
            planner.register(ParameterGroupSpec::new(
                format!("{prefix}.mamba.conv1d"),
                ParameterRole::Channels,
                convolution,
            )?)?;
            for (name, value) in [
                ("dt_bias", mamba.dt_bias.as_ref()),
                ("A_log", mamba.A_log.as_ref()),
                ("D", mamba.D.as_ref()),
            ] {
                planner.register(ParameterGroupSpec::new(
                    format!("{prefix}.mamba.{name}"),
                    ParameterRole::Channels,
                    [array_member(
                        format!("{prefix}.mamba.{name}"),
                        value,
                        MemberSharding::Equal { axis: 0 },
                    )?],
                )?)?;
            }
            planner.register(ParameterGroupSpec::new(
                format!("{prefix}.mamba.norm"),
                ParameterRole::Channels,
                [array_member(
                    format!("{prefix}.mamba.norm.weight"),
                    mamba.norm.weight.as_ref(),
                    MemberSharding::Equal { axis: 0 },
                )?],
            )?)?;
            register_projection(
                planner,
                &mamba.out_proj,
                &format!("{prefix}.mamba.out_proj"),
                ProjectionPlacement::Row,
            )?;
        }
        nemotron_h::LayerPolicy::SelfAttention(_) => {
            let attention = layer.attention.as_ref().ok_or_else(|| {
                Error::Parallel(format!(
                    "Nemotron-H layer {index} is missing its attention mixer"
                ))
            })?;
            for (name, projection, placement) in [
                ("q_proj", &attention.q_proj, ProjectionPlacement::Column),
                ("k_proj", &attention.k_proj, ProjectionPlacement::Column),
                ("v_proj", &attention.v_proj, ProjectionPlacement::Column),
                ("o_proj", &attention.o_proj, ProjectionPlacement::Row),
            ] {
                register_projection(
                    planner,
                    projection,
                    &format!("{prefix}.attention.{name}"),
                    placement,
                )?;
            }
        }
        nemotron_h::LayerPolicy::DenseMlp => {
            let mlp = layer.mlp.as_ref().ok_or_else(|| {
                Error::Parallel(format!("Nemotron-H layer {index} is missing its dense MLP"))
            })?;
            register_projection(
                planner,
                &mlp.up_proj,
                &format!("{prefix}.mlp.up_proj"),
                ProjectionPlacement::Column,
            )?;
            register_projection(
                planner,
                &mlp.down_proj,
                &format!("{prefix}.mlp.down_proj"),
                ProjectionPlacement::Row,
            )?;
        }
        nemotron_h::LayerPolicy::SparseMoe => {
            let moe = layer.moe.as_ref().ok_or_else(|| {
                Error::Parallel(format!(
                    "Nemotron-H layer {index} is missing its sparse MoE"
                ))
            })?;
            register_replicated_module(planner, &moe.gate, &format!("{prefix}.moe.gate"))?;
            for (name, projection, placement) in [
                (
                    "up_proj",
                    &moe.shared_experts.up_proj,
                    ProjectionPlacement::Column,
                ),
                (
                    "down_proj",
                    &moe.shared_experts.down_proj,
                    ProjectionPlacement::Row,
                ),
            ] {
                register_projection(
                    planner,
                    projection,
                    &format!("{prefix}.moe.shared_experts.{name}"),
                    placement,
                )?;
            }
            let experts = &moe.experts;
            if split_experts {
                let hidden = usize::try_from(experts.hidden_size)
                    .map_err(|_| Error::Parallel("Nemotron hidden width exceeds usize".into()))?;
                let intermediate = usize::try_from(experts.intermediate_size)
                    .map_err(|_| Error::Parallel("Nemotron expert width exceeds usize".into()))?;
                for expert in 0..experts.num_experts {
                    for (projection, shape, sharding) in [
                        (
                            "up_proj",
                            vec![intermediate, hidden],
                            MemberSharding::Equal { axis: 0 },
                        ),
                        (
                            "down_proj",
                            vec![hidden, intermediate],
                            MemberSharding::Equal { axis: 1 },
                        ),
                    ] {
                        let target = format!("{prefix}.moe.experts.{expert}.{projection}.weight");
                        planner.register(ParameterGroupSpec::new(
                            target.clone(),
                            ParameterRole::ExpertIntermediate,
                            [ParameterMemberSpec::new(target, shape, sharding)],
                        )?)?;
                    }
                }
            } else {
                let mut up = vec![array_member(
                    format!("{prefix}.moe.experts.up_proj"),
                    experts.up_proj.as_ref(),
                    MemberSharding::Equal { axis: 1 },
                )?];
                for (name, value) in [
                    ("up_proj_scales", experts.up_proj_scales.as_ref().as_ref()),
                    ("up_proj_biases", experts.up_proj_biases.as_ref().as_ref()),
                ] {
                    if let Some(value) = value {
                        up.push(array_member(
                            format!("{prefix}.moe.experts.{name}"),
                            value,
                            MemberSharding::Equal { axis: 1 },
                        )?);
                    }
                }
                planner.register(ParameterGroupSpec::new(
                    format!("{prefix}.moe.experts.up"),
                    ParameterRole::ExpertIntermediate,
                    up,
                )?)?;
                let mut down = vec![array_member(
                    format!("{prefix}.moe.experts.down_proj"),
                    experts.down_proj.as_ref(),
                    MemberSharding::Equal { axis: 2 },
                )?];
                for (name, value) in [
                    (
                        "down_proj_scales",
                        experts.down_proj_scales.as_ref().as_ref(),
                    ),
                    (
                        "down_proj_biases",
                        experts.down_proj_biases.as_ref().as_ref(),
                    ),
                ] {
                    if let Some(value) = value {
                        down.push(array_member(
                            format!("{prefix}.moe.experts.{name}"),
                            value,
                            MemberSharding::Equal { axis: 2 },
                        )?);
                    }
                }
                planner.register(ParameterGroupSpec::new(
                    format!("{prefix}.moe.experts.down"),
                    ParameterRole::ExpertIntermediate,
                    down,
                )?)?;
            }
        }
    }
    Ok(())
}

pub(crate) fn insert_deepseek_layer_plan(
    planner: &mut ParallelPlanBuilder,
    layer: &deepseek_v3::DecoderLayer,
    index: usize,
) -> Result<(), Error> {
    let prefix = format!("model.layers.{index}");
    let attention = &layer.self_attn;
    for (name, projection) in [
        ("q_proj", attention.q_proj.as_ref()),
        ("q_b_proj", attention.q_b_proj.as_ref()),
        ("kv_b_proj", attention.kv_b_proj.as_ref()),
    ] {
        if let Some(projection) = projection {
            register_projection_module(
                planner,
                projection,
                &format!("{prefix}.self_attn.{name}"),
                ProjectionPlacement::Column,
            )?;
        }
    }
    for (name, projection) in [
        ("k_b_proj", attention.k_b_proj.as_ref()),
        ("v_b_proj", attention.v_b_proj.as_ref()),
    ] {
        if let Some(projection) = projection {
            register_projection_module(
                planner,
                projection,
                &format!("{prefix}.self_attn.{name}"),
                ProjectionPlacement::Column,
            )?;
        }
    }
    for (name, projection) in [
        ("q_a_proj", attention.q_a_proj.as_ref()),
        ("kv_a_proj_with_mqa", Some(&attention.kv_a_proj_with_mqa)),
    ] {
        if let Some(projection) = projection {
            register_projection_module(
                planner,
                projection,
                &format!("{prefix}.self_attn.{name}"),
                ProjectionPlacement::Replicated,
            )?;
        }
    }
    register_projection_module(
        planner,
        &attention.o_proj,
        &format!("{prefix}.self_attn.o_proj"),
        ProjectionPlacement::Row,
    )?;
    for (name, module) in [
        ("q_a_layernorm", attention.q_a_layernorm.as_ref()),
        ("kv_a_layernorm", Some(&attention.kv_a_layernorm)),
    ] {
        if let Some(module) = module {
            register_replicated_module(planner, module, &format!("{prefix}.self_attn.{name}"))?;
        }
    }
    register_replicated_module(
        planner,
        &attention.rope,
        &format!("{prefix}.self_attn.rope"),
    )?;
    for (name, norm) in [
        ("input_layernorm", &layer.input_layernorm),
        ("post_attention_layernorm", &layer.post_attention_layernorm),
    ] {
        register_replicated_module(planner, norm, &format!("{prefix}.{name}"))?;
    }
    let register_mlp = |planner: &mut ParallelPlanBuilder,
                        mlp: &deepseek_v3::Mlp,
                        prefix: &str|
     -> Result<(), Error> {
        for (name, projection, placement) in [
            ("gate_proj", &mlp.gate_proj, ProjectionPlacement::Column),
            ("up_proj", &mlp.up_proj, ProjectionPlacement::Column),
            ("down_proj", &mlp.down_proj, ProjectionPlacement::Row),
        ] {
            register_projection_module(
                planner,
                projection,
                &format!("{prefix}.{name}"),
                placement,
            )?;
        }
        Ok(())
    };
    match &layer.mlp {
        deepseek_v3::FeedForward::Dense(mlp) => {
            register_mlp(planner, mlp, &format!("{prefix}.mlp"))?;
        }
        deepseek_v3::FeedForward::Moe(moe) => {
            register_replicated_module(planner, &moe.gate, &format!("{prefix}.mlp.gate"))?;
            register_mlp(
                planner,
                &moe.shared_experts,
                &format!("{prefix}.mlp.shared_experts"),
            )?;
            let experts = &moe.experts;
            let banks = [
                (
                    "gate_proj",
                    experts.gate_proj.as_ref().as_ref(),
                    MemberSharding::Equal { axis: 0 },
                ),
                (
                    "gate_proj.weight_scale_inv",
                    experts.gate_proj_scale_inv.as_ref().as_ref(),
                    MemberSharding::Equal { axis: 0 },
                ),
                (
                    "gate_proj.scales",
                    experts.gate_proj_scales.as_ref().as_ref(),
                    MemberSharding::Equal { axis: 0 },
                ),
                (
                    "gate_proj.biases",
                    experts.gate_proj_biases.as_ref().as_ref(),
                    MemberSharding::Equal { axis: 0 },
                ),
                (
                    "up_proj",
                    experts.up_proj.as_ref().as_ref(),
                    MemberSharding::Equal { axis: 0 },
                ),
                (
                    "up_proj.weight_scale_inv",
                    experts.up_proj_scale_inv.as_ref().as_ref(),
                    MemberSharding::Equal { axis: 0 },
                ),
                (
                    "up_proj.scales",
                    experts.up_proj_scales.as_ref().as_ref(),
                    MemberSharding::Equal { axis: 0 },
                ),
                (
                    "up_proj.biases",
                    experts.up_proj_biases.as_ref().as_ref(),
                    MemberSharding::Equal { axis: 0 },
                ),
                (
                    "down_proj",
                    experts.down_proj.as_ref().as_ref(),
                    MemberSharding::Equal { axis: 1 },
                ),
                (
                    "down_proj.weight_scale_inv",
                    experts.down_proj_scale_inv.as_ref().as_ref(),
                    MemberSharding::Equal { axis: 1 },
                ),
                (
                    "down_proj.scales",
                    experts.down_proj_scales.as_ref().as_ref(),
                    MemberSharding::Equal { axis: 1 },
                ),
                (
                    "down_proj.biases",
                    experts.down_proj_biases.as_ref().as_ref(),
                    MemberSharding::Equal { axis: 1 },
                ),
            ];
            for expert in 0..experts.num_experts {
                for (suffix, bank, sharding) in &banks {
                    let Some(bank) = bank else { continue };
                    let shape = bank.shape()[1..]
                        .iter()
                        .map(|dimension| {
                            usize::try_from(*dimension).map_err(|_| {
                                Error::Parallel("DeepSeek expert shape exceeds usize".into())
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let (projection, component) =
                        suffix.split_once('.').unwrap_or((suffix, "weight"));
                    let target = format!("{prefix}.mlp.experts.{expert}.{projection}.{component}");
                    planner.register(ParameterGroupSpec::new(
                        target.clone(),
                        ParameterRole::ExpertIntermediate,
                        [ParameterMemberSpec::new(target, shape, sharding.clone())],
                    )?)?;
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn insert_kimi_layer_plan(
    planner: &mut ParallelPlanBuilder,
    layer: &kimi_linear::DecoderLayer,
    index: usize,
    split_experts: bool,
) -> Result<(), Error> {
    insert_kimi_layer_plan_with_names(planner, layer, index, split_experts, false)
}

/// Registers Kimi runtime parameter names for the generalized layerwise
/// loader, whose derived bindings independently describe official checkpoint
/// source names.
pub(crate) fn insert_kimi_layerwise_plan(
    planner: &mut ParallelPlanBuilder,
    layer: &kimi_linear::DecoderLayer,
    index: usize,
) -> Result<(), Error> {
    insert_kimi_layer_plan_with_names(planner, layer, index, false, true)
}

fn insert_kimi_layer_plan_with_names(
    planner: &mut ParallelPlanBuilder,
    layer: &kimi_linear::DecoderLayer,
    index: usize,
    split_experts: bool,
    runtime_names: bool,
) -> Result<(), Error> {
    let prefix = format!("model.layers.{index}");
    match &layer.self_attn {
        kimi_linear::Attention::Kda(attention) => {
            for (name, projection) in [
                ("q_proj", &attention.q_proj),
                ("k_proj", &attention.k_proj),
                ("v_proj", &attention.v_proj),
                ("f_b_proj", &attention.f_b_proj),
                ("b_proj", &attention.b_proj),
                ("g_b_proj", &attention.g_b_proj),
            ] {
                register_projection_module(
                    planner,
                    projection,
                    &format!("{prefix}.self_attn.{name}"),
                    ProjectionPlacement::Column,
                )?;
            }
            for name in ["q_conv1d", "k_conv1d", "v_conv1d"] {
                let convolution = match name {
                    "q_conv1d" => &attention.q_conv1d,
                    "k_conv1d" => &attention.k_conv1d,
                    "v_conv1d" => &attention.v_conv1d,
                    _ => unreachable!(),
                };
                planner.register(ParameterGroupSpec::new(
                    format!("{prefix}.self_attn.{name}"),
                    ParameterRole::Channels,
                    [array_member(
                        format!("{prefix}.self_attn.{name}.weight"),
                        convolution.weight.as_ref(),
                        MemberSharding::Equal { axis: 0 },
                    )?],
                )?)?;
            }
            for (name, projection) in [
                ("f_a_proj", &attention.f_a_proj),
                ("g_a_proj", &attention.g_a_proj),
            ] {
                register_projection_module(
                    planner,
                    projection,
                    &format!("{prefix}.self_attn.{name}"),
                    ProjectionPlacement::Replicated,
                )?;
            }
            for (name, value, axis) in [
                ("A_log", attention.A_log.as_ref(), 2usize),
                ("dt_bias", attention.dt_bias.as_ref(), 0usize),
            ] {
                planner.register(ParameterGroupSpec::new(
                    format!("{prefix}.self_attn.{name}"),
                    ParameterRole::Channels,
                    [array_member(
                        format!("{prefix}.self_attn.{name}"),
                        value,
                        MemberSharding::Equal { axis },
                    )?],
                )?)?;
            }
            register_replicated_module(
                planner,
                &attention.o_norm,
                &format!("{prefix}.self_attn.o_norm"),
            )?;
            register_projection_module(
                planner,
                &attention.o_proj,
                &format!("{prefix}.self_attn.o_proj"),
                ProjectionPlacement::Row,
            )?;
        }
        kimi_linear::Attention::Mla(attention) => {
            for (name, projection) in [
                ("q_proj", attention.q_proj.as_ref()),
                ("q_b_proj", attention.q_b_proj.as_ref()),
                ("kv_b_proj", attention.kv_b_proj.as_ref()),
            ] {
                if let Some(projection) = projection {
                    register_projection_module(
                        planner,
                        projection,
                        &format!("{prefix}.self_attn.{name}"),
                        ProjectionPlacement::Column,
                    )?;
                }
            }
            for (name, projection) in [
                ("k_b_proj", attention.k_b_proj.as_ref()),
                ("v_b_proj", attention.v_b_proj.as_ref()),
            ] {
                if let Some(projection) = projection {
                    register_projection_module(
                        planner,
                        projection,
                        &format!("{prefix}.self_attn.{name}"),
                        ProjectionPlacement::Column,
                    )?;
                }
            }
            for (name, projection) in [
                ("q_a_proj", attention.q_a_proj.as_ref()),
                ("kv_a_proj_with_mqa", Some(&attention.kv_a_proj_with_mqa)),
            ] {
                if let Some(projection) = projection {
                    register_projection_module(
                        planner,
                        projection,
                        &format!("{prefix}.self_attn.{name}"),
                        ProjectionPlacement::Replicated,
                    )?;
                }
            }
            register_projection_module(
                planner,
                &attention.o_proj,
                &format!("{prefix}.self_attn.o_proj"),
                ProjectionPlacement::Row,
            )?;
            for (name, norm) in [
                ("q_a_layernorm", attention.q_a_layernorm.as_ref()),
                ("kv_a_layernorm", Some(&attention.kv_a_layernorm)),
            ] {
                if let Some(norm) = norm {
                    register_replicated_module(
                        planner,
                        norm,
                        &format!("{prefix}.self_attn.{name}"),
                    )?;
                }
            }
            register_replicated_module(
                planner,
                &attention.rope,
                &format!("{prefix}.self_attn.rope"),
            )?;
        }
    }
    for (name, norm) in [
        ("input_layernorm", &layer.input_layernorm),
        ("post_attention_layernorm", &layer.post_attention_layernorm),
    ] {
        register_replicated_module(planner, norm, &format!("{prefix}.{name}"))?;
    }
    let register_swiglu = |planner: &mut ParallelPlanBuilder,
                           mlp: &crate::nn::layers::SwiGluMlp,
                           prefix: &str|
     -> Result<(), Error> {
        for (name, projection, placement) in [
            ("gate_proj", &mlp.gate_proj, ProjectionPlacement::Column),
            ("up_proj", &mlp.up_proj, ProjectionPlacement::Column),
            ("down_proj", &mlp.down_proj, ProjectionPlacement::Row),
        ] {
            register_projection(planner, projection, &format!("{prefix}.{name}"), placement)?;
        }
        Ok(())
    };
    match &layer.mlp {
        kimi_linear::FeedForward::Dense(mlp) => {
            register_swiglu(planner, mlp, &format!("{prefix}.mlp"))?;
        }
        kimi_linear::FeedForward::Moe(moe) => {
            let source_prefix = if runtime_names {
                format!("{prefix}.mlp")
            } else {
                format!("{prefix}.block_sparse_moe")
            };
            register_replicated_module(planner, &moe.gate, &format!("{source_prefix}.gate"))?;
            register_swiglu(
                planner,
                &moe.shared_experts,
                &format!("{source_prefix}.shared_experts"),
            )?;
            if split_experts {
                let experts = &moe.experts;
                let hidden = usize::try_from(experts.hidden_dim)
                    .map_err(|_| Error::Parallel("Kimi hidden width exceeds usize".into()))?;
                let intermediate = usize::try_from(experts.intermediate_dim)
                    .map_err(|_| Error::Parallel("Kimi expert width exceeds usize".into()))?;
                for expert in 0..experts.num_experts {
                    for (projection, shape, sharding) in [
                        (
                            "gate_proj",
                            vec![intermediate, hidden],
                            MemberSharding::Equal { axis: 0 },
                        ),
                        (
                            "up_proj",
                            vec![intermediate, hidden],
                            MemberSharding::Equal { axis: 0 },
                        ),
                        (
                            "down_proj",
                            vec![hidden, intermediate],
                            MemberSharding::Equal { axis: 1 },
                        ),
                    ] {
                        let target =
                            format!("{source_prefix}.experts.{expert}.{projection}.weight");
                        planner.register(ParameterGroupSpec::new(
                            target.clone(),
                            ParameterRole::ExpertIntermediate,
                            [ParameterMemberSpec::new(target, shape, sharding)],
                        )?)?;
                    }
                }
            } else {
                let experts = &moe.experts;
                let intermediate = usize::try_from(experts.intermediate_dim)
                    .map_err(|_| Error::Parallel("Kimi expert width exceeds usize".into()))?;
                let segments = vec![0..intermediate, intermediate..2 * intermediate];
                let mut gate_up = vec![array_member(
                    format!("{source_prefix}.experts.gate_up_proj"),
                    experts.gate_up_proj.as_ref(),
                    MemberSharding::Segmented {
                        axis: 1,
                        segments: segments.clone(),
                    },
                )?];
                for (name, value) in [
                    (
                        "gate_up_proj_scales",
                        experts.gate_up_proj_scales.as_ref().as_ref(),
                    ),
                    (
                        "gate_up_proj_biases",
                        experts.gate_up_proj_biases.as_ref().as_ref(),
                    ),
                ] {
                    if let Some(value) = value {
                        gate_up.push(array_member(
                            format!("{source_prefix}.experts.{name}"),
                            value,
                            MemberSharding::Segmented {
                                axis: 1,
                                segments: segments.clone(),
                            },
                        )?);
                    }
                }
                planner.register(ParameterGroupSpec::new(
                    format!("{source_prefix}.experts.gate_up"),
                    ParameterRole::ExpertIntermediate,
                    gate_up,
                )?)?;
                let mut down = vec![array_member(
                    format!("{source_prefix}.experts.down_proj"),
                    experts.down_proj.as_ref(),
                    MemberSharding::Equal { axis: 2 },
                )?];
                for (name, value) in [
                    (
                        "down_proj_scales",
                        experts.down_proj_scales.as_ref().as_ref(),
                    ),
                    (
                        "down_proj_biases",
                        experts.down_proj_biases.as_ref().as_ref(),
                    ),
                ] {
                    if let Some(value) = value {
                        down.push(array_member(
                            format!("{source_prefix}.experts.{name}"),
                            value,
                            MemberSharding::Equal { axis: 2 },
                        )?);
                    }
                }
                planner.register(ParameterGroupSpec::new(
                    format!("{source_prefix}.experts.down"),
                    ParameterRole::ExpertIntermediate,
                    down,
                )?)?;
            }
        }
    }
    Ok(())
}

pub(crate) fn insert_lfm2_layer_plan(
    planner: &mut ParallelPlanBuilder,
    layer: &crate::api::lfm2::DecoderLayer,
    index: usize,
    split_experts: bool,
) -> Result<(), Error> {
    let prefix = format!("model.layers.{index}");
    if let Some(attention) = &layer.self_attn {
        for (name, projection) in [
            ("q_proj", &attention.q_proj),
            ("k_proj", &attention.k_proj),
            ("v_proj", &attention.v_proj),
        ] {
            register_projection(
                planner,
                projection,
                &format!("{prefix}.self_attn.{name}"),
                ProjectionPlacement::Column,
            )?;
        }
        register_projection(
            planner,
            &attention.out_proj,
            &format!("{prefix}.self_attn.out_proj"),
            ProjectionPlacement::Row,
        )?;
        for (name, module) in [
            ("q_layernorm", &attention.q_layernorm),
            ("k_layernorm", &attention.k_layernorm),
        ] {
            register_replicated_module(planner, module, &format!("{prefix}.self_attn.{name}"))?;
        }
        register_replicated_module(
            planner,
            &attention.rope,
            &format!("{prefix}.self_attn.rope"),
        )?;
    }
    if let Some(conv) = &layer.conv {
        register_replicated_module(planner, conv, &format!("{prefix}.conv"))?;
    }
    for (name, norm) in [
        ("operator_norm", &layer.operator_norm),
        ("ffn_norm", &layer.ffn_norm),
    ] {
        register_replicated_module(planner, norm, &format!("{prefix}.{name}"))?;
    }
    let feed_forward = &layer.feed_forward;
    if feed_forward.is_moe {
        let gate = feed_forward.gate.as_ref().ok_or_else(|| {
            Error::Parallel(format!("LFM2 layer {index} is missing its MoE gate"))
        })?;
        register_replicated_module(planner, gate, &format!("{prefix}.feed_forward.gate"))?;
        if let Some(bias) = feed_forward.expert_bias.as_ref().as_ref() {
            planner.register(ParameterGroupSpec::new(
                format!("{prefix}.feed_forward.expert_bias"),
                ParameterRole::Replicated,
                [array_member(
                    format!("{prefix}.feed_forward.expert_bias"),
                    bias,
                    MemberSharding::Replicated,
                )?],
            )?)?;
        }
        let experts = feed_forward.experts.as_ref().ok_or_else(|| {
            Error::Parallel(format!("LFM2 layer {index} is missing its expert bank"))
        })?;
        let expert_prefix = format!("{prefix}.feed_forward.experts");
        if split_experts {
            let hidden = usize::try_from(experts.hidden_dim)
                .map_err(|_| Error::Parallel("LFM2 hidden width exceeds usize".into()))?;
            let intermediate = usize::try_from(experts.intermediate_dim)
                .map_err(|_| Error::Parallel("LFM2 expert width exceeds usize".into()))?;
            for expert in 0..experts.num_experts {
                for (projection, shape, sharding) in [
                    (
                        "w1",
                        vec![intermediate, hidden],
                        MemberSharding::Equal { axis: 0 },
                    ),
                    (
                        "w3",
                        vec![intermediate, hidden],
                        MemberSharding::Equal { axis: 0 },
                    ),
                    (
                        "w2",
                        vec![hidden, intermediate],
                        MemberSharding::Equal { axis: 1 },
                    ),
                ] {
                    let target = format!("{expert_prefix}.{expert}.{projection}.weight");
                    planner.register(ParameterGroupSpec::new(
                        target.clone(),
                        ParameterRole::ExpertIntermediate,
                        [ParameterMemberSpec::new(target, shape, sharding)],
                    )?)?;
                }
            }
        } else {
            let intermediate = usize::try_from(experts.intermediate_dim)
                .map_err(|_| Error::Parallel("LFM2 expert width exceeds usize".into()))?;
            let segments = vec![0..intermediate, intermediate..2 * intermediate];
            let mut gate_up = vec![array_member(
                format!("{expert_prefix}.gate_up_proj"),
                experts.gate_up_proj.as_ref(),
                MemberSharding::Segmented {
                    axis: 1,
                    segments: segments.clone(),
                },
            )?];
            for (name, value) in [
                (
                    "gate_up_proj_scales",
                    experts.gate_up_proj_scales.as_ref().as_ref(),
                ),
                (
                    "gate_up_proj_biases",
                    experts.gate_up_proj_biases.as_ref().as_ref(),
                ),
            ] {
                if let Some(value) = value {
                    gate_up.push(array_member(
                        format!("{expert_prefix}.{name}"),
                        value,
                        MemberSharding::Segmented {
                            axis: 1,
                            segments: segments.clone(),
                        },
                    )?);
                }
            }
            planner.register(ParameterGroupSpec::new(
                format!("{expert_prefix}.gate_up"),
                ParameterRole::ExpertIntermediate,
                gate_up,
            )?)?;
            let mut down = vec![array_member(
                format!("{expert_prefix}.down_proj"),
                experts.down_proj.as_ref(),
                MemberSharding::Equal { axis: 2 },
            )?];
            for (name, value) in [
                (
                    "down_proj_scales",
                    experts.down_proj_scales.as_ref().as_ref(),
                ),
                (
                    "down_proj_biases",
                    experts.down_proj_biases.as_ref().as_ref(),
                ),
            ] {
                if let Some(value) = value {
                    down.push(array_member(
                        format!("{expert_prefix}.{name}"),
                        value,
                        MemberSharding::Equal { axis: 2 },
                    )?);
                }
            }
            planner.register(ParameterGroupSpec::new(
                format!("{expert_prefix}.down"),
                ParameterRole::ExpertIntermediate,
                down,
            )?)?;
        }
    } else {
        for (name, projection, placement) in [
            ("w1", feed_forward.w1.as_ref(), ProjectionPlacement::Column),
            ("w3", feed_forward.w3.as_ref(), ProjectionPlacement::Column),
            ("w2", feed_forward.w2.as_ref(), ProjectionPlacement::Row),
        ] {
            let projection = projection.ok_or_else(|| {
                Error::Parallel(format!("LFM2 dense layer {index} is missing {name}"))
            })?;
            register_projection(
                planner,
                projection,
                &format!("{prefix}.feed_forward.{name}"),
                placement,
            )?;
        }
    }
    Ok(())
}

pub(crate) fn insert_qwen_hybrid_layer_plan(
    planner: &mut ParallelPlanBuilder,
    layer: &qwen_hybrid::TransformerBlock,
    index: usize,
    args: &qwen_hybrid::ModelArgs,
    split_experts: bool,
) -> Result<(), Error> {
    let prefix = format!("model.layers.{index}");
    if let Some(attention) = &layer.self_attn {
        let query_width = usize::try_from(attention.n_heads * attention.head_dim)
            .map_err(|_| Error::Parallel("Qwen hybrid query width exceeds usize".into()))?;
        let query_blocks = query_width.div_ceil(128);
        planner.register(typed_module_group(
            &format!("{prefix}.self_attn.q_proj"),
            ParameterRole::Segmented,
            &attention.q_proj,
            &format!("{prefix}.self_attn.q_proj"),
            |name, _| {
                let width = if name == "weight_scale_inv" {
                    query_blocks
                } else {
                    query_width
                };
                Ok(MemberSharding::Segmented {
                    axis: 0,
                    segments: vec![0..width, width..2 * width],
                })
            },
        )?)?;
        for (name, projection) in [("k_proj", &attention.k_proj), ("v_proj", &attention.v_proj)] {
            register_projection_module(
                planner,
                projection,
                &format!("{prefix}.self_attn.{name}"),
                ProjectionPlacement::Column,
            )?;
        }
        register_projection_module(
            planner,
            &attention.o_proj,
            &format!("{prefix}.self_attn.o_proj"),
            ProjectionPlacement::Row,
        )?;
        for (name, module) in [("q_norm", &attention.q_norm), ("k_norm", &attention.k_norm)] {
            register_replicated_module(planner, module, &format!("{prefix}.self_attn.{name}"))?;
        }
        register_replicated_module(
            planner,
            &attention.rope,
            &format!("{prefix}.self_attn.rope"),
        )?;
    }
    if let Some(linear_attention) = &layer.linear_attn {
        if args.model_type == "qwen3_next" {
            let fused_group = |planner: &mut ParallelPlanBuilder,
                               name: &str,
                               first: &qwen_hybrid::QwenLinear,
                               second: &qwen_hybrid::QwenLinear|
             -> Result<(), Error> {
                let first_params = first.parameters().flatten();
                let second_params = second.parameters().flatten();
                let mut members = Vec::new();
                for component in ["weight", "weight_scale_inv", "scales", "biases", "bias"] {
                    let (Some(first), Some(second)) =
                        (first_params.get(component), second_params.get(component))
                    else {
                        continue;
                    };
                    let mut shape = first.shape().to_vec();
                    shape[0] += second.dim(0);
                    members.push(ParameterMemberSpec::new(
                        format!("{prefix}.linear_attn.{name}.{component}"),
                        shape
                            .into_iter()
                            .map(|dimension| {
                                usize::try_from(dimension).map_err(|_| {
                                    Error::Parallel(
                                        "Qwen fused projection shape exceeds usize".into(),
                                    )
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                        MemberSharding::Equal { axis: 0 },
                    ));
                }
                planner.register(ParameterGroupSpec::new(
                    format!("{prefix}.linear_attn.{name}"),
                    ParameterRole::Channels,
                    members,
                )?)
            };
            fused_group(
                planner,
                "in_proj_qkvz",
                &linear_attention.in_proj_qkv,
                &linear_attention.in_proj_z,
            )?;
            fused_group(
                planner,
                "in_proj_ba",
                &linear_attention.in_proj_b,
                &linear_attention.in_proj_a,
            )?;
        } else {
            let key_width = usize::try_from(linear_attention.key_dim)
                .map_err(|_| Error::Parallel("Qwen recurrent key width exceeds usize".into()))?;
            let value_width = usize::try_from(linear_attention.value_dim)
                .map_err(|_| Error::Parallel("Qwen recurrent value width exceeds usize".into()))?;
            let standard_segments = vec![
                0..key_width,
                key_width..2 * key_width,
                2 * key_width..2 * key_width + value_width,
            ];
            let key_blocks = key_width.div_ceil(128);
            let value_blocks = value_width.div_ceil(128);
            let scale_segments = vec![
                0..key_blocks,
                key_blocks..2 * key_blocks,
                2 * key_blocks..2 * key_blocks + value_blocks,
            ];
            planner.register(typed_module_group(
                &format!("{prefix}.linear_attn.in_proj_qkv"),
                ParameterRole::Segmented,
                &linear_attention.in_proj_qkv,
                &format!("{prefix}.linear_attn.in_proj_qkv"),
                |name, _| {
                    Ok(MemberSharding::Segmented {
                        axis: 0,
                        segments: if name == "weight_scale_inv" {
                            scale_segments.clone()
                        } else {
                            standard_segments.clone()
                        },
                    })
                },
            )?)?;
            for (name, projection) in [
                ("in_proj_z", &linear_attention.in_proj_z),
                ("in_proj_b", &linear_attention.in_proj_b),
                ("in_proj_a", &linear_attention.in_proj_a),
            ] {
                register_projection_module(
                    planner,
                    projection,
                    &format!("{prefix}.linear_attn.{name}"),
                    ProjectionPlacement::Column,
                )?;
            }
        }
        let key_width = usize::try_from(linear_attention.key_dim)
            .map_err(|_| Error::Parallel("Qwen recurrent key width exceeds usize".into()))?;
        let value_width = usize::try_from(linear_attention.value_dim)
            .map_err(|_| Error::Parallel("Qwen recurrent value width exceeds usize".into()))?;
        planner.register(ParameterGroupSpec::new(
            format!("{prefix}.linear_attn.conv1d"),
            ParameterRole::Channels,
            [array_member(
                format!("{prefix}.linear_attn.conv1d.weight"),
                linear_attention.conv1d.weight.as_ref(),
                MemberSharding::Segmented {
                    axis: 0,
                    segments: vec![
                        0..key_width,
                        key_width..2 * key_width,
                        2 * key_width..2 * key_width + value_width,
                    ],
                },
            )?],
        )?)?;
        for (name, module) in [
            ("dt_bias", linear_attention.dt_bias.as_ref()),
            ("A_log", linear_attention.A_log.as_ref()),
        ] {
            planner.register(ParameterGroupSpec::new(
                format!("{prefix}.linear_attn.{name}"),
                ParameterRole::Channels,
                [array_member(
                    format!("{prefix}.linear_attn.{name}"),
                    module,
                    MemberSharding::Equal { axis: 0 },
                )?],
            )?)?;
        }
        register_replicated_module(
            planner,
            &linear_attention.norm,
            &format!("{prefix}.linear_attn.norm"),
        )?;
        register_projection_module(
            planner,
            &linear_attention.out_proj,
            &format!("{prefix}.linear_attn.out_proj"),
            ProjectionPlacement::Row,
        )?;
    }
    for (name, norm) in [
        ("input_layernorm", &layer.input_layernorm),
        ("post_attention_layernorm", &layer.post_attention_layernorm),
    ] {
        register_replicated_module(planner, norm, &format!("{prefix}.{name}"))?;
    }
    let register_mlp = |planner: &mut ParallelPlanBuilder,
                        mlp: &qwen_hybrid::Mlp,
                        prefix: &str|
     -> Result<(), Error> {
        for (name, projection, placement) in [
            ("gate_proj", &mlp.gate_proj, ProjectionPlacement::Column),
            ("up_proj", &mlp.up_proj, ProjectionPlacement::Column),
            ("down_proj", &mlp.down_proj, ProjectionPlacement::Row),
        ] {
            register_projection_module(
                planner,
                projection,
                &format!("{prefix}.{name}"),
                placement,
            )?;
        }
        Ok(())
    };
    match &layer.mlp {
        qwen_hybrid::FeedForward::Dense(mlp) => {
            register_mlp(planner, mlp, &format!("{prefix}.mlp"))?;
        }
        qwen_hybrid::FeedForward::Moe(moe) => {
            register_replicated_module(planner, &moe.gate, &format!("{prefix}.mlp.gate"))?;
            register_replicated_module(
                planner,
                &moe.shared_expert_gate,
                &format!("{prefix}.mlp.shared_expert_gate"),
            )?;
            register_mlp(
                planner,
                &moe.shared_expert,
                &format!("{prefix}.mlp.shared_expert"),
            )?;
            let experts = &moe.experts;
            let expert_prefix = format!("{prefix}.mlp.experts");
            if split_experts || experts.use_fp8 {
                let hidden = usize::try_from(experts.hidden_dim).map_err(|_| {
                    Error::Parallel("Qwen hybrid hidden width exceeds usize".into())
                })?;
                let intermediate = usize::try_from(experts.intermediate_dim).map_err(|_| {
                    Error::Parallel("Qwen hybrid expert width exceeds usize".into())
                })?;
                for expert in 0..experts.num_experts {
                    for (projection, shape, sharding) in [
                        (
                            "gate_proj",
                            vec![intermediate, hidden],
                            MemberSharding::Equal { axis: 0 },
                        ),
                        (
                            "up_proj",
                            vec![intermediate, hidden],
                            MemberSharding::Equal { axis: 0 },
                        ),
                        (
                            "down_proj",
                            vec![hidden, intermediate],
                            MemberSharding::Equal { axis: 1 },
                        ),
                    ] {
                        let target = format!("{expert_prefix}.{expert}.{projection}.weight");
                        planner.register(ParameterGroupSpec::new(
                            target.clone(),
                            ParameterRole::ExpertIntermediate,
                            [ParameterMemberSpec::new(
                                target,
                                shape.clone(),
                                sharding.clone(),
                            )],
                        )?)?;
                        if experts.use_fp8 {
                            let scale_shape = vec![shape[0].div_ceil(128), shape[1].div_ceil(128)];
                            let target =
                                format!("{expert_prefix}.{expert}.{projection}.weight_scale_inv");
                            planner.register(ParameterGroupSpec::new(
                                target.clone(),
                                ParameterRole::ExpertIntermediate,
                                [ParameterMemberSpec::new(target, scale_shape, sharding)],
                            )?)?;
                        }
                    }
                }
            } else {
                let intermediate = usize::try_from(experts.intermediate_dim).map_err(|_| {
                    Error::Parallel("Qwen hybrid expert width exceeds usize".into())
                })?;
                let segments = vec![0..intermediate, intermediate..2 * intermediate];
                let mut gate_up = vec![array_member(
                    format!("{expert_prefix}.gate_up_proj"),
                    experts.gate_up_proj.as_ref(),
                    MemberSharding::Segmented {
                        axis: 1,
                        segments: segments.clone(),
                    },
                )?];
                for (name, value) in [
                    (
                        "gate_up_proj_scales",
                        experts.gate_up_proj_scales.as_ref().as_ref(),
                    ),
                    (
                        "gate_up_proj_biases",
                        experts.gate_up_proj_biases.as_ref().as_ref(),
                    ),
                ] {
                    if let Some(value) = value {
                        gate_up.push(array_member(
                            format!("{expert_prefix}.{name}"),
                            value,
                            MemberSharding::Segmented {
                                axis: 1,
                                segments: segments.clone(),
                            },
                        )?);
                    }
                }
                planner.register(ParameterGroupSpec::new(
                    format!("{expert_prefix}.gate_up"),
                    ParameterRole::ExpertIntermediate,
                    gate_up,
                )?)?;
                let mut down = vec![array_member(
                    format!("{expert_prefix}.down_proj"),
                    experts.down_proj.as_ref(),
                    MemberSharding::Equal { axis: 2 },
                )?];
                for (name, value) in [
                    (
                        "down_proj_scales",
                        experts.down_proj_scales.as_ref().as_ref(),
                    ),
                    (
                        "down_proj_biases",
                        experts.down_proj_biases.as_ref().as_ref(),
                    ),
                ] {
                    if let Some(value) = value {
                        down.push(array_member(
                            format!("{expert_prefix}.{name}"),
                            value,
                            MemberSharding::Equal { axis: 2 },
                        )?);
                    }
                }
                planner.register(ParameterGroupSpec::new(
                    format!("{expert_prefix}.down"),
                    ParameterRole::ExpertIntermediate,
                    down,
                )?)?;
            }
        }
    }
    Ok(())
}

fn partition_info(
    partition: &RankPartition,
    topology: ParallelTopology,
    kind: ModelKind,
    heads: (i32, i32),
    local_heads: (i32, i32),
    vocabulary: Range<usize>,
) -> TensorParallelInfo {
    let mut owned_tensors = partition
        .tensors()
        .map(|(name, _)| canonical_checkpoint_name(name))
        .collect::<Vec<_>>();
    owned_tensors.sort();
    TensorParallelInfo {
        global_rank: topology.global_rank,
        tensor_parallel_rank: topology.tensor_parallel_rank,
        tensor_parallel_size: topology.tensor_parallel_size,
        model_kind: kind,
        global_attention_heads: heads.0,
        local_attention_heads: local_heads.0,
        global_kv_heads: heads.1,
        local_kv_heads: local_heads.1,
        local_vocabulary_range: vocabulary,
        owned_tensors,
        local_parameter_bytes: partition.tensors().map(|(_, value)| value.nbytes()).sum(),
        opened_checkpoint_shards: partition.opened_shards().to_vec(),
    }
}

/// Loads an executable rank-local pure tensor-parallel model.
pub fn load_tensor_parallel_model(
    model_dir: impl AsRef<Path>,
    topology: ParallelTopology,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<TensorParallelModel, Error> {
    load_tensor_parallel_model_with_options(
        model_dir,
        ModelLoadOptions::with_parallel(topology),
        stream,
        weights_stream,
    )
}

/// Loads an executable rank-local pure tensor-parallel model with options.
pub fn load_tensor_parallel_model_with_options(
    model_dir: impl AsRef<Path>,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<TensorParallelModel, Error> {
    let model_dir = model_dir.as_ref();
    let topology = options.parallel.ok_or_else(|| {
        Error::Parallel("tensor-parallel loading requires ModelLoadOptions::parallel".into())
    })?;
    validate_pure_tensor(topology)?;
    topology.validate_execution_stream(stream)?;
    if model_dir
        .extension()
        .is_some_and(|extension| extension == "gguf")
    {
        return Err(Error::Parallel(
            "tensor-parallel GGUF loading is unsupported because bounded local-range selection is unavailable; use safetensors"
                .into(),
        ));
    }
    let config: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(model_dir.join("config.json"))?)?;
    match config.get("model_type").and_then(serde_json::Value::as_str) {
        Some("llama" | "mistral") => {
            load_llama(model_dir, topology, options.quantization, stream, weights_stream)
        }
        Some("qwen2" | "qwen3") => {
            load_qwen(model_dir, topology, options.quantization, stream, weights_stream)
        }
        Some("gpt_oss") => load_gpt_oss(
            model_dir,
            topology,
            options.quantization,
            stream,
            weights_stream,
        ),
        Some("kimi_linear") => load_kimi(
            model_dir,
            topology,
            options.quantization,
            stream,
            weights_stream,
        ),
        Some("lfm2" | "lfm2_moe") => load_lfm2(
            model_dir,
            topology,
            options.quantization,
            stream,
            weights_stream,
        ),
        Some("nemotron_h") => load_nemotron_h(
            model_dir,
            topology,
            options.quantization,
            stream,
            weights_stream,
        ),
        Some(
            "qwen3_next"
            | "qwen3_5"
            | "qwen3_5_text"
            | "qwen3_5_moe"
            | "qwen3_5_moe_text",
        ) => load_qwen_hybrid(
            model_dir,
            topology,
            options.quantization,
            stream,
            weights_stream,
        ),
        Some("gemma4" | "gemma4_text" | "gemma4_unified" | "gemma4_unified_text") => {
            load_gemma(
                model_dir,
                &config,
                topology,
                options.quantization,
                stream,
                weights_stream,
            )
        }
        Some("deepseek_v3") => {
            load_deepseek(model_dir, topology, options.quantization, stream, weights_stream)
        }
        Some(model_type) => Err(Error::UnsupportedArchitecture(format!(
            "tensor-parallel execution supports Llama/Mistral, Qwen2/Qwen3, Gemma text, GPT-OSS, DeepSeek-V3/R1, Kimi Linear, LFM2, Nemotron-H, and Qwen3-Next/Qwen3.5 text models, not {model_type}"
        ))),
        None => Err(Error::UnsupportedArchitecture(
            "tensor-parallel model config is missing model_type".into(),
        )),
    }
}

fn load_partition(
    model_dir: &Path,
    plan: &PlacementPlan,
    weights_stream: &Stream,
    stream: &Stream,
    strict: &StrictLoadConfig,
) -> Result<RankPartition, Error> {
    load_safetensors_partition_on_streams(model_dir, plan, weights_stream, stream, strict)
}

fn load_llama(
    model_dir: &Path,
    topology: ParallelTopology,
    requested_quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<TensorParallelModel, Error> {
    let source_args = llama::get_llama_model_args(model_dir)?;
    if source_args.hidden_size <= 0 || source_args.head_dim <= 0 {
        return Err(Error::Parallel(
            "Llama hidden_size and head_dim must be positive".into(),
        ));
    }
    if source_args.hidden_size != source_args.num_attention_heads * source_args.head_dim {
        return Err(Error::Parallel(format!(
            "Llama hidden size {} does not match {} attention heads * head dimension {}",
            source_args.hidden_size, source_args.num_attention_heads, source_args.head_dim
        )));
    }
    let local_heads = exact_division(
        "Llama attention heads",
        source_args.num_attention_heads,
        topology.tensor_parallel_size,
    )?;
    let local_kv_heads = exact_division(
        "Llama KV heads",
        source_args.num_key_value_heads,
        topology.tensor_parallel_size,
    )?;
    let local_intermediate = exact_division(
        "Llama intermediate size",
        source_args.intermediate_size,
        topology.tensor_parallel_size,
    )?;
    if local_heads % local_kv_heads != 0 {
        return Err(Error::Parallel(
            "Llama local query heads must be divisible by local KV heads for GQA".into(),
        ));
    }
    let quantize_on_load = requested_quantization
        .map(|requested| {
            should_quantize_on_load(
                "Llama tensor parallel",
                source_args.weight_quantization(),
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    if let Some(quantization) = quantize_on_load.or(source_args.weight_quantization()) {
        require_alignment(
            "model.layers.*.self_attn.o_proj.weight",
            local_heads * source_args.head_dim,
            quantization.group_size(),
            topology,
        )?;
        require_alignment(
            "model.layers.*.mlp.down_proj.weight",
            local_intermediate,
            quantization.group_size(),
            topology,
        )?;
    }
    let vocabulary = balanced_contiguous_range(
        source_args.vocab_size as usize,
        topology.tensor_parallel_size,
        topology.tensor_parallel_rank,
        false,
    )?;
    let mut planner = ParallelPlanBuilder::new(topology);
    let source_embedding = linear::unloaded_maybe_quantized_embedding(
        source_args.vocab_size,
        source_args.hidden_size,
        source_args.affine_quantization_for("model.embed_tokens.weight"),
        stream,
    )
    .map_err(|error| Error::Parallel(format!("construct source Llama embedding: {error}")))?;
    register_vocabulary_module(&mut planner, &source_embedding, "model.embed_tokens")?;
    for index in 0..source_args.attention_schedule.len() {
        let layer = llama::TransformerBlock::new_for_layer(&source_args, index as i32, stream)
            .map_err(|error| {
                Error::Parallel(format!(
                    "construct source Llama tensor-parallel layer {index}: {error}"
                ))
            })?;
        insert_llama_layer_plan(&mut planner, &layer, index)?;
    }
    let source_norm = nn::RmsNorm::unloaded(
        source_args.hidden_size,
        source_args.rms_norm_eps,
        Dtype::Float32,
        stream,
    )?;
    register_replicated_module(&mut planner, &source_norm, "model.norm")?;
    if !source_args.tie_word_embeddings {
        let source_head = linear::build_unloaded_maybe_quantized_lm_head_with_quantization(
            source_args.hidden_size,
            source_args.vocab_size,
            source_args.affine_quantization_for("lm_head.weight"),
            stream,
        )?;
        register_vocabulary_module(&mut planner, &source_head, "lm_head")?;
    }
    let (plan, _local_layout) = planner.finish()?;
    let partition = load_partition(
        model_dir,
        &plan,
        weights_stream,
        stream,
        &StrictLoadConfig::default(),
    )
    .map_err(|error| Error::Parallel(format!("load Llama tensor partition: {error}")))?;
    let info = partition_info(
        &partition,
        topology,
        ModelKind::Llama,
        (
            source_args.num_attention_heads,
            source_args.num_key_value_heads,
        ),
        (local_heads, local_kv_heads),
        vocabulary.clone(),
    );
    let mut target_args = source_args.clone();
    target_args.num_attention_heads = local_heads;
    target_args.num_key_value_heads = local_kv_heads;
    target_args.intermediate_size = local_intermediate;
    if let Some(quantization) = quantize_on_load {
        target_args.quantization = Some(quantization);
        target_args.quantization_config = None;
    }
    let local_vocab = i32::try_from(vocabulary.len())
        .map_err(|_| Error::Parallel("local vocabulary does not fit in i32".into()))?;
    let mut embedding = linear::unloaded_maybe_quantized_embedding(
        local_vocab,
        target_args.hidden_size,
        target_args.affine_quantization_for("model.embed_tokens.weight"),
        stream,
    )
    .map_err(|error| Error::Parallel(format!("construct local Llama embedding: {error}")))?;
    let mut layers = (0..target_args.num_hidden_layers)
        .map(|index| {
            llama::TransformerBlock::new_for_layer(&target_args, index, stream).map_err(|error| {
                Error::Parallel(format!(
                    "construct local Llama tensor-parallel layer {index}: {error}"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut norm = nn::RmsNorm::unloaded(
        target_args.hidden_size,
        target_args.rms_norm_eps,
        Dtype::Float32,
        stream,
    )?;
    let mut lm_head = if target_args.tie_word_embeddings {
        None
    } else {
        Some(
            linear::build_unloaded_maybe_quantized_lm_head_with_quantization(
                target_args.hidden_size,
                local_vocab,
                target_args.affine_quantization_for("lm_head.weight"),
                stream,
            )?,
        )
    };
    let mut tensors = partition.into_tensors();
    assign_module(
        &mut embedding,
        "model.embed_tokens",
        &mut tensors,
        quantize_on_load,
        stream,
    )?;
    for (index, layer) in layers.iter_mut().enumerate() {
        assign_module(
            layer,
            &format!("model.layers.{index}"),
            &mut tensors,
            quantize_on_load,
            stream,
        )?;
    }
    assign_module(&mut norm, "model.norm", &mut tensors, None, stream)?;
    if let Some(head) = lm_head.as_mut() {
        assign_module(head, "lm_head", &mut tensors, quantize_on_load, stream)?;
    }
    ensure_no_unused(tensors)?;
    Ok(TensorParallelModel {
        topology,
        info,
        architecture: TensorArchitecture::Llama(LlamaTensorModel {
            global_args: source_args,
            embedding,
            layers,
            norm,
            lm_head,
        }),
    })
}

fn load_qwen(
    model_dir: &Path,
    topology: ParallelTopology,
    requested_quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<TensorParallelModel, Error> {
    let source_args = qwen::load_config(model_dir)?;
    if source_args.hidden_size <= 0 || source_args.head_dim <= 0 {
        return Err(Error::Parallel(
            "Qwen hidden_size and head_dim must be positive".into(),
        ));
    }
    if source_args.hidden_size != source_args.num_attention_heads * source_args.head_dim {
        return Err(Error::Parallel(format!(
            "Qwen hidden size {} does not match {} attention heads * head dimension {}",
            source_args.hidden_size, source_args.num_attention_heads, source_args.head_dim
        )));
    }
    let local_heads = exact_division(
        "Qwen attention heads",
        source_args.num_attention_heads,
        topology.tensor_parallel_size,
    )?;
    let local_kv_heads = exact_division(
        "Qwen KV heads",
        source_args.num_key_value_heads,
        topology.tensor_parallel_size,
    )?;
    let local_intermediate = (!source_args.is_moe())
        .then(|| {
            exact_division(
                "Qwen intermediate size",
                source_args.intermediate_size,
                topology.tensor_parallel_size,
            )
        })
        .transpose()?;
    let local_moe_intermediate = source_args
        .is_moe()
        .then(|| {
            exact_division(
                "Qwen expert intermediate size",
                source_args.moe_intermediate_size,
                topology.tensor_parallel_size,
            )
        })
        .transpose()?;
    if local_heads % local_kv_heads != 0 {
        return Err(Error::Parallel(
            "Qwen local query heads must be divisible by local KV heads for GQA".into(),
        ));
    }
    let quantize_on_load = requested_quantization
        .map(|requested| {
            should_quantize_on_load(
                "dense Qwen tensor parallel",
                source_args.weight_quantization(),
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    if let Some(quantization) = quantize_on_load.or(source_args.weight_quantization()) {
        require_alignment(
            "model.layers.*.self_attn.o_proj.weight",
            local_heads * source_args.head_dim,
            quantization.group_size(),
            topology,
        )?;
        require_alignment(
            if source_args.is_moe() {
                "model.layers.*.mlp.experts.down_proj"
            } else {
                "model.layers.*.mlp.down_proj.weight"
            },
            local_moe_intermediate
                .or(local_intermediate)
                .ok_or_else(|| {
                    Error::Parallel(
                        "Qwen layer has neither dense nor expert intermediate width".into(),
                    )
                })?,
            quantization.group_size(),
            topology,
        )?;
    }
    let vocabulary = balanced_contiguous_range(
        usize::try_from(source_args.vocab_size)
            .map_err(|_| Error::Parallel("Qwen vocabulary does not fit usize".into()))?,
        topology.tensor_parallel_size,
        topology.tensor_parallel_rank,
        false,
    )?;
    let mut planner = ParallelPlanBuilder::new(topology);
    let source_embedding = linear::unloaded_maybe_quantized_embedding(
        source_args.vocab_size,
        source_args.hidden_size,
        source_args.weight_quantization_for("model.embed_tokens.weight"),
        stream,
    )?;
    register_vocabulary_module(&mut planner, &source_embedding, "model.embed_tokens")?;
    for index in 0..usize::try_from(source_args.num_hidden_layers)
        .map_err(|_| Error::Parallel("Qwen layer count does not fit usize".into()))?
    {
        let layer = qwen::TransformerBlock::new_for_layer(&source_args, index as i32, stream)?;
        insert_qwen_layer_plan(&mut planner, &layer, index)?;
    }
    let source_norm = nn::RmsNorm::unloaded(
        source_args.hidden_size,
        source_args.rms_norm_eps,
        Dtype::Float32,
        stream,
    )?;
    register_replicated_module(&mut planner, &source_norm, "model.norm")?;
    if !source_args.tie_word_embeddings {
        let source_head = linear::build_unloaded_maybe_quantized_lm_head_with_quantization(
            source_args.hidden_size,
            source_args.vocab_size,
            source_args.weight_quantization_for("lm_head.weight"),
            stream,
        )?;
        register_vocabulary_module(&mut planner, &source_head, "lm_head")?;
    }
    let (plan, _local_layout) = planner.finish()?;
    let partition = load_partition(
        model_dir,
        &plan,
        weights_stream,
        stream,
        &StrictLoadConfig::default(),
    )
    .map_err(|error| Error::Parallel(format!("load Qwen tensor partition: {error}")))?;
    let info = partition_info(
        &partition,
        topology,
        source_args.model_kind(),
        (
            source_args.num_attention_heads,
            source_args.num_key_value_heads,
        ),
        (local_heads, local_kv_heads),
        vocabulary.clone(),
    );
    let mut target_args = source_args.clone();
    target_args.num_attention_heads = local_heads;
    target_args.num_key_value_heads = local_kv_heads;
    if let Some(local_intermediate) = local_intermediate {
        target_args.intermediate_size = local_intermediate;
    }
    if let Some(local_moe_intermediate) = local_moe_intermediate {
        target_args.moe_intermediate_size = local_moe_intermediate;
    }
    if let Some(quantization) = quantize_on_load {
        target_args.quantization = Some(quantization);
        target_args.quantization_config = None;
    }
    let local_vocab = i32::try_from(vocabulary.len())
        .map_err(|_| Error::Parallel("local Qwen vocabulary does not fit i32".into()))?;
    let mut embedding = linear::unloaded_maybe_quantized_embedding(
        local_vocab,
        target_args.hidden_size,
        target_args.weight_quantization_for("model.embed_tokens.weight"),
        stream,
    )?;
    let mut layers = (0..target_args.num_hidden_layers)
        .map(|index| qwen::TransformerBlock::new_for_layer(&target_args, index, stream))
        .collect::<Result<Vec<_>, _>>()?;
    let mut norm = nn::RmsNorm::unloaded(
        target_args.hidden_size,
        target_args.rms_norm_eps,
        Dtype::Float32,
        stream,
    )?;
    let mut lm_head = if target_args.tie_word_embeddings {
        None
    } else {
        Some(
            linear::build_unloaded_maybe_quantized_lm_head_with_quantization(
                target_args.hidden_size,
                local_vocab,
                target_args.weight_quantization_for("lm_head.weight"),
                stream,
            )?,
        )
    };
    let mut tensors = partition.into_tensors();
    assign_module(
        &mut embedding,
        "model.embed_tokens",
        &mut tensors,
        quantize_on_load,
        stream,
    )?;
    for (index, layer) in layers.iter_mut().enumerate() {
        assign_module(
            layer,
            &format!("model.layers.{index}"),
            &mut tensors,
            quantize_on_load,
            stream,
        )?;
    }
    assign_module(&mut norm, "model.norm", &mut tensors, None, stream)?;
    if let Some(head) = lm_head.as_mut() {
        assign_module(head, "lm_head", &mut tensors, quantize_on_load, stream)?;
    }
    ensure_no_unused(tensors)?;
    Ok(TensorParallelModel {
        topology,
        info,
        architecture: TensorArchitecture::Qwen(QwenTensorModel {
            global_args: source_args,
            embedding,
            layers,
            norm,
            lm_head,
        }),
    })
}

fn load_kimi(
    model_dir: &Path,
    topology: ParallelTopology,
    requested_quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<TensorParallelModel, Error> {
    let source_args = kimi_linear::get_model_args(model_dir)?;
    let local_heads = exact_division(
        "Kimi MLA attention heads",
        source_args.num_attention_heads,
        topology.tensor_parallel_size,
    )?;
    let local_kv_heads = exact_division(
        "Kimi MLA KV heads",
        source_args.num_key_value_heads,
        topology.tensor_parallel_size,
    )?;
    let local_kda_heads = exact_division(
        "Kimi KDA heads",
        source_args.kda_config.num_heads,
        topology.tensor_parallel_size,
    )?;
    let local_intermediate = exact_division(
        "Kimi dense intermediate size",
        source_args.intermediate_size,
        topology.tensor_parallel_size,
    )?;
    let local_moe_intermediate = exact_division(
        "Kimi expert intermediate size",
        source_args.moe_intermediate_size,
        topology.tensor_parallel_size,
    )?;
    let quantize_on_load = requested_quantization
        .map(|requested| {
            should_quantize_on_load(
                "Kimi Linear tensor parallel",
                source_args.quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    if let Some(quantization) = quantize_on_load.or(source_args.quantization) {
        for (name, dimension) in [
            (
                "model.layers.*.self_attn.o_proj.weight",
                local_heads * source_args.v_head_dim,
            ),
            (
                "model.layers.*.self_attn.kda.o_proj.weight",
                local_kda_heads * source_args.kda_config.head_dim,
            ),
            ("model.layers.*.mlp.down_proj.weight", local_intermediate),
            (
                "model.layers.*.mlp.experts.down_proj",
                local_moe_intermediate,
            ),
        ] {
            require_alignment(name, dimension, quantization.group_size(), topology)?;
        }
    }
    let vocabulary = balanced_contiguous_range(
        usize::try_from(source_args.vocab_size)
            .map_err(|_| Error::Parallel("Kimi vocabulary does not fit usize".into()))?,
        topology.tensor_parallel_size,
        topology.tensor_parallel_rank,
        false,
    )?;
    let split_experts = source_args.quantization.is_none();
    let source_model = kimi_linear::Model::new(source_args.clone(), stream)?;
    let mut planner = ParallelPlanBuilder::new(topology);
    register_vocabulary_module(
        &mut planner,
        &source_model.model.embed_tokens,
        "model.embed_tokens",
    )?;
    for (index, layer) in source_model.model.layers.iter().enumerate() {
        insert_kimi_layer_plan(&mut planner, layer, index, split_experts)?;
    }
    register_replicated_module(&mut planner, &source_model.model.norm, "model.norm")?;
    if let Some(head) = source_model.lm_head.as_ref() {
        register_vocabulary_module(&mut planner, head, "lm_head")?;
    }
    let (plan, _local_layout) = planner.finish()?;
    let partition = load_partition(
        model_dir,
        &plan,
        weights_stream,
        stream,
        &StrictLoadConfig::default(),
    )
    .map_err(|error| Error::Parallel(format!("load Kimi tensor partition: {error}")))?;
    let info = partition_info(
        &partition,
        topology,
        ModelKind::KimiLinear,
        (
            source_args.num_attention_heads,
            source_args.num_key_value_heads,
        ),
        (local_heads, local_kv_heads),
        vocabulary.clone(),
    );
    let mut target_args = source_args.clone();
    target_args.num_attention_heads = local_heads;
    target_args.num_key_value_heads = local_kv_heads;
    target_args.kda_config.num_heads = local_kda_heads;
    target_args.intermediate_size = local_intermediate;
    target_args.moe_intermediate_size = local_moe_intermediate;
    if let Some(quantization) = quantize_on_load {
        target_args.quantization = Some(quantization);
        target_args.quantized_weight_configs = None;
    }
    let local_vocab = i32::try_from(vocabulary.len())
        .map_err(|_| Error::Parallel("local Kimi vocabulary does not fit i32".into()))?;
    let global_vocab = target_args.vocab_size;
    target_args.vocab_size = local_vocab;
    let mut model = kimi_linear::Model::new(target_args.clone(), stream)?;
    model.args.vocab_size = global_vocab;
    let mut transformed = HashMap::new();
    for (name, value) in partition.into_tensors() {
        for (name, value) in
            kimi_linear::transform_safetensors_weight(&target_args, name, value, stream)?
        {
            transformed.insert(name, value);
        }
    }
    let mut tensors = if split_experts {
        transform_split_swiglu_experts(transformed, source_args.num_experts, stream)?
    } else {
        transformed
    };
    assign_module(
        &mut model.model.embed_tokens,
        "model.embed_tokens",
        &mut tensors,
        quantize_on_load,
        stream,
    )?;
    for (index, layer) in model.model.layers.iter_mut().enumerate() {
        assign_module(
            layer,
            &format!("model.layers.{index}"),
            &mut tensors,
            quantize_on_load,
            stream,
        )?;
    }
    assign_module(
        &mut model.model.norm,
        "model.norm",
        &mut tensors,
        None,
        stream,
    )?;
    if let Some(head) = model.lm_head.as_mut() {
        assign_module(head, "lm_head", &mut tensors, quantize_on_load, stream)?;
    }
    ensure_no_unused(tensors)?;
    Ok(TensorParallelModel {
        topology,
        info,
        architecture: TensorArchitecture::Kimi(Box::new(KimiTensorModel {
            global_args: source_args,
            model,
        })),
    })
}

fn load_lfm2(
    model_dir: &Path,
    topology: ParallelTopology,
    requested_quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<TensorParallelModel, Error> {
    let source_args = lfm2::get_model_args(model_dir)?;
    let local_heads = exact_division(
        "LFM2 attention heads",
        source_args.num_attention_heads,
        topology.tensor_parallel_size,
    )?;
    let local_kv_heads = exact_division(
        "LFM2 KV heads",
        source_args.num_key_value_heads,
        topology.tensor_parallel_size,
    )?;
    let local_dense_intermediate = exact_division(
        "LFM2 dense intermediate size",
        source_args.dense_layer_intermediate_size(),
        topology.tensor_parallel_size,
    )?;
    let local_moe_intermediate = source_args
        .has_sparse_moe_layers()
        .then(|| {
            exact_division(
                "LFM2 expert intermediate size",
                source_args.moe_intermediate_size,
                topology.tensor_parallel_size,
            )
        })
        .transpose()?;
    let source_quantization = source_args.quantization.or(source_args.quantization_config);
    let quantize_on_load = requested_quantization
        .map(|requested| {
            should_quantize_on_load("LFM2 tensor parallel", source_quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    if let Some(quantization) = quantize_on_load.or(source_quantization) {
        for (name, dimension) in [
            (
                "model.layers.*.self_attn.out_proj.weight",
                local_heads * (source_args.hidden_size / source_args.num_attention_heads),
            ),
            (
                "model.layers.*.feed_forward.w2.weight",
                local_dense_intermediate,
            ),
            (
                "model.layers.*.feed_forward.experts.down_proj",
                local_moe_intermediate.unwrap_or(local_dense_intermediate),
            ),
        ] {
            require_alignment(name, dimension, quantization.group_size(), topology)?;
        }
    }
    let vocabulary = balanced_contiguous_range(
        usize::try_from(source_args.vocab_size)
            .map_err(|_| Error::Parallel("LFM2 vocabulary does not fit usize".into()))?,
        topology.tensor_parallel_size,
        topology.tensor_parallel_rank,
        false,
    )?;
    let split_experts = source_args.has_sparse_moe_layers() && source_quantization.is_none();
    let source_model = lfm2::Model::new(source_args.clone(), stream)?;
    let mut planner = ParallelPlanBuilder::new(topology);
    register_vocabulary_module(
        &mut planner,
        &source_model.model.embed_tokens,
        "model.embed_tokens",
    )?;
    for (index, layer) in source_model.model.layers.iter().enumerate() {
        insert_lfm2_layer_plan(&mut planner, layer, index, split_experts)?;
    }
    register_replicated_module(
        &mut planner,
        &source_model.model.embedding_norm,
        "model.embedding_norm",
    )?;
    if let Some(head) = source_model.lm_head.as_ref() {
        register_vocabulary_module(&mut planner, head, "lm_head")?;
    }
    let (plan, _local_layout) = planner.finish()?;
    let partition = load_partition(
        model_dir,
        &plan,
        weights_stream,
        stream,
        &StrictLoadConfig::default(),
    )
    .map_err(|error| Error::Parallel(format!("load LFM2 tensor partition: {error}")))?;
    let info = partition_info(
        &partition,
        topology,
        ModelKind::Lfm2,
        (
            source_args.num_attention_heads,
            source_args.num_key_value_heads,
        ),
        (local_heads, local_kv_heads),
        vocabulary.clone(),
    );
    let local_vocab = i32::try_from(vocabulary.len())
        .map_err(|_| Error::Parallel("local LFM2 vocabulary does not fit i32".into()))?;
    let mut model = lfm2::Model::new_tensor_parallel(
        source_args.clone(),
        lfm2::TensorParallelGeometry {
            vocab_size: local_vocab,
            attention_heads: local_heads,
            kv_heads: local_kv_heads,
            dense_intermediate: local_dense_intermediate,
            moe_intermediate: local_moe_intermediate.unwrap_or(source_args.moe_intermediate_size),
        },
        quantize_on_load,
        stream,
    )?;
    let mut tensors = partition.into_tensors();
    if split_experts {
        tensors = transform_split_swiglu_experts(tensors, source_args.num_experts, stream)?;
    }
    assign_module(
        &mut model.model.embed_tokens,
        "model.embed_tokens",
        &mut tensors,
        quantize_on_load,
        stream,
    )?;
    for (index, layer) in model.model.layers.iter_mut().enumerate() {
        assign_module(
            layer,
            &format!("model.layers.{index}"),
            &mut tensors,
            quantize_on_load,
            stream,
        )?;
    }
    assign_module(
        &mut model.model.embedding_norm,
        "model.embedding_norm",
        &mut tensors,
        None,
        stream,
    )?;
    if let Some(head) = model.lm_head.as_mut() {
        assign_module(head, "lm_head", &mut tensors, quantize_on_load, stream)?;
    }
    ensure_no_unused(tensors)?;
    Ok(TensorParallelModel {
        topology,
        info,
        architecture: TensorArchitecture::Lfm2(Box::new(Lfm2TensorModel {
            global_args: source_args,
            model,
        })),
    })
}

fn load_qwen_hybrid(
    model_dir: &Path,
    topology: ParallelTopology,
    requested_quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<TensorParallelModel, Error> {
    let (source_args, image_token_id, video_token_id, vision_config) =
        qwen_hybrid::get_qwen3_5_moe_model_args(model_dir)?;
    if vision_config.is_some() {
        return Err(Error::UnsupportedArchitecture(
            "tensor-parallel Qwen3.5 currently supports the text architecture, not the vision tower"
                .into(),
        ));
    }
    if source_args.mtp_num_hidden_layers > 0 {
        return Err(Error::UnsupportedArchitecture(
            "tensor-parallel Qwen hybrid loading does not yet execute embedded MTP layers".into(),
        ));
    }
    if requested_quantization.is_some() && source_args.uses_fp8() {
        return Err(Error::Quantization(
            "native Qwen FP8 tensor-parallel weights cannot be implicitly requantized".into(),
        ));
    }
    let local_heads = exact_division(
        "Qwen hybrid attention heads",
        source_args.num_attention_heads,
        topology.tensor_parallel_size,
    )?;
    let local_kv_heads = exact_division(
        "Qwen hybrid KV heads",
        source_args.num_key_value_heads,
        topology.tensor_parallel_size,
    )?;
    let local_linear_key_heads = exact_division(
        "Qwen recurrent key heads",
        source_args.linear_num_key_heads,
        topology.tensor_parallel_size,
    )?;
    let local_linear_value_heads = exact_division(
        "Qwen recurrent value heads",
        source_args.linear_num_value_heads,
        topology.tensor_parallel_size,
    )?;
    if local_linear_value_heads % local_linear_key_heads != 0 {
        return Err(Error::Parallel(
            "Qwen recurrent local value heads must be divisible by local key heads".into(),
        ));
    }
    if local_heads % local_kv_heads != 0 {
        return Err(Error::Parallel(
            "Qwen hybrid local query heads must be divisible by local KV heads".into(),
        ));
    }
    let local_intermediate = (!source_args.is_moe())
        .then(|| {
            exact_division(
                "Qwen hybrid dense intermediate size",
                source_args.intermediate_size,
                topology.tensor_parallel_size,
            )
        })
        .transpose()?;
    let local_moe_intermediate = source_args
        .is_moe()
        .then(|| {
            exact_division(
                "Qwen hybrid routed expert intermediate size",
                source_args.moe_intermediate_size,
                topology.tensor_parallel_size,
            )
        })
        .transpose()?;
    let local_shared_intermediate = source_args
        .is_moe()
        .then(|| {
            exact_division(
                "Qwen hybrid shared expert intermediate size",
                source_args.shared_expert_intermediate_size,
                topology.tensor_parallel_size,
            )
        })
        .transpose()?;
    let quantize_on_load = requested_quantization
        .map(|requested| {
            should_quantize_on_load(
                "Qwen hybrid tensor parallel",
                source_args.quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let row_alignment = if source_args.uses_fp8() {
        Some(128)
    } else {
        quantize_on_load
            .or(source_args.quantization)
            .map(WeightQuantization::group_size)
    };
    if let Some(alignment) = row_alignment {
        for (name, dimension) in [
            (
                "model.layers.*.self_attn.o_proj.weight",
                local_heads * source_args.head_dim,
            ),
            (
                "model.layers.*.linear_attn.out_proj.weight",
                local_linear_value_heads * source_args.linear_value_head_dim,
            ),
            (
                "model.layers.*.mlp.down_proj.weight",
                local_intermediate.unwrap_or(0),
            ),
            (
                "model.layers.*.mlp.experts.down_proj",
                local_moe_intermediate.unwrap_or(0),
            ),
            (
                "model.layers.*.mlp.shared_expert.down_proj.weight",
                local_shared_intermediate.unwrap_or(0),
            ),
        ] {
            if dimension > 0 {
                require_alignment(name, dimension, alignment, topology)?;
            }
        }
    }
    let vocabulary = balanced_contiguous_range(
        usize::try_from(source_args.vocab_size)
            .map_err(|_| Error::Parallel("Qwen hybrid vocabulary does not fit usize".into()))?,
        topology.tensor_parallel_size,
        topology.tensor_parallel_rank,
        false,
    )?;
    let split_experts =
        source_args.is_moe() && !source_args.uses_fp8() && source_args.quantization.is_none();
    let source_model = qwen_hybrid::Model::new(
        source_args.clone(),
        image_token_id,
        video_token_id,
        None,
        stream,
    )?;
    let mut planner = ParallelPlanBuilder::new(topology);
    register_vocabulary_module(
        &mut planner,
        &source_model.model.embed_tokens,
        "model.embed_tokens",
    )?;
    for (index, layer) in source_model.model.layers.iter().enumerate() {
        insert_qwen_hybrid_layer_plan(&mut planner, layer, index, &source_args, split_experts)?;
    }
    register_replicated_module(&mut planner, &source_model.model.norm, "model.norm")?;
    if let Some(head) = source_model.lm_head.as_ref() {
        register_vocabulary_module(&mut planner, head, "lm_head")?;
    }
    let (plan, _local_layout) = planner.finish()?;
    let strict = qwen_hybrid::qwen3_5_moe_strict_load_config(false)
        .allow_unused_prefix("mtp.")
        .allow_unused_prefix("model.mtp.");
    let partition = load_partition(model_dir, &plan, weights_stream, stream, &strict)
        .map_err(|error| Error::Parallel(format!("load Qwen hybrid tensor partition: {error}")))?;
    let kind = if source_args.model_type == "qwen3_next" {
        ModelKind::Qwen3Next
    } else {
        ModelKind::Qwen35Moe
    };
    let info = partition_info(
        &partition,
        topology,
        kind,
        (
            source_args.num_attention_heads,
            source_args.num_key_value_heads,
        ),
        (local_heads, local_kv_heads),
        vocabulary.clone(),
    );
    let mut target_args = source_args.clone();
    target_args.vocab_size = i32::try_from(vocabulary.len())
        .map_err(|_| Error::Parallel("local Qwen hybrid vocabulary does not fit i32".into()))?;
    target_args.num_attention_heads = local_heads;
    target_args.num_key_value_heads = local_kv_heads;
    target_args.linear_num_key_heads = local_linear_key_heads;
    target_args.linear_num_value_heads = local_linear_value_heads;
    if let Some(width) = local_intermediate {
        target_args.intermediate_size = width;
    }
    if let Some(width) = local_moe_intermediate {
        target_args.moe_intermediate_size = width;
    }
    if let Some(width) = local_shared_intermediate {
        target_args.shared_expert_intermediate_size = width;
    }
    if let Some(quantization) = quantize_on_load {
        target_args.quantization = Some(quantization);
        target_args.quantized_weight_configs = None;
    }
    let mut model = qwen_hybrid::Model::new(
        target_args.clone(),
        image_token_id,
        video_token_id,
        None,
        stream,
    )?;
    model.args = source_args.clone();
    let mut transformed = HashMap::new();
    for (name, value) in partition.into_tensors() {
        let values = if source_args.model_type == "qwen3_next" {
            crate::architectures::qwen::hybrid::qwen3_next::split_fused_projection(
                &name,
                value,
                &target_args,
                stream,
            )?
        } else {
            vec![(name, value)]
        };
        transformed.extend(values);
    }
    let mut tensors = if source_args.uses_fp8() && source_args.is_moe() {
        qwen_hybrid::transform_split_qwen_fp8_experts(transformed, source_args.num_experts, stream)?
    } else if split_experts {
        transform_split_swiglu_experts(transformed, source_args.num_experts, stream)?
    } else {
        transformed
    };
    assign_module(
        &mut model.model.embed_tokens,
        "model.embed_tokens",
        &mut tensors,
        quantize_on_load,
        stream,
    )?;
    for (index, layer) in model.model.layers.iter_mut().enumerate() {
        assign_module(
            layer,
            &format!("model.layers.{index}"),
            &mut tensors,
            quantize_on_load,
            stream,
        )?;
    }
    assign_module(
        &mut model.model.norm,
        "model.norm",
        &mut tensors,
        None,
        stream,
    )?;
    if let Some(head) = model.lm_head.as_mut() {
        assign_module(head, "lm_head", &mut tensors, quantize_on_load, stream)?;
    }
    ensure_no_unused(tensors)?;
    Ok(TensorParallelModel {
        topology,
        info,
        architecture: TensorArchitecture::QwenHybrid(Box::new(QwenHybridTensorModel {
            global_args: source_args,
            model,
        })),
    })
}

fn load_gemma(
    model_dir: &Path,
    config: &serde_json::Value,
    topology: ParallelTopology,
    requested_quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<TensorParallelModel, Error> {
    let source_args = if config.get("text_config").is_some() {
        gemma4::get_gemma4_model_args(model_dir)?
    } else {
        gemma4::model_args_from_config_value(config)?
    };
    if source_args
        .layer_schedule
        .iter()
        .any(|policy| policy.feed_forward != gemma4::FeedForwardPolicy::Dense)
    {
        return Err(Error::UnsupportedArchitecture(
            "Gemma MoE uses the execution-group tensor-parallel loader".into(),
        ));
    }
    let local_heads = exact_division(
        "Gemma attention heads",
        source_args.num_attention_heads,
        topology.tensor_parallel_size,
    )?;
    let local_moe_intermediate = source_args
        .moe_intermediate_size
        .map(|width| {
            exact_division(
                "Gemma expert intermediate size",
                width,
                topology.tensor_parallel_size,
            )
        })
        .transpose()?;
    let mut local_policies = Vec::with_capacity(source_args.layer_schedule.len());
    for (index, policy) in source_args.layer_schedule.iter().copied().enumerate() {
        let local_kv_heads = exact_division(
            &format!("Gemma layer {index} KV heads"),
            i32::try_from(policy.num_key_value_heads.get())
                .map_err(|_| Error::Parallel("Gemma KV heads exceed i32".into()))?,
            topology.tensor_parallel_size,
        )?;
        let local_intermediate = exact_division(
            &format!("Gemma layer {index} intermediate size"),
            i32::try_from(policy.intermediate_size.get())
                .map_err(|_| Error::Parallel("Gemma intermediate size exceeds i32".into()))?,
            topology.tensor_parallel_size,
        )?;
        local_policies.push(gemma4::LayerPolicy {
            num_key_value_heads: NonZeroU32::new(
                u32::try_from(local_kv_heads)
                    .map_err(|_| Error::Parallel("local Gemma KV heads exceed u32".into()))?,
            )
            .ok_or_else(|| Error::Parallel("local Gemma KV heads must be positive".into()))?,
            intermediate_size: NonZeroU32::new(u32::try_from(local_intermediate).map_err(
                |_| Error::Parallel("local Gemma intermediate size exceeds u32".into()),
            )?)
            .ok_or_else(|| {
                Error::Parallel("local Gemma intermediate size must be positive".into())
            })?,
            ..policy
        });
    }
    let quantize_on_load = requested_quantization
        .map(|requested| {
            should_quantize_on_load(
                "Gemma text tensor parallel",
                source_args.weight_quantization(),
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    if let Some(quantization) = quantize_on_load.or(source_args.weight_quantization()) {
        for (index, policy) in local_policies.iter().enumerate() {
            require_alignment(
                &format!("model.language_model.layers.{index}.self_attn.o_proj.weight"),
                local_heads
                    * i32::try_from(policy.head_dim.get())
                        .map_err(|_| Error::Parallel("Gemma head dimension exceeds i32".into()))?,
                quantization.group_size(),
                topology,
            )?;
            require_alignment(
                &format!("model.language_model.layers.{index}.mlp.down_proj.weight"),
                i32::try_from(policy.intermediate_size.get()).map_err(|_| {
                    Error::Parallel("local Gemma intermediate size exceeds i32".into())
                })?,
                quantization.group_size(),
                topology,
            )?;
        }
    }
    let vocabulary = balanced_contiguous_range(
        usize::try_from(source_args.vocab_size)
            .map_err(|_| Error::Parallel("Gemma vocabulary does not fit usize".into()))?,
        topology.tensor_parallel_size,
        topology.tensor_parallel_rank,
        false,
    )?;
    let per_layer_vocabulary = source_args
        .vocab_size_per_layer_input
        .map(|global| {
            balanced_contiguous_range(
                global as usize,
                topology.tensor_parallel_size,
                topology.tensor_parallel_rank,
                false,
            )
        })
        .transpose()?;
    let source_text = gemma4::Gemma4TextModel::new(&source_args, stream)?;
    let mut planner = ParallelPlanBuilder::new(topology);
    register_vocabulary_module(
        &mut planner,
        &source_text.embed_tokens,
        "model.language_model.embed_tokens",
    )?;
    if let Some(embedding) = &source_text.embed_tokens_per_layer {
        register_vocabulary_module(
            &mut planner,
            embedding,
            "model.language_model.embed_tokens_per_layer",
        )?;
    }
    if let Some(projection) = &source_text.per_layer_model_projection {
        register_projection(
            &mut planner,
            projection,
            "model.language_model.per_layer_model_projection",
            ProjectionPlacement::Replicated,
        )?;
    }
    if let Some(norm) = &source_text.per_layer_projection_norm {
        register_replicated_module(
            &mut planner,
            norm,
            "model.language_model.per_layer_projection_norm",
        )?;
    }
    for (index, layer) in source_text.layers.iter().enumerate() {
        insert_gemma_layer_plan(&mut planner, layer, index)?;
    }
    register_replicated_module(&mut planner, &source_text.norm, "model.language_model.norm")?;
    let source_head = if source_args.tie_word_embeddings {
        None
    } else {
        Some(linear::unloaded_maybe_quantized_linear(
            source_args.hidden_size,
            source_args.vocab_size,
            false,
            source_args.quantization_for("lm_head.weight"),
            stream,
        )?)
    };
    if let Some(head) = &source_head {
        register_vocabulary_module(&mut planner, head, "lm_head")?;
    }
    let (plan, _local_layout) = planner.finish()?;
    let strict = StrictLoadConfig::default()
        .rewrite_prefix("language_model.model.", "model.language_model.")
        .rewrite_prefix("model.language_model.", "model.language_model.")
        .allow_unused_prefix("vision_tower.")
        .allow_unused_prefix("model.vision_tower.")
        .allow_unused_prefix("embed_vision.")
        .allow_unused_prefix("model.embed_vision.")
        .allow_unused_prefix("audio_tower.")
        .allow_unused_prefix("model.audio_tower.")
        .allow_unused_prefix("embed_audio.")
        .allow_unused_prefix("model.embed_audio.")
        .allow_unused_prefix("multi_modal_projector.")
        .allow_unused_prefix("model.multi_modal_projector.")
        .allow_unused_prefix("model.vision_embedder.")
        .allow_missing_suffix(".bias");
    let partition = load_partition(model_dir, &plan, weights_stream, stream, &strict)
        .map_err(|error| Error::Parallel(format!("load Gemma text tensor partition: {error}")))?;
    let representative_kv_heads = source_args
        .layer_schedule
        .iter()
        .find(|policy| policy.key_value.owns_state())
        .map(|policy| i32::try_from(policy.num_key_value_heads.get()))
        .transpose()
        .map_err(|_| Error::Parallel("Gemma KV heads exceed i32".into()))?
        .unwrap_or(source_args.num_attention_heads);
    let local_representative_kv_heads = exact_division(
        "Gemma representative KV heads",
        representative_kv_heads,
        topology.tensor_parallel_size,
    )?;
    let info = partition_info(
        &partition,
        topology,
        ModelKind::Gemma4,
        (source_args.num_attention_heads, representative_kv_heads),
        (local_heads, local_representative_kv_heads),
        vocabulary.clone(),
    );
    let mut target_args = source_args.clone();
    target_args.num_attention_heads = local_heads;
    target_args.moe_intermediate_size = local_moe_intermediate;
    target_args.vocab_size = i32::try_from(vocabulary.len())
        .map_err(|_| Error::Parallel("local Gemma vocabulary does not fit i32".into()))?;
    if let Some(range) = &per_layer_vocabulary {
        target_args.vocab_size_per_layer_input = Some(range.len() as i32);
    }
    target_args.layer_schedule = LayerSchedule::new(local_policies.len(), local_policies)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    if let Some(quantization) = quantize_on_load {
        target_args.quantized = true;
        target_args.weight_quantization = Some(quantization);
        target_args.quantization_group_size = quantization.group_size();
        target_args.quantization_bits = quantization.bits();
    }
    let mut text = gemma4::Gemma4TextModel::new(&target_args, stream)?;
    let mut lm_head = if target_args.tie_word_embeddings {
        None
    } else {
        Some(linear::unloaded_maybe_quantized_linear(
            target_args.hidden_size,
            target_args.vocab_size,
            false,
            target_args.quantization_for("lm_head.weight"),
            stream,
        )?)
    };
    let mut tensors = partition.into_tensors();
    assign_module(
        &mut text,
        "model.language_model",
        &mut tensors,
        quantize_on_load,
        stream,
    )?;
    if let Some(head) = lm_head.as_mut() {
        assign_module(head, "lm_head", &mut tensors, quantize_on_load, stream)?;
    }
    ensure_no_unused(tensors)?;
    Ok(TensorParallelModel {
        topology,
        info,
        architecture: TensorArchitecture::Gemma(Box::new(GemmaTensorModel {
            global_args: source_args,
            text,
            lm_head,
            per_layer_vocabulary_range: per_layer_vocabulary,
        })),
    })
}

fn load_gpt_oss(
    model_dir: &Path,
    topology: ParallelTopology,
    requested_quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<TensorParallelModel, Error> {
    let source_args = gpt_oss::get_model_args(model_dir)?;
    if requested_quantization.is_some_and(|quantization| quantization != WeightQuantization::MxFp4)
    {
        return Err(Error::Quantization(
            "GPT-OSS tensor parallelism only accepts MXFP4 load-time quantization".into(),
        ));
    }
    let local_heads = exact_division(
        "GPT-OSS attention heads",
        source_args.num_attention_heads,
        topology.tensor_parallel_size,
    )?;
    let local_kv_heads = exact_division(
        "GPT-OSS KV heads",
        source_args.num_key_value_heads,
        topology.tensor_parallel_size,
    )?;
    let local_intermediate = exact_division(
        "GPT-OSS expert intermediate size",
        source_args.intermediate_size,
        topology.tensor_parallel_size,
    )?;
    if local_intermediate % 32 != 0 {
        return Err(Error::Parallel(format!(
            "GPT-OSS local expert width {local_intermediate} must remain aligned to MXFP4 group size 32"
        )));
    }
    let vocabulary = balanced_contiguous_range(
        usize::try_from(source_args.vocab_size)
            .map_err(|_| Error::Parallel("GPT-OSS vocabulary does not fit usize".into()))?,
        topology.tensor_parallel_size,
        topology.tensor_parallel_rank,
        false,
    )?;
    let source_model = gpt_oss::Model::new(source_args.clone(), stream)?;
    let mut planner = ParallelPlanBuilder::new(topology);
    register_vocabulary_module(
        &mut planner,
        &source_model.model.embed_tokens,
        "model.embed_tokens",
    )?;
    for (index, layer) in source_model.model.layers.iter().enumerate() {
        insert_gpt_oss_layer_plan(&mut planner, layer, index)?;
    }
    register_replicated_module(&mut planner, &source_model.model.norm, "model.norm")?;
    register_vocabulary_module(&mut planner, &source_model.lm_head, "lm_head")?;
    let (plan, _) = planner.finish()?;
    let partition = load_partition(
        model_dir,
        &plan,
        weights_stream,
        stream,
        &StrictLoadConfig::default(),
    )
    .map_err(|error| Error::Parallel(format!("load GPT-OSS tensor partition: {error}")))?;
    let info = partition_info(
        &partition,
        topology,
        ModelKind::GptOss,
        (
            source_args.num_attention_heads,
            source_args.num_key_value_heads,
        ),
        (local_heads, local_kv_heads),
        vocabulary.clone(),
    );
    let mut target_args = source_args.clone();
    target_args.num_attention_heads = local_heads;
    target_args.num_key_value_heads = local_kv_heads;
    target_args.intermediate_size = local_intermediate;
    target_args.vocab_size = i32::try_from(vocabulary.len())
        .map_err(|_| Error::Parallel("local GPT-OSS vocabulary does not fit i32".into()))?;
    if let Some(quantization) = requested_quantization {
        target_args.quantization = Some(quantization);
    }
    let mut model = gpt_oss::Model::new(target_args, stream)?;
    let mut tensors = partition.into_tensors();
    assign_module(
        &mut model.model.embed_tokens,
        "model.embed_tokens",
        &mut tensors,
        requested_quantization,
        stream,
    )?;
    for (index, layer) in model.model.layers.iter_mut().enumerate() {
        assign_module(
            layer,
            &format!("model.layers.{index}"),
            &mut tensors,
            requested_quantization,
            stream,
        )?;
    }
    assign_module(
        &mut model.model.norm,
        "model.norm",
        &mut tensors,
        None,
        stream,
    )?;
    assign_module(
        &mut model.lm_head,
        "lm_head",
        &mut tensors,
        requested_quantization,
        stream,
    )?;
    ensure_no_unused(tensors)?;
    Ok(TensorParallelModel {
        topology,
        info,
        architecture: TensorArchitecture::GptOss(Box::new(GptOssTensorModel {
            global_args: source_args,
            model,
        })),
    })
}

fn load_nemotron_h(
    model_dir: &Path,
    topology: ParallelTopology,
    requested_quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<TensorParallelModel, Error> {
    let source_args = nemotron_h::get_nemotron_h_model_args(model_dir)?;
    if requested_quantization
        .is_some_and(|quantization| !matches!(quantization, WeightQuantization::Affine(_)))
    {
        return Err(Error::Quantization(
            "Nemotron-H tensor parallelism supports dense or MLX-affine safetensors weights".into(),
        ));
    }
    let local_attention_heads = exact_division(
        "Nemotron-H attention heads",
        source_args.num_attention_heads,
        topology.tensor_parallel_size,
    )?;
    let local_kv_heads = exact_division(
        "Nemotron-H KV heads",
        source_args.num_key_value_heads,
        topology.tensor_parallel_size,
    )?;
    if local_attention_heads % local_kv_heads != 0 {
        return Err(Error::Parallel(
            "Nemotron-H local query heads must be divisible by local KV heads".into(),
        ));
    }
    let local_mamba_heads = exact_division(
        "Nemotron-H Mamba heads",
        source_args.mamba_num_heads,
        topology.tensor_parallel_size,
    )?;
    let local_mamba_groups = exact_division(
        "Nemotron-H Mamba B/C groups",
        source_args.n_groups,
        topology.tensor_parallel_size,
    )?;
    if local_mamba_heads % local_mamba_groups != 0 {
        return Err(Error::Parallel(
            "Nemotron-H local Mamba heads must be divisible by local B/C groups".into(),
        ));
    }
    let local_dense_intermediate = exact_division(
        "Nemotron-H dense intermediate size",
        source_args.intermediate_size,
        topology.tensor_parallel_size,
    )?;
    let local_moe_intermediate = exact_division(
        "Nemotron-H routed expert intermediate size",
        source_args.moe_intermediate_size,
        topology.tensor_parallel_size,
    )?;
    let local_shared_intermediate = exact_division(
        "Nemotron-H shared expert intermediate size",
        source_args.moe_shared_expert_intermediate_size,
        topology.tensor_parallel_size,
    )?;
    let source_quantization = source_args.quantization.map(WeightQuantization::Affine);
    let quantize_on_load = requested_quantization
        .map(|requested| {
            should_quantize_on_load("Nemotron-H tensor parallel", source_quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    if let Some(quantization) = quantize_on_load.or(source_quantization) {
        let local_mamba_width = local_mamba_heads * source_args.mamba_head_dim;
        for (name, width) in [
            ("model.layers.*.mamba.out_proj.weight", local_mamba_width),
            (
                "model.layers.*.attention.o_proj.weight",
                local_attention_heads * source_args.head_dim,
            ),
            (
                "model.layers.*.mlp.down_proj.weight",
                local_dense_intermediate,
            ),
            (
                "model.layers.*.moe.experts.down_proj",
                local_moe_intermediate,
            ),
            (
                "model.layers.*.moe.shared_experts.down_proj.weight",
                local_shared_intermediate,
            ),
        ] {
            require_alignment(name, width, quantization.group_size(), topology)?;
        }
    }
    let vocabulary = balanced_contiguous_range(
        usize::try_from(source_args.vocab_size)
            .map_err(|_| Error::Parallel("Nemotron-H vocabulary does not fit usize".into()))?,
        topology.tensor_parallel_size,
        topology.tensor_parallel_rank,
        false,
    )?;
    let catalog = SafetensorsWeightStore::open(model_dir)?;
    let split_experts = catalog.keys().iter().any(|key| key.contains(".experts.0."));
    let source_model = nemotron_h::Model::new(source_args.clone(), stream)?;
    let mut planner = ParallelPlanBuilder::new(topology);
    register_vocabulary_module(
        &mut planner,
        &source_model.model.embeddings,
        "model.embeddings",
    )?;
    for (index, layer) in source_model.model.layers.iter().enumerate() {
        insert_nemotron_layer_plan(&mut planner, layer, index, split_experts)?;
    }
    register_replicated_module(&mut planner, &source_model.model.norm_f, "model.norm_f")?;
    if let Some(head) = source_model.lm_head.as_ref() {
        register_vocabulary_module(&mut planner, head, "lm_head")?;
    }
    let (plan, _) = planner.finish()?;
    let mut strict = nemotron_h::nemotron_h_strict_load_config();
    for (index, policy) in source_args.layer_schedule.iter().enumerate() {
        let field = match policy {
            nemotron_h::LayerPolicy::Mamba => "mamba",
            nemotron_h::LayerPolicy::SelfAttention(_) => "attention",
            nemotron_h::LayerPolicy::DenseMlp => "mlp",
            nemotron_h::LayerPolicy::SparseMoe => "moe",
        };
        strict = strict
            .rewrite_prefix(
                format!("backbone.layers.{index}.mixer."),
                format!("model.layers.{index}.{field}."),
            )
            .rewrite_prefix(
                format!("model.backbone.layers.{index}.mixer."),
                format!("model.layers.{index}.{field}."),
            );
    }
    let partition = load_partition(model_dir, &plan, weights_stream, stream, &strict)
        .map_err(|error| Error::Parallel(format!("load Nemotron-H tensor partition: {error}")))?;
    let info = partition_info(
        &partition,
        topology,
        ModelKind::NemotronH,
        (
            source_args.num_attention_heads,
            source_args.num_key_value_heads,
        ),
        (local_attention_heads, local_kv_heads),
        vocabulary.clone(),
    );
    let mut target_args = source_args.clone();
    target_args.vocab_size = i32::try_from(vocabulary.len())
        .map_err(|_| Error::Parallel("local Nemotron-H vocabulary does not fit i32".into()))?;
    target_args.num_attention_heads = local_attention_heads;
    target_args.num_key_value_heads = local_kv_heads;
    target_args.mamba_num_heads = local_mamba_heads;
    target_args.n_groups = local_mamba_groups;
    target_args.intermediate_size = local_dense_intermediate;
    target_args.moe_intermediate_size = local_moe_intermediate;
    target_args.moe_shared_expert_intermediate_size = local_shared_intermediate;
    if let Some(WeightQuantization::Affine(quantization)) = quantize_on_load {
        target_args.quantization = Some(quantization);
        target_args.quantized_weight_configs = None;
        target_args.quantized_weights = None;
    }
    let mut model = nemotron_h::Model::new(target_args, stream)?;
    let mut tensors = partition.into_tensors();
    if split_experts {
        tensors = transform_split_relu2_experts(tensors, source_args.n_routed_experts, stream)?;
    }
    assign_module(
        &mut model.model.embeddings,
        "model.embeddings",
        &mut tensors,
        quantize_on_load,
        stream,
    )?;
    for (index, layer) in model.model.layers.iter_mut().enumerate() {
        assign_module(
            layer,
            &format!("model.layers.{index}"),
            &mut tensors,
            quantize_on_load,
            stream,
        )?;
    }
    assign_module(
        &mut model.model.norm_f,
        "model.norm_f",
        &mut tensors,
        None,
        stream,
    )?;
    if let Some(head) = model.lm_head.as_mut() {
        assign_module(head, "lm_head", &mut tensors, quantize_on_load, stream)?;
    }
    ensure_no_unused(tensors)?;
    Ok(TensorParallelModel {
        topology,
        info,
        architecture: TensorArchitecture::NemotronH(Box::new(NemotronHTensorModel {
            global_args: source_args,
            model,
        })),
    })
}

fn load_deepseek(
    model_dir: &Path,
    topology: ParallelTopology,
    requested_quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<TensorParallelModel, Error> {
    let source_args = deepseek_v3::get_model_args(model_dir)?;
    if requested_quantization.is_some() && source_args.native_fp8_config().is_some() {
        return Err(Error::Quantization(
            "native DeepSeek block-FP8 tensor-parallel weights cannot be implicitly requantized"
                .into(),
        ));
    }
    let local_heads = exact_division(
        "DeepSeek attention heads",
        source_args.num_attention_heads,
        topology.tensor_parallel_size,
    )?;
    let local_intermediate = exact_division(
        "DeepSeek dense intermediate size",
        source_args.intermediate_size,
        topology.tensor_parallel_size,
    )?;
    let local_moe_intermediate = exact_division(
        "DeepSeek expert intermediate size",
        source_args.moe_intermediate_size,
        topology.tensor_parallel_size,
    )?;
    let quantize_on_load = requested_quantization
        .map(|requested| {
            should_quantize_on_load(
                "DeepSeek tensor parallel",
                source_args.affine_quantization()?,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    if source_args.native_fp8_config().is_some() {
        for (tensor, dimension) in [
            (
                "model.layers.*.self_attn.q_proj.weight",
                local_heads * (source_args.qk_nope_head_dim + source_args.qk_rope_head_dim),
            ),
            (
                "model.layers.*.self_attn.kv_b_proj.weight",
                local_heads * (source_args.qk_nope_head_dim + source_args.v_head_dim),
            ),
            (
                "model.layers.*.self_attn.o_proj.weight",
                local_heads * source_args.v_head_dim,
            ),
            ("model.layers.*.mlp.down_proj.weight", local_intermediate),
            (
                "model.layers.*.mlp.experts.*.down_proj.weight",
                local_moe_intermediate,
            ),
        ] {
            require_alignment(tensor, dimension, 128, topology)?;
        }
    }
    if let Some(quantization) = quantize_on_load.or(source_args.affine_quantization()?) {
        for (tensor, dimension) in [
            (
                "model.layers.*.self_attn.o_proj.weight",
                local_heads * source_args.v_head_dim,
            ),
            ("model.layers.*.mlp.down_proj.weight", local_intermediate),
            (
                "model.layers.*.mlp.experts.*.down_proj.weight",
                local_moe_intermediate,
            ),
        ] {
            require_alignment(tensor, dimension, quantization.group_size(), topology)?;
        }
    }
    let vocabulary = balanced_contiguous_range(
        source_args.vocab_size as usize,
        topology.tensor_parallel_size,
        topology.tensor_parallel_rank,
        false,
    )?;
    let mut planner = ParallelPlanBuilder::new(topology);
    let source_embedding = linear::unloaded_maybe_quantized_embedding(
        source_args.vocab_size,
        source_args.hidden_size,
        source_args.weight_quantization_for("model.embed_tokens.weight"),
        stream,
    )?;
    register_vocabulary_module(&mut planner, &source_embedding, "model.embed_tokens")?;
    for index in 0..source_args.layer_schedule.len() {
        let layer = deepseek_v3::DecoderLayer::new_layerwise(&source_args, index as i32, stream)?;
        insert_deepseek_layer_plan(&mut planner, &layer, index)?;
    }
    let source_norm = nn::RmsNorm::unloaded(
        source_args.hidden_size,
        source_args.rms_norm_eps,
        Dtype::Float32,
        stream,
    )?;
    register_replicated_module(&mut planner, &source_norm, "model.norm")?;
    let source_head = linear::unloaded_maybe_quantized_linear(
        source_args.hidden_size,
        source_args.vocab_size,
        false,
        source_args.weight_quantization_for("lm_head.weight"),
        stream,
    )?;
    register_vocabulary_module(&mut planner, &source_head, "lm_head")?;
    let (plan, _local_layout) = planner.finish()?;
    let mut strict = StrictLoadConfig::default();
    for index in 0..source_args.num_nextn_predict_layers {
        strict = strict.allow_unused_prefix(format!(
            "model.layers.{}.",
            source_args.num_hidden_layers + index
        ));
    }
    let partition = load_partition(model_dir, &plan, weights_stream, stream, &strict)?;
    let info = partition_info(
        &partition,
        topology,
        ModelKind::DeepSeekV3,
        (
            source_args.num_attention_heads,
            source_args.num_attention_heads,
        ),
        (local_heads, local_heads),
        vocabulary.clone(),
    );
    let mut target_args = source_args.clone();
    target_args.num_attention_heads = local_heads;
    target_args.intermediate_size = local_intermediate;
    target_args.moe_intermediate_size = local_moe_intermediate;
    if let Some(quantization) = quantize_on_load {
        target_args.quantization_config = None;
        target_args.quantization = Some(quantization);
    }
    let local_vocab = i32::try_from(vocabulary.len())
        .map_err(|_| Error::Parallel("local vocabulary does not fit in i32".into()))?;
    let mut embedding = linear::unloaded_maybe_quantized_embedding(
        local_vocab,
        target_args.hidden_size,
        target_args.weight_quantization_for("model.embed_tokens.weight"),
        stream,
    )?;
    let mut layers = target_args
        .layer_schedule
        .iter()
        .enumerate()
        .map(|(index, _)| deepseek_v3::DecoderLayer::new(&target_args, index as i32, stream))
        .collect::<Result<Vec<_>, _>>()?;
    let mut norm = nn::RmsNorm::unloaded(
        target_args.hidden_size,
        target_args.rms_norm_eps,
        Dtype::Float32,
        stream,
    )?;
    let mut lm_head = linear::unloaded_maybe_quantized_linear(
        target_args.hidden_size,
        local_vocab,
        false,
        target_args.weight_quantization_for("lm_head.weight"),
        stream,
    )?;
    let mut tensors = partition.into_tensors();
    assign_module(
        &mut embedding,
        "model.embed_tokens",
        &mut tensors,
        quantize_on_load,
        stream,
    )?;
    for (index, layer) in layers.iter_mut().enumerate() {
        assign_module(
            layer,
            &format!("model.layers.{index}"),
            &mut tensors,
            quantize_on_load,
            stream,
        )?;
        if let Some(moe) = layer.mlp.moe_mut() {
            load_deepseek_experts(
                moe,
                index,
                (
                    target_args.n_routed_experts,
                    target_args.hidden_size,
                    target_args.moe_intermediate_size,
                ),
                &mut tensors,
                quantize_on_load,
                stream,
            )?;
        }
    }
    assign_module(&mut norm, "model.norm", &mut tensors, None, stream)?;
    assign_module(
        &mut lm_head,
        "lm_head",
        &mut tensors,
        quantize_on_load,
        stream,
    )?;
    ensure_no_unused(tensors)?;
    Ok(TensorParallelModel {
        topology,
        info,
        architecture: TensorArchitecture::DeepSeek(DeepSeekTensorModel {
            global_args: source_args,
            embedding,
            layers,
            norm,
            lm_head,
        }),
    })
}

fn ensure_no_unused(tensors: HashMap<String, Array>) -> Result<(), Error> {
    if tensors.is_empty() {
        Ok(())
    } else {
        let mut unused = tensors.into_keys().collect::<Vec<_>>();
        unused.sort();
        Err(Error::StrictLoadValidation {
            missing: Vec::new(),
            unused,
        })
    }
}

fn vocabulary_embedding(
    embedding: &mut MaybeQuantized<nn::Embedding>,
    tokens: &Array,
    range: &Range<usize>,
    group: &Group,
    stream: &Stream,
) -> Result<Array, Error> {
    let start = Array::from_int(
        i32::try_from(range.start)
            .map_err(|_| Error::Parallel("vocabulary range start does not fit in i32".into()))?,
    );
    let end = Array::from_int(
        i32::try_from(range.end)
            .map_err(|_| Error::Parallel("vocabulary range end does not fit in i32".into()))?,
    );
    let valid = tokens
        .ge(&start, stream)?
        .logical_and(tokens.lt(&end, stream)?, stream)?;
    let local_ids = tokens.subtract(&start, stream)?;
    let safe_ids = safemlx::ops::r#where(&valid, &local_ids, Array::from_int(0), stream)?;
    let local = embedding.forward(&safe_ids, stream)?;
    let valid = valid.expand_dims(-1, stream)?;
    let local = safemlx::ops::r#where(&valid, &local, zeros_like(&local, stream)?, stream)?;
    Ok(distributed::all_sum(&local, group, stream)?)
}

fn forward_llama(
    model: &mut LlamaTensorModel,
    tokens: &Array,
    explicit_mask: Option<&Array>,
    caches: &mut [TensorParallelLlamaLayerCache],
    info: &TensorParallelInfo,
    group: &Group,
    stream: &Stream,
) -> Result<Array, Error> {
    if caches.len() != model.layers.len() {
        return Err(Error::Parallel(format!(
            "Llama TP cache has {} layers, expected {}",
            caches.len(),
            model.layers.len()
        )));
    }
    for (layer, cache) in caches.iter().enumerate() {
        let expected = model
            .global_args
            .attention_schedule
            .get(layer)
            .expect("validated Llama tensor-parallel layer range")
            .window()
            .map(|window| {
                i32::try_from(window.get()).expect("validated Llama attention window fits i32")
            });
        let actual = match cache {
            TensorParallelLlamaLayerCache::Standard(cache) => cache.max_size(),
            TensorParallelLlamaLayerCache::Paged(cache) => cache.max_size(),
        };
        if actual != expected {
            return Err(Error::Parallel(format!(
                "Llama tensor-parallel cache policy mismatch at layer {layer}: expected window {expected:?}, got {actual:?}"
            )));
        }
    }
    let mut hidden = vocabulary_embedding(
        &mut model.embedding,
        tokens,
        &info.local_vocabulary_range,
        group,
        stream,
    )?;
    let sequence = tokens.dim(1);
    let offset = caches.first().map_or(0, |cache| match cache {
        TensorParallelLlamaLayerCache::Standard(cache) => cache.offset(),
        TensorParallelLlamaLayerCache::Paged(cache) => cache.offset(),
    });
    let allow_sliding_prefill = explicit_mask.is_none();
    let generated_mask = if explicit_mask.is_some() {
        None
    } else {
        (sequence > 1)
            .then(|| create_causal_mask(sequence, Some(offset), None, None, stream))
            .transpose()?
    };
    let mask = explicit_mask.or(generated_mask.as_ref());
    for (layer, cache) in model.layers.iter_mut().zip(caches) {
        hidden = match cache {
            TensorParallelLlamaLayerCache::Standard(cache) => layer.forward_tensor_parallel(
                &hidden,
                mask,
                Some(cache),
                allow_sliding_prefill,
                group,
                stream,
            )?,
            TensorParallelLlamaLayerCache::Paged(cache) => layer.forward_tensor_parallel(
                &hidden,
                mask,
                Some(cache),
                allow_sliding_prefill,
                group,
                stream,
            )?,
        };
    }
    let hidden = model.norm.forward(&hidden, stream)?;
    match model.lm_head.as_mut() {
        Some(head) => Ok(head.forward(&hidden, stream)?),
        None => match &mut model.embedding {
            MaybeQuantized::Original(embedding) => Ok(embedding.as_linear(&hidden, stream)?),
            MaybeQuantized::Quantized(embedding) => Ok(embedding.as_linear(&hidden, stream)?),
        },
    }
}

fn forward_qwen(
    model: &mut QwenTensorModel,
    tokens: &Array,
    explicit_mask: Option<&Array>,
    caches: &mut [TensorParallelLlamaLayerCache],
    info: &TensorParallelInfo,
    group: &Group,
    stream: &Stream,
) -> Result<Array, Error> {
    if caches.len() != model.layers.len() {
        return Err(Error::Parallel(format!(
            "Qwen TP cache has {} layers, expected {}",
            caches.len(),
            model.layers.len()
        )));
    }
    let mut hidden = vocabulary_embedding(
        &mut model.embedding,
        tokens,
        &info.local_vocabulary_range,
        group,
        stream,
    )?;
    let sequence = tokens.dim(1);
    let offset = caches.first().map_or(0, |cache| match cache {
        TensorParallelLlamaLayerCache::Standard(cache) => cache.offset(),
        TensorParallelLlamaLayerCache::Paged(cache) => cache.offset(),
    });
    let generated_mask = (explicit_mask.is_none() && sequence > 1)
        .then(|| create_causal_mask(sequence, Some(offset), None, None, stream))
        .transpose()?;
    let mask = explicit_mask.or(generated_mask.as_ref());
    for (layer, cache) in model.layers.iter_mut().zip(caches) {
        hidden = match cache {
            TensorParallelLlamaLayerCache::Standard(cache) => {
                layer.forward_tensor_parallel(&hidden, mask, Some(cache), group, stream)?
            }
            TensorParallelLlamaLayerCache::Paged(cache) => {
                layer.forward_tensor_parallel(&hidden, mask, Some(cache), group, stream)?
            }
        };
    }
    let hidden = model.norm.forward(&hidden, stream)?;
    match model.lm_head.as_mut() {
        Some(head) => Ok(head.forward(&hidden, stream)?),
        None => match &mut model.embedding {
            MaybeQuantized::Original(embedding) => Ok(embedding.as_linear(&hidden, stream)?),
            MaybeQuantized::Quantized(embedding) => Ok(embedding.as_linear(&hidden, stream)?),
        },
    }
}

fn gemma_vocabulary_embedding(
    embedding: &mut gemma4::Gemma4Embedding,
    tokens: &Array,
    range: &Range<usize>,
    group: &Group,
    stream: &Stream,
) -> Result<Array, Error> {
    let start = Array::from_int(
        i32::try_from(range.start)
            .map_err(|_| Error::Parallel("Gemma vocabulary start exceeds i32".into()))?,
    );
    let end = Array::from_int(
        i32::try_from(range.end)
            .map_err(|_| Error::Parallel("Gemma vocabulary end exceeds i32".into()))?,
    );
    let valid = tokens
        .ge(&start, stream)?
        .logical_and(tokens.lt(&end, stream)?, stream)?;
    let local_ids = tokens.subtract(&start, stream)?;
    let safe_ids = safemlx::ops::r#where(&valid, &local_ids, Array::from_int(0), stream)?;
    let local = embedding.forward(&safe_ids, stream)?;
    let valid = valid.expand_dims(-1, stream)?;
    let local = safemlx::ops::r#where(&valid, &local, zeros_like(&local, stream)?, stream)?;
    Ok(distributed::all_sum(&local, group, stream)?)
}

fn gemma_cache_offset(caches: &[Option<TensorParallelLlamaLayerCache>]) -> i32 {
    caches
        .iter()
        .flatten()
        .map(|cache| match cache {
            TensorParallelLlamaLayerCache::Standard(cache) => cache.offset(),
            TensorParallelLlamaLayerCache::Paged(cache) => cache.offset(),
        })
        .max()
        .unwrap_or(0)
}

#[allow(clippy::too_many_arguments)]
fn forward_gemma(
    model: &mut GemmaTensorModel,
    tokens: &Array,
    explicit_mask: Option<&Array>,
    caches: &mut [Option<TensorParallelLlamaLayerCache>],
    info: &TensorParallelInfo,
    group: &Group,
    stream: &Stream,
) -> Result<Array, Error> {
    if caches.len() != model.text.layers.len() {
        return Err(Error::Parallel(format!(
            "Gemma TP cache has {} layers, expected {}",
            caches.len(),
            model.text.layers.len()
        )));
    }
    let mut hidden = gemma_vocabulary_embedding(
        &mut model.text.embed_tokens,
        tokens,
        &info.local_vocabulary_range,
        group,
        stream,
    )?
    .multiply(
        Array::from_f32((model.global_args.hidden_size as f32).sqrt()),
        stream,
    )?;
    let per_layer_inputs = if let Some(embedding) = model.text.embed_tokens_per_layer.as_mut() {
        let projection = model
            .text
            .per_layer_model_projection
            .as_mut()
            .ok_or_else(|| Error::Parallel("missing Gemma per-layer model projection".into()))?;
        let norm = model
            .text
            .per_layer_projection_norm
            .as_mut()
            .ok_or_else(|| Error::Parallel("missing Gemma per-layer projection norm".into()))?;
        let dimensions = model.global_args.hidden_size_per_layer_input;
        let token_identity = gemma_vocabulary_embedding(
            embedding,
            tokens,
            model
                .per_layer_vocabulary_range
                .as_ref()
                .unwrap_or(&info.local_vocabulary_range),
            group,
            stream,
        )?
        .multiply(Array::from_f32((dimensions as f32).sqrt()), stream)?
        .reshape(
            &[
                tokens.dim(0),
                tokens.dim(1),
                model.global_args.num_hidden_layers,
                dimensions,
            ],
            stream,
        )?;
        let projected = projection
            .forward(&hidden, stream)?
            .multiply(
                Array::from_f32((model.global_args.hidden_size as f32).sqrt().recip()),
                stream,
            )?
            .reshape(
                &[
                    hidden.dim(0),
                    hidden.dim(1),
                    model.global_args.num_hidden_layers,
                    dimensions,
                ],
                stream,
            )?;
        Some(
            norm.forward(&projected, stream)?
                .add(token_identity, stream)?
                .multiply(Array::from_f32(2.0_f32.powf(-0.5)), stream)?,
        )
    } else {
        None
    };
    let position_offset = gemma_cache_offset(caches);
    let full_mask = match explicit_mask {
        Some(mask) => Some(mask.clone()),
        None if tokens.dim(1) > 1 => Some(create_causal_mask(
            tokens.dim(1),
            Some(position_offset),
            None,
            None,
            stream,
        )?),
        None => None,
    };
    let mut shared_kv = HashMap::new();
    for (index, (layer, cache)) in model.text.layers.iter_mut().zip(caches).enumerate() {
        let sliding_mask = if explicit_mask.is_none() && tokens.dim(1) > 1 {
            layer
                .layer_policy
                .attention
                .window()
                .map(|window| {
                    let window = i32::try_from(window.get()).map_err(|_| {
                        safemlx::error::Exception::custom("Gemma sliding window exceeds i32")
                    })?;
                    create_causal_mask(
                        tokens.dim(1),
                        Some(position_offset),
                        Some(window - 1),
                        None,
                        stream,
                    )
                })
                .transpose()?
        } else {
            None
        };
        let layer_mask = sliding_mask.as_ref().or(full_mask.as_ref());
        let per_layer_input = per_layer_inputs
            .as_ref()
            .map(|inputs| inputs.try_index_device((.., .., index as i32, ..), stream))
            .transpose()?;
        hidden = match cache.as_mut() {
            Some(TensorParallelLlamaLayerCache::Standard(cache)) => layer.forward_tensor_parallel(
                &hidden,
                layer_mask,
                Some(cache),
                position_offset,
                per_layer_input.as_ref(),
                &mut shared_kv,
                group,
                stream,
            )?,
            Some(TensorParallelLlamaLayerCache::Paged(cache)) => layer.forward_tensor_parallel(
                &hidden,
                layer_mask,
                Some(cache),
                position_offset,
                per_layer_input.as_ref(),
                &mut shared_kv,
                group,
                stream,
            )?,
            None => layer.forward_tensor_parallel::<ConcatKeyValueCache>(
                &hidden,
                layer_mask,
                None,
                position_offset,
                per_layer_input.as_ref(),
                &mut shared_kv,
                group,
                stream,
            )?,
        };
    }
    let hidden = model.text.norm.forward(&hidden, stream)?;
    let mut logits = match model.lm_head.as_mut() {
        Some(head) => head.forward(&hidden, stream)?,
        None => model.text.embed_tokens.as_linear(&hidden, stream)?,
    };
    if let Some(softcap) = model.global_args.final_logit_softcapping {
        logits = tanh(&logits.divide(Array::from_f32(softcap), stream)?, stream)?
            .multiply(Array::from_f32(softcap), stream)?;
    }
    Ok(logits)
}

fn forward_deepseek(
    model: &mut DeepSeekTensorModel,
    tokens: &Array,
    explicit_mask: Option<&Array>,
    caches: &mut [CompressedLatentCache],
    info: &TensorParallelInfo,
    group: &Group,
    stream: &Stream,
) -> Result<Array, Error> {
    if caches.len() != model.layers.len() {
        return Err(Error::Parallel(format!(
            "DeepSeek TP cache has {} layers, expected {}",
            caches.len(),
            model.layers.len()
        )));
    }
    let mut hidden = vocabulary_embedding(
        &mut model.embedding,
        tokens,
        &info.local_vocabulary_range,
        group,
        stream,
    )?;
    let sequence = tokens.dim(1);
    let offset = caches.first().map_or(0, CompressedLatentCache::offset);
    let generated_mask = (explicit_mask.is_none() && sequence > 1 && offset > 0)
        .then(|| create_causal_mask(sequence, Some(offset), None, None, stream))
        .transpose()?;
    let mask = explicit_mask.or(generated_mask.as_ref());
    for (layer, cache) in model.layers.iter_mut().zip(caches) {
        hidden = layer.forward_tensor_parallel(&hidden, mask, Some(cache), group, stream)?;
    }
    let hidden = model.norm.forward(&hidden, stream)?;
    Ok(model.lm_head.forward(&hidden, stream)?)
}

fn forward_gpt_oss(
    model: &mut GptOssTensorModel,
    tokens: &Array,
    explicit_mask: Option<&Array>,
    cache: &mut gpt_oss::Cache,
    info: &TensorParallelInfo,
    group: &Group,
    stream: &Stream,
) -> Result<Array, Error> {
    if cache.layers.len() != model.model.model.layers.len() {
        return Err(Error::Parallel(format!(
            "GPT-OSS TP cache has {} layers, expected {}",
            cache.layers.len(),
            model.model.model.layers.len()
        )));
    }
    let mut hidden = vocabulary_embedding(
        &mut model.model.model.embed_tokens,
        tokens,
        &info.local_vocabulary_range,
        group,
        stream,
    )?;
    let length = hidden.dim(1);
    for ((layer, layer_cache), policy) in model
        .model
        .model
        .layers
        .iter_mut()
        .zip(cache.layers.iter_mut())
        .zip(model.global_args.attention_schedule.iter())
    {
        let generated = if explicit_mask.is_none() && length > 1 {
            let offset = layer_cache.offset();
            let window = policy.window().map(|window| window.get() as i32);
            let max_past = window.map(|size| size - 1);
            Some(create_causal_mask(
                length,
                Some(offset.min(max_past.unwrap_or(offset))),
                max_past,
                None,
                stream,
            )?)
        } else {
            None
        };
        hidden = layer.forward_tensor_parallel(
            &hidden,
            explicit_mask.or(generated.as_ref()),
            layer_cache,
            group,
            stream,
        )?;
    }
    let hidden = model.model.model.norm.forward(&hidden, stream)?;
    Ok(model.model.lm_head.forward(&hidden, stream)?)
}

fn forward_kimi(
    model: &mut KimiTensorModel,
    tokens: &Array,
    explicit_mask: Option<&Array>,
    cache: &mut kimi_linear::Cache,
    info: &TensorParallelInfo,
    group: &Group,
    stream: &Stream,
) -> Result<Array, Error> {
    if cache.layers.len() != model.model.model.layers.len() {
        return Err(Error::Parallel(format!(
            "Kimi TP cache has {} layers, expected {}",
            cache.layers.len(),
            model.model.model.layers.len()
        )));
    }
    let mut hidden = vocabulary_embedding(
        &mut model.model.model.embed_tokens,
        tokens,
        &info.local_vocabulary_range,
        group,
        stream,
    )?;
    let sequence = tokens.dim(1);
    let offset = cache.offset();
    let generated_mask = (explicit_mask.is_none() && sequence > 1 && offset > 0)
        .then(|| create_causal_mask(sequence, Some(offset), None, None, stream))
        .transpose()?;
    let mask = explicit_mask.or(generated_mask.as_ref());
    for (layer, cache) in model
        .model
        .model
        .layers
        .iter_mut()
        .zip(cache.layers.iter_mut())
    {
        hidden = layer.forward_tensor_parallel(&hidden, mask, Some(cache), group, stream)?;
    }
    let hidden = model.model.model.norm.forward(&hidden, stream)?;
    match model.model.lm_head.as_mut() {
        Some(head) => Ok(head.forward(&hidden, stream)?),
        None => match &mut model.model.model.embed_tokens {
            MaybeQuantized::Original(embedding) => Ok(embedding.as_linear(&hidden, stream)?),
            MaybeQuantized::Quantized(embedding) => Ok(embedding.as_linear(&hidden, stream)?),
        },
    }
}

fn forward_lfm2(
    model: &mut Lfm2TensorModel,
    tokens: &Array,
    explicit_mask: Option<&Array>,
    cache: &mut lfm2::Cache,
    info: &TensorParallelInfo,
    group: &Group,
    stream: &Stream,
) -> Result<Array, Error> {
    if cache.layers.len() != model.model.model.layers.len() {
        return Err(Error::Parallel(format!(
            "LFM2 TP cache has {} layers, expected {}",
            cache.layers.len(),
            model.model.model.layers.len()
        )));
    }
    let mut hidden = vocabulary_embedding(
        &mut model.model.model.embed_tokens,
        tokens,
        &info.local_vocabulary_range,
        group,
        stream,
    )?;
    let sequence = tokens.dim(1);
    let generated_mask = (explicit_mask.is_none() && sequence > 1)
        .then(|| create_causal_mask(sequence, Some(cache.offset()), None, None, stream))
        .transpose()?;
    let mask = explicit_mask.or(generated_mask.as_ref());
    for (layer, cache) in model
        .model
        .model
        .layers
        .iter_mut()
        .zip(cache.layers.iter_mut())
    {
        hidden = layer.forward_tensor_parallel(&hidden, mask, Some(cache), group, stream)?;
    }
    let hidden = model.model.model.embedding_norm.forward(&hidden, stream)?;
    match model.model.lm_head.as_mut() {
        Some(head) => Ok(head.forward(&hidden, stream)?),
        None => match &mut model.model.model.embed_tokens {
            MaybeQuantized::Original(embedding) => Ok(embedding.as_linear(&hidden, stream)?),
            MaybeQuantized::Quantized(embedding) => Ok(embedding.as_linear(&hidden, stream)?),
        },
    }
}

fn forward_qwen_hybrid(
    model: &mut QwenHybridTensorModel,
    tokens: &Array,
    explicit_mask: Option<&Array>,
    cache: &mut qwen_hybrid::Cache,
    info: &TensorParallelInfo,
    group: &Group,
    stream: &Stream,
) -> Result<Array, Error> {
    if cache.layers.len() != model.model.model.layers.len() {
        return Err(Error::Parallel(format!(
            "Qwen hybrid TP cache has {} layers, expected {}",
            cache.layers.len(),
            model.model.model.layers.len()
        )));
    }
    let mut hidden = vocabulary_embedding(
        &mut model.model.model.embed_tokens,
        tokens,
        &info.local_vocabulary_range,
        group,
        stream,
    )?;
    let sequence = tokens.dim(1);
    let generated_mask = (explicit_mask.is_none() && sequence > 1)
        .then(|| create_causal_mask(sequence, Some(cache.offset()), None, None, stream))
        .transpose()?;
    let mask = explicit_mask.or(generated_mask.as_ref());
    for (layer, cache) in model
        .model
        .model
        .layers
        .iter_mut()
        .zip(cache.layers.iter_mut())
    {
        hidden = layer.forward_tensor_parallel(
            qwen_hybrid::BlockInput {
                x: &hidden,
                mask,
                cache: Some(cache),
            },
            group,
            stream,
        )?;
    }
    let hidden = model.model.model.norm.forward(&hidden, stream)?;
    match model.model.lm_head.as_mut() {
        Some(head) => Ok(head.forward(&hidden, stream)?),
        None => match &mut model.model.model.embed_tokens {
            MaybeQuantized::Original(embedding) => Ok(embedding.as_linear(&hidden, stream)?),
            MaybeQuantized::Quantized(embedding) => Ok(embedding.as_linear(&hidden, stream)?),
        },
    }
}

fn forward_nemotron_h(
    model: &mut NemotronHTensorModel,
    tokens: &Array,
    explicit_mask: Option<&Array>,
    cache: &mut nemotron_h::Cache,
    info: &TensorParallelInfo,
    group: &Group,
    stream: &Stream,
) -> Result<Array, Error> {
    if cache.layers.len() != model.model.model.layers.len() {
        return Err(Error::Parallel(format!(
            "Nemotron-H TP cache has {} layers, expected {}",
            cache.layers.len(),
            model.model.model.layers.len()
        )));
    }
    let mut hidden = vocabulary_embedding(
        &mut model.model.model.embeddings,
        tokens,
        &info.local_vocabulary_range,
        group,
        stream,
    )?;
    let sequence = tokens.dim(1);
    let generated_mask = (explicit_mask.is_none() && sequence > 1)
        .then(|| create_causal_mask(sequence, Some(cache.offset()), None, None, stream))
        .transpose()?;
    let mask = explicit_mask.or(generated_mask.as_ref());
    for (layer, layer_cache) in model
        .model
        .model
        .layers
        .iter_mut()
        .zip(cache.layers.iter_mut())
    {
        hidden = layer.forward_tensor_parallel(
            nemotron_h::BlockInput {
                x: &hidden,
                mask,
                cache: Some(layer_cache),
            },
            group,
            stream,
        )?;
    }
    let hidden = model.model.model.norm_f.forward(&hidden, stream)?;
    match model.model.lm_head.as_mut() {
        Some(head) => Ok(head.forward(&hidden, stream)?),
        None => match &mut model.model.model.embeddings {
            MaybeQuantized::Original(embedding) => Ok(embedding.as_linear(&hidden, stream)?),
            MaybeQuantized::Quantized(embedding) => Ok(embedding.as_linear(&hidden, stream)?),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::attention::{AttentionPolicy, LayerSchedule};
    use crate::runtime::checkpoint::binding::canonical_checkpoint_name;
    use crate::runtime::distributed::topology::DeviceAssignment;
    use safemlx::{Device, DeviceType};

    fn topology(world: usize, rank: usize, tp: usize) -> ParallelTopology {
        ParallelTopology::from_rank(
            world,
            rank,
            tp,
            1,
            1,
            DeviceAssignment::new(DeviceType::Cpu, 0),
        )
        .unwrap()
    }

    fn filled(shape: &[i32], value: f32, stream: &Stream) -> Array {
        Array::full::<f32>(shape, Array::from_f32(value), stream).unwrap()
    }

    fn save_fixture(directory: &Path, config: &serde_json::Value, tensors: &[(String, Array)]) {
        Array::save_safetensors(
            tensors.iter().map(|(name, value)| (name.as_str(), value)),
            None,
            directory.join("model.safetensors"),
        )
        .unwrap();
        std::fs::write(
            directory.join("config.json"),
            serde_json::to_vec_pretty(config).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn pure_tensor_topology_rejects_hybrids_before_loading() {
        assert!(validate_pure_tensor(topology(2, 0, 2)).is_ok());
        let hybrid =
            ParallelTopology::from_rank(4, 0, 2, 2, 1, DeviceAssignment::new(DeviceType::Cpu, 0))
                .unwrap();
        assert!(validate_pure_tensor(hybrid)
            .unwrap_err()
            .to_string()
            .contains("hybrid"));
    }

    #[test]
    fn uneven_vocabulary_widths_cover_global_vocabulary() {
        assert_eq!(vocabulary_widths(11, 3).unwrap(), vec![4, 4, 3]);
    }

    #[test]
    fn dense_family_layer_plans_use_typed_projection_roles() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let topology = topology(2, 0, 2);

        let llama_args = llama::ModelArgs {
            model_type: "llama".into(),
            hidden_size: 8,
            num_hidden_layers: 1,
            intermediate_size: 16,
            num_attention_heads: 2,
            rms_norm_eps: 1e-5,
            vocab_size: 11,
            num_key_value_heads: 2,
            max_position_embeddings: 32,
            rope_theta: 10_000.0,
            rope_traditional: false,
            head_dim: 4,
            tie_word_embeddings: true,
            attention_bias: false,
            mlp_bias: false,
            rope_scaling: None,
            attention_schedule: LayerSchedule::all_full(1).unwrap(),
            quantization: None,
            quantization_config: None,
            quantized_weights: None,
            quantized_weight_configs: None,
        };
        let llama_layer = llama::TransformerBlock::new_for_layer(&llama_args, 0, &stream).unwrap();
        let mut llama_plan = ParallelPlanBuilder::new(topology);
        insert_llama_layer_plan(&mut llama_plan, &llama_layer, 0).unwrap();
        let (_, llama_local) = llama_plan.finish().unwrap();
        assert_eq!(
            llama_local
                .tensor("model.layers.0.self_attn.q_proj.weight")
                .unwrap()
                .role(),
            ParameterRole::ColumnProjection
        );
        assert_eq!(
            llama_local
                .tensor("model.layers.0.self_attn.o_proj.weight")
                .unwrap()
                .role(),
            ParameterRole::RowProjection
        );

        let qwen_args = qwen::DecoderConfig {
            model_type: "qwen3".into(),
            hidden_size: 8,
            num_hidden_layers: 1,
            intermediate_size: 16,
            num_attention_heads: 2,
            rms_norm_eps: 1e-6,
            vocab_size: 11,
            num_key_value_heads: 2,
            max_position_embeddings: 32,
            rope_theta: 10_000.0,
            head_dim: 4,
            tie_word_embeddings: true,
            rope_scaling: None,
            hidden_act: "silu".into(),
            attention_dropout: 0.0,
            attention_bias: Some(false),
            mlp_bias: Some(false),
            attention_schedule: LayerSchedule::all_full(1).unwrap(),
            quantization: None,
            quantization_config: None,
            quantized_weights: None,
            moe_intermediate_size: 0,
            num_experts: 0,
            num_experts_per_tok: 0,
            norm_topk_prob: false,
            quantized_weight_configs: None,
        };
        let qwen_layer = qwen::TransformerBlock::new_for_layer(&qwen_args, 0, &stream).unwrap();
        let mut qwen_plan = ParallelPlanBuilder::new(topology);
        insert_qwen_layer_plan(&mut qwen_plan, &qwen_layer, 0).unwrap();
        let (_, qwen_local) = qwen_plan.finish().unwrap();
        assert_eq!(
            qwen_local
                .tensor("model.layers.0.mlp.gate_proj.weight")
                .unwrap()
                .role(),
            ParameterRole::ColumnProjection
        );
        assert_eq!(
            qwen_local
                .tensor("model.layers.0.mlp.down_proj.weight")
                .unwrap()
                .role(),
            ParameterRole::RowProjection
        );

        let gemma_policy = gemma4::LayerPolicy {
            attention: AttentionPolicy::Full,
            head_dim: NonZeroU32::new(4).unwrap(),
            num_key_value_heads: NonZeroU32::new(2).unwrap(),
            key_value: gemma4::KeyValuePolicy::Local {
                value: gemma4::ValuePolicy::Projected,
            },
            intermediate_size: NonZeroU32::new(16).unwrap(),
            feed_forward: gemma4::FeedForwardPolicy::Dense,
        };
        let gemma_args = gemma4::ModelArgs {
            model_type: "gemma4_unified_text".into(),
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            rms_norm_eps: 1e-6,
            vocab_size: 11,
            pad_token_id: 0,
            max_position_embeddings: 32,
            rope_theta: 10_000.0,
            tie_word_embeddings: true,
            attention_bias: false,
            quantized: false,
            weight_quantization: None,
            quantized_weights: None,
            quantized_weight_configs: None,
            quantization_group_size: 64,
            quantization_bits: 4,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: None,
            layer_schedule: LayerSchedule::new(1, vec![gemma_policy]).unwrap(),
            final_logit_softcapping: None,
            num_experts: None,
            top_k_experts: None,
            moe_intermediate_size: None,
            rope_scaling: None,
            rope_parameters: None,
        };
        let gemma_layer =
            gemma4::TransformerBlock::new(&gemma_args, gemma_policy, 0, &stream).unwrap();
        let mut gemma_plan = ParallelPlanBuilder::new(topology);
        insert_gemma_layer_plan(&mut gemma_plan, &gemma_layer, 0).unwrap();
        let (_, gemma_local) = gemma_plan.finish().unwrap();
        assert_eq!(
            gemma_local
                .tensor("model.language_model.layers.0.self_attn.q_proj.weight")
                .unwrap()
                .role(),
            ParameterRole::ColumnProjection
        );
        assert_eq!(
            gemma_local
                .tensor("model.language_model.layers.0.mlp.down_proj.weight")
                .unwrap()
                .role(),
            ParameterRole::RowProjection
        );
    }

    #[test]
    fn mla_and_moe_family_plans_preserve_packed_geometry() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let topology = topology(2, 0, 2);

        let qwen_args = qwen::DecoderConfig {
            model_type: "qwen3".into(),
            hidden_size: 8,
            num_hidden_layers: 1,
            intermediate_size: 0,
            num_attention_heads: 2,
            rms_norm_eps: 1e-6,
            vocab_size: 16,
            num_key_value_heads: 2,
            max_position_embeddings: 32,
            rope_theta: 10_000.0,
            head_dim: 4,
            tie_word_embeddings: true,
            rope_scaling: None,
            hidden_act: "silu".into(),
            attention_dropout: 0.0,
            attention_bias: Some(false),
            mlp_bias: Some(false),
            attention_schedule: LayerSchedule::all_full(1).unwrap(),
            quantization: None,
            quantization_config: None,
            quantized_weights: None,
            moe_intermediate_size: 8,
            num_experts: 4,
            num_experts_per_tok: 2,
            norm_topk_prob: true,
            quantized_weight_configs: None,
        };
        let qwen_layer = qwen::TransformerBlock::new_for_layer(&qwen_args, 0, &stream).unwrap();
        let mut qwen_plan = ParallelPlanBuilder::new(topology);
        insert_qwen_layer_plan(&mut qwen_plan, &qwen_layer, 0).unwrap();
        let (_, qwen_local) = qwen_plan.finish().unwrap();
        assert_eq!(
            qwen_local
                .tensor("model.layers.0.mlp.experts.gate_up_proj")
                .unwrap()
                .local_shape(),
            &[4, 8, 8]
        );
        assert_eq!(
            qwen_local
                .tensor("model.layers.0.mlp.experts.down_proj")
                .unwrap()
                .local_shape(),
            &[4, 8, 4]
        );

        let deepseek_args = deepseek_v3::model_args_from_config_value(&serde_json::json!({
            "model_type": "deepseek_v3",
            "hidden_size": 8,
            "intermediate_size": 16,
            "moe_intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "vocab_size": 16,
            "rms_norm_eps": 0.000001,
            "max_position_embeddings": 64,
            "q_lora_rank": null,
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "v_head_dim": 2,
            "first_k_dense_replace": 0,
            "moe_layer_freq": 1,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "n_group": 1,
            "topk_group": 1
        }))
        .unwrap();
        let deepseek_layer =
            deepseek_v3::DecoderLayer::new_layerwise(&deepseek_args, 0, &stream).unwrap();
        let mut deepseek_plan = ParallelPlanBuilder::new(topology);
        insert_deepseek_layer_plan(&mut deepseek_plan, &deepseek_layer, 0).unwrap();
        let (_, deepseek_local) = deepseek_plan.finish().unwrap();
        assert_eq!(
            deepseek_local
                .tensor("model.layers.0.mlp.experts.0.gate_proj.weight")
                .unwrap()
                .local_shape(),
            &[4, 8]
        );

        let hybrid_args = qwen_hybrid::model_args_from_config_value(&serde_json::json!({
            "model_type": "qwen3_5_moe_text",
            "vocab_size": 16,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "linear_conv_kernel_dim": 2,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "intermediate_size": 0,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
            "num_experts_per_tok": 2,
            "num_experts": 4,
            "layer_types": ["full_attention"]
        }))
        .unwrap();
        let hybrid_layer = qwen_hybrid::TransformerBlock::new(&hybrid_args, 0, &stream).unwrap();
        let mut hybrid_plan = ParallelPlanBuilder::new(topology);
        insert_qwen_hybrid_layer_plan(&mut hybrid_plan, &hybrid_layer, 0, &hybrid_args, true)
            .unwrap();
        let (_, hybrid_local) = hybrid_plan.finish().unwrap();
        assert_eq!(
            hybrid_local
                .tensor("model.layers.0.self_attn.q_proj.weight")
                .unwrap()
                .local_shape(),
            &[16, 16]
        );
        assert_eq!(
            hybrid_local
                .tensor("model.layers.0.mlp.experts.0.gate_proj.weight")
                .unwrap()
                .local_shape(),
            &[8, 16]
        );
    }

    #[test]
    fn qwen_moe_loader_materializes_rank_local_packed_experts() {
        let context = safemlx::ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let config = serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 0,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "rms_norm_eps": 0.000001,
            "vocab_size": 16,
            "max_position_embeddings": 32,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false,
            "hidden_act": "silu",
            "attention_bias": false,
            "mlp_bias": false,
            "moe_intermediate_size": 8,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "norm_topk_prob": true
        });
        let args = qwen::DecoderConfig {
            model_type: "qwen3".into(),
            hidden_size: 8,
            num_hidden_layers: 1,
            intermediate_size: 0,
            num_attention_heads: 2,
            rms_norm_eps: 1e-6,
            vocab_size: 16,
            num_key_value_heads: 2,
            max_position_embeddings: 32,
            rope_theta: 10_000.0,
            head_dim: 4,
            tie_word_embeddings: false,
            rope_scaling: None,
            hidden_act: "silu".into(),
            attention_dropout: 0.0,
            attention_bias: Some(false),
            mlp_bias: Some(false),
            attention_schedule: LayerSchedule::all_full(1).unwrap(),
            quantization: None,
            quantization_config: None,
            quantized_weights: None,
            moe_intermediate_size: 8,
            num_experts: 4,
            num_experts_per_tok: 2,
            norm_topk_prob: true,
            quantized_weight_configs: None,
        };
        let mut source = qwen::Model::new(args, stream).unwrap();
        for (name, parameter) in source.parameters_mut().flatten() {
            let shape = parameter.shape().to_vec();
            *parameter = if name.ends_with("norm.weight") {
                Array::ones::<f32>(&shape, stream).unwrap()
            } else {
                Array::full::<f32>(&shape, Array::from_f32(0.01), stream).unwrap()
            };
        }
        let tensors = source
            .parameters()
            .flatten()
            .into_iter()
            .map(|(name, value)| (canonical_checkpoint_name(&name), value.clone()))
            .collect::<Vec<_>>();
        let directory = tempfile::tempdir().unwrap();
        Array::save_safetensors(
            tensors.iter().map(|(name, value)| (name.as_str(), value)),
            None,
            directory.path().join("model.safetensors"),
        )
        .unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();

        let model = load_tensor_parallel_model(directory.path(), topology(2, 0, 2), stream, stream)
            .unwrap();
        let TensorArchitecture::Qwen(model) = &model.architecture else {
            panic!("expected Qwen tensor model")
        };
        let qwen::FeedForward::Moe(moe) = &model.layers[0].mlp else {
            panic!("expected Qwen MoE layer")
        };
        assert_eq!(moe.experts.gate_up_proj.shape(), &[4, 8, 8]);
        assert_eq!(moe.experts.down_proj.shape(), &[4, 8, 4]);
    }

    #[test]
    fn gpt_oss_loader_shards_native_mxfp4_expert_groups() {
        let context = safemlx::ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let config = serde_json::json!({
            "model_type": "gpt_oss",
            "hidden_size": 64,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": 16,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
            "rms_norm_eps": 0.00001,
            "sliding_window": 8,
            "max_position_embeddings": 128,
            "rope_theta": 150000.0,
            "layer_types": ["full_attention"],
            "quantization_config": {"quant_method": "mxfp4"}
        });
        let args = gpt_oss::model_args_from_config_value(&config).unwrap();
        let mut source = gpt_oss::Model::new(args, stream).unwrap();
        let tensors = source
            .parameters_mut()
            .flatten()
            .into_iter()
            .map(|(name, parameter)| {
                let value = safemlx::ops::zeros_dtype(parameter.shape(), parameter.dtype(), stream)
                    .unwrap();
                *parameter = value.clone();
                (canonical_checkpoint_name(&name), value)
            })
            .collect::<Vec<_>>();
        let directory = tempfile::tempdir().unwrap();
        save_fixture(directory.path(), &config, &tensors);

        let loaded =
            load_tensor_parallel_model(directory.path(), topology(2, 0, 2), stream, stream)
                .unwrap();
        let TensorArchitecture::GptOss(loaded) = &loaded.architecture else {
            panic!("expected GPT-OSS tensor model")
        };
        let layer = &loaded.model.model.layers[0];
        let MaybeQuantized::Original(q_proj) = &layer.self_attn.q_proj else {
            panic!("expected dense GPT-OSS query projection")
        };
        assert_eq!(q_proj.weight.shape(), &[32, 64]);
        assert_eq!(layer.self_attn.sinks.shape(), &[1]);
        assert_eq!(
            layer.mlp.experts.gate_up_proj_blocks.shape(),
            &[2, 64, 2, 16]
        );
        assert_eq!(layer.mlp.experts.gate_up_proj_scales.shape(), &[2, 64, 2]);
        assert_eq!(layer.mlp.experts.down_proj_blocks.shape(), &[2, 64, 1, 16]);
        assert_eq!(layer.mlp.experts.down_proj_scales.shape(), &[2, 64, 1]);
        assert_eq!(layer.mlp.experts.down_proj_bias.shape(), &[2, 64]);
    }

    #[test]
    fn hybrid_moe_loaders_pack_official_split_experts_after_sharding() {
        let context = safemlx::ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();

        let kimi_config = serde_json::json!({
            "model_type": "kimi_linear",
            "vocab_size": 16,
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 16,
            "head_dim": 4,
            "model_max_length": 128,
            "rms_norm_eps": 0.00001,
            "rope_theta": 10000.0,
            "linear_attn_config": {
                "kda_layers": [1],
                "full_attn_layers": [2],
                "num_heads": 2,
                "head_dim": 4,
                "short_conv_kernel_size": 2
            },
            "num_experts": 4,
            "moe_intermediate_size": 8,
            "kv_lora_rank": 4,
            "q_lora_rank": null,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "v_head_dim": 2,
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
        });
        let kimi_args = kimi_linear::model_args_from_config_value(&kimi_config).unwrap();
        let mut kimi_source = kimi_linear::Model::new(kimi_args, stream).unwrap();
        let mut kimi_tensors = Vec::new();
        for (name, parameter) in kimi_source.parameters_mut().flatten() {
            let canonical = canonical_checkpoint_name(&name);
            if canonical.starts_with("model.layers.1.mlp.experts.") {
                continue;
            }
            let source_name = if canonical.starts_with("model.layers.1.mlp.") {
                canonical.replace("model.layers.1.mlp.", "model.layers.1.block_sparse_moe.")
            } else {
                canonical
            };
            kimi_tensors.push((source_name, filled(parameter.shape(), 0.01, stream)));
        }
        for expert in 0..4 {
            for (projection, shape) in [
                ("gate_proj", vec![8, 8]),
                ("up_proj", vec![8, 8]),
                ("down_proj", vec![8, 8]),
            ] {
                kimi_tensors.push((
                    format!("model.layers.1.block_sparse_moe.experts.{expert}.{projection}.weight"),
                    filled(&shape, 0.01, stream),
                ));
            }
        }
        let kimi_dir = tempfile::tempdir().unwrap();
        save_fixture(kimi_dir.path(), &kimi_config, &kimi_tensors);
        let kimi =
            load_tensor_parallel_model(kimi_dir.path(), topology(2, 0, 2), stream, stream).unwrap();
        let TensorArchitecture::Kimi(kimi) = &kimi.architecture else {
            panic!("expected Kimi tensor model")
        };
        let kimi_linear::Attention::Kda(kda) = &kimi.model.model.layers[0].self_attn else {
            panic!("expected Kimi KDA layer")
        };
        assert_eq!(kda.num_heads, 1);
        assert_eq!(kda.q_proj.weight.shape(), &[4, 8]);
        assert_eq!(kda.q_conv1d.weight.shape(), &[4, 1, 2]);
        assert_eq!(kda.A_log.shape(), &[1, 1, 1, 1]);
        assert_eq!(kda.dt_bias.shape(), &[4]);
        assert_eq!(kda.o_proj.weight.shape(), &[8, 4]);
        let kimi_linear::FeedForward::Moe(moe) = &kimi.model.model.layers[1].mlp else {
            panic!("expected Kimi MoE layer")
        };
        assert_eq!(moe.experts.gate_up_proj.shape(), &[4, 8, 8]);
        assert_eq!(moe.experts.down_proj.shape(), &[4, 8, 4]);

        let lfm_config = serde_json::json!({
            "model_type": "lfm2_moe",
            "vocab_size": 16,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "norm_eps": 0.00001,
            "conv_L_cache": 3,
            "block_multiple_of": 4,
            "block_ffn_dim_multiplier": 1.0,
            "block_auto_adjust_ff_dim": true,
            "layer_types": ["conv", "full_attention"],
            "tie_embedding": false,
            "moe_intermediate_size": 8,
            "num_dense_layers": 1,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "norm_topk_prob": true,
            "use_expert_bias": true
        });
        let lfm_args = lfm2::model_args_from_config_value(&lfm_config).unwrap();
        let mut lfm_source = lfm2::Model::new(lfm_args, stream).unwrap();
        let mut lfm_tensors = Vec::new();
        for (name, parameter) in lfm_source.parameters_mut().flatten() {
            let canonical = canonical_checkpoint_name(&name);
            if canonical.starts_with("model.layers.1.feed_forward.experts.") {
                continue;
            }
            lfm_tensors.push((canonical, filled(parameter.shape(), 0.01, stream)));
        }
        for expert in 0..4 {
            for (projection, shape) in [
                ("w1", vec![8, 16]),
                ("w3", vec![8, 16]),
                ("w2", vec![16, 8]),
            ] {
                lfm_tensors.push((
                    format!("model.layers.1.feed_forward.experts.{expert}.{projection}.weight"),
                    filled(&shape, 0.01, stream),
                ));
            }
        }
        let lfm_dir = tempfile::tempdir().unwrap();
        save_fixture(lfm_dir.path(), &lfm_config, &lfm_tensors);
        let lfm =
            load_tensor_parallel_model(lfm_dir.path(), topology(2, 0, 2), stream, stream).unwrap();
        let TensorArchitecture::Lfm2(lfm) = &lfm.architecture else {
            panic!("expected LFM2 tensor model")
        };
        let experts = lfm.model.model.layers[1]
            .feed_forward
            .experts
            .as_ref()
            .expect("LFM2 MoE experts");
        assert_eq!(experts.gate_up_proj.shape(), &[4, 8, 16]);
        assert_eq!(experts.down_proj.shape(), &[4, 16, 4]);
    }

    #[test]
    fn nemotron_h_loader_shards_every_hybrid_operator_and_packs_split_experts() {
        let context = safemlx::ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let config = serde_json::json!({
            "model_type": "nemotron_h",
            "vocab_size": 16,
            "hidden_size": 16,
            "num_hidden_layers": 4,
            "hybrid_override_pattern": "M*-E",
            "intermediate_size": 8,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "max_position_embeddings": 64,
            "layer_norm_epsilon": 0.00001,
            "norm_eps": 0.00001,
            "ssm_state_size": 2,
            "mamba_num_heads": 4,
            "n_groups": 2,
            "mamba_head_dim": 4,
            "conv_kernel": 2,
            "chunk_size": 2,
            "mamba_hidden_act": "silu",
            "mlp_hidden_act": "relu2",
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "moe_intermediate_size": 8,
            "moe_shared_expert_intermediate_size": 8,
            "num_experts_per_tok": 2,
            "n_group": 1,
            "topk_group": 1,
            "tie_word_embeddings": false,
            "use_conv_bias": true
        });
        let args = nemotron_h::model_args_from_config_value(&config).unwrap();
        let mut source = nemotron_h::Model::new(args, stream).unwrap();
        let mut tensors = Vec::new();
        let layer_fields = ["mamba", "attention", "mlp", "moe"];
        for (name, parameter) in source.parameters_mut().flatten() {
            let canonical = canonical_checkpoint_name(&name);
            if canonical.starts_with("model.layers.3.moe.experts.") {
                continue;
            }
            let mut checkpoint = canonical
                .strip_prefix("model.")
                .map_or_else(|| canonical.clone(), |suffix| format!("backbone.{suffix}"));
            for (layer, field) in layer_fields.iter().enumerate() {
                let runtime = format!("backbone.layers.{layer}.{field}.");
                if checkpoint.starts_with(&runtime) {
                    checkpoint = checkpoint.replacen(
                        &runtime,
                        &format!("backbone.layers.{layer}.mixer."),
                        1,
                    );
                    break;
                }
            }
            tensors.push((checkpoint, filled(parameter.shape(), 0.01, stream)));
        }
        for expert in 0..4 {
            for (projection, shape) in [("up_proj", vec![8, 16]), ("down_proj", vec![16, 8])] {
                tensors.push((
                    format!("backbone.layers.3.mixer.experts.{expert}.{projection}.weight"),
                    filled(&shape, 0.01, stream),
                ));
            }
        }
        let directory = tempfile::tempdir().unwrap();
        save_fixture(directory.path(), &config, &tensors);

        let loaded =
            load_tensor_parallel_model(directory.path(), topology(2, 0, 2), stream, stream)
                .unwrap();
        let TensorArchitecture::NemotronH(loaded) = &loaded.architecture else {
            panic!("expected Nemotron-H tensor model")
        };

        let mamba = loaded.model.model.layers[0]
            .mamba
            .as_ref()
            .expect("Mamba layer");
        assert_eq!(mamba.num_heads, 2);
        assert_eq!(mamba.n_groups, 1);
        assert_eq!(
            mamba.in_proj.parameters().flatten()["weight"].shape(),
            &[22, 16]
        );
        assert_eq!(mamba.conv1d.weight.shape(), &[12, 1, 2]);
        assert_eq!(mamba.A_log.shape(), &[2]);
        assert_eq!(mamba.norm.weight.shape(), &[8]);
        assert_eq!(
            mamba.out_proj.parameters().flatten()["weight"].shape(),
            &[16, 8]
        );

        let attention = loaded.model.model.layers[1]
            .attention
            .as_ref()
            .expect("attention layer");
        assert_eq!(attention.n_heads, 2);
        assert_eq!(attention.n_kv_heads, 1);
        assert_eq!(
            attention.q_proj.parameters().flatten()["weight"].shape(),
            &[8, 16]
        );
        assert_eq!(
            attention.k_proj.parameters().flatten()["weight"].shape(),
            &[4, 16]
        );
        assert_eq!(
            attention.o_proj.parameters().flatten()["weight"].shape(),
            &[16, 8]
        );

        let mlp = loaded.model.model.layers[2]
            .mlp
            .as_ref()
            .expect("dense MLP");
        assert_eq!(
            mlp.up_proj.parameters().flatten()["weight"].shape(),
            &[4, 16]
        );
        assert_eq!(
            mlp.down_proj.parameters().flatten()["weight"].shape(),
            &[16, 4]
        );

        let moe = loaded.model.model.layers[3]
            .moe
            .as_ref()
            .expect("sparse MoE");
        assert_eq!(moe.experts.up_proj.shape(), &[4, 4, 16]);
        assert_eq!(moe.experts.down_proj.shape(), &[4, 16, 4]);
        assert_eq!(
            moe.shared_experts.up_proj.parameters().flatten()["weight"].shape(),
            &[4, 16]
        );
    }

    #[test]
    fn qwen_hybrid_loader_shards_fused_query_gate_and_split_experts() {
        let context = safemlx::ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let config = serde_json::json!({
            "model_type": "qwen3_5_moe_text",
            "vocab_size": 16,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "rms_norm_eps": 0.000001,
            "linear_conv_kernel_dim": 2,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "intermediate_size": 0,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
            "num_experts_per_tok": 2,
            "num_experts": 4,
            "layer_types": ["full_attention", "linear_attention"],
            "tie_word_embeddings": false
        });
        let args = qwen_hybrid::model_args_from_config_value(&config).unwrap();
        let mut source = qwen_hybrid::Model::new(args, None, None, None, stream).unwrap();
        let mut tensors = Vec::new();
        for (name, parameter) in source.parameters_mut().flatten() {
            let canonical = canonical_checkpoint_name(&name);
            if canonical.contains(".mlp.experts.") {
                continue;
            }
            tensors.push((canonical, filled(parameter.shape(), 0.01, stream)));
        }
        for layer in 0..2 {
            for expert in 0..4 {
                for (projection, shape) in [
                    ("gate_proj", vec![16, 16]),
                    ("up_proj", vec![16, 16]),
                    ("down_proj", vec![16, 16]),
                ] {
                    tensors.push((
                        format!("model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"),
                        filled(&shape, 0.01, stream),
                    ));
                }
            }
        }
        let directory = tempfile::tempdir().unwrap();
        save_fixture(directory.path(), &config, &tensors);
        let loaded =
            load_tensor_parallel_model(directory.path(), topology(2, 0, 2), stream, stream)
                .unwrap();
        let TensorArchitecture::QwenHybrid(loaded) = &loaded.architecture else {
            panic!("expected Qwen hybrid tensor model")
        };
        let attention = loaded.model.model.layers[0]
            .self_attn
            .as_ref()
            .expect("full attention");
        assert_eq!(attention.q_proj.weight.shape(), &[16, 16]);
        let linear = loaded.model.model.layers[1]
            .linear_attn
            .as_ref()
            .expect("linear attention");
        assert_eq!(linear.in_proj_qkv.weight.shape(), &[12, 16]);
        assert_eq!(linear.in_proj_z.weight.shape(), &[4, 16]);
        assert_eq!(linear.in_proj_b.weight.shape(), &[1, 16]);
        assert_eq!(linear.in_proj_a.weight.shape(), &[1, 16]);
        assert_eq!(linear.conv1d.weight.shape(), &[12, 1, 2]);
        assert_eq!(linear.A_log.shape(), &[1]);
        assert_eq!(linear.dt_bias.shape(), &[1]);
        assert_eq!(linear.out_proj.weight.shape(), &[16, 4]);
        let qwen_hybrid::FeedForward::Moe(moe) = &loaded.model.model.layers[0].mlp else {
            panic!("expected Qwen hybrid MoE")
        };
        assert_eq!(moe.experts.gate_up_proj.shape(), &[4, 16, 16]);
        assert_eq!(moe.experts.down_proj.shape(), &[4, 16, 8]);
    }

    #[test]
    fn qwen3_next_loader_shards_and_splits_fused_linear_attention() {
        let context = safemlx::ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let config = serde_json::json!({
            "model_type": "qwen3_next",
            "vocab_size": 16,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "rms_norm_eps": 0.000001,
            "linear_conv_kernel_dim": 2,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "intermediate_size": 0,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
            "num_experts_per_tok": 2,
            "num_experts": 4,
            "layer_types": ["linear_attention", "full_attention"],
            "tie_word_embeddings": false
        });
        let args = qwen_hybrid::model_args_from_config_value(&config).unwrap();
        let mut source = qwen_hybrid::Model::new(args, None, None, None, stream).unwrap();
        let mut tensors = Vec::new();
        for (name, parameter) in source.parameters_mut().flatten() {
            let canonical = canonical_checkpoint_name(&name);
            if canonical.contains(".mlp.experts.")
                || canonical.starts_with("model.layers.0.linear_attn.in_proj_qkv.")
                || canonical.starts_with("model.layers.0.linear_attn.in_proj_z.")
                || canonical.starts_with("model.layers.0.linear_attn.in_proj_b.")
                || canonical.starts_with("model.layers.0.linear_attn.in_proj_a.")
            {
                continue;
            }
            tensors.push((canonical, filled(parameter.shape(), 0.01, stream)));
        }
        tensors.push((
            "model.layers.0.linear_attn.in_proj_qkvz.weight".into(),
            filled(&[32, 16], 0.01, stream),
        ));
        tensors.push((
            "model.layers.0.linear_attn.in_proj_ba.weight".into(),
            filled(&[4, 16], 0.01, stream),
        ));
        for layer in 0..2 {
            for expert in 0..4 {
                for (projection, shape) in [
                    ("gate_proj", vec![16, 16]),
                    ("up_proj", vec![16, 16]),
                    ("down_proj", vec![16, 16]),
                ] {
                    tensors.push((
                        format!("model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"),
                        filled(&shape, 0.01, stream),
                    ));
                }
            }
        }
        let directory = tempfile::tempdir().unwrap();
        save_fixture(directory.path(), &config, &tensors);
        let loaded =
            load_tensor_parallel_model(directory.path(), topology(2, 0, 2), stream, stream)
                .unwrap();
        let TensorArchitecture::QwenHybrid(loaded) = &loaded.architecture else {
            panic!("expected Qwen3-Next tensor model")
        };
        let linear = loaded.model.model.layers[0]
            .linear_attn
            .as_ref()
            .expect("linear attention");
        assert_eq!(linear.in_proj_qkv.weight.shape(), &[12, 16]);
        assert_eq!(linear.in_proj_z.weight.shape(), &[4, 16]);
        assert_eq!(linear.in_proj_b.weight.shape(), &[1, 16]);
        assert_eq!(linear.in_proj_a.weight.shape(), &[1, 16]);
        assert_eq!(linear.conv1d.weight.shape(), &[12, 1, 2]);
        assert_eq!(linear.A_log.shape(), &[1]);
        assert_eq!(linear.dt_bias.shape(), &[1]);
        let full = loaded.model.model.layers[1]
            .self_attn
            .as_ref()
            .expect("full attention");
        assert_eq!(full.q_proj.weight.shape(), &[16, 16]);
    }
}

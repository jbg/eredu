//! Bounded layer execution for the shared dense-Qwen decoder.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::Path,
    sync::Arc,
    time::Instant,
};

use safemlx::{
    error::Exception,
    module::{Module, Param},
    nn,
    ops::indexing::TryIndexOp,
    ops::{GgufCheckpoint, GgufMetadataValue},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};

use super::{self as resident, DecoderConfig, Experts, FeedForward, TransformerBlock};
use crate::core::cache::{
    PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
    PromptCacheTopology,
};

use crate::{
    api::{
        common::{
            attention::AttentionInput,
            generation::CausalLm,
            linear::{
                build_unloaded_maybe_quantized_lm_head_with_quantization,
                project_logits_maybe_quantized, unloaded_maybe_quantized_embedding,
            },
        },
        input,
    },
    core::cache::LayerCachePolicy,
    error::Error,
    nn::{
        parallel::{
            planned_kv_head_layout, register_gqa_projection_group,
            register_swiglu_projection_group, GqaProjectionNames, SwiGluProjectionNames,
            VocabParallelEmbedding, VocabParallelLmHead,
        },
        tensor::{create_attention_mask, AttentionMask},
    },
    runtime::cache::residency::{CacheResidencyPolicy, CacheResidencyReport, PagedCacheOptions},
    runtime::cache::{
        ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache, SlidingKeyValueCache,
    },
    runtime::checkpoint::binding::{
        build_module_bindings, build_module_bindings_excluding, build_module_bindings_with_recipes,
        populate_module_from_lease, populate_module_from_lease_excluding,
    },
    runtime::checkpoint::store::{GgufWeightStore, TensorSelection, WeightStore},
    runtime::checkpoint::{
        quantization::{should_quantize_on_load, WeightQuantization},
        recipe::DerivedWeightRecipe,
    },
    runtime::distributed::parallel::{
        aligned_partition_units, array_parameter_member, register_replicated_module,
        MemberSharding, ParallelPlanBuilder, ParameterGroupSpec, ParameterRole,
    },
    runtime::execution::layerwise::{
        load_layerwise_model, load_layerwise_model_with_quantization,
        load_safetensors_layerwise_model, load_tensor_parallel_layerwise_model,
        open_safetensors_weight_store, transformed_module_weight_store, ArchitectureAdapter,
        LayerWeightResidency, LayerwiseForwardState, LayerwiseModel, LoadTimeQuantizableAdapter,
        StaticUnitBindings, WeightResidency,
    },
    runtime::residency::expert_cache::{
        ExpertCache, ExpertCacheError, ExpertCacheLoadOptions, ExpertCacheReport,
        ExpertCatalogEntry, ExpertIdentity, ExpertPass, ExpertRouteBatch,
    },
    runtime::residency::manager::{OffloadUnit, ResidencyReport, ResidentUnitLease, WeightBinding},
};

const EMBEDDING_UNIT: &str = "dense_qwen.static.embedding";
const NORM_UNIT: &str = "dense_qwen.static.norm";
const HEAD_UNIT: &str = "dense_qwen.static.output";

/// Architecture-owned KV cache accepted by the canonical dense-Qwen adapter.
pub enum DenseQwenLayerwiseCache {
    /// Append-only device KV caches.
    Concat(Vec<Option<ConcatKeyValueCache>>),
    /// Sliding device KV caches used by expert-parallel execution.
    Sliding(Vec<Option<SlidingKeyValueCache>>),
    /// Paged KV caches used by expert-parallel execution.
    Paged(Vec<Option<PagedKeyValueCache>>),
}

/// Host-backed dense-Qwen causal LM.
pub struct LayerwiseDecoder {
    execution: LayerwiseModel<DenseQwenLayerwiseAdapter>,
}

impl LayerwiseDecoder {
    /// Returns the normalized decoder configuration.
    pub fn args(&self) -> &DecoderConfig {
        self.execution.adapter().args()
    }

    pub(crate) fn bind_parallel_topology(&mut self, topology: crate::ParallelTopology) {
        self.execution.bind_parallel_topology(topology);
    }

    /// Creates one standard device-resident KV cache per decoder block.
    pub fn new_cache(&self) -> Vec<Option<ConcatKeyValueCache>> {
        self.args()
            .attention_schedule
            .iter()
            .map(|policy| {
                Some(match policy.window() {
                    Some(window) => ConcatKeyValueCache::new_for_sliding_attention(
                        i32::try_from(window.get())
                            .expect("validated dense-Qwen attention window fits i32"),
                    ),
                    None => ConcatKeyValueCache::new(),
                })
            })
            .collect()
    }

    /// Creates device-resident or globally budgeted paged KV state without
    /// changing decoder-weight residency.
    pub fn new_cache_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<DenseQwenLayerwiseCache, Error> {
        match policy {
            CacheResidencyPolicy::Device => Ok(DenseQwenLayerwiseCache::Concat(self.new_cache())),
            CacheResidencyPolicy::Paged(options) => {
                let manager = crate::CacheResidencyManager::new(options)
                    .map_err(|error| Exception::custom(error.to_string()))?;
                let caches = resident::new_paged_cache_with_manager(
                    self.args(),
                    manager,
                    self.execution.prompt_cache_rank_identity(),
                )?;
                Ok(DenseQwenLayerwiseCache::Paged(caches))
            }
        }
    }

    /// Returns aggregate live KV paging observations, if paging is enabled.
    pub fn cache_residency_report(
        &self,
        cache: &DenseQwenLayerwiseCache,
    ) -> Result<Option<CacheResidencyReport>, Error> {
        match cache {
            DenseQwenLayerwiseCache::Paged(caches) => caches
                .iter()
                .flatten()
                .next()
                .map(PagedKeyValueCache::report)
                .transpose()
                .map_err(Into::into),
            DenseQwenLayerwiseCache::Concat(_) | DenseQwenLayerwiseCache::Sliding(_) => Ok(None),
        }
    }

    /// Returns rank-local generalized parallel information when applicable.
    pub fn parallel_info(&self) -> Option<&crate::ParallelModelInfo> {
        self.execution.parallel_info()
    }

    /// Returns generalized parameter-residency and memory metadata.
    pub fn residency_metadata(&self) -> &crate::LayerwiseModelMetadata {
        self.execution.metadata()
    }

    /// Returns this rank's exact prompt-cache state layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        self.execution.prompt_cache_layer_layout()
    }

    /// Returns the architecture identity used to validate persisted prompt caches.
    pub fn prompt_cache_architecture_fingerprint(&self) -> String {
        resident::prompt_cache_architecture_fingerprint(self.args())
    }

    /// Returns the complete rank-local prompt-cache identity.
    pub fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        self.execution.prompt_cache_model_identity()
    }

    /// Persists a compatible standard prefix cache.
    pub fn save_prompt_cache(
        &self,
        cache: &mut Vec<Option<ConcatKeyValueCache>>,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        let mut owned = DenseQwenLayerwiseCache::Concat(std::mem::take(cache));
        let result = self.execution.save_prompt_cache(
            &mut owned,
            destination,
            descriptor,
            prefix_token_ids,
            options,
            stream,
        );
        let DenseQwenLayerwiseCache::Concat(owned) = owned else {
            unreachable!("dense-Qwen prompt-cache wrapper changed variants")
        };
        *cache = owned;
        result
    }

    /// Restores a compatible standard prefix cache.
    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(Vec<Option<ConcatKeyValueCache>>, PromptCacheManifest), Error> {
        let (cache, manifest) = self.execution.load_prompt_cache(
            directory,
            expected,
            prefix_token_ids,
            options,
            stream,
        )?;
        let DenseQwenLayerwiseCache::Concat(cache) = cache else {
            return Err(Error::Parallel(
                "dense-Qwen prompt-cache restore returned a non-concat representation".into(),
            ));
        };
        Ok((cache, manifest))
    }

    /// Returns current logical residency and transfer telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        self.execution.residency_report()
    }
    /// Returns dense-stream observations when that policy is active.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<crate::runtime::execution::layerwise::DenseDiskStreamReport>, Error> {
        self.execution.dense_stream_report()
    }

    /// Returns sparse expert-cache telemetry when that residency mode is active.
    pub fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        self.execution
            .adapter()
            .expert_cache
            .as_ref()
            .map(ExpertCache::report)
            .transpose()
            .map_err(Error::from)
    }

    /// Returns the persistent checkpoint store.
    pub fn checkpoint_store(&self) -> &(dyn WeightStore + Send + Sync) {
        self.execution.checkpoint_store()
    }

    /// Runs a rank-local tensor-parallel forward pass through the generalized
    /// execution-group engine.
    pub fn forward_tensor_parallel(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut DenseQwenLayerwiseCache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution.forward_tensor_parallel(
            DenseQwenAdapterInput { inputs, mask },
            cache,
            group,
            stream,
        )
    }

    /// Runs Qwen2/Qwen2.5 or Qwen3 with a standard KV cache.
    pub fn forward(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<ConcatKeyValueCache>>,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let mut owned = DenseQwenLayerwiseCache::Concat(std::mem::take(cache));
        let result =
            self.execution
                .forward(DenseQwenAdapterInput { inputs, mask }, &mut owned, stream);
        let DenseQwenLayerwiseCache::Concat(owned) = owned else {
            unreachable!("dense-Qwen concat cache wrapper changed variants")
        };
        *cache = owned;
        result
    }

    /// Runs dense Qwen through the canonical observer contract.
    pub fn forward_with_observer(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<ConcatKeyValueCache>>,
        stream: &Stream,
        observer: &mut dyn crate::runtime::execution::inspection::ActivationObserver,
    ) -> Result<Array, Error> {
        let mut owned = DenseQwenLayerwiseCache::Concat(std::mem::take(cache));
        let result = self.execution.forward_with_observer(
            DenseQwenAdapterInput { inputs, mask },
            &mut owned,
            stream,
            observer,
        );
        let DenseQwenLayerwiseCache::Concat(owned) = owned else {
            unreachable!("dense-Qwen concat cache wrapper changed variants")
        };
        *cache = owned;
        result
    }

    /// Runs paged dense Qwen through the canonical observer contract.
    pub fn forward_paged_with_observer(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<PagedKeyValueCache>>,
        stream: &Stream,
        observer: &mut dyn crate::runtime::execution::inspection::ActivationObserver,
    ) -> Result<Array, Error> {
        let mut owned = DenseQwenLayerwiseCache::Paged(std::mem::take(cache));
        let result = self.execution.forward_with_observer(
            DenseQwenAdapterInput { inputs, mask },
            &mut owned,
            stream,
            observer,
        );
        let DenseQwenLayerwiseCache::Paged(owned) = owned else {
            unreachable!("dense-Qwen paged cache wrapper changed variants")
        };
        *cache = owned;
        result
    }

    /// Runs Qwen2/Qwen2.5 or Qwen3 with a block-addressable paged KV cache.
    pub fn forward_paged(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<PagedKeyValueCache>>,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let mut owned = DenseQwenLayerwiseCache::Paged(std::mem::take(cache));
        let result =
            self.execution
                .forward(DenseQwenAdapterInput { inputs, mask }, &mut owned, stream);
        let DenseQwenLayerwiseCache::Paged(owned) = owned else {
            unreachable!("dense-Qwen paged cache wrapper changed variants")
        };
        *cache = owned;
        result
    }

    /// Runs streamed layers while delegating routed experts to a caller.
    pub(crate) fn forward_with_expert_executor<F>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<ConcatKeyValueCache>>,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut owned = DenseQwenLayerwiseCache::Concat(std::mem::take(cache));
        let result =
            self.forward_with_expert_executor_cache(inputs, mask, &mut owned, &mut execute, stream);
        let DenseQwenLayerwiseCache::Concat(owned) = owned else {
            unreachable!("dense-Qwen concat cache wrapper changed variants")
        };
        *cache = owned;
        result
    }

    pub(crate) fn forward_with_sliding_expert_executor<F>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<SlidingKeyValueCache>>,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut owned = DenseQwenLayerwiseCache::Sliding(std::mem::take(cache));
        let result =
            self.forward_with_expert_executor_cache(inputs, mask, &mut owned, &mut execute, stream);
        let DenseQwenLayerwiseCache::Sliding(owned) = owned else {
            unreachable!("dense-Qwen sliding cache wrapper changed variants")
        };
        *cache = owned;
        result
    }

    pub(crate) fn forward_with_paged_expert_executor<F>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<PagedKeyValueCache>>,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut owned = DenseQwenLayerwiseCache::Paged(std::mem::take(cache));
        let result =
            self.forward_with_expert_executor_cache(inputs, mask, &mut owned, &mut execute, stream);
        let DenseQwenLayerwiseCache::Paged(owned) = owned else {
            unreachable!("dense-Qwen paged cache wrapper changed variants")
        };
        *cache = owned;
        result
    }

    fn forward_with_expert_executor_cache<F>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut DenseQwenLayerwiseCache,
        execute: &mut F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        self.execution.forward_with_layer_executor(
            DenseQwenAdapterInput { inputs, mask },
            cache,
            stream,
            |_adapter, _group, index, layer, hidden, cache, context, stream| match cache {
                DenseQwenLayerwiseCache::Concat(cache) => forward_sparse_with_executor(
                    layer,
                    hidden,
                    cache[index].as_mut(),
                    context,
                    index,
                    execute,
                    stream,
                ),
                DenseQwenLayerwiseCache::Sliding(cache) => forward_sparse_with_executor(
                    layer,
                    hidden,
                    cache[index].as_mut(),
                    context,
                    index,
                    execute,
                    stream,
                ),
                DenseQwenLayerwiseCache::Paged(cache) => forward_sparse_with_executor(
                    layer,
                    hidden,
                    cache[index].as_mut(),
                    context,
                    index,
                    execute,
                    stream,
                ),
            },
        )
    }

    /// Runs the shared tensor-parallel model while delegating routed experts
    /// to a topology-scoped expert-parallel executor.
    pub(crate) fn forward_tensor_expert_parallel<F>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<ConcatKeyValueCache>>,
        tensor_group: &safemlx::distributed::Group,
        execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut owned = DenseQwenLayerwiseCache::Concat(std::mem::take(cache));
        let result = self.forward_tensor_expert_parallel_cache(
            inputs,
            mask,
            &mut owned,
            tensor_group,
            execute,
            stream,
        );
        let DenseQwenLayerwiseCache::Concat(owned) = owned else {
            unreachable!("dense-Qwen concat cache wrapper changed variants")
        };
        *cache = owned;
        result
    }

    pub(crate) fn forward_tensor_expert_parallel_sliding<F>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<SlidingKeyValueCache>>,
        tensor_group: &safemlx::distributed::Group,
        execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut owned = DenseQwenLayerwiseCache::Sliding(std::mem::take(cache));
        let result = self.forward_tensor_expert_parallel_cache(
            inputs,
            mask,
            &mut owned,
            tensor_group,
            execute,
            stream,
        );
        let DenseQwenLayerwiseCache::Sliding(owned) = owned else {
            unreachable!("dense-Qwen sliding cache wrapper changed variants")
        };
        *cache = owned;
        result
    }

    pub(crate) fn forward_tensor_expert_parallel_paged<F>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<PagedKeyValueCache>>,
        tensor_group: &safemlx::distributed::Group,
        execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut owned = DenseQwenLayerwiseCache::Paged(std::mem::take(cache));
        let result = self.forward_tensor_expert_parallel_cache(
            inputs,
            mask,
            &mut owned,
            tensor_group,
            execute,
            stream,
        );
        let DenseQwenLayerwiseCache::Paged(owned) = owned else {
            unreachable!("dense-Qwen paged cache wrapper changed variants")
        };
        *cache = owned;
        result
    }

    fn forward_tensor_expert_parallel_cache<F>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut DenseQwenLayerwiseCache,
        tensor_group: &safemlx::distributed::Group,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        self.execution.forward_tensor_parallel_with_layer_executor(
            DenseQwenAdapterInput { inputs, mask },
            cache,
            tensor_group,
            stream,
            |_adapter, _group, index, layer, hidden, cache, context, execution| {
                let tp_group = execution.group().ok_or_else(|| {
                    Error::Parallel("TP+EP execution requires an active TP group".into())
                })?;
                match cache {
                    DenseQwenLayerwiseCache::Concat(cache) => forward_sparse_tp_with_executor(
                        layer,
                        hidden,
                        cache[index].as_mut(),
                        context,
                        index,
                        tp_group,
                        &mut execute,
                        execution.stream(),
                    ),
                    DenseQwenLayerwiseCache::Sliding(cache) => forward_sparse_tp_with_executor(
                        layer,
                        hidden,
                        cache[index].as_mut(),
                        context,
                        index,
                        tp_group,
                        &mut execute,
                        execution.stream(),
                    ),
                    DenseQwenLayerwiseCache::Paged(cache) => forward_sparse_tp_with_executor(
                        layer,
                        hidden,
                        cache[index].as_mut(),
                        context,
                        index,
                        tp_group,
                        &mut execute,
                        execution.stream(),
                    ),
                }
            },
        )
    }

    /// Clears temporary device decoder copies.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        self.execution.clear_device_group("text_decoder")
    }
}

fn forward_sparse_with_executor<C, F>(
    layer: &mut TransformerBlock,
    hidden: &Array,
    cache: Option<&mut C>,
    context: &DenseQwenForwardContext,
    index: usize,
    execute: &mut F,
    stream: &Stream,
) -> Result<Array, Error>
where
    C: KeyValueCache,
    F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
{
    Ok(layer.forward_sparse_experts(
        AttentionInput {
            x: hidden,
            mask: context.mask.as_ref(),
            cache,
        },
        stream,
        |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
    )?)
}

#[allow(clippy::too_many_arguments)]
fn forward_sparse_tp_with_executor<C, F>(
    layer: &mut TransformerBlock,
    hidden: &Array,
    cache: Option<&mut C>,
    context: &DenseQwenForwardContext,
    index: usize,
    tensor_group: &safemlx::distributed::Group,
    execute: &mut F,
    stream: &Stream,
) -> Result<Array, Error>
where
    C: KeyValueCache,
    F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
{
    Ok(layer.forward_sparse_experts_tensor_parallel(
        AttentionInput {
            x: hidden,
            mask: context.mask.as_ref(),
            cache,
        },
        tensor_group,
        stream,
        |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
    )?)
}

impl CausalLm<Vec<Option<ConcatKeyValueCache>>> for LayerwiseDecoder {
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Vec<Option<ConcatKeyValueCache>>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        self.forward(&tokens, None, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Vec<Option<ConcatKeyValueCache>>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward(input_tokens, None, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }
}

impl CausalLm<Vec<Option<PagedKeyValueCache>>> for LayerwiseDecoder {
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Vec<Option<PagedKeyValueCache>>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        self.forward_paged(&tokens, None, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Vec<Option<PagedKeyValueCache>>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward_paged(input_tokens, None, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }
}

/// Loads Qwen2/Qwen2.5 or Qwen3 through the generalized residency engine.
pub fn load_safetensors(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseDecoder, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let residency = options.weight_residency();
    let args = resident::load_config(model_dir)?;
    crate::api::structural::validate_safetensors_load_path(
        args.model_kind(),
        model_dir,
        crate::api::ModelLoadOptions::default().with_weight_residency(residency),
    )?;
    let adapter = DenseQwenLayerwiseAdapter::new(args, stream)?;
    Ok(LayerwiseDecoder {
        execution: load_safetensors_layerwise_model(
            model_dir,
            adapter,
            options,
            stream,
            weights_stream,
        )?,
    })
}

pub(crate) fn load_safetensors_quantized_residency(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    quantization: WeightQuantization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseDecoder, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let args = resident::load_config(model_dir)?;
    let store = open_safetensors_weight_store(model_dir, options.max_mapped_shards())?;
    if !should_quantize_on_load(
        "dense Qwen residency",
        args.weight_quantization(),
        quantization,
    )? {
        return Ok(LayerwiseDecoder {
            execution: load_layerwise_model(
                store,
                DenseQwenLayerwiseAdapter::new(args, stream)?,
                options,
                stream,
                weights_stream,
            )?,
        });
    }
    let source_adapter = DenseQwenLayerwiseAdapter::new(args, stream)?;
    Ok(LayerwiseDecoder {
        execution: load_layerwise_model_with_quantization(
            store,
            source_adapter,
            options,
            Some(quantization),
            stream,
            weights_stream,
        )?,
    })
}

pub(crate) fn execute_transformed_model(
    model: resident::Model,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseDecoder, Error> {
    let adapter = DenseQwenLayerwiseAdapter::new(model.args.clone(), stream)?;
    let store = transformed_module_weight_store(&model)?;
    Ok(LayerwiseDecoder {
        execution: load_layerwise_model(
            store,
            adapter,
            LayerWeightResidency::FullyResident,
            stream,
            weights_stream,
        )?,
    })
}

/// Loads Qwen2/3 dense or MoE checkpoints through the generalized
/// tensor-parallel execution-group engine.
pub fn load_tensor_parallel_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    build: crate::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseDecoder, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let residency = options.weight_residency();
    if model_dir
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        let checkpoint = GgufCheckpoint::open(model_dir)?;
        let metadata = crate::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        let architecture = match metadata.get("general.architecture") {
            Some(GgufMetadataValue::String(architecture)) => architecture.as_str(),
            Some(_) => {
                return Err(Error::UnsupportedArchitecture(
                    "GGUF metadata key general.architecture has the wrong type".into(),
                ));
            }
            None => {
                return Err(Error::UnsupportedArchitecture(
                    "GGUF metadata is missing general.architecture".into(),
                ));
            }
        };
        return load_gguf_tensor_parallel_model(
            &checkpoint,
            &metadata,
            architecture,
            options,
            build,
            stream,
            weights_stream,
        )
        .map(|(model, _)| model);
    }
    let args = resident::load_config(model_dir)?;
    crate::api::structural::validate_safetensors_load_path(
        args.model_kind(),
        model_dir,
        crate::api::ModelLoadOptions::default().with_weight_residency(residency),
    )?;
    let adapter = DenseQwenLayerwiseAdapter::new(args, stream)?;
    Ok(LayerwiseDecoder {
        execution: load_tensor_parallel_layerwise_model(
            open_safetensors_weight_store(model_dir, options.max_mapped_shards())?,
            adapter,
            options,
            build,
            stream,
            weights_stream,
        )?,
    })
}

pub(crate) fn load_gguf_tensor_parallel_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    architecture: &str,
    options: LayerWeightResidency,
    build: crate::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(LayerwiseDecoder, Vec<u32>), Error> {
    crate::runtime::execution::layerwise::validate_gguf_layerwise_source(
        checkpoint, metadata, options,
    )?;
    let is_moe = architecture == "qwen3moe";
    let (args, eos_token_ids) =
        resident::prepare_gguf_checkpoint(checkpoint, metadata, architecture, is_moe)?;
    let store: Arc<dyn WeightStore + Send + Sync> =
        Arc::new(GgufWeightStore::new_with_max_mapped_shards(
            checkpoint.clone(),
            move |name| resident::translate_gguf_weight_name(name, is_moe),
            options.max_mapped_shards(),
        )?);
    let execution = load_tensor_parallel_layerwise_model(
        store,
        DenseQwenLayerwiseAdapter::new(args, stream)?,
        options,
        build,
        stream,
        weights_stream,
    )?;
    Ok((LayerwiseDecoder { execution }, eos_token_ids))
}

pub(crate) fn load_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    architecture: &str,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(LayerwiseDecoder, Vec<u32>), Error> {
    let is_moe = architecture == "qwen3moe";
    let (args, eos_token_ids) =
        resident::prepare_gguf_checkpoint(checkpoint, metadata, architecture, is_moe)?;
    let store: Arc<dyn WeightStore + Send + Sync> =
        Arc::new(GgufWeightStore::new_with_max_mapped_shards(
            checkpoint.clone(),
            |name| resident::translate_gguf_weight_name(name, is_moe),
            residency.max_mapped_shards(),
        )?);

    if let Some(expert_options) = residency.expert_cache() {
        return Ok((
            load_qwen3_gguf_sparse_with_store(
                store,
                args,
                expert_options,
                residency.layers(),
                quantization,
                stream,
                weights_stream,
            )?,
            eos_token_ids,
        ));
    }
    let execution = load_layerwise_model_with_quantization(
        store,
        DenseQwenLayerwiseAdapter::new(args, stream)?,
        residency.layers(),
        quantization,
        stream,
        weights_stream,
    )?;
    Ok((LayerwiseDecoder { execution }, eos_token_ids))
}

/// Loads replicated Qwen3-MoE GGUF parameters for sparse expert-parallel
/// execution without materializing routed experts.
fn load_qwen3_gguf_sparse_with_store(
    store: Arc<dyn WeightStore + Send + Sync>,
    args: DecoderConfig,
    options: ExpertCacheLoadOptions,
    non_expert: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseDecoder, Error> {
    if !args.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires a Qwen3 sparse-MoE GGUF checkpoint".into(),
        ));
    }
    let adapter = DenseQwenLayerwiseAdapter::new_external_experts(args.clone(), stream)?;
    let mut execution = load_layerwise_model_with_quantization(
        store,
        adapter,
        non_expert,
        quantization,
        stream,
        weights_stream,
    )?;
    let checkpoint_store = execution.checkpoint_store_arc();
    let entries = qwen3_expert_catalog(&args, checkpoint_store.as_ref())?;
    execution.adapter_mut().expert_cache = Some(match quantization {
        Some(quantization) => ExpertCache::new_quantized_shared(
            checkpoint_store,
            entries,
            options,
            quantization,
            weights_stream.clone(),
            stream.clone(),
        )?,
        None => ExpertCache::new_shared(
            checkpoint_store,
            entries,
            options,
            weights_stream.clone(),
            stream.clone(),
        )?,
    });
    Ok(LayerwiseDecoder { execution })
}

/// Builds the streamed nonexpert Qwen3 execution base used by distributed EP.
pub(crate) fn load_qwen3_sparse_ep_base_with_store(
    store: Arc<dyn WeightStore + Send + Sync>,
    args: DecoderConfig,
    non_expert: impl Into<LayerWeightResidency>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseDecoder, Error> {
    if !args.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "streamed sparse expert parallelism requires Qwen3 MoE".into(),
        ));
    }
    let adapter = DenseQwenLayerwiseAdapter::new_external_experts(args, stream)?;
    let execution = load_layerwise_model(store, adapter, non_expert, stream, weights_stream)?;
    Ok(LayerwiseDecoder { execution })
}

/// Builds the shared TP-sharded nonexpert base used by combined TP+EP.
pub(crate) fn load_qwen3_sparse_tp_ep_base_with_store(
    store: Arc<dyn WeightStore + Send + Sync>,
    args: DecoderConfig,
    non_expert: impl Into<LayerWeightResidency>,
    build: crate::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseDecoder, Error> {
    if !args.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "combined tensor/expert parallelism requires Qwen3 MoE".into(),
        ));
    }
    let adapter = DenseQwenLayerwiseAdapter::new_external_experts(args, stream)?;
    let execution = load_tensor_parallel_layerwise_model(
        store,
        adapter,
        non_expert,
        build,
        stream,
        weights_stream,
    )?;
    Ok(LayerwiseDecoder { execution })
}

/// Loads sparse Qwen3 with independently cached experts and bounded non-expert units.
pub fn load_qwen3_expert_cache_model(
    model_dir: impl AsRef<Path>,
    non_expert: crate::NonExpertWeightResidency,
    options: ExpertCacheLoadOptions,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseDecoder, Error> {
    let model_dir = model_dir.as_ref();
    let args = resident::load_config(model_dir)?;
    if !args.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires a Qwen3 sparse-MoE checkpoint".into(),
        ));
    }
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load(
                "Qwen3 independent expert cache",
                args.weight_quantization(),
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let source_adapter = DenseQwenLayerwiseAdapter::new_external_experts(args.clone(), stream)?;
    let store = open_safetensors_weight_store(model_dir, non_expert.layers().max_mapped_shards())?;
    let mut execution = load_layerwise_model_with_quantization(
        store,
        source_adapter,
        non_expert,
        quantize_on_load,
        stream,
        weights_stream,
    )?;
    let store = execution.checkpoint_store_arc();
    let entries = qwen3_expert_catalog(&args, store.as_ref())?;
    let cache = match quantize_on_load {
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
    execution.adapter_mut().expert_cache = Some(cache);
    Ok(LayerwiseDecoder { execution })
}

/// Dense-Qwen adapter sharing one complete-block execution path.
pub struct DenseQwenLayerwiseAdapter {
    args: DecoderConfig,
    embedding: MaybeQuantized<nn::Embedding>,
    norm: nn::RmsNorm,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_embedding: Option<VocabParallelEmbedding>,
    parallel_lm_head: Option<VocabParallelLmHead>,
    parallel_kv_heads: Option<Vec<i32>>,
    sparse_expert_cache: bool,
    expert_cache: Option<ExpertCache>,
}

impl DenseQwenLayerwiseAdapter {
    /// Creates metadata-only static dense-Qwen modules.
    pub fn new(args: DecoderConfig, stream: &Stream) -> Result<Self, Error> {
        let embedding = unloaded_maybe_quantized_embedding(
            args.vocab_size,
            args.hidden_size,
            args.weight_quantization_for("model.embed_tokens.weight"),
            stream,
        )?;
        let norm =
            nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(build_unloaded_maybe_quantized_lm_head_with_quantization(
                args.hidden_size,
                args.vocab_size,
                args.weight_quantization_for("lm_head.weight"),
                stream,
            )?)
        };
        Ok(Self {
            args,
            embedding,
            norm,
            lm_head,
            parallel_embedding: None,
            parallel_lm_head: None,
            parallel_kv_heads: None,
            sparse_expert_cache: false,
            expert_cache: None,
        })
    }

    pub(crate) fn new_external_experts(
        args: DecoderConfig,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let mut adapter = Self::new(args, stream)?;
        adapter.sparse_expert_cache = true;
        Ok(adapter)
    }

    /// Returns normalized model arguments.
    pub const fn args(&self) -> &DecoderConfig {
        &self.args
    }
}

/// Attention mask shared by every temporary dense-Qwen decoder block.
pub struct DenseQwenForwardContext {
    mask: Option<Array>,
}

/// Dense-Qwen input consumed by the architecture-neutral layerwise engine.
pub struct DenseQwenAdapterInput<'a> {
    inputs: &'a Array,
    mask: Option<&'a Array>,
}

fn replace_qwen_expert_bank(
    layer: &mut TransformerBlock,
    args: &DecoderConfig,
    layer_index: usize,
    experts: i32,
    intermediate: Option<i32>,
    stream: &Stream,
) -> Result<(), Error> {
    let FeedForward::Moe(moe) = &mut layer.mlp else {
        return Err(Error::Parallel(format!(
            "dense Qwen layer {layer_index} is not an MoE layer"
        )));
    };
    let prefix = format!("model.layers.{layer_index}.mlp.experts");
    moe.experts = Experts::new(
        experts,
        args.hidden_size,
        intermediate.unwrap_or(args.moe_intermediate_size),
        args.weight_quantization_for(&format!("{prefix}.gate_up_proj")),
        args.weight_quantization_for(&format!("{prefix}.down_proj")),
        stream,
    )?;
    Ok(())
}

pub(crate) fn register_qwen_layer_parallel_plan(
    planner: &mut ParallelPlanBuilder,
    layer: &TransformerBlock,
    args: &DecoderConfig,
    prefix: &str,
) -> Result<(), Error> {
    let attention = &layer.self_attn;
    register_gqa_projection_group(
        planner,
        &format!("{prefix}.self_attn"),
        GqaProjectionNames {
            query: "q_proj",
            key: "k_proj",
            value: "v_proj",
            output: "o_proj",
        },
        &attention.q_proj,
        &attention.k_proj,
        &attention.v_proj,
        &attention.o_proj,
        attention.n_heads,
        attention.n_kv_heads,
        args.head_dim,
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
        resident::FeedForward::Dense(mlp) => {
            register_swiglu_projection_group(
                planner,
                &format!("{prefix}.mlp"),
                SwiGluProjectionNames {
                    gate: "gate_proj",
                    up: "up_proj",
                    down: "down_proj",
                },
                &mlp.gate_proj,
                &mlp.up_proj,
                &mlp.down_proj,
                args.intermediate_size,
            )?;
        }
        resident::FeedForward::Moe(moe) => {
            register_replicated_module(planner, &moe.gate, &format!("{prefix}.mlp.gate"))?;
            let experts = &moe.experts;
            let intermediate = usize::try_from(experts.intermediate_dim)
                .map_err(|_| Error::Parallel("Qwen expert width exceeds usize".into()))?;
            let down_alignment =
                experts
                    .down_affine
                    .or(experts.down_iquant)
                    .map_or(Ok(1usize), |quantization| {
                        usize::try_from(quantization.group_size()).map_err(|_| {
                            Error::Parallel("Qwen expert quantization group exceeds usize".into())
                        })
                    })?;
            let expert_units = aligned_partition_units(
                &format!("{prefix}.mlp.experts"),
                intermediate,
                1,
                down_alignment,
            )?;
            let segments = vec![0..intermediate, intermediate..2 * intermediate];
            let mut members = vec![array_parameter_member(
                format!("{prefix}.mlp.experts.gate_up_proj"),
                experts.gate_up_proj.as_ref(),
                MemberSharding::PartitionedSegments {
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
                    members.push(array_parameter_member(
                        format!("{prefix}.mlp.experts.{name}"),
                        value,
                        MemberSharding::PartitionedSegments {
                            axis: 1,
                            segments: segments.clone(),
                        },
                    )?);
                }
            }
            members.push(array_parameter_member(
                format!("{prefix}.mlp.experts.down_proj"),
                experts.down_proj.as_ref(),
                MemberSharding::Partitioned { axis: 2 },
            )?);
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
                    members.push(array_parameter_member(
                        format!("{prefix}.mlp.experts.{name}"),
                        value,
                        MemberSharding::Partitioned { axis: 2 },
                    )?);
                }
            }
            planner.register(ParameterGroupSpec::partitioned(
                format!("{prefix}.mlp.experts.intermediate"),
                ParameterRole::ExpertIntermediate,
                expert_units,
                members,
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
    )
}

impl LoadTimeQuantizableAdapter for DenseQwenLayerwiseAdapter {
    fn load_time_quantized(
        &self,
        quantization: WeightQuantization,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let mut args = self.args.clone();
        args.quantization = Some(quantization);
        args.quantization_config = None;
        args.quantized_weight_configs = None;
        if self.sparse_expert_cache {
            Self::new_external_experts(args, stream)
        } else {
            Self::new(args, stream)
        }
    }
}

impl ArchitectureAdapter for DenseQwenLayerwiseAdapter {
    type Input<'a> = DenseQwenAdapterInput<'a>;
    type Cache = DenseQwenLayerwiseCache;
    type Layer = TransformerBlock;
    type ForwardContext = DenseQwenForwardContext;

    fn model_type(&self) -> &str {
        &self.args.model_type
    }

    fn quantization(&self) -> Option<crate::runtime::checkpoint::quantization::WeightQuantization> {
        self.args.quantization.or(self.args.quantization_config)
    }

    fn prompt_cache_model_identity(
        &self,
        topology: Option<crate::ParallelTopology>,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let layer_count = usize::try_from(self.args.num_hidden_layers)
            .map_err(|_| Exception::custom("invalid dense-Qwen cache layer count"))?;
        let local_kv_heads = match topology {
            Some(topology) if topology.is_axis_active(crate::ParallelAxis::Tensor) => {
                self.parallel_kv_heads.clone().ok_or_else(|| {
                    Error::Parallel(
                    "dense-Qwen parallel cache identity requested before local layout configuration"
                        .into(),
                )
                })?
            }
            _ => vec![self.args.num_key_value_heads; layer_count],
        };
        if local_kv_heads.len() != layer_count {
            return Err(Error::Parallel(format!(
                "dense-Qwen parallel cache geometry has {} layers, expected {layer_count}",
                local_kv_heads.len()
            )));
        }
        let layer_layout = crate::LayerSchedule::new(
            layer_count,
            self.args
                .attention_schedule
                .iter()
                .zip(local_kv_heads)
                .map(|(attention, kv_heads)| {
                    LayerCachePolicy::key_value(*attention, kv_heads, self.args.head_dim)
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| Error::Parallel(error.to_string()))?,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(PromptCacheModelIdentity {
            model_family: "dense_qwen".into(),
            effective_model_type: self.args.model_type.clone(),
            architecture_fingerprint: resident::prompt_cache_architecture_fingerprint(&self.args),
            layer_count,
            global_layer_start: 0,
            global_layer_end: layer_count,
            sink_tokens: 0,
            layer_prefix_offsets: vec![0; layer_count],
            topology: topology.map_or_else(
                PromptCacheTopology::default,
                crate::backend::mlx::cache::prompt_cache_topology,
            ),
            layer_layout,
        })
    }

    fn save_prompt_cache(
        &self,
        cache: &mut Self::Cache,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        match cache {
            DenseQwenLayerwiseCache::Concat(cache) => resident::save_prompt_cache(
                &self.args,
                cache,
                destination,
                descriptor,
                prefix_token_ids,
                options,
                stream,
            )
            .map_err(Into::into),
            DenseQwenLayerwiseCache::Paged(caches) => {
                for cache in caches.iter_mut().flatten() {
                    cache.finalize()?;
                }
                caches
                    .iter()
                    .flatten()
                    .next()
                    .ok_or_else(|| Error::Parallel("cannot persist an empty dense-Qwen cache".into()))?
                    .manager()
                    .save_prompt_cache(destination, descriptor, prefix_token_ids, &[], options)
                    .map_err(|error| Error::Parallel(error.to_string()))
            }
            DenseQwenLayerwiseCache::Sliding(_) => Err(Error::Parallel(
                "dense-Qwen sliding-cache persistence is unsupported; use concat or paged cache state".into(),
            )),
        }
    }

    fn load_prompt_cache(
        &self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        _options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(Self::Cache, PromptCacheManifest), Error> {
        let (cache, manifest) = resident::load_prompt_cache_with_identity(
            &self.args,
            directory,
            expected,
            prefix_token_ids,
            identity,
            stream,
        )?;
        Ok((DenseQwenLayerwiseCache::Concat(cache), manifest))
    }

    fn validate_cache(&self, cache: &mut Self::Cache) -> Result<(), Error> {
        let expected = usize::try_from(self.args.num_hidden_layers).map_err(|_| {
            Error::UnsupportedArchitecture(format!(
                "dense-Qwen layer count {} is invalid",
                self.args.num_hidden_layers
            ))
        })?;
        match cache {
            DenseQwenLayerwiseCache::Concat(caches) => {
                if caches.is_empty() {
                    *caches = self
                        .args
                        .attention_schedule
                        .iter()
                        .map(|policy| {
                            Some(match policy.window() {
                                Some(window) => ConcatKeyValueCache::new_for_sliding_attention(
                                    i32::try_from(window.get())
                                        .expect("validated dense-Qwen attention window fits i32"),
                                ),
                                None => ConcatKeyValueCache::new(),
                            })
                        })
                        .collect();
                }
                validate_dense_qwen_cache(caches, expected)
            }
            DenseQwenLayerwiseCache::Sliding(caches) => validate_dense_qwen_cache(caches, expected),
            DenseQwenLayerwiseCache::Paged(caches) => validate_dense_qwen_cache(caches, expected),
        }
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        let hidden = self.embedding.forward(input.inputs, stream)?;
        let mask = match cache {
            DenseQwenLayerwiseCache::Concat(caches) => {
                dense_qwen_attention_mask(&hidden, input.mask, caches, stream)?
            }
            DenseQwenLayerwiseCache::Sliding(caches) => {
                dense_qwen_attention_mask(&hidden, input.mask, caches, stream)?
            }
            DenseQwenLayerwiseCache::Paged(caches) => {
                dense_qwen_attention_mask(&hidden, input.mask, caches, stream)?
            }
        };
        Ok(LayerwiseForwardState {
            hidden,
            context: DenseQwenForwardContext { mask },
        })
    }

    fn begin_forward_with_execution<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Self::Cache,
        execution: &crate::runtime::distributed::parallel::ParallelExecutionContext<'_>,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        let Some(embedding) = &mut self.parallel_embedding else {
            return self.begin_forward(input, cache, execution.stream());
        };
        let hidden = embedding.forward(input.inputs, execution)?;
        let mask = match cache {
            DenseQwenLayerwiseCache::Concat(caches) => {
                dense_qwen_attention_mask(&hidden, input.mask, caches, execution.stream())?
            }
            DenseQwenLayerwiseCache::Sliding(caches) => {
                dense_qwen_attention_mask(&hidden, input.mask, caches, execution.stream())?
            }
            DenseQwenLayerwiseCache::Paged(caches) => {
                dense_qwen_attention_mask(&hidden, input.mask, caches, execution.stream())?
            }
        };
        Ok(LayerwiseForwardState {
            hidden,
            context: DenseQwenForwardContext { mask },
        })
    }

    fn execution_graph(
        &self,
    ) -> Result<crate::runtime::execution::layerwise::ExecutionGroupDag, Error> {
        crate::runtime::execution::layerwise::ExecutionGroupDag::chain(["text_decoder"])
    }

    fn layer_count(&self, group: usize) -> Result<usize, Error> {
        if group != 0 {
            return Err(Error::UnsupportedArchitecture(format!(
                "dense Qwen has no group {group}"
            )));
        }
        usize::try_from(self.args.num_hidden_layers).map_err(|_| {
            Error::UnsupportedArchitecture(format!(
                "dense-Qwen layer count {} is invalid",
                self.args.num_hidden_layers
            ))
        })
    }

    fn static_units(&self, store: &dyn WeightStore) -> Result<Vec<StaticUnitBindings>, Error> {
        self.selected_static_units(store, &|_| true)
    }

    fn selected_static_units(
        &self,
        store: &dyn WeightStore,
        select: &dyn Fn(&str) -> bool,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        let mut units = Vec::new();
        if select(EMBEDDING_UNIT) {
            units.push(StaticUnitBindings::new(
                EMBEDDING_UNIT,
                build_module_bindings(&self.embedding, "model.embed_tokens", store)?,
            )?);
        }
        if select(NORM_UNIT) {
            units.push(StaticUnitBindings::new(
                NORM_UNIT,
                build_module_bindings(&self.norm, "model.norm", store)?,
            )?);
        }
        if select(HEAD_UNIT) {
            if let Some(head) = &self.lm_head {
                units.push(StaticUnitBindings::new(
                    HEAD_UNIT,
                    build_module_bindings(head, "lm_head", store)?,
                )?);
            }
        }
        Ok(units)
    }

    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error> {
        let expected = if self.lm_head.is_some() { 3 } else { 2 };
        if leases.len() != expected {
            return Err(Error::UnsupportedArchitecture(format!(
                "dense-Qwen adapter received {} static leases, expected {expected}",
                leases.len()
            )));
        }
        if let Some(embedding) = &mut self.parallel_embedding {
            populate_module_from_lease(embedding.inner_mut(), &leases[0])?;
        } else {
            populate_module_from_lease(&mut self.embedding, &leases[0])?;
        }
        populate_module_from_lease(&mut self.norm, &leases[1])?;
        if let Some(head) = &mut self.parallel_lm_head {
            populate_module_from_lease(head.inner_mut(), &leases[2])?;
        } else if let Some(head) = &mut self.lm_head {
            populate_module_from_lease(head, &leases[2])?;
        }
        Ok(())
    }

    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        if group != 0 {
            return Err(Error::UnsupportedArchitecture(format!(
                "dense Qwen has no group {group}"
            )));
        }
        let index = i32::try_from(index).map_err(|_| {
            Error::UnsupportedArchitecture("dense-Qwen layer index exceeds i32".into())
        })?;
        let mut layer = TransformerBlock::new_for_layer(&self.args, index, stream)?;
        if self.sparse_expert_cache {
            replace_qwen_expert_bank(&mut layer, &self.args, index as usize, 0, None, stream)?;
        }
        Ok(layer)
    }

    fn register_parallel_parameters(
        &self,
        _context: crate::runtime::distributed::parallel::ParallelBuildContext,
        planner: &mut crate::runtime::distributed::parallel::ParallelPlanBuilder,
        stream: &Stream,
    ) -> Result<(), Error> {
        planner.register(crate::nn::parallel::vocab_embedding_parameter_group(
            &self.embedding,
            "model.embed_tokens",
            self.args.vocab_size as usize,
            self.args.hidden_size,
            false,
        )?)?;
        crate::nn::parallel::register_replicated_parameter_group(
            planner,
            &self.norm,
            "model.norm",
        )?;
        if let Some(head) = &self.lm_head {
            planner.register(crate::nn::parallel::vocab_lm_head_parameter_group(
                head,
                "lm_head",
                self.args.hidden_size,
                self.args.vocab_size as usize,
                false,
            )?)?;
        }
        for index in 0..self.args.num_hidden_layers as usize {
            let layer = TransformerBlock::new_for_layer(&self.args, index as i32, stream)?;
            register_qwen_layer_parallel_plan(
                planner,
                &layer,
                &self.args,
                &format!("model.layers.{index}"),
            )?;
        }
        Ok(())
    }

    fn configure_parallel_static(
        &mut self,
        context: crate::runtime::distributed::parallel::ParallelBuildContext,
        layout: &crate::runtime::distributed::parallel::LocalModelLayout,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.parallel_kv_heads = Some(planned_kv_head_layout(
            layout,
            self.args.num_hidden_layers as usize,
            self.args.head_dim,
            "model.layers",
        )?);
        self.parallel_embedding = Some(VocabParallelEmbedding::unloaded(
            self.args.vocab_size as usize,
            self.args.hidden_size,
            self.args
                .weight_quantization_for("model.embed_tokens.weight"),
            context,
            stream,
        )?);
        if self.lm_head.is_some() {
            self.parallel_lm_head = Some(VocabParallelLmHead::unloaded(
                self.args.hidden_size,
                self.args.vocab_size as usize,
                self.args.weight_quantization_for("lm_head.weight"),
                context,
                stream,
            )?);
        }
        Ok(())
    }

    fn new_parallel_layer(
        &self,
        group: usize,
        index: usize,
        layout: &crate::runtime::distributed::parallel::LocalModelLayout,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        if group != 0 {
            return Err(Error::UnsupportedArchitecture(format!(
                "dense Qwen has no group {group}"
            )));
        }
        let prefix = format!("model.layers.{index}");
        let tensor = |suffix: &str| {
            layout
                .tensor(&format!("{prefix}.{suffix}.weight"))
                .or_else(|| layout.tensor(&format!("{prefix}.{suffix}.inner.weight")))
        };
        let q = tensor("self_attn.q_proj")
            .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} query")))?;
        let k = tensor("self_attn.k_proj")
            .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} key")))?;
        let mut args = self.args.clone();
        args.num_attention_heads = q.local_shape()[0] as i32 / args.head_dim;
        args.num_key_value_heads = k.local_shape()[0] as i32 / args.head_dim;
        if args.is_moe() {
            let expert = layout
                .tensor(&format!("{prefix}.mlp.experts.gate_up_proj"))
                .ok_or_else(|| {
                    Error::Parallel(format!("missing TP layout for {prefix} experts"))
                })?;
            args.moe_intermediate_size = expert.local_shape()[1] as i32 / 2;
        } else {
            let gate = tensor("mlp.gate_proj")
                .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} MLP")))?;
            args.intermediate_size = gate.local_shape()[0] as i32;
        }
        let mut layer = TransformerBlock::new_for_layer(&args, index as i32, stream)?;
        if self.sparse_expert_cache {
            replace_qwen_expert_bank(
                &mut layer,
                &self.args,
                index,
                0,
                Some(args.moe_intermediate_size),
                stream,
            )?;
        }
        Ok(layer)
    }

    fn new_expert_parallel_layer(
        &self,
        group: usize,
        index: usize,
        assignment: &crate::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        if !self.args.is_moe() {
            return Err(Error::Parallel(
                "dense Qwen has no routed experts for expert-local layer construction".into(),
            ));
        }
        let mut layer = self.new_layer(group, index, stream)?;
        let local_experts = if self.sparse_expert_cache {
            0
        } else {
            i32::try_from(assignment.local_global_expert_ids().len())
                .map_err(|_| Error::Parallel("local Qwen expert count exceeds i32".into()))?
        };
        let FeedForward::Moe(moe) = &mut layer.mlp else {
            return Err(Error::Parallel(format!(
                "dense Qwen layer {index} is not an MoE layer"
            )));
        };
        moe.experts = Experts::new(
            local_experts,
            self.args.hidden_size,
            self.args.moe_intermediate_size,
            self.args
                .weight_quantization_for(&format!("model.layers.{index}.mlp.experts.gate_up_proj")),
            self.args
                .weight_quantization_for(&format!("model.layers.{index}.mlp.experts.down_proj")),
            stream,
        )?;
        Ok(layer)
    }

    fn new_tensor_expert_parallel_layer(
        &self,
        group: usize,
        index: usize,
        layout: &crate::runtime::distributed::parallel::LocalModelLayout,
        assignment: &crate::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        if !self.args.is_moe() {
            return Err(Error::Parallel(
                "dense Qwen has no routed experts for combined tensor/expert layer construction"
                    .into(),
            ));
        }
        let mut layer = self.new_parallel_layer(group, index, layout, stream)?;
        let local_experts = if self.sparse_expert_cache {
            0
        } else {
            i32::try_from(assignment.local_global_expert_ids().len())
                .map_err(|_| Error::Parallel("local Qwen expert count exceeds i32".into()))?
        };
        let FeedForward::Moe(moe) = &mut layer.mlp else {
            return Err(Error::Parallel(format!(
                "dense Qwen layer {index} is not an MoE layer"
            )));
        };
        let local_intermediate = moe.experts.intermediate_dim;
        moe.experts = Experts::new(
            local_experts,
            self.args.hidden_size,
            local_intermediate,
            self.args
                .weight_quantization_for(&format!("model.layers.{index}.mlp.experts.gate_up_proj")),
            self.args
                .weight_quantization_for(&format!("model.layers.{index}.mlp.experts.down_proj")),
            stream,
        )?;
        Ok(layer)
    }

    fn expert_parallel_assignment(
        &self,
        topology: crate::runtime::distributed::topology::ParallelTopology,
    ) -> Result<Option<crate::runtime::distributed::expert::ExpertAssignment>, Error> {
        if topology.expert_parallel_size == 1 && !self.sparse_expert_cache {
            return Ok(None);
        }
        if !self.args.is_moe() {
            return Err(Error::Parallel(
                "dense Qwen has no routed experts for expert-parallel ownership".into(),
            ));
        }
        Ok(Some(
            crate::runtime::distributed::expert::ExpertAssignment::balanced(
                self.args.num_experts as usize,
                topology.expert_parallel_size,
                topology.expert_parallel_rank,
            )?,
        ))
    }

    fn layer_checkpoint_prefix(&self, _group: usize, index: usize) -> String {
        format!("model.layers.{index}")
    }

    fn layer_unit_name(&self, _group: usize, index: usize) -> String {
        format!("dense_qwen.layer.{index:05}")
    }

    fn populate_layer(
        &self,
        _group: usize,
        _index: usize,
        layer: &mut Self::Layer,
        lease: &ResidentUnitLease,
    ) -> Result<(), Error> {
        if self.sparse_expert_cache {
            Ok(populate_module_from_lease_excluding(
                layer,
                lease,
                |name| name.starts_with("mlp.experts."),
            )?)
        } else {
            Ok(populate_module_from_lease(layer, lease)?)
        }
    }

    fn layer_bindings(
        &self,
        _group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn WeightStore,
    ) -> Result<Vec<WeightBinding>, Error> {
        qwen_text_layer_bindings(
            layer,
            &self.args,
            &format!("model.layers.{index}"),
            store,
            self.sparse_expert_cache,
        )
    }

    fn parallel_layer_bindings(
        &self,
        group: usize,
        index: usize,
        _layer: &Self::Layer,
        store: &dyn WeightStore,
        layout: &crate::runtime::distributed::parallel::LocalModelLayout,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        let global = self.new_layer(group, index, stream)?;
        crate::runtime::execution::layerwise::shard_layer_bindings(
            self.layer_bindings(group, index, &global, store)?,
            &self.layer_checkpoint_prefix(group, index),
            store,
            layout,
        )
    }

    fn expert_parallel_layer_bindings(
        &self,
        group: usize,
        index: usize,
        _layer: &Self::Layer,
        store: &dyn WeightStore,
        assignment: &crate::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        let global = self.new_layer(group, index, stream)?;
        let indices = assignment.local_global_expert_ids().to_vec();
        self.layer_bindings(group, index, &global, store)?
            .into_iter()
            .map(|binding| {
                let target = binding.logical_target().unwrap_or_else(|| binding.name());
                if target.contains(".experts.") {
                    binding
                        .select_bounded_output(
                            store,
                            TensorSelection::Indices {
                                axis: 0,
                                indices: indices.clone(),
                            },
                        )
                        .map_err(Error::from)
                } else {
                    Ok(binding)
                }
            })
            .collect()
    }

    fn additional_consumed_checkpoint_keys(&self, store: &dyn WeightStore) -> Vec<String> {
        if self.sparse_expert_cache {
            store
                .keys()
                .into_iter()
                .filter(|key| key.contains(".mlp.experts."))
                .collect()
        } else {
            Vec::new()
        }
    }

    fn forward_layer(
        &mut self,
        _group: usize,
        index: usize,
        layer: &mut Self::Layer,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match cache {
            DenseQwenLayerwiseCache::Concat(caches) => self.forward_cached_layer(
                index,
                layer,
                hidden,
                caches[index].as_mut().expect("validated dense-Qwen cache"),
                context,
                stream,
            ),
            DenseQwenLayerwiseCache::Sliding(caches) => self.forward_cached_layer(
                index,
                layer,
                hidden,
                caches[index].as_mut().expect("validated dense-Qwen cache"),
                context,
                stream,
            ),
            DenseQwenLayerwiseCache::Paged(caches) => self.forward_cached_layer(
                index,
                layer,
                hidden,
                caches[index].as_mut().expect("validated dense-Qwen cache"),
                context,
                stream,
            ),
        }
    }

    fn forward_layer_with_observer<O: crate::runtime::execution::inspection::ActivationObserver>(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Self::Layer,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        stream: &Stream,
        observer: &mut O,
    ) -> Result<Array, Error> {
        if self.sparse_expert_cache {
            let prefix = self.layer_checkpoint_prefix(group, index);
            observer.observe(&format!("{prefix}.input"), hidden)?;
            let output = self.forward_layer(group, index, layer, hidden, cache, context, stream)?;
            observer.observe(&format!("{prefix}.output"), &output)?;
            return Ok(observer
                .intervene(&format!("{prefix}.output"), &output)?
                .unwrap_or(output));
        }
        let prefix = self.layer_checkpoint_prefix(group, index);
        Ok(match cache {
            DenseQwenLayerwiseCache::Concat(caches) => layer.forward_with_observer(
                AttentionInput {
                    x: hidden,
                    mask: context.mask.as_ref(),
                    cache: Some(caches[index].as_mut().expect("validated dense-Qwen cache")),
                },
                stream,
                &prefix,
                observer,
            )?,
            DenseQwenLayerwiseCache::Sliding(caches) => layer.forward_with_observer(
                AttentionInput {
                    x: hidden,
                    mask: context.mask.as_ref(),
                    cache: Some(caches[index].as_mut().expect("validated dense-Qwen cache")),
                },
                stream,
                &prefix,
                observer,
            )?,
            DenseQwenLayerwiseCache::Paged(caches) => layer.forward_with_observer(
                AttentionInput {
                    x: hidden,
                    mask: context.mask.as_ref(),
                    cache: Some(caches[index].as_mut().expect("validated dense-Qwen cache")),
                },
                stream,
                &prefix,
                observer,
            )?,
        })
    }

    fn forward_layer_with_execution(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Self::Layer,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        execution: &crate::runtime::distributed::parallel::ParallelExecutionContext<'_>,
    ) -> Result<Array, Error> {
        let Some(tp_group) = execution.group() else {
            return self.forward_layer(
                group,
                index,
                layer,
                hidden,
                cache,
                context,
                execution.stream(),
            );
        };
        match cache {
            DenseQwenLayerwiseCache::Concat(caches) => Ok(layer.forward_tensor_parallel(
                hidden,
                context.mask.as_ref(),
                caches[index].as_mut(),
                tp_group,
                execution.stream(),
            )?),
            DenseQwenLayerwiseCache::Sliding(caches) => Ok(layer.forward_tensor_parallel(
                hidden,
                context.mask.as_ref(),
                caches[index].as_mut(),
                tp_group,
                execution.stream(),
            )?),
            DenseQwenLayerwiseCache::Paged(caches) => Ok(layer.forward_tensor_parallel(
                hidden,
                context.mask.as_ref(),
                caches[index].as_mut(),
                tp_group,
                execution.stream(),
            )?),
        }
    }

    fn retained_arrays<'a>(
        &self,
        cache: &'a Self::Cache,
        _group: usize,
        index: usize,
    ) -> Vec<&'a Array> {
        match cache {
            DenseQwenLayerwiseCache::Concat(caches) => retained_cache_arrays(caches, index),
            DenseQwenLayerwiseCache::Sliding(caches) => retained_cache_arrays(caches, index),
            DenseQwenLayerwiseCache::Paged(caches) => retained_cache_arrays(caches, index),
        }
    }

    fn finish(
        &mut self,
        hidden: &Array,
        _cache: &mut Self::Cache,
        _context: &Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let hidden = self.norm.forward(hidden, stream)?;
        Ok(project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.embedding,
            &hidden,
            stream,
        )?)
    }

    fn finish_with_execution(
        &mut self,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &Self::ForwardContext,
        execution: &crate::runtime::distributed::parallel::ParallelExecutionContext<'_>,
    ) -> Result<Array, Error> {
        let Some(embedding) = &mut self.parallel_embedding else {
            return self.finish(hidden, cache, context, execution.stream());
        };
        let hidden = self.norm.forward(hidden, execution.stream())?;
        let logits = match &mut self.parallel_lm_head {
            Some(head) => head.forward(&hidden, execution)?,
            None => embedding.project_logits(&hidden, execution)?,
        };
        logits.all_gather(execution)
    }
}

impl DenseQwenLayerwiseAdapter {
    fn forward_cached_layer<C: KeyValueCache>(
        &self,
        index: usize,
        layer: &mut TransformerBlock,
        hidden: &Array,
        cache: &mut C,
        context: &DenseQwenForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if self.sparse_expert_cache {
            let expert_cache = self.expert_cache.as_ref().ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "Qwen3 sparse expert cache was not initialized".into(),
                )
            })?;
            let pass = if hidden.dim(1) > 1 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            };
            return Ok(layer.forward_sparse_experts(
                AttentionInput {
                    x: hidden,
                    mask: context.mask.as_ref(),
                    cache: Some(cache),
                },
                stream,
                |flat, indices, weights, stream| {
                    expert_cache
                        .execute_routes_bounded(
                            ExpertRouteBatch::new(index, flat, indices, weights, pass),
                            stream,
                            |flat, acquired, weights, stream| {
                                if acquired.is_empty() {
                                    return Err(ExpertCacheError::EmptyRoutedBank {
                                        architecture: "Qwen3",
                                    });
                                }
                                let started = Instant::now();
                                let prefix = format!("model.layers.{index}.mlp.experts");
                                let load_time = expert_cache.weight_quantization();
                                let mut bank = resident::Experts::new(
                                    acquired.identities().len() as i32,
                                    self.args.hidden_size,
                                    self.args.moe_intermediate_size,
                                    load_time.or_else(|| {
                                        self.args.weight_quantization_for(&format!(
                                            "{prefix}.gate_up_proj"
                                        ))
                                    }),
                                    load_time.or_else(|| {
                                        self.args
                                            .weight_quantization_for(&format!("{prefix}.down_proj"))
                                    }),
                                    stream,
                                )?;
                                bank.gate_up_proj = Param::new(
                                    acquired
                                        .compact_binding("gate_up_proj", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                bank.gate_up_proj_scales = Param::new(
                                    acquired
                                        .optional_compact_binding("gate_up_proj_scales", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                bank.gate_up_proj_biases = Param::new(
                                    acquired
                                        .optional_compact_binding("gate_up_proj_biases", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                bank.down_proj = Param::new(
                                    acquired
                                        .compact_binding("down_proj", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                bank.down_proj_scales = Param::new(
                                    acquired
                                        .optional_compact_binding("down_proj_scales", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                bank.down_proj_biases = Param::new(
                                    acquired
                                        .optional_compact_binding("down_proj_biases", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                expert_cache.record_compact_bank(
                                    pass,
                                    acquired.scratch_bytes(),
                                    started.elapsed(),
                                )?;
                                Ok(bank.forward(
                                    flat,
                                    acquired.compact_routes(),
                                    weights,
                                    stream,
                                )?)
                            },
                        )
                        .map_err(|error| Exception::custom(error.to_string()))
                },
            )?);
        }
        Ok(layer.forward(
            AttentionInput {
                x: hidden,
                mask: context.mask.as_ref(),
                cache: Some(cache),
            },
            stream,
        )?)
    }
}

fn validate_dense_qwen_cache<C: KeyValueCache>(
    caches: &[Option<C>],
    expected: usize,
) -> Result<(), Error> {
    if caches.len() != expected {
        return Err(Exception::custom(format!(
            "dense-Qwen cache has {} layers, expected {expected}",
            caches.len()
        ))
        .into());
    }
    for (index, cache) in caches.iter().enumerate() {
        cache.as_ref().ok_or_else(|| {
            Exception::custom(format!("dense-Qwen cache is missing layer {index}"))
        })?;
    }
    Ok(())
}

fn dense_qwen_attention_mask<C: KeyValueCache>(
    hidden: &Array,
    explicit: Option<&Array>,
    caches: &[Option<C>],
    stream: &Stream,
) -> Result<Option<Array>, Error> {
    if let Some(mask) = explicit {
        return Ok(Some(mask.clone()));
    }
    match create_attention_mask(hidden, caches, Some(true), stream)? {
        Some(AttentionMask::Array(mask)) => Ok(Some(mask)),
        Some(AttentionMask::Causal) => Err(Error::UnsupportedArchitecture(
            "dense-Qwen layerwise execution requires an array attention mask".into(),
        )),
        None => Ok(None),
    }
}

fn retained_cache_arrays<C: KeyValueCache>(caches: &[Option<C>], index: usize) -> Vec<&Array> {
    caches[index]
        .as_ref()
        .map(KeyValueCache::retained_arrays)
        .unwrap_or_default()
}

pub(crate) fn qwen_text_layer_bindings(
    layer: &TransformerBlock,
    args: &DecoderConfig,
    prefix: &str,
    store: &dyn WeightStore,
    external_experts: bool,
) -> Result<Vec<WeightBinding>, Error> {
    if external_experts {
        return Ok(build_module_bindings_excluding(
            layer,
            prefix,
            store,
            |name| name.starts_with("mlp.experts."),
        )?);
    }
    let expert_prefix = format!("{prefix}.mlp.experts");
    let keys = store.keys().into_iter().collect::<BTreeSet<_>>();
    let mut recipes = BTreeMap::new();
    if keys.contains(&format!("{expert_prefix}.gate_proj"))
        && keys.contains(&format!("{expert_prefix}.up_proj"))
    {
        recipes.insert(
            "mlp.experts.gate_up_proj".to_string(),
            DerivedWeightRecipe::Concatenate {
                axis: 1,
                inputs: vec![
                    DerivedWeightRecipe::source(
                        format!("{expert_prefix}.gate_proj"),
                        TensorSelection::Full,
                    ),
                    DerivedWeightRecipe::source(
                        format!("{expert_prefix}.up_proj"),
                        TensorSelection::Full,
                    ),
                ],
            },
        );
        for suffix in ["_scales", "_biases"] {
            let gate = format!("{expert_prefix}.gate_proj{suffix}");
            let up = format!("{expert_prefix}.up_proj{suffix}");
            if keys.contains(&gate) && keys.contains(&up) {
                recipes.insert(
                    format!("mlp.experts.gate_up_proj{suffix}"),
                    DerivedWeightRecipe::Concatenate {
                        axis: 1,
                        inputs: vec![
                            DerivedWeightRecipe::source(gate, TensorSelection::Full),
                            DerivedWeightRecipe::source(up, TensorSelection::Full),
                        ],
                    },
                );
            }
        }
    } else if args.is_moe() && !keys.contains(&format!("{expert_prefix}.gate_up_proj")) {
        let mut gate_up = Vec::with_capacity(args.num_experts as usize);
        let mut down = Vec::with_capacity(args.num_experts as usize);
        for expert in 0..args.num_experts as usize {
            let gate = split_expert_key(&keys, &expert_prefix, expert, &["gate_proj", "w1"])?;
            let up = split_expert_key(&keys, &expert_prefix, expert, &["up_proj", "w3"])?;
            let down_key = split_expert_key(&keys, &expert_prefix, expert, &["down_proj", "w2"])?;
            gate_up.push(DerivedWeightRecipe::Concatenate {
                axis: 0,
                inputs: vec![
                    DerivedWeightRecipe::source(gate, TensorSelection::Full),
                    DerivedWeightRecipe::source(up, TensorSelection::Full),
                ],
            });
            down.push(DerivedWeightRecipe::source(down_key, TensorSelection::Full));
        }
        recipes.insert(
            "mlp.experts.gate_up_proj".to_string(),
            DerivedWeightRecipe::Stack {
                axis: 0,
                inputs: gate_up,
            },
        );
        recipes.insert(
            "mlp.experts.down_proj".to_string(),
            DerivedWeightRecipe::Stack {
                axis: 0,
                inputs: down,
            },
        );
    }
    Ok(build_module_bindings_with_recipes(
        layer, prefix, store, recipes,
    )?)
}

pub(crate) fn qwen3_expert_catalog(
    args: &DecoderConfig,
    store: &dyn WeightStore,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    qwen3_expert_catalog_cartesian(args, store, "model.layers", None)
}

pub(crate) fn qwen3_expert_catalog_at(
    args: &DecoderConfig,
    store: &dyn WeightStore,
    layer_root: &str,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    qwen3_expert_catalog_cartesian(args, store, layer_root, None)
}

/// Builds expert-granular bindings under an optional TP semantic layout.
///
/// Expert-axis selection is applied by the catalog recipe first; TP selection
/// is then composed over each expert's output geometry. This preserves atomic
/// expert caching while avoiding a full expert copy on every TP coordinate.
pub(crate) fn qwen3_expert_catalog_cartesian(
    args: &DecoderConfig,
    store: &dyn WeightStore,
    layer_root: &str,
    layout: Option<&crate::runtime::distributed::parallel::LocalModelLayout>,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let keys = store.keys().into_iter().collect::<BTreeSet<_>>();
    let mut entries = Vec::new();
    for layer in 0..usize::try_from(args.num_hidden_layers)
        .map_err(|_| Error::UnsupportedArchitecture("Qwen3 layer count is negative".into()))?
    {
        let prefix = format!("{layer_root}.{layer}.mlp.experts");
        let packed_gate_up = format!("{prefix}.gate_up_proj");
        let packed_down = format!("{prefix}.down_proj");
        for expert in 0..usize::try_from(args.num_experts)
            .map_err(|_| Error::UnsupportedArchitecture("Qwen3 expert count is negative".into()))?
        {
            let identity = ExpertIdentity::new(layer, expert);
            let mut bindings = Vec::new();
            if keys.contains(&packed_gate_up) && keys.contains(&packed_down) {
                for (name, key) in [
                    ("gate_up_proj", packed_gate_up.clone()),
                    ("down_proj", packed_down.clone()),
                ] {
                    bindings.push(recipe_binding(
                        name,
                        DerivedWeightRecipe::source(
                            key,
                            TensorSelection::Range {
                                axis: 0,
                                start: expert,
                                end: expert + 1,
                            },
                        ),
                        store,
                    )?);
                }
                for (name, key) in [
                    ("gate_up_proj_scales", format!("{packed_gate_up}_scales")),
                    ("gate_up_proj_biases", format!("{packed_gate_up}_biases")),
                    ("down_proj_scales", format!("{packed_down}_scales")),
                    ("down_proj_biases", format!("{packed_down}_biases")),
                ] {
                    if keys.contains(&key) {
                        bindings.push(recipe_binding(
                            name,
                            DerivedWeightRecipe::source(
                                key,
                                TensorSelection::Range {
                                    axis: 0,
                                    start: expert,
                                    end: expert + 1,
                                },
                            ),
                            store,
                        )?);
                    }
                }
            } else if keys.contains(&format!("{prefix}.gate_proj"))
                && keys.contains(&format!("{prefix}.up_proj"))
                && keys.contains(&packed_down)
            {
                let selection = TensorSelection::Range {
                    axis: 0,
                    start: expert,
                    end: expert + 1,
                };
                bindings.push(recipe_binding(
                    "gate_up_proj",
                    DerivedWeightRecipe::Concatenate {
                        axis: 1,
                        inputs: vec![
                            DerivedWeightRecipe::source(
                                format!("{prefix}.gate_proj"),
                                selection.clone(),
                            ),
                            DerivedWeightRecipe::source(
                                format!("{prefix}.up_proj"),
                                selection.clone(),
                            ),
                        ],
                    },
                    store,
                )?);
                bindings.push(recipe_binding(
                    "down_proj",
                    DerivedWeightRecipe::source(packed_down.clone(), selection.clone()),
                    store,
                )?);
                for suffix in ["_scales", "_biases"] {
                    let gate = format!("{prefix}.gate_proj{suffix}");
                    let up = format!("{prefix}.up_proj{suffix}");
                    if keys.contains(&gate) && keys.contains(&up) {
                        bindings.push(recipe_binding(
                            &format!("gate_up_proj{suffix}"),
                            DerivedWeightRecipe::Concatenate {
                                axis: 1,
                                inputs: vec![
                                    DerivedWeightRecipe::source(gate, selection.clone()),
                                    DerivedWeightRecipe::source(up, selection.clone()),
                                ],
                            },
                            store,
                        )?);
                    }
                    let down = format!("{packed_down}{suffix}");
                    if keys.contains(&down) {
                        bindings.push(recipe_binding(
                            &format!("down_proj{suffix}"),
                            DerivedWeightRecipe::source(down, selection.clone()),
                            store,
                        )?);
                    }
                }
            } else {
                if args
                    .weight_quantization_for(&format!("{prefix}.gate_up_proj"))
                    .is_some()
                    || args
                        .weight_quantization_for(&format!("{prefix}.down_proj"))
                        .is_some()
                {
                    return Err(Error::Quantization(
                        "split Qwen3 experts cannot be lazily load-time quantized; use checkpoint-native packed expert weights"
                            .into(),
                    ));
                }
                let gate = split_expert_key(&keys, &prefix, expert, &["gate_proj", "w1"])?;
                let up = split_expert_key(&keys, &prefix, expert, &["up_proj", "w3"])?;
                let down = split_expert_key(&keys, &prefix, expert, &["down_proj", "w2"])?;
                bindings.push(recipe_binding(
                    "gate_up_proj",
                    DerivedWeightRecipe::Stack {
                        axis: 0,
                        inputs: vec![DerivedWeightRecipe::Concatenate {
                            axis: 0,
                            inputs: vec![
                                DerivedWeightRecipe::source(gate, TensorSelection::Full),
                                DerivedWeightRecipe::source(up, TensorSelection::Full),
                            ],
                        }],
                    },
                    store,
                )?);
                bindings.push(recipe_binding(
                    "down_proj",
                    DerivedWeightRecipe::Stack {
                        axis: 0,
                        inputs: vec![DerivedWeightRecipe::source(down, TensorSelection::Full)],
                    },
                    store,
                )?);
            }
            let bindings = match layout {
                Some(layout) => crate::runtime::execution::layerwise::shard_layer_bindings(
                    bindings, &prefix, store, layout,
                )?,
                None => bindings,
            };
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    Error::UnsupportedArchitecture("Qwen3 expert byte total overflowed".into())
                })
            })?;
            let unit = OffloadUnit::new(identity.unit_id(), bindings)?;
            entries.push(ExpertCatalogEntry::new(identity, unit, bytes)?);
        }
    }
    Ok(entries)
}

fn recipe_binding(
    name: &str,
    recipe: DerivedWeightRecipe,
    store: &dyn WeightStore,
) -> Result<WeightBinding, Error> {
    let bytes = recipe.infer(store)?.byte_len();
    Ok(WeightBinding::from_recipe(name, recipe, bytes)?)
}

fn split_expert_key(
    keys: &BTreeSet<String>,
    prefix: &str,
    expert: usize,
    projections: &[&str],
) -> Result<String, Error> {
    projections
        .iter()
        .map(|projection| format!("{prefix}.{expert}.{projection}.weight"))
        .find(|key| keys.contains(key))
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Qwen3 checkpoint is missing split expert {expert} projection {:?}",
                projections
            ))
        })
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use safemlx::{
        module::ModuleParameters,
        ops::{indexing::TryIndexOp, ones_dtype},
        Array, Device, DeviceType, ExecutionContext,
    };

    use super::*;
    use crate::{
        architectures::qwen::dense as dense_qwen,
        core::residency::{MemoryTier, OffloadConfig, ResidencyPolicy},
        runtime::{
            distributed::{
                parallel::{ParallelBuildContext, ShardingPolicy},
                topology::{DeviceAssignment, ParallelTopology},
            },
            execution::layerwise::{
                ExecutionResidency, LayerWeightResidency, LayerwiseLoadOptions,
            },
        },
    };

    fn tensor_parallel_context_for(rank: usize) -> ParallelBuildContext {
        ParallelBuildContext::new(
            ParallelTopology::from_rank(
                2,
                rank,
                2,
                1,
                1,
                DeviceAssignment::new(DeviceType::Cpu, 0),
            )
            .unwrap(),
            ShardingPolicy::Require,
        )
    }

    fn tensor_parallel_context() -> ParallelBuildContext {
        tensor_parallel_context_for(0)
    }

    fn args(moe: bool) -> DecoderConfig {
        DecoderConfig {
            model_type: "qwen3".into(),
            hidden_size: 8,
            num_hidden_layers: 3,
            intermediate_size: if moe { 0 } else { 16 },
            num_attention_heads: 2,
            rms_norm_eps: 1e-5,
            vocab_size: 16,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            head_dim: 4,
            tie_word_embeddings: false,
            rope_scaling: None,
            hidden_act: "silu".into(),
            attention_dropout: 0.0,
            attention_bias: Some(false),
            mlp_bias: Some(false),
            attention_schedule: crate::runtime::attention::LayerSchedule::all_full(3).unwrap(),
            quantization: None,
            quantization_config: None,
            quantized_weights: None,
            moe_intermediate_size: if moe { 9 } else { 0 },
            num_experts: if moe { 4 } else { 0 },
            num_experts_per_tok: if moe { 2 } else { 0 },
            norm_topk_prob: moe,
            quantized_weight_configs: None,
        }
    }

    fn initialize(module: &mut impl ModuleParameters, stream: &Stream) {
        let mut names = module
            .parameters()
            .flatten()
            .keys()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        names.sort();
        let mut params = module.parameters_mut().flatten();
        for (index, name) in names.iter().enumerate() {
            let parameter = params.get_mut(name.as_str()).unwrap();
            let shape = parameter.shape().to_vec();
            let dtype = parameter.dtype();
            **parameter = if name.ends_with("layernorm.weight") || name == "model.norm.weight" {
                ones_dtype(&shape, dtype, stream).unwrap()
            } else {
                Array::full::<f32>(&shape, Array::from_f32(0.001 * (index + 1) as f32), stream)
                    .unwrap()
                    .as_dtype(dtype, stream)
                    .unwrap()
            };
        }
    }

    fn write_fixture(dir: &Path, model: &dense_qwen::Model, split_experts: bool, stream: &Stream) {
        let params = model.parameters().flatten();
        let mut arrays = Vec::<(String, Array)>::new();
        for (name, value) in params {
            let name = crate::runtime::checkpoint::binding::canonical_checkpoint_name(&name);
            if split_experts {
                if let Some(prefix) = name.strip_suffix(".mlp.experts.gate_up_proj") {
                    for expert in 0..model.args.num_experts {
                        let selected = value.try_index_device(expert, stream).unwrap();
                        let intermediate = model.args.moe_intermediate_size;
                        arrays.push((
                            format!("{prefix}.mlp.experts.{expert}.gate_proj.weight"),
                            selected
                                .try_index_device((..intermediate, ..), stream)
                                .unwrap(),
                        ));
                        arrays.push((
                            format!("{prefix}.mlp.experts.{expert}.up_proj.weight"),
                            selected
                                .try_index_device((intermediate.., ..), stream)
                                .unwrap(),
                        ));
                    }
                    continue;
                }
                if let Some(prefix) = name.strip_suffix(".mlp.experts.down_proj") {
                    for expert in 0..model.args.num_experts {
                        arrays.push((
                            format!("{prefix}.mlp.experts.{expert}.down_proj.weight"),
                            value.try_index_device(expert, stream).unwrap(),
                        ));
                    }
                    continue;
                }
            }
            arrays.push((name, value.clone()));
        }
        Array::save_safetensors(
            arrays.iter().map(|(name, value)| (name.as_str(), value)),
            None,
            dir.join("model.safetensors"),
        )
        .unwrap();
        let policies = model.args.attention_schedule.iter().collect::<Vec<_>>();
        let candidate_first = policies.iter().position(|policy| policy.window().is_some());
        let sliding_window = candidate_first.and_then(|first| {
            let window = policies[first].window().unwrap().get();
            policies[first..]
                .iter()
                .all(|policy| policy.window().is_some_and(|value| value.get() == window))
                .then_some(window)
        });
        let first_sliding = sliding_window.and(candidate_first);
        fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&serde_json::json!({
                "model_type": model.args.model_type,
                "hidden_size": model.args.hidden_size,
                "num_hidden_layers": model.args.num_hidden_layers,
                "intermediate_size": model.args.intermediate_size,
                "num_attention_heads": model.args.num_attention_heads,
                "num_key_value_heads": model.args.num_key_value_heads,
                "rms_norm_eps": model.args.rms_norm_eps,
                "vocab_size": model.args.vocab_size,
                "max_position_embeddings": model.args.max_position_embeddings,
                "rope_theta": model.args.rope_theta,
                "head_dim": model.args.head_dim,
                "tie_word_embeddings": model.args.tie_word_embeddings,
                "attention_bias": model.args.attention_bias,
                "use_sliding_window": first_sliding.is_some(),
                "sliding_window": sliding_window,
                "max_window_layers": first_sliding,
                "moe_intermediate_size": model.args.moe_intermediate_size,
                "num_experts": model.args.num_experts,
                "num_experts_per_tok": model.args.num_experts_per_tok,
                "norm_topk_prob": model.args.norm_topk_prob
            }))
            .unwrap(),
        )
        .unwrap();
    }

    fn assert_close(left: &Array, right: &Array) {
        let left = left.evaluated().unwrap();
        let right = right.evaluated().unwrap();
        assert_eq!(left.as_array().shape(), right.as_array().shape());
        for (left, right) in left.as_slice::<f32>().iter().zip(right.as_slice::<f32>()) {
            assert!((left - right).abs() <= 3e-5, "{left} != {right}");
        }
    }

    fn parity_with_args(model_args: DecoderConfig, depth: usize) {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut fixture = dense_qwen::Model::new(model_args, gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, false, gpu.stream());

        let mut resident =
            dense_qwen::load_safetensors(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let mut layerwise = load_safetensors(
            dir.path(),
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, depth).unwrap()),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut resident_cache = resident.new_cache();
        let mut layerwise_cache = layerwise.new_cache();
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
            Array::from_slice(&[4u32], &[1, 1]),
            Array::from_slice(&[5u32], &[1, 1]),
        ] {
            let expected = resident
                .forward(
                    dense_qwen::ModelInput {
                        inputs: &tokens,
                        mask: None,
                        cache: &mut resident_cache,
                    },
                    gpu.stream(),
                )
                .unwrap();
            let actual = layerwise
                .forward(&tokens, None, &mut layerwise_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected);
            let report = layerwise.residency_report().unwrap();
            let layers = report
                .units()
                .iter()
                .filter(|unit| unit.id().as_str().starts_with("dense_qwen.layer."))
                .collect::<Vec<_>>();
            assert!(layers.iter().all(|unit| unit.host_resident()));
            assert!(layers.iter().filter(|unit| unit.device_resident()).count() <= depth);
            assert!(report
                .units()
                .iter()
                .filter(|unit| unit.device_resident() && !layers.contains(unit))
                .all(|unit| unit.policy() == ResidencyPolicy::Pinned));
        }
    }

    fn parity(moe: bool, depth: usize) {
        parity_with_args(args(moe), depth);
    }

    #[test]
    fn generalized_tensor_parallel_loader_preserves_packed_moe_geometry() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let weights = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut fixture = dense_qwen::Model::new(args(true), execution.stream()).unwrap();
        initialize(&mut fixture, execution.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, false, execution.stream());

        let model = load_tensor_parallel_model(
            dir.path(),
            LayerwiseLoadOptions::default(),
            tensor_parallel_context(),
            execution.stream(),
            weights.stream(),
        )
        .unwrap();
        let layout = model.execution.parallel_layout().unwrap();
        assert_eq!(
            layout
                .tensor("model.layers.0.self_attn.q_proj.weight")
                .unwrap()
                .local_shape(),
            &[4, 8]
        );
        assert_eq!(
            layout
                .tensor("model.layers.0.mlp.experts.gate_up_proj")
                .unwrap()
                .local_shape(),
            &[4, 10, 8]
        );
        assert_eq!(
            layout
                .tensor("model.layers.0.mlp.experts.down_proj")
                .unwrap()
                .local_shape(),
            &[4, 8, 5]
        );
        assert!(model
            .parallel_info()
            .unwrap()
            .owned_tensors()
            .iter()
            .any(|name| name == "model.layers.0.mlp.experts.gate_up_proj"));
    }

    #[test]
    fn qwen3_moe_quantized_experts_share_uneven_block_aligned_partitions() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut config = args(true);
        config.hidden_size = 64;
        config.num_hidden_layers = 1;
        config.num_attention_heads = 4;
        config.num_key_value_heads = 2;
        config.head_dim = 16;
        config.vocab_size = 64;
        config.moe_intermediate_size = 96;
        config.attention_schedule = crate::runtime::attention::LayerSchedule::all_full(1).unwrap();
        config.quantized_weight_configs = Some(HashMap::from([(
            "model.layers.0.mlp.experts.down_proj".into(),
            crate::runtime::checkpoint::quantization::AffineQuantization::new(32, 4)
                .unwrap()
                .into(),
        )]));
        let layer = TransformerBlock::new_for_layer(&config, 0, execution.stream()).unwrap();

        for (rank, gate_up_width, packed_width, scale_width) in [(0, 128, 8, 2), (1, 64, 4, 1)] {
            let mut planner = tensor_parallel_context_for(rank).planner();
            register_qwen_layer_parallel_plan(&mut planner, &layer, &config, "model.layers.0")
                .unwrap();
            let (_, layout) = planner.finish().unwrap();
            let gate_up = layout
                .tensor("model.layers.0.mlp.experts.gate_up_proj")
                .unwrap();
            let down = layout
                .tensor("model.layers.0.mlp.experts.down_proj")
                .unwrap();
            let scales = layout
                .tensor("model.layers.0.mlp.experts.down_proj_scales")
                .unwrap();
            assert_eq!(gate_up.local_shape(), &[4, gate_up_width, 64]);
            assert_eq!(down.local_shape(), &[4, 64, packed_width]);
            assert_eq!(scales.local_shape(), &[4, 64, scale_width]);
            assert_eq!(
                gate_up.logical_name(),
                "model.layers.0.mlp.experts.intermediate"
            );
            assert_eq!(down.logical_name(), gate_up.logical_name());
            assert_eq!(scales.logical_name(), gate_up.logical_name());
        }
    }

    #[test]
    fn generalized_fully_resident_execution_materializes_layers_once() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let weights = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut fixture = dense_qwen::Model::new(args(false), execution.stream()).unwrap();
        initialize(&mut fixture, execution.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, false, execution.stream());

        let mut model = load_safetensors(
            dir.path(),
            LayerWeightResidency::FullyResident,
            execution.stream(),
            weights.stream(),
        )
        .unwrap();
        let metadata = model.residency_metadata();
        assert_eq!(metadata.residency(), ExecutionResidency::FullyResident);
        assert_eq!(metadata.device_layer_capacity(), metadata.layer_count());
        assert_eq!(
            metadata.maximum_device_layer_bytes(),
            metadata.layer_parameter_bytes()
        );
        let before = model.residency_report().unwrap();
        let layers = before
            .units()
            .iter()
            .filter(|unit| unit.id().as_str().starts_with("dense_qwen.layer."))
            .collect::<Vec<_>>();
        assert_eq!(layers.len(), metadata.layer_count());
        assert!(layers.iter().all(|unit| {
            unit.policy() == ResidencyPolicy::Pinned
                && unit.planned_tier() == MemoryTier::Device
                && unit.device_resident()
                && !unit.host_resident()
                && unit.device_pins() == 1
        }));

        let initial_telemetry = before.offload().clone();
        let mut cache = model.new_cache();
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
        ] {
            model
                .forward(&tokens, None, &mut cache, execution.stream())
                .unwrap();
        }
        let after = model.residency_report().unwrap();
        assert_eq!(after.offload(), &initial_telemetry);
        assert!(after
            .units()
            .iter()
            .all(|unit| { unit.policy() == ResidencyPolicy::Pinned && unit.device_resident() }));
        model.clear_device_layer_window().unwrap();
        assert!(model
            .residency_report()
            .unwrap()
            .units()
            .iter()
            .all(|unit| unit.device_resident()));
    }

    #[test]
    fn high_level_resident_and_layerwise_dispatch_execute_and_report_paged_kv_state() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut fixture = dense_qwen::Model::new(args(false), gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, false, gpu.stream());

        let fully_resident = load_safetensors(
            dir.path(),
            LayerWeightResidency::FullyResident,
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let layerwise = load_safetensors(
            dir.path(),
            LayerwiseLoadOptions::default(),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        for mut model in [
            crate::api::Model::DenseQwen(fully_resident),
            crate::api::Model::DenseQwen(layerwise),
        ] {
            let options = PagedCacheOptions::new(1, 16 * 1024, 16 * 1024, 1)
                .unwrap()
                .with_full_attention(true);
            let mut cache = model
                .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                .unwrap();
            assert_eq!(
                cache
                    .residency_report()
                    .unwrap()
                    .unwrap()
                    .logical_cached_tokens,
                0
            );

            let tokens = Array::from_slice(&[1u32, 2, 3], &[1, 3]);
            let parts = [crate::runtime::media::input::InputPart::text_token_ids(
                &tokens,
            )];
            model
                .submit_prefill(
                    crate::runtime::media::input::ModelInput::new(&parts),
                    &mut cache,
                    gpu.stream(),
                )
                .unwrap()
                .wait()
                .unwrap()
                .evaluated()
                .unwrap();
            let report = cache.residency_report().unwrap().unwrap();
            assert_eq!(report.logical_cached_tokens, 3);
            assert!(report.current_device_bytes > 0);
        }
    }

    #[test]
    fn generalized_fully_resident_tensor_parallel_reports_local_and_global_memory() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let weights = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut fixture = dense_qwen::Model::new(args(false), execution.stream()).unwrap();
        initialize(&mut fixture, execution.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, false, execution.stream());

        let model = load_tensor_parallel_model(
            dir.path(),
            LayerWeightResidency::FullyResident,
            tensor_parallel_context(),
            execution.stream(),
            weights.stream(),
        )
        .unwrap();
        let info = model.parallel_info().unwrap();
        assert_eq!(
            info.pinned_device_parameter_bytes(),
            info.local_parameter_bytes()
        );
        assert_eq!(
            info.maximum_device_parameter_bytes(),
            info.local_parameter_bytes()
        );
        assert!(info.global_parameter_bytes() > info.local_parameter_bytes());
        let metadata = model.residency_metadata();
        assert_eq!(metadata.residency(), ExecutionResidency::FullyResident);
        assert_eq!(metadata.device_layer_capacity(), metadata.layer_count());
        let report = model.residency_report().unwrap();
        assert!(report.units().iter().all(|unit| {
            unit.policy() == ResidencyPolicy::Pinned
                && unit.planned_tier() == MemoryTier::Device
                && unit.device_resident()
                && !unit.host_resident()
        }));
    }

    #[test]
    fn qwen3_dense_layerwise_prefill_and_cached_decode_parity() {
        parity(false, 1);
        parity(false, 2);
    }

    #[test]
    fn qwen2_resident_and_layerwise_full_and_sliding_attention_parity() {
        let mut model_args = args(false);
        model_args.model_type = "qwen2".into();
        model_args.attention_bias = Some(true);
        model_args.attention_schedule =
            crate::runtime::attention::LayerSchedule::from_sliding_pattern(
                3,
                &[false, true, true],
                Some(2),
            )
            .unwrap();
        parity_with_args(model_args, 1);
    }

    #[test]
    fn qwen2_resident_and_layerwise_arbitrary_pattern_parity() {
        use crate::runtime::checkpoint::store::SafetensorsWeightStore;

        let gpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut model_args = args(false);
        model_args.model_type = "qwen2".into();
        model_args.attention_bias = Some(true);
        model_args.attention_schedule =
            crate::runtime::attention::LayerSchedule::from_sliding_pattern(
                3,
                &[true, false, true],
                Some(2),
            )
            .unwrap();
        let mut resident = dense_qwen::Model::new(model_args.clone(), gpu.stream()).unwrap();
        initialize(&mut resident, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &resident, false, gpu.stream());

        let store: Arc<dyn WeightStore + Send + Sync> =
            Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap());
        let adapter = DenseQwenLayerwiseAdapter::new(model_args, gpu.stream()).unwrap();
        let execution = load_layerwise_model(
            store,
            adapter,
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut layerwise = LayerwiseDecoder { execution };
        let mut resident_cache = resident.new_cache();
        let mut layerwise_cache = layerwise.new_cache();
        for tokens in [
            Array::from_slice(&[1u32, 2, 3], &[1, 3]),
            Array::from_slice(&[4u32], &[1, 1]),
            Array::from_slice(&[5u32], &[1, 1]),
        ] {
            let expected = resident
                .forward(
                    dense_qwen::ModelInput {
                        inputs: &tokens,
                        mask: None,
                        cache: &mut resident_cache,
                    },
                    gpu.stream(),
                )
                .unwrap();
            let actual = layerwise
                .forward(&tokens, None, &mut layerwise_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected);
        }
    }

    #[test]
    fn qwen3_sparse_moe_layerwise_prefill_and_cached_decode_parity() {
        parity(true, 1);
    }

    #[test]
    fn qwen3_sparse_expert_cache_prefill_and_decode_parity() {
        sparse_expert_cache_parity(false);
        sparse_expert_cache_parity(true);
    }

    #[test]
    fn qwen3_sparse_expert_cache_streams_non_expert_layers() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut fixture = dense_qwen::Model::new(args(true), gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, false, gpu.stream());

        let expert_options =
            ExpertCacheLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap(), 1 << 20, 1)
                .unwrap();
        let dense = crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions::new(
            u64::MAX,
            u64::MAX,
            1,
            1,
        )
        .unwrap();
        let mut cached = load_qwen3_expert_cache_model(
            dir.path(),
            crate::NonExpertWeightResidency::DenseDiskStream(dense),
            expert_options,
            None,
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let initial = cached.dense_stream_report().unwrap().unwrap();
        assert!(initial
            .residency()
            .units()
            .iter()
            .filter(|unit| unit.id().as_str().starts_with("dense_qwen.layer."))
            .all(|unit| {
                unit.planned_tier() == MemoryTier::Disk
                    && !unit.host_resident()
                    && !unit.device_resident()
            }));

        let mut resident =
            dense_qwen::load_safetensors(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let mut resident_cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();
        let mut cached_cache = cached.new_cache();
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
        ] {
            let expected = resident
                .forward(
                    dense_qwen::ModelInput {
                        inputs: &tokens,
                        mask: None,
                        cache: &mut resident_cache,
                    },
                    gpu.stream(),
                )
                .unwrap();
            let actual = cached
                .forward(&tokens, None, &mut cached_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected);
        }
        assert!(cached.expert_cache_report().unwrap().is_some());
    }

    #[test]
    fn qwen3_load_time_quantization_packs_complete_and_cached_expert_banks() {
        use crate::runtime::checkpoint::quantization::AffineQuantization;

        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut model_args = args(true);
        model_args.hidden_size = 32;
        model_args.vocab_size = 32;
        model_args.num_attention_heads = 4;
        model_args.num_key_value_heads = 4;
        model_args.head_dim = 8;
        model_args.moe_intermediate_size = 32;
        let mut fixture = dense_qwen::Model::new(model_args, gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, false, gpu.stream());

        let quantization = WeightQuantization::Affine(AffineQuantization::new(32, 4).unwrap());
        let expert_options =
            ExpertCacheLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap(), 1 << 20, 1)
                .unwrap();
        let dense = crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions::new(
            u64::MAX,
            u64::MAX,
            1,
            1,
        )
        .unwrap();
        let mut loaded = load_qwen3_expert_cache_model(
            dir.path(),
            crate::NonExpertWeightResidency::DenseDiskStream(dense),
            expert_options,
            Some(quantization),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut complete = load_safetensors_quantized_residency(
            dir.path(),
            crate::LayerWeightResidency::FullyResident,
            quantization,
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut streamed = load_safetensors_quantized_residency(
            dir.path(),
            dense,
            quantization,
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut layerwise = load_safetensors_quantized_residency(
            dir.path(),
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
            quantization,
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();

        let ordinary = loaded.residency_metadata().materialization().unwrap();
        assert!(ordinary.transformed_weights > 0);
        assert!(ordinary.output_bytes < ordinary.source_bytes_read);
        assert!(ordinary.peak_planned_working_set_bytes <= ordinary.output_bytes);
        let experts = loaded.expert_cache_report().unwrap().unwrap();
        let expert_materialization = experts.materialization.unwrap();
        assert!(expert_materialization.transformed_weights > 0);
        assert!(expert_materialization.output_bytes < expert_materialization.source_bytes_read);
        assert!(loaded
            .dense_stream_report()
            .unwrap()
            .unwrap()
            .residency()
            .units()
            .iter()
            .filter(|unit| unit.id().as_str().starts_with("dense_qwen.layer."))
            .all(|unit| unit.planned_tier() == MemoryTier::Disk));
        let complete_materialization = complete.residency_metadata().materialization().unwrap();
        assert!(complete_materialization.transformed_weights > 0);
        assert!(complete_materialization.output_bytes < complete_materialization.source_bytes_read);
        assert!(
            complete_materialization.peak_planned_working_set_bytes
                <= complete_materialization.output_bytes
        );
        assert!(complete.expert_cache_report().unwrap().is_none());
        assert!(streamed.expert_cache_report().unwrap().is_none());
        assert!(streamed
            .dense_stream_report()
            .unwrap()
            .unwrap()
            .residency()
            .units()
            .iter()
            .filter(|unit| unit.id().as_str().starts_with("dense_qwen.layer."))
            .all(|unit| unit.planned_tier() == MemoryTier::Disk));
        assert!(layerwise.expert_cache_report().unwrap().is_none());

        let tokens = Array::from_slice(&[1u32, 2], &[1, 2]);
        let mut cache = loaded.new_cache();
        let mut complete_cache = complete.new_cache();
        let mut streamed_cache = streamed.new_cache();
        let mut layerwise_cache = layerwise.new_cache();
        let logits = loaded
            .forward(&tokens, None, &mut cache, gpu.stream())
            .unwrap();
        let complete_logits = complete
            .forward(&tokens, None, &mut complete_cache, gpu.stream())
            .unwrap();
        let streamed_logits = streamed
            .forward(&tokens, None, &mut streamed_cache, gpu.stream())
            .unwrap();
        let layerwise_logits = layerwise
            .forward(&tokens, None, &mut layerwise_cache, gpu.stream())
            .unwrap();
        assert_close(&complete_logits, &logits);
        assert_close(&streamed_logits, &logits);
        assert_close(&layerwise_logits, &logits);
        safemlx::transforms::async_eval_with_event([&logits])
            .unwrap()
            .synchronize()
            .unwrap();
        assert_eq!(logits.shape(), &[1, 2, 32]);
    }

    #[test]
    fn qwen3_dense_disk_load_time_quantization_accounts_only_packed_bytes() {
        use crate::runtime::checkpoint::quantization::AffineQuantization;

        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut model_args = args(false);
        model_args.hidden_size = 32;
        model_args.vocab_size = 32;
        model_args.num_attention_heads = 4;
        model_args.num_key_value_heads = 4;
        model_args.head_dim = 8;
        model_args.intermediate_size = 32;
        let mut fixture = dense_qwen::Model::new(model_args, gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, false, gpu.stream());
        let dense = crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions::new(
            u64::MAX,
            u64::MAX,
            1,
            1,
        )
        .unwrap();
        let loaded = load_safetensors_quantized_residency(
            dir.path(),
            dense,
            WeightQuantization::Affine(AffineQuantization::new(32, 4).unwrap()),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let materialization = loaded.residency_metadata().materialization().unwrap();
        assert!(materialization.output_bytes < materialization.source_bytes_read);
        let report = loaded.dense_stream_report().unwrap().unwrap();
        assert_eq!(
            report.planned_layer_bytes(),
            loaded.residency_metadata().layer_parameter_bytes()
        );
        assert!(report.planned_layer_bytes() < materialization.source_bytes_read);
    }

    fn sparse_expert_cache_parity(split_experts: bool) {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut fixture = dense_qwen::Model::new(args(true), gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, split_experts, gpu.stream());

        let resident_dir = tempfile::tempdir().unwrap();
        write_fixture(resident_dir.path(), &fixture, false, gpu.stream());

        let mut resident =
            dense_qwen::load_safetensors(resident_dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let non_expert = LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap());
        let expert_options =
            ExpertCacheLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap(), 1 << 20, 1)
                .unwrap();
        let mut cached = load_qwen3_expert_cache_model(
            dir.path(),
            crate::NonExpertWeightResidency::LayerwiseHost(non_expert),
            expert_options,
            None,
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut resident_cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();
        let mut cached_cache = cached.new_cache();
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
            Array::from_slice(&[4u32], &[1, 1]),
        ] {
            let expected = resident
                .forward(
                    dense_qwen::ModelInput {
                        inputs: &tokens,
                        mask: None,
                        cache: &mut resident_cache,
                    },
                    gpu.stream(),
                )
                .unwrap();
            let actual = cached
                .forward(&tokens, None, &mut cached_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected);
        }
        let report = cached.expert_cache_report().unwrap().unwrap();
        assert_eq!(report.owned_experts, 12);
        assert!(report.prefill.requested_routes > 0);
        assert!(report.decode.requested_routes > 0);
        assert!(report.prefill.compact_banks > 1);
        assert!(report.decode.compact_banks > 0);
        assert_eq!(
            cached_cache[0].as_ref().unwrap().offset(),
            resident_cache[0].as_ref().unwrap().offset()
        );
    }
}

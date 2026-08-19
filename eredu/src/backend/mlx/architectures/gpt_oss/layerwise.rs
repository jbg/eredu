//! Unified fully resident and bounded layer execution for GPT-OSS.

use eredu_checkpoint::WeightQuantization;
use eredu_runtime::{OffloadUnit, WeightBinding};

use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
    sync::Arc,
    time::Instant,
};

use safemlx::{
    error::Exception,
    module::{Module, Param},
    nn,
    ops::{indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};

use crate::core::cache::{
    validate_prompt_cache_model_identity, PromptCacheDescriptor, PromptCacheManifest,
    PromptCacheModelIdentity, PromptCacheOptions, PromptCacheTopology,
};

use crate::{
    backend::mlx::architectures::gpt_oss::model::{
        self as resident, Cache, Experts, LayerCache, ModelArgs, TransformerBlock,
    },
    backend::mlx::error::Error,
    backend::mlx::nn::parallel::{
        gqa_projection_members, planned_kv_head_layout, GqaProjectionNames, VocabParallelEmbedding,
        VocabParallelLmHead,
    },
    backend::mlx::nn::{self as common, generation::CausalLm},
    backend::mlx::runtime::cache::residency::{
        open_prompt_cache, CacheResidencyManager, CacheResidencyPolicy, PagedCacheOptions,
    },
    backend::mlx::runtime::cache::{ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache},
    backend::mlx::runtime::checkpoint::binding::{
        build_module_binding_plan_with_recipes_excluding, build_module_bindings,
        populate_module_from_lease, populate_module_from_lease_excluding,
    },
    backend::mlx::runtime::checkpoint::binding_plan::{BindingPlan, PlannedBinding},
    backend::mlx::runtime::checkpoint::store::{open_gguf_checkpoint_source, TensorSelection},
    backend::mlx::runtime::checkpoint::{
        quantization::should_quantize_on_load,
        recipe::{DerivedWeightRecipe, RecipeDtype},
    },
    backend::mlx::runtime::distributed::parallel::{
        aligned_partition_units, array_parameter_member, register_projection_module,
        register_replicated_module, MemberSharding, ParallelPlanBuilder, ParameterGroupSpec,
        ParameterRole, ProjectionSharding,
    },
    backend::mlx::runtime::execution::layerwise::{
        load_layerwise_model, load_layerwise_model_with_quantization,
        load_tensor_parallel_layerwise_model, open_safetensors_weight_store, ArchitectureAdapter,
        LayerWeightResidency, LayerwiseForwardState, LayerwiseModel, LoadTimeQuantizableAdapter,
        StaticUnitBindings, WeightResidency,
    },
    backend::mlx::runtime::media::input,
    backend::mlx::runtime::residency::expert_cache::{
        ExpertCache, ExpertCacheLoadOptions, ExpertCacheReport, ExpertCatalogEntry, ExpertIdentity,
        ExpertPass, ExpertRouteBatch,
    },
    backend::mlx::runtime::residency::manager::ResidentUnitLease,
    core::attention::{AttentionPolicy, LayerSchedule},
};

use eredu_runtime::ResidencyReport;

const EMBEDDING_UNIT: &str = "gpt_oss.static.embedding";
const NORM_UNIT: &str = "gpt_oss.static.norm";
const HEAD_UNIT: &str = "gpt_oss.static.output";

/// GPT-OSS causal LM using bounded residency for complete decoder blocks.
pub struct GptOssLayerwiseModel {
    execution: LayerwiseModel<GptOssLayerwiseAdapter>,
}

impl GptOssLayerwiseModel {
    /// Returns the validated model arguments.
    pub fn args(&self) -> &ModelArgs {
        self.execution.adapter().args()
    }

    pub(crate) fn bind_parallel_topology(
        &mut self,
        topology: crate::backend::mlx::MlxParallelContext,
    ) {
        self.execution.bind_parallel_topology(topology);
    }

    /// Returns the canonical cache-relevant architecture identity.
    pub fn prompt_cache_architecture_fingerprint(&self) -> String {
        resident::prompt_cache_architecture_fingerprint(self.args())
    }

    /// Returns this rank's exact prompt-cache state layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        self.execution.prompt_cache_layer_layout()
    }

    /// Returns the complete rank-local prompt-cache identity.
    pub fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        self.execution.prompt_cache_model_identity()
    }

    /// Returns rank-local generalized parallel information when applicable.
    pub fn parallel_info(
        &self,
    ) -> Option<&crate::backend::mlx::runtime::execution::layerwise::ParallelModelInfo> {
        self.execution.parallel_info()
    }

    /// Returns generalized parameter-residency and encoding metadata.
    pub fn residency_metadata(
        &self,
    ) -> &crate::backend::mlx::runtime::execution::layerwise::LayerwiseModelMetadata {
        self.execution.metadata()
    }

    /// Creates caches matching the canonical per-layer attention schedule.
    pub fn new_cache(&self) -> Cache {
        self.execution.adapter().new_cache()
    }

    /// Creates scheduled attention caches independently of weight residency.
    pub fn new_cache_with_options(&self, policy: CacheResidencyPolicy) -> Result<Cache, Error> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => {
                let adapter = self.execution.adapter();
                let manager = CacheResidencyManager::new(options)
                    .map_err(|error| Exception::custom(error.to_string()))?;
                let rank = self.execution.prompt_cache_rank_identity();
                let layers = adapter
                    .attention_schedule
                    .iter()
                    .enumerate()
                    .map(|(layer, policy)| {
                        let window = policy.window().map(|window| {
                            i32::try_from(window.get())
                                .expect("validated GPT-OSS sliding window fits i32")
                        });
                        PagedKeyValueCache::new_with_layout(manager.clone(), layer, window, 0, rank)
                            .map(LayerCache::Paged)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Cache { layers })
            }
        }
    }

    /// Lazily catalogs a compatible persisted scheduled-attention prefix.
    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(Cache, PromptCacheManifest), Error> {
        if self.execution.parallel_info().is_some() {
            return self.execution.load_prompt_cache(
                directory,
                expected,
                prefix_token_ids,
                options,
                stream,
            );
        }
        let args = self.args();
        let layer_count = usize::try_from(args.num_hidden_layers)
            .map_err(|_| Exception::custom("invalid GPT-OSS cache layer count"))?;
        let identity = PromptCacheModelIdentity {
            model_family: "gpt_oss".into(),
            effective_model_type: args.model_type.clone(),
            architecture_fingerprint: resident::prompt_cache_architecture_fingerprint(args),
            layer_count,
            global_layer_start: 0,
            global_layer_end: layer_count,
            sink_tokens: 0,
            layer_prefix_offsets: vec![0; layer_count],
            topology: Default::default(),
            layer_layout: PromptCacheModelIdentity::key_value_layouts(
                args.attention_schedule.iter().map(|policy| {
                    policy.window().map(|window| {
                        i32::try_from(window.get())
                            .expect("validated GPT-OSS sliding window fits i32")
                    })
                }),
                args.num_key_value_heads,
                args.head_dim,
            )
            .map_err(|error| Exception::custom(error.to_string()))?,
        };
        validate_prompt_cache_model_identity(expected, &identity)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let (manager, manifest) =
            open_prompt_cache(directory, expected, &identity, prefix_token_ids, options)
                .map_err(|error| Exception::custom(error.to_string()))?;
        let adapter = self.execution.adapter();
        let layers = adapter
            .attention_schedule
            .iter()
            .enumerate()
            .map(|(layer, policy)| {
                let window = policy.window().map(|window| {
                    i32::try_from(window.get()).expect("validated GPT-OSS sliding window fits i32")
                });
                PagedKeyValueCache::new(manager.clone(), layer, window).map(LayerCache::Paged)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((Cache { layers }, manifest))
    }

    /// Persists a prefix through the generalized execution contract.
    pub fn save_prompt_cache(
        &self,
        cache: &mut Cache,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        self.execution.save_prompt_cache(
            cache,
            destination,
            descriptor,
            prefix_token_ids,
            options,
            stream,
        )
    }

    /// Returns current logical residency and transfer telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        self.execution.residency_report()
    }
    /// Returns dense-stream observations when that policy is active.
    pub fn dense_stream_report(
        &self,
    ) -> Result<
        Option<crate::backend::mlx::runtime::execution::layerwise::DenseDiskStreamReport>,
        Error,
    > {
        self.execution.dense_stream_report()
    }

    /// Returns sparse expert-cache telemetry when enabled.
    pub fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        self.execution
            .adapter()
            .expert_cache
            .as_ref()
            .map(ExpertCache::report)
            .transpose()
            .map_err(Into::into)
    }

    /// Returns the persistent checkpoint store.
    pub fn checkpoint_store(&self) -> &(dyn eredu_checkpoint::store::CheckpointSource) {
        self.execution.checkpoint_store()
    }

    pub(crate) fn checkpoint_store_arc(
        &self,
    ) -> Arc<dyn eredu_checkpoint::store::CheckpointSource> {
        self.execution.checkpoint_store_arc()
    }

    /// Runs a rank-local tensor-parallel forward pass through the generalized engine.
    pub(crate) fn forward_tensor_parallel(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution
            .forward_tensor_parallel(inputs, cache, group, stream)
    }

    /// Runs GPT-OSS while preserving its heterogeneous cache schedule.
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution.forward(inputs, cache, stream)
    }

    /// Runs streamed layers while delegating routed experts to a caller.
    pub(crate) fn forward_with_expert_executor<F>(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        self.execution.forward_with_layer_executor(
            inputs,
            cache,
            stream,
            |adapter, _group, index, layer, hidden, cache, context, stream| {
                let layer_cache = &mut cache.layers[index];
                let offset = layer_cache.offset();
                let policy = adapter
                    .attention_schedule
                    .get(index)
                    .expect("validated GPT-OSS layer index");
                let mask =
                    resident::attention_mask(policy, context.sequence_length, offset, stream)?;
                Ok(layer.forward_with_expert_executor(
                    hidden,
                    mask.as_ref(),
                    layer_cache,
                    stream,
                    |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                )?)
            },
        )
    }

    /// Runs TP-sharded nonexpert layers while delegating routed experts to EP.
    pub(crate) fn forward_tensor_expert_parallel<F>(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        tensor_group: &safemlx::distributed::Group,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        self.execution.forward_tensor_parallel_with_layer_executor(
            inputs,
            cache,
            tensor_group,
            stream,
            |adapter, _group, index, layer, hidden, cache, context, execution| {
                let tp_group = execution.group().ok_or_else(|| {
                    Error::Parallel("GPT-OSS TP+EP execution requires an active TP group".into())
                })?;
                let layer_cache = &mut cache.layers[index];
                let offset = layer_cache.offset();
                let policy = adapter
                    .attention_schedule
                    .get(index)
                    .expect("validated GPT-OSS TP+EP layer index");
                let mask = resident::attention_mask(
                    policy,
                    context.sequence_length,
                    offset,
                    execution.stream(),
                )?;
                Ok(layer.forward_tensor_with_expert_executor(
                    hidden,
                    mask.as_ref(),
                    layer_cache,
                    tp_group,
                    execution.stream(),
                    |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                )?)
            },
        )
    }

    /// Clears temporary decoder copies from the execution device.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        self.execution.clear_device_group("text_decoder")
    }
}

impl CausalLm<Cache> for GptOssLayerwiseModel {
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        self.forward(&tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward(input_tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }
}

/// Loads GPT-OSS through the generalized execution engine.
pub fn load_gpt_oss_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let args = resident::get_model_args(model_dir)?;
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load("GPT-OSS", args.quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let adapter = GptOssLayerwiseAdapter::new(args, stream)?;
    let store = open_safetensors_weight_store(model_dir, options.max_mapped_shards())?;
    Ok(GptOssLayerwiseModel {
        execution: load_layerwise_model_with_quantization(
            store,
            adapter,
            options,
            quantize_on_load,
            stream,
            weights_stream,
        )?,
    })
}

/// Loads GPT-OSS through the generalized tensor-parallel execution engine.
pub(crate) fn load_gpt_oss_tensor_parallel_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    if model_dir
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        let checkpoint = GgufCheckpoint::open(model_dir)?;
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        return load_gpt_oss_gguf_tensor_parallel_model(
            &checkpoint,
            &metadata,
            options,
            build,
            stream,
            weights_stream,
        )
        .map(|(model, _)| model);
    }
    let adapter = GptOssLayerwiseAdapter::new(resident::get_model_args(model_dir)?, stream)?;
    Ok(GptOssLayerwiseModel {
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

pub(crate) fn load_gpt_oss_gguf_tensor_parallel_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(GptOssLayerwiseModel, Vec<u32>), Error> {
    crate::backend::mlx::runtime::execution::layerwise::validate_gguf_layerwise_source(
        checkpoint, metadata, options,
    )?;
    let prepared = resident::prepare_gguf_checkpoint(checkpoint, metadata, weights_stream)?;
    let gguf_plan =
        super::checkpoint::gguf_plan(&prepared.args).map_err(Error::UnsupportedArchitecture)?;
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
        Arc::new(open_gguf_checkpoint_source(
            checkpoint.clone(),
            &gguf_plan,
            resident::translate_gguf_weight_name,
            options.max_mapped_shards(),
        )?);
    let execution = load_tensor_parallel_layerwise_model(
        store,
        GptOssLayerwiseAdapter::new(prepared.args, stream)?,
        options,
        build,
        stream,
        weights_stream,
    )?;
    Ok((GptOssLayerwiseModel { execution }, prepared.eos_token_ids))
}

pub(crate) fn load_gpt_oss_gguf_layerwise_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(GptOssLayerwiseModel, Vec<u32>), Error> {
    let prepared = resident::prepare_gguf_checkpoint(checkpoint, metadata, weights_stream)?;
    let gguf_plan =
        super::checkpoint::gguf_plan(&prepared.args).map_err(Error::UnsupportedArchitecture)?;
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
        Arc::new(open_gguf_checkpoint_source(
            checkpoint.clone(),
            &gguf_plan,
            resident::translate_gguf_weight_name,
            residency.max_mapped_shards(),
        )?);
    let args = prepared.args;
    if let Some(expert_options) = residency.expert_cache() {
        return Ok((
            load_gpt_oss_gguf_sparse_with_store(
                store,
                args,
                expert_options,
                residency.layers(),
                quantization,
                stream,
                weights_stream,
            )?,
            prepared.eos_token_ids,
        ));
    }
    let execution = load_layerwise_model_with_quantization(
        store,
        GptOssLayerwiseAdapter::new(args, stream)?,
        residency.layers(),
        quantization,
        stream,
        weights_stream,
    )?;
    Ok((GptOssLayerwiseModel { execution }, prepared.eos_token_ids))
}

fn load_gpt_oss_gguf_sparse_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    options: ExpertCacheLoadOptions,
    non_expert: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssLayerwiseModel, Error> {
    let mut adapter = GptOssLayerwiseAdapter::new(args.clone(), stream)?;
    adapter.sparse_expert_cache = true;
    let mut execution = load_layerwise_model_with_quantization(
        store,
        adapter,
        non_expert,
        quantization,
        stream,
        weights_stream,
    )?;
    let checkpoint_store = execution.checkpoint_store_arc();
    let entries = gpt_oss_expert_catalog(&args, checkpoint_store.as_ref())?;
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
    Ok(GptOssLayerwiseModel { execution })
}

/// Loads GPT-OSS with independently cached experts and bounded non-expert units.
pub fn load_gpt_oss_expert_cache_model(
    model_dir: impl AsRef<Path>,
    non_expert: crate::backend::mlx::runtime::execution::layerwise::NonExpertWeightResidency,
    options: ExpertCacheLoadOptions,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = resident::get_model_args(model_dir)?;
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load(
                "GPT-OSS independent expert cache",
                args.quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let mut source_adapter = GptOssLayerwiseAdapter::new(args.clone(), stream)?;
    source_adapter.sparse_expert_cache = true;
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
    let entries = gpt_oss_expert_catalog(&args, store.as_ref())?;
    execution.adapter_mut().expert_cache = Some(match quantize_on_load {
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
    });
    Ok(GptOssLayerwiseModel { execution })
}

/// Builds the streamed nonexpert GPT-OSS execution base used by distributed EP.
pub(crate) fn load_gpt_oss_sparse_ep_base_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    non_expert: impl Into<LayerWeightResidency>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssLayerwiseModel, Error> {
    let mut adapter = GptOssLayerwiseAdapter::new(args, stream)?;
    adapter.sparse_expert_cache = true;
    let execution = load_layerwise_model(store, adapter, non_expert, stream, weights_stream)?;
    Ok(GptOssLayerwiseModel { execution })
}

/// Builds the shared TP-sharded nonexpert base used by combined TP+EP.
pub(crate) fn load_gpt_oss_sparse_tp_ep_base_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    non_expert: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssLayerwiseModel, Error> {
    let mut adapter = GptOssLayerwiseAdapter::new(args, stream)?;
    adapter.sparse_expert_cache = true;
    let execution = load_tensor_parallel_layerwise_model(
        store,
        adapter,
        non_expert,
        build,
        stream,
        weights_stream,
    )?;
    Ok(GptOssLayerwiseModel { execution })
}

/// Generalized adapter for GPT-OSS native MXFP4 sparse decoder blocks.
pub struct GptOssLayerwiseAdapter {
    args: ModelArgs,
    attention_schedule: LayerSchedule<AttentionPolicy>,
    embedding: MaybeQuantized<nn::Embedding>,
    norm: nn::RmsNorm,
    lm_head: MaybeQuantized<nn::Linear>,
    parallel_embedding: Option<VocabParallelEmbedding>,
    parallel_lm_head: Option<VocabParallelLmHead>,
    parallel_kv_heads: Option<Vec<i32>>,
    sparse_expert_cache: bool,
    expert_cache: Option<ExpertCache>,
}

impl GptOssLayerwiseAdapter {
    /// Creates metadata-only pinned modules for a validated configuration.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        args.validate()?;
        let attention_schedule = args.attention_schedule.clone();
        let embedding = common::linear::unloaded_maybe_quantized_embedding(
            args.vocab_size,
            args.hidden_size,
            args.weight_quantization_for("model.embed_tokens.weight"),
            stream,
        )?;
        let norm =
            nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)?;
        let lm_head = common::linear::unloaded_maybe_quantized_linear(
            args.hidden_size,
            args.vocab_size,
            false,
            args.weight_quantization_for("lm_head.weight"),
            stream,
        )?;
        Ok(Self {
            args,
            attention_schedule,
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

    /// Creates the semantic adapter with routed experts supplied externally.
    pub(crate) fn new_external_experts(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let mut adapter = Self::new(args, stream)?;
        adapter.sparse_expert_cache = true;
        Ok(adapter)
    }

    /// Returns the validated model arguments.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    fn new_cache(&self) -> Cache {
        Cache {
            layers: self
                .attention_schedule
                .iter()
                .map(|policy| match policy.window() {
                    Some(window) => {
                        LayerCache::Sliding(ConcatKeyValueCache::new_for_sliding_attention(
                            i32::try_from(window.get())
                                .expect("validated GPT-OSS sliding window fits i32"),
                        ))
                    }
                    None => LayerCache::Full(ConcatKeyValueCache::new()),
                })
                .collect(),
        }
    }

    fn layer_recipes(
        &self,
        index: usize,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
        let prefix = format!("model.layers.{index}.mlp.experts");
        if !store
            .source_keys()
            .contains(&format!("{prefix}.gate_proj.weight"))
        {
            return Ok(BTreeMap::new());
        }
        let experts = self.args.num_local_experts as usize;
        let hidden = self.args.hidden_size as usize;
        let intermediate = self.args.intermediate_size as usize;
        let source = |name: &str| {
            DerivedWeightRecipe::source(format!("{prefix}.{name}"), TensorSelection::Full)
        };
        let stack_reshape =
            |gate: &str, up: &str, shape: Vec<usize>| DerivedWeightRecipe::Reshape {
                input: Box::new(DerivedWeightRecipe::Stack {
                    axis: 2,
                    inputs: vec![source(gate), source(up)],
                }),
                shape,
            };
        let gate_up_u32 = stack_reshape(
            "gate_proj.weight",
            "up_proj.weight",
            vec![experts, 2 * intermediate, hidden / 8],
        );
        Ok(BTreeMap::from([
            (
                "mlp.experts.gate_up_proj_blocks".into(),
                DerivedWeightRecipe::View {
                    input: Box::new(gate_up_u32),
                    dtype: RecipeDtype::U8,
                    shape: vec![experts, 2 * intermediate, hidden / 32, 16],
                },
            ),
            (
                "mlp.experts.gate_up_proj_scales".into(),
                stack_reshape(
                    "gate_proj.scales",
                    "up_proj.scales",
                    vec![experts, 2 * intermediate, hidden / 32],
                ),
            ),
            (
                "mlp.experts.gate_up_proj_bias".into(),
                stack_reshape(
                    "gate_proj.bias",
                    "up_proj.bias",
                    vec![experts, 2 * intermediate],
                ),
            ),
            (
                "mlp.experts.down_proj_blocks".into(),
                DerivedWeightRecipe::View {
                    input: Box::new(source("down_proj.weight")),
                    dtype: RecipeDtype::U8,
                    shape: vec![experts, hidden, intermediate / 32, 16],
                },
            ),
            (
                "mlp.experts.down_proj_scales".into(),
                source("down_proj.scales"),
            ),
            (
                "mlp.experts.down_proj_bias".into(),
                source("down_proj.bias"),
            ),
        ]))
    }
}

/// GPT-OSS state shared across temporary decoder blocks.
pub struct GptOssForwardContext {
    sequence_length: i32,
}

fn register_gpt_oss_layer_parallel_plan(
    planner: &mut ParallelPlanBuilder,
    layer: &TransformerBlock,
    args: &ModelArgs,
    index: usize,
) -> Result<(), Error> {
    let prefix = format!("model.layers.{index}");
    let attention = &layer.self_attn;
    let attention_prefix = format!("{prefix}.self_attn");
    let (head_units, mut attention_members) = gqa_projection_members(
        &attention_prefix,
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
        args.num_attention_heads,
        args.num_key_value_heads,
        args.head_dim,
    )?;
    attention_members.push(array_parameter_member(
        format!("{prefix}.self_attn.sinks"),
        attention.sinks.as_ref(),
        MemberSharding::Partitioned { axis: 0 },
    )?);
    planner.register(ParameterGroupSpec::partitioned(
        format!("{prefix}.self_attn.heads"),
        ParameterRole::AttentionHeads,
        head_units,
        attention_members,
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
    register_projection_module(
        planner,
        &layer.mlp.router,
        &format!("{prefix}.mlp.router"),
        ProjectionSharding::Replicated,
    )?;
    let experts = &layer.mlp.experts;
    let intermediate = usize::try_from(args.intermediate_size)
        .map_err(|_| Error::Parallel("GPT-OSS expert width exceeds usize".into()))?;
    let intermediate_units = aligned_partition_units(
        &format!("{prefix}.mlp.experts.intermediate"),
        intermediate,
        1,
        32,
    )?;
    planner.register(ParameterGroupSpec::partitioned(
        format!("{prefix}.mlp.experts.intermediate"),
        ParameterRole::ExpertIntermediate,
        intermediate_units,
        [
            array_parameter_member(
                format!("{prefix}.mlp.experts.gate_up_proj_blocks"),
                experts.gate_up_proj_blocks.as_ref(),
                MemberSharding::Partitioned { axis: 1 },
            )?,
            array_parameter_member(
                format!("{prefix}.mlp.experts.gate_up_proj_scales"),
                experts.gate_up_proj_scales.as_ref(),
                MemberSharding::Partitioned { axis: 1 },
            )?,
            array_parameter_member(
                format!("{prefix}.mlp.experts.gate_up_proj_bias"),
                experts.gate_up_proj_bias.as_ref(),
                MemberSharding::Partitioned { axis: 1 },
            )?,
            array_parameter_member(
                format!("{prefix}.mlp.experts.down_proj_blocks"),
                experts.down_proj_blocks.as_ref(),
                MemberSharding::Partitioned { axis: 2 },
            )?,
            array_parameter_member(
                format!("{prefix}.mlp.experts.down_proj_scales"),
                experts.down_proj_scales.as_ref(),
                MemberSharding::Partitioned { axis: 2 },
            )?,
            array_parameter_member(
                format!("{prefix}.mlp.experts.down_proj_bias"),
                experts.down_proj_bias.as_ref(),
                MemberSharding::Replicated,
            )?,
        ],
    )?)?;
    Ok(())
}

impl LoadTimeQuantizableAdapter for GptOssLayerwiseAdapter {
    fn load_time_quantized(
        &self,
        quantization: WeightQuantization,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let mut args = self.args.clone();
        args.quantization = Some(quantization);
        args.quantized_weight_configs = None;
        let mut adapter = Self::new(args, stream)?;
        adapter.sparse_expert_cache = self.sparse_expert_cache;
        Ok(adapter)
    }
}

impl ArchitectureAdapter for GptOssLayerwiseAdapter {
    type Input<'a> = &'a Array;
    type Cache = Cache;
    type Layer = TransformerBlock;
    type ForwardContext = GptOssForwardContext;

    fn model_type(&self) -> &str {
        &self.args.model_type
    }

    fn safetensors_checkpoint_plan(
        &self,
    ) -> Result<crate::backend::mlx::runtime::execution::layerwise::ArchitectureCheckpointPlan, Error>
    {
        super::checkpoint::safetensors_plan(&self.args)
            .map_err(Error::UnsupportedArchitecture)
            .map(Into::into)
    }

    fn quantization(&self) -> Option<eredu_checkpoint::WeightQuantization> {
        self.args.quantization
    }

    fn prompt_cache_model_identity(
        &self,
        topology: Option<crate::backend::mlx::MlxParallelContext>,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let layer_count = usize::try_from(self.args.num_hidden_layers)
            .map_err(|_| Exception::custom("invalid GPT-OSS cache layer count"))?;
        let local_kv_heads = match topology {
            Some(topology) if topology.tensor_parallel_size > 1 => {
                let heads = self.parallel_kv_heads.as_ref().ok_or_else(|| {
                    Error::Parallel(
                        "GPT-OSS prompt-cache identity requires planner-derived KV geometry".into(),
                    )
                })?;
                let first = *heads.first().ok_or_else(|| {
                    Error::Parallel("GPT-OSS planner returned no KV-head geometry".into())
                })?;
                if heads.iter().any(|heads| *heads != first) {
                    return Err(Error::Parallel(
                        "GPT-OSS layers unexpectedly received different local KV-head counts"
                            .into(),
                    ));
                }
                first
            }
            _ => self.args.num_key_value_heads,
        };
        Ok(PromptCacheModelIdentity {
            model_family: "gpt_oss".into(),
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
            layer_layout: PromptCacheModelIdentity::key_value_layouts(
                self.args
                    .attention_schedule
                    .iter()
                    .map(|policy| policy.window().map(|window| window.get() as i32)),
                local_kv_heads,
                self.args.head_dim,
            )
            .map_err(|error| Exception::custom(error.to_string()))?,
        })
    }

    fn save_prompt_cache(
        &self,
        cache: &mut Self::Cache,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        _stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        cache
            .save_prompt_cache(destination, descriptor, prefix_token_ids, options)
            .map_err(Into::into)
    }

    fn load_prompt_cache(
        &self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        _stream: &Stream,
    ) -> Result<(Self::Cache, PromptCacheManifest), Error> {
        let (manager, manifest) =
            open_prompt_cache(directory, expected, identity, prefix_token_ids, options)
                .map_err(|error| Exception::custom(error.to_string()))?;
        Ok((
            Cache::new_paged(
                &self.args.attention_schedule,
                manager,
                identity.topology.cache_rank_identity(),
            )?,
            manifest,
        ))
    }

    fn static_units(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        self.selected_static_units(store, &|_| true)
    }

    fn selected_static_units(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
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
            units.push(StaticUnitBindings::new(
                HEAD_UNIT,
                build_module_bindings(&self.lm_head, "lm_head", store)?,
            )?);
        }
        Ok(units)
    }

    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error> {
        if leases.len() != 3 {
            return Err(Error::UnsupportedArchitecture(format!(
                "GPT-OSS adapter received {} static leases, expected 3",
                leases.len()
            )));
        }
        if let Some(v) = &mut self.parallel_embedding {
            populate_module_from_lease(v.inner_mut(), &leases[0])?;
        } else {
            populate_module_from_lease(&mut self.embedding, &leases[0])?;
        }
        populate_module_from_lease(&mut self.norm, &leases[1])?;
        if let Some(v) = &mut self.parallel_lm_head {
            populate_module_from_lease(v.inner_mut(), &leases[2])?;
        } else {
            populate_module_from_lease(&mut self.lm_head, &leases[2])?;
        }
        Ok(())
    }

    fn validate_cache(&self, cache: &mut Cache) -> Result<(), Error> {
        if cache.layers.is_empty() {
            *cache = self.new_cache();
            return Ok(());
        }
        if cache.layers.len() != self.attention_schedule.len() {
            return Err(Error::UnsupportedArchitecture(format!(
                "GPT-OSS cache has {} layers, expected {}",
                cache.layers.len(),
                self.attention_schedule.len()
            )));
        }
        for (index, (cache, policy)) in cache
            .layers
            .iter()
            .zip(self.attention_schedule.iter())
            .enumerate()
        {
            let actual = cache
                .attention_policy()
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            if actual != *policy {
                return Err(Error::UnsupportedArchitecture(format!(
                    "GPT-OSS cache policy mismatch at layer {index}: expected {policy:?}, got {actual:?}"
                )));
            }
        }
        Ok(())
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        _cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        let hidden = self.embedding.forward(input, stream)?;
        Ok(LayerwiseForwardState {
            context: GptOssForwardContext {
                sequence_length: hidden.dim(1),
            },
            hidden,
        })
    }

    fn begin_forward_with_execution<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Self::Cache,
        execution: &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<
            '_,
        >,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        let Some(v) = &mut self.parallel_embedding else {
            return self.begin_forward(input, cache, execution.stream());
        };
        let hidden = v.forward(input, execution)?;
        Ok(LayerwiseForwardState {
            context: GptOssForwardContext {
                sequence_length: hidden.dim(1),
            },
            hidden,
        })
    }

    fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Error> {
        eredu_runtime::ExecutionGraph::chain(["text_decoder"]).map_err(Into::into)
    }

    fn layer_count(&self, group: usize) -> Result<usize, Error> {
        if group == 0 {
            Ok(self.attention_schedule.len())
        } else {
            Err(Error::UnsupportedArchitecture(format!(
                "GPT-OSS has no execution group {group}"
            )))
        }
    }

    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        self.layer_count(group)?;
        Ok(TransformerBlock::new(&self.args, index, stream)?)
    }

    fn new_expert_parallel_layer(
        &self,
        group: usize,
        index: usize,
        assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        let mut layer = self.new_layer(group, index, stream)?;
        let mut local_args = self.args.clone();
        local_args.num_local_experts = if self.sparse_expert_cache {
            0
        } else {
            i32::try_from(assignment.local_expert_count())
                .map_err(|_| Error::Parallel("local GPT-OSS expert count exceeds i32".into()))?
        };
        layer.mlp.experts = Experts::new(&local_args, stream)?;
        Ok(layer)
    }

    fn new_tensor_expert_parallel_layer(
        &self,
        group: usize,
        index: usize,
        layout: &crate::backend::mlx::runtime::distributed::parallel::LocalModelLayout,
        assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        let mut layer = self.new_parallel_layer(group, index, layout, stream)?;
        let prefix = format!("model.layers.{index}");
        let expert = layout
            .tensor(&format!("{prefix}.mlp.experts.gate_up_proj_bias"))
            .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} experts")))?;
        let mut local_args = self.args.clone();
        local_args.intermediate_size = i32::try_from(expert.local_shape()[1] / 2)
            .map_err(|_| Error::Parallel("local GPT-OSS expert width exceeds i32".into()))?;
        local_args.num_local_experts = if self.sparse_expert_cache {
            0
        } else {
            i32::try_from(assignment.local_expert_count())
                .map_err(|_| Error::Parallel("local GPT-OSS expert count exceeds i32".into()))?
        };
        layer.mlp.experts = Experts::new(&local_args, stream)?;
        Ok(layer)
    }

    fn expert_parallel_assignment(
        &self,
        topology: crate::backend::mlx::MlxParallelContext,
    ) -> Result<Option<crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>, Error>
    {
        if topology.expert_parallel_size == 1 && !self.sparse_expert_cache {
            return Ok(None);
        }
        Ok(Some(
            crate::backend::mlx::runtime::distributed::expert::ExpertAssignment::balanced(
                self.args.num_local_experts as usize,
                topology.expert_parallel_size,
                topology.expert_parallel_rank,
            )?,
        ))
    }

    fn register_parallel_parameters(
        &self,
        _context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        planner: &mut crate::backend::mlx::runtime::distributed::parallel::ParallelPlanBuilder,
        stream: &Stream,
    ) -> Result<(), Error> {
        planner.register(
            crate::backend::mlx::nn::parallel::vocab_embedding_parameter_group(
                &self.embedding,
                "model.embed_tokens",
                self.args.vocab_size as usize,
                self.args.hidden_size,
                false,
            )?,
        )?;
        crate::backend::mlx::nn::parallel::register_replicated_parameter_group(
            planner,
            &self.norm,
            "model.norm",
        )?;
        planner.register(
            crate::backend::mlx::nn::parallel::vocab_lm_head_parameter_group(
                &self.lm_head,
                "lm_head",
                self.args.hidden_size,
                self.args.vocab_size as usize,
                false,
            )?,
        )?;
        for index in 0..self.args.num_hidden_layers as usize {
            let layer = TransformerBlock::new(&self.args, index, stream)?;
            register_gpt_oss_layer_parallel_plan(planner, &layer, &self.args, index)?;
        }
        Ok(())
    }
    fn configure_parallel_static(
        &mut self,
        context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        layout: &crate::backend::mlx::runtime::distributed::parallel::LocalModelLayout,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.parallel_embedding = Some(VocabParallelEmbedding::unloaded(
            self.args.vocab_size as usize,
            self.args.hidden_size,
            self.args
                .weight_quantization_for("model.embed_tokens.weight"),
            context,
            stream,
        )?);
        self.parallel_lm_head = Some(VocabParallelLmHead::unloaded(
            self.args.hidden_size,
            self.args.vocab_size as usize,
            self.args.weight_quantization_for("lm_head.weight"),
            context,
            stream,
        )?);
        self.parallel_kv_heads = Some(planned_kv_head_layout(
            layout,
            self.attention_schedule.len(),
            self.args.head_dim,
            "model.layers",
        )?);
        Ok(())
    }
    fn new_parallel_layer(
        &self,
        group: usize,
        index: usize,
        layout: &crate::backend::mlx::runtime::distributed::parallel::LocalModelLayout,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        self.layer_count(group)?;
        let prefix = format!("model.layers.{index}");
        let find = |n: &str| {
            layout
                .tensor(&format!("{prefix}.{n}.weight"))
                .or_else(|| layout.tensor(&format!("{prefix}.{n}.inner.weight")))
        };
        let q = find("self_attn.q_proj")
            .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} query")))?;
        let k = find("self_attn.k_proj")
            .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} key")))?;
        let expert = layout
            .tensor(&format!("{prefix}.mlp.experts.gate_up_proj_bias"))
            .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} experts")))?;
        let mut args = self.args.clone();
        args.num_attention_heads = q.local_shape()[0] as i32 / args.head_dim;
        args.num_key_value_heads = k.local_shape()[0] as i32 / args.head_dim;
        args.intermediate_size = expert.local_shape()[1] as i32 / 2;
        Ok(TransformerBlock::new(&args, index, stream)?)
    }

    fn layer_checkpoint_prefix(&self, _group: usize, index: usize) -> String {
        format!("model.layers.{index}")
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
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        let prefix = format!("model.layers.{index}");
        Ok(build_module_binding_plan_with_recipes_excluding(
            layer,
            &prefix,
            store,
            self.layer_recipes(index, store)?,
            |name| self.sparse_expert_cache && name.starts_with("mlp.experts."),
        )?
        .build_bindings(store)?)
    }

    fn parallel_layer_bindings(
        &self,
        group: usize,
        index: usize,
        _layer: &Self::Layer,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        layout: &crate::backend::mlx::runtime::distributed::parallel::LocalModelLayout,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        let global = self.new_layer(group, index, stream)?;
        crate::backend::mlx::runtime::execution::layerwise::shard_layer_bindings(
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
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        let global = self.new_layer(group, index, stream)?;
        let indices = assignment.local_global_expert_ids().to_vec();
        self.layer_bindings(group, index, &global, store)?
            .into_iter()
            .map(|binding| {
                let target = binding.logical_target().unwrap_or_else(|| binding.name());
                if target.contains("mlp.experts.") {
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

    fn additional_consumed_checkpoint_keys(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Vec<String> {
        if self.sparse_expert_cache {
            store
                .source_keys()
                .into_iter()
                .filter(|key| key.contains(".mlp.experts."))
                .collect()
        } else {
            Vec::new()
        }
    }

    fn layer_unit_name(&self, _group: usize, index: usize) -> String {
        format!("gpt_oss.layer.{index:05}")
    }

    fn forward_layer(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Self::Layer,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.layer_count(group)?;
        let layer_cache = &mut cache.layers[index];
        let offset = layer_cache.offset();
        let policy = self
            .attention_schedule
            .get(index)
            .expect("validated GPT-OSS layer index");
        let mask = resident::attention_mask(policy, context.sequence_length, offset, stream)?;
        if self.sparse_expert_cache {
            let expert_cache = self.expert_cache.as_ref().ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "GPT-OSS sparse expert cache was not initialized".into(),
                )
            })?;
            let pass = if hidden.dim(1) > 1 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            };
            return Ok(layer.forward_with_expert_executor(
                hidden,
                mask.as_ref(),
                layer_cache,
                stream,
                |flat, indices, weights, stream| {
                    expert_cache
                        .execute_routes_bounded(
                            ExpertRouteBatch::new(index, flat, indices, weights, pass),
                            stream,
                            |flat, acquired, weights, stream| {
                                let started = Instant::now();
                                let mut compact_args = self.args.clone();
                                compact_args.num_local_experts = acquired.identities().len() as i32;
                                let mut bank = Experts::new(&compact_args, stream)?;
                                bank.gate_up_proj_blocks = Param::new(
                                    acquired
                                        .compact_binding("gate_up_proj_blocks", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                bank.gate_up_proj_scales = Param::new(
                                    acquired
                                        .compact_binding("gate_up_proj_scales", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                bank.gate_up_proj_bias = Param::new(
                                    acquired
                                        .compact_binding("gate_up_proj_bias", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                bank.down_proj_blocks = Param::new(
                                    acquired
                                        .compact_binding("down_proj_blocks", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                bank.down_proj_scales = Param::new(
                                    acquired
                                        .compact_binding("down_proj_scales", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                bank.down_proj_bias = Param::new(
                                    acquired
                                        .compact_binding("down_proj_bias", stream)
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
        Ok(layer.forward(hidden, mask.as_ref(), layer_cache, stream)?)
    }

    fn forward_layer_with_execution(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Self::Layer,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        execution: &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<
            '_,
        >,
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
        self.layer_count(group)?;
        let layer_cache = &mut cache.layers[index];
        let offset = layer_cache.offset();
        let policy = self
            .attention_schedule
            .get(index)
            .expect("validated GPT-OSS layer index");
        let mask =
            resident::attention_mask(policy, context.sequence_length, offset, execution.stream())?;
        Ok(layer.forward_tensor_parallel(
            hidden,
            mask.as_ref(),
            layer_cache,
            tp_group,
            execution.stream(),
        )?)
    }

    fn retained_arrays<'a>(
        &self,
        cache: &'a Self::Cache,
        _group: usize,
        index: usize,
    ) -> Vec<&'a Array> {
        cache.layers[index].retained_arrays()
    }

    fn finish(
        &mut self,
        hidden: &Array,
        _cache: &mut Self::Cache,
        _context: &Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let hidden = self.norm.forward(hidden, stream)?;
        Ok(self.lm_head.forward(&hidden, stream)?)
    }
    fn finish_with_execution(
        &mut self,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &Self::ForwardContext,
        execution: &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<
            '_,
        >,
    ) -> Result<Array, Error> {
        let Some(head) = &mut self.parallel_lm_head else {
            return self.finish(hidden, cache, context, execution.stream());
        };
        let hidden = self.norm.forward(hidden, execution.stream())?;
        head.forward(&hidden, execution)?.all_gather(execution)
    }
}

pub(crate) fn gpt_oss_expert_catalog(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    gpt_oss_expert_catalog_cartesian(args, store, None)
}

/// Builds expert-granular GPT-OSS bindings under an optional TP layout.
///
/// Expert selection is resolved before the shared semantic TP selection so
/// native MXFP4 blocks, E8M0 scales, and biases remain one atomic cache unit.
pub(crate) fn gpt_oss_expert_catalog_cartesian(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    layout: Option<&crate::backend::mlx::runtime::distributed::parallel::LocalModelLayout>,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let mut entries = Vec::new();
    for layer in 0..args.num_hidden_layers as usize {
        let prefix = format!("model.layers.{layer}.mlp.experts");
        let gguf = store
            .source_keys()
            .contains(&format!("{prefix}.gate_proj.weight"));
        for expert in 0..args.num_local_experts as usize {
            let identity = ExpertIdentity::new(layer, expert);
            let mut bindings = Vec::new();
            let selected = |name: &str| {
                DerivedWeightRecipe::source(
                    format!("{prefix}.{name}"),
                    TensorSelection::Range {
                        axis: 0,
                        start: expert,
                        end: expert + 1,
                    },
                )
            };
            let recipes = if gguf {
                let hidden = args.hidden_size as usize;
                let intermediate = args.intermediate_size as usize;
                let stack_reshape =
                    |gate: &str, up: &str, shape: Vec<usize>| DerivedWeightRecipe::Reshape {
                        input: Box::new(DerivedWeightRecipe::Stack {
                            axis: 2,
                            inputs: vec![selected(gate), selected(up)],
                        }),
                        shape,
                    };
                vec![
                    (
                        "gate_up_proj_blocks",
                        DerivedWeightRecipe::View {
                            input: Box::new(stack_reshape(
                                "gate_proj.weight",
                                "up_proj.weight",
                                vec![1, 2 * intermediate, hidden / 8],
                            )),
                            dtype: RecipeDtype::U8,
                            shape: vec![1, 2 * intermediate, hidden / 32, 16],
                        },
                    ),
                    (
                        "gate_up_proj_scales",
                        stack_reshape(
                            "gate_proj.scales",
                            "up_proj.scales",
                            vec![1, 2 * intermediate, hidden / 32],
                        ),
                    ),
                    (
                        "gate_up_proj_bias",
                        stack_reshape("gate_proj.bias", "up_proj.bias", vec![1, 2 * intermediate]),
                    ),
                    (
                        "down_proj_blocks",
                        DerivedWeightRecipe::View {
                            input: Box::new(selected("down_proj.weight")),
                            dtype: RecipeDtype::U8,
                            shape: vec![1, hidden, intermediate / 32, 16],
                        },
                    ),
                    ("down_proj_scales", selected("down_proj.scales")),
                    ("down_proj_bias", selected("down_proj.bias")),
                ]
            } else {
                [
                    "gate_up_proj_blocks",
                    "gate_up_proj_scales",
                    "gate_up_proj_bias",
                    "down_proj_blocks",
                    "down_proj_scales",
                    "down_proj_bias",
                ]
                .into_iter()
                .map(|name| (name, selected(name)))
                .collect()
            };
            let mut planned = Vec::with_capacity(recipes.len());
            for (name, recipe) in recipes {
                let metadata = recipe.infer(store)?;
                planned.push(PlannedBinding {
                    target_name: name.into(),
                    expected_shape: metadata.shape().to_vec(),
                    expected_dtype: metadata.dtype().clone(),
                    recipe,
                });
            }
            bindings.extend(
                BindingPlan::new(planned)
                    .and_then(|plan| plan.build_bindings(store))
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
            );
            let bindings = match layout {
                Some(layout) => {
                    crate::backend::mlx::runtime::execution::layerwise::shard_layer_bindings(
                        bindings, &prefix, store, layout,
                    )?
                }
                None => bindings,
            };
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    Error::UnsupportedArchitecture("GPT-OSS expert byte total overflowed".into())
                })
            })?;
            entries.push(ExpertCatalogEntry::new(
                identity,
                OffloadUnit::new(identity.unit_id(), bindings)?,
                bytes,
            )?);
        }
    }
    Ok(entries)
}

/// GPT-OSS token generation iterator using bounded layer execution.
pub type Generate<'a, S = crate::backend::mlx::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, GptOssLayerwiseModel, Cache, S>;

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

use super::{self as resident, DecoderConfig, TransformerBlock};
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
    error::Error,
    nn::{
        parallel::{VocabParallelEmbedding, VocabParallelLmHead},
        tensor::{create_attention_mask, AttentionMask},
    },
    runtime::cache::residency::{
        PagedCacheOptions, PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity,
        PromptCacheOptions, PromptCacheTopology,
    },
    runtime::cache::{
        ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache, SlidingKeyValueCache,
    },
    runtime::checkpoint::binding::{
        build_module_bindings, build_module_bindings_excluding, build_module_bindings_with_recipes,
        materialize_module_bindings, populate_module_from_arrays_excluding,
        populate_module_from_lease, populate_module_from_lease_excluding,
    },
    runtime::checkpoint::recipe::DerivedWeightRecipe,
    runtime::checkpoint::store::{GgufWeightStore, TensorSelection, WeightStore},
    runtime::distributed::parallel::{
        array_parameter_member, exact_parallel_division, register_projection_module,
        register_replicated_module, MemberSharding, ParallelPlanBuilder, ParameterGroupSpec,
        ParameterRole, ProjectionSharding,
    },
    runtime::execution::layerwise::{
        load_layerwise_model, load_safetensors_layerwise_model,
        load_tensor_parallel_layerwise_model, open_safetensors_weight_store, ArchitectureAdapter,
        LayerExecutionLoadOptions, LayerwiseForwardState, LayerwiseModel, StaticUnitBindings,
        WeightResidency,
    },
    runtime::residency::expert_cache::{
        ExpertCache, ExpertCacheError, ExpertCacheLoadOptions, ExpertCacheReport,
        ExpertCatalogEntry, ExpertIdentity, ExpertPass,
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

/// Loads Qwen2/Qwen2.5 or Qwen3 through the generalized residency engine.
pub fn load_safetensors(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerExecutionLoadOptions>,
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

/// Loads Qwen2/3 dense or MoE checkpoints through the generalized
/// tensor-parallel execution-group engine.
pub fn load_tensor_parallel_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerExecutionLoadOptions>,
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
    options: LayerExecutionLoadOptions,
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

    let execution = match residency {
        WeightResidency::LayerwiseHost(options) => {
            let adapter = DenseQwenLayerwiseAdapter::new(args, stream)?;
            load_layerwise_model(store, adapter, options, stream, weights_stream)?
        }
        WeightResidency::DenseDiskStream(options) => {
            let adapter = DenseQwenLayerwiseAdapter::new(args, stream)?;
            load_layerwise_model(store, adapter, options, stream, weights_stream)?
        }
        WeightResidency::SparseExpertCache(options) => {
            return Ok((
                load_qwen3_gguf_sparse_with_store(
                    store,
                    args,
                    options,
                    options.non_expert,
                    stream,
                    weights_stream,
                )?,
                eos_token_ids,
            ));
        }
        WeightResidency::SparseExpertCacheWithDenseLayers(options) => {
            return Ok((
                load_qwen3_gguf_sparse_with_store(
                    store,
                    args,
                    options.expert_cache,
                    options.non_expert,
                    stream,
                    weights_stream,
                )?,
                eos_token_ids,
            ));
        }
        WeightResidency::FullyResident => {
            return Err(Error::UnsupportedArchitecture(
                "the bounded GGUF Qwen3 loader does not accept fully resident policy".into(),
            ));
        }
    };
    Ok((LayerwiseDecoder { execution }, eos_token_ids))
}

/// Loads replicated Qwen3-MoE GGUF parameters for sparse expert-parallel
/// execution without materializing routed experts.
pub(crate) fn load_qwen3_gguf_sparse_ep_base(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    max_mapped_shards: usize,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(resident::Model, Arc<dyn WeightStore + Send + Sync>), Error> {
    let (args, _) = resident::prepare_gguf_checkpoint(checkpoint, metadata, "qwen3moe", true)?;
    if !args.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "sparse GGUF expert parallelism requires a Qwen3 MoE checkpoint".into(),
        ));
    }
    let store: Arc<dyn WeightStore + Send + Sync> =
        Arc::new(GgufWeightStore::new_with_max_mapped_shards(
            checkpoint.clone(),
            |name| resident::translate_gguf_weight_name(name, true),
            max_mapped_shards,
        )?);
    let adapter = DenseQwenLayerwiseAdapter::new_sparse(args.clone(), stream)?;
    let mut model = resident::Model::new(args, stream)?;

    let bindings = build_module_bindings(
        &model.model.embed_tokens,
        "model.embed_tokens",
        store.as_ref(),
    )?;
    let arrays = materialize_module_bindings(store.as_ref(), &bindings, weights_stream, stream)?;
    populate_module_from_arrays_excluding(&mut model.model.embed_tokens, &arrays, |_| false)?;

    let bindings = build_module_bindings(&model.model.norm, "model.norm", store.as_ref())?;
    let arrays = materialize_module_bindings(store.as_ref(), &bindings, weights_stream, stream)?;
    populate_module_from_arrays_excluding(&mut model.model.norm, &arrays, |_| false)?;

    if let Some(head) = &mut model.lm_head {
        let bindings = build_module_bindings(head, "lm_head", store.as_ref())?;
        let arrays =
            materialize_module_bindings(store.as_ref(), &bindings, weights_stream, stream)?;
        populate_module_from_arrays_excluding(head, &arrays, |_| false)?;
    }

    for (index, layer) in model.model.layers.iter_mut().enumerate() {
        let bindings = adapter.layer_bindings(0, index, layer, store.as_ref())?;
        let arrays =
            materialize_module_bindings(store.as_ref(), &bindings, weights_stream, stream)?;
        populate_module_from_arrays_excluding(layer, &arrays, |name| {
            name.starts_with("mlp.experts.")
        })?;
    }
    Ok((model, store))
}

fn load_qwen3_gguf_sparse_with_store(
    store: Arc<dyn WeightStore + Send + Sync>,
    args: DecoderConfig,
    options: ExpertCacheLoadOptions,
    non_expert: impl Into<LayerExecutionLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseDecoder, Error> {
    if !args.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires a Qwen3 sparse-MoE GGUF checkpoint".into(),
        ));
    }
    let adapter = DenseQwenLayerwiseAdapter::new_sparse(args.clone(), stream)?;
    let mut execution = load_layerwise_model(store, adapter, non_expert, stream, weights_stream)?;
    let checkpoint_store = execution.weight_store_arc();
    let entries = qwen3_expert_catalog(&args, checkpoint_store.as_ref())?;
    execution.adapter_mut().expert_cache = Some(ExpertCache::new_shared(
        checkpoint_store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(LayerwiseDecoder { execution })
}

/// Builds the streamed nonexpert Qwen3 execution base used by distributed EP.
pub(crate) fn load_qwen3_sparse_ep_base_with_store(
    store: Arc<dyn WeightStore + Send + Sync>,
    args: DecoderConfig,
    non_expert: crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseDecoder, Error> {
    if !args.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "streamed sparse expert parallelism requires Qwen3 MoE".into(),
        ));
    }
    let adapter = DenseQwenLayerwiseAdapter::new_sparse(args, stream)?;
    let execution = load_layerwise_model(store, adapter, non_expert, stream, weights_stream)?;
    Ok(LayerwiseDecoder { execution })
}

/// Loads sparse Qwen3 with layerwise non-expert weights and expert-granular caching.
pub fn load_qwen3_sparse_expert_cache_model(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseDecoder, Error> {
    load_qwen3_sparse_expert_cache_model_with_non_expert(
        model_dir,
        options,
        options.non_expert,
        stream,
        weights_stream,
    )
}

/// Loads sparse Qwen3 with expert caching and disk-streamed non-expert units.
pub fn load_qwen3_sparse_expert_cache_model_with_dense_layers(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    non_expert: crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseDecoder, Error> {
    load_qwen3_sparse_expert_cache_model_with_non_expert(
        model_dir,
        options,
        non_expert,
        stream,
        weights_stream,
    )
}

fn load_qwen3_sparse_expert_cache_model_with_non_expert(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    non_expert: impl Into<LayerExecutionLoadOptions>,
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
    let adapter = DenseQwenLayerwiseAdapter::new_sparse(args.clone(), stream)?;
    let mut execution =
        load_safetensors_layerwise_model(model_dir, adapter, non_expert, stream, weights_stream)?;
    let store = execution.weight_store_arc();
    let entries = qwen3_expert_catalog(&args, store.as_ref())?;
    let cache = ExpertCache::new_shared(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?;
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
            sparse_expert_cache: false,
            expert_cache: None,
        })
    }

    fn new_sparse(args: DecoderConfig, stream: &Stream) -> Result<Self, Error> {
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

pub(crate) fn register_qwen_layer_parallel_plan(
    planner: &mut ParallelPlanBuilder,
    layer: &TransformerBlock,
    prefix: &str,
) -> Result<(), Error> {
    let attention = &layer.self_attn;
    for (name, projection) in [
        ("q_proj", &attention.q_proj),
        ("k_proj", &attention.k_proj),
        ("v_proj", &attention.v_proj),
    ] {
        register_projection_module(
            planner,
            projection,
            &format!("{prefix}.self_attn.{name}"),
            ProjectionSharding::Column,
        )?;
    }
    register_projection_module(
        planner,
        &attention.o_proj,
        &format!("{prefix}.self_attn.o_proj"),
        ProjectionSharding::Row,
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
            for (name, projection, sharding) in [
                ("gate_proj", &mlp.gate_proj, ProjectionSharding::Column),
                ("up_proj", &mlp.up_proj, ProjectionSharding::Column),
                ("down_proj", &mlp.down_proj, ProjectionSharding::Row),
            ] {
                register_projection_module(
                    planner,
                    projection,
                    &format!("{prefix}.mlp.{name}"),
                    sharding,
                )?;
            }
        }
        resident::FeedForward::Moe(moe) => {
            register_replicated_module(planner, &moe.gate, &format!("{prefix}.mlp.gate"))?;
            let experts = &moe.experts;
            let intermediate = usize::try_from(experts.intermediate_dim)
                .map_err(|_| Error::Parallel("Qwen expert width exceeds usize".into()))?;
            let segments = vec![0..intermediate, intermediate..2 * intermediate];
            let mut gate_up = vec![array_parameter_member(
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
                    gate_up.push(array_parameter_member(
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
            let mut down = vec![array_parameter_member(
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
                    down.push(array_parameter_member(
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
    )
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
        let local_kv_heads = topology.map_or(Ok(self.args.num_key_value_heads), |topology| {
            exact_parallel_division(
                "dense-Qwen prompt-cache KV heads",
                self.args.num_key_value_heads,
                topology.tensor_parallel_size,
            )
        })?;
        Ok(PromptCacheModelIdentity {
            model_family: "dense_qwen".into(),
            effective_model_type: self.args.model_type.clone(),
            architecture_fingerprint: resident::prompt_cache_architecture_fingerprint(&self.args),
            layer_count,
            global_layer_start: 0,
            global_layer_end: layer_count,
            sink_tokens: 0,
            topology: topology.map_or_else(
                PromptCacheTopology::default,
                PromptCacheTopology::for_parallel_topology,
            ),
            layer_layout: PromptCacheModelIdentity::key_value_layouts(
                self.args.attention_schedule.iter().map(|policy| {
                    policy.window().map(|window| {
                        i32::try_from(window.get())
                            .expect("validated dense-Qwen attention window fits i32")
                    })
                }),
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

    fn execution_group_count(&self) -> usize {
        1
    }

    fn execution_group_id(&self, group: usize) -> Result<String, Error> {
        (group == 0).then(|| "text_decoder".into()).ok_or_else(|| {
            Error::UnsupportedArchitecture(format!("dense Qwen has no group {group}"))
        })
    }

    fn layer_count(&self, group: usize) -> Result<usize, Error> {
        self.execution_group_id(group)?;
        usize::try_from(self.args.num_hidden_layers).map_err(|_| {
            Error::UnsupportedArchitecture(format!(
                "dense-Qwen layer count {} is invalid",
                self.args.num_hidden_layers
            ))
        })
    }

    fn static_units(&self, store: &dyn WeightStore) -> Result<Vec<StaticUnitBindings>, Error> {
        let mut units = vec![
            StaticUnitBindings::new(
                EMBEDDING_UNIT,
                build_module_bindings(&self.embedding, "model.embed_tokens", store)?,
            )?,
            StaticUnitBindings::new(
                NORM_UNIT,
                build_module_bindings(&self.norm, "model.norm", store)?,
            )?,
        ];
        if let Some(head) = &self.lm_head {
            units.push(StaticUnitBindings::new(
                HEAD_UNIT,
                build_module_bindings(head, "lm_head", store)?,
            )?);
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
        self.execution_group_id(group)?;
        let index = i32::try_from(index).map_err(|_| {
            Error::UnsupportedArchitecture("dense-Qwen layer index exceeds i32".into())
        })?;
        Ok(TransformerBlock::new_for_layer(&self.args, index, stream)?)
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
            register_qwen_layer_parallel_plan(planner, &layer, &format!("model.layers.{index}"))?;
        }
        Ok(())
    }

    fn configure_parallel_static(
        &mut self,
        context: crate::runtime::distributed::parallel::ParallelBuildContext,
        _layout: &crate::runtime::distributed::parallel::LocalModelLayout,
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
        self.execution_group_id(group)?;
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
        Ok(TransformerBlock::new_for_layer(
            &args,
            index as i32,
            stream,
        )?)
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
        if self.sparse_expert_cache {
            Ok(build_module_bindings_excluding(
                layer,
                &format!("model.layers.{index}"),
                store,
                |name| name.starts_with("mlp.experts."),
            )?)
        } else {
            let prefix = format!("model.layers.{index}");
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
            }
            Ok(build_module_bindings_with_recipes(
                layer, &prefix, store, recipes,
            )?)
        }
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
                            index,
                            flat,
                            indices,
                            weights,
                            pass,
                            stream,
                            |flat, acquired, weights, stream| {
                                if acquired.is_empty() {
                                    return Err(ExpertCacheError::EmptyRoutedBank {
                                        architecture: "Qwen3",
                                    });
                                }
                                let started = Instant::now();
                                let prefix = format!("model.layers.{index}.mlp.experts");
                                let mut bank = resident::Experts::new(
                                    acquired.identities().len() as i32,
                                    self.args.hidden_size,
                                    self.args.moe_intermediate_size,
                                    self.args
                                        .weight_quantization_for(&format!("{prefix}.gate_up_proj")),
                                    self.args
                                        .weight_quantization_for(&format!("{prefix}.down_proj")),
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

pub(crate) fn qwen3_expert_catalog(
    args: &DecoderConfig,
    store: &dyn WeightStore,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    qwen3_expert_catalog_at(args, store, "model.layers")
}

pub(crate) fn qwen3_expert_catalog_at(
    args: &DecoderConfig,
    store: &dyn WeightStore,
    layer_root: &str,
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
        runtime::residency::policy::{MemoryTier, OffloadConfig, ResidencyPolicy},
        runtime::{
            distributed::{
                parallel::{ParallelBuildContext, ShardingPolicy},
                topology::{DeviceAssignment, ParallelTopology},
            },
            execution::layerwise::{
                ExecutionResidency, LayerExecutionLoadOptions, LayerwiseLoadOptions,
            },
        },
    };

    fn tensor_parallel_context() -> ParallelBuildContext {
        ParallelBuildContext::new(
            ParallelTopology::from_rank(2, 0, 2, 1, 1, DeviceAssignment::new(DeviceType::Cpu, 0))
                .unwrap(),
            ShardingPolicy::Require,
        )
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
            moe_intermediate_size: if moe { 8 } else { 0 },
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
            &[4, 8, 8]
        );
        assert_eq!(
            layout
                .tensor("model.layers.0.mlp.experts.down_proj")
                .unwrap()
                .local_shape(),
            &[4, 8, 4]
        );
        assert!(model
            .parallel_info()
            .unwrap()
            .owned_tensors()
            .iter()
            .any(|name| name == "model.layers.0.mlp.experts.gate_up_proj"));
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
            LayerExecutionLoadOptions::FullyResident,
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
    fn generalized_fully_resident_tensor_parallel_reports_local_and_global_memory() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let weights = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut fixture = dense_qwen::Model::new(args(false), execution.stream()).unwrap();
        initialize(&mut fixture, execution.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, false, execution.stream());

        let model = load_tensor_parallel_model(
            dir.path(),
            LayerExecutionLoadOptions::FullyResident,
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

        let non_expert = LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap());
        let expert_options = ExpertCacheLoadOptions::new(
            non_expert,
            OffloadConfig::new(None, None, 1).unwrap(),
            1 << 20,
            1,
        )
        .unwrap();
        let dense = crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions::new(
            u64::MAX,
            u64::MAX,
            1,
            1,
            1,
        )
        .unwrap();
        let mut cached = load_qwen3_sparse_expert_cache_model_with_dense_layers(
            dir.path(),
            expert_options,
            dense,
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
        let expert_options = ExpertCacheLoadOptions::new(
            non_expert,
            OffloadConfig::new(None, None, 1).unwrap(),
            1 << 20,
            1,
        )
        .unwrap();
        let mut cached = load_qwen3_sparse_expert_cache_model(
            dir.path(),
            expert_options,
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

//! Bounded layer execution for DeepSeek-V3 and DeepSeek-R1 checkpoints.

use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
    sync::Arc,
    time::Instant,
};

use safemlx::{
    error::Exception,
    module::{Module, ModuleParameters, Param},
    nn,
    ops::{indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};

use crate::{
    api::{
        common::{self, generation::CausalLm},
        deepseek_v3::{self as resident, Cache, DecoderLayer, LayerPolicy, ModelArgs},
        input,
    },
    error::Error,
    nn::{
        parallel::{VocabParallelEmbedding, VocabParallelLmHead},
        tensor::create_causal_mask,
    },
    runtime::cache::residency::{
        validate_prompt_cache_model_identity, CacheResidencyPolicy, PagedCacheOptions,
        PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
        PromptCacheTopology,
    },
    runtime::checkpoint::binding::{
        build_module_bindings_with_recipes, canonical_checkpoint_name, materialize_module_bindings,
        populate_module_from_arrays_excluding, populate_module_from_lease,
        populate_module_from_lease_excluding,
    },
    runtime::checkpoint::recipe::DerivedWeightRecipe,
    runtime::checkpoint::store::{GgufWeightStore, TensorSelection, WeightStore},
    runtime::distributed::parallel::{
        aligned_partition_units, array_parameter_member, partitioned_projection_members,
        register_partitioned_projection_group, register_projection_module,
        register_replicated_module, MemberSharding, ParallelPlanBuilder, ParameterGroupSpec,
        ParameterRole, ProjectionSharding,
    },
    runtime::execution::layerwise::{
        load_layerwise_model, load_safetensors_layerwise_model,
        load_tensor_parallel_layerwise_model, open_safetensors_weight_store,
        transformed_module_weight_store, ArchitectureAdapter, LayerExecutionLoadOptions,
        LayerwiseForwardState, LayerwiseModel, StaticUnitBindings, WeightResidency,
    },
    runtime::residency::expert_cache::{
        ExpertCache, ExpertCacheError, ExpertCacheLoadOptions, ExpertCacheReport,
        ExpertCatalogEntry, ExpertIdentity, ExpertPass, ExpertRouteBatch,
    },
    runtime::residency::manager::{OffloadUnit, ResidencyReport, ResidentUnitLease, WeightBinding},
};

const EMBEDDING_UNIT: &str = "deepseek_v3.static.embedding";
const NORM_UNIT: &str = "deepseek_v3.static.norm";
const HEAD_UNIT: &str = "deepseek_v3.static.output";

/// DeepSeek-V3/R1 causal LM using bounded residency for decoder blocks.
pub struct DeepSeekV3LayerwiseModel {
    execution: LayerwiseModel<DeepSeekV3LayerwiseAdapter>,
}

impl DeepSeekV3LayerwiseModel {
    /// Returns the validated architecture arguments.
    pub fn args(&self) -> &ModelArgs {
        self.execution.adapter().args()
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

    /// Returns rank-local generalized parallel information when applicable.
    pub fn parallel_info(
        &self,
    ) -> Option<&crate::runtime::execution::layerwise::ParallelModelInfo> {
        self.execution.parallel_info()
    }

    /// Creates one compressed MLA cache per decoder block.
    pub fn new_cache(&self) -> Cache {
        self.execution.adapter().new_cache()
    }

    /// Creates ordinary or paged compressed attention state independently of weight residency.
    pub fn new_cache_with_options(&self, policy: CacheResidencyPolicy) -> Result<Cache, Error> {
        Cache::new_with_options_and_rank(
            &self.args().layer_schedule,
            policy,
            self.execution.prompt_cache_rank_identity(),
        )
        .map_err(Into::into)
    }

    /// Lazily catalogs a compatible persisted compressed prefix.
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
        let layer_count = args.layer_schedule.len();
        let identity = PromptCacheModelIdentity {
            model_family: "deepseek_v3".into(),
            effective_model_type: args.model_type.clone(),
            architecture_fingerprint: resident::prompt_cache_architecture_fingerprint(args),
            layer_count,
            global_layer_start: 0,
            global_layer_end: layer_count,
            sink_tokens: 0,
            topology: Default::default(),
            layer_layout: PromptCacheModelIdentity::compressed_layouts(
                layer_count,
                args.kv_lora_rank,
                args.qk_rope_head_dim,
            )
            .map_err(|error| Exception::custom(error.to_string()))?,
        };
        validate_prompt_cache_model_identity(expected, &identity)
            .map_err(|error| Exception::custom(error.to_string()))?;
        Cache::load_prompt_cache(
            &self.args().layer_schedule,
            directory,
            expected,
            &identity,
            prefix_token_ids,
            options,
        )
        .map_err(Into::into)
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

    /// Runs a rank-local tensor-parallel forward pass through the generalized engine.
    pub fn forward_tensor_parallel(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution
            .forward_tensor_parallel(inputs, cache, group, stream)
    }

    /// Runs MLA and dense/MoE decoder blocks while preserving compressed state.
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution.forward(inputs, cache, stream)
    }

    /// Runs the canonical execution path with stable per-layer observation points.
    pub fn forward_with_observer(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
        observer: &mut dyn crate::runtime::execution::inspection::ActivationObserver,
    ) -> Result<Array, Error> {
        self.execution
            .forward_with_observer(inputs, cache, stream, observer)
    }

    /// Runs streamed layers while delegating routed experts to a caller.
    pub(crate) fn forward_with_expert_executor<F>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
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
            |_adapter, _group, index, layer, hidden, cache, context, stream| {
                Ok(layer.forward_sparse_experts(
                    hidden,
                    mask.or(context.mask.as_ref()),
                    Some(&mut cache.layers[index]),
                    stream,
                    |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                )?)
            },
        )
    }

    /// Clears temporary decoder blocks from the execution device.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        self.execution.clear_device_group("text_decoder")
    }
}

impl CausalLm<Cache> for DeepSeekV3LayerwiseModel {
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

/// Loads DeepSeek-V3/R1 through the generalized execution engine.
pub fn load_deepseek_v3_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerExecutionLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV3LayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let residency = options.weight_residency();
    crate::api::structural::validate_safetensors_load_path(
        crate::api::ModelKind::DeepSeekV3,
        model_dir,
        crate::api::ModelLoadOptions::default().with_weight_residency(residency),
    )?;
    let args = resident::get_model_args(model_dir)?;
    args.validate()?;
    let adapter = DeepSeekV3LayerwiseAdapter::new(args, stream)?;
    Ok(DeepSeekV3LayerwiseModel {
        execution: load_safetensors_layerwise_model(
            model_dir,
            adapter,
            options,
            stream,
            weights_stream,
        )?,
    })
}

pub(crate) fn execute_transformed_deepseek_v3_model(
    model: resident::Model,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV3LayerwiseModel, Error> {
    let adapter = DeepSeekV3LayerwiseAdapter::new(model.args.clone(), stream)?;
    let store = transformed_module_weight_store(&model)?;
    Ok(DeepSeekV3LayerwiseModel {
        execution: load_layerwise_model(
            store,
            adapter,
            LayerExecutionLoadOptions::FullyResident,
            stream,
            weights_stream,
        )?,
    })
}

/// Loads DeepSeek-V3/R1 through the generalized tensor-parallel engine.
pub fn load_deepseek_v3_tensor_parallel_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerExecutionLoadOptions>,
    build: crate::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV3LayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let residency = options.weight_residency();
    if model_dir
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        let checkpoint = GgufCheckpoint::open(model_dir)?;
        let metadata = crate::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        return load_deepseek_v3_gguf_tensor_parallel_model(
            &checkpoint,
            &metadata,
            options,
            build,
            stream,
            weights_stream,
        )
        .map(|(model, _)| model);
    }
    crate::api::structural::validate_safetensors_load_path(
        crate::api::ModelKind::DeepSeekV3,
        model_dir,
        crate::api::ModelLoadOptions::default().with_weight_residency(residency),
    )?;
    let args = resident::get_model_args(model_dir)?;
    args.validate()?;
    let adapter = DeepSeekV3LayerwiseAdapter::new(args, stream)?;
    Ok(DeepSeekV3LayerwiseModel {
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

pub(crate) fn load_deepseek_v3_gguf_tensor_parallel_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: LayerExecutionLoadOptions,
    build: crate::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(DeepSeekV3LayerwiseModel, Vec<u32>), Error> {
    let residency = options.weight_residency();
    crate::api::structural::validate_gguf(
        crate::api::GgufArchitecture::DeepSeek2,
        checkpoint,
        metadata,
        crate::api::ModelLoadOptions::default().with_weight_residency(residency),
    )
    .into_loader_result()?;
    let prepared = resident::prepare_gguf_checkpoint(checkpoint, metadata, None, weights_stream)?;
    let store: Arc<dyn WeightStore + Send + Sync> =
        Arc::new(GgufWeightStore::new_with_max_mapped_shards(
            checkpoint.clone(),
            resident::translate_gguf_weight_name,
            options.max_mapped_shards(),
        )?);
    let execution = load_tensor_parallel_layerwise_model(
        store,
        DeepSeekV3LayerwiseAdapter::new(prepared.args, stream)?,
        options,
        build,
        stream,
        weights_stream,
    )?;
    Ok((
        DeepSeekV3LayerwiseModel { execution },
        prepared.eos_token_ids,
    ))
}

pub(crate) fn load_deepseek_v3_gguf_layerwise_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    residency: WeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(DeepSeekV3LayerwiseModel, Vec<u32>), Error> {
    crate::api::structural::validate_gguf(
        crate::api::GgufArchitecture::DeepSeek2,
        checkpoint,
        metadata,
        crate::api::ModelLoadOptions::default().with_weight_residency(residency),
    )
    .into_loader_result()?;
    let prepared = resident::prepare_gguf_checkpoint(checkpoint, metadata, None, weights_stream)?;
    let args = prepared.args;
    let store: Arc<dyn WeightStore + Send + Sync> =
        Arc::new(GgufWeightStore::new_with_max_mapped_shards(
            checkpoint.clone(),
            resident::translate_gguf_weight_name,
            residency.max_mapped_shards(),
        )?);
    let execution = match residency {
        WeightResidency::LayerwiseHost(options) => load_layerwise_model(
            store,
            DeepSeekV3LayerwiseAdapter::new(args, stream)?,
            options,
            stream,
            weights_stream,
        )?,
        WeightResidency::DenseDiskStream(options) => load_layerwise_model(
            store,
            DeepSeekV3LayerwiseAdapter::new(args, stream)?,
            options,
            stream,
            weights_stream,
        )?,
        WeightResidency::SparseExpertCache(options) => {
            return Ok((
                load_deepseek_gguf_sparse_with_store(
                    store,
                    args,
                    options,
                    options.non_expert,
                    stream,
                    weights_stream,
                )?,
                prepared.eos_token_ids,
            ));
        }
        WeightResidency::SparseExpertCacheWithDenseLayers(options) => {
            return Ok((
                load_deepseek_gguf_sparse_with_store(
                    store,
                    args,
                    options.expert_cache,
                    options.non_expert,
                    stream,
                    weights_stream,
                )?,
                prepared.eos_token_ids,
            ));
        }
        WeightResidency::FullyResident => load_layerwise_model(
            store,
            DeepSeekV3LayerwiseAdapter::new(args, stream)?,
            LayerExecutionLoadOptions::FullyResident,
            stream,
            weights_stream,
        )?,
    };
    Ok((
        DeepSeekV3LayerwiseModel { execution },
        prepared.eos_token_ids,
    ))
}

/// Loads replicated DeepSeek GGUF parameters for sparse expert-parallel
/// execution without materializing any routed-expert bank.
pub(crate) fn load_deepseek_v3_gguf_sparse_ep_base(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    max_mapped_shards: usize,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(resident::Model, Arc<dyn WeightStore + Send + Sync>), Error> {
    let prepared = resident::prepare_gguf_checkpoint(checkpoint, metadata, None, weights_stream)?;
    let args = prepared.args;
    let store: Arc<dyn WeightStore + Send + Sync> =
        Arc::new(GgufWeightStore::new_with_max_mapped_shards(
            checkpoint.clone(),
            resident::translate_gguf_weight_name,
            max_mapped_shards,
        )?);
    let adapter = DeepSeekV3LayerwiseAdapter::new_sparse(args.clone(), stream)?;
    let mut model = resident::Model::new(args, stream)?;

    let bindings = build_module_bindings_with_recipes(
        &model.model.embed_tokens,
        "model.embed_tokens",
        store.as_ref(),
        BTreeMap::new(),
    )?;
    let arrays = materialize_module_bindings(store.as_ref(), &bindings, weights_stream, stream)?;
    populate_module_from_arrays_excluding(&mut model.model.embed_tokens, &arrays, |_| false)?;

    let bindings = build_module_bindings_with_recipes(
        &model.model.norm,
        "model.norm",
        store.as_ref(),
        BTreeMap::new(),
    )?;
    let arrays = materialize_module_bindings(store.as_ref(), &bindings, weights_stream, stream)?;
    populate_module_from_arrays_excluding(&mut model.model.norm, &arrays, |_| false)?;

    let bindings = build_module_bindings_with_recipes(
        &model.lm_head,
        "lm_head",
        store.as_ref(),
        BTreeMap::new(),
    )?;
    let arrays = materialize_module_bindings(store.as_ref(), &bindings, weights_stream, stream)?;
    populate_module_from_arrays_excluding(&mut model.lm_head, &arrays, |_| false)?;

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

fn load_deepseek_gguf_sparse_with_store(
    store: Arc<dyn WeightStore + Send + Sync>,
    args: ModelArgs,
    options: ExpertCacheLoadOptions,
    non_expert: impl Into<LayerExecutionLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV3LayerwiseModel, Error> {
    let adapter = DeepSeekV3LayerwiseAdapter::new_sparse(args.clone(), stream)?;
    let mut execution = load_layerwise_model(store, adapter, non_expert, stream, weights_stream)?;
    let checkpoint_store = execution.checkpoint_store_arc();
    let entries = deepseek_expert_catalog(&args, checkpoint_store.as_ref())?;
    execution.adapter_mut().expert_cache = Some(ExpertCache::new_shared(
        checkpoint_store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(DeepSeekV3LayerwiseModel { execution })
}

/// Builds the streamed nonexpert DeepSeek execution base used by distributed EP.
pub(crate) fn load_deepseek_v3_sparse_ep_base_with_store(
    store: Arc<dyn WeightStore + Send + Sync>,
    args: ModelArgs,
    non_expert: crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV3LayerwiseModel, Error> {
    args.validate()?;
    let adapter = DeepSeekV3LayerwiseAdapter::new_sparse(args, stream)?;
    let execution = load_layerwise_model(store, adapter, non_expert, stream, weights_stream)?;
    Ok(DeepSeekV3LayerwiseModel { execution })
}

/// Loads DeepSeek-V3/R1 with layerwise non-expert weights and expert-granular caching.
pub fn load_deepseek_v3_sparse_expert_cache_model(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV3LayerwiseModel, Error> {
    load_deepseek_v3_sparse_expert_cache_model_with_non_expert(
        model_dir,
        options,
        options.non_expert,
        stream,
        weights_stream,
    )
}

/// Loads DeepSeek-V3/R1 with expert caching and disk-streamed non-expert units.
pub fn load_deepseek_v3_sparse_expert_cache_model_with_dense_layers(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    non_expert: crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV3LayerwiseModel, Error> {
    load_deepseek_v3_sparse_expert_cache_model_with_non_expert(
        model_dir,
        options,
        non_expert,
        stream,
        weights_stream,
    )
}

fn load_deepseek_v3_sparse_expert_cache_model_with_non_expert(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    non_expert: impl Into<LayerExecutionLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV3LayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    crate::api::structural::validate_safetensors_load_path(
        crate::api::ModelKind::DeepSeekV3,
        model_dir,
        crate::api::ModelLoadOptions::default()
            .with_weight_residency(WeightResidency::SparseExpertCache(options)),
    )?;
    let args = resident::get_model_args(model_dir)?;
    args.validate()?;
    let adapter = DeepSeekV3LayerwiseAdapter::new_sparse(args.clone(), stream)?;
    let mut execution =
        load_safetensors_layerwise_model(model_dir, adapter, non_expert, stream, weights_stream)?;
    let store = execution.checkpoint_store_arc();
    let entries = deepseek_expert_catalog(&args, store.as_ref())?;
    let cache = ExpertCache::new_shared(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?;
    execution.adapter_mut().expert_cache = Some(cache);
    Ok(DeepSeekV3LayerwiseModel { execution })
}

/// Adapter for compressed MLA and mixed dense/MoE DeepSeek decoder blocks.
pub struct DeepSeekV3LayerwiseAdapter {
    args: ModelArgs,
    embedding: MaybeQuantized<nn::Embedding>,
    norm: nn::RmsNorm,
    lm_head: MaybeQuantized<nn::Linear>,
    parallel_embedding: Option<VocabParallelEmbedding>,
    parallel_lm_head: Option<VocabParallelLmHead>,
    sparse_expert_cache: bool,
    expert_cache: Option<ExpertCache>,
}

impl DeepSeekV3LayerwiseAdapter {
    pub(crate) fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        Ok(Self {
            embedding: common::linear::unloaded_maybe_quantized_embedding(
                args.vocab_size,
                args.hidden_size,
                args.weight_quantization_for("model.embed_tokens.weight"),
                stream,
            )?,
            norm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
            lm_head: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.vocab_size,
                false,
                args.weight_quantization_for("lm_head.weight"),
                stream,
            )?,
            parallel_embedding: None,
            parallel_lm_head: None,
            sparse_expert_cache: false,
            expert_cache: None,
            args,
        })
    }

    fn new_sparse(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let mut adapter = Self::new(args, stream)?;
        adapter.sparse_expert_cache = true;
        Ok(adapter)
    }

    /// Returns the validated architecture arguments.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    fn new_cache(&self) -> Cache {
        Cache::new(&self.args.layer_schedule)
    }

    fn recipes_for_layer(
        &self,
        layer: &DecoderLayer,
        index: usize,
        store: &dyn WeightStore,
    ) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
        let prefix = format!("model.layers.{index}");
        let normalized = normalized_checkpoint_keys(store);
        let keys = store.keys();
        let mut recipes = BTreeMap::new();

        for local_name in layer.parameters().flatten().keys() {
            let destination = format!("{prefix}.{local_name}");
            let canonical = canonical_checkpoint_name(&destination);
            if keys.contains(&destination) || keys.contains(&canonical) {
                continue;
            }
            if let Some((projection, component)) = expert_destination(local_name.as_ref()) {
                let mut inputs = Vec::with_capacity(self.args.n_routed_experts as usize);
                for expert in 0..self.args.n_routed_experts {
                    let runtime = format!("{prefix}.mlp.experts.{expert}.{projection}.{component}");
                    let raw = normalized.get(&runtime).ok_or_else(|| {
                        Error::UnsupportedArchitecture(format!(
                            "DeepSeek-V3 checkpoint is missing split expert tensor {runtime}"
                        ))
                    })?;
                    inputs.push(DerivedWeightRecipe::source(
                        raw.clone(),
                        TensorSelection::Full,
                    ));
                }
                recipes.insert(
                    local_name.to_string(),
                    DerivedWeightRecipe::Stack { axis: 0, inputs },
                );
                continue;
            }
            let raw = normalized
                .get(&destination)
                .or_else(|| normalized.get(&canonical))
                .ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "DeepSeek-V3 checkpoint is missing runtime parameter {canonical}"
                    ))
                })?;
            recipes.insert(
                local_name.to_string(),
                DerivedWeightRecipe::source(raw.clone(), TensorSelection::Full),
            );
        }
        Ok(recipes)
    }
}

fn normalized_checkpoint_keys(store: &dyn WeightStore) -> BTreeMap<String, String> {
    store
        .keys()
        .into_iter()
        .map(|raw| (canonical_checkpoint_name(&raw), raw))
        .collect()
}

fn expert_destination(local_name: &str) -> Option<(&'static str, &'static str)> {
    ["gate_proj", "up_proj", "down_proj"]
        .into_iter()
        .find_map(|projection| {
            [
                ("", "weight"),
                ("_scale_inv", "weight_scale_inv"),
                ("_scales", "scales"),
                ("_biases", "biases"),
            ]
            .into_iter()
            .find_map(|(runtime_suffix, checkpoint_component)| {
                (local_name == format!("mlp.experts.{projection}{runtime_suffix}"))
                    .then_some((projection, checkpoint_component))
            })
        })
}

/// Per-forward causal mask shared by all MLA blocks.
pub struct DeepSeekV3ForwardContext {
    mask: Option<Array>,
}

fn register_deepseek_layer_parallel_plan(
    planner: &mut ParallelPlanBuilder,
    layer: &DecoderLayer,
    index: usize,
) -> Result<(), Error> {
    let prefix = format!("model.layers.{index}");
    let attention = &layer.self_attn;
    let attention_prefix = format!("{prefix}.self_attn");
    let mut projection_names = Vec::new();
    for (name, projection) in [
        ("q_proj", attention.q_proj.as_ref()),
        ("q_b_proj", attention.q_b_proj.as_ref()),
        ("kv_b_proj", attention.kv_b_proj.as_ref()),
    ] {
        if let Some(projection) = projection {
            projection_names.push((
                projection,
                format!("{attention_prefix}.{name}"),
                ProjectionSharding::Column,
            ));
        }
    }
    projection_names.push((
        &attention.o_proj,
        format!("{attention_prefix}.o_proj"),
        ProjectionSharding::Row,
    ));
    let projections = projection_names
        .iter()
        .map(|(projection, name, sharding)| (*projection, name.as_str(), *sharding))
        .collect::<Vec<_>>();
    let preferred_heads = usize::try_from(attention.num_heads)
        .map_err(|_| Error::Parallel("DeepSeek MLA head count exceeds usize".into()))?;
    let (mut head_units, mut head_members) =
        partitioned_projection_members(&projections, preferred_heads)?;
    let mut packed_names = Vec::new();
    for (name, projection) in [
        ("k_b_proj", attention.k_b_proj.as_ref()),
        ("v_b_proj", attention.v_b_proj.as_ref()),
    ] {
        if let Some(projection) = projection {
            packed_names.push((
                projection,
                format!("{attention_prefix}.{name}"),
                ProjectionSharding::Column,
            ));
        }
    }
    if !packed_names.is_empty() {
        let packed = packed_names
            .iter()
            .map(|(projection, name, sharding)| (*projection, name.as_str(), *sharding))
            .collect::<Vec<_>>();
        let (packed_units, packed_members) = partitioned_projection_members(&packed, head_units)?;
        head_units = packed_units;
        head_members.extend(packed_members);
    }
    planner.register(ParameterGroupSpec::partitioned(
        format!("{attention_prefix}.heads"),
        ParameterRole::AttentionHeads,
        head_units,
        head_members,
    )?)?;
    for (name, projection) in [
        ("q_a_proj", attention.q_a_proj.as_ref()),
        ("kv_a_proj_with_mqa", Some(&attention.kv_a_proj_with_mqa)),
    ] {
        if let Some(projection) = projection {
            register_projection_module(
                planner,
                projection,
                &format!("{prefix}.self_attn.{name}"),
                ProjectionSharding::Replicated,
            )?;
        }
    }
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
                        mlp: &resident::Mlp,
                        prefix: &str,
                        intermediate: i32|
     -> Result<(), Error> {
        let intermediate = usize::try_from(intermediate)
            .map_err(|_| Error::Parallel("DeepSeek feed-forward width exceeds usize".into()))?;
        let gate = format!("{prefix}.gate_proj");
        let up = format!("{prefix}.up_proj");
        let down = format!("{prefix}.down_proj");
        register_partitioned_projection_group(
            planner,
            &format!("{prefix}.intermediate"),
            ParameterRole::FeedForwardIntermediate,
            &[
                (&mlp.gate_proj, gate.as_str(), ProjectionSharding::Column),
                (&mlp.up_proj, up.as_str(), ProjectionSharding::Column),
                (&mlp.down_proj, down.as_str(), ProjectionSharding::Row),
            ],
            intermediate,
        )
    };
    match &layer.mlp {
        resident::FeedForward::Dense(mlp) => register_mlp(
            planner,
            mlp,
            &format!("{prefix}.mlp"),
            mlp.gate_proj.output_dims,
        )?,
        resident::FeedForward::Moe(moe) => {
            register_replicated_module(planner, &moe.gate, &format!("{prefix}.mlp.gate"))?;
            register_mlp(
                planner,
                &moe.shared_experts,
                &format!("{prefix}.mlp.shared_experts"),
                moe.shared_experts.gate_proj.output_dims,
            )?;
            let experts = &moe.experts;
            let intermediate = usize::try_from(experts.intermediate_size).map_err(|_| {
                Error::Parallel("DeepSeek routed-expert width exceeds usize".into())
            })?;
            let down_alignment = if experts.use_fp8 {
                128
            } else {
                experts
                    .down_affine
                    .or(experts.down_iquant)
                    .map_or(Ok(1usize), |quantization| {
                        usize::try_from(quantization.group_size()).map_err(|_| {
                            Error::Parallel(
                                "DeepSeek expert quantization group exceeds usize".into(),
                            )
                        })
                    })?
            };
            let expert_units = aligned_partition_units(
                &format!("{prefix}.mlp.experts.intermediate"),
                intermediate,
                1,
                down_alignment,
            )?;
            let mut members = Vec::new();
            for (name, value) in [
                ("gate_proj", experts.gate_proj.as_ref().as_ref()),
                (
                    "gate_proj_scale_inv",
                    experts.gate_proj_scale_inv.as_ref().as_ref(),
                ),
                (
                    "gate_proj_scales",
                    experts.gate_proj_scales.as_ref().as_ref(),
                ),
                (
                    "gate_proj_biases",
                    experts.gate_proj_biases.as_ref().as_ref(),
                ),
                ("up_proj", experts.up_proj.as_ref().as_ref()),
                (
                    "up_proj_scale_inv",
                    experts.up_proj_scale_inv.as_ref().as_ref(),
                ),
                ("up_proj_scales", experts.up_proj_scales.as_ref().as_ref()),
                ("up_proj_biases", experts.up_proj_biases.as_ref().as_ref()),
            ] {
                if let Some(value) = value {
                    members.push(array_parameter_member(
                        format!("{prefix}.mlp.experts.{name}"),
                        value,
                        MemberSharding::Partitioned { axis: 1 },
                    )?);
                }
            }
            for (name, value) in [
                ("down_proj", experts.down_proj.as_ref().as_ref()),
                (
                    "down_proj_scale_inv",
                    experts.down_proj_scale_inv.as_ref().as_ref(),
                ),
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
    Ok(())
}

impl ArchitectureAdapter for DeepSeekV3LayerwiseAdapter {
    type Input<'a> = &'a Array;
    type Cache = Cache;
    type Layer = DecoderLayer;
    type ForwardContext = DeepSeekV3ForwardContext;

    fn model_type(&self) -> &str {
        &self.args.model_type
    }

    fn prompt_cache_model_identity(
        &self,
        topology: Option<crate::ParallelTopology>,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let layer_count = self.args.layer_schedule.len();
        Ok(PromptCacheModelIdentity {
            model_family: "deepseek_v3".into(),
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
            layer_layout: PromptCacheModelIdentity::compressed_layouts(
                layer_count,
                self.args.kv_lora_rank,
                self.args.qk_rope_head_dim,
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
        Cache::load_prompt_cache(
            &self.args.layer_schedule,
            directory,
            expected,
            identity,
            prefix_token_ids,
            options,
        )
        .map_err(Into::into)
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
                build_module_bindings_with_recipes(
                    &self.embedding,
                    "model.embed_tokens",
                    store,
                    BTreeMap::new(),
                )?,
            )?);
        }
        if select(NORM_UNIT) {
            units.push(StaticUnitBindings::new(
                NORM_UNIT,
                build_module_bindings_with_recipes(
                    &self.norm,
                    "model.norm",
                    store,
                    BTreeMap::new(),
                )?,
            )?);
        }
        if select(HEAD_UNIT) {
            units.push(StaticUnitBindings::new(
                HEAD_UNIT,
                build_module_bindings_with_recipes(
                    &self.lm_head,
                    "lm_head",
                    store,
                    BTreeMap::new(),
                )?,
            )?);
        }
        Ok(units)
    }

    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error> {
        if leases.len() != 3 {
            return Err(Error::UnsupportedArchitecture(format!(
                "DeepSeek-V3 adapter received {} static leases, expected 3",
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
        } else {
            populate_module_from_lease(&mut self.lm_head, &leases[2])?;
        }
        Ok(())
    }

    fn validate_cache(&self, cache: &mut Cache) -> Result<(), Error> {
        if cache.layers.is_empty() {
            *cache = self.new_cache();
        }
        if cache.layers.len() != self.args.layer_schedule.len() {
            return Err(Error::UnsupportedArchitecture(format!(
                "DeepSeek-V3 cache has {} layers, expected {}",
                cache.layers.len(),
                self.args.layer_schedule.len()
            )));
        }
        Ok(())
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        let hidden = self.embedding.forward(input, stream)?;
        let offset = cache.offset();
        let mask = if hidden.dim(1) > 1 && offset > 0 {
            Some(create_causal_mask(
                hidden.dim(1),
                Some(offset),
                None,
                None,
                stream,
            )?)
        } else {
            None
        };
        Ok(LayerwiseForwardState {
            hidden,
            context: DeepSeekV3ForwardContext { mask },
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
        let hidden = embedding.forward(input, execution)?;
        let offset = cache.offset();
        let mask = if hidden.dim(1) > 1 && offset > 0 {
            Some(create_causal_mask(
                hidden.dim(1),
                Some(offset),
                None,
                None,
                execution.stream(),
            )?)
        } else {
            None
        };
        Ok(LayerwiseForwardState {
            hidden,
            context: DeepSeekV3ForwardContext { mask },
        })
    }

    fn execution_graph(
        &self,
    ) -> Result<crate::runtime::execution::layerwise::ExecutionGroupDag, Error> {
        crate::runtime::execution::layerwise::ExecutionGroupDag::chain(["text_decoder"])
    }

    fn layer_count(&self, group: usize) -> Result<usize, Error> {
        if group == 0 {
            Ok(self.args.layer_schedule.len())
        } else {
            Err(Error::UnsupportedArchitecture(format!(
                "DeepSeek-V3 decoder has no execution group {group}"
            )))
        }
    }

    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        self.layer_count(group)?;
        Ok(DecoderLayer::new_layerwise(
            &self.args,
            index as i32,
            stream,
        )?)
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
        planner.register(crate::nn::parallel::vocab_lm_head_parameter_group(
            &self.lm_head,
            "lm_head",
            self.args.hidden_size,
            self.args.vocab_size as usize,
            false,
        )?)?;
        for index in 0..self.args.layer_schedule.len() {
            let layer = DecoderLayer::new_layerwise(&self.args, index as i32, stream)?;
            register_deepseek_layer_parallel_plan(planner, &layer, index)?;
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
        self.parallel_lm_head = Some(VocabParallelLmHead::unloaded(
            self.args.hidden_size,
            self.args.vocab_size as usize,
            self.args.weight_quantization_for("lm_head.weight"),
            context,
            stream,
        )?);
        Ok(())
    }

    fn new_parallel_layer(
        &self,
        group: usize,
        index: usize,
        layout: &crate::runtime::distributed::parallel::LocalModelLayout,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        self.layer_count(group)?;
        let prefix = format!("model.layers.{index}");
        let tensor = |suffix: &str| {
            layout
                .tensor(&format!("{prefix}.{suffix}.weight"))
                .or_else(|| layout.tensor(&format!("{prefix}.{suffix}.inner.weight")))
        };
        let attention = tensor("self_attn.q_proj")
            .or_else(|| tensor("self_attn.q_b_proj"))
            .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} MLA query")))?;
        let local_heads = i32::try_from(attention.local_shape()[0])
            .map_err(|_| Error::Parallel("DeepSeek local query width exceeds i32".into()))?
            / (self.args.qk_nope_head_dim + self.args.qk_rope_head_dim);
        let local_width = |suffix: &str, axis: usize, fallback: i32| -> Result<i32, Error> {
            tensor(suffix)
                .map(|value| {
                    value
                        .local_shape()
                        .get(axis)
                        .copied()
                        .ok_or_else(|| {
                            Error::Parallel(format!(
                                "DeepSeek TP layout for {prefix}.{suffix} has no axis {axis}"
                            ))
                        })
                        .and_then(|width| {
                            i32::try_from(width).map_err(|_| {
                                Error::Parallel(format!(
                                    "DeepSeek local width for {prefix}.{suffix} exceeds i32"
                                ))
                            })
                        })
                })
                .transpose()
                .map(|value| value.unwrap_or(fallback))
        };
        let dense_intermediate = local_width("mlp.gate_proj", 0, self.args.intermediate_size)?;
        let routed_intermediate = layout
            .tensor(&format!("{prefix}.mlp.experts.gate_proj"))
            .map(|value| {
                i32::try_from(value.local_shape()[1]).map_err(|_| {
                    Error::Parallel("DeepSeek local routed-expert width exceeds i32".into())
                })
            })
            .transpose()?
            .unwrap_or(self.args.moe_intermediate_size);
        let shared_intermediate = local_width(
            "mlp.shared_experts.gate_proj",
            0,
            self.args.moe_intermediate_size * self.args.n_shared_experts,
        )?;
        Ok(DecoderLayer::new_parallel_layerwise(
            &self.args,
            index as i32,
            local_heads,
            dense_intermediate,
            routed_intermediate,
            shared_intermediate,
            stream,
        )?)
    }

    fn layer_checkpoint_prefix(&self, _group: usize, index: usize) -> String {
        format!("model.layers.{index}")
    }

    fn layer_unit_name(&self, _group: usize, index: usize) -> String {
        format!("deepseek_v3.layer.{index:05}")
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
        let prefix = format!("model.layers.{index}");
        let bindings = build_module_bindings_with_recipes(
            layer,
            &prefix,
            store,
            self.recipes_for_layer(layer, index, store)?,
        )?;
        if self.sparse_expert_cache {
            Ok(bindings
                .into_iter()
                .filter(|binding| !binding.name().starts_with("mlp.experts."))
                .collect())
        } else {
            Ok(bindings)
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
        let prediction_start = self.args.num_hidden_layers;
        let prediction_end = prediction_start + self.args.num_nextn_predict_layers;
        store
            .keys()
            .into_iter()
            .filter(|key| {
                let cached_expert = self.sparse_expert_cache && key.contains(".mlp.experts.");
                let prediction_layer = key
                    .strip_prefix("model.layers.")
                    .and_then(|tail| tail.split_once('.'))
                    .and_then(|(layer, _)| layer.parse::<i32>().ok())
                    .is_some_and(|layer| layer >= prediction_start && layer < prediction_end);
                cached_expert || prediction_layer
            })
            .collect()
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
        if self.sparse_expert_cache
            && self.args.layer_policy(index) == Some(&LayerPolicy::SparseMoe)
        {
            let expert_cache = self.expert_cache.as_ref().ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "DeepSeek-V3 sparse expert cache was not initialized".into(),
                )
            })?;
            let pass = if hidden.dim(1) > 1 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            };
            let output = layer.forward_sparse_experts(
                hidden,
                context.mask.as_ref(),
                Some(&mut cache.layers[index]),
                stream,
                |flat, indices, weights, stream| {
                    expert_cache
                        .execute_routes_bounded(
                            ExpertRouteBatch::new(index, flat, indices, weights, pass),
                            stream,
                            |flat, acquired, weights, stream| {
                                if acquired.is_empty() {
                                    return Err(ExpertCacheError::EmptyRoutedBank {
                                        architecture: "DeepSeek-V3",
                                    });
                                }
                                let started = Instant::now();
                                let mut bank = resident::RoutedExperts::new_compact(
                                    &self.args,
                                    index as i32,
                                    acquired.identities().len() as i32,
                                    stream,
                                )?;
                                bank.gate_proj = Param::new(Some(
                                    acquired
                                        .compact_binding("gate_proj", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                ));
                                bank.gate_proj_scale_inv = Param::new(
                                    acquired
                                        .optional_compact_binding("gate_proj_scale_inv", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                bank.gate_proj_scales = Param::new(
                                    acquired
                                        .optional_compact_binding("gate_proj_scales", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                bank.gate_proj_biases = Param::new(
                                    acquired
                                        .optional_compact_binding("gate_proj_biases", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                bank.up_proj = Param::new(Some(
                                    acquired
                                        .compact_binding("up_proj", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                ));
                                bank.up_proj_scale_inv = Param::new(
                                    acquired
                                        .optional_compact_binding("up_proj_scale_inv", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                bank.up_proj_scales = Param::new(
                                    acquired
                                        .optional_compact_binding("up_proj_scales", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                bank.up_proj_biases = Param::new(
                                    acquired
                                        .optional_compact_binding("up_proj_biases", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                bank.down_proj = Param::new(Some(
                                    acquired
                                        .compact_binding("down_proj", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                ));
                                bank.down_proj_scale_inv = Param::new(
                                    acquired
                                        .optional_compact_binding("down_proj_scale_inv", stream)
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
                                Ok(bank.forward_local(
                                    flat,
                                    acquired.compact_routes(),
                                    weights,
                                    stream,
                                )?)
                            },
                        )
                        .map_err(|error| Exception::custom(error.to_string()))
                },
            )?;
            return Ok(output);
        }
        Ok(layer.forward_stage(
            hidden,
            context.mask.as_ref(),
            Some(&mut cache.layers[index]),
            stream,
        )?)
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
        self.layer_count(group)?;
        if self.sparse_expert_cache {
            let prefix = self.layer_checkpoint_prefix(group, index);
            observer.observe(&format!("{prefix}.input"), hidden)?;
            let output = <Self as ArchitectureAdapter>::forward_layer(
                self, group, index, layer, hidden, cache, context, stream,
            )?;
            observer.observe(&format!("{prefix}.output"), &output)?;
            return Ok(observer
                .intervene(&format!("{prefix}.output"), &output)?
                .unwrap_or(output));
        }
        Ok(layer.forward_stage_with_observer(
            hidden,
            context.mask.as_ref(),
            Some(&mut cache.layers[index]),
            stream,
            &self.layer_checkpoint_prefix(group, index),
            observer,
        )?)
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
        self.layer_count(group)?;
        Ok(layer.forward_tensor_parallel(
            hidden,
            context.mask.as_ref(),
            Some(&mut cache.layers[index]),
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
        cache.layers[index]
            .arrays()
            .map(|(latent, rotary)| vec![latent, rotary])
            .unwrap_or_default()
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
        execution: &crate::runtime::distributed::parallel::ParallelExecutionContext<'_>,
    ) -> Result<Array, Error> {
        let Some(head) = &mut self.parallel_lm_head else {
            return self.finish(hidden, cache, context, execution.stream());
        };
        let hidden = self.norm.forward(hidden, execution.stream())?;
        head.forward(&hidden, execution)?.all_gather(execution)
    }

    fn ignores_checkpoint_key(&self, key: &str) -> bool {
        (0..self.args.num_nextn_predict_layers).any(|index| {
            key.starts_with(&format!(
                "model.layers.{}.",
                self.args.num_hidden_layers + index
            ))
        })
    }
}

pub(crate) fn deepseek_expert_catalog(
    args: &ModelArgs,
    store: &dyn WeightStore,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let normalized = normalized_checkpoint_keys(store);
    let mut entries = Vec::new();
    for (layer, policy) in args.layer_schedule.iter().enumerate() {
        if *policy != LayerPolicy::SparseMoe {
            continue;
        }
        let prefix = format!("model.layers.{layer}.mlp.experts");
        for expert in 0..usize::try_from(args.n_routed_experts).map_err(|_| {
            Error::UnsupportedArchitecture("DeepSeek-V3 expert count is negative".into())
        })? {
            let identity = ExpertIdentity::new(layer, expert);
            let mut bindings = Vec::new();
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let packed = normalized.get(&format!("{prefix}.{projection}"));
                for (runtime_suffix, checkpoint_component, required) in [
                    ("", "weight", true),
                    ("_scale_inv", "weight_scale_inv", false),
                    ("_scales", "scales", false),
                    ("_biases", "biases", false),
                ] {
                    let binding_name = format!("{projection}{runtime_suffix}");
                    let recipe = if let Some(packed_key) = packed {
                        let runtime = format!("{prefix}.{projection}{runtime_suffix}");
                        match normalized.get(&runtime) {
                            Some(raw) => Some(DerivedWeightRecipe::source(
                                raw.clone(),
                                TensorSelection::Range {
                                    axis: 0,
                                    start: expert,
                                    end: expert + 1,
                                },
                            )),
                            None if required => Some(DerivedWeightRecipe::source(
                                packed_key.clone(),
                                TensorSelection::Range {
                                    axis: 0,
                                    start: expert,
                                    end: expert + 1,
                                },
                            )),
                            None => None,
                        }
                    } else {
                        let runtime =
                            format!("{prefix}.{expert}.{projection}.{checkpoint_component}");
                        match normalized.get(&runtime) {
                            Some(raw) => Some(DerivedWeightRecipe::Stack {
                                axis: 0,
                                inputs: vec![DerivedWeightRecipe::source(
                                    raw.clone(),
                                    TensorSelection::Full,
                                )],
                            }),
                            None if required => {
                                return Err(Error::UnsupportedArchitecture(format!(
                                    "DeepSeek-V3 checkpoint is missing expert tensor {runtime}"
                                )));
                            }
                            None => None,
                        }
                    };
                    if let Some(recipe) = recipe {
                        bindings.push(deepseek_recipe_binding(&binding_name, recipe, store)?);
                    }
                }
            }
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    Error::UnsupportedArchitecture(
                        "DeepSeek-V3 expert byte total overflowed".into(),
                    )
                })
            })?;
            let unit = OffloadUnit::new(identity.unit_id(), bindings)?;
            entries.push(ExpertCatalogEntry::new(identity, unit, bytes)?);
        }
    }
    Ok(entries)
}

fn deepseek_recipe_binding(
    name: &str,
    recipe: DerivedWeightRecipe,
    store: &dyn WeightStore,
) -> Result<WeightBinding, Error> {
    let bytes = recipe.infer(store)?.byte_len();
    Ok(WeightBinding::from_recipe(name, recipe, bytes)?)
}

/// DeepSeek token generation using bounded layer execution.
pub type Generate<'a, S = crate::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, DeepSeekV3LayerwiseModel, Cache, S>;

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use safemlx::{
        module::{ModuleParameters, Param},
        ops::{indexing::TryIndexOp, ones_dtype, zeros_dtype},
        Array, Device, DeviceType, Dtype, ExecutionContext, Stream,
    };

    use super::{
        load_deepseek_v3_layerwise_model, load_deepseek_v3_sparse_expert_cache_model,
        register_deepseek_layer_parallel_plan, DeepSeekV3LayerwiseAdapter,
        DeepSeekV3LayerwiseModel,
    };
    use crate::{
        architectures::deepseek_v3::model::{
            self as resident, FeedForward, LayerPolicy, Model, ModelArgs, ModelInput,
        },
        runtime::attention::LayerSchedule,
        runtime::checkpoint::binding::canonical_checkpoint_name,
        runtime::distributed::{
            parallel::{ParallelBuildContext, ShardingPolicy},
            topology::{DeviceAssignment, ParallelTopology},
        },
        runtime::execution::layerwise::{load_safetensors_layerwise_model, LayerwiseLoadOptions},
        runtime::residency::expert_cache::ExpertCacheLoadOptions,
        runtime::residency::policy::{OffloadConfig, ResidencyPolicy},
    };

    fn config(fp8: bool) -> serde_json::Value {
        let mut value = serde_json::json!({
            "architectures": ["DeepseekV3ForCausalLM"],
            "model_type": "deepseek_v3",
            "hidden_size": 8,
            "intermediate_size": 16,
            "moe_intermediate_size": 4,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "vocab_size": 32,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 128,
            "rope_theta": 10000,
            "q_lora_rank": 4,
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "v_head_dim": 2,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "n_group": 2,
            "topk_group": 1,
            "topk_method": "noaux_tc",
            "scoring_func": "sigmoid",
            "norm_topk_prob": true,
            "routed_scaling_factor": 1.5,
            "num_nextn_predict_layers": 1,
            "tie_word_embeddings": false,
            "attention_bias": false,
            "attention_dropout": 0.0,
            "hidden_act": "silu",
            "eos_token_id": 1
        });
        if fp8 {
            value.as_object_mut().unwrap().insert(
                "quantization_config".into(),
                serde_json::json!({
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128]
                }),
            );
        }
        value
    }

    fn args(fp8: bool) -> ModelArgs {
        resident::model_args_from_config_value(&config(fp8)).unwrap()
    }

    #[test]
    fn tensor_parallel_plan_balances_mla_dense_and_expert_domains() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut args = args(false);
        args.hidden_size = 12;
        args.intermediate_size = 17;
        args.moe_intermediate_size = 5;
        args.num_attention_heads = 3;
        args.layer_schedule =
            LayerSchedule::new(2, vec![LayerPolicy::DenseMlp, LayerPolicy::SparseMoe]).unwrap();
        args.validate().unwrap();
        for rank in 0..2 {
            let context = ParallelBuildContext::new(
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
            );
            let mut planner = context.planner();
            for layer in 0..2 {
                let block =
                    resident::DecoderLayer::new_layerwise(&args, layer as i32, execution.stream())
                        .unwrap();
                register_deepseek_layer_parallel_plan(&mut planner, &block, layer).unwrap();
            }
            let (_, layout) = planner.finish().unwrap();
            let heads = if rank == 0 { 2 } else { 1 };
            let dense = if rank == 0 { 9 } else { 8 };
            let expert = if rank == 0 { 3 } else { 2 };
            assert_eq!(
                layout
                    .tensor("model.layers.0.self_attn.q_b_proj.weight")
                    .unwrap()
                    .local_shape(),
                &[heads * 4, 4]
            );
            assert_eq!(
                layout
                    .tensor("model.layers.0.mlp.gate_proj.weight")
                    .unwrap()
                    .local_shape(),
                &[dense, 12]
            );
            assert_eq!(
                layout
                    .tensor("model.layers.1.mlp.experts.gate_proj")
                    .unwrap()
                    .local_shape(),
                &[4, expert, 12]
            );
            assert_eq!(
                layout
                    .tensor("model.layers.1.mlp.experts.down_proj")
                    .unwrap()
                    .local_shape(),
                &[4, 12, expert]
            );
            assert_eq!(
                layout
                    .tensor("model.layers.1.mlp.shared_experts.gate_proj.weight")
                    .unwrap()
                    .local_shape(),
                &[expert, 12]
            );
        }
    }

    fn initialize(model: &mut Model, stream: &Stream) {
        for layer in &mut model.model.layers {
            if let FeedForward::Moe(moe) = &mut layer.mlp {
                let experts = model.args.n_routed_experts;
                let hidden = model.args.hidden_size;
                let intermediate = model.args.moe_intermediate_size;
                let weight = |shape: &[i32]| {
                    if model.args.native_fp8_config().is_some() {
                        Array::full::<u8>(shape, Array::from_slice(&[0x38u8], &[]), stream).unwrap()
                    } else {
                        Array::full::<f32>(shape, Array::from_f32(0.01), stream).unwrap()
                    }
                };
                moe.experts.gate_proj = Param::new(Some(weight(&[experts, intermediate, hidden])));
                moe.experts.up_proj = Param::new(Some(weight(&[experts, intermediate, hidden])));
                moe.experts.down_proj = Param::new(Some(weight(&[experts, hidden, intermediate])));
                if model.args.native_fp8_config().is_some() {
                    moe.experts.gate_proj_scale_inv =
                        Param::new(Some(Array::ones::<f32>(&[experts, 1, 1], stream).unwrap()));
                    moe.experts.up_proj_scale_inv =
                        Param::new(Some(Array::ones::<f32>(&[experts, 1, 1], stream).unwrap()));
                    moe.experts.down_proj_scale_inv =
                        Param::new(Some(Array::ones::<f32>(&[experts, 1, 1], stream).unwrap()));
                }
            }
        }
        for (name, parameter) in model.parameters_mut().flatten() {
            let shape = parameter.shape().to_vec();
            let dtype = parameter.dtype();
            *parameter = if dtype == Dtype::Uint8 {
                Array::full::<u8>(&shape, Array::from_slice(&[0x38u8], &[]), stream).unwrap()
            } else if name.ends_with("layernorm.weight")
                || name.as_ref() == "model.norm.weight"
                || name.ends_with("weight_scale_inv")
                || name.ends_with("_scale_inv")
            {
                ones_dtype(&shape, dtype, stream).unwrap()
            } else if dtype == Dtype::Float32 {
                Array::full::<f32>(&shape, Array::from_f32(0.01), stream).unwrap()
            } else {
                zeros_dtype(&shape, dtype, stream).unwrap()
            };
        }
    }

    fn write_fixture(dir: &Path, model: &Model, fp8: bool, split_experts: bool, stream: &Stream) {
        let mut arrays = Vec::<(String, Array)>::new();
        for (name, value) in model.parameters().flatten() {
            let name = canonical_checkpoint_name(&name);
            let packed = ["gate_proj", "up_proj", "down_proj"]
                .into_iter()
                .find_map(|projection| {
                    [
                        ("", "weight"),
                        ("_scale_inv", "weight_scale_inv"),
                        ("_scales", "scales"),
                        ("_biases", "biases"),
                    ]
                    .into_iter()
                    .find_map(|(runtime_suffix, checkpoint_component)| {
                        name.ends_with(&format!(".mlp.experts.{projection}{runtime_suffix}"))
                            .then_some((projection, runtime_suffix, checkpoint_component))
                    })
                });
            if let Some((projection, runtime_suffix, checkpoint_component)) =
                packed.filter(|_| split_experts)
            {
                let suffix = format!(".experts.{projection}{runtime_suffix}");
                let prefix = name.strip_suffix(&suffix).unwrap();
                for expert in 0..model.args.n_routed_experts {
                    arrays.push((
                        format!("{prefix}.experts.{expert}.{projection}.{checkpoint_component}"),
                        value.try_index_device(expert, stream).unwrap(),
                    ));
                }
            } else {
                arrays.push((name, value.clone()));
            }
        }
        Array::save_safetensors(
            arrays.iter().map(|(name, value)| (name.as_str(), value)),
            None,
            dir.join("model.safetensors"),
        )
        .unwrap();
        fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&config(fp8)).unwrap(),
        )
        .unwrap();
    }

    fn assert_close(left: &Array, right: &Array, tolerance: f32) {
        let left = left.evaluated().unwrap();
        let right = right.evaluated().unwrap();
        assert_eq!(left.as_array().shape(), right.as_array().shape());
        for (left, right) in left.as_slice::<f32>().iter().zip(right.as_slice::<f32>()) {
            assert!((left - right).abs() <= tolerance, "{left} != {right}");
        }
    }

    fn parity(fp8: bool, depth: usize) {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut fixture = Model::new(args(fp8), gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, fp8, true, gpu.stream());

        let mut resident = resident::load_model(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let options = LayerwiseLoadOptions::new(OffloadConfig::new(None, None, depth).unwrap());
        let mut layerwise =
            load_deepseek_v3_layerwise_model(dir.path(), options, gpu.stream(), cpu.stream())
                .unwrap();
        let mut resident_cache = resident.new_cache();
        let mut layerwise_cache = resident::Cache { layers: Vec::new() };
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
            Array::from_slice(&[4u32], &[1, 1]),
            Array::from_slice(&[5u32], &[1, 1]),
        ] {
            let expected = resident
                .forward_logits(
                    ModelInput {
                        inputs: &tokens,
                        mask: None,
                        cache: Some(&mut resident_cache),
                    },
                    false,
                    gpu.stream(),
                )
                .unwrap();
            let actual = layerwise
                .forward(&tokens, &mut layerwise_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected, if fp8 { 2e-4 } else { 3e-5 });
            assert_eq!(resident_cache.offset(), layerwise_cache.offset());
            for (expected, actual) in resident_cache.layers.iter().zip(&layerwise_cache.layers) {
                assert_eq!(expected.offset(), actual.offset());
                let (expected_latent, expected_rotary) = expected.arrays().unwrap();
                let (actual_latent, actual_rotary) = actual.arrays().unwrap();
                assert_eq!(expected_latent.shape(), actual_latent.shape());
                assert_eq!(expected_rotary.shape(), actual_rotary.shape());
            }
            let report = layerwise.residency_report().unwrap();
            let layers = report
                .units()
                .iter()
                .filter(|unit| unit.id().as_str().starts_with("deepseek_v3.layer."))
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

    #[test]
    fn arbitrary_moe_dense_order_matches_resident_execution() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut custom_args = args(false);
        custom_args.layer_schedule =
            LayerSchedule::new(2, vec![LayerPolicy::SparseMoe, LayerPolicy::DenseMlp]).unwrap();
        custom_args.validate().unwrap();
        let mut resident = Model::new(custom_args.clone(), gpu.stream()).unwrap();
        initialize(&mut resident, gpu.stream());
        let directory = tempfile::tempdir().unwrap();
        write_fixture(directory.path(), &resident, false, true, gpu.stream());

        let options = LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap());
        let adapter = DeepSeekV3LayerwiseAdapter::new(custom_args, gpu.stream()).unwrap();
        let execution = load_safetensors_layerwise_model(
            directory.path(),
            adapter,
            options,
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut layerwise = DeepSeekV3LayerwiseModel { execution };
        let mut resident_cache = resident.new_cache();
        let mut layerwise_cache = resident::Cache { layers: Vec::new() };
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
        ] {
            let expected = resident
                .forward_logits(
                    ModelInput {
                        inputs: &tokens,
                        mask: None,
                        cache: Some(&mut resident_cache),
                    },
                    false,
                    gpu.stream(),
                )
                .unwrap();
            let actual = layerwise
                .forward(&tokens, &mut layerwise_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected, 3e-5);
        }
        assert_eq!(resident_cache.offset(), 3);
        assert_eq!(layerwise_cache.offset(), 3);
        assert!(resident_cache
            .layers
            .iter()
            .all(|cache| cache.offset() == 3));
        assert!(layerwise_cache
            .layers
            .iter()
            .all(|cache| cache.offset() == 3));
    }

    #[test]
    fn deepseek_v3_split_moe_layerwise_parity() {
        parity(false, 1);
        parity(false, 2);
    }

    #[test]
    fn deepseek_v3_native_fp8_split_moe_layerwise_parity() {
        parity(true, 1);
    }

    #[test]
    fn deepseek_v3_sparse_expert_cache_layout_parity_and_telemetry() {
        sparse_expert_cache_parity(false, true);
        sparse_expert_cache_parity(false, false);
        sparse_expert_cache_parity(true, true);
    }

    fn sparse_expert_cache_parity(fp8: bool, split_experts: bool) {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut fixture = Model::new(args(fp8), gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, fp8, split_experts, gpu.stream());

        let mut resident = if split_experts {
            resident::load_model(dir.path(), gpu.stream(), cpu.stream()).unwrap()
        } else {
            fixture
        };
        let expert_options = ExpertCacheLoadOptions::new(
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
            OffloadConfig::new(None, None, 1).unwrap(),
            1 << 20,
            1,
        )
        .unwrap();
        let mut cached = load_deepseek_v3_sparse_expert_cache_model(
            dir.path(),
            expert_options,
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut resident_cache = resident.new_cache();
        let mut cached_cache = resident::Cache { layers: Vec::new() };
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
            Array::from_slice(&[4u32], &[1, 1]),
        ] {
            let expected = resident
                .forward_logits(
                    ModelInput {
                        inputs: &tokens,
                        mask: None,
                        cache: Some(&mut resident_cache),
                    },
                    false,
                    gpu.stream(),
                )
                .unwrap();
            let actual = cached
                .forward(&tokens, &mut cached_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected, if fp8 { 2e-4 } else { 3e-5 });
            assert_eq!(cached_cache.offset(), resident_cache.offset());
        }
        let report = cached.expert_cache_report().unwrap().unwrap();
        assert_eq!(report.owned_experts, 4);
        assert!(report.prefill.requested_routes > 0);
        assert!(report.decode.requested_routes > 0);
        assert!(report.prefill.compact_banks > 1);
        assert!(report.decode.compact_banks > 0);
    }
}

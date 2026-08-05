//! Text-decoder bounded layer execution for Thinking Machines Lab Inkling.

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
    ops::{
        concatenate_axis, indexing::NewAxis, indexing::TryIndexOp, GgufCheckpoint,
        GgufMetadataValue,
    },
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};

use crate::{
    api::{
        common::{self, generation::CausalLm, moe::PackedSwiGluExperts},
        inkling::{
            self as resident, AudioModel, Cache, DecoderLayer, ModelArgs, VisionLayer, VisionModel,
        },
        input,
    },
    error::Error,
    nn::parallel::{
        vocab_embedding_parameter_group, vocab_lm_head_parameter_group, VocabParallelEmbedding,
        VocabParallelLmHead,
    },
    runtime::cache::residency::{
        CacheResidencyPolicy, CacheResidencyReport, PagedCacheOptions, PromptCacheDescriptor,
        PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions, PromptCacheTopology,
    },
    runtime::cache::KeyValueCache,
    runtime::checkpoint::binding::{
        build_module_bindings_with_recipes, populate_module_from_lease,
        populate_module_from_lease_excluding,
    },
    runtime::checkpoint::recipe::DerivedWeightRecipe,
    runtime::checkpoint::store::{GgufWeightStore, TensorSelection, WeightStore},
    runtime::distributed::parallel::exact_parallel_division,
    runtime::execution::layerwise::{
        load_layerwise_model, load_safetensors_layerwise_model,
        load_tensor_parallel_layerwise_model, open_safetensors_weight_store, ArchitectureAdapter,
        LayerExecutionLoadOptions, LayerwiseForwardState, LayerwiseModel, StaticUnitBindings,
        WeightResidency,
    },
    runtime::residency::expert_cache::{
        AcquiredExperts, ExpertCache, ExpertCacheError, ExpertCacheLoadOptions, ExpertCacheReport,
        ExpertCatalogEntry, ExpertIdentity, ExpertPass,
    },
    runtime::residency::manager::{OffloadUnit, ResidencyReport, ResidentUnitLease, WeightBinding},
};

const EMBEDDING_UNIT: &str = "inkling.static.embedding";
const EMBED_NORM_UNIT: &str = "inkling.static.embed_norm";
const NORM_UNIT: &str = "inkling.static.norm";
const HEAD_UNIT: &str = "inkling.static.output";
const AUDIO_UNIT: &str = "inkling.static.audio";
const VISION_NORM_UNIT: &str = "inkling.static.vision_norm";

/// Inkling multimodal model using bounded residency for hMLP and decoder blocks.
pub struct InklingLayerwiseModel {
    execution: LayerwiseModel<InklingLayerwiseAdapter>,
}

impl InklingLayerwiseModel {
    /// Returns the parsed Inkling configuration.
    pub fn args(&self) -> &ModelArgs {
        self.execution.adapter().args()
    }

    /// Creates global/sliding KV and short-convolution state for every layer.
    pub fn new_cache(&self) -> Cache {
        self.execution.adapter().new_cache()
    }

    /// Returns rank-local generalized parallel information when applicable.
    pub fn parallel_info(&self) -> Option<&crate::ParallelModelInfo> {
        self.execution.parallel_info()
    }

    /// Returns this rank's exact prompt-cache state layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        self.execution.prompt_cache_layer_layout()
    }

    /// Persists a compatible multimodal prefix cache.
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

    /// Restores a compatible multimodal prefix cache.
    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(Cache, PromptCacheManifest), Error> {
        self.execution
            .load_prompt_cache(directory, expected, prefix_token_ids, options, stream)
    }

    /// Creates global/sliding paged attention state while retaining the small
    /// short-convolution state on device.
    pub fn new_cache_with_options(&self, policy: CacheResidencyPolicy) -> Result<Cache, Error> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => Cache::new_paged(
                &self.args().text_config,
                options,
                self.execution.prompt_cache_rank_identity(),
            )
            .map_err(Into::into),
        }
    }

    /// Returns aggregate KV residency telemetry when paging is active.
    pub fn cache_residency_report(
        &self,
        cache: &Cache,
    ) -> Result<Option<CacheResidencyReport>, Error> {
        cache.residency_report().map_err(Into::into)
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
    pub fn checkpoint_store(&self) -> &(dyn WeightStore + Send + Sync) {
        self.execution.checkpoint_store()
    }

    /// Backward-compatible alias for [`Self::checkpoint_store`].
    pub fn weight_store(&self) -> &(dyn WeightStore + Send + Sync) {
        self.checkpoint_store()
    }

    /// Runs the text decoder while preserving KV and convolution state.
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution.forward(
            InklingExecutionInput {
                input: InklingInput::Decode(inputs),
                last_token_only: false,
            },
            cache,
            stream,
        )
    }

    /// Runs a typed multimodal prefill through rank-local hMLP units.
    pub fn prefill_tensor_parallel(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution.forward_tensor_parallel(
            InklingExecutionInput {
                input: InklingInput::Prefill(input),
                last_token_only: false,
            },
            cache,
            group,
            stream,
        )
    }

    /// Runs decode on a TP-loaded Inkling model.
    pub fn decode_tensor_parallel(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution.forward_tensor_parallel(
            InklingExecutionInput {
                input: InklingInput::Decode(tokens),
                last_token_only: false,
            },
            cache,
            group,
            stream,
        )
    }

    /// Runs streamed text layers while delegating routed experts to a caller.
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
            InklingExecutionInput {
                input: InklingInput::Decode(inputs),
                last_token_only: false,
            },
            cache,
            stream,
            |_adapter, _group, index, layer, hidden, cache, context, stream| match layer {
                InklingLayer::Vision(layer) => {
                    for job in &mut context.vision_jobs {
                        job.hidden = layer.forward(&job.hidden, stream)?;
                    }
                    Ok(context.vision_jobs[0].hidden.clone())
                }
                InklingLayer::Text(layer) => Ok(layer.forward_with_expert_executor(
                    hidden,
                    Some(&mut cache.layers[index]),
                    stream,
                    |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                )?),
            },
        )
    }

    /// Clears temporary vision and decoder blocks from the execution device.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        self.execution.clear_all_device_groups()
    }
}

impl CausalLm<Cache> for InklingLayerwiseModel {
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.execution
            .forward(
                InklingExecutionInput {
                    input: InklingInput::Prefill(input),
                    last_token_only: true,
                },
                cache,
                stream,
            )
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.execution
            .forward(
                InklingExecutionInput {
                    input: InklingInput::Decode(input_tokens),
                    last_token_only: true,
                },
                cache,
                stream,
            )
            .map_err(|error| Exception::custom(error.to_string()))
    }
}

/// Loads Inkling's multimodal model through the generalized execution engine.
pub fn load_inkling_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerExecutionLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<InklingLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let residency = options.weight_residency();
    crate::api::structural::validate_safetensors_load_path(
        crate::api::ModelKind::Inkling,
        model_dir,
        crate::api::ModelLoadOptions::default().with_weight_residency(residency),
    )?;
    let args = resident::get_model_args(model_dir)?;
    let adapter = InklingLayerwiseAdapter::new(args, stream)?;
    Ok(InklingLayerwiseModel {
        execution: load_safetensors_layerwise_model(
            model_dir,
            adapter,
            options,
            stream,
            weights_stream,
        )?,
    })
}

/// Loads Inkling with a rank-local hierarchical vision execution group.
pub fn load_inkling_tensor_parallel_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerExecutionLoadOptions>,
    build: crate::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<InklingLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let residency = options.weight_residency();
    if model_dir
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        let checkpoint = GgufCheckpoint::open(model_dir)?;
        let metadata = crate::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        let mmproj = resident::open_sibling_mmproj(model_dir)?;
        return load_inkling_gguf_tensor_parallel_model(
            &checkpoint,
            &metadata,
            mmproj.as_ref(),
            options,
            build,
            stream,
            weights_stream,
        )
        .map(|(model, _)| model);
    }
    crate::api::structural::validate_safetensors_load_path(
        crate::api::ModelKind::Inkling,
        model_dir,
        crate::api::ModelLoadOptions::default().with_weight_residency(residency),
    )?;
    let args = resident::get_model_args(model_dir)?;
    let adapter = InklingLayerwiseAdapter::new(args, stream)?;
    Ok(InklingLayerwiseModel {
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

pub(crate) fn load_inkling_gguf_tensor_parallel_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    mmproj: Option<&resident::InklingMmprojGguf>,
    options: LayerExecutionLoadOptions,
    build: crate::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(InklingLayerwiseModel, Vec<u32>), Error> {
    let residency = options.weight_residency();
    crate::api::structural::validate_gguf(
        crate::api::GgufArchitecture::Inkling,
        checkpoint,
        metadata,
        crate::api::ModelLoadOptions::default().with_weight_residency(residency),
    )
    .into_loader_result()?;
    let prepared = resident::prepare_gguf_checkpoint_with_mmproj(checkpoint, metadata, mmproj)?;
    let store = inkling_gguf_store(checkpoint, mmproj, options.max_mapped_shards())?;
    let execution = load_tensor_parallel_layerwise_model(
        store,
        InklingLayerwiseAdapter::new(prepared.args, stream)?,
        options,
        build,
        stream,
        weights_stream,
    )?;
    Ok((InklingLayerwiseModel { execution }, prepared.eos_token_ids))
}

pub(crate) fn load_inkling_gguf_layerwise_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    mmproj: Option<&resident::InklingMmprojGguf>,
    residency: WeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(InklingLayerwiseModel, Vec<u32>), Error> {
    crate::api::structural::validate_gguf(
        crate::api::GgufArchitecture::Inkling,
        checkpoint,
        metadata,
        crate::api::ModelLoadOptions::default().with_weight_residency(residency),
    )
    .into_loader_result()?;
    let prepared = resident::prepare_gguf_checkpoint_with_mmproj(checkpoint, metadata, mmproj)?;
    let store = inkling_gguf_store(checkpoint, mmproj, residency.max_mapped_shards())?;
    let args = prepared.args;
    let execution = match residency {
        WeightResidency::LayerwiseHost(options) => load_layerwise_model(
            store,
            InklingLayerwiseAdapter::new(args, stream)?,
            options,
            stream,
            weights_stream,
        )?,
        WeightResidency::DenseDiskStream(options) => load_layerwise_model(
            store,
            InklingLayerwiseAdapter::new(args, stream)?,
            options,
            stream,
            weights_stream,
        )?,
        WeightResidency::SparseExpertCache(options) => {
            return Ok((
                load_inkling_gguf_sparse_with_store(
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
                load_inkling_gguf_sparse_with_store(
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
        WeightResidency::FullyResident => {
            return Err(Error::UnsupportedArchitecture(
                "the bounded Inkling GGUF loader does not accept fully resident policy".into(),
            ));
        }
    };
    Ok((InklingLayerwiseModel { execution }, prepared.eos_token_ids))
}

pub(crate) fn inkling_gguf_store(
    checkpoint: &GgufCheckpoint,
    mmproj: Option<&resident::InklingMmprojGguf>,
    max_mapped_shards: usize,
) -> Result<Arc<dyn WeightStore + Send + Sync>, Error> {
    let mut builder = GgufWeightStore::builder()
        .max_cached_readers(max_mapped_shards)?
        .add_checkpoint(checkpoint.clone(), resident::translate_gguf_weight_name)?;
    if let Some(mmproj) = mmproj {
        builder = builder.add_checkpoint(
            mmproj.checkpoint.clone(),
            resident::translate_mmproj_weight_name,
        )?;
    }
    Ok(Arc::new(builder.build()?))
}

fn load_inkling_gguf_sparse_with_store(
    store: Arc<dyn WeightStore + Send + Sync>,
    args: ModelArgs,
    options: ExpertCacheLoadOptions,
    non_expert: impl Into<LayerExecutionLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<InklingLayerwiseModel, Error> {
    let mut adapter = InklingLayerwiseAdapter::new(args.clone(), stream)?;
    adapter.sparse_expert_cache = true;
    let mut execution = load_layerwise_model(store, adapter, non_expert, stream, weights_stream)?;
    let checkpoint_store = execution.weight_store_arc();
    let entries = inkling_expert_catalog(&args, checkpoint_store.as_ref())?;
    execution.adapter_mut().expert_cache = Some(ExpertCache::new_shared(
        checkpoint_store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(InklingLayerwiseModel { execution })
}

/// Loads Inkling with expert-granular sparse caching for routed text experts.
pub fn load_inkling_sparse_expert_cache_model(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<InklingLayerwiseModel, Error> {
    load_inkling_sparse_expert_cache_model_with_non_expert(
        model_dir,
        options,
        options.non_expert,
        stream,
        weights_stream,
    )
}

/// Loads Inkling with expert caching and disk-streamed non-expert units.
pub fn load_inkling_sparse_expert_cache_model_with_dense_layers(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    non_expert: crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<InklingLayerwiseModel, Error> {
    load_inkling_sparse_expert_cache_model_with_non_expert(
        model_dir,
        options,
        non_expert,
        stream,
        weights_stream,
    )
}

fn load_inkling_sparse_expert_cache_model_with_non_expert(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    non_expert: impl Into<LayerExecutionLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<InklingLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    crate::api::structural::validate_safetensors_load_path(
        crate::api::ModelKind::Inkling,
        model_dir,
        crate::api::ModelLoadOptions::default()
            .with_weight_residency(WeightResidency::SparseExpertCache(options)),
    )?;
    let args = resident::get_model_args(model_dir)?;
    if args.text_config.n_routed_experts <= 0
        || !args
            .text_config
            .layer_schedule
            .iter()
            .any(|policy| policy.feed_forward == resident::FeedForwardPolicy::SparseMoe)
    {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires an Inkling checkpoint with routed MoE layers".into(),
        ));
    }
    let mut adapter = InklingLayerwiseAdapter::new(args.clone(), stream)?;
    adapter.sparse_expert_cache = true;
    let mut execution =
        load_safetensors_layerwise_model(model_dir, adapter, non_expert, stream, weights_stream)?;
    let store = execution.weight_store_arc();
    let entries = inkling_expert_catalog(&args, store.as_ref())?;
    execution.adapter_mut().expert_cache = Some(ExpertCache::new_shared(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(InklingLayerwiseModel { execution })
}

/// Builds the streamed nonexpert Inkling execution base used by distributed EP.
pub(crate) fn load_inkling_sparse_ep_base_with_store(
    store: Arc<dyn WeightStore + Send + Sync>,
    args: ModelArgs,
    non_expert: impl Into<LayerExecutionLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<InklingLayerwiseModel, Error> {
    let mut adapter = InklingLayerwiseAdapter::new(args, stream)?;
    adapter.sparse_expert_cache = true;
    let execution = load_layerwise_model(store, adapter, non_expert, stream, weights_stream)?;
    Ok(InklingLayerwiseModel { execution })
}

/// Adapter for Inkling local/global attention and dense/MoE text blocks.
struct InklingLayerwiseAdapter {
    args: ModelArgs,
    embedding: MaybeQuantized<nn::Embedding>,
    parallel_embedding: Option<VocabParallelEmbedding>,
    embed_norm: nn::RmsNorm,
    norm: nn::RmsNorm,
    lm_head: MaybeQuantized<nn::Linear>,
    parallel_lm_head: Option<VocabParallelLmHead>,
    audio: Option<AudioModel>,
    vision_norm: Option<nn::RmsNorm>,
    vision_depth: usize,
    sparse_expert_cache: bool,
    expert_cache: Option<ExpertCache>,
}

impl InklingLayerwiseAdapter {
    fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let text = &args.text_config;
        let audio = args
            .audio_config
            .as_ref()
            .map(|config| AudioModel::new(config, text.weight_dtype(), stream))
            .transpose()?;
        let vision = args
            .vision_config
            .as_ref()
            .map(|config| VisionModel::new(config, text.weight_dtype(), stream))
            .transpose()?;
        let (vision_norm, vision_depth) = match vision {
            Some(vision) => (Some(vision.final_norm), vision.layers.len()),
            None => (None, 0),
        };
        Ok(Self {
            embedding: common::linear::unloaded_maybe_quantized_embedding_with_dtype(
                text.vocab_size,
                text.hidden_size,
                text.weight_quantization_for("model.embed_tokens.weight"),
                text.weight_dtype(),
                stream,
            )?,
            parallel_embedding: None,
            embed_norm: nn::RmsNorm::unloaded(
                text.hidden_size,
                text.rms_norm_eps,
                text.weight_dtype(),
                stream,
            )?,
            norm: nn::RmsNorm::unloaded(
                text.hidden_size,
                text.rms_norm_eps,
                text.weight_dtype(),
                stream,
            )?,
            lm_head: common::linear::unloaded_maybe_quantized_linear_with_dtype(
                text.hidden_size,
                text.vocab_size,
                false,
                text.weight_quantization_for("lm_head.weight"),
                text.weight_dtype(),
                stream,
            )?,
            parallel_lm_head: None,
            audio,
            vision_norm,
            vision_depth,
            sparse_expert_cache: false,
            expert_cache: None,
            args,
        })
    }

    /// Returns the parsed Inkling configuration.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    fn new_cache(&self) -> Cache {
        Cache::new(&self.args.text_config)
    }

    fn forward_cached_expert_bank(
        &self,
        layer: usize,
        flat: &Array,
        acquired: &AcquiredExperts,
        weights: &Array,
        stream: &Stream,
    ) -> Result<Array, ExpertCacheError> {
        let expert_cache =
            self.expert_cache
                .as_ref()
                .ok_or(ExpertCacheError::CacheUnavailable {
                    architecture: "Inkling",
                })?;
        let started = Instant::now();
        let text = &self.args.text_config;
        let prefix = format!("model.layers.{layer}.moe.experts");
        let gate_format = text.weight_quantization_for(&format!("{prefix}.gate_up_proj"));
        let down_format = text.weight_quantization_for(&format!("{prefix}.down_proj"));
        let mut bank = PackedSwiGluExperts::new_with_dtype(
            acquired.identities().len() as i32,
            text.hidden_size,
            text.moe_intermediate_size(),
            gate_format,
            down_format,
            text.weight_dtype(),
            stream,
        )?;
        bank.gate_up_proj = Param::new(acquired.compact_binding("gate_up_proj", stream)?);
        bank.down_proj = Param::new(acquired.compact_binding("down_proj", stream)?);
        if gate_format.is_some_and(|format| format.gguf_iquant().is_none()) {
            bank.gate_up_proj_scales = Param::new(Some(
                acquired.compact_binding("gate_up_proj_scales", stream)?,
            ));
        }
        if gate_format.is_some_and(|format| format.has_biases()) {
            bank.gate_up_proj_biases = Param::new(Some(
                acquired.compact_binding("gate_up_proj_biases", stream)?,
            ));
        }
        if down_format.is_some_and(|format| format.gguf_iquant().is_none()) {
            bank.down_proj_scales =
                Param::new(Some(acquired.compact_binding("down_proj_scales", stream)?));
        }
        if down_format.is_some_and(|format| format.has_biases()) {
            bank.down_proj_biases =
                Param::new(Some(acquired.compact_binding("down_proj_biases", stream)?));
        }
        expert_cache.record_compact_bank(
            acquired.pass(),
            acquired.scratch_bytes(),
            started.elapsed(),
        )?;
        Ok(bank.forward(flat, acquired.compact_routes(), weights, stream)?)
    }

    fn recipes_for_module(
        &self,
        module: &impl ModuleParameters,
        prefix: &str,
        store: &dyn WeightStore,
    ) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
        let normalized = normalized_checkpoint_keys(store);
        let direct = store.keys();
        let mut recipes = BTreeMap::new();
        let parameters = module.parameters().flatten();
        for (local_name, parameter) in &parameters {
            let destination = format!("{prefix}.{local_name}");
            if let Some(inner) = destination.strip_suffix(".inner.weight") {
                let checkpoint_name = format!("{inner}.weight");
                if direct.contains(&checkpoint_name) {
                    recipes.insert(
                        local_name.to_string(),
                        DerivedWeightRecipe::source(checkpoint_name, TensorSelection::Full),
                    );
                    continue;
                }
            }
            if direct.contains(&destination) {
                if destination.contains("_sconv.weight")
                    && store.metadata(&destination)?.shape.len() == 2
                {
                    recipes.insert(
                        local_name.to_string(),
                        DerivedWeightRecipe::Reshape {
                            input: Box::new(DerivedWeightRecipe::source(
                                destination,
                                TensorSelection::Full,
                            )),
                            shape: parameter
                                .shape()
                                .iter()
                                .map(|value| *value as usize)
                                .collect(),
                        },
                    );
                }
                continue;
            }
            if destination.ends_with(".dense_global_scale")
                || destination.ends_with(".moe.router.global_scale")
            {
                let layer_prefix = destination
                    .split_once(".dense_global_scale")
                    .map(|(prefix, _)| prefix)
                    .or_else(|| {
                        destination
                            .split_once(".moe.router.global_scale")
                            .map(|(prefix, _)| prefix)
                    })
                    .expect("global-scale suffix matched");
                let raw = format!("{layer_prefix}.global_scale");
                if direct.contains(&raw) {
                    recipes.insert(
                        local_name.to_string(),
                        DerivedWeightRecipe::source(raw, TensorSelection::Full),
                    );
                    continue;
                }
            }
            if let Some(recipe) = inkling_w13_recipe(&destination, &normalized, store)? {
                recipes.insert(local_name.to_string(), recipe);
                continue;
            }
            let raw = normalized.get(&destination).ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "Inkling checkpoint is missing runtime parameter {destination}"
                ))
            })?;
            let source = DerivedWeightRecipe::source(raw.clone(), TensorSelection::Full);
            recipes.insert(
                local_name.to_string(),
                if raw.contains("_sconv.weight") && store.metadata(raw)?.shape.len() == 2 {
                    DerivedWeightRecipe::Reshape {
                        input: Box::new(source),
                        shape: parameter
                            .shape()
                            .iter()
                            .map(|value| *value as usize)
                            .collect(),
                    }
                } else if raw.ends_with("_sconv.weight") {
                    DerivedWeightRecipe::Cast {
                        input: Box::new(source),
                        dtype: Dtype::Float32,
                    }
                } else {
                    source
                },
            );
        }
        Ok(recipes)
    }
}

fn normalized_checkpoint_keys(store: &dyn WeightStore) -> BTreeMap<String, String> {
    store
        .keys()
        .into_iter()
        .filter_map(|raw| normalize_checkpoint_key(&raw).map(|runtime| (runtime, raw)))
        .collect()
}

fn normalize_checkpoint_key(raw: &str) -> Option<String> {
    if raw.starts_with("model.mtp.") {
        return None;
    }
    if let Some(suffix) = raw.strip_prefix("model.audio.") {
        return Some(format!("audio.{suffix}"));
    }
    if let Some(suffix) = raw.strip_prefix("model.visual.") {
        let mut suffix = suffix.to_string();
        for layer in 0..4 {
            suffix = suffix
                .replace(
                    &format!("layers.linear_{layer}.weight"),
                    &format!("layers.{layer}.projection.weight"),
                )
                .replace(
                    &format!("layers.norm_{layer}.weight"),
                    &format!("layers.{layer}.layer_norm.weight"),
                );
        }
        return Some(format!("visual.{suffix}"));
    }
    if !raw.starts_with("model.llm.") {
        return Some(raw.to_string());
    }
    let mut key = raw.replacen("model.llm.", "model.", 1);
    key = key
        .replace("model.embed.weight", "model.embed_tokens.weight")
        .replace("model.unembed.weight", "lm_head.weight")
        .replace(".attn_norm.weight", ".input_layernorm.weight")
        .replace(".mlp_norm.weight", ".post_attention_layernorm.weight")
        .replace(".attn.wq_du.weight", ".self_attn.q_proj.weight")
        .replace(".attn.wk_dv.weight", ".self_attn.k_proj.weight")
        .replace(".attn.wv_dv.weight", ".self_attn.v_proj.weight")
        .replace(".attn.wr_du.weight", ".self_attn.r_proj.weight")
        .replace(".attn.wo_ud.weight", ".self_attn.o_proj.weight")
        .replace(".attn.q_norm.weight", ".self_attn.q_norm.weight")
        .replace(".attn.k_norm.weight", ".self_attn.k_norm.weight")
        .replace(".attn.rel_logits_proj.proj", ".self_attn.rel_proj")
        .replace(".attn.k_sconv.weight", ".self_attn.k_sconv.weight")
        .replace(".attn.v_sconv.weight", ".self_attn.v_sconv.weight")
        .replace(".mlp.w2_md.weight", ".dense.down_proj.weight")
        .replace(".mlp.global_scale", ".dense_global_scale")
        .replace(".mlp.gate.weight", ".moe.router.weight")
        .replace(".mlp.gate.bias", ".moe.router.bias")
        .replace(".mlp.gate.global_scale", ".moe.router.global_scale")
        .replace(".mlp.experts.w2_weight", ".moe.experts.down_proj")
        .replace(
            ".mlp.shared_experts.shared_w2_weight",
            ".moe.shared_experts.down_proj",
        );
    Some(key)
}

/// Input mode for typed prefill and cached text decode.
pub enum InklingInput<'a> {
    /// Ordered multimodal prompt parts.
    Prefill(input::ModelInput<'a>),
    /// Text tokens for a cached decode step.
    Decode(&'a Array),
}

struct InklingExecutionInput<'a> {
    input: InklingInput<'a>,
    last_token_only: bool,
}

enum PreparedPart {
    Ready { tokens: Array, embeddings: Array },
    Vision { tokens: Array, job: usize },
}

struct VisionJob {
    hidden: Array,
}

/// Transient media and ordered prompt assembly state.
struct InklingForwardContext {
    parts: Vec<PreparedPart>,
    vision_jobs: Vec<VisionJob>,
    needs_assembly: bool,
    last_token_only: bool,
}

/// One leased Inkling hMLP or decoder unit.
enum InklingLayer {
    /// One hMLP projection/fold layer.
    Vision(VisionLayer),
    /// One text decoder block.
    Text(Box<DecoderLayer>),
}

impl ModuleParameters for InklingLayer {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Vision(layer) => layer.num_parameters(),
            Self::Text(layer) => layer.num_parameters(),
        }
    }

    fn parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            Self::Vision(layer) => layer.parameters(),
            Self::Text(layer) => layer.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> safemlx::module::ModuleParamMut<'_> {
        match self {
            Self::Vision(layer) => layer.parameters_mut(),
            Self::Text(layer) => layer.parameters_mut(),
        }
    }

    fn trainable_parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            Self::Vision(layer) => layer.trainable_parameters(),
            Self::Text(layer) => layer.trainable_parameters(),
        }
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Vision(layer) => layer.freeze_parameters(recursive),
            Self::Text(layer) => layer.freeze_parameters(recursive),
        }
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Vision(layer) => layer.unfreeze_parameters(recursive),
            Self::Text(layer) => layer.unfreeze_parameters(recursive),
        }
    }

    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Vision(layer) => layer.all_frozen(),
            Self::Text(layer) => layer.all_frozen(),
        }
    }

    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Vision(layer) => layer.any_frozen(),
            Self::Text(layer) => layer.any_frozen(),
        }
    }
}

fn inkling_w13_recipe(
    destination: &str,
    normalized: &BTreeMap<String, String>,
    store: &dyn WeightStore,
) -> Result<Option<DerivedWeightRecipe>, Error> {
    for bank in ["moe.experts", "moe.shared_experts"] {
        if let Some(prefix) = destination.strip_suffix(&format!(".{bank}.gate_up_proj")) {
            let gate = format!("{prefix}.{bank}.gate_proj");
            let up = format!("{prefix}.{bank}.up_proj");
            if let (Some(gate), Some(up)) = (normalized.get(&gate), normalized.get(&up)) {
                return Ok(Some(DerivedWeightRecipe::Concatenate {
                    axis: 1,
                    inputs: vec![
                        DerivedWeightRecipe::source(gate.clone(), TensorSelection::Full),
                        DerivedWeightRecipe::source(up.clone(), TensorSelection::Full),
                    ],
                }));
            }
        }
        for suffix in ["_scales", "_biases"] {
            if let Some(prefix) = destination.strip_suffix(&format!(".{bank}.gate_up_proj{suffix}"))
            {
                let gate = format!("{prefix}.{bank}.gate_proj{suffix}");
                let up = format!("{prefix}.{bank}.up_proj{suffix}");
                if let (Some(gate), Some(up)) = (normalized.get(&gate), normalized.get(&up)) {
                    return Ok(Some(DerivedWeightRecipe::Concatenate {
                        axis: 1,
                        inputs: vec![
                            DerivedWeightRecipe::source(gate.clone(), TensorSelection::Full),
                            DerivedWeightRecipe::source(up.clone(), TensorSelection::Full),
                        ],
                    }));
                }
            }
        }
    }
    let (source_runtime, axis, parity, concatenate) =
        if let Some(prefix) = destination.strip_suffix(".dense.gate_proj.weight") {
            (format!("{prefix}.mlp.w13_dn.weight"), 0, 0, false)
        } else if let Some(prefix) = destination.strip_suffix(".dense.up_proj.weight") {
            (format!("{prefix}.mlp.w13_dn.weight"), 0, 1, false)
        } else if let Some(prefix) = destination.strip_suffix(".moe.experts.gate_up_proj") {
            (format!("{prefix}.mlp.experts.w13_weight"), 1, 0, true)
        } else if let Some(prefix) = destination.strip_suffix(".moe.shared_experts.gate_up_proj") {
            (
                format!("{prefix}.mlp.shared_experts.shared_w13_weight"),
                1,
                0,
                true,
            )
        } else {
            return Ok(None);
        };
    let Some(raw) = normalized.get(&source_runtime) else {
        return Ok(None);
    };
    let metadata = store.metadata(raw)?;
    let rows = metadata
        .shape
        .get(axis)
        .copied()
        .ok_or_else(|| Error::UnsupportedArchitecture("Inkling w13 rank is invalid".into()))?;
    if rows % 2 != 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "Inkling w13 tensor {raw} has odd interleaved width {rows}"
        )));
    }
    let selected = |parity: usize| {
        DerivedWeightRecipe::source(
            raw.clone(),
            TensorSelection::Indices {
                axis,
                indices: (parity..rows).step_by(2).collect(),
            },
        )
    };
    Ok(Some(if concatenate {
        DerivedWeightRecipe::Concatenate {
            axis,
            inputs: vec![selected(0), selected(1)],
        }
    } else {
        selected(parity)
    }))
}

impl ArchitectureAdapter for InklingLayerwiseAdapter {
    type Input<'a> = InklingExecutionInput<'a>;
    type Cache = Cache;
    type Layer = InklingLayer;
    type ForwardContext = InklingForwardContext;

    fn model_type(&self) -> &str {
        &self.args.model_type
    }

    fn prompt_cache_model_identity(
        &self,
        topology: Option<crate::ParallelTopology>,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let mut local = self.args.clone();
        if let Some(topology) = topology {
            let text = &mut local.text_config;
            text.num_attention_heads = exact_parallel_division(
                "Inkling prompt-cache attention heads",
                text.num_attention_heads,
                topology.tensor_parallel_size,
            )?;
            text.num_key_value_heads = exact_parallel_division(
                "Inkling prompt-cache KV heads",
                text.num_key_value_heads,
                topology.tensor_parallel_size,
            )?;
            text.swa_num_attention_heads = text
                .swa_num_attention_heads
                .map(|value| {
                    exact_parallel_division(
                        "Inkling prompt-cache sliding attention heads",
                        value,
                        topology.tensor_parallel_size,
                    )
                })
                .transpose()?;
            text.swa_num_key_value_heads = text
                .swa_num_key_value_heads
                .map(|value| {
                    exact_parallel_division(
                        "Inkling prompt-cache sliding KV heads",
                        value,
                        topology.tensor_parallel_size,
                    )
                })
                .transpose()?;
        }
        let layer_count = self.args.text_config.num_hidden_layers as usize;
        Ok(PromptCacheModelIdentity {
            model_family: "inkling".into(),
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
            layer_layout: resident::prompt_cache_layer_layout(&local)?,
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
        resident::Model::save_prompt_cache(
            cache,
            destination,
            descriptor,
            prefix_token_ids,
            options,
            stream,
        )
        .map_err(Into::into)
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
        resident::Model::load_prompt_cache_with_identity(
            &self.args,
            directory,
            expected,
            prefix_token_ids,
            identity,
            stream,
        )
        .map_err(Into::into)
    }

    fn static_units(&self, store: &dyn WeightStore) -> Result<Vec<StaticUnitBindings>, Error> {
        let mut units = vec![
            StaticUnitBindings::new(
                EMBEDDING_UNIT,
                build_module_bindings_with_recipes(
                    &self.embedding,
                    "model.embed_tokens",
                    store,
                    self.recipes_for_module(&self.embedding, "model.embed_tokens", store)?,
                )?,
            )?,
            StaticUnitBindings::new(
                EMBED_NORM_UNIT,
                build_module_bindings_with_recipes(
                    &self.embed_norm,
                    "model.embed_norm",
                    store,
                    self.recipes_for_module(&self.embed_norm, "model.embed_norm", store)?,
                )?,
            )?,
            StaticUnitBindings::new(
                NORM_UNIT,
                build_module_bindings_with_recipes(
                    &self.norm,
                    "model.norm",
                    store,
                    self.recipes_for_module(&self.norm, "model.norm", store)?,
                )?,
            )?,
            StaticUnitBindings::new(
                HEAD_UNIT,
                build_module_bindings_with_recipes(
                    &self.lm_head,
                    "lm_head",
                    store,
                    self.recipes_for_module(&self.lm_head, "lm_head", store)?,
                )?,
            )?,
        ];
        if let Some(audio) = &self.audio {
            units.push(StaticUnitBindings::new(
                AUDIO_UNIT,
                build_module_bindings_with_recipes(
                    audio,
                    "audio",
                    store,
                    self.recipes_for_module(audio, "audio", store)?,
                )?,
            )?);
        }
        if let Some(norm) = &self.vision_norm {
            units.push(StaticUnitBindings::new(
                VISION_NORM_UNIT,
                build_module_bindings_with_recipes(
                    norm,
                    "visual.final_norm",
                    store,
                    self.recipes_for_module(norm, "visual.final_norm", store)?,
                )?,
            )?);
        }
        Ok(units)
    }

    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error> {
        let expected =
            4 + usize::from(self.audio.is_some()) + usize::from(self.vision_norm.is_some());
        if leases.len() != expected {
            return Err(Error::UnsupportedArchitecture(format!(
                "Inkling adapter received {} static leases, expected {expected}",
                leases.len()
            )));
        }
        if let Some(embedding) = &mut self.parallel_embedding {
            populate_module_from_lease(embedding.inner_mut(), &leases[0])?;
        } else {
            populate_module_from_lease(&mut self.embedding, &leases[0])?;
        }
        populate_module_from_lease(&mut self.embed_norm, &leases[1])?;
        populate_module_from_lease(&mut self.norm, &leases[2])?;
        if let Some(head) = &mut self.parallel_lm_head {
            populate_module_from_lease(head.inner_mut(), &leases[3])?;
        } else {
            populate_module_from_lease(&mut self.lm_head, &leases[3])?;
        }
        let mut index = 4;
        if let Some(audio) = &mut self.audio {
            populate_module_from_lease(audio, &leases[index])?;
            index += 1;
        }
        if let Some(norm) = &mut self.vision_norm {
            populate_module_from_lease(norm, &leases[index])?;
        }
        Ok(())
    }

    fn validate_cache(&self, cache: &mut Cache) -> Result<(), Error> {
        if cache.layers.is_empty() {
            *cache = self.new_cache();
        }
        cache.validate(&self.args.text_config.layer_schedule)
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        _cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        let InklingExecutionInput {
            input,
            last_token_only,
        } = input;
        if let InklingInput::Decode(tokens) = input {
            let hidden = self
                .embed_norm
                .forward(&self.embedding.forward(tokens, stream)?, stream)?;
            return Ok(LayerwiseForwardState {
                hidden,
                context: InklingForwardContext {
                    parts: Vec::new(),
                    vision_jobs: Vec::new(),
                    needs_assembly: false,
                    last_token_only,
                },
            });
        }
        let InklingInput::Prefill(typed) = input else {
            unreachable!()
        };
        input::validate(typed)?;
        let mut parts = Vec::with_capacity(typed.parts.len());
        let mut vision_jobs = Vec::new();
        for part in typed.parts {
            match (part.modality, part.payload) {
                (input::Modality::Text, input::InputPayload::TokenIds(tokens)) => {
                    let embeddings = self
                        .embed_norm
                        .forward(&self.embedding.forward(tokens, stream)?, stream)?;
                    parts.push(PreparedPart::Ready {
                        tokens: tokens.clone(),
                        embeddings,
                    });
                }
                (input::Modality::Image, input::InputPayload::Tensor(pixels)) => {
                    if self.vision_norm.is_none() {
                        return Err(Error::UnsupportedArchitecture(
                            "Inkling image input requires vision_config and vision weights".into(),
                        ));
                    }
                    let job = vision_jobs.len();
                    vision_jobs.push(VisionJob {
                        hidden: pixels.clone(),
                    });
                    let count = pixels.dim(0) as usize;
                    parts.push(PreparedPart::Vision {
                        tokens: input::token_ids_array(
                            &vec![self.args.image_token_id; count],
                            stream,
                        )?,
                        job,
                    });
                }
                (input::Modality::Audio, input::InputPayload::Tensor(ids)) => {
                    let embeddings = self
                        .audio
                        .as_mut()
                        .ok_or_else(|| {
                            Error::UnsupportedArchitecture(
                                "Inkling audio input requires audio_config and audio weights"
                                    .into(),
                            )
                        })?
                        .forward(ids, part.metadata.audio_mask, stream)?;
                    parts.push(PreparedPart::Ready {
                        tokens: input::token_ids_array(
                            &vec![self.args.audio_token_id; embeddings.dim(1) as usize],
                            stream,
                        )?,
                        embeddings,
                    });
                }
                (
                    input::Modality::Image | input::Modality::Audio,
                    input::InputPayload::Embeddings(embeddings),
                ) => {
                    input::ensure_hidden_size(
                        embeddings,
                        self.args.text_config.hidden_size,
                        "Inkling media embeddings",
                    )?;
                    let token = if part.modality == input::Modality::Image {
                        self.args.image_token_id
                    } else {
                        self.args.audio_token_id
                    };
                    parts.push(PreparedPart::Ready {
                        tokens: input::token_ids_array(
                            &vec![token; embeddings.dim(1) as usize],
                            stream,
                        )?,
                        embeddings: embeddings.clone(),
                    });
                }
                (modality, _) => {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "Inkling layerwise input does not support {} payloads of this kind",
                        modality.as_str()
                    )));
                }
            }
        }
        if vision_jobs.is_empty() {
            let token_parts = parts
                .iter()
                .map(|part| match part {
                    PreparedPart::Ready { tokens, .. } => tokens,
                    PreparedPart::Vision { .. } => unreachable!(),
                })
                .collect::<Vec<_>>();
            let embedding_parts = parts
                .iter()
                .map(|part| match part {
                    PreparedPart::Ready { embeddings, .. } => embeddings,
                    PreparedPart::Vision { .. } => unreachable!(),
                })
                .collect::<Vec<_>>();
            let _tokens = concatenate_axis(&token_parts, 1, stream)?;
            let hidden = concatenate_axis(&embedding_parts, 1, stream)?;
            return Ok(LayerwiseForwardState {
                hidden,
                context: InklingForwardContext {
                    parts,
                    vision_jobs,
                    needs_assembly: false,
                    last_token_only,
                },
            });
        }
        let hidden = vision_jobs
            .first()
            .map(|job| job.hidden.clone())
            .unwrap_or_else(|| {
                parts
                    .first()
                    .map(|part| match part {
                        PreparedPart::Ready { embeddings, .. } => embeddings.clone(),
                        PreparedPart::Vision { .. } => unreachable!(),
                    })
                    .expect("validated non-empty Inkling input")
            });
        Ok(LayerwiseForwardState {
            hidden,
            context: InklingForwardContext {
                parts,
                vision_jobs,
                needs_assembly: true,
                last_token_only,
            },
        })
    }

    fn begin_forward_with_execution<'a>(
        &mut self,
        input: Self::Input<'a>,
        _cache: &mut Self::Cache,
        execution: &crate::runtime::distributed::parallel::ParallelExecutionContext<'_>,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        let Some(embedding) = &mut self.parallel_embedding else {
            return self.begin_forward(input, _cache, execution.stream());
        };
        let InklingExecutionInput {
            input,
            last_token_only,
        } = input;
        let stream = execution.stream();
        if let InklingInput::Decode(tokens) = input {
            let hidden = self
                .embed_norm
                .forward(&embedding.forward(tokens, execution)?, stream)?;
            return Ok(LayerwiseForwardState {
                hidden,
                context: InklingForwardContext {
                    parts: Vec::new(),
                    vision_jobs: Vec::new(),
                    needs_assembly: false,
                    last_token_only,
                },
            });
        }
        let InklingInput::Prefill(typed) = input else {
            unreachable!()
        };
        input::validate(typed)?;
        let mut parts = Vec::with_capacity(typed.parts.len());
        let mut vision_jobs = Vec::new();
        for part in typed.parts {
            match (part.modality, part.payload) {
                (input::Modality::Text, input::InputPayload::TokenIds(tokens)) => {
                    let embeddings = self
                        .embed_norm
                        .forward(&embedding.forward(tokens, execution)?, stream)?;
                    parts.push(PreparedPart::Ready {
                        tokens: tokens.clone(),
                        embeddings,
                    });
                }
                (input::Modality::Image, input::InputPayload::Tensor(pixels)) => {
                    if self.vision_norm.is_none() {
                        return Err(Error::UnsupportedArchitecture(
                            "Inkling image input requires vision_config and vision weights".into(),
                        ));
                    }
                    let job = vision_jobs.len();
                    vision_jobs.push(VisionJob {
                        hidden: pixels.clone(),
                    });
                    parts.push(PreparedPart::Vision {
                        tokens: input::token_ids_array(
                            &vec![self.args.image_token_id; pixels.dim(0) as usize],
                            stream,
                        )?,
                        job,
                    });
                }
                (input::Modality::Audio, input::InputPayload::Tensor(ids)) => {
                    let embeddings = self
                        .audio
                        .as_mut()
                        .ok_or_else(|| {
                            Error::UnsupportedArchitecture(
                                "Inkling audio input requires audio_config and audio weights"
                                    .into(),
                            )
                        })?
                        .forward_tensor_parallel(
                            ids,
                            part.metadata.audio_mask,
                            execution.group().ok_or_else(|| {
                                Error::Parallel("missing Inkling TP group".into())
                            })?,
                            stream,
                        )?;
                    parts.push(PreparedPart::Ready {
                        tokens: input::token_ids_array(
                            &vec![self.args.audio_token_id; embeddings.dim(1) as usize],
                            stream,
                        )?,
                        embeddings,
                    });
                }
                (
                    input::Modality::Image | input::Modality::Audio,
                    input::InputPayload::Embeddings(embeddings),
                ) => {
                    input::ensure_hidden_size(
                        embeddings,
                        self.args.text_config.hidden_size,
                        "Inkling media embeddings",
                    )?;
                    let token = if part.modality == input::Modality::Image {
                        self.args.image_token_id
                    } else {
                        self.args.audio_token_id
                    };
                    parts.push(PreparedPart::Ready {
                        tokens: input::token_ids_array(
                            &vec![token; embeddings.dim(1) as usize],
                            stream,
                        )?,
                        embeddings: embeddings.clone(),
                    });
                }
                (modality, _) => {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "Inkling layerwise input does not support {} payloads of this kind",
                        modality.as_str()
                    )));
                }
            }
        }
        if vision_jobs.is_empty() {
            let token_parts = parts
                .iter()
                .map(|part| match part {
                    PreparedPart::Ready { tokens, .. } => tokens,
                    PreparedPart::Vision { .. } => unreachable!(),
                })
                .collect::<Vec<_>>();
            let embedding_parts = parts
                .iter()
                .map(|part| match part {
                    PreparedPart::Ready { embeddings, .. } => embeddings,
                    PreparedPart::Vision { .. } => unreachable!(),
                })
                .collect::<Vec<_>>();
            let _tokens = concatenate_axis(&token_parts, 1, stream)?;
            let hidden = concatenate_axis(&embedding_parts, 1, stream)?;
            return Ok(LayerwiseForwardState {
                hidden,
                context: InklingForwardContext {
                    parts,
                    vision_jobs,
                    needs_assembly: false,
                    last_token_only,
                },
            });
        }
        let hidden = vision_jobs
            .first()
            .map(|job| job.hidden.clone())
            .or_else(|| {
                parts.iter().find_map(|part| match part {
                    PreparedPart::Ready { embeddings, .. } => Some(embeddings.clone()),
                    PreparedPart::Vision { .. } => None,
                })
            })
            .expect("validated non-empty Inkling input");
        Ok(LayerwiseForwardState {
            hidden,
            context: InklingForwardContext {
                parts,
                vision_jobs,
                needs_assembly: true,
                last_token_only,
            },
        })
    }

    fn execution_group_count(&self) -> usize {
        1 + usize::from(self.vision_depth > 0)
    }

    fn execution_group_id(&self, group: usize) -> Result<String, Error> {
        match (self.vision_depth > 0, group) {
            (true, 0) => Ok("vision_encoder".into()),
            (true, 1) | (false, 0) => Ok("text_decoder".into()),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Inkling has no execution group {group}"
            ))),
        }
    }

    fn should_execute_group(&self, group: usize, context: &Self::ForwardContext) -> bool {
        self.execution_group_id(group)
            .is_ok_and(|id| id != "vision_encoder" || !context.vision_jobs.is_empty())
    }

    fn layer_count(&self, group: usize) -> Result<usize, Error> {
        match self.execution_group_id(group)?.as_str() {
            "vision_encoder" => Ok(self.vision_depth),
            "text_decoder" => Ok(self.args.text_config.num_hidden_layers as usize),
            _ => unreachable!(),
        }
    }

    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        self.layer_count(group)?;
        if self.execution_group_id(group)? == "vision_encoder" {
            let args = self
                .args
                .vision_config
                .as_ref()
                .expect("vision group config");
            let specs = args.layer_specs();
            let (input_dim, output_dim, t_fold, hw_fold) = specs[index];
            Ok(InklingLayer::Vision(VisionLayer::new(
                (input_dim, output_dim, t_fold, hw_fold),
                index + 1 != specs.len(),
                args.rms_norm_eps,
                args.weight_quantization_for(&format!("visual.layers.{index}.projection.weight")),
                self.args.text_config.weight_dtype(),
                stream,
            )?))
        } else {
            Ok(InklingLayer::Text(Box::new(DecoderLayer::new(
                &self.args.text_config,
                index as i32,
                stream,
            )?)))
        }
    }

    fn parallel_parameter_groups(
        &self,
        _context: crate::runtime::distributed::parallel::ParallelBuildContext,
    ) -> Result<Vec<crate::runtime::distributed::parallel::ParameterGroupSpec>, Error> {
        use crate::runtime::distributed::parallel::{
            MemberSharding, ParameterGroupSpec, ParameterMemberSpec, ParameterRole,
        };
        let text = &self.args.text_config;
        let mut groups = vec![
            vocab_embedding_parameter_group(
                &self.embedding,
                "model.embed_tokens",
                text.vocab_size as usize,
                text.hidden_size,
                false,
            )?,
            vocab_lm_head_parameter_group(
                &self.lm_head,
                "lm_head",
                text.hidden_size,
                text.vocab_size as usize,
                false,
            )?,
        ];
        let Some(args) = &self.args.vision_config else {
            return Ok(groups);
        };
        for (index, (input, output, _, _)) in args.layer_specs().into_iter().enumerate() {
            let name = format!("visual.layers.{index}.projection.weight");
            let quantization = args.weight_quantization_for(&name);
            let mut members = vec![ParameterMemberSpec::new(
                name,
                [
                    output as usize,
                    quantization.map_or(input as usize, |quantization| {
                        safemlx::ops::quantized_packed_dimension(input, quantization.bits())
                            as usize
                    }),
                ],
                MemberSharding::Equal { axis: 1 },
            )];
            if let Some(quantization) = quantization {
                let companion_shape = [
                    output as usize,
                    (input / quantization.group_size()) as usize,
                ];
                members.push(ParameterMemberSpec::new(
                    format!("visual.layers.{index}.projection.scales"),
                    companion_shape,
                    MemberSharding::Equal { axis: 1 },
                ));
                if quantization.has_biases() {
                    members.push(ParameterMemberSpec::new(
                        format!("visual.layers.{index}.projection.biases"),
                        companion_shape,
                        MemberSharding::Equal { axis: 1 },
                    ));
                }
            }
            groups.push(ParameterGroupSpec::new(
                format!("visual.layers.{index}.projection"),
                ParameterRole::RowProjection,
                members,
            )?);
        }
        Ok(groups)
    }

    fn configure_parallel_static(
        &mut self,
        context: crate::runtime::distributed::parallel::ParallelBuildContext,
        _layout: &crate::runtime::distributed::parallel::LocalModelLayout,
        stream: &Stream,
    ) -> Result<(), Error> {
        let text = &self.args.text_config;
        self.parallel_embedding = Some(VocabParallelEmbedding::unloaded_with_dtype(
            text.vocab_size as usize,
            text.hidden_size,
            text.weight_quantization_for("model.embed_tokens.weight"),
            text.weight_dtype(),
            context,
            stream,
        )?);
        self.parallel_lm_head = Some(VocabParallelLmHead::unloaded_with_dtype(
            text.hidden_size,
            text.vocab_size as usize,
            text.weight_quantization_for("lm_head.weight"),
            text.weight_dtype(),
            context,
            stream,
        )?);
        if let Some(audio) = &self.args.audio_config {
            self.audio = Some(AudioModel::new_tensor_parallel(
                audio,
                text.weight_dtype(),
                context.topology(),
                stream,
            )?);
        }
        Ok(())
    }

    fn register_parallel_parameters(
        &self,
        context: crate::runtime::distributed::parallel::ParallelBuildContext,
        planner: &mut crate::runtime::distributed::parallel::ParallelPlanBuilder,
        stream: &Stream,
    ) -> Result<(), Error> {
        for group in self.parallel_parameter_groups(context)? {
            planner.register(group)?;
        }
        for index in 0..self.args.text_config.num_hidden_layers as usize {
            let layer = DecoderLayer::new(&self.args.text_config, index as i32, stream)?;
            layer.register_tensor_parallel_parameters(planner, &format!("model.layers.{index}"))?;
        }
        if let Some(audio) = &self.audio {
            audio.register_tensor_parallel_parameters(planner, "audio")?;
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
        if self.execution_group_id(group)? != "vision_encoder" {
            let prefix = format!("model.layers.{index}.self_attn.q_proj");
            let query = layout
                .tensor(&format!("{prefix}.weight"))
                .or_else(|| layout.tensor(&format!("{prefix}.inner.weight")))
                .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix}")))?;
            let parts = query.global_shape()[0]
                .checked_div(query.local_shape()[0])
                .ok_or_else(|| Error::Parallel("invalid Inkling TP query geometry".into()))?;
            return Ok(InklingLayer::Text(Box::new(
                DecoderLayer::new_tensor_parallel(
                    &self.args.text_config,
                    index as i32,
                    parts,
                    stream,
                )?,
            )));
        }
        let args = self
            .args
            .vision_config
            .as_ref()
            .expect("vision group config");
        let specs = args.layer_specs();
        let (_, output, t_fold, hw_fold) = specs[index];
        let target = format!("visual.layers.{index}.projection.weight");
        let local_input = i32::try_from(
            layout
                .tensor(&target)
                .ok_or_else(|| Error::Parallel(format!("missing TP layout for {target}")))?
                .local_shape()[1],
        )
        .map_err(|_| Error::Parallel("Inkling vision local input exceeds i32".into()))?;
        Ok(InklingLayer::Vision(VisionLayer::new(
            (local_input, output, t_fold, hw_fold),
            index + 1 != specs.len(),
            args.rms_norm_eps,
            args.weight_quantization_for(&target),
            self.args.text_config.weight_dtype(),
            stream,
        )?))
    }

    fn layer_checkpoint_prefix(&self, group: usize, index: usize) -> String {
        if self.execution_group_id(group).ok().as_deref() == Some("vision_encoder") {
            format!("visual.layers.{index}")
        } else {
            format!("model.layers.{index}")
        }
    }

    fn layer_unit_name(&self, group: usize, index: usize) -> String {
        if self.execution_group_id(group).ok().as_deref() == Some("vision_encoder") {
            format!("inkling.vision.{index:05}")
        } else {
            format!("inkling.layer.{index:05}")
        }
    }

    fn layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn WeightStore,
    ) -> Result<Vec<WeightBinding>, Error> {
        let prefix = self.layer_checkpoint_prefix(group, index);
        let bindings = build_module_bindings_with_recipes(
            layer,
            &prefix,
            store,
            self.recipes_for_module(layer, &prefix, store)?,
        )?;
        Ok(
            if self.sparse_expert_cache && self.execution_group_id(group)? == "text_decoder" {
                bindings
                    .into_iter()
                    .filter(|binding| !binding.name().starts_with("moe.experts."))
                    .collect()
            } else {
                bindings
            },
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
                |name| name.starts_with("moe.experts."),
            )?)
        } else {
            Ok(populate_module_from_lease(layer, lease)?)
        }
    }

    fn additional_consumed_checkpoint_keys(&self, store: &dyn WeightStore) -> Vec<String> {
        if self.sparse_expert_cache {
            store
                .keys()
                .into_iter()
                .filter(|key| key.contains(".mlp.experts.") || key.contains(".moe.experts."))
                .collect()
        } else {
            Vec::new()
        }
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
        match (self.execution_group_id(group)?.as_str(), layer) {
            ("vision_encoder", InklingLayer::Vision(layer)) => {
                for job in &mut context.vision_jobs {
                    job.hidden = layer.forward(&job.hidden, stream)?;
                }
                Ok(context.vision_jobs[0].hidden.clone())
            }
            ("text_decoder", InklingLayer::Text(layer)) => {
                let policy = self.args.text_config.layer_policy(index).ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "Inkling layer schedule has no layer {index}"
                    ))
                })?;
                if self.sparse_expert_cache
                    && policy.feed_forward == resident::FeedForwardPolicy::SparseMoe
                {
                    let expert_cache = self.expert_cache.as_ref().ok_or_else(|| {
                        Error::UnsupportedArchitecture(
                            "Inkling sparse expert cache was not initialized".into(),
                        )
                    })?;
                    let pass = if hidden.dim(1) > 1 {
                        ExpertPass::Prefill
                    } else {
                        ExpertPass::Decode
                    };
                    return Ok(layer.forward_with_expert_executor(
                        hidden,
                        Some(&mut cache.layers[index]),
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
                                        self.forward_cached_expert_bank(
                                            index, flat, acquired, weights, stream,
                                        )
                                    },
                                )
                                .map_err(|error| Exception::custom(error.to_string()))
                        },
                    )?);
                }
                Ok(layer.forward(hidden, Some(&mut cache.layers[index]), stream)?)
            }
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Inkling execution unit does not match group {group}"
            ))),
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
        if self.execution_group_id(group)? == "vision_encoder" {
            if let InklingLayer::Vision(layer) = layer {
                for job in &mut context.vision_jobs {
                    job.hidden =
                        layer.forward_tensor_parallel(&job.hidden, tp_group, execution.stream())?;
                }
                return Ok(context.vision_jobs[0].hidden.clone());
            }
        } else if let InklingLayer::Text(layer) = layer {
            if self.sparse_expert_cache {
                return Err(Error::Parallel(
                    "Inkling tensor parallelism cannot be combined with sparse expert caching"
                        .into(),
                ));
            }
            return Ok(layer.forward_tensor_parallel(
                hidden,
                Some(&mut cache.layers[index]),
                tp_group,
                execution.stream(),
            )?);
        }
        self.forward_layer(
            group,
            index,
            layer,
            hidden,
            cache,
            context,
            execution.stream(),
        )
    }

    fn retained_arrays<'a>(
        &self,
        cache: &'a Self::Cache,
        group: usize,
        index: usize,
    ) -> Vec<&'a Array> {
        if self.execution_group_id(group).ok().as_deref() == Some("text_decoder") {
            let layer = &cache.layers[index];
            let mut arrays = layer.kv.retained_arrays();
            arrays.extend(
                layer
                    .convolutions
                    .iter()
                    .filter_map(|cache| cache.state.as_ref()),
            );
            arrays
        } else {
            Vec::new()
        }
    }

    fn retained_context_arrays<'a>(
        &self,
        context: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Vec<&'a Array> {
        context.vision_jobs.iter().map(|job| &job.hidden).collect()
    }

    fn finish_execution_group(
        &mut self,
        group: usize,
        hidden: &Array,
        _cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let group_id = self.execution_group_id(group)?;
        let should_assemble = context.needs_assembly && group_id == "vision_encoder";
        if !should_assemble {
            return Ok(hidden.clone());
        }
        if let Some(norm) = &mut self.vision_norm {
            for job in &mut context.vision_jobs {
                job.hidden = norm
                    .forward(&job.hidden, stream)?
                    .reshape(&[-1, self.args.text_config.hidden_size], stream)?
                    .try_index_device(NewAxis, stream)?;
            }
        }
        let mut tokens = Vec::with_capacity(context.parts.len());
        let mut embeddings = Vec::with_capacity(context.parts.len());
        for part in &context.parts {
            match part {
                PreparedPart::Ready {
                    tokens: ids,
                    embeddings: value,
                } => {
                    tokens.push(ids);
                    embeddings.push(value);
                }
                PreparedPart::Vision { tokens: ids, job } => {
                    tokens.push(ids);
                    embeddings.push(&context.vision_jobs[*job].hidden);
                }
            }
        }
        let _tokens = concatenate_axis(&tokens, 1, stream)?;
        context.needs_assembly = false;
        Ok(concatenate_axis(&embeddings, 1, stream)?)
    }

    fn finish(
        &mut self,
        hidden: &Array,
        _cache: &mut Self::Cache,
        context: &Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let mut hidden = self.norm.forward(hidden, stream)?;
        if context.last_token_only {
            hidden = hidden.try_index_device((.., -1, ..), stream)?;
        }
        let hidden = hidden.divide(
            Array::from_f32(self.args.text_config.logits_mup_width_multiplier),
            stream,
        )?;
        let logits = self.lm_head.forward(&hidden, stream)?;
        if let Some(size) = self.args.text_config.unpadded_vocab_size {
            if size < logits.dim(-1) {
                return Ok(match logits.ndim() {
                    2 => logits.try_index_device((.., ..size), stream)?,
                    3 => logits.try_index_device((.., .., ..size), stream)?,
                    rank => {
                        return Err(Error::UnsupportedArchitecture(format!(
                            "Inkling logits have unsupported rank {rank}"
                        )));
                    }
                });
            }
        }
        Ok(logits)
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
        let mut hidden = self.norm.forward(hidden, execution.stream())?;
        if context.last_token_only {
            hidden = hidden.try_index_device((.., -1, ..), execution.stream())?;
        }
        let hidden = hidden.divide(
            Array::from_f32(self.args.text_config.logits_mup_width_multiplier),
            execution.stream(),
        )?;
        let logits = head.forward(&hidden, execution)?.all_gather(execution)?;
        if let Some(size) = self.args.text_config.unpadded_vocab_size {
            if size < logits.dim(-1) {
                return Ok(match logits.ndim() {
                    2 => logits.try_index_device((.., ..size), execution.stream())?,
                    3 => logits.try_index_device((.., .., ..size), execution.stream())?,
                    rank => {
                        return Err(Error::UnsupportedArchitecture(format!(
                            "Inkling logits have unsupported rank {rank}"
                        )));
                    }
                });
            }
        }
        Ok(logits)
    }

    fn ignores_checkpoint_key(&self, key: &str) -> bool {
        key.starts_with("model.mtp.")
    }
}

pub(crate) fn inkling_expert_catalog(
    args: &ModelArgs,
    store: &dyn WeightStore,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let normalized = normalized_checkpoint_keys(store);
    let text = &args.text_config;
    let mut entries = Vec::new();
    for layer in 0..text.num_hidden_layers as usize {
        let policy = text.layer_policy(layer).ok_or_else(|| {
            Error::UnsupportedArchitecture(format!("Inkling layer schedule has no layer {layer}"))
        })?;
        if policy.feed_forward == resident::FeedForwardPolicy::Dense {
            continue;
        }
        let runtime_prefix = format!("model.layers.{layer}");
        let gate_up_runtime = format!("{runtime_prefix}.moe.experts.gate_up_proj");
        let down_runtime = format!("{runtime_prefix}.moe.experts.down_proj");
        let gate_up_raw = normalized.get(&gate_up_runtime).cloned().or_else(|| {
            normalized
                .get(&format!("{runtime_prefix}.mlp.experts.w13_weight"))
                .cloned()
        });
        let split_gate = normalized
            .get(&format!("{runtime_prefix}.moe.experts.gate_proj"))
            .cloned();
        let split_up = normalized
            .get(&format!("{runtime_prefix}.moe.experts.up_proj"))
            .cloned();
        if gate_up_raw.is_none() && (split_gate.is_none() || split_up.is_none()) {
            return Err(Error::UnsupportedArchitecture(format!(
                "Inkling checkpoint is missing routed gate/up bank for layer {layer}"
            )));
        }
        let down_raw = normalized.get(&down_runtime).cloned().ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Inkling checkpoint is missing routed down bank for layer {layer}"
            ))
        })?;
        let interleaved = gate_up_raw
            .as_ref()
            .map(|raw| store.metadata(raw))
            .transpose()?
            .and_then(|metadata| metadata.shape.get(1).copied());
        if gate_up_raw.is_some()
            && !normalized.contains_key(&gate_up_runtime)
            && interleaved.is_none_or(|width| width % 2 != 0)
        {
            return Err(Error::UnsupportedArchitecture(format!(
                "Inkling routed w13 bank for layer {layer} has invalid interleaved width"
            )));
        }
        let gate_format = text.weight_quantization_for(&gate_up_runtime);
        let down_format = text.weight_quantization_for(&down_runtime);
        for expert in 0..text.n_routed_experts as usize {
            let identity = ExpertIdentity::new(layer, expert);
            let selected = |raw: String| {
                DerivedWeightRecipe::source(
                    raw,
                    TensorSelection::Range {
                        axis: 0,
                        start: expert,
                        end: expert + 1,
                    },
                )
            };
            let gate_up = if let (Some(gate), Some(up)) = (&split_gate, &split_up) {
                DerivedWeightRecipe::Concatenate {
                    axis: 1,
                    inputs: vec![selected(gate.clone()), selected(up.clone())],
                }
            } else {
                let gate_up_raw = gate_up_raw.clone().expect("validated gate/up source");
                let selected_expert = selected(gate_up_raw);
                if normalized.contains_key(&gate_up_runtime) {
                    selected_expert
                } else {
                    let width = interleaved.expect("validated interleaved width");
                    let select = |parity| DerivedWeightRecipe::Select {
                        input: Box::new(selected_expert.clone()),
                        selection: TensorSelection::Indices {
                            axis: 1,
                            indices: (parity..width).step_by(2).collect(),
                        },
                    };
                    DerivedWeightRecipe::Concatenate {
                        axis: 1,
                        inputs: vec![select(0), select(1)],
                    }
                }
            };
            let down = selected(down_raw.clone());
            let mut recipes = vec![("gate_up_proj", gate_up), ("down_proj", down)];
            if gate_format.is_some_and(|format| format.gguf_iquant().is_none()) {
                let gate = normalized
                    .get(&format!("{runtime_prefix}.moe.experts.gate_proj_scales"))
                    .cloned()
                    .ok_or_else(|| {
                        Error::UnsupportedArchitecture(
                            "Inkling GGUF gate scales are missing".into(),
                        )
                    })?;
                let up = normalized
                    .get(&format!("{runtime_prefix}.moe.experts.up_proj_scales"))
                    .cloned()
                    .ok_or_else(|| {
                        Error::UnsupportedArchitecture("Inkling GGUF up scales are missing".into())
                    })?;
                recipes.push((
                    "gate_up_proj_scales",
                    DerivedWeightRecipe::Concatenate {
                        axis: 1,
                        inputs: vec![selected(gate), selected(up)],
                    },
                ));
            }
            if gate_format.is_some_and(|format| format.has_biases()) {
                let gate = normalized
                    .get(&format!("{runtime_prefix}.moe.experts.gate_proj_biases"))
                    .cloned()
                    .ok_or_else(|| {
                        Error::UnsupportedArchitecture(
                            "Inkling GGUF gate biases are missing".into(),
                        )
                    })?;
                let up = normalized
                    .get(&format!("{runtime_prefix}.moe.experts.up_proj_biases"))
                    .cloned()
                    .ok_or_else(|| {
                        Error::UnsupportedArchitecture("Inkling GGUF up biases are missing".into())
                    })?;
                recipes.push((
                    "gate_up_proj_biases",
                    DerivedWeightRecipe::Concatenate {
                        axis: 1,
                        inputs: vec![selected(gate), selected(up)],
                    },
                ));
            }
            if down_format.is_some_and(|format| format.gguf_iquant().is_none()) {
                let raw = normalized
                    .get(&format!("{down_runtime}_scales"))
                    .cloned()
                    .ok_or_else(|| {
                        Error::UnsupportedArchitecture(
                            "Inkling GGUF down scales are missing".into(),
                        )
                    })?;
                recipes.push(("down_proj_scales", selected(raw)));
            }
            if down_format.is_some_and(|format| format.has_biases()) {
                let raw = normalized
                    .get(&format!("{down_runtime}_biases"))
                    .cloned()
                    .ok_or_else(|| {
                        Error::UnsupportedArchitecture(
                            "Inkling GGUF down biases are missing".into(),
                        )
                    })?;
                recipes.push(("down_proj_biases", selected(raw)));
            }
            let mut bindings = Vec::new();
            for (name, recipe) in recipes {
                let bytes = recipe.infer(store)?.byte_len();
                bindings.push(WeightBinding::from_recipe(name, recipe, bytes)?);
            }
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    Error::UnsupportedArchitecture("Inkling expert byte total overflowed".into())
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

/// Inkling text token generation using bounded layer execution.
pub type Generate<'a, S = crate::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, InklingLayerwiseModel, Cache, S>;

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use safemlx::{
        distributed::{Backend, Group},
        module::ModuleParameters,
        ops::{indexing::TryIndexOp, ones_dtype, stack_axis},
        Array, Device, DeviceType, Dtype, ExecutionContext, Stream,
    };

    use super::{
        load_inkling_layerwise_model, load_inkling_sparse_expert_cache_model,
        load_inkling_tensor_parallel_layerwise_model,
    };
    use crate::{
        api::{
            common::generation::CausalLm,
            inkling::{self as resident, Model, ModelArgs},
            input as runtime_input,
        },
        runtime::cache::KeyValueCache,
        runtime::distributed::{
            parallel::{ParallelBuildContext, ShardingPolicy},
            topology::{DeviceAssignment, ParallelTopology},
        },
        runtime::execution::layerwise::{LayerExecutionLoadOptions, LayerwiseLoadOptions},
        runtime::residency::dense_stream::DenseDiskStreamLoadOptions,
        runtime::residency::expert_cache::ExpertCacheLoadOptions,
        runtime::residency::policy::{OffloadConfig, ResidencyPolicy},
        PagedCacheOptions,
    };

    fn config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "inkling_mm_model",
            "eos_token_id": 1,
            "text_config": {
                "torch_dtype": "bfloat16",
                "hidden_size": 16,
                "num_hidden_layers": 3,
                "vocab_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "swa_num_attention_heads": 2,
                "swa_num_key_value_heads": 1,
                "swa_head_dim": 8,
                "sliding_window_size": 4,
                "layer_types": ["full_attention", "sliding_attention", "full_attention"],
                "dense_mlp_idx": 1,
                "sconv_kernel_size": 3,
                "d_rel": 4,
                "rel_extent": 8,
                "intermediate_size": 8,
                "dense_intermediate_size": 16,
                "moe_intermediate_size": 8,
                "n_routed_experts": 2,
                "num_experts_per_tok": 1,
                "n_shared_experts": 1,
                "route_scale": 1.0,
                "use_sconv": true,
                "use_embed_norm": true,
                "shared_expert_sink": true,
                "use_gate_bias": true,
                "norm_after_topk": true,
                "use_global_scale": true,
                "gate_activation": "sigmoid",
                "hidden_act": "silu",
                "attention_dropout": 0.0,
                "q_bias": false,
                "o_bias": false,
                "logits_mup_width_multiplier": 2.0,
                "unpadded_vocab_size": 30
            }
        })
    }

    fn args() -> ModelArgs {
        resident::model_args_from_config_value(&config()).unwrap()
    }

    #[test]
    fn released_mixed_dtype_policy_keeps_only_router_scalars_in_f32() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let model = Model::new(args(), context.stream()).unwrap();
        let parameters = model.parameters().flatten();
        assert_eq!(
            parameters["model.layers.0.dense_global_scale"].dtype(),
            Dtype::Bfloat16
        );
        assert_eq!(
            parameters["model.layers.1.moe.router.bias"].dtype(),
            Dtype::Float32
        );
        assert_eq!(
            parameters["model.layers.1.moe.router.global_scale"].dtype(),
            Dtype::Float32
        );
    }

    fn initialize(model: &mut Model, stream: &Stream) {
        for (name, parameter) in model.parameters_mut().flatten() {
            let shape = parameter.shape().to_vec();
            *parameter = if name.ends_with("norm.weight")
                || name.ends_with("layernorm.weight")
                || name.ends_with("global_scale")
            {
                ones_dtype(&shape, parameter.dtype(), stream).unwrap()
            } else {
                Array::full::<f32>(&shape, Array::from_f32(0.01), stream)
                    .unwrap()
                    .as_dtype(parameter.dtype(), stream)
                    .unwrap()
            };
        }
    }

    fn released_name(runtime: &str) -> String {
        if runtime == "lm_head.weight" {
            return "model.llm.unembed.weight".into();
        }
        if let Some(rest) = runtime.strip_prefix("audio.") {
            return format!("model.audio.{rest}");
        }
        if let Some(rest) = runtime.strip_prefix("visual.") {
            return format!("model.visual.{rest}");
        }
        let rest = runtime.strip_prefix("model.").unwrap();
        let mut raw = format!("model.llm.{rest}");
        raw = raw
            .replace("model.llm.embed_tokens.weight", "model.llm.embed.weight")
            .replace(".input_layernorm.weight", ".attn_norm.weight")
            .replace(".post_attention_layernorm.weight", ".mlp_norm.weight")
            .replace(".self_attn.q_proj.weight", ".attn.wq_du.weight")
            .replace(".self_attn.k_proj.weight", ".attn.wk_dv.weight")
            .replace(".self_attn.v_proj.weight", ".attn.wv_dv.weight")
            .replace(".self_attn.r_proj.weight", ".attn.wr_du.weight")
            .replace(".self_attn.o_proj.weight", ".attn.wo_ud.weight")
            .replace(".self_attn.q_norm.weight", ".attn.q_norm.weight")
            .replace(".self_attn.k_norm.weight", ".attn.k_norm.weight")
            .replace(".self_attn.rel_proj", ".attn.rel_logits_proj.proj")
            .replace(".self_attn.k_sconv.weight", ".attn.k_sconv.weight")
            .replace(".self_attn.v_sconv.weight", ".attn.v_sconv.weight")
            .replace(".dense.down_proj.weight", ".mlp.w2_md.weight")
            .replace(".dense_global_scale", ".mlp.global_scale")
            .replace(".moe.router.weight", ".mlp.gate.weight")
            .replace(".moe.router.bias", ".mlp.gate.bias")
            .replace(".moe.router.global_scale", ".mlp.gate.global_scale")
            .replace(".moe.experts.down_proj", ".mlp.experts.w2_weight")
            .replace(
                ".moe.shared_experts.down_proj",
                ".mlp.shared_experts.shared_w2_weight",
            );
        raw
    }

    fn interleave(gate: &Array, up: &Array, axis: i32, stream: &Stream) -> Array {
        let stacked = stack_axis(&[gate.clone(), up.clone()], axis, stream).unwrap();
        let mut shape = gate.shape().to_vec();
        let row_axis = shape.len() - 2;
        shape[row_axis] *= 2;
        stacked.reshape(&shape, stream).unwrap()
    }

    fn write_fixture(dir: &Path, model: &Model, stream: &Stream) {
        let parameters = model.parameters().flatten();
        let mut arrays = Vec::<(String, Array)>::new();
        for (name, value) in &parameters {
            let name = name.as_ref();
            if name.ends_with(".dense.up_proj.weight") {
                continue;
            }
            if let Some(prefix) = name.strip_suffix(".dense.gate_proj.weight") {
                let up_name = format!("{prefix}.dense.up_proj.weight");
                let up = parameters.get(up_name.as_str()).unwrap();
                arrays.push((
                    format!("model.llm.{}.mlp.w13_dn.weight", &prefix["model.".len()..]),
                    interleave(value, up, 1, stream),
                ));
                continue;
            }
            if let Some(prefix) = name.strip_suffix(".moe.experts.gate_up_proj") {
                let intermediate = model.args.text_config.moe_intermediate_size.unwrap();
                let gate = value
                    .try_index_device((.., ..intermediate, ..), stream)
                    .unwrap();
                let up = value
                    .try_index_device((.., intermediate.., ..), stream)
                    .unwrap();
                arrays.push((
                    format!(
                        "model.llm.{}.mlp.experts.w13_weight",
                        &prefix["model.".len()..]
                    ),
                    interleave(&gate, &up, 2, stream),
                ));
                continue;
            }
            if let Some(prefix) = name.strip_suffix(".moe.shared_experts.gate_up_proj") {
                let intermediate = model.args.text_config.moe_intermediate_size.unwrap();
                let gate = value
                    .try_index_device((.., ..intermediate, ..), stream)
                    .unwrap();
                let up = value
                    .try_index_device((.., intermediate.., ..), stream)
                    .unwrap();
                arrays.push((
                    format!(
                        "model.llm.{}.mlp.shared_experts.shared_w13_weight",
                        &prefix["model.".len()..]
                    ),
                    interleave(&gate, &up, 2, stream),
                ));
                continue;
            }
            let raw = released_name(name);
            let value = if raw.ends_with("_sconv.weight") {
                value.as_dtype(Dtype::Bfloat16, stream).unwrap()
            } else {
                (*value).clone()
            };
            arrays.push((raw, value));
        }
        Array::save_safetensors(
            arrays.iter().map(|(name, value)| (name.as_str(), value)),
            None,
            dir.join("model.safetensors"),
        )
        .unwrap();
        fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&config()).unwrap(),
        )
        .unwrap();
    }

    fn assert_close(left: &Array, right: &Array, stream: &Stream) {
        let left_f32 = left.as_dtype(Dtype::Float32, stream).unwrap();
        let right_f32 = right.as_dtype(Dtype::Float32, stream).unwrap();
        let left = left_f32.evaluated().unwrap();
        let right = right_f32.evaluated().unwrap();
        assert_eq!(left.as_array().shape(), right.as_array().shape());
        for (left, right) in left.as_slice::<f32>().iter().zip(right.as_slice::<f32>()) {
            assert!((left - right).abs() <= 5e-5, "{left} != {right}");
        }
    }

    fn parity(depth: usize) {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut fixture = Model::new(args(), gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, gpu.stream());

        let mut resident = resident::load_model(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let options = LayerwiseLoadOptions::new(OffloadConfig::new(None, None, depth).unwrap());
        let mut layerwise =
            load_inkling_layerwise_model(dir.path(), options, gpu.stream(), cpu.stream()).unwrap();
        let mut resident_cache = resident.new_cache();
        let mut layerwise_cache = resident::Cache { layers: Vec::new() };
        for tokens in [
            Array::from_slice(&[1u32, 2, 3], &[1, 3]),
            Array::from_slice(&[4u32], &[1, 1]),
            Array::from_slice(&[5u32], &[1, 1]),
            Array::from_slice(&[6u32], &[1, 1]),
        ] {
            let expected = resident
                .forward_logits(
                    &tokens,
                    None,
                    Some(&mut resident_cache),
                    false,
                    gpu.stream(),
                )
                .unwrap();
            let actual = layerwise
                .forward(&tokens, &mut layerwise_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected, gpu.stream());
            assert_eq!(resident_cache.offset(), layerwise_cache.offset());
            for (expected, actual) in resident_cache.layers.iter().zip(&layerwise_cache.layers) {
                assert_eq!(expected.kv.offset(), actual.kv.offset());
                for (expected, actual) in expected.convolutions.iter().zip(&actual.convolutions) {
                    assert_eq!(expected.offset, actual.offset);
                    assert_eq!(
                        expected.state.as_ref().map(Array::shape),
                        actual.state.as_ref().map(Array::shape)
                    );
                }
            }
            let report = layerwise.residency_report().unwrap();
            let layers = report
                .units()
                .iter()
                .filter(|unit| unit.id().as_str().starts_with("inkling.layer."))
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
    #[ignore = "requires an MLX runtime with a Metal device"]
    fn tensor_parallel_dense_stream_loads_multimodal_static_and_text_group() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut fixture = Model::new(args(), gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, gpu.stream());
        let group = Group::init(false, Backend::Any).unwrap();
        assert_eq!(group.size(), 1);
        let topology =
            ParallelTopology::from_rank(1, 0, 1, 1, 1, DeviceAssignment::new(DeviceType::Gpu, 0))
                .unwrap();
        let build = ParallelBuildContext::new(topology, ShardingPolicy::Require);
        let options = DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1, 1).unwrap();
        let model = load_inkling_tensor_parallel_layerwise_model(
            dir.path(),
            LayerExecutionLoadOptions::DenseDiskStream(options),
            build,
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let report = model.dense_stream_report().unwrap().unwrap();
        assert!(
            report
                .residency()
                .units()
                .iter()
                .filter(|unit| unit.id().as_str().contains("inkling.text."))
                .all(|unit| unit.planned_tier()
                    == crate::runtime::residency::policy::MemoryTier::Disk)
        );
    }

    #[test]
    fn inkling_released_layout_layerwise_parity() {
        parity(1);
        parity(2);
    }

    #[test]
    fn inkling_global_and_sliding_attention_paged_parity() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let mut args = args();
        args.text_config.layer_schedule = crate::runtime::attention::LayerSchedule::new(
            3,
            vec![
                resident::LayerPolicy {
                    attention: crate::runtime::attention::AttentionPolicy::Full,
                    feed_forward: resident::FeedForwardPolicy::Dense,
                },
                resident::LayerPolicy {
                    attention: crate::runtime::attention::AttentionPolicy::sliding(4).unwrap(),
                    feed_forward: resident::FeedForwardPolicy::SparseMoe,
                },
                resident::LayerPolicy {
                    attention: crate::runtime::attention::AttentionPolicy::sliding(2).unwrap(),
                    feed_forward: resident::FeedForwardPolicy::SparseMoe,
                },
            ],
        )
        .unwrap();
        let mut expected_model = Model::new(args.clone(), gpu.stream()).unwrap();
        let mut paged_model = Model::new(args, gpu.stream()).unwrap();
        initialize(&mut expected_model, gpu.stream());
        initialize(&mut paged_model, gpu.stream());
        let mut expected_cache = expected_model.new_cache();
        let paging = PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1)
            .unwrap()
            .with_full_attention(true);
        let mut paged_cache = paged_model.new_paged_cache(paging).unwrap();

        for tokens in [
            Array::from_slice(&[1u32, 2, 3, 4, 5], &[1, 5]),
            Array::from_slice(&[6u32], &[1, 1]),
            Array::from_slice(&[7u32], &[1, 1]),
        ] {
            let expected = expected_model
                .forward_logits(
                    &tokens,
                    None,
                    Some(&mut expected_cache),
                    false,
                    gpu.stream(),
                )
                .unwrap();
            let actual = paged_model
                .forward_logits(&tokens, None, Some(&mut paged_cache), false, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected, gpu.stream());
            assert_eq!(paged_cache.offset(), expected_cache.offset());
        }

        let report = paged_cache.residency_report().unwrap().unwrap();
        assert!(report.key_value_blocks > 0);
        assert!(report.prefill_full_attention_blocks > 0);
        assert!(report.decode_full_attention_blocks > 0);
        assert_eq!(
            expected_cache
                .layers
                .iter()
                .map(|layer| {
                    layer
                        .kv
                        .retained_arrays()
                        .first()
                        .map(|array| array.dim(-2))
                        .unwrap_or(0)
                })
                .collect::<Vec<_>>(),
            vec![7, 3, 1]
        );
    }

    #[test]
    fn inkling_sparse_expert_cache_prefill_and_decode_parity() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut fixture = Model::new(args(), gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, gpu.stream());
        let mut resident = resident::load_model(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let options = ExpertCacheLoadOptions::new(
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
            OffloadConfig::new(None, None, 1).unwrap(),
            768,
            768,
        )
        .unwrap();
        let mut cached =
            load_inkling_sparse_expert_cache_model(dir.path(), options, gpu.stream(), cpu.stream())
                .unwrap();
        let mut resident_cache = resident.new_cache();
        let mut cached_cache = resident::Cache { layers: Vec::new() };
        for tokens in [
            Array::from_slice(&[1u32, 2, 3], &[1, 3]),
            Array::from_slice(&[4u32], &[1, 1]),
        ] {
            let expected = resident
                .forward_logits(
                    &tokens,
                    None,
                    Some(&mut resident_cache),
                    false,
                    gpu.stream(),
                )
                .unwrap();
            let actual = cached
                .forward(&tokens, &mut cached_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected, gpu.stream());
        }
        let report = cached.expert_cache_report().unwrap().unwrap();
        assert_eq!(report.owned_experts, 4);
        assert_eq!(report.prefill.compact_banks, 6);
        assert!(report.prefill.requested_routes > 0);
        assert!(report.decode.requested_routes > 0);
        crate::architectures::distributed::expert::assert_rank_owned_sparse_ep_load(
            dir.path(),
            options,
            crate::api::ModelKind::Inkling,
            report.owned_experts / 2,
            gpu.stream(),
            cpu.stream(),
        );
    }

    #[test]
    fn inkling_audio_and_text_layerwise_parity() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut value = config();
        value["audio_config"] = serde_json::json!({
            "text_hidden_size": 16,
            "num_codebooks": 2,
            "codebook_size": 8,
            "bias": false,
            "use_audio_norm": true,
            "audio_mode": "dmel",
            "rms_norm_eps": 1e-6,
        });
        value["audio_token_id"] = serde_json::json!(20);
        let mut fixture = Model::new(
            resident::model_args_from_config_value(&value).unwrap(),
            gpu.stream(),
        )
        .unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, gpu.stream());
        fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();

        let mut resident = resident::load_model(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let mut layerwise = load_inkling_layerwise_model(
            dir.path(),
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let text = runtime_input::token_ids_array(&[1, 2], gpu.stream()).unwrap();
        let audio_ids = Array::from_slice(&[0u32, 1, 2, 3, 4, 5], &[3, 2]);
        let mask = Array::from_slice(&[true, true, false], &[1, 3]);
        let parts = [
            runtime_input::InputPart::text_token_ids(&text),
            runtime_input::InputPart::audio_tensor(
                &audio_ids,
                runtime_input::InputMetadata::audio_mask(&mask),
            ),
        ];
        let typed = runtime_input::ModelInput::new(&parts);
        let mut resident_cache = resident.new_cache();
        let mut layerwise_cache = layerwise.new_cache();
        let expected = resident
            .prefill_input_logits(typed, &mut resident_cache, gpu.stream())
            .unwrap();
        let actual = layerwise
            .prefill_input_logits(typed, &mut layerwise_cache, gpu.stream())
            .unwrap();
        assert_close(&actual, &expected, gpu.stream());
        assert_eq!(resident_cache.offset(), layerwise_cache.offset());

        let next = runtime_input::token_ids_array(&[6], gpu.stream()).unwrap();
        let expected = resident
            .decode_logits(&next, &mut resident_cache, gpu.stream())
            .unwrap();
        let actual = layerwise
            .decode_logits(&next, &mut layerwise_cache, gpu.stream())
            .unwrap();
        assert_close(&actual, &expected, gpu.stream());
        assert_eq!(resident_cache.offset(), layerwise_cache.offset());
    }
}

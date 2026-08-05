//! Unified fully resident and bounded layer execution for LFM2/LFM2.5.

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
        common::moe::PackedSwiGluExperts,
        common::{self, generation::CausalLm, linear::project_logits_maybe_quantized},
        input,
        lfm2::{
            self as resident, Cache, DecoderLayer, FeedForwardPolicy, LayerCache, ModelArgs,
            OperatorPolicy,
        },
    },
    error::Error,
    nn::{
        parallel::{VocabParallelEmbedding, VocabParallelLmHead},
        tensor::{create_attention_mask, AttentionMask},
    },
    runtime::cache::{
        residency::{
            PagedCacheOptions, PromptCacheDescriptor, PromptCacheManifest,
            PromptCacheModelIdentity, PromptCacheOptions, PromptCacheTopology,
        },
        KeyValueCache,
    },
    runtime::checkpoint::binding::{
        build_module_bindings, build_module_bindings_with_recipes, populate_module_from_lease,
        populate_module_from_lease_excluding,
    },
    runtime::checkpoint::recipe::DerivedWeightRecipe,
    runtime::checkpoint::store::{GgufWeightStore, TensorSelection, WeightStore},
    runtime::distributed::parallel::exact_parallel_division,
    runtime::execution::layerwise::{
        load_layerwise_model, load_layerwise_model_with_store,
        load_tensor_parallel_layerwise_model, ArchitectureAdapter, LayerExecutionLoadOptions,
        LayerwiseForwardState, LayerwiseModel, StaticUnitBindings, WeightResidency,
    },
    runtime::residency::expert_cache::{
        ExpertCache, ExpertCacheLoadOptions, ExpertCacheReport, ExpertCatalogEntry, ExpertIdentity,
        ExpertPass,
    },
    runtime::residency::manager::{OffloadUnit, ResidencyReport, ResidentUnitLease, WeightBinding},
};

const EMBEDDING_UNIT: &str = "lfm2.static.embedding";
const NORM_UNIT: &str = "lfm2.static.norm";
const HEAD_UNIT: &str = "lfm2.static.output";

/// LFM2/LFM2.5 causal LM with host-backed decoder blocks.
pub struct Lfm2LayerwiseModel {
    execution: LayerwiseModel<Lfm2LayerwiseAdapter>,
}

impl Lfm2LayerwiseModel {
    /// Returns the validated model arguments.
    pub fn args(&self) -> &ModelArgs {
        self.execution.adapter().args()
    }

    /// Creates heterogeneous attention and convolution state.
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

    /// Persists a compatible prefix cache.
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

    /// Restores a compatible prefix cache.
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

    /// Runs the hybrid decoder while preserving recurrent and KV state.
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution.forward(inputs, cache, stream)
    }
    /// Runs a rank-local tensor-parallel hybrid forward pass.
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
            |_adapter, _group, index, layer, hidden, cache, context, stream| {
                Ok(layer.forward_with_expert_executor(
                    hidden,
                    context.mask.as_ref(),
                    Some(&mut cache.layers[index]),
                    stream,
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

impl CausalLm<Cache> for Lfm2LayerwiseModel {
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

/// Loads dense or MoE LFM2 through bounded layer residency.
pub fn load_lfm2_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerExecutionLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Lfm2LayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let residency = match options {
        LayerExecutionLoadOptions::LayerwiseHost(options) => {
            WeightResidency::LayerwiseHost(options)
        }
        LayerExecutionLoadOptions::DenseDiskStream(options) => {
            WeightResidency::DenseDiskStream(options)
        }
    };
    crate::api::structural::validate_safetensors_load_path(
        crate::api::ModelKind::Lfm2,
        model_dir,
        crate::api::ModelLoadOptions::default().with_weight_residency(residency),
    )?;
    let args = resident::get_model_args(model_dir)?;
    let adapter = Lfm2LayerwiseAdapter::new(args, stream)?;
    Ok(Lfm2LayerwiseModel {
        execution: load_layerwise_model(model_dir, adapter, options, stream, weights_stream)?,
    })
}
/// Loads dense or MoE LFM2 through the generalized tensor-parallel engine.
pub fn load_lfm2_tensor_parallel_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerExecutionLoadOptions>,
    build: crate::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Lfm2LayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let residency = match options {
        LayerExecutionLoadOptions::LayerwiseHost(v) => WeightResidency::LayerwiseHost(v),
        LayerExecutionLoadOptions::DenseDiskStream(v) => WeightResidency::DenseDiskStream(v),
    };
    crate::api::structural::validate_safetensors_load_path(
        crate::api::ModelKind::Lfm2,
        model_dir,
        crate::api::ModelLoadOptions::default().with_weight_residency(residency),
    )?;
    Ok(Lfm2LayerwiseModel {
        execution: load_tensor_parallel_layerwise_model(
            model_dir,
            Lfm2LayerwiseAdapter::new(resident::get_model_args(model_dir)?, stream)?,
            options,
            build,
            stream,
            weights_stream,
        )?,
    })
}

pub(crate) fn load_lfm2_gguf_layerwise_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    residency: WeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(Lfm2LayerwiseModel, Vec<u32>), Error> {
    let prepared = resident::prepare_gguf_checkpoint(checkpoint, metadata, weights_stream)?;
    let args = prepared.args;
    let is_moe = args.model_type == "lfm2_moe";
    let store: Arc<dyn WeightStore + Send + Sync> =
        Arc::new(GgufWeightStore::new_with_max_mapped_shards(
            checkpoint.clone(),
            |name| resident::translate_gguf_weight_name(name, is_moe),
            residency.max_mapped_shards(),
        )?);
    let execution = match residency {
        WeightResidency::LayerwiseHost(options) => load_layerwise_model_with_store(
            store,
            Lfm2LayerwiseAdapter::new(args, stream)?,
            options,
            stream,
            weights_stream,
        )?,
        WeightResidency::DenseDiskStream(options) => load_layerwise_model_with_store(
            store,
            Lfm2LayerwiseAdapter::new(args, stream)?,
            options,
            stream,
            weights_stream,
        )?,
        WeightResidency::SparseExpertCache(options) => {
            return Ok((
                load_lfm2_gguf_sparse_with_store(
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
                load_lfm2_gguf_sparse_with_store(
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
                "the bounded GGUF LFM2 loader does not accept fully resident policy".into(),
            ));
        }
    };
    Ok((Lfm2LayerwiseModel { execution }, prepared.eos_token_ids))
}

fn load_lfm2_gguf_sparse_with_store(
    store: Arc<dyn WeightStore + Send + Sync>,
    args: ModelArgs,
    options: ExpertCacheLoadOptions,
    non_expert: impl Into<LayerExecutionLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Lfm2LayerwiseModel, Error> {
    if !args.has_sparse_moe_layers() {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires an LFM2 MoE GGUF checkpoint".into(),
        ));
    }
    let mut adapter = Lfm2LayerwiseAdapter::new(args.clone(), stream)?;
    adapter.sparse_expert_cache = true;
    let mut execution =
        load_layerwise_model_with_store(store, adapter, non_expert, stream, weights_stream)?;
    let checkpoint_store = execution.weight_store_arc();
    let entries = lfm2_expert_catalog(&args, checkpoint_store.as_ref())?;
    execution.adapter_mut().expert_cache = Some(ExpertCache::new_shared(
        checkpoint_store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(Lfm2LayerwiseModel { execution })
}

/// Builds the streamed nonexpert LFM2 execution base used by distributed EP.
pub(crate) fn load_lfm2_sparse_ep_base_with_store(
    store: Arc<dyn WeightStore + Send + Sync>,
    args: ModelArgs,
    non_expert: crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Lfm2LayerwiseModel, Error> {
    let mut adapter = Lfm2LayerwiseAdapter::new(args, stream)?;
    adapter.sparse_expert_cache = true;
    let execution =
        load_layerwise_model_with_store(store, adapter, non_expert, stream, weights_stream)?;
    Ok(Lfm2LayerwiseModel { execution })
}

/// Loads MoE LFM2 with expert-granular sparse caching.
pub fn load_lfm2_sparse_expert_cache_model(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Lfm2LayerwiseModel, Error> {
    load_lfm2_sparse_expert_cache_model_with_non_expert(
        model_dir,
        options,
        options.non_expert,
        stream,
        weights_stream,
    )
}

/// Loads LFM2 with expert caching and disk-streamed non-expert units.
pub fn load_lfm2_sparse_expert_cache_model_with_dense_layers(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    non_expert: crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Lfm2LayerwiseModel, Error> {
    load_lfm2_sparse_expert_cache_model_with_non_expert(
        model_dir,
        options,
        non_expert,
        stream,
        weights_stream,
    )
}

fn load_lfm2_sparse_expert_cache_model_with_non_expert(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    non_expert: impl Into<LayerExecutionLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Lfm2LayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = resident::get_model_args(model_dir)?;
    if !args.has_sparse_moe_layers() {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires an LFM2 MoE checkpoint".into(),
        ));
    }
    let mut adapter = Lfm2LayerwiseAdapter::new(args.clone(), stream)?;
    adapter.sparse_expert_cache = true;
    let mut execution =
        load_layerwise_model(model_dir, adapter, non_expert, stream, weights_stream)?;
    let store = execution.weight_store_arc();
    let entries = lfm2_expert_catalog(&args, store.as_ref())?;
    execution.adapter_mut().expert_cache = Some(ExpertCache::new_shared(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(Lfm2LayerwiseModel { execution })
}

/// Adapter shared by dense, MoE, attention, and short-convolution LFM2 blocks.
pub struct Lfm2LayerwiseAdapter {
    args: ModelArgs,
    embedding: MaybeQuantized<nn::Embedding>,
    norm: nn::RmsNorm,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_embedding: Option<VocabParallelEmbedding>,
    parallel_lm_head: Option<VocabParallelLmHead>,
    sparse_expert_cache: bool,
    expert_cache: Option<ExpertCache>,
}

impl Lfm2LayerwiseAdapter {
    /// Creates metadata-only pinned modules.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let embedding = common::linear::unloaded_maybe_quantized_embedding(
            args.vocab_size,
            args.hidden_size,
            args.weight_quantization_for("model.embed_tokens.weight"),
            stream,
        )?;
        let norm = nn::RmsNorm::unloaded(args.hidden_size, args.norm_eps, Dtype::Float32, stream)?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.vocab_size,
                false,
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

    /// Returns the validated model arguments.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    fn new_cache(&self) -> Cache {
        Cache::new(&self.args).expect("validated LFM2 layer schedule")
    }

    fn split_expert_recipes(
        &self,
        index: usize,
        store: &dyn WeightStore,
    ) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
        if self
            .args
            .layer_policy(index)
            .ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "LFM2 layer schedule has no policy for layer {index}"
                ))
            })?
            .feed_forward
            != FeedForwardPolicy::SparseMoe
        {
            return Ok(BTreeMap::new());
        }
        let runtime_prefix = format!("model.layers.{index}.feed_forward.experts");
        let keys = store.keys();
        if keys.contains(&format!("{runtime_prefix}.gate_up_proj")) {
            return Ok(BTreeMap::new());
        }
        if keys.contains(&format!("{runtime_prefix}.gate_proj"))
            && keys.contains(&format!("{runtime_prefix}.up_proj"))
        {
            let mut recipes = BTreeMap::from([(
                "feed_forward.experts.gate_up_proj".into(),
                DerivedWeightRecipe::Concatenate {
                    axis: 1,
                    inputs: vec![
                        DerivedWeightRecipe::source(
                            format!("{runtime_prefix}.gate_proj"),
                            TensorSelection::Full,
                        ),
                        DerivedWeightRecipe::source(
                            format!("{runtime_prefix}.up_proj"),
                            TensorSelection::Full,
                        ),
                    ],
                },
            )]);
            for suffix in ["_scales", "_biases"] {
                let gate = format!("{runtime_prefix}.gate_proj{suffix}");
                let up = format!("{runtime_prefix}.up_proj{suffix}");
                if keys.contains(&gate) && keys.contains(&up) {
                    recipes.insert(
                        format!("feed_forward.experts.gate_up_proj{suffix}"),
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
            return Ok(recipes);
        }
        let mut gate_up = Vec::with_capacity(self.args.num_experts as usize);
        let mut down = Vec::with_capacity(self.args.num_experts as usize);
        for expert in 0..self.args.num_experts {
            let gate = expert_source(store, &runtime_prefix, expert, &["w1", "gate_proj"])?;
            let up = expert_source(store, &runtime_prefix, expert, &["w3", "up_proj"])?;
            let down_source = expert_source(store, &runtime_prefix, expert, &["w2", "down_proj"])?;
            gate_up.push(DerivedWeightRecipe::Concatenate {
                axis: 0,
                inputs: vec![gate, up],
            });
            down.push(down_source);
        }
        Ok(BTreeMap::from([
            (
                "feed_forward.experts.gate_up_proj".into(),
                DerivedWeightRecipe::Stack {
                    axis: 0,
                    inputs: gate_up,
                },
            ),
            (
                "feed_forward.experts.down_proj".into(),
                DerivedWeightRecipe::Stack {
                    axis: 0,
                    inputs: down,
                },
            ),
        ]))
    }
}

fn expert_source(
    store: &dyn WeightStore,
    prefix: &str,
    expert: i32,
    projections: &[&str],
) -> Result<DerivedWeightRecipe, Error> {
    let keys = store.keys();
    let key = projections
        .iter()
        .map(|projection| format!("{prefix}.{expert}.{projection}.weight"))
        .find(|candidate| keys.contains(candidate))
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "LFM2 checkpoint is missing expert {expert} projection under {prefix}"
            ))
        })?;
    Ok(DerivedWeightRecipe::source(key, TensorSelection::Full))
}

/// Per-forward attention mask shared by the temporary hybrid blocks.
pub struct Lfm2ForwardContext {
    mask: Option<Array>,
}

struct OffsetOnlyCache(i32);

impl KeyValueCache for OffsetOnlyCache {
    fn offset(&self) -> i32 {
        self.0
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
        _stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        Ok((keys, values))
    }
}

impl ArchitectureAdapter for Lfm2LayerwiseAdapter {
    type Input<'a> = &'a Array;
    type Cache = Cache;
    type Layer = DecoderLayer;
    type ForwardContext = Lfm2ForwardContext;

    fn model_type(&self) -> &str {
        &self.args.model_type
    }

    fn quantization(&self) -> Option<crate::runtime::checkpoint::quantization::WeightQuantization> {
        self.args.weight_quantization()
    }

    fn prompt_cache_model_identity(
        &self,
        topology: Option<crate::ParallelTopology>,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let mut local = self.args.clone();
        if let Some(topology) = topology {
            local.num_attention_heads = exact_parallel_division(
                "LFM2 prompt-cache attention heads",
                local.num_attention_heads,
                topology.tensor_parallel_size,
            )?;
            local.num_key_value_heads = exact_parallel_division(
                "LFM2 prompt-cache KV heads",
                local.num_key_value_heads,
                topology.tensor_parallel_size,
            )?;
        }
        let layer_count = usize::try_from(self.args.num_hidden_layers)
            .map_err(|_| Exception::custom("invalid LFM2 cache layer count"))?;
        Ok(PromptCacheModelIdentity {
            model_family: "lfm2".into(),
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
            layer_layout: resident::prompt_cache_layer_layout(&local)
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
        let rank = descriptor.topology.cache_rank_identity();
        resident::Model::save_prompt_cache_with_rank(
            cache,
            destination,
            descriptor,
            prefix_token_ids,
            options,
            rank,
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
            identity.clone(),
            stream,
        )
        .map_err(Into::into)
    }

    fn static_units(&self, store: &dyn WeightStore) -> Result<Vec<StaticUnitBindings>, Error> {
        let mut units = vec![
            StaticUnitBindings::new(
                EMBEDDING_UNIT,
                build_module_bindings(&self.embedding, "model.embed_tokens", store)?,
            )?,
            StaticUnitBindings::new(
                NORM_UNIT,
                build_module_bindings(&self.norm, "model.embedding_norm", store)?,
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
                "LFM2 adapter received {} static leases, expected {expected}",
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
        } else if let Some(head) = &mut self.lm_head {
            populate_module_from_lease(head, &leases[2])?;
        }
        Ok(())
    }

    fn validate_cache(&self, cache: &mut Cache) -> Result<(), Error> {
        if cache.layers.is_empty() {
            *cache = self.new_cache();
            return Ok(());
        }
        if cache.layers.len() != self.args.num_hidden_layers as usize {
            return Err(Error::UnsupportedArchitecture(format!(
                "LFM2 cache has {} layers, expected {}",
                cache.layers.len(),
                self.args.num_hidden_layers
            )));
        }
        for (index, layer_cache) in cache.layers.iter().enumerate() {
            let policy = self.args.layer_schedule.get(index).ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "LFM2 layer schedule has no policy for layer {index}"
                ))
            })?;
            let matches = matches!(
                (policy.operator, layer_cache),
                (OperatorPolicy::CausalConvolution, LayerCache::Conv(_))
                    | (OperatorPolicy::SelfAttention(_), LayerCache::Attention(_))
            );
            if !matches {
                return Err(Error::UnsupportedArchitecture(format!(
                    "LFM2 cache kind does not match the layer schedule at layer {index}"
                )));
            }
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
        let mask = if hidden.dim(1) > 1 {
            let offset_cache = vec![Some(OffsetOnlyCache(cache.offset()))];
            match create_attention_mask(&hidden, &offset_cache, Some(true), stream)? {
                Some(AttentionMask::Array(mask)) => Some(mask),
                Some(AttentionMask::Causal) => {
                    return Err(Error::UnsupportedArchitecture(
                        "LFM2 requires an array causal mask".into(),
                    ));
                }
                None => None,
            }
        } else {
            None
        };
        Ok(LayerwiseForwardState {
            hidden,
            context: Lfm2ForwardContext { mask },
        })
    }
    fn begin_forward_with_execution<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Self::Cache,
        execution: &crate::runtime::distributed::parallel::ParallelExecutionContext<'_>,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        let Some(v) = &mut self.parallel_embedding else {
            return self.begin_forward(input, cache, execution.stream());
        };
        let hidden = v.forward(input, execution)?;
        let mask = if hidden.dim(1) > 1 {
            let offset_cache = vec![Some(OffsetOnlyCache(cache.offset()))];
            match create_attention_mask(&hidden, &offset_cache, Some(true), execution.stream())? {
                Some(AttentionMask::Array(v)) => Some(v),
                Some(AttentionMask::Causal) => {
                    return Err(Error::UnsupportedArchitecture(
                        "LFM2 requires an array causal mask".into(),
                    ))
                }
                None => None,
            }
        } else {
            None
        };
        Ok(LayerwiseForwardState {
            hidden,
            context: Lfm2ForwardContext { mask },
        })
    }

    fn execution_group_count(&self) -> usize {
        1
    }

    fn execution_group_id(&self, group: usize) -> Result<String, Error> {
        if group == 0 {
            Ok("text_decoder".into())
        } else {
            Err(Error::UnsupportedArchitecture(format!(
                "LFM2 has no execution group {group}"
            )))
        }
    }

    fn layer_count(&self, group: usize) -> Result<usize, Error> {
        if group == 0 {
            Ok(self.args.num_hidden_layers as usize)
        } else {
            Err(Error::UnsupportedArchitecture(format!(
                "LFM2 has no execution group {group}"
            )))
        }
    }

    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        self.layer_count(group)?;
        let index = i32::try_from(index)
            .map_err(|_| Error::UnsupportedArchitecture("LFM2 layer index exceeds i32".into()))?;
        DecoderLayer::new(&self.args, index, stream)
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
            "model.embedding_norm",
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
            let layer = DecoderLayer::new(&self.args, index as i32, stream)?;
            crate::architectures::distributed::tensor::insert_lfm2_layer_plan(
                planner, &layer, index, false,
            )?;
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
        self.layer_count(group)?;
        let prefix = format!("model.layers.{index}");
        let find = |n: &str| {
            layout
                .tensor(&format!("{prefix}.{n}.weight"))
                .or_else(|| layout.tensor(&format!("{prefix}.{n}.inner.weight")))
        };
        let mut args = self.args.clone();
        let head_dim = self.args.hidden_size / self.args.num_attention_heads;
        if let Some(q) = find("self_attn.q_proj") {
            args.num_attention_heads = q.local_shape()[0] as i32 / head_dim;
        }
        if let Some(k) = find("self_attn.k_proj") {
            args.num_key_value_heads = k.local_shape()[0] as i32 / head_dim;
        }
        let dense = find("feed_forward.w1").map_or(args.dense_layer_intermediate_size(), |v| {
            v.local_shape()[0] as i32
        });
        let moe = layout
            .tensor(&format!("{prefix}.feed_forward.experts.gate_up_proj"))
            .map_or(args.moe_intermediate_size, |v| {
                v.local_shape()[1] as i32 / 2
            });
        DecoderLayer::new_with_widths(&args, index as i32, dense, moe, Some(head_dim), stream)
    }

    fn layer_checkpoint_prefix(&self, _group: usize, index: usize) -> String {
        format!("model.layers.{index}")
    }

    fn layer_unit_name(&self, _group: usize, index: usize) -> String {
        format!("lfm2.layer.{index:05}")
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
                |name| name.starts_with("feed_forward.experts."),
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
        let mut recipes = self.split_expert_recipes(index, store)?;
        let conv_key = format!("{prefix}.conv.conv.weight");
        if store
            .metadata(&conv_key)
            .is_ok_and(|metadata| metadata.shape.len() == 2)
        {
            if let Some(parameter) = layer.parameters().flatten().get("conv.conv.weight") {
                recipes.insert(
                    "conv.conv.weight".into(),
                    DerivedWeightRecipe::Reshape {
                        input: Box::new(DerivedWeightRecipe::source(
                            conv_key,
                            TensorSelection::Full,
                        )),
                        shape: parameter
                            .shape()
                            .iter()
                            .map(|dimension| *dimension as usize)
                            .collect(),
                    },
                );
            }
        }
        let bindings = build_module_bindings_with_recipes(layer, &prefix, store, recipes)?;
        Ok(if self.sparse_expert_cache {
            bindings
                .into_iter()
                .filter(|binding| !binding.name().starts_with("feed_forward.experts."))
                .collect()
        } else {
            bindings
        })
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
            layout,
        )
    }

    fn additional_consumed_checkpoint_keys(&self, store: &dyn WeightStore) -> Vec<String> {
        if self.sparse_expert_cache {
            store
                .keys()
                .into_iter()
                .filter(|key| key.contains(".feed_forward.experts."))
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
        self.layer_count(group)?;
        let policy = self.args.layer_policy(index).ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "LFM2 layer schedule has no policy for layer {index}"
            ))
        })?;
        if self.sparse_expert_cache && policy.feed_forward == FeedForwardPolicy::SparseMoe {
            let expert_cache = self.expert_cache.as_ref().ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "LFM2 sparse expert cache was not initialized".into(),
                )
            })?;
            let pass = if hidden.dim(1) > 1 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            };
            return Ok(layer.forward_with_expert_executor(
                hidden,
                context.mask.as_ref(),
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
                                let started = Instant::now();
                                let prefix = format!("model.layers.{index}.feed_forward.experts");
                                let mut bank = PackedSwiGluExperts::new(
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
                                bank.down_proj = Param::new(
                                    acquired
                                        .compact_binding("down_proj", stream)
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
            hidden,
            context.mask.as_ref(),
            Some(&mut cache.layers[index]),
            stream,
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

pub(crate) fn lfm2_expert_catalog(
    args: &ModelArgs,
    store: &dyn WeightStore,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let keys = store
        .keys()
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    let mut entries = Vec::new();
    for (layer, _) in args
        .layer_schedule
        .iter()
        .enumerate()
        .filter(|(_, policy)| policy.feed_forward == FeedForwardPolicy::SparseMoe)
    {
        let prefix = format!("model.layers.{layer}.feed_forward.experts");
        let packed_gate_up = format!("{prefix}.gate_up_proj");
        let packed_down = format!("{prefix}.down_proj");
        for expert in 0..args.num_experts as usize {
            let identity = ExpertIdentity::new(layer, expert);
            let mut bindings = Vec::new();
            if keys.contains(&packed_gate_up) && keys.contains(&packed_down) {
                for (name, key) in [
                    ("gate_up_proj", &packed_gate_up),
                    ("down_proj", &packed_down),
                ] {
                    let recipe = DerivedWeightRecipe::source(
                        key.clone(),
                        TensorSelection::Range {
                            axis: 0,
                            start: expert,
                            end: expert + 1,
                        },
                    );
                    let bytes = recipe.infer(store)?.byte_len();
                    bindings.push(WeightBinding::from_recipe(name, recipe, bytes)?);
                }
                for (name, key) in [
                    ("gate_up_proj_scales", format!("{packed_gate_up}_scales")),
                    ("gate_up_proj_biases", format!("{packed_gate_up}_biases")),
                    ("down_proj_scales", format!("{packed_down}_scales")),
                    ("down_proj_biases", format!("{packed_down}_biases")),
                ] {
                    if keys.contains(&key) {
                        let recipe = DerivedWeightRecipe::source(
                            key,
                            TensorSelection::Range {
                                axis: 0,
                                start: expert,
                                end: expert + 1,
                            },
                        );
                        let bytes = recipe.infer(store)?.byte_len();
                        bindings.push(WeightBinding::from_recipe(name, recipe, bytes)?);
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
                for (name, recipe) in [
                    (
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
                    ),
                    (
                        "down_proj",
                        DerivedWeightRecipe::source(packed_down.clone(), selection.clone()),
                    ),
                ] {
                    let bytes = recipe.infer(store)?.byte_len();
                    bindings.push(WeightBinding::from_recipe(name, recipe, bytes)?);
                }
                for suffix in ["_scales", "_biases"] {
                    let gate = format!("{prefix}.gate_proj{suffix}");
                    let up = format!("{prefix}.up_proj{suffix}");
                    if keys.contains(&gate) && keys.contains(&up) {
                        let recipe = DerivedWeightRecipe::Concatenate {
                            axis: 1,
                            inputs: vec![
                                DerivedWeightRecipe::source(gate, selection.clone()),
                                DerivedWeightRecipe::source(up, selection.clone()),
                            ],
                        };
                        let bytes = recipe.infer(store)?.byte_len();
                        bindings.push(WeightBinding::from_recipe(
                            format!("gate_up_proj{suffix}"),
                            recipe,
                            bytes,
                        )?);
                    }
                    let down = format!("{packed_down}{suffix}");
                    if keys.contains(&down) {
                        let recipe = DerivedWeightRecipe::source(down, selection.clone());
                        let bytes = recipe.infer(store)?.byte_len();
                        bindings.push(WeightBinding::from_recipe(
                            format!("down_proj{suffix}"),
                            recipe,
                            bytes,
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
                        "split LFM2 experts cannot be lazily load-time quantized; use checkpoint-native packed expert weights"
                            .into(),
                    ));
                }
                let gate = expert_source(store, &prefix, expert as i32, &["w1", "gate_proj"])?;
                let up = expert_source(store, &prefix, expert as i32, &["w3", "up_proj"])?;
                let down = expert_source(store, &prefix, expert as i32, &["w2", "down_proj"])?;
                for (name, recipe) in [
                    (
                        "gate_up_proj",
                        DerivedWeightRecipe::Stack {
                            axis: 0,
                            inputs: vec![DerivedWeightRecipe::Concatenate {
                                axis: 0,
                                inputs: vec![gate, up],
                            }],
                        },
                    ),
                    (
                        "down_proj",
                        DerivedWeightRecipe::Stack {
                            axis: 0,
                            inputs: vec![down],
                        },
                    ),
                ] {
                    let bytes = recipe.infer(store)?.byte_len();
                    bindings.push(WeightBinding::from_recipe(name, recipe, bytes)?);
                }
            }
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    Error::UnsupportedArchitecture("LFM2 expert byte total overflowed".into())
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

/// LFM2 token generation iterator using bounded layer execution.
pub type Generate<'a, S = crate::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, Lfm2LayerwiseModel, Cache, S>;

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use safemlx::{
        module::ModuleParameters,
        ops::{indexing::TryIndexOp, ones_dtype, zeros_dtype},
        Array, Device, DeviceType, ExecutionContext, Stream,
    };

    use super::{
        load_lfm2_layerwise_model, load_lfm2_sparse_expert_cache_model, Lfm2LayerwiseAdapter,
        Lfm2LayerwiseModel,
    };
    use crate::{
        architectures::lfm2::model::{
            self as resident, Cache, FeedForwardPolicy, LayerCache, LayerPolicy, Model, ModelArgs,
            OperatorPolicy,
        },
        runtime::residency::dense_stream::DenseDiskStreamLoadOptions,
        runtime::residency::expert_cache::ExpertCacheLoadOptions,
        runtime::residency::policy::{MemoryTier, OffloadConfig, ResidencyPolicy},
        runtime::{
            attention::LayerSchedule,
            execution::layerwise::{
                load_layerwise_model, LayerExecutionLoadOptions, LayerwiseLoadOptions,
            },
        },
    };

    fn args(moe: bool) -> ModelArgs {
        resident::model_args_from_config_value(&serde_json::json!({
            "model_type": if moe { "lfm2_moe" } else { "lfm2" },
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "norm_eps": 1e-5,
            "layer_types": ["conv", "full_attention", "conv"],
            "conv_L_cache": 3,
            "conv_bias": false,
            "block_auto_adjust_ff_dim": false,
            "tie_word_embeddings": false,
            "moe_intermediate_size": if moe { 8 } else { 0 },
            "num_dense_layers": if moe { 1 } else { 0 },
            "num_experts": if moe { 2 } else { 0 },
            "num_experts_per_tok": if moe { 1 } else { 0 },
            "norm_topk_prob": moe,
            "use_expert_bias": moe
        }))
        .unwrap()
    }

    fn initialize(model: &mut Model, stream: &Stream) {
        for (name, parameter) in model.parameters_mut().flatten() {
            let shape = parameter.shape().to_vec();
            let dtype = parameter.dtype();
            *parameter = if name.ends_with("norm.weight") {
                ones_dtype(&shape, dtype, stream).unwrap()
            } else {
                zeros_dtype(&shape, dtype, stream).unwrap()
            };
        }
    }

    fn write_fixture(dir: &Path, model: &Model, stream: &Stream) {
        let params = model.parameters().flatten();
        let mut arrays = Vec::<(String, Array)>::new();
        for (name, value) in params {
            let name = crate::runtime::checkpoint::binding::canonical_checkpoint_name(&name);
            if name.ends_with("feed_forward.experts.gate_up_proj") {
                let prefix = name.trim_end_matches(".gate_up_proj");
                for expert in 0..model.args.num_experts {
                    arrays.push((
                        format!("{prefix}.{expert}.w1.weight"),
                        value
                            .try_index_device(
                                (expert, ..model.args.moe_intermediate_size, ..),
                                stream,
                            )
                            .unwrap(),
                    ));
                    arrays.push((
                        format!("{prefix}.{expert}.w3.weight"),
                        value
                            .try_index_device(
                                (expert, model.args.moe_intermediate_size.., ..),
                                stream,
                            )
                            .unwrap(),
                    ));
                }
            } else if name.ends_with("feed_forward.experts.down_proj") {
                let prefix = name.trim_end_matches(".down_proj");
                for expert in 0..model.args.num_experts {
                    arrays.push((
                        format!("{prefix}.{expert}.w2.weight"),
                        value.try_index_device((expert, .., ..), stream).unwrap(),
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
        let num_dense_layers = if model.args.model_type == "lfm2_moe" {
            model
                .args
                .layer_schedule
                .iter()
                .position(|policy| policy.feed_forward == resident::FeedForwardPolicy::SparseMoe)
                .unwrap_or(model.args.layer_schedule.len())
        } else {
            0
        };
        fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&serde_json::json!({
                "model_type": model.args.model_type,
                "vocab_size": model.args.vocab_size,
                "hidden_size": model.args.hidden_size,
                "intermediate_size": model.args.intermediate_size,
                "num_hidden_layers": model.args.num_hidden_layers,
                "num_attention_heads": model.args.num_attention_heads,
                "num_key_value_heads": model.args.num_key_value_heads,
                "max_position_embeddings": model.args.max_position_embeddings,
                "norm_eps": model.args.norm_eps,
                "layer_types": model.args.layer_schedule.iter().map(|policy| match policy.operator {
                    resident::OperatorPolicy::CausalConvolution => "conv",
                    resident::OperatorPolicy::SelfAttention(crate::AttentionPolicy::Full) => "full_attention",
                    resident::OperatorPolicy::SelfAttention(crate::AttentionPolicy::Sliding { .. }) => unreachable!("validated LFM2 test schedule"),
                }).collect::<Vec<_>>(),
                "conv_L_cache": model.args.conv_l_cache,
                "conv_bias": model.args.conv_bias,
                "block_auto_adjust_ff_dim": false,
                "tie_word_embeddings": model.args.tie_word_embeddings,
                "moe_intermediate_size": model.args.moe_intermediate_size,
                "num_dense_layers": num_dense_layers,
                "num_experts": model.args.num_experts,
                "num_experts_per_tok": model.args.num_experts_per_tok,
                "norm_topk_prob": model.args.norm_topk_prob,
                "use_expert_bias": model.args.use_expert_bias
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

    fn parity(moe: bool, depth: usize, dense_stream: bool) {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut fixture = Model::new(args(moe), gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, gpu.stream());

        let mut resident = resident::load_model(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let options = if dense_stream {
            LayerExecutionLoadOptions::DenseDiskStream(
                DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, depth, depth, depth).unwrap(),
            )
        } else {
            LayerExecutionLoadOptions::LayerwiseHost(LayerwiseLoadOptions::new(
                OffloadConfig::new(None, None, depth).unwrap(),
            ))
        };
        let mut layerwise =
            load_lfm2_layerwise_model(dir.path(), options, gpu.stream(), cpu.stream()).unwrap();
        if dense_stream {
            let report = layerwise.dense_stream_report().unwrap().unwrap();
            assert!(report
                .residency()
                .units()
                .iter()
                .filter(|unit| unit.id().as_str().starts_with("lfm2.layer."))
                .all(|unit| {
                    unit.planned_tier() == MemoryTier::Disk
                        && !unit.host_resident()
                        && !unit.device_resident()
                }));
        }
        let mut resident_cache = resident.new_cache();
        let mut layerwise_cache = Cache { layers: Vec::new() };
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
            Array::from_slice(&[4u32], &[1, 1]),
            Array::from_slice(&[5u32], &[1, 1]),
        ] {
            let expected = resident
                .forward_logits(&tokens, Some(&mut resident_cache), false, gpu.stream())
                .unwrap();
            let actual = layerwise
                .forward(&tokens, &mut layerwise_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected);
            for (expected, actual) in resident_cache.layers.iter().zip(&layerwise_cache.layers) {
                let expected_offset = match expected {
                    LayerCache::Attention(cache) => {
                        crate::runtime::cache::KeyValueCache::offset(cache)
                    }
                    LayerCache::Conv(cache) => cache.offset,
                };
                let actual_offset = match actual {
                    LayerCache::Attention(cache) => {
                        crate::runtime::cache::KeyValueCache::offset(cache)
                    }
                    LayerCache::Conv(cache) => cache.offset,
                };
                assert_eq!(expected_offset, actual_offset);
            }
            if !dense_stream {
                let report = layerwise.residency_report().unwrap();
                let layers = report
                    .units()
                    .iter()
                    .filter(|unit| unit.id().as_str().starts_with("lfm2.layer."))
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
    }

    #[test]
    fn lfm2_dense_hybrid_layerwise_prefill_and_cached_decode_parity() {
        parity(false, 1, false);
        parity(false, 2, false);
    }

    #[test]
    fn lfm2_split_moe_hybrid_layerwise_prefill_and_cached_decode_parity() {
        parity(true, 1, false);
    }

    #[test]
    fn arbitrary_feed_forward_schedule_matches_resident_execution() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut custom_args = args(true);
        custom_args.layer_schedule = LayerSchedule::new(
            3,
            vec![
                LayerPolicy {
                    operator: OperatorPolicy::CausalConvolution,
                    feed_forward: FeedForwardPolicy::SparseMoe,
                },
                LayerPolicy {
                    operator: OperatorPolicy::SelfAttention(crate::AttentionPolicy::Full),
                    feed_forward: FeedForwardPolicy::Dense,
                },
                LayerPolicy {
                    operator: OperatorPolicy::CausalConvolution,
                    feed_forward: FeedForwardPolicy::SparseMoe,
                },
            ],
        )
        .unwrap();
        let mut resident = Model::new(custom_args.clone(), gpu.stream()).unwrap();
        assert!(resident.model.layers[0].feed_forward.is_moe);
        assert!(!resident.model.layers[1].feed_forward.is_moe);
        assert!(resident.model.layers[2].feed_forward.is_moe);
        initialize(&mut resident, gpu.stream());
        let directory = tempfile::tempdir().unwrap();
        write_fixture(directory.path(), &resident, gpu.stream());

        let adapter = Lfm2LayerwiseAdapter::new(custom_args, gpu.stream()).unwrap();
        let execution = load_layerwise_model(
            directory.path(),
            adapter,
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut layerwise = Lfm2LayerwiseModel { execution };
        let mut resident_cache = resident.new_cache();
        let mut layerwise_cache = Cache { layers: Vec::new() };
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
        ] {
            let expected = resident
                .forward_logits(&tokens, Some(&mut resident_cache), false, gpu.stream())
                .unwrap();
            let actual = layerwise
                .forward(&tokens, &mut layerwise_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected);
        }
        assert_eq!(resident_cache.offset(), 3);
        assert_eq!(layerwise_cache.offset(), 3);
    }

    #[test]
    fn lfm2_dense_stream_hybrid_prefill_and_cached_decode_parity() {
        parity(false, 1, true);
    }

    #[test]
    fn lfm2_sparse_expert_cache_prefill_and_decode_parity() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut fixture = Model::new(args(true), gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture, gpu.stream());
        let mut resident = resident::load_model(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let options = ExpertCacheLoadOptions::new(
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
            OffloadConfig::new(None, None, 1).unwrap(),
            1 << 20,
            1,
        )
        .unwrap();
        let mut cached =
            load_lfm2_sparse_expert_cache_model(dir.path(), options, gpu.stream(), cpu.stream())
                .unwrap();
        let mut resident_cache = resident.new_cache();
        let mut cached_cache = Cache { layers: Vec::new() };
        for tokens in [
            Array::from_slice(&[1u32, 2], &[1, 2]),
            Array::from_slice(&[3u32], &[1, 1]),
        ] {
            let expected = resident
                .forward_logits(&tokens, Some(&mut resident_cache), false, gpu.stream())
                .unwrap();
            let actual = cached
                .forward(&tokens, &mut cached_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected);
        }
        let report = cached.expert_cache_report().unwrap().unwrap();
        assert_eq!(report.owned_experts, 4);
        assert!(report.prefill.requested_routes > 0);
        assert!(report.decode.requested_routes > 0);
        assert!(report.prefill.compact_banks > 1);
        crate::architectures::distributed::expert::assert_rank_owned_sparse_ep_load(
            dir.path(),
            options,
            crate::api::ModelKind::Lfm2,
            report.owned_experts / 2,
            gpu.stream(),
            cpu.stream(),
        );
    }
}

//! Bounded-residency execution for Kimi Linear safetensors and GGUF checkpoints.

use std::{collections::BTreeMap, path::Path, sync::Arc, time::Instant};

use safemlx::{
    error::Exception,
    module::{Module, ModuleParameters, Param},
    nn,
    ops::{indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};

use crate::{
    api::input,
    error::Error,
    nn::{
        generation::CausalLm,
        linear::{
            project_logits_maybe_quantized, unloaded_maybe_quantized_embedding,
            unloaded_maybe_quantized_linear,
        },
        tensor::create_causal_mask,
    },
    runtime::{
        checkpoint::{
            binding::{
                build_module_bindings_with_recipes, canonical_checkpoint_name,
                materialize_module_bindings, populate_module_from_arrays_excluding,
                populate_module_from_lease, populate_module_from_lease_excluding,
            },
            recipe::DerivedWeightRecipe,
            store::{GgufWeightStore, TensorSelection, WeightStore, WeightStoreBackend},
        },
        execution::layerwise::{
            load_general_layerwise_model, load_general_layerwise_model_with_store,
            GeneralLayerwiseModel, GeneralLayerwiseModelAdapter, LayerExecutionLoadOptions,
            LayerwiseForwardState, StaticUnitBindings, WeightResidency,
        },
        residency::{
            expert_cache::{
                ExpertCache, ExpertCacheLoadOptions, ExpertCacheReport, ExpertCatalogEntry,
                ExpertIdentity, ExpertPass,
            },
            manager::{OffloadUnit, ResidencyReport, ResidentUnitLease, WeightBinding},
        },
    },
};

use super::model::{self as resident, Cache, DecoderLayer, FeedForwardPolicy, ModelArgs};

const EMBEDDING_UNIT: &str = "kimi_linear.static.embedding";
const NORM_UNIT: &str = "kimi_linear.static.norm";
const HEAD_UNIT: &str = "kimi_linear.static.output";

/// Kimi Linear causal LM with a bounded decoder-layer window.
pub struct KimiLinearLayerwiseModel {
    execution: GeneralLayerwiseModel<KimiLinearLayerwiseAdapter>,
}

impl KimiLinearLayerwiseModel {
    /// Returns the validated architecture configuration.
    pub fn args(&self) -> &ModelArgs {
        self.execution.adapter().args()
    }

    /// Creates an empty heterogeneous KDA/MLA cache.
    pub fn new_cache(&self) -> Cache {
        Cache::new(self.args())
    }

    /// Returns current weight-residency telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        self.execution.residency_report()
    }

    /// Returns disk-stream telemetry when that policy is active.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<crate::runtime::execution::layerwise::DenseDiskStreamReport>, Error> {
        self.execution.dense_stream_report()
    }

    /// Returns sparse routed-expert cache telemetry when enabled.
    pub fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        self.execution
            .adapter()
            .expert_cache
            .as_ref()
            .map(ExpertCache::report)
            .transpose()
            .map_err(Error::from)
    }

    /// Runs the embedding, hybrid decoder stack, norm, and output projection.
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

    /// Evicts temporary decoder layers from the execution device.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        self.execution.clear_device_group("text_decoder")
    }
}

impl CausalLm<Cache> for KimiLinearLayerwiseModel {
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

/// Loads Kimi Linear through the shared bounded host/disk execution engine.
pub fn load_kimi_linear_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerExecutionLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<KimiLinearLayerwiseModel, Error> {
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
        crate::api::ModelKind::KimiLinear,
        model_dir,
        crate::api::ModelLoadOptions::default().with_weight_residency(residency),
    )?;
    let args = resident::get_model_args(model_dir)?;
    args.validate()?;
    let adapter = KimiLinearLayerwiseAdapter::new(args, stream)?;
    Ok(KimiLinearLayerwiseModel {
        execution: load_general_layerwise_model(
            model_dir,
            adapter,
            options,
            stream,
            weights_stream,
        )?,
    })
}

pub(crate) fn load_kimi_linear_gguf_layerwise_model(
    checkpoint: &GgufCheckpoint,
    metadata: &std::collections::HashMap<String, GgufMetadataValue>,
    residency: WeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(KimiLinearLayerwiseModel, Vec<u32>), Error> {
    let prepared = resident::prepare_gguf_checkpoint(checkpoint, metadata, None, weights_stream)?;
    let args = prepared.args;
    let store: Arc<dyn WeightStore + Send + Sync> =
        Arc::new(GgufWeightStore::new_with_max_mapped_shards(
            checkpoint.clone(),
            resident::translate_gguf_weight_name,
            residency.max_mapped_shards(),
        )?);
    let execution = match residency {
        WeightResidency::LayerwiseHost(options) => load_general_layerwise_model_with_store(
            store,
            KimiLinearLayerwiseAdapter::new(args, stream)?,
            options,
            stream,
            weights_stream,
        )?,
        WeightResidency::DenseDiskStream(options) => load_general_layerwise_model_with_store(
            store,
            KimiLinearLayerwiseAdapter::new(args, stream)?,
            options,
            stream,
            weights_stream,
        )?,
        WeightResidency::SparseExpertCache(options) => {
            return Ok((
                load_kimi_linear_sparse_with_store(
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
                load_kimi_linear_sparse_with_store(
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
                "the bounded Kimi Linear GGUF loader does not accept fully resident policy".into(),
            ));
        }
    };
    Ok((
        KimiLinearLayerwiseModel { execution },
        prepared.eos_token_ids,
    ))
}

/// Loads only the replicated Kimi Linear GGUF parameters needed by sparse
/// expert-parallel execution and returns the shared lazy checkpoint store.
pub(crate) fn load_kimi_linear_gguf_sparse_ep_base(
    checkpoint: &GgufCheckpoint,
    metadata: &std::collections::HashMap<String, GgufMetadataValue>,
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
    let adapter = KimiLinearLayerwiseAdapter::new_sparse(args.clone(), stream)?;
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

    if let Some(head) = &mut model.lm_head {
        let bindings =
            build_module_bindings_with_recipes(head, "lm_head", store.as_ref(), BTreeMap::new())?;
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

/// Loads Kimi Linear with independently cached routed experts.
pub fn load_kimi_linear_sparse_expert_cache_model(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<KimiLinearLayerwiseModel, Error> {
    load_kimi_linear_sparse_expert_cache_model_with_non_expert(
        model_dir,
        options,
        options.non_expert,
        stream,
        weights_stream,
    )
}

/// Loads Kimi Linear with expert caching and disk-streamed nonexpert layers.
pub fn load_kimi_linear_sparse_expert_cache_model_with_dense_layers(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    non_expert: crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<KimiLinearLayerwiseModel, Error> {
    load_kimi_linear_sparse_expert_cache_model_with_non_expert(
        model_dir,
        options,
        non_expert,
        stream,
        weights_stream,
    )
}

fn load_kimi_linear_sparse_expert_cache_model_with_non_expert(
    model_dir: impl AsRef<Path>,
    options: ExpertCacheLoadOptions,
    non_expert: impl Into<LayerExecutionLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<KimiLinearLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    crate::api::structural::validate_safetensors_load_path(
        crate::api::ModelKind::KimiLinear,
        model_dir,
        crate::api::ModelLoadOptions::default(),
    )?;
    let args = resident::get_model_args(model_dir)?;
    args.validate()?;
    let adapter = KimiLinearLayerwiseAdapter::new_sparse(args.clone(), stream)?;
    let mut execution =
        load_general_layerwise_model(model_dir, adapter, non_expert, stream, weights_stream)?;
    let store = execution.weight_store_arc();
    let entries = kimi_expert_catalog(&args, store.as_ref())?;
    execution.adapter_mut().expert_cache = Some(ExpertCache::new_shared(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(KimiLinearLayerwiseModel { execution })
}

fn load_kimi_linear_sparse_with_store(
    store: Arc<dyn WeightStore + Send + Sync>,
    args: ModelArgs,
    options: ExpertCacheLoadOptions,
    non_expert: impl Into<LayerExecutionLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<KimiLinearLayerwiseModel, Error> {
    let adapter = KimiLinearLayerwiseAdapter::new_sparse(args.clone(), stream)?;
    let mut execution = load_general_layerwise_model_with_store(
        store.clone(),
        adapter,
        non_expert,
        stream,
        weights_stream,
    )?;
    let entries = kimi_expert_catalog(&args, store.as_ref())?;
    execution.adapter_mut().expert_cache = Some(ExpertCache::new_shared(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(KimiLinearLayerwiseModel { execution })
}

/// Builds the streamed nonexpert Kimi execution base used by distributed EP.
pub(crate) fn load_kimi_linear_sparse_ep_base_with_store(
    store: Arc<dyn WeightStore + Send + Sync>,
    args: ModelArgs,
    non_expert: crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<KimiLinearLayerwiseModel, Error> {
    args.validate()?;
    let adapter = KimiLinearLayerwiseAdapter::new_sparse(args, stream)?;
    let execution = load_general_layerwise_model_with_store(
        store,
        adapter,
        non_expert,
        stream,
        weights_stream,
    )?;
    Ok(KimiLinearLayerwiseModel { execution })
}

/// Adapter for Kimi's heterogeneous KDA/MLA decoder layers.
pub struct KimiLinearLayerwiseAdapter {
    args: ModelArgs,
    embedding: MaybeQuantized<nn::Embedding>,
    norm: nn::RmsNorm,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    sparse_expert_cache: bool,
    expert_cache: Option<ExpertCache>,
}

impl KimiLinearLayerwiseAdapter {
    fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.vocab_size,
                false,
                args.weight_quantization_for("lm_head.weight"),
                stream,
            )?)
        };
        Ok(Self {
            embedding: unloaded_maybe_quantized_embedding(
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
            lm_head,
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

    /// Returns the architecture configuration.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
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
        let gguf = store.backend() == WeightStoreBackend::Gguf;
        let mut recipes = BTreeMap::new();

        for local_name in layer.parameters().flatten().keys() {
            let destination = format!("{prefix}.{local_name}");
            let canonical = canonical_checkpoint_name(&destination);
            if let Some(raw) = normalized.get(&canonical) {
                let mut recipe = DerivedWeightRecipe::source(raw.clone(), TensorSelection::Full);
                if local_name.ends_with("q_conv1d.weight")
                    || local_name.ends_with("k_conv1d.weight")
                    || local_name.ends_with("v_conv1d.weight")
                {
                    recipe = DerivedWeightRecipe::Reshape {
                        input: Box::new(recipe),
                        shape: vec![
                            (self.args.kda_config.num_heads * self.args.kda_config.head_dim)
                                as usize,
                            1,
                            self.args.kda_config.short_conv_kernel_size as usize,
                        ],
                    };
                } else if local_name.ends_with("A_log") {
                    recipe = DerivedWeightRecipe::Reshape {
                        input: Box::new(recipe),
                        shape: vec![1, 1, self.args.kda_config.num_heads as usize, 1],
                    };
                    if gguf {
                        recipe = DerivedWeightRecipe::NegLog {
                            input: Box::new(recipe),
                        };
                    }
                } else if keys.contains(&destination) || keys.contains(&canonical) {
                    continue;
                }
                recipes.insert(local_name.to_string(), recipe);
                continue;
            }
            if let Some(recipe) =
                self.split_expert_recipe(local_name.as_ref(), &prefix, &normalized)?
            {
                recipes.insert(local_name.to_string(), recipe);
                continue;
            }
            return Err(Error::UnsupportedArchitecture(format!(
                "Kimi Linear checkpoint is missing runtime parameter {canonical}"
            )));
        }
        Ok(recipes)
    }

    fn split_expert_recipe(
        &self,
        local_name: &str,
        prefix: &str,
        normalized: &BTreeMap<String, String>,
    ) -> Result<Option<DerivedWeightRecipe>, Error> {
        let component = if local_name == "mlp.experts.gate_up_proj" {
            Some(("gate_up", "weight", ""))
        } else if local_name == "mlp.experts.gate_up_proj_scales" {
            Some(("gate_up", "scales", "_scales"))
        } else if local_name == "mlp.experts.gate_up_proj_biases" {
            Some(("gate_up", "biases", "_biases"))
        } else if local_name == "mlp.experts.down_proj" {
            Some(("down", "weight", ""))
        } else {
            None
        };
        let Some((projection, checkpoint_component, packed_suffix)) = component else {
            return Ok(None);
        };
        if projection == "gate_up" {
            let gate = normalized.get(&format!("{prefix}.mlp.experts.gate_proj{packed_suffix}"));
            let up = normalized.get(&format!("{prefix}.mlp.experts.up_proj{packed_suffix}"));
            match (gate, up) {
                (Some(gate), Some(up)) => {
                    return Ok(Some(DerivedWeightRecipe::Concatenate {
                        axis: 1,
                        inputs: vec![
                            DerivedWeightRecipe::source(gate.clone(), TensorSelection::Full),
                            DerivedWeightRecipe::source(up.clone(), TensorSelection::Full),
                        ],
                    }));
                }
                (None, None) => {}
                _ => {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "Kimi Linear checkpoint layer {prefix} has mismatched packed gate/up expert tensors"
                    )));
                }
            }
        }
        if checkpoint_component != "weight" {
            return Ok(None);
        }
        let mut experts = Vec::with_capacity(self.args.num_experts as usize);
        for expert in 0..self.args.num_experts {
            let source = |name: &str| -> Result<DerivedWeightRecipe, Error> {
                let runtime =
                    format!("{prefix}.mlp.experts.{expert}.{name}.{checkpoint_component}");
                let raw = normalized.get(&runtime).ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "Kimi Linear checkpoint is missing expert tensor {runtime}"
                    ))
                })?;
                Ok(DerivedWeightRecipe::source(
                    raw.clone(),
                    TensorSelection::Full,
                ))
            };
            experts.push(if projection == "gate_up" {
                DerivedWeightRecipe::Concatenate {
                    axis: 0,
                    inputs: vec![source("w1")?, source("w3")?],
                }
            } else {
                source("w2")?
            });
        }
        Ok(Some(DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: experts,
        }))
    }
}

fn normalized_checkpoint_keys(store: &dyn WeightStore) -> BTreeMap<String, String> {
    store
        .keys()
        .into_iter()
        .map(|raw| {
            let runtime = canonical_checkpoint_name(&raw).replace(".block_sparse_moe.", ".mlp.");
            (runtime, raw)
        })
        .collect()
}

/// Per-forward causal state shared by all MLA layers.
pub struct KimiLinearForwardContext {
    mask: Option<Array>,
}

impl GeneralLayerwiseModelAdapter for KimiLinearLayerwiseAdapter {
    type Input<'a> = &'a Array;
    type Cache = Cache;
    type Layer = DecoderLayer;
    type ForwardContext = KimiLinearForwardContext;

    fn static_units(&self, store: &dyn WeightStore) -> Result<Vec<StaticUnitBindings>, Error> {
        let mut units = vec![
            StaticUnitBindings::new(
                EMBEDDING_UNIT,
                build_module_bindings_with_recipes(
                    &self.embedding,
                    "model.embed_tokens",
                    store,
                    BTreeMap::new(),
                )?,
            )?,
            StaticUnitBindings::new(
                NORM_UNIT,
                build_module_bindings_with_recipes(
                    &self.norm,
                    "model.norm",
                    store,
                    BTreeMap::new(),
                )?,
            )?,
        ];
        if let Some(lm_head) = &self.lm_head {
            units.push(StaticUnitBindings::new(
                HEAD_UNIT,
                build_module_bindings_with_recipes(lm_head, "lm_head", store, BTreeMap::new())?,
            )?);
        }
        Ok(units)
    }

    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error> {
        let expected = if self.lm_head.is_some() { 3 } else { 2 };
        if leases.len() != expected {
            return Err(Error::UnsupportedArchitecture(format!(
                "Kimi Linear adapter received {} static leases, expected {expected}",
                leases.len()
            )));
        }
        populate_module_from_lease(&mut self.embedding, &leases[0])?;
        populate_module_from_lease(&mut self.norm, &leases[1])?;
        if let Some(lm_head) = &mut self.lm_head {
            populate_module_from_lease(lm_head, &leases[2])?;
        }
        Ok(())
    }

    fn validate_cache(&self, cache: &mut Cache) -> Result<(), Error> {
        if cache.layers.is_empty() {
            *cache = Cache::new(&self.args);
        }
        cache.validate(&self.args.layer_schedule)
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
            context: KimiLinearForwardContext { mask },
        })
    }

    fn execution_group_count(&self) -> usize {
        1
    }

    fn execution_group_id(&self, group: usize) -> Result<String, Error> {
        (group == 0).then(|| "text_decoder".into()).ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Kimi Linear decoder has no execution group {group}"
            ))
        })
    }

    fn layer_count(&self, group: usize) -> Result<usize, Error> {
        if group == 0 {
            Ok(self.args.layer_schedule.len())
        } else {
            Err(Error::UnsupportedArchitecture(format!(
                "Kimi Linear decoder has no execution group {group}"
            )))
        }
    }

    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        self.layer_count(group)?;
        Ok(DecoderLayer::new(&self.args, index, stream)?)
    }

    fn layer_checkpoint_prefix(&self, _group: usize, index: usize) -> String {
        format!("model.layers.{index}")
    }

    fn layer_unit_name(&self, _group: usize, index: usize) -> String {
        format!("kimi_linear.layer.{index:05}")
    }

    fn layer_bindings(
        &self,
        _group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn WeightStore,
    ) -> Result<Vec<WeightBinding>, Error> {
        let bindings = build_module_bindings_with_recipes(
            layer,
            &format!("model.layers.{index}"),
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

    fn additional_consumed_checkpoint_keys(&self, store: &dyn WeightStore) -> Vec<String> {
        if self.sparse_expert_cache {
            store
                .keys()
                .into_iter()
                .filter(|key| {
                    key.contains(".mlp.experts.") || key.contains(".block_sparse_moe.experts.")
                })
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
                "Kimi Linear layer schedule has no policy for layer {index}"
            ))
        })?;
        if self.sparse_expert_cache && policy.feed_forward == FeedForwardPolicy::SparseMoe {
            let expert_cache = self.expert_cache.as_ref().ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "Kimi Linear sparse expert cache was not initialized".into(),
                )
            })?;
            let pass = if hidden.dim(1) > 1 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            };
            return Ok(layer.forward_sparse_experts(
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
                                let gate_up_quantization = self.args.weight_quantization_for(
                                    &format!("model.layers.{index}.mlp.experts.gate_up_proj"),
                                );
                                let down_quantization = self.args.weight_quantization_for(
                                    &format!("model.layers.{index}.mlp.experts.down_proj"),
                                );
                                let mut bank = crate::nn::moe::PackedSwiGluExperts::new(
                                    acquired.identities().len() as i32,
                                    self.args.hidden_size,
                                    self.args.moe_intermediate_size,
                                    gate_up_quantization,
                                    down_quantization,
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
        let mut observer = None;
        Ok(layer.forward_impl(
            hidden,
            context.mask.as_ref(),
            Some(&mut cache.layers[index]),
            stream,
            &format!("model.layers.{index}"),
            &mut observer,
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

    fn ignores_checkpoint_key(&self, key: &str) -> bool {
        key.starts_with("model.mtp.")
    }
}

pub(crate) fn kimi_expert_catalog(
    args: &ModelArgs,
    store: &dyn WeightStore,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let normalized = normalized_checkpoint_keys(store);
    let mut entries = Vec::new();
    for (layer, policy) in args.layer_schedule.iter().enumerate() {
        if policy.feed_forward != FeedForwardPolicy::SparseMoe {
            continue;
        }
        let prefix = format!("model.layers.{layer}.mlp.experts");
        for expert in 0..args.num_experts as usize {
            let mut bindings = Vec::new();
            for (binding_name, recipe) in [
                (
                    "gate_up_proj",
                    expert_projection_recipe(&normalized, &prefix, expert, "gate_up_proj")?,
                ),
                (
                    "down_proj",
                    expert_projection_recipe(&normalized, &prefix, expert, "down_proj")?,
                ),
            ] {
                bindings.push(recipe_binding(binding_name, recipe, store)?);
            }
            for (name, projection, suffix) in [
                ("gate_up_proj_scales", "gate_up_proj", "_scales"),
                ("gate_up_proj_biases", "gate_up_proj", "_biases"),
                ("down_proj_scales", "down_proj", "_scales"),
                ("down_proj_biases", "down_proj", "_biases"),
            ] {
                if let Some(recipe) = optional_expert_component_recipe(
                    &normalized,
                    &prefix,
                    expert,
                    projection,
                    suffix,
                )? {
                    bindings.push(recipe_binding(name, recipe, store)?);
                }
            }
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    Error::UnsupportedArchitecture(
                        "Kimi Linear expert byte total overflowed".into(),
                    )
                })
            })?;
            let identity = ExpertIdentity::new(layer, expert);
            let unit = OffloadUnit::new(identity.unit_id(), bindings)?;
            entries.push(ExpertCatalogEntry::new(identity, unit, bytes)?);
        }
    }
    Ok(entries)
}

fn expert_projection_recipe(
    normalized: &BTreeMap<String, String>,
    prefix: &str,
    expert: usize,
    projection: &str,
) -> Result<DerivedWeightRecipe, Error> {
    if let Some(raw) = normalized.get(&format!("{prefix}.{projection}")) {
        return Ok(DerivedWeightRecipe::source(
            raw.clone(),
            TensorSelection::Range {
                axis: 0,
                start: expert,
                end: expert + 1,
            },
        ));
    }
    if projection == "gate_up_proj" {
        let gate = normalized.get(&format!("{prefix}.gate_proj"));
        let up = normalized.get(&format!("{prefix}.up_proj"));
        match (gate, up) {
            (Some(gate), Some(up)) => {
                let selection = TensorSelection::Range {
                    axis: 0,
                    start: expert,
                    end: expert + 1,
                };
                return Ok(DerivedWeightRecipe::Concatenate {
                    axis: 1,
                    inputs: vec![
                        DerivedWeightRecipe::source(gate.clone(), selection.clone()),
                        DerivedWeightRecipe::source(up.clone(), selection),
                    ],
                });
            }
            (None, None) => {}
            _ => {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Kimi Linear checkpoint {prefix} has mismatched packed gate/up expert tensors"
                )));
            }
        }
        let source = |name: &str| -> Result<DerivedWeightRecipe, Error> {
            let runtime = format!("{prefix}.{expert}.{name}.weight");
            let raw = normalized.get(&runtime).ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "Kimi Linear checkpoint is missing expert tensor {runtime}"
                ))
            })?;
            Ok(DerivedWeightRecipe::source(
                raw.clone(),
                TensorSelection::Full,
            ))
        };
        return Ok(DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: vec![DerivedWeightRecipe::Concatenate {
                axis: 0,
                inputs: vec![source("w1")?, source("w3")?],
            }],
        });
    }
    let runtime = format!("{prefix}.{expert}.w2.weight");
    let raw = normalized.get(&runtime).ok_or_else(|| {
        Error::UnsupportedArchitecture(format!(
            "Kimi Linear checkpoint is missing expert tensor {runtime}"
        ))
    })?;
    Ok(DerivedWeightRecipe::Stack {
        axis: 0,
        inputs: vec![DerivedWeightRecipe::source(
            raw.clone(),
            TensorSelection::Full,
        )],
    })
}

fn optional_expert_component_recipe(
    normalized: &BTreeMap<String, String>,
    prefix: &str,
    expert: usize,
    projection: &str,
    suffix: &str,
) -> Result<Option<DerivedWeightRecipe>, Error> {
    let selection = TensorSelection::Range {
        axis: 0,
        start: expert,
        end: expert + 1,
    };
    if let Some(raw) = normalized.get(&format!("{prefix}.{projection}{suffix}")) {
        return Ok(Some(DerivedWeightRecipe::source(raw.clone(), selection)));
    }
    if projection != "gate_up_proj" {
        return Ok(None);
    }
    let gate = normalized.get(&format!("{prefix}.gate_proj{suffix}"));
    let up = normalized.get(&format!("{prefix}.up_proj{suffix}"));
    match (gate, up) {
        (Some(gate), Some(up)) => Ok(Some(DerivedWeightRecipe::Concatenate {
            axis: 1,
            inputs: vec![
                DerivedWeightRecipe::source(gate.clone(), selection.clone()),
                DerivedWeightRecipe::source(up.clone(), selection),
            ],
        })),
        (None, None) => Ok(None),
        _ => Err(Error::UnsupportedArchitecture(format!(
            "Kimi Linear checkpoint {prefix} has mismatched packed gate/up expert components {suffix:?}"
        ))),
    }
}

fn recipe_binding(
    name: &str,
    recipe: DerivedWeightRecipe,
    store: &dyn WeightStore,
) -> Result<WeightBinding, Error> {
    let bytes = recipe.infer(store)?.byte_len();
    Ok(WeightBinding::from_recipe(name, recipe, bytes)?)
}

/// Token generation over a bounded Kimi model.
pub type Generate<'a, S = crate::runtime::generation::sampler::DefaultSampler> =
    crate::nn::generation::Generate<'a, KimiLinearLayerwiseModel, Cache, S>;

#[cfg(test)]
mod tests {
    use std::fs;

    use safemlx::{
        module::ModuleParameters,
        ops::{indexing::TryIndexOp, zeros_dtype},
        Array, Device, DeviceType, ExecutionContext, Stream,
    };

    use super::{
        expert_projection_recipe, load_kimi_linear_sparse_expert_cache_model,
        normalized_checkpoint_keys, optional_expert_component_recipe,
    };
    use crate::{
        api::ModelKind,
        architectures::kimi_linear::model::{
            load_model, model_args_from_config_value, Model, ModelInput,
        },
        runtime::{
            checkpoint::store::{SafetensorsWeightStore, WeightStore},
            execution::layerwise::LayerwiseLoadOptions,
            residency::{expert_cache::ExpertCacheLoadOptions, policy::OffloadConfig},
        },
    };

    fn tiny_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "kimi_linear",
            "vocab_size": 32,
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
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
        })
    }

    fn write_official_style_fixture(directory: &std::path::Path, model: &Model, stream: &Stream) {
        let mut arrays = Vec::<(String, Array)>::new();
        for (name, parameter) in model.parameters().flatten() {
            let value = zeros_dtype(parameter.shape(), parameter.dtype(), stream).unwrap();
            if name.as_ref() == "model.layers.1.mlp.experts.gate_up_proj" {
                for expert in 0..model.args.num_experts {
                    arrays.push((
                        format!("model.layers.1.block_sparse_moe.experts.{expert}.w1.weight"),
                        value
                            .try_index_device(
                                (expert, ..model.args.moe_intermediate_size, ..),
                                stream,
                            )
                            .unwrap(),
                    ));
                    arrays.push((
                        format!("model.layers.1.block_sparse_moe.experts.{expert}.w3.weight"),
                        value
                            .try_index_device(
                                (expert, model.args.moe_intermediate_size.., ..),
                                stream,
                            )
                            .unwrap(),
                    ));
                }
                continue;
            }
            if name.as_ref() == "model.layers.1.mlp.experts.down_proj" {
                for expert in 0..model.args.num_experts {
                    arrays.push((
                        format!("model.layers.1.block_sparse_moe.experts.{expert}.w2.weight"),
                        value.try_index_device((expert, .., ..), stream).unwrap(),
                    ));
                }
                continue;
            }
            let checkpoint_name = if name.starts_with("model.layers.1.mlp.") {
                name.replacen("model.layers.1.mlp.", "model.layers.1.block_sparse_moe.", 1)
            } else {
                name.to_string()
            };
            let value = if checkpoint_name.ends_with("_conv1d.weight") {
                value
                    .reshape(
                        &[
                            model.args.kda_config.num_heads * model.args.kda_config.head_dim,
                            model.args.kda_config.short_conv_kernel_size,
                        ],
                        stream,
                    )
                    .unwrap()
            } else {
                value
            };
            arrays.push((checkpoint_name, value));
        }
        Array::save_safetensors(
            arrays.iter().map(|(name, value)| (name.as_str(), value)),
            None,
            directory.join("model.safetensors"),
        )
        .unwrap();
        fs::write(
            directory.join("config.json"),
            serde_json::to_vec(&tiny_config()).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn packed_gguf_style_expert_recipes_combine_selected_gate_and_up_components() {
        let dir = tempfile::tempdir().unwrap();
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let prefix = "model.layers.1.mlp.experts";
        let arrays = [
            (
                format!("{prefix}.gate_proj"),
                Array::from_slice(
                    &(0..8).map(|value| value as f32).collect::<Vec<_>>(),
                    &[2, 2, 2],
                ),
            ),
            (
                format!("{prefix}.up_proj"),
                Array::from_slice(
                    &(100..108).map(|value| value as f32).collect::<Vec<_>>(),
                    &[2, 2, 2],
                ),
            ),
            (
                format!("{prefix}.gate_proj_scales"),
                Array::from_slice(&[10.0f32, 11.0, 12.0, 13.0], &[2, 2, 1]),
            ),
            (
                format!("{prefix}.up_proj_scales"),
                Array::from_slice(&[20.0f32, 21.0, 22.0, 23.0], &[2, 2, 1]),
            ),
        ];
        Array::save_safetensors(
            arrays.iter().map(|(name, value)| (name.as_str(), value)),
            None,
            dir.path().join("model.safetensors"),
        )
        .unwrap();
        let store = SafetensorsWeightStore::open(dir.path()).unwrap();
        let normalized = normalized_checkpoint_keys(&store);

        let weights = expert_projection_recipe(&normalized, prefix, 1, "gate_up_proj")
            .unwrap()
            .materialize(&store, &stream)
            .unwrap();
        assert_eq!(weights.shape(), &[1, 4, 2]);
        let weights = weights.evaluated().unwrap();
        assert_eq!(
            weights.as_slice::<f32>(),
            &[4.0, 5.0, 6.0, 7.0, 104.0, 105.0, 106.0, 107.0]
        );

        let scales =
            optional_expert_component_recipe(&normalized, prefix, 1, "gate_up_proj", "_scales")
                .unwrap()
                .unwrap()
                .materialize(&store, &stream)
                .unwrap();
        assert_eq!(scales.shape(), &[1, 4, 1]);
        let scales = scales.evaluated().unwrap();
        assert_eq!(scales.as_slice::<f32>(), &[12.0, 13.0, 22.0, 23.0]);
        assert_eq!(store.keys().len(), 4);
    }

    #[test]
    fn sparse_cache_and_rank_owned_ep_load_official_split_experts() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let fixture = Model::new(
            model_args_from_config_value(&tiny_config()).unwrap(),
            gpu.stream(),
        )
        .unwrap();
        let directory = tempfile::tempdir().unwrap();
        write_official_style_fixture(directory.path(), &fixture, gpu.stream());

        let mut resident = load_model(directory.path(), gpu.stream(), cpu.stream()).unwrap();
        let options = ExpertCacheLoadOptions::new(
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
            OffloadConfig::new(None, None, 1).unwrap(),
            1 << 20,
            1,
        )
        .unwrap();
        let mut sparse = load_kimi_linear_sparse_expert_cache_model(
            directory.path(),
            options,
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut resident_cache = resident.new_cache();
        let mut sparse_cache = sparse.new_cache();
        for tokens in [
            Array::from_slice(&[1i32, 2], &[1, 2]),
            Array::from_slice(&[3i32], &[1, 1]),
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
            let actual = sparse
                .forward(&tokens, &mut sparse_cache, gpu.stream())
                .unwrap();
            let expected = expected.evaluated().unwrap();
            let actual = actual.evaluated().unwrap();
            assert_eq!(actual.as_slice::<f32>(), expected.as_slice::<f32>());
        }
        let report = sparse.expert_cache_report().unwrap().unwrap();
        assert_eq!(report.owned_experts, 4);
        assert!(report.prefill.requested_routes > 0);
        assert!(report.decode.requested_routes > 0);
        assert!(report.prefill.compact_banks > 1);
        crate::architectures::distributed::expert::assert_rank_owned_sparse_ep_load(
            directory.path(),
            options,
            ModelKind::KimiLinear,
            2,
            gpu.stream(),
            cpu.stream(),
        );
    }
}

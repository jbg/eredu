//! Bounded-residency execution for DeepSeek V4 SafeTensors checkpoints.

use std::{collections::BTreeMap, path::Path, sync::Arc, time::Instant};

use safemlx::{
    error::Exception,
    module::{Module, ModuleParameters, Param},
    ops::{broadcast_to, indexing::NewAxis, indexing::TryIndexOp},
    Array, Dtype, Stream,
};

use crate::{
    api::input,
    error::Error,
    nn::generation::CausalLm,
    runtime::{
        cache::residency::{PromptCacheModelIdentity, PromptCacheTopology},
        checkpoint::{
            binding::{
                build_module_bindings_with_recipes, populate_module_from_lease,
                populate_module_from_lease_excluding,
            },
            quantization::WeightQuantization,
            recipe::DerivedWeightRecipe,
            store::{TensorSelection, WeightStore},
        },
        execution::layerwise::{
            load_layerwise_model, load_safetensors_layerwise_model, open_safetensors_weight_store,
            ArchitectureAdapter, LayerWeightResidency, LayerwiseForwardState, LayerwiseModel,
            StaticUnitBindings,
        },
        residency::{
            expert_cache::{
                ExpertCache, ExpertCacheError, ExpertCacheLoadOptions, ExpertCacheReport,
                ExpertCatalogEntry, ExpertIdentity, ExpertPass, ExpertRouteBatch,
            },
            manager::{OffloadUnit, ResidencyReport, ResidentUnitLease, WeightBinding},
        },
    },
};

use super::{
    attention::AttentionCache,
    model::{
        prompt_cache_architecture_fingerprint, Cache, DecoderLayer, Model as ResidentModel,
        ModelArgs,
    },
};

const EMBEDDING_UNIT: &str = "deepseek_v4.static.embedding";
const NORM_UNIT: &str = "deepseek_v4.static.norm";
const HC_HEAD_UNIT: &str = "deepseek_v4.static.hc_head";
const HEAD_UNIT: &str = "deepseek_v4.static.output";
const DRAFT_UNIT: &str = "deepseek_v4.static.draft";

/// DeepSeek V4 decoder using the generalized residency executor.
pub struct DeepSeekV4LayerwiseModel {
    execution: LayerwiseModel<DeepSeekV4LayerwiseAdapter>,
}

impl DeepSeekV4LayerwiseModel {
    /// Validated architecture arguments.
    pub fn args(&self) -> &ModelArgs {
        &self.execution.adapter().args
    }

    /// Creates target and embedded-draft cache state.
    pub fn new_cache(&self) -> Result<Cache, Exception> {
        self.execution.adapter().new_cache()
    }

    /// Runs target decoding.
    pub fn forward(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution.forward(tokens, cache, stream)
    }

    pub(crate) fn mtp_len(&self) -> usize {
        self.execution.adapter().static_model.mtp_len()
    }

    fn forward_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput, Exception> {
        let (logits, context) = self
            .execution
            .forward_with_context_hook(tokens, cache, stream, |_, _, _| Ok(()))
            .map_err(|error| Exception::custom(error.to_string()))?;
        let hidden = context.draft_hidden.ok_or_else(|| {
            Exception::custom("DeepSeek V4 layerwise pass did not retain draft hidden state")
        })?;
        Ok(
            crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput {
                logits,
                hidden,
                tokens: tokens.clone(),
            },
        )
    }

    /// Current bounded-residency telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        self.execution.residency_report()
    }

    /// Dense disk-stream telemetry when that policy is active.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<crate::runtime::execution::layerwise::DenseDiskStreamReport>, Error> {
        self.execution.dense_stream_report()
    }

    /// Independent expert-cache telemetry when configured.
    pub fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        Ok(self
            .execution
            .adapter()
            .expert_cache
            .as_ref()
            .map(ExpertCache::report)
            .transpose()?)
    }

    /// Returns the generic prompt-cache layout when the V4 attention schedule
    /// does not require compressed pooling/indexer state.
    pub(crate) fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        self.execution
            .adapter()
            .static_model
            .prompt_cache_layer_layout()
    }
}

impl CausalLm<Cache> for DeepSeekV4LayerwiseModel {
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
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward(tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }
}

impl crate::runtime::generation::embedded_mtp::EmbeddedMtpTarget for DeepSeekV4LayerwiseModel {
    type Cache = Cache;
    type DraftCache = Vec<AttentionCache>;

    fn prefill_target(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        *cache = self.new_cache()?;
        self.forward_mtp_target(&tokens, cache, stream)
    }

    fn verify_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput, Exception> {
        self.forward_mtp_target(tokens, cache, stream)
    }

    fn prefill_draft_cache(
        &mut self,
        output: &crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        <ResidentModel as crate::runtime::generation::embedded_mtp::EmbeddedMtpTarget>::prefill_draft_cache(
            &mut self.execution.adapter_mut().static_model,
            output,
            tokens,
            cache,
            stream,
        )
    }

    fn draft_cache(cache: &Cache) -> Self::DraftCache {
        cache.mtp_layers.clone()
    }

    fn commit_draft_cache(cache: &mut Cache, draft: &Self::DraftCache) {
        cache.mtp_layers.clone_from(draft);
    }

    fn draft_logits(
        &mut self,
        hidden: &Array,
        last_token: u32,
        draft_index: usize,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        <ResidentModel as crate::runtime::generation::embedded_mtp::EmbeddedMtpTarget>::draft_logits(
            &mut self.execution.adapter_mut().static_model,
            hidden,
            last_token,
            draft_index,
            cache,
            stream,
        )
    }

    fn fused_draft_logits(
        &mut self,
        hidden: &Array,
        last_token: u32,
        proposal_capacity: usize,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<Option<Array>, Exception> {
        <ResidentModel as crate::runtime::generation::embedded_mtp::EmbeddedMtpTarget>::fused_draft_logits(
            &mut self.execution.adapter_mut().static_model,
            hidden,
            last_token,
            proposal_capacity,
            cache,
            stream,
        )
    }

    fn adjust_fused_draft_logits(
        &mut self,
        logits: Array,
        last_token: u32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        <ResidentModel as crate::runtime::generation::embedded_mtp::EmbeddedMtpTarget>::adjust_fused_draft_logits(
            &mut self.execution.adapter_mut().static_model,
            logits,
            last_token,
            stream,
        )
    }

    fn advance_draft_cache(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        <ResidentModel as crate::runtime::generation::embedded_mtp::EmbeddedMtpTarget>::advance_draft_cache(
            &mut self.execution.adapter_mut().static_model,
            hidden,
            tokens,
            cache,
            stream,
        )
    }

    fn max_draft_tokens(&self) -> usize {
        self.mtp_len()
    }
}

/// Token generation over a bounded V4 model.
pub type Generate<'a, S = crate::runtime::generation::sampler::DefaultSampler> =
    crate::nn::generation::Generate<'a, DeepSeekV4LayerwiseModel, Cache, S>;

/// Per-forward token state retained across streamed decoder units.
pub struct DeepSeekV4ForwardContext {
    input_ids: Array,
    captures: Vec<(usize, Array)>,
    draft_hidden: Option<Array>,
}

/// Architecture adapter used by the generalized layerwise executor.
pub struct DeepSeekV4LayerwiseAdapter {
    args: ModelArgs,
    static_model: ResidentModel,
    sparse_expert_cache: bool,
    expert_cache: Option<ExpertCache>,
}

impl DeepSeekV4LayerwiseAdapter {
    fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let mut resident = ResidentModel::new(args.clone(), stream)?;
        // Decoder layers are residency units owned by `LayerwiseModel`. Keep only
        // the shared static target and draft modules in this holder.
        resident.model.layers.clear();
        Ok(Self {
            static_model: resident,
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

    fn new_cache(&self) -> Result<Cache, Exception> {
        let layers = self.args.compress_ratios[..self.args.num_hidden_layers as usize]
            .iter()
            .map(|ratio| AttentionCache::new_for_ratio(*ratio, self.args.sliding_window))
            .collect::<Result<_, _>>()?;
        let mut cache = self.static_model.new_cache()?;
        cache.layers = layers;
        Ok(cache)
    }

    fn layer_recipes(
        &self,
        layer: &DecoderLayer,
        index: usize,
        store: &dyn WeightStore,
    ) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
        let root = format!("layers.{index}");
        let mut recipes = BTreeMap::new();
        for (local, target) in layer.parameters().flatten() {
            let recipe = if let Some(recipe) =
                expert_bank_recipe(local.as_ref(), target.shape(), &root, &self.args, store)?
            {
                recipe
            } else if let Some(rest) = local.strip_prefix("attn.wo_a.projections.") {
                grouped_output_recipe(rest, &root, &self.args, store)?
            } else {
                let raw = raw_layer_key(&root, local.as_ref());
                DerivedWeightRecipe::source(raw, TensorSelection::Full)
            };
            recipes.insert(local.to_string(), recipe);
        }
        Ok(recipes)
    }

    fn draft_static_unit(
        &self,
        store: &dyn WeightStore,
    ) -> Result<Option<StaticUnitBindings>, Error> {
        if let Some(mtp) = &self.static_model.mtp {
            return Ok(Some(StaticUnitBindings::new(
                DRAFT_UNIT,
                build_module_bindings_with_recipes(
                    mtp,
                    "mtp",
                    store,
                    draft_recipes(mtp, &self.args, store, false)?,
                )?,
            )?));
        }
        if let Some(dspark) = &self.static_model.dspark {
            return Ok(Some(StaticUnitBindings::new(
                DRAFT_UNIT,
                build_module_bindings_with_recipes(
                    dspark,
                    "mtp",
                    store,
                    draft_recipes(dspark, &self.args, store, true)?,
                )?,
            )?));
        }
        Ok(None)
    }
}

impl ArchitectureAdapter for DeepSeekV4LayerwiseAdapter {
    type Input<'a> = &'a Array;
    type Cache = Cache;
    type Layer = DecoderLayer;
    type ForwardContext = DeepSeekV4ForwardContext;

    fn model_type(&self) -> &str {
        &self.args.model_type
    }

    fn prompt_cache_model_identity(
        &self,
        topology: Option<crate::ParallelTopology>,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let layer_count = self.args.num_hidden_layers as usize;
        if self.args.compress_ratios[..layer_count]
            .iter()
            .any(|ratio| *ratio != 0)
        {
            return Err(Error::UnsupportedArchitecture(
                "DeepSeek-V4 compressed sparse prompt caches require pooled/indexer state".into(),
            ));
        }
        let policy = crate::LayerCachePolicy::key_value(
            crate::AttentionPolicy::sliding(self.args.sliding_window as u32)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
            self.args.num_attention_heads,
            self.args.head_dim,
        )
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let layer_layout = crate::LayerSchedule::new(layer_count, vec![policy; layer_count])
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        Ok(PromptCacheModelIdentity {
            model_family: "deepseek_v4".into(),
            effective_model_type: self.args.model_type.clone(),
            architecture_fingerprint: prompt_cache_architecture_fingerprint(&self.args),
            layer_count,
            global_layer_start: 0,
            global_layer_end: layer_count,
            sink_tokens: 0,
            topology: topology.map_or_else(
                PromptCacheTopology::default,
                PromptCacheTopology::for_parallel_topology,
            ),
            layer_layout,
        })
    }

    fn static_units(&self, store: &dyn WeightStore) -> Result<Vec<StaticUnitBindings>, Error> {
        let source = |key: &str| DerivedWeightRecipe::source(key, TensorSelection::Full);
        Ok(vec![
            StaticUnitBindings::new(
                EMBEDDING_UNIT,
                build_module_bindings_with_recipes(
                    &self.static_model.model.embed_tokens,
                    "embed",
                    store,
                    BTreeMap::new(),
                )?,
            )?,
            StaticUnitBindings::new(
                NORM_UNIT,
                build_module_bindings_with_recipes(
                    &self.static_model.model.norm,
                    "norm",
                    store,
                    BTreeMap::new(),
                )?,
            )?,
            StaticUnitBindings::new(
                HC_HEAD_UNIT,
                build_module_bindings_with_recipes(
                    &self.static_model.model.hc_head,
                    "hc_head",
                    store,
                    BTreeMap::from([
                        ("function".into(), source("hc_head_fn")),
                        ("base".into(), source("hc_head_base")),
                        ("scale".into(), source("hc_head_scale")),
                    ]),
                )?,
            )?,
            StaticUnitBindings::new(
                HEAD_UNIT,
                build_module_bindings_with_recipes(
                    &self.static_model.lm_head,
                    "head",
                    store,
                    qwen_linear_recipes("head", &self.static_model.lm_head),
                )?,
            )?,
        ]
        .into_iter()
        .chain(self.draft_static_unit(store)?)
        .collect())
    }

    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error> {
        let expected =
            4 + usize::from(self.static_model.mtp.is_some() || self.static_model.dspark.is_some());
        if leases.len() != expected {
            return Err(Error::UnsupportedArchitecture(format!(
                "DeepSeek V4 adapter received {} static leases, expected {expected}",
                leases.len(),
            )));
        }
        populate_module_from_lease(&mut self.static_model.model.embed_tokens, &leases[0])?;
        populate_module_from_lease(&mut self.static_model.model.norm, &leases[1])?;
        populate_module_from_lease(&mut self.static_model.model.hc_head, &leases[2])?;
        populate_module_from_lease(&mut self.static_model.lm_head, &leases[3])?;
        if let Some(lease) = leases.get(4) {
            if let Some(mtp) = &mut self.static_model.mtp {
                populate_module_from_lease(mtp, lease)?;
            } else if let Some(dspark) = &mut self.static_model.dspark {
                populate_module_from_lease(dspark, lease)?;
            }
        }
        Ok(())
    }

    fn validate_cache(&self, cache: &mut Cache) -> Result<(), Error> {
        if cache.layers.is_empty() {
            *cache = self.new_cache()?;
        }
        if cache.layers.len() != self.args.num_hidden_layers as usize {
            return Err(Error::UnsupportedArchitecture(
                "DeepSeek V4 cache does not match decoder depth".into(),
            ));
        }
        Ok(())
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        _cache: &mut Cache,
        stream: &Stream,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        let embedded = self
            .static_model
            .model
            .embed_tokens
            .forward(input, stream)?;
        let hidden = embedded.try_index_device((.., .., NewAxis, ..), stream)?;
        let hidden = broadcast_to(
            &hidden,
            &[
                embedded.dim(0),
                embedded.dim(1),
                self.args.hc_mult,
                self.args.hidden_size,
            ],
            stream,
        )?;
        Ok(LayerwiseForwardState {
            hidden,
            context: DeepSeekV4ForwardContext {
                input_ids: input.clone(),
                captures: Vec::new(),
                draft_hidden: None,
            },
        })
    }

    fn execution_graph(
        &self,
    ) -> Result<crate::runtime::execution::layerwise::ExecutionGroupDag, Error> {
        crate::runtime::execution::layerwise::ExecutionGroupDag::chain(["text_decoder"])
    }

    fn layer_count(&self, group: usize) -> Result<usize, Error> {
        if group == 0 {
            Ok(self.args.num_hidden_layers as usize)
        } else {
            Err(Error::UnsupportedArchitecture(format!(
                "DeepSeek V4 decoder has no execution group {group}"
            )))
        }
    }

    fn new_layer(
        &self,
        group: usize,
        index: usize,
        stream: &Stream,
    ) -> Result<DecoderLayer, Error> {
        self.layer_count(group)?;
        Ok(DecoderLayer::new(&self.args, index, stream)?)
    }

    fn layer_checkpoint_prefix(&self, _group: usize, index: usize) -> String {
        format!("layers.{index}")
    }

    fn layer_unit_name(&self, _group: usize, index: usize) -> String {
        format!("deepseek_v4.layer.{index:05}")
    }

    fn layer_bindings(
        &self,
        _group: usize,
        index: usize,
        layer: &DecoderLayer,
        store: &dyn WeightStore,
    ) -> Result<Vec<WeightBinding>, Error> {
        let bindings = build_module_bindings_with_recipes(
            layer,
            &format!("layers.{index}"),
            store,
            self.layer_recipes(layer, index, store)?,
        )?;
        if self.sparse_expert_cache {
            Ok(bindings
                .into_iter()
                .filter(|binding| !binding.name().starts_with("ffn.switch_mlp."))
                .collect())
        } else {
            Ok(bindings)
        }
    }

    fn populate_layer(
        &self,
        _group: usize,
        _index: usize,
        layer: &mut DecoderLayer,
        lease: &ResidentUnitLease,
    ) -> Result<(), Error> {
        if self.sparse_expert_cache {
            populate_module_from_lease_excluding(layer, lease, |name| {
                name.starts_with("ffn.switch_mlp.")
            })?;
        } else {
            populate_module_from_lease(layer, lease)?;
        }
        Ok(())
    }

    fn additional_consumed_checkpoint_keys(&self, store: &dyn WeightStore) -> Vec<String> {
        if self.sparse_expert_cache {
            store
                .keys()
                .into_iter()
                .filter(|key| key.contains(".ffn.experts."))
                .collect()
        } else {
            Vec::new()
        }
    }

    fn forward_layer(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut DecoderLayer,
        hidden: &Array,
        cache: &mut Cache,
        context: &mut DeepSeekV4ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.layer_count(group)?;
        let output = if let Some(expert_cache) = &self.expert_cache {
            let pass = if hidden.dim(1) > 1 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            };
            layer.forward_with_expert_executor(
                hidden,
                None,
                Some(&mut cache.layers[index]),
                &context.input_ids,
                stream,
                |flat, indices, weights, stream| {
                    execute_cached_experts(
                        &self.args,
                        expert_cache,
                        index,
                        flat,
                        indices,
                        weights,
                        pass,
                        stream,
                    )
                },
            )?
        } else {
            layer.forward(
                hidden,
                None,
                Some(&mut cache.layers[index]),
                &context.input_ids,
                stream,
            )?
        };
        if let Some(dspark) = &self.args.dspark {
            if let Some(position) = dspark
                .target_layer_ids
                .iter()
                .position(|wanted| *wanted == index as i32)
            {
                context.captures.push((
                    position,
                    safemlx::ops::mean_axis(&output, 2, false, stream)?,
                ));
            }
        }
        if index + 1 == self.args.num_hidden_layers as usize {
            context.draft_hidden = if self.args.dspark.is_some() {
                context.captures.sort_by_key(|(position, _)| *position);
                Some(safemlx::ops::concatenate_axis(
                    &context
                        .captures
                        .iter()
                        .map(|(_, value)| value)
                        .collect::<Vec<_>>(),
                    -1,
                    stream,
                )?)
            } else {
                Some(output.clone())
            };
        }
        Ok(output)
    }

    fn retained_arrays<'a>(&self, cache: &'a Cache, _group: usize, index: usize) -> Vec<&'a Array> {
        cache.layers[index].retained_arrays()
    }

    fn finish(
        &mut self,
        hidden: &Array,
        _cache: &mut Cache,
        _context: &DeepSeekV4ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let hidden = self.static_model.model.hc_head.forward(hidden, stream)?;
        let hidden = self.static_model.model.norm.forward(&hidden, stream)?;
        Ok(self.static_model.lm_head.forward(&hidden, stream)?)
    }

    fn ignores_checkpoint_key(&self, _key: &str) -> bool {
        false
    }
}

/// Loads V4 with resident, host-windowed, or dense disk-streamed layers.
pub fn load_deepseek_v4_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV4LayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let args = super::model::get_model_args(model_dir)?;
    let adapter = DeepSeekV4LayerwiseAdapter::new(args, stream)?;
    Ok(DeepSeekV4LayerwiseModel {
        execution: load_safetensors_layerwise_model(
            model_dir,
            adapter,
            options,
            stream,
            weights_stream,
        )?,
    })
}

/// Loads V4 with routed experts in independent cache units.
pub fn load_deepseek_v4_expert_cache_model(
    model_dir: impl AsRef<Path>,
    non_expert: crate::NonExpertWeightResidency,
    options: ExpertCacheLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV4LayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = super::model::get_model_args(model_dir)?;
    let store = open_safetensors_weight_store(model_dir, non_expert.layers().max_mapped_shards())?;
    let adapter = DeepSeekV4LayerwiseAdapter::new_sparse(args.clone(), stream)?;
    let mut execution = load_layerwise_model(
        Arc::clone(&store),
        adapter,
        non_expert,
        stream,
        weights_stream,
    )?;
    let entries = expert_catalog(&args, store.as_ref())?;
    execution.adapter_mut().expert_cache = Some(ExpertCache::new_shared(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(DeepSeekV4LayerwiseModel { execution })
}

fn raw_layer_key(root: &str, local: &str) -> String {
    let mut key = format!("{root}.{local}");
    for sublayer in ["attn", "ffn"] {
        for (runtime, raw) in [("function", "fn"), ("base", "base"), ("scale", "scale")] {
            key = key.replace(
                &format!(".{sublayer}_hc.{runtime}"),
                &format!(".hc_{sublayer}_{raw}"),
            );
        }
    }
    for (runtime, raw) in [("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3")] {
        key = key.replace(
            &format!(".ffn.shared_experts.{runtime}."),
            &format!(".ffn.shared_experts.{raw}."),
        );
    }
    key = key
        .replace(".ffn.gate.router.weight", ".ffn.gate.weight")
        .replace(".ffn.gate.router.e_score_correction_bias", ".ffn.gate.bias");
    if key.ends_with(".weight_scale_inv") {
        key.truncate(key.len() - ".weight_scale_inv".len());
        key.push_str(".scale");
    }
    key
}

fn grouped_output_recipe(
    rest: &str,
    root: &str,
    args: &ModelArgs,
    _store: &dyn WeightStore,
) -> Result<DerivedWeightRecipe, Error> {
    let (group, component) = rest.split_once('.').ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("invalid V4 grouped output parameter {rest:?}"))
    })?;
    let group: usize = group.parse().map_err(|_| {
        Error::UnsupportedArchitecture(format!("invalid V4 output group {group:?}"))
    })?;
    let (source, rows) = match component {
        "weight" => (
            format!("{root}.attn.wo_a.weight"),
            args.o_lora_rank as usize,
        ),
        "weight_scale_inv" => (
            format!("{root}.attn.wo_a.scale"),
            ((args.o_lora_rank + 127) / 128) as usize,
        ),
        other => {
            return Err(Error::UnsupportedArchitecture(format!(
                "unsupported V4 grouped output component {other:?}"
            )))
        }
    };
    Ok(DerivedWeightRecipe::source(
        source,
        TensorSelection::Range {
            axis: 0,
            start: group * rows,
            end: (group + 1) * rows,
        },
    ))
}

fn expert_bank_recipe(
    local: &str,
    target_shape: &[i32],
    root: &str,
    args: &ModelArgs,
    store: &dyn WeightStore,
) -> Result<Option<DerivedWeightRecipe>, Error> {
    let component = match local {
        "ffn.switch_mlp.gate_up_proj" => Some((true, "weight")),
        "ffn.switch_mlp.gate_up_proj_scales" => Some((true, "scale")),
        "ffn.switch_mlp.down_proj" => Some((false, "weight")),
        "ffn.switch_mlp.down_proj_scales" => Some((false, "scale")),
        _ => None,
    };
    let Some((gate_up, component)) = component else {
        return Ok(None);
    };
    let source = |expert: i32, projection: &str| {
        DerivedWeightRecipe::source(
            format!("{root}.ffn.experts.{expert}.{projection}.{component}"),
            TensorSelection::Full,
        )
    };
    let mut experts = Vec::with_capacity(args.n_routed_experts as usize);
    for expert in 0..args.n_routed_experts {
        experts.push(if gate_up {
            DerivedWeightRecipe::Concatenate {
                axis: 0,
                inputs: vec![source(expert, "w1"), source(expert, "w3")],
            }
        } else {
            source(expert, "w2")
        });
    }
    let mut recipe = DerivedWeightRecipe::Stack {
        axis: 0,
        inputs: experts,
    };
    if component == "weight" && args.expert_dtype.as_deref() == Some("fp4") {
        recipe = DerivedWeightRecipe::View {
            input: Box::new(recipe),
            dtype: Dtype::Uint32,
            shape: target_shape.iter().map(|value| *value as usize).collect(),
        };
    }
    recipe.infer(store)?;
    Ok(Some(recipe))
}

fn qwen_linear_recipes(
    raw_prefix: &str,
    linear: &crate::api::qwen3_5::QwenLinear,
) -> BTreeMap<String, DerivedWeightRecipe> {
    let mut recipes = BTreeMap::new();
    if linear.weight_scale_inv.as_ref().is_some() {
        recipes.insert(
            "weight_scale_inv".into(),
            DerivedWeightRecipe::source(format!("{raw_prefix}.scale"), TensorSelection::Full),
        );
    }
    recipes
}

fn draft_recipes<M: ModuleParameters>(
    module: &M,
    args: &ModelArgs,
    store: &dyn WeightStore,
    dspark: bool,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
    let last = args.num_nextn_predict_layers as usize - 1;
    let mut recipes = BTreeMap::new();
    for (local, target) in module.parameters().flatten() {
        let recipe = if dspark {
            if let Some(rest) = local.strip_prefix("layers.") {
                let (depth, field) = rest.split_once('.').ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!("invalid DSpark parameter {local:?}"))
                })?;
                draft_decoder_recipe(field, target.shape(), &format!("mtp.{depth}"), args, store)?
            } else {
                let raw = match local.as_ref() {
                    "main_proj.weight" => "mtp.0.main_proj.weight".into(),
                    "main_proj.weight_scale_inv" => "mtp.0.main_proj.scale".into(),
                    "main_norm.weight" => "mtp.0.main_norm.weight".into(),
                    "norm.weight" => format!("mtp.{last}.norm.weight"),
                    "hc_head.function" => format!("mtp.{last}.hc_head_fn"),
                    "hc_head.base" => format!("mtp.{last}.hc_head_base"),
                    "hc_head.scale" => format!("mtp.{last}.hc_head_scale"),
                    "markov_w1.weight" => {
                        format!("mtp.{last}.markov_head.markov_w1.weight")
                    }
                    "markov_w2.weight" => {
                        format!("mtp.{last}.markov_head.markov_w2.weight")
                    }
                    "confidence_head.weight" => {
                        format!("mtp.{last}.confidence_head.proj.weight")
                    }
                    other => {
                        return Err(Error::UnsupportedArchitecture(format!(
                            "unsupported DSpark static parameter {other:?}"
                        )))
                    }
                };
                DerivedWeightRecipe::source(raw, TensorSelection::Full)
            }
        } else {
            let rest = local.strip_prefix("layers.").ok_or_else(|| {
                Error::UnsupportedArchitecture(format!("invalid V4 MTP parameter {local:?}"))
            })?;
            let (depth, field) = rest.split_once('.').ok_or_else(|| {
                Error::UnsupportedArchitecture(format!("invalid V4 MTP parameter {local:?}"))
            })?;
            let root = format!("mtp.{depth}");
            if let Some(field) = field.strip_prefix("decoder.") {
                draft_decoder_recipe(field, target.shape(), &root, args, store)?
            } else {
                let raw = match field {
                    "hc_head.function" => format!("{root}.hc_head_fn"),
                    "hc_head.base" => format!("{root}.hc_head_base"),
                    "hc_head.scale" => format!("{root}.hc_head_scale"),
                    other if other.ends_with(".weight_scale_inv") => {
                        format!(
                            "{root}.{}.scale",
                            other.trim_end_matches(".weight_scale_inv")
                        )
                    }
                    other => format!("{root}.{other}"),
                };
                DerivedWeightRecipe::source(raw, TensorSelection::Full)
            }
        };
        recipes.insert(local.to_string(), recipe);
    }
    Ok(recipes)
}

fn draft_decoder_recipe(
    field: &str,
    target_shape: &[i32],
    root: &str,
    args: &ModelArgs,
    store: &dyn WeightStore,
) -> Result<DerivedWeightRecipe, Error> {
    if let Some(recipe) = expert_bank_recipe(field, target_shape, root, args, store)? {
        return Ok(recipe);
    }
    if let Some(rest) = field.strip_prefix("attn.wo_a.projections.") {
        return grouped_output_recipe(rest, root, args, store);
    }
    Ok(DerivedWeightRecipe::source(
        raw_layer_key(root, field),
        TensorSelection::Full,
    ))
}

fn expert_catalog(
    args: &ModelArgs,
    store: &dyn WeightStore,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let mut entries = Vec::new();
    for layer in 0..args.num_hidden_layers as usize {
        let root = format!("layers.{layer}.ffn.experts");
        for expert in 0..args.n_routed_experts as usize {
            let mut bindings = Vec::new();
            for (name, gate_up, component) in [
                ("gate_up_proj", true, "weight"),
                ("down_proj", false, "weight"),
                ("gate_up_proj_scales", true, "scale"),
                ("down_proj_scales", false, "scale"),
            ] {
                if component == "scale"
                    && args.expert_dtype.as_deref().is_none()
                    && store
                        .metadata(&format!("{root}.{expert}.w1.scale"))
                        .is_err()
                {
                    continue;
                }
                let source = |projection: &str| {
                    DerivedWeightRecipe::source(
                        format!("{root}.{expert}.{projection}.{component}"),
                        TensorSelection::Full,
                    )
                };
                let mut recipe = if gate_up {
                    DerivedWeightRecipe::Stack {
                        axis: 0,
                        inputs: vec![DerivedWeightRecipe::Concatenate {
                            axis: 0,
                            inputs: vec![source("w1"), source("w3")],
                        }],
                    }
                } else {
                    DerivedWeightRecipe::Stack {
                        axis: 0,
                        inputs: vec![source("w2")],
                    }
                };
                if component == "weight" && args.expert_dtype.as_deref() == Some("fp4") {
                    let output = if gate_up {
                        2 * args.moe_intermediate_size
                    } else {
                        args.hidden_size
                    };
                    let input = if gate_up {
                        args.hidden_size
                    } else {
                        args.moe_intermediate_size
                    };
                    recipe = DerivedWeightRecipe::View {
                        input: Box::new(recipe),
                        dtype: Dtype::Uint32,
                        shape: vec![1, output as usize, (input / 8) as usize],
                    };
                }
                let bytes = recipe.infer(store)?.byte_len();
                bindings.push(WeightBinding::from_recipe(name, recipe, bytes)?);
            }
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    Error::UnsupportedArchitecture("V4 expert byte total overflowed".into())
                })
            })?;
            let identity = ExpertIdentity::new(layer, expert);
            entries.push(ExpertCatalogEntry::new(
                identity,
                OffloadUnit::new(identity.unit_id(), bindings)?,
                bytes,
            )?);
        }
    }
    Ok(entries)
}

#[allow(clippy::too_many_arguments)]
fn execute_cached_experts(
    args: &ModelArgs,
    cache: &ExpertCache,
    layer: usize,
    flat: &Array,
    indices: &Array,
    weights: &Array,
    pass: ExpertPass,
    stream: &Stream,
) -> Result<Array, Exception> {
    cache
        .execute_routes_bounded(
            ExpertRouteBatch::new(layer, flat, indices, weights, pass),
            stream,
            |flat, acquired, weights, stream| {
                if acquired.is_empty() {
                    return Err(ExpertCacheError::EmptyRoutedBank {
                        architecture: "DeepSeek-V4",
                    });
                }
                let started = Instant::now();
                let quantization = (args.expert_dtype.as_deref() == Some("fp4"))
                    .then_some(WeightQuantization::MxFp4);
                let bank = crate::nn::moe::PackedSwiGluExperts::new(
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
                cache.record_compact_bank(pass, acquired.scratch_bytes(), started.elapsed())?;
                Ok(bank.forward(flat, acquired.compact_routes(), weights, stream)?)
            },
        )
        .map_err(|error| Exception::custom(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::raw_layer_key;

    #[test]
    fn maps_runtime_v4_layer_names_to_official_checkpoint_names() {
        assert_eq!(
            raw_layer_key("layers.7", "attn_hc.function"),
            "layers.7.hc_attn_fn"
        );
        assert_eq!(
            raw_layer_key("layers.7", "ffn_hc.base"),
            "layers.7.hc_ffn_base"
        );
        assert_eq!(
            raw_layer_key("layers.7", "ffn.shared_experts.gate_proj.weight"),
            "layers.7.ffn.shared_experts.w1.weight"
        );
        assert_eq!(
            raw_layer_key("layers.7", "attn.wq_a.weight_scale_inv"),
            "layers.7.attn.wq_a.scale"
        );
    }
}

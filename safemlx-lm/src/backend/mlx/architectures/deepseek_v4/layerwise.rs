//! Checkpoint-format-independent bounded-residency execution for DeepSeek V4.

use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
    sync::Arc,
    time::Instant,
};

use safemlx::{
    error::Exception,
    module::{Module, ModuleParameters, Param},
    ops::{
        broadcast_to, indexing::NewAxis, indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue,
    },
    Array, Dtype, Stream,
};

use crate::core::cache::{
    PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
    PromptCacheTopology,
};

use crate::{
    backend::mlx::error::Error,
    backend::mlx::nn::generation::CausalLm,
    backend::mlx::runtime::media::input,
    backend::mlx::runtime::{
        cache::residency::PagedCacheOptions,
        checkpoint::{
            binding::{
                build_module_bindings_with_recipes, populate_module_from_lease,
                populate_module_from_lease_excluding,
            },
            quantization::WeightQuantization,
            recipe::DerivedWeightRecipe,
            store::{GgufWeightStore, TensorSelection, WeightStore},
        },
        execution::layerwise::{
            load_layerwise_model_with_quantization,
            load_tensor_parallel_layerwise_model_with_quantization, open_safetensors_weight_store,
            ArchitectureAdapter, LayerWeightResidency, LayerwiseForwardState, LayerwiseModel,
            LoadTimeQuantizableAdapter, StaticUnitBindings,
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
        load_prompt_cache_with_identity, prompt_cache_model_identity, save_prompt_cache_with_rank,
        Cache, DecoderLayer, Model as ResidentModel, ModelArgs,
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

    /// Creates resident or explicitly bounded cache state independently of
    /// parameter residency.
    pub fn new_cache_with_options(
        &self,
        policy: crate::backend::mlx::runtime::cache::residency::CacheResidencyPolicy,
    ) -> Result<Cache, Error> {
        let rank = self.execution.prompt_cache_rank_identity();
        match policy {
            crate::backend::mlx::runtime::cache::residency::CacheResidencyPolicy::Device => {
                self.new_cache().map_err(Into::into)
            }
            crate::backend::mlx::runtime::cache::residency::CacheResidencyPolicy::Paged(
                options,
            ) => {
                let manager =
                    crate::backend::mlx::runtime::cache::residency::CacheResidencyManager::new(
                        options,
                    )
                    .map_err(|error| Error::Parallel(error.to_string()))?;
                self.execution
                    .adapter()
                    .static_model
                    .new_cache_with_manager(manager, rank)
                    .map_err(Into::into)
            }
        }
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

    /// Runs a rank-local tensor-parallel target pass through the generalized executor.
    pub(crate) fn forward_tensor_parallel(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution
            .forward_tensor_parallel(tokens, cache, group, stream)
    }

    /// Executes replicated layers while delegating routed experts to an EP owner.
    pub(crate) fn forward_with_expert_executor<F>(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut Cache,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        self.execution.forward_with_layer_executor(
            tokens,
            cache,
            stream,
            |_adapter, _group, index, layer, hidden, cache, context, stream| {
                let output = layer.forward_with_expert_executor(
                    hidden,
                    mask,
                    Some(&mut cache.layers[index]),
                    &context.input_ids,
                    stream,
                    |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                )?;
                capture_draft_hidden(&_adapter.args, index, &output, context, stream)?;
                Ok(output)
            },
        )
    }

    /// Executes TP-sharded attention while delegating routed experts to EP.
    pub(crate) fn forward_tensor_expert_parallel<F>(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut Cache,
        tensor_group: &safemlx::distributed::Group,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        self.execution.forward_tensor_parallel_with_layer_executor(
            tokens,
            cache,
            tensor_group,
            stream,
            |_adapter, _group, index, layer, hidden, cache, context, execution| {
                let group = execution.group().ok_or_else(|| {
                    Error::Parallel("DeepSeek V4 TP+EP execution has no TP group".into())
                })?;
                let output = layer.forward_tensor_with_expert_executor(
                    hidden,
                    mask,
                    Some(&mut cache.layers[index]),
                    &context.input_ids,
                    group,
                    execution.stream(),
                    |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                )?;
                capture_draft_hidden(&_adapter.args, index, &output, context, execution.stream())?;
                Ok(output)
            },
        )
    }

    /// Returns the active Cartesian topology and rank-local parameter accounting.
    pub fn parallel_info(
        &self,
    ) -> Option<&crate::backend::mlx::runtime::execution::layerwise::ParallelModelInfo> {
        self.execution.parallel_info()
    }

    pub(crate) fn bind_parallel_topology(
        &mut self,
        topology: crate::backend::mlx::MlxParallelContext,
    ) {
        self.execution.bind_parallel_topology(topology);
    }

    pub(crate) fn mtp_len(&self) -> usize {
        self.execution.adapter().static_model.mtp_len()
    }

    fn forward_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<crate::backend::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        let (logits, context) = self
            .execution
            .forward_with_context_hook(tokens, cache, stream, |_, _, _| Ok(()))
            .map_err(|error| Exception::custom(error.to_string()))?;
        let hidden = context.draft_hidden.ok_or_else(|| {
            Exception::custom("DeepSeek V4 layerwise pass did not retain draft hidden state")
        })?;
        Ok(
            crate::backend::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits,
                hidden,
                tokens: tokens.clone(),
            },
        )
    }

    pub(crate) fn forward_mtp_target_with_expert_executor<F>(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        tensor_group: Option<&safemlx::distributed::Group>,
        mut execute: F,
        stream: &Stream,
    ) -> Result<crate::backend::mlx::speculative::embedded::EmbeddedMtpOutput, Exception>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let (logits, context) = match tensor_group {
            Some(tensor_group) => self
                .execution
                .forward_tensor_parallel_with_layer_executor_and_context(
                    tokens,
                    cache,
                    tensor_group,
                    stream,
                    |_adapter, _group, index, layer, hidden, cache, context, execution| {
                        let group = execution.group().ok_or_else(|| {
                            Error::Parallel(
                                "DeepSeek V4 TP+EP MTP target is missing its TP group".into(),
                            )
                        })?;
                        let output = layer.forward_tensor_with_expert_executor(
                            hidden,
                            None,
                            Some(&mut cache.layers[index]),
                            &context.input_ids,
                            group,
                            execution.stream(),
                            |hidden, ids, weights, stream| {
                                execute(index, hidden, ids, weights, stream)
                            },
                        )?;
                        capture_draft_hidden(
                            &_adapter.args,
                            index,
                            &output,
                            context,
                            execution.stream(),
                        )?;
                        Ok(output)
                    },
                ),
            None => self.execution.forward_with_layer_executor_and_context(
                tokens,
                cache,
                stream,
                |_adapter, _group, index, layer, hidden, cache, context, stream| {
                    let output = layer.forward_with_expert_executor(
                        hidden,
                        None,
                        Some(&mut cache.layers[index]),
                        &context.input_ids,
                        stream,
                        |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                    )?;
                    capture_draft_hidden(&_adapter.args, index, &output, context, stream)?;
                    Ok(output)
                },
            ),
        }
        .map_err(|error| Exception::custom(error.to_string()))?;
        Ok(
            crate::backend::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits,
                hidden: context.draft_hidden.ok_or_else(|| {
                    Exception::custom("DeepSeek V4 EP pass did not retain MTP/DSpark hidden state")
                })?,
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
    ) -> Result<
        Option<crate::backend::mlx::runtime::execution::layerwise::DenseDiskStreamReport>,
        Error,
    > {
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

    /// Returns the exact generic prompt-cache layout including pooling state.
    pub(crate) fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        self.execution
            .adapter()
            .static_model
            .prompt_cache_layer_layout()
    }

    pub(crate) fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        self.execution.prompt_cache_model_identity()
    }

    /// Atomically persists target KV plus compressed pooling/indexer state.
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

    /// Restores target KV plus compressed pooling/indexer state exactly.
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

impl crate::backend::mlx::speculative::embedded::EmbeddedMtpTarget for DeepSeekV4LayerwiseModel {
    type Cache = Cache;
    type DraftCache = super::model::DraftCache;

    fn prefill_target(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<crate::backend::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        cache.reset()?;
        self.forward_mtp_target(&tokens, cache, stream)
    }

    fn verify_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<crate::backend::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        self.forward_mtp_target(tokens, cache, stream)
    }

    fn prefill_draft_cache(
        &mut self,
        output: &crate::backend::mlx::speculative::embedded::EmbeddedMtpOutput,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        <ResidentModel as crate::backend::mlx::speculative::embedded::EmbeddedMtpTarget>::prefill_draft_cache(
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

    fn restore_target_checkpoint(
        cache: &mut Self::Cache,
        checkpoint: &Self::Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        cache.restore_target_checkpoint(checkpoint, stream)
    }

    fn draft_logits(
        &mut self,
        hidden: &Array,
        last_token: u32,
        draft_index: usize,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        <ResidentModel as crate::backend::mlx::speculative::embedded::EmbeddedMtpTarget>::draft_logits(
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
        <ResidentModel as crate::backend::mlx::speculative::embedded::EmbeddedMtpTarget>::fused_draft_logits(
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
        <ResidentModel as crate::backend::mlx::speculative::embedded::EmbeddedMtpTarget>::adjust_fused_draft_logits(
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
        <ResidentModel as crate::backend::mlx::speculative::embedded::EmbeddedMtpTarget>::advance_draft_cache(
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
pub type Generate<'a, S = crate::backend::mlx::runtime::generation::sampler::DefaultSampler> =
    crate::backend::mlx::nn::generation::Generate<'a, DeepSeekV4LayerwiseModel, Cache, S>;

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
    parallel_topology: Option<crate::backend::mlx::MlxParallelContext>,
}

impl DeepSeekV4LayerwiseAdapter {
    pub(crate) fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let mut resident = ResidentModel::new(args.clone(), stream)?;
        // Decoder layers are residency units owned by `LayerwiseModel`. Keep only
        // the shared static target and draft modules in this holder.
        resident.model.layers.clear();
        Ok(Self {
            static_model: resident,
            sparse_expert_cache: false,
            expert_cache: None,
            parallel_topology: None,
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

    pub(crate) fn new_external_experts(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        Self::new_sparse(args, stream)
    }

    pub(crate) fn configure_cartesian_layout(
        &mut self,
        build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        layout: &crate::backend::mlx::runtime::distributed::parallel::LocalModelLayout,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.configure_parallel_static(build, layout, stream)
    }

    pub(crate) fn pipeline_static_mut(&mut self, role: &str) -> Option<&mut dyn ModuleParameters> {
        match role {
            "embedding" => Some(&mut self.static_model.model.embed_tokens),
            "norm" => Some(&mut self.static_model.model.norm),
            "hc_head" => Some(&mut self.static_model.model.hc_head),
            "output" => Some(&mut self.static_model.lm_head),
            "draft" => self
                .static_model
                .mtp
                .as_mut()
                .map(|module| module as &mut dyn ModuleParameters)
                .or_else(|| {
                    self.static_model
                        .dspark
                        .as_mut()
                        .map(|module| module as &mut dyn ModuleParameters)
                }),
            _ => None,
        }
    }

    pub(crate) fn pipeline_embed(
        &mut self,
        tokens: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let embedded = self
            .static_model
            .model
            .embed_tokens
            .forward(tokens, stream)?;
        let hidden = embedded.try_index_device((.., .., NewAxis, ..), stream)?;
        broadcast_to(
            &hidden,
            &[
                embedded.dim(0),
                embedded.dim(1),
                self.args.hc_mult,
                self.args.hidden_size,
            ],
            stream,
        )
    }

    pub(crate) fn pipeline_finish(
        &mut self,
        hidden: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let hidden = self.static_model.model.hc_head.forward(hidden, stream)?;
        let hidden = self.static_model.model.norm.forward(&hidden, stream)?;
        self.static_model.lm_head.forward(&hidden, stream)
    }

    pub(crate) fn embedded_mtp_len(&self) -> usize {
        self.static_model.mtp_len()
    }

    pub(crate) fn embedded_mtp_cache(&self) -> super::model::DraftCache {
        self.static_model
            .new_cache()
            .expect("validated DeepSeek V4 draft cache geometry")
            .mtp_layers
    }

    pub(crate) fn embedded_mtp_cache_with_manager(
        &self,
        manager: crate::backend::mlx::runtime::cache::residency::CacheResidencyManager,
        rank: Option<crate::CacheRankIdentity>,
    ) -> Result<super::model::DraftCache, Error> {
        Ok(self
            .static_model
            .new_cache_with_manager(manager, rank)?
            .mtp_layers)
    }

    pub(crate) fn prefill_pipeline_draft_cache(
        &mut self,
        output: &crate::backend::mlx::speculative::embedded::EmbeddedMtpOutput,
        tokens: &Array,
        cache: &mut super::model::DraftCache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let mut target_cache = self.static_model.new_cache()?;
        target_cache.mtp_layers.clone_from(cache);
        let pipeline_output;
        let output = if self.args.dspark.is_none() && output.hidden.ndim() == 3 {
            pipeline_output = crate::backend::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits: output.logits.clone(),
                hidden: output.hidden.reshape(
                    &[
                        output.hidden.dim(0),
                        output.hidden.dim(1),
                        self.args.hc_mult,
                        self.args.hidden_size,
                    ],
                    stream,
                )?,
                tokens: output.tokens.clone(),
            };
            &pipeline_output
        } else {
            output
        };
        <ResidentModel as crate::backend::mlx::speculative::embedded::EmbeddedMtpTarget>::prefill_draft_cache(
            &mut self.static_model,
            output,
            tokens,
            &mut target_cache,
            stream,
        )?;
        cache.clone_from(&target_cache.mtp_layers);
        Ok(())
    }

    pub(crate) fn forward_pipeline_mtp(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut super::model::DraftCache,
        stream: &Stream,
    ) -> Result<crate::backend::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        let hidden = if hidden.ndim() == 3 {
            hidden.reshape(
                &[
                    hidden.dim(0),
                    hidden.dim(1),
                    self.args.hc_mult,
                    self.args.hidden_size,
                ],
                stream,
            )?
        } else {
            hidden.clone()
        };
        let (logits, hidden) = self
            .static_model
            .forward_mtp_draft(&hidden, tokens, depth, cache, stream)?;
        let hidden = hidden.reshape(
            &[
                hidden.dim(0),
                hidden.dim(1),
                self.args.hc_mult * self.args.hidden_size,
            ],
            stream,
        )?;
        Ok(
            crate::backend::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits,
                hidden,
                tokens: tokens.clone(),
            },
        )
    }

    pub(crate) fn fused_pipeline_draft_logits(
        &mut self,
        hidden: &Array,
        last_token: u32,
        proposal_capacity: usize,
        cache: &mut super::model::DraftCache,
        stream: &Stream,
    ) -> Result<Option<Array>, Exception> {
        <ResidentModel as crate::backend::mlx::speculative::embedded::EmbeddedMtpTarget>::fused_draft_logits(
            &mut self.static_model,
            hidden,
            last_token,
            proposal_capacity,
            cache,
            stream,
        )
    }

    pub(crate) fn adjust_pipeline_fused_logits(
        &mut self,
        logits: Array,
        last_token: u32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        <ResidentModel as crate::backend::mlx::speculative::embedded::EmbeddedMtpTarget>::adjust_fused_draft_logits(
            &mut self.static_model,
            logits,
            last_token,
            stream,
        )
    }

    pub(crate) fn advance_pipeline_draft_cache(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut super::model::DraftCache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let hidden = if self.args.dspark.is_none() && hidden.ndim() == 3 {
            hidden.reshape(
                &[
                    hidden.dim(0),
                    hidden.dim(1),
                    self.args.hc_mult,
                    self.args.hidden_size,
                ],
                stream,
            )?
        } else {
            hidden.clone()
        };
        <ResidentModel as crate::backend::mlx::speculative::embedded::EmbeddedMtpTarget>::advance_draft_cache(
            &mut self.static_model,
            &hidden,
            tokens,
            cache,
            stream,
        )
    }
}

impl LoadTimeQuantizableAdapter for DeepSeekV4LayerwiseAdapter {
    fn load_time_quantized(
        &self,
        quantization: WeightQuantization,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let args = self.args.with_load_time_quantization(quantization)?;
        if self.sparse_expert_cache {
            Self::new_sparse(args, stream)
        } else {
            Self::new(args, stream)
        }
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

    fn quantization(&self) -> Option<WeightQuantization> {
        self.args.quantization
    }

    fn quantizes_static_binding(&self, binding: &WeightBinding) -> bool {
        let target = binding.logical_target().unwrap_or(binding.checkpoint_key());
        if target == "head.weight" {
            return true;
        }
        if !target.starts_with("mtp.") {
            return false;
        }
        if target.ends_with(".gate_up_proj") || target.ends_with(".down_proj") {
            return true;
        }
        target.ends_with(".weight")
            && !target.ends_with("norm.weight")
            && !target.contains(".markov_w1.")
    }

    fn prompt_cache_model_identity(
        &self,
        topology: Option<crate::backend::mlx::MlxParallelContext>,
    ) -> Result<PromptCacheModelIdentity, Error> {
        prompt_cache_model_identity(
            &self.args,
            topology.map_or_else(
                PromptCacheTopology::default,
                crate::backend::mlx::cache::prompt_cache_topology,
            ),
        )
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
        save_prompt_cache_with_rank(
            cache,
            destination,
            descriptor.clone(),
            prefix_token_ids,
            options,
            descriptor.topology.cache_rank_identity(),
            stream,
        )
    }

    fn load_prompt_cache(
        &self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(Self::Cache, PromptCacheManifest), Error> {
        load_prompt_cache_with_identity(
            &self.args,
            directory,
            expected,
            prefix_token_ids,
            identity.clone(),
            options,
            stream,
        )
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
    ) -> Result<crate::backend::mlx::runtime::execution::layerwise::ExecutionGroupDag, Error> {
        crate::backend::mlx::runtime::execution::layerwise::ExecutionGroupDag::chain([
            "text_decoder",
        ])
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

    fn register_parallel_parameters(
        &self,
        context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        planner: &mut crate::backend::mlx::runtime::distributed::parallel::ParallelPlanBuilder,
        stream: &Stream,
    ) -> Result<(), Error> {
        let _ = context;
        use crate::backend::mlx::runtime::distributed::parallel::register_replicated_module;

        register_replicated_module(planner, &self.static_model.model.embed_tokens, "embed")?;
        register_replicated_module(planner, &self.static_model.model.norm, "norm")?;
        register_replicated_module(planner, &self.static_model.model.hc_head, "hc_head")?;
        register_replicated_module(planner, &self.static_model.lm_head, "head")?;
        if let Some(mtp) = &self.static_model.mtp {
            register_replicated_module(planner, mtp, "mtp")?;
        }
        if let Some(dspark) = &self.static_model.dspark {
            register_replicated_module(planner, dspark, "mtp")?;
        }
        for index in 0..self.args.num_hidden_layers as usize {
            let layer = DecoderLayer::new(&self.args, index, stream)?;
            register_v4_layer_parallel_plan(planner, &layer, index)?;
        }
        Ok(())
    }

    fn configure_parallel_static(
        &mut self,
        context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        _layout: &crate::backend::mlx::runtime::distributed::parallel::LocalModelLayout,
        _stream: &Stream,
    ) -> Result<(), Error> {
        self.parallel_topology = Some(context.topology());
        Ok(())
    }

    fn new_parallel_layer(
        &self,
        group: usize,
        index: usize,
        layout: &crate::backend::mlx::runtime::distributed::parallel::LocalModelLayout,
        stream: &Stream,
    ) -> Result<DecoderLayer, Error> {
        self.layer_count(group)?;
        let target = format!("layers.{index}.attn.wq_b.weight");
        let query = layout.tensor(&target).ok_or_else(|| {
            Error::Parallel(format!("missing DeepSeek V4 TP layout for {target}"))
        })?;
        if query.fell_back_to_replication() {
            return Ok(DecoderLayer::new(&self.args, index, stream)?);
        }
        let units = query.logical_units().ok_or_else(|| {
            Error::Parallel(format!("DeepSeek V4 TP query {target} has no head domain"))
        })?;
        let global_heads = usize::try_from(self.args.num_attention_heads)
            .map_err(|_| Error::Parallel("DeepSeek V4 head count exceeds usize".into()))?;
        if units == 0 || !global_heads.is_multiple_of(units) {
            return Err(Error::Parallel(format!(
                "DeepSeek V4 query head domain {units} does not divide {global_heads} heads"
            )));
        }
        let topology = self.parallel_topology.ok_or_else(|| {
            Error::Parallel("DeepSeek V4 TP layer was built before topology binding".into())
        })?;
        let heads_per_unit = global_heads / units;
        let widths = (0..topology.tensor_parallel_size)
            .map(|rank| {
                crate::core::balanced_contiguous_range(
                    units,
                    topology.tensor_parallel_size,
                    rank,
                    false,
                )
                .map(|range| range.len() * heads_per_unit)
                .map_err(|error| Error::Parallel(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let local_heads = i32::try_from(widths[topology.tensor_parallel_rank])
            .map_err(|_| Error::Parallel("DeepSeek V4 local head count exceeds i32".into()))?;
        Ok(DecoderLayer::new_parallel(
            &self.args,
            index,
            local_heads,
            widths,
            stream,
        )?)
    }

    fn new_expert_parallel_layer(
        &self,
        group: usize,
        index: usize,
        _assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<DecoderLayer, Error> {
        self.new_layer(group, index, stream)
    }

    fn new_tensor_expert_parallel_layer(
        &self,
        group: usize,
        index: usize,
        layout: &crate::backend::mlx::runtime::distributed::parallel::LocalModelLayout,
        _assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<DecoderLayer, Error> {
        self.new_parallel_layer(group, index, layout, stream)
    }

    fn expert_parallel_assignment(
        &self,
        topology: crate::backend::mlx::MlxParallelContext,
    ) -> Result<Option<crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>, Error>
    {
        if topology.expert_parallel_size == 1 && !self.sparse_expert_cache {
            return Ok(None);
        }
        if self.args.n_routed_experts <= 0 {
            return Err(Error::Parallel(
                "DeepSeek V4 PP+EP requires routed experts".into(),
            ));
        }
        Ok(Some(
            crate::backend::mlx::runtime::distributed::expert::ExpertAssignment::balanced(
                self.args.n_routed_experts as usize,
                topology.expert_parallel_size,
                topology.expert_parallel_rank,
            )?,
        ))
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

    fn parallel_layer_bindings(
        &self,
        group: usize,
        index: usize,
        _layer: &DecoderLayer,
        store: &dyn WeightStore,
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
        store: &dyn WeightStore,
        assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        let global = self.new_layer(group, index, stream)?;
        let indices = assignment.local_global_expert_ids().to_vec();
        self.layer_bindings(group, index, &global, store)?
            .into_iter()
            .map(|binding| {
                let target = binding.logical_target().unwrap_or_else(|| binding.name());
                if target.starts_with("ffn.switch_mlp.") {
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
                .filter(|key| is_routed_expert_source(key))
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
        capture_draft_hidden(&self.args, index, &output, context, stream)?;
        Ok(output)
    }

    fn forward_layer_with_execution(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut DecoderLayer,
        hidden: &Array,
        cache: &mut Cache,
        context: &mut DeepSeekV4ForwardContext,
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
        let output = if let Some(expert_cache) = &self.expert_cache {
            let pass = if hidden.dim(1) > 1 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            };
            layer.forward_tensor_with_expert_executor(
                hidden,
                None,
                Some(&mut cache.layers[index]),
                &context.input_ids,
                tp_group,
                execution.stream(),
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
            layer.forward_tensor_parallel(
                hidden,
                None,
                Some(&mut cache.layers[index]),
                &context.input_ids,
                tp_group,
                execution.stream(),
            )?
        };
        capture_draft_hidden(&self.args, index, &output, context, execution.stream())?;
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

fn is_routed_expert_source(key: &str) -> bool {
    key.contains(".ffn.experts.") || key.contains(".ffn.expert_banks.")
}

/// Loads V4 with resident, host-windowed, or dense disk-streamed layers.
pub fn load_deepseek_v4_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    requested_quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV4LayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let args = super::model::get_model_args(model_dir)?;
    let quantization =
        args.resolve_load_time_quantization("DeepSeek V4", requested_quantization)?;
    let adapter = DeepSeekV4LayerwiseAdapter::new(args, stream)?;
    Ok(DeepSeekV4LayerwiseModel {
        execution: load_layerwise_model_with_quantization(
            open_safetensors_weight_store(model_dir, options.max_mapped_shards())?,
            adapter,
            options,
            quantization,
            stream,
            weights_stream,
        )?,
    })
}

/// Loads a canonical llama.cpp `deepseek4` GGUF through the generalized
/// resident/layerwise/dense-stream and independent-expert-cache engine.
pub(crate) fn load_deepseek_v4_gguf_layerwise_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    residency: crate::backend::mlx::runtime::execution::layerwise::WeightResidency,
    requested_quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(DeepSeekV4LayerwiseModel, Vec<u32>), Error> {
    crate::backend::mlx::structural::validate_gguf(
        crate::core::GgufArchitecture::DeepSeek4,
        checkpoint,
        metadata,
        crate::backend::mlx::ModelLoadOptions::default().with_weight_residency(residency),
    )
    .into_loader_result()?;
    let prepared = super::model::prepare_gguf_checkpoint(checkpoint, metadata)?;
    let quantization = prepared
        .args
        .resolve_load_time_quantization("DeepSeek V4 GGUF", requested_quantization)?;
    let store: Arc<dyn WeightStore + Send + Sync> =
        Arc::new(GgufWeightStore::new_with_max_mapped_shards(
            checkpoint.clone(),
            super::model::translate_gguf_weight_name,
            residency.max_mapped_shards(),
        )?);
    let mut execution = load_layerwise_model_with_quantization(
        Arc::clone(&store),
        if residency.expert_cache().is_some() {
            DeepSeekV4LayerwiseAdapter::new_sparse(prepared.args.clone(), stream)?
        } else {
            DeepSeekV4LayerwiseAdapter::new(prepared.args.clone(), stream)?
        },
        residency.layers(),
        quantization,
        stream,
        weights_stream,
    )?;
    if let Some(options) = residency.expert_cache() {
        let entries = expert_catalog(&prepared.args, store.as_ref())?;
        execution.adapter_mut().expert_cache = Some(match quantization {
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
    }
    Ok((
        DeepSeekV4LayerwiseModel { execution },
        prepared.eos_token_ids,
    ))
}

/// Loads a canonical `deepseek4` GGUF with TP composed with any layer residency.
pub(crate) fn load_deepseek_v4_gguf_tensor_parallel_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: LayerWeightResidency,
    requested_quantization: Option<WeightQuantization>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(DeepSeekV4LayerwiseModel, Vec<u32>), Error> {
    let residency = options.weight_residency();
    crate::backend::mlx::structural::validate_gguf(
        crate::core::GgufArchitecture::DeepSeek4,
        checkpoint,
        metadata,
        crate::backend::mlx::ModelLoadOptions::default().with_weight_residency(residency),
    )
    .into_loader_result()?;
    let prepared = super::model::prepare_gguf_checkpoint(checkpoint, metadata)?;
    let quantization = prepared.args.resolve_load_time_quantization(
        "DeepSeek V4 GGUF tensor parallel",
        requested_quantization,
    )?;
    let store: Arc<dyn WeightStore + Send + Sync> =
        Arc::new(GgufWeightStore::new_with_max_mapped_shards(
            checkpoint.clone(),
            super::model::translate_gguf_weight_name,
            options.max_mapped_shards(),
        )?);
    let execution = load_tensor_parallel_layerwise_model_with_quantization(
        store,
        DeepSeekV4LayerwiseAdapter::new(prepared.args, stream)?,
        options,
        quantization,
        build,
        stream,
        weights_stream,
    )?;
    Ok((
        DeepSeekV4LayerwiseModel { execution },
        prepared.eos_token_ids,
    ))
}

/// Loads V4 through the generalized tensor-parallel and residency engine.
pub(crate) fn load_deepseek_v4_tensor_parallel_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    requested_quantization: Option<WeightQuantization>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV4LayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    if model_dir
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        let checkpoint = GgufCheckpoint::open(model_dir)?;
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        return load_deepseek_v4_gguf_tensor_parallel_model(
            &checkpoint,
            &metadata,
            options,
            requested_quantization,
            build,
            stream,
            weights_stream,
        )
        .map(|(model, _)| model);
    }
    crate::backend::mlx::structural::validate_safetensors_load_path(
        crate::core::ModelKind::DeepSeekV4,
        model_dir,
        crate::backend::mlx::ModelLoadOptions::default()
            .with_weight_residency(options.weight_residency()),
    )?;
    let args = super::model::get_model_args(model_dir)?;
    let quantization =
        args.resolve_load_time_quantization("DeepSeek V4 tensor parallel", requested_quantization)?;
    let adapter = DeepSeekV4LayerwiseAdapter::new(args, stream)?;
    Ok(DeepSeekV4LayerwiseModel {
        execution: load_tensor_parallel_layerwise_model_with_quantization(
            open_safetensors_weight_store(model_dir, options.max_mapped_shards())?,
            adapter,
            options,
            quantization,
            build,
            stream,
            weights_stream,
        )?,
    })
}

/// Loads V4 with routed experts in independent cache units.
pub fn load_deepseek_v4_expert_cache_model(
    model_dir: impl AsRef<Path>,
    non_expert: crate::backend::mlx::runtime::execution::layerwise::NonExpertWeightResidency,
    options: ExpertCacheLoadOptions,
    requested_quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV4LayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = super::model::get_model_args(model_dir)?;
    let quantization = args.resolve_load_time_quantization(
        "DeepSeek V4 independent expert cache",
        requested_quantization,
    )?;
    let store = open_safetensors_weight_store(model_dir, non_expert.layers().max_mapped_shards())?;
    let adapter = DeepSeekV4LayerwiseAdapter::new_sparse(args.clone(), stream)?;
    let mut execution = load_layerwise_model_with_quantization(
        Arc::clone(&store),
        adapter,
        non_expert,
        quantization,
        stream,
        weights_stream,
    )?;
    let entries = expert_catalog(&args, store.as_ref())?;
    execution.adapter_mut().expert_cache = Some(match quantization {
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
    Ok(DeepSeekV4LayerwiseModel { execution })
}

/// Builds the non-expert V4 base used by pure EP execution.
pub(crate) fn load_deepseek_v4_sparse_ep_base_with_store(
    store: Arc<dyn WeightStore + Send + Sync>,
    args: ModelArgs,
    non_expert: impl Into<LayerWeightResidency>,
    requested_quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV4LayerwiseModel, Error> {
    let quantization =
        args.resolve_load_time_quantization("DeepSeek V4 expert parallel", requested_quantization)?;
    let adapter = DeepSeekV4LayerwiseAdapter::new_sparse(args, stream)?;
    Ok(DeepSeekV4LayerwiseModel {
        execution: load_layerwise_model_with_quantization(
            store,
            adapter,
            non_expert,
            quantization,
            stream,
            weights_stream,
        )?,
    })
}

/// Builds the TP-sharded non-expert V4 base used by TP+EP execution.
pub(crate) fn load_deepseek_v4_sparse_tp_ep_base_with_store(
    store: Arc<dyn WeightStore + Send + Sync>,
    args: ModelArgs,
    non_expert: impl Into<LayerWeightResidency>,
    requested_quantization: Option<WeightQuantization>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV4LayerwiseModel, Error> {
    let quantization = args.resolve_load_time_quantization(
        "DeepSeek V4 tensor/expert parallel",
        requested_quantization,
    )?;
    let adapter = DeepSeekV4LayerwiseAdapter::new_sparse(args, stream)?;
    Ok(DeepSeekV4LayerwiseModel {
        execution: load_tensor_parallel_layerwise_model_with_quantization(
            store,
            adapter,
            non_expert,
            quantization,
            build,
            stream,
            weights_stream,
        )?,
    })
}

fn capture_draft_hidden(
    args: &ModelArgs,
    index: usize,
    output: &Array,
    context: &mut DeepSeekV4ForwardContext,
    stream: &Stream,
) -> Result<(), Error> {
    if let Some(dspark) = &args.dspark {
        if let Some(position) = dspark
            .target_layer_ids
            .iter()
            .position(|wanted| *wanted == index as i32)
        {
            context
                .captures
                .push((position, safemlx::ops::mean_axis(output, 2, false, stream)?));
        }
    }
    if index + 1 == args.num_hidden_layers as usize {
        context.draft_hidden = if args.dspark.is_some() {
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
    Ok(())
}

fn register_v4_layer_parallel_plan(
    planner: &mut crate::backend::mlx::runtime::distributed::parallel::ParallelPlanBuilder,
    layer: &DecoderLayer,
    index: usize,
) -> Result<(), Error> {
    use crate::backend::mlx::runtime::distributed::parallel::{
        array_parameter_member, partitioned_projection_members, register_replicated_module,
        MemberSharding, ParameterGroupSpec, ParameterRole, ProjectionSharding,
    };

    let prefix = format!("layers.{index}");
    let attention = &layer.attn;
    let query_prefix = format!("{prefix}.attn.wq_b");
    let preferred_heads = usize::try_from(attention.heads)
        .map_err(|_| Error::Parallel("DeepSeek V4 query head count exceeds usize".into()))?;
    let (units, mut members) = partitioned_projection_members(
        &[(
            &attention.wq_b,
            query_prefix.as_str(),
            ProjectionSharding::Column,
        )],
        preferred_heads,
    )?;
    members.push(array_parameter_member(
        format!("{prefix}.attn.attn_sink"),
        attention.attn_sink.as_ref(),
        MemberSharding::Partitioned { axis: 0 },
    )?);
    planner.register(ParameterGroupSpec::partitioned(
        format!("{prefix}.attn.query_heads"),
        ParameterRole::AttentionHeads,
        units,
        members,
    )?)?;

    register_replicated_module(planner, &attention.wq_a, &format!("{prefix}.attn.wq_a"))?;
    register_replicated_module(planner, &attention.q_norm, &format!("{prefix}.attn.q_norm"))?;
    register_replicated_module(planner, &attention.wkv, &format!("{prefix}.attn.wkv"))?;
    register_replicated_module(
        planner,
        &attention.kv_norm,
        &format!("{prefix}.attn.kv_norm"),
    )?;
    register_replicated_module(planner, &attention.wo_a, &format!("{prefix}.attn.wo_a"))?;
    register_replicated_module(planner, &attention.wo_b, &format!("{prefix}.attn.wo_b"))?;
    register_replicated_module(planner, &layer.ffn, &format!("{prefix}.ffn"))?;
    register_replicated_module(planner, &layer.attn_norm, &format!("{prefix}.attn_norm"))?;
    register_replicated_module(planner, &layer.ffn_norm, &format!("{prefix}.ffn_norm"))?;
    register_replicated_module(planner, &layer.attn_hc, &format!("{prefix}.attn_hc"))?;
    register_replicated_module(planner, &layer.ffn_hc, &format!("{prefix}.ffn_hc"))?;
    if let Some(compressor) = &attention.compressor {
        register_replicated_module(planner, compressor, &format!("{prefix}.attn.compressor"))?;
    }
    if let Some(indexer) = &attention.indexer {
        register_replicated_module(planner, indexer, &format!("{prefix}.attn.indexer"))?;
    }
    Ok(())
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
        "scales" => (
            format!("{root}.attn.wo_a.scales"),
            args.o_lora_rank as usize,
        ),
        "biases" => (
            format!("{root}.attn.wo_a.biases"),
            args.o_lora_rank as usize,
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
    let bank = |projection: &str| format!("{root}.ffn.expert_banks.{projection}.{component}");
    if store.metadata(&bank("w1")).is_ok() {
        let source =
            |projection: &str| DerivedWeightRecipe::source(bank(projection), TensorSelection::Full);
        let recipe = if gate_up {
            DerivedWeightRecipe::Concatenate {
                axis: 1,
                inputs: vec![source("w1"), source("w3")],
            }
        } else {
            source("w2")
        };
        recipe.infer(store)?;
        return Ok(Some(recipe));
    }
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
    linear: &crate::backend::mlx::architectures::qwen::hybrid::qwen3_5::QwenLinear,
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

pub(crate) fn expert_catalog(
    args: &ModelArgs,
    store: &dyn WeightStore,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let mut entries = Vec::new();
    for layer in 0..args.num_hidden_layers as usize {
        let root = format!("layers.{layer}.ffn.experts");
        let bank_root = format!("layers.{layer}.ffn.expert_banks");
        let fused_banks = store.metadata(&format!("{bank_root}.w1.weight")).is_ok();
        for expert in 0..args.n_routed_experts as usize {
            let mut bindings = Vec::new();
            for (name, gate_up, component) in [
                ("gate_up_proj", true, "weight"),
                ("down_proj", false, "weight"),
                ("gate_up_proj_scales", true, "scale"),
                ("down_proj_scales", false, "scale"),
            ] {
                let scale_probe = if fused_banks {
                    format!("{bank_root}.w1.scale")
                } else {
                    format!("{root}.{expert}.w1.scale")
                };
                if component == "scale"
                    && args.expert_dtype.as_deref().is_none()
                    && store.metadata(&scale_probe).is_err()
                {
                    continue;
                }
                let source = |projection: &str| {
                    DerivedWeightRecipe::source(
                        if fused_banks {
                            format!("{bank_root}.{projection}.{component}")
                        } else {
                            format!("{root}.{expert}.{projection}.{component}")
                        },
                        if fused_banks {
                            TensorSelection::Range {
                                axis: 0,
                                start: expert,
                                end: expert + 1,
                            }
                        } else {
                            TensorSelection::Full
                        },
                    )
                };
                let mut recipe = if gate_up {
                    let recipe = DerivedWeightRecipe::Concatenate {
                        axis: 1,
                        inputs: vec![source("w1"), source("w3")],
                    };
                    if fused_banks {
                        recipe
                    } else {
                        DerivedWeightRecipe::Stack {
                            axis: 0,
                            inputs: vec![DerivedWeightRecipe::Concatenate {
                                axis: 0,
                                inputs: vec![source("w1"), source("w3")],
                            }],
                        }
                    }
                } else if fused_banks {
                    source("w2")
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
                let quantization = match args.expert_dtype.as_deref() {
                    Some("fp4") => Some(WeightQuantization::MxFp4),
                    Some("fp8") => None,
                    None => args.quantization,
                    Some(_) => unreachable!("validated expert dtype"),
                };
                let bank = crate::backend::mlx::nn::moe::PackedSwiGluExperts::new(
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
    use crate::backend::mlx::{DeviceAssignment, MlxParallelContext};
    use safemlx::{module::ModuleParameters, Device, DeviceType, Dtype, ExecutionContext};

    use super::{raw_layer_key, DeepSeekV4LayerwiseAdapter};
    use crate::{
        backend::mlx::architectures::deepseek_v4::model::ModelArgs,
        backend::mlx::runtime::{
            checkpoint::quantization::WeightQuantization,
            checkpoint::store::TensorSelection,
            distributed::parallel::{ParallelBuildContext, ShardingPolicy},
            execution::layerwise::{ArchitectureAdapter, LoadTimeQuantizableAdapter},
            residency::manager::WeightBinding,
        },
    };

    fn args() -> ModelArgs {
        ModelArgs::from_value(serde_json::json!({
            "model_type": "deepseek_v4",
            "hidden_size": 8,
            "moe_intermediate_size": 4,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "qk_rope_head_dim": 4,
            "q_lora_rank": 8,
            "o_lora_rank": 8,
            "o_groups": 2,
            "vocab_size": 32,
            "max_position_embeddings": 4096,
            "compress_ratios": [0, 4],
            "index_n_heads": 4,
            "index_head_dim": 4,
            "index_topk": 2,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2
        }))
        .unwrap()
    }

    fn quantizable_args(dspark: bool) -> ModelArgs {
        let mut value = serde_json::json!({
            "model_type": "deepseek_v4",
            "hidden_size": 32,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "qk_rope_head_dim": 4,
            "q_lora_rank": 32,
            "o_lora_rank": 32,
            "o_groups": 2,
            "vocab_size": 64,
            "max_position_embeddings": 4096,
            "compress_ratios": [0, 0],
            "index_n_heads": 8,
            "index_head_dim": 8,
            "index_topk": 2,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "num_nextn_predict_layers": 1
        });
        if dspark {
            value["dspark_block_size"] = 4.into();
            value["dspark_noise_token_id"] = 0.into();
            value["dspark_target_layer_ids"] = serde_json::json!([0]);
            value["dspark_markov_rank"] = 32.into();
        }
        ModelArgs::from_value(value).unwrap()
    }

    fn assert_packed_parameter(module: &impl ModuleParameters, suffix: &str) {
        let parameters = module.parameters().flatten();
        let (_, parameter) = parameters
            .iter()
            .find(|(name, _)| name.ends_with(suffix))
            .unwrap_or_else(|| panic!("missing packed parameter ending in {suffix:?}"));
        assert_eq!(parameter.dtype(), Dtype::Uint32, "{suffix}");
    }

    fn binding(target: &str) -> WeightBinding {
        WeightBinding::new("parameter", target, TensorSelection::Full, 4).unwrap()
    }

    #[test]
    fn static_quantization_selects_only_packed_v4_modules() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let adapter =
            DeepSeekV4LayerwiseAdapter::new(quantizable_args(true), execution.stream()).unwrap();

        assert!(adapter.quantizes_static_binding(&binding("head.weight")));
        assert!(adapter.quantizes_static_binding(&binding("mtp.main_proj.weight")));
        assert!(adapter.quantizes_static_binding(&binding(
            "mtp.layers.0.decoder.ffn.switch_mlp.gate_up_proj"
        )));
        assert!(!adapter.quantizes_static_binding(&binding("embed.weight")));
        assert!(!adapter.quantizes_static_binding(&binding("hc_head.function")));
        assert!(!adapter.quantizes_static_binding(&binding("mtp.main_norm.weight")));
        assert!(!adapter.quantizes_static_binding(&binding("mtp.markov_w1.weight")));
        assert!(!adapter.quantizes_static_binding(&binding("mtp.layers.0.attn.compressor.ape")));
    }

    #[test]
    fn uniform_load_time_format_covers_target_moe_and_embedded_mtp() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let source =
            DeepSeekV4LayerwiseAdapter::new(quantizable_args(false), execution.stream()).unwrap();
        let target = source
            .load_time_quantized(WeightQuantization::MxFp4, execution.stream())
            .unwrap();
        let layer = target.new_layer(0, 0, execution.stream()).unwrap();

        assert_eq!(target.quantization(), Some(WeightQuantization::MxFp4));
        assert_packed_parameter(&layer, "attn.wq_a.weight");
        assert_packed_parameter(&layer, "ffn.switch_mlp.gate_up_proj");
        assert_packed_parameter(&layer, "ffn.shared_experts.gate_proj.weight");
        assert_packed_parameter(
            target.static_model.mtp.as_ref().unwrap(),
            "layers.0.e_proj.weight",
        );
        assert_packed_parameter(
            target.static_model.mtp.as_ref().unwrap(),
            "layers.0.decoder.ffn.switch_mlp.down_proj",
        );
        assert_packed_parameter(&target.static_model.lm_head, "weight");
    }

    #[test]
    fn uniform_load_time_format_covers_fused_dspark_projections() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let source =
            DeepSeekV4LayerwiseAdapter::new(quantizable_args(true), execution.stream()).unwrap();
        let target = source
            .load_time_quantized(WeightQuantization::MxFp4, execution.stream())
            .unwrap();
        let dspark = target.static_model.dspark.as_ref().unwrap();

        assert_packed_parameter(dspark, "main_proj.weight");
        assert_packed_parameter(dspark, "markov_w2.weight");
        assert_packed_parameter(dspark, "confidence_head.weight");
        assert_packed_parameter(dspark, "layers.0.attn.wq_a.weight");
        assert_packed_parameter(dspark, "layers.0.ffn.switch_mlp.down_proj");
    }

    #[test]
    fn checkpoint_native_formats_reject_implicit_transcoding() {
        let mut args = quantizable_args(false);
        args.expert_dtype = Some("fp4".into());
        let error = args
            .resolve_load_time_quantization("DeepSeek V4", Some(WeightQuantization::MxFp4))
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("implicit dequantization and requantization"));
    }

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

    #[test]
    fn cartesian_plan_shards_query_heads_and_balances_expert_ownership() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        for rank in 0..8 {
            let topology = MlxParallelContext::for_rank(
                rank,
                2,
                2,
                2,
                DeviceAssignment::new(DeviceType::Cpu, 0),
            )
            .unwrap();
            let build = ParallelBuildContext::new(topology, ShardingPolicy::Require);
            let adapter = DeepSeekV4LayerwiseAdapter::new(args(), execution.stream()).unwrap();
            let assignment = adapter
                .expert_parallel_assignment(topology)
                .unwrap()
                .unwrap();
            assert_eq!(assignment.global_expert_count(), 8);
            assert_eq!(assignment.local_expert_count(), 4);

            let mut planner = build.planner();
            adapter
                .register_parallel_parameters(build, &mut planner, execution.stream())
                .unwrap();
            let (_, layout) = planner.finish().unwrap();
            assert_eq!(
                layout
                    .tensor("layers.0.attn.wq_b.weight")
                    .unwrap()
                    .local_shape(),
                &[16, 8]
            );
            assert_eq!(
                layout
                    .tensor("layers.0.attn.attn_sink")
                    .unwrap()
                    .local_shape(),
                &[2]
            );
            assert_eq!(
                layout
                    .tensor("layers.0.attn.wkv.weight")
                    .unwrap()
                    .local_shape(),
                &[8, 8]
            );
        }
    }
}

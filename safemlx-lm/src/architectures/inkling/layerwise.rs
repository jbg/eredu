//! Text-decoder bounded layer execution for Thinking Machines Lab Inkling.

use std::{
    collections::{BTreeMap, HashMap},
    ops::Range,
    path::Path,
    sync::Arc,
    time::Instant,
};

use safemlx::{
    error::Exception,
    macros::ModuleParameters,
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
        build_module_binding_plan_with_recipes, build_module_binding_plan_with_recipes_excluding,
        packed_companion_checkpoint_name, populate_module_from_lease,
        populate_module_from_lease_excluding,
    },
    runtime::checkpoint::binding_plan::{BindingPlan, PlannedBinding},
    runtime::checkpoint::store::{GgufWeightStore, TensorSelection, WeightStore},
    runtime::checkpoint::{
        quantization::{should_quantize_on_load, WeightQuantization},
        recipe::DerivedWeightRecipe,
    },
    runtime::execution::layerwise::{
        load_layerwise_model, load_layerwise_model_with_quantization,
        load_tensor_parallel_layerwise_model, open_safetensors_weight_store, ArchitectureAdapter,
        ExecutionGroupDag, LayerWeightResidency, LayerwiseForwardState, LayerwiseModel,
        LoadTimeQuantizableAdapter, StaticUnitBindings, WeightResidency,
    },
    runtime::residency::expert_cache::{
        AcquiredExperts, ExpertCache, ExpertCacheError, ExpertCacheLoadOptions, ExpertCacheReport,
        ExpertCatalogEntry, ExpertIdentity, ExpertPass, ExpertRouteBatch,
    },
    runtime::residency::manager::{OffloadUnit, ResidencyReport, ResidentUnitLease, WeightBinding},
};

const EMBEDDING_UNIT: &str = "inkling.static.embedding";
const EMBED_NORM_UNIT: &str = "inkling.static.embed_norm";
const NORM_UNIT: &str = "inkling.static.norm";
const HEAD_UNIT: &str = "inkling.static.output";
const AUDIO_UNIT: &str = "inkling.static.audio";
const VISION_NORM_UNIT: &str = "inkling.static.vision_norm";
const MTP_UNIT: &str = "inkling.static.mtp";

#[derive(Debug, Clone, ModuleParameters)]
struct InklingMtpDepth {
    #[param]
    hidden_norm: nn::RmsNorm,
    #[param]
    embed_norm: nn::RmsNorm,
    #[param]
    input_proj: MaybeQuantized<nn::Linear>,
    #[param]
    transformer_block: DecoderLayer,
}

#[derive(Debug, Clone, ModuleParameters)]
struct InklingMtpModule {
    #[param]
    layers: Vec<InklingMtpDepth>,
    #[param]
    chain_norm: Option<nn::RmsNorm>,
    policies: Vec<crate::AttentionPolicy>,
}

fn inkling_mtp_text_args(
    args: &ModelArgs,
    config: &resident::InklingMtpConfig,
    attention: crate::AttentionPolicy,
) -> Result<resident::TextArgs, Error> {
    let mut text = args.text_config.clone();
    text.num_attention_heads = config
        .num_attention_heads
        .unwrap_or(text.num_attention_heads);
    text.num_key_value_heads = config
        .num_key_value_heads
        .unwrap_or(text.num_key_value_heads);
    text.head_dim = config.head_dim.unwrap_or(text.head_dim);
    text.swa_num_attention_heads = config
        .swa_num_attention_heads
        .or(text.swa_num_attention_heads);
    text.swa_num_key_value_heads = config
        .swa_num_key_value_heads
        .or(text.swa_num_key_value_heads);
    text.swa_head_dim = config.swa_head_dim.or(text.swa_head_dim);
    text.dense_intermediate_size = config
        .dense_intermediate_size
        .or(text.dense_intermediate_size);
    text.intermediate_size = config.intermediate_size.unwrap_or(text.intermediate_size);
    text.sconv_kernel_size = config.sconv_kernel_size.unwrap_or(text.sconv_kernel_size);
    text.rel_extent = config.rel_extent.unwrap_or(text.rel_extent);
    text.d_rel = config.d_rel.unwrap_or(text.d_rel);
    text.layer_schedule = crate::LayerSchedule::new(
        1,
        vec![resident::LayerPolicy {
            attention,
            feed_forward: resident::FeedForwardPolicy::Dense,
        }],
    )
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    text.num_hidden_layers = 1;
    Ok(text)
}

impl InklingMtpModule {
    fn new(args: &ModelArgs, stream: &Stream) -> Result<Option<Self>, Error> {
        let Some(config) = args.mtp_config.as_ref() else {
            return Ok(None);
        };
        let count = usize::try_from(config.num_nextn_predict_layers).map_err(|_| {
            Error::UnsupportedArchitecture("Inkling MTP layer count is negative".into())
        })?;
        if count == 0 {
            return Ok(None);
        }
        let sliding = args
            .text_config
            .layer_schedule
            .iter()
            .find_map(|policy| policy.attention.window())
            .map(|window| crate::AttentionPolicy::Sliding { window });
        if !config.local_layer_ids.is_empty() && sliding.is_none() {
            return Err(Error::UnsupportedArchitecture(
                "Inkling MTP local layers require a backbone sliding-attention window".into(),
            ));
        }
        let policies = (0..count)
            .map(|depth| {
                if config.local_layer_ids.contains(&depth) {
                    sliding.expect("validated MTP sliding policy")
                } else {
                    crate::AttentionPolicy::Full
                }
            })
            .collect::<Vec<_>>();
        let layers = policies
            .iter()
            .copied()
            .map(|attention| {
                let text = inkling_mtp_text_args(args, config, attention)?;
                Ok(InklingMtpDepth {
                    hidden_norm: nn::RmsNorm::unloaded(
                        text.hidden_size,
                        text.rms_norm_eps,
                        text.weight_dtype(),
                        stream,
                    )?,
                    embed_norm: nn::RmsNorm::unloaded(
                        text.hidden_size,
                        text.rms_norm_eps,
                        text.weight_dtype(),
                        stream,
                    )?,
                    input_proj: common::linear::unloaded_maybe_quantized_linear_with_dtype(
                        text.hidden_size * 2,
                        text.hidden_size,
                        false,
                        None,
                        text.weight_dtype(),
                        stream,
                    )?,
                    transformer_block: DecoderLayer::new(&text, 0, stream)?,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let chain_norm = config
            .chain_hidden_post_norm
            .then(|| {
                nn::RmsNorm::unloaded(
                    args.text_config.hidden_size,
                    args.text_config.rms_norm_eps,
                    args.text_config.weight_dtype(),
                    stream,
                )
            })
            .transpose()?;
        Ok(Some(Self {
            layers,
            chain_norm,
            policies,
        }))
    }

    fn len(&self) -> usize {
        self.layers.len()
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_step(
        &mut self,
        hidden: &Array,
        embeddings: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut [resident::LayerCache],
        execution: Option<&crate::runtime::distributed::parallel::ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput, Exception> {
        if self.layers.is_empty() || cache.len() != self.layers.len() {
            return Err(Exception::custom(
                "Inkling MTP cache does not match prediction layers",
            ));
        }
        let depth = depth % self.layers.len();
        let layer = &mut self.layers[depth];
        let hidden = layer.hidden_norm.forward(hidden, stream)?;
        let embeddings = layer.embed_norm.forward(embeddings, stream)?;
        let combined = concatenate_axis(&[&hidden, &embeddings], -1, stream)?;
        let fused = layer.input_proj.forward(&combined, stream)?;
        let mut hidden = match execution.filter(|execution| execution.is_tensor_parallel()) {
            Some(execution) => layer.transformer_block.forward_tensor_parallel(
                &fused,
                Some(&mut cache[depth]),
                execution.group().ok_or_else(|| {
                    Exception::custom("Inkling MTP TP execution is missing its group")
                })?,
                stream,
            )?,
            None => layer
                .transformer_block
                .forward(&fused, Some(&mut cache[depth]), stream)?,
        };
        if let Some(norm) = &mut self.chain_norm {
            hidden = norm.forward(&hidden, stream)?;
        }
        Ok(
            crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput {
                logits: hidden.clone(),
                hidden,
                tokens: tokens.clone(),
            },
        )
    }
}

/// Inkling multimodal model using bounded residency for hMLP and decoder blocks.
pub struct InklingLayerwiseModel {
    execution: LayerwiseModel<InklingLayerwiseAdapter>,
}

pub(crate) struct InklingTensorMtpTarget<'a> {
    model: &'a mut InklingLayerwiseModel,
    group: &'a safemlx::distributed::Group,
}

impl<'a> InklingTensorMtpTarget<'a> {
    pub(crate) fn new(
        model: &'a mut InklingLayerwiseModel,
        group: &'a safemlx::distributed::Group,
    ) -> Self {
        Self { model, group }
    }
}

impl InklingLayerwiseModel {
    /// Returns the parsed Inkling configuration.
    pub fn args(&self) -> &ModelArgs {
        self.execution.adapter().args()
    }

    pub(crate) fn bind_parallel_topology(&mut self, topology: crate::ParallelTopology) {
        self.execution.bind_parallel_topology(topology);
    }

    /// Creates global/sliding KV and short-convolution state for every layer.
    pub fn new_cache(&self) -> Cache {
        self.execution.adapter().new_cache()
    }

    pub(crate) fn mtp_len(&self) -> usize {
        self.execution
            .adapter()
            .mtp
            .as_ref()
            .map_or(0, InklingMtpModule::len)
    }

    fn forward_mtp_target(
        &mut self,
        input: InklingInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput, Exception> {
        let (logits, context) = self
            .execution
            .forward_with_context_hook(
                InklingExecutionInput {
                    input,
                    last_token_only: false,
                },
                cache,
                stream,
                |_, _, _| Ok(()),
            )
            .map_err(|error| Exception::custom(error.to_string()))?;
        let hidden = context.draft_hidden.ok_or_else(|| {
            Exception::custom("Inkling layerwise pass did not retain MTP hidden state")
        })?;
        let tokens = context.draft_tokens.ok_or_else(|| {
            Exception::custom("Inkling layerwise pass did not retain MTP token identity")
        })?;
        Ok(
            crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput {
                logits,
                hidden,
                tokens,
            },
        )
    }

    fn forward_mtp_draft(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut [resident::LayerCache],
        stream: &Stream,
    ) -> Result<crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput, Exception> {
        let adapter = self.execution.adapter_mut();
        let embeddings = adapter.embedding.forward(tokens, stream)?;
        let embeddings = adapter.embed_norm.forward(&embeddings, stream)?;
        let mut output = adapter
            .mtp
            .as_mut()
            .ok_or_else(|| Exception::custom("Inkling checkpoint does not contain MTP layers"))?
            .forward_step(hidden, &embeddings, tokens, depth, cache, None, stream)?;
        output.logits = resident::project_text_logits(
            &output.hidden,
            &adapter.args.text_config,
            false,
            stream,
            |hidden, stream| adapter.lm_head.forward(hidden, stream),
        )?;
        Ok(output)
    }

    fn forward_mtp_target_tensor(
        &mut self,
        input: InklingInput<'_>,
        cache: &mut Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput, Exception> {
        let (logits, context) = self
            .execution
            .forward_tensor_parallel_with_context(
                InklingExecutionInput {
                    input,
                    last_token_only: false,
                },
                cache,
                group,
                stream,
            )
            .map_err(|error| Exception::custom(error.to_string()))?;
        Ok(
            crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput {
                logits,
                hidden: context.draft_hidden.ok_or_else(|| {
                    Exception::custom("Inkling tensor pass did not retain MTP hidden state")
                })?,
                tokens: context.draft_tokens.ok_or_else(|| {
                    Exception::custom("Inkling tensor pass did not retain MTP token identity")
                })?,
            },
        )
    }

    fn forward_mtp_draft_tensor(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut [resident::LayerCache],
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput, Exception> {
        let topology = self
            .parallel_info()
            .ok_or_else(|| Exception::custom("Inkling MTP target has no parallel topology"))?
            .topology();
        let execution =
            crate::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(
                topology, group, stream,
            )
            .map_err(|error| Exception::custom(error.to_string()))?;
        let adapter = self.execution.adapter_mut();
        let embeddings = adapter
            .parallel_embedding
            .as_mut()
            .ok_or_else(|| Exception::custom("Inkling MTP has no TP embedding shard"))?
            .forward(tokens, &execution)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let embeddings = adapter.embed_norm.forward(&embeddings, stream)?;
        let mut output = adapter
            .mtp
            .as_mut()
            .ok_or_else(|| Exception::custom("Inkling checkpoint does not contain MTP layers"))?
            .forward_step(
                hidden,
                &embeddings,
                tokens,
                depth,
                cache,
                Some(&execution),
                stream,
            )?;
        output.logits = adapter
            .parallel_lm_head
            .as_mut()
            .ok_or_else(|| Exception::custom("Inkling MTP has no TP output-head shard"))?
            .forward(&output.hidden, &execution)
            .and_then(|output| output.all_gather(&execution))
            .map_err(|error| Exception::custom(error.to_string()))?;
        Ok(output)
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

    /// Returns the complete rank-local prompt-cache identity.
    pub fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        self.execution.prompt_cache_model_identity()
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
            CacheResidencyPolicy::Paged(options) => {
                let rank = self.execution.prompt_cache_rank_identity();
                let cache = Cache::new_paged(&self.args().text_config, options, rank)?;
                match &self.execution.adapter().mtp {
                    Some(mtp) => cache
                        .with_paged_mtp_policies(&mtp.policies, rank)
                        .map_err(Into::into),
                    None => Ok(cache),
                }
            }
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

    pub(crate) fn checkpoint_store_arc(&self) -> Arc<dyn WeightStore + Send + Sync> {
        self.execution.checkpoint_store_arc()
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

    /// Runs TP-sharded attention, dense/shared projections, and rank-local
    /// cache state while delegating routed experts to the matching EP group.
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
            InklingExecutionInput {
                input: InklingInput::Decode(inputs),
                last_token_only: false,
            },
            cache,
            tensor_group,
            stream,
            |_adapter, _group, index, layer, hidden, cache, _context, execution| match layer {
                InklingLayer::Vision(_) => Err(Error::Parallel(
                    "Inkling TP+EP execution is restricted to the text decoder group".into(),
                )),
                InklingLayer::Text(layer) => {
                    let tp_group = execution.group().ok_or_else(|| {
                        Error::Parallel(
                            "Inkling TP+EP execution requires an active TP group".into(),
                        )
                    })?;
                    Ok(layer.forward_tensor_with_expert_executor(
                        hidden,
                        Some(&mut cache.layers[index]),
                        tp_group,
                        execution.stream(),
                        |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                    )?)
                }
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
    ) -> Result<crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput, Exception>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let input = InklingExecutionInput {
            input: InklingInput::Decode(tokens),
            last_token_only: false,
        };
        let (logits, context) = match tensor_group {
            Some(tensor_group) => self
                .execution
                .forward_tensor_parallel_with_layer_executor_and_context(
                    input,
                    cache,
                    tensor_group,
                    stream,
                    |_adapter, _group, index, layer, hidden, cache, _context, execution| match layer
                    {
                        InklingLayer::Vision(_) => Err(Error::Parallel(
                            "Inkling TP+EP MTP text target received a vision unit".into(),
                        )),
                        InklingLayer::Text(layer) => Ok(layer
                            .forward_tensor_with_expert_executor(
                                hidden,
                                Some(&mut cache.layers[index]),
                                execution.group().ok_or_else(|| {
                                    Error::Parallel(
                                        "Inkling TP+EP MTP target is missing its TP group".into(),
                                    )
                                })?,
                                execution.stream(),
                                |hidden, ids, weights, stream| {
                                    execute(index, hidden, ids, weights, stream)
                                },
                            )?),
                    },
                ),
            None => self.execution.forward_with_layer_executor_and_context(
                input,
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
            ),
        }
        .map_err(|error| Exception::custom(error.to_string()))?;
        Ok(
            crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput {
                logits,
                hidden: context.draft_hidden.ok_or_else(|| {
                    Exception::custom("Inkling EP pass did not retain MTP hidden state")
                })?,
                tokens: context.draft_tokens.ok_or_else(|| {
                    Exception::custom("Inkling EP pass did not retain MTP token identity")
                })?,
            },
        )
    }

    pub(crate) fn forward_mtp_draft_cartesian(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut [resident::LayerCache],
        tensor_group: Option<&safemlx::distributed::Group>,
        stream: &Stream,
    ) -> Result<crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput, Exception> {
        let topology = self
            .parallel_info()
            .ok_or_else(|| Exception::custom("Inkling EP MTP target has no topology"))?
            .topology();
        let execution = tensor_group
            .map(|group| {
                crate::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(
                    topology, group, stream,
                )
            })
            .transpose()
            .map_err(|error| Exception::custom(error.to_string()))?;
        let adapter = self.execution.adapter_mut();
        let embeddings = match execution.as_ref() {
            Some(execution) => adapter
                .parallel_embedding
                .as_mut()
                .ok_or_else(|| Exception::custom("Inkling MTP has no TP embedding shard"))?
                .forward(tokens, execution)
                .map_err(|error| Exception::custom(error.to_string()))?,
            None => adapter.embedding.forward(tokens, stream)?,
        };
        let embeddings = adapter.embed_norm.forward(&embeddings, stream)?;
        let mut output = adapter
            .mtp
            .as_mut()
            .ok_or_else(|| Exception::custom("Inkling checkpoint does not contain MTP layers"))?
            .forward_step(
                hidden,
                &embeddings,
                tokens,
                depth,
                cache,
                execution.as_ref(),
                stream,
            )?;
        output.logits = match execution.as_ref() {
            Some(execution) => adapter
                .parallel_lm_head
                .as_mut()
                .ok_or_else(|| Exception::custom("Inkling MTP has no TP output shard"))?
                .forward(&output.hidden, execution)
                .and_then(|output| output.all_gather(execution))
                .map_err(|error| Exception::custom(error.to_string()))?,
            None => adapter.lm_head.forward(&output.hidden, stream)?,
        };
        Ok(output)
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

impl crate::runtime::generation::embedded_mtp::EmbeddedMtpTarget for InklingLayerwiseModel {
    type Cache = Cache;
    type DraftCache = Vec<resident::LayerCache>;

    fn prefill_target(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput, Exception> {
        cache.reset()?;
        self.forward_mtp_target(InklingInput::Prefill(input), cache, stream)
    }

    fn verify_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput, Exception> {
        self.forward_mtp_target(InklingInput::Decode(tokens), cache, stream)
    }

    fn prefill_draft_cache(
        &mut self,
        output: &crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let sequence = tokens.dim(1);
        if sequence <= 1 {
            return Ok(());
        }
        let hidden = output
            .hidden
            .try_index_device((.., ..sequence - 1, ..), stream)?;
        let next = tokens.try_index_device((.., 1..), stream)?;
        for depth in 0..cache.mtp_layers.len() {
            let _ = self.forward_mtp_draft(&hidden, &next, depth, &mut cache.mtp_layers, stream)?;
        }
        Ok(())
    }

    fn draft_cache(cache: &Cache) -> Self::DraftCache {
        cache.mtp_layers.clone()
    }
    fn commit_draft_cache(cache: &mut Cache, draft: &Self::DraftCache) {
        cache.mtp_layers.clone_from(draft);
    }
    fn restore_target_checkpoint(
        cache: &mut Cache,
        checkpoint: &Cache,
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
        let token = Array::from_slice(&[last_token], &[1, 1]);
        let output = self.forward_mtp_draft(hidden, &token, draft_index, cache, stream)?;
        Ok((output.logits, output.hidden))
    }
    fn advance_draft_cache(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        for depth in 0..cache.len() {
            let _ = self.forward_mtp_draft(hidden, tokens, depth, cache, stream)?;
        }
        Ok(())
    }
    fn max_draft_tokens(&self) -> usize {
        self.mtp_len()
    }
}

impl crate::runtime::generation::embedded_mtp::EmbeddedMtpTarget for InklingTensorMtpTarget<'_> {
    type Cache = Cache;
    type DraftCache = Vec<resident::LayerCache>;

    fn prefill_target(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput, Exception> {
        cache.reset()?;
        self.model.forward_mtp_target_tensor(
            InklingInput::Prefill(input),
            cache,
            self.group,
            stream,
        )
    }

    fn verify_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput, Exception> {
        self.model.forward_mtp_target_tensor(
            InklingInput::Decode(tokens),
            cache,
            self.group,
            stream,
        )
    }

    fn prefill_draft_cache(
        &mut self,
        output: &crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let sequence = tokens.dim(1);
        if sequence <= 1 {
            return Ok(());
        }
        let hidden = output
            .hidden
            .try_index_device((.., ..sequence - 1, ..), stream)?;
        let next = tokens.try_index_device((.., 1..), stream)?;
        for depth in 0..cache.mtp_layers.len() {
            let _ = self.model.forward_mtp_draft_tensor(
                &hidden,
                &next,
                depth,
                &mut cache.mtp_layers,
                self.group,
                stream,
            )?;
        }
        Ok(())
    }

    fn draft_cache(cache: &Cache) -> Self::DraftCache {
        cache.mtp_layers.clone()
    }

    fn commit_draft_cache(cache: &mut Cache, draft: &Self::DraftCache) {
        cache.mtp_layers.clone_from(draft);
    }

    fn restore_target_checkpoint(
        cache: &mut Cache,
        checkpoint: &Cache,
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
        let token = Array::from_slice(&[last_token], &[1, 1]);
        let output = self.model.forward_mtp_draft_tensor(
            hidden,
            &token,
            draft_index,
            cache,
            self.group,
            stream,
        )?;
        Ok((output.logits, output.hidden))
    }

    fn advance_draft_cache(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        for depth in 0..cache.len() {
            let _ = self
                .model
                .forward_mtp_draft_tensor(hidden, tokens, depth, cache, self.group, stream)?;
        }
        Ok(())
    }

    fn max_draft_tokens(&self) -> usize {
        self.model.mtp_len()
    }
}

/// Loads Inkling's multimodal model through the generalized execution engine.
pub fn load_inkling_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<InklingLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let args = resident::get_model_args(model_dir)?;
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load(
                "Inkling layerwise model",
                args.text_config.weight_quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let adapter = InklingLayerwiseAdapter::new(args, stream)?;
    let store = open_safetensors_weight_store(model_dir, options.max_mapped_shards())?;
    Ok(InklingLayerwiseModel {
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

/// Loads Inkling with a rank-local hierarchical vision execution group.
pub fn load_inkling_tensor_parallel_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    build: crate::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<InklingLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
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
    options: LayerWeightResidency,
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
    let store = inkling_gguf_store(
        checkpoint,
        mmproj,
        &prepared.args,
        options.max_mapped_shards(),
    )?;
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
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(InklingLayerwiseModel, Vec<u32>), Error> {
    let load_options = quantization
        .map(crate::api::ModelLoadOptions::with_quantization)
        .unwrap_or_default()
        .with_weight_residency(residency);
    crate::api::structural::validate_gguf(
        crate::api::GgufArchitecture::Inkling,
        checkpoint,
        metadata,
        load_options,
    )
    .into_loader_result()?;
    let prepared = resident::prepare_gguf_checkpoint_with_mmproj(checkpoint, metadata, mmproj)?;
    let store = inkling_gguf_store(
        checkpoint,
        mmproj,
        &prepared.args,
        residency.max_mapped_shards(),
    )?;
    let args = prepared.args;
    if let Some(expert_options) = residency.expert_cache() {
        return Ok((
            load_inkling_gguf_sparse_with_store(
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
        InklingLayerwiseAdapter::new(args, stream)?,
        residency.layers(),
        quantization,
        stream,
        weights_stream,
    )?;
    Ok((InklingLayerwiseModel { execution }, prepared.eos_token_ids))
}

pub(crate) fn inkling_gguf_store(
    checkpoint: &GgufCheckpoint,
    mmproj: Option<&resident::InklingMmprojGguf>,
    args: &ModelArgs,
    max_mapped_shards: usize,
) -> Result<Arc<dyn WeightStore + Send + Sync>, Error> {
    let text_plan = super::checkpoint::gguf_plan(args).map_err(Error::UnsupportedArchitecture)?;
    let mut builder = GgufWeightStore::builder()
        .max_cached_readers(max_mapped_shards)?
        .add_checkpoint(
            checkpoint.clone(),
            &text_plan,
            resident::translate_gguf_weight_name,
        )?;
    if let Some(mmproj) = mmproj {
        let projector_plan =
            super::checkpoint::mmproj_gguf_plan(args).map_err(Error::UnsupportedArchitecture)?;
        builder = builder.add_checkpoint(
            mmproj.checkpoint.clone(),
            &projector_plan,
            resident::translate_mmproj_weight_name,
        )?;
    }
    Ok(Arc::new(builder.build()?))
}

fn load_inkling_gguf_sparse_with_store(
    store: Arc<dyn WeightStore + Send + Sync>,
    args: ModelArgs,
    options: ExpertCacheLoadOptions,
    non_expert: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<InklingLayerwiseModel, Error> {
    let mut adapter = InklingLayerwiseAdapter::new(args.clone(), stream)?;
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
    let entries = inkling_expert_catalog(&args, checkpoint_store.as_ref())?;
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
    Ok(InklingLayerwiseModel { execution })
}

/// Loads Inkling with independently cached experts and bounded non-expert units.
pub fn load_inkling_expert_cache_model(
    model_dir: impl AsRef<Path>,
    non_expert: crate::NonExpertWeightResidency,
    options: ExpertCacheLoadOptions,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<InklingLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
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
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load(
                "Inkling independent expert cache",
                args.text_config.weight_quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let mut adapter = InklingLayerwiseAdapter::new(args.clone(), stream)?;
    adapter.sparse_expert_cache = true;
    let store = open_safetensors_weight_store(model_dir, non_expert.layers().max_mapped_shards())?;
    let mut execution = load_layerwise_model_with_quantization(
        store,
        adapter,
        non_expert,
        quantize_on_load,
        stream,
        weights_stream,
    )?;
    let store = execution.checkpoint_store_arc();
    let entries = inkling_expert_catalog(&args, store.as_ref())?;
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
    Ok(InklingLayerwiseModel { execution })
}

/// Builds the streamed nonexpert Inkling execution base used by distributed EP.
pub(crate) fn load_inkling_sparse_ep_base_with_store(
    store: Arc<dyn WeightStore + Send + Sync>,
    args: ModelArgs,
    non_expert: impl Into<LayerWeightResidency>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<InklingLayerwiseModel, Error> {
    let mut adapter = InklingLayerwiseAdapter::new(args, stream)?;
    adapter.sparse_expert_cache = true;
    let execution = load_layerwise_model(store, adapter, non_expert, stream, weights_stream)?;
    Ok(InklingLayerwiseModel { execution })
}

/// Builds the TP-sharded nonexpert Inkling base used by combined TP+EP.
pub(crate) fn load_inkling_sparse_tp_ep_base_with_store(
    store: Arc<dyn WeightStore + Send + Sync>,
    args: ModelArgs,
    non_expert: impl Into<LayerWeightResidency>,
    build: crate::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<InklingLayerwiseModel, Error> {
    let mut adapter = InklingLayerwiseAdapter::new(args, stream)?;
    adapter.sparse_expert_cache = true;
    let execution = load_tensor_parallel_layerwise_model(
        store,
        adapter,
        non_expert,
        build,
        stream,
        weights_stream,
    )?;
    Ok(InklingLayerwiseModel { execution })
}

/// Adapter for Inkling local/global attention and dense/MoE text blocks.
pub(crate) struct InklingLayerwiseAdapter {
    args: ModelArgs,
    embedding: MaybeQuantized<nn::Embedding>,
    parallel_embedding: Option<VocabParallelEmbedding>,
    embed_norm: nn::RmsNorm,
    norm: nn::RmsNorm,
    lm_head: MaybeQuantized<nn::Linear>,
    parallel_lm_head: Option<VocabParallelLmHead>,
    mtp: Option<InklingMtpModule>,
    audio: Option<AudioModel>,
    vision_norm: Option<nn::RmsNorm>,
    vision_depth: usize,
    parallel_text_geometry: Option<Vec<resident::ParallelLayerGeometry>>,
    parallel_vision_input_ranges: Option<Vec<Range<i32>>>,
    sparse_expert_cache: bool,
    expert_cache: Option<ExpertCache>,
}

impl InklingLayerwiseAdapter {
    /// Exports hMLP job state for the next PP owner.
    pub(crate) fn pipeline_ingress_arrays(
        &self,
        state: &InklingPipelineIngressState,
    ) -> Vec<Array> {
        state
            .forward
            .context
            .vision_jobs
            .iter()
            .map(|job| job.hidden.clone())
            .collect()
    }

    /// Imports hMLP job state from the previous PP owner.
    pub(crate) fn replace_pipeline_ingress_arrays(
        &self,
        state: &mut InklingPipelineIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        if arrays.len() != state.forward.context.vision_jobs.len() {
            return Err(Error::Parallel(format!(
                "Inkling distributed vision payload has {} jobs, expected {}",
                arrays.len(),
                state.forward.context.vision_jobs.len()
            )));
        }
        for (job, hidden) in state.forward.context.vision_jobs.iter_mut().zip(arrays) {
            job.hidden = hidden;
        }
        state.forward.hidden = state
            .forward
            .context
            .vision_jobs
            .first()
            .map(|job| job.hidden.clone())
            .unwrap_or_else(|| state.forward.hidden.clone());
        Ok(())
    }

    pub(crate) fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let mtp = InklingMtpModule::new(&args, stream)?;
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
            mtp,
            audio,
            vision_norm,
            vision_depth,
            parallel_text_geometry: None,
            parallel_vision_input_ranges: None,
            sparse_expert_cache: false,
            expert_cache: None,
            args,
        })
    }

    pub(crate) fn new_external_experts(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let mut adapter = Self::new(args, stream)?;
        adapter.sparse_expert_cache = true;
        Ok(adapter)
    }

    /// Returns the parsed Inkling configuration.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    fn execution_group_name(&self, group: usize) -> Result<&'static str, Error> {
        match (self.vision_depth > 0, group) {
            (true, 0) => Ok("vision_encoder"),
            (true, 1) | (false, 0) => Ok("text_decoder"),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Inkling has no execution group {group}"
            ))),
        }
    }

    fn new_cache(&self) -> Cache {
        let cache = Cache::new(&self.args.text_config);
        match &self.mtp {
            Some(mtp) => cache.with_mtp_policies(&mtp.policies),
            None => cache,
        }
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
            if self.sparse_expert_cache && local_name.starts_with("moe.experts.") {
                continue;
            }
            let destination = format!("{prefix}.{local_name}");
            if parameter.dtype() == Dtype::Uint32 {
                let packed_weight = format!("{destination}.weight");
                if direct.contains(&packed_weight) {
                    recipes.insert(
                        local_name.to_string(),
                        DerivedWeightRecipe::source(packed_weight, TensorSelection::Full),
                    );
                    continue;
                }
            }
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
            if let Some(companion) = packed_companion_checkpoint_name(&destination)
                .filter(|companion| direct.contains(companion))
            {
                recipes.insert(
                    local_name.to_string(),
                    DerivedWeightRecipe::source(companion, TensorSelection::Full),
                );
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
    if let Some(suffix) = raw.strip_prefix("model.mtp.") {
        let mut key = format!("mtp.{suffix}");
        key = key
            .replace(
                ".transformer_block.attn_norm.weight",
                ".transformer_block.input_layernorm.weight",
            )
            .replace(
                ".transformer_block.mlp_norm.weight",
                ".transformer_block.post_attention_layernorm.weight",
            )
            .replace(
                ".transformer_block.attn.wq_du.weight",
                ".transformer_block.self_attn.q_proj.weight",
            )
            .replace(
                ".transformer_block.attn.wk_dv.weight",
                ".transformer_block.self_attn.k_proj.weight",
            )
            .replace(
                ".transformer_block.attn.wv_dv.weight",
                ".transformer_block.self_attn.v_proj.weight",
            )
            .replace(
                ".transformer_block.attn.wr_du.weight",
                ".transformer_block.self_attn.r_proj.weight",
            )
            .replace(
                ".transformer_block.attn.wo_ud.weight",
                ".transformer_block.self_attn.o_proj.weight",
            )
            .replace(
                ".transformer_block.attn.q_norm.weight",
                ".transformer_block.self_attn.q_norm.weight",
            )
            .replace(
                ".transformer_block.attn.k_norm.weight",
                ".transformer_block.self_attn.k_norm.weight",
            )
            .replace(
                ".transformer_block.attn.rel_logits_proj.proj",
                ".transformer_block.self_attn.rel_proj",
            )
            .replace(
                ".transformer_block.attn.k_sconv.weight",
                ".transformer_block.self_attn.k_sconv.weight",
            )
            .replace(
                ".transformer_block.attn.v_sconv.weight",
                ".transformer_block.self_attn.v_sconv.weight",
            )
            .replace(
                ".transformer_block.mlp.w2_md.weight",
                ".transformer_block.dense.down_proj.weight",
            )
            .replace(
                ".transformer_block.mlp.global_scale",
                ".transformer_block.dense_global_scale",
            );
        return Some(key);
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

pub(crate) struct InklingExecutionInput<'a> {
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
pub(crate) struct InklingForwardContext {
    parts: Vec<PreparedPart>,
    vision_jobs: Vec<VisionJob>,
    needs_assembly: bool,
    last_token_only: bool,
    draft_hidden: Option<Array>,
    draft_tokens: Option<Array>,
}

/// Opaque semantic state retained while a pipeline ingress stage executes its
/// configured Inkling vision root.
pub(crate) struct InklingPipelineIngressState {
    cache: Cache,
    forward: LayerwiseForwardState<InklingForwardContext>,
}

impl InklingLayerwiseAdapter {
    /// Embeds a decoder token step with the same stage-zero static ownership
    /// used by typed multimodal ingress.
    pub(crate) fn embed_pipeline_tokens(
        &mut self,
        tokens: &Array,
        execution: Option<&crate::runtime::distributed::parallel::ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let embedded = match execution {
            Some(execution) if execution.is_tensor_parallel() => self
                .parallel_embedding
                .as_mut()
                .ok_or_else(|| {
                    Error::Parallel("Inkling pipeline adapter has no TP embedding shard".into())
                })?
                .forward(tokens, execution)?,
            _ => self.embedding.forward(tokens, stream)?,
        };
        self.embed_norm
            .forward(&embedded, stream)
            .map_err(Into::into)
    }

    /// Returns the execution-group coordinates of configured media towers.
    pub(crate) fn pipeline_media_groups(&self) -> Vec<(usize, usize)> {
        (self.vision_depth > 0)
            .then_some((0, self.vision_depth))
            .into_iter()
            .collect()
    }

    /// Returns the decoder execution group after any configured vision root.
    pub(crate) fn pipeline_text_group(&self) -> usize {
        usize::from(self.vision_depth > 0)
    }

    /// Selects one configured media static target for pipeline ownership.
    pub(crate) fn pipeline_static_mut(&mut self, role: &str) -> Option<&mut dyn ModuleParameters> {
        match role {
            "embedding" => {
                if let Some(module) = &mut self.parallel_embedding {
                    Some(module.inner_mut())
                } else {
                    Some(&mut self.embedding)
                }
            }
            "embed_norm" => Some(&mut self.embed_norm),
            "output" => self
                .parallel_lm_head
                .as_mut()
                .map(|module| module.inner_mut() as &mut dyn ModuleParameters)
                .or(Some(&mut self.lm_head)),
            "mtp" => self
                .mtp
                .as_mut()
                .map(|module| module as &mut dyn ModuleParameters),
            "audio" => self
                .audio
                .as_mut()
                .map(|module| module as &mut dyn ModuleParameters),
            "vision_norm" => self
                .vision_norm
                .as_mut()
                .map(|module| module as &mut dyn ModuleParameters),
            _ => None,
        }
    }

    pub(crate) fn embedded_mtp_len(&self) -> usize {
        self.mtp.as_ref().map_or(0, InklingMtpModule::len)
    }

    pub(crate) fn embedded_mtp_cache(&self) -> Vec<resident::LayerCache> {
        self.new_cache().mtp_layers
    }

    pub(crate) fn forward_pipeline_mtp(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut [resident::LayerCache],
        execution: Option<&crate::runtime::distributed::parallel::ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<crate::runtime::generation::embedded_mtp::EmbeddedMtpOutput, Exception> {
        let embeddings = match execution.filter(|execution| execution.is_tensor_parallel()) {
            Some(execution) => self
                .parallel_embedding
                .as_mut()
                .ok_or_else(|| Exception::custom("Inkling pipeline MTP has no TP embedding shard"))?
                .forward(tokens, execution)
                .map_err(|error| Exception::custom(error.to_string()))?,
            None => self.embedding.forward(tokens, stream)?,
        };
        let embeddings = self.embed_norm.forward(&embeddings, stream)?;
        let mut output = self
            .mtp
            .as_mut()
            .ok_or_else(|| Exception::custom("Inkling checkpoint does not contain MTP layers"))?
            .forward_step(hidden, &embeddings, tokens, depth, cache, execution, stream)?;
        output.logits = match execution.filter(|execution| execution.is_tensor_parallel()) {
            Some(execution) => self
                .parallel_lm_head
                .as_mut()
                .ok_or_else(|| Exception::custom("Inkling pipeline MTP has no TP output shard"))?
                .forward(&output.hidden, execution)
                .and_then(|output| output.all_gather(execution))
                .map_err(|error| Exception::custom(error.to_string()))?,
            None => self.lm_head.forward(&output.hidden, stream)?,
        };
        Ok(output)
    }

    /// Starts typed pipeline ingress through the same adapter lifecycle used
    /// by resident and bounded layerwise execution.
    pub(crate) fn begin_pipeline_ingress(
        &mut self,
        input: input::ModelInput<'_>,
        execution: Option<&crate::runtime::distributed::parallel::ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<InklingPipelineIngressState, Error> {
        let mut cache = Cache::new(&self.args.text_config);
        let forward = match execution {
            Some(execution) if execution.is_tensor_parallel() => self
                .begin_forward_with_execution(
                    InklingExecutionInput {
                        input: InklingInput::Prefill(input),
                        last_token_only: false,
                    },
                    &mut cache,
                    execution,
                )?,
            _ => self.begin_forward(
                InklingExecutionInput {
                    input: InklingInput::Prefill(input),
                    last_token_only: false,
                },
                &mut cache,
                stream,
            )?,
        };
        Ok(InklingPipelineIngressState { cache, forward })
    }

    /// Builds the parameter-free hMLP job skeleton used by downstream PP
    /// owners. Text embedding, dMel audio ingress, normalization, and modality
    /// assembly remain on the placement-declared ingress/finalization owner.
    pub(crate) fn begin_pipeline_continuation(
        &self,
        input: input::ModelInput<'_>,
    ) -> Result<InklingPipelineIngressState, Error> {
        input::validate(input)?;
        let vision_jobs = input
            .parts
            .iter()
            .filter_map(|part| match (part.modality, part.payload) {
                (input::Modality::Image, input::InputPayload::Tensor(pixels)) => Some(VisionJob {
                    hidden: pixels.clone(),
                }),
                _ => None,
            })
            .collect::<Vec<_>>();
        let hidden = vision_jobs
            .first()
            .map(|job| job.hidden.clone())
            .or_else(|| {
                input.parts.first().map(|part| match part.payload {
                    input::InputPayload::Tensor(value)
                    | input::InputPayload::Embeddings(value)
                    | input::InputPayload::TokenIds(value) => value.clone(),
                })
            })
            .ok_or_else(|| Error::Parallel("Inkling continuation has no payload".into()))?;
        Ok(InklingPipelineIngressState {
            cache: Cache::new(&self.args.text_config),
            forward: LayerwiseForwardState {
                hidden,
                context: InklingForwardContext {
                    parts: Vec::new(),
                    vision_jobs,
                    needs_assembly: true,
                    last_token_only: false,
                    draft_hidden: None,
                    draft_tokens: None,
                },
            },
        })
    }

    /// Returns whether a configured media group has work for this input.
    pub(crate) fn should_execute_pipeline_group(
        &self,
        group: usize,
        state: &InklingPipelineIngressState,
    ) -> bool {
        self.should_execute_group(group, &state.forward.context)
    }

    /// Executes one resident or leased hMLP block using the canonical Inkling
    /// layerwise hooks.
    pub(crate) fn forward_pipeline_media_layer(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut InklingLayer,
        state: &mut InklingPipelineIngressState,
        execution: Option<&crate::runtime::distributed::parallel::ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Vec<Array>, Error> {
        state.forward.hidden = match execution {
            Some(execution) if execution.is_tensor_parallel() => self
                .forward_layer_with_execution(
                    group,
                    index,
                    layer,
                    &state.forward.hidden,
                    &mut state.cache,
                    &mut state.forward.context,
                    execution,
                )?,
            _ => self.forward_layer(
                group,
                index,
                layer,
                &state.forward.hidden,
                &mut state.cache,
                &mut state.forward.context,
                stream,
            )?,
        };
        Ok(std::iter::once(state.forward.hidden.clone())
            .chain(
                self.retained_context_arrays(&state.forward.context, group, index)
                    .into_iter()
                    .cloned(),
            )
            .collect())
    }

    /// Completes media roots and assembles the exact decoder ingress tensor.
    pub(crate) fn finish_pipeline_ingress(
        &mut self,
        mut state: InklingPipelineIngressState,
        execution: Option<&crate::runtime::distributed::parallel::ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let text_group = self.pipeline_text_group();
        state.forward.hidden = match execution {
            Some(execution) if execution.is_tensor_parallel() => self
                .begin_execution_group_with_execution(
                    text_group,
                    &state.forward.hidden,
                    &[],
                    &mut state.cache,
                    &mut state.forward.context,
                    execution,
                )?,
            _ => self.begin_execution_group(
                text_group,
                &state.forward.hidden,
                &[],
                &mut state.cache,
                &mut state.forward.context,
                stream,
            )?,
        };
        Ok(state.forward.hidden)
    }
}

/// One leased Inkling hMLP or decoder unit.
pub(crate) enum InklingLayer {
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

impl LoadTimeQuantizableAdapter for InklingLayerwiseAdapter {
    fn load_time_quantized(
        &self,
        quantization: WeightQuantization,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let mut args = self.args.clone();
        args.text_config.weight_quantization = Some(quantization);
        args.text_config.quantized_weight_configs = None;
        let aligned =
            |input: i32| input > 0 && input % quantization.group_size() == 0 && input % 32 == 0;
        if let Some(audio) = &mut args.audio_config {
            let mut formats = HashMap::new();
            if aligned(audio.text_hidden_size) {
                formats.insert("audio.encoder.weight".into(), quantization);
            }
            audio.quantized_weight_configs = Some(formats);
        }
        if let Some(vision) = &mut args.vision_config {
            let formats = vision
                .layer_specs()
                .into_iter()
                .enumerate()
                .filter(|(_, (input, _, _, _))| aligned(*input))
                .map(|(index, _)| {
                    (
                        format!("visual.layers.{index}.projection.weight"),
                        quantization,
                    )
                })
                .collect();
            vision.quantized_weight_configs = Some(formats);
        }
        let mut adapter = Self::new(args, stream)?;
        adapter.sparse_expert_cache = self.sparse_expert_cache;
        Ok(adapter)
    }
}

impl ArchitectureAdapter for InklingLayerwiseAdapter {
    type Input<'a> = InklingExecutionInput<'a>;
    type Cache = Cache;
    type Layer = InklingLayer;
    type ForwardContext = InklingForwardContext;

    fn model_type(&self) -> &str {
        &self.args.model_type
    }

    fn safetensors_checkpoint_plan(
        &self,
    ) -> Result<crate::runtime::execution::layerwise::ArchitectureCheckpointPlan, Error> {
        super::checkpoint::safetensors_plan(&self.args)
            .map_err(Error::UnsupportedArchitecture)
            .map(Into::into)
    }

    fn quantization(&self) -> Option<WeightQuantization> {
        self.args.text_config.weight_quantization
    }

    fn quantizes_static_binding(&self, binding: &WeightBinding) -> bool {
        let target = binding.logical_target().unwrap_or(binding.checkpoint_key());
        if target.starts_with("visual.") || target.contains(".visual.") {
            return false;
        }
        let audio = target
            .strip_prefix("audio.")
            .map(|suffix| format!("audio.{suffix}"))
            .or_else(|| {
                target
                    .split_once(".audio.")
                    .map(|(_, suffix)| format!("audio.{suffix}"))
            });
        match audio {
            Some(target) => self
                .args
                .audio_config
                .as_ref()
                .and_then(|args| args.quantized_weight_configs.as_ref())
                .is_some_and(|formats| formats.contains_key(&target)),
            None => true,
        }
    }

    fn prompt_cache_model_identity(
        &self,
        topology: Option<crate::ParallelTopology>,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let layer_layout = match topology {
            Some(topology) if topology.is_axis_active(crate::ParallelAxis::Tensor) => {
                resident::prompt_cache_layer_layout_with_geometry(
                &self.args,
                self.parallel_text_geometry.as_ref().ok_or_else(|| {
                    Error::Parallel(
                        "Inkling parallel cache identity requested before local layout configuration"
                            .into(),
                    )
                })?,
            )
            }
            _ => resident::prompt_cache_layer_layout(&self.args),
        }?;
        let layer_count = self.args.text_config.num_hidden_layers as usize;
        Ok(PromptCacheModelIdentity {
            model_family: "inkling".into(),
            effective_model_type: self.args.model_type.clone(),
            architecture_fingerprint: resident::prompt_cache_architecture_fingerprint(&self.args),
            layer_count,
            global_layer_start: 0,
            global_layer_end: layer_count,
            sink_tokens: 0,
            layer_prefix_offsets: vec![0; layer_count],
            topology: topology.map_or_else(
                PromptCacheTopology::default,
                PromptCacheTopology::for_parallel_topology,
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
                build_module_binding_plan_with_recipes(
                    &self.embedding,
                    "model.embed_tokens",
                    store,
                    self.recipes_for_module(&self.embedding, "model.embed_tokens", store)?,
                )?
                .build_bindings(store)?,
            )?);
        }
        if select(EMBED_NORM_UNIT) {
            units.push(StaticUnitBindings::new(
                EMBED_NORM_UNIT,
                build_module_binding_plan_with_recipes(
                    &self.embed_norm,
                    "model.embed_norm",
                    store,
                    self.recipes_for_module(&self.embed_norm, "model.embed_norm", store)?,
                )?
                .build_bindings(store)?,
            )?);
        }
        if select(NORM_UNIT) {
            units.push(StaticUnitBindings::new(
                NORM_UNIT,
                build_module_binding_plan_with_recipes(
                    &self.norm,
                    "model.norm",
                    store,
                    self.recipes_for_module(&self.norm, "model.norm", store)?,
                )?
                .build_bindings(store)?,
            )?);
        }
        if select(HEAD_UNIT) {
            units.push(StaticUnitBindings::new(
                HEAD_UNIT,
                build_module_binding_plan_with_recipes(
                    &self.lm_head,
                    "lm_head",
                    store,
                    self.recipes_for_module(&self.lm_head, "lm_head", store)?,
                )?
                .build_bindings(store)?,
            )?);
        }
        if select(MTP_UNIT) {
            if let Some(mtp) = &self.mtp {
                units.push(StaticUnitBindings::new(
                    MTP_UNIT,
                    build_module_binding_plan_with_recipes(
                        mtp,
                        "mtp",
                        store,
                        self.recipes_for_module(mtp, "mtp", store)?,
                    )?
                    .build_bindings(store)?,
                )?);
            }
        }
        if select(AUDIO_UNIT) {
            if let Some(audio) = &self.audio {
                units.push(StaticUnitBindings::new(
                    AUDIO_UNIT,
                    build_module_binding_plan_with_recipes(
                        audio,
                        "audio",
                        store,
                        self.recipes_for_module(audio, "audio", store)?,
                    )?
                    .build_bindings(store)?,
                )?);
            }
        }
        if select(VISION_NORM_UNIT) {
            if let Some(norm) = &self.vision_norm {
                units.push(StaticUnitBindings::new(
                    VISION_NORM_UNIT,
                    build_module_binding_plan_with_recipes(
                        norm,
                        "visual.final_norm",
                        store,
                        self.recipes_for_module(norm, "visual.final_norm", store)?,
                    )?
                    .build_bindings(store)?,
                )?);
            }
        }
        Ok(units)
    }

    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error> {
        let expected = 4
            + usize::from(self.mtp.is_some())
            + usize::from(self.audio.is_some())
            + usize::from(self.vision_norm.is_some());
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
        if let Some(mtp) = &mut self.mtp {
            populate_module_from_lease(mtp, &leases[index])?;
            index += 1;
        }
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
                    draft_hidden: None,
                    draft_tokens: Some(tokens.clone()),
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
            let tokens = concatenate_axis(&token_parts, 1, stream)?;
            let hidden = concatenate_axis(&embedding_parts, 1, stream)?;
            return Ok(LayerwiseForwardState {
                hidden,
                context: InklingForwardContext {
                    parts,
                    vision_jobs,
                    needs_assembly: false,
                    last_token_only,
                    draft_hidden: None,
                    draft_tokens: Some(tokens),
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
                draft_hidden: None,
                draft_tokens: None,
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
                    draft_hidden: None,
                    draft_tokens: Some(tokens.clone()),
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
            let tokens = concatenate_axis(&token_parts, 1, stream)?;
            let hidden = concatenate_axis(&embedding_parts, 1, stream)?;
            return Ok(LayerwiseForwardState {
                hidden,
                context: InklingForwardContext {
                    parts,
                    vision_jobs,
                    needs_assembly: false,
                    last_token_only,
                    draft_hidden: None,
                    draft_tokens: Some(tokens),
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
                draft_hidden: None,
                draft_tokens: None,
            },
        })
    }

    fn execution_graph(&self) -> Result<ExecutionGroupDag, Error> {
        if self.vision_depth > 0 {
            ExecutionGroupDag::chain(["vision_encoder", "text_decoder"])
        } else {
            ExecutionGroupDag::chain(["text_decoder"])
        }
    }

    fn should_execute_group(&self, group: usize, context: &Self::ForwardContext) -> bool {
        self.execution_group_name(group)
            .is_ok_and(|name| name != "vision_encoder" || !context.vision_jobs.is_empty())
    }

    fn layer_count(&self, group: usize) -> Result<usize, Error> {
        match self.execution_group_name(group)? {
            "vision_encoder" => Ok(self.vision_depth),
            "text_decoder" => Ok(self.args.text_config.num_hidden_layers as usize),
            _ => unreachable!(),
        }
    }

    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        self.layer_count(group)?;
        if self.execution_group_name(group)? == "vision_encoder" {
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

    fn new_expert_parallel_layer(
        &self,
        group: usize,
        index: usize,
        assignment: &crate::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        if self.execution_group_name(group)? == "vision_encoder" {
            return self.new_layer(group, index, stream);
        }
        Ok(InklingLayer::Text(Box::new(
            DecoderLayer::new_expert_parallel(
                &self.args.text_config,
                index as i32,
                if self.sparse_expert_cache {
                    0
                } else {
                    i32::try_from(assignment.local_expert_count()).map_err(|_| {
                        Error::Parallel("local Inkling expert count exceeds i32".into())
                    })?
                },
                stream,
            )?,
        )))
    }

    fn new_tensor_expert_parallel_layer(
        &self,
        group: usize,
        index: usize,
        layout: &crate::runtime::distributed::parallel::LocalModelLayout,
        assignment: &crate::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        if self.execution_group_name(group)? == "vision_encoder" {
            return self.new_parallel_layer(group, index, layout, stream);
        }
        let geometry = self
            .parallel_text_geometry
            .as_ref()
            .and_then(|geometry| geometry.get(index))
            .copied()
            .ok_or_else(|| {
                Error::Parallel(format!(
                    "Inkling local geometry is unavailable for decoder layer {index}"
                ))
            })?;
        let local_experts = if self.sparse_expert_cache {
            0
        } else {
            i32::try_from(assignment.local_expert_count())
                .map_err(|_| Error::Parallel("local Inkling expert count exceeds i32".into()))?
        };
        Ok(InklingLayer::Text(Box::new(
            DecoderLayer::new_tensor_expert_parallel(
                &self.args.text_config,
                index as i32,
                geometry,
                local_experts,
                stream,
            )?,
        )))
    }

    fn expert_parallel_assignment(
        &self,
        topology: crate::runtime::distributed::topology::ParallelTopology,
    ) -> Result<Option<crate::runtime::distributed::expert::ExpertAssignment>, Error> {
        if topology.expert_parallel_size == 1 && !self.sparse_expert_cache {
            return Ok(None);
        }
        if self.args.text_config.n_routed_experts <= 0
            || !self
                .args
                .text_config
                .layer_schedule
                .iter()
                .any(|policy| policy.feed_forward == resident::FeedForwardPolicy::SparseMoe)
        {
            return Err(Error::Parallel(
                "Inkling PP+EP requires a checkpoint with sparse MoE text layers".into(),
            ));
        }
        Ok(Some(
            crate::runtime::distributed::expert::ExpertAssignment::balanced(
                self.args.text_config.n_routed_experts as usize,
                topology.expert_parallel_size,
                topology.expert_parallel_rank,
            )?,
        ))
    }

    fn parallel_parameter_groups(
        &self,
        _context: crate::runtime::distributed::parallel::ParallelBuildContext,
    ) -> Result<Vec<crate::runtime::distributed::parallel::ParameterGroupSpec>, Error> {
        use crate::runtime::distributed::parallel::{
            aligned_partition_units, MemberSharding, ParameterGroupSpec, ParameterMemberSpec,
            ParameterRole,
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
            let weight_input = quantization.map_or(input as usize, |quantization| {
                safemlx::ops::quantized_packed_dimension(input, quantization.bits()) as usize
            });
            let mut members = vec![ParameterMemberSpec::new(
                name,
                [output as usize, weight_input],
                MemberSharding::Partitioned { axis: 1 },
            )];
            if let Some(quantization) = quantization {
                let companion_shape = [
                    output as usize,
                    (input / quantization.group_size()) as usize,
                ];
                members.push(ParameterMemberSpec::new(
                    format!("visual.layers.{index}.projection.scales"),
                    companion_shape,
                    MemberSharding::Partitioned { axis: 1 },
                ));
                if quantization.has_biases() {
                    members.push(ParameterMemberSpec::new(
                        format!("visual.layers.{index}.projection.biases"),
                        companion_shape,
                        MemberSharding::Partitioned { axis: 1 },
                    ));
                }
            }
            groups.push(ParameterGroupSpec::partitioned(
                format!("visual.layers.{index}.projection"),
                ParameterRole::RowProjection,
                aligned_partition_units(
                    &format!("visual.layers.{index}.projection"),
                    input as usize,
                    1,
                    usize::try_from(quantization.map_or(1, |value| value.group_size())).map_err(
                        |_| Error::Parallel("Inkling vision alignment exceeds usize".into()),
                    )?,
                )?,
                members,
            )?);
        }
        Ok(groups)
    }

    fn configure_parallel_static(
        &mut self,
        context: crate::runtime::distributed::parallel::ParallelBuildContext,
        layout: &crate::runtime::distributed::parallel::LocalModelLayout,
        stream: &Stream,
    ) -> Result<(), Error> {
        let text = &self.args.text_config;
        let local_dimension = |target: &str, axis: usize| -> Result<i32, Error> {
            let tensor = layout.tensor(target).ok_or_else(|| {
                Error::Parallel(format!("missing Inkling TP layout for {target}"))
            })?;
            let dimension = tensor.local_shape().get(axis).copied().ok_or_else(|| {
                Error::Parallel(format!("Inkling TP layout for {target} has no axis {axis}"))
            })?;
            i32::try_from(dimension)
                .map_err(|_| Error::Parallel(format!("Inkling local {target} exceeds i32")))
        };
        let projection_dimension = |prefix: &str, axis: usize| -> Result<i32, Error> {
            for target in [format!("{prefix}.weight"), format!("{prefix}.inner.weight")] {
                if layout.tensor(&target).is_some() {
                    return local_dimension(&target, axis);
                }
            }
            Err(Error::Parallel(format!(
                "missing Inkling TP projection layout for {prefix}"
            )))
        };
        let mut geometry = Vec::with_capacity(text.layer_schedule.len());
        for (index, policy) in text.layer_schedule.iter().enumerate() {
            let prefix = format!("model.layers.{index}");
            let sliding = policy.attention.window().is_some();
            let head_dim = text.attention_head_dim(sliding);
            let query_width = projection_dimension(&format!("{prefix}.self_attn.q_proj"), 0)?;
            let kv_width = projection_dimension(&format!("{prefix}.self_attn.k_proj"), 0)?;
            if head_dim <= 0 || query_width % head_dim != 0 || kv_width % head_dim != 0 {
                return Err(Error::Parallel(format!(
                    "Inkling layer {index} local attention widths ({query_width}, {kv_width}) do not contain integral heads of width {head_dim}"
                )));
            }
            let query_heads = query_width / head_dim;
            let kv_heads = kv_width / head_dim;
            if kv_heads <= 0 || query_heads % kv_heads != 0 {
                return Err(Error::Parallel(format!(
                    "Inkling layer {index} local attention geometry q={query_heads}, kv={kv_heads} does not preserve complete GQA groups"
                )));
            }
            let sparse_width = |bank: &str| -> Result<i32, Error> {
                let fused = local_dimension(&format!("{prefix}.moe.{bank}.gate_up_proj"), 1)?;
                if fused % 2 != 0 {
                    return Err(Error::Parallel(format!(
                        "Inkling layer {index} local {bank} fused width {fused} is not even"
                    )));
                }
                Ok(fused / 2)
            };
            let feed_forward = match policy.feed_forward {
                resident::FeedForwardPolicy::Dense => {
                    resident::ParallelFeedForwardGeometry::Dense {
                        intermediate: projection_dimension(
                            &format!("{prefix}.dense.gate_proj"),
                            0,
                        )?,
                    }
                }
                resident::FeedForwardPolicy::SparseMoe => {
                    resident::ParallelFeedForwardGeometry::SparseMoe {
                        routed_intermediate: sparse_width("experts")?,
                        shared_intermediate: sparse_width("shared_experts")?,
                    }
                }
            };
            geometry.push(resident::ParallelLayerGeometry {
                query_heads,
                kv_heads,
                feed_forward,
            });
        }
        self.parallel_text_geometry = Some(geometry);
        self.parallel_vision_input_ranges = self
            .args
            .vision_config
            .as_ref()
            .map(|vision| {
                vision
                    .layer_specs()
                    .into_iter()
                    .enumerate()
                    .map(|(index, (input, _, _, _))| {
                        let target = format!("visual.layers.{index}.projection.weight");
                        let tensor = layout.tensor(&target).ok_or_else(|| {
                            Error::Parallel(format!("missing Inkling TP layout for {target}"))
                        })?;
                        let units = tensor.logical_units().ok_or_else(|| {
                            Error::Parallel(format!(
                                "Inkling TP layout for {target} has no logical partition"
                            ))
                        })?;
                        let range = tensor.logical_range().ok_or_else(|| {
                            Error::Parallel(format!(
                                "Inkling TP layout for {target} has no local logical range"
                            ))
                        })?;
                        let input = usize::try_from(input).map_err(|_| {
                            Error::Parallel("Inkling vision input width exceeds usize".into())
                        })?;
                        if input % units != 0 {
                            return Err(Error::Parallel(format!(
                                "Inkling vision input width {input} does not contain {units} aligned planner units"
                            )));
                        }
                        let channels_per_unit = input / units;
                        Ok(i32::try_from(range.start * channels_per_unit)
                            .map_err(|_| Error::Parallel("Inkling vision range exceeds i32".into()))?
                            ..i32::try_from(range.end * channels_per_unit).map_err(|_| {
                                Error::Parallel("Inkling vision range exceeds i32".into())
                            })?)
                    })
                    .collect::<Result<Vec<_>, Error>>()
            })
            .transpose()?;
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
        if let (Some(mtp), Some(config)) = (&mut self.mtp, self.args.mtp_config.as_ref()) {
            for (index, layer) in mtp.layers.iter_mut().enumerate() {
                let prefix = format!("mtp.layers.{index}.transformer_block");
                let mtp_text = inkling_mtp_text_args(&self.args, config, mtp.policies[index])?;
                let sliding = mtp.policies[index].window().is_some();
                let head_dim = mtp_text.attention_head_dim(sliding);
                let query_width = projection_dimension(&format!("{prefix}.self_attn.q_proj"), 0)?;
                let kv_width = projection_dimension(&format!("{prefix}.self_attn.k_proj"), 0)?;
                if head_dim <= 0 || query_width % head_dim != 0 || kv_width % head_dim != 0 {
                    return Err(Error::Parallel(format!(
                        "Inkling MTP layer {index} local attention widths ({query_width}, {kv_width}) do not contain integral heads of width {head_dim}"
                    )));
                }
                let geometry = resident::ParallelLayerGeometry {
                    query_heads: query_width / head_dim,
                    kv_heads: kv_width / head_dim,
                    feed_forward: resident::ParallelFeedForwardGeometry::Dense {
                        intermediate: projection_dimension(
                            &format!("{prefix}.dense.gate_proj"),
                            0,
                        )?,
                    },
                };
                layer.transformer_block =
                    DecoderLayer::new_parallel_layerwise(&mtp_text, 0, geometry, stream)?;
            }
        }
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
        if let Some(mtp) = &self.mtp {
            for (index, layer) in mtp.layers.iter().enumerate() {
                let prefix = format!("mtp.layers.{index}");
                crate::runtime::distributed::parallel::register_replicated_module(
                    planner,
                    &layer.hidden_norm,
                    &format!("{prefix}.hidden_norm"),
                )?;
                crate::runtime::distributed::parallel::register_replicated_module(
                    planner,
                    &layer.embed_norm,
                    &format!("{prefix}.embed_norm"),
                )?;
                crate::runtime::distributed::parallel::register_projection_module(
                    planner,
                    &layer.input_proj,
                    &format!("{prefix}.input_proj"),
                    crate::runtime::distributed::parallel::ProjectionSharding::Replicated,
                )?;
                layer
                    .transformer_block
                    .register_tensor_parallel_parameters(
                        planner,
                        &format!("{prefix}.transformer_block"),
                    )?;
            }
            if let Some(chain_norm) = &mtp.chain_norm {
                crate::runtime::distributed::parallel::register_replicated_module(
                    planner,
                    chain_norm,
                    "mtp.chain_norm",
                )?;
            }
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
        if self.execution_group_name(group)? != "vision_encoder" {
            let _ = layout;
            let geometry = self
                .parallel_text_geometry
                .as_ref()
                .and_then(|geometry| geometry.get(index))
                .copied()
                .ok_or_else(|| {
                    Error::Parallel(format!(
                        "Inkling local geometry is unavailable for decoder layer {index}"
                    ))
                })?;
            return Ok(InklingLayer::Text(Box::new(
                DecoderLayer::new_parallel_layerwise(
                    &self.args.text_config,
                    index as i32,
                    geometry,
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
        let spec = specs[index];
        let target = format!("visual.layers.{index}.projection.weight");
        let _ = layout
            .tensor(&target)
            .ok_or_else(|| Error::Parallel(format!("missing TP layout for {target}")))?;
        let input_range = self
            .parallel_vision_input_ranges
            .as_ref()
            .and_then(|ranges| ranges.get(index))
            .cloned()
            .ok_or_else(|| {
                Error::Parallel(format!(
                    "Inkling local vision geometry is unavailable for layer {index}"
                ))
            })?;
        Ok(InklingLayer::Vision(VisionLayer::new_parallel_layerwise(
            spec,
            input_range,
            index + 1 != specs.len(),
            args.rms_norm_eps,
            args.weight_quantization_for(&target),
            self.args.text_config.weight_dtype(),
            stream,
        )?))
    }

    fn layer_checkpoint_prefix(&self, group: usize, index: usize) -> String {
        if self.execution_group_name(group).ok() == Some("vision_encoder") {
            format!("visual.layers.{index}")
        } else {
            format!("model.layers.{index}")
        }
    }

    fn layer_unit_name(&self, group: usize, index: usize) -> String {
        if self.execution_group_name(group).ok() == Some("vision_encoder") {
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
        let recipes = self.recipes_for_module(layer, &prefix, store)?;
        let external_experts =
            self.sparse_expert_cache && self.execution_group_name(group)? == "text_decoder";
        let bindings = build_module_binding_plan_with_recipes_excluding(
            layer,
            &prefix,
            store,
            recipes,
            |name| external_experts && name.starts_with("moe.experts."),
        )?
        .build_bindings(store)?;
        bindings
            .into_iter()
            .map(|binding| {
                if matches!(
                    binding.name(),
                    "moe.experts.gate_up_proj"
                        | "moe.experts.down_proj"
                        | "moe.shared_experts.gate_up_proj"
                        | "moe.shared_experts.down_proj"
                ) {
                    let target = format!("{prefix}.{}.weight", binding.name());
                    binding.with_logical_target(target).map_err(Error::from)
                } else {
                    Ok(binding)
                }
            })
            .collect()
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
                if target.contains("moe.experts.") {
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
        match (self.execution_group_name(group)?, layer) {
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
                                    ExpertRouteBatch::new(index, flat, indices, weights, pass),
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
        if self.execution_group_name(group)? == "vision_encoder" {
            if let InklingLayer::Vision(layer) = layer {
                for job in &mut context.vision_jobs {
                    job.hidden =
                        layer.forward_tensor_parallel(&job.hidden, tp_group, execution.stream())?;
                }
                return Ok(context.vision_jobs[0].hidden.clone());
            }
        } else if let InklingLayer::Text(layer) = layer {
            if self.sparse_expert_cache
                && self
                    .args
                    .text_config
                    .layer_policy(index)
                    .is_some_and(|policy| {
                        policy.feed_forward == resident::FeedForwardPolicy::SparseMoe
                    })
            {
                let expert_cache = self.expert_cache.as_ref().ok_or_else(|| {
                    Error::Parallel("Inkling sparse expert cache was not initialized".into())
                })?;
                let pass = if hidden.dim(1) > 1 {
                    ExpertPass::Prefill
                } else {
                    ExpertPass::Decode
                };
                return Ok(layer.forward_tensor_with_expert_executor(
                    hidden,
                    Some(&mut cache.layers[index]),
                    tp_group,
                    execution.stream(),
                    |flat, indices, weights, stream| {
                        expert_cache
                            .execute_routes_bounded(
                                ExpertRouteBatch::new(index, flat, indices, weights, pass),
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
        if self.execution_group_name(group).ok() == Some("text_decoder") {
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

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial_hidden: &Array,
        dependency_outputs: &[Array],
        _cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let group_name = self.execution_group_name(group)?;
        let hidden = match dependency_outputs {
            [] => initial_hidden,
            [dependency] => dependency,
            _ => {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Inkling execution group {group_name} received {} dependency outputs",
                    dependency_outputs.len()
                )))
            }
        };
        let should_assemble = context.needs_assembly && group_name == "text_decoder";
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
        let tokens = concatenate_axis(&tokens, 1, stream)?;
        context.draft_tokens = Some(tokens);
        context.needs_assembly = false;
        Ok(concatenate_axis(&embeddings, 1, stream)?)
    }

    fn complete_execution_group(
        &mut self,
        group: usize,
        hidden: &Array,
        _cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        if self.execution_group_name(group)? == "text_decoder" {
            context.draft_hidden = Some(hidden.clone());
        }
        Ok(hidden.clone())
    }

    fn finish(
        &mut self,
        hidden: &Array,
        _cache: &mut Self::Cache,
        context: &Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let hidden = self.norm.forward(hidden, stream)?;
        Ok(resident::project_text_logits(
            &hidden,
            &self.args.text_config,
            context.last_token_only,
            stream,
            |hidden, stream| self.lm_head.forward(hidden, stream),
        )?)
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
        resident::project_text_logits(
            &hidden,
            &self.args.text_config,
            context.last_token_only,
            execution.stream(),
            |hidden, _| head.forward(hidden, execution)?.all_gather(execution),
        )
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
            let bindings = BindingPlan::new(planned)
                .and_then(|plan| plan.build_bindings(store))
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
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
    use std::{collections::HashMap, fs, path::Path};

    use safemlx::{
        distributed::{Backend, Group},
        module::ModuleParameters,
        ops::{indexing::TryIndexOp, ones_dtype, stack_axis, zeros_dtype},
        Array, Device, DeviceType, Dtype, ExecutionContext, Stream,
    };

    use super::{
        load_inkling_expert_cache_model, load_inkling_layerwise_model,
        load_inkling_tensor_parallel_layerwise_model, InklingLayer, InklingLayerwiseAdapter,
        InklingLayerwiseModel,
    };
    use crate::{
        api::{
            common::generation::CausalLm,
            inkling::{self as resident, Model, ModelArgs},
            input as runtime_input,
        },
        runtime::cache::{
            residency::{CacheResidencyPolicy, PromptCacheDescriptor, PromptCacheOptions},
            KeyValueCache,
        },
        runtime::checkpoint::quantization::{AffineQuantization, WeightQuantization},
        runtime::distributed::{
            parallel::{ParallelBuildContext, ShardingPolicy},
            topology::{DeviceAssignment, ParallelTopology},
        },
        runtime::execution::layerwise::{
            load_layerwise_model_with_quantization, transformed_module_weight_store,
            ArchitectureAdapter, LayerWeightResidency, LayerwiseLoadOptions,
            LoadTimeQuantizableAdapter,
        },
        runtime::residency::dense_stream::DenseDiskStreamLoadOptions,
        runtime::residency::expert_cache::{ExpertCacheLoadOptions, ExpertPass, ExpertRouteBatch},
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
    fn published_mtp_geometry_builds_distinct_full_and_local_draft_state() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut value = config();
        value["mtp_config"] = serde_json::json!({
            "num_nextn_predict_layers": 3,
            "chain_hidden_post_norm": true,
            "local_layer_ids": [1],
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "intermediate_size": 32,
            "sconv_kernel_size": 3,
            "rel_extent": 32,
            "d_rel": 32
        });
        let args = resident::model_args_from_config_value(&value).unwrap();
        let adapter = InklingLayerwiseAdapter::new(args, execution.stream()).unwrap();
        let mtp = adapter.mtp.as_ref().unwrap();
        assert_eq!(mtp.len(), 3);
        assert_eq!(mtp.policies[0], crate::AttentionPolicy::Full);
        assert!(matches!(
            mtp.policies[1],
            crate::AttentionPolicy::Sliding { .. }
        ));
        assert_eq!(adapter.new_cache().mtp_layers.len(), 3);
    }

    fn quantizable_config() -> serde_json::Value {
        let mut value = config();
        value["text_config"]["hidden_size"] = 32.into();
        value["text_config"]["vocab_size"] = 64.into();
        value["text_config"]["num_attention_heads"] = 4.into();
        value["text_config"]["num_key_value_heads"] = 2.into();
        value["text_config"]["head_dim"] = 8.into();
        value["text_config"]["swa_num_attention_heads"] = 4.into();
        value["text_config"]["swa_num_key_value_heads"] = 2.into();
        value["text_config"]["swa_head_dim"] = 8.into();
        value["text_config"]["d_rel"] = 32.into();
        value["text_config"]["rel_extent"] = 32.into();
        value["text_config"]["intermediate_size"] = 32.into();
        value["text_config"]["dense_intermediate_size"] = 32.into();
        value["text_config"]["moe_intermediate_size"] = 32.into();
        value
    }

    #[test]
    #[ignore = "requires an MLX Metal device"]
    fn load_time_adapter_packs_text_and_aligned_media_with_external_experts() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut value = config();
        value["text_config"]["hidden_size"] = 32.into();
        value["text_config"]["vocab_size"] = 64.into();
        value["text_config"]["num_attention_heads"] = 4.into();
        value["text_config"]["num_key_value_heads"] = 2.into();
        value["text_config"]["head_dim"] = 8.into();
        value["text_config"]["swa_num_attention_heads"] = 4.into();
        value["text_config"]["swa_num_key_value_heads"] = 2.into();
        value["text_config"]["swa_head_dim"] = 8.into();
        value["text_config"]["intermediate_size"] = 32.into();
        value["text_config"]["dense_intermediate_size"] = 32.into();
        value["text_config"]["moe_intermediate_size"] = 32.into();
        value["vision_config"] = serde_json::json!({
            "decoder_dmodel": 32,
            "patch_size": 40,
            "temporal_patch_size": 2,
            "n_channels": 3,
            "n_layers": 4
        });
        value["audio_config"] = serde_json::json!({
            "text_hidden_size": 32,
            "num_codebooks": 2,
            "codebook_size": 8,
            "bias": false,
            "use_audio_norm": true,
            "audio_mode": "dmel",
            "rms_norm_eps": 1e-6
        });
        let mut args = resident::model_args_from_config_value(&value).unwrap();
        args.text_config.quantized_weight_configs = Some(HashMap::from([(
            "model.layers.0.dense.down_proj.weight".into(),
            WeightQuantization::MxFp4,
        )]));
        let source =
            InklingLayerwiseAdapter::new_external_experts(args.clone(), execution.stream())
                .unwrap();
        let quantization = AffineQuantization::new(32, 4).unwrap().into();
        let target = source
            .load_time_quantized(quantization, execution.stream())
            .unwrap();

        assert_eq!(target.quantization(), Some(quantization));
        assert_eq!(
            target.args.text_config.weight_quantization,
            Some(quantization)
        );
        assert!(target.args.text_config.quantized_weight_configs.is_none());
        assert!(target.sparse_expert_cache);
        assert!(target.audio.is_some());
        assert_eq!(target.vision_depth, 4);
        assert!(matches!(
            target.embedding,
            safemlx::quantization::MaybeQuantized::Quantized(_)
        ));
        assert!(target
            .audio
            .as_ref()
            .unwrap()
            .parameters()
            .flatten()
            .values()
            .any(|parameter| parameter.dtype() == Dtype::Uint32));

        let InklingLayer::Text(text) = target.new_layer(1, 0, execution.stream()).unwrap() else {
            panic!("Inkling decoder group must build a text layer")
        };
        assert!(text
            .parameters()
            .flatten()
            .values()
            .any(|parameter| parameter.dtype() == Dtype::Uint32));
        let InklingLayer::Vision(vision) = target.new_layer(0, 0, execution.stream()).unwrap()
        else {
            panic!("Inkling vision group must build a vision layer")
        };
        assert!(vision
            .parameters()
            .flatten()
            .values()
            .all(|parameter| parameter.dtype() != Dtype::Uint32));
        let InklingLayer::Vision(vision) = target.new_layer(0, 1, execution.stream()).unwrap()
        else {
            panic!("Inkling vision group must build a vision layer")
        };
        assert!(vision
            .parameters()
            .flatten()
            .values()
            .any(|parameter| parameter.dtype() == Dtype::Uint32));

        let mxfp4 = source
            .load_time_quantized(WeightQuantization::MxFp4, execution.stream())
            .unwrap();
        assert_eq!(mxfp4.quantization(), Some(WeightQuantization::MxFp4));
        let InklingLayer::Text(text) = mxfp4.new_layer(1, 1, execution.stream()).unwrap() else {
            panic!("Inkling MXFP4 target must build a text layer")
        };
        assert!(text
            .parameters()
            .flatten()
            .values()
            .any(|parameter| parameter.dtype() == Dtype::Uint32));

        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        args.text_config.quantized_weight_configs = None;
        let mut fixture = Model::new(args.clone(), gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let store = transformed_module_weight_store(&fixture).unwrap();
        let execution = load_layerwise_model_with_quantization(
            store,
            InklingLayerwiseAdapter::new(args, gpu.stream()).unwrap(),
            LayerWeightResidency::FullyResident,
            Some(quantization),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let keys = execution.checkpoint_store_arc().keys();
        assert!(keys.iter().any(|key| key == "audio.encoder.scales"));
        assert!(keys
            .iter()
            .any(|key| key == "visual.layers.1.projection.scales"));
        assert!(!keys
            .iter()
            .any(|key| key == "visual.layers.0.projection.scales"));
        let report = execution.residency_report().unwrap();
        let materialization = report.materialization().unwrap();
        assert!(materialization.transformed_weights > 20);
        assert!(materialization.output_bytes < materialization.source_bytes_read);
        assert!(materialization.peak_planned_working_set_bytes <= materialization.output_bytes);

        let mut quantized = InklingLayerwiseModel { execution };
        let text = runtime_input::token_ids_array(&[1, 2], gpu.stream()).unwrap();
        let pixels = Array::zeros::<f32>(&[1, 2, 40, 40, 3], gpu.stream()).unwrap();
        let audio_ids = Array::from_slice(&[0u32, 1, 2, 3, 4, 5], &[3, 2]);
        let audio_mask = Array::from_slice(&[true, true, false], &[1, 3]);
        let parts = [
            runtime_input::InputPart::text_token_ids(&text),
            runtime_input::InputPart::image_tensor(
                &pixels,
                runtime_input::InputMetadata::default(),
            ),
            runtime_input::InputPart::audio_tensor(
                &audio_ids,
                runtime_input::InputMetadata::audio_mask(&audio_mask),
            ),
        ];
        let typed = runtime_input::ModelInput::new(&parts);
        let mut dense_cache = fixture.new_cache();
        let mut quantized_cache = quantized.new_cache();
        let expected = fixture
            .prefill_input_logits(typed, &mut dense_cache, gpu.stream())
            .unwrap();
        let actual = quantized
            .prefill_input_logits(typed, &mut quantized_cache, gpu.stream())
            .unwrap();
        assert!(actual
            .all_close(&expected, Some(2e-2), Some(2e-2), None, gpu.stream())
            .unwrap()
            .item::<bool>(gpu.stream()));
    }

    #[test]
    #[ignore = "requires an MLX Metal device"]
    fn inkling_fully_resident_load_time_quantization_packs_complete_expert_banks() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let config = quantizable_config();
        let args = resident::model_args_from_config_value(&config).unwrap();
        let mut fixture = Model::new(args, gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture_with_config(dir.path(), &fixture, &config, gpu.stream());

        for quantization in [
            WeightQuantization::Affine(AffineQuantization::new(32, 4).unwrap()),
            WeightQuantization::MxFp4,
        ] {
            let expert_options = ExpertCacheLoadOptions::new(
                OffloadConfig::new(Some(1 << 20), Some(1 << 20), 1).unwrap(),
                1 << 16,
                1 << 16,
            )
            .unwrap();
            let mut cached = load_inkling_expert_cache_model(
                dir.path(),
                crate::NonExpertWeightResidency::FullyResident,
                expert_options,
                Some(quantization),
                gpu.stream(),
                cpu.stream(),
            )
            .unwrap();
            let loaded = crate::api::load_model_with_options(
                dir.path(),
                crate::api::ModelLoadOptions::with_quantization(quantization),
                gpu.stream(),
                cpu.stream(),
            )
            .unwrap();
            let crate::api::Model::Inkling(mut quantized) = loaded else {
                panic!("high-level dispatch did not return an Inkling model")
            };
            assert_eq!(
                quantized.args().text_config.weight_quantization,
                Some(quantization)
            );
            assert!(quantized.expert_cache_report().unwrap().is_none());
            let report = quantized.residency_report().unwrap();
            assert!(report.initialized());
            assert!(report.units().iter().all(|unit| unit.device_resident()));
            let materialization = report.materialization().unwrap();
            assert!(materialization.transformed_weights > 0);
            assert!(materialization.source_bytes_read > materialization.output_bytes);
            assert!(materialization.peak_planned_working_set_bytes <= materialization.output_bytes);

            let mut cached_cache = cached.new_cache();
            let mut quantized_cache = quantized.new_cache();
            for tokens in [
                Array::from_slice(&[1u32, 2, 3], &[1, 3]),
                Array::from_slice(&[4u32], &[1, 1]),
            ] {
                let expected = cached
                    .forward(&tokens, &mut cached_cache, gpu.stream())
                    .unwrap();
                let actual = quantized
                    .forward(&tokens, &mut quantized_cache, gpu.stream())
                    .unwrap();
                assert_close(&actual, &expected, gpu.stream());
            }
        }
    }

    #[test]
    fn inkling_multimodal_execution_graph_declares_vision_dependency() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut value = config();
        value["image_token_id"] = 20.into();
        value["vision_config"] = serde_json::json!({
            "decoder_dmodel": 16,
            "patch_size": 40,
            "temporal_patch_size": 2,
            "n_channels": 3,
            "n_layers": 4
        });
        let adapter = InklingLayerwiseAdapter::new(
            resident::model_args_from_config_value(&value).unwrap(),
            execution.stream(),
        )
        .unwrap();
        let graph = adapter.execution_graph().unwrap();
        assert_eq!(
            graph
                .groups()
                .iter()
                .map(|group| group.id())
                .collect::<Vec<_>>(),
            ["vision_encoder", "text_decoder"]
        );
        assert_eq!(graph.dependencies(1), Some([0].as_slice()));
        assert_eq!(graph.output(), 1);
    }

    #[test]
    fn tensor_parallel_plan_supports_uneven_text_and_folded_vision_geometry() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut config = config();
        config["text_config"]["num_attention_heads"] = 6.into();
        config["text_config"]["num_key_value_heads"] = 3.into();
        config["text_config"]["head_dim"] = 2.into();
        config["text_config"]["swa_num_attention_heads"] = 6.into();
        config["text_config"]["swa_num_key_value_heads"] = 3.into();
        config["text_config"]["swa_head_dim"] = 2.into();
        config["text_config"]["dense_intermediate_size"] = 17.into();
        config["text_config"]["moe_intermediate_size"] = 9.into();
        config["vision_config"] = serde_json::json!({
            "decoder_dmodel": 16,
            "patch_size": 40,
            "temporal_patch_size": 2,
            "n_channels": 3,
            "n_layers": 4
        });
        let args = resident::model_args_from_config_value(&config).unwrap();

        for (rank, query_heads, kv_heads, dense, expert, first_vision) in
            [(0, 4, 2, 9, 5, 0..38), (1, 2, 1, 8, 4, 38..75)]
        {
            let mut adapter =
                InklingLayerwiseAdapter::new(args.clone(), execution.stream()).unwrap();
            let topology = ParallelTopology::from_rank(
                2,
                rank,
                2,
                1,
                1,
                DeviceAssignment::new(DeviceType::Cpu, 0),
            )
            .unwrap();
            let context = ParallelBuildContext::new(topology, ShardingPolicy::Require);
            let mut planner = context.planner();
            adapter
                .register_parallel_parameters(context, &mut planner, execution.stream())
                .unwrap();
            let (_, layout) = planner.finish().unwrap();
            adapter
                .configure_parallel_static(context, &layout, execution.stream())
                .unwrap();

            let geometry = adapter.parallel_text_geometry.as_ref().unwrap();
            assert_eq!(geometry[0].query_heads, query_heads);
            assert_eq!(geometry[0].kv_heads, kv_heads);
            assert_eq!(
                geometry[0].feed_forward,
                resident::ParallelFeedForwardGeometry::Dense {
                    intermediate: dense
                }
            );
            assert_eq!(geometry[1].query_heads, query_heads);
            assert_eq!(geometry[1].kv_heads, kv_heads);
            assert_eq!(
                geometry[1].feed_forward,
                resident::ParallelFeedForwardGeometry::SparseMoe {
                    routed_intermediate: expert,
                    shared_intermediate: expert,
                }
            );
            assert_eq!(
                adapter.parallel_vision_input_ranges.as_ref().unwrap()[0],
                first_vision
            );

            let attention = layout
                .tensor("model.layers.0.self_attn.q_proj.weight")
                .unwrap();
            assert_eq!(attention.logical_units(), Some(3));
            assert_eq!(attention.local_shape(), &[query_heads as usize * 2, 16]);
            assert_eq!(
                layout
                    .tensor("model.layers.1.moe.experts.gate_up_proj")
                    .unwrap()
                    .local_shape(),
                &[2, 2 * expert as usize, 16]
            );

            let identity = adapter.prompt_cache_model_identity(Some(topology)).unwrap();
            match identity.layer_layout.get(0).unwrap() {
                crate::LayerCachePolicy::KeyValueWithFixedState {
                    num_key_value_heads,
                    ..
                } => assert_eq!(num_key_value_heads.get(), kv_heads as u32),
                policy => panic!("unexpected Inkling cache policy {policy:?}"),
            }
        }
    }

    #[test]
    fn cartesian_layer_composes_uneven_ep_ownership_with_tp_geometry() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut value = config();
        value["text_config"]["n_routed_experts"] = 5.into();
        value["text_config"]["num_attention_heads"] = 4.into();
        value["text_config"]["num_key_value_heads"] = 2.into();
        value["text_config"]["head_dim"] = 4.into();
        value["text_config"]["swa_num_attention_heads"] = 4.into();
        value["text_config"]["swa_num_key_value_heads"] = 2.into();
        value["text_config"]["swa_head_dim"] = 4.into();
        let args = resident::model_args_from_config_value(&value).unwrap();

        for rank in 0..12 {
            let topology = ParallelTopology::from_rank(
                12,
                rank,
                2,
                2,
                3,
                DeviceAssignment::new(DeviceType::Cpu, 0),
            )
            .unwrap();
            let mut adapter =
                InklingLayerwiseAdapter::new(args.clone(), execution.stream()).unwrap();
            let context = ParallelBuildContext::new(topology, ShardingPolicy::Require);
            let mut planner = context.planner();
            adapter
                .register_parallel_parameters(context, &mut planner, execution.stream())
                .unwrap();
            let (_, layout) = planner.finish().unwrap();
            adapter
                .configure_parallel_static(context, &layout, execution.stream())
                .unwrap();
            let assignment = adapter
                .expert_parallel_assignment(topology)
                .unwrap()
                .unwrap();
            let layer = adapter
                .new_cartesian_layer(0, 1, Some(&layout), Some(&assignment), execution.stream())
                .unwrap();
            let parameters = layer.parameters().flatten();
            assert_eq!(
                parameters["moe.experts.gate_up_proj"].shape(),
                &[assignment.local_expert_count() as i32, 8, 16]
            );
            assert_eq!(assignment.group_size(), 3);
        }

        let topology =
            ParallelTopology::from_rank(4, 0, 2, 2, 1, DeviceAssignment::new(DeviceType::Cpu, 0))
                .unwrap();
        let adapter =
            InklingLayerwiseAdapter::new_external_experts(args, execution.stream()).unwrap();
        let assignment = adapter
            .expert_parallel_assignment(topology)
            .unwrap()
            .unwrap();
        assert_eq!(assignment.group_size(), 1);
        assert_eq!(assignment.local_expert_count(), 5);
    }

    #[test]
    fn tensor_parallel_plan_keeps_quantized_inkling_intermediates_block_aligned() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut config = config();
        config["text_config"]["num_attention_heads"] = 6.into();
        config["text_config"]["num_key_value_heads"] = 3.into();
        config["text_config"]["head_dim"] = 2.into();
        config["text_config"]["swa_num_attention_heads"] = 6.into();
        config["text_config"]["swa_num_key_value_heads"] = 3.into();
        config["text_config"]["swa_head_dim"] = 2.into();
        config["text_config"]["dense_intermediate_size"] = 160.into();
        config["text_config"]["moe_intermediate_size"] = 160.into();
        let mut args = resident::model_args_from_config_value(&config).unwrap();
        let affine = AffineQuantization::new(32, 4).unwrap().into();
        args.text_config.quantized_weight_configs = Some(HashMap::from([
            ("model.layers.0.dense.down_proj.weight".into(), affine),
            ("model.layers.1.moe.experts.down_proj".into(), affine),
            ("model.layers.2.moe.experts.down_proj".into(), affine),
        ]));

        for (rank, local_width, packed_width, scale_width) in
            [(0, 96usize, 12usize, 3usize), (1, 64, 8, 2)]
        {
            let mut adapter =
                InklingLayerwiseAdapter::new(args.clone(), execution.stream()).unwrap();
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
            adapter
                .register_parallel_parameters(context, &mut planner, execution.stream())
                .unwrap();
            let (_, layout) = planner.finish().unwrap();
            adapter
                .configure_parallel_static(context, &layout, execution.stream())
                .unwrap();

            assert_eq!(
                layout
                    .tensor("model.layers.0.dense.down_proj.inner.weight")
                    .unwrap()
                    .local_shape(),
                &[16, packed_width]
            );
            assert_eq!(
                layout
                    .tensor("model.layers.0.dense.down_proj.scales")
                    .unwrap()
                    .local_shape(),
                &[16, scale_width]
            );
            assert_eq!(
                layout
                    .tensor("model.layers.1.moe.experts.gate_up_proj")
                    .unwrap()
                    .local_shape(),
                &[2, 2 * local_width, 16]
            );
            assert_eq!(
                layout
                    .tensor("model.layers.1.moe.experts.down_proj")
                    .unwrap()
                    .local_shape(),
                &[2, 16, packed_width]
            );
            assert_eq!(
                layout
                    .tensor("model.layers.1.moe.experts.down_proj_scales")
                    .unwrap()
                    .local_shape(),
                &[2, 16, scale_width]
            );
            assert_eq!(
                layout
                    .tensor("model.layers.1.moe.shared_experts.gate_up_proj")
                    .unwrap()
                    .local_shape(),
                &[1, 160, 16]
            );
            assert_eq!(
                adapter.parallel_text_geometry.as_ref().unwrap()[1].feed_forward,
                resident::ParallelFeedForwardGeometry::SparseMoe {
                    routed_intermediate: local_width as i32,
                    shared_intermediate: 80,
                }
            );
        }
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

    fn write_fixture_with_config(
        dir: &Path,
        model: &Model,
        config: &serde_json::Value,
        stream: &Stream,
    ) {
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
        fs::write(dir.join("config.json"), serde_json::to_vec(config).unwrap()).unwrap();
    }

    fn write_fixture(dir: &Path, model: &Model, stream: &Stream) {
        write_fixture_with_config(dir, model, &config(), stream);
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

        let mut resident = load_inkling_layerwise_model(
            dir.path(),
            LayerWeightResidency::FullyResident,
            None,
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let options = LayerwiseLoadOptions::new(OffloadConfig::new(None, None, depth).unwrap());
        let mut layerwise =
            load_inkling_layerwise_model(dir.path(), options, None, gpu.stream(), cpu.stream())
                .unwrap();
        let mut resident_cache = resident.new_cache();
        let mut layerwise_cache = layerwise.new_cache();
        for tokens in [
            Array::from_slice(&[1u32, 2, 3], &[1, 3]),
            Array::from_slice(&[4u32], &[1, 1]),
            Array::from_slice(&[5u32], &[1, 1]),
            Array::from_slice(&[6u32], &[1, 1]),
        ] {
            let expected = resident
                .forward(&tokens, &mut resident_cache, gpu.stream())
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
        let options = DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1).unwrap();
        let model = load_inkling_tensor_parallel_layerwise_model(
            dir.path(),
            LayerWeightResidency::DenseDiskStream(options),
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
        let mut resident = load_inkling_layerwise_model(
            dir.path(),
            LayerWeightResidency::FullyResident,
            None,
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let options =
            ExpertCacheLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap(), 768, 768)
                .unwrap();
        let mut cached = load_inkling_expert_cache_model(
            dir.path(),
            crate::NonExpertWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                OffloadConfig::new(None, None, 1).unwrap(),
            )),
            options,
            None,
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut resident_cache = resident.new_cache();
        let mut cached_cache = cached.new_cache();
        for tokens in [
            Array::from_slice(&[1u32, 2, 3], &[1, 3]),
            Array::from_slice(&[4u32], &[1, 1]),
        ] {
            let expected = resident
                .forward(&tokens, &mut resident_cache, gpu.stream())
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
    #[ignore = "requires an MLX Metal device"]
    fn inkling_quantized_expert_cache_is_bounded_empty_route_safe_and_persistent() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let config = quantizable_config();
        let args = resident::model_args_from_config_value(&config).unwrap();
        let mut fixture = Model::new(args, gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture_with_config(dir.path(), &fixture, &config, gpu.stream());
        let mut resident = load_inkling_layerwise_model(
            dir.path(),
            LayerWeightResidency::FullyResident,
            None,
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let expert_options = ExpertCacheLoadOptions::new(
            OffloadConfig::new(Some(1 << 20), Some(1 << 20), 1).unwrap(),
            1 << 16,
            1 << 16,
        )
        .unwrap();
        let quantization: WeightQuantization = AffineQuantization::new(32, 4).unwrap().into();
        let mut cached = load_inkling_expert_cache_model(
            dir.path(),
            crate::NonExpertWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                OffloadConfig::new(Some(1 << 20), Some(1 << 20), 1).unwrap(),
            )),
            expert_options,
            Some(quantization),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();

        let mut resident_cache = resident.new_cache();
        let mut cached_cache = cached.new_cache();
        for tokens in [
            Array::from_slice(&[1u32, 2, 3], &[1, 3]),
            Array::from_slice(&[4u32], &[1, 1]),
        ] {
            let expected = resident
                .forward(&tokens, &mut resident_cache, gpu.stream())
                .unwrap();
            let actual = cached
                .forward(&tokens, &mut cached_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected, gpu.stream());
        }

        let ordinary = cached.residency_report().unwrap();
        let ordinary_materialization = ordinary.materialization().unwrap();
        assert!(ordinary_materialization.transformed_weights > 0);
        assert!(ordinary_materialization.source_bytes_read > ordinary_materialization.output_bytes);
        let report = cached.expert_cache_report().unwrap().unwrap();
        assert_eq!(report.weight_quantization, Some(quantization));
        let expert_materialization = report.materialization.as_ref().unwrap();
        assert!(expert_materialization.transformed_weights > 0);
        assert!(expert_materialization.source_bytes_read > expert_materialization.output_bytes);
        assert!(
            expert_materialization.peak_planned_working_set_bytes
                <= expert_options.compact_bank_scratch_bytes
        );
        assert!(report.owned_bytes < expert_materialization.source_bytes_read);

        let empty_hidden = zeros_dtype(&[0, 32], Dtype::Float32, gpu.stream()).unwrap();
        let empty_ids = zeros_dtype(&[0, 1], Dtype::Int32, gpu.stream()).unwrap();
        let empty_weights = zeros_dtype(&[0, 1], Dtype::Float32, gpu.stream()).unwrap();
        let sparse_layer = cached
            .args()
            .text_config
            .layer_schedule
            .iter()
            .position(|policy| policy.feed_forward == resident::FeedForwardPolicy::SparseMoe)
            .unwrap();
        let empty = cached
            .execution
            .adapter()
            .expert_cache
            .as_ref()
            .unwrap()
            .execute_routes_bounded(
                ExpertRouteBatch::new(
                    sparse_layer,
                    &empty_hidden,
                    &empty_ids,
                    &empty_weights,
                    ExpertPass::Decode,
                ),
                gpu.stream(),
                |hidden, acquired, _, _| {
                    assert!(acquired.identities().is_empty());
                    Ok(hidden.clone())
                },
            )
            .unwrap();
        assert_eq!(empty.shape(), &[0, 32]);

        let paged = PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1)
            .unwrap()
            .with_full_attention(true);
        let mut original = cached
            .new_cache_with_options(CacheResidencyPolicy::Paged(paged.clone()))
            .unwrap();
        let prefix = [1_u32, 2, 3];
        cached
            .forward(
                &Array::from_slice(&prefix, &[1, prefix.len() as i32]),
                &mut original,
                gpu.stream(),
            )
            .unwrap();
        let identity = cached.prompt_cache_model_identity().unwrap();
        let descriptor = PromptCacheDescriptor {
            model_family: identity.model_family,
            effective_model_type: identity.effective_model_type,
            checkpoint_fingerprint: "inkling-quantized-expert-cache".into(),
            prefix_content_fingerprint: "tokens:1,2,3".into(),
            architecture_fingerprint: identity.architecture_fingerprint,
            layer_count: identity.layer_count,
            global_layer_start: identity.global_layer_start,
            global_layer_end: identity.global_layer_end,
            batch_size: 1,
            layer_prefix_offsets: identity.layer_prefix_offsets,
            layer_layout: identity.layer_layout,
            sink_tokens: identity.sink_tokens,
            topology: identity.topology,
        };
        let persisted = tempfile::tempdir().unwrap();
        let destination = persisted.path().join("prompt-cache");
        cached
            .save_prompt_cache(
                &mut original,
                &destination,
                descriptor.clone(),
                &prefix,
                &PromptCacheOptions::default(),
                gpu.stream(),
            )
            .unwrap();
        let (mut restored, _) = cached
            .load_prompt_cache(&destination, &descriptor, &prefix, paged, gpu.stream())
            .unwrap();
        let next = Array::from_slice(&[4u32], &[1, 1]);
        let expected = cached.forward(&next, &mut original, gpu.stream()).unwrap();
        let actual = cached.forward(&next, &mut restored, gpu.stream()).unwrap();
        assert_close(&actual, &expected, gpu.stream());

        crate::architectures::distributed::expert::assert_rank_owned_quantized_sparse_ep_load(
            dir.path(),
            expert_options,
            quantization,
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

        let mut resident = load_inkling_layerwise_model(
            dir.path(),
            LayerWeightResidency::FullyResident,
            None,
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut layerwise = load_inkling_layerwise_model(
            dir.path(),
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
            None,
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

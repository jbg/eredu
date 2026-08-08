//! Unified Llama/Mistral loading across weight-residency policies.

use std::{collections::HashMap, path::Path, sync::Arc};

use safemlx::{
    error::Exception,
    module::Module,
    nn,
    ops::indexing::TryIndexOp,
    ops::{GgufCheckpoint, GgufMetadataValue},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};

use crate::{
    api::{
        common::{
            generation::CausalLm,
            linear::{
                build_unloaded_maybe_quantized_lm_head_with_quantization,
                project_logits_maybe_quantized, unloaded_maybe_quantized_embedding,
            },
        },
        input,
        llama::{self as resident, AttentionInput, ModelArgs, TransformerBlock},
    },
    error::Error,
    nn::{
        parallel::{
            planned_kv_head_layout, register_gqa_projection_group,
            register_swiglu_projection_group, GqaProjectionNames, SwiGluProjectionNames,
            VocabParallelEmbedding, VocabParallelLmHead,
        },
        tensor::{create_attention_mask, AttentionMask},
    },
    runtime::cache::residency::{
        open_prompt_cache, CacheResidencyManager, CacheResidencyPolicy, CacheResidencyReport,
        LayerCachePolicy, PagedCacheOptions, PromptCacheDescriptor, PromptCacheManifest,
        PromptCacheModelIdentity, PromptCacheOptions, PromptCacheTopology,
    },
    runtime::cache::{ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache},
    runtime::checkpoint::binding::{build_module_bindings, populate_module_from_lease},
    runtime::checkpoint::{
        quantization::WeightQuantization,
        store::{GgufWeightStore, WeightStore},
    },
    runtime::distributed::parallel::{register_replicated_module, ParallelPlanBuilder},
    runtime::execution::layerwise::{
        load_layerwise_model, load_layerwise_model_with_quantization,
        load_safetensors_layerwise_model, load_tensor_parallel_layerwise_model,
        open_safetensors_weight_store, transformed_module_weight_store, ArchitectureAdapter,
        DenseDiskStreamReport, ExecutionResidency, LayerWeightResidency, LayerwiseForwardState,
        LayerwiseLoadOptions, LayerwiseModel, LayerwiseModelMetadata, LoadTimeQuantizableAdapter,
        StaticUnitBindings, WeightResidency,
    },
    runtime::residency::manager::{ResidencyReport, ResidentUnitLease, WeightBinding},
};

const EMBEDDING_UNIT: &str = "llama.static.embedding";
const NORM_UNIT: &str = "llama.static.norm";
const HEAD_UNIT: &str = "llama.static.output";

/// Options for the unified Llama/Mistral loader.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Default)]
pub struct LlamaLoadOptions {
    /// Determines where decoder weights live and how they execute.
    pub weight_residency: WeightResidency,
}

impl LlamaLoadOptions {
    /// Selects the eager engine with every parameter on the execution device.
    pub const fn fully_resident() -> Self {
        Self {
            weight_residency: WeightResidency::fully_resident(),
        }
    }

    /// Selects host-backed decoder layers with a bounded device window.
    pub const fn layerwise_host(options: LayerwiseLoadOptions) -> Self {
        Self {
            weight_residency: WeightResidency::layerwise_host(options),
        }
    }

    /// Selects experimental dense disk streaming with finite tier budgets.
    pub const fn dense_disk_stream(
        options: crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions,
    ) -> Self {
        Self {
            weight_residency: WeightResidency::dense_disk_stream(options),
        }
    }
}

/// Per-layer Llama/Mistral KV state selected from the canonical schedule.
#[derive(Debug, Clone)]
pub enum LlamaCache {
    /// Device caches, independently full or bounded for each layer.
    Device(Vec<Option<ConcatKeyValueCache>>),
    /// Block-addressable caches sharing one model-wide residency manager.
    Paged(Vec<Option<PagedKeyValueCache>>),
}

impl LlamaCache {
    /// Returns the common absolute token offset, or zero for an empty cache.
    pub fn offset(&self) -> i32 {
        match self {
            Self::Device(caches) => caches
                .first()
                .and_then(Option::as_ref)
                .map_or(0, KeyValueCache::offset),
            Self::Paged(caches) => caches
                .first()
                .and_then(Option::as_ref)
                .map_or(0, KeyValueCache::offset),
        }
    }

    /// Clears retained arrays without changing cache type or window size.
    pub fn clear(&mut self) -> Result<(), Error> {
        match self {
            Self::Device(caches) => caches.iter_mut().flatten().for_each(|cache| cache.clear()),
            Self::Paged(caches) => {
                for cache in caches.iter_mut().flatten() {
                    cache.clear()?;
                }
            }
        }
        Ok(())
    }

    /// Returns aggregate cache-residency telemetry for a paged cache.
    pub fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Error> {
        match self {
            Self::Paged(caches) => caches
                .iter()
                .flatten()
                .next()
                .map(|cache| cache.report().map_err(Into::into))
                .transpose(),
            Self::Device(_) => Ok(None),
        }
    }

    /// Finalizes every mutable tail and atomically persists a completed text prefix.
    pub fn save_prompt_cache(
        &mut self,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        let Self::Paged(caches) = self else {
            return Err(Exception::custom(
                "prompt-cache persistence requires an explicitly configured paged cache",
            )
            .into());
        };
        for cache in caches.iter_mut().flatten() {
            cache.finalize()?;
        }
        let manager = caches
            .iter()
            .flatten()
            .next()
            .ok_or_else(|| Exception::custom("cannot persist an empty paged cache"))?
            .manager()
            .clone();
        manager
            .save_prompt_cache(destination, descriptor, prefix_token_ids, &[], options)
            .map_err(|error| Exception::custom(error.to_string()).into())
    }
}

/// Llama/Mistral causal LM whose execution engine follows its residency policy.
pub struct LlamaModel {
    execution: Box<LayerwiseModel<LlamaLayerwiseAdapter>>,
}

impl LlamaModel {
    /// Returns normalized model arguments regardless of execution engine.
    pub fn args(&self) -> &ModelArgs {
        self.execution.adapter().args()
    }

    /// Returns the canonical cache-relevant architecture identity.
    pub fn prompt_cache_architecture_fingerprint(&self) -> String {
        crate::architectures::llama::model::prompt_cache_architecture_fingerprint(self.args())
    }

    /// Returns this rank's exact prompt-cache state layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        self.execution.prompt_cache_layer_layout()
    }

    /// Returns whether all parameters use the eager execution-device engine.
    pub fn is_fully_resident(&self) -> bool {
        self.execution.metadata().residency() == ExecutionResidency::FullyResident
    }

    /// Returns canonical parameter and residency metadata.
    pub fn metadata(&self) -> &LayerwiseModelMetadata {
        self.execution.metadata()
    }

    /// Returns rank-local generalized parallel information when applicable.
    pub fn parallel_info(
        &self,
    ) -> Option<&crate::runtime::execution::layerwise::ParallelModelInfo> {
        self.execution.parallel_info()
    }

    /// Returns logical residency and transfer telemetry for a layerwise model.
    pub fn residency_report(&self) -> Result<Option<ResidencyReport>, Error> {
        Ok(Some(self.execution.residency_report()?))
    }

    /// Returns dense-stream observations when that policy is active.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        self.execution.dense_stream_report()
    }

    /// Returns the persistent checkpoint store used by a layerwise model.
    pub fn checkpoint_store(&self) -> &(dyn WeightStore + Send + Sync) {
        self.execution.checkpoint_store()
    }

    /// Returns the number of pinned static leases used by the layerwise engine.
    pub fn static_lease_count(&self) -> usize {
        self.execution.static_lease_count()
    }

    /// Creates the cache representation required by the model configuration.
    pub fn new_cache(&self) -> LlamaCache {
        let args = self.args();
        LlamaCache::Device(
            args.attention_schedule
                .iter()
                .map(|policy| {
                    Some(match policy.window() {
                        Some(window) => ConcatKeyValueCache::new_for_sliding_attention(
                            i32::try_from(window.get())
                                .expect("validated Llama attention window fits i32"),
                        ),
                        None => ConcatKeyValueCache::new(),
                    })
                })
                .collect(),
        )
    }

    /// Creates a device-resident or explicitly bounded paged model cache.
    pub fn new_cache_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<LlamaCache, Error> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => {
                self.new_paged_cache(options, None, self.execution.prompt_cache_rank_identity())
            }
        }
    }

    /// Catalogs a compatible reusable prefix without loading all cache blocks.
    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(LlamaCache, PromptCacheManifest), Error> {
        self.execution
            .load_prompt_cache(directory, expected, prefix_token_ids, options, stream)
    }

    /// Persists a prefix through the generalized execution contract.
    pub fn save_prompt_cache(
        &self,
        cache: &mut LlamaCache,
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

    fn new_paged_cache(
        &self,
        options: PagedCacheOptions,
        manager: Option<CacheResidencyManager>,
        rank: Option<crate::CacheRankIdentity>,
    ) -> Result<LlamaCache, Error> {
        let manager = match manager {
            Some(manager) => manager,
            None => CacheResidencyManager::new(options)
                .map_err(|error| Exception::custom(error.to_string()))?,
        };
        self.new_paged_cache_from_manager(manager, rank)
    }

    fn new_paged_cache_from_manager(
        &self,
        manager: CacheResidencyManager,
        rank: Option<crate::CacheRankIdentity>,
    ) -> Result<LlamaCache, Error> {
        let args = self.args();
        let caches = args
            .attention_schedule
            .iter()
            .enumerate()
            .map(|(layer, policy)| {
                let window = policy.window().map(|window| {
                    i32::try_from(window.get()).expect("validated Llama attention window fits i32")
                });
                PagedKeyValueCache::new_with_layout(manager.clone(), layer, window, 0, rank)
                    .map(Some)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(LlamaCache::Paged(caches))
    }

    /// Runs embedding, decoder layers, final normalization, and projection.
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut LlamaCache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.validate_cache(cache)?;
        self.execution
            .forward(LlamaAdapterInput { inputs, mask: None }, cache, stream)
    }

    /// Runs the canonical execution path with stable per-layer observation points.
    pub fn forward_with_observer(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut LlamaCache,
        stream: &Stream,
        observer: &mut dyn crate::runtime::execution::inspection::ActivationObserver,
    ) -> Result<Array, Error> {
        self.validate_cache(cache)?;
        self.execution.forward_with_observer(
            LlamaAdapterInput { inputs, mask },
            cache,
            stream,
            observer,
        )
    }

    /// Runs a rank-local tensor-parallel forward pass.
    pub fn forward_tensor_parallel(
        &mut self,
        inputs: &Array,
        cache: &mut LlamaCache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.validate_cache(cache)?;
        self.execution.forward_tensor_parallel(
            LlamaAdapterInput { inputs, mask: None },
            cache,
            group,
            stream,
        )
    }

    /// Runs prompt prefill and returns last-token logits.
    pub fn prefill(
        &mut self,
        inputs: &Array,
        cache: &mut LlamaCache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward(inputs, cache, stream)?
            .try_index_device((.., -1, ..), stream)
            .map_err(Into::into)
    }

    /// Runs cached decode and returns last-token logits.
    pub fn decode(
        &mut self,
        input_tokens: &Array,
        cache: &mut LlamaCache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.prefill(input_tokens, cache, stream)
    }

    /// Clears temporary execution-device decoder copies when layerwise residency is active.
    ///
    /// Returns `true` when a layerwise window was cleared and `false` for the
    /// fully resident engine.
    pub fn clear_device_layer_window(&self) -> Result<bool, Error> {
        if self.is_fully_resident() {
            return Ok(false);
        }
        self.execution.clear_device_group("text_decoder")?;
        Ok(true)
    }

    fn validate_cache(&self, cache: &LlamaCache) -> Result<(), Error> {
        let expected_layers = usize::try_from(self.args().num_hidden_layers).map_err(|_| {
            LlamaModelError::InvalidLayerCount {
                count: self.args().num_hidden_layers,
            }
        })?;
        let actual_layers = match cache {
            LlamaCache::Device(caches) => caches.len(),
            LlamaCache::Paged(caches) => caches.len(),
        };
        if actual_layers != expected_layers {
            return Err(LlamaModelError::CacheLengthMismatch {
                expected: expected_layers,
                actual: actual_layers,
            }
            .into());
        }
        match cache {
            LlamaCache::Device(caches) => {
                validate_cache_policies(caches, &self.args().attention_schedule)?
            }
            LlamaCache::Paged(caches) => {
                validate_cache_policies(caches, &self.args().attention_schedule)?
            }
        }
        Ok(())
    }
}

fn validate_cache_policies<C: KeyValueCache>(
    caches: &[Option<C>],
    schedule: &crate::runtime::attention::LayerSchedule<crate::runtime::attention::AttentionPolicy>,
) -> Result<(), Error> {
    for (layer, (cache, policy)) in caches.iter().zip(schedule.iter()).enumerate() {
        let cache = cache
            .as_ref()
            .ok_or_else(|| Exception::custom(format!("Llama cache is missing layer {layer}")))?;
        let expected = policy.window().map(|window| {
            i32::try_from(window.get()).expect("validated Llama attention window fits i32")
        });
        if cache.max_size() != expected {
            return Err(Exception::custom(format!(
                "Llama cache policy mismatch at layer {layer}: expected {policy:?}, cache window is {:?}",
                cache.max_size()
            ))
            .into());
        }
    }
    Ok(())
}

impl CausalLm<LlamaCache> for LlamaModel {
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut LlamaCache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        self.prefill(&tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut LlamaCache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.decode(input_tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))
    }
}

/// Loads a Llama/Mistral safetensors model using the selected residency policy.
pub fn load_llama_model(
    model_dir: impl AsRef<Path>,
    options: LlamaLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LlamaModel, Error> {
    let model_dir = model_dir.as_ref();
    crate::api::structural::validate_safetensors_load_path(
        crate::api::ModelKind::Llama,
        model_dir,
        crate::api::ModelLoadOptions::default().with_weight_residency(options.weight_residency),
    )?;
    if options.weight_residency.expert_cache().is_some() {
        return Err(Error::UnsupportedArchitecture(
            "independent expert caching is not supported for Llama checkpoints".into(),
        ));
    }
    let execution_options = options.weight_residency.layers();
    let args = resident::get_llama_model_args(model_dir)?;
    let adapter = LlamaLayerwiseAdapter::new(args, stream)?;
    Ok(LlamaModel {
        execution: Box::new(load_safetensors_layerwise_model(
            model_dir,
            adapter,
            execution_options,
            stream,
            weights_stream,
        )?),
    })
}

pub(crate) fn execute_transformed_llama_model(
    model: resident::ResidentModel,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LlamaModel, Error> {
    let adapter = LlamaLayerwiseAdapter::new(model.args.clone(), stream)?;
    let store = transformed_module_weight_store(&model)?;
    Ok(LlamaModel {
        execution: Box::new(load_layerwise_model(
            store,
            adapter,
            LayerWeightResidency::FullyResident,
            stream,
            weights_stream,
        )?),
    })
}

/// Loads Llama/Mistral through the generalized tensor-parallel execution engine.
pub fn load_llama_tensor_parallel_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    build: crate::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LlamaModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let residency = options.weight_residency();
    if model_dir
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        let checkpoint = GgufCheckpoint::open(model_dir)?;
        let metadata = crate::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        return load_llama_gguf_tensor_parallel_model(
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
        crate::api::ModelKind::Llama,
        model_dir,
        crate::api::ModelLoadOptions::default().with_weight_residency(residency),
    )?;
    let args = resident::get_llama_model_args(model_dir)?;
    let adapter = LlamaLayerwiseAdapter::new(args, stream)?;
    Ok(LlamaModel {
        execution: Box::new(load_tensor_parallel_layerwise_model(
            open_safetensors_weight_store(model_dir, options.max_mapped_shards())?,
            adapter,
            options,
            build,
            stream,
            weights_stream,
        )?),
    })
}

pub(crate) fn load_llama_gguf_tensor_parallel_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: LayerWeightResidency,
    build: crate::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(LlamaModel, Vec<u32>), Error> {
    crate::runtime::execution::layerwise::validate_gguf_layerwise_source(
        checkpoint, metadata, options,
    )?;
    let prepared =
        resident::prepare_llama_gguf_checkpoint(checkpoint, metadata, None, weights_stream)?;
    let store: Arc<dyn WeightStore + Send + Sync> =
        Arc::new(GgufWeightStore::new_with_max_mapped_shards(
            checkpoint.clone(),
            resident::translate_gguf_weight_name,
            options.max_mapped_shards(),
        )?);
    let adapter = LlamaLayerwiseAdapter::new(prepared.args, stream)?;
    let execution = load_tensor_parallel_layerwise_model(
        store,
        adapter,
        options,
        build,
        stream,
        weights_stream,
    )?;
    Ok((
        LlamaModel {
            execution: Box::new(execution),
        },
        prepared.eos_token_ids,
    ))
}

/// Loads a Llama/Mistral GGUF checkpoint using the selected residency policy.
pub(crate) fn load_llama_gguf_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(LlamaModel, Vec<u32>), Error> {
    let prepared =
        resident::prepare_llama_gguf_checkpoint(checkpoint, metadata, None, weights_stream)?;
    let store: Arc<dyn WeightStore + Send + Sync> =
        Arc::new(GgufWeightStore::new_with_max_mapped_shards(
            checkpoint.clone(),
            resident::translate_gguf_weight_name,
            residency.max_mapped_shards(),
        )?);
    let adapter = LlamaLayerwiseAdapter::new(prepared.args, stream)?;
    if residency.expert_cache().is_some() {
        return Err(Error::UnsupportedArchitecture(
            "independent expert caching is not supported for Llama GGUF checkpoints".into(),
        ));
    }
    let execution_options = residency.layers();
    Ok((
        LlamaModel {
            execution: Box::new(load_layerwise_model_with_quantization(
                store,
                adapter,
                execution_options,
                quantization,
                stream,
                weights_stream,
            )?),
        },
        prepared.eos_token_ids,
    ))
}

/// Llama implementation of the generic layerwise model-family contract.
pub struct LlamaLayerwiseAdapter {
    args: ModelArgs,
    embedding: MaybeQuantized<nn::Embedding>,
    norm: nn::RmsNorm,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_embedding: Option<VocabParallelEmbedding>,
    parallel_lm_head: Option<VocabParallelLmHead>,
    parallel_kv_heads: Option<Vec<i32>>,
}

impl LlamaLayerwiseAdapter {
    /// Creates metadata-only static modules for a normalized Llama configuration.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let embedding = unloaded_maybe_quantized_embedding(
            args.vocab_size,
            args.hidden_size,
            args.affine_quantization_for("model.embed_tokens.weight"),
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
                args.affine_quantization_for("lm_head.weight"),
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
        })
    }

    /// Returns normalized Llama arguments.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }
}

impl LoadTimeQuantizableAdapter for LlamaLayerwiseAdapter {
    fn load_time_quantized(
        &self,
        quantization: WeightQuantization,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let mut args = self.args.clone();
        args.quantization = Some(quantization);
        args.quantization_config = None;
        args.quantized_weights = None;
        args.quantized_weight_configs = None;
        Self::new(args, stream)
    }
}

/// Llama mask state shared by every temporary decoder block.
pub struct LlamaForwardContext {
    mask: Option<Array>,
    allow_sliding_prefill: bool,
}

/// Borrowed tokens and optional mask consumed by the canonical Llama adapter.
pub struct LlamaAdapterInput<'a> {
    /// Token ids with shape `[batch, sequence]`.
    pub inputs: &'a Array,
    /// Optional caller-provided attention mask.
    pub mask: Option<&'a Array>,
}

fn register_llama_layer_parallel_plan(
    planner: &mut ParallelPlanBuilder,
    layer: &TransformerBlock,
    args: &ModelArgs,
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
    register_replicated_module(
        planner,
        &attention.rope,
        &format!("{prefix}.self_attn.rope"),
    )?;
    register_swiglu_projection_group(
        planner,
        &format!("{prefix}.mlp"),
        SwiGluProjectionNames {
            gate: "gate_proj",
            up: "up_proj",
            down: "down_proj",
        },
        &layer.mlp.gate_proj,
        &layer.mlp.up_proj,
        &layer.mlp.down_proj,
        args.intermediate_size,
    )?;
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

impl ArchitectureAdapter for LlamaLayerwiseAdapter {
    type Input<'a> = LlamaAdapterInput<'a>;
    type Cache = LlamaCache;
    type Layer = TransformerBlock;
    type ForwardContext = LlamaForwardContext;

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
        let layer_count = usize::try_from(self.args.num_hidden_layers)
            .map_err(|_| Exception::custom("invalid Llama cache layer count"))?;
        let local_kv_heads = match topology {
            None => vec![self.args.num_key_value_heads; layer_count],
            Some(_) => self.parallel_kv_heads.clone().ok_or_else(|| {
                Error::Parallel(
                    "Llama parallel cache identity requested before local layout configuration"
                        .into(),
                )
            })?,
        };
        if local_kv_heads.len() != layer_count {
            return Err(Error::Parallel(format!(
                "Llama parallel cache geometry has {} layers, expected {layer_count}",
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
            model_family: "llama".into(),
            effective_model_type: self.args.model_type.clone(),
            architecture_fingerprint:
                crate::architectures::llama::model::prompt_cache_architecture_fingerprint(
                    &self.args,
                ),
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

    fn save_prompt_cache(
        &self,
        cache: &mut Self::Cache,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        _stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        cache.save_prompt_cache(destination, descriptor, prefix_token_ids, options)
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
        let rank = identity.topology.cache_rank_identity();
        let caches = self
            .args
            .attention_schedule
            .iter()
            .enumerate()
            .map(|(layer, policy)| {
                PagedKeyValueCache::new_with_layout(
                    manager.clone(),
                    layer,
                    policy.window().map(|window| {
                        i32::try_from(window.get())
                            .expect("validated Llama attention window fits i32")
                    }),
                    0,
                    rank,
                )
                .map(Some)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((LlamaCache::Paged(caches), manifest))
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
            return Err(Exception::custom(format!(
                "Llama adapter received {} static leases, expected {expected}",
                leases.len()
            ))
            .into());
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

    fn validate_cache(&self, cache: &mut Self::Cache) -> Result<(), Error> {
        let expected = usize::try_from(self.args.num_hidden_layers).map_err(|_| {
            LlamaModelError::InvalidLayerCount {
                count: self.args.num_hidden_layers,
            }
        })?;
        let actual = match cache {
            LlamaCache::Device(caches) => {
                if caches.is_empty() {
                    *caches = self
                        .args
                        .attention_schedule
                        .iter()
                        .map(|policy| {
                            Some(match policy.window() {
                                Some(window) => ConcatKeyValueCache::new_for_sliding_attention(
                                    i32::try_from(window.get())
                                        .expect("validated Llama attention window fits i32"),
                                ),
                                None => ConcatKeyValueCache::new(),
                            })
                        })
                        .collect();
                }
                validate_cache_policies(caches, &self.args.attention_schedule)?;
                caches.len()
            }
            LlamaCache::Paged(caches) => {
                validate_cache_policies(caches, &self.args.attention_schedule)?;
                caches.len()
            }
        };
        if actual != expected {
            return Err(LlamaModelError::CacheLengthMismatch { expected, actual }.into());
        }
        Ok(())
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        let hidden = self.embedding.forward(input.inputs, stream)?;
        let allow_sliding_prefill = input.mask.is_none();
        let mask = match cache {
            LlamaCache::Device(caches) => match input.mask {
                Some(mask) => Some(mask.clone()),
                None => llama_attention_mask(&hidden, caches, stream)?,
            },
            LlamaCache::Paged(caches) => match input.mask {
                Some(mask) => Some(mask.clone()),
                None => llama_attention_mask(&hidden, caches, stream)?,
            },
        };
        Ok(LayerwiseForwardState {
            hidden,
            context: LlamaForwardContext {
                mask,
                allow_sliding_prefill,
            },
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
        let allow_sliding_prefill = input.mask.is_none();
        let mask = match cache {
            LlamaCache::Device(caches) => match input.mask {
                Some(mask) => Some(mask.clone()),
                None => llama_attention_mask(&hidden, caches, execution.stream())?,
            },
            LlamaCache::Paged(caches) => match input.mask {
                Some(mask) => Some(mask.clone()),
                None => llama_attention_mask(&hidden, caches, execution.stream())?,
            },
        };
        Ok(LayerwiseForwardState {
            hidden,
            context: LlamaForwardContext {
                mask,
                allow_sliding_prefill,
            },
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
                "Llama has no group {group}"
            )));
        }
        usize::try_from(self.args.num_hidden_layers).map_err(|_| {
            LlamaModelError::InvalidLayerCount {
                count: self.args.num_hidden_layers,
            }
            .into()
        })
    }

    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        if group != 0 {
            return Err(Error::UnsupportedArchitecture(format!(
                "Llama has no group {group}"
            )));
        }
        let index =
            i32::try_from(index).map_err(|_| LlamaModelError::LayerIndexOverflow { index })?;
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
            register_llama_layer_parallel_plan(
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
                .affine_quantization_for("model.embed_tokens.weight"),
            context,
            stream,
        )?);
        if self.lm_head.is_some() {
            self.parallel_lm_head = Some(VocabParallelLmHead::unloaded(
                self.args.hidden_size,
                self.args.vocab_size as usize,
                self.args.affine_quantization_for("lm_head.weight"),
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
                "Llama has no group {group}"
            )));
        }
        let prefix = format!("model.layers.{index}");
        let q = layout
            .tensor(&format!("{prefix}.self_attn.q_proj.weight"))
            .or_else(|| layout.tensor(&format!("{prefix}.self_attn.q_proj.inner.weight")))
            .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} query")))?;
        let k = layout
            .tensor(&format!("{prefix}.self_attn.k_proj.weight"))
            .or_else(|| layout.tensor(&format!("{prefix}.self_attn.k_proj.inner.weight")))
            .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} key")))?;
        let gate = layout
            .tensor(&format!("{prefix}.mlp.gate_proj.weight"))
            .or_else(|| layout.tensor(&format!("{prefix}.mlp.gate_proj.inner.weight")))
            .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} MLP")))?;
        let mut args = self.args.clone();
        let query_width = i32::try_from(q.local_shape()[0])
            .map_err(|_| Error::Parallel("local Llama query width exceeds i32".into()))?;
        let key_width = i32::try_from(k.local_shape()[0])
            .map_err(|_| Error::Parallel("local Llama key width exceeds i32".into()))?;
        if query_width % args.head_dim != 0 || key_width % args.head_dim != 0 {
            return Err(Error::Parallel(format!(
                "local Llama attention widths q={query_width}, k={key_width} split head dimension {}",
                args.head_dim
            )));
        }
        args.num_attention_heads = query_width / args.head_dim;
        args.num_key_value_heads = key_width / args.head_dim;
        args.intermediate_size = i32::try_from(gate.local_shape()[0])
            .map_err(|_| Error::Parallel("local Llama MLP width exceeds i32".into()))?;
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
        format!("llama.layer.{index:05}")
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
            LlamaCache::Device(caches) => Ok(layer.forward(
                AttentionInput {
                    x: hidden,
                    mask: context.mask.as_ref(),
                    cache: Some(
                        caches[index]
                            .as_mut()
                            .expect("validated Llama device cache"),
                    ),
                    allow_sliding_prefill: context.allow_sliding_prefill,
                },
                stream,
            )?),
            LlamaCache::Paged(caches) => Ok(layer.forward(
                AttentionInput {
                    x: hidden,
                    mask: context.mask.as_ref(),
                    cache: Some(caches[index].as_mut().expect("validated Llama paged cache")),
                    allow_sliding_prefill: context.allow_sliding_prefill,
                },
                stream,
            )?),
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
            LlamaCache::Device(caches) => Ok(layer.forward_tensor_parallel(
                hidden,
                context.mask.as_ref(),
                caches[index].as_mut(),
                context.allow_sliding_prefill,
                tp_group,
                execution.stream(),
            )?),
            LlamaCache::Paged(caches) => Ok(layer.forward_tensor_parallel(
                hidden,
                context.mask.as_ref(),
                caches[index].as_mut(),
                context.allow_sliding_prefill,
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
            LlamaCache::Device(caches) => caches[index]
                .as_ref()
                .map(KeyValueCache::retained_arrays)
                .unwrap_or_default(),
            LlamaCache::Paged(caches) => caches[index]
                .as_ref()
                .map(KeyValueCache::retained_arrays)
                .unwrap_or_default(),
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

    fn ignores_checkpoint_key(&self, key: &str) -> bool {
        key.starts_with("rope_freqs.") || key.ends_with(".rotary_emb.inv_freq")
    }
}

#[cfg(test)]
mod tests {
    use safemlx::{Device, DeviceType, ExecutionContext};

    use super::*;
    use crate::runtime::{
        attention::LayerSchedule,
        checkpoint::quantization::AffineQuantization,
        distributed::{
            parallel::{ParallelBuildContext, ShardingPolicy},
            topology::{DeviceAssignment, ParallelTopology},
        },
    };

    fn build_context(rank: usize) -> ParallelBuildContext {
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

    #[test]
    fn quantized_llama_groups_share_uneven_aligned_head_and_mlp_ranges() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let quantization = AffineQuantization::new(32, 4).unwrap().into();
        let args = ModelArgs {
            model_type: "llama".into(),
            hidden_size: 96,
            num_hidden_layers: 1,
            intermediate_size: 96,
            num_attention_heads: 6,
            rms_norm_eps: 1e-5,
            vocab_size: 64,
            num_key_value_heads: 3,
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            rope_traditional: false,
            head_dim: 16,
            tie_word_embeddings: false,
            attention_bias: false,
            mlp_bias: false,
            rope_scaling: None,
            attention_schedule: LayerSchedule::all_full(1).unwrap(),
            quantization: None,
            quantization_config: None,
            quantized_weights: Some(
                [
                    "model.layers.0.self_attn.o_proj.weight".to_string(),
                    "model.layers.0.mlp.down_proj.weight".to_string(),
                ]
                .into_iter()
                .collect(),
            ),
            quantized_weight_configs: Some(HashMap::from([
                (
                    "model.layers.0.self_attn.o_proj.weight".into(),
                    quantization,
                ),
                ("model.layers.0.mlp.down_proj.weight".into(), quantization),
            ])),
        };
        let layer = TransformerBlock::new_for_layer(&args, 0, execution.stream()).unwrap();

        for (rank, local_heads, local_intermediate, packed_width, scale_width) in
            [(0, 2, 64, 8, 2), (1, 1, 32, 4, 1)]
        {
            let mut planner = build_context(rank).planner();
            register_llama_layer_parallel_plan(&mut planner, &layer, &args, "model.layers.0")
                .unwrap();
            let (_, layout) = planner.finish().unwrap();
            assert_eq!(
                planned_kv_head_layout(&layout, 1, 16, "model.layers").unwrap(),
                vec![local_heads]
            );
            assert_eq!(
                layout
                    .tensor("model.layers.0.self_attn.o_proj.inner.weight")
                    .unwrap()
                    .local_shape(),
                &[96, packed_width]
            );
            assert_eq!(
                layout
                    .tensor("model.layers.0.self_attn.o_proj.scales")
                    .unwrap()
                    .local_shape(),
                &[96, scale_width]
            );
            assert_eq!(
                layout
                    .tensor("model.layers.0.mlp.gate_proj.weight")
                    .unwrap()
                    .local_shape(),
                &[local_intermediate, 96]
            );
            assert_eq!(
                layout
                    .tensor("model.layers.0.mlp.down_proj.inner.weight")
                    .unwrap()
                    .local_shape(),
                &[96, packed_width]
            );
        }
    }
}

fn llama_attention_mask<C: KeyValueCache>(
    hidden: &Array,
    cache: &[Option<C>],
    stream: &Stream,
) -> Result<Option<Array>, Error> {
    match create_attention_mask(hidden, cache, Some(true), stream)? {
        Some(AttentionMask::Array(mask)) => Ok(Some(mask)),
        Some(AttentionMask::Causal) => Err(Exception::custom(
            "Llama-compatible decoders require an explicit attention mask",
        )
        .into()),
        None => Ok(None),
    }
}

/// Structured failures at the unified Llama model boundary.
#[derive(Debug, thiserror::Error)]
pub enum LlamaModelError {
    /// The normalized decoder count cannot be represented by this runtime.
    #[error("invalid Llama decoder layer count {count}")]
    InvalidLayerCount {
        /// Invalid configured count.
        count: i32,
    },
    /// A decoder index cannot be represented by the model implementation.
    #[error("Llama decoder index {index} exceeds the supported range")]
    LayerIndexOverflow {
        /// Invalid decoder index.
        index: usize,
    },
    /// A cache vector had the wrong number of layers.
    #[error("Llama cache has {actual} layers, expected {expected}")]
    CacheLengthMismatch {
        /// Model decoder count.
        expected: usize,
        /// Supplied cache count.
        actual: usize,
    },
}

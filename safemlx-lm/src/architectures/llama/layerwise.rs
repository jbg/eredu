//! Unified Llama/Mistral loading across weight-residency policies.

use std::{collections::HashMap, path::Path, sync::Arc};

use safemlx::{
    error::Exception,
    module::{Module, ModuleParameters},
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
        parallel::{VocabParallelEmbedding, VocabParallelLmHead},
        tensor::{create_attention_mask, AttentionMask},
    },
    runtime::cache::residency::{
        open_prompt_cache, validate_prompt_cache_model_identity, CacheResidencyManager,
        CacheResidencyPolicy, CacheResidencyReport, PagedCacheOptions, PromptCacheDescriptor,
        PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions, PromptCacheTopology,
    },
    runtime::cache::{ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache},
    runtime::checkpoint::binding::{build_module_bindings, populate_module_from_lease},
    runtime::checkpoint::store::{GgufWeightStore, WeightStore},
    runtime::distributed::parallel::{
        exact_parallel_division, register_projection_module, register_replicated_module,
        ParallelPlanBuilder, ProjectionSharding,
    },
    runtime::execution::layerwise::{
        load_layerwise_model, load_safetensors_layerwise_model,
        load_tensor_parallel_layerwise_model, open_safetensors_weight_store, ArchitectureAdapter,
        DenseDiskStreamReport, LayerExecutionLoadOptions, LayerwiseForwardState,
        LayerwiseLoadOptions, LayerwiseModel, LayerwiseModelMetadata, StaticUnitBindings,
        WeightResidency,
    },
    runtime::residency::manager::{ResidencyReport, ResidentUnitLease},
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
            weight_residency: WeightResidency::FullyResident,
        }
    }

    /// Selects host-backed decoder layers with a bounded device window.
    pub const fn layerwise_host(options: LayerwiseLoadOptions) -> Self {
        Self {
            weight_residency: WeightResidency::LayerwiseHost(options),
        }
    }

    /// Selects experimental dense disk streaming with finite tier budgets.
    pub const fn dense_disk_stream(
        options: crate::runtime::residency::dense_stream::DenseDiskStreamLoadOptions,
    ) -> Self {
        Self {
            weight_residency: WeightResidency::DenseDiskStream(options),
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

enum LlamaExecution {
    FullyResident(Box<resident::ResidentModel>),
    LayerwiseHost(Box<LayerwiseModel<LlamaLayerwiseAdapter>>),
}

/// Llama/Mistral causal LM whose execution engine follows its residency policy.
pub struct LlamaModel {
    execution: LlamaExecution,
}

impl LlamaModel {
    /// Returns normalized model arguments regardless of execution engine.
    pub fn args(&self) -> &ModelArgs {
        match &self.execution {
            LlamaExecution::FullyResident(model) => &model.args,
            LlamaExecution::LayerwiseHost(model) => model.adapter().args(),
        }
    }

    /// Returns the canonical cache-relevant architecture identity.
    pub fn prompt_cache_architecture_fingerprint(&self) -> String {
        crate::architectures::llama::model::prompt_cache_architecture_fingerprint(self.args())
    }

    /// Returns this rank's exact prompt-cache state layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        match &self.execution {
            LlamaExecution::LayerwiseHost(model) => model.prompt_cache_layer_layout(),
            LlamaExecution::FullyResident(_) => {
                let args = self.args();
                PromptCacheModelIdentity::key_value_layouts(
                    args.attention_schedule.iter().map(|policy| {
                        policy.window().map(|window| {
                            i32::try_from(window.get())
                                .expect("validated Llama attention window fits i32")
                        })
                    }),
                    args.num_key_value_heads,
                    args.head_dim,
                )
                .map_err(|error| Error::Parallel(error.to_string()))
            }
        }
    }

    /// Returns whether all parameters use the eager execution-device engine.
    pub const fn is_fully_resident(&self) -> bool {
        matches!(&self.execution, LlamaExecution::FullyResident(_))
    }

    /// Returns layerwise parameter metadata when that engine is selected.
    pub fn layerwise_metadata(&self) -> Option<&LayerwiseModelMetadata> {
        match &self.execution {
            LlamaExecution::FullyResident(_) => None,
            LlamaExecution::LayerwiseHost(model) => Some(model.metadata()),
        }
    }

    /// Returns rank-local generalized parallel information when applicable.
    pub fn parallel_info(
        &self,
    ) -> Option<&crate::runtime::execution::layerwise::ParallelModelInfo> {
        match &self.execution {
            LlamaExecution::FullyResident(_) => None,
            LlamaExecution::LayerwiseHost(model) => model.parallel_info(),
        }
    }

    pub(crate) fn resident_parameter_bytes(&self) -> Option<Result<u64, &'static str>> {
        match &self.execution {
            LlamaExecution::FullyResident(model) => Some(
                model
                    .parameters()
                    .flatten()
                    .values()
                    .try_fold(0u64, |total, parameter| {
                        let bytes = u64::try_from(parameter.nbytes())
                            .map_err(|_| "Llama parameter byte count does not fit u64")?;
                        total
                            .checked_add(bytes)
                            .ok_or("Llama parameter byte total overflowed u64")
                    }),
            ),
            LlamaExecution::LayerwiseHost(_) => None,
        }
    }

    /// Returns logical residency and transfer telemetry for a layerwise model.
    pub fn residency_report(&self) -> Result<Option<ResidencyReport>, Error> {
        match &self.execution {
            LlamaExecution::FullyResident(_) => Ok(None),
            LlamaExecution::LayerwiseHost(model) => Ok(Some(model.residency_report()?)),
        }
    }

    /// Returns dense-stream observations when that policy is active.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        match &self.execution {
            LlamaExecution::FullyResident(_) => Ok(None),
            LlamaExecution::LayerwiseHost(model) => model.dense_stream_report(),
        }
    }

    /// Returns the persistent checkpoint store used by a layerwise model.
    pub fn checkpoint_store(&self) -> Option<&(dyn WeightStore + Send + Sync)> {
        match &self.execution {
            LlamaExecution::FullyResident(_) => None,
            LlamaExecution::LayerwiseHost(model) => Some(model.checkpoint_store()),
        }
    }

    /// Backward-compatible alias for [`Self::checkpoint_store`].
    pub fn weight_store(&self) -> Option<&(dyn WeightStore + Send + Sync)> {
        self.checkpoint_store()
    }

    /// Returns the number of pinned static leases used by the layerwise engine.
    pub fn layerwise_static_lease_count(&self) -> Option<usize> {
        match &self.execution {
            LlamaExecution::FullyResident(_) => None,
            LlamaExecution::LayerwiseHost(model) => Some(model.static_lease_count()),
        }
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
            CacheResidencyPolicy::Paged(options) => self.new_paged_cache(
                options,
                None,
                match &self.execution {
                    LlamaExecution::FullyResident(_) => None,
                    LlamaExecution::LayerwiseHost(model) => model.prompt_cache_rank_identity(),
                },
            ),
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
        if let LlamaExecution::LayerwiseHost(model) = &self.execution {
            return model.load_prompt_cache(directory, expected, prefix_token_ids, options, stream);
        }
        let args = self.args();
        let layer_count = usize::try_from(args.num_hidden_layers)
            .map_err(|_| Exception::custom("invalid Llama cache layer count"))?;
        let identity = PromptCacheModelIdentity {
            model_family: "llama".into(),
            effective_model_type: args.model_type.clone(),
            architecture_fingerprint:
                crate::architectures::llama::model::prompt_cache_architecture_fingerprint(args),
            layer_count,
            global_layer_start: 0,
            global_layer_end: layer_count,
            sink_tokens: 0,
            topology: Default::default(),
            layer_layout: PromptCacheModelIdentity::key_value_layouts(
                args.attention_schedule.iter().map(|policy| {
                    policy.window().map(|window| {
                        i32::try_from(window.get())
                            .expect("validated Llama attention window fits i32")
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
        let cache = self.new_paged_cache_from_manager(manager, None)?;
        Ok((cache, manifest))
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
        match &self.execution {
            LlamaExecution::LayerwiseHost(model) => model.save_prompt_cache(
                cache,
                destination,
                descriptor,
                prefix_token_ids,
                options,
                stream,
            ),
            LlamaExecution::FullyResident(_) => {
                cache.save_prompt_cache(destination, descriptor, prefix_token_ids, options)
            }
        }
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
        match &mut self.execution {
            LlamaExecution::FullyResident(model) => match cache {
                LlamaCache::Device(caches) => Ok(model.forward(
                    resident::ModelInput {
                        inputs,
                        mask: None,
                        cache: caches,
                    },
                    stream,
                )?),
                LlamaCache::Paged(caches) => Ok(model.forward(
                    resident::ModelInput {
                        inputs,
                        mask: None,
                        cache: caches,
                    },
                    stream,
                )?),
            },
            LlamaExecution::LayerwiseHost(model) => {
                model.forward(LlamaAdapterInput { inputs, mask: None }, cache, stream)
            }
        }
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
        match &mut self.execution {
            LlamaExecution::LayerwiseHost(model) => model.forward_tensor_parallel(
                LlamaAdapterInput { inputs, mask: None },
                cache,
                group,
                stream,
            ),
            LlamaExecution::FullyResident(_) => Err(Error::Parallel(
                "Llama model was not loaded for tensor-parallel execution".into(),
            )),
        }
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
        match &self.execution {
            LlamaExecution::FullyResident(_) => Ok(false),
            LlamaExecution::LayerwiseHost(model) => {
                model.clear_device_group("text_decoder")?;
                Ok(true)
            }
        }
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
    let execution = match options.weight_residency {
        WeightResidency::FullyResident => LlamaExecution::FullyResident(Box::new(
            resident::load_resident_llama_model(model_dir, stream, weights_stream)?,
        )),
        WeightResidency::LayerwiseHost(options) => {
            let args = resident::get_llama_model_args(model_dir)?;
            let adapter = LlamaLayerwiseAdapter::new(args, stream)?;
            LlamaExecution::LayerwiseHost(Box::new(load_safetensors_layerwise_model(
                model_dir,
                adapter,
                options,
                stream,
                weights_stream,
            )?))
        }
        WeightResidency::DenseDiskStream(options) => {
            let args = resident::get_llama_model_args(model_dir)?;
            let adapter = LlamaLayerwiseAdapter::new(args, stream)?;
            LlamaExecution::LayerwiseHost(Box::new(load_safetensors_layerwise_model(
                model_dir,
                adapter,
                options,
                stream,
                weights_stream,
            )?))
        }
        WeightResidency::SparseExpertCache(_)
        | WeightResidency::SparseExpertCacheWithDenseLayers(_) => {
            return Err(Error::UnsupportedArchitecture(
                "sparse expert caching is not supported for Llama checkpoints".into(),
            ));
        }
    };
    Ok(LlamaModel { execution })
}

/// Loads Llama/Mistral through the generalized tensor-parallel execution engine.
pub fn load_llama_tensor_parallel_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerExecutionLoadOptions>,
    build: crate::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LlamaModel, Error> {
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
        execution: LlamaExecution::LayerwiseHost(Box::new(load_tensor_parallel_layerwise_model(
            open_safetensors_weight_store(model_dir, options.max_mapped_shards())?,
            adapter,
            options,
            build,
            stream,
            weights_stream,
        )?)),
    })
}

pub(crate) fn load_llama_gguf_tensor_parallel_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: LayerExecutionLoadOptions,
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
            execution: LlamaExecution::LayerwiseHost(Box::new(execution)),
        },
        prepared.eos_token_ids,
    ))
}

/// Loads a Llama/Mistral GGUF checkpoint using the selected residency policy.
pub(crate) fn load_llama_gguf_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    residency: WeightResidency,
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
    let execution = match residency {
        WeightResidency::LayerwiseHost(options) => LlamaExecution::LayerwiseHost(Box::new(
            load_layerwise_model(store, adapter, options, stream, weights_stream)?,
        )),
        WeightResidency::DenseDiskStream(options) => LlamaExecution::LayerwiseHost(Box::new(
            load_layerwise_model(store, adapter, options, stream, weights_stream)?,
        )),
        WeightResidency::SparseExpertCache(_)
        | WeightResidency::SparseExpertCacheWithDenseLayers(_) => {
            return Err(Error::UnsupportedArchitecture(
                "sparse expert caching is not supported for Llama GGUF checkpoints".into(),
            ));
        }
        WeightResidency::FullyResident => {
            return Err(Error::UnsupportedArchitecture(
                "the bounded GGUF Llama loader does not accept fully resident policy".into(),
            ));
        }
    };
    Ok((LlamaModel { execution }, prepared.eos_token_ids))
}

/// Llama implementation of the generic layerwise model-family contract.
pub struct LlamaLayerwiseAdapter {
    args: ModelArgs,
    embedding: MaybeQuantized<nn::Embedding>,
    norm: nn::RmsNorm,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_embedding: Option<VocabParallelEmbedding>,
    parallel_lm_head: Option<VocabParallelLmHead>,
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
        })
    }

    /// Returns normalized Llama arguments.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
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
    prefix: &str,
) -> Result<(), Error> {
    let attention = &layer.self_attn;
    for (name, projection, sharding) in [
        ("q_proj", &attention.q_proj, ProjectionSharding::Column),
        ("k_proj", &attention.k_proj, ProjectionSharding::Column),
        ("v_proj", &attention.v_proj, ProjectionSharding::Column),
        ("o_proj", &attention.o_proj, ProjectionSharding::Row),
    ] {
        register_projection_module(
            planner,
            projection,
            &format!("{prefix}.self_attn.{name}"),
            sharding,
        )?;
    }
    register_replicated_module(
        planner,
        &attention.rope,
        &format!("{prefix}.self_attn.rope"),
    )?;
    for (name, projection, sharding) in [
        (
            "gate_proj",
            &layer.mlp.gate_proj,
            ProjectionSharding::Column,
        ),
        ("up_proj", &layer.mlp.up_proj, ProjectionSharding::Column),
        ("down_proj", &layer.mlp.down_proj, ProjectionSharding::Row),
    ] {
        register_projection_module(
            planner,
            projection,
            &format!("{prefix}.mlp.{name}"),
            sharding,
        )?;
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
            Some(topology) => exact_parallel_division(
                "Llama prompt-cache KV heads",
                self.args.num_key_value_heads,
                topology.tensor_parallel_size,
            )?,
            None => self.args.num_key_value_heads,
        };
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
            layer_layout: PromptCacheModelIdentity::key_value_layouts(
                self.args.attention_schedule.iter().map(|policy| {
                    policy.window().map(|window| {
                        i32::try_from(window.get())
                            .expect("validated Llama attention window fits i32")
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

    fn execution_group_count(&self) -> usize {
        1
    }

    fn execution_group_id(&self, group: usize) -> Result<String, Error> {
        (group == 0)
            .then(|| "text_decoder".into())
            .ok_or_else(|| Error::UnsupportedArchitecture(format!("Llama has no group {group}")))
    }

    fn layer_count(&self, group: usize) -> Result<usize, Error> {
        self.execution_group_id(group)?;
        usize::try_from(self.args.num_hidden_layers).map_err(|_| {
            LlamaModelError::InvalidLayerCount {
                count: self.args.num_hidden_layers,
            }
            .into()
        })
    }

    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        self.execution_group_id(group)?;
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
            register_llama_layer_parallel_plan(planner, &layer, &format!("model.layers.{index}"))?;
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
        self.execution_group_id(group)?;
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
        args.num_attention_heads = q.local_shape()[0] as i32 / args.head_dim;
        args.num_key_value_heads = k.local_shape()[0] as i32 / args.head_dim;
        args.intermediate_size = gate.local_shape()[0] as i32;
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

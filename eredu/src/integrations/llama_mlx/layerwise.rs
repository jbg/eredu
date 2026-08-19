//! Unified Llama/Mistral loading across weight-residency policies.

use eredu_checkpoint::WeightQuantization;
use eredu_runtime::{CausalModel, RuntimeState, WeightBinding};

use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
    sync::Arc,
};

use eredu_architectures::llama::ModelArgs;
use eredu_nn::{
    EmbeddingOperator, EmbeddingSpec, LinearOperator, LinearSpec, NeuralBackend,
    NormalizationOperator, NormalizationSpec, ParameterSpec,
};
use safemlx::{
    error::Exception,
    ops::indexing::TryIndexOp,
    ops::{GgufCheckpoint, GgufMetadataValue},
    Array, Stream,
};

use crate::core::cache::{
    PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
    PromptCacheTopology,
};

use crate::{
    backend::mlx::error::Error,
    backend::mlx::nn::shared::{MlxBackend, MlxEmbedding, MlxLinear, MlxModule, MlxRmsNorm},
    backend::mlx::nn::{
        parallel::{planned_kv_head_layout, VocabParallelEmbedding, VocabParallelLmHead},
        tensor::{create_attention_mask_from_cache, AttentionMask},
    },
    backend::mlx::runtime::cache::residency::{
        open_prompt_cache, CacheResidencyManager, CacheResidencyPolicy, PagedCacheOptions,
    },
    backend::mlx::runtime::cache::{state::MlxKeyValueState, KeyValueCache},
    backend::mlx::runtime::checkpoint::binding::{
        build_module_binding_plan_with_recipes, populate_module_from_lease,
    },
    backend::mlx::runtime::checkpoint::{
        quantization::should_quantize_on_load, store::open_gguf_checkpoint_source,
    },
    backend::mlx::runtime::distributed::parallel::ParallelPlanBuilder,
    backend::mlx::runtime::execution::layerwise::{
        load_layerwise_model_with_quantization, load_tensor_parallel_layerwise_model,
        open_safetensors_weight_store, ArchitectureAdapter, DenseDiskStreamReport,
        ExecutionResidency, LayerWeightResidency, LayerwiseForwardState, LayerwiseModel,
        LayerwiseModelMetadata, LoadTimeQuantizableAdapter, StaticUnitBindings, WeightResidency,
    },
    backend::mlx::runtime::media::input,
    backend::mlx::runtime::residency::manager::ResidentUnitLease,
    integrations::llama_mlx::model as resident,
};

use eredu_runtime::ResidencyReport;

const EMBEDDING_UNIT: &str = "llama.static.embedding";
const NORM_UNIT: &str = "llama.static.norm";
const HEAD_UNIT: &str = "llama.static.output";

type AttentionInput<'a, C> = eredu_architectures::llama::AttentionInput<'a, Array, C>;
type TransformerBlock = MlxModule<eredu_architectures::llama::TransformerBlock<MlxBackend>>;

fn new_transformer_block(
    args: &ModelArgs,
    layer: usize,
    stream: &Stream,
) -> Result<TransformerBlock, Error> {
    eredu_architectures::llama::TransformerBlock::<MlxBackend>::new(args, layer, stream)
        .map(MlxModule::new)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

fn load_model_args(model_dir: &Path) -> Result<ModelArgs, Error> {
    let file = std::fs::File::open(model_dir.join("config.json"))?;
    eredu_architectures::llama::model_args_from_config_reader(file)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
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
        eredu_architectures::llama::prompt_cache_architecture_fingerprint(self.args())
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
    ) -> Option<&crate::backend::mlx::runtime::execution::layerwise::ParallelModelInfo> {
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
    pub fn checkpoint_store(&self) -> &dyn eredu_checkpoint::store::CheckpointSource {
        self.execution.checkpoint_store()
    }

    /// Returns the number of pinned static leases used by the layerwise engine.
    pub fn static_lease_count(&self) -> usize {
        self.execution.static_lease_count()
    }

    /// Creates the cache representation required by the model configuration.
    pub fn new_cache(&self) -> MlxKeyValueState {
        MlxKeyValueState::device(
            eredu_architectures::llama::state_layout(self.args())
                .expect("validated Llama configuration has a valid state layout"),
        )
        .expect("MLX key/value state supports the validated Llama layout")
    }

    /// Creates a device-resident or explicitly bounded paged model cache.
    pub fn new_cache_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<MlxKeyValueState, Error> {
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
    ) -> Result<(MlxKeyValueState, PromptCacheManifest), Error> {
        self.execution
            .load_prompt_cache(directory, expected, prefix_token_ids, options, stream)
    }

    /// Persists a prefix through the generalized execution contract.
    pub fn save_prompt_cache(
        &self,
        cache: &mut MlxKeyValueState,
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
    ) -> Result<MlxKeyValueState, Error> {
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
    ) -> Result<MlxKeyValueState, Error> {
        MlxKeyValueState::paged(
            eredu_architectures::llama::state_layout(self.args())
                .map_err(|error| Exception::custom(error.to_string()))?,
            manager,
            rank,
        )
        .map_err(Into::into)
    }

    /// Runs embedding, decoder layers, final normalization, and projection.
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut MlxKeyValueState,
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
        cache: &mut MlxKeyValueState,
        stream: &Stream,
        observer: &mut dyn crate::backend::mlx::runtime::execution::inspection::ActivationObserver,
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
    pub(crate) fn forward_tensor_parallel(
        &mut self,
        inputs: &Array,
        cache: &mut MlxKeyValueState,
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
        cache: &mut MlxKeyValueState,
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
        cache: &mut MlxKeyValueState,
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

    fn validate_cache(&self, cache: &MlxKeyValueState) -> Result<(), Error> {
        let expected = eredu_architectures::llama::state_layout(self.args())
            .map_err(|error| Exception::custom(error.to_string()))?;
        if cache.layout() != &expected {
            return Err(Exception::custom(format!(
                "MLX key/value state layout {:?} does not match Llama layout {expected:?}",
                cache.layout()
            ))
            .into());
        }
        Ok(())
    }
}

impl CausalModel<MlxKeyValueState> for LlamaModel {
    type Tensor = Array;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut MlxKeyValueState,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        self.prefill(&tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut MlxKeyValueState,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.decode(input_tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))
    }
}

pub(crate) fn load_llama_safetensors_mlx(
    model_dir: impl AsRef<Path>,
    weight_residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LlamaModel, Error> {
    let model_dir = model_dir.as_ref();
    if weight_residency.expert_cache().is_some() {
        return Err(Error::UnsupportedArchitecture(
            "independent expert caching is not supported for Llama checkpoints".into(),
        ));
    }
    let execution_options = weight_residency.layers();
    let args = load_model_args(model_dir)?;
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load("Llama", args.weight_quantization(), requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let adapter = LlamaLayerwiseAdapter::new(args, stream)?;
    let store = open_safetensors_weight_store(model_dir, execution_options.max_mapped_shards())?;
    Ok(LlamaModel {
        execution: Box::new(load_layerwise_model_with_quantization(
            store,
            adapter,
            execution_options,
            quantize_on_load,
            stream,
            weights_stream,
        )?),
    })
}

/// Loads Llama/Mistral through the generalized tensor-parallel execution engine.
pub(crate) fn load_llama_tensor_parallel_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LlamaModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    if model_dir
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        let checkpoint = GgufCheckpoint::open(model_dir)?;
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
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
    let args = load_model_args(model_dir)?;
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
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(LlamaModel, Vec<u32>), Error> {
    crate::backend::mlx::runtime::execution::layerwise::validate_gguf_layerwise_source(
        checkpoint, metadata, options,
    )?;
    let prepared =
        resident::prepare_llama_gguf_checkpoint(checkpoint, metadata, None, weights_stream)?;
    let gguf_plan = eredu_architectures::llama::gguf_plan(&prepared.args)
        .map_err(Error::UnsupportedArchitecture)?;
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
        Arc::new(open_gguf_checkpoint_source(
            checkpoint.clone(),
            &gguf_plan,
            eredu_architectures::llama::translate_gguf_weight_name,
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
    let gguf_plan = eredu_architectures::llama::gguf_plan(&prepared.args)
        .map_err(Error::UnsupportedArchitecture)?;
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
        Arc::new(open_gguf_checkpoint_source(
            checkpoint.clone(),
            &gguf_plan,
            eredu_architectures::llama::translate_gguf_weight_name,
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
    embedding: MlxEmbedding,
    norm: MlxRmsNorm,
    lm_head: Option<MlxLinear>,
    parallel_embedding: Option<VocabParallelEmbedding>,
    parallel_lm_head: Option<VocabParallelLmHead>,
    parallel_kv_heads: Option<Vec<i32>>,
}

impl LlamaLayerwiseAdapter {
    /// Creates metadata-only static modules for a normalized Llama configuration.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let embedding = <MlxBackend as NeuralBackend>::embedding(
            EmbeddingSpec {
                vocabulary: args.vocab_size,
                dimensions: args.hidden_size,
                weight: ParameterSpec::trainable("model.embed_tokens.weight")
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
                quantization: args.weight_quantization_for("model.embed_tokens.weight"),
            },
            stream,
        )
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let norm = <MlxBackend as NeuralBackend>::rms_norm(
            NormalizationSpec {
                dimensions: args.hidden_size,
                epsilon: args.rms_norm_eps,
                weight: ParameterSpec::trainable("model.norm.weight")
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
            },
            stream,
        )
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(
                <MlxBackend as NeuralBackend>::linear(
                    LinearSpec {
                        input: args.hidden_size,
                        output: args.vocab_size,
                        weight: ParameterSpec::trainable("lm_head.weight")
                            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
                        bias: None,
                        quantization: args.weight_quantization_for("lm_head.weight"),
                    },
                    stream,
                )
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
            )
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

impl ArchitectureAdapter for LlamaLayerwiseAdapter {
    type Input<'a> = LlamaAdapterInput<'a>;
    type Cache = MlxKeyValueState;
    type Layer = TransformerBlock;
    type ForwardContext = LlamaForwardContext;

    fn model_type(&self) -> &str {
        &self.args.model_type
    }

    fn safetensors_checkpoint_plan(
        &self,
    ) -> Result<crate::backend::mlx::runtime::execution::layerwise::ArchitectureCheckpointPlan, Error>
    {
        eredu_architectures::llama::safetensors_plan(&self.args)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
            .map(Into::into)
    }

    fn quantization(&self) -> Option<eredu_checkpoint::WeightQuantization> {
        self.args.weight_quantization()
    }

    fn prompt_cache_model_identity(
        &self,
        topology: Option<crate::backend::mlx::MlxParallelContext>,
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
        let layer_layout = eredu_architectures::llama::cache_layout_with_key_value_heads(
            &self.args,
            local_kv_heads,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(PromptCacheModelIdentity {
            model_family: "llama".into(),
            effective_model_type: self.args.model_type.clone(),
            architecture_fingerprint:
                eredu_architectures::llama::prompt_cache_architecture_fingerprint(&self.args),
            layer_count,
            global_layer_start: 0,
            global_layer_end: layer_count,
            sink_tokens: 0,
            layer_prefix_offsets: vec![0; layer_count],
            topology: topology.map_or_else(
                PromptCacheTopology::default,
                crate::backend::mlx::cache::prompt_cache_topology,
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
        let (manager, manifest) =
            open_prompt_cache(directory, expected, identity, prefix_token_ids, options)
                .map_err(|error| Exception::custom(error.to_string()))?;
        let rank = identity.topology.cache_rank_identity();
        let state = MlxKeyValueState::paged(
            eredu_architectures::llama::state_layout(&self.args)
                .map_err(|error| Exception::custom(error.to_string()))?,
            manager,
            rank,
        )?;
        Ok((state, manifest))
    }

    fn static_units(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        self.selected_static_units(store, &|_| true)
    }

    fn selected_static_units(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        select: &dyn Fn(&str) -> bool,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        let mut units = Vec::new();
        if select(EMBEDDING_UNIT) {
            units.push(StaticUnitBindings::new(
                EMBEDDING_UNIT,
                build_module_binding_plan_with_recipes(
                    &self.embedding,
                    "",
                    store,
                    BTreeMap::new(),
                )?
                .build_bindings(store)?,
            )?);
        }
        if select(NORM_UNIT) {
            units.push(StaticUnitBindings::new(
                NORM_UNIT,
                build_module_binding_plan_with_recipes(&self.norm, "", store, BTreeMap::new())?
                    .build_bindings(store)?,
            )?);
        }
        if select(HEAD_UNIT) {
            if let Some(head) = &self.lm_head {
                units.push(StaticUnitBindings::new(
                    HEAD_UNIT,
                    build_module_binding_plan_with_recipes(head, "", store, BTreeMap::new())?
                        .build_bindings(store)?,
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
        let expected = eredu_architectures::llama::state_layout(&self.args)
            .map_err(|error| Exception::custom(error.to_string()))?;
        if cache.layout() != &expected {
            return Err(Exception::custom(format!(
                "MLX key/value state layout {:?} does not match Llama layout {expected:?}",
                cache.layout()
            ))
            .into());
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
        let mask = match input.mask {
            Some(mask) => Some(mask.clone()),
            None => llama_attention_mask(&hidden, cache.as_ref(), stream)?,
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
        execution: &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<
            '_,
        >,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        let Some(embedding) = &mut self.parallel_embedding else {
            return self.begin_forward(input, cache, execution.stream());
        };
        let hidden = embedding.forward(input.inputs, execution)?;
        let allow_sliding_prefill = input.mask.is_none();
        let mask = match input.mask {
            Some(mask) => Some(mask.clone()),
            None => llama_attention_mask(&hidden, cache.as_ref(), execution.stream())?,
        };
        Ok(LayerwiseForwardState {
            hidden,
            context: LlamaForwardContext {
                mask,
                allow_sliding_prefill,
            },
        })
    }

    fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Error> {
        eredu_runtime::ExecutionGraph::chain(["text_decoder"]).map_err(Into::into)
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
        new_transformer_block(&self.args, index, stream)
    }

    fn register_parallel_parameters(
        &self,
        _context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        planner: &mut crate::backend::mlx::runtime::distributed::parallel::ParallelPlanBuilder,
        stream: &Stream,
    ) -> Result<(), Error> {
        for group in eredu_architectures::llama::static_parallel_parameter_groups::<MlxBackend>(
            &self.embedding,
            &self.norm,
            self.lm_head.as_ref(),
        )? {
            planner.register(group)?;
        }
        for index in 0..self.args.num_hidden_layers as usize {
            let layer = new_transformer_block(&self.args, index, stream)?;
            for group in eredu_architectures::llama::layer_parallel_parameter_groups::<MlxBackend>(
                &layer.inner,
                &self.args,
                index,
            )? {
                planner.register(group)?;
            }
        }
        Ok(())
    }

    fn configure_parallel_static(
        &mut self,
        context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        layout: &eredu_runtime::LocalModelLayout,
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
        layout: &eredu_runtime::LocalModelLayout,
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
        new_transformer_block(&args, index, stream)
    }

    fn layer_checkpoint_prefix(&self, _group: usize, index: usize) -> String {
        format!("model.layers.{index}")
    }

    fn layer_bindings(
        &self,
        _group: usize,
        _index: usize,
        layer: &Self::Layer,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        Ok(
            build_module_binding_plan_with_recipes(layer, "", store, BTreeMap::new())?
                .build_bindings(store)?,
        )
    }

    fn layer_unit_name(&self, _group: usize, index: usize) -> String {
        format!("llama.layer.{index:05}")
    }

    fn parallel_layer_bindings(
        &self,
        group: usize,
        index: usize,
        _layer: &Self::Layer,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        layout: &eredu_runtime::LocalModelLayout,
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
        let layer_state = cache
            .layer(index)
            .map_err(|error| Exception::custom(error.to_string()))?;
        Ok(layer
            .inner
            .forward(
                AttentionInput {
                    hidden,
                    mask: context.mask.as_ref(),
                    cache: Some(layer_state),
                    allow_sliding_prefill: context.allow_sliding_prefill,
                },
                stream,
            )
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?)
    }

    fn forward_layer_with_execution(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Self::Layer,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
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
        let layer_state = cache
            .layer(index)
            .map_err(|error| Exception::custom(error.to_string()))?;
        Ok(layer
            .inner
            .forward_tensor_parallel(
                AttentionInput {
                    hidden,
                    mask: context.mask.as_ref(),
                    cache: Some(layer_state),
                    allow_sliding_prefill: context.allow_sliding_prefill,
                },
                tp_group,
                execution.stream(),
            )
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?)
    }

    fn retained_arrays<'a>(
        &self,
        cache: &'a Self::Cache,
        _group: usize,
        index: usize,
    ) -> Vec<&'a Array> {
        cache
            .as_ref()
            .get(index)
            .map(KeyValueCache::retained_arrays)
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
        match &mut self.lm_head {
            Some(head) => head
                .forward(&hidden, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string())),
            None => self
                .embedding
                .as_linear(&hidden, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string())),
        }
    }

    fn finish_with_execution(
        &mut self,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &Self::ForwardContext,
        execution: &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<
            '_,
        >,
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
    cache: &[C],
    stream: &Stream,
) -> Result<Option<Array>, Error> {
    match create_attention_mask_from_cache(hidden, cache.first(), Some(true), stream)? {
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
}

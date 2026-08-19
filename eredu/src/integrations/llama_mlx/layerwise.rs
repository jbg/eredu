//! Unified Llama/Mistral loading across weight-residency policies.

use eredu_checkpoint::WeightQuantization;
use eredu_runtime::{CausalModel, LayerwiseRuntime, RuntimeState, WeightBinding};

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
        parallel::{VocabParallelEmbedding, VocabParallelLmHead},
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
    backend::mlx::runtime::execution::generic::{
        prepare_layerwise_policy, MlxLayerwisePolicy, MlxResidentPolicy, MlxUnitFactory,
    },
    backend::mlx::runtime::execution::layerwise::{
        load_tensor_parallel_layerwise_model, open_safetensors_weight_store,
        quantize_parameterized_store, ArchitectureAdapter, DenseDiskStreamReport,
        ExecutionResidency, LayerWeightResidency, LayerwiseForwardState, LayerwiseModel,
        LayerwiseModelMetadata, LoadTimeQuantizableAdapter, StaticUnitBindings, WeightResidency,
    },
    backend::mlx::runtime::media::input,
    backend::mlx::runtime::residency::manager::ResidentUnitLease,
    composition::llama_mlx as resident,
};

use eredu_runtime::ResidencyReport;

const EMBEDDING_UNIT: &str = "llama.static.embedding";
const NORM_UNIT: &str = "llama.static.norm";
const HEAD_UNIT: &str = "llama.static.output";

type AttentionInput<'a, C> = eredu_architectures::llama::AttentionInput<'a, Array, C>;
type TransformerBlock = MlxModule<eredu_architectures::llama::TransformerBlock<MlxBackend>>;
type NeutralBlock = eredu_architectures::llama::TransformerBlock<MlxBackend>;

#[derive(Clone)]
struct LlamaUnitFactory {
    args: ModelArgs,
}

impl MlxUnitFactory<NeutralBlock> for LlamaUnitFactory {
    fn build(&mut self, index: usize, stream: &Stream) -> Result<NeutralBlock, Error> {
        NeutralBlock::new(&self.args, index, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }
}

type NeutralArchitecture = eredu_architectures::llama::LayeredModel<MlxBackend>;
type NeutralResidentRuntime = LayerwiseRuntime<
    NeutralArchitecture,
    MlxBackend,
    MlxKeyValueState,
    MlxResidentPolicy<NeutralBlock>,
>;
type NeutralLayerwiseRuntime = LayerwiseRuntime<
    NeutralArchitecture,
    MlxBackend,
    MlxKeyValueState,
    MlxLayerwisePolicy<NeutralBlock, LlamaUnitFactory>,
>;

enum LlamaExecution {
    Resident(NeutralResidentRuntime),
    Layerwise(NeutralLayerwiseRuntime),
    TensorParallel(Box<LayerwiseModel<LlamaLayerwiseAdapter>>),
}

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

fn resolve_llama_safetensors_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: &ModelArgs,
) -> Result<Arc<dyn eredu_checkpoint::store::CheckpointSource>, Error> {
    if store.is_checkpoint_contract_resolved()
        || store.source_diagnostics()?.backend
            != eredu_checkpoint::store::WeightStoreBackend::Safetensors
    {
        return Ok(store);
    }
    let plan = eredu_architectures::llama::safetensors_plan(args)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let resolved = eredu_checkpoint::validation::resolve_safetensors_plan(store.as_ref(), &plan)
        .map_err(|validation| {
            Error::UnsupportedArchitecture(format!(
                "{} checkpoint contract did not resolve: {validation:?}",
                args.model_type
            ))
        })?;
    Ok(Arc::new(
        eredu_checkpoint::store::ResolvedCheckpointSource::new(store, resolved),
    ))
}

fn load_neutral_llama(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
    materialization: Option<eredu_runtime::WeightMaterializationReport>,
) -> Result<LlamaModel, Error> {
    let layer_count = usize::try_from(args.num_hidden_layers)
        .map_err(|_| Error::UnsupportedArchitecture("invalid Llama layer count".into()))?;
    let mut architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let factory = LlamaUnitFactory { args: args.clone() };
    let (policy, mut metadata) = prepare_layerwise_policy(
        store,
        architecture.static_modules_mut(),
        factory,
        layer_count,
        options,
        stream,
        weights_stream,
        |key| key.starts_with("rope_freqs.") || key.ends_with(".rotary_emb.inv_freq"),
    )?;
    metadata.model_type = args.model_type.clone();
    metadata.quantization = args.weight_quantization();
    metadata.materialization = materialization;
    let execution = if options.is_fully_resident() {
        LlamaExecution::Resident(LayerwiseRuntime::new(
            architecture,
            policy.into_resident(stream)?,
        ))
    } else {
        LlamaExecution::Layerwise(LayerwiseRuntime::new(architecture, policy))
    };
    Ok(LlamaModel {
        args,
        metadata,
        execution,
    })
}

fn quantize_neutral_llama_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    source_args: &ModelArgs,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<
    (
        Arc<dyn eredu_checkpoint::store::CheckpointSource>,
        ModelArgs,
        eredu_runtime::WeightMaterializationReport,
    ),
    Error,
> {
    let mut target_args = source_args.clone();
    target_args.quantization = Some(quantization);
    target_args.quantization_config = None;
    target_args.quantized_weights = None;
    target_args.quantized_weight_configs = None;
    let source = NeutralArchitecture::new(source_args.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let target = NeutralArchitecture::new(target_args.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let count = usize::try_from(source_args.num_hidden_layers)
        .map_err(|_| Error::UnsupportedArchitecture("invalid Llama layer count".into()))?;
    let source_unit_args = source_args.clone();
    let target_unit_args = target_args.clone();
    let (store, report) = quantize_parameterized_store(
        store,
        source.static_modules(),
        target.static_modules(),
        move |index, stream| {
            NeutralBlock::new(&source_unit_args, index, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        },
        move |index, stream| {
            NeutralBlock::new(&target_unit_args, index, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        },
        count,
        quantization,
        stream,
    )?;
    Ok((store, target_args, report))
}

/// Llama/Mistral causal LM whose execution engine follows its residency policy.
pub struct LlamaModel {
    args: ModelArgs,
    metadata: LayerwiseModelMetadata,
    execution: LlamaExecution,
}

impl LlamaModel {
    /// Returns normalized model arguments regardless of execution engine.
    pub fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Returns the canonical cache-relevant architecture identity.
    pub fn prompt_cache_architecture_fingerprint(&self) -> String {
        eredu_architectures::llama::prompt_cache_architecture_fingerprint(self.args())
    }

    /// Returns this rank's exact prompt-cache state layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        Ok(self.prompt_cache_model_identity()?.layer_layout)
    }

    /// Returns whether all parameters use the eager execution-device engine.
    pub fn is_fully_resident(&self) -> bool {
        self.metadata.residency() == ExecutionResidency::FullyResident
    }

    /// Returns canonical parameter and residency metadata.
    pub fn metadata(&self) -> &LayerwiseModelMetadata {
        &self.metadata
    }

    /// Returns rank-local generalized parallel information when applicable.
    pub fn parallel_info(
        &self,
    ) -> Option<&crate::backend::mlx::runtime::execution::layerwise::ParallelModelInfo> {
        match &self.execution {
            LlamaExecution::TensorParallel(execution) => execution.parallel_info(),
            LlamaExecution::Resident(_) | LlamaExecution::Layerwise(_) => None,
        }
    }

    /// Returns logical residency and transfer telemetry for a layerwise model.
    pub fn residency_report(&self) -> Result<Option<ResidencyReport>, Error> {
        let report = match &self.execution {
            LlamaExecution::Resident(execution) => execution.policy().residency_report()?,
            LlamaExecution::Layerwise(execution) => execution.policy().residency_report()?,
            LlamaExecution::TensorParallel(execution) => execution.residency_report()?,
        };
        Ok(Some(report))
    }

    /// Returns dense-stream observations when that policy is active.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        match &self.execution {
            LlamaExecution::TensorParallel(execution) => execution.dense_stream_report(),
            LlamaExecution::Layerwise(execution) => execution.policy().dense_stream_report(),
            LlamaExecution::Resident(_) => Ok(None),
        }
    }

    /// Returns the persistent checkpoint store used by a layerwise model.
    pub fn checkpoint_store(&self) -> &dyn eredu_checkpoint::store::CheckpointSource {
        match &self.execution {
            LlamaExecution::Resident(execution) => execution.policy().checkpoint_store(),
            LlamaExecution::Layerwise(execution) => execution.policy().checkpoint_store(),
            LlamaExecution::TensorParallel(execution) => execution.checkpoint_store(),
        }
    }

    /// Returns the number of pinned static leases used by the layerwise engine.
    pub fn static_lease_count(&self) -> usize {
        match &self.execution {
            LlamaExecution::Resident(execution) => execution.policy().static_lease_count(),
            LlamaExecution::Layerwise(execution) => execution.policy().static_lease_count(),
            LlamaExecution::TensorParallel(execution) => execution.static_lease_count(),
        }
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
                let rank = match &self.execution {
                    LlamaExecution::TensorParallel(execution) => {
                        execution.prompt_cache_rank_identity()
                    }
                    LlamaExecution::Resident(_) | LlamaExecution::Layerwise(_) => None,
                };
                self.new_paged_cache(options, None, rank)
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
        let identity = self.prompt_cache_model_identity()?;
        eredu_core::cache::validate_prompt_cache_model_identity(expected, &identity)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let (manager, manifest) = open_prompt_cache(
            directory.as_ref(),
            expected,
            &identity,
            prefix_token_ids,
            options,
        )
        .map_err(|error| Exception::custom(error.to_string()))?;
        let state =
            self.new_paged_cache_from_manager(manager, identity.topology.cache_rank_identity())?;
        let _ = stream;
        Ok((state, manifest))
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
        let identity = self.prompt_cache_model_identity()?;
        eredu_core::cache::validate_prompt_cache_model_identity(&descriptor, &identity)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let _ = stream;
        cache
            .save_prompt_cache(destination, descriptor, prefix_token_ids, options)
            .map_err(Into::into)
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
        let input = eredu_architectures::llama::LayeredInput {
            tokens: inputs,
            mask: None,
        };
        match &mut self.execution {
            LlamaExecution::Resident(execution) => execution
                .forward(input, cache, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string())),
            LlamaExecution::Layerwise(execution) => execution
                .forward(input, cache, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string())),
            LlamaExecution::TensorParallel(execution) => {
                execution.forward(LlamaAdapterInput { inputs, mask: None }, cache, stream)
            }
        }
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
        match &mut self.execution {
            LlamaExecution::TensorParallel(execution) => execution.forward_with_observer(
                LlamaAdapterInput { inputs, mask },
                cache,
                stream,
                observer,
            ),
            LlamaExecution::Resident(execution) => {
                let output = execution
                    .forward_with_unit_hook(
                        eredu_architectures::llama::LayeredInput {
                            tokens: inputs,
                            mask,
                        },
                        cache,
                        stream,
                        |path, input, output| {
                            observer
                                .observe(&format!("{path}.input"), input)
                                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                            observer
                                .observe(&format!("{path}.output"), output)
                                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                            observer
                                .intervene(&format!("{path}.output"), output)
                                .map_err(|error| eredu_nn::Error::backend(error.to_string()))
                        },
                    )
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
                observer.observe("model.logits", &output)?;
                Ok(output)
            }
            LlamaExecution::Layerwise(execution) => {
                let output = execution
                    .forward_with_unit_hook(
                        eredu_architectures::llama::LayeredInput {
                            tokens: inputs,
                            mask,
                        },
                        cache,
                        stream,
                        |path, input, output| {
                            observer
                                .observe(&format!("{path}.input"), input)
                                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                            observer
                                .observe(&format!("{path}.output"), output)
                                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                            observer
                                .intervene(&format!("{path}.output"), output)
                                .map_err(|error| eredu_nn::Error::backend(error.to_string()))
                        },
                    )
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
                observer.observe("model.logits", &output)?;
                Ok(output)
            }
        }
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
        match &mut self.execution {
            LlamaExecution::TensorParallel(execution) => execution.forward_tensor_parallel(
                LlamaAdapterInput { inputs, mask: None },
                cache,
                group,
                stream,
            ),
            LlamaExecution::Resident(_) | LlamaExecution::Layerwise(_) => Err(Error::Parallel(
                "model was not loaded for tensor-parallel execution".into(),
            )),
        }
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
        match &self.execution {
            LlamaExecution::Layerwise(_) => {}
            LlamaExecution::TensorParallel(execution) => {
                execution.clear_device_group("text_decoder")?
            }
            LlamaExecution::Resident(_) => return Ok(false),
        }
        Ok(true)
    }

    fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        let layout = eredu_architectures::llama::state_layout(self.args())
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let identity = eredu_architectures::llama::state_identity(
            self.args(),
            &layout,
            0,
            PromptCacheTopology::default(),
        )
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        identity
            .prompt_cache_identity(&layout)
            .map_err(|error| Error::Parallel(error.to_string()))
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
    let store = open_safetensors_weight_store(model_dir, execution_options.max_mapped_shards())?;
    let store = resolve_llama_safetensors_store(store, &args)?;
    if let Some(quantization) = quantize_on_load {
        let (store, args, report) =
            quantize_neutral_llama_store(store, &args, quantization, stream)?;
        return load_neutral_llama(
            store,
            args,
            execution_options,
            stream,
            weights_stream,
            Some(report),
        );
    }
    load_neutral_llama(store, args, execution_options, stream, weights_stream, None)
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
    let adapter = LlamaLayerwiseAdapter::new(args.clone(), stream)?;
    let execution = load_tensor_parallel_layerwise_model(
        open_safetensors_weight_store(model_dir, options.max_mapped_shards())?,
        adapter,
        options,
        build,
        stream,
        weights_stream,
    )?;
    let metadata = execution.metadata().clone();
    Ok(LlamaModel {
        args,
        metadata,
        execution: LlamaExecution::TensorParallel(Box::new(execution)),
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
    let args = prepared.args;
    let adapter = LlamaLayerwiseAdapter::new(args.clone(), stream)?;
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
            args,
            metadata: execution.metadata().clone(),
            execution: LlamaExecution::TensorParallel(Box::new(execution)),
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
    let args = prepared.args;
    if residency.expert_cache().is_some() {
        return Err(Error::UnsupportedArchitecture(
            "independent expert caching is not supported for Llama GGUF checkpoints".into(),
        ));
    }
    let execution_options = residency.layers();
    let model = if let Some(quantization) = quantization {
        let (store, args, report) =
            quantize_neutral_llama_store(store, &args, quantization, stream)?;
        load_neutral_llama(
            store,
            args,
            execution_options,
            stream,
            weights_stream,
            Some(report),
        )?
    } else {
        load_neutral_llama(store, args, execution_options, stream, weights_stream, None)?
    };
    Ok((model, prepared.eos_token_ids))
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
        self.parallel_kv_heads = Some(
            eredu_architectures::llama::local_key_value_heads(&self.args, layout)
                .map_err(|error| Error::Parallel(error.to_string()))?,
        );
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
        let args = eredu_architectures::llama::local_block_args(&self.args, index, layout)
            .map_err(|error| Error::Parallel(error.to_string()))?;
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

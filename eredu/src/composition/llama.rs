//! Unified Llama/Mistral loading across weight-residency policies.

use eredu_checkpoint::WeightQuantization;
use eredu_runtime::{
    CausalModel, LayeredArchitecture, LayeredForwardState, LayerwiseRuntime,
    ParallelLayeredArchitecture, RuntimeState,
};

use std::{collections::HashMap, path::Path, sync::Arc};

use eredu_architectures::llama::ModelArgs;
use eredu_nn::{
    NormalizationOperator, ParameterSpec, ParameterVisitor, ParameterVisitorMut, Parameterized,
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
    backend::mlx::nn::parallel::{VocabParallelEmbedding, VocabParallelLmHead},
    backend::mlx::nn::shared::{MlxBackend, MlxModule, MlxNamedModule},
    backend::mlx::runtime::cache::residency::{open_prompt_cache, CacheResidencyManager},
    backend::mlx::runtime::cache::state::MlxKeyValueState,
    backend::mlx::runtime::checkpoint::binding::{
        binding_bytes, build_module_binding_plan_with_recipes, build_module_bindings,
    },
    backend::mlx::runtime::checkpoint::{
        quantization::should_quantize_on_load, store::open_gguf_checkpoint_source,
    },
    backend::mlx::runtime::execution::generic::{
        prepare_layerwise_policy, prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy,
        MlxResidentPolicy, MlxUnitFactory,
    },
    backend::mlx::runtime::execution::layerwise::{
        open_safetensors_weight_store, quantize_parameterized_store, shard_layer_bindings,
        DenseDiskStreamReport, ExecutionResidency, LayerWeightResidency, LayerwiseModelMetadata,
        ParallelModelInfo, StaticUnitBindings, WeightResidency,
    },
    backend::mlx::runtime::media::input,
    composition::llama_mlx as resident,
};
use eredu_runtime::{CacheResidencyPolicy, PagedCacheOptions};

use eredu_runtime::{ResidencyReport, WeightBinding};

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

#[derive(Clone)]
struct LlamaParallelUnitFactory {
    local_args: Arc<Vec<ModelArgs>>,
}

impl MlxUnitFactory<NeutralBlock> for LlamaParallelUnitFactory {
    fn build(&mut self, index: usize, stream: &Stream) -> Result<NeutralBlock, Error> {
        let args = self.local_args.get(index).ok_or_else(|| {
            Error::Parallel(format!(
                "parallel Llama unit {index} is outside {} local layouts",
                self.local_args.len()
            ))
        })?;
        NeutralBlock::new(args, index, stream)
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

type NeutralParallelResidentRuntime = LayerwiseRuntime<
    LlamaParallelComposition,
    MlxBackend,
    MlxKeyValueState,
    MlxResidentPolicy<NeutralBlock>,
>;
type NeutralParallelLayerwiseRuntime = LayerwiseRuntime<
    LlamaParallelComposition,
    MlxBackend,
    MlxKeyValueState,
    MlxLayerwisePolicy<NeutralBlock, LlamaParallelUnitFactory>,
>;

enum LlamaExecution {
    Resident(NeutralResidentRuntime),
    Layerwise(NeutralLayerwiseRuntime),
    TensorParallelResident(Box<NeutralParallelResidentRuntime>),
    TensorParallelLayerwise(Box<NeutralParallelLayerwiseRuntime>),
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
        state_layout: eredu_architectures::llama::state_layout(&args)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
        args,
        metadata,
        parallel_info: None,
        parallel_rank: None,
        execution,
    })
}

pub(crate) fn quantize_neutral_llama_store(
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
    state_layout: eredu_runtime::StateLayout,
    metadata: LayerwiseModelMetadata,
    parallel_info: Option<ParallelModelInfo>,
    parallel_rank: Option<crate::CacheRankIdentity>,
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
        self.parallel_info.as_ref()
    }

    /// Returns logical residency and transfer telemetry for a layerwise model.
    pub fn residency_report(&self) -> Result<Option<ResidencyReport>, Error> {
        let report = match &self.execution {
            LlamaExecution::Resident(execution) => execution.policy().residency_report()?,
            LlamaExecution::Layerwise(execution) => execution.policy().residency_report()?,
            LlamaExecution::TensorParallelResident(execution) => {
                execution.policy().residency_report()?
            }
            LlamaExecution::TensorParallelLayerwise(execution) => {
                execution.policy().residency_report()?
            }
        };
        Ok(Some(report))
    }

    /// Returns dense-stream observations when that policy is active.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        match &self.execution {
            LlamaExecution::TensorParallelLayerwise(execution) => {
                execution.policy().dense_stream_report()
            }
            LlamaExecution::TensorParallelResident(_) => Ok(None),
            LlamaExecution::Layerwise(execution) => execution.policy().dense_stream_report(),
            LlamaExecution::Resident(_) => Ok(None),
        }
    }

    /// Returns the persistent checkpoint store used by a layerwise model.
    pub fn checkpoint_store(&self) -> &dyn eredu_checkpoint::store::CheckpointSource {
        match &self.execution {
            LlamaExecution::Resident(execution) => execution.policy().checkpoint_store(),
            LlamaExecution::Layerwise(execution) => execution.policy().checkpoint_store(),
            LlamaExecution::TensorParallelResident(execution) => {
                execution.policy().checkpoint_store()
            }
            LlamaExecution::TensorParallelLayerwise(execution) => {
                execution.policy().checkpoint_store()
            }
        }
    }

    /// Returns the number of pinned static leases used by the layerwise engine.
    pub fn static_lease_count(&self) -> usize {
        match &self.execution {
            LlamaExecution::Resident(execution) => execution.policy().static_lease_count(),
            LlamaExecution::Layerwise(execution) => execution.policy().static_lease_count(),
            LlamaExecution::TensorParallelResident(execution) => {
                execution.policy().static_lease_count()
            }
            LlamaExecution::TensorParallelLayerwise(execution) => {
                execution.policy().static_lease_count()
            }
        }
    }

    /// Creates the cache representation required by the model configuration.
    pub fn new_cache(&self) -> MlxKeyValueState {
        MlxKeyValueState::device(self.state_layout.clone())
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
                self.new_paged_cache(options, None, self.parallel_rank)
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
        MlxKeyValueState::paged(self.state_layout.clone(), manager, rank).map_err(Into::into)
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
            LlamaExecution::TensorParallelResident(_)
            | LlamaExecution::TensorParallelLayerwise(_) => Err(Error::Parallel(
                "tensor-parallel Llama requires its collective execution context".into(),
            )),
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
            LlamaExecution::TensorParallelResident(_)
            | LlamaExecution::TensorParallelLayerwise(_) => Err(Error::Parallel(
                "tensor-parallel observation requires its collective execution context".into(),
            )),
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
            LlamaExecution::TensorParallelResident(execution) => execution
                .forward_parallel(
                    eredu_architectures::llama::LayeredInput {
                        tokens: inputs,
                        mask: None,
                    },
                    cache,
                    group,
                    stream,
                )
                .map_err(|error| Error::Parallel(error.to_string())),
            LlamaExecution::TensorParallelLayerwise(execution) => execution
                .forward_parallel(
                    eredu_architectures::llama::LayeredInput {
                        tokens: inputs,
                        mask: None,
                    },
                    cache,
                    group,
                    stream,
                )
                .map_err(|error| Error::Parallel(error.to_string())),
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
            LlamaExecution::TensorParallelLayerwise(execution) => {
                execution.policy().clear_device_window()?
            }
            LlamaExecution::Resident(_) | LlamaExecution::TensorParallelResident(_) => {
                return Ok(false)
            }
        }
        Ok(true)
    }

    fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        let layout = self.state_layout.clone();
        let topology = self
            .parallel_info
            .as_ref()
            .map_or_else(PromptCacheTopology::default, |info| {
                crate::backend::mlx::cache::prompt_cache_topology(info.topology())
            });
        let identity =
            eredu_architectures::llama::state_identity(self.args(), &layout, 0, topology)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        identity
            .prompt_cache_identity(&layout)
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn validate_cache(&self, cache: &MlxKeyValueState) -> Result<(), Error> {
        let expected = &self.state_layout;
        if cache.layout() != expected {
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

fn load_neutral_llama_parallel(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LlamaModel, Error> {
    let layer_count = usize::try_from(args.num_hidden_layers)
        .map_err(|_| Error::UnsupportedArchitecture("invalid Llama layer count".into()))?;
    let mut composition = LlamaParallelComposition::new(args.clone(), stream)?;
    let mut planner = build.planner();
    let static_modules = composition.architecture.static_modules();
    for group in eredu_architectures::llama::static_parallel_parameter_groups::<MlxBackend>(
        &static_modules.embeddings,
        &static_modules.norm,
        static_modules.lm_head.as_ref(),
    )? {
        planner.register(group)?;
    }
    for index in 0..layer_count {
        let unit = NeutralBlock::new(&args, index, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        for group in eredu_architectures::llama::layer_parallel_parameter_groups::<MlxBackend>(
            &unit, &args, index,
        )? {
            planner.register(group)?;
        }
    }
    let (_, layout) = planner.finish()?;
    if layout.is_empty() {
        return Err(Error::Parallel(
            "Llama declared no tensor-parallel parameters".into(),
        ));
    }
    composition.configure_parallel(build, &layout, stream)?;
    let state_layout = composition.local_state_layout()?;
    let factory = composition.unit_factory()?;

    let global_static_bindings = build_module_bindings(
        &MlxModule::new(composition.architecture.static_modules().clone()),
        "",
        store.as_ref(),
    )?;
    let mut global_parameter_bytes = binding_bytes(&global_static_bindings)?;
    for index in 0..layer_count {
        let unit = NeutralBlock::new(&args, index, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let bytes = binding_bytes(&build_module_bindings(
            &MlxModule::new(unit),
            "",
            store.as_ref(),
        )?)?;
        global_parameter_bytes = global_parameter_bytes
            .checked_add(bytes)
            .ok_or_else(|| Error::Parallel("global Llama parameter bytes overflowed".into()))?;
    }

    let binding_args = args.clone();
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        Arc::clone(&store),
        &mut composition,
        factory,
        layer_count,
        options,
        stream,
        weights_stream,
        |key| key.starts_with("rope_freqs.") || key.ends_with(".rotary_emb.inv_freq"),
        |modules, store| {
            let global = MlxModule::new(modules.architecture.static_modules().clone());
            let bindings = build_module_bindings(&global, "", store)?;
            shard_layer_bindings(bindings, "", store, &layout)
        },
        |index, _local, store, stream| {
            let global = NeutralBlock::new(&binding_args, index, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            let bindings = build_module_bindings(&MlxModule::new(global), "", store)?;
            shard_layer_bindings(bindings, &format!("model.layers.{index}"), store, &layout)
        },
    )?;
    metadata.model_type = args.model_type.clone();
    metadata.quantization = args.weight_quantization();
    let local_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.layer_parameter_bytes())
        .ok_or_else(|| Error::Parallel("local Llama parameter bytes overflowed".into()))?;
    let maximum_device_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.maximum_device_layer_bytes())
        .ok_or_else(|| Error::Parallel("device Llama parameter bytes overflowed".into()))?;
    let parallel_info = ParallelModelInfo::new(
        build.topology(),
        args.model_type.clone(),
        layout
            .tensors()
            .map(|(target, _)| target.to_owned())
            .collect(),
        local_parameter_bytes,
        global_parameter_bytes,
        if options.is_fully_resident() {
            local_parameter_bytes
        } else {
            metadata.static_device_bytes()
        },
        maximum_device_parameter_bytes,
    );
    let parallel_rank =
        crate::backend::mlx::cache::prompt_cache_topology(build.topology()).cache_rank_identity();
    let execution = if options.is_fully_resident() {
        LlamaExecution::TensorParallelResident(Box::new(LayerwiseRuntime::new(
            composition,
            policy.into_resident(stream)?,
        )))
    } else {
        LlamaExecution::TensorParallelLayerwise(Box::new(LayerwiseRuntime::new(
            composition,
            policy,
        )))
    };
    Ok(LlamaModel {
        args,
        state_layout,
        metadata,
        parallel_info: Some(parallel_info),
        parallel_rank,
        execution,
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
    let store = open_safetensors_weight_store(model_dir, options.max_mapped_shards())?;
    let store = resolve_llama_safetensors_store(store, &args)?;
    load_neutral_llama_parallel(store, args, options, build, stream, weights_stream)
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
    let model =
        load_neutral_llama_parallel(store, prepared.args, options, build, stream, weights_stream)?;
    Ok((model, prepared.eos_token_ids))
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
pub struct LlamaParallelComposition {
    architecture: NeutralArchitecture,
    parallel_embedding: Option<MlxNamedModule<VocabParallelEmbedding>>,
    parallel_lm_head: Option<MlxNamedModule<VocabParallelLmHead>>,
    parallel_kv_heads: Option<Vec<i32>>,
    local_args: Option<Arc<Vec<ModelArgs>>>,
    topology: Option<crate::backend::mlx::MlxParallelContext>,
}

impl LlamaParallelComposition {
    /// Creates metadata-only static modules for a normalized Llama configuration.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let architecture = NeutralArchitecture::new(args, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        Ok(Self {
            architecture,
            parallel_embedding: None,
            parallel_lm_head: None,
            parallel_kv_heads: None,
            local_args: None,
            topology: None,
        })
    }

    /// Returns normalized Llama arguments.
    pub const fn args(&self) -> &ModelArgs {
        self.architecture.args()
    }

    pub(crate) fn register_parallel_parameters(
        &self,
        planner: &mut crate::backend::mlx::runtime::distributed::parallel::ParallelPlanBuilder,
        stream: &Stream,
    ) -> Result<(), Error> {
        let static_modules = self.architecture.static_modules();
        for group in eredu_architectures::llama::static_parallel_parameter_groups::<MlxBackend>(
            &static_modules.embeddings,
            &static_modules.norm,
            static_modules.lm_head.as_ref(),
        )? {
            planner.register(group)?;
        }
        for index in 0..self.args().num_hidden_layers as usize {
            let unit = NeutralBlock::new(self.args(), index, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            for group in eredu_architectures::llama::layer_parallel_parameter_groups::<MlxBackend>(
                &unit,
                self.args(),
                index,
            )? {
                planner.register(group)?;
            }
        }
        Ok(())
    }

    pub(crate) fn new_cartesian_layer(
        &self,
        index: usize,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        stream: &Stream,
    ) -> Result<MlxModule<NeutralBlock>, Error> {
        let local;
        let args = match layout {
            Some(layout) => {
                local = eredu_architectures::llama::local_block_args(self.args(), index, layout)
                    .map_err(|error| Error::Parallel(error.to_string()))?;
                &local
            }
            None => self.args(),
        };
        NeutralBlock::new(args, index, stream)
            .map(MlxModule::new)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    pub(crate) fn cartesian_layer_bindings(
        &self,
        index: usize,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        let global = self.new_cartesian_layer(index, None, stream)?;
        let bindings = build_module_bindings(&global, "", store)?;
        match layout {
            Some(layout) => {
                shard_layer_bindings(bindings, &format!("model.layers.{index}"), store, layout)
            }
            None => Ok(bindings),
        }
    }

    pub(crate) fn selected_static_units(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        roles: &[&str],
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        let selected = |role: &str| roles.contains(&role);
        let static_modules = self.architecture.static_modules();
        let mut units = Vec::new();
        if selected("embedding") {
            units.push(StaticUnitBindings::new(
                "llama.static.embedding",
                build_module_binding_plan_with_recipes(
                    &static_modules.embeddings,
                    "",
                    store,
                    Default::default(),
                )?
                .build_bindings(store)?,
            )?);
        }
        if selected("norm") {
            units.push(StaticUnitBindings::new(
                "llama.static.norm",
                build_module_binding_plan_with_recipes(
                    &static_modules.norm,
                    "",
                    store,
                    Default::default(),
                )?
                .build_bindings(store)?,
            )?);
        }
        if selected("output") {
            if let Some(head) = &static_modules.lm_head {
                units.push(StaticUnitBindings::new(
                    "llama.static.output",
                    build_module_binding_plan_with_recipes(head, "", store, Default::default())?
                        .build_bindings(store)?,
                )?);
            }
        }
        Ok(units)
    }
}

impl LlamaParallelComposition {
    fn configure_parallel(
        &mut self,
        context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.parallel_kv_heads = Some(
            eredu_architectures::llama::local_key_value_heads(self.args(), layout)
                .map_err(|error| Error::Parallel(error.to_string()))?,
        );
        self.parallel_embedding = Some(MlxNamedModule::new(
            VocabParallelEmbedding::unloaded(
                self.args().vocab_size as usize,
                self.args().hidden_size,
                self.args()
                    .weight_quantization_for("model.embed_tokens.weight"),
                context,
                stream,
            )?,
            ParameterSpec::trainable("model.embed_tokens.weight")
                .map_err(|error| Error::Parallel(error.to_string()))?,
            None,
        )?);
        if self.architecture.static_modules().lm_head.is_some() {
            self.parallel_lm_head = Some(MlxNamedModule::new(
                VocabParallelLmHead::unloaded(
                    self.args().hidden_size,
                    self.args().vocab_size as usize,
                    self.args().weight_quantization_for("lm_head.weight"),
                    context,
                    stream,
                )?,
                ParameterSpec::trainable("lm_head.weight")
                    .map_err(|error| Error::Parallel(error.to_string()))?,
                None,
            )?);
        }
        self.local_args = Some(Arc::new(
            (0..self.args().num_hidden_layers as usize)
                .map(|index| {
                    eredu_architectures::llama::local_block_args(self.args(), index, layout)
                        .map_err(|error| Error::Parallel(error.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?,
        ));
        self.topology = Some(context.topology());
        Ok(())
    }

    fn unit_factory(&self) -> Result<LlamaParallelUnitFactory, Error> {
        Ok(LlamaParallelUnitFactory {
            local_args: Arc::clone(self.local_args.as_ref().ok_or_else(|| {
                Error::Parallel("parallel Llama unit layout is not configured".into())
            })?),
        })
    }

    fn local_state_layout(&self) -> Result<eredu_runtime::StateLayout, Error> {
        let layout = eredu_architectures::llama::cache_layout_with_key_value_heads(
            self.args(),
            self.parallel_kv_heads.clone().ok_or_else(|| {
                Error::Parallel("parallel Llama cache layout is not configured".into())
            })?,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        eredu_runtime::StateLayout::new(layout).map_err(|error| Error::Parallel(error.to_string()))
    }

    fn execution_context<'a>(
        &self,
        group: &'a safemlx::distributed::Group,
        stream: &'a Stream,
    ) -> Result<
        crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'a>,
        Error,
    > {
        let topology = self
            .topology
            .ok_or_else(|| Error::Parallel("parallel Llama topology is not configured".into()))?;
        crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(
            topology, group, stream,
        )
    }
}

impl Parameterized<Array> for LlamaParallelComposition {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, Array>,
    {
        if let Some(embedding) = &self.parallel_embedding {
            embedding.visit_parameters(visitor);
        }
        self.architecture
            .static_modules()
            .norm
            .visit_parameters(visitor);
        if let Some(head) = &self.parallel_lm_head {
            head.visit_parameters(visitor);
        }
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, Array>,
    {
        if let Some(embedding) = &mut self.parallel_embedding {
            embedding.visit_parameters_mut(visitor);
        }
        self.architecture
            .static_modules_mut()
            .norm
            .visit_parameters_mut(visitor);
        if let Some(head) = &mut self.parallel_lm_head {
            head.visit_parameters_mut(visitor);
        }
    }

    fn set_trainable(&mut self, trainable: bool) {
        if let Some(embedding) = &mut self.parallel_embedding {
            embedding.set_trainable(trainable);
        }
        self.architecture
            .static_modules_mut()
            .norm
            .set_trainable(trainable);
        if let Some(head) = &mut self.parallel_lm_head {
            head.set_trainable(trainable);
        }
    }
}

impl LayeredArchitecture<MlxBackend, MlxKeyValueState> for LlamaParallelComposition {
    type Input<'a> = eredu_architectures::llama::LayeredInput<'a, Array>;
    type StaticModules = Self;
    type Unit = NeutralBlock;
    type ForwardContext = eredu_architectures::llama::ForwardContext<Array>;
    type RetainedContextValues<'a> = std::option::Iter<'a, Array>;
    type Error = Error;

    fn model_identity(&self) -> &str {
        &self.args().model_type
    }

    fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Self::Error> {
        eredu_runtime::ExecutionGraph::chain(["text_decoder"]).map_err(Into::into)
    }

    fn unit_count(&self) -> Result<usize, Self::Error> {
        usize::try_from(self.args().num_hidden_layers).map_err(|_| {
            LlamaModelError::InvalidLayerCount {
                count: self.args().num_hidden_layers,
            }
            .into()
        })
    }

    fn unit_path(&self, index: usize) -> Result<String, Self::Error> {
        if index >= self.unit_count()? {
            return Err(Error::Parallel(format!(
                "parallel Llama unit {index} is outside the decoder"
            )));
        }
        Ok(format!("model.layers.{index}"))
    }

    fn static_modules(&self) -> &Self::StaticModules {
        self
    }

    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        self
    }

    fn build_unit(&self, index: usize, stream: &Stream) -> Result<Self::Unit, Self::Error> {
        let args = self
            .local_args
            .as_ref()
            .and_then(|args| args.get(index))
            .ok_or_else(|| {
                Error::Parallel(format!("parallel Llama unit {index} is not configured"))
            })?;
        NeutralBlock::new(args, index, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    fn begin_forward<'a>(
        &mut self,
        _input: Self::Input<'a>,
        _state: &mut MlxKeyValueState,
        _stream: &Stream,
    ) -> Result<LayeredForwardState<Array, Self::ForwardContext>, Self::Error> {
        Err(Error::Parallel(
            "parallel Llama composition requires a collective context".into(),
        ))
    }

    fn forward_unit(
        &mut self,
        _index: usize,
        _unit: &mut Self::Unit,
        _hidden: &Array,
        _state: &mut MlxKeyValueState,
        _forward: &mut Self::ForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Self::Error> {
        Err(Error::Parallel(
            "parallel Llama composition requires a collective context".into(),
        ))
    }

    fn finish_forward(
        &mut self,
        _hidden: &Array,
        _state: &mut MlxKeyValueState,
        _forward: &Self::ForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Self::Error> {
        Err(Error::Parallel(
            "parallel Llama composition requires a collective context".into(),
        ))
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        index: usize,
    ) -> Self::RetainedContextValues<'a> {
        <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxKeyValueState>>::retained_context_values(
            &self.architecture,
            forward,
            index,
        )
    }
}

impl ParallelLayeredArchitecture<MlxBackend, MlxKeyValueState> for LlamaParallelComposition {
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut MlxKeyValueState,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<LayeredForwardState<Array, Self::ForwardContext>, Self::Error> {
        let execution = self.execution_context(group, stream)?;
        let hidden = self
            .parallel_embedding
            .as_mut()
            .ok_or_else(|| Error::Parallel("parallel Llama embedding is not configured".into()))?
            .forward(input.tokens, &execution)?;
        let expected = self.local_state_layout()?;
        self.architecture
            .begin_embedded_with_layout(hidden, input.mask, state, &expected, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    fn forward_unit_parallel(
        &mut self,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &Array,
        state: &mut MlxKeyValueState,
        forward: &mut Self::ForwardContext,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Self::Error> {
        self.architecture
            .forward_block_parallel(index, unit, hidden, state, forward, group, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    fn finish_forward_parallel(
        &mut self,
        hidden: &Array,
        _state: &mut MlxKeyValueState,
        _forward: &Self::ForwardContext,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Self::Error> {
        let execution = self.execution_context(group, stream)?;
        let hidden = self
            .architecture
            .static_modules_mut()
            .norm
            .forward(hidden, stream)?;
        let logits = match &mut self.parallel_lm_head {
            Some(head) => head.forward(&hidden, &execution)?,
            None => self
                .parallel_embedding
                .as_mut()
                .ok_or_else(|| {
                    Error::Parallel("parallel Llama embedding is not configured".into())
                })?
                .project_logits(&hidden, &execution)?,
        };
        logits.all_gather(&execution)
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

impl From<LlamaModelError> for crate::backend::mlx::error::Error {
    fn from(error: LlamaModelError) -> Self {
        Self::ArchitectureModel(error.to_string())
    }
}

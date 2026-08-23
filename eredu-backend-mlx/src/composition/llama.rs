//! Unified Llama/Mistral loading across weight-residency policies.

#[path = "llama/checkpoint.rs"]
pub mod checkpoint;

use eredu_checkpoint::WeightQuantization;
use eredu_runtime::{
    CausalModel, ExecutionResidency, LayerWeightResidency, LayerwiseRuntime, RuntimeState,
    WeightResidency,
};

use std::{path::Path, sync::Arc};

use eredu_architectures::llama::ModelArgs;
use safemlx::{error::Exception, ops::indexing::TryIndexOp, Array, Stream};

use eredu_core::cache::{
    PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
    PromptCacheTopology,
};

use crate::{
    backend::mlx::error::Error,
    backend::mlx::nn::shared::{MlxBackend, MlxModule},
    backend::mlx::runtime::cache::residency::{open_prompt_cache, CacheResidencyManager},
    backend::mlx::runtime::cache::state::MlxKeyValueState,
    backend::mlx::runtime::checkpoint::binding::{binding_bytes, build_module_bindings},
    backend::mlx::runtime::checkpoint::{
        quantization::should_quantize_on_load, store::open_gguf_checkpoint_source,
    },
    backend::mlx::runtime::execution::generic::{
        prepare_layerwise_policy, prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy,
        MlxResidentPolicy,
    },
    backend::mlx::runtime::execution::layerwise::{
        open_safetensors_weight_store, quantize_parameterized_store, shard_layer_bindings,
    },
    backend::mlx::runtime::media::input,
};
use eredu_runtime::{
    CacheResidencyPolicy, DenseDiskStreamReport, LayerwiseModelMetadata, PagedCacheOptions,
    ParallelModelInfo, StaticUnitBindings,
};

use eredu_runtime::{ResidencyReport, WeightBinding};

type NeutralBlock = eredu_architectures::llama::TransformerBlock<MlxBackend>;

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
    MlxLayerwisePolicy<NeutralBlock>,
>;

type NeutralParallelResidentRuntime = LayerwiseRuntime<
    NeutralArchitecture,
    MlxBackend,
    MlxKeyValueState,
    MlxResidentPolicy<NeutralBlock>,
>;
type NeutralParallelLayerwiseRuntime = LayerwiseRuntime<
    NeutralArchitecture,
    MlxBackend,
    MlxKeyValueState,
    MlxLayerwisePolicy<NeutralBlock>,
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
    let mut architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let (policy, mut metadata) = prepare_layerwise_policy(
        store,
        &mut architecture,
        (),
        std::marker::PhantomData::<MlxKeyValueState>,
        options,
        stream,
        weights_stream,
        |key| key.starts_with("rope_freqs.") || key.ends_with(".rotary_emb.inv_freq"),
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.weight_quantization());
    metadata.set_materialization(materialization);
    let execution = if options.is_fully_resident() {
        LlamaExecution::Resident(LayerwiseRuntime::new_policy_first(
            policy.into_resident(
                &architecture,
                stream,
                std::marker::PhantomData::<MlxKeyValueState>,
            )?,
            architecture,
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

pub fn quantize_neutral_llama_store(
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
    parallel_info: Option<ParallelModelInfo<crate::backend::mlx::MlxParallelContext>>,
    parallel_rank: Option<eredu_core::cache::CacheRankIdentity>,
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
    ) -> Result<eredu_core::LayerSchedule<eredu_core::cache::LayerCachePolicy>, Error> {
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
    ) -> Option<&ParallelModelInfo<crate::backend::mlx::MlxParallelContext>> {
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
        rank: Option<eredu_core::cache::CacheRankIdentity>,
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
        rank: Option<eredu_core::cache::CacheRankIdentity>,
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
        let inputs = crate::MlxTensor::from_array(inputs.clone());
        let input = eredu_architectures::llama::LayeredInput {
            tokens: &inputs,
            mask: None,
        };
        let output = match &mut self.execution {
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
        }?;
        Ok(output.into_array())
    }

    /// Runs the canonical execution path with stable per-layer observation points.
    pub fn forward_with_observer(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut MlxKeyValueState,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, safemlx::error::Exception>,
    ) -> Result<Array, Error> {
        self.validate_cache(cache)?;
        let inputs = crate::MlxTensor::from_array(inputs.clone());
        let mask = mask.cloned().map(crate::MlxTensor::from_array);
        match &mut self.execution {
            LlamaExecution::TensorParallelResident(_)
            | LlamaExecution::TensorParallelLayerwise(_) => Err(Error::Parallel(
                "tensor-parallel observation requires its collective execution context".into(),
            )),
            LlamaExecution::Resident(execution) => {
                let output = execution
                    .forward_with_unit_hook(
                        eredu_architectures::llama::LayeredInput {
                            tokens: &inputs,
                            mask: mask.as_ref(),
                        },
                        cache,
                        stream,
                        |path, input, output| {
                            observer
                                .observe(&format!("{path}.input"), input.as_array())
                                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                            observer
                                .observe(&format!("{path}.output"), output.as_array())
                                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                            observer
                                .intervene(&format!("{path}.output"), output.as_array())
                                .map(|replacement| replacement.map(crate::MlxTensor::from_array))
                                .map_err(|error| eredu_nn::Error::backend(error.to_string()))
                        },
                    )
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
                observer.observe("model.logits", output.as_array())?;
                Ok(output.into_array())
            }
            LlamaExecution::Layerwise(execution) => {
                let output = execution
                    .forward_with_unit_hook(
                        eredu_architectures::llama::LayeredInput {
                            tokens: &inputs,
                            mask: mask.as_ref(),
                        },
                        cache,
                        stream,
                        |path, input, output| {
                            observer
                                .observe(&format!("{path}.input"), input.as_array())
                                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                            observer
                                .observe(&format!("{path}.output"), output.as_array())
                                .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                            observer
                                .intervene(&format!("{path}.output"), output.as_array())
                                .map(|replacement| replacement.map(crate::MlxTensor::from_array))
                                .map_err(|error| eredu_nn::Error::backend(error.to_string()))
                        },
                    )
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
                observer.observe("model.logits", output.as_array())?;
                Ok(output.into_array())
            }
        }
    }

    /// Runs a rank-local tensor-parallel forward pass.
    pub fn forward_tensor_parallel(
        &mut self,
        inputs: &Array,
        cache: &mut MlxKeyValueState,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.validate_cache(cache)?;
        let inputs = crate::MlxTensor::from_array(inputs.clone());
        let output = match &mut self.execution {
            LlamaExecution::TensorParallelResident(execution) => execution
                .forward_parallel(
                    eredu_architectures::llama::LayeredInput {
                        tokens: &inputs,
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
                        tokens: &inputs,
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
        }?;
        Ok(output.into_array())
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
    type Tensor = crate::MlxTensor;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut MlxKeyValueState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        self.prefill(&tokens, cache, stream)
            .map(crate::MlxTensor::from_array)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn decode_logits(
        &mut self,
        input_tokens: &crate::MlxTensor,
        cache: &mut MlxKeyValueState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Exception> {
        self.decode(input_tokens.as_array(), cache, stream)
            .map(crate::MlxTensor::from_array)
            .map_err(|error| Exception::custom(error.to_string()))
    }
}

pub fn load_llama_safetensors_mlx(
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
    let global_architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let mut planner = build.planner();
    let static_modules = global_architecture.static_modules();
    for group in eredu_architectures::llama::static_parallel_parameter_groups::<MlxBackend>(
        &static_modules.embeddings,
        &static_modules.norm,
        static_modules.lm_head.as_ref(),
        "model",
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
    let geometry = eredu_architectures::llama::local_geometry(&args, &layout)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let mut architecture = NeutralArchitecture::new_parallel(args.clone(), geometry, stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let state_layout = architecture
        .runtime_state_layout()
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let global_static_bindings = build_module_bindings(
        &MlxModule::new(global_architecture.static_modules().clone()),
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
    let global_static_modules = global_architecture.static_modules().clone();
    let binding_layout = layout.clone();
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        Arc::clone(&store),
        &mut architecture,
        (),
        std::marker::PhantomData::<MlxKeyValueState>,
        options,
        stream,
        weights_stream,
        |key| key.starts_with("rope_freqs.") || key.ends_with(".rotary_emb.inv_freq"),
        move |_modules, store| {
            let global = MlxModule::new(global_static_modules.clone());
            let bindings = build_module_bindings(&global, "", store)?;
            shard_layer_bindings(bindings, "", store, &binding_layout)
        },
        |address, path, _local, store, stream| {
            let index = address.index();
            let global = NeutralBlock::new(&binding_args, index, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            let bindings = build_module_bindings(&MlxModule::new(global), "", store)?;
            shard_layer_bindings(bindings, path, store, &layout)
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.weight_quantization());
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
        LlamaExecution::TensorParallelResident(Box::new(LayerwiseRuntime::new_policy_first(
            policy.into_resident(
                &architecture,
                stream,
                std::marker::PhantomData::<MlxKeyValueState>,
            )?,
            architecture,
        )))
    } else {
        LlamaExecution::TensorParallelLayerwise(Box::new(LayerwiseRuntime::new(
            architecture,
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
pub fn load_llama_tensor_parallel_model(
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
        let admitted = crate::composition::mlx::structural::admit_gguf_path(
            model_dir,
            crate::backend::mlx::ModelLoadOptions::default()
                .with_weight_residency(WeightResidency::with_layers(options)),
        )?;
        return load_llama_gguf_tensor_parallel_model(
            &admitted,
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
    source: &crate::composition::mlx::structural::AdmittedGguf,
    options: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(LlamaModel, Vec<u32>), Error> {
    let checkpoint = source.checkpoint();
    let prepared = checkpoint::prepare_llama_gguf_checkpoint(source, None, weights_stream)?;
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
    source: &crate::composition::mlx::structural::AdmittedGguf,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(LlamaModel, Vec<u32>), Error> {
    let checkpoint = source.checkpoint();
    let prepared = checkpoint::prepare_llama_gguf_checkpoint(source, None, weights_stream)?;
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
#[derive(Default)]
pub struct LlamaPipelineBindings;

impl LlamaPipelineBindings {
    /// Creates a stateless checkpoint-binding adapter.
    pub const fn new() -> Self {
        Self
    }

    pub fn cartesian_layer_bindings(
        &self,
        architecture: &NeutralArchitecture,
        index: usize,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        let global = architecture
            .construct_unit(index, stream)
            .map(MlxModule::new)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let bindings = build_module_bindings(&global, "", store)?;
        match layout {
            Some(layout) => {
                shard_layer_bindings(bindings, &format!("model.layers.{index}"), store, layout)
            }
            None => Ok(bindings),
        }
    }

    pub fn static_units(
        &self,
        architecture: &NeutralArchitecture,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        crate::composition::architecture_static_units(architecture, store)
    }
}

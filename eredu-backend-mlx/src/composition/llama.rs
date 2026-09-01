//! Unified Llama/Mistral loading across weight-residency policies.

#[path = "llama/checkpoint.rs"]
pub mod checkpoint;

use eredu_checkpoint::WeightQuantization;
use eredu_runtime::{
    ArchitectureParameters, CausalModel, LayerWeightResidency, LayerwiseRuntime, RuntimeState,
};

use std::{path::Path, sync::Arc};

use eredu_architectures::llama::ModelArgs;
use safemlx::{error::Exception, ops::indexing::TryIndexOp, Array, Stream};

use eredu_core::cache::{
    PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
    PromptCacheTopology,
};

use crate::{
    backend::error::Error,
    backend::nn::shared::{MlxModule, MlxNeuralBackend},
    backend::runtime::cache::residency::{open_prompt_cache, CacheResidencyManager},
    backend::runtime::cache::state::MlxKeyValueState,
    backend::runtime::checkpoint::binding::{binding_bytes, build_module_bindings},
    backend::runtime::checkpoint::store::open_gguf_checkpoint_source,
    backend::runtime::execution::generic::{
        architecture_execution_layout, construct_architecture_unit,
        prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
    },
    backend::runtime::execution::layerwise::{quantize_parameterized_store, shard_layer_bindings},
    backend::runtime::media::input,
};
use eredu_runtime::{
    CacheResidencyPolicy, DenseDiskStreamReport, PagedCacheOptions, ParallelModelInfo,
    StaticUnitBindings,
};

use eredu_runtime::{ResidencyReport, WeightBinding};

type NeutralBlock = eredu_architectures::llama::TransformerBlock<MlxNeuralBackend>;

type NeutralArchitecture = eredu_architectures::llama::LayeredModel<MlxNeuralBackend>;
type NeutralResidentRuntime = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxKeyValueState,
    MlxResidentPolicy<NeutralBlock>,
>;
type NeutralLayerwiseRuntime = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxKeyValueState,
    MlxLayerwisePolicy<NeutralBlock>,
>;

enum PartitionedLlamaExecution {
    TensorParallelResident(Box<NeutralResidentRuntime>),
    TensorParallelLayerwise(Box<NeutralLayerwiseRuntime>),
}

impl PartitionedLlamaExecution {
    fn architecture(&self) -> &NeutralArchitecture {
        match self {
            Self::TensorParallelResident(runtime) => runtime.architecture(),
            Self::TensorParallelLayerwise(runtime) => runtime.architecture(),
        }
    }
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
    let target_args = eredu_architectures::llama::load_time_quantization(source_args, quantization)
        .map_err(Error::ArchitectureModel)?;
    let source = NeutralArchitecture::new(source_args.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let target = NeutralArchitecture::new(target_args.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let source_layout = architecture_execution_layout::<_, MlxKeyValueState>(&source)?;
    let target_layout = architecture_execution_layout::<_, MlxKeyValueState>(&target)?;
    if source_layout != target_layout {
        return Err(Error::Quantization(
            "Llama quantization changed the architecture execution layout".into(),
        ));
    }
    let unit_count = source_layout.len();
    let source_static = source.static_modules().clone();
    let target_static = target.static_modules().clone();
    let (store, report) = quantize_parameterized_store(
        store,
        &source_static,
        &target_static,
        move |ordinal, stream| {
            construct_architecture_unit(
                &source,
                &source_layout,
                ordinal,
                stream,
                std::marker::PhantomData::<MlxKeyValueState>,
            )
        },
        move |ordinal, stream| {
            construct_architecture_unit(
                &target,
                &target_layout,
                ordinal,
                stream,
                std::marker::PhantomData::<MlxKeyValueState>,
            )
        },
        unit_count,
        quantization,
        stream,
    )?;
    Ok((store, target_args, report))
}

/// Llama/Mistral causal LM whose execution engine follows its residency policy.
pub struct PartitionedLlamaModel {
    args: ModelArgs,
    state_layout: eredu_runtime::StateLayout,
    parallel_info:
        Option<ParallelModelInfo<crate::composition::mlx::distributed::topology::MlxParallelPlan>>,
    parallel_rank: Option<eredu_core::cache::CacheRankIdentity>,
    execution: PartitionedLlamaExecution,
}

impl PartitionedLlamaModel {
    /// Returns normalized model arguments regardless of execution engine.
    pub fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Returns rank-local generalized parallel information when applicable.
    pub fn parallel_info(
        &self,
    ) -> Option<&ParallelModelInfo<crate::composition::mlx::distributed::topology::MlxParallelPlan>>
    {
        self.parallel_info.as_ref()
    }

    /// Returns logical residency and transfer telemetry for a layerwise model.
    pub fn residency_report(&self) -> Result<Option<ResidencyReport>, Error> {
        let report = match &self.execution {
            PartitionedLlamaExecution::TensorParallelResident(execution) => {
                execution.policy().residency_report()?
            }
            PartitionedLlamaExecution::TensorParallelLayerwise(execution) => {
                execution.policy().residency_report()?
            }
        };
        Ok(Some(report))
    }

    /// Returns dense-stream observations when that policy is active.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        match &self.execution {
            PartitionedLlamaExecution::TensorParallelLayerwise(execution) => {
                execution.policy().dense_stream_report()
            }
            PartitionedLlamaExecution::TensorParallelResident(_) => Ok(None),
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
            self.new_paged_cache_from_manager(manager, identity.topology().cache_rank_identity())?;
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
        _inputs: &Array,
        cache: &mut MlxKeyValueState,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        self.validate_cache(cache)?;
        Err(Error::Parallel(
            "tensor-parallel Llama requires its collective execution context".into(),
        ))
    }

    /// Runs the canonical execution path with stable per-layer observation points.
    pub fn forward_with_observer(
        &mut self,
        _inputs: &Array,
        _mask: Option<&Array>,
        cache: &mut MlxKeyValueState,
        _stream: &Stream,
        _observer: &mut dyn eredu_runtime::ActivationObserver<Array, safemlx::error::Exception>,
    ) -> Result<Array, Error> {
        self.validate_cache(cache)?;
        Err(Error::Parallel(
            "tensor-parallel observation requires its collective execution context".into(),
        ))
    }

    /// Runs a rank-local tensor-parallel forward pass.
    pub fn forward_tensor_parallel(
        &mut self,
        inputs: &Array,
        cache: &mut MlxKeyValueState,
        group: &crate::backend::runtime::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.validate_cache(cache)?;
        let inputs = crate::MlxTensor::from_array(inputs.clone());
        let output = match &mut self.execution {
            PartitionedLlamaExecution::TensorParallelResident(execution) => execution
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
            PartitionedLlamaExecution::TensorParallelLayerwise(execution) => execution
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
        }?;
        Ok(output.into_array())
    }

    pub fn forward_tensor_parallel_with_observer(
        &mut self,
        inputs: &Array,
        cache: &mut MlxKeyValueState,
        group: &crate::backend::runtime::distributed::Group,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, safemlx::error::Exception>,
    ) -> Result<Array, Error> {
        self.validate_cache(cache)?;
        let inputs = crate::MlxTensor::from_array(inputs.clone());
        let mut neutral = crate::composition::NeutralActivationObserver::new(observer);
        let output = match &mut self.execution {
            PartitionedLlamaExecution::TensorParallelResident(runtime) => runtime
                .forward_parallel_with_observer(
                    eredu_architectures::llama::LayeredInput {
                        tokens: &inputs,
                        mask: None,
                    },
                    cache,
                    group,
                    stream,
                    &mut neutral,
                ),
            PartitionedLlamaExecution::TensorParallelLayerwise(runtime) => runtime
                .forward_parallel_with_observer(
                    eredu_architectures::llama::LayeredInput {
                        tokens: &inputs,
                        mask: None,
                    },
                    cache,
                    group,
                    stream,
                    &mut neutral,
                ),
        }
        .map_err(|error| Error::Parallel(error.to_string()))?;
        eredu_runtime::observe_model_logits(&mut neutral, &output)
            .map(crate::MlxTensor::into_array)
            .map_err(Into::into)
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

    pub(crate) fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        let topology =
            self.parallel_info
                .as_ref()
                .map_or_else(PromptCacheTopology::default, |info| {
                    crate::composition::mlx::distributed::topology::prompt_cache_topology(
                        info.topology(),
                    )
                });
        crate::composition::replicated_prompt_cache_identity(
            self.execution.architecture(),
            topology,
        )
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

impl CausalModel<MlxKeyValueState> for PartitionedLlamaModel {
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

fn load_partitioned_llama(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    build: crate::composition::mlx::distributed::topology::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PartitionedLlamaModel, Error> {
    let global_architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let parameter_description = global_architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let execution_layout = parameter_description.unit_layout().clone();
    let layout =
        crate::composition::parallel_layout_from_description(build, &parameter_description)?;
    if layout.is_empty() {
        return Err(Error::Parallel(
            "Llama declared no tensor-parallel parameters".into(),
        ));
    }
    let geometry = eredu_architectures::llama::local_geometry(&args, &layout)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let mut architecture = NeutralArchitecture::new_parallel(args.clone(), geometry, stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let state_layout = architecture
        .state_layout()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let global_static_bindings = build_module_bindings(
        &MlxModule::new(global_architecture.static_modules().clone()),
        "",
        store.as_ref(),
    )?;
    let mut global_parameter_bytes = binding_bytes(&global_static_bindings)?;
    for ordinal in 0..execution_layout.len() {
        let unit = construct_architecture_unit(
            &global_architecture,
            &execution_layout,
            ordinal,
            stream,
            std::marker::PhantomData::<MlxKeyValueState>,
        )?;
        let bytes = binding_bytes(&build_module_bindings(
            &MlxModule::new(unit),
            "",
            store.as_ref(),
        )?)?;
        global_parameter_bytes = global_parameter_bytes
            .checked_add(bytes)
            .ok_or_else(|| Error::Parallel("global Llama parameter bytes overflowed".into()))?;
    }

    let global_static_modules = global_architecture.static_modules().clone();
    let binding_execution_layout = execution_layout;
    let binding_layout = layout.clone();
    let unit_binding_layout = layout.clone();
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        Arc::clone(&store),
        &mut architecture,
        (),
        std::marker::PhantomData::<MlxKeyValueState>,
        options,
        stream,
        weights_stream,
        |_| false,
        move |_modules, store| {
            let global = MlxModule::new(global_static_modules.clone());
            let bindings = build_module_bindings(&global, "", store)?;
            shard_layer_bindings(bindings, store, &binding_layout)
        },
        move |ordinal, _address, _path, _local, store, stream| {
            let global = construct_architecture_unit(
                &global_architecture,
                &binding_execution_layout,
                ordinal,
                stream,
                std::marker::PhantomData::<MlxKeyValueState>,
            )?;
            let bindings = build_module_bindings(&MlxModule::new(global), "", store)?;
            shard_layer_bindings(bindings, store, &unit_binding_layout)
        },
    )?;
    metadata.set_effective_model_type(args.model_type.clone());
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
        crate::composition::mlx::distributed::topology::prompt_cache_topology(build.topology())
            .cache_rank_identity();
    let execution = if options.is_fully_resident() {
        PartitionedLlamaExecution::TensorParallelResident(Box::new(
            LayerwiseRuntime::new_policy_first(
                policy.into_resident(
                    &architecture,
                    stream,
                    std::marker::PhantomData::<MlxKeyValueState>,
                )?,
                architecture,
            ),
        ))
    } else {
        PartitionedLlamaExecution::TensorParallelLayerwise(Box::new(LayerwiseRuntime::new(
            architecture,
            policy,
        )))
    };
    Ok(PartitionedLlamaModel {
        args,
        state_layout,
        parallel_info: Some(parallel_info),
        parallel_rank,
        execution,
    })
}

/// Loads one tensor-parallel Llama/Mistral rank from an admitted SafeTensors artifact.
pub fn load_partitioned_llama_safetensors(
    artifact: &crate::composition::mlx::artifact::PreparedSafetensorsArtifact,
    options: impl Into<LayerWeightResidency>,
    build: crate::composition::mlx::distributed::topology::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PartitionedLlamaModel, Error> {
    let options = options.into();
    let eredu_architectures::configuration::SafetensorsModelConfig::Llama(args) = artifact.model()
    else {
        return Err(Error::ArchitectureModel(
            "Llama loader received a different prepared architecture".into(),
        ));
    };
    let args = args.clone();
    let store = artifact.store();
    load_partitioned_llama(store, args, options, build, stream, weights_stream)
}

pub(crate) fn load_partitioned_llama_gguf(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    options: LayerWeightResidency,
    build: crate::composition::mlx::distributed::topology::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PartitionedLlamaModel, Error> {
    let checkpoint = source.checkpoint();
    let prepared = checkpoint::prepare_llama_gguf_checkpoint(source)?;
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
        Arc::new(open_gguf_checkpoint_source(
            checkpoint.clone(),
            source.plan().checkpoint(),
            source.plan().tensor_mapping(),
            options.max_cached_shards(),
        )?);
    let model =
        load_partitioned_llama(store, prepared.args, options, build, stream, weights_stream)?;
    Ok(model)
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
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let bindings = build_module_bindings(&global, "", store)?;
        match layout {
            Some(layout) => shard_layer_bindings(bindings, store, layout),
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

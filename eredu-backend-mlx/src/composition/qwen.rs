//! Unified Qwen loading across weight-residency policies.

use eredu_checkpoint::WeightQuantization;
use eredu_runtime::{
    ArchitectureParameters, CausalModel, ExecutionResidency, LayerWeightResidency,
    LayerwiseRuntime, RuntimeState, WeightResidency,
};

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::Arc,
};

use eredu_architectures::{media_plan::QwenVisionIngressPlan, qwen::ModelArgs};
use eredu_nn::RoutedNeuralBackend;
use safemlx::{error::Exception, ops::indexing::TryIndexOp, Array, Stream};

use eredu_core::cache::{
    PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
    PromptCacheTopology,
};

use crate::backend::runtime::checkpoint::load::gguf_quantization_configs;
use crate::{
    backend::error::Error,
    backend::nn::shared::{MlxModule, MlxNeuralBackend},
    backend::runtime::cache::residency::{open_prompt_cache, CacheResidencyManager},
    backend::runtime::cache::state::MlxKeyValueState,
    backend::runtime::checkpoint::binding::{
        apply_rank_local_parameter_recipes, binding_bytes,
        build_module_binding_plan_with_recipes_excluding, build_module_bindings,
        build_module_bindings_with_recipes_excluding, parameter_name_in_targets,
        parameter_role_targets, populate_module_from_lease_excluding,
    },
    backend::runtime::checkpoint::{
        quantization::should_quantize_on_load, store::open_gguf_checkpoint_source,
    },
    backend::runtime::execution::generic::{
        prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
        MlxUnitPopulator,
    },
    backend::runtime::execution::layerwise::{quantize_parameterized_store, shard_layer_bindings},
    backend::runtime::media::input,
    backend::runtime::residency::expert_cache::{ExpertCache, ExpertCacheReport},
    backend::runtime::residency::manager::ResidentUnitLease,
};

struct MaterializedQwenIngress {
    tokens: Array,
    patch_grid: Vec<(i32, i32, i32)>,
}

fn materialize_qwen_media_ingress(
    plan: QwenVisionIngressPlan,
    stream: &Stream,
) -> Result<MaterializedQwenIngress, Exception> {
    let placeholder_count = usize::try_from(plan.placeholder_count)
        .map_err(|_| Exception::custom("Qwen media placeholder span exceeds host capacity"))?;
    let tokens =
        input::token_ids_array(&vec![plan.placeholder_token_id; placeholder_count], stream)?;
    Ok(MaterializedQwenIngress {
        tokens,
        patch_grid: plan.patch_grid,
    })
}

pub mod expert {
    include!("qwen_expert.rs");
}

pub mod hybrid {
    include!("qwen_hybrid.rs");
}

#[cfg(feature = "image")]
pub mod processor {
    include!("qwen_processor.rs");
}

pub mod vl {
    include!("qwen_vl.rs");
}
use eredu_runtime::{
    CacheResidencyPolicy, DenseDiskStreamReport, LayerwiseModelMetadata, PagedCacheOptions,
    ParallelModelInfo, ParameterRole, StaticUnitBindings,
};

use eredu_runtime::{ResidencyReport, WeightBinding};

type NeutralBlock = eredu_architectures::qwen::RoutedTransformerBlock<MlxNeuralBackend>;

type NeutralArchitecture = eredu_architectures::qwen::RoutedLayeredModel<MlxNeuralBackend>;

fn require_decoder_group(architecture: &NeutralArchitecture, group: usize) -> Result<(), Error> {
    let transport = <NeutralArchitecture as eredu_runtime::LayeredArchitecture<
        MlxNeuralBackend,
        MlxKeyValueState,
    >>::group_transport(architecture, group);
    if transport.kind == eredu_runtime::ArchitectureGroupKind::Decoder {
        Ok(())
    } else {
        Err(Error::ArchitectureModel(format!(
            "Qwen checkpoint bindings require the decoder execution group, got {group}"
        )))
    }
}

fn decoder_unit_path(
    architecture: &NeutralArchitecture,
    group: usize,
    index: usize,
) -> Result<String, Error> {
    require_decoder_group(architecture, group)?;
    <NeutralArchitecture as eredu_runtime::LayeredArchitecture<
        MlxNeuralBackend,
        MlxKeyValueState,
    >>::unit_path(architecture, group, index)
    .map_err(|error| Error::ArchitectureModel(error.to_string()))
}

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
    MlxLayerwisePolicy<NeutralBlock, QwenUnitPopulator>,
>;

type NeutralParallelResidentRuntime = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxKeyValueState,
    MlxResidentPolicy<NeutralBlock>,
>;
type NeutralParallelLayerwiseRuntime = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxKeyValueState,
    MlxLayerwisePolicy<NeutralBlock, QwenUnitPopulator>,
>;

#[derive(Clone)]
struct QwenUnitPopulator {
    external_experts: bool,
    expert_targets: Arc<BTreeSet<String>>,
}

impl MlxUnitPopulator<NeutralBlock> for QwenUnitPopulator {
    fn populate(
        &mut self,
        unit: &mut MlxModule<NeutralBlock>,
        lease: &ResidentUnitLease,
    ) -> Result<(), Error> {
        populate_module_from_lease_excluding(unit, lease, |name| {
            self.external_experts && parameter_name_in_targets(name, &self.expert_targets)
        })?;
        Ok(())
    }
}

enum QwenExecution {
    Resident(Box<NeutralResidentRuntime>),
    Layerwise(Box<NeutralLayerwiseRuntime>),
    TensorParallelResident(Box<NeutralParallelResidentRuntime>),
    TensorParallelLayerwise(Box<NeutralParallelLayerwiseRuntime>),
}

struct NeutralQwenObserver<'a> {
    inner: &'a mut dyn eredu_runtime::ActivationObserver<Array, safemlx::error::Exception>,
}

impl eredu_runtime::ActivationObserver<crate::MlxTensor, eredu_nn::Error>
    for NeutralQwenObserver<'_>
{
    fn observe(&mut self, path: &str, value: &crate::MlxTensor) -> Result<(), eredu_nn::Error> {
        self.inner
            .observe(path, value.as_array())
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))
    }

    fn intervene(
        &mut self,
        path: &str,
        value: &crate::MlxTensor,
    ) -> Result<Option<crate::MlxTensor>, eredu_nn::Error> {
        self.inner
            .intervene(path, value.as_array())
            .map(|value| value.map(crate::MlxTensor::from_array))
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))
    }

    fn observe_routing(
        &mut self,
        routing: eredu_runtime::RoutingObservation<'_, crate::MlxTensor>,
    ) -> Result<(), eredu_nn::Error> {
        let routing = eredu_runtime::RoutingObservation {
            path: routing.path,
            selected_experts: routing.selected_experts.as_array(),
            selected_scores: routing.selected_scores.as_array(),
            route_weights: routing.route_weights.as_array(),
            routed_output: routing.routed_output.as_array(),
            local_routed_output: routing.local_routed_output.map(crate::MlxTensor::as_array),
            reduced_routed_output: routing
                .reduced_routed_output
                .map(crate::MlxTensor::as_array),
            shared_output: routing.shared_output.map(crate::MlxTensor::as_array),
            combined_output: routing.combined_output.map(crate::MlxTensor::as_array),
            expert_count: routing.expert_count,
        };
        self.inner
            .observe_routing(routing)
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))
    }
}

fn qwen_unit_recipes(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    args: &ModelArgs,
    layer: usize,
) -> Result<BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>, Error> {
    if !args.is_moe() {
        return Ok(BTreeMap::new());
    }
    let resolved =
        eredu_architectures::qwen::expert_recipes(store, args, &args.parameter_root, layer)
            .map_err(Error::ArchitectureModel)?;
    Ok(BTreeMap::from([
        (resolved.target_gate_up, resolved.gate_up),
        (resolved.target_down, resolved.down),
    ]))
}

fn load_neutral_qwen(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
    materialization: Option<eredu_runtime::WeightMaterializationReport>,
    external_experts: bool,
) -> Result<QwenModel, Error> {
    let mut architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let expert_targets = Arc::new(
        architecture
            .parameter_description(stream)
            .map_err(|error| Error::Parallel(error.to_string()))?
            .targets_for_role(ParameterRole::ExpertIntermediate),
    );
    let binding_args = args.clone();
    let excluded_expert_targets = Arc::clone(&expert_targets);
    let binding_expert_targets = Arc::clone(&expert_targets);
    let factory = QwenUnitPopulator {
        external_experts,
        expert_targets: Arc::clone(&expert_targets),
    };
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        store,
        &mut architecture,
        factory,
        std::marker::PhantomData::<MlxKeyValueState>,
        options,
        stream,
        weights_stream,
        move |key| external_experts && parameter_name_in_targets(key, &excluded_expert_targets),
        |modules, store| {
            build_module_bindings(&MlxModule::new(modules.clone()), "", store).map_err(Into::into)
        },
        move |_ordinal, address, _path, unit, store, _stream| {
            let index = address.index();
            let recipes = if external_experts {
                BTreeMap::new()
            } else {
                qwen_unit_recipes(store, &binding_args, index)?
            };
            build_module_bindings_with_recipes_excluding(
                &MlxModule::new(unit),
                "",
                store,
                recipes,
                |name| external_experts && parameter_name_in_targets(name, &binding_expert_targets),
            )
            .map_err(Into::into)
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.weight_quantization());
    metadata.set_materialization(materialization);
    let execution = if options.is_fully_resident() {
        QwenExecution::Resident(Box::new(LayerwiseRuntime::new_policy_first(
            policy.into_resident(
                &architecture,
                stream,
                std::marker::PhantomData::<MlxKeyValueState>,
            )?,
            architecture,
        )))
    } else {
        QwenExecution::Layerwise(Box::new(LayerwiseRuntime::new(architecture, policy)))
    };
    Ok(QwenModel {
        state_layout: eredu_architectures::qwen::state_layout(&args)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
        args,
        metadata,
        parallel_info: None,
        parallel_rank: None,
        execution,
        expert_cache: None,
    })
}

pub fn quantize_neutral_qwen_store(
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
    let target_args = eredu_architectures::qwen::load_time_quantization(source_args, quantization)
        .map_err(Error::ArchitectureModel)?;
    let source = NeutralArchitecture::new(source_args.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let target = NeutralArchitecture::new(target_args.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let count = usize::try_from(source_args.num_hidden_layers)
        .map_err(|_| Error::ArchitectureModel("invalid Qwen layer count".into()))?;
    let source_unit_args = source_args.clone();
    let target_unit_args = target_args.clone();
    let (store, report) = quantize_parameterized_store(
        store,
        source.static_modules(),
        target.static_modules(),
        move |index, stream| {
            eredu_architectures::qwen::new_routed_block::<MlxNeuralBackend>(
                &source_unit_args,
                index,
                stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        },
        move |index, stream| {
            eredu_architectures::qwen::new_routed_block::<MlxNeuralBackend>(
                &target_unit_args,
                index,
                stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        },
        count,
        quantization,
        stream,
    )?;
    Ok((store, target_args, report))
}

/// Qwen causal LM whose execution engine follows its residency policy.
pub struct QwenModel {
    args: ModelArgs,
    state_layout: eredu_runtime::StateLayout,
    metadata: LayerwiseModelMetadata,
    parallel_info: Option<ParallelModelInfo<crate::backend::MlxParallelContext>>,
    parallel_rank: Option<eredu_core::cache::CacheRankIdentity>,
    execution: QwenExecution,
    expert_cache: Option<ExpertCache>,
}

impl QwenModel {
    /// Returns normalized model arguments regardless of execution engine.
    pub fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Returns the canonical cache-relevant architecture identity.
    pub fn prompt_cache_architecture_fingerprint(&self) -> String {
        eredu_architectures::qwen::prompt_cache_architecture_fingerprint(self.args())
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
    pub fn parallel_info(&self) -> Option<&ParallelModelInfo<crate::backend::MlxParallelContext>> {
        self.parallel_info.as_ref()
    }

    /// Returns logical residency and transfer telemetry for a layerwise model.
    pub fn residency_report(&self) -> Result<Option<ResidencyReport>, Error> {
        let report = match &self.execution {
            QwenExecution::Resident(execution) => execution.policy().residency_report()?,
            QwenExecution::Layerwise(execution) => execution.policy().residency_report()?,
            QwenExecution::TensorParallelResident(execution) => {
                execution.policy().residency_report()?
            }
            QwenExecution::TensorParallelLayerwise(execution) => {
                execution.policy().residency_report()?
            }
        };
        Ok(Some(report))
    }

    /// Returns dense-stream observations when that policy is active.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        match &self.execution {
            QwenExecution::TensorParallelLayerwise(execution) => {
                execution.policy().dense_stream_report()
            }
            QwenExecution::TensorParallelResident(_) => Ok(None),
            QwenExecution::Layerwise(execution) => execution.policy().dense_stream_report(),
            QwenExecution::Resident(_) => Ok(None),
        }
    }

    /// Returns independent expert residency telemetry when configured.
    pub fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        self.expert_cache
            .as_ref()
            .map(ExpertCache::report)
            .transpose()
            .map_err(Error::from)
    }

    /// Returns the persistent checkpoint store used by a layerwise model.
    pub fn checkpoint_store(&self) -> &dyn eredu_checkpoint::store::CheckpointSource {
        match &self.execution {
            QwenExecution::Resident(execution) => execution.policy().checkpoint_store(),
            QwenExecution::Layerwise(execution) => execution.policy().checkpoint_store(),
            QwenExecution::TensorParallelResident(execution) => {
                execution.policy().checkpoint_store()
            }
            QwenExecution::TensorParallelLayerwise(execution) => {
                execution.policy().checkpoint_store()
            }
        }
    }

    pub fn checkpoint_store_arc(&self) -> Arc<dyn eredu_checkpoint::store::CheckpointSource> {
        match &self.execution {
            QwenExecution::Resident(execution) => execution.policy().checkpoint_store_arc(),
            QwenExecution::Layerwise(execution) => execution.policy().checkpoint_store_arc(),
            QwenExecution::TensorParallelResident(execution) => {
                execution.policy().checkpoint_store_arc()
            }
            QwenExecution::TensorParallelLayerwise(execution) => {
                execution.policy().checkpoint_store_arc()
            }
        }
    }

    /// Returns the number of pinned static leases used by the layerwise engine.
    pub fn static_lease_count(&self) -> usize {
        match &self.execution {
            QwenExecution::Resident(execution) => execution.policy().static_lease_count(),
            QwenExecution::Layerwise(execution) => execution.policy().static_lease_count(),
            QwenExecution::TensorParallelResident(execution) => {
                execution.policy().static_lease_count()
            }
            QwenExecution::TensorParallelLayerwise(execution) => {
                execution.policy().static_lease_count()
            }
        }
    }

    /// Creates the cache representation required by the model configuration.
    pub fn new_cache(&self) -> MlxKeyValueState {
        MlxKeyValueState::device(self.state_layout.clone())
            .expect("MLX key/value state supports the validated Qwen layout")
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
        if let Some(expert_cache) = self.expert_cache.take() {
            let args = self.args.clone();
            let result = {
                let mut provider = expert::cached_provider(&expert_cache, &args);
                self.forward_with_expert_provider(inputs, None, cache, &mut provider, stream)
            };
            self.expert_cache = Some(expert_cache);
            return result;
        }
        self.validate_cache(cache)?;
        let inputs = crate::MlxTensor::from_array(inputs.clone());
        let input = eredu_architectures::qwen::LayeredInput {
            tokens: &inputs,
            mask: None,
        };
        let output = match &mut self.execution {
            QwenExecution::Resident(execution) => execution
                .forward(input, cache, stream)
                .map_err(|error| Error::ArchitectureModel(error.to_string())),
            QwenExecution::Layerwise(execution) => execution
                .forward(input, cache, stream)
                .map_err(|error| Error::ArchitectureModel(error.to_string())),
            QwenExecution::TensorParallelResident(_)
            | QwenExecution::TensorParallelLayerwise(_) => Err(Error::Parallel(
                "tensor-parallel Qwen requires its collective execution context".into(),
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
        let args = self.args.clone();
        let pass = if inputs.dim(1) > 1 {
            eredu_runtime::ExpertPass::Prefill
        } else {
            eredu_runtime::ExpertPass::Decode
        };
        let expert_cache = self.expert_cache.take();
        let mut observer = NeutralQwenObserver { inner: observer };
        let result = match expert_cache.as_ref() {
            Some(expert_cache) => {
                let mut provider = expert::cached_provider(expert_cache, &args);
                self.forward_observed_with_provider(
                    inputs,
                    mask,
                    cache,
                    pass,
                    &mut provider,
                    stream,
                    &mut observer,
                )
            }
            None => {
                let mut provider = eredu_runtime::ResidentExpertProvider;
                self.forward_observed_with_provider(
                    inputs,
                    mask,
                    cache,
                    pass,
                    &mut provider,
                    stream,
                    &mut observer,
                )
            }
        };
        self.expert_cache = expert_cache;
        let output = result?;
        observer
            .inner
            .observe("model.logits", &output)
            .map_err(Error::from)?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_observed_with_provider<P>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut MlxKeyValueState,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        stream: &Stream,
        observer: &mut NeutralQwenObserver<'_>,
    ) -> Result<Array, Error>
    where
        P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
        P::Error: std::fmt::Display,
    {
        let inputs = crate::MlxTensor::from_array(inputs.clone());
        let mask = mask.cloned().map(crate::MlxTensor::from_array);
        let output = match &mut self.execution {
            QwenExecution::Resident(runtime) => runtime.forward_with_routed_observer(
                eredu_architectures::qwen::LayeredInput {
                    tokens: &inputs,
                    mask: mask.as_ref(),
                },
                cache,
                pass,
                provider,
                stream,
                observer,
            ),
            QwenExecution::Layerwise(runtime) => runtime.forward_with_routed_observer(
                eredu_architectures::qwen::LayeredInput {
                    tokens: &inputs,
                    mask: mask.as_ref(),
                },
                cache,
                pass,
                provider,
                stream,
                observer,
            ),
            QwenExecution::TensorParallelResident(_)
            | QwenExecution::TensorParallelLayerwise(_) => {
                return Err(Error::Parallel(
                    "tensor-parallel observation requires its collective execution context".into(),
                ))
            }
        }
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(output.into_array())
    }

    /// Runs a rank-local tensor-parallel forward pass.
    pub fn forward_tensor_parallel(
        &mut self,
        inputs: &Array,
        cache: &mut MlxKeyValueState,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if let Some(expert_cache) = self.expert_cache.take() {
            let args = self.args.clone();
            let result = {
                let mut provider = expert::cached_provider(&expert_cache, &args);
                self.forward_tensor_expert_provider(
                    inputs,
                    None,
                    cache,
                    group,
                    &mut provider,
                    stream,
                )
            };
            self.expert_cache = Some(expert_cache);
            return result;
        }
        self.validate_cache(cache)?;
        let inputs = crate::MlxTensor::from_array(inputs.clone());
        let output = match &mut self.execution {
            QwenExecution::TensorParallelResident(execution) => execution
                .forward_parallel(
                    eredu_architectures::qwen::LayeredInput {
                        tokens: &inputs,
                        mask: None,
                    },
                    cache,
                    group,
                    stream,
                )
                .map_err(|error| Error::Parallel(error.to_string())),
            QwenExecution::TensorParallelLayerwise(execution) => execution
                .forward_parallel(
                    eredu_architectures::qwen::LayeredInput {
                        tokens: &inputs,
                        mask: None,
                    },
                    cache,
                    group,
                    stream,
                )
                .map_err(|error| Error::Parallel(error.to_string())),
            QwenExecution::Resident(_) | QwenExecution::Layerwise(_) => Err(Error::Parallel(
                "model was not loaded for tensor-parallel execution".into(),
            )),
        }?;
        Ok(output.into_array())
    }

    /// Runs the neutral decoder while delegating routed experts to a runtime
    /// provider such as bounded expert residency or expert-parallel exchange.
    pub fn forward_with_expert_executor<F>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut MlxKeyValueState,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut provider =
            crate::backend::runtime::residency::expert_provider::ExpertExecutorProvider::new(
                &mut execute,
            );
        self.forward_with_expert_provider(inputs, mask, cache, &mut provider, stream)
    }

    fn forward_with_expert_provider<P>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut MlxKeyValueState,
        provider: &mut P,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
        P::Error: std::fmt::Display,
    {
        self.validate_cache(cache)?;
        let pass = if inputs.dim(1) > 1 {
            eredu_runtime::ExpertPass::Prefill
        } else {
            eredu_runtime::ExpertPass::Decode
        };
        let hook = |architecture: &mut NeutralArchitecture,
                    group: usize,
                    index: usize,
                    block: &mut NeutralBlock,
                    hidden: &crate::MlxTensor,
                    state: &mut MlxKeyValueState,
                    forward: &mut eredu_architectures::qwen::ForwardContext<crate::MlxTensor>,
                    context: &Stream| {
            <NeutralArchitecture as eredu_runtime::RoutedLayeredArchitecture<
                MlxNeuralBackend,
                MlxKeyValueState,
            >>::forward_unit_with_provider(
                architecture,
                group,
                index,
                block,
                hidden,
                state,
                forward,
                pass,
                provider,
                context,
            )
        };
        let inputs = crate::MlxTensor::from_array(inputs.clone());
        let mask = mask.cloned().map(crate::MlxTensor::from_array);
        let input = eredu_architectures::qwen::LayeredInput {
            tokens: &inputs,
            mask: mask.as_ref(),
        };
        let output = match &mut self.execution {
            QwenExecution::Resident(runtime) => runtime
                .forward_with_unit_executor(input, cache, stream, hook)
                .map_err(|error| Error::Parallel(error.to_string())),
            QwenExecution::Layerwise(runtime) => runtime
                .forward_with_unit_executor(input, cache, stream, hook)
                .map_err(|error| Error::Parallel(error.to_string())),
            QwenExecution::TensorParallelResident(_)
            | QwenExecution::TensorParallelLayerwise(_) => Err(Error::Parallel(
                "tensor-parallel Qwen requires its collective expert executor".into(),
            )),
        }?;
        Ok(output.into_array())
    }

    /// Runs tensor-parallel attention and projections while delegating routed
    /// experts to the matching-coordinate expert-parallel provider.
    pub fn forward_tensor_expert_parallel<F>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut MlxKeyValueState,
        group: &safemlx::distributed::Group,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut provider =
            crate::backend::runtime::residency::expert_provider::ExpertExecutorProvider::new(
                &mut execute,
            );
        self.forward_tensor_expert_provider(inputs, mask, cache, group, &mut provider, stream)
    }

    fn forward_tensor_expert_provider<P>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut MlxKeyValueState,
        group: &safemlx::distributed::Group,
        provider: &mut P,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
        P::Error: std::fmt::Display,
    {
        self.validate_cache(cache)?;
        let pass = if inputs.dim(1) > 1 {
            eredu_runtime::ExpertPass::Prefill
        } else {
            eredu_runtime::ExpertPass::Decode
        };
        let hook = |architecture: &mut NeutralArchitecture,
                    execution_group: usize,
                    index: usize,
                    block: &mut NeutralBlock,
                    hidden: &crate::MlxTensor,
                    state: &mut MlxKeyValueState,
                    forward: &mut eredu_architectures::qwen::ForwardContext<crate::MlxTensor>,
                    parallel: &safemlx::distributed::Group,
                    context: &Stream| {
            <NeutralArchitecture as eredu_runtime::ParallelRoutedLayeredArchitecture<
                MlxNeuralBackend,
                MlxKeyValueState,
            >>::forward_unit_parallel_with_provider(
                architecture,
                execution_group,
                index,
                block,
                hidden,
                state,
                forward,
                pass,
                provider,
                parallel,
                context,
            )
        };
        let inputs = crate::MlxTensor::from_array(inputs.clone());
        let mask = mask.cloned().map(crate::MlxTensor::from_array);
        let input = eredu_architectures::qwen::LayeredInput {
            tokens: &inputs,
            mask: mask.as_ref(),
        };
        let output = match &mut self.execution {
            QwenExecution::TensorParallelResident(runtime) => runtime
                .forward_parallel_with_unit_executor(input, cache, group, stream, hook)
                .map_err(|error| Error::Parallel(error.to_string())),
            QwenExecution::TensorParallelLayerwise(runtime) => runtime
                .forward_parallel_with_unit_executor(input, cache, group, stream, hook)
                .map_err(|error| Error::Parallel(error.to_string())),
            QwenExecution::Resident(_) | QwenExecution::Layerwise(_) => Err(Error::Parallel(
                "model was not loaded for tensor-parallel expert execution".into(),
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
            QwenExecution::Layerwise(_) => {}
            QwenExecution::TensorParallelLayerwise(execution) => {
                execution.policy().clear_device_window()?
            }
            QwenExecution::Resident(_) | QwenExecution::TensorParallelResident(_) => {
                return Ok(false)
            }
        }
        Ok(true)
    }

    pub fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        let layout = self.state_layout.clone();
        let topology = self
            .parallel_info
            .as_ref()
            .map_or_else(PromptCacheTopology::default, |info| {
                crate::backend::cache::prompt_cache_topology(info.topology())
            });
        let identity = eredu_architectures::qwen::state_identity(self.args(), &layout, 0, topology)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        identity
            .prompt_cache_identity(&layout)
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn validate_cache(&self, cache: &MlxKeyValueState) -> Result<(), Error> {
        let expected = &self.state_layout;
        if cache.layout() != expected {
            return Err(Exception::custom(format!(
                "MLX key/value state layout {:?} does not match Qwen layout {expected:?}",
                cache.layout()
            ))
            .into());
        }
        Ok(())
    }
}

impl CausalModel<MlxKeyValueState> for QwenModel {
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

pub fn load_qwen_safetensors_mlx(
    artifact: &crate::composition::mlx::artifact::PreparedSafetensorsArtifact,
    weight_residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenModel, Error> {
    let expert_options = weight_residency.expert_cache();
    let execution_options = weight_residency.layers();
    let eredu_architectures::configuration::SafetensorsModelConfig::Qwen(args) = artifact.model()
    else {
        return Err(Error::ArchitectureModel(
            "Qwen loader received a different prepared architecture".into(),
        ));
    };
    let args = args.clone();
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load("Qwen", args.weight_quantization(), requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let store = artifact.store();
    if let Some(quantization) = quantize_on_load {
        let (store, args, report) =
            quantize_neutral_qwen_store(store, &args, quantization, stream)?;
        let mut model = load_neutral_qwen(
            store,
            args,
            execution_options,
            stream,
            weights_stream,
            Some(report),
            expert_options.is_some(),
        )?;
        if let Some(options) = expert_options {
            attach_qwen_expert_cache(&mut model, options, stream, weights_stream)?;
        }
        return Ok(model);
    }
    let mut model = load_neutral_qwen(
        store,
        args,
        execution_options,
        stream,
        weights_stream,
        None,
        expert_options.is_some(),
    )?;
    if let Some(options) = expert_options {
        attach_qwen_expert_cache(&mut model, options, stream, weights_stream)?;
    }
    Ok(model)
}

fn attach_qwen_expert_cache(
    model: &mut QwenModel,
    options: eredu_runtime::ExpertCacheLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(), Error> {
    if !model.args.is_moe() {
        return Err(Error::ArchitectureModel(
            "independent expert caching requires Qwen3-MoE arguments".into(),
        ));
    }
    let store = model.checkpoint_store_arc();
    let entries = expert::expert_catalog(&model.args, store.as_ref())?;
    model.expert_cache = Some(ExpertCache::new_shared(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(())
}

fn load_neutral_qwen_parallel(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    build: crate::backend::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
    external_experts: bool,
) -> Result<QwenModel, Error> {
    let layer_count = usize::try_from(args.num_hidden_layers)
        .map_err(|_| Error::ArchitectureModel("invalid Qwen layer count".into()))?;
    let global_architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let expert_targets = Arc::new(
        global_architecture
            .parameter_description(stream)
            .map_err(|error| Error::Parallel(error.to_string()))?
            .targets_for_role(ParameterRole::ExpertIntermediate),
    );
    let mut planner = build.planner();
    let static_modules = global_architecture.static_modules();
    for group in eredu_architectures::qwen::static_parallel_parameter_groups::<MlxNeuralBackend>(
        &static_modules.embeddings,
        &static_modules.norm,
        static_modules.lm_head.as_ref(),
        &args.parameter_root,
    )? {
        planner.register(group)?;
    }
    for index in 0..layer_count {
        let unit =
            eredu_architectures::qwen::new_routed_block::<MlxNeuralBackend>(&args, index, stream)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        for group in eredu_architectures::qwen::routed_layer_parallel_parameter_groups::<
            MlxNeuralBackend,
        >(&unit, &args, index)?
        {
            planner.register(group)?;
        }
    }
    let (_, layout) = planner.finish()?;
    if layout.is_empty() {
        return Err(Error::Parallel(
            "Qwen declared no tensor-parallel parameters".into(),
        ));
    }
    let geometry = eredu_architectures::qwen::local_geometry(&args, &layout)
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
    for index in 0..layer_count {
        let unit =
            eredu_architectures::qwen::new_routed_block::<MlxNeuralBackend>(&args, index, stream)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let recipes = if external_experts {
            BTreeMap::new()
        } else {
            qwen_unit_recipes(store.as_ref(), &args, index)?
        };
        let bytes = binding_bytes(&build_module_bindings_with_recipes_excluding(
            &MlxModule::new(unit),
            "",
            store.as_ref(),
            recipes,
            |name| external_experts && parameter_name_in_targets(name, &expert_targets),
        )?)?;
        global_parameter_bytes = global_parameter_bytes
            .checked_add(bytes)
            .ok_or_else(|| Error::Parallel("global Qwen parameter bytes overflowed".into()))?;
    }

    let binding_args = args.clone();
    let global_static_modules = global_architecture.static_modules().clone();
    let binding_layout = layout.clone();
    let excluded_expert_targets = Arc::clone(&expert_targets);
    let binding_expert_targets = Arc::clone(&expert_targets);
    let factory = QwenUnitPopulator {
        external_experts,
        expert_targets: Arc::clone(&expert_targets),
    };
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        Arc::clone(&store),
        &mut architecture,
        factory,
        std::marker::PhantomData::<MlxKeyValueState>,
        options,
        stream,
        weights_stream,
        move |key| external_experts && parameter_name_in_targets(key, &excluded_expert_targets),
        move |_modules, store| {
            let global = MlxModule::new(global_static_modules.clone());
            let bindings = build_module_bindings(&global, "", store)?;
            shard_layer_bindings(bindings, "", store, &binding_layout)
        },
        |_ordinal, address, path, _local, store, stream| {
            let index = address.index();
            let global = eredu_architectures::qwen::new_routed_block::<MlxNeuralBackend>(
                &binding_args,
                index,
                stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let recipes = if external_experts {
                BTreeMap::new()
            } else {
                qwen_unit_recipes(store, &binding_args, index)?
            };
            let bindings = build_module_bindings_with_recipes_excluding(
                &MlxModule::new(global),
                "",
                store,
                recipes,
                |name| external_experts && parameter_name_in_targets(name, &binding_expert_targets),
            )?;
            shard_layer_bindings(bindings, path, store, &layout)
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.weight_quantization());
    let local_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.layer_parameter_bytes())
        .ok_or_else(|| Error::Parallel("local Qwen parameter bytes overflowed".into()))?;
    let maximum_device_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.maximum_device_layer_bytes())
        .ok_or_else(|| Error::Parallel("device Qwen parameter bytes overflowed".into()))?;
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
        crate::backend::cache::prompt_cache_topology(build.topology()).cache_rank_identity();
    let execution = if options.is_fully_resident() {
        QwenExecution::TensorParallelResident(Box::new(LayerwiseRuntime::new_policy_first(
            policy.into_resident(
                &architecture,
                stream,
                std::marker::PhantomData::<MlxKeyValueState>,
            )?,
            architecture,
        )))
    } else {
        QwenExecution::TensorParallelLayerwise(Box::new(LayerwiseRuntime::new(
            architecture,
            policy,
        )))
    };
    Ok(QwenModel {
        args,
        state_layout,
        metadata,
        parallel_info: Some(parallel_info),
        parallel_rank,
        execution,
        expert_cache: None,
    })
}

/// Loads Qwen through the generalized tensor-parallel execution engine.
pub fn load_qwen_tensor_parallel_model(
    artifact: &crate::composition::mlx::artifact::PreparedSafetensorsArtifact,
    options: impl Into<LayerWeightResidency>,
    build: crate::backend::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenModel, Error> {
    let options = options.into();
    let eredu_architectures::configuration::SafetensorsModelConfig::Qwen(args) = artifact.model()
    else {
        return Err(Error::ArchitectureModel(
            "Qwen loader received a different prepared architecture".into(),
        ));
    };
    let args = args.clone();
    let store = artifact.store();
    load_neutral_qwen_parallel(store, args, options, build, stream, weights_stream, false)
}

pub(crate) struct PreparedQwenGguf {
    pub args: ModelArgs,
}

pub(crate) fn prepare_qwen_gguf_checkpoint(
    source: &crate::composition::mlx::structural::AdmittedGguf,
) -> Result<PreparedQwenGguf, Error> {
    if !matches!(
        source.architecture(),
        eredu_architectures::GgufArchitecture::Qwen2
            | eredu_architectures::GgufArchitecture::Qwen3
            | eredu_architectures::GgufArchitecture::Qwen3Moe
    ) {
        return Err(Error::ArchitectureModel(format!(
            "Qwen GGUF loader received architecture {:?}",
            source.architecture()
        )));
    }
    let checkpoint = source.checkpoint();
    let eredu_architectures::configuration::GgufModelConfig::Qwen(args) = source.model() else {
        return Err(Error::ArchitectureModel(
            "Qwen GGUF loader received a different prepared model".into(),
        ));
    };
    let is_moe = args.is_moe();
    let translate =
        |name: &str| eredu_architectures::qwen::translate_gguf_weight_name(name, is_moe);
    let configs = gguf_quantization_configs(checkpoint, translate)?;
    let args = eredu_architectures::qwen::with_checkpoint_formats(args, configs)
        .map_err(Error::ArchitectureModel)?;
    Ok(PreparedQwenGguf { args })
}

pub(crate) fn load_qwen_gguf_tensor_parallel_model(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    options: LayerWeightResidency,
    build: crate::backend::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenModel, Error> {
    let checkpoint = source.checkpoint();
    let prepared = prepare_qwen_gguf_checkpoint(source)?;
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
        Arc::new(open_gguf_checkpoint_source(
            checkpoint.clone(),
            source.plan().checkpoint(),
            |name| {
                eredu_architectures::qwen::translate_gguf_weight_name(name, prepared.args.is_moe())
            },
            options.max_mapped_shards(),
        )?);
    let model = load_neutral_qwen_parallel(
        store,
        prepared.args,
        options,
        build,
        stream,
        weights_stream,
        false,
    )?;
    Ok(model)
}

/// Loads a Qwen GGUF checkpoint using the selected residency policy.
pub(crate) fn load_qwen_gguf_model(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenModel, Error> {
    let checkpoint = source.checkpoint();
    let prepared = prepare_qwen_gguf_checkpoint(source)?;
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
        Arc::new(open_gguf_checkpoint_source(
            checkpoint.clone(),
            source.plan().checkpoint(),
            |name| {
                eredu_architectures::qwen::translate_gguf_weight_name(name, prepared.args.is_moe())
            },
            residency.max_mapped_shards(),
        )?);
    let args = prepared.args;
    let expert_options = residency.expert_cache();
    let execution_options = residency.layers();
    let model = if let Some(quantization) = quantization {
        let (store, args, report) =
            quantize_neutral_qwen_store(store, &args, quantization, stream)?;
        load_neutral_qwen(
            store,
            args,
            execution_options,
            stream,
            weights_stream,
            Some(report),
            expert_options.is_some(),
        )?
    } else {
        load_neutral_qwen(
            store,
            args,
            execution_options,
            stream,
            weights_stream,
            None,
            expert_options.is_some(),
        )?
    };
    let mut model = model;
    if let Some(options) = expert_options {
        attach_qwen_expert_cache(&mut model, options, stream, weights_stream)?;
    }
    Ok(model)
}

/// Qwen binding and placement helper for pipeline-parallel stages.
#[derive(Default)]
pub struct QwenPipelineBindings {
    external_experts: bool,
}

impl QwenPipelineBindings {
    /// Creates a stateless checkpoint-binding adapter.
    pub const fn new() -> Self {
        Self {
            external_experts: false,
        }
    }

    pub const fn new_external_experts() -> Self {
        Self {
            external_experts: true,
        }
    }

    pub fn model_type<'a>(&self, architecture: &'a NeutralArchitecture) -> &'a str {
        &architecture.args().model_type
    }

    pub fn static_units(
        &self,
        architecture: &NeutralArchitecture,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        crate::composition::architecture_static_units(architecture, store)
    }

    pub fn quantizes_static_binding(&self, _binding: &WeightBinding) -> bool {
        true
    }

    /// Applies rank-local expert residency to an architecture-constructed unit.
    pub fn prepare_unit_expert_residency(
        &self,
        architecture: &NeutralArchitecture,
        index: usize,
        layer: &mut MlxModule<NeutralBlock>,
        local_intermediate_size: i32,
        assignment: Option<&crate::backend::runtime::distributed::expert::ExpertAssignment>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let args = architecture.args();
        if let (Some(assignment), eredu_architectures::qwen::FeedForward::Routed(moe)) =
            (assignment, &mut layer.inner.mlp)
        {
            let count = i32::try_from(assignment.local_global_expert_ids().len())
                .map_err(|_| Error::Parallel("local Qwen expert count exceeds i32".into()))?;
            if count > 0 {
                moe.experts = <MlxNeuralBackend as RoutedNeuralBackend>::gated_product_expert_bank(
                    eredu_architectures::qwen::localized_expert_bank_spec(
                        args,
                        index,
                        count,
                        local_intermediate_size,
                    )
                    .map_err(|error| Error::Parallel(error.to_string()))?,
                    stream,
                )
                .map_err(|error| Error::Parallel(error.to_string()))?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cartesian_layer_bindings(
        &self,
        architecture: &NeutralArchitecture,
        group: usize,
        index: usize,
        global_layer: &MlxModule<NeutralBlock>,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        assignment: Option<&crate::backend::runtime::distributed::expert::ExpertAssignment>,
    ) -> Result<Vec<WeightBinding>, Error> {
        require_decoder_group(architecture, group)?;
        let expert_targets = parameter_role_targets(
            &eredu_architectures::qwen::routed_layer_parallel_parameter_groups(
                global_layer,
                architecture.args(),
                index,
            )
            .map_err(|error| Error::Parallel(error.to_string()))?,
            ParameterRole::ExpertIntermediate,
        );
        let recipes = if self.external_experts {
            BTreeMap::new()
        } else {
            qwen_unit_recipes(store, architecture.args(), index)?
        };
        // Build against canonical bank recipes first. Architecture-produced
        // rank-local recipes then lower EP identity before generic TP sharding.
        let bindings = build_module_binding_plan_with_recipes_excluding(
            global_layer,
            "",
            store,
            recipes,
            |name| self.external_experts && parameter_name_in_targets(name, &expert_targets),
        )?
        .build_bindings(store)?;
        let bindings = match assignment {
            Some(assignment) if !self.external_experts => apply_rank_local_parameter_recipes(
                bindings,
                store,
                eredu_architectures::qwen::rank_local_expert_recipes(
                    store,
                    architecture.args(),
                    index,
                    assignment.local_global_expert_ids(),
                )
                .map_err(Error::ArchitectureModel)?,
            )?,
            _ => bindings,
        };
        match layout {
            Some(layout) => shard_layer_bindings(
                bindings,
                &decoder_unit_path(architecture, group, index)?,
                store,
                layout,
            ),
            None => Ok(bindings),
        }
    }

    pub fn expert_parallel_assignment(
        &self,
        architecture: &NeutralArchitecture,
        topology: crate::backend::MlxParallelContext,
    ) -> Result<Option<crate::backend::runtime::distributed::expert::ExpertAssignment>, Error> {
        if topology.expert_parallel_size == 1 && !self.external_experts {
            return Ok(None);
        }
        let args = architecture.args();
        if !args.is_moe() {
            return Err(Error::Parallel(
                "Qwen has no routed experts for expert-parallel ownership".into(),
            ));
        }
        Ok(Some(
            crate::backend::runtime::distributed::expert::ExpertAssignment::balanced(
                args.num_experts as usize,
                topology.expert_parallel_size,
                topology.expert_parallel_rank,
            )?,
        ))
    }

    pub fn layer_bindings(
        &self,
        architecture: &NeutralArchitecture,
        group: usize,
        index: usize,
        layer: &MlxModule<NeutralBlock>,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        require_decoder_group(architecture, group)?;
        let expert_targets = parameter_role_targets(
            &eredu_architectures::qwen::routed_layer_parallel_parameter_groups(
                layer,
                architecture.args(),
                index,
            )
            .map_err(|error| Error::Parallel(error.to_string()))?,
            ParameterRole::ExpertIntermediate,
        );
        let recipes = if self.external_experts {
            BTreeMap::new()
        } else {
            qwen_unit_recipes(store, architecture.args(), index)?
        };
        Ok(
            build_module_binding_plan_with_recipes_excluding(layer, "", store, recipes, |name| {
                self.external_experts && parameter_name_in_targets(name, &expert_targets)
            })?
            .build_bindings(store)?,
        )
    }
}

/// Structured failures at the unified Qwen model boundary.
#[derive(Debug, thiserror::Error)]
pub enum QwenModelError {
    /// The normalized decoder count cannot be represented by this runtime.
    #[error("invalid Qwen decoder layer count {count}")]
    InvalidLayerCount {
        /// Invalid configured count.
        count: i32,
    },
}

impl From<QwenModelError> for crate::backend::error::Error {
    fn from(error: QwenModelError) -> Self {
        Self::ArchitectureModel(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use eredu_checkpoint::store::MemoryWeightStore;
    use safemlx::{Device, DeviceType};

    use super::*;

    fn f32_tensor(
        name: &str,
        shape: Vec<usize>,
    ) -> (String, safetensors::Dtype, Vec<usize>, Vec<u8>) {
        let elements = shape.iter().product::<usize>();
        (
            name.into(),
            safetensors::Dtype::F32,
            shape,
            vec![0; elements * size_of::<f32>()],
        )
    }

    #[test]
    fn qwen_static_bindings_use_architecture_parameter_identities() {
        let args = eredu_architectures::qwen::model_args_from_config_value(&serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "rms_norm_eps": 0.00001,
            "vocab_size": 32,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false
        }))
        .unwrap();
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let architecture = NeutralArchitecture::new(args, &stream).unwrap();
        let store = MemoryWeightStore::from_safetensors([
            f32_tensor("model.embed_tokens.weight", vec![32, 8]),
            f32_tensor("model.norm.weight", vec![8]),
            f32_tensor("lm_head.weight", vec![32, 8]),
        ])
        .unwrap();

        let units = QwenPipelineBindings::new()
            .static_units(&architecture, &store)
            .unwrap();
        let actual = units
            .iter()
            .map(|unit| {
                (
                    unit.id().as_str(),
                    unit.bindings()[0].name(),
                    unit.bindings()[0].logical_target(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            [
                (
                    "embedding",
                    "model.embed_tokens.weight",
                    Some("model.embed_tokens.weight")
                ),
                ("norm", "model.norm.weight", Some("model.norm.weight")),
                ("output", "lm_head.weight", Some("lm_head.weight")),
            ]
        );
    }
}

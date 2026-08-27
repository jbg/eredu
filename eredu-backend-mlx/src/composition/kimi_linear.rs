//! Neutral Kimi Linear/Kimi Linear-MoE composition over MLX execution policies.

use std::{collections::BTreeSet, path::Path, sync::Arc};

use eredu_architectures::kimi_linear::{Block, LayeredModel, ModelArgs};
use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::CheckpointSource, WeightQuantization};
use eredu_runtime::{
    ArchitectureParameters, CacheResidencyPolicy, CausalModel, DenseDiskStreamReport,
    LayerWeightResidency, LayerwiseModelMetadata, LayerwiseRuntime, PagedCacheOptions,
    ParallelModelInfo, ParameterRole, ResidencyReport, WeightBinding, WeightResidency,
};
use safemlx::{error::Exception, ops::indexing::TryIndexOp, Array, Stream};

use crate::backend::{
    error::Error,
    nn::shared::{MlxModule, MlxNeuralBackend},
    runtime::{
        cache::{
            residency::{
                load_prompt_cache_state_tensors, open_prompt_cache, CacheResidencyManager,
            },
            state::MlxHybridState,
        },
        checkpoint::{
            binding::{
                binding_bytes, build_module_bindings, build_module_bindings_with_recipes_excluding,
                parameter_name_in_targets, parameter_role_targets,
                populate_module_from_lease_excluding,
            },
            load::gguf_quantization_configs,
            quantization::should_quantize_on_load,
            store::open_gguf_checkpoint_source,
        },
        execution::{
            generic::{
                architecture_execution_layout, construct_architecture_unit,
                prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
                MlxUnitPopulator,
            },
            layerwise::{quantize_parameterized_store, shard_layer_bindings},
        },
        media::input,
        residency::expert_cache::ExpertCatalogEntry,
        residency::expert_cache::{ExpertCache, ExpertCacheReport},
        residency::expert_provider::{CachedGatedProductExpertProvider, ExpertExecutorProvider},
    },
};
use eredu_core::cache::{
    PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
    PromptCacheTopology,
};

type NeutralBlock = Block<MlxNeuralBackend>;
type NeutralArchitecture = LayeredModel<MlxNeuralBackend>;
type ResidentRuntime = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxHybridState,
    MlxResidentPolicy<NeutralBlock>,
>;
type BoundedRuntime = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxHybridState,
    MlxLayerwisePolicy<NeutralBlock, KimiLinearUnitPopulator>,
>;
type ParallelResidentRuntime = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxHybridState,
    MlxResidentPolicy<NeutralBlock>,
>;
type ParallelBoundedRuntime = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxHybridState,
    MlxLayerwisePolicy<NeutralBlock, KimiLinearParallelUnitPopulator>,
>;

#[derive(eredu_nn::Parameterized)]
#[parameterized(tensor = "crate::MlxTensor")]
#[doc(hidden)]
#[cfg(test)]
pub struct KimiLinearCheckpointTemplate {
    pub static_modules: eredu_architectures::decoder::StaticModules<MlxNeuralBackend>,
    pub layers: Vec<NeutralBlock>,
}

#[cfg(test)]
impl KimiLinearCheckpointTemplate {
    /// Builds one neutral full-parameter template for checkpoint tooling.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let architecture = NeutralArchitecture::new(args, stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let layout = architecture_execution_layout::<_, MlxHybridState>(&architecture)?;
        let layers = (0..layout.len())
            .map(|ordinal| {
                construct_architecture_unit(
                    &architecture,
                    &layout,
                    ordinal,
                    stream,
                    std::marker::PhantomData::<MlxHybridState>,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            static_modules: architecture.static_modules().clone(),
            layers,
        })
    }
}

#[derive(Clone)]
struct KimiLinearUnitPopulator {
    external_experts: bool,
    expert_targets: Arc<BTreeSet<String>>,
}

/// Pipeline/loading adapter over the same neutral Kimi Linear blocks used by resident
/// and bounded execution.
#[derive(Default)]
pub struct KimiLinearBindings {
    external_experts: bool,
}

impl KimiLinearBindings {
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

    pub fn layer_count(
        &self,
        architecture: &NeutralArchitecture,
        group: usize,
    ) -> Result<usize, Error> {
        <NeutralArchitecture as eredu_runtime::LayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::group_unit_count(architecture, group)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    pub fn layer_bindings(
        &self,
        architecture: &NeutralArchitecture,
        _group: usize,
        index: usize,
        layer: &MlxModule<NeutralBlock>,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        let expert_targets = parameter_role_targets(
            &eredu_architectures::kimi_linear::layer_parallel_parameter_groups(
                layer,
                architecture.args(),
                index,
            )?,
            ParameterRole::ExpertIntermediate,
        );
        let recipes = unit_recipes(store, architecture.args(), index, !self.external_experts)?;
        build_module_bindings_with_recipes_excluding(layer, "", store, recipes, |name| {
            self.external_experts && parameter_name_in_targets(name, &expert_targets)
        })
        .map_err(Into::into)
    }

    pub fn quantizes_static_binding(&self, _binding: &WeightBinding) -> bool {
        true
    }

    pub fn expert_parallel_assignment(
        &self,
        realization: Option<
            &eredu_architectures::ExpertRealizationPlan<eredu_nn::GatedProductExpertBankSpec>,
        >,
    ) -> Result<Option<crate::backend::runtime::distributed::expert::ExpertAssignment>, Error> {
        match realization {
            None if self.external_experts => Err(Error::Parallel(
                "Kimi Linear has no architecture expert realization".into(),
            )),
            None => Ok(None),
            Some(plan) if plan.expert_parallel_size() == 1 && !self.external_experts => Ok(None),
            Some(plan) => {
                crate::backend::runtime::distributed::expert::ExpertAssignment::from_realization(
                    plan,
                )
                .map(Some)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cartesian_layer_bindings(
        &self,
        architecture: &NeutralArchitecture,
        group: usize,
        index: usize,
        global_layer: &MlxModule<NeutralBlock>,
        store: &dyn CheckpointSource,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        _assignment: Option<&crate::backend::runtime::distributed::expert::ExpertAssignment>,
    ) -> Result<Vec<WeightBinding>, Error> {
        self.layer_count(architecture, group)?;
        let bindings = self.layer_bindings(architecture, group, index, global_layer, store)?;
        match layout {
            Some(layout) => shard_layer_bindings(bindings, store, layout),
            None => Ok(bindings),
        }
    }
}

impl MlxUnitPopulator<NeutralBlock> for KimiLinearUnitPopulator {
    fn populate(
        &mut self,
        unit: &mut MlxModule<NeutralBlock>,
        lease: &crate::backend::runtime::residency::manager::ResidentUnitLease,
    ) -> Result<(), Error> {
        populate_module_from_lease_excluding(unit, lease, |name| {
            self.external_experts && parameter_name_in_targets(name, &self.expert_targets)
        })?;
        Ok(())
    }
}

#[derive(Clone)]
struct KimiLinearParallelUnitPopulator {
    external_experts: bool,
    expert_targets: Arc<BTreeSet<String>>,
}

impl MlxUnitPopulator<NeutralBlock> for KimiLinearParallelUnitPopulator {
    fn populate(
        &mut self,
        unit: &mut MlxModule<NeutralBlock>,
        lease: &crate::backend::runtime::residency::manager::ResidentUnitLease,
    ) -> Result<(), Error> {
        populate_module_from_lease_excluding(unit, lease, |name| {
            self.external_experts && parameter_name_in_targets(name, &self.expert_targets)
        })?;
        Ok(())
    }
}

enum KimiLinearExecution {
    Resident(Box<ResidentRuntime>),
    Layerwise(Box<BoundedRuntime>),
    TensorParallelResident(Box<ParallelResidentRuntime>),
    TensorParallelLayerwise(Box<ParallelBoundedRuntime>),
}

fn unit_recipes(
    store: &dyn CheckpointSource,
    args: &ModelArgs,
    layer: usize,
    include_experts: bool,
) -> Result<std::collections::BTreeMap<String, DerivedWeightRecipe>, Error> {
    eredu_architectures::kimi_linear::unit_recipes(store, args, layer, include_experts)
        .map_err(Error::ArchitectureModel)
}

pub fn expert_catalog(
    args: &ModelArgs,
    store: &dyn CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let catalog = eredu_architectures::kimi_linear::expert_residency_catalog(store, args)
        .map_err(Error::ArchitectureModel)?;
    crate::composition::architecture_expert_units(catalog, store, None)
}

const fn cached_provider<'a>(
    cache: &'a ExpertCache,
    _args: &ModelArgs,
) -> CachedGatedProductExpertProvider<'a> {
    CachedGatedProductExpertProvider::new(cache)
}

fn load_neutral(
    store: Arc<dyn CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
    materialization: Option<eredu_runtime::WeightMaterializationReport>,
    external_experts: bool,
) -> Result<KimiLinearModel, Error> {
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
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        store,
        &mut architecture,
        KimiLinearUnitPopulator {
            external_experts,
            expert_targets: Arc::clone(&expert_targets),
        },
        std::marker::PhantomData::<MlxHybridState>,
        options,
        stream,
        weights_stream,
        move |key| external_experts && parameter_name_in_targets(key, &excluded_expert_targets),
        |modules, store| {
            build_module_bindings(&MlxModule::new(modules.clone()), "", store).map_err(Into::into)
        },
        move |_ordinal, address, _path, unit, store, _| {
            let index = address.index();
            build_module_bindings_with_recipes_excluding(
                &MlxModule::new(unit),
                "",
                store,
                unit_recipes(store, &binding_args, index, !external_experts)?,
                |name| external_experts && parameter_name_in_targets(name, &binding_expert_targets),
            )
            .map_err(Into::into)
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.weight_quantization);
    metadata.set_materialization(materialization);
    let state_layout = architecture
        .state_layout()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let execution = if options.is_fully_resident() {
        KimiLinearExecution::Resident(Box::new(LayerwiseRuntime::new_policy_first(
            policy.into_resident(
                &architecture,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )?,
            architecture,
        )))
    } else {
        KimiLinearExecution::Layerwise(Box::new(LayerwiseRuntime::new(architecture, policy)))
    };
    Ok(KimiLinearModel {
        args,
        state_layout,
        metadata,
        execution,
        expert_cache: None,
        parallel_info: None,
        parallel_rank: None,
    })
}

fn load_neutral_parallel(
    store: Arc<dyn CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    build: crate::backend::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
    external_experts: bool,
) -> Result<KimiLinearModel, Error> {
    let global_architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let parameter_description = global_architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let execution_layout = parameter_description.unit_layout().clone();
    let expert_targets =
        Arc::new(parameter_description.targets_for_role(ParameterRole::ExpertIntermediate));
    let layout =
        crate::composition::parallel_layout_from_description(build, &parameter_description)?;
    if layout.is_empty() {
        return Err(Error::Parallel(
            "Kimi Linear declared no tensor-parallel parameters".into(),
        ));
    }
    let geometry = eredu_architectures::kimi_linear::local_geometry(&args, &layout)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let mut architecture = NeutralArchitecture::new_parallel(args.clone(), geometry, stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let state_layout = architecture
        .state_layout()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let factory = KimiLinearParallelUnitPopulator {
        external_experts,
        expert_targets: Arc::clone(&expert_targets),
    };

    let global_static = MlxModule::new(global_architecture.static_modules().clone());
    let global_static_bindings = build_module_bindings(&global_static, "", store.as_ref())?;
    let mut global_parameter_bytes = binding_bytes(&global_static_bindings)?;
    for ordinal in 0..execution_layout.len() {
        let address = execution_layout
            .address(ordinal)
            .expect("canonical execution ordinal");
        let block = construct_architecture_unit(
            &global_architecture,
            &execution_layout,
            ordinal,
            stream,
            std::marker::PhantomData::<MlxHybridState>,
        )?;
        let bindings = build_module_bindings_with_recipes_excluding(
            &MlxModule::new(block),
            "",
            store.as_ref(),
            unit_recipes(store.as_ref(), &args, address.index(), !external_experts)?,
            |name| external_experts && parameter_name_in_targets(name, &expert_targets),
        )?;
        global_parameter_bytes = global_parameter_bytes
            .checked_add(binding_bytes(&bindings)?)
            .ok_or_else(|| {
                Error::Parallel("global Kimi Linear parameter bytes overflowed".into())
            })?;
    }

    let shared_layout = Arc::new(layout);
    let static_layout = Arc::clone(&shared_layout);
    let unit_layout = Arc::clone(&shared_layout);
    let binding_args = args.clone();
    let global_static_modules = global_architecture.static_modules().clone();
    let binding_execution_layout = execution_layout;
    let excluded_expert_targets = Arc::clone(&expert_targets);
    let binding_expert_targets = Arc::clone(&expert_targets);
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        Arc::clone(&store),
        &mut architecture,
        factory,
        std::marker::PhantomData::<MlxHybridState>,
        options,
        stream,
        weights_stream,
        move |key| external_experts && parameter_name_in_targets(key, &excluded_expert_targets),
        move |_modules, store| {
            let global = MlxModule::new(global_static_modules.clone());
            let bindings = build_module_bindings(&global, "", store)?;
            shard_layer_bindings(bindings, store, &static_layout)
        },
        move |ordinal, address, _path, _local, store, stream| {
            let layer = address.index();
            let global = construct_architecture_unit(
                &global_architecture,
                &binding_execution_layout,
                ordinal,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )?;
            let bindings = build_module_bindings_with_recipes_excluding(
                &MlxModule::new(global),
                "",
                store,
                unit_recipes(store, &binding_args, layer, !external_experts)?,
                |name| external_experts && parameter_name_in_targets(name, &binding_expert_targets),
            )?;
            shard_layer_bindings(bindings, store, &unit_layout)
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.weight_quantization);
    let local_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.layer_parameter_bytes())
        .ok_or_else(|| Error::Parallel("local Kimi Linear parameter bytes overflowed".into()))?;
    let maximum_device_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.maximum_device_layer_bytes())
        .ok_or_else(|| Error::Parallel("device Kimi Linear parameter bytes overflowed".into()))?;
    let info = ParallelModelInfo::new(
        build.topology(),
        args.model_type.clone(),
        shared_layout
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
    let rank = crate::backend::cache::prompt_cache_topology(build.topology()).cache_rank_identity();
    let execution = if options.is_fully_resident() {
        KimiLinearExecution::TensorParallelResident(Box::new(LayerwiseRuntime::new_policy_first(
            policy.into_resident(
                &architecture,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )?,
            architecture,
        )))
    } else {
        KimiLinearExecution::TensorParallelLayerwise(Box::new(LayerwiseRuntime::new(
            architecture,
            policy,
        )))
    };
    Ok(KimiLinearModel {
        args,
        state_layout,
        metadata,
        execution,
        expert_cache: None,
        parallel_info: Some(info),
        parallel_rank: rank,
    })
}

fn quantize_store(
    store: Arc<dyn CheckpointSource>,
    args: &ModelArgs,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<
    (
        Arc<dyn CheckpointSource>,
        ModelArgs,
        eredu_runtime::WeightMaterializationReport,
    ),
    Error,
> {
    let target = eredu_architectures::kimi_linear::load_time_quantization(args, quantization)
        .map_err(Error::ArchitectureModel)?;
    let source = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let destination = NeutralArchitecture::new(target.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let source_layout = architecture_execution_layout::<_, MlxHybridState>(&source)?;
    let target_layout = architecture_execution_layout::<_, MlxHybridState>(&destination)?;
    if source_layout != target_layout {
        return Err(Error::Quantization(
            "Kimi Linear quantization changed the architecture execution layout".into(),
        ));
    }
    let unit_count = source_layout.len();
    let source_static = source.static_modules().clone();
    let target_static = destination.static_modules().clone();
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
                std::marker::PhantomData::<MlxHybridState>,
            )
        },
        move |ordinal, stream| {
            construct_architecture_unit(
                &destination,
                &target_layout,
                ordinal,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )
        },
        unit_count,
        quantization,
        stream,
    )?;
    Ok((store, target, report))
}

/// Kimi Linear causal model whose equations are owned by `eredu-architectures`.
pub struct KimiLinearModel {
    args: ModelArgs,
    state_layout: eredu_runtime::StateLayout,
    metadata: LayerwiseModelMetadata,
    execution: KimiLinearExecution,
    expert_cache: Option<ExpertCache>,
    parallel_info: Option<ParallelModelInfo<crate::backend::MlxParallelContext>>,
    parallel_rank: Option<eredu_core::cache::CacheRankIdentity>,
}

impl KimiLinearModel {
    /// Returns validated family policy.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Returns canonical residency metadata.
    pub const fn residency_metadata(&self) -> &LayerwiseModelMetadata {
        &self.metadata
    }

    /// Returns parallel metadata when a distributed binder supplied it.
    pub fn parallel_info(&self) -> Option<&ParallelModelInfo<crate::backend::MlxParallelContext>> {
        self.parallel_info.as_ref()
    }

    /// Creates device-resident heterogeneous state.
    pub fn new_cache(&self) -> MlxHybridState {
        MlxHybridState::device(self.state_layout.clone())
            .expect("validated Kimi Linear state must be realizable by MLX")
    }

    /// Creates state with the requested attention residency.
    pub fn new_cache_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<MlxHybridState, Error> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => {
                let manager = CacheResidencyManager::new(options)
                    .map_err(|error| Exception::custom(error.to_string()))?;
                MlxHybridState::paged(self.state_layout.clone(), manager, self.parallel_rank)
                    .map_err(Into::into)
            }
        }
    }

    /// Returns cache residency telemetry.
    pub fn cache_residency_report(
        &self,
        cache: &MlxHybridState,
    ) -> Result<Option<eredu_runtime::CacheResidencyReport>, Error> {
        cache.residency_report().map_err(Into::into)
    }

    /// Returns weight residency telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        match &self.execution {
            KimiLinearExecution::Resident(runtime) => runtime.policy().residency_report(),
            KimiLinearExecution::Layerwise(runtime) => runtime.policy().residency_report(),
            KimiLinearExecution::TensorParallelResident(runtime) => {
                runtime.policy().residency_report()
            }
            KimiLinearExecution::TensorParallelLayerwise(runtime) => {
                runtime.policy().residency_report()
            }
        }
    }

    /// Returns disk streaming telemetry when enabled.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        match &self.execution {
            KimiLinearExecution::Resident(_) => Ok(None),
            KimiLinearExecution::Layerwise(runtime) => runtime.policy().dense_stream_report(),
            KimiLinearExecution::TensorParallelResident(_) => Ok(None),
            KimiLinearExecution::TensorParallelLayerwise(runtime) => {
                runtime.policy().dense_stream_report()
            }
        }
    }

    /// Returns independent expert-cache telemetry when enabled.
    pub fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        self.expert_cache
            .as_ref()
            .map(ExpertCache::report)
            .transpose()
            .map_err(Into::into)
    }

    /// Returns the persistent checkpoint source.
    pub fn checkpoint_store(&self) -> &dyn CheckpointSource {
        match &self.execution {
            KimiLinearExecution::Resident(runtime) => runtime.policy().checkpoint_store(),
            KimiLinearExecution::Layerwise(runtime) => runtime.policy().checkpoint_store(),
            KimiLinearExecution::TensorParallelResident(runtime) => {
                runtime.policy().checkpoint_store()
            }
            KimiLinearExecution::TensorParallelLayerwise(runtime) => {
                runtime.policy().checkpoint_store()
            }
        }
    }

    pub fn checkpoint_store_arc(&self) -> Arc<dyn CheckpointSource> {
        match &self.execution {
            KimiLinearExecution::Resident(runtime) => runtime.policy().checkpoint_store_arc(),
            KimiLinearExecution::Layerwise(runtime) => runtime.policy().checkpoint_store_arc(),
            KimiLinearExecution::TensorParallelResident(runtime) => {
                runtime.policy().checkpoint_store_arc()
            }
            KimiLinearExecution::TensorParallelLayerwise(runtime) => {
                runtime.policy().checkpoint_store_arc()
            }
        }
    }

    /// Returns the canonical prompt-cache fingerprint.
    pub fn prompt_cache_architecture_fingerprint(&self) -> Result<String, Error> {
        Ok(eredu_architectures::kimi_linear::prompt_cache_architecture_fingerprint(&self.args))
    }

    /// Returns the exact rank-local state layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<eredu_core::LayerSchedule<eredu_core::cache::LayerCachePolicy>, Error> {
        Ok(self.state_layout.layers().clone())
    }

    /// Returns the canonical prompt-cache identity.
    pub fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        eredu_architectures::kimi_linear::state_identity(
            &self.args,
            &self.state_layout,
            0,
            self.parallel_info
                .as_ref()
                .map_or_else(PromptCacheTopology::default, |info| {
                    crate::backend::cache::prompt_cache_topology(info.topology())
                }),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        .prompt_cache_identity(&self.state_layout)
        .map_err(|error| Error::Parallel(error.to_string()))
    }

    /// Persists paged attention and all fixed components atomically.
    pub fn save_prompt_cache(
        &self,
        cache: &mut MlxHybridState,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        _stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        self.save_prompt_cache_with_identity(
            cache,
            destination,
            descriptor,
            prefix_token_ids,
            options,
            &self.prompt_cache_model_identity()?,
        )
    }

    pub fn save_prompt_cache_with_identity(
        &self,
        cache: &mut MlxHybridState,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        identity: &PromptCacheModelIdentity,
    ) -> Result<PromptCacheManifest, Error> {
        eredu_core::cache::validate_prompt_cache_model_identity(&descriptor, identity)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        cache
            .save_prompt_cache(destination, descriptor, prefix_token_ids, options)
            .map_err(Into::into)
    }

    /// Opens paged attention and restores all fixed components.
    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(MlxHybridState, PromptCacheManifest), Error> {
        let identity = self.prompt_cache_model_identity()?;
        self.load_prompt_cache_with_identity(
            directory,
            expected,
            prefix_token_ids,
            options,
            &identity,
            stream,
        )
    }

    pub fn load_prompt_cache_with_identity(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        identity: &PromptCacheModelIdentity,
        stream: &Stream,
    ) -> Result<(MlxHybridState, PromptCacheManifest), Error> {
        let (manager, manifest) = open_prompt_cache(
            directory.as_ref(),
            expected,
            identity,
            prefix_token_ids,
            options,
        )
        .map_err(|error| Exception::custom(error.to_string()))?;
        let tensors = load_prompt_cache_state_tensors(directory, &manifest, stream)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let mut cache =
            MlxHybridState::paged(self.state_layout.clone(), manager, self.parallel_rank)?;
        cache.restore_prompt_cache_state(
            tensors,
            i32::try_from(prefix_token_ids.len())
                .map_err(|_| Exception::custom("prompt-cache prefix exceeds i32"))?,
            &identity.layer_prefix_offsets,
        )?;
        Ok((cache, manifest))
    }

    /// Executes embedding, the scheduled physical blocks, normalization, and logits.
    pub fn forward(
        &mut self,
        tokens: &Array,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if let Some(expert_cache) = self.expert_cache.take() {
            let args = self.args.clone();
            let result = {
                let mut provider = cached_provider(&expert_cache, &args);
                self.forward_with_provider(tokens, None, cache, &mut provider, stream)
            };
            self.expert_cache = Some(expert_cache);
            return result;
        }
        let input = eredu_architectures::decoder::LayeredInput {
            tokens: crate::composition::tensor_ref(tokens),
            mask: None,
        };
        let output = match &mut self.execution {
            KimiLinearExecution::Resident(runtime) => runtime.forward(input, cache, stream),
            KimiLinearExecution::Layerwise(runtime) => runtime.forward(input, cache, stream),
            KimiLinearExecution::TensorParallelResident(_)
            | KimiLinearExecution::TensorParallelLayerwise(_) => {
                return Err(Error::Parallel(
                    "tensor-parallel Kimi Linear requires collective execution".into(),
                ))
            }
        }
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(output.into_array())
    }

    fn forward_with_provider<P>(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut MlxHybridState,
        provider: &mut P,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
        P::Error: std::fmt::Display,
    {
        let pass = if tokens.dim(1) > 1 {
            eredu_runtime::ExpertPass::Prefill
        } else {
            eredu_runtime::ExpertPass::Decode
        };
        let input = eredu_architectures::decoder::LayeredInput {
            tokens: crate::composition::tensor_ref(tokens),
            mask: crate::composition::tensor_opt(mask),
        };
        let hook =
            |architecture: &mut NeutralArchitecture,
             group: usize,
             index: usize,
             block: &mut NeutralBlock,
             hidden: &crate::MlxTensor,
             state: &mut MlxHybridState,
             forward: &mut eredu_architectures::kimi_linear::ForwardContext<crate::MlxTensor>,
             context: &Stream| {
                <NeutralArchitecture as eredu_runtime::RoutedLayeredArchitecture<
                    MlxNeuralBackend,
                    MlxHybridState,
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
        let output = match &mut self.execution {
            KimiLinearExecution::Resident(runtime) => {
                runtime.forward_with_unit_executor(input, cache, stream, hook)
            }
            KimiLinearExecution::Layerwise(runtime) => {
                runtime.forward_with_unit_executor(input, cache, stream, hook)
            }
            _ => {
                return Err(Error::Parallel(
                    "tensor-parallel Kimi Linear expert cache requires collective execution".into(),
                ))
            }
        }
        .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(output.into_array())
    }

    /// Executes a replicated neutral pass with activation intervention and
    /// normalized routing observations.
    pub fn forward_with_observer(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut MlxHybridState,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error> {
        let expert_cache = self.expert_cache.take();
        let result = {
            let mut observer = crate::composition::NeutralActivationObserver::new(observer);
            match expert_cache.as_ref() {
                Some(expert_cache) => {
                    let args = self.args.clone();
                    let mut provider = cached_provider(expert_cache, &args);
                    self.forward_observed_with_provider(
                        tokens,
                        mask,
                        cache,
                        &mut provider,
                        stream,
                        &mut observer,
                    )
                }
                None => {
                    let mut provider = eredu_runtime::ResidentExpertProvider;
                    self.forward_observed_with_provider(
                        tokens,
                        mask,
                        cache,
                        &mut provider,
                        stream,
                        &mut observer,
                    )
                }
            }
        };
        self.expert_cache = expert_cache;
        result
    }

    fn forward_observed_with_provider<P>(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut MlxHybridState,
        provider: &mut P,
        stream: &Stream,
        observer: &mut crate::composition::NeutralActivationObserver<'_>,
    ) -> Result<Array, Error>
    where
        P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
        P::Error: std::fmt::Display,
    {
        let pass = if tokens.dim(1) > 1 {
            eredu_runtime::ExpertPass::Prefill
        } else {
            eredu_runtime::ExpertPass::Decode
        };
        let output = match &mut self.execution {
            KimiLinearExecution::Resident(runtime) => runtime.forward_with_provider_and_observer(
                eredu_architectures::decoder::LayeredInput {
                    tokens: crate::composition::tensor_ref(tokens),
                    mask: crate::composition::tensor_opt(mask),
                },
                cache,
                pass,
                provider,
                stream,
                observer,
            ),
            KimiLinearExecution::Layerwise(runtime) => runtime.forward_with_provider_and_observer(
                eredu_architectures::decoder::LayeredInput {
                    tokens: crate::composition::tensor_ref(tokens),
                    mask: crate::composition::tensor_opt(mask),
                },
                cache,
                pass,
                provider,
                stream,
                observer,
            ),
            _ => {
                return Err(Error::Parallel(
                    "tensor-parallel Kimi Linear observation requires distributed observation"
                        .into(),
                ))
            }
        }
        .map_err(|error| Error::Parallel(error.to_string()))?;
        eredu_runtime::observe_and_intervene(
            observer,
            eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
            &output,
        )
        .map(crate::MlxTensor::into_array)
        .map_err(Into::into)
    }

    /// Runs the neutral decoder while delegating routed experts to an
    /// external placement policy such as expert parallelism.
    pub fn forward_with_expert_executor<F>(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut MlxHybridState,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut provider = ExpertExecutorProvider::new(&mut execute);
        self.forward_with_provider(tokens, mask, cache, &mut provider, stream)
    }

    /// Runs TP-sharded neutral blocks and dense operators while delegating
    /// routed experts to the matching expert-parallel subgroup.
    pub fn forward_tensor_expert_parallel<F>(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut MlxHybridState,
        group: &safemlx::distributed::Group,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let pass = if tokens.dim(1) > 1 {
            eredu_runtime::ExpertPass::Prefill
        } else {
            eredu_runtime::ExpertPass::Decode
        };
        let input = eredu_architectures::decoder::LayeredInput {
            tokens: crate::composition::tensor_ref(tokens),
            mask: crate::composition::tensor_opt(mask),
        };
        let mut provider = ExpertExecutorProvider::new(&mut execute);
        let hook =
            |architecture: &mut NeutralArchitecture,
             group_index: usize,
             index: usize,
             block: &mut NeutralBlock,
             hidden: &crate::MlxTensor,
             state: &mut MlxHybridState,
             forward: &mut eredu_architectures::kimi_linear::ForwardContext<crate::MlxTensor>,
             parallel: &safemlx::distributed::Group,
             context: &Stream| {
                <NeutralArchitecture as eredu_runtime::ParallelRoutedLayeredArchitecture<
                    MlxNeuralBackend,
                    MlxHybridState,
                >>::forward_unit_parallel_with_provider(
                    architecture,
                    group_index,
                    index,
                    block,
                    hidden,
                    state,
                    forward,
                    pass,
                    &mut provider,
                    parallel,
                    context,
                )
            };
        let output = match &mut self.execution {
            KimiLinearExecution::TensorParallelResident(runtime) => {
                runtime.forward_parallel_with_unit_executor(input, cache, group, stream, hook)
            }
            KimiLinearExecution::TensorParallelLayerwise(runtime) => {
                runtime.forward_parallel_with_unit_executor(input, cache, group, stream, hook)
            }
            _ => {
                return Err(Error::Parallel(
                    "Kimi Linear was not loaded for tensor plus expert parallelism".into(),
                ))
            }
        }
        .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(output.into_array())
    }

    /// Clears transient decoder weights for bounded execution.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        match &self.execution {
            KimiLinearExecution::Resident(_) => Ok(()),
            KimiLinearExecution::Layerwise(runtime) => {
                let group = crate::composition::architecture_group_name::<_, MlxHybridState>(
                    runtime.architecture(),
                    eredu_runtime::ArchitectureGroupKind::Decoder,
                )?;
                runtime.policy().clear_device_group(&group)
            }
            KimiLinearExecution::TensorParallelResident(_) => Ok(()),
            KimiLinearExecution::TensorParallelLayerwise(runtime) => {
                let group = crate::composition::architecture_group_name::<_, MlxHybridState>(
                    runtime.architecture(),
                    eredu_runtime::ArchitectureGroupKind::Decoder,
                )?;
                runtime.policy().clear_device_group(&group)
            }
        }
    }

    /// Executes a rank-local tensor-parallel forward pass.
    pub fn forward_tensor_parallel(
        &mut self,
        tokens: &Array,
        cache: &mut MlxHybridState,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let input = eredu_architectures::decoder::LayeredInput {
            tokens: crate::composition::tensor_ref(tokens),
            mask: None,
        };
        let output = match &mut self.execution {
            KimiLinearExecution::TensorParallelResident(runtime) => {
                runtime.forward_parallel(input, cache, group, stream)
            }
            KimiLinearExecution::TensorParallelLayerwise(runtime) => {
                runtime.forward_parallel(input, cache, group, stream)
            }
            _ => {
                return Err(Error::Parallel(
                    "Kimi Linear was not loaded for tensor parallelism".into(),
                ))
            }
        }
        .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(output.into_array())
    }

    pub fn forward_tensor_parallel_with_observer(
        &mut self,
        tokens: &Array,
        cache: &mut MlxHybridState,
        group: &safemlx::distributed::Group,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error> {
        let input = eredu_architectures::decoder::LayeredInput {
            tokens: crate::composition::tensor_ref(tokens),
            mask: None,
        };
        let mut neutral = crate::composition::NeutralActivationObserver::new(observer);
        let output =
            match &mut self.execution {
                KimiLinearExecution::TensorParallelResident(runtime) => runtime
                    .forward_parallel_with_observer(input, cache, group, stream, &mut neutral),
                KimiLinearExecution::TensorParallelLayerwise(runtime) => runtime
                    .forward_parallel_with_observer(input, cache, group, stream, &mut neutral),
                _ => {
                    return Err(Error::Parallel(
                        "Kimi Linear was not loaded for tensor parallelism".into(),
                    ))
                }
            }
            .map_err(|error| Error::Parallel(error.to_string()))?;
        eredu_runtime::observe_and_intervene(
            &mut neutral,
            eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
            &output,
        )
        .map(crate::MlxTensor::into_array)
        .map_err(Into::into)
    }
}

impl CausalModel<MlxHybridState> for KimiLinearModel {
    type Tensor = crate::MlxTensor;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        self.forward(&tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
            .map(crate::MlxTensor::from_array)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &crate::MlxTensor,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Exception> {
        self.forward(input_tokens.as_array(), cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
            .map(crate::MlxTensor::from_array)
    }
}

/// Loads SafeTensors Kimi Linear through one neutral model object.
pub fn load_kimi_linear_model(
    artifact: &crate::composition::mlx::artifact::PreparedSafetensorsArtifact,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<KimiLinearModel, Error> {
    let expert_options = residency.expert_cache();
    let options = residency.layers();
    let eredu_architectures::configuration::SafetensorsModelConfig::KimiLinear(args) =
        artifact.model()
    else {
        return Err(Error::ArchitectureModel(
            "Kimi Linear loader received a different prepared architecture".into(),
        ));
    };
    let args = args.clone();
    let quantize = quantization
        .map(|requested| {
            should_quantize_on_load("Kimi Linear", args.weight_quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let store = artifact.store();
    if let Some(quantization) = quantize {
        let (store, target, report) = quantize_store(store, &args, quantization, stream)?;
        let mut model = load_neutral(
            store,
            target,
            options,
            stream,
            weights_stream,
            Some(report),
            expert_options.is_some(),
        )?;
        if let Some(expert_options) = expert_options {
            attach_expert_cache(&mut model, expert_options, stream, weights_stream)?;
        }
        return Ok(model);
    }
    let mut model = load_neutral(
        store,
        args,
        options,
        stream,
        weights_stream,
        None,
        expert_options.is_some(),
    )?;
    if let Some(expert_options) = expert_options {
        attach_expert_cache(&mut model, expert_options, stream, weights_stream)?;
    }
    Ok(model)
}

fn attach_expert_cache(
    model: &mut KimiLinearModel,
    options: eredu_runtime::ExpertCacheLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(), Error> {
    let store = model.checkpoint_store_arc();
    let entries = expert_catalog(&model.args, store.as_ref())?;
    model.expert_cache = Some(ExpertCache::new_shared(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(())
}

/// Loads SafeTensors Kimi Linear through generalized tensor-parallel placement.
pub fn load_kimi_linear_tensor_parallel_model(
    artifact: &crate::composition::mlx::artifact::PreparedSafetensorsArtifact,
    options: impl Into<LayerWeightResidency>,
    build: crate::backend::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<KimiLinearModel, Error> {
    let options = options.into();
    let eredu_architectures::configuration::SafetensorsModelConfig::KimiLinear(args) =
        artifact.model()
    else {
        return Err(Error::ArchitectureModel(
            "Kimi Linear loader received a different prepared architecture".into(),
        ));
    };
    let args = args.clone();
    let store = artifact.store();
    load_neutral_parallel(store, args, options, build, stream, weights_stream, false)
}

pub(crate) struct PreparedGguf {
    pub args: ModelArgs,
}

pub(crate) fn prepare_gguf(
    source: &crate::composition::mlx::structural::AdmittedGguf,
) -> Result<PreparedGguf, Error> {
    if source.architecture() != eredu_architectures::GgufArchitecture::KimiLinear {
        return Err(Error::ArchitectureModel(format!(
            "Kimi Linear GGUF loader received architecture {:?}",
            source.architecture()
        )));
    }
    let checkpoint = source.checkpoint();
    let eredu_architectures::configuration::GgufModelConfig::KimiLinear(args) = source.model()
    else {
        return Err(Error::ArchitectureModel(
            "Kimi Linear GGUF loader received a different prepared model".into(),
        ));
    };
    let translate = eredu_architectures::kimi_linear::translate_gguf_weight_name;
    let configs = gguf_quantization_configs(checkpoint, translate)?;
    let args = eredu_architectures::kimi_linear::with_checkpoint_formats(args, configs)
        .map_err(Error::ArchitectureModel)?;
    Ok(PreparedGguf { args })
}

/// Loads a GGUF checkpoint through the same neutral Kimi Linear model object.
pub(crate) fn load_kimi_linear_gguf_model(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<KimiLinearModel, Error> {
    let checkpoint = source.checkpoint();
    let prepared = prepare_gguf(source)?;
    let expert_options = residency.expert_cache();
    let store: Arc<dyn CheckpointSource> = Arc::new(open_gguf_checkpoint_source(
        checkpoint.clone(),
        source.plan().checkpoint(),
        eredu_architectures::kimi_linear::translate_gguf_weight_name,
        residency.max_mapped_shards(),
    )?);
    let (store, args, materialization) = match quantization {
        Some(quantization) => {
            let (store, args, report) =
                quantize_store(store, &prepared.args, quantization, stream)?;
            (store, args, Some(report))
        }
        None => (store, prepared.args, None),
    };
    let mut model = load_neutral(
        store,
        args,
        residency.layers(),
        stream,
        weights_stream,
        materialization,
        expert_options.is_some(),
    )?;
    if let Some(expert_options) = expert_options {
        attach_expert_cache(&mut model, expert_options, stream, weights_stream)?;
    }
    Ok(model)
}

/// Loads GGUF Kimi Linear with tensor-parallel placement.
pub(crate) fn load_kimi_linear_gguf_tensor_parallel_model(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    options: LayerWeightResidency,
    build: crate::backend::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<KimiLinearModel, Error> {
    let checkpoint = source.checkpoint();
    let prepared = prepare_gguf(source)?;
    let store: Arc<dyn CheckpointSource> = Arc::new(open_gguf_checkpoint_source(
        checkpoint.clone(),
        source.plan().checkpoint(),
        eredu_architectures::kimi_linear::translate_gguf_weight_name,
        options.max_mapped_shards(),
    )?);
    let model = load_neutral_parallel(
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

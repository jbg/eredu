//! Neutral Nemotron-H/Nemotron-H composition over MLX execution policies.

use std::{path::Path, sync::Arc};

use eredu_architectures::nemotron_h::{LayeredModel, ModelArgs, Unit, PREDICTION_STATE_SEGMENT};
use eredu_checkpoint::{store::CheckpointSource, WeightQuantization};
use eredu_runtime::{
    ActivationObserver, ArchitectureParameters, CacheResidencyPolicy, CausalModel,
    DenseDiskStreamReport, LayerWeightResidency, LayeredArchitecture, LayerwiseModelMetadata,
    LayerwiseRuntime, PagedCacheOptions, ParallelModelInfo, ParameterRole, ResidencyReport,
    StaticUnitBindings, WeightBinding, WeightResidency,
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
                binding_bytes, build_module_bindings_with_recipes,
                build_module_bindings_with_recipes_excluding, parameter_name_in_targets,
                parameter_role_targets, populate_module_from_lease_excluding,
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
            layerwise::{quantize_module_store_with_bindings, shard_layer_bindings},
        },
        media::input,
        residency::expert_cache::ExpertCatalogEntry,
        residency::expert_cache::{ExpertCache, ExpertCacheReport},
        residency::expert_provider::{CachedRelu2ExpertProvider, ExpertExecutorProvider},
    },
};
use eredu_core::cache::{
    PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
    PromptCacheTopology,
};

type NeutralBlock = Unit<MlxNeuralBackend>;
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
    MlxLayerwisePolicy<NeutralBlock, NemotronHUnitPopulator>,
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
    MlxLayerwisePolicy<NeutralBlock, NemotronHParallelUnitPopulator>,
>;

#[derive(eredu_nn::Parameterized)]
#[parameterized(tensor = "crate::MlxTensor")]
#[doc(hidden)]
#[cfg(test)]
pub struct NemotronHCheckpointTemplate {
    pub static_modules: eredu_architectures::decoder::StaticModules<MlxNeuralBackend>,
    pub layers: Vec<NeutralBlock>,
}

#[cfg(test)]
impl NemotronHCheckpointTemplate {
    /// Builds one neutral full-parameter template for checkpoint tooling.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let architecture = NeutralArchitecture::new(args.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let layout = architecture_execution_layout::<_, MlxHybridState>(&architecture)?;
        let layers = (0..layout.len())
            .map(|index| {
                construct_architecture_unit(
                    &architecture,
                    &layout,
                    index,
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

struct NeutralNemotronHObserver<'a> {
    inner: &'a mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
}

impl eredu_runtime::ActivationObserver<crate::MlxTensor, eredu_nn::Error>
    for NeutralNemotronHObserver<'_>
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

fn neutral_embedded_input<'a>(
    input: eredu_architectures::nemotron_h::EmbeddedInput<'a, Array>,
) -> eredu_architectures::nemotron_h::EmbeddedInput<'a, crate::MlxTensor> {
    match input {
        eredu_architectures::nemotron_h::EmbeddedInput::Target { tokens, mask } => {
            eredu_architectures::nemotron_h::EmbeddedInput::target(
                crate::composition::tensor_ref(tokens),
                crate::composition::tensor_opt(mask),
            )
        }
        eredu_architectures::nemotron_h::EmbeddedInput::Draft {
            tokens,
            hidden,
            depth,
        } => eredu_architectures::nemotron_h::EmbeddedInput::draft(
            crate::composition::tensor_ref(tokens),
            crate::composition::tensor_ref(hidden),
            depth,
        ),
    }
}

#[derive(Clone)]
struct NemotronHUnitPopulator {
    external_experts: bool,
    expert_targets: Arc<std::collections::BTreeSet<String>>,
}

/// Pipeline/loading adapter over the same neutral Nemotron-H blocks used by resident
/// and bounded execution.
#[derive(Default)]
pub struct NemotronHBindings {
    external_experts: bool,
}

impl NemotronHBindings {
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
        store: &dyn CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        crate::composition::architecture_static_units(architecture, store)
    }

    pub fn layer_bindings(
        &self,
        architecture: &NeutralArchitecture,
        group: usize,
        index: usize,
        layer: &MlxModule<NeutralBlock>,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        let flat = architecture_execution_layout::<_, MlxHybridState>(architecture)?
            .ordinal(group, index)
            .ok_or_else(|| {
                Error::Parallel(format!("Nemotron-H has no unit {index} in group {group}"))
            })?;
        let expert_targets = parameter_role_targets(
            &eredu_architectures::nemotron_h::unit_parallel_parameter_groups(
                layer,
                architecture.args(),
                flat,
            )?,
            ParameterRole::ExpertIntermediate,
        );
        let recipes = eredu_architectures::nemotron_h::unit_recipes(
            store,
            architecture.args(),
            group,
            index,
            !self.external_experts,
        )
        .map_err(Error::ArchitectureModel)?;
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
        architecture: &NeutralArchitecture,
        topology: crate::backend::MlxParallelContext,
    ) -> Result<Option<crate::backend::runtime::distributed::expert::ExpertAssignment>, Error> {
        if topology.expert_parallel_size == 1 && !self.external_experts {
            return Ok(None);
        }
        let args = architecture.args();
        if !args.has_sparse_moe_layers() {
            return Err(Error::Parallel(
                "Nemotron-H PP+EP requires a sparse-MoE checkpoint".into(),
            ));
        }
        Ok(Some(
            crate::backend::runtime::distributed::expert::ExpertAssignment::balanced(
                args.n_routed_experts as usize,
                topology.expert_parallel_size,
                topology.expert_parallel_rank,
            )?,
        ))
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
        let bindings = self.layer_bindings(architecture, group, index, global_layer, store)?;
        match layout {
            Some(layout) => {
                let root = <NeutralArchitecture as eredu_runtime::LayeredArchitecture<
                    MlxNeuralBackend,
                    MlxHybridState,
                >>::unit_path(architecture, group, index)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                shard_layer_bindings(bindings, &root, store, layout)
            }
            None => Ok(bindings),
        }
    }
}

impl MlxUnitPopulator<NeutralBlock> for NemotronHUnitPopulator {
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
struct NemotronHParallelUnitPopulator {
    external_experts: bool,
    expert_targets: Arc<std::collections::BTreeSet<String>>,
}

impl MlxUnitPopulator<NeutralBlock> for NemotronHParallelUnitPopulator {
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

enum NemotronHExecution {
    Resident(Box<ResidentRuntime>),
    Layerwise(Box<BoundedRuntime>),
    TensorParallelResident(Box<ParallelResidentRuntime>),
    TensorParallelLayerwise(Box<ParallelBoundedRuntime>),
}

pub fn expert_catalog(
    args: &ModelArgs,
    store: &dyn CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    expert_catalog_selected(args, store, |_, _| true)
}

/// Canonical independent-expert catalog for selected architecture-owned units.
pub fn expert_catalog_selected(
    args: &ModelArgs,
    store: &dyn CheckpointSource,
    owns_unit: impl FnMut(&eredu_runtime::ExecutionGroupId, usize) -> bool,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let catalog = eredu_architectures::nemotron_h::expert_residency_catalog(store, args)
        .map_err(Error::ArchitectureModel)?;
    let units = catalog.into_units_selected_by_owner(owns_unit);
    crate::composition::architecture_expert_units(units, store, None)
}

fn cached_provider<'a>(
    cache: &'a ExpertCache,
    args: &'a ModelArgs,
) -> CachedRelu2ExpertProvider<
    'a,
    impl FnMut(usize) -> Result<eredu_nn::Relu2ExpertBankSpec, Error> + 'a,
> {
    CachedRelu2ExpertProvider::new(cache, move |layer| {
        eredu_architectures::nemotron_h::expert_bank_spec(args, layer).map_err(Error::from)
    })
}

fn load_neutral(
    store: Arc<dyn CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
    materialization: Option<eredu_runtime::WeightMaterializationReport>,
    external_experts: bool,
) -> Result<NemotronHModel, Error> {
    let mut architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let expert_targets = Arc::new(
        architecture
            .parameter_description(stream)
            .map_err(|error| Error::Parallel(error.to_string()))?
            .targets_for_role(ParameterRole::ExpertIntermediate),
    );
    let static_binding_args = args.clone();
    let binding_args = args.clone();
    let excluded_expert_targets = Arc::clone(&expert_targets);
    let binding_expert_targets = Arc::clone(&expert_targets);
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        store,
        &mut architecture,
        NemotronHUnitPopulator {
            external_experts,
            expert_targets: Arc::clone(&expert_targets),
        },
        std::marker::PhantomData::<MlxHybridState>,
        options,
        stream,
        weights_stream,
        move |key| external_experts && parameter_name_in_targets(key, &excluded_expert_targets),
        move |modules, store| {
            build_module_bindings_with_recipes(
                &MlxModule::new(modules.clone()),
                "",
                store,
                eredu_architectures::nemotron_h::static_recipes(store, &static_binding_args, None)
                    .map_err(Error::ArchitectureModel)?,
            )
            .map_err(Into::into)
        },
        move |_ordinal, address, _path, unit, store, _| {
            build_module_bindings_with_recipes_excluding(
                &MlxModule::new(unit),
                "",
                store,
                eredu_architectures::nemotron_h::unit_recipes(
                    store,
                    &binding_args,
                    address.group(),
                    address.index(),
                    !external_experts,
                )
                .map_err(Error::ArchitectureModel)?,
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
        NemotronHExecution::Resident(Box::new(LayerwiseRuntime::new_policy_first(
            policy.into_resident(
                &architecture,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )?,
            architecture,
        )))
    } else {
        NemotronHExecution::Layerwise(Box::new(LayerwiseRuntime::new(architecture, policy)))
    };
    Ok(NemotronHModel {
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
) -> Result<NemotronHModel, Error> {
    let global_architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let global_execution =
        architecture_execution_layout::<_, MlxHybridState>(&global_architecture)?;
    let expert_targets = Arc::new(
        global_architecture
            .parameter_description(stream)
            .map_err(|error| Error::Parallel(error.to_string()))?
            .targets_for_role(ParameterRole::ExpertIntermediate),
    );
    let mut planner = build.planner();
    for group in eredu_architectures::nemotron_h::static_parallel_parameter_groups::<
        MlxNeuralBackend,
    >(global_architecture.static_modules())?
    {
        planner.register(group)?;
    }
    for ordinal in 0..global_execution.len() {
        let unit = construct_architecture_unit(
            &global_architecture,
            &global_execution,
            ordinal,
            stream,
            std::marker::PhantomData::<MlxHybridState>,
        )?;
        for group in
            eredu_architectures::nemotron_h::unit_parallel_parameter_groups(&unit, &args, ordinal)?
        {
            planner.register(group)?;
        }
    }
    let (_, layout) = planner.finish()?;
    if layout.is_empty() {
        return Err(Error::Parallel(
            "Nemotron-H declared no tensor-parallel parameters".into(),
        ));
    }
    let geometry = eredu_architectures::nemotron_h::local_geometry(&args, &layout)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let mut architecture = NeutralArchitecture::new_parallel(args.clone(), geometry, stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let state_layout = architecture
        .state_layout()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let factory = NemotronHParallelUnitPopulator {
        external_experts,
        expert_targets: Arc::clone(&expert_targets),
    };

    let global_static = MlxModule::new(global_architecture.static_modules().clone());
    let global_static_bindings = build_module_bindings_with_recipes(
        &global_static,
        "",
        store.as_ref(),
        eredu_architectures::nemotron_h::static_recipes(store.as_ref(), &args, None)
            .map_err(Error::ArchitectureModel)?,
    )?;
    let mut global_parameter_bytes = binding_bytes(&global_static_bindings)?;
    for ordinal in 0..global_execution.len() {
        let unit = construct_architecture_unit(
            &global_architecture,
            &global_execution,
            ordinal,
            stream,
            std::marker::PhantomData::<MlxHybridState>,
        )?;
        let address = global_execution
            .address(ordinal)
            .expect("architecture execution layout contains every ordinal");
        let bindings = build_module_bindings_with_recipes_excluding(
            &MlxModule::new(unit),
            "",
            store.as_ref(),
            eredu_architectures::nemotron_h::unit_recipes(
                store.as_ref(),
                &args,
                address.group(),
                address.index(),
                !external_experts,
            )
            .map_err(Error::ArchitectureModel)?,
            |name| external_experts && parameter_name_in_targets(name, &expert_targets),
        )?;
        global_parameter_bytes = global_parameter_bytes
            .checked_add(binding_bytes(&bindings)?)
            .ok_or_else(|| {
                Error::Parallel("global Nemotron-H parameter bytes overflowed".into())
            })?;
    }

    let shared_layout = Arc::new(layout);
    let static_layout = Arc::clone(&shared_layout);
    let unit_layout = Arc::clone(&shared_layout);
    let static_binding_args = args.clone();
    let binding_args = args.clone();
    let global_static_modules = global_architecture.static_modules().clone();
    let binding_architecture = global_architecture;
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
            let bindings = build_module_bindings_with_recipes(
                &global,
                "",
                store,
                eredu_architectures::nemotron_h::static_recipes(store, &static_binding_args, None)
                    .map_err(Error::ArchitectureModel)?,
            )?;
            shard_layer_bindings(bindings, "", store, &static_layout)
        },
        move |_ordinal, address, path, _local, store, stream| {
            let global = <NeutralArchitecture as LayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::build_unit(
                &binding_architecture,
                address.group(),
                address.index(),
                stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let bindings = build_module_bindings_with_recipes_excluding(
                &MlxModule::new(global),
                "",
                store,
                eredu_architectures::nemotron_h::unit_recipes(
                    store,
                    &binding_args,
                    address.group(),
                    address.index(),
                    !external_experts,
                )
                .map_err(Error::ArchitectureModel)?,
                |name| external_experts && parameter_name_in_targets(name, &binding_expert_targets),
            )?;
            shard_layer_bindings(bindings, path, store, &unit_layout)
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.weight_quantization);
    let local_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.layer_parameter_bytes())
        .ok_or_else(|| Error::Parallel("local Nemotron-H parameter bytes overflowed".into()))?;
    let maximum_device_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.maximum_device_layer_bytes())
        .ok_or_else(|| Error::Parallel("device Nemotron-H parameter bytes overflowed".into()))?;
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
        NemotronHExecution::TensorParallelResident(Box::new(LayerwiseRuntime::new_policy_first(
            policy.into_resident(
                &architecture,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )?,
            architecture,
        )))
    } else {
        NemotronHExecution::TensorParallelLayerwise(Box::new(LayerwiseRuntime::new(
            architecture,
            policy,
        )))
    };
    Ok(NemotronHModel {
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
    let target = eredu_architectures::nemotron_h::load_time_quantization(args, quantization)
        .map_err(Error::ArchitectureModel)?;
    let source = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let destination = NeutralArchitecture::new(target.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let source_layout = architecture_execution_layout::<_, MlxHybridState>(&source)?;
    let target_layout = architecture_execution_layout::<_, MlxHybridState>(&destination)?;
    if source_layout != target_layout {
        return Err(Error::Quantization(
            "Nemotron-H quantization changed the architecture execution layout".into(),
        ));
    }
    let unit_count = source_layout.len();
    let binding_layout = source_layout.clone();
    let binding_args = args.clone();
    let static_binding_args = args.clone();
    let source_static = MlxModule::new(source.static_modules().clone());
    let target_static = MlxModule::new(destination.static_modules().clone());
    let (store, report) = quantize_module_store_with_bindings(
        store,
        &source_static,
        &target_static,
        move |index, stream| {
            construct_architecture_unit(
                &source,
                &source_layout,
                index,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )
            .map(MlxModule::new)
        },
        move |index, stream| {
            construct_architecture_unit(
                &destination,
                &target_layout,
                index,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )
            .map(MlxModule::new)
        },
        unit_count,
        quantization,
        stream,
        move |modules, store| {
            build_module_bindings_with_recipes(
                modules,
                "",
                store,
                eredu_architectures::nemotron_h::static_recipes(store, &static_binding_args, None)
                    .map_err(Error::ArchitectureModel)?,
            )
            .map_err(Into::into)
        },
        move |index, unit, store| {
            let address = binding_layout.address(index).ok_or_else(|| {
                Error::ArchitectureModel(format!(
                    "Nemotron-H has no execution unit at ordinal {index}"
                ))
            })?;
            build_module_bindings_with_recipes(
                unit,
                "",
                store,
                eredu_architectures::nemotron_h::unit_recipes(
                    store,
                    &binding_args,
                    address.group(),
                    address.index(),
                    true,
                )
                .map_err(Error::ArchitectureModel)?,
            )
            .map_err(Into::into)
        },
    )?;
    Ok((store, target, report))
}

/// Nemotron-H causal model whose equations are owned by `eredu-architectures`.
pub struct NemotronHModel {
    args: ModelArgs,
    state_layout: eredu_runtime::StateLayout,
    metadata: LayerwiseModelMetadata,
    execution: NemotronHExecution,
    expert_cache: Option<ExpertCache>,
    parallel_info: Option<ParallelModelInfo<crate::backend::MlxParallelContext>>,
    parallel_rank: Option<eredu_core::cache::CacheRankIdentity>,
}

/// Collective context adapter for the same neutral embedded-MTP target.
pub struct NemotronHTensorMtpTarget<'a> {
    model: &'a mut NemotronHModel,
    group: &'a safemlx::distributed::Group,
}

impl<'a> NemotronHTensorMtpTarget<'a> {
    pub const fn new(
        model: &'a mut NemotronHModel,
        group: &'a safemlx::distributed::Group,
    ) -> Self {
        Self { model, group }
    }
}

impl NemotronHModel {
    /// Returns validated family policy.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Returns the number of embedded prediction depths.
    pub fn mtp_len(&self) -> usize {
        usize::try_from(self.args.num_nextn_predict_layers).unwrap_or(0)
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
            .expect("validated Nemotron-H state must be realizable by MLX")
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
            NemotronHExecution::Resident(runtime) => runtime.policy().residency_report(),
            NemotronHExecution::Layerwise(runtime) => runtime.policy().residency_report(),
            NemotronHExecution::TensorParallelResident(runtime) => {
                runtime.policy().residency_report()
            }
            NemotronHExecution::TensorParallelLayerwise(runtime) => {
                runtime.policy().residency_report()
            }
        }
    }

    /// Returns disk streaming telemetry when enabled.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        match &self.execution {
            NemotronHExecution::Resident(_) => Ok(None),
            NemotronHExecution::Layerwise(runtime) => runtime.policy().dense_stream_report(),
            NemotronHExecution::TensorParallelResident(_) => Ok(None),
            NemotronHExecution::TensorParallelLayerwise(runtime) => {
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
            NemotronHExecution::Resident(runtime) => runtime.policy().checkpoint_store(),
            NemotronHExecution::Layerwise(runtime) => runtime.policy().checkpoint_store(),
            NemotronHExecution::TensorParallelResident(runtime) => {
                runtime.policy().checkpoint_store()
            }
            NemotronHExecution::TensorParallelLayerwise(runtime) => {
                runtime.policy().checkpoint_store()
            }
        }
    }

    pub fn checkpoint_store_arc(&self) -> Arc<dyn CheckpointSource> {
        match &self.execution {
            NemotronHExecution::Resident(runtime) => runtime.policy().checkpoint_store_arc(),
            NemotronHExecution::Layerwise(runtime) => runtime.policy().checkpoint_store_arc(),
            NemotronHExecution::TensorParallelResident(runtime) => {
                runtime.policy().checkpoint_store_arc()
            }
            NemotronHExecution::TensorParallelLayerwise(runtime) => {
                runtime.policy().checkpoint_store_arc()
            }
        }
    }

    /// Returns the canonical prompt-cache fingerprint.
    pub fn prompt_cache_architecture_fingerprint(&self) -> Result<String, Error> {
        Ok(eredu_architectures::nemotron_h::prompt_cache_architecture_fingerprint(&self.args))
    }

    /// Returns the exact rank-local state layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<eredu_core::LayerSchedule<eredu_core::cache::LayerCachePolicy>, Error> {
        Ok(self.state_layout.layers().clone())
    }

    /// Returns the canonical prompt-cache identity.
    pub fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        eredu_architectures::nemotron_h::state_identity(
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
        let input = eredu_architectures::nemotron_h::EmbeddedInput::target(
            crate::composition::tensor_ref(tokens),
            None,
        );
        match &mut self.execution {
            NemotronHExecution::Resident(runtime) => runtime.forward(input, cache, stream),
            NemotronHExecution::Layerwise(runtime) => runtime.forward(input, cache, stream),
            NemotronHExecution::TensorParallelResident(_)
            | NemotronHExecution::TensorParallelLayerwise(_) => {
                return Err(Error::Parallel(
                    "tensor-parallel Nemotron-H requires collective execution".into(),
                ))
            }
        }
        .map(crate::MlxTensor::into_array)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
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
        self.forward_with_provider_context(tokens, mask, cache, provider, stream)
            .map(|(output, _)| output.into_array())
    }

    fn forward_with_provider_context<P>(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut MlxHybridState,
        provider: &mut P,
        stream: &Stream,
    ) -> Result<
        (
            crate::MlxTensor,
            eredu_architectures::nemotron_h::ForwardContext<crate::MlxTensor>,
        ),
        Error,
    >
    where
        P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
        P::Error: std::fmt::Display,
    {
        let input = eredu_architectures::nemotron_h::EmbeddedInput::target(
            crate::composition::tensor_ref(tokens),
            crate::composition::tensor_opt(mask),
        );
        let hook =
            |architecture: &mut NeutralArchitecture,
             group: usize,
             index: usize,
             block: &mut NeutralBlock,
             hidden: &crate::MlxTensor,
             state: &mut MlxHybridState,
             forward: &mut eredu_architectures::nemotron_h::ForwardContext<crate::MlxTensor>,
             context: &Stream| {
                architecture.forward_unit_with_provider(
                    group, index, block, hidden, state, forward, provider, context,
                )
            };
        match &mut self.execution {
            NemotronHExecution::Resident(runtime) => runtime
                .forward_with_unit_executor_and_context_hook(
                    input,
                    cache,
                    stream,
                    hook,
                    |_, _, _| Ok(()),
                ),
            NemotronHExecution::Layerwise(runtime) => runtime
                .forward_with_unit_executor_and_context_hook(
                    input,
                    cache,
                    stream,
                    hook,
                    |_, _, _| Ok(()),
                ),
            _ => {
                return Err(Error::Parallel(
                    "tensor-parallel Nemotron-H expert cache requires collective execution".into(),
                ))
            }
        }
        .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn forward_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        let expert_cache = self.expert_cache.take();
        let result = match expert_cache.as_ref() {
            Some(cache_store) => {
                let args = self.args.clone();
                let mut provider = cached_provider(cache_store, &args);
                self.forward_with_provider_context(tokens, None, cache, &mut provider, stream)
            }
            None => self.forward_with_provider_context(
                tokens,
                None,
                cache,
                &mut eredu_runtime::ResidentExpertProvider,
                stream,
            ),
        };
        self.expert_cache = expert_cache;
        let (logits, context) = result.map_err(|error| Exception::custom(error.to_string()))?;
        let hidden = context
            .target_capture()
            .cloned()
            .ok_or_else(|| Exception::custom("Nemotron-H target pass retained no hidden state"))?;
        Ok(
            crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits,
                hidden,
                tokens: crate::MlxTensor::from_array(tokens.clone()),
            },
        )
    }

    fn forward_mtp_draft(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        let input = eredu_architectures::nemotron_h::EmbeddedInput::draft(
            crate::composition::tensor_ref(tokens),
            crate::composition::tensor_ref(hidden),
            depth,
        );
        let expert_cache = self.expert_cache.take();
        let result = match expert_cache.as_ref() {
            Some(cache_store) => {
                let args = self.args.clone();
                let mut provider = cached_provider(cache_store, &args);
                self.forward_input_with_provider_context(input, cache, &mut provider, stream)
            }
            None => self.forward_input_with_provider_context(
                input,
                cache,
                &mut eredu_runtime::ResidentExpertProvider,
                stream,
            ),
        };
        self.expert_cache = expert_cache;
        let (logits, context) = result.map_err(|error| Exception::custom(error.to_string()))?;
        let hidden = context
            .target_capture()
            .cloned()
            .ok_or_else(|| Exception::custom("Nemotron-H draft pass retained no hidden state"))?;
        Ok(
            crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits,
                hidden,
                tokens: crate::MlxTensor::from_array(tokens.clone()),
            },
        )
    }

    fn forward_mtp_parallel(
        &mut self,
        input: eredu_architectures::nemotron_h::EmbeddedInput<'_, Array>,
        cache: &mut MlxHybridState,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        let tokens = match &input {
            eredu_architectures::nemotron_h::EmbeddedInput::Target { tokens, .. }
            | eredu_architectures::nemotron_h::EmbeddedInput::Draft { tokens, .. } => {
                (*tokens).clone()
            }
        };
        let input = neutral_embedded_input(input);
        let result = match &mut self.execution {
            NemotronHExecution::TensorParallelResident(runtime) => runtime
                .forward_parallel_with_context_hook(input, cache, group, stream, |_, _, _| Ok(())),
            NemotronHExecution::TensorParallelLayerwise(runtime) => runtime
                .forward_parallel_with_context_hook(input, cache, group, stream, |_, _, _| Ok(())),
            _ => {
                return Err(Exception::custom(
                    "Nemotron-H was not loaded for tensor-parallel MTP",
                ))
            }
        }
        .map_err(|error| Exception::custom(error.to_string()))?;
        let hidden = result
            .1
            .target_capture()
            .cloned()
            .ok_or_else(|| Exception::custom("Nemotron-H MTP pass retained no hidden state"))?;
        Ok(
            crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits: result.0,
                hidden,
                tokens: crate::MlxTensor::from_array(tokens),
            },
        )
    }

    fn forward_input_with_provider_context<'a, P>(
        &mut self,
        input: eredu_architectures::nemotron_h::EmbeddedInput<'a, crate::MlxTensor>,
        cache: &mut MlxHybridState,
        provider: &mut P,
        stream: &Stream,
    ) -> Result<
        (
            crate::MlxTensor,
            eredu_architectures::nemotron_h::ForwardContext<crate::MlxTensor>,
        ),
        Error,
    >
    where
        P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
        P::Error: std::fmt::Display,
    {
        let hook =
            |architecture: &mut NeutralArchitecture,
             group: usize,
             index: usize,
             unit: &mut NeutralBlock,
             hidden: &crate::MlxTensor,
             state: &mut MlxHybridState,
             forward: &mut eredu_architectures::nemotron_h::ForwardContext<crate::MlxTensor>,
             context: &Stream| {
                architecture.forward_unit_with_provider(
                    group, index, unit, hidden, state, forward, provider, context,
                )
            };
        match &mut self.execution {
            NemotronHExecution::Resident(runtime) => runtime
                .forward_with_unit_executor_and_context_hook(
                    input,
                    cache,
                    stream,
                    hook,
                    |_, _, _| Ok(()),
                ),
            NemotronHExecution::Layerwise(runtime) => runtime
                .forward_with_unit_executor_and_context_hook(
                    input,
                    cache,
                    stream,
                    hook,
                    |_, _, _| Ok(()),
                ),
            _ => {
                return Err(Error::Parallel(
                    "tensor-parallel Nemotron-H requires collective MTP execution".into(),
                ))
            }
        }
        .map_err(|error| Error::Parallel(error.to_string()))
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
            let mut observer = NeutralNemotronHObserver { inner: observer };
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
        observer: &mut NeutralNemotronHObserver<'_>,
    ) -> Result<Array, Error>
    where
        P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
        P::Error: std::fmt::Display,
    {
        let input = eredu_architectures::nemotron_h::EmbeddedInput::target(
            crate::composition::tensor_ref(tokens),
            crate::composition::tensor_opt(mask),
        );
        let hook =
            |architecture: &mut NeutralArchitecture,
             group: usize,
             index: usize,
             block: &mut NeutralBlock,
             hidden: &crate::MlxTensor,
             state: &mut MlxHybridState,
             forward: &mut eredu_architectures::nemotron_h::ForwardContext<crate::MlxTensor>,
             context: &Stream| {
                let path = <NeutralArchitecture as LayeredArchitecture<
                    MlxNeuralBackend,
                    MlxHybridState,
                >>::unit_path(architecture, group, index)?;
                observer.observe(&format!("{path}.input"), hidden)?;
                let output = architecture.forward_unit_observed_with_provider(
                    group, index, block, hidden, state, forward, observer, provider, context,
                )?;
                eredu_runtime::observe_and_intervene(observer, &format!("{path}.output"), &output)
            };
        match &mut self.execution {
            NemotronHExecution::Resident(runtime) => {
                runtime.forward_with_unit_executor(input, cache, stream, hook)
            }
            NemotronHExecution::Layerwise(runtime) => {
                runtime.forward_with_unit_executor(input, cache, stream, hook)
            }
            _ => {
                return Err(Error::Parallel(
                    "tensor-parallel Nemotron-H observation requires distributed observation"
                        .into(),
                ))
            }
        }
        .map(crate::MlxTensor::into_array)
        .map_err(|error| Error::Parallel(error.to_string()))
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
        let input = eredu_architectures::nemotron_h::EmbeddedInput::target(
            crate::composition::tensor_ref(tokens),
            crate::composition::tensor_opt(mask),
        );
        let mut provider = ExpertExecutorProvider::new(&mut execute);
        let hook =
            |architecture: &mut NeutralArchitecture,
             group_index: usize,
             index: usize,
             block: &mut NeutralBlock,
             hidden: &crate::MlxTensor,
             state: &mut MlxHybridState,
             forward: &mut eredu_architectures::nemotron_h::ForwardContext<crate::MlxTensor>,
             parallel: &safemlx::distributed::Group,
             context: &Stream| {
                architecture.forward_unit_parallel_with_provider(
                    group_index,
                    index,
                    block,
                    hidden,
                    state,
                    forward,
                    parallel,
                    &mut provider,
                    context,
                )
            };
        match &mut self.execution {
            NemotronHExecution::TensorParallelResident(runtime) => {
                runtime.forward_parallel_with_unit_executor(input, cache, group, stream, hook)
            }
            NemotronHExecution::TensorParallelLayerwise(runtime) => {
                runtime.forward_parallel_with_unit_executor(input, cache, group, stream, hook)
            }
            _ => {
                return Err(Error::Parallel(
                    "Nemotron-H was not loaded for tensor plus expert parallelism".into(),
                ))
            }
        }
        .map(crate::MlxTensor::into_array)
        .map_err(|error| Error::Parallel(error.to_string()))
    }

    pub fn forward_mtp_target_with_expert_executor<F>(
        &mut self,
        tokens: &Array,
        cache: &mut MlxHybridState,
        tensor_group: Option<&safemlx::distributed::Group>,
        mut execute: F,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        self.forward_mtp_input_with_expert_executor(
            eredu_architectures::nemotron_h::EmbeddedInput::target(tokens, None),
            cache,
            tensor_group,
            &mut execute,
            stream,
        )
    }

    pub fn forward_mtp_draft_with_expert_executor<F>(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut MlxHybridState,
        tensor_group: Option<&safemlx::distributed::Group>,
        mut execute: F,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        self.forward_mtp_input_with_expert_executor(
            eredu_architectures::nemotron_h::EmbeddedInput::draft(tokens, hidden, depth),
            cache,
            tensor_group,
            &mut execute,
            stream,
        )
    }

    fn forward_mtp_input_with_expert_executor<'a, F>(
        &mut self,
        input: eredu_architectures::nemotron_h::EmbeddedInput<'a, Array>,
        cache: &mut MlxHybridState,
        tensor_group: Option<&safemlx::distributed::Group>,
        execute: &mut F,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let tokens = match &input {
            eredu_architectures::nemotron_h::EmbeddedInput::Target { tokens, .. }
            | eredu_architectures::nemotron_h::EmbeddedInput::Draft { tokens, .. } => {
                (*tokens).clone()
            }
        };
        let input = neutral_embedded_input(input);
        let mut provider = ExpertExecutorProvider::new(execute);
        let (logits, context) = match tensor_group {
            Some(group) => {
                let hook = |architecture: &mut NeutralArchitecture,
                            group_index: usize,
                            index: usize,
                            unit: &mut NeutralBlock,
                            hidden: &crate::MlxTensor,
                            state: &mut MlxHybridState,
                            forward: &mut eredu_architectures::nemotron_h::ForwardContext<
                    crate::MlxTensor,
                >,
                            parallel: &safemlx::distributed::Group,
                            context: &Stream| {
                    architecture.forward_unit_parallel_with_provider(
                        group_index,
                        index,
                        unit,
                        hidden,
                        state,
                        forward,
                        parallel,
                        &mut provider,
                        context,
                    )
                };
                match &mut self.execution {
                    NemotronHExecution::TensorParallelResident(runtime) => runtime
                        .forward_parallel_with_unit_executor_and_context_hook(
                            input,
                            cache,
                            group,
                            stream,
                            hook,
                            |_, _, _| Ok(()),
                        ),
                    NemotronHExecution::TensorParallelLayerwise(runtime) => runtime
                        .forward_parallel_with_unit_executor_and_context_hook(
                            input,
                            cache,
                            group,
                            stream,
                            hook,
                            |_, _, _| Ok(()),
                        ),
                    _ => return Err(Exception::custom("Nemotron-H was not loaded for TP+EP MTP")),
                }
                .map_err(|error| Exception::custom(error.to_string()))?
            }
            None => self
                .forward_input_with_provider_context(input, cache, &mut provider, stream)
                .map_err(|error| Exception::custom(error.to_string()))?,
        };
        let hidden = context
            .target_capture()
            .cloned()
            .ok_or_else(|| Exception::custom("Nemotron-H MTP pass retained no hidden state"))?;
        Ok(
            crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits,
                hidden,
                tokens: crate::MlxTensor::from_array(tokens),
            },
        )
    }

    /// Clears transient decoder weights for bounded execution.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        match &self.execution {
            NemotronHExecution::Resident(_) => Ok(()),
            NemotronHExecution::Layerwise(runtime) => runtime.policy().clear_device_group("target"),
            NemotronHExecution::TensorParallelResident(_) => Ok(()),
            NemotronHExecution::TensorParallelLayerwise(runtime) => {
                runtime.policy().clear_device_group("target")
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
        let input = eredu_architectures::nemotron_h::EmbeddedInput::target(
            crate::composition::tensor_ref(tokens),
            None,
        );
        match &mut self.execution {
            NemotronHExecution::TensorParallelResident(runtime) => {
                runtime.forward_parallel(input, cache, group, stream)
            }
            NemotronHExecution::TensorParallelLayerwise(runtime) => {
                runtime.forward_parallel(input, cache, group, stream)
            }
            _ => {
                return Err(Error::Parallel(
                    "Nemotron-H was not loaded for tensor parallelism".into(),
                ))
            }
        }
        .map(crate::MlxTensor::into_array)
        .map_err(|error| Error::Parallel(error.to_string()))
    }
}

impl CausalModel<MlxHybridState> for NemotronHModel {
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

impl crate::composition::mlx::speculative::embedded::EmbeddedMtpTarget for NemotronHModel {
    type Cache = MlxHybridState;
    type DraftCache = MlxHybridState;

    fn prefill_target(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        cache.clear()?;
        self.forward_mtp_target(&tokens, cache, stream)
    }

    fn verify_target(
        &mut self,
        tokens: &crate::MlxTensor,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        self.forward_mtp_target(tokens.as_array(), cache, stream)
    }

    fn prefill_draft_cache(
        &mut self,
        output: &crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput,
        tokens: &crate::MlxTensor,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let sequence = tokens.as_array().dim(1);
        if sequence <= 1 {
            return Ok(());
        }
        let hidden = output
            .hidden
            .as_array()
            .try_index_device((.., ..sequence - 1, ..), stream)?;
        let next = tokens.as_array().try_index_device((.., 1..), stream)?;
        for depth in 0..self.mtp_len() {
            let _ = self.forward_mtp_draft(&hidden, &next, depth, cache, stream)?;
        }
        Ok(())
    }

    fn draft_cache(&self, cache: &Self::Cache) -> Self::DraftCache {
        cache
            .deep_clone_state()
            .expect("evaluated Nemotron-H draft state must be forkable")
    }

    fn commit_draft_cache(&self, cache: &mut Self::Cache, draft: &Self::DraftCache) {
        cache
            .commit_segment_from(draft, PREDICTION_STATE_SEGMENT)
            .expect("validated Nemotron-H prediction state segment");
    }

    fn restore_target_checkpoint(
        cache: &mut Self::Cache,
        checkpoint: &Self::Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        cache.restore_checkpoint(checkpoint, stream)
    }

    fn draft_logits(
        &mut self,
        hidden: &crate::MlxTensor,
        last_token: u32,
        draft_index: usize,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(crate::MlxTensor, crate::MlxTensor), Exception> {
        let token = Array::from_slice(&[last_token], &[1, 1]);
        let output =
            self.forward_mtp_draft(hidden.as_array(), &token, draft_index, cache, stream)?;
        Ok((output.logits, output.hidden))
    }

    fn advance_draft_cache(
        &mut self,
        hidden: &crate::MlxTensor,
        tokens: &crate::MlxTensor,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        for depth in 0..self.mtp_len() {
            let _ =
                self.forward_mtp_draft(hidden.as_array(), tokens.as_array(), depth, cache, stream)?;
        }
        Ok(())
    }

    fn max_draft_tokens(&self) -> usize {
        self.mtp_len()
    }
}

impl crate::composition::mlx::speculative::embedded::EmbeddedMtpTarget
    for NemotronHTensorMtpTarget<'_>
{
    type Cache = MlxHybridState;
    type DraftCache = MlxHybridState;

    fn prefill_target(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        cache.clear()?;
        self.model.forward_mtp_parallel(
            eredu_architectures::nemotron_h::EmbeddedInput::target(&tokens, None),
            cache,
            self.group,
            stream,
        )
    }

    fn verify_target(
        &mut self,
        tokens: &crate::MlxTensor,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        self.model.forward_mtp_parallel(
            eredu_architectures::nemotron_h::EmbeddedInput::target(tokens.as_array(), None),
            cache,
            self.group,
            stream,
        )
    }

    fn prefill_draft_cache(
        &mut self,
        output: &crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput,
        tokens: &crate::MlxTensor,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let sequence = tokens.as_array().dim(1);
        if sequence <= 1 {
            return Ok(());
        }
        let hidden = output
            .hidden
            .as_array()
            .try_index_device((.., ..sequence - 1, ..), stream)?;
        let next = tokens.as_array().try_index_device((.., 1..), stream)?;
        for depth in 0..self.model.mtp_len() {
            let _ = self.model.forward_mtp_parallel(
                eredu_architectures::nemotron_h::EmbeddedInput::draft(&next, &hidden, depth),
                cache,
                self.group,
                stream,
            )?;
        }
        Ok(())
    }

    fn draft_cache(&self, cache: &Self::Cache) -> Self::DraftCache {
        cache
            .deep_clone_state()
            .expect("evaluated Nemotron-H tensor draft state must be forkable")
    }

    fn commit_draft_cache(&self, cache: &mut Self::Cache, draft: &Self::DraftCache) {
        cache
            .commit_segment_from(draft, PREDICTION_STATE_SEGMENT)
            .expect("validated Nemotron-H tensor prediction state segment");
    }

    fn restore_target_checkpoint(
        cache: &mut Self::Cache,
        checkpoint: &Self::Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        cache.restore_checkpoint(checkpoint, stream)
    }

    fn draft_logits(
        &mut self,
        hidden: &crate::MlxTensor,
        last_token: u32,
        draft_index: usize,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(crate::MlxTensor, crate::MlxTensor), Exception> {
        let token = Array::from_slice(&[last_token], &[1, 1]);
        let output = self.model.forward_mtp_parallel(
            eredu_architectures::nemotron_h::EmbeddedInput::draft(
                &token,
                hidden.as_array(),
                draft_index,
            ),
            cache,
            self.group,
            stream,
        )?;
        Ok((output.logits, output.hidden))
    }

    fn advance_draft_cache(
        &mut self,
        hidden: &crate::MlxTensor,
        tokens: &crate::MlxTensor,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        for depth in 0..self.model.mtp_len() {
            let _ = self.model.forward_mtp_parallel(
                eredu_architectures::nemotron_h::EmbeddedInput::draft(
                    tokens.as_array(),
                    hidden.as_array(),
                    depth,
                ),
                cache,
                self.group,
                stream,
            )?;
        }
        Ok(())
    }

    fn max_draft_tokens(&self) -> usize {
        self.model.mtp_len()
    }
}

/// Loads SafeTensors Nemotron-H through one neutral model object.
pub fn load_nemotron_h_model(
    artifact: &crate::composition::mlx::artifact::PreparedSafetensorsArtifact,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<NemotronHModel, Error> {
    let expert_options = residency.expert_cache();
    let options = residency.layers();
    let eredu_architectures::configuration::SafetensorsModelConfig::NemotronH(args) =
        artifact.model()
    else {
        return Err(Error::ArchitectureModel(
            "Nemotron-H loader received a different prepared architecture".into(),
        ));
    };
    let args = args.clone();
    let quantize = quantization
        .map(|requested| {
            should_quantize_on_load("Nemotron-H", args.weight_quantization, requested)
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
    model: &mut NemotronHModel,
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

/// Loads SafeTensors Nemotron-H through generalized tensor-parallel placement.
pub fn load_nemotron_h_tensor_parallel_model(
    artifact: &crate::composition::mlx::artifact::PreparedSafetensorsArtifact,
    options: impl Into<LayerWeightResidency>,
    build: crate::backend::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<NemotronHModel, Error> {
    let options = options.into();
    let eredu_architectures::configuration::SafetensorsModelConfig::NemotronH(args) =
        artifact.model()
    else {
        return Err(Error::ArchitectureModel(
            "Nemotron-H loader received a different prepared architecture".into(),
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
    if !matches!(
        source.architecture(),
        eredu_architectures::GgufArchitecture::NemotronH
            | eredu_architectures::GgufArchitecture::NemotronHMoe
    ) {
        return Err(Error::ArchitectureModel(format!(
            "Nemotron-H GGUF loader received architecture {:?}",
            source.architecture()
        )));
    }
    let checkpoint = source.checkpoint();
    let eredu_architectures::configuration::GgufModelConfig::NemotronH(args) = source.model()
    else {
        return Err(Error::ArchitectureModel(
            "Nemotron-H GGUF loader received a different prepared model".into(),
        ));
    };
    let translate = eredu_architectures::nemotron_h::translate_gguf_weight_name;
    let configs = gguf_quantization_configs(checkpoint, translate)?;
    let args = eredu_architectures::nemotron_h::with_checkpoint_formats(args, configs)
        .map_err(Error::ArchitectureModel)?;
    Ok(PreparedGguf { args })
}

/// Loads a GGUF checkpoint through the same neutral Nemotron-H model object.
pub(crate) fn load_nemotron_h_gguf_model(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<NemotronHModel, Error> {
    let checkpoint = source.checkpoint();
    let prepared = prepare_gguf(source)?;
    let expert_options = residency.expert_cache();
    let store: Arc<dyn CheckpointSource> = Arc::new(open_gguf_checkpoint_source(
        checkpoint.clone(),
        source.plan().checkpoint(),
        eredu_architectures::nemotron_h::translate_gguf_weight_name,
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

/// Loads GGUF Nemotron-H with tensor-parallel placement.
pub(crate) fn load_nemotron_h_gguf_tensor_parallel_model(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    options: LayerWeightResidency,
    build: crate::backend::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<NemotronHModel, Error> {
    let checkpoint = source.checkpoint();
    let prepared = prepare_gguf(source)?;
    let store: Arc<dyn CheckpointSource> = Arc::new(open_gguf_checkpoint_source(
        checkpoint.clone(),
        source.plan().checkpoint(),
        eredu_architectures::nemotron_h::translate_gguf_weight_name,
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

//! MLX loading and runtime binding for the backend-neutral GPT-OSS decoder.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::Arc,
};

use eredu_architectures::gpt_oss::ModelArgs;
use eredu_checkpoint::{store::CheckpointSource, WeightQuantization};
use eredu_nn::{
    ParameterMetadata, ParameterVisitor, ParameterVisitorMut, Parameterized, RoutedNeuralBackend,
};
use eredu_runtime::{
    ArchitectureParameters, CacheResidencyPolicy, CausalModel, DenseDiskStreamReport,
    ExecutionResidency, LayerWeightResidency, LayerwiseModelMetadata, LayerwiseRuntime,
    PagedCacheOptions, ParallelModelInfo, ParameterRole, ResidencyReport, RuntimeState,
    WeightBinding, WeightResidency,
};
use safemlx::{error::Exception, ops::indexing::TryIndexOp, Array, Stream};

use crate::backend::{
    error::Error,
    nn::shared::{MlxModule, MlxNeuralBackend},
    runtime::{
        cache::{
            residency::{open_prompt_cache, CacheResidencyManager},
            state::MlxKeyValueState,
        },
        checkpoint::{
            binding::{
                apply_rank_local_parameter_recipes, binding_bytes,
                build_module_binding_plan_with_recipes_excluding, build_module_bindings,
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
            layerwise::{quantize_parameterized_store, shard_layer_bindings},
        },
        media::input,
        residency::expert_cache::{ExpertCache, ExpertCacheReport, ExpertCatalogEntry},
        residency::manager::ResidentUnitLease,
    },
};
use eredu_core::cache::{
    PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
    PromptCacheTopology,
};

pub mod expert {
    include!("gpt_oss_expert.rs");
}

/// The architecture-erased cache representation used by GPT-OSS.
pub type Cache = MlxKeyValueState;

type NeutralBlock = eredu_architectures::gpt_oss::TransformerBlock<MlxNeuralBackend>;
type NeutralArchitecture = eredu_architectures::gpt_oss::LayeredModel<MlxNeuralBackend>;

fn expert_parameter_targets(
    architecture: &NeutralArchitecture,
    stream: &Stream,
) -> Result<BTreeSet<String>, Error> {
    let mut targets = architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?
        .targets_for_role(ParameterRole::ExpertIntermediate);
    targets.extend(
        eredu_architectures::gpt_oss::safetensors_expert_tensors(architecture.args())
            .map_err(Error::ArchitectureModel)?
            .into_iter()
            .map(|tensor| tensor.key),
    );
    targets.extend(
        eredu_architectures::gpt_oss::gguf_expert_quantization_targets(architecture.args())
            .map_err(Error::ArchitectureModel)?,
    );
    Ok(targets)
}

fn require_decoder_group(architecture: &NeutralArchitecture, group: usize) -> Result<(), Error> {
    let transport = <NeutralArchitecture as eredu_runtime::LayeredArchitecture<
        MlxNeuralBackend,
        MlxKeyValueState,
    >>::group_transport(architecture, group);
    if transport.kind == eredu_runtime::ArchitectureGroupKind::Decoder {
        Ok(())
    } else {
        Err(Error::ArchitectureModel(format!(
            "GPT-OSS checkpoint bindings require the decoder execution group, got {group}"
        )))
    }
}

type ResidentRuntime = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxKeyValueState,
    MlxResidentPolicy<NeutralBlock>,
>;
type LayerwiseExecution = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxKeyValueState,
    MlxLayerwisePolicy<NeutralBlock, GptOssUnitPopulator>,
>;
type ParallelResidentRuntime = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxKeyValueState,
    MlxResidentPolicy<NeutralBlock>,
>;
type ParallelLayerwiseExecution = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxKeyValueState,
    MlxLayerwisePolicy<NeutralBlock, GptOssParallelUnitPopulator>,
>;

#[derive(Clone)]
struct GptOssUnitPopulator {
    external_experts: bool,
    expert_targets: Arc<BTreeSet<String>>,
}

impl MlxUnitPopulator<NeutralBlock> for GptOssUnitPopulator {
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

#[derive(Clone)]
struct GptOssParallelUnitPopulator {
    external_experts: bool,
    expert_targets: Arc<BTreeSet<String>>,
}

impl MlxUnitPopulator<NeutralBlock> for GptOssParallelUnitPopulator {
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

enum GptOssExecution {
    Resident(Box<ResidentRuntime>),
    Layerwise(Box<LayerwiseExecution>),
    TensorParallelResident(Box<ParallelResidentRuntime>),
    TensorParallelLayerwise(Box<ParallelLayerwiseExecution>),
}

/// Parameter view used only to select ordinary dense matrices for load-time
/// quantization. Native expert matrices retain their exact MXFP4 recipes.
#[derive(Debug, Clone)]
struct DenseUnit {
    block: NeutralBlock,
    expert_targets: Arc<BTreeSet<String>>,
}

impl Parameterized<crate::MlxTensor> for DenseUnit {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, crate::MlxTensor>,
    {
        struct Filter<'v, V>(&'v mut V, &'v BTreeSet<String>);
        impl<'a, V: ParameterVisitor<'a, crate::MlxTensor>> ParameterVisitor<'a, crate::MlxTensor>
            for Filter<'_, V>
        {
            fn visit(&mut self, metadata: ParameterMetadata, value: &'a crate::MlxTensor) {
                if !parameter_name_in_targets(metadata.id.as_str(), self.1) {
                    self.0.visit(metadata, value);
                }
            }
        }
        self.block
            .visit_parameters(&mut Filter(visitor, &self.expert_targets));
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, crate::MlxTensor>,
    {
        struct Filter<'v, V>(&'v mut V, &'v BTreeSet<String>);
        impl<'a, V: ParameterVisitorMut<'a, crate::MlxTensor>>
            ParameterVisitorMut<'a, crate::MlxTensor> for Filter<'_, V>
        {
            fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut crate::MlxTensor) {
                if !parameter_name_in_targets(metadata.id.as_str(), self.1) {
                    self.0.visit_mut(metadata, value);
                }
            }
        }
        self.block
            .visit_parameters_mut(&mut Filter(visitor, &self.expert_targets));
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.block.set_trainable(trainable);
    }
}

fn unit_recipes(
    store: &dyn CheckpointSource,
    args: &ModelArgs,
    layer: usize,
) -> Result<BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>, Error> {
    let recipes = eredu_architectures::gpt_oss::expert_recipes(store, args, layer)
        .map(|family| family.into_outputs().into_outputs())
        .map_err(Error::ArchitectureModel)?;
    recipes
        .into_iter()
        .map(|(name, mut recipe)| {
            if recipe.infer(store)?.dtype() == &eredu_checkpoint::recipe::RecipeDtype::F4 {
                recipe =
                    crate::backend::runtime::checkpoint::recipe::lower_mxfp4_recipe(recipe, store)?;
            }
            Ok((name, recipe))
        })
        .collect()
}

/// Builds one neutral GPT-OSS runtime from an already opened checkpoint store.
pub fn load_neutral_with_store(
    store: Arc<dyn CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
    materialization: Option<eredu_runtime::WeightMaterializationReport>,
    external_experts: bool,
) -> Result<GptOssModel, Error> {
    let mut architecture =
        eredu_architectures::gpt_oss::new_layered_model::<MlxNeuralBackend>(args.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let expert_targets = Arc::new(expert_parameter_targets(&architecture, stream)?);
    let factory = GptOssUnitPopulator {
        external_experts,
        expert_targets: Arc::clone(&expert_targets),
    };
    let binding_args = args.clone();
    let excluded_expert_targets = Arc::clone(&expert_targets);
    let binding_expert_targets = Arc::clone(&expert_targets);
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
                unit_recipes(store, &binding_args, index)?
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
    metadata.set_quantization(args.quantization);
    metadata.set_materialization(materialization);
    let execution = if options.is_fully_resident() {
        GptOssExecution::Resident(Box::new(LayerwiseRuntime::new_policy_first(
            policy.into_resident(
                &architecture,
                stream,
                std::marker::PhantomData::<MlxKeyValueState>,
            )?,
            architecture,
        )))
    } else {
        GptOssExecution::Layerwise(Box::new(LayerwiseRuntime::new(architecture, policy)))
    };
    let state_layout = eredu_architectures::gpt_oss::state_layout(&args)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    Ok(GptOssModel {
        args,
        state_layout,
        metadata,
        parallel_info: None,
        parallel_rank: None,
        planned_external_experts: None,
        prompt_cache_topology: PromptCacheTopology::default(),
        execution,
        expert_cache: None,
    })
}

fn load_neutral_parallel_with_store(
    store: Arc<dyn CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    build: crate::backend::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
    external_experts: bool,
) -> Result<GptOssModel, Error> {
    let global_architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let expert_targets = Arc::new(expert_parameter_targets(&global_architecture, stream)?);
    let parameter_description = global_architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let execution_layout = parameter_description.unit_layout().clone();
    let layout =
        crate::composition::parallel_layout_from_description(build, &parameter_description)?;
    if layout.is_empty() {
        return Err(Error::Parallel(
            "GPT-OSS declared no tensor-parallel parameters".into(),
        ));
    }
    let geometry = eredu_architectures::gpt_oss::local_geometry(&args, &layout)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let mut architecture = NeutralArchitecture::new_parallel(args.clone(), geometry, stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let state_layout = architecture
        .state_layout()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let factory = GptOssParallelUnitPopulator {
        external_experts,
        expert_targets: Arc::clone(&expert_targets),
    };

    let global_static = MlxModule::new(global_architecture.static_modules().clone());
    let mut global_parameter_bytes =
        binding_bytes(&build_module_bindings(&global_static, "", store.as_ref())?)?;
    for ordinal in 0..execution_layout.len() {
        let address = execution_layout
            .address(ordinal)
            .expect("canonical execution ordinal");
        let block = construct_architecture_unit(
            &global_architecture,
            &execution_layout,
            ordinal,
            stream,
            std::marker::PhantomData::<MlxKeyValueState>,
        )?;
        let recipes = if external_experts {
            BTreeMap::new()
        } else {
            unit_recipes(store.as_ref(), &args, address.index())?
        };
        let bindings = build_module_bindings_with_recipes_excluding(
            &MlxModule::new(block),
            "",
            store.as_ref(),
            recipes,
            |name| external_experts && parameter_name_in_targets(name, &expert_targets),
        )?;
        global_parameter_bytes = global_parameter_bytes
            .checked_add(binding_bytes(&bindings)?)
            .ok_or_else(|| Error::Parallel("global GPT-OSS parameter bytes overflowed".into()))?;
    }

    let binding_args = args.clone();
    let global_static_modules = global_architecture.static_modules().clone();
    let binding_execution_layout = execution_layout;
    let local_layout = Arc::new(layout);
    let static_layout = Arc::clone(&local_layout);
    let unit_local_layout = Arc::clone(&local_layout);
    let excluded_expert_targets = Arc::clone(&expert_targets);
    let binding_expert_targets = Arc::clone(&expert_targets);
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
            shard_layer_bindings(bindings, store, &static_layout)
        },
        move |ordinal, address, _path, _local, store, stream| {
            let layer = address.index();
            let global = construct_architecture_unit(
                &global_architecture,
                &binding_execution_layout,
                ordinal,
                stream,
                std::marker::PhantomData::<MlxKeyValueState>,
            )?;
            let recipes = if external_experts {
                BTreeMap::new()
            } else {
                unit_recipes(store, &binding_args, layer)?
            };
            let bindings = build_module_bindings_with_recipes_excluding(
                &MlxModule::new(global),
                "",
                store,
                recipes,
                |name| external_experts && parameter_name_in_targets(name, &binding_expert_targets),
            )?;
            shard_layer_bindings(bindings, store, &unit_local_layout)
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.quantization);
    let local_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.layer_parameter_bytes())
        .ok_or_else(|| Error::Parallel("local GPT-OSS parameter bytes overflowed".into()))?;
    let maximum_device_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.maximum_device_layer_bytes())
        .ok_or_else(|| Error::Parallel("device GPT-OSS parameter bytes overflowed".into()))?;
    let parallel_info = ParallelModelInfo::new(
        build.topology(),
        args.model_type.clone(),
        local_layout
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
        GptOssExecution::TensorParallelResident(Box::new(LayerwiseRuntime::new_policy_first(
            policy.into_resident(
                &architecture,
                stream,
                std::marker::PhantomData::<MlxKeyValueState>,
            )?,
            architecture,
        )))
    } else {
        GptOssExecution::TensorParallelLayerwise(Box::new(LayerwiseRuntime::new(
            architecture,
            policy,
        )))
    };
    let planned_external_experts = external_experts
        .then(|| expert::expert_catalog(&args, store.as_ref(), Some(&local_layout)))
        .transpose()?;
    Ok(GptOssModel {
        args,
        state_layout,
        metadata,
        parallel_info: Some(parallel_info),
        parallel_rank,
        planned_external_experts,
        prompt_cache_topology: crate::backend::cache::prompt_cache_topology(build.topology()),
        execution,
        expert_cache: None,
    })
}

/// Builds a bounded packed overlay for eligible dense matrices while leaving
/// GPT-OSS native expert tensors in their original checkpoint representation.
pub fn quantize_neutral_store(
    store: Arc<dyn CheckpointSource>,
    source_args: &ModelArgs,
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
    let target_args =
        eredu_architectures::gpt_oss::load_time_quantization(source_args, quantization)
            .map_err(Error::ArchitectureModel)?;
    let source = eredu_architectures::gpt_oss::new_layered_model::<MlxNeuralBackend>(
        source_args.clone(),
        stream,
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let target = eredu_architectures::gpt_oss::new_layered_model::<MlxNeuralBackend>(
        target_args.clone(),
        stream,
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let source_expert_targets = Arc::new(expert_parameter_targets(&source, stream)?);
    let target_expert_targets = Arc::new(expert_parameter_targets(&target, stream)?);
    let source_layout = architecture_execution_layout::<_, MlxKeyValueState>(&source)?;
    let target_layout = architecture_execution_layout::<_, MlxKeyValueState>(&target)?;
    if source_layout != target_layout {
        return Err(Error::Quantization(
            "GPT-OSS quantization changed the architecture execution layout".into(),
        ));
    }
    let unit_count = source_layout.len();
    let source_static = source.static_modules().clone();
    let target_static = target.static_modules().clone();
    let source_unit_expert_targets = Arc::clone(&source_expert_targets);
    let target_unit_expert_targets = Arc::clone(&target_expert_targets);
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
            .map(|block| DenseUnit {
                block,
                expert_targets: Arc::clone(&source_unit_expert_targets),
            })
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        },
        move |ordinal, stream| {
            construct_architecture_unit(
                &target,
                &target_layout,
                ordinal,
                stream,
                std::marker::PhantomData::<MlxKeyValueState>,
            )
            .map(|block| DenseUnit {
                block,
                expert_targets: Arc::clone(&target_unit_expert_targets),
            })
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        },
        unit_count,
        quantization,
        stream,
    )?;
    Ok((store, target_args, report))
}

/// Thin checkpoint-binding adapter used by GPT-OSS pipeline stages.
#[derive(Default)]
pub struct GptOssPipelineBindings {
    external_experts: bool,
}

impl GptOssPipelineBindings {
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
                "GPT-OSS has no architecture expert realization".into(),
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

    /// Applies rank-local expert residency to an architecture-constructed unit.
    pub fn prepare_unit_expert_residency(
        &self,
        index: usize,
        layer: &mut MlxModule<NeutralBlock>,
        realization: Option<
            &eredu_architectures::ExpertRealizationPlan<eredu_nn::GatedProductExpertBankSpec>,
        >,
        assignment: Option<&crate::backend::runtime::distributed::expert::ExpertAssignment>,
        stream: &Stream,
    ) -> Result<(), Error> {
        if let Some(assignment) = assignment {
            let plan = realization.ok_or_else(|| {
                Error::Parallel("GPT-OSS expert assignment has no architecture realization".into())
            })?;
            if assignment.local_global_expert_ids() != plan.local_global_expert_ids() {
                return Err(Error::Parallel(
                    "GPT-OSS native assignment disagrees with architecture realization".into(),
                ));
            }
            let spec = plan
                .unit_spec("text_decoder", index)
                .cloned()
                .ok_or_else(|| {
                    Error::Parallel(format!(
                        "GPT-OSS architecture realization has no expert bank for unit {index}"
                    ))
                })?;
            layer.inner.mlp.experts =
                <MlxNeuralBackend as RoutedNeuralBackend>::gated_product_expert_bank(spec, stream)
                    .map_err(|error| Error::Parallel(error.to_string()))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cartesian_layer_bindings(
        &self,
        architecture: &NeutralArchitecture,
        group: usize,
        index: usize,
        store: &dyn CheckpointSource,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        assignment: Option<&crate::backend::runtime::distributed::expert::ExpertAssignment>,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        require_decoder_group(architecture, group)?;
        let global_layer = MlxModule::new(
            architecture
                .construct_unit(index, stream)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
        );
        let expert_targets = parameter_role_targets(
            &eredu_architectures::gpt_oss::layer_parallel_parameter_groups(
                &global_layer,
                architecture.args(),
                index,
            )
            .map_err(|error| Error::Parallel(error.to_string()))?,
            ParameterRole::ExpertIntermediate,
        );
        let recipes = if self.external_experts {
            BTreeMap::new()
        } else {
            unit_recipes(store, architecture.args(), index)?
        };
        let bindings = build_module_binding_plan_with_recipes_excluding(
            &global_layer,
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
                eredu_architectures::gpt_oss::rank_local_expert_recipes(
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
            Some(layout) => shard_layer_bindings(bindings, store, layout),
            None => Ok(bindings),
        }
    }

    pub fn layer_bindings(
        &self,
        architecture: &NeutralArchitecture,
        group: usize,
        index: usize,
        layer: &MlxModule<NeutralBlock>,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        require_decoder_group(architecture, group)?;
        let expert_targets = parameter_role_targets(
            &eredu_architectures::gpt_oss::layer_parallel_parameter_groups(
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
            unit_recipes(store, architecture.args(), index)?
        };
        Ok(
            build_module_binding_plan_with_recipes_excluding(layer, "", store, recipes, |name| {
                self.external_experts && parameter_name_in_targets(name, &expert_targets)
            })?
            .build_bindings(store)?,
        )
    }
}

/// Neutral GPT-OSS causal LM with resident or bounded layer execution.
pub struct GptOssModel {
    args: ModelArgs,
    state_layout: eredu_runtime::StateLayout,
    metadata: LayerwiseModelMetadata,
    parallel_info: Option<ParallelModelInfo<crate::backend::MlxParallelContext>>,
    parallel_rank: Option<eredu_core::cache::CacheRankIdentity>,
    planned_external_experts: Option<Vec<ExpertCatalogEntry>>,
    prompt_cache_topology: PromptCacheTopology,
    execution: GptOssExecution,
    expert_cache: Option<ExpertCache>,
}

impl GptOssModel {
    /// Returns normalized model arguments.
    pub fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Returns canonical parameter and residency metadata.
    pub fn metadata(&self) -> &LayerwiseModelMetadata {
        &self.metadata
    }

    pub fn parallel_info(&self) -> Option<&ParallelModelInfo<crate::backend::MlxParallelContext>> {
        self.parallel_info.as_ref()
    }

    /// Builds expert-cache units with this rank's exact TP selections.
    pub fn external_expert_catalog(&self) -> Result<Vec<ExpertCatalogEntry>, Error> {
        self.planned_external_experts.clone().map_or_else(
            || expert::expert_catalog(&self.args, self.checkpoint_store(), None),
            Ok,
        )
    }

    /// Records the complete distributed coordinates used by EP/TP+EP wrappers.
    pub fn bind_parallel_topology(&mut self, topology: crate::backend::MlxParallelContext) {
        self.parallel_rank =
            crate::backend::cache::prompt_cache_topology(topology).cache_rank_identity();
        self.prompt_cache_topology = crate::backend::cache::prompt_cache_topology(topology);
    }

    /// Returns whether all decoder blocks remain resident on the execution device.
    pub fn is_fully_resident(&self) -> bool {
        self.metadata.residency() == ExecutionResidency::FullyResident
    }

    /// Returns the canonical cache-relevant architecture identity.
    pub fn prompt_cache_architecture_fingerprint(&self) -> String {
        eredu_architectures::gpt_oss::prompt_cache_architecture_fingerprint(&self.args)
    }

    /// Returns this model's exact ordered prompt-cache layer layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<eredu_core::LayerSchedule<eredu_core::cache::LayerCachePolicy>, Error> {
        Ok(self.prompt_cache_model_identity()?.layer_layout)
    }

    /// Returns logical layer-residency telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        match &self.execution {
            GptOssExecution::Resident(execution) => execution.policy().residency_report(),
            GptOssExecution::Layerwise(execution) => execution.policy().residency_report(),
            GptOssExecution::TensorParallelResident(execution) => {
                execution.policy().residency_report()
            }
            GptOssExecution::TensorParallelLayerwise(execution) => {
                execution.policy().residency_report()
            }
        }
    }

    /// Returns dense disk-stream telemetry when active.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        match &self.execution {
            GptOssExecution::Resident(_) => Ok(None),
            GptOssExecution::Layerwise(execution) => execution.policy().dense_stream_report(),
            GptOssExecution::TensorParallelResident(_) => Ok(None),
            GptOssExecution::TensorParallelLayerwise(execution) => {
                execution.policy().dense_stream_report()
            }
        }
    }

    /// Returns independent expert-cache telemetry when configured.
    pub fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        self.expert_cache
            .as_ref()
            .map(ExpertCache::report)
            .transpose()
            .map_err(Error::from)
    }

    /// Returns the persistent checkpoint store used by either execution policy.
    pub fn checkpoint_store(&self) -> &dyn CheckpointSource {
        match &self.execution {
            GptOssExecution::Resident(execution) => execution.policy().checkpoint_store(),
            GptOssExecution::Layerwise(execution) => execution.policy().checkpoint_store(),
            GptOssExecution::TensorParallelResident(execution) => {
                execution.policy().checkpoint_store()
            }
            GptOssExecution::TensorParallelLayerwise(execution) => {
                execution.policy().checkpoint_store()
            }
        }
    }

    pub fn checkpoint_store_arc(&self) -> Arc<dyn CheckpointSource> {
        match &self.execution {
            GptOssExecution::Resident(execution) => execution.policy().checkpoint_store_arc(),
            GptOssExecution::Layerwise(execution) => execution.policy().checkpoint_store_arc(),
            GptOssExecution::TensorParallelResident(execution) => {
                execution.policy().checkpoint_store_arc()
            }
            GptOssExecution::TensorParallelLayerwise(execution) => {
                execution.policy().checkpoint_store_arc()
            }
        }
    }

    /// Returns the number of static checkpoint leases.
    pub fn static_lease_count(&self) -> usize {
        match &self.execution {
            GptOssExecution::Resident(execution) => execution.policy().static_lease_count(),
            GptOssExecution::Layerwise(execution) => execution.policy().static_lease_count(),
            GptOssExecution::TensorParallelResident(execution) => {
                execution.policy().static_lease_count()
            }
            GptOssExecution::TensorParallelLayerwise(execution) => {
                execution.policy().static_lease_count()
            }
        }
    }

    /// Creates empty device-resident state.
    pub fn new_cache(&self) -> Cache {
        MlxKeyValueState::device(self.state_layout.clone())
            .expect("MLX key/value state supports validated GPT-OSS geometry")
    }

    /// Creates device or explicitly bounded paged state.
    pub fn new_cache_with_options(&self, policy: CacheResidencyPolicy) -> Result<Cache, Error> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => {
                self.new_paged_cache(options, None, self.parallel_rank)
            }
        }
    }

    /// Lazily catalogs a compatible persisted prefix.
    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(Cache, PromptCacheManifest), Error> {
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

    /// Persists a completed prefix after validating model identity.
    pub fn save_prompt_cache(
        &self,
        cache: &mut Cache,
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
    ) -> Result<Cache, Error> {
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
    ) -> Result<Cache, Error> {
        MlxKeyValueState::paged(self.state_layout.clone(), manager, rank).map_err(Into::into)
    }

    /// Executes embedding, all neutral blocks, final normalization, and head.
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
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
        let input = eredu_architectures::decoder::LayeredInput {
            tokens: crate::composition::tensor_ref(inputs),
            mask: None,
        };
        let output = match &mut self.execution {
            GptOssExecution::Resident(execution) => execution
                .forward(input, cache, stream)
                .map_err(|error| Error::ArchitectureModel(error.to_string())),
            GptOssExecution::Layerwise(execution) => execution
                .forward(input, cache, stream)
                .map_err(|error| Error::ArchitectureModel(error.to_string())),
            GptOssExecution::TensorParallelResident(_)
            | GptOssExecution::TensorParallelLayerwise(_) => Err(Error::Parallel(
                "tensor-parallel GPT-OSS execution requires a collective group".into(),
            )),
        }?;
        Ok(output.into_array())
    }

    /// Runs the neutral decoder with runtime-owned expert residency.
    pub fn forward_with_expert_provider<P>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Cache,
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
        let hook =
            |architecture: &mut NeutralArchitecture,
             group: usize,
             index: usize,
             block: &mut NeutralBlock,
             hidden: &crate::MlxTensor,
             state: &mut Cache,
             forward: &mut eredu_architectures::gpt_oss::ForwardContext<crate::MlxTensor>,
             context: &Stream| {
                <NeutralArchitecture as eredu_runtime::RoutedLayeredArchitecture<
                    MlxNeuralBackend,
                    Cache,
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
        let input = eredu_architectures::decoder::LayeredInput {
            tokens: crate::composition::tensor_ref(inputs),
            mask: crate::composition::tensor_opt(mask),
        };
        let output = match &mut self.execution {
            GptOssExecution::Resident(runtime) => runtime
                .forward_with_unit_executor(input, cache, stream, hook)
                .map_err(|error| Error::ArchitectureModel(error.to_string())),
            GptOssExecution::Layerwise(runtime) => runtime
                .forward_with_unit_executor(input, cache, stream, hook)
                .map_err(|error| Error::ArchitectureModel(error.to_string())),
            GptOssExecution::TensorParallelResident(_)
            | GptOssExecution::TensorParallelLayerwise(_) => Err(Error::Parallel(
                "tensor-parallel GPT-OSS expert execution requires a collective group".into(),
            )),
        }?;
        Ok(output.into_array())
    }

    /// Runs a rank-local neutral GPT-OSS tensor-parallel forward pass.
    pub fn forward_tensor_parallel(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
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
        let input = eredu_architectures::gpt_oss::LayeredInput {
            tokens: crate::composition::tensor_ref(inputs),
            mask: None,
        };
        let output = match &mut self.execution {
            GptOssExecution::TensorParallelResident(runtime) => runtime
                .forward_parallel(input, cache, group, stream)
                .map_err(|error| Error::Parallel(error.to_string())),
            GptOssExecution::TensorParallelLayerwise(runtime) => runtime
                .forward_parallel(input, cache, group, stream)
                .map_err(|error| Error::Parallel(error.to_string())),
            GptOssExecution::Resident(_) | GptOssExecution::Layerwise(_) => Err(Error::Parallel(
                "model was not loaded for GPT-OSS tensor-parallel execution".into(),
            )),
        }?;
        Ok(output.into_array())
    }

    pub fn forward_tensor_parallel_with_observer(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error> {
        self.validate_cache(cache)?;
        let args = self.args.clone();
        let expert_cache = self.expert_cache.take();
        let result = {
            let input = eredu_architectures::gpt_oss::LayeredInput {
                tokens: crate::composition::tensor_ref(inputs),
                mask: None,
            };
            let pass = if inputs.dim(1) > 1 {
                eredu_runtime::ExpertPass::Prefill
            } else {
                eredu_runtime::ExpertPass::Decode
            };
            let mut neutral = crate::composition::NeutralActivationObserver::new(observer);
            let output = match expert_cache.as_ref() {
                Some(expert_cache) => {
                    let mut provider = expert::cached_provider(expert_cache, &args);
                    match &mut self.execution {
                        GptOssExecution::TensorParallelResident(runtime) => runtime
                            .forward_parallel_with_provider_and_observer(
                                input,
                                cache,
                                pass,
                                &mut provider,
                                group,
                                stream,
                                &mut neutral,
                            ),
                        GptOssExecution::TensorParallelLayerwise(runtime) => runtime
                            .forward_parallel_with_provider_and_observer(
                                input,
                                cache,
                                pass,
                                &mut provider,
                                group,
                                stream,
                                &mut neutral,
                            ),
                        _ => {
                            return Err(Error::Parallel(
                                "GPT-OSS was not loaded for tensor parallelism".into(),
                            ))
                        }
                    }
                }
                None => {
                    let mut provider = eredu_runtime::ResidentExpertProvider;
                    match &mut self.execution {
                        GptOssExecution::TensorParallelResident(runtime) => runtime
                            .forward_parallel_with_provider_and_observer(
                                input,
                                cache,
                                pass,
                                &mut provider,
                                group,
                                stream,
                                &mut neutral,
                            ),
                        GptOssExecution::TensorParallelLayerwise(runtime) => runtime
                            .forward_parallel_with_provider_and_observer(
                                input,
                                cache,
                                pass,
                                &mut provider,
                                group,
                                stream,
                                &mut neutral,
                            ),
                        _ => {
                            return Err(Error::Parallel(
                                "GPT-OSS was not loaded for tensor parallelism".into(),
                            ))
                        }
                    }
                }
            }
            .map_err(|error| Error::Parallel(error.to_string()))?;
            eredu_runtime::observe_model_logits(&mut neutral, &output)
                .map(crate::MlxTensor::into_array)
                .map_err(Error::from)
        };
        self.expert_cache = expert_cache;
        result
    }

    /// Runs tensor-parallel attention and provider-owned routed experts.
    pub fn forward_tensor_expert_provider<P>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Cache,
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
        let hook =
            |architecture: &mut NeutralArchitecture,
             execution_group: usize,
             index: usize,
             block: &mut NeutralBlock,
             hidden: &crate::MlxTensor,
             state: &mut Cache,
             forward: &mut eredu_architectures::gpt_oss::ForwardContext<crate::MlxTensor>,
             parallel: &safemlx::distributed::Group,
             context: &Stream| {
                <NeutralArchitecture as eredu_runtime::ParallelRoutedLayeredArchitecture<
                    MlxNeuralBackend,
                    Cache,
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
        let input = eredu_architectures::gpt_oss::LayeredInput {
            tokens: crate::composition::tensor_ref(inputs),
            mask: crate::composition::tensor_opt(mask),
        };
        let output = match &mut self.execution {
            GptOssExecution::TensorParallelResident(runtime) => runtime
                .forward_parallel_with_unit_executor(input, cache, group, stream, hook)
                .map_err(|error| Error::Parallel(error.to_string())),
            GptOssExecution::TensorParallelLayerwise(runtime) => runtime
                .forward_parallel_with_unit_executor(input, cache, group, stream, hook)
                .map_err(|error| Error::Parallel(error.to_string())),
            GptOssExecution::Resident(_) | GptOssExecution::Layerwise(_) => Err(Error::Parallel(
                "model was not loaded for GPT-OSS tensor-parallel expert execution".into(),
            )),
        }?;
        Ok(output.into_array())
    }

    /// Runs with stable layer-input, layer-output, and logits observations.
    pub fn forward_with_observer(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Cache,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error> {
        let expert_cache = self.expert_cache.take();
        let mut observer = crate::composition::NeutralActivationObserver::new(observer);
        let result = match expert_cache.as_ref() {
            Some(expert_cache) => {
                let args = self.args.clone();
                let mut provider = expert::cached_provider(expert_cache, &args);
                self.forward_observed_with_provider(
                    inputs,
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
                    inputs,
                    mask,
                    cache,
                    &mut provider,
                    stream,
                    &mut observer,
                )
            }
        };
        self.expert_cache = expert_cache;
        result
    }

    fn forward_observed_with_provider<P>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Cache,
        provider: &mut P,
        stream: &Stream,
        observer: &mut crate::composition::NeutralActivationObserver<'_>,
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
        let output = match &mut self.execution {
            GptOssExecution::Resident(runtime) => runtime
                .forward_with_provider_and_observer(
                    eredu_architectures::decoder::LayeredInput {
                        tokens: crate::composition::tensor_ref(inputs),
                        mask: crate::composition::tensor_opt(mask),
                    },
                    cache,
                    pass,
                    provider,
                    stream,
                    observer,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
            GptOssExecution::Layerwise(runtime) => runtime
                .forward_with_provider_and_observer(
                    eredu_architectures::decoder::LayeredInput {
                        tokens: crate::composition::tensor_ref(inputs),
                        mask: crate::composition::tensor_opt(mask),
                    },
                    cache,
                    pass,
                    provider,
                    stream,
                    observer,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
            GptOssExecution::TensorParallelResident(_)
            | GptOssExecution::TensorParallelLayerwise(_) => {
                return Err(Error::Parallel(
                    "tensor-parallel GPT-OSS observation requires a collective observer path"
                        .into(),
                ))
            }
        };
        eredu_runtime::observe_model_logits(observer, &output)
            .map(crate::MlxTensor::into_array)
            .map_err(Into::into)
    }

    /// Runs prompt prefill and returns final-token logits.
    pub fn prefill(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward(inputs, cache, stream)?
            .try_index_device((.., -1, ..), stream)
            .map_err(Into::into)
    }

    /// Runs cached decode and returns final-token logits.
    pub fn decode(
        &mut self,
        input_tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.prefill(input_tokens, cache, stream)
    }

    /// Clears temporary device block windows for nonresident execution.
    pub fn clear_device_layer_window(&self) -> Result<bool, Error> {
        match &self.execution {
            GptOssExecution::Resident(_) => Ok(false),
            GptOssExecution::Layerwise(execution) => {
                execution.policy().clear_device_window()?;
                Ok(true)
            }
            GptOssExecution::TensorParallelResident(_) => Ok(false),
            GptOssExecution::TensorParallelLayerwise(execution) => {
                execution.policy().clear_device_window()?;
                Ok(true)
            }
        }
    }

    pub fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        let identity = eredu_architectures::gpt_oss::state_identity(
            &self.args,
            &self.state_layout,
            0,
            self.prompt_cache_topology.clone(),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        identity
            .prompt_cache_identity(&self.state_layout)
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn validate_cache(&self, cache: &Cache) -> Result<(), Error> {
        if cache.layout() != &self.state_layout {
            return Err(Exception::custom(format!(
                "MLX key/value state layout {:?} does not match GPT-OSS layout {:?}",
                cache.layout(),
                self.state_layout
            ))
            .into());
        }
        Ok(())
    }
}

impl CausalModel<Cache> for GptOssModel {
    type Tensor = crate::MlxTensor;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
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
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Exception> {
        self.decode(input_tokens.as_array(), cache, stream)
            .map(crate::MlxTensor::from_array)
            .map_err(|error| Exception::custom(error.to_string()))
    }
}

fn attach_expert_cache(
    model: &mut GptOssModel,
    options: eredu_runtime::ExpertCacheLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(), Error> {
    let store = model.checkpoint_store_arc();
    let entries = model.external_expert_catalog()?;
    model.expert_cache = Some(ExpertCache::new_shared(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(())
}

/// Loads SafeTensors GPT-OSS using the selected weight-residency policy.
pub fn load_safetensors(
    artifact: &crate::composition::mlx::artifact::PreparedSafetensorsArtifact,
    weight_residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssModel, Error> {
    let expert_options = weight_residency.expert_cache();
    let execution_options = weight_residency.layers();
    let eredu_architectures::configuration::SafetensorsModelConfig::GptOss(args) = artifact.model()
    else {
        return Err(Error::ArchitectureModel(
            "GPT-OSS loader received a different prepared architecture".into(),
        ));
    };
    let args = args.clone();
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load("GPT-OSS", args.quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let store = artifact.store();
    let (store, args, materialization) = match quantize_on_load {
        Some(quantization) => {
            let (store, args, report) = quantize_neutral_store(store, &args, quantization, stream)?;
            (store, args, Some(report))
        }
        None => (store, args, None),
    };
    let mut model = load_neutral_with_store(
        store,
        args,
        execution_options,
        stream,
        weights_stream,
        materialization,
        expert_options.is_some(),
    )?;
    if let Some(options) = expert_options {
        attach_expert_cache(&mut model, options, stream, weights_stream)?;
    }
    Ok(model)
}

/// Loads SafeTensors or an inspected GGUF through the neutral GPT-OSS tensor-parallel graph.
pub fn load_gpt_oss_tensor_parallel_model(
    artifact: &crate::composition::mlx::artifact::PreparedSafetensorsArtifact,
    options: impl Into<LayerWeightResidency>,
    build: crate::backend::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssModel, Error> {
    let options = options.into();
    let eredu_architectures::configuration::SafetensorsModelConfig::GptOss(args) = artifact.model()
    else {
        return Err(Error::ArchitectureModel(
            "GPT-OSS loader received a different prepared architecture".into(),
        ));
    };
    let args = args.clone();
    let store = artifact.store();
    load_neutral_parallel_with_store(store, args, options, build, stream, weights_stream, false)
}

/// Header-only results needed to open a portable GGUF GPT-OSS checkpoint.
pub(crate) struct PreparedGptOssGguf {
    pub args: ModelArgs,
}

/// Normalizes backend-native GGUF encoding metadata without reading payloads.
pub(crate) fn prepare_gpt_oss_gguf_checkpoint(
    source: &crate::composition::mlx::structural::AdmittedGguf,
) -> Result<PreparedGptOssGguf, Error> {
    if source.architecture() != eredu_architectures::GgufArchitecture::GptOss {
        return Err(Error::ArchitectureModel(format!(
            "GPT-OSS GGUF loader received architecture {:?}",
            source.architecture()
        )));
    }
    let checkpoint = source.checkpoint();
    let eredu_architectures::configuration::GgufModelConfig::GptOss(args) = source.model() else {
        return Err(Error::ArchitectureModel(
            "GPT-OSS GGUF loader received a different prepared model".into(),
        ));
    };
    let configs = gguf_quantization_configs(checkpoint, source.plan().tensor_mapping())?;
    let args = eredu_architectures::gpt_oss::with_checkpoint_formats(args, configs)
        .map_err(Error::ArchitectureModel)?;
    Ok(PreparedGptOssGguf { args })
}

/// Loads a GGUF checkpoint through the same neutral model/runtime object.
pub(crate) fn load_gpt_oss_gguf_model(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssModel, Error> {
    let checkpoint = source.checkpoint();
    let prepared = prepare_gpt_oss_gguf_checkpoint(source)?;
    let store: Arc<dyn CheckpointSource> = Arc::new(open_gguf_checkpoint_source(
        checkpoint.clone(),
        source.plan().checkpoint(),
        source.plan().tensor_mapping(),
        residency.max_mapped_shards(),
    )?);
    let expert_options = residency.expert_cache();
    let execution_options = residency.layers();
    let (store, args, materialization) = match quantization {
        Some(quantization) => {
            let (store, args, report) =
                quantize_neutral_store(store, &prepared.args, quantization, stream)?;
            (store, args, Some(report))
        }
        None => (store, prepared.args, None),
    };
    let mut model = load_neutral_with_store(
        store,
        args,
        execution_options,
        stream,
        weights_stream,
        materialization,
        expert_options.is_some(),
    )?;
    if let Some(options) = expert_options {
        attach_expert_cache(&mut model, options, stream, weights_stream)?;
    }
    Ok(model)
}

/// Loads a validated GGUF checkpoint through the neutral tensor-parallel graph.
pub(crate) fn load_gpt_oss_gguf_tensor_parallel_model(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    options: LayerWeightResidency,
    build: crate::backend::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssModel, Error> {
    let checkpoint = source.checkpoint();
    let prepared = prepare_gpt_oss_gguf_checkpoint(source)?;
    let store: Arc<dyn CheckpointSource> = Arc::new(open_gguf_checkpoint_source(
        checkpoint.clone(),
        source.plan().checkpoint(),
        source.plan().tensor_mapping(),
        options.max_mapped_shards(),
    )?);
    let model = load_neutral_parallel_with_store(
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

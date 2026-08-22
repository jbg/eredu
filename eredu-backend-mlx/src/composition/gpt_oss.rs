//! MLX loading and runtime binding for the backend-neutral GPT-OSS decoder.

use std::{collections::BTreeMap, path::Path, sync::Arc};

use eredu_architectures::gpt_oss::ModelArgs;
use eredu_checkpoint::{store::CheckpointSource, store::TensorSelection, WeightQuantization};
#[cfg(any(test, feature = "test-support"))]
use eredu_nn::ParameterSpec;
use eredu_nn::{
    ParameterMetadata, ParameterVisitor, ParameterVisitorMut, Parameterized, RoutedNeuralBackend,
};
use eredu_runtime::{
    CacheResidencyPolicy, CausalModel, DenseDiskStreamReport, ExecutionResidency,
    LayerWeightResidency, LayerwiseModelMetadata, LayerwiseRuntime, PagedCacheOptions,
    ParallelModelInfo, ParameterRole, ResidencyReport, RuntimeState, StaticUnitBindings,
    WeightBinding, WeightResidency,
};
use safemlx::{error::Exception, ops::indexing::TryIndexOp, Array, Stream};

use crate::{
    backend::mlx::{
        error::Error,
        nn::shared::{MlxBackend, MlxModule},
        runtime::{
            cache::{
                residency::{open_prompt_cache, CacheResidencyManager},
                state::MlxKeyValueState,
            },
            checkpoint::{
                binding::{
                    binding_bytes, build_module_binding_plan_with_recipes,
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
                    prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
                    MlxUnitPopulator,
                },
                layerwise::{
                    open_safetensors_weight_store, quantize_parameterized_store,
                    shard_layer_bindings,
                },
            },
            media::input,
            residency::expert_cache::{ExpertCache, ExpertCacheReport, ExpertCatalogEntry},
            residency::manager::ResidentUnitLease,
        },
    },
    core::cache::{
        PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
        PromptCacheTopology,
    },
};

pub mod expert {
    include!("gpt_oss_expert.rs");
}

/// The architecture-erased cache representation used by GPT-OSS.
pub type Cache = MlxKeyValueState;

type NeutralBlock = eredu_architectures::gpt_oss::TransformerBlock<MlxBackend>;
type NeutralArchitecture = eredu_architectures::gpt_oss::LayeredModel<MlxBackend>;

struct NeutralGptOssObserver<'a> {
    inner: &'a mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
}

impl eredu_runtime::ActivationObserver<crate::MlxTensor, eredu_nn::Error>
    for NeutralGptOssObserver<'_>
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

fn require_decoder_group(architecture: &NeutralArchitecture, group: usize) -> Result<(), Error> {
    let transport = <NeutralArchitecture as eredu_runtime::LayeredArchitecture<
        MlxBackend,
        MlxKeyValueState,
    >>::group_transport(architecture, group);
    if transport.kind == eredu_runtime::ArchitectureGroupKind::Decoder {
        Ok(())
    } else {
        Err(Error::UnsupportedArchitecture(format!(
            "GPT-OSS checkpoint bindings require the decoder execution group, got {group}"
        )))
    }
}
type ResidentRuntime = LayerwiseRuntime<
    NeutralArchitecture,
    MlxBackend,
    MlxKeyValueState,
    MlxResidentPolicy<NeutralBlock>,
>;
type LayerwiseExecution = LayerwiseRuntime<
    NeutralArchitecture,
    MlxBackend,
    MlxKeyValueState,
    MlxLayerwisePolicy<NeutralBlock, GptOssUnitPopulator>,
>;
type ParallelResidentRuntime = LayerwiseRuntime<
    NeutralArchitecture,
    MlxBackend,
    MlxKeyValueState,
    MlxResidentPolicy<NeutralBlock>,
>;
type ParallelLayerwiseExecution = LayerwiseRuntime<
    NeutralArchitecture,
    MlxBackend,
    MlxKeyValueState,
    MlxLayerwisePolicy<NeutralBlock, GptOssParallelUnitPopulator>,
>;

#[doc(hidden)]
#[cfg(any(test, feature = "test-support"))]
pub struct GptOssCheckpointTemplate {
    pub static_modules: eredu_architectures::decoder::StaticModules<MlxBackend>,
    pub layers: Vec<NeutralBlock>,
    native_experts: Vec<GptOssCheckpointParameter>,
}

#[cfg(any(test, feature = "test-support"))]
struct GptOssCheckpointParameter {
    spec: ParameterSpec,
    trainable: bool,
    value: crate::MlxTensor,
}

#[cfg(any(test, feature = "test-support"))]
impl GptOssCheckpointTemplate {
    /// Builds one neutral full-parameter template for checkpoint tooling.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let architecture = NeutralArchitecture::new(args.clone(), stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let layers = (0..args.num_hidden_layers as usize)
            .map(|index| {
                eredu_architectures::gpt_oss::new_block::<MlxBackend>(&args, index, stream)
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let plan = eredu_architectures::gpt_oss::safetensors_plan(&args)
            .map_err(Error::UnsupportedArchitecture)?;
        let native_experts = plan
            .common_tensors
            .into_iter()
            .filter(|tensor| tensor.key.contains(".mlp.experts."))
            .map(|tensor| {
                let shape = tensor
                    .shape
                    .iter()
                    .copied()
                    .map(i32::try_from)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| {
                        Error::UnsupportedArchitecture(format!(
                            "GPT-OSS fixture shape exceeds i32 for {:?}",
                            tensor.key
                        ))
                    })?;
                let value = match tensor.dtype {
                    eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                        eredu_checkpoint::StoredDtype::U8,
                    ) => Array::zeros::<u8>(&shape, stream),
                    eredu_checkpoint::schema::StoredDtypeConstraint::Floating => {
                        Array::zeros::<f32>(&shape, stream)
                    }
                    dtype => {
                        return Err(Error::UnsupportedArchitecture(format!(
                        "GPT-OSS fixture cannot materialize native expert dtype {dtype:?} for {:?}",
                        tensor.key
                    )))
                    }
                }
                .map_err(Error::from)?;
                Ok(GptOssCheckpointParameter {
                    spec: ParameterSpec::trainable(tensor.key)
                        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
                    trainable: true,
                    value: crate::MlxTensor::from_array(value),
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Self {
            static_modules: architecture.static_modules().clone(),
            layers,
            native_experts,
        })
    }
}

#[cfg(any(test, feature = "test-support"))]
impl Parameterized<crate::MlxTensor> for GptOssCheckpointTemplate {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, crate::MlxTensor>,
    {
        struct NonExpert<'v, V>(&'v mut V);
        impl<'a, V: ParameterVisitor<'a, crate::MlxTensor>> ParameterVisitor<'a, crate::MlxTensor>
            for NonExpert<'_, V>
        {
            fn visit(&mut self, metadata: ParameterMetadata, value: &'a crate::MlxTensor) {
                if !is_expert_parameter(&metadata) {
                    self.0.visit(metadata, value);
                }
            }
        }

        self.static_modules
            .visit_parameters(&mut NonExpert(visitor));
        for layer in &self.layers {
            layer.visit_parameters(&mut NonExpert(visitor));
        }
        for parameter in &self.native_experts {
            visitor.visit(
                ParameterMetadata::from_spec(&parameter.spec, parameter.trainable),
                &parameter.value,
            );
        }
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, crate::MlxTensor>,
    {
        struct NonExpert<'v, V>(&'v mut V);
        impl<'a, V: ParameterVisitorMut<'a, crate::MlxTensor>>
            ParameterVisitorMut<'a, crate::MlxTensor> for NonExpert<'_, V>
        {
            fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut crate::MlxTensor) {
                if !is_expert_parameter(&metadata) {
                    self.0.visit_mut(metadata, value);
                }
            }
        }

        self.static_modules
            .visit_parameters_mut(&mut NonExpert(visitor));
        for layer in &mut self.layers {
            layer.visit_parameters_mut(&mut NonExpert(visitor));
        }
        for parameter in &mut self.native_experts {
            visitor.visit_mut(
                ParameterMetadata::from_spec(&parameter.spec, parameter.trainable),
                &mut parameter.value,
            );
        }
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.static_modules.set_trainable(trainable);
        self.layers.set_trainable(trainable);
        for parameter in &mut self.native_experts {
            parameter.trainable = trainable;
        }
    }
}

#[derive(Clone)]
struct GptOssUnitPopulator {
    external_experts: bool,
}

impl MlxUnitPopulator<NeutralBlock> for GptOssUnitPopulator {
    fn populate(
        &mut self,
        unit: &mut MlxModule<NeutralBlock>,
        lease: &ResidentUnitLease,
    ) -> Result<(), Error> {
        populate_module_from_lease_excluding(unit, lease, |name| {
            self.external_experts && name.contains(".mlp.experts.")
        })?;
        Ok(())
    }
}

#[derive(Clone)]
struct GptOssParallelUnitPopulator {
    external_experts: bool,
}

impl MlxUnitPopulator<NeutralBlock> for GptOssParallelUnitPopulator {
    fn populate(
        &mut self,
        unit: &mut MlxModule<NeutralBlock>,
        lease: &ResidentUnitLease,
    ) -> Result<(), Error> {
        populate_module_from_lease_excluding(unit, lease, |name| {
            self.external_experts && name.contains(".mlp.experts.")
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
struct DenseUnit(NeutralBlock);

fn is_expert_parameter(metadata: &ParameterMetadata) -> bool {
    metadata.id.as_str().contains(".mlp.experts.")
}

impl Parameterized<crate::MlxTensor> for DenseUnit {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, crate::MlxTensor>,
    {
        struct Filter<'v, V>(&'v mut V);
        impl<'a, V: ParameterVisitor<'a, crate::MlxTensor>> ParameterVisitor<'a, crate::MlxTensor>
            for Filter<'_, V>
        {
            fn visit(&mut self, metadata: ParameterMetadata, value: &'a crate::MlxTensor) {
                if !is_expert_parameter(&metadata) {
                    self.0.visit(metadata, value);
                }
            }
        }
        self.0.visit_parameters(&mut Filter(visitor));
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, crate::MlxTensor>,
    {
        struct Filter<'v, V>(&'v mut V);
        impl<'a, V: ParameterVisitorMut<'a, crate::MlxTensor>>
            ParameterVisitorMut<'a, crate::MlxTensor> for Filter<'_, V>
        {
            fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut crate::MlxTensor) {
                if !is_expert_parameter(&metadata) {
                    self.0.visit_mut(metadata, value);
                }
            }
        }
        self.0.visit_parameters_mut(&mut Filter(visitor));
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.0.set_trainable(trainable);
    }
}

/// Reads normalized GPT-OSS arguments from a SafeTensors model directory.
pub fn load_model_args(model_dir: &Path) -> Result<ModelArgs, Error> {
    let file = std::fs::File::open(model_dir.join("config.json"))?;
    eredu_architectures::gpt_oss::model_args_from_config_reader(file)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

fn resolve_safetensors_store(
    store: Arc<dyn CheckpointSource>,
    args: &ModelArgs,
) -> Result<Arc<dyn CheckpointSource>, Error> {
    if store.is_checkpoint_contract_resolved()
        || store.source_diagnostics()?.backend
            != eredu_checkpoint::store::WeightStoreBackend::Safetensors
    {
        return Ok(store);
    }
    let plan = eredu_architectures::gpt_oss::safetensors_plan(args)
        .map_err(Error::UnsupportedArchitecture)?;
    let resolved = eredu_checkpoint::validation::resolve_safetensors_plan(store.as_ref(), &plan)
        .map_err(|validation| {
            Error::UnsupportedArchitecture(format!(
                "GPT-OSS SafeTensors contract did not resolve: {validation:?}"
            ))
        })?;
    Ok(Arc::new(
        eredu_checkpoint::store::ResolvedCheckpointSource::new(store, resolved),
    ))
}

fn unit_recipes(
    store: &dyn CheckpointSource,
    args: &ModelArgs,
    layer: usize,
) -> Result<BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>, Error> {
    eredu_architectures::gpt_oss::expert_recipes(store, args, layer)
        .map(|family| family.into_outputs().into_outputs())
        .map_err(Error::UnsupportedArchitecture)
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
        eredu_architectures::gpt_oss::new_layered_model::<MlxBackend>(args.clone(), stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let factory = GptOssUnitPopulator { external_experts };
    let binding_args = args.clone();
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        store,
        &mut architecture,
        factory,
        std::marker::PhantomData::<MlxKeyValueState>,
        options,
        stream,
        weights_stream,
        move |key| {
            key.starts_with("rope_freqs.")
                || key.ends_with(".rotary_emb.inv_freq")
                || (external_experts && key.contains(".mlp.experts."))
        },
        |modules, store| {
            build_module_bindings(&MlxModule::new(modules.clone()), "", store).map_err(Into::into)
        },
        move |address, _path, unit, store, _stream| {
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
                |name| external_experts && name.contains(".mlp.experts."),
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
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
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
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
    external_experts: bool,
) -> Result<GptOssModel, Error> {
    let layer_count = usize::try_from(args.num_hidden_layers)
        .map_err(|_| Error::UnsupportedArchitecture("invalid GPT-OSS layer count".into()))?;
    let global_architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let mut planner = build.planner();
    for group in eredu_architectures::gpt_oss::static_parameter_groups::<MlxBackend>(
        global_architecture.static_modules(),
        &args,
    )? {
        planner.register(group)?;
    }
    for layer in 0..layer_count {
        let block = eredu_architectures::gpt_oss::new_block::<MlxBackend>(&args, layer, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        for group in eredu_architectures::gpt_oss::layer_parallel_parameter_groups::<MlxBackend>(
            &block, &args, layer,
        )? {
            planner.register(group)?;
        }
    }
    let (_, layout) = planner.finish()?;
    if layout.is_empty() {
        return Err(Error::Parallel(
            "GPT-OSS declared no tensor-parallel parameters".into(),
        ));
    }
    let geometry = eredu_architectures::gpt_oss::local_geometry(&args, &layout)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let mut architecture = NeutralArchitecture::new_parallel(args.clone(), geometry, stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let state_layout = architecture
        .runtime_state_layout()
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let factory = GptOssParallelUnitPopulator { external_experts };

    let global_static = MlxModule::new(global_architecture.static_modules().clone());
    let mut global_parameter_bytes =
        binding_bytes(&build_module_bindings(&global_static, "", store.as_ref())?)?;
    for layer in 0..layer_count {
        let block = eredu_architectures::gpt_oss::new_block::<MlxBackend>(&args, layer, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let recipes = if external_experts {
            BTreeMap::new()
        } else {
            unit_recipes(store.as_ref(), &args, layer)?
        };
        let bindings = build_module_bindings_with_recipes_excluding(
            &MlxModule::new(block),
            "",
            store.as_ref(),
            recipes,
            |name| external_experts && name.contains(".mlp.experts."),
        )?;
        global_parameter_bytes = global_parameter_bytes
            .checked_add(binding_bytes(&bindings)?)
            .ok_or_else(|| Error::Parallel("global GPT-OSS parameter bytes overflowed".into()))?;
    }

    let binding_args = args.clone();
    let global_static_modules = global_architecture.static_modules().clone();
    let local_layout = Arc::new(layout);
    let static_layout = Arc::clone(&local_layout);
    let unit_local_layout = Arc::clone(&local_layout);
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        Arc::clone(&store),
        &mut architecture,
        factory,
        std::marker::PhantomData::<MlxKeyValueState>,
        options,
        stream,
        weights_stream,
        move |key| {
            key.starts_with("rope_freqs.")
                || key.ends_with(".rotary_emb.inv_freq")
                || (external_experts && key.contains(".mlp.experts."))
        },
        move |_modules, store| {
            let global = MlxModule::new(global_static_modules.clone());
            let bindings = build_module_bindings(&global, "", store)?;
            shard_layer_bindings(bindings, "", store, &static_layout)
        },
        move |address, path, _local, store, stream| {
            let layer = address.index();
            let global =
                eredu_architectures::gpt_oss::new_block::<MlxBackend>(&binding_args, layer, stream)
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
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
                |name| external_experts && name.contains(".mlp.experts."),
            )?;
            shard_layer_bindings(bindings, path, store, &unit_local_layout)
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
        crate::backend::mlx::cache::prompt_cache_topology(build.topology()).cache_rank_identity();
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
        .then(|| expert::expert_catalog_cartesian(&args, store.as_ref(), Some(&local_layout)))
        .transpose()?;
    Ok(GptOssModel {
        args,
        state_layout,
        metadata,
        parallel_info: Some(parallel_info),
        parallel_rank,
        planned_external_experts,
        prompt_cache_topology: crate::backend::mlx::cache::prompt_cache_topology(build.topology()),
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
    let mut target_args = source_args.clone();
    target_args.quantization = Some(quantization);
    let source =
        eredu_architectures::gpt_oss::new_layered_model::<MlxBackend>(source_args.clone(), stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let target =
        eredu_architectures::gpt_oss::new_layered_model::<MlxBackend>(target_args.clone(), stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let count = usize::try_from(source_args.num_hidden_layers)
        .map_err(|_| Error::UnsupportedArchitecture("invalid GPT-OSS layer count".into()))?;
    let source_unit_args = source_args.clone();
    let target_unit_args = target_args.clone();
    let (store, report) = quantize_parameterized_store(
        store,
        source.static_modules(),
        target.static_modules(),
        move |index, stream| {
            eredu_architectures::gpt_oss::new_block::<MlxBackend>(&source_unit_args, index, stream)
                .map(DenseUnit)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        },
        move |index, stream| {
            eredu_architectures::gpt_oss::new_block::<MlxBackend>(&target_unit_args, index, stream)
                .map(DenseUnit)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        },
        count,
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

    pub fn static_units(
        &self,
        architecture: &NeutralArchitecture,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        self.selected_static_units(architecture, store, &|_| true)
    }

    pub fn selected_static_units(
        &self,
        architecture: &NeutralArchitecture,
        store: &dyn CheckpointSource,
        select: &dyn Fn(&str) -> bool,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        let roles = [
            ("embedding", "gpt_oss.static.embedding"),
            ("norm", "gpt_oss.static.norm"),
            ("output", "gpt_oss.static.output"),
        ]
        .into_iter()
        .filter_map(|(role, unit)| select(unit).then_some(role))
        .collect::<Vec<_>>();
        self.selected_static_units_for_roles(architecture, store, &roles)
    }

    pub fn quantizes_static_binding(&self, _binding: &WeightBinding) -> bool {
        true
    }

    pub fn expert_parallel_assignment(
        &self,
        architecture: &NeutralArchitecture,
        topology: crate::backend::mlx::MlxParallelContext,
    ) -> Result<Option<crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>, Error>
    {
        if topology.expert_parallel_size == 1 && !self.external_experts {
            return Ok(None);
        }
        Ok(Some(
            crate::backend::mlx::runtime::distributed::expert::ExpertAssignment::balanced(
                architecture.args().num_local_experts as usize,
                topology.expert_parallel_size,
                topology.expert_parallel_rank,
            )?,
        ))
    }

    /// Applies rank-local expert residency to an architecture-constructed unit.
    pub fn prepare_unit_expert_residency(
        &self,
        architecture: &NeutralArchitecture,
        index: usize,
        layer: &mut MlxModule<NeutralBlock>,
        local_intermediate_size: i32,
        assignment: Option<&crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let args = architecture.args();
        if let Some(assignment) = assignment {
            let count = i32::try_from(assignment.local_global_expert_ids().len())
                .map_err(|_| Error::Parallel("local GPT-OSS expert count exceeds i32".into()))?;
            if count <= 0 {
                return Err(Error::Parallel(
                    "GPT-OSS expert-parallel rank owns no experts".into(),
                ));
            }
            layer.inner.mlp.experts =
                <MlxBackend as RoutedNeuralBackend>::gated_product_expert_bank(
                    eredu_architectures::gpt_oss::localized_expert_bank_spec(
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
        Ok(())
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
        assignment: Option<&crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>,
    ) -> Result<Vec<WeightBinding>, Error> {
        require_decoder_group(architecture, group)?;
        let expert_targets = parameter_role_targets(
            &eredu_architectures::gpt_oss::layer_parallel_parameter_groups(
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
            unit_recipes(store, architecture.args(), index)?
        };
        let mut bindings = build_module_binding_plan_with_recipes_excluding(
            global_layer,
            "",
            store,
            recipes,
            |name| self.external_experts && parameter_name_in_targets(name, &expert_targets),
        )?
        .build_bindings(store)?;
        if let Some(assignment) = assignment {
            let indices = assignment.local_global_expert_ids().to_vec();
            bindings = bindings
                .into_iter()
                .map(|binding| {
                    let target = binding.logical_target().unwrap_or_else(|| binding.name());
                    if parameter_name_in_targets(target, &expert_targets) {
                        binding
                            .select_bounded_output(
                                store,
                                TensorSelection::Indices {
                                    axis: 0,
                                    indices: indices.clone(),
                                },
                            )
                            .map_err(Error::from)
                    } else {
                        Ok(binding)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
        }
        match layout {
            Some(layout) => shard_layer_bindings(
                bindings,
                &format!("{}.layers.{index}", architecture.args().parameter_root),
                store,
                layout,
            ),
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

    fn selected_static_units_for_roles(
        &self,
        architecture: &NeutralArchitecture,
        store: &dyn CheckpointSource,
        roles: &[&str],
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        let selected = |role: &str| roles.contains(&role);
        let args = architecture.args();
        let static_modules = architecture.static_modules();
        let mut units = Vec::new();
        if selected("embedding") {
            units.push(StaticUnitBindings::new(
                "gpt_oss.static.embedding",
                build_module_binding_plan_with_recipes(
                    &static_modules.embeddings,
                    &format!("{}.embed_tokens", args.parameter_root),
                    store,
                    Default::default(),
                )?
                .build_bindings(store)?,
            )?);
        }
        if selected("norm") {
            let norm_root = format!("{}.norm.", args.parameter_root);
            let bindings = build_module_binding_plan_with_recipes(
                &static_modules.norm,
                "",
                store,
                Default::default(),
            )?
            .build_bindings(store)?
            .into_iter()
            .map(|binding| {
                let local = binding
                    .name()
                    .strip_prefix(&norm_root)
                    .unwrap_or(binding.name())
                    .to_owned();
                binding.with_name(local).map_err(Error::from)
            })
            .collect::<Result<Vec<_>, Error>>()?;
            units.push(StaticUnitBindings::new("gpt_oss.static.norm", bindings)?);
        }
        if selected("output") {
            let head = static_modules.lm_head.as_ref().ok_or_else(|| {
                Error::UnsupportedArchitecture("GPT-OSS requires a separate LM head".into())
            })?;
            units.push(StaticUnitBindings::new(
                "gpt_oss.static.output",
                build_module_binding_plan_with_recipes(head, "lm_head", store, Default::default())?
                    .build_bindings(store)?,
            )?);
        }
        Ok(units)
    }
}

/// Neutral GPT-OSS causal LM with resident or bounded layer execution.
pub struct GptOssModel {
    args: ModelArgs,
    state_layout: eredu_runtime::StateLayout,
    metadata: LayerwiseModelMetadata,
    parallel_info: Option<ParallelModelInfo<crate::backend::mlx::MlxParallelContext>>,
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

    pub fn parallel_info(
        &self,
    ) -> Option<&ParallelModelInfo<crate::backend::mlx::MlxParallelContext>> {
        self.parallel_info.as_ref()
    }

    /// Builds expert-cache units with this rank's exact TP selections.
    pub fn external_expert_catalog(&self) -> Result<Vec<ExpertCatalogEntry>, Error> {
        self.planned_external_experts.clone().map_or_else(
            || expert::expert_catalog_cartesian(&self.args, self.checkpoint_store(), None),
            Ok,
        )
    }

    /// Records the complete distributed coordinates used by EP/TP+EP wrappers.
    pub fn bind_parallel_topology(&mut self, topology: crate::backend::mlx::MlxParallelContext) {
        self.parallel_rank =
            crate::backend::mlx::cache::prompt_cache_topology(topology).cache_rank_identity();
        self.prompt_cache_topology = crate::backend::mlx::cache::prompt_cache_topology(topology);
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
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string())),
            GptOssExecution::Layerwise(execution) => execution
                .forward(input, cache, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string())),
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
        P: eredu_runtime::RoutedExpertProvider<MlxBackend>,
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
                    MlxBackend,
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
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string())),
            GptOssExecution::Layerwise(runtime) => runtime
                .forward_with_unit_executor(input, cache, stream, hook)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string())),
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
        P: eredu_runtime::RoutedExpertProvider<MlxBackend>,
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
                    MlxBackend,
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
        let mut observer = NeutralGptOssObserver { inner: observer };
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
        observer: &mut NeutralGptOssObserver<'_>,
    ) -> Result<Array, Error>
    where
        P: eredu_runtime::RoutedExpertProvider<MlxBackend>,
        P::Error: std::fmt::Display,
    {
        self.validate_cache(cache)?;
        let pass = if inputs.dim(1) > 1 {
            eredu_runtime::ExpertPass::Prefill
        } else {
            eredu_runtime::ExpertPass::Decode
        };
        let expert_count = self.args.num_local_experts;
        let output = match &mut self.execution {
            GptOssExecution::Resident(runtime) => runtime
                .forward_with_routed_observer(
                    eredu_architectures::decoder::LayeredInput {
                        tokens: crate::composition::tensor_ref(inputs),
                        mask: crate::composition::tensor_opt(mask),
                    },
                    cache,
                    pass,
                    provider,
                    stream,
                    observer,
                    |path, _, _| {
                        Some(eredu_runtime::RoutedObservationPoint::new(
                            format!("{path}.mlp"),
                            expert_count,
                        ))
                    },
                )
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
            GptOssExecution::Layerwise(runtime) => runtime
                .forward_with_routed_observer(
                    eredu_architectures::decoder::LayeredInput {
                        tokens: crate::composition::tensor_ref(inputs),
                        mask: crate::composition::tensor_opt(mask),
                    },
                    cache,
                    pass,
                    provider,
                    stream,
                    observer,
                    |path, _, _| {
                        Some(eredu_runtime::RoutedObservationPoint::new(
                            format!("{path}.mlp"),
                            expert_count,
                        ))
                    },
                )
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
            GptOssExecution::TensorParallelResident(_)
            | GptOssExecution::TensorParallelLayerwise(_) => {
                return Err(Error::Parallel(
                    "tensor-parallel GPT-OSS observation requires a collective observer path"
                        .into(),
                ))
            }
        };
        eredu_runtime::observe_and_intervene(observer, "model.logits", &output)
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
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
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
pub fn load_gpt_oss_safetensors_mlx(
    model_dir: impl AsRef<Path>,
    weight_residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssModel, Error> {
    let model_dir = model_dir.as_ref();
    let expert_options = weight_residency.expert_cache();
    let execution_options = weight_residency.layers();
    let args = load_model_args(model_dir)?;
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load("GPT-OSS", args.quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let store = open_safetensors_weight_store(model_dir, execution_options.max_mapped_shards())?;
    let store = resolve_safetensors_store(store, &args)?;
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

/// Loads a SafeTensors checkpoint through unified layered residency.
pub fn load_gpt_oss_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssModel, Error> {
    load_gpt_oss_safetensors_mlx(
        model_dir,
        WeightResidency::with_layers(options.into()),
        quantization,
        stream,
        weights_stream,
    )
}

/// Loads GPT-OSS with experts managed independently from ordinary blocks.
pub fn load_gpt_oss_expert_cache_model(
    model_dir: impl AsRef<Path>,
    non_expert: eredu_runtime::NonExpertWeightResidency,
    options: eredu_runtime::ExpertCacheLoadOptions,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssModel, Error> {
    load_gpt_oss_safetensors_mlx(
        model_dir,
        WeightResidency::with_expert_cache(non_expert, options),
        quantization,
        stream,
        weights_stream,
    )
}

/// Loads SafeTensors or an inspected GGUF through the neutral GPT-OSS tensor-parallel graph.
pub fn load_gpt_oss_tensor_parallel_model(
    model_path: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssModel, Error> {
    let model_path = model_path.as_ref();
    let options = options.into();
    if model_path
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        let admitted = crate::composition::mlx::structural::admit_gguf_path(
            model_path,
            crate::backend::mlx::ModelLoadOptions::default()
                .with_weight_residency(WeightResidency::with_layers(options)),
        )?;
        return load_gpt_oss_gguf_tensor_parallel_model(
            &admitted,
            options,
            build,
            stream,
            weights_stream,
        )
        .map(|(model, _)| model);
    }
    let args = load_model_args(model_path)?;
    let store = open_safetensors_weight_store(model_path, options.max_mapped_shards())?;
    let store = resolve_safetensors_store(store, &args)?;
    load_neutral_parallel_with_store(store, args, options, build, stream, weights_stream, false)
}

/// Loads only static and nonexpert block weights for an external provider.
pub fn load_external_experts_with_store(
    store: Arc<dyn CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    build: Option<crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssModel, Error> {
    let store = resolve_safetensors_store(store, &args)?;
    match build {
        Some(build) if build.topology().tensor_parallel_size > 1 => {
            load_neutral_parallel_with_store(
                store,
                args,
                options,
                build,
                stream,
                weights_stream,
                true,
            )
        }
        _ => load_neutral_with_store(store, args, options, stream, weights_stream, None, true),
    }
}

/// Header-only results needed to open a portable GGUF GPT-OSS checkpoint.
pub(crate) struct PreparedGptOssGguf {
    pub args: ModelArgs,
    pub eos_token_ids: Vec<u32>,
}

/// Validates and normalizes portable GGUF metadata without reading payloads.
pub(crate) fn prepare_gpt_oss_gguf_checkpoint(
    source: &crate::composition::mlx::structural::AdmittedGguf,
) -> Result<PreparedGptOssGguf, Error> {
    if source.architecture() != eredu_core::GgufArchitecture::GptOss {
        return Err(Error::UnsupportedArchitecture(format!(
            "GPT-OSS GGUF loader received architecture {:?}",
            source.architecture()
        )));
    }
    let checkpoint = source.checkpoint();
    let metadata = source.metadata();
    let mut args = eredu_architectures::gpt_oss::model_args_from_gguf_catalog(metadata)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    match eredu_architectures::gpt_oss::validate_gguf(checkpoint, &args) {
        eredu_checkpoint::validation::CheckpointValidation::Exact => {}
        validation => {
            return Err(Error::UnsupportedArchitecture(format!(
                "GPT-OSS GGUF checkpoint contract did not resolve: {validation:?}"
            )))
        }
    }
    let translate = eredu_architectures::gpt_oss::translate_gguf_weight_name;
    checkpoint
        .catalog()
        .translated_outputs(translate)
        .map_err(safemlx::error::IoError::from)?;
    let mut configs = gguf_quantization_configs(checkpoint, translate)?;
    configs.retain(|name, _| !name.contains(".mlp.experts."));
    args.quantized_weight_configs = Some(configs);
    args.quantization = None;
    args.validate()
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    Ok(PreparedGptOssGguf {
        args,
        eos_token_ids: crate::composition::mlx::gguf_eos_token_ids(metadata)?,
    })
}

/// Loads a GGUF checkpoint through the same neutral model/runtime object.
pub(crate) fn load_gpt_oss_gguf_model(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(GptOssModel, Vec<u32>), Error> {
    let checkpoint = source.checkpoint();
    let prepared = prepare_gpt_oss_gguf_checkpoint(source)?;
    let plan = eredu_architectures::gpt_oss::gguf_plan(&prepared.args)
        .map_err(Error::UnsupportedArchitecture)?;
    let store: Arc<dyn CheckpointSource> = Arc::new(open_gguf_checkpoint_source(
        checkpoint.clone(),
        &plan,
        eredu_architectures::gpt_oss::translate_gguf_weight_name,
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
    Ok((model, prepared.eos_token_ids))
}

/// Loads a validated GGUF checkpoint through the neutral tensor-parallel graph.
pub(crate) fn load_gpt_oss_gguf_tensor_parallel_model(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    options: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(GptOssModel, Vec<u32>), Error> {
    let checkpoint = source.checkpoint();
    let prepared = prepare_gpt_oss_gguf_checkpoint(source)?;
    let plan = eredu_architectures::gpt_oss::gguf_plan(&prepared.args)
        .map_err(Error::UnsupportedArchitecture)?;
    let store: Arc<dyn CheckpointSource> = Arc::new(open_gguf_checkpoint_source(
        checkpoint.clone(),
        &plan,
        eredu_architectures::gpt_oss::translate_gguf_weight_name,
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
    Ok((model, prepared.eos_token_ids))
}

/// Loads portable GGUF weights with the requested unified residency policy.
pub(crate) fn load_gpt_oss_gguf_layerwise_model(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(GptOssModel, Vec<u32>), Error> {
    load_gpt_oss_gguf_model(source, residency, quantization, stream, weights_stream)
}

//! Production MLX composition for the backend-neutral Moshi-family model.

use std::{collections::BTreeMap, path::Path, sync::Arc};

use eredu_architectures::moshi::{self, LayeredModel, MoshiConfig, RealtimePreparationPlan, Unit};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{CheckpointSource, ResolvedCheckpointSource, SharedCheckpointSource},
};
use eredu_nn::Parameterized;
use eredu_runtime::{
    ArchitectureParameters, DenseDiskStreamReport, ExecutionUnitLayout, LayeredArchitecture,
    LayeredTraversalHook, LayerwiseModelMetadata, LayerwiseRuntime, ResidencyReport,
    ResidentLayerGroupReport, Sampler, SequentialDecisionDriver, SequentialDecisionTraversal,
    ShardingPolicy,
};
use safemlx::{Array, Stream};

use crate::backend::{
    error::Error,
    nn::shared::{MlxModule, MlxNeuralBackend},
    runtime::{
        cache::state::MlxKeyValueState,
        checkpoint::{
            artifact::{fingerprint_artifact, ArtifactFile, LoadedArtifactIdentity},
            binding::{
                build_neutral_module_bindings_with_recipes_excluding, canonical_checkpoint_name,
            },
            quantization::should_quantize_on_load,
        },
        execution::{
            generic::{
                prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
            },
            layerwise::{
                open_safetensors_weight_store, quantize_module_store_with_bindings,
                shard_layer_bindings,
            },
        },
        generation::sampler::MlxSamplingBackend,
    },
    ModelLoadOptions,
};

type Architecture = LayeredModel<MlxNeuralBackend>;
type MoshiUnit = Unit<MlxNeuralBackend>;
type ResidentRuntime = LayerwiseRuntime<
    Architecture,
    MlxNeuralBackend,
    MlxKeyValueState,
    MlxResidentPolicy<MoshiUnit>,
>;
type BoundedRuntime = LayerwiseRuntime<
    Architecture,
    MlxNeuralBackend,
    MlxKeyValueState,
    MlxLayerwisePolicy<MoshiUnit>,
>;
type ParallelResidentRuntime = LayerwiseRuntime<
    Architecture,
    MlxNeuralBackend,
    MlxKeyValueState,
    MlxResidentPolicy<MoshiUnit>,
>;
type ParallelBoundedRuntime = LayerwiseRuntime<
    Architecture,
    MlxNeuralBackend,
    MlxKeyValueState,
    MlxLayerwisePolicy<MoshiUnit>,
>;

enum Execution {
    Resident(ResidentRuntime),
    Bounded(BoundedRuntime),
    ParallelResident(Box<ParallelResidentRuntime>),
    ParallelBounded(Box<ParallelBoundedRuntime>),
}

#[derive(Clone)]
struct CanonicalBindingRecipes {
    outputs: BTreeMap<String, DerivedWeightRecipe>,
    aliases: BTreeMap<String, String>,
}

/// Stable source-artifact and execution-topology identities for a loaded model.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MoshiModelIdentity {
    source_architecture: String,
    execution_architecture: String,
}

impl MoshiModelIdentity {
    /// Fingerprint of the unmodified source artifact configuration.
    pub fn source_architecture(&self) -> &str {
        &self.source_architecture
    }

    /// Fingerprint of the materialized execution topology.
    pub fn execution_architecture(&self) -> &str {
        &self.execution_architecture
    }
}

/// One neutral Moshi-family MLX model across every replicated residency policy.
pub struct MoshiModel {
    source_config: MoshiConfig,
    target_config: MoshiConfig,
    identity: MoshiModelIdentity,
    artifact_identity: LoadedArtifactIdentity,
    state_layout: eredu_runtime::StateLayout,
    metadata: LayerwiseModelMetadata,
    topology: Option<crate::backend::MlxParallelContext>,
    execution: Execution,
}

impl MoshiModel {
    /// Original normalized artifact policy used for physical checkpoint planning.
    pub fn source_config(&self) -> &MoshiConfig {
        &self.source_config
    }

    /// Materialized execution policy, including optional load-time quantization.
    pub fn config(&self) -> &MoshiConfig {
        &self.target_config
    }

    /// Stable source and execution identities.
    pub fn identity(&self) -> &MoshiModelIdentity {
        &self.identity
    }

    /// Canonical runtime state layout.
    pub fn state_layout(&self) -> &eredu_runtime::StateLayout {
        &self.state_layout
    }

    /// Parameter topology and residency metadata.
    pub fn metadata(&self) -> &LayerwiseModelMetadata {
        &self.metadata
    }

    /// Rank-local topology when this instance owns tensor-parallel parameters.
    pub fn topology(&self) -> Option<crate::backend::MlxParallelContext> {
        self.topology
    }

    /// Logical residency and transfer telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        match &self.execution {
            Execution::Resident(runtime) => runtime.policy().residency_report(),
            Execution::Bounded(runtime) => runtime.policy().residency_report(),
            Execution::ParallelResident(runtime) => runtime.policy().residency_report(),
            Execution::ParallelBounded(runtime) => runtime.policy().residency_report(),
        }
    }

    /// Disk-stream telemetry when that policy is active.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        match &self.execution {
            Execution::Resident(_) => Ok(None),
            Execution::Bounded(runtime) => runtime.policy().dense_stream_report(),
            Execution::ParallelResident(_) => Ok(None),
            Execution::ParallelBounded(runtime) => runtime.policy().dense_stream_report(),
        }
    }

    /// Per-execution-group residency reports.
    pub fn execution_group_reports(&self) -> Result<Vec<ResidentLayerGroupReport>, Error> {
        match &self.execution {
            Execution::Resident(runtime) => runtime.policy().execution_group_reports(),
            Execution::Bounded(runtime) => runtime.policy().execution_group_reports(),
            Execution::ParallelResident(runtime) => runtime.policy().execution_group_reports(),
            Execution::ParallelBounded(runtime) => runtime.policy().execution_group_reports(),
        }
    }

    pub fn artifact_identity(&self) -> &LoadedArtifactIdentity {
        &self.artifact_identity
    }

    /// Creates request-local resident key/value state from the neutral layout.
    pub fn new_realtime_state(&self) -> Result<MlxKeyValueState, Error> {
        MlxKeyValueState::device(self.state_layout.clone()).map_err(Into::into)
    }

    /// Runs one neutral temporal/depth pass through the shared decision traversal.
    pub fn forward_realtime<'a, S>(
        &mut self,
        input: moshi::Input<'a, Array>,
        state: &mut MlxKeyValueState,
        driver: &mut SequentialDecisionDriver<MlxSamplingBackend, S>,
        stream: &Stream,
    ) -> Result<(Array, moshi::ForwardContext<crate::MlxTensor>), Error>
    where
        S: Sampler<MlxSamplingBackend>,
    {
        let mut boundary = moshi::DecisionBoundary::new(&self.target_config)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let mut traversal = SequentialDecisionTraversal::new(driver, &mut boundary);
        self.forward_realtime_with_traversal_hook(input, state, &mut traversal, stream)
    }

    pub fn forward_realtime_with_traversal_hook<'a, H>(
        &mut self,
        input: moshi::Input<'a, Array>,
        state: &mut MlxKeyValueState,
        hook: &mut H,
        stream: &Stream,
    ) -> Result<(Array, moshi::ForwardContext<crate::MlxTensor>), Error>
    where
        H: LayeredTraversalHook<
            MlxNeuralBackend,
            moshi::ForwardContext<crate::MlxTensor>,
            eredu_nn::Error,
        >,
    {
        let text = crate::MlxTensor::from_array(input.text.clone());
        let audio = input
            .audio
            .iter()
            .map(|token| crate::MlxTensor::from_array((*token).clone()))
            .collect::<Vec<_>>();
        let audio = audio.iter().collect::<Vec<_>>();
        let mask = input.mask.cloned().map(crate::MlxTensor::from_array);
        let input = moshi::Input {
            text: &text,
            audio: &audio,
            mask: mask.as_ref(),
        };
        match &mut self.execution {
            Execution::Resident(runtime) => runtime
                .forward_with_traversal_hook(input, state, stream, hook)
                .map(|(output, context)| (output.into_array(), context))
                .map_err(|error| Error::ArchitectureModel(error.to_string())),
            Execution::Bounded(runtime) => runtime
                .forward_with_traversal_hook(input, state, stream, hook)
                .map(|(output, context)| (output.into_array(), context))
                .map_err(|error| Error::ArchitectureModel(error.to_string())),
            Execution::ParallelResident(_) | Execution::ParallelBounded(_) => Err(Error::Parallel(
                "tensor-parallel Moshi execution requires the rank's TP collective group".into(),
            )),
        }
    }

    /// Runs one rank-local tensor-parallel temporal/depth pass.
    pub fn forward_realtime_parallel<'a, S>(
        &mut self,
        input: moshi::Input<'a, Array>,
        state: &mut MlxKeyValueState,
        driver: &mut SequentialDecisionDriver<MlxSamplingBackend, S>,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<(Array, moshi::ForwardContext<crate::MlxTensor>), Error>
    where
        S: Sampler<MlxSamplingBackend>,
    {
        let mut boundary = moshi::DecisionBoundary::new(&self.target_config)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let mut traversal = SequentialDecisionTraversal::new(driver, &mut boundary);
        let text = crate::MlxTensor::from_array(input.text.clone());
        let audio = input
            .audio
            .iter()
            .map(|token| crate::MlxTensor::from_array((*token).clone()))
            .collect::<Vec<_>>();
        let audio = audio.iter().collect::<Vec<_>>();
        let mask = input.mask.cloned().map(crate::MlxTensor::from_array);
        let input = moshi::Input {
            text: &text,
            audio: &audio,
            mask: mask.as_ref(),
        };
        match &mut self.execution {
            Execution::ParallelResident(runtime) => runtime
                .forward_parallel_with_traversal_hook(input, state, group, stream, &mut traversal)
                .map(|(output, context)| (output.into_array(), context))
                .map_err(|error| Error::Parallel(error.to_string())),
            Execution::ParallelBounded(runtime) => runtime
                .forward_parallel_with_traversal_hook(input, state, group, stream, &mut traversal)
                .map(|(output, context)| (output.into_array(), context))
                .map_err(|error| Error::Parallel(error.to_string())),
            Execution::Resident(_) | Execution::Bounded(_) => Err(Error::Parallel(
                "model was not loaded for tensor-parallel Moshi execution".into(),
            )),
        }
    }
}

/// Loads either admitted Moshi-family SafeTensors layout into the neutral model.
pub fn load(
    preparation: RealtimePreparationPlan,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MoshiModel, Error> {
    let (model_dir, source_path, source_config, checkpoint_plan, source_recipes) =
        preparation.into_parts();
    let artifact_identity = artifact_identity(&model_dir, &source_path, &source_config)?;
    let source_store = open_safetensors_weight_store(
        &source_path,
        options.weight_residency.layers().max_mapped_shards(),
    )?;
    let checkpoint_contract = eredu_checkpoint::validation::resolve_safetensors_plan(
        source_store.as_ref(),
        &checkpoint_plan,
    )
    .map_err(|validation| {
        Error::ArchitectureModel(format!(
            "prepared Moshi checkpoint contract no longer resolves: {validation:?}"
        ))
    })?;
    let source_store: SharedCheckpointSource = Arc::new(ResolvedCheckpointSource::new(
        source_store,
        checkpoint_contract,
    ));
    let (source_outputs, source_aliases) = source_recipes.into_parts();
    let source_recipes = Arc::new(CanonicalBindingRecipes {
        outputs: source_outputs,
        aliases: source_aliases,
    });

    let quantize = options
        .quantization
        .map(|requested| {
            should_quantize_on_load(
                "Moshi family",
                source_config.native_quantization(),
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let target_config = match quantize {
        Some(quantization) => source_config
            .with_native_quantization(Some(quantization))
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
        None => source_config.clone(),
    };

    let (store, materialization) = match quantize {
        Some(quantization) => {
            let source_architecture = Architecture::new(source_config.clone(), stream)?;
            let target_architecture = Architecture::new(target_config.clone(), stream)?;
            let layout = execution_layout(&source_architecture)?;
            let source_unit_config = source_config.clone();
            let target_unit_config = target_config.clone();
            let source_unit_layout = layout.clone();
            let target_unit_layout = layout.clone();
            let static_recipes = Arc::clone(&source_recipes);
            let unit_recipes = Arc::clone(&source_recipes);
            let (store, report) = quantize_module_store_with_bindings(
                Arc::clone(&source_store),
                &MlxModule::new(source_architecture.static_modules().clone()),
                &MlxModule::new(target_architecture.static_modules().clone()),
                move |ordinal, stream| {
                    build_unit(&source_unit_config, &source_unit_layout, ordinal, stream)
                        .map(MlxModule::new)
                },
                move |ordinal, stream| {
                    build_unit(&target_unit_config, &target_unit_layout, ordinal, stream)
                        .map(MlxModule::new)
                },
                layout.len(),
                quantization,
                stream,
                move |module, store| bindings(module, store, static_recipes.as_ref()),
                move |_ordinal, module, store| bindings(module, store, unit_recipes.as_ref()),
            )?;
            (store, Some(report))
        }
        None => (source_store, None),
    };

    if let Some(topology) = options
        .parallel_topology()
        .filter(|topology| !topology.is_replicated())
    {
        return load_parallel(
            store,
            source_config,
            target_config,
            source_recipes,
            artifact_identity,
            materialization,
            options.weight_residency.layers(),
            topology,
            stream,
            weights_stream,
        );
    }
    options.validate_replicated()?;

    let mut architecture = Architecture::new(target_config.clone(), stream)?;
    let static_recipes = Arc::clone(&source_recipes);
    let unit_recipes = Arc::clone(&source_recipes);
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        store,
        &mut architecture,
        (),
        std::marker::PhantomData::<MlxKeyValueState>,
        options.weight_residency.layers(),
        stream,
        weights_stream,
        |_| false,
        move |modules, store| bindings(modules, store, static_recipes.as_ref()),
        move |_ordinal, _address, _path, unit, store, _stream| {
            bindings(&unit, store, unit_recipes.as_ref())
        },
    )?;
    metadata.set_model_type(target_config.effective_model_type().as_str());
    metadata.set_quantization(target_config.native_quantization());
    metadata.set_materialization(materialization);
    let state_layout = moshi::state_layout(&target_config)?;
    let execution = if options.weight_residency.layers().is_fully_resident() {
        Execution::Resident(LayerwiseRuntime::new_policy_first(
            policy.into_resident(
                &architecture,
                stream,
                std::marker::PhantomData::<MlxKeyValueState>,
            )?,
            architecture,
        ))
    } else {
        Execution::Bounded(LayerwiseRuntime::new(architecture, policy))
    };
    let identity = MoshiModelIdentity {
        source_architecture: source_config.architecture_fingerprint().to_owned(),
        execution_architecture: target_config.architecture_fingerprint().to_owned(),
    };
    Ok(MoshiModel {
        source_config,
        target_config,
        identity,
        artifact_identity,
        state_layout,
        metadata,
        topology: None,
        execution,
    })
}

#[allow(clippy::too_many_arguments)]
fn load_parallel(
    store: SharedCheckpointSource,
    source_config: MoshiConfig,
    target_config: MoshiConfig,
    source_recipes: Arc<CanonicalBindingRecipes>,
    artifact_identity: LoadedArtifactIdentity,
    materialization: Option<eredu_runtime::WeightMaterializationReport>,
    residency: eredu_runtime::LayerWeightResidency,
    topology: crate::backend::MlxParallelContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MoshiModel, Error> {
    if topology.pipeline_parallel_size != 1 || topology.expert_parallel_size != 1 {
        return Err(Error::Parallel(format!(
            "neutral Moshi supports tensor parallelism only, got TP/PP/EP={}/{}/{}",
            topology.tensor_parallel_size,
            topology.pipeline_parallel_size,
            topology.expert_parallel_size
        )));
    }
    topology.validate_execution_stream(stream)?;
    let build = crate::backend::runtime::distributed::parallel::ParallelBuildContext::new(
        topology,
        ShardingPolicy::Require,
    );
    let global = Architecture::new(target_config.clone(), stream)?;
    let parameter_description = global
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let local_layout =
        crate::composition::parallel_layout_from_description(build, &parameter_description)?;
    if local_layout.is_empty() {
        return Err(Error::Parallel(
            "Moshi declared no tensor-parallel parameters".into(),
        ));
    }
    let geometry = moshi::local_geometry(
        &target_config,
        &local_layout,
        source_recipes
            .aliases
            .iter()
            .map(|(alias, owner)| (alias.as_str(), owner.as_str())),
    )
    .map_err(|error| Error::Parallel(error.to_string()))?;
    let mut composition = Architecture::new_parallel(target_config.clone(), geometry, stream)?;
    let state_layout = composition.state_layout()?;
    let local_layout = Arc::new(local_layout);
    let static_module = global.static_modules().clone();
    let static_recipes = Arc::clone(&source_recipes);
    let static_sharding = Arc::clone(&local_layout);
    let unit_config = target_config.clone();
    let unit_recipes = Arc::clone(&source_recipes);
    let unit_sharding = Arc::clone(&local_layout);
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        store,
        &mut composition,
        (),
        std::marker::PhantomData::<MlxKeyValueState>,
        residency,
        stream,
        weights_stream,
        |_| false,
        move |_modules, store| {
            shard_layer_bindings(
                bindings(&static_module, store, static_recipes.as_ref())?,
                store,
                static_sharding.as_ref(),
            )
        },
        move |_ordinal, address, _path, _local, store, stream| {
            let global = build_addressed_unit(&unit_config, address, stream)?;
            shard_layer_bindings(
                bindings(&global, store, unit_recipes.as_ref())?,
                store,
                unit_sharding.as_ref(),
            )
        },
    )?;
    metadata.set_model_type(target_config.effective_model_type().as_str());
    metadata.set_quantization(target_config.native_quantization());
    metadata.set_materialization(materialization);
    let execution = if residency.is_fully_resident() {
        Execution::ParallelResident(Box::new(LayerwiseRuntime::new_policy_first(
            policy.into_resident(
                &composition,
                stream,
                std::marker::PhantomData::<MlxKeyValueState>,
            )?,
            composition,
        )))
    } else {
        Execution::ParallelBounded(Box::new(LayerwiseRuntime::new(composition, policy)))
    };
    let identity = MoshiModelIdentity {
        source_architecture: source_config.architecture_fingerprint().to_owned(),
        execution_architecture: format!(
            "{};tp-rank={}/{}",
            target_config.architecture_fingerprint(),
            topology.tensor_parallel_rank,
            topology.tensor_parallel_size
        ),
    };
    Ok(MoshiModel {
        source_config,
        target_config,
        identity,
        artifact_identity,
        state_layout,
        metadata,
        topology: Some(topology),
        execution,
    })
}

fn artifact_identity(
    model_dir: &Path,
    source: &Path,
    config: &MoshiConfig,
) -> Result<LoadedArtifactIdentity, Error> {
    let paths = if source.is_dir() {
        crate::backend::runtime::checkpoint::load::safetensors_files(source)?
    } else {
        vec![source.to_owned()]
    };
    let files = paths.into_iter().map(|path| {
        let logical = path
            .strip_prefix(model_dir)
            .unwrap_or(&path)
            .to_string_lossy()
            .into_owned();
        ArtifactFile::new(logical, path)
    });
    fingerprint_artifact(config.effective_model_type().as_str(), files)
}

fn bindings<M>(
    module: &M,
    store: &dyn CheckpointSource,
    recipes: &CanonicalBindingRecipes,
) -> Result<Vec<eredu_runtime::WeightBinding>, Error>
where
    M: Parameterized<crate::MlxTensor>,
{
    let names = eredu_nn::validate_parameter_topology(module)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        .into_iter()
        .map(|metadata| metadata.id.as_str().to_owned())
        .collect::<Vec<_>>();
    let selected = names
        .iter()
        .filter_map(|name| {
            let canonical = canonical_checkpoint_name(name);
            recipes
                .outputs
                .get(&canonical)
                .cloned()
                .map(|recipe| (name.clone(), recipe))
        })
        .collect::<BTreeMap<_, _>>();
    let aliases = names
        .into_iter()
        .filter_map(|name| {
            let canonical = canonical_checkpoint_name(&name);
            recipes
                .aliases
                .get(&canonical)
                .map(|owner| (name, canonical, owner.clone()))
        })
        .collect::<Vec<_>>();
    let alias_names = aliases
        .iter()
        .map(|(name, _, _)| name.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let mut selected = selected;
    let mut bindings = build_neutral_module_bindings_with_recipes_excluding(
        module,
        store,
        &mut selected,
        |name| alias_names.contains(name),
    )?;
    for (local, logical, owner) in aliases {
        let owner_recipe = recipes.outputs.get(&owner).ok_or_else(|| {
            Error::ArchitectureModel(format!(
                "Moshi logical alias {logical:?} names missing owner {owner:?}"
            ))
        })?;
        let expected_bytes = owner_recipe
            .infer(store)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?
            .byte_len();
        bindings.push(
            eredu_runtime::WeightBinding::alias(local, owner, expected_bytes)?
                .with_logical_target(logical)?,
        );
    }
    Ok(bindings)
}

fn execution_layout(architecture: &Architecture) -> Result<ExecutionUnitLayout, Error> {
    let graph =
        <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxKeyValueState>>::execution_graph(
            architecture,
        )?;
    let counts = (0..graph.groups().len())
        .map(|group| {
            <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxKeyValueState>>::group_unit_count(
                architecture,
                group,
            )
            .map_err(Error::from)
        })
        .collect::<Result<Vec<_>, _>>()?;
    ExecutionUnitLayout::new(&graph, counts)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
}

fn build_unit(
    config: &MoshiConfig,
    layout: &ExecutionUnitLayout,
    ordinal: usize,
    stream: &Stream,
) -> Result<MoshiUnit, Error> {
    let address = layout.address(ordinal).ok_or_else(|| {
        Error::ArchitectureModel(format!(
            "Moshi execution ordinal {ordinal} is outside 0..{}",
            layout.len()
        ))
    })?;
    build_addressed_unit(config, address, stream)
}

fn build_addressed_unit(
    config: &MoshiConfig,
    address: eredu_runtime::ExecutionUnitAddress,
    stream: &Stream,
) -> Result<MoshiUnit, Error> {
    let architecture = Architecture::new(config.clone(), stream)?;
    <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxKeyValueState>>::build_unit(
        &architecture,
        address.group(),
        address.index(),
        stream,
    )
    .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_checkpoint::{AffineQuantization, WeightQuantization};

    #[test]
    fn source_and_target_configuration_identities_are_distinct() {
        let source =
            MoshiConfig::from_json(r#"{"model_type":"personaplex","version":"7b-v1"}"#).unwrap();
        let target = source
            .with_native_quantization(Some(WeightQuantization::Affine(
                AffineQuantization::new(32, 4).unwrap(),
            )))
            .unwrap();
        assert_eq!(source.native_quantization(), None);
        assert_eq!(source.checkpoint_layout(), target.checkpoint_layout());
        assert_ne!(
            source.architecture_fingerprint(),
            target.architecture_fingerprint()
        );
    }

    #[test]
    #[ignore = "requires EREDU_MOSHI_FIXTURE pointing at a complete released artifact"]
    fn moshi_native_fixture_loads_through_neutral_entrypoint() {
        let fixture = std::env::var_os("EREDU_MOSHI_FIXTURE").expect(
            "EREDU_MOSHI_FIXTURE must point at a complete released artifact when this ignored fixture test is explicitly enabled",
        );
        assert!(
            Path::new(&fixture).exists(),
            "EREDU_MOSHI_FIXTURE does not exist: {}",
            Path::new(&fixture).display()
        );
        let device = safemlx::Device::new(safemlx::DeviceType::Cpu, 0);
        let stream = Stream::new_with_device(&device);
        let weights_stream = Stream::new_with_device(&device);
        let preparation = eredu_architectures::moshi::prepare_realtime_model(&fixture).unwrap();
        let model = load(
            preparation,
            ModelLoadOptions::default(),
            &stream,
            &weights_stream,
        )
        .unwrap();
        assert_eq!(model.source_config().family(), "moshi");
    }
}

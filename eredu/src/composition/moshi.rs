//! Production MLX composition for the backend-neutral Moshi-family model.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use eredu_architectures::moshi::{self, CheckpointLayout, LayeredModel, MoshiConfig, Unit};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{CheckpointSource, ResolvedCheckpointSource, SharedCheckpointSource},
};
use eredu_runtime::{
    DenseDiskStreamReport, ExecutionUnitLayout, LayeredArchitecture, LayeredTraversalHook,
    LayerwiseModelMetadata, LayerwiseRuntime, ResidencyReport, ResidentLayerGroupReport,
    SequentialDecisionDriver, SequentialDecisionTraversal, ShardingPolicy,
};
use safemlx::{module::ModuleParameters, Array, Stream};

use crate::backend::mlx::{
    ensure_replicated_load_options,
    error::Error,
    nn::shared::{MlxBackend, MlxModule},
    runtime::{
        cache::state::MlxKeyValueState,
        checkpoint::{
            artifact::{fingerprint_artifact, ArtifactFile, LoadedArtifactIdentity},
            binding::{
                build_module_bindings_with_recipes_excluding, canonical_checkpoint_name,
                full_parameter_names,
            },
            quantization::should_quantize_on_load,
        },
        execution::{
            generic::{
                prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
                MlxUnitFactory,
            },
            layerwise::{
                open_safetensors_weight_store, quantize_module_store_with_bindings,
                shard_layer_bindings,
            },
        },
        generation::sampler::{DefaultSampler, MlxSamplingBackend},
    },
    ModelLoadOptions,
};

type Architecture = LayeredModel<MlxBackend>;
type MoshiUnit = Unit<MlxBackend>;
type ResidentRuntime =
    LayerwiseRuntime<Architecture, MlxBackend, MlxKeyValueState, MlxResidentPolicy<MoshiUnit>>;
type BoundedRuntime = LayerwiseRuntime<
    Architecture,
    MlxBackend,
    MlxKeyValueState,
    MlxLayerwisePolicy<MoshiUnit, MoshiUnitFactory>,
>;
type ParallelResidentRuntime =
    LayerwiseRuntime<Architecture, MlxBackend, MlxKeyValueState, MlxResidentPolicy<MoshiUnit>>;
type ParallelBoundedRuntime = LayerwiseRuntime<
    Architecture,
    MlxBackend,
    MlxKeyValueState,
    MlxLayerwisePolicy<MoshiUnit, MoshiParallelUnitFactory>,
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

struct MoshiUnitFactory {
    config: MoshiConfig,
    layout: ExecutionUnitLayout,
}

#[derive(Clone)]
struct MoshiParallelUnitFactory {
    config: MoshiConfig,
    geometry: moshi::LocalGeometry,
    layout: ExecutionUnitLayout,
}

impl MlxUnitFactory<MoshiUnit> for MoshiParallelUnitFactory {
    fn build(&mut self, ordinal: usize, stream: &Stream) -> Result<MoshiUnit, Error> {
        let address = self.layout.address(ordinal).ok_or_else(|| {
            Error::Parallel(format!(
                "Moshi TP ordinal {ordinal} is outside the unit layout"
            ))
        })?;
        self.geometry
            .build_unit(&self.config, address.group(), address.index(), stream)
            .map_err(Into::into)
    }
}

impl MlxUnitFactory<MoshiUnit> for MoshiUnitFactory {
    fn build(&mut self, ordinal: usize, stream: &Stream) -> Result<MoshiUnit, Error> {
        build_unit(&self.config, &self.layout, ordinal, stream)
    }
}

/// Stable source-artifact and execution-topology identities for a loaded model.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MoshiModelIdentity {
    artifact: String,
    source_architecture: String,
    execution_architecture: String,
}

impl MoshiModelIdentity {
    /// Content identity of the source SafeTensors files.
    pub fn artifact(&self) -> &str {
        &self.artifact
    }

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
    topology: Option<crate::backend::mlx::MlxParallelContext>,
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
    pub fn topology(&self) -> Option<crate::backend::mlx::MlxParallelContext> {
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

    pub(crate) fn artifact_identity(&self) -> &LoadedArtifactIdentity {
        &self.artifact_identity
    }

    /// Creates request-local resident key/value state from the neutral layout.
    pub(crate) fn new_realtime_state(&self) -> Result<MlxKeyValueState, Error> {
        MlxKeyValueState::device(self.state_layout.clone()).map_err(Into::into)
    }

    /// Runs one neutral temporal/depth pass through the shared decision traversal.
    pub(crate) fn forward_realtime<'a>(
        &mut self,
        input: moshi::Input<'a, Array>,
        state: &mut MlxKeyValueState,
        driver: &mut SequentialDecisionDriver<MlxSamplingBackend, DefaultSampler>,
        stream: &Stream,
    ) -> Result<(Array, moshi::ForwardContext<Array>), Error> {
        let mut boundary = moshi::DecisionBoundary::new(&self.target_config)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let mut traversal = SequentialDecisionTraversal::new(driver, &mut boundary);
        self.forward_realtime_with_traversal_hook(input, state, &mut traversal, stream)
    }

    pub(crate) fn forward_realtime_with_traversal_hook<'a, H>(
        &mut self,
        input: moshi::Input<'a, Array>,
        state: &mut MlxKeyValueState,
        hook: &mut H,
        stream: &Stream,
    ) -> Result<(Array, moshi::ForwardContext<Array>), Error>
    where
        H: LayeredTraversalHook<MlxBackend, moshi::ForwardContext<Array>, eredu_nn::Error>,
    {
        match &mut self.execution {
            Execution::Resident(runtime) => runtime
                .forward_with_traversal_hook(input, state, stream, hook)
                .map_err(|error| Error::ArchitectureModel(error.to_string())),
            Execution::Bounded(runtime) => runtime
                .forward_with_traversal_hook(input, state, stream, hook)
                .map_err(|error| Error::ArchitectureModel(error.to_string())),
            Execution::ParallelResident(_) | Execution::ParallelBounded(_) => Err(Error::Parallel(
                "tensor-parallel Moshi execution requires the rank's TP collective group".into(),
            )),
        }
    }

    /// Runs one rank-local tensor-parallel temporal/depth pass.
    pub(crate) fn forward_realtime_parallel<'a>(
        &mut self,
        input: moshi::Input<'a, Array>,
        state: &mut MlxKeyValueState,
        driver: &mut SequentialDecisionDriver<MlxSamplingBackend, DefaultSampler>,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<(Array, moshi::ForwardContext<Array>), Error> {
        let mut boundary = moshi::DecisionBoundary::new(&self.target_config)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let mut traversal = SequentialDecisionTraversal::new(driver, &mut boundary);
        match &mut self.execution {
            Execution::ParallelResident(runtime) => runtime
                .forward_parallel_with_traversal_hook(input, state, group, stream, &mut traversal)
                .map_err(|error| Error::Parallel(error.to_string())),
            Execution::ParallelBounded(runtime) => runtime
                .forward_parallel_with_traversal_hook(input, state, group, stream, &mut traversal)
                .map_err(|error| Error::Parallel(error.to_string())),
            Execution::Resident(_) | Execution::Bounded(_) => Err(Error::Parallel(
                "model was not loaded for tensor-parallel Moshi execution".into(),
            )),
        }
    }
}

/// Loads either admitted Moshi-family SafeTensors layout into the neutral model.
pub fn load(
    model_dir: impl AsRef<Path>,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MoshiModel, Error> {
    let model_dir = model_dir.as_ref();
    let (source_config, config_value) = load_config(model_dir)?;
    let source_path = source_checkpoint_path(model_dir, &source_config, config_value.as_ref())?;
    let artifact_identity = artifact_identity(model_dir, &source_path, &source_config)?;
    let source_store = open_safetensors_weight_store(
        &source_path,
        options.weight_residency.layers().max_mapped_shards(),
    )?;
    let source_store = resolve_source_store(source_store, &source_config)?;
    let source_recipe_set = moshi::canonical_recipes(&source_config, source_store.as_ref())
        .map_err(Error::UnsupportedArchitecture)?;
    let (source_outputs, source_aliases) = source_recipe_set.into_parts();
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
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
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
        .parallel
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
    ensure_replicated_load_options(options)?;

    let mut architecture = Architecture::new(target_config.clone(), stream)?;
    let layout = execution_layout(&architecture)?;
    let factory = MoshiUnitFactory {
        config: target_config.clone(),
        layout: layout.clone(),
    };
    let static_recipes = Arc::clone(&source_recipes);
    let unit_recipes = Arc::clone(&source_recipes);
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        store,
        architecture.static_modules_mut(),
        factory,
        layout,
        options.weight_residency.layers(),
        stream,
        weights_stream,
        |_| false,
        move |modules, store| {
            bindings(
                &MlxModule::new(modules.clone()),
                store,
                static_recipes.as_ref(),
            )
        },
        move |_ordinal, unit, store, _stream| {
            bindings(&MlxModule::new(unit), store, unit_recipes.as_ref())
        },
    )?;
    metadata.set_model_type(target_config.effective_model_type().as_str());
    metadata.set_quantization(target_config.native_quantization());
    metadata.set_materialization(materialization);
    let state_layout = moshi::state_layout(&target_config)?;
    let execution = if options.weight_residency.layers().is_fully_resident() {
        Execution::Resident(LayerwiseRuntime::new(
            architecture,
            policy.into_resident(stream)?,
        ))
    } else {
        Execution::Bounded(LayerwiseRuntime::new(architecture, policy))
    };
    let identity = MoshiModelIdentity {
        artifact: format!("{artifact_identity:?}"),
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
    topology: crate::backend::mlx::MlxParallelContext,
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
    let build = crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext::new(
        topology,
        ShardingPolicy::Require,
    );
    let global = Architecture::new(target_config.clone(), stream)?;
    let layout = execution_layout(&global)?;
    let mut planner = build.planner();
    crate::composition::moshi_parallel::register_parallel_parameters(
        &global,
        &mut planner,
        stream,
    )?;
    let (_, local_layout) = planner.finish()?;
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
    let mut composition =
        Architecture::new_parallel(target_config.clone(), geometry.clone(), stream)?;
    let state_layout = composition.runtime_state_layout()?;
    let factory = MoshiParallelUnitFactory {
        config: target_config.clone(),
        geometry,
        layout: layout.clone(),
    };
    let local_layout = Arc::new(local_layout);
    let static_module = MlxModule::new(global.static_modules().clone());
    let static_recipes = Arc::clone(&source_recipes);
    let static_sharding = Arc::clone(&local_layout);
    let unit_config = target_config.clone();
    let unit_layout = layout.clone();
    let unit_recipes = Arc::clone(&source_recipes);
    let unit_sharding = Arc::clone(&local_layout);
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        store,
        composition.static_modules_mut(),
        factory,
        layout,
        residency,
        stream,
        weights_stream,
        |_| false,
        move |_modules, store| {
            shard_layer_bindings(
                bindings(&static_module, store, static_recipes.as_ref())?,
                "",
                store,
                static_sharding.as_ref(),
            )
        },
        move |ordinal, _local, store, stream| {
            let global = build_unit(&unit_config, &unit_layout, ordinal, stream)?;
            shard_layer_bindings(
                bindings(&MlxModule::new(global), store, unit_recipes.as_ref())?,
                "",
                store,
                unit_sharding.as_ref(),
            )
        },
    )?;
    metadata.set_model_type(target_config.effective_model_type().as_str());
    metadata.set_quantization(target_config.native_quantization());
    metadata.set_materialization(materialization);
    let execution = if residency.is_fully_resident() {
        Execution::ParallelResident(Box::new(LayerwiseRuntime::new(
            composition,
            policy.into_resident(stream)?,
        )))
    } else {
        Execution::ParallelBounded(Box::new(LayerwiseRuntime::new(composition, policy)))
    };
    let identity = MoshiModelIdentity {
        artifact: format!("{artifact_identity:?}"),
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

fn load_config(model_dir: &Path) -> Result<(MoshiConfig, Option<serde_json::Value>), Error> {
    let path = model_dir.join("config.json");
    if !path.exists() {
        return MoshiConfig::native_v0_1()
            .map(|config| (config, None))
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()));
    }
    let text = std::fs::read_to_string(path)?;
    let value = serde_json::from_str(&text)?;
    let config = MoshiConfig::from_config_value(Some(&value))
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    Ok((config, Some(value)))
}

fn source_checkpoint_path(
    model_dir: &Path,
    config: &MoshiConfig,
    value: Option<&serde_json::Value>,
) -> Result<PathBuf, Error> {
    if model_dir.join("model.safetensors.index.json").exists()
        || config.checkpoint_layout() == CheckpointLayout::PersonaPlexPytorch
    {
        return Ok(model_dir.to_owned());
    }
    let name = value
        .and_then(serde_json::Value::as_object)
        .and_then(|object| object.get("moshi_name"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("model.safetensors");
    let name = Path::new(name);
    if name.is_absolute() || name.components().count() != 1 {
        return Err(Error::UnsupportedArchitecture(format!(
            "Moshi artifact filename must be a single relative component, got {:?}",
            name
        )));
    }
    Ok(model_dir.join(name))
}

fn artifact_identity(
    model_dir: &Path,
    source: &Path,
    config: &MoshiConfig,
) -> Result<LoadedArtifactIdentity, Error> {
    let paths = if source.is_dir() {
        crate::backend::mlx::runtime::checkpoint::load::safetensors_files(source)?
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

fn resolve_source_store(
    store: SharedCheckpointSource,
    source_config: &MoshiConfig,
) -> Result<SharedCheckpointSource, Error> {
    let plan = moshi::safetensors_plan(source_config)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let resolved = eredu_checkpoint::validation::resolve_safetensors_plan(store.as_ref(), &plan)
        .map_err(|validation| {
            Error::UnsupportedArchitecture(format!(
                "Moshi checkpoint contract did not resolve: {validation:?}"
            ))
        })?;
    Ok(Arc::new(ResolvedCheckpointSource::new(store, resolved)))
}

fn bindings(
    module: &impl ModuleParameters,
    store: &dyn CheckpointSource,
    recipes: &CanonicalBindingRecipes,
) -> Result<Vec<eredu_runtime::WeightBinding>, Error> {
    let names = full_parameter_names(module, "");
    let selected = names
        .iter()
        .filter_map(|name| {
            let canonical = canonical_checkpoint_name(&name);
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
    let mut bindings =
        build_module_bindings_with_recipes_excluding(module, "", store, selected, |name| {
            alias_names.contains(name)
        })?;
    for (local, logical, owner) in aliases {
        let owner_recipe = recipes.outputs.get(&owner).ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Moshi logical alias {logical:?} names missing owner {owner:?}"
            ))
        })?;
        let expected_bytes = owner_recipe
            .infer(store)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?
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
        <Architecture as LayeredArchitecture<MlxBackend, MlxKeyValueState>>::execution_graph(
            architecture,
        )?;
    let counts = (0..graph.groups().len())
        .map(|group| {
            <Architecture as LayeredArchitecture<MlxBackend, MlxKeyValueState>>::group_unit_count(
                architecture,
                group,
            )
            .map_err(Error::from)
        })
        .collect::<Result<Vec<_>, _>>()?;
    ExecutionUnitLayout::new(&graph, counts)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

fn build_unit(
    config: &MoshiConfig,
    layout: &ExecutionUnitLayout,
    ordinal: usize,
    stream: &Stream,
) -> Result<MoshiUnit, Error> {
    let address = layout.address(ordinal).ok_or_else(|| {
        Error::UnsupportedArchitecture(format!(
            "Moshi execution ordinal {ordinal} is outside 0..{}",
            layout.len()
        ))
    })?;
    let architecture = Architecture::new(config.clone(), stream)?;
    <Architecture as LayeredArchitecture<MlxBackend, MlxKeyValueState>>::build_unit(
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
    fn artifact_filename_is_resolution_only_and_confined_to_model_directory() {
        let config = MoshiConfig::native_v0_1().unwrap();
        let root = Path::new("fixture");
        let value = serde_json::json!({"moshi_name":"weights.safetensors"});
        assert_eq!(
            source_checkpoint_path(root, &config, Some(&value)).unwrap(),
            root.join("weights.safetensors")
        );
        let escape = serde_json::json!({"moshi_name":"../weights.safetensors"});
        assert!(source_checkpoint_path(root, &config, Some(&escape)).is_err());
    }

    #[test]
    #[ignore = "requires EREDU_MOSHI_FIXTURE pointing at a complete released artifact"]
    fn moshi_stage8_native_fixture_loads_through_neutral_entrypoint() {
        let Some(fixture) = std::env::var_os("EREDU_MOSHI_FIXTURE") else {
            return;
        };
        assert!(
            Path::new(&fixture).exists(),
            "EREDU_MOSHI_FIXTURE does not exist: {}",
            Path::new(&fixture).display()
        );
        let device = safemlx::Device::new(safemlx::DeviceType::Cpu, 0);
        let stream = Stream::new_with_device(&device);
        let weights_stream = Stream::new_with_device(&device);
        let model = load(
            fixture,
            ModelLoadOptions::default(),
            &stream,
            &weights_stream,
        )
        .unwrap();
        assert_eq!(model.source_config().family(), "moshi");
    }
}

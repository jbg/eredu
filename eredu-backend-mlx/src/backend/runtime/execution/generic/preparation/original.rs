//! Original immutable source preparation before ordinary native model construction.

use super::*;
use eredu_architectures::{
    prepared_execution::{project_replicated_text_binding_destinations, PreparedExecutionError},
    prepared_sources::PreparedModelSources,
};
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceMechanisms, WorkspaceOperation, WorkspaceOperationBound,
};
use eredu_runtime::working_memory::WorkingMemoryPool;
mod partitioned;

/// Move-only adoption of a manager born under its original source account.
/// Cold planning maps are discarded; validation borrows the manager's funded
/// declarations instead of retaining another unaccounted copy in this wrapper.
pub(crate) struct PreparedLayerwiseManager {
    manager: ResidencyManager,
}

impl PreparedLayerwiseManager {
    pub(crate) fn validate_and_take(
        self,
        declarations: &PreparedLayerwiseDeclarations,
        source_stream: &Stream,
        execution_stream: &Stream,
    ) -> Result<ResidencyManager, Error> {
        self.manager.validate_original_layerwise_preparation(
            &declarations.store,
            &declarations.sources,
            &declarations.plan,
            &declarations.definitions,
            source_stream,
            execution_stream,
        )?;
        Ok(self.manager)
    }
}

/// Tries the selected original layerwise source producer before load exclusion is
/// acquired. A pure unsupported plan preserves ordinary loading. Once source
/// admission starts, every failure propagates; no fallback can promote its work.
pub(crate) fn prepare_layerwise_manager(
    sources: &PreparedModelSources,
    pool: &WorkingMemoryPool,
    source_stream: &Stream,
    execution_stream: &Stream,
) -> Result<Option<PreparedLayerwiseManager>, Error> {
    let selected = sources.selected().text_realization().residency();
    if !(matches!(selected, LayerWeightResidency::FullyResident) && sources.prediction_extension().is_some())
        && !matches!(selected, LayerWeightResidency::LayerwiseHost(_))
        && !matches!(selected, LayerWeightResidency::DenseDiskStream(_))
    {
        return Ok(None);
    }
    prepare_selected_layerwise_manager(sources, pool, source_stream, execution_stream)
}

/// Explicit cold disk source handoff. This constructs the manager and its static
/// owners, but certifies no request read slots or native disk operation bounds.
/// Request read slots and native operation bounds are established separately.
pub(crate) fn prepare_foreground_layerwise_manager(
    sources: &PreparedModelSources,
    pool: &WorkingMemoryPool,
    source_stream: &Stream,
    execution_stream: &Stream,
) -> Result<Option<PreparedLayerwiseManager>, Error> {
    if !matches!(
        sources.selected().text_realization().residency(),
        LayerWeightResidency::DenseDiskStream(_)
    ) {
        return Ok(None);
    }
    prepare_selected_layerwise_manager(sources, pool, source_stream, execution_stream)
}

fn prepare_selected_layerwise_manager(
    sources: &PreparedModelSources,
    pool: &WorkingMemoryPool,
    source_stream: &Stream,
    execution_stream: &Stream,
) -> Result<Option<PreparedLayerwiseManager>, Error> {
    let selected = sources.selected().text_realization();
    if matches!(selected.residency(), LayerWeightResidency::DenseDiskStream(options)
        if options.samples_backend_memory() || options.samples_process_memory()) {
        return Ok(None);
    }
    // Transforming payload producers must fund their own source and output
    // construction; the direct ready-host reader cannot certify them.
    if selected.materialization_tasks().iter().chain(selected.auxiliary_materialization_tasks()).any(|task| {
        matches!(
            task.lowering(),
            eredu_runtime::WeightLoweringKind::Transform
                | eredu_runtime::WeightLoweringKind::DerivedTransform
        )
    }) {
        return Ok(None);
    }
    if sources.selected().execution().parallel_topology().is_some() {
        return partitioned::prepare(sources,pool,source_stream,execution_stream);
    }
    let context = WorkspaceContext::new(DestinationFacts);
    let projected = match project_replicated_text_binding_destinations(sources, &context) {
        Ok(projected) => projected,
        Err(
            PreparedExecutionError::UnavailableExecution
            | PreparedExecutionError::UnavailablePrediction
            | PreparedExecutionError::MissingCommunication,
        ) => {
            return Ok(None);
        },
        Err(cause) => return Err(Error::ArchitectureModel(cause.to_string())),
    };
    let contract = projected.contract();
    let selected = contract.selected();
    let layout = selected.requirements().execution_units();
    let tasks = contract.materialization_tasks();
    let partitions = eredu_runtime::plan_replicated_text_materialization_tasks(tasks, layout)
        .map_err(|cause| Error::ArchitectureModel(cause.to_string()))?;
    let static_tasks = partitions
        .static_tasks(tasks)
        .map_err(|cause| Error::ArchitectureModel(cause.to_string()))?;
    let unit_tasks = partitions
        .unit_tasks(tasks)
        .map_err(|cause| Error::ArchitectureModel(cause.to_string()))?;
    let addressable = contract
        .addressable_parameters()
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let bindings = |destinations, tasks: &[&eredu_runtime::ReplicatedTextMaterializationTask]| {
        let targets = crate::backend::runtime::checkpoint::binding::mlx_workspace_binding_targets(
            destinations,
        )
        .ok_or_else(|| {
            Error::ArchitectureModel("invalid cold MLX destination representation".into())
        })?;
        eredu_runtime::build_exact_replicated_text_bindings_for_targets(
            &targets,
            sources.target().as_ref(),
            tasks,
            &addressable,
            None,
            |_task, recipe, source| {
                crate::backend::runtime::checkpoint::recipe::lower_mxfp4_recipe(recipe, source)
            },
        )
        .map_err(|cause| Error::ArchitectureModel(cause.to_string()))
    };
    let static_bindings = bindings(projected.static_parameters(), &static_tasks)?;
    let unit_bindings = projected
        .units()
        .iter()
        .zip(&unit_tasks)
        .map(|(destinations, tasks)| bindings(destinations, tasks))
        .collect::<Result<Vec<_>, _>>()?;
    let mut ignored = selected
        .requirements()
        .parameters()
        .iter()
        .flat_map(|parameter| parameter.admitted_redundant_sources().iter().cloned())
        .collect::<BTreeSet<_>>();
    for parameter in selected
        .requirements()
        .parameters()
        .iter()
        .filter(|parameter| addressable.contains(parameter.name()))
    {
        ignored.extend(parameter.sources().iter().cloned());
        if let Some(recipe) = selected
            .requirements()
            .derived_recipes()
            .get(parameter.name())
        {
            ignored.extend(recipe.source_keys().into_iter().map(str::to_owned));
        }
    }
    let mut supplementary = Vec::with_capacity(projected.prediction_modules().len());
    for module in projected.prediction_modules() {
        let store = sources.extension().ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
        ))?;
        let targets = crate::backend::runtime::checkpoint::binding::mlx_workspace_binding_targets(&module.parameters)
            .ok_or_else(|| Error::ArchitectureModel("invalid cold MLX prediction destination representation".into()))?;
        let tasks = module.tasks.iter().collect::<Vec<_>>();
        let bindings = eredu_runtime::build_exact_replicated_text_bindings_for_targets(
            &targets, store.as_ref(), &tasks, &BTreeSet::new(), module.layout.as_ref(),
            |_task, recipe, source| crate::backend::runtime::checkpoint::recipe::lower_mxfp4_recipe(recipe, source),
        ).map_err(|cause| Error::ArchitectureModel(cause.to_string()))?;
        supplementary.push(SupplementaryResidencyUnit {
            definition: eredu_runtime::OffloadUnit::new(
                OffloadUnitId::new(format!("prediction.module.{:05}", module.ordinal))?, bindings,
            )?,
            source: store.clone(),
            shared: module.shared,
        });
    }
    let declarations = prepare_layerwise_declarations(
        sources.target().clone(),
        selected.residency(),
        |key| ignored.contains(key),
        layout,
        static_bindings,
        unit_bindings,
        supplementary,
    )?;
    prepare_manager_from_declarations(declarations, selected.residency(), layout,
        pool, source_stream, execution_stream)
}

pub(crate) fn prepare_manager_from_declarations(
    declarations: PreparedLayerwiseDeclarations,
    residency: LayerWeightResidency,
    layout: &ExecutionUnitLayout,
    pool: &WorkingMemoryPool,
    source_stream: &Stream,
    execution_stream: &Stream,
) -> Result<Option<PreparedLayerwiseManager>, Error> {
    let mut groups = (0..layout.group_count())
        .map(|group| {
            layout
                .group_id(group)
                .expect("validated execution group")
                .as_str()
                .to_owned()
        })
        .collect::<Vec<_>>();
    if matches!(
        residency,
        LayerWeightResidency::DenseDiskStream(_)
    ) {
        // Ordinary sessions may use the same source-prepared manager. Declare
        // the shared scheduler's exact finite protection names at construction.
        let names = groups
            .iter()
            .flat_map(|group| {
                crate::backend::runtime::execution::layerwise::dense_window_names(group)
            })
            .collect::<Vec<_>>();
        groups.extend(names);
    }
    let manager = match residency {
        LayerWeightResidency::FullyResident | LayerWeightResidency::LayerwiseHost(_) => ResidencyManager::prepare_original_host(
            declarations.store.clone(),
            declarations.sources.clone(),
            &declarations.plan,
            &declarations.definitions,
            &groups,
            &declarations.unit_ids,
            layout,
            declarations.depth,
            source_stream,
            execution_stream,
            pool,
        ),
        LayerWeightResidency::DenseDiskStream(options) => {
            ResidencyManager::prepare_original_foreground_disk_with_controller(
                declarations.store.clone(),
                declarations.sources.clone(),
                &declarations.plan,
                &declarations.definitions,
                &groups,
                &declarations.unit_ids,
                layout,
                declarations.depth,
                source_stream,
                execution_stream,
                pool,
                OriginalDenseControllerFacts {
                    options,
                    planned_layers: declarations.unit_count,
                    planned_bytes: declarations.layer_parameter_bytes,
                    maximum_host_bytes: declarations.maximum_host_bytes,
                    static_bytes: declarations.static_bytes,
                    stream_index: execution_stream.get_index()?,
                },
            )
        }
        _ => return Ok(None),
    }
    .map_err(|cause| Error::Other(Box::new(cause)))?;
    Ok(manager.map(|manager| PreparedLayerwiseManager { manager }))
}

/// Destination construction uses only geometry. This planner supplies no
/// execution/storage bound, so none can accidentally become admission evidence.
#[derive(Debug)]
struct DestinationFacts;
impl WorkspaceMechanisms for DestinationFacts {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        Ok(None)
    }
}

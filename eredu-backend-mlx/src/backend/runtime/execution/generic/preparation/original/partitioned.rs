//! Exact local binding declarations fed to the existing Host/Disk source owner.
use super::*;
use eredu_architectures::prepared_execution::project_partitioned_text_binding_destinations;

pub(super) fn prepare(sources:&PreparedModelSources,pool:&WorkingMemoryPool,
    source_stream:&Stream,execution_stream:&Stream)->Result<Option<PreparedLayerwiseManager>,Error> {
    let context=WorkspaceContext::new(DestinationFacts);
    let projected=match project_partitioned_text_binding_destinations(sources,&context) {
        Ok(projected)=>projected,
        Err(PreparedExecutionError::UnavailableExecution|PreparedExecutionError::UnavailablePrediction
            |PreparedExecutionError::MissingCommunication)=>return Ok(None),
        Err(cause)=>return Err(Error::ArchitectureModel(cause.to_string())),
    };
    let tasks=projected.tasks();
    let plan=eredu_runtime::plan_local_replicated_text_materialization_tasks(
        tasks,projected.global_layout(),projected.addresses())
        .map_err(|cause|Error::ArchitectureModel(cause.to_string()))?;
    let static_tasks=plan.static_tasks(tasks).map_err(|cause|Error::ArchitectureModel(cause.to_string()))?;
    let unit_tasks=plan.unit_tasks(tasks).map_err(|cause|Error::ArchitectureModel(cause.to_string()))?;
    let local=tasks.iter().flat_map(|task|std::iter::once(task.name().to_owned())
        .chain(task.output_companions().iter().map(|companion|companion.name().to_owned())))
        .collect::<BTreeSet<_>>();
    let selected_static=static_tasks.iter().flat_map(|task|std::iter::once(task.name().to_owned())
        .chain(task.output_companions().iter().map(|companion|companion.name().to_owned())))
        .collect::<BTreeSet<_>>();
    let excluded=projected.static_parameters().keys().filter(|name|!selected_static.contains(*name))
        .cloned().collect::<BTreeSet<_>>();
    let bind=|destinations, tasks:&[&eredu_runtime::ReplicatedTextMaterializationTask],excluded:&BTreeSet<String>| {
        let targets=crate::backend::runtime::checkpoint::binding::mlx_workspace_binding_targets(destinations)
            .ok_or_else(||Error::ArchitectureModel("invalid cold partition destination representation".into()))?;
        eredu_runtime::build_exact_replicated_text_bindings_for_targets(&targets,sources.target().as_ref(),
            tasks,excluded,Some(projected.physical_layout()),|_task,recipe,source|
                crate::backend::runtime::checkpoint::recipe::lower_mxfp4_recipe(recipe,source))
            .map_err(|cause|Error::ArchitectureModel(cause.to_string()))
    };
    let static_bindings=bind(projected.static_parameters(),&static_tasks,&excluded)?;
    if projected.units().len()!=unit_tasks.len() {
        return Err(Error::ArchitectureModel("partition source destinations differ from local task rows".into()));
    }
    let unit_bindings=projected.units().iter().zip(&unit_tasks)
        .map(|(destinations,tasks)|bind(destinations,tasks,&BTreeSet::new()))
        .collect::<Result<Vec<_>,_>>()?;
    let selected=projected.selected();
    let mut ignored=selected.requirements().parameters().iter()
        .flat_map(|parameter|parameter.admitted_redundant_sources().iter().cloned()).collect::<BTreeSet<_>>();
    for parameter in selected.requirements().parameters().iter()
        .filter(|parameter|excluded.contains(parameter.name())||!local.contains(parameter.name())) {
        ignored.extend(parameter.sources().iter().cloned());
        if let Some(recipe)=selected.requirements().derived_recipes().get(parameter.name()) {
            ignored.extend(recipe.source_keys().into_iter().map(str::to_owned));
        }
    }
    if let Some(extension)=sources.extension(){ignored.extend(extension.source_keys());}
    let declarations=prepare_layerwise_declarations(sources.target().clone(),selected.residency(),
        |key|ignored.contains(key),projected.local_layout(),static_bindings,unit_bindings,Vec::new())?;
    prepare_manager_from_declarations(declarations,selected.residency(),projected.local_layout(),
        pool,source_stream,execution_stream)
}

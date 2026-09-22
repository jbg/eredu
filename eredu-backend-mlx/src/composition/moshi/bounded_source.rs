//! Exact selected Host/Disk source manager, before ordinary model construction.
use super::*;
use crate::backend::runtime::execution::generic::{
    prepare_layerwise_declarations, prepare_manager_from_declarations, PreparedLayerwiseManager,
};
use eredu_runtime::{ExecutionUnitLayout, LayerWeightResidency};

/// Same ownership/order projection for cold manager construction and the final
/// typed binder. All keys came from the retained neutral materialization tasks.
/// A missing/extra owner refuses; no backend checkpoint naming is reconstructed.
pub(super) fn ordered_units(
    layout: &ExecutionUnitLayout,
    mut units: BTreeMap<ParameterGroupOwner, Vec<WeightBinding>>,
) -> Result<Vec<Vec<WeightBinding>>, Error> {
    let mut result = Vec::with_capacity(layout.len());
    for ordinal in 0..layout.len() {
        let address = layout.address(ordinal).ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
        ))?;
        let group = layout
            .group_id(address.group())
            .ok_or(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ))?;
        let key = ParameterGroupOwner::execution_unit(group.clone(), address.index());
        result.push(units.remove(&key).ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
        ))?);
    }
    if !units.is_empty() {
        return Err(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
        ));
    }
    Ok(result)
}

/// Payload reading and source registration use the existing manager producer.
/// Unsupported source transforms remain on their actual
/// ordinary path until their independent source producer is connected; this
/// helper does not describe those missing sources as complete.
pub(super) fn prepare(
    prepared: &moshi::PreparedMoshiRealtimeSource,
    backend: &crate::backend::MlxBackend<'static>,
) -> Result<Option<PreparedLayerwiseManager>, Error> {
    let source = prepared.selected();
    let selected = source.selected();
    let residency = selected.residency();
    if !matches!(
        residency,
        LayerWeightResidency::LayerwiseHost(_) | LayerWeightResidency::DenseDiskStream(_)
    ) || prepared.lowering().transform().is_some()
    {
        return Ok(None);
    }
    if matches!(residency,LayerWeightResidency::DenseDiskStream(options)
        if options.samples_backend_memory()||options.samples_process_memory())
    {
        return Ok(None);
    }
    match backend.original_copy_environment() {
        Ok(_) => {}
        Err(crate::backend::OriginalCopyEnvironmentError::Memory(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        )) => return Ok(None),
        Err(cause) => return Err(Error::Other(Box::new(cause))),
    }
    let tasks = source.materialization_tasks();
    let bindings = eredu_runtime::preflight_realtime_materialization_tasks::<MlxNeuralBackend>(
        tasks,
        prepared.source().as_ref(),
    )
    .map_err(|cause| Error::ArchitectureModel(cause.to_string()))?;
    // The final binder applies this same architecture-retained local layout to
    // the same physical task recipes. Pure TP keeps all execution-unit owners;
    // only their exact parameter slices change. No native context is needed.
    let local_layout = source.parallel().map(|parallel| parallel.layout());
    let (static_bindings, unit_bindings) =
        selected_task_bindings(bindings, prepared.source().as_ref(), local_layout)?;
    let layout = selected.execution_units();
    let units = ordered_units(layout, unit_bindings)?;
    let declarations = prepare_layerwise_declarations(
        prepared.source().clone(),
        residency,
        |_| false,
        layout,
        static_bindings,
        units,
        Vec::new(),
    )?;
    prepare_manager_from_declarations(
        declarations,
        residency,
        layout,
        &std::collections::BTreeSet::new(),
        None,
        backend.memory_ledger(),
        backend.weights_stream(),
        backend.stream(),
    )
}

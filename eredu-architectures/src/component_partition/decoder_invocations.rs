//! Complete-unit evidence follows declared logical invocations, not checkpoint
//! storage aliases, child component paths, or input/output name conventions.
use super::*;
use eredu_core::{
    ArchitectureNodeKind, ObservationValueType, capture::CaptureError,
    speculative::SpeculativeCaptureScope,
};

pub(super) fn register(
    observations: &mut SourceMap<String, PartitionedObservation>,
    descriptor: &ArchitectureDescriptor,
    parameters: &ArchitectureParameterDescription,
    owns: impl Fn(&eredu_runtime::ParameterGroupOwner) -> bool,
) -> Result<(), ComponentPartitionError> {
    worker(
        observations,
        descriptor,
        parameters,
        owns,
        Destination(None),
    )
}
pub(super) fn worker(
    observations: &mut SourceMap<String, PartitionedObservation>,
    descriptor: &ArchitectureDescriptor,
    parameters: &ArchitectureParameterDescription,
    owns: impl Fn(&eredu_runtime::ParameterGroupOwner) -> bool,
    allocation: Destination<'_>,
) -> Result<(), ComponentPartitionError> {
    allocation.controls::<(
        &mut SourceMap<String, PartitionedObservation>,
        &ArchitectureDescriptor,
        &ArchitectureParameterDescription,
        SourceMap<&str, ()>,
        Option<bool>,
    )>()?;
    allocation.controls_of(&owns)?;
    let mut visited = SourceMap::new();
    for group in &descriptor.layer_groups {
        for execution in group.passes.iter().flat_map(|pass| &pass.executions) {
            let invalid = || {
                allocation.capture_invalid(format_args!(
                    "invalid declared decoder invocation: {}",
                    execution.node_id
                ))
            };
            let node = descriptor.node(&execution.node_id).ok_or_else(&invalid)?;
            if node.kind != ArchitectureNodeKind::DecoderBlock
                || execution.physical_layer_index >= group.physical_layer_count
                || allocation
                    .insert(&mut visited, node.id.as_str(), ())?
                    .is_some()
            {
                return Err(invalid().into());
            }
            if allocation.scope(descriptor, &node.id)? != SpeculativeCaptureScope::Target {
                continue;
            }
            let mut local = None;
            for path in &node.observation_paths {
                let point = descriptor
                    .observations
                    .get(path)
                    .ok_or_else(|| allocation.capture_missing(path))?;
                if point.node_id != node.id {
                    return Err(invalid().into());
                }
                if !matches!(point.value_type, ObservationValueType::Tensor)
                    || !point
                        .axes
                        .as_ref()
                        .is_some_and(|axes| axes.iter().any(|axis| axis.name == "hidden"))
                {
                    continue;
                }
                let local = match local {
                    Some(value) => value,
                    None => {
                        let value = owns(invocations::owner(
                            descriptor, parameters, &node.id, allocation,
                        )?);
                        local = Some(value);
                        value
                    }
                };
                // The other axes (including V4 streams) remain in the catalog.
                // Existing placements must agree; no overwrite or suffix pairing.
                let placement = observations::placement(
                    descriptor,
                    path,
                    "hidden",
                    local,
                    ObservationHookSite::Unit,
                    allocation,
                )?;
                observations::insert(observations, path, placement, allocation)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;

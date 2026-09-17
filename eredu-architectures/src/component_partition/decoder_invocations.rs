//! Complete-unit evidence follows declared logical invocations, not checkpoint
//! storage aliases, child component paths, or input/output name conventions.
use super::*;
use eredu_core::{
    capture::CaptureError, speculative::SpeculativeCaptureScope, ArchitectureNodeKind,
    ObservationValueType,
};

pub(super) fn register(
    observations: &mut BTreeMap<String, PartitionedObservation>,
    descriptor: &ArchitectureDescriptor,
    parameters: &ArchitectureParameterDescription,
    owns: impl Fn(&eredu_runtime::ParameterGroupOwner) -> bool,
) -> Result<(), ComponentPartitionError> {
    let mut visited = std::collections::BTreeSet::new();
    for group in &descriptor.layer_groups {
        for execution in group.passes.iter().flat_map(|pass| &pass.executions) {
            let invalid = || {
                CaptureError::Invalid(format!(
                    "invalid declared decoder invocation: {}",
                    execution.node_id
                ))
            };
            let node = descriptor.node(&execution.node_id).ok_or_else(&invalid)?;
            if node.kind != ArchitectureNodeKind::DecoderBlock
                || execution.physical_layer_index >= group.physical_layer_count
                || !visited.insert(node.id.as_str())
            {
                return Err(invalid().into());
            }
            if crate::speculative_execution::speculative_capture_scope(descriptor, &node.id)?
                != SpeculativeCaptureScope::Target
            {
                continue;
            }
            let mut local = None;
            for path in &node.observation_paths {
                let point = descriptor
                    .observations
                    .get(path)
                    .ok_or_else(|| CaptureError::MissingPath(path.clone()))?;
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
                        let value = owns(node_invocation_owner(descriptor, parameters, &node.id)?);
                        local = Some(value);
                        value
                    }
                };
                // The other axes (including V4 streams) remain in the catalog.
                // Existing placements must agree; no overwrite or suffix pairing.
                let placement = replicated_placement(
                    descriptor,
                    path,
                    "hidden",
                    local,
                    ObservationHookSite::Unit,
                )?;
                insert_observation(observations, path, placement)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;

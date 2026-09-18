//! Borrowed invocation resolution and original boundary registration.
use super::*;
pub(super) fn routed(
    observations: &mut SourceMap<String, PartitionedObservation>,
    descriptor: &ArchitectureDescriptor,
    components: &[eredu_core::component::RoutedComponentGroup],
    owns_node: impl Fn(&str) -> Result<bool, ComponentPartitionError>,
    allocation: Destination<'_>,
) -> Result<(), ComponentPartitionError> {
    allocation.controls::<(
        &mut SourceMap<String, PartitionedObservation>,
        &ArchitectureDescriptor,
        &[eredu_core::component::RoutedComponentGroup],
        &str,
        usize,
    )>()?;
    allocation.controls_of(&owns_node)?;
    for component in components {
        for (input, width) in component
            .input
            .iter()
            .map(|path| (path, component.input_width))
            .chain(
                component
                    .write_output
                    .iter()
                    .chain(component.output.iter())
                    .map(|path| (path, component.output_width)),
            )
        {
            let point = descriptor
                .observations
                .points
                .iter()
                .find(|point| point.path == *input)
                .ok_or_else(|| allocation.capture_missing(input))?;
            if !point.axes.as_ref().is_some_and(|axes| {
                axes.iter().any(|axis| {
                    axis.name == "hidden"
                        && axis.dimension == eredu_core::SymbolicDimension::Known(width)
                })
            }) {
                return Err(allocation.invalid(&component.id));
            }
            observations::replicated(
                observations,
                descriptor,
                input,
                "hidden",
                owns_node(&point.node_id)?,
                ObservationHookSite::Unit,
                allocation,
            )?;
        }
    }
    Ok(())
}

pub(super) fn owner<'a>(
    descriptor: &ArchitectureDescriptor,
    parameters: &'a ArchitectureParameterDescription,
    node_id: &str,
    allocation: Destination<'_>,
) -> Result<&'a eredu_runtime::ParameterGroupOwner, ComponentPartitionError> {
    allocation.controls::<(
        &ArchitectureDescriptor,
        &ArchitectureParameterDescription,
        &str,
        Option<&eredu_runtime::ParameterGroupOwner>,
        bool,
    )>()?;
    let invalid = || {
        allocation.capture_invalid(format_args!(
            "residual write has no unique unit owner: {node_id}"
        ))
    };
    let node = descriptor
        .nodes
        .iter()
        .find(|node| node.id == node_id)
        .ok_or_else(invalid)?;
    let mut owner = None;
    for group_id in &node.parameter_groups {
        let group = descriptor
            .parameter_groups
            .iter()
            .find(|group| &group.id == group_id)
            .ok_or_else(invalid)?;
        let mut matched = false;
        for owned in parameters.groups().iter().filter(|owned| {
            owned.members().iter().any(|member| {
                member
                    .target()
                    .strip_prefix(&group.canonical_prefix)
                    .is_some_and(|suffix| suffix.starts_with('.'))
            })
        }) {
            matched = true;
            if !matches!(
                owned.owner(),
                eredu_runtime::ParameterGroupOwner::ExecutionUnit { .. }
            ) || owner.is_some_and(|previous| previous != owned.owner())
            {
                return Err(invalid().into());
            }
            owner = Some(owned.owner());
        }
        if !matched {
            return Err(invalid().into());
        }
    }
    owner.ok_or_else(|| invalid().into())
}

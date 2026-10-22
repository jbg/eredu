//! Shared DeepSeek semantic declarations; no backend resources are constructed here.
use eredu_nn::Error;
use eredu_runtime::{
    ArchitectureParameterDescription, ComponentDefinition, ComponentUnit, ExecutionUnitAddress,
    StateLayout,
};

pub(super) fn definition(
    id: &'static str,
    parameters: ArchitectureParameterDescription,
    state: StateLayout,
    target_transport: eredu_runtime::ArchitectureGroupTransport,
    path: impl Fn(ExecutionUnitAddress, usize) -> String,
) -> Result<ComponentDefinition, Error> {
    let layout = parameters.unit_layout();
    let units = (0..layout.len())
        .map(|ordinal| {
            let address = layout.address(ordinal).expect("declared DeepSeek unit");
            ComponentUnit {
                path: path(address, ordinal),
                state: ordinal..ordinal + 1,
            }
        })
        .collect();
    let predictions = parameters
        .graph()
        .groups()
        .iter()
        .skip(1)
        .map(|group| group.id().to_owned())
        .collect();
    let transport = (0..parameters.graph().groups().len())
        .map(|group| {
            if group == 0 {
                target_transport.clone()
            } else {
                let mut transport = crate::transport::prediction();
                if group == 1 {
                    transport.first_owner_static_roles.push("mtp".into());
                }
                transport
            }
        })
        .collect();
    let state_partition = state_partition_plan(&parameters, &state);
    ComponentDefinition::new(
        id,
        parameters,
        units,
        transport,
        crate::decoder::TARGET_EXECUTION_GROUP,
        predictions,
        state,
        state_partition,
    )
    .map_err(Error::from)
}

pub(super) fn begin_group<T: Clone>(
    definition: &ComponentDefinition,
    group: usize,
    initial: &T,
    dependencies: &[&T],
) -> Result<T, Error> {
    definition.validate_group_dependencies(group, dependencies.len())?;
    Ok(dependencies.first().copied().unwrap_or(initial).clone())
}

pub(super) fn state_partition_plan(
    parameters: &ArchitectureParameterDescription,
    state: &StateLayout,
) -> eredu_runtime::ArchitectureStatePartitionPlan {
    let targets = parameters
        .unit_layout()
        .group_range(0)
        .expect("declared DeepSeek target group")
        .len();
    crate::transport::pipeline_with_output_state(0, targets, state)
}

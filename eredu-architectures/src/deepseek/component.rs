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
    let units = units(&parameters, path);
    let predictions = predictions(&parameters);
    let transport = transport(&parameters, target_transport);
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

fn units(
    parameters: &ArchitectureParameterDescription,
    path: impl Fn(ExecutionUnitAddress, usize) -> String,
) -> Vec<ComponentUnit> {
    let layout = parameters.unit_layout();
    (0..layout.len())
        .map(|ordinal| {
            let address = layout.address(ordinal).expect("declared DeepSeek unit");
            ComponentUnit {
                path: path(address, ordinal),
                state: ordinal..ordinal + 1,
            }
        })
        .collect()
}
fn predictions(parameters: &ArchitectureParameterDescription) -> Vec<String> {
    parameters
        .graph()
        .groups()
        .iter()
        .skip(1)
        .map(|group| group.id().to_owned())
        .collect()
}
fn transport(
    parameters: &ArchitectureParameterDescription,
    target_transport: eredu_runtime::ArchitectureGroupTransport,
) -> Vec<eredu_runtime::ArchitectureGroupTransport> {
    (0..parameters.graph().groups().len())
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
        .collect()
}
pub(super) fn validate(
    definition: &ComponentDefinition,
    target_transport: eredu_runtime::ArchitectureGroupTransport,
    path: impl Fn(ExecutionUnitAddress, usize) -> String,
    observations: impl Fn(
        ExecutionUnitAddress,
        &ComponentUnit,
        &eredu_runtime::UnitObservationBoundaries,
    ) -> bool,
) -> Result<(), Error> {
    definition.validate_execution_contract(
        crate::decoder::TARGET_EXECUTION_GROUP,
        &predictions(definition.parameters()),
        &units(definition.parameters(), path),
        &transport(definition.parameters(), target_transport),
        &state_partition_plan(definition.parameters(), definition.state_layout()),
        observations,
    )?;
    Ok(())
}
pub(super) fn project_parameters(
    id: &str,
    parameters: ArchitectureParameterDescription,
    definition: &std::sync::Arc<ComponentDefinition>,
) -> Result<std::sync::Arc<ComponentDefinition>, Error> {
    if definition.id() != id {
        return Err(eredu_runtime::ComponentDefinitionError::Invalid {
            component: definition.id().into(),
            resource: format!("expected component {id}"),
        }
        .into());
    }
    definition
        .project_parameters(parameters)
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

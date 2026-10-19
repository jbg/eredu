//! Exact ownership and replicated scalar geometry of post-projection gates.
use super::*;
use eredu_core::{
    capture::CaptureError,
    component::{ComponentReadRole, ComponentRowMapping},
    ObservationPosition as Position, SymbolicDimension, TensorAxis,
};

pub(super) fn insert_observations(
    observations: &mut BTreeMap<String, PartitionedObservation>,
    descriptor: &ArchitectureDescriptor,
    component: &ComponentGroup,
    layout: &LocalModelLayout,
    local: bool,
    owns: &impl Fn(&str) -> Result<bool, ComponentPartitionError>,
) -> Result<(), ComponentPartitionError> {
    let gate = component
        .output_gate
        .as_ref()
        .expect("declared output gate");
    let invalid =
        || CaptureError::Invalid(format!("invalid scalar output gate for {}", component.id));
    let point = |path: &str, position: Option<Position>| {
        let point = descriptor.observations.get(path).ok_or_else(&invalid)?;
        if position.is_some_and(|position| point.position != position) {
            return Err(invalid());
        }
        point.axes.as_deref().ok_or_else(&invalid)
    };
    let input = point(&gate.input, None)?;
    if !descriptor
        .observations
        .get(&gate.input)
        .is_some_and(|point| {
            matches!(
                point.position,
                Position::ReadOnly | Position::AfterIntervention
            )
        })
        || input.len() != 3
        || input[0].name != "batch"
        || input[1].name != "sequence"
        || gate.read.role != ComponentReadRole::OutputGate
        || gate.read.head_normalization.is_some()
        || component.count == 0
        || input != point(&gate.projection_input, Some(Position::ReadOnly))?
    {
        return Err(invalid().into());
    }
    let TensorAxis {
        name,
        dimension: SymbolicDimension::Known(width),
    } = &input[2]
    else {
        return Err(invalid().into());
    };
    if name != "hidden" || *width == 0 {
        return Err(invalid().into());
    }
    // A scalar gate has one read row shared by every component. Validate the
    // complete mapping algebraically, without enumerating a large unit group.
    let shared_row = match gate.read.rows {
        ComponentRowMapping::HeadRows {
            offset: 0,
            component_head_width,
            read_head_width: 1,
            component_heads_per_read_head,
            ..
        } => component_head_width
            .checked_mul(component_heads_per_read_head)
            .is_some_and(|covered| covered >= component.count),
        ComponentRowMapping::Direct { offset: 0 } => component.count == 1,
        _ => false,
    };
    let output = point(&gate.output, Some(Position::BeforeIntervention))?;
    if !shared_row
        || output.len() != 3
        || output[..2] != input[..2]
        || output[2].name != "gate"
        || output[2].dimension != SymbolicDimension::Known(1)
        || output != point(&gate.effective_output, Some(Position::AfterIntervention))?
        || owns(&gate.read.weight)? != local
    {
        return Err(invalid().into());
    }
    for (parameter, shape) in std::iter::once((&gate.read.weight, vec![1, *width]))
        .chain(gate.read.bias.iter().map(|bias| (bias, vec![1])))
    {
        if owns(parameter)? != local {
            return Err(invalid().into());
        }
        if local {
            let tensor = layout
                .tensor(parameter)
                .ok_or_else(|| ComponentPartitionError::MissingWeight(parameter.clone()))?;
            if tensor.global_shape() != shape
                || tensor.local_shape() != shape
                || !matches!(
                    tensor.placement(),
                    TensorPlacement::Local | TensorPlacement::Replicated
                )
                || !tensor.additional_placements().is_empty()
            {
                return Err(ComponentPartitionError::InvalidPlacement(parameter.clone()));
            }
        }
    }
    for (path, axis) in [
        (&gate.input, "hidden"),
        (&gate.projection_input, "hidden"),
        (&gate.output, "gate"),
        (&gate.effective_output, "gate"),
    ] {
        replicated_observation(
            observations,
            descriptor,
            path,
            axis,
            local,
            ObservationHookSite::Unit,
        )?;
    }
    Ok(())
}

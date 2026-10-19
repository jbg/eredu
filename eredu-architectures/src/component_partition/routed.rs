//! Exact routed coordinates from retained bank ownership and physical write layout.
use super::*;
use eredu_core::component::{
    RoutedComponentCoordinateMap, RoutedComponentGroup, RoutedComponentParameter,
};

/// Derives one selected bank's global expert and scalar-unit coordinates. The
/// grouped realization supplies execution ownership; storing an independent
/// expert matrix does not establish that this rank executes that expert.
/// Invocation ownership, capture admission and distributed receipts stay separate.
pub fn derive_routed_component_coordinates(
    component: &RoutedComponentGroup,
    layout: &LocalModelLayout,
    plan: &crate::routed_text::RoutedGroupedPlan,
) -> Result<RoutedComponentCoordinateMap, ComponentPartitionError> {
    let invalid = || ComponentPartitionError::InvalidPlacement(component.id.clone());
    if component.expert_count != plan.global_group_count()
        || component.units_per_expert == 0
        || matches!(plan, crate::routed_text::RoutedGroupedPlan::Linear(_))
    {
        return Err(invalid());
    }
    derive_coordinates_for_experts(component, layout, plan.local_global_group_indices())
}

pub(crate) fn derive_coordinates_for_experts(
    component: &RoutedComponentGroup,
    layout: &LocalModelLayout,
    local_groups: &[usize],
) -> Result<RoutedComponentCoordinateMap, ComponentPartitionError> {
    let invalid = || ComponentPartitionError::InvalidPlacement(component.id.clone());
    let experts = ComponentCoordinateMap::indices(component.expert_count, local_groups.to_vec())?;
    if experts.local_count() == 0 {
        return Ok(RoutedComponentCoordinateMap::new(
            experts,
            ComponentCoordinateMap::range(component.units_per_expert, 0..0)?,
        ));
    }
    let units = match &component.write_weight {
        RoutedComponentParameter::Packed { name } => {
            let tensor = layout
                .tensor(&name.parameter)
                .ok_or_else(|| ComponentPartitionError::MissingWeight(name.parameter.clone()))?;
            let (stored_experts, units) = packed_coordinates(
                &component.id,
                component.expert_count,
                component.units_per_expert,
                component.output_width,
                tensor,
            )?;
            if local_groups
                .iter()
                .any(|expert| stored_experts.global_to_local(*expert).is_none())
            {
                return Err(invalid());
            }
            units
        }
        RoutedComponentParameter::Independent { names } => {
            if names.len() != component.expert_count {
                return Err(invalid());
            }
            let mut units = None;
            for expert in local_groups {
                let name = names[*expert].as_ref().ok_or_else(invalid)?;
                let tensor = layout.tensor(&name.parameter).ok_or_else(|| {
                    ComponentPartitionError::MissingWeight(name.parameter.clone())
                })?;
                if tensor.global_shape().first() != Some(&component.output_width) {
                    return Err(invalid());
                }
                let next =
                    derive_write_coordinates(&name.parameter, component.units_per_expert, tensor)?;
                if units.as_ref().is_some_and(|units| *units != next) {
                    return Err(invalid());
                }
                units = Some(next);
            }
            units.ok_or_else(invalid)?
        }
    };
    Ok(RoutedComponentCoordinateMap::new(experts, units))
}

fn packed_coordinates(
    name: &str,
    expert_count: usize,
    units_per_expert: usize,
    output_width: usize,
    tensor: &LocalTensorLayout,
) -> Result<(ComponentCoordinateMap, ComponentCoordinateMap), ComponentPartitionError> {
    let invalid = || ComponentPartitionError::InvalidPlacement(name.into());
    let global = tensor.global_shape();
    let local = tensor.local_shape();
    if global.len() != 3
        || local.len() != 3
        || global.contains(&0)
        || local.contains(&0)
        || global[0] != expert_count
        || global[1] != output_width
        || global[1] != local[1]
    {
        return Err(invalid());
    }
    let mut expert_map = None;
    let mut unit_placement = None;
    for placement in tensor
        .additional_placements()
        .iter()
        .chain(std::iter::once(tensor.placement()))
    {
        let (axis, normalized) = match placement {
            TensorPlacement::Replicated | TensorPlacement::Local => continue,
            TensorPlacement::Range { axis, start, end } => (
                *axis,
                TensorPlacement::Range {
                    axis: 1,
                    start: *start,
                    end: *end,
                },
            ),
            TensorPlacement::Shard { axis, index, parts } => (
                *axis,
                TensorPlacement::Shard {
                    axis: 1,
                    index: *index,
                    parts: *parts,
                },
            ),
            TensorPlacement::Indices { axis, indices } => (
                *axis,
                TensorPlacement::Indices {
                    axis: 1,
                    indices: indices.clone(),
                },
            ),
            _ => return Err(invalid()),
        };
        match axis {
            0 if expert_map.is_none() => {
                expert_map = Some(match normalized {
                    TensorPlacement::Range { start, end, .. } => {
                        ComponentCoordinateMap::range(global[0], start..end)?
                    }
                    TensorPlacement::Indices { indices, .. } => {
                        ComponentCoordinateMap::indices(global[0], indices)?
                    }
                    TensorPlacement::Shard { index, parts, .. }
                        if parts > 0 && index < parts && global[0].is_multiple_of(parts) =>
                    {
                        let width = global[0] / parts;
                        ComponentCoordinateMap::range(
                            global[0],
                            index * width..(index + 1) * width,
                        )?
                    }
                    _ => return Err(invalid()),
                });
            }
            2 if unit_placement.is_none() => unit_placement = Some(normalized),
            _ => return Err(invalid()),
        }
    }
    let experts = expert_map.unwrap_or(ComponentCoordinateMap::range(global[0], 0..global[0])?);
    if experts.local_count() != local[0] {
        return Err(invalid());
    }
    // Reuse the dense write-axis proof, including packed-column scaling and
    // retained semantic ranges. No encoding factor is guessed from dtype/names.
    let matrix = LocalTensorLayout::new(
        tensor.logical_name(),
        tensor.role(),
        global[1..].to_vec(),
        local[1..].to_vec(),
        unit_placement.unwrap_or(TensorPlacement::Replicated),
        tensor.logical_units(),
        tensor.logical_range().cloned(),
        tensor.fell_back_to_replication(),
    )
    .with_partition_chunk_size(tensor.partition_chunk_size());
    let units = derive_write_coordinates(name, units_per_expert, &matrix)?;
    Ok((experts, units))
}

mod bank;
pub(crate) use bank::derive_bank_unit_coordinates;

pub(super) mod placement;
pub use placement::PartitionedRoutedObservation;

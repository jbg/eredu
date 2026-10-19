//! Grouped write factors retain scalar ownership independently of latent rank.
use super::*;
use eredu_core::{component::ComponentGroupedWriteProjection, SymbolicDimension};

fn latent_width(stage: &ComponentGroupedWriteProjection) -> Result<usize, ComponentPartitionError> {
    stage
        .groups
        .checked_mul(stage.rank)
        .filter(|n| *n > 0)
        .ok_or_else(|| ComponentPartitionError::InvalidPlacement(stage.weight.clone()))
}

fn row_coordinates(
    stage: &ComponentGroupedWriteProjection,
    tensor: &LocalTensorLayout,
) -> Result<ComponentCoordinateMap, ComponentPartitionError> {
    let width = latent_width(stage)?;
    if tensor.global_shape().first() != Some(&width) {
        return Err(ComponentPartitionError::InvalidPlacement(
            stage.weight.clone(),
        ));
    }
    derive_matrix_axis_coordinates(&stage.weight, width, tensor, 0)
}

pub(super) fn component_coordinates(
    count: usize,
    stage: &ComponentGroupedWriteProjection,
    tensor: &LocalTensorLayout,
) -> Result<ComponentCoordinateMap, ComponentPartitionError> {
    let invalid = || ComponentPartitionError::InvalidPlacement(stage.weight.clone());
    let rows = row_coordinates(stage, tensor)?;
    if count == 0 || !count.is_multiple_of(stage.groups) {
        return Err(invalid());
    }
    let per_group = count / stage.groups;
    if let Some(range) = rows.contiguous_range() {
        if !range.start.is_multiple_of(stage.rank) || !range.end.is_multiple_of(stage.rank) {
            return Err(invalid());
        }
        return ComponentCoordinateMap::range(
            count,
            range.start / stage.rank * per_group..range.end / stage.rank * per_group,
        )
        .map_err(Into::into);
    }
    if !rows.local_count().is_multiple_of(stage.rank) {
        return Err(invalid());
    }
    let mut components = Vec::new();
    for start in (0..rows.local_count()).step_by(stage.rank) {
        let first = rows.local_to_global(start).ok_or_else(invalid)?;
        if !first.is_multiple_of(stage.rank)
            || (0..stage.rank)
                .any(|offset| rows.local_to_global(start + offset) != Some(first + offset))
        {
            return Err(invalid());
        }
        let first = first / stage.rank * per_group;
        components.extend(first..first + per_group);
    }
    ComponentCoordinateMap::indices(count, components).map_err(Into::into)
}

fn paired_coordinates(
    stage: &ComponentGroupedWriteProjection,
    output_name: &str,
    first: &LocalTensorLayout,
    last: &LocalTensorLayout,
) -> Result<ComponentCoordinateMap, ComponentPartitionError> {
    let input = row_coordinates(stage, first)?;
    let output = derive_matrix_axis_coordinates(output_name, latent_width(stage)?, last, 1)?;
    // Compare actual local order, accepting equivalent compact representations.
    let same = input.global_count() == output.global_count()
        && input.local_count() == output.local_count()
        && match (input.contiguous_range(), output.contiguous_range()) {
            (Some(a), Some(b)) => a == b,
            _ => (0..input.local_count())
                .all(|i| input.local_to_global(i) == output.local_to_global(i)),
        };
    if !same {
        return Err(ComponentPartitionError::InvalidPlacement(
            output_name.into(),
        ));
    }
    Ok(input)
}

pub(super) fn insert_observations(
    observations: &mut BTreeMap<String, PartitionedObservation>,
    descriptor: &ArchitectureDescriptor,
    component: &ComponentGroup,
    stage: &ComponentGroupedWriteProjection,
    layout: Option<&LocalModelLayout>,
) -> Result<(), ComponentPartitionError> {
    let width = latent_width(stage)?;
    let coordinates = layout
        .map(|layout| {
            let tensor = |name: &str| {
                layout
                    .tensor(name)
                    .ok_or_else(|| ComponentPartitionError::MissingWeight(name.into()))
            };
            paired_coordinates(
                stage,
                &component.write_weight,
                tensor(&stage.weight)?,
                tensor(&component.write_weight)?,
            )
        })
        .transpose()?;
    let groups = layout
        .map(|layout| {
            let tensor = layout
                .tensor(&stage.weight)
                .ok_or_else(|| ComponentPartitionError::MissingWeight(stage.weight.clone()))?;
            component_coordinates(stage.groups, stage, tensor)
        })
        .transpose()?;
    for (path, axis, count, coordinates) in [
        (&stage.input, "group", stage.groups, &groups),
        (&stage.output, "projection", width, &coordinates),
        (&stage.final_input, "projection", width, &coordinates),
    ] {
        let point = descriptor
            .observations
            .get(path)
            .ok_or_else(|| eredu_core::capture::CaptureError::MissingPath(path.clone()))?;
        let mut axes = point
            .axes
            .iter()
            .flatten()
            .filter(|candidate| candidate.name == axis);
        if !axes
            .next()
            .is_some_and(|axis| axis.dimension == SymbolicDimension::Known(count))
            || axes.next().is_some()
        {
            return Err(ComponentPartitionError::InvalidPlacement(
                stage.weight.clone(),
            ));
        }
        super::insert_observation(
            observations,
            path,
            PartitionedObservation {
                axis: axis.into(),
                coordinates: coordinates.clone(),
                exports: layout.is_some(),
                site: ObservationHookSite::Unit,
                combination: PartitionCaptureCombination::Disjoint,
            },
        )?;
    }
    Ok(())
}

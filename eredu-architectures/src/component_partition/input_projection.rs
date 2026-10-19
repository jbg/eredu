//! Coordinates of declared intermediate read stages, before the terminal read.

use super::*;
use eredu_core::{component::ComponentInputProjection, SymbolicDimension};

pub(super) fn insert_observation(
    observations: &mut BTreeMap<String, PartitionedObservation>,
    descriptor: &ArchitectureDescriptor,
    stage: &ComponentInputProjection,
    tensor: Option<&LocalTensorLayout>,
) -> Result<(), ComponentPartitionError> {
    insert_row_output(
        observations,
        descriptor,
        &stage.weight,
        &stage.rows,
        &stage.output,
        tensor,
    )
}

/// Registers an actual affine read output independently of later normalization.
pub(super) fn insert_read_output(
    observations: &mut BTreeMap<String, PartitionedObservation>,
    descriptor: &ArchitectureDescriptor,
    read: &eredu_core::component::ComponentRead,
    tensor: Option<&LocalTensorLayout>,
) -> Result<(), ComponentPartitionError> {
    let Some(path) = &read.projection_output else {
        return Ok(());
    };
    let point = descriptor
        .observations
        .get(path)
        .ok_or_else(|| eredu_core::capture::CaptureError::MissingPath(path.clone()))?;
    let width = point
        .axes
        .as_ref()
        .and_then(|axes| axes.iter().find(|axis| axis.name == "hidden"))
        .and_then(|axis| match axis.dimension {
            SymbolicDimension::Known(width) => Some(width),
            _ => None,
        })
        .ok_or_else(|| ComponentPartitionError::InvalidPlacement(read.weight.clone()))?;
    if point.position != eredu_core::ObservationPosition::ReadOnly
        || tensor.is_some_and(|tensor| tensor.global_shape().first() != Some(&width))
    {
        return Err(ComponentPartitionError::InvalidPlacement(
            read.weight.clone(),
        ));
    }
    insert_row_output(
        observations,
        descriptor,
        &read.weight,
        &(0..width),
        path,
        tensor,
    )
}

fn insert_row_output(
    observations: &mut BTreeMap<String, PartitionedObservation>,
    descriptor: &ArchitectureDescriptor,
    weight: &str,
    rows: &std::ops::Range<usize>,
    output: &str,
    tensor: Option<&LocalTensorLayout>,
) -> Result<(), ComponentPartitionError> {
    let width = rows
        .end
        .checked_sub(rows.start)
        .filter(|n| *n > 0)
        .ok_or_else(|| ComponentPartitionError::InvalidPlacement(weight.to_owned()))?;
    let coordinates = tensor
        .map(|tensor| derive_row_coordinates(weight, rows, tensor))
        .transpose()?;
    let placement = PartitionedObservation {
        axis: "hidden".into(),
        coordinates,
        exports: tensor.is_some(),
        site: ObservationHookSite::Unit,
        combination: PartitionCaptureCombination::Disjoint,
    };
    // Stage declarations name the effective tensor. Register its original seam
    // as well so interventions act before the value is consumed or cached.
    for path in std::iter::once(output).chain(output.strip_suffix(".effective")) {
        let point = descriptor
            .observations
            .points
            .iter()
            .find(|point| point.path == path)
            .ok_or_else(|| eredu_core::capture::CaptureError::MissingPath(path.into()))?;
        let mut axes = point
            .axes
            .iter()
            .flatten()
            .filter(|axis| axis.name == "hidden");
        if !axes
            .next()
            .is_some_and(|axis| axis.dimension == SymbolicDimension::Known(width))
            || axes.next().is_some()
        {
            return Err(ComponentPartitionError::InvalidPlacement(weight.to_owned()));
        }
        super::insert_observation(observations, path, placement.clone())?;
    }
    Ok(())
}

fn derive_row_coordinates(
    weight: &str,
    selected: &std::ops::Range<usize>,
    tensor: &LocalTensorLayout,
) -> Result<ComponentCoordinateMap, ComponentPartitionError> {
    let invalid = || ComponentPartitionError::InvalidPlacement(weight.to_owned());
    let count = *tensor.global_shape().first().ok_or_else(invalid)?;
    if selected.start >= selected.end || selected.end > count {
        return Err(invalid());
    }
    // Output rows are scalar coordinates even if the input columns are packed.
    // Column-sharded partial products and multiple physical placements need an
    // explicit complete-output contract; they cannot masquerade as row shards.
    let rows = derive_matrix_axis_coordinates(weight, count, tensor, 0)?;
    let width = selected.end - selected.start;
    if let Some(range) = rows.contiguous_range() {
        let start = range.start.clamp(selected.start, selected.end) - selected.start;
        let end = range.end.clamp(selected.start, selected.end) - selected.start;
        ComponentCoordinateMap::range(width, start..end).map_err(Into::into)
    } else {
        ComponentCoordinateMap::indices(
            width,
            (0..rows.local_count())
                .filter_map(|index| rows.local_to_global(index))
                .filter(|row| selected.contains(row))
                .map(|row| row - selected.start)
                .collect(),
        )
        .map_err(Into::into)
    }
}

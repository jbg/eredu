//! One coordinate projection worker for ordinary and original source construction.
use super::*;

/// Derives the actual scalar input axis of an architecture-declared write matrix.
/// Semantic units determine scalar offsets even when physical columns are packed.
/// Pipeline invocation ownership must be established separately by the caller.
pub(super) fn component_worker(
    component: &ComponentGroup,
    tensor: &LocalTensorLayout,
    allocation: Destination<'_>,
) -> Result<ComponentCoordinateMap, ComponentPartitionError> {
    allocation.controls::<(
        &str,
        usize,
        &LocalTensorLayout,
        usize,
        ComponentCoordinateMap,
        ComponentPartitionError,
    )>()?;
    allocation.controls::<&ComponentGroup>()?;
    if let Some(stage) = &component.write_input_projection {
        return output_projection::component_worker(component.count, stage, tensor, allocation);
    }
    write_worker(&component.write_weight, component.count, tensor, allocation)
}

pub(super) fn write_worker(
    name: &str,
    count: usize,
    tensor: &LocalTensorLayout,
    allocation: Destination<'_>,
) -> Result<ComponentCoordinateMap, ComponentPartitionError> {
    allocation.controls::<(
        &str,
        usize,
        &LocalTensorLayout,
        usize,
        ComponentCoordinateMap,
        ComponentPartitionError,
    )>()?;
    matrix_worker(name, count, tensor, 1, allocation)
}

pub(super) fn matrix_worker(
    name: &str,
    count: usize,
    tensor: &LocalTensorLayout,
    axis: usize,
    allocation: Destination<'_>,
) -> Result<ComponentCoordinateMap, ComponentPartitionError> {
    allocation.controls::<(
        &str,
        usize,
        &LocalTensorLayout,
        usize,
        ComponentCoordinateMap,
        ComponentPartitionError,
    )>()?;
    if tensor.global_shape().len() != 2 || axis >= 2 {
        return Err(allocation.invalid(name));
    }
    tensor_worker(name, count, tensor, axis, allocation)
}

pub(super) fn tensor_worker(
    name: &str,
    count: usize,
    tensor: &LocalTensorLayout,
    axis: usize,
    allocation: Destination<'_>,
) -> Result<ComponentCoordinateMap, ComponentPartitionError> {
    allocation.controls::<(
        &str,
        usize,
        &LocalTensorLayout,
        usize,
        ComponentCoordinateMap,
        ComponentPartitionError,
    )>()?;
    let invalid = || allocation.invalid(name);
    let global = tensor.global_shape();
    let local = tensor.local_shape();
    if axis >= global.len()
        || global.len() != local.len()
        || global.contains(&0)
        || local.contains(&0)
        || global
            .iter()
            .zip(local)
            .enumerate()
            .any(|(index, (global, local))| index != axis && global != local)
        || !tensor.additional_placements().is_empty()
    {
        return Err(invalid());
    }
    let full = || ComponentCoordinateMap::range(count, 0..count).map_err(Into::into);
    match tensor.placement() {
        TensorPlacement::Replicated | TensorPlacement::Local if global == local => full(),
        TensorPlacement::Range {
            axis: selected_axis,
            start,
            end,
        } if *selected_axis == axis
            && start <= end
            && *end <= global[axis]
            && end - start == local[axis] =>
        {
            let units = tensor.logical_units().ok_or_else(invalid)?;
            let range = tensor.logical_range().ok_or_else(invalid)?;
            if let Some(chunk) = tensor.partition_chunk_size() {
                if chunk == 0 || global[axis].div_ceil(chunk) != units {
                    return Err(invalid());
                }
                let physical =
                    eredu_runtime::partition_chunk_range(global[axis], chunk, range.clone())
                        .map_err(|_| invalid())?;
                if physical != (*start..*end) {
                    return Err(invalid());
                }
                // FP8 retains one physical code per component. A short final
                // chunk therefore maps directly, without uniform-unit rounding.
                if global[axis] == count {
                    return ComponentCoordinateMap::range(count, physical).map_err(Into::into);
                }
                // Packed columns can use uniform semantic units only when
                // every physical chunk is complete. Partial packed encodings
                // need a separately declared scalar layout.
                if global[axis].is_multiple_of(chunk) {
                    return ComponentCoordinateMap::partition_units(count, units, range.clone())
                        .map_err(Into::into);
                }
                return Err(invalid());
            }
            // Verify the physical selection agrees with the retained semantic
            // units before expanding them to scalar component coordinates.
            if units == 0
                || !global[axis].is_multiple_of(units)
                || range.start > range.end
                || range.end > units
            {
                return Err(invalid());
            }
            let physical_per_unit = global[axis] / units;
            if *start != range.start * physical_per_unit || *end != range.end * physical_per_unit {
                return Err(invalid());
            }
            ComponentCoordinateMap::partition_units(count, units, range.clone()).map_err(Into::into)
        }
        TensorPlacement::Shard {
            axis: selected_axis,
            index,
            parts,
        } if *selected_axis == axis
            && *parts > 0
            && index < parts
            && global[axis].is_multiple_of(*parts)
            && local[axis] == global[axis] / parts =>
        {
            let units = tensor.logical_units().ok_or_else(invalid)?;
            let range = tensor.logical_range().ok_or_else(invalid)?;
            if units == 0
                || !units.is_multiple_of(*parts)
                || !global[axis].is_multiple_of(units)
                || range != &(index * (units / parts)..(index + 1) * (units / parts))
            {
                return Err(invalid());
            }
            ComponentCoordinateMap::partition_units(count, units, range.clone()).map_err(Into::into)
        }
        // An indexed physical axis is a scalar axis only when unpacked. Packed
        // layouts need an explicit semantic index map, not multiplication guesses.
        TensorPlacement::Indices {
            axis: selected_axis,
            indices,
        } if *selected_axis == axis && global[axis] == count && local[axis] == indices.len() => {
            allocation.indices(count, allocation.copy(indices)?)
        }
        _ => Err(invalid()),
    }
}

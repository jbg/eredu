//! Grouped write factors retain scalar ownership independently of latent rank.
use super::*;
use eredu_core::{SymbolicDimension, component::ComponentGroupedWriteProjection};

fn latent_width(stage: &ComponentGroupedWriteProjection) -> Result<usize, ComponentPartitionError> {
    latent_width_worker(stage, Destination(None))
}
fn row_coordinates(
    stage: &ComponentGroupedWriteProjection,
    tensor: &LocalTensorLayout,
) -> Result<ComponentCoordinateMap, ComponentPartitionError> {
    row_worker(stage, tensor, Destination(None))
}
pub(super) fn component_coordinates(
    count: usize,
    stage: &ComponentGroupedWriteProjection,
    tensor: &LocalTensorLayout,
) -> Result<ComponentCoordinateMap, ComponentPartitionError> {
    component_worker(count, stage, tensor, Destination(None))
}
fn latent_width_worker(
    stage: &ComponentGroupedWriteProjection,
    allocation: Destination<'_>,
) -> Result<usize, ComponentPartitionError> {
    allocation.controls::<(&ComponentGroupedWriteProjection, usize)>()?;
    stage
        .groups
        .checked_mul(stage.rank)
        .filter(|n| *n > 0)
        .ok_or_else(|| allocation.invalid(&stage.weight))
}

fn row_worker(
    stage: &ComponentGroupedWriteProjection,
    tensor: &LocalTensorLayout,
    allocation: Destination<'_>,
) -> Result<ComponentCoordinateMap, ComponentPartitionError> {
    allocation.controls::<(
        usize,
        &ComponentGroupedWriteProjection,
        &LocalTensorLayout,
        ComponentCoordinateMap,
        Vec<usize>,
        std::ops::Range<usize>,
        [usize; 4],
    )>()?;
    let width = latent_width_worker(stage, allocation)?;
    if tensor.global_shape().first() != Some(&width) {
        return Err(allocation.invalid(&stage.weight));
    }
    coordinates::matrix_worker(&stage.weight, width, tensor, 0, allocation)
}

pub(super) fn component_worker(
    count: usize,
    stage: &ComponentGroupedWriteProjection,
    tensor: &LocalTensorLayout,
    allocation: Destination<'_>,
) -> Result<ComponentCoordinateMap, ComponentPartitionError> {
    allocation.controls::<(
        usize,
        &ComponentGroupedWriteProjection,
        &LocalTensorLayout,
        ComponentCoordinateMap,
        Vec<usize>,
        std::ops::Range<usize>,
        [usize; 4],
    )>()?;
    let invalid = || allocation.invalid(&stage.weight);
    let rows = row_worker(stage, tensor, allocation)?;
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
        for index in first..first + per_group {
            allocation.push(&mut components, index)?;
        }
    }
    allocation.indices(count, components)
}

fn paired_coordinates(
    stage: &ComponentGroupedWriteProjection,
    output_name: &str,
    first: &LocalTensorLayout,
    last: &LocalTensorLayout,
) -> Result<ComponentCoordinateMap, ComponentPartitionError> {
    paired_worker(stage, output_name, first, last, Destination(None))
}
pub(super) fn insert_observations(
    observations: &mut SourceMap<String, PartitionedObservation>,
    descriptor: &ArchitectureDescriptor,
    component: &ComponentGroup,
    stage: &ComponentGroupedWriteProjection,
    layout: Option<&LocalModelLayout>,
) -> Result<(), ComponentPartitionError> {
    worker(
        observations,
        descriptor,
        component,
        stage,
        layout,
        Destination(None),
    )
}
fn paired_worker(
    stage: &ComponentGroupedWriteProjection,
    output_name: &str,
    first: &LocalTensorLayout,
    last: &LocalTensorLayout,
    allocation: Destination<'_>,
) -> Result<ComponentCoordinateMap, ComponentPartitionError> {
    allocation.controls::<(
        &ComponentGroupedWriteProjection,
        &str,
        &LocalTensorLayout,
        &LocalTensorLayout,
        ComponentCoordinateMap,
        ComponentCoordinateMap,
        bool,
    )>()?;
    let input = row_worker(stage, first, allocation)?;
    let output = coordinates::matrix_worker(
        output_name,
        latent_width_worker(stage, allocation)?,
        last,
        1,
        allocation,
    )?;
    // Compare actual local order, accepting equivalent compact representations.
    let same = input.global_count() == output.global_count()
        && input.local_count() == output.local_count()
        && match (input.contiguous_range(), output.contiguous_range()) {
            (Some(a), Some(b)) => a == b,
            _ => (0..input.local_count())
                .all(|i| input.local_to_global(i) == output.local_to_global(i)),
        };
    if !same {
        return Err(allocation.invalid(output_name));
    }
    Ok(input)
}

pub(super) fn worker(
    observations: &mut SourceMap<String, PartitionedObservation>,
    descriptor: &ArchitectureDescriptor,
    component: &ComponentGroup,
    stage: &ComponentGroupedWriteProjection,
    layout: Option<&LocalModelLayout>,
    allocation: Destination<'_>,
) -> Result<(), ComponentPartitionError> {
    allocation.controls::<(
        &mut SourceMap<String, PartitionedObservation>,
        &ArchitectureDescriptor,
        &ComponentGroup,
        &ComponentGroupedWriteProjection,
        Option<&LocalModelLayout>,
        Option<ComponentCoordinateMap>,
        Option<ComponentCoordinateMap>,
        usize,
    )>()?;
    let width = latent_width_worker(stage, allocation)?;
    let coordinates = layout
        .map(|layout| {
            let tensor = |name: &str| layout.tensor(name).ok_or_else(|| allocation.missing(name));
            paired_worker(
                stage,
                &component.write_weight,
                tensor(&stage.weight)?,
                tensor(&component.write_weight)?,
                allocation,
            )
        })
        .transpose()?;
    let groups = layout
        .map(|layout| {
            let tensor = layout
                .tensor(&stage.weight)
                .ok_or_else(|| allocation.missing(&stage.weight))?;
            component_worker(stage.groups, stage, tensor, allocation)
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
            .ok_or_else(|| allocation.capture_missing(path))?;
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
            return Err(allocation.invalid(&stage.weight));
        }
        observations::insert(
            observations,
            path,
            PartitionedObservation {
                axis: allocation.text(axis)?,
                coordinates: allocation.optional_coordinates(coordinates.as_ref())?,
                exports: layout.is_some(),
                site: ObservationHookSite::Unit,
                combination: PartitionCaptureCombination::Disjoint,
            },
            allocation,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage() -> ComponentGroupedWriteProjection {
        ComponentGroupedWriteProjection {
            weight: "first".into(),
            parameter_group: "groups".into(),
            shared_weight: "first".into(),
            bias: None,
            groups: 4,
            rank: 2,
            input: "grouped_input".into(),
            output: "latent".into(),
            final_input: "final_input".into(),
        }
    }

    #[test]
    fn grouped_rows_map_scalar_channels_and_preserve_group_permutations() {
        let stage = stage();
        let layout = |placement, rows, units, range| {
            LocalTensorLayout::new(
                "first",
                eredu_runtime::ParameterRole::AttentionHeads,
                vec![8, 3],
                vec![rows, 3],
                placement,
                units,
                range,
                false,
            )
        };
        let full = layout(TensorPlacement::Replicated, 8, None, None);
        assert_eq!(
            component_coordinates(12, &stage, &full)
                .unwrap()
                .contiguous_range(),
            Some(0..12)
        );
        let shard = layout(
            TensorPlacement::Range {
                axis: 0,
                start: 4,
                end: 8,
            },
            4,
            Some(4),
            Some(2..4),
        );
        assert_eq!(
            component_coordinates(12, &stage, &shard)
                .unwrap()
                .contiguous_range(),
            Some(6..12)
        );
        let permuted = layout(
            TensorPlacement::Indices {
                axis: 0,
                indices: vec![6, 7, 0, 1],
            },
            4,
            None,
            None,
        );
        let coordinates = component_coordinates(12, &stage, &permuted).unwrap();
        assert_eq!(
            (0..coordinates.local_count())
                .map(|i| coordinates.local_to_global(i).unwrap())
                .collect::<Vec<_>>(),
            [9, 10, 11, 0, 1, 2]
        );
        let split = layout(
            TensorPlacement::Indices {
                axis: 0,
                indices: vec![7, 6, 0, 1],
            },
            4,
            None,
            None,
        );
        assert!(component_coordinates(12, &stage, &split).is_err());
        assert!(component_coordinates(13, &stage, &full).is_err());
        let wrong = ComponentGroupedWriteProjection { groups: 2, ..stage };
        assert!(component_coordinates(12, &wrong, &full).is_err());
    }

    #[test]
    fn paired_factors_require_matching_latent_ownership_not_matching_encoding_of_maps() {
        let stage = stage();
        let first = LocalTensorLayout::new(
            "first",
            eredu_runtime::ParameterRole::AttentionHeads,
            vec![8, 3],
            vec![4, 3],
            TensorPlacement::Indices {
                axis: 0,
                indices: vec![4, 5, 6, 7],
            },
            None,
            None,
            false,
        );
        let last = |start| {
            LocalTensorLayout::new(
                "last",
                eredu_runtime::ParameterRole::AttentionHeads,
                vec![7, 8],
                vec![7, 4],
                TensorPlacement::Range {
                    axis: 1,
                    start,
                    end: start + 4,
                },
                Some(4),
                Some(start / 2..start / 2 + 2),
                false,
            )
        };
        let coordinates = paired_coordinates(&stage, "last", &first, &last(4)).unwrap();
        assert_eq!(
            (0..4)
                .map(|i| coordinates.local_to_global(i).unwrap())
                .collect::<Vec<_>>(),
            [4, 5, 6, 7]
        );
        assert!(paired_coordinates(&stage, "last", &first, &last(0)).is_err());
    }
}

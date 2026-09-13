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

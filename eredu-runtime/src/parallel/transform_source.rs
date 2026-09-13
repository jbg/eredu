//! Source coordinates for pointwise format conversion into retained target shards.

use super::{LocalModelLayout, LocalTensorLayout, ParallelPlanError, TensorPlacement};

/// Projects selected target ownership onto the original source tensor geometry.
///
/// Both layouts must describe the same canonical parameters, semantic groups,
/// and axis order. A format may uniformly pack an axis; this operation scales
/// its exact boundaries without rebalancing the source execution. Architecture
/// construction must use the returned shapes for its source-only modules.
pub fn derive_transform_source_layout(
    source: &LocalModelLayout,
    target: &LocalModelLayout,
) -> Result<LocalModelLayout, ParallelPlanError> {
    let mut result = LocalModelLayout::default();
    for (name, original) in source.tensors() {
        let invalid = |reason: &str| {
            ParallelPlanError::InvalidTensor(format!("{name}: transform source {reason}"))
        };
        let selected = target
            .tensor(name)
            .ok_or_else(|| invalid("has no target owner"))?;
        if original.logical_name() != selected.logical_name() || original.role() != selected.role()
        {
            return Err(invalid("changed semantic group or role"));
        }
        let source_shape = original.global_shape();
        let target_shape = selected.global_shape();
        if source_shape.len() != target_shape.len()
            || selected.local_shape().len() != target_shape.len()
            || source_shape.contains(&0)
            || target_shape.contains(&0)
        {
            return Err(invalid("has incompatible axis geometry"));
        }
        let scale = |axis: usize, value: usize| -> Result<usize, ParallelPlanError> {
            let source = *source_shape
                .get(axis)
                .ok_or_else(|| invalid("axis is absent"))?;
            let target = target_shape[axis];
            let scaled = value
                .checked_mul(source)
                .ok_or_else(|| invalid("coordinate overflow"))?;
            if scaled % target != 0 {
                return Err(invalid("boundary cuts through a source element"));
            }
            Ok(scaled / target)
        };
        let remap = |placement: &TensorPlacement| -> Result<TensorPlacement, ParallelPlanError> {
            Ok(match placement {
                TensorPlacement::Range { axis, start, end } => {
                    if start > end
                        || *end
                            > *target_shape
                                .get(*axis)
                                .ok_or_else(|| invalid("axis is absent"))?
                    {
                        return Err(invalid("range exceeds the target axis"));
                    }
                    TensorPlacement::Range {
                        axis: *axis,
                        start: scale(*axis, *start)?,
                        end: scale(*axis, *end)?,
                    }
                }
                TensorPlacement::Indices { axis, indices } => {
                    let extent = *target_shape
                        .get(*axis)
                        .ok_or_else(|| invalid("axis is absent"))?;
                    let mut expanded = Vec::new();
                    for &index in indices {
                        if index >= extent {
                            return Err(invalid("index exceeds the target axis"));
                        }
                        expanded.extend(scale(*axis, index)?..scale(*axis, index + 1)?);
                    }
                    TensorPlacement::Indices {
                        axis: *axis,
                        indices: expanded,
                    }
                }
                TensorPlacement::Shard { axis, index, parts } => {
                    let extent = *source_shape
                        .get(*axis)
                        .ok_or_else(|| invalid("axis is absent"))?;
                    if *parts == 0 || index >= parts || extent % parts != 0 {
                        return Err(invalid("equal shard does not divide the source axis"));
                    }
                    placement.clone()
                }
                _ => placement.clone(),
            })
        };
        let local_shape = selected
            .local_shape()
            .iter()
            .enumerate()
            .map(|(axis, &size)| scale(axis, size))
            .collect::<Result<Vec<_>, _>>()?;
        let chunk = selected
            .partition_chunk_size()
            .map(|width| match selected.placement() {
                TensorPlacement::Range { axis, .. }
                | TensorPlacement::Indices { axis, .. }
                | TensorPlacement::Shard { axis, .. } => scale(*axis, width),
                _ if source_shape == target_shape => Ok(width),
                _ => Err(invalid("packed chunk has no declared partition axis")),
            })
            .transpose()?;
        let mut mapped = LocalTensorLayout::new(
            original.logical_name(),
            original.role(),
            source_shape.to_vec(),
            local_shape,
            remap(selected.placement())?,
            selected.logical_units(),
            selected.logical_range().cloned(),
            selected.fell_back_to_replication(),
        )
        .with_partition_chunk_size(chunk);
        for placement in selected.additional_placements() {
            mapped = mapped.with_additional_placement(remap(placement)?);
        }
        result.insert(name.to_owned(), mapped);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ParameterRole;

    fn layout(
        global: Vec<usize>,
        local: Vec<usize>,
        placement: TensorPlacement,
        units: usize,
        range: std::ops::Range<usize>,
    ) -> LocalModelLayout {
        let mut layout = LocalModelLayout::default();
        layout.insert(
            "down.weight".into(),
            LocalTensorLayout::new(
                "ffn",
                ParameterRole::FeedForwardIntermediate,
                global,
                local,
                placement,
                Some(units),
                Some(range),
                false,
            ),
        );
        layout
    }

    #[test]
    fn unequal_encoded_cuts_select_the_corresponding_source_values() {
        for (source_start, target_start, target_end) in [(0, 0, 8), (48, 8, 12)] {
            let source = layout(
                vec![64, 96],
                vec![64, 48],
                TensorPlacement::Range {
                    axis: 1,
                    start: source_start,
                    end: source_start + 48,
                },
                96,
                source_start..source_start + 48,
            );
            let target = layout(
                vec![64, 12],
                vec![64, target_end - target_start],
                TensorPlacement::Range {
                    axis: 1,
                    start: target_start,
                    end: target_end,
                },
                3,
                target_start / 4..target_end / 4,
            );
            let mapped = derive_transform_source_layout(&source, &target).unwrap();
            let tensor = mapped.tensor("down.weight").unwrap();
            let expected = target_start * 8..target_end * 8;
            assert_eq!(tensor.local_shape(), [64, expected.len()]);
            assert_eq!(
                tensor.placement(),
                &TensorPlacement::Range {
                    axis: 1,
                    start: expected.start,
                    end: expected.end
                }
            );
            assert_eq!(tensor.expanded_logical_range(96).unwrap(), expected);
        }
    }

    #[test]
    fn independent_expert_indices_and_tensor_cuts_preserve_axis_ownership() {
        let source = layout(
            vec![4, 64, 96],
            vec![2, 64, 48],
            TensorPlacement::Range {
                axis: 2,
                start: 0,
                end: 48,
            },
            96,
            0..48,
        );
        let mut target = layout(
            vec![4, 64, 12],
            vec![2, 64, 8],
            TensorPlacement::Range {
                axis: 2,
                start: 0,
                end: 8,
            },
            3,
            0..2,
        );
        let tensor = target
            .tensor("down.weight")
            .unwrap()
            .clone()
            .with_additional_placement(TensorPlacement::Indices {
                axis: 0,
                indices: vec![1, 3],
            });
        target.insert("down.weight".into(), tensor);
        let mapped = derive_transform_source_layout(&source, &target).unwrap();
        let tensor = mapped.tensor("down.weight").unwrap();
        assert_eq!(tensor.local_shape(), [2, 64, 64]);
        assert_eq!(
            tensor.additional_placements(),
            &[TensorPlacement::Indices {
                axis: 0,
                indices: vec![1, 3]
            }]
        );
    }

    #[test]
    fn nonintegral_source_boundaries_are_rejected() {
        let source = layout(
            vec![8],
            vec![4],
            TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 4,
            },
            8,
            0..4,
        );
        let target = layout(
            vec![3],
            vec![1],
            TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 1,
            },
            3,
            0..1,
        );
        assert!(derive_transform_source_layout(&source, &target).is_err());
    }
}

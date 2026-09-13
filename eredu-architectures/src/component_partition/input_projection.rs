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

#[cfg(test)]
fn derive_coordinates(
    stage: &ComponentInputProjection,
    tensor: &LocalTensorLayout,
) -> Result<ComponentCoordinateMap, ComponentPartitionError> {
    derive_row_coordinates(&stage.weight, &stage.rows, tensor)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_read_outputs_follow_actual_rows_and_reject_partial_products() {
        use eredu_core::ModelConfigurationResolver;
        let descriptor = crate::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&serde_json::json!({
                "model_type": "inkling_mm_model", "image_token_id": 5,
                "text_config": {"hidden_size": 8, "num_hidden_layers": 1, "vocab_size": 16,
                    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 2,
                    "layer_types": ["full_attention"], "mlp_layer_types": ["dense"],
                    "sconv_kernel_size": 3, "d_rel": 2, "rel_extent": 8, "intermediate_size": 12,
                "n_routed_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 1,
                "moe_intermediate_size": 6, "dense_intermediate_size": 12}
            }))
            .unwrap()
            .architecture_plan()
            .architecture_descriptor();
        let attention = &descriptor.components[0];
        for read in &attention.reads {
            let path = read.projection_output.as_ref().unwrap();
            let SymbolicDimension::Known(width) = descriptor
                .observations
                .get(path)
                .unwrap()
                .axes
                .as_ref()
                .unwrap()[2]
                .dimension
            else {
                panic!("read width")
            };
            let tensor = LocalTensorLayout::new(
                &read.weight,
                eredu_runtime::ParameterRole::ColumnProjection,
                vec![width, 8],
                vec![width / 2, 8],
                TensorPlacement::Shard {
                    axis: 0,
                    index: 1,
                    parts: 2,
                },
                Some(width),
                Some(width / 2..width),
                false,
            );
            let mut observations = BTreeMap::new();
            insert_read_output(&mut observations, &descriptor, read, Some(&tensor)).unwrap();
            assert_eq!(
                observations[path].coordinates().unwrap().contiguous_range(),
                Some(width / 2..width)
            );
            assert!(observations[path].exports());
            let mut absent = BTreeMap::new();
            insert_read_output(&mut absent, &descriptor, read, None).unwrap();
            assert!(absent[path].coordinates().is_none());
            assert!(!absent[path].exports());
            let partial = LocalTensorLayout::new(
                &read.weight,
                eredu_runtime::ParameterRole::RowProjection,
                vec![width, 8],
                vec![width, 4],
                TensorPlacement::Shard {
                    axis: 1,
                    index: 1,
                    parts: 2,
                },
                Some(8),
                Some(4..8),
                false,
            );
            assert!(
                insert_read_output(&mut BTreeMap::new(), &descriptor, read, Some(&partial))
                    .is_err()
            );
            let mut stale = descriptor.clone();
            stale
                .observations
                .points
                .iter_mut()
                .find(|point| &point.path == path)
                .unwrap()
                .position = eredu_core::ObservationPosition::BeforeIntervention;
            assert!(insert_read_output(&mut BTreeMap::new(), &stale, read, Some(&tensor)).is_err());
        }
    }

    fn stage(rows: std::ops::Range<usize>) -> ComponentInputProjection {
        ComponentInputProjection {
            weight: "projection.weight".into(),
            parameter_group: "projection".into(),
            shared_weight: "shared.weight".into(),
            bias: None,
            rows,
            normalization: None,
            output: "projection.latent.effective".into(),
        }
    }

    fn layout(
        global: usize,
        local: usize,
        placement: TensorPlacement,
        units: Option<usize>,
        range: Option<std::ops::Range<usize>>,
    ) -> LocalTensorLayout {
        LocalTensorLayout::new(
            "projection",
            eredu_runtime::ParameterRole::ColumnProjection,
            vec![global, 5],
            vec![local, 5],
            placement,
            units,
            range,
            false,
        )
    }

    #[test]
    fn stage_rows_intersect_replication_shards_and_permutations_without_alias_ownership() {
        let projection = stage(3..10);
        let replicated = layout(12, 12, TensorPlacement::Replicated, None, None);
        assert_eq!(
            derive_coordinates(&projection, &replicated)
                .unwrap()
                .contiguous_range(),
            Some(0..7)
        );
        for (rank, expected) in [(0, 0..3), (1, 3..7)] {
            let tensor = layout(
                12,
                6,
                TensorPlacement::Shard {
                    axis: 0,
                    index: rank,
                    parts: 2,
                },
                Some(2),
                Some(rank..rank + 1),
            );
            let map = derive_coordinates(&projection, &tensor).unwrap();
            assert_eq!(map.contiguous_range(), Some(expected));
            for local in 0..map.local_count() {
                let global = map.local_to_global(local).unwrap();
                assert_eq!(map.global_to_local(global), Some(local));
                assert!(projection.rows.contains(&(global + projection.rows.start)));
            }
        }
        let tensor = layout(
            12,
            5,
            TensorPlacement::Indices {
                axis: 0,
                indices: vec![11, 7, 3, 0, 9],
            },
            None,
            None,
        );
        let map = derive_coordinates(&projection, &tensor).unwrap();
        assert_eq!(
            (0..map.local_count())
                .map(|i| map.local_to_global(i).unwrap())
                .collect::<Vec<_>>(),
            [4, 0, 6]
        );
        assert_eq!(map.localize_indices(&[0, 1, 4, 6]).unwrap(), [1, 0, 2]);
        let empty = derive_coordinates(
            &stage(0..3),
            &layout(
                12,
                6,
                TensorPlacement::Range {
                    axis: 0,
                    start: 6,
                    end: 12,
                },
                Some(2),
                Some(1..2),
            ),
        )
        .unwrap();
        assert_eq!(empty.local_count(), 0);
    }

    #[test]
    fn latent_fp8_row_chunks_keep_the_short_tail_and_reject_partial_products() {
        let projection = stage(129..259);
        let tail = layout(
            259,
            3,
            TensorPlacement::Range {
                axis: 0,
                start: 256,
                end: 259,
            },
            Some(3),
            Some(2..3),
        )
        .with_partition_chunk_size(Some(128));
        let map = derive_coordinates(&projection, &tail).unwrap();
        assert_eq!(map.contiguous_range(), Some(127..130));
        assert_eq!(map.localize_indices(&[0, 127, 129]).unwrap(), [0, 2]);
        let wrong_units = layout(
            259,
            3,
            TensorPlacement::Range {
                axis: 0,
                start: 256,
                end: 259,
            },
            Some(3),
            Some(1..2),
        )
        .with_partition_chunk_size(Some(128));
        assert!(derive_coordinates(&projection, &wrong_units).is_err());
        let partial = LocalTensorLayout::new(
            "projection",
            eredu_runtime::ParameterRole::ColumnProjection,
            vec![259, 10],
            vec![259, 5],
            TensorPlacement::Shard {
                axis: 1,
                index: 0,
                parts: 2,
            },
            Some(2),
            Some(0..1),
            false,
        );
        assert!(derive_coordinates(&projection, &partial).is_err());
        assert!(derive_coordinates(&stage(0..260), &tail).is_err());
        assert!(derive_coordinates(&stage(3..3), &tail).is_err());
    }
}

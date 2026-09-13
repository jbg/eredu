use super::*;
use crate::capture::{CaptureBudget, CaptureSkipReason};

struct Budget {
    limit: CaptureUsage,
    total: CaptureUsage,
}
impl Budget {
    fn total(&self) -> CaptureUsage {
        self.total
    }
}
impl CaptureReservation for Budget {
    fn reserve(&mut self, cost: CaptureUsage) -> Result<Option<CaptureSkipReason>, CaptureError> {
        let total = self.total.checked_add(cost)?;
        if let Some(budget) = total.exceeded(self.limit) {
            return Err(CaptureError::Limit {
                budget,
                cumulative: true,
            });
        }
        self.total = total;
        Ok(None)
    }
}
fn ledger(host: u64) -> Budget {
    Budget {
        limit: CaptureUsage {
            host_bytes: host,
            ..Default::default()
        },
        total: CaptureUsage::default(),
    }
}
fn unravel(mut index: usize, shape: &[u64]) -> Vec<usize> {
    let mut coordinates = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        coordinates[axis] = index % shape[axis] as usize;
        index /= shape[axis] as usize;
    }
    coordinates
}
fn offset(coordinates: &[usize], shape: &[u64]) -> usize {
    coordinates
        .iter()
        .zip(shape)
        .fold(0, |index, (coordinate, size)| {
            index * *size as usize + coordinate
        })
}
fn indices(region: &ParameterRegion, shape: &[u64]) -> Vec<usize> {
    (0..elements(&region.shape).unwrap() as usize)
        .map(|index| {
            let mut coordinates = unravel(index, &region.shape);
            for (coordinate, start) in coordinates.iter_mut().zip(&region.starts) {
                *coordinate += *start as usize;
            }
            offset(&coordinates, shape)
        })
        .collect()
}
fn map(global: &[u64], axes: Vec<Vec<usize>>) -> ParameterCoordinateMap {
    ParameterCoordinateMap::new(
        global.to_vec(),
        axes.into_iter()
            .zip(global)
            .map(|(indices, count)| {
                ComponentCoordinateMap::indices(*count as usize, indices).unwrap()
            })
            .collect(),
    )
    .unwrap()
}
fn local_values(global: &[f32], coordinates: &ParameterCoordinateMap) -> Vec<f32> {
    (0..elements(coordinates.local_shape()).unwrap() as usize)
        .map(|index| {
            let local = unravel(index, coordinates.local_shape());
            let global_coordinates = local
                .iter()
                .zip(coordinates.axes())
                .map(|(local, map)| map.local_to_global(*local).unwrap())
                .collect::<Vec<_>>();
            global[offset(&global_coordinates, coordinates.global_shape())]
        })
        .collect()
}
fn contract(values: &[f32], source: &[u64], projection: &ParameterProjection) -> Vec<f64> {
    let selected = indices(&projection.region, source)
        .iter()
        .map(|index| values[*index])
        .collect::<Vec<_>>();
    let shape = projection.output_shape(source).unwrap();
    (0..elements(&shape).unwrap() as usize)
        .map(|index| {
            let mut coordinate = unravel(index, &shape);
            let direction = coordinate.pop().unwrap();
            coordinate.insert(projection.axis, 0);
            let width = projection.region.shape[projection.axis] as usize;
            (0..width)
                .map(|contracted| {
                    coordinate[projection.axis] = contracted;
                    f64::from(selected[offset(&coordinate, &projection.region.shape)])
                        * f64::from(projection.coefficients[direction * width + contracted])
                })
                .sum()
        })
        .collect()
}

#[test]
fn simultaneous_axis_shards_preserve_queries_updates_and_contractions() {
    let shape = vec![3, 5, 7];
    let values = (0..105)
        .map(|i| ((i * 13 % 37) as f32 - 18.) * 0.125)
        .collect::<Vec<_>>();
    let region = ParameterRegion {
        starts: vec![0, 1, 1],
        shape: vec![3, 3, 5],
    };
    let selected_indices = indices(&region, &shape);
    let expected = selected_indices
        .iter()
        .map(|i| values[*i])
        .collect::<Vec<_>>();
    let update = (0..expected.len())
        .map(|i| (i as f32 - 11.) * 0.25)
        .collect::<Vec<_>>();
    let mut budget = ledger(1 << 26);
    let mut pieces = Vec::new();
    for experts in [vec![2, 0], vec![1]] {
        for rows in [vec![3, 0], vec![4, 1, 2]] {
            for columns in [vec![6, 1, 4, 0], vec![5, 2, 3]] {
                let coordinates = map(&shape, vec![experts.clone(), rows.clone(), columns]);
                let partition = coordinates
                    .project_region(&region, 256, &mut budget)
                    .unwrap();
                let local = local_values(&values, &coordinates);
                pieces.push((coordinates, partition, local));
            }
        }
    }
    let mut received = vec![None; expected.len()];
    for (coordinates, partition, local) in &pieces {
        assert_eq!(partition.local_shape(), coordinates.local_shape());
        assert_eq!(partition.region(), &region);
        for fragment in partition.fragments() {
            for (source, destination) in indices(fragment.local(), coordinates.local_shape())
                .into_iter()
                .zip(indices(fragment.destination(), &region.shape))
            {
                assert!(
                    received[destination].replace(local[source]).is_none(),
                    "unique global coverage"
                );
            }
        }
        for additive in [false, true] {
            let action = if additive {
                ParameterUpdate::Add {
                    values: update.clone(),
                }
            } else {
                ParameterUpdate::Replace {
                    values: update.clone(),
                }
            };
            let mut actual = local.clone();
            for (index, fragment) in partition.fragments().iter().enumerate() {
                let projected = partition
                    .project_update(index, &action, &mut budget)
                    .unwrap();
                assert_eq!(matches!(projected, ParameterUpdate::Add { .. }), additive);
                for (destination, updated) in indices(fragment.local(), coordinates.local_shape())
                    .into_iter()
                    .zip(projected.values())
                {
                    actual[destination] = if additive {
                        actual[destination] + updated
                    } else {
                        *updated
                    };
                }
            }
            let mut global = values.clone();
            for (index, update) in selected_indices.iter().zip(&update) {
                global[*index] = if additive {
                    global[*index] + update
                } else {
                    *update
                };
            }
            assert_eq!(
                actual,
                local_values(&global, coordinates),
                "updates preserve every unselected parameter entry"
            );
        }
    }
    assert_eq!(
        received.into_iter().collect::<Option<Vec<_>>>().unwrap(),
        expected
    );
    for axis in 0..shape.len() {
        let directions = 3;
        let width = region.shape[axis] as usize;
        let coefficients = (0..directions * width)
            .map(|i| {
                if i % 2 == 0 {
                    (i + 1) as f32 * 0.25
                } else {
                    -(i as f32) * 0.5
                }
            })
            .collect();
        let projection = ParameterProjection {
            region: region.clone(),
            axis,
            directions: directions as u64,
            coefficients,
        };
        let result_shape = projection.output_shape(&shape).unwrap();
        let mut actual = vec![0.; elements(&result_shape).unwrap() as usize];
        for (coordinates, partition, local) in &pieces {
            for (index, source) in partition.fragments().iter().enumerate() {
                let fragment = partition
                    .project_contraction(index, &projection, &mut budget)
                    .unwrap();
                assert_eq!(fragment.source_destination(), source.destination());
                let partial = contract(local, coordinates.local_shape(), fragment.projection());
                for (index, value) in indices(fragment.destination(), &result_shape)
                    .into_iter()
                    .zip(partial)
                {
                    actual[index] += value;
                }
            }
        }
        assert_eq!(
            actual,
            contract(&values, &shape, &projection),
            "all-axis partial sums agree with independent full contraction"
        );
        assert!(actual.iter().any(|value| *value != 0.));
    }
    assert!(budget.total().host_bytes > 0);
    assert_eq!(
        budget.total().retained_bytes,
        0,
        "projection touches no native parameters"
    );
}

#[test]
fn compact_ranges_empty_shards_scalars_and_coordinate_identity() {
    let mut budget = ledger(1 << 20);
    let large = usize::MAX / 64;
    let contiguous = ParameterCoordinateMap::new(
        vec![large as u64, 32],
        vec![
            ComponentCoordinateMap::range(large, 50_000..large).unwrap(),
            ComponentCoordinateMap::range(32, 0..32).unwrap(),
        ],
    )
    .unwrap();
    let partition = contiguous
        .project_region(
            &ParameterRegion {
                starts: vec![75_000, 4],
                shape: vec![2, 3],
            },
            1,
            &mut budget,
        )
        .unwrap();
    assert_eq!(partition.fragments().len(), 1);
    assert_eq!(partition.fragments()[0].local().starts, [25_000, 4]);
    assert!(
        budget.total().host_bytes < 4096,
        "metadata does not scale with full parameter elements"
    );
    let empty = map(&[4, 4], vec![vec![3, 1, 0, 2], vec![]]);
    assert!(empty
        .project_region(
            &ParameterRegion {
                starts: vec![0, 0],
                shape: vec![4, 4]
            },
            1,
            &mut budget
        )
        .unwrap()
        .fragments()
        .is_empty());
    let nonoverlap = map(&[4, 4], vec![vec![3, 1, 0, 2], vec![3]]);
    assert!(nonoverlap
        .project_region(
            &ParameterRegion {
                starts: vec![0, 0],
                shape: vec![4, 2]
            },
            1,
            &mut budget
        )
        .unwrap()
        .fragments()
        .is_empty());
    let scalar = ParameterCoordinateMap::new(vec![], vec![]).unwrap();
    let scalar = scalar
        .project_region(
            &ParameterRegion {
                starts: vec![],
                shape: vec![],
            },
            1,
            &mut budget,
        )
        .unwrap();
    assert_eq!(scalar.fragments().len(), 1);
    assert_eq!(
        scalar
            .project_update(0, &ParameterUpdate::Add { values: vec![2.] }, &mut budget)
            .unwrap()
            .values(),
        [2.]
    );
    let range = ParameterCoordinateMap::new(
        vec![4],
        vec![ComponentCoordinateMap::range(4, 0..4).unwrap()],
    )
    .unwrap();
    assert_eq!(
        range.identity(),
        map(&[4], vec![vec![0, 1, 2, 3]]).identity()
    );
    assert_ne!(
        range.identity(),
        map(&[4], vec![vec![0, 2, 1, 3]]).identity()
    );
}

#[test]
fn invalid_geometry_and_insufficient_projection_credit_prevent_payload_copies() {
    assert!(ParameterCoordinateMap::new(
        vec![4],
        vec![ComponentCoordinateMap::range(3, 0..3).unwrap()]
    )
    .is_err());
    assert!(ParameterCoordinateMap::new(
        vec![u64::MAX, 2],
        vec![
            ComponentCoordinateMap::range(usize::MAX, 0..1).unwrap(),
            ComponentCoordinateMap::range(2, 0..2).unwrap(),
        ]
    )
    .is_err());
    let coordinates = map(&[4, 4], vec![vec![3, 1, 0, 2], vec![1, 0, 3, 2]]);
    let region = ParameterRegion {
        starts: vec![0, 0],
        shape: vec![4, 4],
    };
    let mut zero = ledger(0);
    assert!(matches!(
        coordinates.project_region(&region, 64, &mut zero),
        Err(ParameterError::Budget(CaptureError::Limit {
            budget: CaptureBudget::Host,
            ..
        }))
    ));
    assert_eq!(zero.total(), CaptureUsage::default());
    let mut budget = ledger(1 << 20);
    assert!(coordinates.project_region(&region, 1, &mut budget).is_err());
    let failed_charge = budget.total();
    assert!(
        failed_charge.host_bytes > 0,
        "failed preparation does not refund prior metadata"
    );
    let partition = coordinates
        .project_region(&region, 64, &mut budget)
        .unwrap();
    let action = ParameterUpdate::Replace {
        values: (0..16).map(|i| i as f32).collect(),
    };
    assert!(partition.project_update(0, &action, &mut zero).is_err());
    assert_eq!(zero.total(), CaptureUsage::default());
    let used = budget.total();
    assert!(partition
        .project_update(usize::MAX, &action, &mut budget)
        .is_err());
    assert!(partition
        .project_update(
            0,
            &ParameterUpdate::Replace {
                values: vec![f32::NAN; 16]
            },
            &mut budget
        )
        .is_err());
    assert!(partition
        .project_update(
            0,
            &ParameterUpdate::Add {
                values: vec![1.; 15]
            },
            &mut budget
        )
        .is_err());
    assert_eq!(budget.total(), used);
    let wrong = ParameterProjection {
        region: ParameterRegion {
            starts: vec![0, 0],
            shape: vec![1, 4],
        },
        axis: 1,
        directions: 1,
        coefficients: vec![1.; 4],
    };
    assert!(partition
        .project_contraction(0, &wrong, &mut budget)
        .is_err());
    assert_eq!(budget.total(), used);
    drop(partition);
    assert_eq!(
        budget.total(),
        used,
        "dropping geometry cannot refund live cumulative work"
    );
}

use super::*;
use eredu_core::component::ComponentCoordinateMap;

struct Budget {
    total: CaptureUsage,
    limit: u64,
}
impl CaptureReservation for Budget {
    fn reserve(&mut self, cost: CaptureUsage) -> Result<Option<CaptureSkipReason>, CaptureError> {
        let next = self.total.checked_add(cost)?;
        if next.host_bytes > self.limit {
            return Err(CaptureError::Limit {
                budget: CaptureBudget::Host,
                cumulative: true,
            });
        }
        self.total = next;
        Ok(None)
    }
}
fn budget() -> Budget {
    Budget {
        total: CaptureUsage::default(),
        limit: 64 << 20,
    }
}
fn map(global: &[u64], axes: Vec<Vec<usize>>) -> ParameterCoordinateMap {
    ParameterCoordinateMap::new(
        global.to_vec(),
        axes.into_iter()
            .zip(global)
            .map(|(axis, size)| ComponentCoordinateMap::indices(*size as usize, axis).unwrap())
            .collect(),
    )
    .unwrap()
}
fn coordinates(mut index: usize, shape: &[u64]) -> Vec<usize> {
    let mut result = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        result[axis] = index % shape[axis] as usize;
        index /= shape[axis] as usize;
    }
    result
}
fn offset(point: &[usize], shape: &[u64]) -> usize {
    point
        .iter()
        .zip(shape)
        .fold(0, |offset, (point, size)| offset * *size as usize + point)
}
fn source(global: &[u64]) -> Vec<f32> {
    (0..elements(global).unwrap())
        .map(|i| ((i % 11) as f32 - 5.0) * 0.37 + (i as f32 * 0.03))
        .collect()
}
fn selected(values: &[f32], shape: &[u64], region: &ParameterRegion) -> Vec<f32> {
    (0..elements(&region.shape).unwrap() as usize)
        .map(|i| {
            let point = coordinates(i, &region.shape)
                .iter()
                .zip(&region.starts)
                .map(|(coordinate, start)| coordinate + *start as usize)
                .collect::<Vec<_>>();
            values[offset(&point, shape)]
        })
        .collect()
}
fn local(values: &[f32], map: &ParameterCoordinateMap) -> Vec<f32> {
    (0..elements(map.local_shape()).unwrap() as usize)
        .map(|i| {
            let point = coordinates(i, map.local_shape())
                .iter()
                .zip(map.axes())
                .map(|(coordinate, axis)| axis.local_to_global(*coordinate).unwrap())
                .collect::<Vec<_>>();
            values[offset(&point, map.global_shape())]
        })
        .collect()
}
fn contract(values: &[f32], shape: &[u64], projection: &ParameterProjection) -> Vec<f32> {
    let chosen = selected(values, shape, &projection.region);
    let output = projection.output_shape(shape).unwrap();
    (0..elements(&output).unwrap() as usize)
        .map(|i| {
            let mut point = coordinates(i, &output);
            let direction = point.pop().unwrap();
            point.insert(projection.axis, 0);
            (0..projection.region.shape[projection.axis] as usize)
                .map(|k| {
                    point[projection.axis] = k;
                    f64::from(chosen[offset(&point, &projection.region.shape)])
                        * f64::from(
                            projection.coefficients
                                [direction * projection.region.shape[projection.axis] as usize + k],
                        )
                })
                .sum::<f64>() as f32
        })
        .collect()
}

#[test]
fn partial_replicas_permuted_axes_and_empty_owners_assemble_queries_and_all_contractions() {
    let global = [4, 5, 6];
    let values = source(&global);
    let mut maps = vec![
        None,
        Some(map(&global, vec![vec![2, 1], vec![2, 1, 3], vec![2, 3]])),
    ];
    for a in [vec![0, 1], vec![3, 2]] {
        for b in [vec![4, 2, 0], vec![3, 1]] {
            for c in [vec![0, 2, 4], vec![5, 3, 1]] {
                maps.push(Some(map(&global, vec![a.clone(), b.clone(), c])));
            }
        }
    }
    maps.push(Some(map(
        &global,
        vec![(0..4).collect(), (0..5).collect(), (0..6).collect()],
    )));
    maps.push(Some(map(
        &global,
        vec![vec![], (0..5).collect(), (0..6).collect()],
    )));
    let refs = maps.iter().map(Option::as_ref).collect::<Vec<_>>();
    let region = ParameterRegion {
        starts: vec![0, 1, 1],
        shape: vec![4, 4, 5],
    };
    let mut ledger = budget();
    for axis in [None, Some(0), Some(1), Some(2)] {
        let projection = axis.map(|axis| ParameterProjection {
            region: region.clone(),
            axis,
            directions: 3,
            coefficients: (0..3 * region.shape[axis])
                .map(|i| ((i % 5) as f32 - 2.0) * 0.71)
                .collect(),
        });
        let plan = PartitionParameterReadPlan::new(
            &global,
            &region,
            projection.as_ref(),
            &refs,
            1024,
            &mut ledger,
        )
        .unwrap();
        let mut payloads = vec![Vec::new(); refs.len()];
        for fragment in plan.fragments() {
            let map = refs[fragment.rank()].unwrap();
            let local = local(&values, map);
            payloads[fragment.rank()].extend(match &projection {
                Some(projection) => {
                    let piece = fragment
                        .source()
                        .project_contraction(projection, &global, &mut ledger)
                        .unwrap();
                    assert_eq!(piece.destination(), fragment.destination());
                    contract(&local, map.local_shape(), piece.projection())
                }
                None => selected(&local, map.local_shape(), fragment.source().local()),
            });
        }
        assert!(payloads[0].is_empty());
        assert!(
            payloads[10].is_empty(),
            "later complete replica contributes nothing twice"
        );
        assert!(payloads[11].is_empty());
        assert!(plan.fragments().iter().any(|fragment| fragment.rank() == 1));
        let actual = plan.assemble(&payloads, &mut ledger).unwrap();
        let expected = match &projection {
            Some(projection) => contract(&values, &global, projection),
            None => selected(&values, &global, &region),
        };
        for (a, b) in actual.iter().zip(&expected) {
            assert!((a - b).abs() < 3e-5, "{a} vs {b}");
        }
        let before = ledger.total;
        payloads[1].pop();
        assert!(matches!(
            plan.assemble(&payloads, &mut ledger),
            Err(ParameterError::Incomplete(_))
        ));
        assert_eq!(
            ledger.total, before,
            "bad payload rejected before assembly allocation"
        );
    }
}

#[test]
fn missing_coverage_limits_changed_directions_and_nonfinite_values_are_rejected() {
    let global = [3, 5];
    let region = ParameterRegion {
        starts: vec![0, 0],
        shape: global.to_vec(),
    };
    let partial = map(&global, vec![vec![0], vec![0, 1]]);
    let full = map(&global, vec![(0..3).collect(), (0..5).collect()]);
    let mut ledger = budget();
    assert!(matches!(
        PartitionParameterReadPlan::new(
            &global,
            &region,
            None,
            &[Some(&partial)],
            128,
            &mut ledger
        ),
        Err(ParameterError::Incomplete(_))
    ));
    assert!(ledger.total.host_bytes > 0);
    assert!(PartitionParameterReadPlan::new(
        &global,
        &region,
        None,
        &[Some(&partial), Some(&full)],
        1,
        &mut ledger
    )
    .is_err());
    let mut denied = Budget {
        total: CaptureUsage::default(),
        limit: 0,
    };
    assert!(PartitionParameterReadPlan::new(
        &global,
        &region,
        None,
        &[Some(&full)],
        128,
        &mut denied
    )
    .is_err());
    let query =
        PartitionParameterReadPlan::new(&global, &region, None, &[Some(&full)], 128, &mut ledger)
            .unwrap();
    let mut data = source(&global);
    data[2] = f32::NAN;
    assert!(query.assemble(&[data], &mut ledger).is_err());
    let mut projection = ParameterProjection {
        region,
        axis: 1,
        directions: 1,
        coefficients: vec![1.0; 5],
    };
    let left = PartitionParameterReadPlan::new(
        &global,
        &projection.region,
        Some(&projection),
        &[Some(&full)],
        128,
        &mut ledger,
    )
    .unwrap();
    projection.coefficients[2] = -1.0;
    let right = PartitionParameterReadPlan::new(
        &global,
        &projection.region,
        Some(&projection),
        &[Some(&full)],
        128,
        &mut ledger,
    )
    .unwrap();
    assert_ne!(left.identity(), right.identity());
}

#[test]
fn huge_compact_shapes_and_scalar_queries_need_only_selected_storage() {
    let global = [usize::MAX as u64 / 128, 64];
    let map = ParameterCoordinateMap::new(
        global.to_vec(),
        global
            .iter()
            .map(|size| ComponentCoordinateMap::range(*size as usize, 0..*size as usize).unwrap())
            .collect(),
    )
    .unwrap();
    let region = ParameterRegion {
        starts: vec![global[0] - 1, 61],
        shape: vec![1, 3],
    };
    let mut ledger = budget();
    let plan =
        PartitionParameterReadPlan::new(&global, &region, None, &[Some(&map)], 1, &mut ledger)
            .unwrap();
    assert_eq!(plan.rank_counts(), [3]);
    assert_eq!(
        plan.assemble(&[vec![1.0, -2.0, 3.0]], &mut ledger).unwrap(),
        [1.0, -2.0, 3.0]
    );
    assert!(ledger.total.host_bytes < 8192);
    let scalar = ParameterCoordinateMap::new(vec![], vec![]).unwrap();
    let plan = PartitionParameterReadPlan::new(
        &[],
        &ParameterRegion {
            starts: vec![],
            shape: vec![],
        },
        None,
        &[None, Some(&scalar)],
        1,
        &mut ledger,
    )
    .unwrap();
    assert_eq!(
        plan.assemble(&[vec![], vec![-3.25]], &mut ledger).unwrap(),
        [-3.25]
    );
}

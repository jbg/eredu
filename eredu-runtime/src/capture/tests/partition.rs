mod receipts;
mod windows;
use super::*;
use crate::capture::partition::*;
use eredu_core::component::ComponentCoordinateMap;

#[derive(Clone)]
struct Value {
    shape: Vec<u64>,
    data: TensorObservationData,
}

#[derive(Default)]
struct Backend {
    transforms: usize,
    exported: usize,
    fail: bool,
    corrupt_reduction: bool,
}

fn selected_indices(shape: &[u64], slice: &ResolvedCaptureSlice) -> Vec<usize> {
    (0..elements(&slice.shape).unwrap())
        .map(|index| {
            let mut remainder = index;
            let mut source = 0;
            let mut stride = 1;
            for axis in (0..shape.len()).rev() {
                let local = remainder % slice.shape[axis];
                remainder /= slice.shape[axis];
                source += (slice.starts[axis] + local * slice.strides[axis]) * stride;
                stride *= shape[axis];
            }
            source as usize
        })
        .collect()
}

impl CaptureBackend for Backend {
    type Tensor = Value;
    type Error = std::io::Error;
    fn shape(&self, value: &Value) -> Result<Vec<u64>, Self::Error> {
        Ok(value.shape.clone())
    }
    fn source_dtype(&self, value: &Value) -> Option<TensorDtype> {
        Some(match value.data {
            TensorObservationData::F32(_) => TensorDtype::F32,
            TensorObservationData::I64(_) => TensorDtype::I64,
            TensorObservationData::U64(_) => TensorDtype::U64,
            TensorObservationData::Bool(_) => TensorDtype::Bool,
        })
    }
    fn estimate(
        &self,
        _: &Value,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        let count = elements(&slice.shape)?;
        let exported = match &selection.transform {
            CaptureTransform::Summary => 11,
            CaptureTransform::Histogram { edges } => edges.len() as u64 + 2,
            CaptureTransform::Preview { max_elements } => count.min(*max_elements),
            _ => count,
        };
        Ok(CaptureUsage {
            captures: 1,
            retained_bytes: mul(count, 8)?,
            host_bytes: mul(exported, 8)?,
            encoded_bytes: add(1024, mul(exported, 32)?)?,
        })
    }
    fn transform(
        &mut self,
        value: &Value,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Self::Error> {
        self.transforms += 1;
        if self.fail {
            return Err(std::io::Error::other("injected fragment failure"));
        }
        let mut indices = selected_indices(&value.shape, slice);
        if let CaptureTransform::Preview { max_elements } = selection.transform {
            indices.truncate(max_elements as usize);
        }
        if matches!(
            selection.transform,
            CaptureTransform::Summary | CaptureTransform::Histogram { .. }
        ) {
            let values = indices
                .iter()
                .map(|index| match &value.data {
                    TensorObservationData::F32(values) => values[*index],
                    TensorObservationData::I64(values) => values[*index] as f32,
                    TensorObservationData::U64(values) => values[*index] as f32,
                    TensorObservationData::Bool(values) => u8::from(values[*index]) as f32,
                })
                .collect::<Vec<_>>();
            let mut payload = reference_reduction(&values, &selection.transform);
            match &mut payload {
                CapturePayload::Summary(value) => {
                    self.exported += 11;
                    if self.corrupt_reduction {
                        value.finite += 1;
                    }
                }
                CapturePayload::Histogram(value) => {
                    self.exported += value.counts.len() + 3;
                    if self.corrupt_reduction {
                        value.counts[0] += 1;
                    }
                }
                _ => unreachable!(),
            }
            return Ok(payload);
        }
        self.exported += indices.len();
        let data = match &value.data {
            TensorObservationData::F32(values) => {
                TensorObservationData::F32(indices.iter().map(|i| values[*i]).collect())
            }
            TensorObservationData::I64(values) => {
                TensorObservationData::I64(indices.iter().map(|i| values[*i]).collect())
            }
            TensorObservationData::U64(values) => {
                TensorObservationData::U64(indices.iter().map(|i| values[*i]).collect())
            }
            TensorObservationData::Bool(values) => {
                TensorObservationData::Bool(indices.iter().map(|i| values[*i]).collect())
            }
        };
        Ok(CapturePayload::Tensor(
            TensorObservation::new(
                if matches!(selection.transform, CaptureTransform::Preview { .. }) {
                    vec![indices.len()]
                } else {
                    slice.shape.iter().map(|n| *n as usize).collect()
                },
                data,
            )
            .unwrap(),
        ))
    }
}

fn plan(host_limit: u64) -> AdmittedCapturePlan {
    transformed_plan(host_limit, CaptureTransform::Slice)
}

fn transformed_plan(host_limit: u64, transform: CaptureTransform) -> AdmittedCapturePlan {
    let (mut plan, mut catalog, support, capabilities) = fixture(transform);
    catalog.points[0].axes.as_mut().unwrap()[1].dimension = SymbolicDimension::Known(20);
    plan.selections[0].slices = vec![
        CaptureSlice {
            axis: "sequence".into(),
            start: 0,
            end: 3,
            stride: 2,
        },
        CaptureSlice {
            axis: "hidden".into(),
            start: 1,
            end: 20,
            stride: 3,
        },
    ];
    plan.limits.per_step.host_bytes = host_limit;
    plan.selections[0].schedule.decode = false;
    admit(plan, &catalog, &support, &capabilities).unwrap()
}

fn local_value(global: &Value, map: &ComponentCoordinateMap) -> Value {
    let indices = (0..global.shape[0] as usize)
        .flat_map(|row| {
            (0..map.local_count()).map(move |i| row * 20 + map.local_to_global(i).unwrap())
        })
        .collect::<Vec<_>>();
    let data = match &global.data {
        TensorObservationData::F32(v) => {
            TensorObservationData::F32(indices.iter().map(|i| v[*i]).collect())
        }
        TensorObservationData::I64(v) => {
            TensorObservationData::I64(indices.iter().map(|i| v[*i]).collect())
        }
        TensorObservationData::U64(v) => {
            TensorObservationData::U64(indices.iter().map(|i| v[*i]).collect())
        }
        TensorObservationData::Bool(v) => {
            TensorObservationData::Bool(indices.iter().map(|i| v[*i]).collect())
        }
    };
    Value {
        shape: vec![global.shape[0], map.local_count() as u64],
        data,
    }
}

fn capture_maps(
    plan: &AdmittedCapturePlan,
    global: &Value,
    maps: &[ComponentCoordinateMap],
    backend: &mut Backend,
    ledger: &mut CaptureLedger,
) -> Vec<CapturedPartitionFragment> {
    let slice =
        resolve_slice(&plan.points()[0], &plan.plan().selections[0], &global.shape).unwrap();
    let mut fragments = Vec::new();
    for (producer_rank, map) in maps.iter().enumerate() {
        let projection = CaptureSlicePartition::new(&global.shape, &slice, 1, map, 16).unwrap();
        let local = local_value(global, map);
        for fragment_index in 0..projection.fragments().len() {
            fragments.push(
                capture_fragment(
                    backend,
                    &local,
                    PartitionCaptureRequest {
                        invocation: None,
                        plan,
                        selection_index: 0,
                        phase: CapturePhase::Prefill,
                        prediction: 0,
                        projection: &projection,
                        fragment_index,
                        producer_rank,
                    },
                    ledger,
                )
                .unwrap(),
            );
        }
    }
    fragments
}

#[test]
fn partition_fragments_reconstruct_global_strided_values_and_exact_integer_ids() {
    for data in [
        TensorObservationData::F32((0..60).map(|i| (i as f32 * 0.17).sin()).collect()),
        TensorObservationData::I64((0..60).map(|i| i64::MIN + i).collect()),
        TensorObservationData::U64((0..60).map(|i| u64::MAX - i).collect()),
        TensorObservationData::Bool((0..60).map(|i| i % 3 == 0).collect()),
    ] {
        let plan = plan(1_000_000);
        let global = Value {
            shape: vec![3, 20],
            data,
        };
        let mut ordinary = CaptureSession::new(plan.clone());
        ordinary.begin_step(CapturePhase::Prefill, 0).unwrap();
        ordinary
            .observe(&mut Backend::default(), "block.output", &global)
            .unwrap();
        let expected = ordinary.take_step().unwrap().records.remove(0);
        for maps in [
            vec![
                ComponentCoordinateMap::range(20, 0..8).unwrap(),
                ComponentCoordinateMap::range(20, 8..20).unwrap(),
            ],
            vec![ComponentCoordinateMap::indices(20, vec![19, 2, 16, 7, 0, 4, 1, 13, 10]).unwrap()],
        ] {
            let mut backend = Backend::default();
            let mut ledger = CaptureLedger::new(&plan);
            let mut fragments = capture_maps(&plan, &global, &maps, &mut backend, &mut ledger);
            assert_eq!(
                backend.exported, 14,
                "only selected values may reach host buffers"
            );
            assert!(fragments
                .iter()
                .all(|fragment| fragment.plan_identity() == plan.identity()));
            fragments.reverse(); // Arrival order does not define global component order.
            let merged = assemble_tensor_fragments(
                &plan,
                0,
                CapturePhase::Prefill,
                0,
                &global.shape,
                fragments,
                &mut ledger,
            )
            .unwrap();
            assert_eq!(merged.plan_identity(), plan.identity());
            assert!(!merged.contributions().is_empty());
            assert_eq!(merged.record().payload, expected.payload);
            assert_eq!(merged.record().source_dtype, expected.source_dtype);
            assert_eq!(merged.record().source_shape, expected.source_shape);
            assert_eq!(merged.record().selected_shape, expected.selected_shape);
            assert_eq!(merged.record().charged, ledger.total());
        }
    }
}

#[test]
fn partition_fragment_assembly_rejects_missing_duplicate_and_stale_results() {
    let plan = plan(1_000_000);
    let global = Value {
        shape: vec![3, 20],
        data: TensorObservationData::F32((0..60).map(|i| i as f32 + 1.0).collect()),
    };
    let maps = [
        ComponentCoordinateMap::range(20, 0..8).unwrap(),
        ComponentCoordinateMap::range(20, 8..20).unwrap(),
    ];
    let mut ledger = CaptureLedger::new(&plan);
    let fragments = capture_maps(&plan, &global, &maps, &mut Backend::default(), &mut ledger);
    let incomplete = assemble_tensor_fragments(
        &plan,
        0,
        CapturePhase::Prefill,
        0,
        &global.shape,
        vec![fragments[1].clone()],
        &mut ledger,
    );
    assert!(matches!(
        incomplete,
        Err(PartitionCaptureMergeError::Incomplete {
            expected_elements: 14,
            received_elements: 8
        })
    ));
    let duplicate = assemble_tensor_fragments(
        &plan,
        0,
        CapturePhase::Prefill,
        0,
        &global.shape,
        vec![
            fragments[0].clone(),
            fragments[0].clone(),
            fragments[1].clone(),
        ],
        &mut ledger,
    );
    assert!(matches!(
        duplicate,
        Err(PartitionCaptureMergeError::Overlap)
    ));
    assert!(assemble_tensor_fragments(
        &plan,
        0,
        CapturePhase::Prefill,
        1,
        &global.shape,
        fragments,
        &mut ledger
    )
    .is_err());
}

#[test]
fn partition_fragment_limits_and_native_failures_preserve_reservations() {
    let global = Value {
        shape: vec![3, 20],
        data: TensorObservationData::F32(vec![1.0; 60]),
    };
    let map = ComponentCoordinateMap::range(20, 0..8).unwrap();
    let local = local_value(&global, &map);
    for (limit, fail, copies) in [(1, false, 0), (1_000_000, true, 1)] {
        let plan = plan(limit);
        let slice =
            resolve_slice(&plan.points()[0], &plan.plan().selections[0], &global.shape).unwrap();
        let projection = CaptureSlicePartition::new(&global.shape, &slice, 1, &map, 1).unwrap();
        let mut ledger = CaptureLedger::new(&plan);
        let mut backend = Backend {
            fail,
            ..Default::default()
        };
        assert!(capture_fragment(
            &mut backend,
            &local,
            PartitionCaptureRequest {
                invocation: None,
                plan: &plan,
                selection_index: 0,
                phase: CapturePhase::Prefill,
                prediction: 0,
                projection: &projection,
                fragment_index: 0,
                producer_rank: 0
            },
            &mut ledger
        )
        .is_err());
        assert_eq!(backend.transforms, copies);
        if fail {
            assert!(ledger.total().host_bytes > 0);
            assert_eq!(ledger.total().captures, 1);
        }
    }
    // Measure the exact fragment cost, then allow that work but reject assembly.
    let generous = plan(1_000_000);
    let maps = [
        ComponentCoordinateMap::range(20, 0..8).unwrap(),
        ComponentCoordinateMap::range(20, 8..20).unwrap(),
    ];
    let mut measured = CaptureLedger::new(&generous);
    capture_maps(
        &generous,
        &global,
        &maps,
        &mut Backend::default(),
        &mut measured,
    );
    let limited = plan(measured.total().host_bytes);
    let mut ledger = CaptureLedger::new(&limited);
    let mut backend = Backend::default();
    let fragments = capture_maps(&limited, &global, &maps, &mut backend, &mut ledger);
    let consumed = ledger.total();
    assert!(matches!(
        assemble_tensor_fragments(
            &limited,
            0,
            CapturePhase::Prefill,
            0,
            &global.shape,
            fragments,
            &mut ledger
        ),
        Err(PartitionCaptureMergeError::Capture(CaptureError::Limit {
            budget: CaptureBudget::Host,
            ..
        }))
    ));
    assert_eq!(
        ledger.total(),
        consumed,
        "failed assembly cannot refund earlier exports"
    );
    assert_eq!(backend.transforms, 2);
}

#[test]
fn partition_capture_rejects_wrong_geometry_before_native_work() {
    let plan = plan(1_000_000);
    let slice = resolve_slice(&plan.points()[0], &plan.plan().selections[0], &[3, 20]).unwrap();
    let map = ComponentCoordinateMap::range(20, 0..8).unwrap();
    let projection = CaptureSlicePartition::new(&[3, 20], &slice, 1, &map, 1).unwrap();
    let mut backend = Backend::default();
    let mut ledger = CaptureLedger::new(&plan);
    let wrong = Value {
        shape: vec![3, 9],
        data: TensorObservationData::F32(vec![1.0; 27]),
    };
    assert!(capture_fragment(
        &mut backend,
        &wrong,
        PartitionCaptureRequest {
            invocation: None,
            plan: &plan,
            selection_index: 0,
            phase: CapturePhase::Prefill,
            prediction: 0,
            projection: &projection,
            fragment_index: 0,
            producer_rank: 0
        },
        &mut ledger
    )
    .is_err());
    assert_eq!(backend.transforms, 0);
    assert_eq!(ledger.total(), CaptureUsage::default());
}

fn reference_reduction(values: &[f32], transform: &CaptureTransform) -> CapturePayload {
    match transform {
        CaptureTransform::Summary => {
            let finite = values
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .map(f64::from)
                .collect::<Vec<_>>();
            CapturePayload::Summary(CaptureSummary {
                elements: values.len() as u64,
                finite: finite.len() as u64,
                non_finite: (values.len() - finite.len()) as u64,
                nan: values.iter().filter(|v| v.is_nan()).count() as u64,
                positive_infinity: values.iter().filter(|v| **v == f32::INFINITY).count() as u64,
                negative_infinity: values.iter().filter(|v| **v == f32::NEG_INFINITY).count()
                    as u64,
                min: finite.iter().copied().reduce(f64::min),
                max: finite.iter().copied().reduce(f64::max),
                mean: (!finite.is_empty())
                    .then(|| finite.iter().sum::<f64>() / finite.len() as f64),
                rms: (!finite.is_empty()).then(|| {
                    (finite.iter().map(|v| v * v).sum::<f64>() / finite.len() as f64).sqrt()
                }),
            })
        }
        CaptureTransform::Histogram { edges } => CapturePayload::Histogram(CaptureHistogram {
            edges: edges.clone(),
            counts: edges
                .windows(2)
                .enumerate()
                .map(|(i, edge)| {
                    values
                        .iter()
                        .filter(|value| {
                            value.is_finite()
                                && **value >= edge[0]
                                && (**value < edge[1]
                                    || (i + 2 == edges.len() && **value == edge[1]))
                        })
                        .count() as u64
                })
                .collect(),
            below: values
                .iter()
                .filter(|v| v.is_finite() && **v < edges[0])
                .count() as u64,
            above: values
                .iter()
                .filter(|v| v.is_finite() && **v > *edges.last().unwrap())
                .count() as u64,
            non_finite: values.iter().filter(|v| !v.is_finite()).count() as u64,
        }),
        _ => unreachable!(),
    }
}

fn assert_reduced_close(actual: &CapturePayload, expected: &CapturePayload) {
    match (actual, expected) {
        (CapturePayload::Summary(actual), CapturePayload::Summary(expected)) => {
            let mut counts = actual.clone();
            counts.mean = expected.mean;
            counts.rms = expected.rms;
            assert_eq!(&counts, expected);
            for (a, b) in [(actual.mean, expected.mean), (actual.rms, expected.rms)] {
                match (a, b) {
                    (Some(a), Some(b)) => assert!(
                        (a - b).abs() <= 2e-15 * expected.rms.unwrap().max(1.0),
                        "{a} != {b}"
                    ),
                    (None, None) => (),
                    _ => panic!("finite statistics differ"),
                }
            }
        }
        _ => assert_eq!(actual, expected),
    }
}

#[test]
fn partition_reductions_match_global_statistics_and_reject_invalid_counts() {
    for transform in [
        CaptureTransform::Summary,
        CaptureTransform::Histogram {
            edges: vec![-2.0, 0.0, 1.0, 4.0],
        },
    ] {
        for pattern in [
            vec![
                -1e6,
                0.125,
                1e6,
                -0.25,
                1.0,
                -2.0,
                4.0,
                7.0,
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
            ],
            vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            vec![-f32::MAX, 0.0, f32::MAX, 0.0],
        ] {
            let plan = transformed_plan(1_000_000, transform.clone());
            let data = (0..60)
                .map(|i| pattern[i % pattern.len()])
                .collect::<Vec<_>>();
            let global = Value {
                shape: vec![3, 20],
                data: TensorObservationData::F32(data.clone()),
            };
            let slice = resolve_slice(&plan.points()[0], &plan.plan().selections[0], &global.shape)
                .unwrap();
            let expected = reference_reduction(
                &selected_indices(&global.shape, &slice)
                    .into_iter()
                    .map(|i| data[i])
                    .collect::<Vec<_>>(),
                &transform,
            );
            for maps in [
                vec![
                    ComponentCoordinateMap::range(20, 0..8).unwrap(),
                    ComponentCoordinateMap::range(20, 8..20).unwrap(),
                ],
                vec![
                    ComponentCoordinateMap::indices(20, vec![19, 2, 16, 7, 0, 4, 1, 13, 10])
                        .unwrap(),
                ],
                // Disjoint arithmetic progressions with overlapping bounding boxes.
                vec![
                    ComponentCoordinateMap::indices(20, vec![1, 7, 13, 19]).unwrap(),
                    ComponentCoordinateMap::indices(20, vec![4, 10, 16]).unwrap(),
                ],
            ] {
                let mut ledger = CaptureLedger::new(&plan);
                let mut backend = Backend::default();
                let mut fragments = capture_maps(&plan, &global, &maps, &mut backend, &mut ledger);
                let fragments_count = fragments.len();
                fragments.reverse();
                let merged = assemble_reduced_fragments(
                    &plan,
                    0,
                    CapturePhase::Prefill,
                    0,
                    &global.shape,
                    fragments,
                    &mut ledger,
                )
                .unwrap();
                assert_reduced_close(merged.record().payload.as_ref().unwrap(), &expected);
                assert_eq!(merged.contributions().len(), fragments_count);
                assert_eq!(merged.record().charged, ledger.total());
                assert_eq!(
                    backend.exported,
                    fragments_count
                        * if matches!(transform, CaptureTransform::Summary) {
                            11
                        } else {
                            6
                        }
                );
                let fragments = capture_maps(
                    &plan,
                    &global,
                    &maps,
                    &mut Backend {
                        corrupt_reduction: true,
                        ..Default::default()
                    },
                    &mut CaptureLedger::new(&plan),
                );
                assert!(assemble_reduced_fragments(
                    &plan,
                    0,
                    CapturePhase::Prefill,
                    0,
                    &global.shape,
                    fragments,
                    &mut CaptureLedger::new(&plan)
                )
                .is_err());
            }
        }
    }
}

#[test]
fn generated_partition_fragments_reserve_before_factory_and_preserve_precision_and_errors() {
    let map = ComponentCoordinateMap::range(20, 0..8).unwrap();
    let prototype = Value {
        shape: vec![3, 8],
        data: TensorObservationData::I64(vec![0; 24]),
    };
    for (creation, fail, wrong_shape) in [
        (200, false, false),
        (u64::MAX, false, false),
        (200, true, false),
        (200, false, true),
    ] {
        let plan = plan(1_000_000);
        let slice = resolve_slice(&plan.points()[0], &plan.plan().selections[0], &[3, 20]).unwrap();
        let projection = CaptureSlicePartition::new(&[3, 20], &slice, 1, &map, 1).unwrap();
        let mut backend = Backend::default();
        let mut ledger = CaptureLedger::new(&plan);
        let calls = Cell::new(0);
        let result = capture_generated_fragment(
            &mut backend,
            &prototype,
            &generated_source(creation),
            &mut || {
                calls.set(calls.get() + 1);
                if fail {
                    return Err(String::from("original factory failure"));
                }
                Ok(Value {
                    shape: vec![3, if wrong_shape { 9 } else { 8 }],
                    data: TensorObservationData::F32(vec![2.5; if wrong_shape { 27 } else { 24 }]),
                })
            },
            PartitionCaptureRequest {
                invocation: None,
                plan: &plan,
                selection_index: 0,
                phase: CapturePhase::Prefill,
                prediction: 0,
                projection: &projection,
                fragment_index: 0,
                producer_rank: 0,
            },
            &mut ledger,
            &|error| error.to_string(),
        );
        assert_eq!(calls.get(), usize::from(creation != u64::MAX));
        if fail {
            assert_eq!(result.unwrap_err(), "original factory failure");
        } else if wrong_shape || creation == u64::MAX {
            assert!(result.is_err());
        } else {
            let record = result.unwrap();
            assert_eq!(record.record().source_dtype, Some(TensorDtype::F32));
            assert_eq!(record.record().charged.retained_bytes, creation + 6 * 8);
        }
        assert_eq!(
            backend.transforms,
            usize::from(!fail && !wrong_shape && creation != u64::MAX)
        );
        if creation != u64::MAX {
            assert!(ledger.total().retained_bytes >= creation);
        }
    }
}

#[test]
fn reduced_fragment_assembly_storage_is_independent_of_global_component_width() {
    struct ConstantSummary;
    impl CaptureBackend for ConstantSummary {
        type Tensor = Vec<u64>;
        type Error = std::io::Error;
        fn shape(&self, shape: &Self::Tensor) -> Result<Vec<u64>, Self::Error> {
            Ok(shape.clone())
        }
        fn estimate(
            &self,
            _: &Self::Tensor,
            _: &CaptureSelection,
            _: &ResolvedCaptureSlice,
        ) -> Result<CaptureUsage, CaptureError> {
            Ok(CaptureUsage {
                captures: 1,
                retained_bytes: 16,
                host_bytes: 256,
                encoded_bytes: 1024,
            })
        }
        fn transform(
            &mut self,
            _: &Self::Tensor,
            _: &CaptureSelection,
            slice: &ResolvedCaptureSlice,
        ) -> Result<CapturePayload, Self::Error> {
            let count = elements(&slice.shape).unwrap();
            Ok(CapturePayload::Summary(CaptureSummary {
                elements: count,
                finite: count,
                non_finite: 0,
                nan: 0,
                positive_infinity: 0,
                negative_infinity: 0,
                min: Some(1.0),
                max: Some(1.0),
                mean: Some(1.0),
                rms: Some(1.0),
            }))
        }
    }
    let width = usize::try_from(1u64 << 40).unwrap_or(1 << 28);
    let (plan, mut catalog, support, capabilities) = fixture(CaptureTransform::Summary);
    catalog.points[0].axes.as_mut().unwrap()[1].dimension = SymbolicDimension::Known(width);
    let plan = admit(plan, &catalog, &support, &capabilities).unwrap();
    let global = [3, width as u64];
    let slice = resolve_slice(&plan.points()[0], &plan.plan().selections[0], &global).unwrap();
    let mut ledger = CaptureLedger::new(&plan);
    let mut fragments = Vec::new();
    for (rank, range) in [0..width / 2, width / 2..width].into_iter().enumerate() {
        let map = ComponentCoordinateMap::range(width, range).unwrap();
        let projection = CaptureSlicePartition::new(&global, &slice, 1, &map, 1).unwrap();
        fragments.push(
            capture_fragment(
                &mut ConstantSummary,
                &projection.local_shape().to_vec(),
                PartitionCaptureRequest {
                    invocation: None,
                    plan: &plan,
                    selection_index: 0,
                    phase: CapturePhase::Prefill,
                    prediction: 0,
                    projection: &projection,
                    fragment_index: 0,
                    producer_rank: rank,
                },
                &mut ledger,
            )
            .unwrap(),
        );
    }
    let capture = assemble_reduced_fragments(
        &plan,
        0,
        CapturePhase::Prefill,
        0,
        &global,
        fragments,
        &mut ledger,
    )
    .unwrap();
    let Some(CapturePayload::Summary(value)) = &capture.record().payload else {
        panic!()
    };
    assert_eq!(value.elements, 3 * width as u64);
    assert_eq!((value.mean, value.rms), (Some(1.0), Some(1.0)));
    assert!(ledger.total().host_bytes < 50_000);
    assert_eq!(capture.contributions().len(), 2);
}

use super::*;
use eredu_core::{
    component::{ComponentCoordinateMap, RoutedComponentCoordinateMap},
    *,
};
use safemlx::{Device, DeviceType};

mod provider;

fn ledger(
    geometry: RoutedUnitGeometry,
    tokens: u64,
    slice: &ResolvedCaptureSlice,
) -> CaptureLedger {
    CaptureLedger::new(&admission(geometry, tokens, slice))
}

fn admission(
    geometry: RoutedUnitGeometry,
    tokens: u64,
    slice: &ResolvedCaptureSlice,
) -> AdmittedCapturePlan {
    let point = ObservationPoint {
        path: "bank.units".into(),
        node_id: "bank".into(),
        meaning: "selected expert units".into(),
        value_type: ObservationValueType::RoutedUnits {
            routing: "bank".into(),
            geometry,
        },
        dtype: ObservationDtype::Floating,
        axes: Some(
            [
                ("token", SymbolicDimension::TokenRows),
                (
                    "route",
                    SymbolicDimension::Known(geometry.routes_per_token as usize),
                ),
                (
                    "component",
                    SymbolicDimension::Known(geometry.units_per_expert as usize),
                ),
            ]
            .into_iter()
            .map(|(name, dimension)| TensorAxis {
                name: name.into(),
                dimension,
            })
            .collect(),
        ),
        prefill: true,
        decode: false,
        requirements: vec![ObservationRequirement::ActivationHooks],
        position: ObservationPosition::BeforeIntervention,
        retained_bytes: None,
        host_bytes: None,
    };
    let catalog = ObservationCatalog {
        schema_version: 1,
        points: vec![point],
        completeness: DescriptionCompleteness::Complete,
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: Default::default(),
        points: vec![ObservationSupport {
            path: "bank.units".into(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Unsupported("fixture".into()),
            floating_to_f32: true,
        }],
    };
    let usage = CaptureUsage {
        captures: 128,
        retained_bytes: 128_000_000,
        host_bytes: 128_000_000,
        encoded_bytes: 128_000_000,
    };
    CapturePlan {
        schema_version: CAPTURE_SCHEMA_VERSION,
        selections: vec![CaptureSelection {
            id: "units".into(),
            path: "bank.units".into(),
            schedule: CaptureSchedule {
                decode: false,
                ..Default::default()
            },
            slices: ["token", "route", "component"]
                .into_iter()
                .enumerate()
                .map(|(axis, name)| CaptureSlice {
                    axis: name.into(),
                    start: slice.starts[axis],
                    end: slice.ends[axis],
                    stride: slice.strides[axis],
                })
                .collect(),
            transform: CaptureTransform::RoutedUnits,
        }],
        limits: CaptureLimits {
            per_step: usage,
            cumulative: usage,
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    }
    .admit(
        &catalog,
        &support,
        &capabilities(),
        CaptureRequestShape {
            batch: 1,
            prompt_tokens: tokens,
            max_predictions: 1,
        },
    )
    .unwrap()
}

fn values(payload: &RoutedUnitCapture) -> Vec<(Option<u64>, u64, u64, u64, Vec<f32>)> {
    payload
        .rows
        .iter()
        .map(|row| {
            let TensorObservationData::F32(values) = row.values.data() else {
                panic!("F32 capture")
            };
            (
                row.source_peer,
                row.token,
                row.slot,
                row.expert,
                values.clone(),
            )
        })
        .collect()
}

fn verify(device: DeviceType) {
    let stream = Stream::new_with_device(&Device::new(device, 0));
    let geometry = RoutedUnitGeometry {
        experts: 5,
        units_per_expert: 7,
        routes_per_token: 2,
    };
    let units = ComponentCoordinateMap::indices(7, vec![5, 0, 3]).unwrap();
    let ownership = RoutedUnitCaptureOwnership {
        coordinates: RoutedComponentCoordinateMap::new(
            ComponentCoordinateMap::indices(5, vec![4, 1]).unwrap(),
            units.clone(),
        ),
        source_peer: Some(0),
        source_peers: 3,
    };
    let slice = ResolvedCaptureSlice {
        starts: vec![0, 0, 3],
        ends: vec![3, 2, 6],
        strides: vec![2, 1, 2],
        shape: vec![2, 2, 2],
    };
    let request = PartitionRoutedUnitCaptureRequest {
        geometry,
        source_tokens: 3,
        ownership: &ownership,
        slice: &slice,
    };
    let origins = RoutedUnitOrigins::new(&[2, 0, 3], &[5, 0, 0, 5, 3], 2).unwrap();
    let groups = MlxTensor::from_array(Array::from_slice(&[1i32, 0, 1, 0, 1], &[5, 1]));
    let global_groups = [4, 1];
    let mut backend = NativeCapture {
        stream: &stream,
        domain: None,
        partition: None,
    };
    for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
        HOST_READS.with(|reads| reads.set((0, 0)));
        let mut ledger = ledger(geometry, 3, &slice);
        let usage = backend.estimate_partition_routed_units(&request).unwrap();
        assert_eq!(ledger.reserve(usage).unwrap(), None);
        assert_eq!(HOST_READS.with(|reads| reads.get()), (0, 0));
        let mut payloads = vec![];
        for (offset, sorted) in [(0, vec![1i32, 0]), (2, vec![2i32, 0, 1])] {
            let scalar =
                |original: usize, unit: usize| original as f32 * 16. + unit as f32 * 2. - 20.5;
            let data = sorted
                .iter()
                .flat_map(|&token| [5, 0, 3].map(|unit| scalar(offset + token as usize, unit)))
                .collect::<Vec<_>>();
            let value = MlxTensor::from_array(
                Array::from_slice(&data, &[sorted.len() as i32, 3])
                    .as_dtype(dtype, &stream)
                    .unwrap(),
            );
            // Keep IDs unsigned so native normalization aliases this strided
            // view; the host coordinate reader must follow its logical order.
            let padded = sorted
                .iter()
                .flat_map(|n| [99_u32, *n as u32])
                .collect::<Vec<_>>();
            let tokens = MlxTensor::from_array(
                Array::from_slice(&padded, &[sorted.len() as i32, 2])
                    .try_index_device((.., 1), &stream)
                    .unwrap(),
            );
            let coefficients = MlxTensor::from_array(Array::from_slice(
                &vec![0.5f32; sorted.len()],
                &[sorted.len() as i32, 1],
            ));
            let source = PartitionRoutedUnitCaptureSource {
                source: RoutedUnitCaptureSource {
                    values: &value,
                    token_indices: &tokens,
                    selection_indices: &tokens,
                    coefficients: &coefficients,
                    source_groups: &groups,
                    token_offset: offset as u64,
                    global_groups: Some(&global_groups),
                },
                origins: Some(origins),
                unit_coordinates: &units,
            };
            let payload = backend
                .capture_partition_routed_units(&source, &request)
                .unwrap()
                .unwrap();
            payload.validate_rows(&slice).unwrap();
            assert!(payload.rows.iter().all(|row| row.coefficient == 0.5));
            assert_eq!(
                payload.source_token_ranges,
                [[offset as u64, (offset + sorted.len()) as u64]]
            );
            if offset == 0 {
                assert_eq!(
                    values(&payload),
                    vec![
                        (Some(0), 0, 0, 4, vec![scalar(1, 3), scalar(1, 5)]),
                        (Some(0), 2, 1, 1, vec![scalar(0, 3), scalar(0, 5)])
                    ]
                );
            } else {
                assert!(
                    payload.rows.is_empty(),
                    "other source peers are not exported"
                );
            }
            payloads.push(payload);
            let wrong_units = ComponentCoordinateMap::indices(7, vec![0, 3, 5]).unwrap();
            let bad_source = PartitionRoutedUnitCaptureSource {
                unit_coordinates: &wrong_units,
                ..source
            };
            let before = HOST_READS.with(|reads| reads.get());
            assert!(backend
                .capture_partition_routed_units(&bad_source, &request)
                .unwrap()
                .is_err());
            assert_eq!(
                HOST_READS.with(|reads| reads.get()),
                before,
                "placement rejection must precede copies"
            );
        }
        assert_eq!(
            HOST_READS.with(|reads| reads.get()),
            (24, 3),
            "only bounded route metadata and four selected scalar values are copied"
        );
        let encoded: u64 = payloads
            .iter()
            .map(|payload| {
                serde_json::to_vec(&CapturePayload::RoutedUnits(payload.clone()))
                    .unwrap()
                    .len() as u64
            })
            .sum();
        assert!(encoded <= usage.encoded_bytes);
    }

    // Wrong publication topology and unowned global units reject before copying.
    let empty = MlxTensor::from_array(Array::from_slice::<f32>(&[], &[0, 3]));
    let indices = MlxTensor::from_array(Array::from_slice::<i32>(&[], &[0]));
    let coefficients = MlxTensor::from_array(Array::from_slice::<f32>(&[], &[0, 1]));
    let source = PartitionRoutedUnitCaptureSource {
        source: RoutedUnitCaptureSource {
            values: &empty,
            token_indices: &indices,
            selection_indices: &indices,
            coefficients: &coefficients,
            source_groups: &groups,
            token_offset: 0,
            global_groups: None,
        },
        origins: None,
        unit_coordinates: &units,
    };
    let before = HOST_READS.with(|reads| reads.get());
    assert!(backend
        .capture_partition_routed_units(&source, &request)
        .unwrap()
        .is_err());
    let bad_slice = ResolvedCaptureSlice {
        starts: vec![0, 0, 2],
        ends: vec![3, 2, 3],
        strides: vec![1, 1, 1],
        shape: vec![3, 2, 1],
    };
    let bad = PartitionRoutedUnitCaptureRequest {
        slice: &bad_slice,
        ..request
    };
    assert!(backend.estimate_partition_routed_units(&bad).is_err());
    assert_eq!(HOST_READS.with(|reads| reads.get()), before);
}

#[test]
#[ignore = "requires local native MLX execution"]
fn partitioned_routed_collector_preserves_origins_and_permuted_units_cpu() {
    verify(DeviceType::Cpu);
}

#[test]
#[ignore = "requires local MLX Metal execution"]
#[cfg(feature = "metal")]
fn partitioned_routed_collector_preserves_origins_and_permuted_units_metal() {
    verify(DeviceType::Gpu);
}

#[test]
fn sparse_estimator_covers_full_native_chunk_history_and_incoming_routes() {
    let slice = ResolvedCaptureSlice {
        starts: vec![999, 0, 0],
        ends: vec![1000, 2, 1],
        strides: vec![1, 1, 1],
        shape: vec![1, 2, 1],
    };
    let ordinary = estimate(&[1000, 2, 7], &slice).unwrap();
    assert!(ordinary.encoded_bytes >= 1000 * 64 + 2 * 768);
    let ownership = RoutedUnitCaptureOwnership {
        coordinates: RoutedComponentCoordinateMap::new(
            ComponentCoordinateMap::range(5, 0..5).unwrap(),
            ComponentCoordinateMap::range(7, 0..3).unwrap(),
        ),
        source_peer: Some(0),
        source_peers: 3,
    };
    let request = PartitionRoutedUnitCaptureRequest {
        geometry: RoutedUnitGeometry {
            experts: 5,
            units_per_expert: 7,
            routes_per_token: 2,
        },
        source_tokens: 1000,
        ownership: &ownership,
        slice: &slice,
    };
    assert_eq!(request.native_shape().unwrap(), [6000, 1, 3]);
    let partitioned = estimate_partition(&request).unwrap();
    assert!(partitioned.retained_bytes >= 6000 * 3 * 24 + 6000 * 192);
    assert!(partitioned.encoded_bytes >= 6000 * 64);
    let overflow = PartitionRoutedUnitCaptureRequest {
        source_tokens: u64::MAX,
        ..request
    };
    assert!(matches!(
        estimate_partition(&overflow),
        Err(CaptureError::Overflow)
    ));
}

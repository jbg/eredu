use super::*;

fn geometry(width: usize) -> Geometry {
    use eredu_core::{
        ObservationPoint, ObservationPosition, ObservationValueType, SymbolicDimension, TensorAxis,
    };
    let point = ObservationPoint {
        path: "source".into(),
        node_id: "node".into(),
        meaning: "window".into(),
        value_type: ObservationValueType::Tensor,
        dtype: ObservationDtype::Floating,
        axes: Some(vec![
            TensorAxis {
                name: "sequence".into(),
                dimension: SymbolicDimension::Sequence,
            },
            TensorAxis {
                name: "width".into(),
                dimension: SymbolicDimension::Known(width),
            },
        ]),
        prefill: true,
        decode: true,
        requirements: vec![],
        position: ObservationPosition::ReadOnly,
        retained_bytes: None,
        host_bytes: None,
    };
    let selection = CaptureSelection {
        id: "preview".into(),
        path: "source".into(),
        schedule: Default::default(),
        slices: vec![],
        transform: CaptureTransform::Preview { max_elements: 7 },
    };
    Geometry::new(&point, &selection, 1, 5).unwrap()
}

fn record() -> CaptureRecord {
    CaptureRecord {
        schema_version: CAPTURE_SCHEMA_VERSION,
        selection_id: "preview".into(),
        path: "source".into(),
        node_id: "node".into(),
        position: eredu_core::ObservationPosition::ReadOnly,
        source_shape: Some(vec![5, 4]),
        source_dtype: None,
        selected_shape: Some(vec![5, 4]),
        outcome: CaptureOutcome::Missing,
        payload: None,
        charged: CaptureUsage::default(),
    }
}
fn payload(values: TensorObservationData) -> CapturePayload {
    CapturePayload::Tensor(TensorObservation::new(vec![values.len()], values).unwrap())
}

#[test]
fn physical_preview_corruption_is_rejected_before_any_scatter() {
    let g = geometry(4);
    let plan = Plan {
        maximum: 7,
        output: 7,
        declared: ObservationDtype::Floating,
        width: 4,
    };
    for corruption in 0..6 {
        let (mut state, logical) = plan.allocate();
        let mut current = record();
        current.payload = logical;
        let before = serde_json::to_value(&current).unwrap();
        let mut physical = record();
        physical.source_shape = Some(vec![2, 4]);
        physical.selected_shape = Some(vec![2, 4]);
        physical.source_dtype = Some(TensorDtype::F32);
        physical.outcome = CaptureOutcome::Truncated {
            available_elements: 8,
            emitted_elements: 7,
        };
        physical.payload = Some(payload(TensorObservationData::F32(vec![1.; 7])));
        match corruption {
            0 => physical.outcome = CaptureOutcome::Captured,
            1 => {
                physical.outcome = CaptureOutcome::Truncated {
                    available_elements: 9,
                    emitted_elements: 7,
                }
            }
            2 => physical.payload = Some(payload(TensorObservationData::F32(vec![1.; 6]))),
            3 => physical.payload = Some(payload(TensorObservationData::I64(vec![1; 7]))),
            4 => {
                physical.payload = Some(CapturePayload::Tensor(
                    TensorObservation::new(vec![1, 7], TensorObservationData::F32(vec![1.; 7]))
                        .unwrap(),
                ))
            }
            _ => physical.source_dtype = None,
        }
        assert!(state.validate(&g, &physical, &current, 0, 2, 8).is_err());
        assert_eq!(serde_json::to_value(&current).unwrap(), before);
        assert_eq!(state.copied, 0);
    }
}

#[test]
fn prepaid_integer_prefix_chooses_one_buffer_and_accepts_shared_physical_values() {
    let g = geometry(4);
    for unsigned in [false, true] {
        let plan = Plan {
            maximum: 7,
            output: 7,
            declared: ObservationDtype::Integer,
            width: 8,
        };
        assert_eq!(
            plan.usage().unwrap().host_bytes,
            8 * 7 + std::mem::size_of::<usize>() as u64
        );
        let (mut state, logical) = plan.allocate();
        assert!(logical.is_none());
        assert_eq!(state.shape.as_ref().unwrap(), &[7]);
        let mut current = record();
        let mut pointer = None;
        for (start, end) in [(0, 2), (2, 4), (4, 5)] {
            let count = (end - start) * 4;
            let len = count.min(7) as usize;
            let data = if unsigned {
                TensorObservationData::U64(
                    (0..len)
                        .map(|i| (1u64 << 63) + start * 4 + i as u64)
                        .collect(),
                )
            } else {
                TensorObservationData::I64(
                    (0..len)
                        .map(|i| -(1i64 << 54) + (start * 4 + i as u64) as i64)
                        .collect(),
                )
            };
            let mut physical = record();
            physical.source_dtype = Some(if unsigned {
                TensorDtype::U64
            } else {
                TensorDtype::I64
            });
            physical.outcome = crate::capture::completed_capture_outcome(
                &CaptureTransform::Preview { max_elements: 7 },
                count,
            );
            let tensor = TensorObservation::new(vec![len], data).unwrap();
            physical.payload = Some(CapturePayload::SharedTensor(
                eredu_core::SharedTensorObservation::retain(tensor, ()),
            ));
            state
                .validate(&g, &physical, &current, start, end, count)
                .unwrap();
            state.update(&g, &physical, &mut current, start, end);
            state.commit();
            assert!(state.shape.is_none());
            let data = current
                .payload
                .as_ref()
                .unwrap()
                .as_tensor()
                .unwrap()
                .data();
            let actual = match data {
                TensorObservationData::I64(v) => v.as_ptr() as usize,
                TensorObservationData::U64(v) => v.as_ptr() as usize,
                _ => panic!("integer destination changed"),
            };
            assert_eq!(*pointer.get_or_insert(actual), actual);
        }
        assert_eq!(state.copied, 7);
        let expected = if unsigned {
            TensorObservationData::U64((0..7).map(|i| (1u64 << 63) + i).collect())
        } else {
            TensorObservationData::I64((0..7).map(|i| -(1i64 << 54) + i).collect())
        };
        assert_eq!(
            current
                .payload
                .as_ref()
                .unwrap()
                .as_tensor()
                .unwrap()
                .data(),
            &expected
        );
    }
}

#[test]
fn empty_prefix_still_checks_actual_source_and_completes_all_rows() {
    let g = geometry(0);
    let plan = Plan {
        maximum: 7,
        output: 0,
        declared: ObservationDtype::Integer,
        width: 8,
    };
    let (mut state, logical) = plan.allocate();
    assert!(logical.is_none());
    let mut current = record();
    current.source_shape = Some(vec![5, 0]);
    current.selected_shape = Some(vec![5, 0]);
    for (start, end) in [(0, 2), (2, 4), (4, 5)] {
        let mut physical = record();
        physical.source_shape = Some(vec![end - start, 0]);
        physical.selected_shape = Some(vec![end - start, 0]);
        physical.source_dtype = Some(TensorDtype::U64);
        physical.outcome = CaptureOutcome::Captured;
        physical.payload = Some(payload(TensorObservationData::U64(vec![])));
        let mut wrong = physical.clone();
        wrong.source_dtype = None;
        assert!(state.validate(&g, &wrong, &current, start, end, 0).is_err());
        wrong.source_dtype = Some(TensorDtype::F32);
        assert!(state.validate(&g, &wrong, &current, start, end, 0).is_err());
        state
            .validate(&g, &physical, &current, start, end, 0)
            .unwrap();
        state.update(&g, &physical, &mut current, start, end);
        state.commit();
        assert!(
            matches!(current.payload.as_ref().unwrap().as_tensor().unwrap().data(),TensorObservationData::U64(v) if v.is_empty())
        );
    }
    assert_eq!(state.copied, 0);
    assert_eq!(
        crate::capture::encoded::prefill_terminal_outcome(&current),
        Some(CaptureOutcome::Captured)
    );
}
